"""Facial marks of the ORIGINAL person (painted tears, red eye rims, cuts, wet tear streaks) for the soft-face guide.

SOFT_FACE blurs the original face in the guide so the model cannot copy her identity - and erases makeup and marks
the result should keep. marks_mask() finds only those marks, per frame, so they can be pasted back onto the blurred
face and move with it exactly:
  - light removed first (Retinex idea: brightness / its smooth large-scale version), so harsh or changing light does
    not decide what is a mark
  - redness well above her own median skin colour -> cuts, painted tears, red eyelids / eye rims
  - thin bright streaks below the eyes (high-pass of the light-free image) -> wet tear tracks
Eyes, brows, mouth, nostrils, the face outline and dark spots are left out: they carry identity (and dark-spot
detection mistook nostrils and jaw shadow for a mole).
"""
from __future__ import annotations

import cv2
import numpy as np


def marks_mask(roi: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """roi: face crop (BGR), kps: insightface 5 points in roi coordinates (eyes, nose, mouth corners).
    Returns a float mask 0..1 of the marks."""
    h, w = roi.shape[:2]
    if h < 24 or w < 24:
        return np.zeros((h, w), np.float32)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A = lab[..., 0] + 1, lab[..., 1]
    R = L / cv2.GaussianBlur(L, (0, 0), w / 10)                      # light-free detail
    le, re, no, lm, rm = [np.asarray(p, np.float32) for p in kps]
    eye_r = int(0.11 * w)
    face = np.zeros((h, w), np.uint8)
    cv2.ellipse(face, (w // 2, h // 2), (int(w * .40), int(h * .46)), 0, 0, 360, 1, -1)
    inner = np.zeros((h, w), np.uint8)                               # no contour, hairline, ears
    cv2.ellipse(inner, (w // 2, int(h * .55)), (int(w * .30), int(h * .34)), 0, 0, 360, 1, -1)
    excl = np.zeros((h, w), np.uint8)
    for e in (le, re):                                               # eye openings and brows
        cv2.ellipse(excl, (int(e[0]), int(e[1])), (eye_r, int(eye_r * .45)), 0, 0, 360, 1, -1)
        cv2.ellipse(excl, (int(e[0]), int(e[1] - eye_r)), (int(eye_r * 1.3), int(eye_r * .6)), 0, 0, 360, 1, -1)
    mc = ((lm + rm) / 2).astype(int); mw = int(np.linalg.norm(rm - lm) * .75)
    cv2.ellipse(excl, (int(mc[0]), int(mc[1])), (mw, int(mw * .55)), 0, 0, 360, 1, -1)       # mouth
    cv2.ellipse(excl, (int(no[0]), int(no[1])), (int(.06 * w), int(.07 * w)), 0, 0, 360, 1, -1)  # nostrils
    lids = np.zeros((h, w), np.uint8)                                # band around each eye: red rims
    eyes_big = np.zeros((h, w), np.uint8)
    for e in (le, re):
        cv2.ellipse(lids, (int(e[0]), int(e[1])), (int(eye_r * 1.25), int(eye_r * .85)), 0, 0, 360, 1, -1)
        cv2.ellipse(eyes_big, (int(e[0]), int(e[1])), (int(eye_r * 1.1), int(eye_r * .7)), 0, 0, 360, 1, -1)
    skin = (face > 0) & (excl == 0)
    if skin.sum() < 50:
        return np.zeros((h, w), np.float32)
    red = np.clip((A - np.median(A[skin]) - 8) / 10, 0, 1)
    red *= ((face > 0) & ((excl == 0) | (lids > 0))).astype(np.float32)
    red[(excl > 0) & (lids == 0)] = 0
    red[lids > 0] *= (R[lids > 0] > 0.75)                            # rims, not the dark iris / lash line
    under = np.zeros((h, w), np.uint8)                               # tear zone: below the eyes, above the mouth
    for e in (le, re):
        cv2.rectangle(under, (int(e[0] - eye_r), int(e[1] + eye_r * .3)),
                      (int(e[0] + eye_r), int(mc[1] - eye_r * .5)), 1, -1)
    em = ((le + re) / 2).astype(int)
    cv2.line(under, (int(em[0]), int(em[1])), (int(no[0]), int(no[1])), 0, max(2, int(.10 * w)))   # nose ridge
    hp = R - cv2.GaussianBlur(R, (0, 0), max(1.5, w / 80))
    sd = np.std(hp[skin]) + 1e-3
    tear = np.clip((hp - 2.0 * sd) / sd, 0, 1) * ((under > 0) & (inner > 0) & (eyes_big == 0))
    m = np.maximum(red, tear)
    m = cv2.GaussianBlur(cv2.dilate(m, np.ones((3, 3), np.uint8)), (0, 0), max(1.0, w / 150))
    return np.clip(m * 1.5, 0, 1).astype(np.float32)
