#!/usr/bin/env python3
"""Character sheet with Qwen-Image-2.1 (+ lightx2v Lightning 4-step LoRA): the character from a photo, dressed in
the outfit from the video, rendered from several angles on white. The views become MiniMax H3 references
(CHAR_REF + CHAR_REF_EXTRA in run_swap.sh), so BFS sees the identity and the outfit from front, side and back.

  python utils/make_char_sheet.py --person cases/x/input/full_body_cutout.png --outfit cases/x/input/outfit_ref.jpg \
      --face cases/x/input/face_cutout.png --out cases/x/input/sheet
Writes <out>/view_<name>.png per angle and <out>/sheet.png (all views side by side, for viewing).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # project root
sys.path.insert(0, str(ROOT))
from utils import comfy_client as cs  # noqa: E402

VIEWS = {   # name -> camera description (Chinese, like the model's own multi-reference examples)
    "front": "正面视角，人物面向镜头",
    "front_left": "相机向右绕人物转45度：人物的身体、脸和脚尖都朝向画面左前方，呈四分之三侧面，能看到她的右侧脸和右肩",
    "left": "相机绕人物转90度，拍摄完全的侧面：人物的身体、脸和脚尖都朝向画面左侧，只看到她的侧脸轮廓、一只眼睛和身体侧面",
    "back": "人物原地转身180度，背对镜头站立：画面中只看到她的后背、后脑勺和垂在背后的头发，看不到脸",   # turning the PERSON works, moving the camera did not
    "front_right": "相机向左绕人物转45度：人物的身体、脸和脚尖都朝向画面右前方，呈四分之三侧面",
    "face": "正面胸部以上的特写肖像：人物面向镜头，脸部占据画面大部分，清晰呈现五官、皮肤细节、发型和刘海",
}

# stage 1: dress the person (one image, front view). Image 2 is the garment only (head cropped off).
DRESS = ("图1提供人物身份（五官、脸型、发型、发色和体型），图2只提供服装（不提供人物），图3提供面部细节。"
         "画面中只有一个人：图1的人物脱下她原来的衣服，换上图2的服装，{view}。"
         "服装完全保持图2的款式、剪裁、花纹和颜色：{outfit}；不保留图1原来的服装。"
         "保持图1和图3人物的五官、发型和发色不变。身材高挑修长，比例比常人略高（腿部自然修长但不夸张）。表情自然平静，双眼睁开，嘴唇自然闭合，不做任何手势（不眨眼、不嘟嘴、不比手势）。人物自然站立，双臂自然下垂，从头到脚完整入镜，"
         "纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
         "每张输入图只提供指定的信息，不引入其余背景或物体。只生成一个人物。输出一张完整图片。")
# stage 1 without an outfit image: keep the person's own clothes, just a clean neutral front view.
NEUTRAL = ("图1提供人物（五官、脸型、发型、发色、体型和服装），图2提供面部细节。"
           "画面中只有一个人：生成图1人物的全身照，{view}，穿着图1中她自己的服装（保持款式、剪裁、花纹和颜色）。"
           "保持图1和图2人物的五官、发型和发色不变。身材高挑修长，比例比常人略高（腿部自然修长但不夸张）。表情自然平静，双眼睁开，嘴唇自然闭合，不做任何手势（不眨眼、不嘟嘴、不比手势）。人物自然站立，双臂自然下垂放在身体两侧，手中不拿任何物品，"
           "从头到脚完整入镜，纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
           "每张输入图只提供指定的信息，不引入其余背景或物体。只生成一个人物。输出一张完整图片。")
# stage 2: the same person and outfit from another camera angle (image 1 = stage-1 result).
ROTATE = ("图1是一个人物的全身照。生成同一个人物、同一套服装的全身照，改为{view}。"
          "保持图1人物的五官、发型、发色、体型、身高比例以及服装的款式、花纹、颜色和穿法完全不变，只改变相机角度。"
          "表情自然平静，双臂自然下垂，不做任何手势。"
          "人物自然站立，从头到脚完整入镜，纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
          "画面中只有一个人物。输出一张完整图片。")


# side view with a small OpenPose skeleton (utils/assets/pose_side_skeleton.png, not aligned with the output so it is
# not traced): 10/10 true side views and no drawn lines on 5 characters x 2 seeds; text alone gave 5-8/10
SIDE_SKELETON = ROOT / "utils" / "assets" / "pose_side_skeleton.png"
SKEL_PROMPT = ("图1是一个人物的全身照，图2是一张OpenPose姿势骨架示意图，只用来说明身体朝向，不是画面内容。"
               "生成图1同一个人物、同一套服装的全身照，身体朝向与图2骨架一致：{view}。"
               "保持图1人物的五官、发型、发色、体型、身高比例以及服装的款式、花纹、颜色和穿法完全不变。"
               "表情自然平静，双臂自然下垂，不做任何手势。人物自然站立，从头到脚完整入镜，纯白色背景，柔和均匀的影棚光，干净的写实照片。\n\n"
               "画面中只有一个人物。人物身上和画面中绝对不能出现任何彩色线条、圆点、骨架或标记。输出一张完整图片。")

# back view from the SIDE view: a 90-degree turn works first time (10/10 on 5 characters x 2 seeds), a 180-degree
# turn from the front mostly returned the front view unchanged (1-3/10)
SIDE_TO_BACK = ("图1是一个人物的侧面全身照。生成同一个人物、同一套服装的全身照，改为人物原地继续向同一方向转身90度，"
                "后背完全朝向镜头：看不到脸、五官和胸前的衣服；只看到后脑勺、垂在背后的头发、后背、臀部和小腿后侧，脚跟朝向镜头。"
                "保持图1人物的发型、发色、体型、身高比例以及服装的款式、花纹、颜色和穿法完全不变，只改变人物朝向。"
                "双臂自然下垂。人物自然站立，从头到脚完整入镜，纯白色背景，柔和均匀的影棚光，写实照片。\n\n"
                "画面中只有一个人物。输出一张完整图片。")

CLOSEUP = ("图1是一个人物的全身照。生成同一个人物的{view}。保持图1人物的五官、脸型、发型、发色、妆容和服装领口完全不变，"
           "只改变相机距离。纯白色背景，柔和均匀的影棚光，高清写实照片，皮肤纹理清晰。\n\n画面中只有一个人物。输出一张完整图片。")


# one-shot character sheet: a 3x2 grid in a single generation (default mode)
GRID_PROMPT = ("Create a character reference sheet of the person in the reference image. "
               "3x2 grid on a plain light grey background: "
               "1. close-up portrait of the face (neutral expression, both eyes open, lips closed, looking at camera), "
               "2. full-body front view, "
               "3. full-body back view, "
               "4. full-body side view, "
               "5. three-quarter front view, "
               "6. three-quarter back view. "
               "Keep the exact same face, hair, body proportions, clothing, and colors in every panel. "
               "Neutral relaxed expression with both eyes open in every panel, arms down, no hand gestures. "
               "Body proportions as in the reference image, a slightly tall and elegant figure: keep the same leg "
               "length relative to the head and torso, no extra stretching of the legs or the body. "
               "Clean studio lighting, no background elements.")
GREY = (217, 217, 217)          # medium grey was tried: worse sheets (stretched legs, framed panels)


# face-only close-up added to every sheet (identity detail for close-up shots)
FACE_ONLY_PROMPT = ("图1是一个人物头部的局部截图。把它生成为同一个人的高清面部特写照片：脸部占满画面，"
                    "人物正面直视镜头，表情自然平静，双眼睁开，嘴唇自然闭合，画面中没有手，不做任何手势，清晰呈现五官、皮肤纹理、发际线、刘海和耳朵。"
                    "保持图1人物的五官、脸型、肤色、发型、发色和妆容完全不变，不改变构图。纯浅灰色背景，"
                    "柔和均匀的影棚光，高清写实照片。\n\n画面中只有一个人物的脸。输出一张完整图片。")


def face_crop(src: Path, dst: Path, size: int = 1024) -> bool:
    """Square crop around the face (head and hair) of an image, scaled to size x size."""
    import logging
    logging.getLogger("insightface").setLevel(logging.ERROR)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=str(ROOT / "ComfyUI" / "models" / "insightface"),
                       allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    im = cv2.imread(str(src))
    faces = app.get(im)
    if not faces:
        return False
    x0, y0, x1, y1 = max(faces, key=lambda f: f.det_score).bbox
    cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2 - 0.1 * (y1 - y0), 1.9 * max(x1 - x0, y1 - y0)
    H, W = im.shape[:2]
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    cv2.imwrite(str(dst), cv2.resize(im[b:d, a:c], (size, size), interpolation=cv2.INTER_CUBIC))
    return True


def pad_to_aspect(src: Path, dst: Path, aspect: float = 1.5) -> None:
    """Put the reference on a light grey canvas of the grid's aspect (the latent follows the first image)."""
    im = cv2.imread(str(src))
    h, w = im.shape[:2]
    W, H = (int(h * aspect), h) if w / h < aspect else (w, int(w / aspect))
    canvas = np.full((H, W, 3), GREY, np.uint8)
    y, x = (H - h) // 2, (W - w) // 2
    canvas[y:y + h, x:x + w] = im
    cv2.imwrite(str(dst), canvas)


def _runs(profile: np.ndarray, min_len: int) -> list:
    """[start, end) runs where profile is True, at least min_len long."""
    out, i, n = [], 0, len(profile)
    while i < n:
        if profile[i]:
            j = i
            while j < n and profile[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def _segments(prof: np.ndarray, n: int) -> list:
    """Content segments along one axis: thin full-length border lines removed, smoothed coverage > 3%, gaps
    narrower than 0.4% merged, segments narrower than 3% (specks, panel numbers) dropped."""
    prof = prof.copy()
    for a, b in _runs(prof > 0.97, 1):              # thin, (almost) full-length lines = panel borders
        if b - a <= 6:
            prof[a:b] = 0
    sm = np.convolve(prof, np.ones(5) / 5, mode="same")
    runs = _runs(sm > 0.03, 1)
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 0.004 * n:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= 0.03 * n]


def leg_ratio(img_path) -> float | None:
    """(hip -> ankle) / (shoulder -> hip) of the person, from YOLO11-pose keypoints; None if not a full body."""
    import cv2 as _cv2
    from ultralytics import YOLO
    global _POSE
    if "_POSE" not in globals():
        _POSE = YOLO(str(ROOT / "ComfyUI" / "models" / "ultralytics" / "person" / "yolo11m-pose.pt"))
    r = _POSE.predict(_cv2.imread(str(img_path)), conf=0.3, verbose=False, device="cpu")[0]
    if r.keypoints is None or not len(r.boxes):
        return None
    kp = r.keypoints.data.cpu().numpy()[int(r.boxes.conf.argmax())]
    c, y = kp[:, 2], kp[:, 1]
    if min(c[11], c[12]) < 0.4 or max(c[15], c[16]) < 0.4 or max(c[5], c[6]) < 0.4:
        return None
    sh = (y[5] + y[6]) / 2; hip = (y[11] + y[12]) / 2; ank = max(y[15] if c[15] >= 0.4 else 0, y[16] if c[16] >= 0.4 else 0)
    return (ank - hip) / max(hip - sh, 1.0)


def fix_legs(img_path, target: float, tol: float = 0.04) -> float | None:
    """If the panel's legs are longer than `target` (hip->ankle / shoulder->hip), squeeze the part below the hips
    vertically to the target length; face, torso and arms stay untouched, the freed height is filled with the
    background at the bottom. Returns the factor applied, or None."""
    import cv2 as _cv2
    r = leg_ratio(img_path)
    if r is None or r <= target * (1 + tol):
        return None
    im = _cv2.imread(str(img_path)); H, W = im.shape[:2]
    kp = _POSE.predict(im, conf=0.3, verbose=False, device="cpu")[0].keypoints.data.cpu().numpy()[0]
    hip = int((kp[11, 1] + kp[12, 1]) / 2)
    f = target / r
    lower = im[hip:]
    newh = max(1, int(round(lower.shape[0] * f)))
    squeezed = _cv2.resize(lower, (W, newh), interpolation=_cv2.INTER_AREA)
    bg = np.median(im[-6:].reshape(-1, 3), axis=0).astype(np.uint8)
    out = np.empty_like(im); out[:] = bg
    out[:hip] = im[:hip]; out[hip:hip + newh] = squeezed
    _cv2.imwrite(str(img_path), out)
    return f


def facing(img_path) -> str | None:
    """Rough camera angle of the single person: front | three_quarter | side | back (YOLO11-pose)."""
    import cv2 as _cv2
    leg_ratio(img_path)
    r = _POSE.predict(_cv2.imread(str(img_path)), conf=0.35, verbose=False, device="cpu")[0]
    if r.keypoints is None or not len(r.boxes):
        return None
    kp = r.keypoints.data.cpu().numpy()[int(r.boxes.conf.argmax())]; c = kp[:, 2]
    face = max(c[0], c[1], c[2]) >= 0.5                          # nose / eyes visible
    if not face:
        return "back"
    sw = abs(kp[5, 0] - kp[6, 0]); th = abs((kp[11, 1] + kp[12, 1]) / 2 - (kp[5, 1] + kp[6, 1]) / 2) or 1.0
    w = sw / th                                                  # shoulder width relative to torso height
    off = abs(kp[0, 0] - (kp[5, 0] + kp[6, 0]) / 2) / max(sw, 1.0)   # nose offset from the shoulder centre
    if w < 0.30 or (min(c[1], c[2]) < 0.5 and off > 0.4):
        return "side"
    return "three_quarter" if off >= 0.12 else "front"


def panel_ok(img_path) -> bool:
    """A usable panel holds exactly one person with shoulders (not legs only, not a headless fragment) and that
    person fills most of the panel height or is a close-up from the shoulders up."""
    import cv2 as _cv2
    leg_ratio(img_path)                                         # loads the pose model
    im = _cv2.imread(str(img_path))
    r = _POSE.predict(im, conf=0.35, verbose=False, device="cpu")[0]
    if r.keypoints is None or len(r.boxes) != 1:
        return False
    kp = r.keypoints.data.cpu().numpy()[0]; c = kp[:, 2]
    if max(c[5], c[6]) < 0.4:
        return False                                            # no shoulders: legs or a fragment
    head = max(c[0], c[1], c[2], c[3], c[4]) >= 0.3 or kp[5, 1] > 0.08 * im.shape[0]   # face or room for a head (back)
    return bool(head)


def split_grid(grid: Path, out: Path, pad: int = 16) -> list:
    """Cut the sheet into its panels, whatever layout the model chose (3x2, 1x4, ...): figures are separated by
    plain background, panel border lines are too thin to count. Writes view_panel<k>.png (row-major)."""
    im = cv2.imread(str(grid))
    H, W = im.shape[:2]
    bg = np.median(np.concatenate([im[:8].reshape(-1, 3), im[-8:].reshape(-1, 3)]), axis=0)
    fg = np.abs(im.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > 28
    # pale skin on the light grey background is below that threshold, so a leg could be taken for a gap and cut
    # through: every pixel of a segmented person counts as figure, whatever its colour
    try:
        sys.path.insert(0, str(ROOT))
        from utils import tone_match as _tm
        r = _tm.model().predict(im, classes=[0], conf=0.25, verbose=False, device="cpu", retina_masks=True)[0]
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                m = m > 0.5
                if m.shape != fg.shape:
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                fg |= m
    except Exception as e:                                       # segmentation is a refinement, not required
        print(f"split_grid: person masks unavailable ({e})")
    outs, k = [], 0
    for x0, x1 in _segments(fg.mean(axis=0), W):
        for y0, y1 in _segments(fg[:, x0:x1].mean(axis=1), H):
            k += 1
            dst = out / f"view_panel{k}.png"
            cv2.imwrite(str(dst), im[max(0, y0 - pad):min(H, y1 + pad), max(0, x0 - pad):min(W, x1 + pad)])
            outs.append(dst)
    return outs


def build(images: list, prompt: str, seed: int, prefix: str, resolution: int = 1536) -> dict:
    wf = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "qwen_image_2.1_int8_convrot.safetensors", "weight_dtype": "default"}},
        "attn": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["unet", 0], "attention": "pytorch attention"}},
        "lightning": {"class_type": "LoraLoaderModelOnly",
                      "inputs": {"model": ["attn", 0], "lora_name": "qwen_image_2.1_lightning_4steps_lightx2v.safetensors",
                                 "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "enc": {"class_type": "TextEncodeQwenImage21",
                "inputs": {"clip": ["clip", 0], "prompt": prompt, "negative_prompt": "", "vae": ["vae", 0],
                           "resolution": resolution}},
        "ks": {"class_type": "KSampler",
               "inputs": {"model": ["lightning", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                          "scheduler": "simple", "positive": ["enc", 0], "negative": ["enc", 1],
                          "latent_image": ["enc", 2], "denoise": 1.0}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    }
    for i, name in enumerate(images, 1):
        wf[f"img{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        wf["enc"]["inputs"][f"images.image_{i}"] = [f"img{i}", 0]
    return wf


def write_preview(views: list, dst: Path, height: int = 1024) -> None:
    """Labelled side-by-side preview of the views (JPEG, for looking at; not used by the pipeline)."""
    tiles = []
    for p in views:
        im = cv2.imread(str(p))
        im = cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height), interpolation=cv2.INTER_AREA)
        name = Path(p).stem.replace("view_", "")
        cv2.putText(im, name, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(im, name, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(im)
    cv2.imwrite(str(dst), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", type=Path, required=True, help="full-body photo of the character (cut-out on white)")
    ap.add_argument("--outfit", type=Path, default=None,
                    help="image of the outfit to wear (e.g. a video crop without the head); omit to keep her own clothes")
    ap.add_argument("--face", type=Path, required=True, help="face close-up of the character")
    ap.add_argument("--outfit-text", default="与图2完全相同的服装和配饰",
                    help="short description of the outfit image, Chinese works best, e.g. for a kimono: "
                         "红色和服，红白相间的波浪花纹，单肩露出，宽大的白色衣领，花纹腰带，垂下的黄色系带")
    ap.add_argument("--views", default="left,back,face",
                    help="views mode: angles besides the front (three-quarter views did not turn reliably, so not default)")
    ap.add_argument("--view-tries", type=int, default=1, help="views mode: attempts per view until it passes the checks (1 = no retry: faster, the first picture is kept even at an off angle)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sheet-tries", type=int, default=4, help="grid mode: seeds tried until the proportions match")
    ap.add_argument("--taller", type=float, default=1.05,
                    help="grid mode: target leg/torso ratio of the panels relative to the front view (1.05 = about 5% taller)")
    ap.add_argument("--max-stretch", type=float, default=0.10,
                    help="grid mode: largest allowed leg/torso deviation of a panel from the front view")
    ap.add_argument("--mode", default="views", choices=["grid", "views"],
                    help="views: one generation per camera angle from the front view, each checked (default); "
                         "grid: one sheet in a single generation, split into panels (stretched legs, fragments)")
    ap.add_argument("--resolution", type=int, default=1536,
                    help="views mode: Qwen render size per view (~resolution^2 pixels; 1536 -> ~1248x1920)")
    ap.add_argument("--grid-resolution", type=int, default=2048,
                    help="grid mode: render size of the whole sheet (2048 -> ~2496x1664, panels ~830 px)")
    ap.add_argument("--reuse-grid", action="store_true",
                    help="with --reuse-front: keep an existing sheet_grid.png (re-split it, redo only the face close-up)")
    ap.add_argument("--no-face", action="store_true", help="grid mode: skip the face-only close-up (view_face.png)")
    ap.add_argument("--front-resolution", type=int, default=1024,
                    help="stage 1 (dressing) render size; higher values tend to draw the person twice")
    ap.add_argument("--reuse-front", action="store_true", help="keep an existing <out>/view_front.png (stage 1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--comfy-urls", default="http://127.0.0.1:8188,http://127.0.0.1:8189,http://127.0.0.1:8190")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    inp = ROOT / "ComfyUI" / "input"
    names = {}
    for key, p in (("p", args.person), ("o", args.outfit), ("f", args.face)):
        if p is None:
            continue
        names[key] = f"sheet_{os.getpid()}_{key}{p.suffix}"
        shutil.copy2(p, inp / names[key])
    urls = args.comfy_urls.split(",")
    views = [v for v in args.views.split(",") if v != "front"]

    def gen(k, images, prompt, view, res=None, seed=None):
        c = cs.Comfy(urls[k % len(urls)])
        h = c.wait(c.submit(build(images, prompt, args.seed if seed is None else seed, f"sheet/{view}",
                                  res or args.resolution)), poll=2)
        tmp = args.out / f"_{view}"
        tmp.mkdir(parents=True, exist_ok=True)
        files = [f for f in c.download_outputs(h, tmp) if f.suffix == ".png"]
        dst = args.out / f"view_{view}.png"
        shutil.move(str(files[-1]), dst)
        shutil.rmtree(tmp, ignore_errors=True)
        (args.out / f"prompt_{view}.txt").write_text(prompt + "\n")
        print(f"{view}: {dst}", flush=True)
        return dst

    # face close-up from the real face photo: needs only the face photo, so it runs on the last server while the front
    # view is made (that server is then warm for a body view); views mode only, grid mode keeps its own order
    face_job = None
    if args.mode != "grid" and "face" in views and not args.no_face:
        fc = args.out / "_face_crop.png"
        if face_crop(args.face, fc):
            names["face_crop"] = f"sheet_{os.getpid()}_facecrop.png"
            shutil.copy2(fc, inp / names["face_crop"]); fc.unlink()
            face_job = ThreadPoolExecutor(1).submit(gen, len(urls) - 1, [names["face_crop"]], FACE_ONLY_PROMPT,
                                                    "face", 1024)

    # stage 1: front view, dressed
    front = args.out / "front_reference.png"                # grid mode keeps stage 1 here
    if not front.exists():
        front = args.out / "view_front.png"
    if not (args.reuse_front and front.exists()):
        if args.outfit:
            front = gen(0, [names["p"], names["o"], names["f"]],
                        DRESS.format(view=VIEWS["front"], outfit=args.outfit_text), "front", args.front_resolution)
        else:
            front = gen(0, [names["p"], names["f"]], NEUTRAL.format(view=VIEWS["front"]), "front",
                        args.front_resolution)
    names["front"] = f"sheet_{os.getpid()}_front.png"
    shutil.copy2(front, inp / names["front"])
    if args.mode == "grid":
        # stage 2 (grid): the whole sheet in one generation from the front view, then split into panels
        ref = args.out / "_ref_front_padded.png"
        pad_to_aspect(front, ref)
        names["grid_ref"] = f"sheet_{os.getpid()}_gridref.png"
        shutil.copy2(ref, inp / names["grid_ref"])
        ref.unlink()
        front_ref = args.out / "front_reference.png"          # keep the stage-1 image, the panels replace view_front
        if front != front_ref:
            shutil.copy2(front, front_ref)
        existing = args.out / "sheet_grid.png"
        if args.reuse_front and args.reuse_grid and existing.exists():
            grid = existing                           # keep the sheet, only (re)make the face close-up
            outs = split_grid(grid, args.out)
        else:
            # proportion check: the grid layout (tall narrow panels) tends to stretch the legs; compare every
            # full-body panel's leg/torso ratio with the front view and retry other seeds, keep the most faithful
            ref_r = leg_ratio(front)
            best = None
            for t, sd in enumerate([args.seed] + [args.seed + 16 * k for k in range(1, args.sheet_tries)]):
                g_ = gen(0, [names["grid_ref"]], GRID_PROMPT, "sheet", args.grid_resolution, seed=sd)
                trial = args.out / f"_try{t}"; trial.mkdir(exist_ok=True)
                g_ = Path(shutil.move(str(g_), trial / "sheet_grid.png"))
                panels = split_grid(g_, trial)
                rs = [r for r in (leg_ratio(pp) for pp in panels) if r]
                tgt = ref_r * args.taller if ref_r else None         # a bit taller than the front view by default
                err = max((abs(r / tgt - 1) for r in rs), default=0.0) if tgt and rs else 0.0
                bad = [pp.stem.replace("view_panel", "") for pp in panels if not panel_ok(pp)]
                if bad or len(panels) < 4:
                    err += 1.0                                  # broken panels outweigh any proportion error
                print(f"sheet try {t + 1} (seed {sd}): legs {', '.join(f'{r / ref_r:.2f}x' for r in rs) if ref_r else '?'}"
                      f" of the front view" + (f", broken panels {','.join(bad)}" if bad else "")
                      + f" -> {'ok' if err <= args.max_stretch else 'rejected'}", flush=True)
                if best is None or err < best[0]:
                    best = (err, trial)
                if err <= args.max_stretch:
                    break
            for old in args.out.glob("view_panel*.png"):
                old.unlink()
            for f_ in best[1].iterdir():
                shutil.move(str(f_), args.out / f_.name)
            for t_ in args.out.glob("_try*"):
                shutil.rmtree(t_, ignore_errors=True)
            grid = existing
            outs = sorted(args.out.glob("view_panel*.png"), key=lambda q: int(q.stem.replace("view_panel", "")))
            if ref_r:                                     # guarantee the proportions: squeeze legs that are still long
                for pp in outs:
                    total = 1.0
                    for _ in range(5):                    # re-measure: the hip estimate moves after each squeeze
                        f_ = fix_legs(pp, ref_r * args.taller)
                        if not f_:
                            break
                        total *= f_
                    if total < 1.0:
                        r_ = leg_ratio(pp)
                        print(f"{pp.name}: legs shortened to {total:.2f} of their height -> "
                              f"{(r_ or 0) / ref_r:.2f}x the front view (target {args.taller:.2f}x)")
        (args.out / "sheet.png").unlink(missing_ok=True)
        prev = cv2.imread(str(grid))
        if not args.no_face:                          # face-only close-up from the same front reference
            fc = args.out / "_face_crop.png"
            # identity from the real face photo (--face); the generated front view drifts toward a generic face
            if face_crop(args.face, fc) or face_crop(front, fc):
                names["face_crop"] = f"sheet_{os.getpid()}_facecrop.png"
                shutil.copy2(fc, inp / names["face_crop"])
                fc.unlink()
                face = gen(1, [names["face_crop"]], FACE_ONLY_PROMPT, "face", 1024)
                outs.append(face)
            else:
                face = None
        if not args.no_face and face is not None:
            fim = cv2.imread(str(face))
            fim = cv2.resize(fim, (int(fim.shape[1] * prev.shape[0] / fim.shape[0]), prev.shape[0]))
            prev = np.hstack([prev, fim])
        cv2.imwrite(str(args.out / "sheet.jpg"), cv2.resize(prev, (int(prev.shape[1] * 1024 / prev.shape[0]), 1024)),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        for v in names.values():
            (inp / v).unlink(missing_ok=True)
        print(f"sheet: {grid} -> panels {', '.join(p.name for p in outs)}")
        return
    # stage 2 (views mode, default): every view is its own generation from the front view. The output takes the
    # front view's canvas shape (first reference), so nothing pushes the legs longer; no grid to split, no fragments.
    # Each view is checked (one complete person, the right camera angle) and its legs corrected to the target;
    # only a failing view is redone.
    want = {"left": {"side"}, "back": {"back"}, "front_left": {"three_quarter", "side"}, "front_right": {"three_quarter", "side"},
            "front": {"front", "three_quarter"}}
    ref_r = leg_ratio(front)
    body = [v for v in views if v != "face"]

    order = [0, len(urls) - 1] + list(range(1, len(urls) - 1))      # body views go to the warm servers first

    if SIDE_SKELETON.exists():
        names["skel"] = f"sheet_{os.getpid()}_skel.png"
        shutil.copy2(SIDE_SKELETON, inp / names["skel"])

    def one_view(kv, src_name=None, prompt=None):
        k, v = kv
        k = order[k % len(order)]
        best = None
        imgs = [src_name or names["front"]]
        if v == "left" and src_name is None and "skel" in names:      # side view: pose skeleton as image 2
            imgs, prompt = imgs + [names["skel"]], SKEL_PROMPT.format(view=VIEWS["left"])
        for t in range(args.view_tries):
            out = gen(k, imgs, prompt or ROTATE.format(view=VIEWS[v]), v, seed=args.seed + 101 * t)
            ok, ang = panel_ok(out), facing(out)
            good_angle = ang in want.get(v, {ang})
            print(f"view {v} try {t + 1}: {'one complete person' if ok else 'BROKEN'}, angle {ang}"
                  f" -> {'ok' if ok and good_angle else 'redo'}", flush=True)
            if ok and good_angle:
                best = out; break
            if ok and best is None:
                best = out                                 # keep a usable picture even if the angle is off
            if not ok:
                out.unlink(missing_ok=True)
        if best is not None and ref_r:
            total = 1.0
            for _ in range(5):
                f_ = fix_legs(best, ref_r * args.taller, tol=0.02)
                if not f_:
                    break
                total *= f_
            r_ = leg_ratio(best)
            print(f"view {v}: legs {(r_ or 0) / ref_r:.2f}x the front view" + (f" (shortened to {total:.2f})" if total < 1 else ""), flush=True)
        return best

    if "back" in body and "left" in body:
        # side first, then the back turned from the side (one try each, on the warm server); other views in parallel
        others = [v for v in body if v not in ("left", "back")]
        with ThreadPoolExecutor(len(urls)) as ex:
            fut = ex.map(one_view, [(i + 1, v) for i, v in enumerate(others)])
            side = one_view((0, "left"))
            back = None
            if side is not None:
                names["side"] = f"sheet_{os.getpid()}_side.png"
                shutil.copy2(side, inp / names["side"])
                back = one_view((0, "back"), names["side"], SIDE_TO_BACK)
            got = dict(zip(others, fut))
        got.update(left=side, back=back)
        rest = [got[v] for v in body if got.get(v) is not None]
    else:
        with ThreadPoolExecutor(len(urls)) as ex:
            rest = [r_ for r_ in ex.map(one_view, enumerate(body)) if r_ is not None]
    # face close-up from the real face photo (identity detail for close-up shots)
    face_img = face_job.result() if face_job else None
    if face_img is None and "face" in views and not args.no_face:
        fc = args.out / "_face_crop.png"
        if face_crop(args.face, fc) or face_crop(front, fc):
            names["face_crop"] = f"sheet_{os.getpid()}_facecrop.png"
            shutil.copy2(fc, inp / names["face_crop"]); fc.unlink()
            face_img = gen(0, [names["face_crop"]], FACE_ONLY_PROMPT, "face", 1024)
    # the pipeline (utils/sheet_refs.py) reads view_panel<k>.png: 1 = front, then the other body views
    for old in args.out.glob("view_panel*.png"):
        old.unlink()
    panels = [front] + rest
    for k, pth in enumerate(panels, 1):
        shutil.copy2(pth, args.out / f"view_panel{k}.png")
    write_preview(panels + ([face_img] if face_img else []), args.out / "sheet.jpg")
    for v in names.values():
        (inp / v).unlink(missing_ok=True)
    print(f"sheet: views {', '.join(['front'] + body)} -> view_panel1..{len(panels)}.png" + (", view_face.png" if face_img else ""))

if __name__ == "__main__":
    main()
