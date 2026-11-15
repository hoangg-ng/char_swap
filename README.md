# Character swap with MiniMax H3

Replace **one chosen person** in a video with a new character built from reference photos. Everyone else, the
background, the camera and the motion stay as in the source video. The heavy models run in ComfyUI; the pipeline
around them (cuts, target detection, references, assembly, face refinement) is plain Python.

Each video is a **case** folder with its own inputs, settings and outputs:

## Pipeline

```
 0. character sheet      the character photo + face photo -> Qwen-Image-2.1 sheet (front, side, back, face panels),
                         in her own clothes or (KEEP_OUTFIT=1) in the video's outfit
 1. scene cuts           CUTS=auto: sensitive PySceneDetect + colour check + 1-frame flash detection
 2. target check         which shots/segments contain the person to replace (PRESENCE=face|person|head)
 3. references per shot  recoloured to the scene's light (people masked out); face references only where a face
                         is visible, the back view first where none is
 4. generation 480p      MiniMax H3 + BFS v1.1 swap LoRA + 4-step Turbo LoRA (+ SAM lock on PROTECT people)
 5. upscale to 720p      H3 latent upscaler x1.5 + refine pass with the Realism People LoRA
 6. assembly             generated and source shots joined, letterbox restored, tone/light rules for dark shots
 7. FaceRefine + guard   faces re-rendered larger; any frame whose identity drops is reverted
```

| # | Step | Component | What it does |
|---|---|---|---|
| 0 | **Character sheet** | `utils/make_char_sheet.py`, `utils/sheet_refs.py` | Renders the character from several angles with Qwen-Image-2.1 + a 4-step Lightning LoRA. Stage 1 makes a clean front view from the cut-out and face photo (dressed in the video's outfit with `KEEP_OUTFIT=1`); stage 2 makes one character sheet from it.
| 1 | **Shot split** | `bfs_full.py` | Cuts the clip and converts to 24 fps and to the generation size, adds lead frames and pads to H3's frame grid. Shots run in parallel, one ComfyUI server per GPU. |
| 2 | **Target check** | `bfs_full.py` (insightface, YOLO11) | a shot is generated only where the person to replace is seen (ArcFace match against `TARGET_FACE`), because the swap LoRA always replaces the most visible person. Shots are split into segments with and without her; segments with only a sleeve or a silhouette stay source. |
| 3 | **References** | `utils/scene_color.py` | Recolours the references toward each shot's light, with colour statistics from the scene without people, so the original person's clothes cannot tint the new character. |
| 4 | **Guide** | `bfs_full.py` | The original shot is the guide video. The original person's face is softened in it, so the model takes the face from the references instead of copying it.|
| 5 | **Lock the others** (`PROTECT`) | SAM 3.1 | Tracks the people to keep and applies the inverted mask as the latent edit mask, so they cannot change. `SUBJECT` does the opposite: only that person is edited. |
| 6 | **Generation** | MiniMax H3 ref2va + BFS v1.1 + Turbo LoRA | 4 sampling steps with the Turbo LoRA (an 8-step DMD LoRA is optional). |
| 7 | **Upscale** (optional) | H3 latent upscaler + refine pass | The latent is upscaled x1.5 (480p -> 720p) and refined at the new size; the Realism People LoRA runs in this pass. |
| 8 | **Assembly** | `assemble_bfs.py` | Drops the lead frames, scales back to the source size, restores letterbox bars, and applies two light rules: a backlit silhouette keeps the original's darkness, and a character drawn far too bright in a dark scene is toned down. Only lightness is matched, never the original person's colours. |
| 9 | **FaceRefine + identity guard** | `utils/face_refine.py` | Per shot: tracks the face, re-renders the face crop at a larger size and pastes it back. The guard compares each refined frame with the frame before refinement (ArcFace similarity to the character's face photo) and cross-fades back where the refined face is worse. Runs last. |

### Scene cuts (`CUTS=auto`)
A missed cut is the most damaging error: the shot is generated as one clip and the model draws the person into a
scene without her. `auto_cuts` in `bfs_full.py` detects sensitively and then checks every candidate:
1. **Candidates:** PySceneDetect `AdaptiveDetector` at a low threshold, so dissolves and cuts between similar shots
   are found.
2. **Colour check:** HSV histogram distance across the candidate; fast motion alone keeps the colours and is dropped.
3. **Snap:** the cut moves to the sharpest frame change within a few frames.
4. **Same-scene check:** ORB features matched across the cut; an occluder crossing the shot keeps the background
   and is dropped.
5. **Flash frames:** 1-2 frames of another shot get their own cuts, so they are generated, not kept.

Hand-set `CUTS` in `case.env` always win.

## Setup
Accept the MiniMax H3 licence on Hugging Face first, then:
```bash
HF_TOKEN=<your token> bash scripts/setup.sh              # ComfyUI, custom nodes, Python deps, all model weights
HF_TOKEN=<your token> bash scripts/download_models.sh    # weights only (skips files already present)
```

Models:
- MiniMax H3: `minimax_h3_ref2va_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_nvfp4_awq`, video/audio VAEs
- SAM 3.1: `sam3.1_multiplex_fp16`
- Qwen-Image-2.1 (character sheet): `qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`
- LoRAs: BFS v1.1 `minimax_h3_character_swap_v1.1_ref2va_fro`, Realism `h3-realism-people-t2v-i2v-r2v`,
  DMD 8-step `minimax_h3_dmd_ref2va_8step_turbo_pruned`, `qwen_image_2.1_lightning_4steps_lightx2v`
- Turbo LoRA (default sampler): `minimax_h3_turbo_v4_step600_ema` from `larryvrh/MiniMax-H3-Turbo-Lora` with its
  `ComfyUI-MiniMax-H3-Turbo` node (installed by hand)
- latent upscaler: `minimax_h3_latent_upscaler_2d_conv_v1_fp16`
- YOLO11 `yolo11m.pt`, `yolo11m-pose.pt`, `yolo11m-seg.pt`; insightface `buffalo_l` is fetched on first use
- custom nodes installed by `scripts/setup.sh`: `akatz-ai/h3-relay`, `Carasibana/ComfyUI-H3-FaceRefine`,
  `LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`

## Settings
Any setting can be set in `case.env` or overridden on the command line (`CASE=cases/<name> SEED=7 bash scripts/run_swap.sh`).

| setting | meaning |
|---|---|
| `VIDEO` | the source video (any size and fps) |
| `CUTS` | shot boundaries in source frames, or `auto` |
| `SHEET_PERSON` / `SHEET_FACE` | the character's full-body cut-out and face photo; the character sheet is built from them and sets the references |
| `CHAR_REF` / `CHAR_REF_EXTRA` / `CHAR_REF_BACK` / `CHAR_REF_FACE` | references set by hand instead of a sheet: main view, extra views, back view (leads in shots without a face), face references (used only where a face is visible) |
| `CHAR_DETAILS` | the character's head-to-toe description, starting with gender |
| `KEEP_OUTFIT=1` | identity, face, hair and body from the references; clothing as worn in the video (`OUTFIT_REF`, `OUTFIT_TEXT` describe the outfit) |
| `TARGET_FACE` / `TARGET_SEX` | a face crop of the person to replace; shots are generated only where this identity is visible. Without it, `TARGET_SEX=F/M` counts any face of that gender |
| `TARGET_TEXT` | how the prompt names the person to replace |
| `PRESENCE` | `face` (several people), `person` or `head` (one person; segments without her stay source) |
| `PROTECT` / `PROTECT_TEXT` | SAM 3.1 text of the people to keep unchanged, and how the prompt names them |
| `SUBJECT` | SAM 3.1 text of the person to replace; only that person is edited (for a target half in frame next to another person) |
| `FORCE_SHOTS="2,4"` | generate these shots whatever the target check says |
| `SOFT_FACE_MARKS=1` | keep the original person's facial marks (painted tears, cuts, red eye rims) in the guide |
| `REF_COLOR` / `REF_COLOR_SCENE` / `REF_COLOR_MODE` | reference recolouring (`reinhard_lab:1.0`, empty = off), statistics from the scene only (default), mode `scene`, `scene_lum` or `skin` |
| `UPSCALE` (1.5) / `REALISM_STAGE` (2) | two-stage generation: 1.5 = 720p, `0` = 480p only; where the Realism LoRA runs (stage 1, 2 or both) |
| `REALISM` (0.8) / `FR_REALISM` | Realism LoRA strength for generation and for FaceRefine (`FR_REALISM` defaults to `REALISM`) |
| `FACE_REFINE` (1) / `FACE_REFINE_REF` | FaceRefine + identity guard on/off, and the identity image (default: the face photo) |
| `FACE_ONLY=1` | swap only the face and composite it onto the source |
| `TONE_PERSON` (0.8) / `LIGHT_RULES` (1) | tone match for a character drawn much brighter than the original, and the silhouette / dark-shot rules |
| `CROP` | `auto` or `w:h:x:y`: crop black bars for generation and pad them back afterwards |
| `RES` (480), `SEED` (42) | generation size and seed |
| `JOB` / `OUT` | folder names under `output/` |
| `ONLY="2,4"` | generate only these shots (the others reuse existing results in `JOB`) |
| `SPLICE="N=clip.mp4"` | replace shot N with a prepared clip |
| `CHECK_ONLY=1` / `ASSEMBLE_ONLY=1` | print the per-shot decisions and stop / re-assemble `JOB` without generating |

Per-shot overrides go in `output/<JOB>/shot_edits.json`, e.g. `{"13": {"tone": 0.5}, "4": {"soft_face_scale": 0.08}}`.

## References
- **Swap LoRA:** BFS v1.1 by [Alissonerdx](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI).
- **FaceRefine:** [Carasibana/ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine).
- **Realism LoRA:** [fal/MiniMax-H3-Realism-People-LoRA](https://huggingface.co/fal/MiniMax-H3-Realism-People-LoRA).
- **SageAttention:** J. Zhang et al., *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference
  Acceleration*, ICLR 2025 ([thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)).

## Licences
MiniMax H3, the BFS LoRA and the Realism People LoRA follow the MiniMax H3 Community Licence; read its territorial
terms before use. The BFS card asks for appropriate rights and consent for the reference images and guide videos.
Qwen-Image-2.1 follows the Qwen research licence.
