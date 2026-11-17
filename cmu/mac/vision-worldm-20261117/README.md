# David's Task B detector handoff

Related task: [Fetch B, continuous vision and approach](https://github.com/qilinxiaoxiang/microduck_rl/issues/3).
Base revision: `867edf0598cc401763b23f44e7b640ebaac2e0c8`.

This change prepares the three YOLO-World M v2 variants for the existing
`cmu/mac/learned_target.py` ONNX adapter. It adds an exporter and a model selector.
It does not complete target tracking, approach, pickup, or the full fetch task.

## Checkpoints and current evidence

The training notebook saved these paths under
`MyDrive/16831robotproject/microduck_world`:

| Variant | Checkpoint relative to that directory | Export name |
|---|---|---|
| Original pretrained baseline | `pretrained/yolov8m-worldv2.pt` | `zero_shot.onnx` |
| No DR | `runs/worldm_seed0_v1/no_dr/weights/last.pt` | `no_dr.onnx` |
| DR | `runs/worldm_seed0_v1/dr/weights/last.pt` | `dr.onnx` |

Both fine-tuned models start independently from the same pretrained model and
run 50 epochs, seed 0, input 640, batch 4, AdamW, initial learning rate 0.0002.
Training-time image augmentations were disabled in both groups to isolate
render-time DR. The two `last.pt` files are the fixed-epoch checkpoints used
in the recorded comparison. Each group's `best.pt` was selected on its own
validation domain; it is not the checkpoint used in this table.

The [recorded CSV](synthetic_validation_comparison.csv) comes from the completed
Colab notebook's common evaluation, using the **same 300 DR synthetic validation
images** for all three models (228 ball instances, 72 background images).

| Variant | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Original | 0.8601 | 0.7823 | 0.8624 | 0.6656 |
| No DR, last | 0.9613 | 0.8684 | 0.9333 | 0.6890 |
| DR, last | 0.9865 | 0.9635 | 0.9909 | 0.8455 |

These are validation results from one training seed, not an independent test
or evidence of real-robot transfer. Precision/recall are the library's default
reported operating point, not results at the deployment threshold of 0.25.

## 0. Download the published models

The models are published in
[vision-worldm-seed0-v1](https://github.com/wjcdavid/Microduck-RL/releases/tag/vision-worldm-seed0-v1).
All three uploaded archive sizes and SHA256 digests match the generated packages.
Use the current deployment branch for source code and the release for model files.

For a separate checkout on David's Mac, run:

```bash
cd "$HOME/Desktop"
git clone --depth 1 --single-branch --branch codex/david-vision-deployment \
  https://github.com/wjcdavid/Microduck-RL.git microduck_vision
cd microduck_vision
conda activate microduck-camera
python -m pip install -r cmu/mac/requirements-vision-runtime.txt
python cmu/mac/download_vision_models.py
```

If that checkout already exists, switch to its deployment branch and pull the
latest changes instead of cloning again. The download script uses only the
Python standard library. It installs `zero_shot.onnx`, `no_dr.onnx`, `dr.onnx`,
`manifest.json` and a synthetic `example_frame.jpg` under
`artifacts/vision/worldm_seed0_v1/`. It verifies archive and model checksums,
skips already matching files, and stops before replacing a differing local
file unless `--overwrite` is explicitly supplied. `--group dr` installs just
that model. The downloader reads only the model, manifest and example image;
it does not apply the older `deployment.patch` bundled in each ZIP.

Try all three models on the supplied synthetic example:

```bash
for group in zero_shot no_dr dr; do
  python cmu/mac/vision_model.py --group "$group" --conf 0.25 -- \
    python cmu/mac/learned_target.py \
    artifacts/vision/worldm_seed0_v1/example_frame.jpg \
    --out "artifacts/vision_compare/${group}.jpg"
done
open artifacts/vision_compare
```

This runs offline on the laptop. The example is a synthetic validation image;
replace its path with a real head-camera image for a qualitative real-image
comparison. It does not start a camera stream or move the robot.

| Model | Release package |
|---|---|
| Original pretrained | [microduck_zero_shot_deploy.zip](https://github.com/wjcdavid/Microduck-RL/releases/download/vision-worldm-seed0-v1/microduck_zero_shot_deploy.zip) |
| No DR, epoch 50 | [microduck_no_dr_deploy.zip](https://github.com/wjcdavid/Microduck-RL/releases/download/vision-worldm-seed0-v1/microduck_no_dr_deploy.zip) |
| DR, epoch 50 | [microduck_dr_deploy.zip](https://github.com/wjcdavid/Microduck-RL/releases/download/vision-worldm-seed0-v1/microduck_dr_deploy.zip) |

## 1. Export newly trained weights (optional)

The published models are already exported. This section is only needed for a
new training run or to reproduce the export.

Use the training Colab environment (Ultralytics 8.3.253) or a separate local
Python environment with a working PyTorch installation. No GPU is required for
this CPU export. On the Mac shown in the original setup, Python is x86_64;
using the existing Colab environment avoids changing that installation.

From a checkout of this branch:

```bash
python -m pip install -r cmu/mac/requirements-vision-export.txt
```

If the two downloaded checkpoints are on the Mac Desktop:

```bash
python cmu/mac/export_learned_target.py \
  --no-dr "$HOME/Desktop/last(no_dr).pt" \
  --dr "$HOME/Desktop/last(dr).pt"
```

Alternatively, in Colab, mount Drive and run the exporter from the repository
checkout using the original paths:

```bash
python cmu/mac/export_learned_target.py \
  --zero-shot /content/drive/MyDrive/16831robotproject/microduck_world/pretrained/yolov8m-worldv2.pt \
  --no-dr /content/drive/MyDrive/16831robotproject/microduck_world/runs/worldm_seed0_v1/no_dr/weights/last.pt \
  --dr /content/drive/MyDrive/16831robotproject/microduck_world/runs/worldm_seed0_v1/dr/weights/last.pt \
  --out /content/drive/MyDrive/16831robotproject/microduck_world/deploy/worldm_seed0_v1
```

Omitting `--zero-shot` lets Ultralytics download the official
`yolov8m-worldv2.pt`; the exporter checks its SHA256 against the training
baseline. Original checkpoint SHA256:
`b614d33aa35b8e61d988041ff6939dfb3ed627af88ccaf643e4cdb822eb41d71`.

The original model is given the prompt `orange ball`. Fine-tuned classes are
checked and preserved. All exports are static 640x640, batch 1, FP32, opset 17,
with `nms=False`, `dynamic=False`, `half=False`, and `simplify=False`.
The runtime adapter applies NMS itself. Exporting with embedded NMS changes
the output contract and is incompatible with this adapter.

Each ONNX file must pass the ONNX checker, metadata/input/output checks and
one CPU inference before the exporter reports success. The generated
`manifest.json` records checkpoint and ONNX hashes. Export uses temporary
copies and does not modify source `.pt` files.

Default output directory relative to the repo:
`artifacts/vision/worldm_seed0_v1/`. Copy the three ONNX files and manifest
there if they were exported in Colab. PyTorch and CLIP are not needed by the
robot after export.

## 2. Compare offline images first

Use a separate runtime environment if exporting elsewhere:

```bash
python -m pip install -r cmu/mac/requirements-vision-runtime.txt
```

Put a real head-camera image at `real_frame.jpg`, then run:

```bash
for group in zero_shot no_dr dr; do
  python cmu/mac/vision_model.py --group "$group" --conf 0.25 -- \
    python cmu/mac/learned_target.py real_frame.jpg \
    --out "out/vision_compare/${group}.jpg"
done
```

`0.25` is a shared starting threshold for replay, not a calibrated threshold.
The adapter also has a 16-pixel minimum box area and half-frame maximum box
side. Tiny or very close balls may be filtered even when the network finds
them. Validate these filters on real development recordings before final tests.

## 3. Select a model for the existing vision process

The selector injects `LEARNED_TARGET_MODEL`, `LEARNED_TARGET_PROMPTS=orange ball`,
and `LEARNED_TARGET_CONF` into an existing command. It prints shell exports if
no command is supplied:

```bash
python cmu/mac/vision_model.py --group dr --conf 0.25
```

Copy the printed exports into the shell that launches the team's existing
vision process. Change `dr` to `no_dr` or `zero_shot` to select another model.
Restart the process after changing the selection: settings are read at import
time and model sessions are cached. With no environment overrides, the previous
`small orange ball` prompt and 0.10 threshold remain the defaults for Shawn's
existing export.

The current orange-ball path in `seepick.py` uses `BallDetect` to run learned
inference in a background worker. Keep that worker, its frame context, tracking,
and the existing motion controller. Do not put synchronous YOLO inference into
the 50 Hz motor loop. The standalone `approach.py` entry point still uses its
HSV detector; changing the environment alone does not replace every detector
entry point. This handoff does not start robot motion.

Measure end-to-end latency on the actual robot before attempting closed-loop
approach. Colab L4 timing does not predict onboard CPU throughput, and the
latencies in `ball_detect.py` refer to Shawn's earlier model, not World M.

## 4. Share binaries separately from source

The three ONNX models and their manifest are in the published release packages
linked above. Their pinned URLs, archive sizes and checksums are recorded in
[release_assets.json](release_assets.json). The default `artifacts/` directory
and `.pt` files are already ignored by Git. No model binaries are included in
Git history. For a future training run, publish new versioned assets and update
the catalog after verifying them; retain this release for reproducibility.

## Verification and remaining work

```bash
python -m pip install pytest 'onnx>=1.12,<2'
python -m pytest -q cmu/mac/test_learned_target.py cmu/mac/test_download_vision_models.py
```

The automated checks use a tiny constant-output ONNX fixture to exercise
letterboxing, decoding, NMS, the hit/accept contract, class-name compatibility
and configuration propagation. The 10 adapter checks and 6 release-download
checks passed on 2026-11-17. The downloader checks include corrupt archives,
model digest mismatches, preservation of existing files and repeat installs.
All three public release packages were also downloaded to a fresh directory,
verified and used by the existing CLI to write annotated example images;
a second downloader run skipped all three matching models.

The actual uploaded `last(no_dr).pt` and `last(dr).pt` were also exported and
validated on 2026-11-17, together with the checksum-matched original model.
See [export_manifest.json](export_manifest.json) for checkpoint/ONNX hashes and
[export_verification.json](export_verification.json) for the recorded checks.
All three exports have input `[1,3,640,640]`, output `[1,5,8400]`, and one baked
class, `orange ball`. Their outputs were compared with the PyTorch models on
the same six existing synthetic validation images (four positive, two negative):
all 18 model/image comparisons passed. Detection counts agreed; the maximum
box difference was 0.5 pixels (the adapter rounds boxes) and the maximum
confidence difference was approximately 0.000033. This is a conversion check,
not an additional accuracy benchmark. Export used CPU PyTorch 2.15.0,
Ultralytics 8.3.253, ONNX 1.23.2, and the ONNX Runtime version in the report.
Real-image accuracy and onboard timing still require robot recordings/hardware.

Keep real development recordings separate from a final held-out test split
by recording session. Compare all three models on that same test set, including
no-ball frames, self-shadow, orange distractors, detection recall, false
selections, identity switches and end-to-end latency. Detection results alone
do not establish pickup success or completion of Task B.
