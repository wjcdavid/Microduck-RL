#!/usr/bin/env python3
"""Export the original, no-DR and DR YOLO-World checkpoints for learned_target.py.

This is a detector exporter, separate from scripts/export.py for locomotion policies.
Requires requirements-vision-export.txt. Only the original model needs CLIP text encoding.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


PRETRAINED_SHA256 = "b614d33aa35b8e61d988041ff6939dfb3ed627af88ccaf643e4cdb822eb41d71"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "artifacts/vision/worldm_seed0_v1"
LABEL = "orange ball"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export(path):
    """Check the metadata, static input and raw output contract used by the robot adapter."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640] or inputs[0].type != "tensor(float)":
        raise ValueError(f"expected one static FP32 input [1,3,640,640], got {inputs}")
    names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get("names", "{}"))
    if names != {0: LABEL}:
        raise ValueError(f"expected baked names {{0: {LABEL!r}}}, got {names!r}")
    values = session.run(None, {inputs[0].name: np.zeros((1, 3, 640, 640), dtype=np.float32)})
    if len(values) != 1:
        raise ValueError("expected one detection output")
    output = values[0]
    if output.ndim != 3 or output.shape[:2] != (1, 5) or output.shape[2] < 1:
        raise ValueError(f"expected raw [1,5,N] output without embedded NMS, got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("ONNX output contains non-finite values")
    return {"input_shape": inputs[0].shape, "output_shape": list(output.shape), "names": names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-dr", type=Path, required=True, help="no_dr/weights/last.pt (epoch 50)")
    parser.add_argument("--dr", type=Path, required=True, help="dr/weights/last.pt (epoch 50)")
    parser.add_argument("--zero-shot", type=Path,
                        help="original yolov8m-worldv2.pt; omit to download the official checkpoint")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    sources = {"no_dr": args.no_dr.expanduser().resolve(), "dr": args.dr.expanduser().resolve()}
    for source in sources.values():
        if not source.is_file():
            parser.error(f"checkpoint not found: {source}")
    if sha256(sources["no_dr"]) == sha256(sources["dr"]):
        parser.error("no-DR and DR inputs are identical; select the checkpoints from different run folders")

    import ultralytics
    from ultralytics import YOLOWorld

    if ultralytics.__version__ != "8.3.253":
        parser.error(f"use ultralytics==8.3.253 to match training; found {ultralytics.__version__}")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.zero_shot is None:
        # Ultralytics downloads its official model if the filename is not present.
        original = YOLOWorld("yolov8m-worldv2.pt")
        sources["zero_shot"] = Path(original.ckpt_path).resolve()
        del original
    else:
        sources["zero_shot"] = args.zero_shot.expanduser().resolve()
    if not sources["zero_shot"].is_file():
        parser.error(f"original checkpoint not found: {sources['zero_shot']}")
    if sha256(sources["zero_shot"]) != PRETRAINED_SHA256:
        parser.error("original checkpoint differs from the one used in the three-model comparison")

    manifest = {
        "experiment": "worldm_seed0_v1", "ultralytics": ultralytics.__version__,
        "class_name": LABEL, "checkpoint_selection": "last.pt after 50 epochs for both fine-tuned models",
        "export": {"format": "onnx", "imgsz": 640, "batch": 1, "device": "cpu", "half": False,
                   "dynamic": False, "simplify": False, "nms": False, "opset": 17},
        "models": {},
    }
    for group in ("zero_shot", "no_dr", "dr"):
        source = sources[group]
        # Work on a copy: export must not write files onto the user's Desktop or into the training run.
        with tempfile.TemporaryDirectory(prefix=f"export-{group}-", dir=out) as temporary:
            staged = Path(temporary) / f"{group}.pt"
            shutil.copy2(source, staged)
            model = YOLOWorld(str(staged))
            if group == "zero_shot":
                model.set_classes([LABEL])
            if model.names != {0: LABEL}:
                raise ValueError(f"{group} has unexpected classes {model.names}; fine-tuned classes must be preserved")
            exported = Path(model.export(**manifest["export"]))
            contract = validate_export(exported)
            destination = out / f"{group}.onnx"
            shutil.copy2(exported, destination)
            manifest["models"][group] = {
                "checkpoint_file": source.name, "checkpoint_sha256": sha256(source),
                "onnx_file": destination.name, "onnx_sha256": sha256(destination), **contract,
            }
            del model
        print(f"Validated {group}: {destination}", flush=True)
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
