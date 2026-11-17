#!/usr/bin/env python3
"""Select a World detector for an existing command, without changing the motion controller."""
import argparse
import os
from pathlib import Path
import shlex


DEFAULT_MODELS = Path(__file__).resolve().parents[2] / "artifacts/vision/worldm_seed0_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=("zero_shot", "no_dr", "dr"))
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--conf", type=float, default=0.10,
                        help="starting threshold, not calibrated for these models; keep equal in comparisons")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- followed by an existing offline or runtime command; omit to print exports")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be in (0, 1]")
    model = (args.models_dir.expanduser() / f"{args.group}.onnx").resolve()
    if not model.is_file():
        parser.error(f"model missing: {model}; run cmu/mac/export_learned_target.py first")
    settings = {
        "LEARNED_TARGET_MODEL": str(model),
        "LEARNED_TARGET_PROMPTS": "orange ball",
        "LEARNED_TARGET_CONF": str(args.conf),
    }
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        for key, value in settings.items():
            print(f"export {key}={shlex.quote(value)}")
        return
    os.execvpe(command[0], command, dict(os.environ, **settings))


if __name__ == "__main__":
    main()
