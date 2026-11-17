"""CPU-only integration checks using a tiny synthetic ONNX graph, not trained weights.

Run: python -m pytest -q cmu/mac/test_learned_target.py
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("cv2")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import export_learned_target as exporter  # noqa: E402
import learned_target as detector  # noqa: E402


def fixture_model(path, label="orange ball", output_shape=(1, 5, 3)):
    raw = np.array([[[320, 321, 100], [320, 321, 200], [20, 20, 6], [20, 20, 6],
                     [0.9, 0.8, 0.05]]], dtype=np.float32)
    if output_shape != raw.shape:
        raw = np.zeros(output_shape, np.float32)
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["output"], value=onnx.numpy_helper.from_array(raw))],
        "detector-contract-fixture",
        [onnx.helper.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 640, 640])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, list(raw.shape))],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.helper.set_model_props(model, {"names": repr({0: label})})
    onnx.save(model, path)
    return path


def test_raw_export_through_robot_adapter(tmp_path):
    path = fixture_model(tmp_path / "dr.onnx")
    assert exporter.validate_export(path)["output_shape"] == [1, 5, 3]
    frame = np.zeros((480, 640, 3), np.uint8)
    hit = detector.detect(frame, str(path), conf=0.25, prompts=("orange ball",), mask=True)
    # Removes the 80px letterbox padding, suppresses overlapping boxes and keeps the hit contract.
    assert hit["bbox"] == (310, 230, 20, 20)
    assert hit["n_blobs"] == 1 and hit["label"] == "orange ball"
    assert hit["cx"] == hit["base_cx"] == 320
    assert hit["cy"] == 240 and hit["area"] == 400
    assert np.count_nonzero(hit["mask"]) == 400
    assert detector.detect(frame, str(path), conf=0.95, prompts=("orange ball",)) is None
    assert detector.detect(frame, str(path), prompts=("orange ball",), accept=lambda *box: False) is None
    with pytest.raises(ValueError, match="not baked"):
        detector.detect(frame, str(path), prompts=("small orange ball",))


@pytest.mark.parametrize("label,shape,reason", [
    ("small orange ball", (1, 5, 3), "expected baked names"),
    ("orange ball", (1, 300, 6), "expected raw"),
])
def test_rejects_incompatible_exports(tmp_path, label, shape, reason):
    with pytest.raises(ValueError, match=reason):
        exporter.validate_export(fixture_model(tmp_path / "wrong.onnx", label, shape))


def import_settings(**settings):
    env = dict(os.environ)
    for key in ("LEARNED_TARGET_PROMPTS", "LEARNED_TARGET_CONF"):
        env.pop(key, None)
    env.update(settings)
    code = ("import json,sys; sys.path.insert(0,sys.argv[1]); import learned_target as d; "
            "print(json.dumps([d.PROMPTS,d.CONF]))")
    return subprocess.run([sys.executable, "-c", code, str(HERE)], env=env, capture_output=True, text=True)


def test_old_default_and_world_configuration():
    original = import_settings()
    assert original.returncode == 0, original.stderr
    assert json.loads(original.stdout) == [["small orange ball"], 0.1]
    world = import_settings(LEARNED_TARGET_PROMPTS="orange ball", LEARNED_TARGET_CONF="0.25")
    assert world.returncode == 0, world.stderr
    assert json.loads(world.stdout) == [["orange ball"], 0.25]
    all_classes = import_settings(LEARNED_TARGET_PROMPTS="all")
    assert json.loads(all_classes.stdout) == [None, 0.1]


@pytest.mark.parametrize("settings", [
    {"LEARNED_TARGET_PROMPTS": " | "}, {"LEARNED_TARGET_CONF": "nan"}, {"LEARNED_TARGET_CONF": "-1"},
])
def test_invalid_settings_fail_before_inference(settings):
    result = import_settings(**settings)
    assert result.returncode != 0 and "ValueError" in result.stderr


@pytest.mark.parametrize("group", ["zero_shot", "no_dr", "dr"])
def test_launcher_passes_model_settings_to_child(tmp_path, group):
    fixture_model(tmp_path / f"{group}.onnx")
    code = ("import json,os; print(json.dumps({k:os.environ[k] for k in "
            "['LEARNED_TARGET_MODEL','LEARNED_TARGET_PROMPTS','LEARNED_TARGET_CONF']}))")
    result = subprocess.run([
        sys.executable, str(HERE / "vision_model.py"), "--group", group,
        "--models-dir", str(tmp_path), "--conf", "0.25", "--", sys.executable, "-c", code,
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    settings = json.loads(result.stdout)
    assert settings["LEARNED_TARGET_MODEL"] == str(tmp_path / f"{group}.onnx")
    assert settings["LEARNED_TARGET_PROMPTS"] == "orange ball"
    assert settings["LEARNED_TARGET_CONF"] == "0.25"
