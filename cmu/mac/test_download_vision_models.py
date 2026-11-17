"""Offline integrity and installation checks for the model release downloader."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_vision_models as downloader  # noqa: E402


def package(model=b"onnx-model", expected_model=b"onnx-model"):
    common_files = {"manifest.json": b'{"experiment":"test"}', "example_frame.jpg": b"example-image"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("models/dr.onnx", model)
        for name, content in common_files.items():
            archive.writestr(name, content)
        archive.writestr("deployment.patch", b"old patch must not be installed")
        archive.writestr("../outside.txt", b"must not be extracted")
    data = buffer.getvalue()
    entry = {"url": "https://example.invalid/test.zip", "archive_size": len(data),
             "archive_sha256": hashlib.sha256(data).hexdigest(), "member": "models/dr.onnx",
             "onnx_size": len(model), "onnx_sha256": hashlib.sha256(expected_model).hexdigest()}
    common = {name: hashlib.sha256(content).hexdigest() for name, content in common_files.items()}

    def fetch(url, destination):
        destination.write_bytes(data)

    return entry, common, fetch


def test_install_only_verified_model_and_common_files(tmp_path):
    entry, common, fetch = package()
    out = tmp_path / "models"
    downloader.install("dr", entry, common, out, fetch=fetch)
    assert {p.name for p in out.iterdir()} == {"dr.onnx", "manifest.json", "example_frame.jpg"}
    assert (out / "dr.onnx").read_bytes() == b"onnx-model"
    assert not (tmp_path / "outside.txt").exists()


def test_corrupted_archive_does_not_install_files(tmp_path):
    entry, common, _ = package()

    def corrupted(url, destination):
        destination.write_bytes(b"truncated archive")

    with pytest.raises(ValueError, match="archive size/SHA256 mismatch"):
        downloader.install("dr", entry, common, tmp_path, fetch=corrupted)
    assert not list(tmp_path.iterdir())


def test_invalid_model_digest_preserves_existing_files(tmp_path):
    entry, common, fetch = package(model=b"wrong-model")
    (tmp_path / "dr.onnx").write_bytes(b"existing-model")
    with pytest.raises(ValueError, match="dr.onnx SHA256 mismatch"):
        downloader.install("dr", entry, common, tmp_path, overwrite=True, fetch=fetch)
    assert (tmp_path / "dr.onnx").read_bytes() == b"existing-model"
    assert not (tmp_path / "manifest.json").exists()


def test_differing_local_model_is_not_overwritten_without_flag(tmp_path):
    entry, common, fetch = package()
    (tmp_path / "dr.onnx").write_bytes(b"another-model")
    with pytest.raises(ValueError, match="differs from this release"):
        downloader.install("dr", entry, common, tmp_path, fetch=fetch)
    assert (tmp_path / "dr.onnx").read_bytes() == b"another-model"


def test_repeat_install_uses_verified_cache(tmp_path):
    entry, common, fetch = package()
    downloader.install("dr", entry, common, tmp_path, fetch=fetch)

    def no_network(url, destination):
        raise AssertionError("already verified models must not be downloaded again")

    downloader.install("dr", entry, common, tmp_path, fetch=no_network)
    assert (tmp_path / "dr.onnx").read_bytes() == b"onnx-model"


def test_catalog_matches_frozen_export_hashes():
    catalog = json.loads(downloader.CATALOG.read_text())
    exported = json.loads((downloader.HERE / "vision-worldm-20261117/export_manifest.json").read_text())
    for group, entry in catalog["models"].items():
        assert entry["onnx_sha256"] == exported["models"][group]["onnx_sha256"]
