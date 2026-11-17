#!/usr/bin/env python3
"""Download the published World ONNX models and example image; only Python's standard library is needed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
import zipfile


HERE = Path(__file__).resolve().parent
CATALOG = HERE / "vision-worldm-20261117/release_assets.json"
DEFAULT_OUT = HERE.parents[1] / "artifacts/vision/worldm_seed0_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path, digest):
    return path.is_file() and sha256(path) == digest


def download(url, destination):
    request = Request(url, headers={"User-Agent": "MicroDuck-Vision-Downloader"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as target:
        shutil.copyfileobj(response, target)


def install(group, entry, common, out, overwrite=False, fetch=download):
    """Verify the archive and selected members before replacing any installed files."""
    out = Path(out)
    model_name = f"{group}.onnx"
    expected = {model_name: entry["onnx_sha256"], **common}
    if all(matches(out / name, digest) for name, digest in expected.items()):
        print(f"Already verified: {group}", flush=True)
        return
    for name, digest in expected.items():
        path = out / name
        if path.exists() and not matches(path, digest) and not overwrite:
            raise ValueError(f"{path} differs from this release; choose another --out or use --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"download-{group}-", dir=out) as temporary:
        stage = Path(temporary)
        archive_path = stage / "download.zip"
        print(f"Downloading: {group}", flush=True)
        fetch(entry["url"], archive_path)
        if archive_path.stat().st_size != entry["archive_size"] or sha256(archive_path) != entry["archive_sha256"]:
            raise ValueError(f"{group}: archive size/SHA256 mismatch")
        with zipfile.ZipFile(archive_path) as archive:
            if archive.getinfo(entry["member"]).file_size != entry["onnx_size"]:
                raise ValueError(f"{group}: unexpected ONNX size")
            # Read known members explicitly: do not extract/apply the old bundled code patch.
            for name, digest in expected.items():
                member = entry["member"] if name == model_name else name
                staged_file = stage / name
                with archive.open(member) as source, staged_file.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                if sha256(staged_file) != digest:
                    raise ValueError(f"{group}: {name} SHA256 mismatch")
        for name in expected:
            (stage / name).replace(out / name)
    print(f"Installed and verified: {out / model_name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "zero_shot", "no_dr", "dr"), default="all")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--overwrite", action="store_true", help="replace local files that differ from this release")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text())
    groups = list(catalog["models"]) if args.group == "all" else [args.group]
    try:
        for group in groups:
            install(group, catalog["models"][group], catalog["common_sha256"],
                    args.out.expanduser().resolve(), overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
