"""Download only public pinned model releases; never AI Hub training data.

Example: python scripts/fetch_unified_models.py --output ../../work/ai-model-downloads
The 484 MB ASR weights are downloaded at runtime/build preparation, not put in Git.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile

RELEASES = (
    ("kiosk_rec_v2-paddle.zip", "ocr-kiosk-rec-v2",
     "e14ac5d92849b4e1d542c523882c3f010879420ce29248d8c92cf3faf9561326", 10 * 1024 * 1024),
    ("whisper-elder-v3-ct2.zip", "asr-whisper-elder-v3",
     "d6d5b3c3efbc7be6b9d6585fca8155e4418fa00f443eeb9aff464075da37c3ea", 500 * 1024 * 1024),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, checksum, max_bytes):
    if target.is_file() and sha256(target) == checksum:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as file:
            temporary = Path(file.name)
            request = urllib.request.Request(url, headers={"User-Agent": "sonkkeut-model-installer/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                total = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("Model asset exceeded its pinned size limit")
                    file.write(block)
        if sha256(temporary) != checksum:
            raise ValueError("Model SHA256 mismatch; refusing to install")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def safe_extract(archive, output):
    with zipfile.ZipFile(archive) as zipped:
        for item in zipped.infolist():
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or ":" in item.filename:
                raise ValueError("Unsafe model archive path")
            if ((item.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Model archives may not contain symbolic links")
        zipped.extractall(output)


def fetch(output: Path, ocr_only=False):
    output.mkdir(parents=True, exist_ok=True)
    for filename, tag, checksum, size_limit in RELEASES[:1 if ocr_only else 2]:
        asset = output / filename
        url = f"https://github.com/fingertip-vision/sonkkeut-ai/releases/download/{tag}/{filename}"
        download(url, asset, checksum, size_limit)
        safe_extract(asset, output)
        print(f"verified {filename}: {checksum}")
    if not ocr_only:
        # Same frozen Whisper small tokenizer. Team encoder weights remain unchanged.
        download("https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json",
                 output / "whisper-elder-v3/tokenizer.json",
                 "27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566", 5 * 1024 * 1024)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ocr-only", action="store_true")
    args = ap.parse_args()
    fetch(args.output, args.ocr_only)
