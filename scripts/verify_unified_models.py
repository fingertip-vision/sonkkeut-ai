"""Actual team model inference checks using only bundled synthetic OCR fixtures.

Pass --asr and --audio to additionally transcribe an explicit synthetic WAV.
No microphone recording is initiated, and nothing is downloaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sonkkeut_ai.ocr import KioskRecognizer
from sonkkeut_ai.runtime import UnifiedRuntime
from sonkkeut_ai.speech import WhisperSpeech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", type=Path)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    assets = ROOT / "android/react-native-sonkkeut/android/src/main/assets/sonkkeut"
    model = assets / "m3_kiosk_rec_v2.onnx"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    metadata = json.loads((assets / "m3_kiosk_rec_v2_metadata.json").read_text(encoding="utf-8"))
    assert digest == metadata["sha256"], "OCR model differs from its exported manifest"
    recognizer = KioskRecognizer(model, assets / "m3_kiosk_rec_v2_dictionary.txt")
    result = {"ocr_model_sha256": digest, "ocr": [], "data_kind": "synthetic_test_fixtures"}
    manifest = json.loads((ROOT / "tests/fixtures/ocr/manifest.json").read_text(encoding="utf-8"))
    for fixture in manifest:
        image = cv2.imread(str(ROOT / "tests/fixtures/ocr" / fixture["image"]))
        t = time.perf_counter()
        text, confidence = recognizer.read_line(image)
        assert text == fixture["expected"], (fixture["expected"], text)
        result["ocr"].append({"expected": fixture["expected"], "text": text, "confidence": confidence,
                              "ms": round((time.perf_counter() - t) * 1000)})
    if bool(args.asr) != bool(args.audio):
        ap.error("--asr and --audio must be supplied together")
    if args.asr:
        runtime = UnifiedRuntime(recognizer, WhisperSpeech(args.asr))
        t = time.perf_counter()
        result["speech"] = runtime.transcribe(str(args.audio), ["아메리카노", "카페라떼"])
        result["speech_ms"] = round((time.perf_counter() - t) * 1000)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
