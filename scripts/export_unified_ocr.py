"""Export the verified team's Paddle PIR OCR v2 to shared ONNX/CTC assets.

Windows known compatible versions: paddlepaddle==3.1.1, paddle2onnx==2.1.0.
Pass --skip-convert to inspect/package an already exported ONNX file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def export(source: Path, output: Path, skip_convert=False):
    import onnx
    import yaml
    output.mkdir(parents=True, exist_ok=True)
    model = output / "m3_kiosk_rec_v2.onnx"
    if not skip_convert:
        subprocess.run([sys.executable, "-m", "paddle2onnx.command", "--model_dir", str(source),
                        "--model_filename", "inference.json", "--params_filename", "inference.pdiparams",
                        "--save_file", str(model), "--opset_version", "17", "--optimize_tool", "None"], check=True)
    graph = onnx.load(model)
    onnx.checker.check_model(graph)
    config = yaml.safe_load((source / "inference.yml").read_text(encoding="utf-8"))
    characters = [""] + config["PostProcess"]["character_dict"] + [" "]
    output_classes = graph.graph.output[0].type.tensor_type.shape.dim[2].dim_value
    if len(characters) != output_classes or graph.graph.node[-1].op_type != "Softmax":
        raise ValueError("CTC dictionary/output probabilities do not match the trained model")
    metadata = {
        "schema_version": "sonkkeut.ocr.v1", "model_id": "ocr-kiosk-rec-v2-onnx",
        "source_release": "https://github.com/fingertip-vision/sonkkeut-ai/releases/tag/ocr-kiosk-rec-v2",
        "source_zip_sha256": "e14ac5d92849b4e1d542c523882c3f010879420ce29248d8c92cf3faf9561326",
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "bytes": model.stat().st_size,
        "input": {"name": graph.graph.input[0].name, "dtype": "float32", "shape": [1, 3, 48, "dynamic_width"]},
        "output": {"name": graph.graph.output[0].name, "dtype": "float32", "shape": [1, "time", len(characters)], "probabilities": True},
        "preprocess": {"channel_order": "BGR", "height": 48, "min_width": 320, "max_width": 3200,
                       "keep_aspect": True, "right_padding_after_normalization": 0,
                       "normalization": "(pixel / 255.0 - 0.5) / 0.5"},
        "ctc": {"blank_index": 0, "collapse_adjacent_repeats": True, "characters_excluding_blank": len(characters) - 1,
                "trailing_space": True, "unicode_normalization": "NFC", "confidence": "mean probability of emitted nonblank tokens"},
        "runtime": "onnxruntime-android 1.19.2 CPU", "exporter": "paddle2onnx 2.1.0 / paddlepaddle 3.1.1 / opset 17",
    }
    (output / "m3_kiosk_rec_v2_dictionary.txt").write_text("\n".join(characters[1:]) + "\n", encoding="utf-8")
    (output / "m3_kiosk_rec_v2_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dictionary = {**metadata, "characters": characters}
    (output / "m3_kiosk_rec_v2_dictionary.json").write_text(json.dumps(dictionary, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": str(model), "sha256": metadata["sha256"], "classes": len(characters)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()
    export(args.source, args.output, args.skip_convert)
