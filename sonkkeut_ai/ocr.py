"""CPU ONNX adapter for the team's actual PP-OCRv5 kiosk recognition v2.

M2 provides the regions. This adapter reads those regions directly, rather than
claiming that a recognition-only model is a text detector. Native Android uses
the same preprocessing and CTC dictionary exported by export_unified_ocr.py.
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from ocr.ocr_m3 import KioskOCR, Line


def decode_ctc(scores: np.ndarray, characters: list[str]) -> tuple[str, float]:
    """Greedy CTC: blank index 0, suppress only adjacent repeated token IDs."""
    scores = np.asarray(scores)
    if scores.ndim == 3:
        scores = scores[0]
    if scores.ndim != 2 or scores.shape[1] != len(characters):
        raise ValueError("OCR output and exported character dictionary do not match")
    ids = scores.argmax(-1)
    selected = [i for i, token in enumerate(ids) if token != 0 and (i == 0 or token != ids[i - 1])]
    text = "".join(characters[int(ids[i])] for i in selected)
    confidence = float(np.mean(scores[selected, ids[selected]])) if selected else 0.0
    return unicodedata.normalize("NFC", text).strip(), confidence


def preprocess(crop: np.ndarray, height: int = 48, min_width: int = 320, max_width: int = 3200) -> np.ndarray:
    """BGR NCHW, keep aspect, normalize /255 -> (x-.5)/.5, right pad zero."""
    if crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) < 1:
        raise ValueError("OCR requires a nonempty BGR image")
    h, w = crop.shape[:2]
    width = min(max_width, max(min_width, math.ceil(height * w / h)))
    resized_width = min(width, math.ceil(height * w / h))
    img = cv2.resize(crop, (resized_width, height)).astype(np.float32) / 255.0
    normalized = (img.transpose(2, 0, 1) - 0.5) / 0.5
    padded = np.zeros((3, height, width), dtype=np.float32)
    padded[:, :, :resized_width] = normalized
    return padded[None]


class KioskRecognizer:
    model_id = "ocr-kiosk-rec-v2-onnx"

    def __init__(self, model_path: str | Path, dictionary_path: str | Path, menu_dict=None,
                 conf_thresh: float = 0.80, threads: int = 2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        dictionary_path = Path(dictionary_path)
        if dictionary_path.suffix == ".txt":
            dictionary = json.loads(dictionary_path.with_name("m3_kiosk_rec_v2_metadata.json").read_text(encoding="utf-8"))
            self.characters = [""] + dictionary_path.read_text(encoding="utf-8").splitlines()
        else:
            dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
            self.characters = dictionary["characters"]
        self.height = dictionary["preprocess"]["height"]
        self.min_width = dictionary["preprocess"]["min_width"]
        self.max_width = dictionary["preprocess"]["max_width"]
        # Reuse the team's tested price, quantity and menu-dictionary postprocessing.
        # Do not construct PaddleOCR, since ONNX has already exported those weights.
        self.post = object.__new__(KioskOCR)
        self.post.menu_dict = menu_dict or []
        self.post.conf_thresh = conf_thresh
        self.post.fuzzy_cutoff = 80.0

    def read_line(self, crop: np.ndarray) -> tuple[str, float]:
        data = preprocess(crop, self.height, self.min_width, self.max_width)
        output = self.session.run(None, {self.input_name: data})[0]
        return decode_ctc(output, self.characters)

    @staticmethod
    def _line_bands(crop: np.ndarray) -> list[tuple[int, int]]:
        """Split clearly separated text rows; avoid merging name and price in a menu card.

        This is a region heuristic, not a substitute for the original text detector.
        It handles high contrast kiosk cards and falls back to reading the whole region.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if float((mask > 0).mean()) > 0.5:
            mask = 255 - mask
        hits = (mask > 0).sum(axis=1) >= max(2, crop.shape[1] * 0.008)
        # Join tiny blank gaps inside Korean glyphs before finding distinct rows.
        hits = cv2.morphologyEx(hits.astype(np.uint8)[:, None], cv2.MORPH_CLOSE,
                                np.ones((7, 1), np.uint8))[:, 0] > 0
        bands, start = [], None
        for y, active in enumerate(np.r_[hits, False]):
            if active and start is None:
                start = y
            elif not active and start is not None:
                if y - start >= 5:
                    bands.append((max(0, start - 3), min(crop.shape[0], y + 3)))
                start = None
        return bands if 1 <= len(bands) <= 6 else [(0, crop.shape[0])]

    def read_screen(self, image: np.ndarray, elements: list[dict]) -> list[dict]:
        """Image excludes flatten() padding; boxes are normalized to that image."""
        h, w = image.shape[:2]
        output = []
        for element in elements:
            x1, y1, x2, y2 = element["box"]
            x1, y1 = max(0, math.floor(x1 * w)), max(0, math.floor(y1 * h))
            x2, y2 = min(w, math.ceil(x2 * w)), min(h, math.ceil(y2 * h))
            crop = image[y1:y2, x1:x2]
            lines = []
            if crop.size and min(crop.shape[:2]) >= 6:
                for top, bottom in self._line_bands(crop):
                    text, score = self.read_line(crop[top:bottom])
                    if text:
                        lines.append(Line(text, score, (x1 / w, (y1 + top) / h, x2 / w, (y1 + bottom) / h)))
            filled = self.post._fill(element, lines)
            filled["ocr_provider"] = self.model_id
            # A low OCR result may not retain M2's high button confidence and allow a press.
            filled["conf"] = min(float(element.get("conf", 1.0)), filled["conf_ocr"])
            output.append(filled)
        return output
