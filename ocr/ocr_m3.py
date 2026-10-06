"""M3 문자 인식 (F-04) — 손끝길.

F-03가 준 요소 목록(좌표 0~1)과 펼친 화면 이미지를 받아
요소별 텍스트·가격·신뢰도를 채운다.

처리 (명세 F-04)
  ① 영역별 OCR      : 화면 전체를 한 번 검출+인식한 뒤 줄을 요소 박스에 배정
                      (요소마다 따로 돌리는 read_crop 도 제공)
  ② 가격 정규화      : "4,500원" → 4500
  ③ 메뉴 사전 보정   : 매장 메뉴 사전이 있으면 menu 요소의 메뉴명을 가장 가까운 항목으로
예외: 신뢰도 기준 미만이면 uncertain=True (각도 변경 안내·3회 실패 처리는 앱 상태 쪽 몫)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

DET_MODEL = "PP-OCRv5_mobile_det"
REC_MODEL = "korean_PP-OCRv5_mobile_rec"

# ₩ 는 OCR에서 W·# 로 자주 읽힌다
_PRICE_RE = re.compile(r"^[+＋]?\s*[₩W#\\]?\s*[\dOoIl,.\s]{2,}\s*(원|won)?$", re.I)
_PRICE_IN_RE = re.compile(r"[₩W#\\]?\s*\d[\d,.]{2,}\s*원?")
_QTY_RE = re.compile(r"(?<=[가-힣\s)])\s*[xX×]\s*([\dlI|]{1,2})\s*$")  # 'x1' 의 1 은 l·I 로도 읽힌다
_DIGIT_FIX =str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})


def normalize_price(text: str) -> int | None:
    """가격 문자열을 원 단위 정수로. 가격이 아니면 None. '+500' 같은 옵션 추가금도 500."""
    t = text.strip().replace(" ", "").rstrip(",.")
    if not _PRICE_RE.match(t) or not re.search(r"\d", t):
        return None
    digits = re.sub(r"[^\d]", "", t.translate(_DIGIT_FIX))
    if not digits:
        return None
    v = int(digits)
    return v if 100 <= v <= 1_000_000 else None


def find_price(text: str) -> int | None:
    """'합계 28,500원'처럼 글자와 섞인 줄에서 가격만 뽑는다."""
    for m in _PRICE_IN_RE.finditer(text):
        v = normalize_price(m.group())
        if v is not None:
            return v
    return None


def reading_order(lines: list["Line"]) -> list["Line"]:
    """세로로 겹치는 줄끼리 한 행으로 묶고, 위→아래, 왼쪽→오른쪽."""
    rows: list[list[Line]] = []
    for l in sorted(lines, key=lambda l: l.center[1]):
        if rows:
            r = rows[-1]
            top, bot = min(x.box[1] for x in r), max(x.box[3] for x in r)
            ov = min(bot, l.box[3]) - max(top, l.box[1])
            if ov > 0.5 * min(bot - top, l.box[3] - l.box[1]):
                r.append(l)
                continue
        rows.append([l])
    return [l for r in rows for l in sorted(r, key=lambda l: l.box[0])]


@dataclass
class Line:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # 0~1, x1 y1 x2 y2

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


class KioskOCR:
    def __init__(self, menu_dict: list[str] | None = None, conf_thresh: float = 0.80,
                 fuzzy_cutoff: float = 80.0, device: str = "gpu:0", rec_model_dir: str | None = None):
        """rec_model_dir: 파인튜닝 후 export 한 인식 모델 폴더 (없으면 공식 가중치)."""
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(text_detection_model_name=DET_MODEL, text_recognition_model_name=REC_MODEL,
                             text_recognition_model_dir=rec_model_dir,
                             use_doc_orientation_classify=False, use_doc_unwarping=False,
                             use_textline_orientation=False, device=device)
        from paddleocr import TextRecognition
        self.rec = TextRecognition(model_name=REC_MODEL, model_dir=rec_model_dir, device=device)
        self.menu_dict = menu_dict or []
        self.conf_thresh = conf_thresh
        self.fuzzy_cutoff = fuzzy_cutoff

    # ---- 저수준 ----
    def lines(self, img: np.ndarray) -> list[Line]:
        """BGR 이미지 → 텍스트 줄 목록 (좌표 0~1)."""
        h, w = img.shape[:2]
        r = self.ocr.predict(img)[0]
        out = []
        for t, s, b in zip(r["rec_texts"], r["rec_scores"], r["rec_boxes"]):
            if not t.strip():
                continue
            x1, y1, x2, y2 = [float(v) for v in b]
            out.append(Line(t.strip(), float(s), (x1 / w, y1 / h, x2 / w, y2 / h)))
        return out

    def correct_menu(self, text: str) -> tuple[str, float | None]:
        if not self.menu_dict or not text:
            return text, None
        from rapidfuzz import fuzz, process
        key = text.replace(" ", "")
        hit = process.extractOne(key, {m: m.replace(" ", "") for m in self.menu_dict},
                                 scorer=fuzz.ratio, score_cutoff=self.fuzzy_cutoff)
        return (hit[2], hit[1]) if hit else (text, None)

    # ---- 요소 단위 ----
    def _fill(self, el: dict, ls: list[Line]) -> dict:
        ls = reading_order(ls)
        prices = [(normalize_price(l.text), l) for l in ls]
        price_lines = [l for p, l in prices if p is not None]
        text_lines = [l for p, l in prices if p is None]
        # 버튼·탭 문구가 숫자뿐이면(예: 수량 "2") 가격이 아니라 텍스트로 둔다
        if not text_lines and el.get("kind") != "price":
            text_lines, price_lines = price_lines, []
        text = " ".join(l.text for l in text_lines)
        used = text_lines + price_lines
        conf = min((l.conf for l in used), default=0.0)
        out = dict(el, text=text, conf_ocr=round(conf, 4), raw=[l.text for l in used])
        if price_lines:
            out["price"] = normalize_price(price_lines[0].text)
        elif el.get("kind") == "price":
            out["price"] = find_price(text)
        # 장바구니 '초코라떼 x3' — x 는 X·× 로도 읽힌다. 수량은 F-07 이 쓰도록 qty 로 뺀다
        m = _QTY_RE.search(text)
        if m:
            text, out["qty"] = text[:m.start()].strip(), int(m.group(1).translate(_DIGIT_FIX).replace("|", "1"))
        if el.get("kind") in ("menu", "cart_item"):
            fixed, score = self.correct_menu(text)
            if score is not None:
                text, out["dict_score"] = fixed, round(score, 1)
        out["text"] = f"{text} x{out['qty']}" if m else text
        out["uncertain"] = (not used) or conf < self.conf_thresh
        return out

    def read_screen(self, img: np.ndarray, elements: list[dict]) -> list[dict]:
        """펼친 화면 전체 + F-03 요소 목록 → 텍스트가 채워진 요소 목록.
        줄 중심이 들어가는 요소 중 가장 작은 박스에 배정한다(카드 안 버튼 등 중첩 대비)."""
        ls = self.lines(img)
        buckets: dict[int, list[Line]] = {i: [] for i in range(len(elements))}
        for l in ls:
            cx, cy = l.center
            inside = [i for i, e in enumerate(elements)
                      if e["box"][0] <= cx <= e["box"][2] and e["box"][1] <= cy <= e["box"][3]]
            if inside:
                i = min(inside, key=lambda i: (elements[i]["box"][2] - elements[i]["box"][0]) *
                                              (elements[i]["box"][3] - elements[i]["box"][1]))
                buckets[i].append(l)
        self._fallback(img, elements, buckets)
        return [self._fill(e, buckets[i]) for i, e in enumerate(elements)]

    def _fallback(self, img, elements, buckets, min_score=0.5):
        """검출기는 '+', '-', '2', '티' 같은 한 글자를 자주 놓친다 → 빈 요소는 박스를 잘라 인식기에 바로 넣는다."""
        h, w = img.shape[:2]
        idx, crops = [], []
        for i, e in enumerate(elements):
            if buckets[i] or e.get("kind") == "menu":
                continue
            x1, y1, x2, y2 = e["box"]
            c = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
            if c.size and min(c.shape[:2]) >= 6:
                idx.append(i), crops.append(c)
        if not crops:
            return
        for i, r in zip(idx, self.rec.predict(crops, batch_size=len(crops))):
            t, sc = r["rec_text"].strip(), float(r["rec_score"])
            if t and sc >= min_score:
                buckets[i].append(Line(t, sc, tuple(elements[i]["box"])))

    def read_crop(self, crop: np.ndarray, el: dict | None = None) -> dict:
        """요소 영역 이미지 하나만 읽는다(요소별 OCR 방식)."""
        return self._fill(el or {"kind": "unknown", "box": [0, 0, 1, 1]}, self.lines(crop))
