"""One frontend contract for M1/M2/M4 vision, M3 OCR and M5 voice orders."""
from __future__ import annotations

import copy
from dataclasses import asdict, fields

from core.order_nlu import parse
from core.planner import Progress, plan
from core.screen_struct import structure
from . import CONTRACT_VERSION


def frontend_structure(elements: list[dict], keyframe_id=0, previous: dict | None = None) -> dict:
    """Keep F10/RN fields, while retaining the newer core's cart/diff/OCR evidence."""
    if previous is not None:
        previous = {**previous, "cart": previous.get("cart", []), "total": previous.get("total", previous.get("total_price"))}
    current = structure(elements, keyframe_id, previous)
    by_id = {e["id"]: e for e in elements}
    for element in current["elements"]:
        source = by_id.get(element["id"], {})
        for key in ("conf_ocr", "raw", "dict_score", "ocr_provider", "parent"):
            if key in source:
                element[key] = source[key]
    current["total_price"] = current["total"]
    if current["cart"]:
        current["cart_count"] = sum(c["qty"] or 1 for c in current["cart"])
    elif any(e.get("qty") for e in elements):
        current["cart_count"] = sum(e.get("qty", 0) for e in elements if e.get("kind") == "cart_item")
    if current["screen_type"] == "other":
        current["screen_type"] = "start" if current.get("hint") == "start" else "unknown"
    # A single explicit, confidently read pair is required; vague dine-in wording
    # or a partial OCR result must not guess the method screen.
    choices = {e.get("text", "").replace(" ", "") for e in elements
               if e.get("kind") == "button" and not e.get("uncertain")
               and float(e.get("conf_ocr", e.get("conf", 0))) >= .8}
    if {"매장", "포장"}.issubset(choices) and not any(e.get("kind") == "menu" for e in elements):
        current["screen_type"] = "method"
    current["selected"] = [e["id"] for e in elements if "선택됨" in e.get("text", "")]
    return current


class OCRStructure:
    """Drop-in VisionPipeline.structure_fn for the trained recognizer.

    flatten() allocates an image with a 4% margin around the screen. Removing that
    margin here keeps M2 element and hand guidance coordinates in the same plane.
    """
    def __init__(self, recognizer, margin=0.04):
        self.recognizer = recognizer
        self.margin = margin
        self.previous = None

    def __call__(self, elements, crops, flat, keyframe_id):
        h, w = flat.shape[:2]
        factor = 1 + 2 * self.margin
        mx, my = round(w * self.margin / factor), round(h * self.margin / factor)
        image = flat[my:h - my if my else h, mx:w - mx if mx else w]
        source = [e.to_dict() if hasattr(e, "to_dict") else copy.deepcopy(e) for e in elements]
        enriched = self.recognizer.read_screen(image, source)
        result = frontend_structure(enriched, keyframe_id, self.previous)
        self.previous = result
        return result


def create_vision_pipeline(m1, m2, recognizer, **options):
    """The existing trained M1/M2/fingertip guide uses actual M3 at each keyframe."""
    from sonkkeut_vision.pipeline import VisionPipeline
    return VisionPipeline(m1, m2, structure_fn=OCRStructure(recognizer), **options)


def progress_from_json(value: dict | None) -> Progress:
    keys = {f.name for f in fields(Progress)}
    data = {k: v for k, v in (value or {}).items() if k in keys}
    data["opt_done"] = set(data.get("opt_done") or [])
    return Progress(**data)


def progress_to_json(progress: Progress) -> dict:
    value = asdict(progress)
    value["opt_done"] = sorted(value["opt_done"])
    return value


class UnifiedRuntime:
    def __init__(self, recognizer=None, speech=None):
        self.recognizer, self.speech = recognizer, speech

    def capabilities(self):
        return {"schema_version": CONTRACT_VERSION, "ocr": {
            "ready": self.recognizer is not None,
            "provider": "ocr-kiosk-rec-v2-onnx" if self.recognizer else "unavailable",
        }, "asr": {"ready": self.speech is not None,
            "provider": "asr-whisper-elder-v3-ct2" if self.speech else "unavailable"},
            "nlu": {"ready": True, "provider": "korean-order-rules"},
            "vision": {"provider": "M1/M2/M4-on-device", "transport": "react-native-sonkkeut"},
            "payment": {"automatic": False}}

    def order(self, text, screen_menus, store_menus=None):
        intent = parse(text, screen_menus, store_menus)
        return {"schema_version": CONTRACT_VERSION, "transcript": text,
                "intent": intent, "requires_confirmation": True, "plan": None,
                "providers": {"nlu": "korean-order-rules"}}

    def transcribe(self, audio_path, screen_menus, store_menus=None):
        if self.speech is None:
            raise RuntimeError("The trained Whisper provider has not been configured")
        text = self.speech(audio_path, list(dict.fromkeys(screen_menus + (store_menus or []))))
        result = self.order(text, screen_menus, store_menus)
        result["providers"]["asr"] = "asr-whisper-elder-v3-ct2"
        return result

    def read_screen(self, image, elements, keyframe_id=0, previous=None):
        if self.recognizer is None:
            raise RuntimeError("The trained OCR provider has not been configured")
        enriched = self.recognizer.read_screen(image, elements)
        return {"schema_version": CONTRACT_VERSION,
                "structure": frontend_structure(enriched, keyframe_id, previous),
                "providers": {"ocr": self.recognizer.model_id}}

    @staticmethod
    def next_step(screen, intent, progress=None, confirmed=False):
        pg = progress_from_json(progress)
        if not confirmed or not intent.get("items"):
            return {"schema_version": CONTRACT_VERSION, "plan": None,
                    "requires_confirmation": True, "progress": progress_to_json(pg)}
        screen = copy.deepcopy(screen)
        if pg.current is not None and (not isinstance(pg.current, int) or not 0 <= pg.current < len(intent["items"])):
            pg.current = None
        if screen["screen_type"] in ("unknown", "other"):
            return {"schema_version": CONTRACT_VERSION, "plan": {
                "target_id": None, "say": "화면을 다시 읽어 주세요", "why": "uncertain_screen"},
                "requires_confirmation": False, "progress": progress_to_json(pg)}
        # Only classified start/method screens use the prototype's "other" branch.
        if screen["screen_type"] in ("start", "method"):
            screen["screen_type"] = "other"
        trusted = [e for e in screen["elements"] if not e.get("uncertain") and float(e.get("conf", 1)) >= 0.6]
        screen["elements"] = trusted
        if screen["screen_type"] == "payment":
            target = None
        else:
            target = plan(screen, copy.deepcopy(intent), pg)
        return {"schema_version": CONTRACT_VERSION, "plan": target,
                "requires_confirmation": False, "progress": progress_to_json(pg)}
