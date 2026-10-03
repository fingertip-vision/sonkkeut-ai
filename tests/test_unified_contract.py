import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from sonkkeut_ai.api import create_app, decode_bounded_audio
from sonkkeut_ai.ocr import decode_ctc, preprocess
from sonkkeut_ai.runtime import OCRStructure, UnifiedRuntime, frontend_structure


def element(kind="menu", text="아메리카노", **extra):
    return {"id": "e1", "kind": kind, "text": text, "conf": .95, "box": [0.1, 0.1, .9, .3], **extra}


def test_ctc_blank_separator_and_unicode():
    chars = ["", "ᄀ", "ᅡ", "나", " "]
    rows = np.eye(len(chars))[[1, 1, 2, 0, 2, 3, 4]]
    text, confidence = decode_ctc(rows, chars)
    assert text == "가ᅡ나"
    assert confidence == 1


def test_bgr_preprocessing_pads_after_normalization():
    pixels = np.full((48, 100, 3), 255, np.uint8)
    data = preprocess(pixels)
    assert data.shape == (1, 3, 48, 320)
    assert np.all(data[..., :100] == 1)
    assert np.all(data[..., 100:] == 0)


def test_flatten_margin_does_not_change_element_coordinates():
    class OCR:
        def read_screen(self, image, elements):
            assert image.shape == (100, 200, 3)
            assert elements[0]["box"] == [0.1, 0.1, .9, .3]
            assert np.all(image == 255)
            return elements
    flat = np.zeros((108, 216, 3), np.uint8)
    flat[4:104, 8:208] = 255
    result = OCRStructure(OCR())([element()], {}, flat, 3)
    assert result["keyframe_id"] == 3


def test_no_menu_and_one_menu_unknown_order_do_not_crash():
    runtime = UnifiedRuntime()
    assert runtime.order("피자 주세요", [])["intent"]["source"] == "none"
    result = runtime.order("피자 주세요", ["아메리카노"])
    assert result["requires_confirmation"]
    assert result["plan"] is None


def test_real_order_rules_preserve_per_item_options_and_quantity():
    intent = UnifiedRuntime().order("따뜻한 아메리카노 두 잔하고 카페라떼 한 잔 포장해 주세요", ["아메리카노", "카페라떼"])["intent"]
    assert [(i["menu"], i["qty"]) for i in intent["items"]] == [("아메리카노", 2), ("카페라떼", 1)]
    assert intent["items"][0]["options"] == {"temp": "hot"}
    assert intent["items"][1]["options"] == {}
    assert intent["dine"] == "포장"


def test_frontend_structure_keeps_ocr_cart_and_old_rn_previous():
    result = frontend_structure([element("cart_item", "아메리카노", qty=2, conf_ocr=.86, uncertain=False),
                                 element("price", "합계 9,000원", id="p1", price=9000)], 4,
                                {"screen_type": "menu", "elements": [element()]})
    assert result["cart_count"] == 2
    assert result["total_price"] == 9000
    assert result["elements"][0]["conf_ocr"] == .86


def test_plan_requires_explicit_confirmation_and_trusted_ocr():
    screen = {"screen_type": "menu", "elements": [element(uncertain=True)]}
    intent = {"items": [{"menu": "아메리카노", "qty": 1, "options": {}}]}
    assert UnifiedRuntime.next_step(screen, intent)["plan"] is None
    assert UnifiedRuntime.next_step(screen, intent, confirmed=True)["plan"]["target_id"] is None
    screen["elements"][0]["uncertain"] = False
    assert UnifiedRuntime.next_step(screen, intent, confirmed=True)["plan"]["target_id"] == "e1"


def test_payment_screen_does_not_plan_automatic_payment():
    result = UnifiedRuntime.next_step({"screen_type": "payment", "elements": [element()]},
                                     {"items": [{"menu": "아메리카노", "qty": 1, "options": {}}]}, confirmed=True)
    assert result["plan"] is None


def test_unknown_screen_does_not_guess_a_large_button():
    result = UnifiedRuntime.next_step({"screen_type": "unknown", "elements": [element("button", "결제하기")]},
                                     {"items": [{"menu": "아메리카노", "qty": 1, "options": {}}]}, confirmed=True)
    assert result["plan"]["target_id"] is None
    assert result["plan"]["why"] == "uncertain_screen"


def test_exact_method_pair_and_confirmed_takeaway_choice():
    choices = [element("button", "매장", id="d1", conf_ocr=.95), element("button", "포장", id="d2", conf_ocr=.95)]
    screen = frontend_structure(choices)
    assert screen["screen_type"] == "method"
    intent = {"items": [{"menu": "아메리카노", "qty": 1, "options": {}}], "dine": "포장"}
    assert UnifiedRuntime.next_step(screen, intent, confirmed=True)["plan"]["target_id"] == "d2"
    choices[1]["uncertain"] = True
    assert frontend_structure(choices)["screen_type"] != "method"


def test_http_contract_and_unavailable_provider_are_explicit():
    with TestClient(create_app()) as client:
        caps = client.get("/v1/ai/capabilities").json()
        assert caps["ocr"] == {"ready": False, "provider": "unavailable"}
        reply = client.post("/v1/ai/order", json={"text": "아메리카노 한 잔", "screen_menus": ["아메리카노"]})
        assert reply.status_code == 200
        assert reply.headers["cache-control"] == "no-store"
        assert reply.json()["requires_confirmation"]
        assert client.post("/v1/ai/screen", json={"image_base64": "invalid", "elements": []}).status_code == 503
        assert client.post("/v1/ai/speech", files={"audio": ("anything.wav", b"test")}).status_code == 503


def test_http_rejects_coordinates_invalid_progress_and_large_body():
    with TestClient(create_app()) as client:
        assert client.post("/v1/ai/screen", json={"image_base64": "aa==", "elements": [element(box=[0, 0, 1.5, 1])]}).status_code == 422
        data = {"screen": {"screen_type": "menu", "elements": []}, "intent": {"items": []}, "progress": {"qty_pressed": "wrong"}}
        assert client.post("/v1/ai/plan", json=data).status_code == 422
        assert client.post("/v1/ai/order", content=b"x" * (8 * 1024 * 1024 + 1)).status_code == 413


def test_explicit_api_key_and_cors_origin():
    with TestClient(create_app(api_key="test-ai-contract-key", origins=["https://demo.example"])) as client:
        assert client.get("/v1/ai/capabilities").status_code == 401
        assert client.get("/v1/ai/capabilities", headers={"X-AI-Key": "test-ai-contract-key"}).status_code == 200
        response = client.options("/v1/ai/order", headers={"Origin": "https://demo.example", "Access-Control-Request-Method": "POST"})
        assert response.headers["access-control-allow-origin"] == "https://demo.example"


def test_uncertain_ocr_stops_existing_fingertip_press_guidance():
    from sonkkeut_vision import VisionPipeline
    from sonkkeut_vision.schema import Element, Fingertip
    el = Element("e1", "menu", (.1, .1, .9, .3), .99)
    class Observation:
        margin = .04
        def detect(self, frame, plane):
            return [el], np.zeros((108, 216, 3), np.uint8)
    pipe = VisionPipeline(None, None, hand_source=object(), plane_kwargs={"refiner": None},
                          structure_fn=lambda els, crops, flat, kid: {
                              "elements": [{**els[0].to_dict(), "text": "잘못 읽힌 글자", "uncertain": True}]})
    pipe.elem_det = Observation()
    pipe._read_screen(None, None, 1)
    pipe.set_target("e1")
    for tick in range(10):
        event = pipe.guide.update(Fingertip(pos=(.5, .2), conf=.99), tick * .2)
        assert event.type != "press"
    assert event.type == "hold"


def test_audio_duration_limit_stops_decoding_before_whisper(tmp_path):
    import wave
    import pytest
    file = tmp_path / "31-seconds.wav"
    with wave.open(str(file), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * (16000 * 31))
    with pytest.raises(ValueError, match="30 seconds"):
        decode_bounded_audio(file)
