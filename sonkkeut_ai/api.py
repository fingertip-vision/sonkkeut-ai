"""Optional local Python model API; default bind is 127.0.0.1.

Cloudflare Workers cannot execute these Python providers. The Android package
executes the same team models on the device, without needing this API.
"""
from __future__ import annotations

import base64
import binascii
import hmac
import io
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from PIL import Image

from .ocr import KioskRecognizer
from .runtime import UnifiedRuntime
from .speech import WhisperSpeech

MAX_BODY = 8 * 1024 * 1024
Name = Annotated[str, Field(min_length=1, max_length=100)]


class Element(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Annotated[str, Field(min_length=1, max_length=64)]
    kind: Literal["menu", "button", "tab", "back", "price", "cart_item", "text", "title"]
    box: tuple[float, float, float, float]
    conf: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 1.0
    text: Annotated[str, Field(max_length=300)] = ""
    price: Annotated[int, Field(ge=0, le=1_000_000)] | None = None
    qty: Annotated[int, Field(ge=1, le=99)] | None = None
    conf_ocr: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    uncertain: bool = False

    @field_validator("box")
    @classmethod
    def valid_box(cls, box):
        if not all(np.isfinite(v) and 0 <= v <= 1 for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Element boxes must be ordered screen coordinates in [0,1]")
        return box


class Screen(BaseModel):
    model_config = ConfigDict(extra="allow")
    screen_type: Literal["menu", "option", "cart", "payment", "start", "unknown", "method", "other"]
    keyframe_id: Annotated[int, Field(ge=0)] = 0
    elements: Annotated[list[Element], Field(max_length=100)]
    cart: list[dict] = []

    @field_validator("cart")
    @classmethod
    def valid_cart(cls, cart):
        if len(cart) > 100 or any(not isinstance(c.get("qty"), (int, type(None))) or (c.get("qty") is not None and not 1 <= c["qty"] <= 99) for c in cart):
            raise ValueError("Cart rows must have bounded numeric quantities")
        return cart


class OrderInput(BaseModel):
    text: Annotated[str, Field(min_length=1, max_length=4000)]
    screen_menus: Annotated[list[Name], Field(max_length=100)] = []
    store_menus: Annotated[list[Name], Field(max_length=500)] = []


class OCRInput(BaseModel):
    # Image is the already rectified screen; remove flatten() margin first.
    image_base64: Annotated[str, Field(min_length=1, max_length=MAX_BODY)]
    elements: Annotated[list[Element], Field(max_length=100)]
    keyframe_id: Annotated[int, Field(ge=0)] = 0
    previous: Screen | None = None


class Options(BaseModel):
    temp: Literal["hot", "ice"] | None = None
    size: Annotated[str, Field(max_length=100)] | None = None
    extras: Annotated[list[Name], Field(max_length=10)] = []


class Item(BaseModel):
    model_config = ConfigDict(extra="allow")
    menu: Name
    qty: Annotated[int, Field(ge=1, le=99)]
    options: Options = Options()
    status: Literal["pending", "added"] = "pending"


class Intent(BaseModel):
    model_config = ConfigDict(extra="allow")
    items: Annotated[list[Item], Field(max_length=50)]
    dine: Literal["매장", "포장"] | None = None


class PlanInput(BaseModel):
    screen: Screen
    intent: Intent
    confirmed: bool = False
    # Client holds this state; no audio, images or personal orders are persisted.
    progress: "ProgressInput" = Field(default_factory=lambda: ProgressInput())


class ProgressInput(BaseModel):
    current: Annotated[int, Field(ge=0, le=49)] | None = None
    variant_temp: bool = False
    opt_done: Annotated[list[Name], Field(max_length=50)] = []
    qty_pressed: Annotated[int, Field(ge=0, le=98)] = 0
    tabs_tried: Annotated[list[Name], Field(max_length=100)] = []
    pages_moved: Annotated[int, Field(ge=0, le=5)] = 0
    last: dict | None = None
    cur_tab: Name | None = None
    seen: dict[str, Name] = {}
    relook: Annotated[int, Field(ge=0, le=3)] = 0

    @field_validator("last", "seen")
    @classmethod
    def bounded_progress(cls, progress):
        if len(json.dumps(progress)) > 16000:
            raise ValueError("Planner state is too large")
        return progress


PlanInput.model_rebuild()


class ProviderCapability(BaseModel):
    ready: bool
    provider: str


class Capabilities(BaseModel):
    schema_version: Literal["sonkkeut.ai.v1"]
    ocr: ProviderCapability
    asr: ProviderCapability
    nlu: ProviderCapability
    vision: dict[str, str]
    payment: dict[str, bool]


class PlanTarget(BaseModel):
    target_id: str | None
    say: str
    why: str


class OrderResponse(BaseModel):
    schema_version: Literal["sonkkeut.ai.v1"]
    transcript: str
    intent: Intent
    requires_confirmation: bool
    plan: PlanTarget | None = None
    providers: dict[str, str]


class ScreenResponse(BaseModel):
    schema_version: Literal["sonkkeut.ai.v1"]
    structure: Screen
    providers: dict[str, str]


class PlanResponse(BaseModel):
    schema_version: Literal["sonkkeut.ai.v1"]
    plan: PlanTarget | None
    requires_confirmation: bool
    progress: ProgressInput


class InputBoundary:
    """Bound bodies including chunked uploads, and optionally require an API key."""
    def __init__(self, app, api_key=None):
        self.app, self.api_key = app, api_key

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if self.api_key and scope["method"] != "OPTIONS":
            supplied = headers.get(b"x-ai-key", b"").decode("utf-8", errors="replace")
            if not hmac.compare_digest(supplied.encode(), self.api_key.encode()):
                return await self.reject(send, 401, "AI API key required")
        chunks, count = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            count += len(message.get("body", b""))
            if count > MAX_BODY:
                return await self.reject(send, 413, "AI request exceeds 8 MiB")
            chunks.append(message)
            if not message.get("more_body", False):
                break
        index = 0

        async def replay():
            nonlocal index
            if index < len(chunks):
                message = chunks[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay, send)

    @staticmethod
    async def reject(send, status, detail):
        body = json.dumps({"detail": detail}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})


def decode_screen(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 6 * 1024 * 1024:
            raise ValueError("Image exceeds 6 MiB")
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > 2_000_000 or min(image.size) < 6:
                raise ValueError("Screen must contain 6..2,000,000 pixels")
            rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (binascii.Error, ValueError, OSError) as exc:
        raise HTTPException(422, "Invalid or oversized screen image") from exc


def configured_runtime():
    assets = Path(__file__).resolve().parents[1] / "android/react-native-sonkkeut/android/src/main/assets/sonkkeut"
    ocr_model = Path(os.environ.get("SONKKEUT_OCR_MODEL", assets / "m3_kiosk_rec_v2.onnx"))
    dictionary = Path(os.environ.get("SONKKEUT_OCR_DICTIONARY", assets / "m3_kiosk_rec_v2_dictionary.txt"))
    recognizer = KioskRecognizer(ocr_model, dictionary) if ocr_model.is_file() and dictionary.is_file() else None
    speech_path = os.environ.get("SONKKEUT_ASR_MODEL")
    speech = WhisperSpeech(speech_path) if speech_path else None
    return UnifiedRuntime(recognizer, speech)


def decode_bounded_audio(file):
    """Stop compressed audio at 30 seconds, before allocating an unbounded PCM array."""
    import av
    chunks, count = [], 0
    with av.open(str(file), mode="r", metadata_errors="ignore") as container:
        if not container.streams.audio:
            raise ValueError("No audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            frame.pts = None
            for chunk in resampler.resample(frame):
                values = chunk.to_ndarray().ravel()
                count += len(values)
                if count > 480000:
                    raise ValueError("Audio exceeds 30 seconds")
                chunks.append(values)
        for chunk in resampler.resample(None):
            values = chunk.to_ndarray().ravel()
            count += len(values)
            if count > 480000:
                raise ValueError("Audio exceeds 30 seconds")
            chunks.append(values)
    if count < 800:
        raise ValueError("Audio is too short")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def create_app(runtime=None, api_key=None, origins=None):
    app = FastAPI(title="손끝길 unified AI", version="1.0.0")
    runtime = runtime or UnifiedRuntime()
    lock = threading.Lock()
    app.state.ai = runtime
    app.add_middleware(CORSMiddleware, allow_origins=origins or [], allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-AI-Key"])
    app.add_middleware(InputBoundary, api_key=api_key)

    @app.middleware("http")
    async def no_persistence(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/v1/ai/capabilities", response_model=Capabilities)
    def capabilities():
        return runtime.capabilities()

    @app.post("/v1/ai/order", response_model=OrderResponse, response_model_exclude_unset=True)
    def order(data: OrderInput):
        return runtime.order(data.text, data.screen_menus, data.store_menus)

    @app.post("/v1/ai/screen", response_model=ScreenResponse, response_model_exclude_unset=True)
    def screen(data: OCRInput):
        if runtime.recognizer is None:
            raise HTTPException(503, "Trained OCR is not configured")
        image = decode_screen(data.image_base64)
        with lock:
            return runtime.read_screen(image, [e.model_dump(exclude_none=True) for e in data.elements],
                                       data.keyframe_id, data.previous.model_dump(exclude_none=True) if data.previous else None)

    @app.post("/v1/ai/speech", response_model=OrderResponse, response_model_exclude_unset=True)
    async def speech(audio: UploadFile = File(...), menus: str = Form("[]"), store_menus: str = Form("[]")):
        if runtime.speech is None:
            raise HTTPException(503, "Trained Whisper is not configured")
        try:
            names = OrderInput(text="audio", screen_menus=json.loads(menus), store_menus=json.loads(store_menus))
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, "menus/store_menus must be JSON arrays of menu names") from exc
        try:
            # Uploaded bytes stay in a random temporary file only while decoding;
            # ignore the submitted filename and delete on both success and failure.
            with tempfile.TemporaryDirectory(prefix="sonkkeut-audio-") as directory:
                file = Path(directory) / "input.audio"
                raw = await audio.read(MAX_BODY + 1)
                if len(raw) > MAX_BODY:
                    raise HTTPException(413, "Audio too large")
                file.write_bytes(raw)
                samples = decode_bounded_audio(file)
                # A synchronous CPU provider is run outside the event loop.
                from starlette.concurrency import run_in_threadpool
                def infer():
                    with lock:
                        return runtime.transcribe(samples, names.screen_menus, names.store_menus)
                return await run_in_threadpool(infer)
        except HTTPException:
            raise
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(422, "Audio cannot be decoded") from exc
        finally:
            await audio.close()

    @app.post("/v1/ai/plan", response_model=PlanResponse)
    def next_step(data: PlanInput):
        return runtime.next_step(data.screen.model_dump(exclude_none=True), data.intent.model_dump(exclude_none=True),
                                 data.progress.model_dump(), data.confirmed)

    return app


def app_factory():
    return create_app(configured_runtime(), os.environ.get("SONKKEUT_AI_KEY"),
                      [s.strip() for s in os.environ.get("SONKKEUT_AI_ORIGINS", "").split(",") if s.strip()])
