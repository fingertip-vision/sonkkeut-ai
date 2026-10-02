"""화면 표시용 그리기 (데모·디버그 전용, 앱에는 들어가지 않음)"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = {"tab": (255, 140, 0), "menu": (0, 170, 0), "price": (200, 0, 200), "button": (0, 120, 255),
          "back": (0, 0, 220)}
_FONT_CANDIDATES = ["C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/System/Library/Fonts/AppleSDGothicNeo.ttc"]
_font_cache = {}


def _font(size):
    if size not in _font_cache:
        path = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
        _font_cache[size] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return _font_cache[size]


def put_text(img, text, xy, size=28, color=(255, 255, 255), bg=(0, 0, 0)):
    """한글 글자 쓰기 (cv2.putText는 한글을 못 쓴다)"""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    f = _font(size)
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=f)
    if bg is not None:
        d.rectangle((l - 6, t - 4, r + 6, b + 4), fill=bg[::-1])
    d.text((x, y), text, font=f, fill=color[::-1])
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


def draw_frame(img, res, elements=(), target_id=None, last_speak=None):
    """카메라 프레임 위에 화면 사각형·요소·손끝·안내를 그린다"""
    vis = img.copy()
    plane = res.plane
    if plane is not None:
        col = (0, 220, 0) if plane.source == "detect" else (0, 200, 255)
        cv2.polylines(vis, [plane.corners.astype(np.int32)], True, col, 3)
        for e in elements:
            x1, y1, x2, y2 = e.box
            quad = plane.to_image([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).astype(np.int32)
            thick = 5 if e.id == target_id else 2
            c = (0, 0, 255) if e.id == target_id else COLORS.get(e.kind, (200, 200, 200))
            cv2.polylines(vis, [quad], True, c, thick)
            if e.id == target_id or e.kind in ("menu", "button"):
                cv2.putText(vis, e.id, tuple(quad[0] + [3, 16]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    tip = res.tip
    if tip is not None and tip.image_px is not None:
        cv2.circle(vis, tuple(int(v) for v in tip.image_px), 12, (0, 0, 255), 3)
        if plane is not None and tip.pos is not None:
            p = plane.to_image([tip.pos])[0]
            cv2.circle(vis, tuple(int(v) for v in p), 5, (255, 255, 0), -1)
    lines = []
    if res.hint:
        lines.append(res.hint)
    ev = res.event
    if ev is not None:
        lines.append(f"[{ev.type}] {ev.dir or ''} {ev.distance or ''}  진동 {ev.vibe_hz:.0f}Hz")
    if last_speak:
        lines.append(f"음성: {last_speak}")
    if res.verdict is not None:
        lines.append(f"결과: {res.verdict.result} - {res.verdict.speak}")
    ms = res.timings.get("total_ms")
    if ms is not None:
        lines.append(f"{ms:.0f} ms  ({res.timings.get('plane_mode', '-')})" + ("  키프레임" if res.keyframe else ""))
    for i, s in enumerate(lines):
        put_text(vis, s, (12, 12 + 40 * i), size=26)
    return vis
