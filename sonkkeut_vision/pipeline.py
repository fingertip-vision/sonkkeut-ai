"""
영상 쪽 전체 흐름 (기능 명세서 1장 그림의 아랫줄 + 윗줄 중 영상 부분)

  매 프레임  : F-02 화면 평면(추적) → F-08 손끝 → 키프레임 판단 → F-09 오차 계산 → F-10 대기
  키프레임만 : F-03 화면 요소 → (언어 쪽 F-04·F-05로 넘김) → 목표 버튼 좌표 갱신 / F-10 판정

언어 쪽(노현석)과는 structure_fn 하나로만 연결된다.
  structure_fn(elements, crops, flat, keyframe_id) -> 화면 구조 dict (6장)
F-04·F-05가 아직 없으면 기본값(elements_to_structure)이 텍스트 없이 요소만 담아 돌려준다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .elements import ElementDetector, flatten, iou
from .fingertip import FingertipTracker
from .guidance import Guide
from .keyframe import KeyframeDetector
from .plane import ScreenPlaneEstimator
from .schema import Element, elements_to_structure
from .verify import PressVerifier


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


@dataclass
class FrameResult:
    plane: object = None
    hint: Optional[str] = None  # 화면을 못 찾았을 때 휴대폰 위치 안내
    tip: object = None
    keyframe: bool = False
    keyframe_id: int = 0
    structure: Optional[dict] = None  # 키프레임에서만 새로 채워짐
    event: object = None  # GuidanceEvent
    verdict: object = None  # Verdict
    target_missing: bool = False  # 새 화면에 목표 버튼이 없음 → F-07이 다시 계획해야 함
    timings: dict = field(default_factory=dict)


class VisionPipeline:
    def __init__(self, m1, m2, hand_source=None, structure_fn=None, device=None, long_side=960,
                 plane_kwargs=None, guide_kwargs=None, keyframe_kwargs=None):
        if hand_source is None:
            from .fingertip import MediaPipeHandSource
            hand_source = MediaPipeHandSource()
        self.plane_est = ScreenPlaneEstimator(m1, device=device, **(plane_kwargs or {}))
        self.elem_det = ElementDetector(m2, device=device, long_side=long_side)
        self.tracker = FingertipTracker(hand_source)
        self.kf = KeyframeDetector(**(keyframe_kwargs or {}))
        self.guide = Guide(**(guide_kwargs or {}))
        self.verifier = PressVerifier()
        self.structure_fn = structure_fn or (lambda els, crops, flat, kid: elements_to_structure(els, kid))
        self.elements, self.structure, self.flat = [], None, None
        self.expect = None
        self._force_kf = False
        self._last_forced = -1e9

    # ---------- 언어 쪽(F-07)에서 부르는 API ----------
    def set_target(self, element_id, expect=None):
        """목표 버튼 지정. expect는 누른 뒤 기대 결과 (verify.py 참고)"""
        el = next((e for e in self.elements if e.id == element_id), None)
        if el is None:
            raise KeyError(f"현재 화면에 {element_id} 가 없습니다")
        self.guide.set_target(el, self.plane_est.state.aspect if self.plane_est.state else None)
        self.expect = expect
        self.verifier.disarm()

    def clear_target(self):
        self.guide.target = None
        self.verifier.disarm()

    def request_keyframe(self):
        self._force_kf = True

    # ---------- 내부 ----------
    def _rematch_target(self, hand_box=None):
        """키프레임마다 요소 id가 새로 매겨지므로, 위치·종류로 같은 버튼을 다시 찾는다.
        return: 목표가 새 화면에서 사라졌으면 True"""
        tgt = self.guide.target
        if tgt is None:
            return False
        best = max(self.elements, key=lambda e: iou(e.box, tgt.box) if e.kind == tgt.kind else 0.0, default=None)
        if best is not None and best.kind == tgt.kind and iou(best.box, tgt.box) > 0.5:
            self.guide.update_target_box(best)
            return False
        if hand_box is not None and _overlaps(hand_box, tgt.box):
            # 손가락이 버튼을 가려서 못 찾은 것 → 버튼이 사라진 게 아니므로 기존 좌표를 유지
            return False
        self.guide.target = Element(tgt.id, tgt.kind, tgt.box, 0.0, text=tgt.text)
        return True

    def _move_to_new_frame(self, old, new):
        """기준 좌표계가 바뀌면 읽어 둔 요소·목표 좌표를 카메라 영상을 거쳐 새 좌표계로 옮기고 화면을 다시 읽는다"""
        def conv(box):
            x1, y1, x2, y2 = box
            q = new.to_screen(old.to_image([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]))
            return tuple(float(v) for v in (*q.min(0), *q.max(0)))

        for e in self.elements:
            e.box = conv(e.box)
        tgt = self.guide.target
        if tgt is not None and tgt not in self.elements:
            tgt.box = conv(tgt.box)
        self.request_keyframe()

    def _read_screen(self, frame, plane, kid):
        els, flat = self.elem_det.detect(frame, plane)
        crops = ElementDetector.crops(flat, els, margin=self.elem_det.margin)
        self.elements, self.flat = els, flat
        self.structure = self.structure_fn(els, crops, flat, kid)
        # 언어 쪽이 텍스트를 채워 돌려줬다면 요소에도 반영
        by_id = {d["id"]: d for d in self.structure.get("elements", []) if isinstance(d, dict)}
        for e in els:
            d = by_id.get(e.id, {})
            e.text, e.price = d.get("text", e.text), d.get("price", e.price)
            if "conf" in d:
                e.conf = min(e.conf, float(d["conf"]))
            if d.get("uncertain"):
                e.conf = 0.0  # OCR uncertainty must also stop fingertip press guidance.
        return self.structure

    # ---------- 매 프레임 ----------
    def process(self, frame, t=None):
        t = time.monotonic() if t is None else t
        t0 = time.perf_counter()
        res = FrameResult()
        plane, hint = self.plane_est.update(frame, t)
        res.plane, res.hint = plane, hint
        res.timings.update(self.plane_est.last_timing)
        if plane is None:
            res.tip = self.tracker.update(frame, None, t)
            # Losing the plane invalidates screen coordinates and resets dwell.
            res.event = self.guide.update(res.tip, t, target_conf=0.0) if self.guide.target is not None else None
            res.timings["total_ms"] = (time.perf_counter() - t0) * 1000
            return res
        if plane.reframed and plane.ref_prev is not None:
            self._move_to_new_frame(plane.ref_prev, plane)
        if self.guide.target is not None:
            self.guide.aspect = plane.aspect

        tip = self.tracker.update(frame, plane, t)
        res.tip = tip
        res.timings.update(self.tracker.last_timing)

        # 키프레임 판단: 작게 펼친 화면끼리 비교, 손 영역은 제외
        t1 = time.perf_counter()
        small = flatten(frame, plane, long_side=96)
        hb = self.tracker.hand_box_screen(plane)
        is_kf, _ = self.kf.update(small, [hb] if hb else None, force=self._force_kf)
        self._force_kf = False
        res.timings["keyframe_ms"] = (time.perf_counter() - t1) * 1000
        res.keyframe, res.keyframe_id = is_kf, self.kf.keyframe_id

        if is_kf:
            t2 = time.perf_counter()
            before = self.structure
            res.structure = self._read_screen(frame, plane, self.kf.keyframe_id)
            res.timings["read_ms"] = (time.perf_counter() - t2) * 1000
            res.timings.update(self.elem_det.last_timing)
            if self.verifier.armed:
                res.verdict = self.verifier.judge(res.structure)
                if res.verdict.result in ("success", "restarted"):
                    self.guide.target = None  # 다음 목표는 F-07이 정한다
                else:
                    res.target_missing = self._rematch_target(hb)
            elif before is not None:
                res.target_missing = self._rematch_target(hb)
        elif self.verifier.armed:
            status = self.verifier.poll(t, self.kf.changed_since_key(small, [hb] if hb else None))
            if status == "timeout":
                res.verdict = self.verifier.timeout_verdict()
                self.guide.pressed_latch = False  # 다시 누르도록 재안내

        # The app must apply its next target after consuming the new screen.
        if not is_kf and self.guide.target is not None and not self.verifier.armed and res.verdict is None:
            ev = self.guide.update(tip, t)
            res.event = ev
            if ev is not None and ev.type == "press":
                self.verifier.arm(t, self.structure, self.expect)
            elif ev is not None and ev.type == "hold" and t - self._last_forced >= 0.7:
                self.request_keyframe()  # 다시 읽기는 0.7초에 한 번까지 (매 프레임 M2를 돌리지 않게)
                self._last_forced = t
        res.timings["total_ms"] = (time.perf_counter() - t0) * 1000
        return res
