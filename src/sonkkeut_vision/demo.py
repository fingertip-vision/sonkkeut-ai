"""Replay synthetic model outputs; no camera, model, OCR or planner is executed."""

from dataclasses import asdict, replace
import json

from .contracts import Box, Element, Point, ScreenSnapshot
from .elements import Detection, process_detections
from .geometry import Homography, PlaneTracker
from .guidance import GuidanceEngine
from .tracking import FingertipTracker, HandObservation
from .verification import ExpectedEffect, PressVerifier


def hand_at(transform: Homography, screen_point: Point) -> HandObservation:
    pixel = transform.screen_to_camera(screen_point)
    # Only joint 8 is used; the other joints are synthetic placeholders.
    return HandObservation("demo-hand", tuple(pixel for _ in range(21)), 0.95)


def replay_transition(before_type: str, expected: ExpectedEffect, after_type: str,
                      *, text: str, kind: str, keyframe_id: int = 1,
                      cart_quantities: dict[str, int] | None = None) -> dict[str, object]:
    corners = (Point(100, 80), Point(1180, 100), Point(1130, 650), Point(130, 620))
    plane_tracker = PlaneTracker()
    box = Box(0.25, 0.3, 0.45, 0.5)
    processed = process_detections((Detection(kind, box, 0.95),
                                    Detection(kind, Box(0.26, 0.31, 0.46, 0.51), 0.8)), keyframe_id)
    # Synthetic F-04/05 and F-07 fixture inputs, not inferred text or planning.
    target = replace(processed.elements[0], text=text)
    before = ScreenSnapshot(before_type, keyframe_id, (target,))
    fingertip = FingertipTracker()
    guidance = GuidanceEngine()
    events = []
    pressed_at = None
    for index in range(11):
        timestamp = index / 10
        plane = plane_tracker.update(corners, timestamp, 0.98)
        position = Point(0.15, 0.7) if index < 2 else target.box.center
        state = fingertip.update((hand_at(plane.homography, position),), plane.homography,
                                 timestamp, keyframe_id=keyframe_id)
        decision = guidance.update(state.point, target, timestamp, raw_point=state.raw_point,
                                   finger_confidence=state.confidence, keyframe_id=keyframe_id,
                                   hand_id=state.hand_id)
        event = asdict(decision)
        event.update(timestamp=timestamp, target_id=target.id, distance_band=decision.distance_band)
        events.append(event)
        if decision.action == "press":
            pressed_at = timestamp
    if pressed_at is None:
        raise RuntimeError("synthetic fixture did not produce a dwell cue")
    verifier = PressVerifier()
    verifier.begin(before, expected, pressed_at)
    after = ScreenSnapshot(after_type, keyframe_id + 1, (
        Element(f"e{keyframe_id + 1}_1", "button", Box(0.7, 0.8, 0.95, 0.95), 0.95, "다음"),))
    result = verifier.observe(after, pressed_at + 0.5, captured_at=pressed_at + 0.4,
                              cart_quantities=cart_quantities,
                              cart_keyframe_id=after.keyframe_id if cart_quantities is not None else None)
    return {"before": before.to_dict(), "after": after.to_dict(), "events": events,
            "verification": result.to_dict()}


def run_demo() -> dict[str, object]:
    return {"fixture_type": "synthetic_model_outputs",
            "models_executed": False, "android_acceptance_verified": False,
            "flows": [
                replay_transition("menu", ExpectedEffect(screen_type="option"), "option",
                                  text="아메리카노", kind="menu"),
                replay_transition("option", ExpectedEffect(cart_item_key="아메리카노|temp=hot", before_quantity=0),
                                  "cart", text="담기", kind="button", keyframe_id=3,
                                  cart_quantities={"아메리카노|temp=hot": 1}),
                replay_transition("cart", ExpectedEffect(screen_type="payment"), "payment",
                                  text="결제", kind="button", keyframe_id=5)]}


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
