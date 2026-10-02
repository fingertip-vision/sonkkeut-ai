"""F-10 verifies an expected effect supplied by F-07, after a press cue.

Cart quantities are optional sidecar evidence produced by F-05/app code. Keys
must identify the exact menu AND options, not a total cart count. No OCR,
planning, app state transitions, or automatic retry is performed here.
"""

from dataclasses import dataclass
from typing import Mapping

from .contracts import SCREEN_TYPES, ScreenSnapshot, confidence, finite_number


@dataclass(frozen=True)
class ExpectedEffect:
    screen_type: str | None = None
    cart_item_key: str | None = None
    before_quantity: int | None = None
    quantity_delta: int = 1

    def __post_init__(self) -> None:
        if (self.screen_type is None) == (self.cart_item_key is None):
            raise ValueError("provide exactly one expected screen transition or cart item")
        if self.screen_type is not None:
            if self.screen_type not in SCREEN_TYPES - {"other"}:
                raise ValueError("expected screen type must be a known screen")
            if self.before_quantity is not None:
                raise ValueError("screen transition cannot contain cart quantity evidence")
        else:
            if not isinstance(self.cart_item_key, str) or not self.cart_item_key.strip():
                raise ValueError("cart item key must identify menu and options")
            if type(self.before_quantity) is not int or self.before_quantity < 0:
                raise ValueError("before quantity must be a nonnegative integer")
        if type(self.quantity_delta) is not int or self.quantity_delta <= 0:
            raise ValueError("quantity delta must be a positive integer")


@dataclass(frozen=True)
class VerificationResult:
    verdict: str
    reason: str
    terminal: bool
    retain_progress: bool = True

    def to_dict(self) -> dict[str, object]:
        return {"verdict": self.verdict, "reason": self.reason,
                "terminal": self.terminal, "retain_progress": self.retain_progress}


def screen_changed(before: ScreenSnapshot, after: ScreenSnapshot, box_iou: float = 0.85) -> bool:
    """Compare visible semantics, ignoring detection IDs/confidence and small box jitter."""
    confidence(box_iou, "box_iou")
    if before.screen_type != after.screen_type or len(before.elements) != len(after.elements):
        return True
    unmatched = list(after.elements)
    for old in before.elements:
        candidates = [(index, old.box.iou(new.box)) for index, new in enumerate(unmatched)
                      if (old.kind, old.text, old.price) == (new.kind, new.text, new.price)]
        if not candidates:
            return True
        index, overlap = max(candidates, key=lambda pair: pair[1])
        if overlap < box_iou:
            return True
        unmatched.pop(index)
    return False


class PressVerifier:
    def __init__(self, timeout_s: float = 1.5, min_confidence: float = 0.6) -> None:
        finite_number(timeout_s, "timeout_s")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        confidence(min_confidence, "min_confidence")
        self.timeout_s = timeout_s
        self.min_confidence = min_confidence
        self._before: ScreenSnapshot | None = None
        self._effect: ExpectedEffect | None = None
        self._started = 0.0
        self._last_timestamp: float | None = None
        self._last_keyframe: int | None = None
        self._last_capture: float | None = None
        self._saw_change = False
        self._terminal: VerificationResult | None = None

    def begin(self, before: ScreenSnapshot, expected: ExpectedEffect, timestamp: float) -> None:
        """Arm only after GuidanceEngine emits its press action; replace the previous attempt."""
        finite_number(timestamp, "timestamp")
        if not isinstance(before, ScreenSnapshot) or not isinstance(expected, ExpectedEffect):
            raise ValueError("begin requires a ScreenSnapshot and ExpectedEffect")
        if timestamp < 0:
            raise ValueError("timestamp must be nonnegative")
        if expected.screen_type == before.screen_type:
            raise ValueError("expected screen must differ from the pre-press screen")
        self._before = before
        self._effect = expected
        self._started = timestamp
        self._last_timestamp = timestamp
        self._last_keyframe = before.keyframe_id
        self._last_capture = None
        self._saw_change = False
        self._terminal = None

    def observe(self, after: ScreenSnapshot | None, timestamp: float, *,
                captured_at: float | None = None,
                cart_quantities: Mapping[str, int] | None = None,
                cart_keyframe_id: int | None = None,
                returned_to_start: bool = False) -> VerificationResult:
        finite_number(timestamp, "timestamp")
        if after is not None and not isinstance(after, ScreenSnapshot):
            raise ValueError("after must be a ScreenSnapshot or None")
        if type(returned_to_start) is not bool:
            raise ValueError("returned_to_start must be a bool")
        if captured_at is not None:
            finite_number(captured_at, "captured_at")
        if cart_quantities is not None:
            if not isinstance(cart_quantities, Mapping):
                raise ValueError("cart quantities must be a mapping")
            if any(not isinstance(key, str) or not key.strip() or type(qty) is not int or qty < 0
                   for key, qty in cart_quantities.items()):
                raise ValueError("cart quantities must map item keys to nonnegative integers")
        if self._before is None or self._effect is None:
            raise RuntimeError("begin a press verification before observing")
        if timestamp < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        self._last_timestamp = timestamp
        if self._terminal is not None:
            return self._terminal
        elapsed = timestamp - self._started
        expired = elapsed >= self.timeout_s
        if returned_to_start:
            return self._finish("failure", "returned_to_start")
        if after is None:
            return self._uncertain("missing_screen", expired)
        if captured_at is None:
            return self._uncertain("missing_capture_time", expired)
        finite_number(captured_at, "captured_at")
        if (captured_at < self._started or captured_at > timestamp or
                (self._last_capture is not None and captured_at <= self._last_capture)):
            return self._uncertain("stale_capture", expired)
        # Replayed/out-of-order snapshots cannot complete another press attempt.
        if after.keyframe_id <= self._last_keyframe:
            return self._uncertain("stale_keyframe", expired)
        self._last_keyframe = after.keyframe_id
        self._last_capture = captured_at
        if (after.screen_type == "other" or not after.elements or
                any(element.conf < self.min_confidence for element in after.elements) or
                not self._before.elements or self._before.screen_type == "other" or
                any(element.conf < self.min_confidence for element in self._before.elements)):
            return self._uncertain("uncertain_screen", expired)
        changed = screen_changed(self._before, after)
        self._saw_change = self._saw_change or changed
        # Late changes cannot prove that the kiosk reacted within the 1.5s window.
        if captured_at - self._started > self.timeout_s:
            # Cameras need not sample exactly at 1.5s. The first subsequent
            # unchanged frame can close an otherwise unchanged attempt.
            if not self._saw_change:
                if self._effect.screen_type is not None:
                    return self._finish("failure", "no_screen_change")
                item_key = self._effect.cart_item_key
                if cart_quantities is None or item_key not in cart_quantities:
                    return self._uncertain("missing_cart_evidence", True)
                if type(cart_keyframe_id) is not int or cart_keyframe_id != after.keyframe_id:
                    return self._uncertain("unbound_cart_evidence", True)
                if cart_quantities[item_key] == self._effect.before_quantity:
                    return self._finish("failure", "cart_unchanged")
            return self._uncertain("late_observation", True)
        if self._effect.screen_type is not None:
            if after.screen_type == self._effect.screen_type:
                return self._finish("success", "expected_screen")
            if changed and after.screen_type != self._before.screen_type:
                return self._finish("failure", "unexpected_screen")
            if expired:
                if captured_at - self._started < self.timeout_s:
                    return self._uncertain("incomplete_window", True)
                return self._finish("failure", "unexpected_change" if self._saw_change else "no_screen_change")
            return self._uncertain("awaiting_expected_screen", False)
        item_key = self._effect.cart_item_key
        if cart_quantities is None or item_key not in cart_quantities:
            return self._uncertain("missing_cart_evidence", expired)
        if type(cart_keyframe_id) is not int or cart_keyframe_id != after.keyframe_id:
            return self._uncertain("unbound_cart_evidence", expired)
        quantity = cart_quantities[item_key]
        if type(quantity) is not int or quantity < 0:
            raise ValueError("cart quantities must be nonnegative integers")
        expected_quantity = self._effect.before_quantity + self._effect.quantity_delta
        if quantity == expected_quantity:
            return self._finish("success", "expected_cart_quantity")
        if quantity != self._effect.before_quantity:
            return self._finish("failure", "unexpected_cart_quantity")
        if expired:
            if captured_at - self._started < self.timeout_s:
                return self._uncertain("incomplete_window", True)
            return self._finish("failure", "cart_unchanged")
        return self._uncertain("awaiting_cart_quantity", False)

    def _finish(self, verdict: str, reason: str) -> VerificationResult:
        self._terminal = VerificationResult(verdict, reason, True)
        return self._terminal

    def _uncertain(self, reason: str, terminal: bool) -> VerificationResult:
        result = VerificationResult("uncertain", reason, terminal)
        if terminal:
            self._terminal = result
        return result
