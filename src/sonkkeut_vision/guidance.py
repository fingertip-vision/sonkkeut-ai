"""F-09 geometric decisions; device audio and vibration belong to the caller.

Coordinates use the normalized screen plane with positive y pointing down.
Both fresh raw and smoothed fingertip positions must be inside a target for
continuous dwell. Smoothing can never turn a fresh outside sample into a press.
"""

from dataclasses import dataclass
import math

from .contracts import Element, Point, confidence, finite_number


@dataclass(frozen=True)
class GuidanceDecision:
    action: str
    direction: str | None = None
    dx: float | None = None
    dy: float | None = None
    distance: float | None = None
    dwell_seconds: float = 0.0
    reason: str = ""
    recenter: bool = False
    reacquire: bool = False

    @property
    def distance_band(self) -> str | None:
        if self.reason == "awaiting_result":
            return None
        return {"move": "far", "near": "near", "hold": "reached", "press": "reached"}.get(self.action)


def direction_for_delta(dx: float, dy: float) -> str | None:
    """Quantize center-minus-finger delta into eight equal 45-degree sectors."""
    finite_number(dx, "dx")
    finite_number(dy, "dy")
    if dx == 0 and dy == 0:
        return None
    index = int(math.floor((math.atan2(dy, dx) + math.pi / 8) / (math.pi / 4))) % 8
    return ("right", "down_right", "down", "down_left", "left", "up_left", "up", "up_right")[index]


class GuidanceEngine:
    """Emit one press cue, then await the caller's explicit attempt reset.

    Target and keyframe refreshes restart dwell continuity but cannot release a
    pending press cue. F-10 or the interaction controller owns reset_attempt().
    """

    def __init__(self, *, near_distance: float = 0.05, min_confidence: float = 0.7,
                 dwell_duration: float = 0.3, max_frame_gap: float = 0.2,
                 recenter_duration: float = 3.0, increasing_epsilon: float = 1e-6) -> None:
        confidence(min_confidence, "min_confidence")
        for value, name in ((near_distance, "near_distance"), (dwell_duration, "dwell_duration"),
                            (max_frame_gap, "max_frame_gap"), (recenter_duration, "recenter_duration")):
            finite_number(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        finite_number(increasing_epsilon, "increasing_epsilon")
        if increasing_epsilon < 0:
            raise ValueError("increasing_epsilon must be nonnegative")
        self.near_distance = near_distance
        self.min_confidence = min_confidence
        self.dwell_duration = dwell_duration
        self.max_frame_gap = max_frame_gap
        self.recenter_duration = recenter_duration
        self.increasing_epsilon = increasing_epsilon
        self._timestamp: float | None = None
        self._target_key: tuple | None = None
        self._inside_since: float | None = None
        self._pressed = False
        self._previous_distance: float | None = None
        self._previous_valid_at: float | None = None
        self._rising_since: float | None = None
        self._recenter_emitted = False

    def _reset_progress(self) -> None:
        self._inside_since = None
        self._previous_distance = None
        self._previous_valid_at = None
        self._rising_since = None
        self._recenter_emitted = False

    def reset_attempt(self) -> None:
        """Explicitly permit another dwell cue after the caller handles a result.

        A cue is only a request to the output adapter, not proof of a physical
        kiosk press. Dropouts and moving fingers never authorize another cue.
        """
        self._pressed = False
        self._reset_progress()

    def _stop(self, reason: str, *, reacquire: bool = False) -> GuidanceDecision:
        self._reset_progress()
        return GuidanceDecision("stop", reason=reason, reacquire=reacquire)

    def _recenter(self, distance: float, timestamp: float) -> bool:
        rising = (self._previous_distance is not None
                  and distance - self._previous_distance > self.increasing_epsilon)
        if rising:
            if self._rising_since is None:
                self._rising_since = self._previous_valid_at
        else:
            self._rising_since = None
            self._recenter_emitted = False
        self._previous_distance, self._previous_valid_at = distance, timestamp
        if (self._rising_since is not None and not self._recenter_emitted
                and timestamp - self._rising_since + 1e-9 >= self.recenter_duration):
            self._recenter_emitted = True
            return True
        return False

    def update(self, point: Point | None, target: Element | None, timestamp: float, *,
               raw_point: Point | None = None, finger_confidence: float = 1.0,
               keyframe_id: int | None = None, hand_id: str | None = None) -> GuidanceDecision:
        try:
            finite_number(timestamp, "timestamp")
            if self._timestamp is not None and timestamp <= self._timestamp:
                raise ValueError("timestamps must strictly increase")
        except ValueError:
            self._reset_progress()
            raise
        if self._timestamp is not None and timestamp - self._timestamp > self.max_frame_gap + 1e-9:
            self._reset_progress()
        self._timestamp = timestamp
        if keyframe_id is not None and (type(keyframe_id) is not int or keyframe_id < 0):
            return self._stop("invalid_keyframe", reacquire=True)
        if hand_id is not None and (not isinstance(hand_id, str) or not hand_id.strip()):
            return self._stop("invalid_hand_id")
        if target is None:
            self._target_key = None
            return self._stop("missing_target", reacquire=True)
        if not isinstance(target, Element):
            self._target_key = None
            return self._stop("invalid_target", reacquire=True)
        if target.kind == "price":
            return self._stop("noninteractive_target", reacquire=True)
        target_key = (keyframe_id, target.id, target.kind, target.box, hand_id)
        if target_key != self._target_key:
            self._reset_progress()
            self._target_key = target_key
        if target.conf < self.min_confidence:
            return self._stop("low_target_confidence", reacquire=True)
        try:
            confidence(finger_confidence, "finger_confidence")
        except ValueError:
            return self._stop("invalid_finger_confidence")
        if finger_confidence < self.min_confidence:
            return self._stop("low_finger_confidence")
        if point is None or raw_point is None:
            return self._stop("missing_finger")
        if not isinstance(point, Point) or not isinstance(raw_point, Point):
            return self._stop("invalid_finger")
        if not all(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in (point, raw_point)):
            return self._stop("outside_plane")
        center = target.box.center
        dx, dy = center.x - point.x, center.y - point.y
        distance = math.hypot(dx, dy)
        if self._pressed:
            return GuidanceDecision("hold", dx=dx, dy=dy, distance=distance,
                                    reason="awaiting_result")
        if target.box.contains(point) and target.box.contains(raw_point):
            self._previous_distance = None
            self._previous_valid_at = None
            self._rising_since = None
            self._recenter_emitted = False
            if self._inside_since is None:
                self._inside_since = timestamp
            dwell = timestamp - self._inside_since
            if not self._pressed and dwell + 1e-9 >= self.dwell_duration:
                self._pressed = True
                return GuidanceDecision("press", dx=dx, dy=dy, distance=distance,
                                        dwell_seconds=dwell, reason="continuous_inside")
            return GuidanceDecision("hold", dx=dx, dy=dy, distance=distance, dwell_seconds=dwell,
                                    reason="dwell_pending")
        self._inside_since = None
        recenter = self._recenter(distance, timestamp)
        return GuidanceDecision("near" if distance <= self.near_distance else "move",
                                direction_for_delta(dx, dy), dx, dy, distance,
                                reason="raw_outside_target" if not target.box.contains(raw_point)
                                else "smoothed_outside_target", recenter=recenter)
