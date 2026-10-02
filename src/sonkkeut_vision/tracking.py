"""F-08: track joint 8 from caller-supplied M4 hand observations.

Landmarks are camera pixels. A valid, externally established screen transform
is required. Optional plane distances must be real measurements from the
caller; two-dimensional landmarks and relative landmark z do not provide them.
"""

from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import Point, confidence, finite_number


class ScreenProjector(Protocol):
    def camera_to_screen(self, point: Point) -> Point: ...


@dataclass(frozen=True)
class HandObservation:
    hand_id: str
    landmarks: tuple[Point, ...]
    confidence: float
    plane_distance: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hand_id, str) or not self.hand_id.strip():
            raise ValueError("hand_id must be a stable, nonempty identifier")
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        if len(self.landmarks) != 21 or any(not isinstance(p, Point) for p in self.landmarks):
            raise ValueError("M4 must supply exactly 21 camera-pixel Point landmarks")
        confidence(self.confidence)
        if self.plane_distance is not None:
            finite_number(self.plane_distance, "plane_distance")
            if self.plane_distance < 0:
                raise ValueError("plane_distance must be nonnegative")


@dataclass(frozen=True)
class FingertipState:
    raw_point: Point | None
    point: Point | None
    confidence: float
    hand_id: str | None
    reason: str
    selection_basis: str | None = None

    @property
    def valid(self) -> bool:
        return self.reason == "tracking" and self.raw_point is not None and self.point is not None


class FingertipTracker:
    """Recent-five average with explicit validity and continuous-frame resets.

    Timestamp units are seconds from one monotonic clock. Replayed or
    out-of-order timestamps raise ValueError and discard smoothing history.
    Without measured distances for every candidate, an existing identity can
    be retained, but an initial multiple-hand frame is ambiguous.
    """

    def __init__(self, *, min_confidence: float = 0.7, smoothing_frames: int = 5,
                 max_frame_gap: float = 0.2, missing_timeout: float = 1.0) -> None:
        confidence(min_confidence, "min_confidence")
        if type(smoothing_frames) is not int or smoothing_frames <= 0:
            raise ValueError("smoothing_frames must be a positive integer")
        for value, name in ((max_frame_gap, "max_frame_gap"), (missing_timeout, "missing_timeout")):
            finite_number(value, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.min_confidence = min_confidence
        self.max_frame_gap = max_frame_gap
        self.missing_timeout = missing_timeout
        self._history: deque[Point] = deque(maxlen=smoothing_frames)
        self._hand_id: str | None = None
        self._timestamp: float | None = None
        self._last_hand_at: float | None = None
        self._missing_since: float | None = None
        self._transform: ScreenProjector | None = None
        self._keyframe_id: int | None = None

    def _discard(self, *, identity: bool = False) -> None:
        self._history.clear()
        if identity:
            self._hand_id = None

    def _stop(self, reason: str, *, hand: HandObservation | None = None,
              selection_basis: str | None = None) -> FingertipState:
        self._discard(identity=hand is None)
        return FingertipState(None, None, hand.confidence if hand else 0.0,
                              hand.hand_id if hand else None, reason, selection_basis)

    def _choose(self, hands: Sequence[HandObservation]) -> tuple[HandObservation | None, str]:
        if len(hands) == 1:
            return hands[0], "single_hand"
        if all(hand.plane_distance is not None for hand in hands):
            nearest = min(hand.plane_distance for hand in hands)  # all values are measured
            candidates = [hand for hand in hands if hand.plane_distance == nearest]
            if len(candidates) == 1:
                return candidates[0], "measured_plane_distance"
            retained = next((hand for hand in candidates if hand.hand_id == self._hand_id), None)
            return retained, "retained_identity" if retained else "ambiguous_hand"
        retained = next((hand for hand in hands if hand.hand_id == self._hand_id), None)
        return retained, "retained_identity" if retained else "ambiguous_hand"

    def update(self, hands: Sequence[HandObservation], transform: ScreenProjector | None,
               timestamp: float, *, keyframe_id: int | None = None) -> FingertipState:
        try:
            finite_number(timestamp, "timestamp")
            if self._timestamp is not None and timestamp <= self._timestamp:
                raise ValueError("timestamps must strictly increase")
        except ValueError:
            self._discard(identity=True)
            raise
        if self._timestamp is not None and timestamp - self._timestamp > self.max_frame_gap + 1e-9:
            self._discard()
        self._timestamp = timestamp
        if keyframe_id is not None and (type(keyframe_id) is not int or keyframe_id < 0):
            return self._stop("invalid_keyframe")
        # Valid per-frame camera transforms may move or jitter while preserving
        # one canonical screen plane. Average projected points across those
        # updates; changing the plane keyframe or an invalid frame resets it.
        if keyframe_id != self._keyframe_id:
            self._discard()
        self._transform, self._keyframe_id = transform, keyframe_id
        hands = tuple(hands)
        if any(not isinstance(hand, HandObservation) for hand in hands):
            return self._stop("invalid_hand_observation")
        if len({hand.hand_id for hand in hands}) != len(hands):
            return self._stop("ambiguous_hand")
        if not hands:
            if self._missing_since is None:
                self._missing_since = self._last_hand_at if self._last_hand_at is not None else timestamp
            elapsed = timestamp - self._missing_since
            return self._stop("lost_hand" if elapsed + 1e-9 >= self.missing_timeout else "missing_hand")
        self._last_hand_at = timestamp
        self._missing_since = None
        hand, basis = self._choose(hands)
        if hand is None:
            return self._stop("ambiguous_hand")
        if hand.hand_id != self._hand_id:
            self._discard()
        self._hand_id = hand.hand_id
        if hand.confidence < self.min_confidence:
            return self._stop("low_hand_confidence", hand=hand, selection_basis=basis)
        if transform is None or not callable(getattr(transform, "camera_to_screen", None)):
            return self._stop("invalid_plane", hand=hand, selection_basis=basis)
        try:
            point = transform.camera_to_screen(hand.landmarks[8])
        except (ValueError, ArithmeticError):
            return self._stop("invalid_plane", hand=hand, selection_basis=basis)
        if not isinstance(point, Point):
            return self._stop("invalid_plane", hand=hand, selection_basis=basis)
        if not (0 <= point.x <= 1 and 0 <= point.y <= 1):
            return self._stop("outside_plane", hand=hand, selection_basis=basis)
        self._history.append(point)
        count = len(self._history)
        smoothed = Point(sum(p.x for p in self._history) / count, sum(p.y for p in self._history) / count)
        return FingertipState(point, smoothed, hand.confidence, hand.hand_id, "tracking", basis)
