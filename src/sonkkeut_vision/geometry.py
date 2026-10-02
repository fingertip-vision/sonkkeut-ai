"""F-02 geometry for corners supplied by an external M1/tracker adapter.

Camera coordinates use pixels with x right and y down. Corners must arrive in
TL, TR, BR, BL order; this module does not detect, track or warp image pixels.
Raster adapters can use ``screen_to_camera`` to sample a rectified image.
"""

from dataclasses import dataclass
import math
from typing import Sequence

from .contracts import Point, confidence as validate_confidence, finite_number

_EPSILON = 1e-12
_UNIT_CORNERS = (Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))


def _solve(matrix: list[list[float]], values: list[float]) -> tuple[float, ...]:
    """Solve a small normalized linear system with scaled partial pivoting."""
    rows = [row[:] + [value] for row, value in zip(matrix, values)]
    size = len(rows)
    scales = [max(abs(value) for value in row[:-1]) for row in rows]
    for column in range(size):
        pivot = max(range(column, size),
                    key=lambda index: abs(rows[index][column]) / scales[index]
                    if scales[index] else 0)
        if not scales[pivot] or abs(rows[pivot][column]) <= _EPSILON * scales[pivot]:
            raise ValueError("corners do not define a stable homography")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scales[column], scales[pivot] = scales[pivot], scales[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for index in range(size):
            if index == column:
                continue
            factor = rows[index][column]
            rows[index] = [a - factor * b for a, b in zip(rows[index], rows[column])]
    solution = tuple(row[-1] for row in rows)
    if not all(math.isfinite(value) for value in solution):
        raise ValueError("homography calculation was not finite")
    return solution


def _inverse(matrix: tuple[float, ...]) -> tuple[float, ...]:
    a, b, c, d, e, f, g, h, i = matrix
    cofactors = (e * i - f * h, c * h - b * i, b * f - c * e,
                 f * g - d * i, a * i - c * g, c * d - a * f,
                 d * h - e * g, b * g - a * h, a * e - b * d)
    terms = (a * cofactors[0], b * cofactors[3], c * cofactors[6])
    determinant = sum(terms)
    scale = sum(abs(value) for value in terms)
    if not math.isfinite(determinant) or not scale or abs(determinant) <= _EPSILON * scale:
        raise ValueError("homography is singular")
    result = tuple(value / determinant for value in cofactors)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("inverse homography was not finite")
    return result


def _map(matrix: tuple[float, ...], point: Point) -> Point:
    if not isinstance(point, Point):
        raise ValueError("projection input must be a Point")
    a, b, c, d, e, f, g, h, i = matrix
    denominator_terms = (g * point.x, h * point.y, i)
    denominator = sum(denominator_terms)
    denominator_scale = sum(abs(value) for value in denominator_terms)
    if (not math.isfinite(denominator) or not math.isfinite(denominator_scale)
            or not denominator_scale or abs(denominator) <= _EPSILON * denominator_scale):
        raise ValueError("point is on or too close to the projective horizon")
    x = (a * point.x + b * point.y + c) / denominator
    y = (d * point.x + e * point.y + f) / denominator
    return Point(x, y)


@dataclass(frozen=True)
class Homography:
    """An invertible camera-pixel to unit-screen projective transform."""

    matrix: tuple[float, ...]

    def __post_init__(self) -> None:
        values = tuple(self.matrix)
        if len(values) != 9:
            raise ValueError("homography matrix must contain nine values")
        for value in values:
            finite_number(value, "homography coefficient")
        object.__setattr__(self, "matrix", values)
        _inverse(values)

    @classmethod
    def from_corners(cls, corners: Sequence[Point]) -> "Homography":
        points = tuple(corners)
        if len(points) != 4 or any(not isinstance(point, Point) for point in points):
            raise ValueError("four ordered Point corners are required")
        center_x = sum(point.x / 4 for point in points)
        center_y = sum(point.y / 4 for point in points)
        scale = max(max(point.x for point in points) - min(point.x for point in points),
                    max(point.y for point in points) - min(point.y for point in points))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("corner span must be positive and finite")
        normalized = tuple(Point((point.x - center_x) / scale, (point.y - center_y) / scale)
                           for point in points)
        # Positive turns in downward-positive pixel coordinates require a simple,
        # strictly convex clockwise polygon and reject mirrored/crossed order.
        for index in range(4):
            first, second, third = (normalized[(index + offset) % 4] for offset in range(3))
            turn = ((second.x - first.x) * (third.y - second.y)
                    - (second.y - first.y) * (third.x - second.x))
            if turn <= _EPSILON:
                raise ValueError("corners must be strictly convex in TL, TR, BR, BL order")
        rows: list[list[float]] = []
        values: list[float] = []
        for source, target in zip(normalized, _UNIT_CORNERS):
            x, y, u, v = source.x, source.y, target.x, target.y
            rows.extend(([x, y, 1, 0, 0, 0, -u * x, -u * y],
                         [0, 0, 0, x, y, 1, -v * x, -v * y]))
            values.extend((u, v))
        a, b, c, d, e, f, g, h = _solve(rows, values)
        transform = cls((a / scale, b / scale, c - (a * center_x + b * center_y) / scale,
                         d / scale, e / scale, f - (d * center_x + e * center_y) / scale,
                         g / scale, h / scale, 1 - (g * center_x + h * center_y) / scale))
        for source, target in zip(points, _UNIT_CORNERS):
            mapped = transform.camera_to_screen(source)
            if abs(mapped.x - target.x) > 1e-8 or abs(mapped.y - target.y) > 1e-8:
                raise ValueError("corner projection is numerically unstable")
        return transform

    def camera_to_screen(self, point: Point) -> Point:
        return _map(self.matrix, point)

    def project(self, point: Point) -> Point:
        """Alias used by the fingertip tracker projection protocol."""
        return self.camera_to_screen(point)

    def screen_to_camera(self, point: Point) -> Point:
        return _map(_inverse(self.matrix), point)


@dataclass(frozen=True)
class PlaneUpdate:
    homography: Homography | None
    lost_for_s: float
    loss_timeout: bool
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.homography is not None

    @property
    def requires_reacquisition(self) -> bool:
        return not self.valid


class PlaneTracker:
    """Validate fresh external corners and invalidate a lost plane immediately.

    The confidence threshold is an unverified tuning default. A three-second
    loss timeout can trigger the specification's loss feedback; it never keeps
    an old transform alive during that interval. Timestamps use monotonic seconds.
    """

    def __init__(self, min_confidence: float = 0.5, loss_timeout_s: float = 3.0) -> None:
        validate_confidence(min_confidence, "min_confidence")
        finite_number(loss_timeout_s, "loss_timeout_s")
        if loss_timeout_s <= 0:
            raise ValueError("loss_timeout_s must be positive")
        self.min_confidence = min_confidence
        self.loss_timeout_s = loss_timeout_s
        self._last_timestamp: float | None = None
        self._loss_started: float | None = None

    def update(self, corners: Sequence[Point] | None, timestamp_s: float,
               confidence: float = 1.0) -> PlaneUpdate:
        finite_number(timestamp_s, "timestamp_s")
        if timestamp_s < 0 or (self._last_timestamp is not None and timestamp_s < self._last_timestamp):
            raise ValueError("timestamps must be nonnegative and monotonic")
        self._last_timestamp = timestamp_s
        transform = None
        reason = "missing corners"
        try:
            validate_confidence(confidence)
            if confidence < self.min_confidence:
                reason = "low corner confidence"
            elif corners is not None:
                transform = Homography.from_corners(corners)
                reason = None
        except (ValueError, TypeError, OverflowError) as error:
            reason = str(error)
        if transform is not None:
            self._loss_started = None
            return PlaneUpdate(transform, 0.0, False)
        if self._loss_started is None:
            self._loss_started = timestamp_s
        lost_for_s = timestamp_s - self._loss_started
        return PlaneUpdate(None, lost_for_s, lost_for_s >= self.loss_timeout_s, reason)
