"""Typed boundary matching chapter 6 of the 2026-10-01 feature specification.

Screen boxes are [left, top, right, bottom] in screen-plane coordinates.
Point itself is unbounded: camera pixels and off-screen fingertips are valid.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

ELEMENT_KINDS = frozenset({"button", "menu", "price", "tab", "back"})
SCREEN_TYPES = frozenset({"menu", "option", "cart", "payment", "other"})


def finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def confidence(value: float, name: str = "confidence") -> None:
    finite_number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        finite_number(self.x, "x")
        finite_number(self.y, "y")


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            value = getattr(self, name)
            finite_number(value, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("box must have positive width and height")

    @property
    def center(self) -> Point:
        return Point((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, point: Point) -> bool:
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom

    def iou(self, other: "Box") -> float:
        width = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        height = max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))
        overlap = width * height
        return overlap / (self.area + other.area - overlap)

    def to_list(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True)
class Element:
    id: str
    kind: str
    box: Box
    conf: float
    text: str | None = None
    price: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("element id must be nonempty")
        if self.kind not in ELEMENT_KINDS:
            raise ValueError(f"unsupported element kind: {self.kind}")
        if not isinstance(self.box, Box):
            raise ValueError("element box must be a Box")
        confidence(self.conf)
        if self.text is not None and not isinstance(self.text, str):
            raise ValueError("element text must be a string or None")
        if self.price is not None and (type(self.price) is not int or self.price < 0):
            raise ValueError("element price must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"id": self.id, "kind": self.kind, "box": self.box.to_list(), "conf": self.conf}
        if self.text is not None:
            result["text"] = self.text
        if self.price is not None:
            result["price"] = self.price
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Element":
        raw_box = value["box"]
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            raise ValueError("box must contain four coordinates")
        return cls(id=value["id"], kind=value["kind"], box=Box(*raw_box),
                   conf=value["conf"], text=value.get("text"), price=value.get("price"))


@dataclass(frozen=True)
class ScreenSnapshot:
    screen_type: str
    keyframe_id: int
    elements: tuple[Element, ...]

    def __post_init__(self) -> None:
        if self.screen_type not in SCREEN_TYPES:
            raise ValueError(f"unsupported screen type: {self.screen_type}")
        if type(self.keyframe_id) is not int or self.keyframe_id < 0:
            raise ValueError("keyframe_id must be a nonnegative integer")
        object.__setattr__(self, "elements", tuple(self.elements))
        if any(not isinstance(element, Element) for element in self.elements):
            raise ValueError("elements must contain Element values")
        ids = [element.id for element in self.elements]
        if len(ids) != len(set(ids)):
            raise ValueError("element ids must be unique within a keyframe")

    def to_dict(self) -> dict[str, Any]:
        return {"screen_type": self.screen_type, "keyframe_id": self.keyframe_id,
                "elements": [element.to_dict() for element in self.elements]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScreenSnapshot":
        return cls(screen_type=value["screen_type"], keyframe_id=value["keyframe_id"],
                   elements=tuple(Element.from_dict(element) for element in value["elements"]))
