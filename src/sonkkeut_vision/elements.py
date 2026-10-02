"""F-03 postprocessing of external detections in rectified screen coordinates.

This boundary consumes M2 inference outputs; it supplies neither trained weights
nor image inference. Confidence/NMS thresholds are unverified tuning defaults.
"""

from dataclasses import dataclass
from typing import Iterable

from .contracts import Box, ELEMENT_KINDS, Element, confidence


@dataclass(frozen=True)
class Detection:
    kind: str
    box: Box
    conf: float

    def __post_init__(self) -> None:
        if self.kind not in ELEMENT_KINDS:
            raise ValueError(f"unsupported element kind: {self.kind}")
        if not isinstance(self.box, Box):
            raise ValueError("detection box must be a normalized Box")
        confidence(self.conf)


@dataclass(frozen=True)
class ElementResult:
    elements: tuple[Element, ...]

    @property
    def requires_plane_reacquisition(self) -> bool:
        """No surviving elements means F-02 must reacquire the screen plane."""
        return not self.elements


def _reading_key(detection: Detection) -> tuple[float | str, ...]:
    box = detection.box
    return (box.top, box.left, box.bottom, box.right, detection.kind, -detection.conf)


def process_detections(detections: Iterable[Detection], keyframe_id: int,
                       min_confidence: float = 0.5,
                       nms_iou_threshold: float = 0.5) -> ElementResult:
    """Filter confidence, apply class-aware NMS, then assign keyframe IDs.

    Reading order is deterministic top-to-bottom, left-to-right by box top-left.
    IDs are regenerated per keyframe and must not be used as cross-frame tracks.
    Boxes from distinct kinds survive overlap; IoU above the threshold suppresses
    a lower-confidence box of the same kind. At equality both boxes survive.
    """
    if type(keyframe_id) is not int or keyframe_id < 0:
        raise ValueError("keyframe_id must be a nonnegative integer")
    confidence(min_confidence, "min_confidence")
    confidence(nms_iou_threshold, "nms_iou_threshold")
    candidates = tuple(detections)
    if any(not isinstance(detection, Detection) for detection in candidates):
        raise ValueError("detections must contain Detection values")
    ranked = sorted((detection for detection in candidates if detection.conf >= min_confidence),
                    key=lambda detection: (-detection.conf, *_reading_key(detection)))
    kept: list[Detection] = []
    for detection in ranked:
        if any(previous.kind == detection.kind
               and previous.box.iou(detection.box) > nms_iou_threshold for previous in kept):
            continue
        kept.append(detection)
    ordered = sorted(kept, key=_reading_key)
    return ElementResult(tuple(Element(f"e{keyframe_id}_{index}", detection.kind,
                                      detection.box, detection.conf)
                               for index, detection in enumerate(ordered, start=1)))
