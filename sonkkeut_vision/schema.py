"""
기능 사이를 오가는 데이터 형식 (기능 명세서 6장)

좌표는 모두 '화면 평면 기준 0~1 비율'이다.
  x: 화면 왼쪽 0 → 오른쪽 1,  y: 화면 위 0 → 아래 1
카메라 영상의 픽셀 좌표는 이 모듈 밖으로 내보내지 않는다. 그래야 휴대폰이 움직여도
버튼·손끝 좌표가 같은 기준에 머물고, 언어 쪽(노현석)과 앱 쪽(임현승)이 카메라를 몰라도 된다.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

KINDS = ("tab", "menu", "price", "button", "back")


@dataclass
class Element:
    """F-03 출력 한 건. text·price는 F-04(문자 인식, 노현석)가 채운다."""

    id: str
    kind: str
    box: tuple  # (x1, y1, x2, y2), 0~1
    conf: float
    parent: Optional[str] = None  # 메뉴 카드 안의 가격처럼 다른 요소 안에 들어 있으면 그 요소 id
    text: Optional[str] = None
    price: Optional[int] = None

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def contains(self, x, y, shrink=0.0):
        """shrink: 박스 폭·높이 대비 안쪽으로 줄일 비율. 버튼 가장자리에서 '누르세요'가 나오지 않게 한다."""
        x1, y1, x2, y2 = self.box
        dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
        return x1 + dx <= x <= x2 - dx and y1 + dy <= y <= y2 - dy

    def to_dict(self):
        d = asdict(self)
        d["box"] = [round(float(v), 4) for v in self.box]
        d["conf"] = round(float(self.conf), 3)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class PlaneState:
    """F-02 출력. H는 카메라 픽셀 → 화면 0~1 좌표 변환 행렬."""

    corners: np.ndarray  # (4, 2) 카메라 픽셀, 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래
    H: np.ndarray  # 3x3, 카메라 픽셀 → 화면 0~1
    conf: float
    aspect: float  # 화면 가로/세로 비 (펼친 화면 기준)
    source: str  # "detect" | "track"
    frame_idx: int = 0
    reframed: bool = False  # 이번 프레임에 기준 좌표계가 바뀜 (추적을 놓쳐 새로 검출)
    ref_prev: Optional["PlaneState"] = field(default=None, repr=False)  # 바뀌기 전 기준 (좌표 옮기기용)

    def to_screen(self, pts_px):
        """카메라 픽셀 좌표 (N,2) → 화면 0~1 좌표 (N,2)"""
        p = np.asarray(pts_px, np.float64).reshape(-1, 2)
        ph = np.hstack([p, np.ones((len(p), 1))]) @ self.H.T
        return ph[:, :2] / ph[:, 2:3]

    def to_image(self, pts_scr):
        """화면 0~1 좌표 (N,2) → 카메라 픽셀 좌표 (N,2)"""
        p = np.asarray(pts_scr, np.float64).reshape(-1, 2)
        ph = np.hstack([p, np.ones((len(p), 1))]) @ np.linalg.inv(self.H).T
        return ph[:, :2] / ph[:, 2:3]

    def flat_size(self, long_side=960):
        """펼친 화면 이미지 크기 (W, H)"""
        if self.aspect >= 1:
            return long_side, max(1, int(round(long_side / self.aspect)))
        return max(1, int(round(long_side * self.aspect))), long_side


@dataclass
class Fingertip:
    """F-08 출력"""

    pos: Optional[tuple]  # 화면 0~1 (x, y), 손을 못 찾으면 None
    conf: float = 0.0
    image_px: Optional[tuple] = None  # 카메라 픽셀 (화면 표시용)
    lost_for: float = 0.0  # 손을 놓친 지 몇 초
    inside_screen: bool = False
    pointing: float = 1.0  # 검지를 펴서 가리키는 자세인가 (0~1). 낮으면 검지 끝 관절이 틀렸을 가능성이 크다


@dataclass
class GuidanceEvent:
    """F-09 출력 = 안내 이벤트 (임현승의 음성·진동 출력이 받는다)

    type:
      direction  방향 안내 (speak가 None이면 이번 프레임은 진동만)
      press      "지금 누르세요"
      hold       목표 버튼 신뢰도 부족 → 잠시 멈춤, 키프레임 재인식 요청
      reset      오차가 계속 커짐 → 화면 가운데에서 다시
      no_hand    손이 안 보임
      point      손은 보이지만 검지를 펴지 않음 (검지 끝 위치를 믿을 수 없음)
    """

    type: str
    target_id: Optional[str]
    dir: Optional[str] = None
    distance: Optional[str] = None  # far | near | reach
    speak: Optional[str] = None
    vibe_hz: float = 0.0
    error: Optional[tuple] = None  # (dx, dy) 화면 높이 기준 단위, 디버그·로그용

    def to_dict(self):
        d = asdict(self)
        if self.error is not None:
            d["error"] = [round(float(v), 4) for v in self.error]
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class Verdict:
    """F-10 출력"""

    result: str  # success | fail | uncertain | restarted
    reason: str
    speak: str
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def elements_to_structure(elements, keyframe_id, screen_type="unknown"):
    """요소 목록을 6장의 '화면 구조' 모양으로 감싼다.

    화면 종류 분류(F-05)는 노현석 담당이라 여기서는 'unknown'으로 둔다.
    F-05가 완성되면 이 dict에 screen_type·text·price가 채워져 돌아온다.
    """
    return {
        "screen_type": screen_type,
        "keyframe_id": keyframe_id,
        "elements": [e.to_dict() if isinstance(e, Element) else e for e in elements],
    }
