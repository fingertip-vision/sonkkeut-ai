"""
F-09 손끝 유도 — 오차 계산 (음성·진동 '출력'은 임현승 담당, 여기서는 무엇을 말하고 얼마나 떨지 정한다)

시각 서보(visual servoing)를 사람 손에 적용한 것이다.
로봇 팔 대신 사람이 움직이므로, 제어 신호를 '방향 음성'과 '거리 진동' 두 채널로 나눴다.
  - 음성은 정보량이 많지만 느리다 → 0.8초에 한 번까지만, 방향이 바뀌었을 때 우선
  - 진동은 정보량은 적지만 즉각적이다 → 매 프레임, 가까울수록 빠르게

오차는 '화면 높이 = 1' 단위로 계산한다. 0~1 좌표를 그대로 쓰면 가로로 긴 화면에서
같은 0.1이라도 가로·세로 실제 거리가 달라 방향 판정이 틀어지기 때문이다.

잘못된 "지금 누르세요"를 0회로 만들기 위한 안전장치 (수용 기준: 20회 중 0회)
  ① 버튼 가장자리 15%는 '도달'로 치지 않는다 (박스를 안쪽으로 줄여 판정)
  ② 그 안에 0.3초 머물러야 한다 (지나가는 손끝 무시)
  ③ 목표 버튼 신뢰도, 손끝 신뢰도가 모두 기준 이상이어야 한다
  ④ 한 번 말한 뒤에는 손끝이 버튼을 벗어났다 다시 들어오기 전까지 반복하지 않는다
"""
from __future__ import annotations

import math
from collections import deque

from .schema import GuidanceEvent

DIRS = ["right", "up_right", "up", "up_left", "left", "down_left", "down", "down_right"]
DIR_KO = {
    "right": "오른쪽", "up_right": "오른쪽 위", "up": "위", "up_left": "왼쪽 위",
    "left": "왼쪽", "down_left": "왼쪽 아래", "down": "아래", "down_right": "오른쪽 아래",
}


def josa_ro(word):
    """'로/으로' — 받침이 있으면(ㄹ 제외) '으로'"""
    ch = word[-1]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return "으로" if jong not in (0, 8) else "로"
    return "로"


def phrase(direction, distance):
    w = DIR_KO[direction]
    base = f"{w}{josa_ro(w)}"
    return base + " 조금" if distance == "near" else base


def box_distance(px, py, box, aspect):
    """점에서 박스까지 거리(박스 안이면 0), 화면 높이 단위"""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2) * aspect
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


class Guide:
    def __init__(self, speak_interval=0.8, repeat_interval=2.5, dwell=0.3, edge_shrink=0.15,
                 target_conf_min=0.5, tip_conf_min=0.5, near_min=0.06, diverge_window=3.0,
                 diverge_gain=0.1, lost_after=1.0, dir_hysteresis_deg=10.0):
        self.speak_interval, self.repeat_interval = speak_interval, repeat_interval
        self.dwell, self.edge_shrink = dwell, edge_shrink
        self.target_conf_min, self.tip_conf_min = target_conf_min, tip_conf_min
        self.near_min = near_min
        self.diverge_window, self.diverge_gain = diverge_window, diverge_gain
        self.lost_after = lost_after
        self.hyst = dir_hysteresis_deg
        self.target = None
        self.aspect = 1.0
        self._reset_state()

    def _reset_state(self):
        self.last_speak_t = -1e9
        self.last_phrase = None
        self.last_dir = None
        self.inside_since = None
        self.pressed_latch = False
        self.dist_hist = deque()

    def set_target(self, element, aspect=None):
        """목표 버튼 지정 (F-07이 고른 요소). element는 Element 또는 6장 형식 dict"""
        if isinstance(element, dict):
            from .schema import Element
            element = Element(id=element["id"], kind=element.get("kind", "button"), box=tuple(element["box"]),
                              conf=element.get("conf", 1.0), text=element.get("text"))
        self.target = element
        if aspect is not None:
            self.aspect = aspect
        self._reset_state()

    def update_target_box(self, element):
        """키프레임 재인식으로 같은 버튼의 좌표·신뢰도만 갱신 (유도 상태는 유지)"""
        if self.target is not None and element is not None:
            self.target = element

    # ---------- 내부 계산 ----------
    def _direction(self, dx, dy):
        ang = math.degrees(math.atan2(-dy, dx))  # 화면 y는 아래가 +, 사람 기준 위가 +
        idx = int(((ang + 22.5) % 360) // 45)
        cand = DIRS[idx]
        if self.last_dir is not None and cand != self.last_dir:
            center = DIRS.index(self.last_dir) * 45
            diff = (ang - center + 180) % 360 - 180
            if abs(diff) <= 22.5 + self.hyst:  # 경계 근처에서 방향 말이 오락가락하지 않게
                cand = self.last_dir
        self.last_dir = cand
        return cand

    def _vibe(self, distance, d, size):
        if distance == "reach":
            return 10.0
        if distance == "far":
            return 2.0
        near_r = max(self.near_min, size)
        return round(4.0 + 4.0 * (1.0 - min(d / near_r, 1.0)), 1)

    def _can_speak(self, t, text):
        """새 문장은 0.8초 간격, 같은 문장 반복은 2.5초 간격 (명세: 음성은 0.8초에 한 번까지)"""
        gap = t - self.last_speak_t
        return gap >= (self.speak_interval if text != self.last_phrase else self.repeat_interval)

    def _say(self, t, text):
        self.last_speak_t, self.last_phrase = t, text
        return text

    def _diverging(self, t, d):
        self.dist_hist.append((t, d))
        while self.dist_hist and t - self.dist_hist[0][0] > self.diverge_window:
            self.dist_hist.popleft()
        if len(self.dist_hist) < 5 or t - self.dist_hist[0][0] < self.diverge_window * 0.9:
            return False
        ds = [v for _, v in self.dist_hist]
        ups = sum(b > a for a, b in zip(ds, ds[1:])) / (len(ds) - 1)
        return ds[-1] - ds[0] > self.diverge_gain and ups >= 0.6

    # ---------- 매 프레임 ----------
    def update(self, tip, t, target_conf=None):
        """
        tip: schema.Fingertip
        target_conf: 최신 키프레임에서 목표 버튼의 신뢰도 (없으면 target.conf)
        return GuidanceEvent | None
        """
        if self.target is None:
            return None
        tid = self.target.id
        tconf = self.target.conf if target_conf is None else target_conf

        if tconf < self.target_conf_min:
            self.inside_since = None
            text = "잠시 멈춰 주세요"
            return GuidanceEvent("hold", tid, speak=self._say(t, text) if self._can_speak(t, text) else None)

        if tip is None or tip.pos is None:
            self.inside_since = None
            if tip is not None and tip.lost_for >= self.lost_after:
                text = "검지를 화면 앞으로 가져와 주세요"
                return GuidanceEvent("no_hand", tid, speak=self._say(t, text) if self._can_speak(t, text) else None)
            return None

        if tip.pointing < 0.5:
            # 검지를 접고 있으면 검지 끝 좌표를 믿을 수 없으므로 방향 안내도, 누르라는 말도 하지 않는다
            self.inside_since = None
            text = "검지 하나만 펴서 가리켜 주세요"
            return GuidanceEvent("point", tid, speak=self._say(t, text) if self._can_speak(t, text) else None)

        fx, fy = tip.pos
        cx, cy = self.target.center
        x1, y1, x2, y2 = self.target.box
        size = min((x2 - x1) * self.aspect, y2 - y1)
        dx, dy = (cx - fx) * self.aspect, cy - fy
        d = box_distance(fx, fy, self.target.box, self.aspect)
        inside_core = self.target.contains(fx, fy, shrink=self.edge_shrink)
        inside_any = self.target.contains(fx, fy)

        if not inside_any:
            self.pressed_latch = False

        if inside_core:
            distance = "reach"
            if self.inside_since is None:
                self.inside_since = t
            ok = (t - self.inside_since >= self.dwell and tip.conf >= self.tip_conf_min
                  and tconf >= self.target_conf_min
                  and t - self.last_speak_t >= self.speak_interval)  # 직전 안내와 겹치지 않게(0.8초 규칙)
            if ok and not self.pressed_latch:
                self.pressed_latch = True
                self.dist_hist.clear()
                return GuidanceEvent("press", tid, distance="reach", speak=self._say(t, "지금 누르세요"),
                                     vibe_hz=self._vibe("reach", 0, size), error=(dx, dy))
            return GuidanceEvent("direction", tid, dir=None, distance="reach", speak=None,
                                 vibe_hz=self._vibe("reach", 0, size), error=(dx, dy))

        self.inside_since = None
        distance = "near" if d < max(self.near_min, size) else "far"

        if self._diverging(t, d):
            self.dist_hist.clear()
            text = "화면 가운데에서 다시 시작해 주세요"
            return GuidanceEvent("reset", tid, speak=self._say(t, text), error=(dx, dy))

        direction = self._direction(dx, dy)
        text = phrase(direction, distance)
        speak = self._say(t, text) if self._can_speak(t, text) else None
        return GuidanceEvent("direction", tid, dir=direction, distance=distance, speak=speak,
                             vibe_hz=self._vibe(distance, d, size), error=(dx, dy))
