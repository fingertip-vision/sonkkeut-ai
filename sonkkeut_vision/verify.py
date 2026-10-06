"""
F-10 누름 결과 확인

키오스크는 누름 신호를 밖으로 내보내지 않으므로, '화면이 바뀌었는가'로 누름을 감지한다.
  ① "지금 누르세요" 이후 1.5초 동안 펼친 화면이 마지막 키프레임과 달라지는지 본다
  ② 달라졌으면 새 키프레임을 읽고(F-03~F-05) 그 화면 구조를 기대 결과와 비교한다
  ③ 1.5초 안에 변화가 없으면 '눌리지 않음'

기대 결과(expect)는 버튼 순서 계획(F-07, 노현석)이 목표 버튼과 함께 넘긴다. 형식:
  {"screen_type": "option"}            누르면 옵션 화면이 떠야 함
  {"screen_type_not": "menu"}          지금 화면을 벗어나기만 하면 됨
  {"cart_delta": 1}                    장바구니 항목이 1개 늘어야 함 (화면 구조의 cart_count 사용)
  {"selected": "e12"}                  해당 요소가 선택 상태가 되어야 함 (화면 구조의 selected 목록 사용)
  {"changed": true}                    아무 변화든 성공 (기본값)
여러 키를 함께 쓰면 모두 만족해야 성공이다.
"""
from __future__ import annotations

from .schema import Verdict

START_TYPES = ("start", "idle", "home")  # 키오스크가 첫 화면으로 돌아갔다고 보는 화면 종류


class PressVerifier:
    def __init__(self, timeout=1.5):
        self.timeout = timeout
        self.armed_at = None
        self.before = None
        self.expect = None

    @property
    def armed(self):
        return self.armed_at is not None

    def arm(self, t, before_structure, expect=None):
        """'지금 누르세요'를 말한 순간 부른다"""
        self.armed_at = t
        self.before = before_structure
        self.expect = expect or {"changed": True}

    def disarm(self):
        self.armed_at = None

    def poll(self, t, screen_changed):
        """매 프레임: 'waiting' | 'changed' | 'timeout' | 'idle'"""
        if self.armed_at is None:
            return "idle"
        if screen_changed:
            return "changed"
        if t - self.armed_at > self.timeout:
            return "timeout"
        return "waiting"

    def timeout_verdict(self):
        self.disarm()
        return Verdict("fail", "no_change", "버튼이 눌리지 않았습니다. 다시 눌러 주세요")

    def judge(self, after_structure):
        """새 키프레임의 화면 구조로 성공·실패를 판정한다"""
        before, exp = self.before or {}, self.expect or {"changed": True}
        self.disarm()
        b_type, a_type = before.get("screen_type", "unknown"), after_structure.get("screen_type", "unknown")

        # 키오스크가 처음 화면으로 돌아감 → 진행 상태는 앱이 기억하고 처음부터 다시
        if a_type in START_TYPES and b_type not in START_TYPES and exp.get("screen_type") not in START_TYPES:
            return Verdict("restarted", "kiosk_reset", "키오스크가 처음 화면으로 돌아갔습니다. 처음부터 다시 안내할게요",
                           {"from": b_type, "to": a_type})

        checks, unknown = [], []
        if "screen_type" in exp:
            if a_type == "unknown":
                unknown.append("screen_type")
            else:
                checks.append(("screen_type", a_type == exp["screen_type"]))
        if "screen_type_not" in exp:
            if a_type == "unknown":
                unknown.append("screen_type_not")
            else:
                checks.append(("screen_type_not", a_type != exp["screen_type_not"]))
        if "cart_delta" in exp:
            bc, ac = before.get("cart_count"), after_structure.get("cart_count")
            if bc is None or ac is None:
                unknown.append("cart_delta")
            else:
                checks.append(("cart_delta", ac - bc == exp["cart_delta"]))
        if "selected" in exp:
            sel = after_structure.get("selected")
            if sel is None:
                unknown.append("selected")
            else:
                checks.append(("selected", exp["selected"] in sel))
        if exp.get("changed"):
            checks.append(("changed", True))  # 이 함수는 화면이 바뀐 뒤에만 불린다

        failed = [k for k, ok in checks if not ok]
        if failed:
            return Verdict("fail", "unexpected:" + ",".join(failed),
                           "의도와 다른 화면입니다. 이전 화면으로 돌아가는 버튼을 안내할게요",
                           {"from": b_type, "to": a_type})
        if unknown:
            # 확인할 수단이 없음 (예: F-05가 화면 종류를 'unknown'으로 줌)
            return Verdict("uncertain", "unverifiable:" + ",".join(unknown), "화면이 바뀌었습니다. 확인 중입니다",
                           {"from": b_type, "to": a_type})
        return Verdict("success", "ok", exp.get("success_speak", "눌렸습니다"), {"from": b_type, "to": a_type})
