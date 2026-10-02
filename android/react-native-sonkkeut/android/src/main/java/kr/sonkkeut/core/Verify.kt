package kr.sonkkeut.core

/**
 * F-10 누름 결과 확인 (파이썬 verify.py)
 * "지금 누르세요" 후 1.5초 안에 화면이 바뀌는지 보고, 바뀐 화면 구조를 기대 결과(Expect)와 비교한다.
 */
class PressVerifier(val timeout: Double = 1.5) {
    private var armedAt: Double? = null
    private var before: ScreenStructure? = null
    private var expect: Expect = Expect.CHANGED

    val armed get() = armedAt != null

    fun arm(t: Double, beforeStructure: ScreenStructure?, expect: Expect?) {
        armedAt = t
        before = beforeStructure
        this.expect = expect ?: Expect.CHANGED
    }

    fun disarm() {
        armedAt = null
    }

    /** waiting | changed | timeout | idle */
    fun poll(t: Double, screenChanged: Boolean): String {
        val a = armedAt ?: return "idle"
        if (screenChanged) return "changed"
        if (t - a > timeout) return "timeout"
        return "waiting"
    }

    fun timeoutVerdict(): Verdict {
        disarm()
        return Verdict("fail", "no_change", "버튼이 눌리지 않았습니다. 다시 눌러 주세요")
    }

    fun judge(after: ScreenStructure): Verdict {
        val b = before
        val exp = expect
        disarm()
        val bType = b?.screenType ?: "unknown"
        val aType = after.screenType
        val extra = mapOf("from" to bType, "to" to aType)

        if (aType in START_TYPES && bType !in START_TYPES && exp.screenType !in START_TYPES) {
            return Verdict("restarted", "kiosk_reset", "키오스크가 처음 화면으로 돌아갔습니다. 처음부터 다시 안내할게요", extra)
        }
        val checks = mutableListOf<Pair<String, Boolean>>()
        val unknown = mutableListOf<String>()
        exp.screenType?.let { if (aType == "unknown") unknown += "screen_type" else checks += "screen_type" to (aType == it) }
        exp.screenTypeNot?.let { if (aType == "unknown") unknown += "screen_type_not" else checks += "screen_type_not" to (aType != it) }
        exp.cartDelta?.let { d ->
            val bc = b?.cartCount
            val ac = after.cartCount
            if (bc == null || ac == null) unknown += "cart_delta" else checks += "cart_delta" to (ac - bc == d)
        }
        exp.selected?.let { s ->
            val sel = after.selected
            if (sel == null) unknown += "selected" else checks += "selected" to (s in sel)
        }
        if (exp.changed) checks += "changed" to true

        val failed = checks.filter { !it.second }.map { it.first }
        if (failed.isNotEmpty()) {
            return Verdict("fail", "unexpected:" + failed.joinToString(","),
                "의도와 다른 화면입니다. 이전 화면으로 돌아가는 버튼을 안내할게요", extra)
        }
        if (unknown.isNotEmpty() && checks.none { it.first != "changed" }) {
            return Verdict("uncertain", "unverifiable:" + unknown.joinToString(","), "화면이 바뀌었습니다. 확인 중입니다", extra)
        }
        return Verdict("success", "ok", exp.successSpeak ?: "눌렸습니다", extra)
    }

    companion object {
        val START_TYPES = setOf("start", "idle", "home")
    }
}
