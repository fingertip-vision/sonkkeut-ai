package kr.sonkkeut.core

import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * F-09 손끝 유도 — 오차 계산 (파이썬 guidance.py를 그대로 옮김)
 *
 * 음성은 0.8초에 한 번까지(같은 문장 반복은 2.5초), 진동은 매 프레임.
 * 잘못된 "지금 누르세요"를 막는 장치: 버튼 가장자리 15% 제외, 0.3초 머무름, 신뢰도 기준, 한 번 말하면 잠금.
 */
class Guide(
    val speakInterval: Double = 0.8,
    val repeatInterval: Double = 2.5,
    val dwell: Double = 0.3,
    val edgeShrink: Double = 0.15,
    val targetConfMin: Double = 0.5,
    val tipConfMin: Double = 0.5,
    val nearMin: Double = 0.06,
    val divergeWindow: Double = 3.0,
    val divergeGain: Double = 0.1,
    val lostAfter: Double = 1.0,
    val hysteresisDeg: Double = 10.0,
) {
    var target: Element? = null
    var aspect = 1.0
    var pressedLatch = false
    private var lastSpeakT = -1e9
    private var lastPhrase: String? = null
    private var lastDir: String? = null
    private var insideSince: Double? = null
    private val distHist = ArrayDeque<Pair<Double, Double>>()

    private fun resetState() {
        lastSpeakT = -1e9
        lastPhrase = null
        lastDir = null
        insideSince = null
        pressedLatch = false
        distHist.clear()
    }

    fun setTarget(element: Element, aspect: Double? = null) {
        target = element
        if (aspect != null) this.aspect = aspect
        resetState()
    }

    /** 키프레임 재인식으로 같은 버튼의 좌표·신뢰도만 갱신 (유도 상태 유지) */
    fun updateTargetBox(element: Element) {
        if (target != null) target = element
    }

    private fun direction(dx: Double, dy: Double): String {
        val ang = Math.toDegrees(atan2(-dy, dx)) // 화면 y는 아래가 +, 사람 기준 위가 +
        val m = (((ang + 22.5) % 360) + 360) % 360 // 파이썬 %처럼 항상 0 이상
        var cand = DIRS[(m / 45).toInt().coerceIn(0, 7)]
        val last = lastDir
        if (last != null && cand != last) {
            val center = DIRS.indexOf(last) * 45.0
            val diff = ((ang - center + 180) % 360 + 360) % 360 - 180
            if (kotlin.math.abs(diff) <= 22.5 + hysteresisDeg) cand = last
        }
        lastDir = cand
        return cand
    }

    private fun vibe(distance: String, d: Double, size: Double): Double = when (distance) {
        "reach" -> 10.0
        "far" -> 2.0
        else -> {
            val r = max(nearMin, size)
            Math.round((4.0 + 4.0 * (1.0 - min(d / r, 1.0))) * 10) / 10.0
        }
    }

    private fun canSpeak(t: Double, text: String): Boolean {
        val gap = t - lastSpeakT
        return gap >= if (text != lastPhrase) speakInterval else repeatInterval
    }

    private fun say(t: Double, text: String): String {
        lastSpeakT = t
        lastPhrase = text
        return text
    }

    private fun maybeSay(t: Double, text: String) = if (canSpeak(t, text)) say(t, text) else null

    private fun diverging(t: Double, d: Double): Boolean {
        distHist.addLast(t to d)
        while (distHist.isNotEmpty() && t - distHist.first().first > divergeWindow) distHist.removeFirst()
        if (distHist.size < 5 || t - distHist.first().first < divergeWindow * 0.9) return false
        val ds = distHist.map { it.second }
        val ups = ds.zipWithNext().count { (a, b) -> b > a }.toDouble() / (ds.size - 1)
        return ds.last() - ds.first() > divergeGain && ups >= 0.6
    }

    fun update(tip: Fingertip?, t: Double, targetConf: Double? = null): GuidanceEvent? {
        val tgt = target ?: return null
        val tid = tgt.id
        val tconf = targetConf ?: tgt.conf

        if (tconf < targetConfMin) {
            insideSince = null
            return GuidanceEvent("hold", tid, speak = maybeSay(t, "잠시 멈춰 주세요"))
        }
        if (tip?.pos == null) {
            insideSince = null
            if (tip != null && tip.lostFor >= lostAfter) {
                return GuidanceEvent("no_hand", tid, speak = maybeSay(t, "검지를 화면 앞으로 가져와 주세요"))
            }
            return null
        }
        if (tip.pointing < 0.5) {
            insideSince = null
            return GuidanceEvent("point", tid, speak = maybeSay(t, "검지 하나만 펴서 가리켜 주세요"))
        }

        val (fx, fy) = tip.pos
        val b = tgt.box
        val size = min(b.w * aspect, b.h)
        val dx = (b.cx - fx) * aspect
        val dy = b.cy - fy
        val d = boxDistance(fx, fy, b, aspect)
        val insideCore = b.contains(fx, fy, edgeShrink)
        val insideAny = b.contains(fx, fy)
        if (!insideAny) pressedLatch = false

        if (insideCore) {
            // 믿을 수 없는 손끝 위치로 머무른 시간을 쌓으면, 확실한 프레임이 한 번만 와도 바로 누르라고 하게 된다.
            // 안전장치 ②(0.3초 머무름)는 ③(손끝 신뢰도)을 만족하는 프레임으로만 센다. (파이썬 guidance.py와 같음)
            val since = if (tip.conf < tipConfMin) { insideSince = null; null } else insideSince ?: t.also { insideSince = it }
            val ok = since != null && t - since >= dwell && tip.conf >= tipConfMin && tconf >= targetConfMin &&
                t - lastSpeakT >= speakInterval
            if (ok && !pressedLatch) {
                pressedLatch = true
                distHist.clear()
                return GuidanceEvent("press", tid, distance = "reach", speak = say(t, "지금 누르세요"),
                    vibeHz = vibe("reach", 0.0, size), error = Pt(dx, dy))
            }
            return GuidanceEvent("direction", tid, distance = "reach", vibeHz = vibe("reach", 0.0, size), error = Pt(dx, dy))
        }

        insideSince = null
        val distance = if (d < max(nearMin, size)) "near" else "far"
        if (diverging(t, d)) {
            distHist.clear()
            return GuidanceEvent("reset", tid, speak = say(t, "화면 가운데에서 다시 시작해 주세요"), error = Pt(dx, dy))
        }
        val dir = direction(dx, dy)
        val text = phrase(dir, distance)
        return GuidanceEvent("direction", tid, dir = dir, distance = distance, speak = maybeSay(t, text),
            vibeHz = vibe(distance, d, size), error = Pt(dx, dy))
    }

    companion object {
        val DIRS = listOf("right", "up_right", "up", "up_left", "left", "down_left", "down", "down_right")
        val DIR_KO = mapOf(
            "right" to "오른쪽", "up_right" to "오른쪽 위", "up" to "위", "up_left" to "왼쪽 위",
            "left" to "왼쪽", "down_left" to "왼쪽 아래", "down" to "아래", "down_right" to "오른쪽 아래",
        )

        /** '로/으로' — 받침이 있으면(ㄹ 제외) '으로' */
        fun josaRo(word: String): String {
            val ch = word.last()
            if (ch in '가'..'힣') {
                val jong = (ch.code - 0xAC00) % 28
                return if (jong != 0 && jong != 8) "으로" else "로"
            }
            return "로"
        }

        fun phrase(direction: String, distance: String): String {
            val w = DIR_KO.getValue(direction)
            val base = w + josaRo(w)
            return if (distance == "near") "$base 조금" else base
        }

        fun boxDistance(px: Double, py: Double, b: Box, aspect: Double): Double {
            val dx = maxOf(b.x1 - px, 0.0, px - b.x2) * aspect
            val dy = maxOf(b.y1 - py, 0.0, py - b.y2)
            return hypot(dx, dy)
        }
    }
}
