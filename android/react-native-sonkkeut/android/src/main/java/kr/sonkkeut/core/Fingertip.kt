package kr.sonkkeut.core

import kotlin.math.hypot

/**
 * F-08 손끝 추적 (파이썬 fingertip.py)
 * 손 관절 21점 → 검지 끝(8번) → 화면 0~1 좌표 → 최근 5프레임 평균.
 * 손이 여러 개면 화면에 가장 가까운 손. 검지를 접고 있으면 pointing이 낮아진다.
 */
class FingertipTracker(
    private val source: HandSource,
    val smooth: Int = 5,
    val jumpReset: Double = 0.25,
    val maxOutside: Double = 0.15,
) {
    private val hist = ArrayDeque<Pt>()
    private var lastSeen: Double? = null
    var lastHandPx: List<Pt>? = null
        private set

    fun reset() {
        hist.clear(); lastSeen = null
    }

    /** 손 전체를 감싸는 화면 0~1 박스 (키프레임 비교에서 손 영역 제외용), 30% 넓힘 */
    fun handBoxScreen(plane: PlaneState?, pad: Double = 0.3): Box? {
        val pts = lastHandPx ?: return null
        if (plane == null) return null
        val s = pts.map { plane.toScreen(it) }
        val x1 = s.minOf { it.x }; val y1 = s.minOf { it.y }
        val x2 = s.maxOf { it.x }; val y2 = s.maxOf { it.y }
        val px = (x2 - x1) * pad + 0.02
        val py = (y2 - y1) * pad + 0.02
        return Box((x1 - px).coerceIn(-1.0, 2.0), (y1 - py).coerceIn(-1.0, 2.0), (x2 + px).coerceIn(-1.0, 2.0), (y2 + py).coerceIn(-1.0, 2.0))
    }

    fun update(frame: FrameImage, plane: PlaneState?, t: Double): Fingertip {
        val hands = if (plane != null) source.detect(frame, t) else emptyList()
        var best: Hand? = null
        var bestTip: Pt? = null
        var bestD = 0.0
        if (plane != null) {
            for (h in hands) {
                val tip = plane.toScreen(h.points[INDEX_TIP])
                val d = distToUnitSquare(tip)
                if (best == null || d < bestD || (d == bestD && h.score > best.score)) {
                    best = h; bestTip = tip; bestD = d
                }
            }
        }
        if (best == null || bestD > maxOutside) {
            lastHandPx = best?.points
            val ls = lastSeen ?: t.also { lastSeen = it }
            val lost = t - ls
            if (lost > 0.3) hist.clear()
            return Fingertip(pos = null, lostFor = lost)
        }
        lastHandPx = best.points
        val tip = bestTip!!
        if (hist.isNotEmpty() && (hist.last() - tip).norm() > jumpReset) hist.clear()
        hist.addLast(tip)
        while (hist.size > smooth) hist.removeFirst()
        val pos = Pt(hist.map { it.x }.average(), hist.map { it.y }.average())
        lastSeen = t
        val pt = pointingScore(best.points)
        return Fingertip(pos, best.score * pt, best.points[INDEX_TIP], 0.0, bestD == 0.0, pt)
    }

    companion object {
        const val INDEX_TIP = 8

        /** 검지를 펴서 가리키는 자세인지 0~1 (검지 길이 / 손바닥 길이: 0.6 이하 접음, 0.8 이상 폄) */
        fun pointingScore(p: List<Pt>): Double {
            val palm = (p[9] - p[0]).norm()
            if (palm < 1e-6) return 0.0
            val ext = (p[8] - p[5]).norm() / palm
            return ((ext - 0.6) / 0.2).coerceIn(0.0, 1.0)
        }

        fun distToUnitSquare(p: Pt): Double {
            val dx = maxOf(0.0 - p.x, 0.0, p.x - 1.0)
            val dy = maxOf(0.0 - p.y, 0.0, p.y - 1.0)
            return hypot(dx, dy)
        }
    }
}
