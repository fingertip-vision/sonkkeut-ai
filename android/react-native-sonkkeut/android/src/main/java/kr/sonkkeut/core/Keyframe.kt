package kr.sonkkeut.core

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** 8비트 흑백 이미지 (행 우선) */
class GrayImage(val w: Int, val h: Int, val data: FloatArray) {
    operator fun get(x: Int, y: Int) = data[y * w + x]

    companion object {
        fun fromBytes(w: Int, h: Int, bytes: ByteArray) = GrayImage(w, h, FloatArray(w * h) { (bytes[it].toInt() and 0xFF).toFloat() })
    }
}

/**
 * 키프레임 판단 (파이썬 keyframe.py)
 * 펼친 화면을 긴 변 64칸 흑백으로 줄여 직전 키프레임과 비교한다. 손 영역은 비교에서 뺀다.
 * '바뀌었고 + 몇 프레임 멈춰 있을 때' 키프레임을 낸다(전환 애니메이션 도중 읽기 방지).
 */
class KeyframeDetector(
    val size: Int = 64,
    val changeThr: Double = 12.0,
    val stableThr: Double = 4.0,
    val stableFrames: Int = 3,
    val minValid: Double = 0.4,
) {
    private var key: GrayImage? = null
    private var prev: GrayImage? = null
    private var stableCount = 0
    var keyframeId = 0
        private set
    var lastChange = 0.0
        private set

    fun reset() {
        key = null; prev = null; stableCount = 0; keyframeId = 0; lastChange = 0.0
    }

    /** INTER_AREA 축소 + 3×3 가우시안 */
    fun small(flat: GrayImage): GrayImage {
        val s = size.toDouble() / max(flat.w, flat.h)
        val w = max(1, (flat.w * s).toInt())
        val h = max(1, (flat.h * s).toInt())
        val out = FloatArray(w * h)
        val sx = flat.w.toDouble() / w
        val sy = flat.h.toDouble() / h
        for (y in 0 until h) {
            val y0 = y * sy
            val y1 = (y + 1) * sy
            for (x in 0 until w) {
                val x0 = x * sx
                val x1 = (x + 1) * sx
                var acc = 0.0
                var wsum = 0.0
                var yy = y0.toInt()
                while (yy < min(flat.h.toDouble(), y1)) {
                    val wy = min(yy + 1.0, y1) - max(yy.toDouble(), y0)
                    var xx = x0.toInt()
                    while (xx < min(flat.w.toDouble(), x1)) {
                        val wx = min(xx + 1.0, x1) - max(xx.toDouble(), x0)
                        acc += flat[xx, yy] * wx * wy
                        wsum += wx * wy
                        xx++
                    }
                    yy++
                }
                out[y * w + x] = (acc / wsum).toFloat()
            }
        }
        return gaussian3(GrayImage(w, h, out))
    }

    private fun gaussian3(g: GrayImage): GrayImage {
        val k = floatArrayOf(0.25f, 0.5f, 0.25f)
        fun at(x: Int, y: Int): Float { // BORDER_REFLECT_101
            val xx = if (x < 0) -x else if (x >= g.w) 2 * g.w - 2 - x else x
            val yy = if (y < 0) -y else if (y >= g.h) 2 * g.h - 2 - y else y
            return g[xx.coerceIn(0, g.w - 1), yy.coerceIn(0, g.h - 1)]
        }
        val tmp = FloatArray(g.w * g.h)
        for (y in 0 until g.h) for (x in 0 until g.w) tmp[y * g.w + x] = k[0] * at(x - 1, y) + k[1] * at(x, y) + k[2] * at(x + 1, y)
        val t = GrayImage(g.w, g.h, tmp)
        fun at2(x: Int, y: Int): Float {
            val yy = if (y < 0) -y else if (y >= g.h) 2 * g.h - 2 - y else y
            return t[x, yy.coerceIn(0, g.h - 1)]
        }
        val out = FloatArray(g.w * g.h)
        for (y in 0 until g.h) for (x in 0 until g.w) out[y * g.w + x] = k[0] * at2(x, y - 1) + k[1] * at2(x, y) + k[2] * at2(x, y + 1)
        return GrayImage(g.w, g.h, out)
    }

    /** hand: 화면 0~1 박스 목록 → 비교에 쓸 칸 true */
    fun handMask(w: Int, h: Int, hands: List<Box>?): BooleanArray {
        val m = BooleanArray(w * h) { true }
        for (b in hands.orEmpty()) {
            val ys = max(0, (b.y1 * h).toInt() - 1)
            val ye = min(h, (b.y2 * h).toInt() + 2)
            val xs = max(0, (b.x1 * w).toInt() - 1)
            val xe = min(w, (b.x2 * w).toInt() + 2)
            for (y in ys until ye) for (x in xs until xe) m[y * w + x] = false
        }
        return m
    }

    private fun diff(a: GrayImage, b: GrayImage, mask: BooleanArray?): Double? {
        if (a.w != b.w || a.h != b.h) return 255.0 // 화면 비율이 바뀜 = 다른 화면
        var s = 0.0
        var n = 0
        for (i in a.data.indices) {
            if (mask != null && !mask[i]) continue
            s += abs(a.data[i] - b.data[i]); n++
        }
        if (mask != null && n.toDouble() / a.data.size < minValid) return null // 대부분 가려져 판단 불가
        return if (n == 0) null else s / n
    }

    /** return (키프레임인가, 변화량) */
    fun update(flat: GrayImage, hands: List<Box>? = null, force: Boolean = false): Pair<Boolean, Double> {
        val cur = small(flat)
        val k = key
        if (k == null || force) {
            commit(cur); return true to 0.0
        }
        val mask = handMask(cur.w, cur.h, hands)
        val change = diff(cur, k, mask)
        val p = prev
        val motion = if (p != null) diff(cur, p, mask) else 0.0
        prev = cur
        if (change == null || motion == null) {
            stableCount = 0; return false to 0.0
        }
        lastChange = change
        stableCount = if (motion < stableThr) stableCount + 1 else 0
        if (change > changeThr && stableCount >= stableFrames) {
            commit(cur); return true to change
        }
        return false to change
    }

    /** 누름 결과 확인(F-10)용: 지금 화면이 마지막 키프레임과 다른가 */
    fun changedSinceKey(flat: GrayImage, hands: List<Box>? = null): Boolean {
        val k = key ?: return false
        val cur = small(flat)
        val v = diff(cur, k, handMask(cur.w, cur.h, hands))
        return v != null && v > changeThr
    }

    private fun commit(cur: GrayImage) {
        key = cur; prev = cur; stableCount = 0; keyframeId++
    }

    @Suppress("unused")
    private fun Double.r() = roundToInt()
}
