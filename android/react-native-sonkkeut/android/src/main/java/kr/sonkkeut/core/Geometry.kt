package kr.sonkkeut.core

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/** 2차원 점 (픽셀 또는 화면 0~1 좌표) */
data class Pt(val x: Double, val y: Double) {
    operator fun minus(o: Pt) = Pt(x - o.x, y - o.y)
    operator fun plus(o: Pt) = Pt(x + o.x, y + o.y)
    operator fun times(s: Double) = Pt(x * s, y * s)
    fun norm() = hypot(x, y)
}

/** 화면 0~1 좌표의 박스 */
data class Box(val x1: Double, val y1: Double, val x2: Double, val y2: Double) {
    val cx get() = (x1 + x2) / 2
    val cy get() = (y1 + y2) / 2
    val w get() = x2 - x1
    val h get() = y2 - y1

    fun contains(x: Double, y: Double, shrink: Double = 0.0): Boolean {
        val dx = w * shrink
        val dy = h * shrink
        return x >= x1 + dx && x <= x2 - dx && y >= y1 + dy && y <= y2 - dy
    }

    fun toList() = listOf(x1, y1, x2, y2)
}

fun iou(a: Box, b: Box): Double {
    val ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    val iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    val inter = ix * iy
    val ua = a.w * a.h + b.w * b.h - inter
    return if (ua > 0) inter / ua else 0.0
}

/** inner 면적 중 outer 안에 든 비율 */
fun containment(inner: Box, outer: Box): Double {
    val ix = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    val iy = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    val area = inner.w * inner.h
    return if (area > 0) ix * iy / area else 0.0
}

fun overlaps(a: Box, b: Box) = min(a.x2, b.x2) > max(a.x1, b.x1) && min(a.y2, b.y2) > max(a.y1, b.y1)

/** 3×3 행렬 (행 우선, 길이 9) */
class Mat3(val m: DoubleArray) {
    init {
        require(m.size == 9)
    }

    fun apply(p: Pt): Pt {
        val x = m[0] * p.x + m[1] * p.y + m[2]
        val y = m[3] * p.x + m[4] * p.y + m[5]
        val w = m[6] * p.x + m[7] * p.y + m[8]
        return Pt(x / w, y / w)
    }

    operator fun times(o: Mat3): Mat3 {
        val r = DoubleArray(9)
        for (i in 0 until 3) for (j in 0 until 3) {
            var s = 0.0
            for (k in 0 until 3) s += m[i * 3 + k] * o.m[k * 3 + j]
            r[i * 3 + j] = s
        }
        return Mat3(r)
    }

    fun inverse(): Mat3 {
        val a = m
        val c00 = a[4] * a[8] - a[5] * a[7]
        val c01 = a[5] * a[6] - a[3] * a[8]
        val c02 = a[3] * a[7] - a[4] * a[6]
        val det = a[0] * c00 + a[1] * c01 + a[2] * c02
        require(abs(det) > 1e-15) { "특이 행렬" }
        val inv = doubleArrayOf(
            c00, a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
            c01, a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
            c02, a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3],
        )
        return Mat3(DoubleArray(9) { inv[it] / det })
    }

    companion object {
        fun identity() = Mat3(doubleArrayOf(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        fun scale(sx: Double, sy: Double) = Mat3(doubleArrayOf(sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0))

        /** cv2.getPerspectiveTransform과 같은 4점 대응 호모그래피 (src → dst) */
        fun perspective(src: List<Pt>, dst: List<Pt>): Mat3 {
            val a = Array(8) { DoubleArray(8) }
            val b = DoubleArray(8)
            for (i in 0 until 4) {
                val (x, y) = src[i]
                val (u, v) = dst[i]
                a[i] = doubleArrayOf(x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u)
                b[i] = u
                a[i + 4] = doubleArrayOf(0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v)
                b[i + 4] = v
            }
            val h = solve(a, b)
            return Mat3(doubleArrayOf(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0))
        }

        /** 부분 피벗 가우스 소거 */
        fun solve(a0: Array<DoubleArray>, b0: DoubleArray): DoubleArray {
            val n = b0.size
            val a = Array(n) { a0[it].copyOf() }
            val b = b0.copyOf()
            for (c in 0 until n) {
                var p = c
                for (r in c + 1 until n) if (abs(a[r][c]) > abs(a[p][c])) p = r
                require(abs(a[p][c]) > 1e-12) { "해를 구할 수 없습니다(점이 한 직선 위)" }
                if (p != c) {
                    val t = a[p]; a[p] = a[c]; a[c] = t
                    val tb = b[p]; b[p] = b[c]; b[c] = tb
                }
                for (r in c + 1 until n) {
                    val f = a[r][c] / a[c][c]
                    if (f == 0.0) continue
                    for (k in c until n) a[r][k] -= f * a[c][k]
                    b[r] -= f * b[c]
                }
            }
            val x = DoubleArray(n)
            for (r in n - 1 downTo 0) {
                var s = b[r]
                for (k in r + 1 until n) s -= a[r][k] * x[k]
                x[r] = s / a[r][r]
            }
            return x
        }
    }
}

val UNIT_SQUARE = listOf(Pt(0.0, 0.0), Pt(1.0, 0.0), Pt(1.0, 1.0), Pt(0.0, 1.0))

fun meanSide(c: List<Pt>): Double = (0 until 4).sumOf { (c[(it + 1) % 4] - c[it]).norm() } / 4.0

/** 꼭짓점 4개(카메라 픽셀) → (H: 픽셀→화면 0~1, 화면 가로/세로 비) — plane.homography_from_corners */
fun homographyFromCorners(c: List<Pt>): Pair<Mat3, Double> {
    val h = Mat3.perspective(c, UNIT_SQUARE)
    val w = ((c[1] - c[0]).norm() + (c[2] - c[3]).norm()) / 2
    val hh = ((c[3] - c[0]).norm() + (c[2] - c[1]).norm()) / 2
    return h to w / max(hh, 1e-6)
}

fun quadArea(c: List<Pt>): Double {
    var s = 0.0
    for (i in 0 until 4) {
        val p = c[i]
        val q = c[(i + 1) % 4]
        s += p.x * q.y - q.x * p.y
    }
    return abs(s) / 2
}

fun isConvexQuad(c: List<Pt>): Boolean {
    var sign = 0
    for (i in 0 until 4) {
        val a = c[i]
        val b = c[(i + 1) % 4]
        val d = c[(i + 2) % 4]
        val cr = (b.x - a.x) * (d.y - b.y) - (b.y - a.y) * (d.x - b.x)
        if (abs(cr) < 1e-9) continue
        val s = if (cr > 0) 1 else -1
        if (sign == 0) sign = s else if (s != sign) return false
    }
    return sign != 0
}

/** 볼록하고 너무 작지 않은 사각형인지 — plane.is_valid_quad */
fun isValidQuad(c: List<Pt>, frameW: Int, frameH: Int, minAreaRatio: Double = 0.02) =
    isConvexQuad(c) && quadArea(c) >= minAreaRatio * frameW * frameH
