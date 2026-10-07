package kr.sonkkeut.core

/**
 * F-02 화면 평면 추정 (파이썬 plane.py + corner_net.py)
 *
 * M1으로 꼭짓점 검출 → M1-R로 정밀 보정 → 호모그래피. 그 사이 프레임은 광류로 추적(수 ms).
 * 30프레임마다 재검출해 추적값과 비교하고, 변 길이의 8% 넘게 다를 때만 기준 좌표계를 바꾼다.
 */
class CornerRefiner(private val model: TensorModel, private val ops: ImageOps, val iters: Int = 2, val maxShift: Double = 0.12) {
    fun refine(gray: GrayFrame, corners: List<Pt>): List<Pt> {
        val side = meanSide(corners)
        val size = SCALE * side
        var c = corners
        repeat(iters) {
            val input = FloatArray(4 * PATCH * PATCH)
            for (k in 0 until 4) ops.cropPatch(gray, c[k], size, k).copyInto(input, k * PATCH * PATCH)
            val d = model.run(input, longArrayOf(4, 1, PATCH.toLong(), PATCH.toLong()))
            c = (0 until 4).map { k -> offsetToImage(c[k], size, k, d[2 * k].toDouble(), d[2 * k + 1].toDouble()) }
        }
        if ((0 until 4).maxOf { (c[it] - corners[it]).norm() } > maxShift * side) return corners
        if (!isConvexQuad(c)) return corners
        return c
    }

    companion object {
        const val PATCH = 64
        const val SCALE = 0.25
        val FLIPS = listOf(false to false, true to false, true to true, false to true) // TL, TR, BR, BL

        /** training/corner_refiner.py offset_to_image */
        fun offsetToImage(center: Pt, size: Double, k: Int, dx: Double, dy: Double): Pt {
            val (fx, fy) = FLIPS[k]
            return Pt(center.x + (if (fx) -dx else dx) * size, center.y + (if (fy) -dy else dy) * size)
        }
    }
}

class ScreenPlaneEstimator(
    private val m1: TensorModel,
    private val ops: ImageOps,
    private val refiner: CornerRefiner? = null,
    val conf: Double = 0.25,
    val kpConf: Double = 0.5,
    val redetectEvery: Int = 30,
    val minInliers: Int = 25,
    val minInlierRatio: Double = 0.6,
    val lostAfter: Double = 3.0,
    val reframeTol: Double = 0.08,
    val imgsz: Int = 640,
) {
    var state: PlaneState? = null
        private set
    private var prevGray: GrayFrame? = null
    private var pts: TrackPoints? = null
        set(v) { // 바뀐 특징점 묶음은 네이티브 메모리를 쓰므로 이전 것을 해제
            if (field !== v) ops.release(field)
            field = v
        }
    private var sinceDetect = 0
    private var frameIdx = 0
    private var lastSeen: Double? = null
    var lastMode = "detect"
        private set

    fun reset() {
        state = null; prevGray = null; pts = null; sinceDetect = 0; frameIdx = 0; lastSeen = null
    }

    fun forceRedetect() {
        sinceDetect = redetectEvery
    }

    class CornerDetection(val corners: List<Pt>, val conf: Double, val kpConf: List<Double>)

    fun detect(frame: FrameImage): CornerDetection? {
        val lb = Letterbox.of(frame.width, frame.height, imgsz)
        val out = m1.run(ops.letterboxTensor(frame, lb), longArrayOf(1, 3, imgsz.toLong(), imgsz.toLong()))
        val n = out.size / 17
        val d = YoloDecoder.decodePoseBest(out, 4, n, lb, conf) ?: return null
        return CornerDetection(d.keypoints!!, d.score * d.kpConf!!.average(), d.kpConf)
    }

    private fun make(c: List<Pt>, conf: Double, source: String) = PlaneState.of(c, conf, source, frameIdx)

    private fun adopt(gray: GrayFrame, c: List<Pt>, conf: Double, reframeFrom: PlaneState?) {
        val s = make(c, conf, "detect")
        if (reframeFrom != null) {
            s.reframed = true
            s.refPrev = reframeFrom
        }
        state = s
        pts = ops.seedPoints(gray, c)
        sinceDetect = 0
    }

    /** return (PlaneState?, hint 문구?) */
    fun update(frame: FrameImage, t: Double): Pair<PlaneState?, String?> {
        val gray = ops.toGray(frame)
        frameIdx++
        var hint: String? = null
        val prev = state
        var tracked: Pair<List<Pt>, Double>? = null
        val pg = prevGray
        val p = pts
        if (prev != null && pg != null && p != null && p.count >= minInliers) {
            val tr = ops.track(pg, gray, p, minInliers)
            if (tr != null && tr.inlierRatio >= minInlierRatio && tr.inliers.count >= minInliers) {
                val c = prev.corners.map { tr.homography.apply(it) }
                if (isValidQuad(c, frame.width, frame.height)) {
                    tracked = c to tr.inlierRatio
                    pts = tr.inliers
                }
            }
            if (tr != null && tracked == null) ops.release(tr.inliers)
        }
        val needDetect = tracked == null || sinceDetect >= redetectEvery
        var det: CornerDetection? = if (needDetect) detect(frame) else null
        var kp: List<Pt>? = null
        if (det != null) {
            hint = missingHint(det.corners, det.kpConf, frame.width, frame.height, kpConf)
            kp = det.corners
            if (hint == null && refiner != null) kp = refiner.refine(gray, kp)
            if (hint != null || !isValidQuad(kp, frame.width, frame.height)) det = null
        }

        if (tracked != null) {
            val (corners, ratio) = tracked
            if (det != null && kp != null) {
                val side = meanSide(corners)
                val maxDiff = (0 until 4).maxOf { (kp[it] - corners[it]).norm() }
                if (maxDiff > reframeTol * side) {
                    adopt(gray, kp, det.conf, PlaneState.of(corners, prev!!.conf, "track", frameIdx))
                    lastMode = "reframe"
                } else {
                    sinceDetect = 0
                    state = make(corners, maxOf(det.conf, prev!!.conf * 0.98), "track")
                    lastMode = "verify"
                }
            } else {
                state = make(corners, prev!!.conf * (0.98 + 0.02 * ratio), "track")
                sinceDetect++
                lastMode = "track"
            }
            if ((pts?.count ?: 0) < 80) pts = ops.seedPoints(gray, state!!.corners)
        } else if (det != null && kp != null) {
            // 이 프레임의 추적값이 없으므로 prev(이전 프레임 좌표계)를 거쳐 버튼 좌표를 옮기면
            // 두 프레임 사이 휴대폰 움직임만큼 좌표가 밀린다. 옮기지 않고 다시 읽도록 rebased로 알린다.
            adopt(gray, kp, det.conf, null)
            state!!.rebased = true
            lastMode = "detect"
        } else {
            state = null; pts = null
            lastMode = "detect"
        }
        ops.release(prevGray)
        prevGray = gray
        if (state != null) lastSeen = t
        else if (hint == null) {
            val ls = lastSeen ?: t.also { lastSeen = it }
            if (t - ls >= lostAfter) hint = "화면이 보이지 않습니다. 천천히 움직여 주세요"
        }
        return state to hint
    }

    companion object {
        /** 보이지 않는 꼭짓점이 있으면 휴대폰을 어느 쪽으로 옮길지 (plane.missing_hint) */
        fun missingHint(kp: List<Pt>, kc: List<Double>, w: Int, h: Int, kpConf: Double = 0.5, margin: Int = 4): String? {
            val miss = (0 until 4).map { j ->
                kc[j] < kpConf || !(kp[j].x >= margin && kp[j].x < w - margin && kp[j].y >= margin && kp[j].y < h - margin)
            }
            if (miss.none { it }) return null
            var left = false; var right = false; var top = false; var bottom = false
            for (j in 0 until 4) {
                if (!miss[j]) continue
                val (x, y) = kp[j]
                var out = false
                if (x < margin) { left = true; out = true }
                if (x >= w - margin) { right = true; out = true }
                if (y < margin) { top = true; out = true }
                if (y >= h - margin) { bottom = true; out = true }
                if (!out) {
                    left = left || j == 0 || j == 3
                    right = right || j == 1 || j == 2
                    top = top || j == 0 || j == 1
                    bottom = bottom || j == 2 || j == 3
                }
            }
            if ((left && right) || (top && bottom)) return "휴대폰을 조금 뒤로 빼 주세요"
            val parts = listOf("왼쪽" to left, "오른쪽" to right, "위" to top, "아래" to bottom).filter { it.second }.map { it.first }
            val josa = if (parts.last().endsWith("쪽")) "으로" else "로"
            return "휴대폰을 조금 ${parts.joinToString(" ")}$josa"
        }
    }
}
