package kr.sonkkeut.core

/**
 * 영상 쪽 전체 흐름 (파이썬 pipeline.py)
 *
 *   매 프레임  : F-02 화면 평면(추적) → F-08 손끝 → 키프레임 판단 → F-09 오차 계산 → F-10 대기
 *   키프레임만 : F-03 화면 요소 → (언어 쪽 F-04·F-05) → 목표 버튼 좌표 갱신 / F-10 판정
 */
class FrameResult {
    var plane: PlaneState? = null
    var hint: String? = null
    var tip: Fingertip? = null
    var keyframe = false
    var keyframeId = 0
    var structure: ScreenStructure? = null
    var event: GuidanceEvent? = null
    var verdict: Verdict? = null
    var targetMissing = false
    val timings = LinkedHashMap<String, Double>()

    /** React Native로 넘길 형태 */
    fun toMap(): Map<String, Any?> = buildMap {
        put("found", plane != null)
        hint?.let { put("hint", it) }
        put("keyframe", keyframe)
        put("keyframe_id", keyframeId)
        structure?.let { put("structure", it.toMap()) }
        event?.let { put("event", it.toMap()) }
        verdict?.let { put("verdict", it.toMap()) }
        if (targetMissing) put("target_missing", true)
        tip?.pos?.let { put("tip", listOf(round4(it.x), round4(it.y))) }
        tip?.let { put("tip_conf",round3(it.conf)); put("tip_pointing",round3(it.pointing)) }
        plane?.let { p -> put("corners", p.corners.flatMap { listOf(it.x, it.y) }) }
        put("timings", timings.mapValues { Math.round(it.value * 10) / 10.0 })
    }
}

class VisionPipeline(
    m1: TensorModel,
    private val m2: TensorModel,
    private val ops: ImageOps,
    hands: HandSource,
    refiner: TensorModel? = null,
    private val structureProvider: StructureProvider = StructureProvider { els, _, _, kid -> ScreenStructure.basic(els, kid) },
    val longSide: Int = 960,
    val margin: Double = 0.04,
    val elemConf: Double = 0.3,
) {
    val plane = ScreenPlaneEstimator(m1, ops, refiner?.let { CornerRefiner(it, ops) })
    val tracker = FingertipTracker(hands)
    val kf = KeyframeDetector()
    val guide = Guide()
    val verifier = PressVerifier()
    var elements: List<Element> = emptyList()
        private set
    /** elements를 읽은 키프레임 번호 (id e1, e2…는 키프레임마다 새로 매겨진다) */
    var elementsKeyframeId = 0
        private set
    var structure: ScreenStructure? = null
        private set
    private var expect: Expect? = null
    private var forceKf = false
    private var lastForced = -1e9

    // ---------- 언어 쪽(F-07)·앱에서 부르는 API ----------
    /**
     * keyframeId: elementId를 고른 화면 구조의 keyframe_id. 그 사이 화면을 다시 읽었다면 같은 id가 다른 버튼일 수 있으므로
     * 거부한다. 넘기지 않으면 예전처럼 검사하지 않는다.
     */
    fun setTarget(elementId: String, expect: Expect? = null, keyframeId: Int? = null) {
        if (keyframeId != null && keyframeId != elementsKeyframeId) {
            throw NoSuchElementException("화면을 다시 읽었습니다 (고른 화면 $keyframeId, 현재 $elementsKeyframeId). 새 화면 구조에서 다시 골라 주세요")
        }
        val el = elements.firstOrNull { it.id == elementId } ?: throw NoSuchElementException("현재 화면에 $elementId 가 없습니다")
        require(el.uncertain != true && (el.confOcr?.let { it.isFinite() && it >= 0.8 } ?: true)) {
            "글자를 확실하게 읽지 못한 요소에는 안내할 수 없습니다"
        }
        guide.setTarget(el, plane.state?.aspect)
        this.expect = expect
        verifier.disarm()
    }

    /** 화면 좌표(0~1)를 눌러 목표 지정 (데모·저시력 모드용). 겹치면 가장 작은 요소 */
    fun setTargetAt(x: Double, y: Double, expect: Expect? = null): Element? {
        val hit = elements.filter { it.uncertain != true && (it.confOcr?.let { c -> c.isFinite() && c >= 0.8 } ?: true) && it.box.contains(x, y) }
            .minByOrNull { it.box.w * it.box.h } ?: return null
        setTarget(hit.id, expect)
        return hit
    }

    fun clearTarget() {
        guide.target = null
        verifier.disarm()
    }

    fun requestKeyframe() {
        forceKf = true
    }

    // ---------- 내부 ----------
    private fun rematchTarget(handBox: Box?): Boolean {
        val tgt = guide.target ?: return false
        val best = elements.maxByOrNull { if (it.kind == tgt.kind) iou(it.box, tgt.box) else 0.0 }
        if (best != null && best.kind == tgt.kind && iou(best.box, tgt.box) > 0.5) {
            if (best.uncertain == true || (best.confOcr?.let { !it.isFinite() || it < 0.8 } ?: false)) {
                clearTarget(); return true
            }
            if (!sameLabel(tgt, best)) {
                // 같은 자리에 다른 글자의 버튼 = 화면이 바뀜 (탭 전환, 잘못 누름). 다른 버튼으로 바꿔 끼우지 않는다.
                guide.target = Element(tgt.id, tgt.kind, tgt.box, 0.0, text = tgt.text)
                return true
            }
            guide.updateTargetBox(best); return false
        }
        if (handBox != null && overlaps(handBox, tgt.box)) return false // 손가락이 가린 것
        guide.target = Element(tgt.id, tgt.kind, tgt.box, 0.0, text = tgt.text)
        return true
    }

    /** 목표와 같은 글자의 버튼인가. 목표에 글자가 없으면(아이콘 등) 비교할 근거가 없어 위치만 본다. */
    private fun sameLabel(target: Element, element: Element): Boolean {
        val want = target.text?.takeIf { it.isNotBlank() } ?: return true
        val got = element.text?.takeIf { it.isNotBlank() } ?: return false
        fun norm(s: String) = s.filterNot { it.isWhitespace() }.lowercase()
        return norm(want) == norm(got)
    }

    /**
     * 이전 프레임과 이어지지 않는 새 좌표계 → 읽어 둔 버튼 좌표를 버리고 다시 읽는다.
     * 목표는 신뢰도 0으로 두어("잠시 멈춰 주세요") 다시 읽은 뒤 rematchTarget으로 확인될 때까지 누르라고 하지 않는다.
     */
    private fun invalidateCoordinates() {
        elements = emptyList()
        guide.target?.let { guide.target = Element(it.id, it.kind, it.box, 0.0, text = it.text) }
        requestKeyframe()
    }

    private fun moveToNewFrame(old: PlaneState, new: PlaneState) {
        fun conv(b: Box): Box {
            val q = listOf(Pt(b.x1, b.y1), Pt(b.x2, b.y1), Pt(b.x2, b.y2), Pt(b.x1, b.y2)).map { new.toScreen(old.toImage(it)) }
            return Box(q.minOf { it.x }, q.minOf { it.y }, q.maxOf { it.x }, q.maxOf { it.y })
        }
        for (e in elements) e.box = conv(e.box)
        val tgt = guide.target
        if (tgt != null && elements.none { it === tgt }) tgt.box = conv(tgt.box)
        requestKeyframe()
    }

    /** 카메라 픽셀 → 펼친 이미지(margin 포함) 픽셀 */
    private fun flatMatrix(p: PlaneState, w: Int, h: Int, m: Double): Mat3 {
        val s = 1.0 + 2 * m
        return Mat3(doubleArrayOf(w / s, 0.0, w * m / s, 0.0, h / s, h * m / s, 0.0, 0.0, 1.0)) * p.H
    }

    private fun readScreen(frame: FrameImage, p: PlaneState, kid: Int, timings: MutableMap<String, Double>): ScreenStructure {
        val t0 = System.nanoTime()
        val (w, h) = p.flatSize(longSide)
        val flat = ops.warp(frame, flatMatrix(p, w, h, margin), w, h)
        val crops = LinkedHashMap<String, FrameImage>()
        try {
        val t1 = System.nanoTime()
        val lb = Letterbox.of(w, h, 640)
        val out = m2.run(ops.letterboxTensor(flat, lb), longArrayOf(1, 3, 640, 640))
        val dets = YoloDecoder.decodeDetect(out, 5, out.size / 9, lb, elemConf)
        val els = ElementPost.toElements(dets, w, h, margin)
        timings["flatten_ms"] = (t1 - t0) / 1e6
        timings["m2_ms"] = (System.nanoTime() - t1) / 1e6
        val k = 1.0 + 2 * margin
        val pad = 0.008
        for (e in els) {
            val x1 = ((e.box.x1 + margin) / k - pad) * w
            val y1 = ((e.box.y1 + margin) / k - pad) * h
            val x2 = ((e.box.x2 + margin) / k + pad) * w
            val y2 = ((e.box.y2 + margin) / k + pad) * h
            val left = x1.toInt().coerceIn(0, w)
            val top = y1.toInt().coerceIn(0, h)
            val right = x2.toInt().coerceIn(0, w)
            val bottom = y2.toInt().coerceIn(0, h)
            if (right > left && bottom > top) crops[e.id] = ops.crop(flat, left, top, right, bottom)
        }
        val s = structureProvider.build(els, crops, flat, kid)
        // 언어 쪽이 텍스트를 채워 돌려줬다면 요소에도 반영
        val byId = s.elements.associateBy { it.id }
        for (e in els) byId[e.id]?.let { d ->
            e.text = d.text ?: e.text; e.price = d.price ?: e.price
            e.confOcr = d.confOcr; e.uncertain = d.uncertain; e.ocrSource = d.ocrSource; e.qty = d.qty
        }
        elements = els
        elementsKeyframeId = kid
        structure = s
        return s
        } finally {
            crops.values.forEach { ops.release(it) }
            ops.release(flat)
        }
    }

    // ---------- 매 프레임 ----------
    fun process(frame: FrameImage, t: Double): FrameResult {
        val t0 = System.nanoTime()
        val res = FrameResult()
        val (pl, hint) = plane.update(frame, t)
        res.plane = pl
        res.hint = hint
        res.timings["plane_ms"] = (System.nanoTime() - t0) / 1e6
        if (pl == null) {
            res.tip = tracker.update(frame, null, t)
            // A stale screen coordinate must never produce a press instruction.
            // Reset dwell even when the plane is lost for only a single frame.
            res.event = if (guide.target != null) guide.update(res.tip, t, targetConf = 0.0) else null
            res.timings["total_ms"] = (System.nanoTime() - t0) / 1e6
            return res
        }
        val ref = pl.refPrev
        if (pl.reframed && ref != null) moveToNewFrame(ref, pl)
        else if (pl.rebased) invalidateCoordinates()
        if (guide.target != null) guide.aspect = pl.aspect

        val t1 = System.nanoTime()
        val tip = tracker.update(frame, pl, t)
        res.tip = tip
        res.timings["tip_ms"] = (System.nanoTime() - t1) / 1e6

        val t2 = System.nanoTime()
        val (sw, sh) = pl.flatSize(96)
        val smallFlat = ops.warp(frame, flatMatrix(pl, sw, sh, 0.0), sw, sh)
        val small = ops.grayImage(smallFlat)
        ops.release(smallFlat)
        val hb = tracker.handBoxScreen(pl)
        val hands = hb?.let { listOf(it) }
        val forced = forceKf
        // 판정 대기 중의 강제 키프레임(좌표계 교체·앱 요청)은 화면이 바뀌어서 생긴 게 아니다.
        // 강제하기 전에 실제로 바뀌었는지 봐 두고, 바뀌지 않았으면 판정하지 않는다.
        val changedBefore = if (forced && verifier.armed) kf.changedSinceKey(small, hands) else true
        val (isKf, _) = kf.update(small, hands, forced)
        forceKf = false
        res.timings["keyframe_ms"] = (System.nanoTime() - t2) / 1e6
        res.keyframe = isKf
        res.keyframeId = kf.keyframeId

        if (isKf) {
            val t3 = System.nanoTime()
            val before = structure
            res.structure = readScreen(frame, pl, kf.keyframeId, res.timings)
            res.timings["read_ms"] = (System.nanoTime() - t3) / 1e6
            if (verifier.armed && changedBefore) {
                val v = verifier.judge(res.structure!!)
                res.verdict = v
                if (v.result == "success" || v.result == "restarted") guide.target = null
                else res.targetMissing = rematchTarget(hb)
            } else if (before != null) {
                res.targetMissing = rematchTarget(hb)
            }
        } else if (verifier.armed) {
            if (verifier.poll(t, kf.changedSinceKey(small, hands)) == "timeout") {
                res.verdict = verifier.timeoutVerdict()
                guide.pressedLatch = false
            }
        }

        // Let the app consume a new screen and apply its next target before guiding.
        if (!isKf && guide.target != null && !verifier.armed && res.verdict == null) {
            val ev = guide.update(tip, t)
            res.event = ev
            if (ev?.type == "press") verifier.arm(t, structure, expect)
            else if (ev?.type == "hold" && t - lastForced >= 0.7) {
                requestKeyframe(); lastForced = t
            }
        }
        res.timings["total_ms"] = (System.nanoTime() - t0) / 1e6
        return res
    }
}
