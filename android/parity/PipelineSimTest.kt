// Kotlin VisionPipeline 전체 흐름 시뮬레이션 테스트 (모델·영상처리는 정답을 돌려주는 가짜로 대체)
//
// 파이썬 sim.run_scenario와 같은 시나리오를 Kotlin에서 돌린다:
//   메뉴 화면 → 목표 지정 → 가짜 사용자가 방향 안내를 (조금 부정확하게) 따름 → "지금 누르세요"
//   → 0.2초 뒤 키오스크 화면이 바뀜 → 누름 결과 '성공'
// 가짜 M1은 정답 꼭짓점에 화면 폭의 ±1.5% 잡음을, 가짜 M2는 정답 버튼 위치를 '펼친 화면 좌표'로 바꿔 돌려준다.
// 그래서 레터박스·펼치기 행렬·margin·좌표 변환·재검출·키프레임·판정 로직이 모두 실제로 실행된다.

import kr.sonkkeut.core.Box
import kr.sonkkeut.core.FrameImage
import kr.sonkkeut.core.GrayFrame
import kr.sonkkeut.core.GrayImage
import kr.sonkkeut.core.Hand
import kr.sonkkeut.core.HandSource
import kr.sonkkeut.core.ImageOps
import kr.sonkkeut.core.Letterbox
import kr.sonkkeut.core.Mat3
import kr.sonkkeut.core.Pt
import kr.sonkkeut.core.TensorModel
import kr.sonkkeut.core.TrackPoints
import kr.sonkkeut.core.TrackResult
import kr.sonkkeut.core.UNIT_SQUARE
import kr.sonkkeut.core.VisionPipeline
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin
import kotlin.random.Random

const val FW = 960
const val FH = 1280

class Screen(val id: Int, val boxes: List<Pair<String, Box>>)

fun makeScreen(id: Int, rng: Random): Screen {
    val b = ArrayList<Pair<String, Box>>()
    b += "back" to Box(0.02, 0.015, 0.19, 0.063)
    for (i in 0 until 4) b += "tab" to Box(0.01 + i * 0.247, 0.09, 0.247 + i * 0.247, 0.153)
    val rows = 3 + rng.nextInt(2)
    for (r in 0 until rows) for (c in 0 until 2) {
        val x1 = 0.02 + c * 0.49
        val y1 = 0.17 + r * (0.62 / rows)
        b += "menu" to Box(x1, y1, x1 + 0.46, y1 + 0.62 / rows - 0.015)
        b += "price" to Box(x1 + 0.15, y1 + 0.62 / rows - 0.06, x1 + 0.31, y1 + 0.62 / rows - 0.03)
    }
    b += "button" to Box(0.33, 0.9, 0.53, 0.98)
    b += "button" to Box(0.55, 0.9, 0.75, 0.98)
    b += "button" to Box(0.77, 0.9, 0.97, 0.98)
    return Screen(id, b)
}

class Camera(seed: Int) {
    private val rng = Random(seed)
    private val jit = List(4) { Pt(rng.nextDouble(-0.08, 0.08), rng.nextDouble(-0.08, 0.08)) }
    private val ph = List(3) { rng.nextDouble(0.0, 6.28) }
    fun corners(t: Double): List<Pt> {
        val cw = 0.8 * FW * 0.62
        val ch = cw * 16 / 9
        val base = listOf(Pt(-cw / 2, -ch / 2), Pt(cw / 2, -ch / 2), Pt(cw / 2, ch / 2), Pt(-cw / 2, ch / 2))
            .mapIndexed { i, p -> p + Pt(jit[i].x * cw, jit[i].y * ch) }
        val wob = Pt(sin(0.9 * t + ph[0]), cos(0.7 * t + ph[1])) * 6.0
        val rot = Math.toRadians(2.0 * sin(0.5 * t + ph[2]))
        return base.map { Pt(it.x * cos(rot) - it.y * sin(rot), it.x * sin(rot) + it.y * cos(rot)) + Pt(FW / 2.0, FH / 2.0) + wob }
    }
}

/** 정답 정보를 들고 다니는 가짜 프레임 */
class SimFrame(val screen: Screen, val corners: List<Pt>, val hand: List<Pt>?, override val width: Int = FW, override val height: Int = FH) : FrameImage {
    val scr2img: Mat3 = Mat3.perspective(UNIT_SQUARE, corners)
}

/** 펼친 이미지: 원본 프레임 + (카메라 픽셀 → 펼친 픽셀) 행렬 */
class SimFlat(val src: SimFrame, val H: Mat3, override val width: Int, override val height: Int) : FrameImage
class SimGray(val f: SimFrame) : GrayFrame
class SimPts(override val count: Int) : TrackPoints

class SimOps : ImageOps {
    var lastTensorFrame: FrameImage? = null
    var lastLb: Letterbox? = null
    override fun toGray(frame: FrameImage) = SimGray(frame as SimFrame)
    override fun letterboxTensor(frame: FrameImage, lb: Letterbox): FloatArray {
        lastTensorFrame = frame; lastLb = lb; return FloatArray(0)
    }
    override fun warp(frame: FrameImage, H: Mat3, outW: Int, outH: Int) = SimFlat(frame as SimFrame, H, outW, outH)

    /** 화면 id마다 다른 무늬 (키프레임 판단이 화면 전환을 보도록) */
    override fun grayImage(frame: FrameImage): GrayImage {
        val f = frame as SimFlat
        val r = Random(f.src.screen.id * 7919)
        val pattern = FloatArray(16) { r.nextInt(30, 230).toFloat() }
        return GrayImage(f.width, f.height, FloatArray(f.width * f.height) { i ->
            val x = (i % f.width) * 4 / f.width
            val y = (i / f.width) * 4 / f.height
            pattern[y * 4 + x]
        })
    }
    override fun seedPoints(gray: GrayFrame, quad: List<Pt>) = SimPts(150)
    var prevFrame: SimFrame? = null
    override fun track(prev: GrayFrame, cur: GrayFrame, pts: TrackPoints, minInliers: Int): TrackResult {
        val a = (prev as SimGray).f.corners
        val b = (cur as SimGray).f.corners
        return TrackResult(Mat3.perspective(a, b), SimPts(140), 0.93)
    }
    override fun cropPatch(gray: GrayFrame, center: Pt, size: Double, k: Int) = FloatArray(64 * 64)
    override fun crop(frame: FrameImage, x1: Int, y1: Int, x2: Int, y2: Int): FrameImage = frame
}

/** 가짜 M1: 정답 꼭짓점(+잡음)을 YOLO pose 원시 출력 형식으로 */
class FakeM1(private val ops: SimOps, seed: Int, private val noise: Double = (System.getenv("M1_NOISE") ?: "0.015").toDouble()) : TensorModel {
    private val rng = Random(seed)
    override fun run(input: FloatArray, shape: LongArray): FloatArray {
        val f = ops.lastTensorFrame as SimFrame
        val lb = ops.lastLb!!
        val n = 8400
        val out = FloatArray(17 * n)
        val side = f.corners.let { (it[1] - it[0]).norm() }
        val kp = f.corners.map { it + Pt(rng.nextDouble(-1.0, 1.0), rng.nextDouble(-1.0, 1.0)) * (noise * side) }
        val li = kp.map { Pt(it.x * lb.scale + lb.padX, it.y * lb.scale + lb.padY) }
        val i = 1234
        out[i] = 320f; out[n + i] = 320f; out[2 * n + i] = 500f; out[3 * n + i] = 600f; out[4 * n + i] = 0.95f
        for (k in 0 until 4) {
            out[(5 + 3 * k) * n + i] = li[k].x.toFloat()
            out[(6 + 3 * k) * n + i] = li[k].y.toFloat()
            out[(7 + 3 * k) * n + i] = 0.99f
        }
        return out
    }
}

/** 가짜 M2: 정답 요소 박스를 '펼친 이미지' 좌표 → 레터박스 좌표로 */
class FakeM2(private val ops: SimOps) : TensorModel {
    override fun run(input: FloatArray, shape: LongArray): FloatArray {
        val flat = ops.lastTensorFrame as SimFlat
        val lb = ops.lastLb!!
        val n = 8400
        val out = FloatArray(9 * n)
        val names = listOf("tab", "menu", "price", "button", "back")
        val toFlat = flat.H * flat.src.scr2img // 화면 0~1 → 카메라 → 펼친 픽셀
        for ((i, kb) in flat.src.screen.boxes.withIndex()) {
            val (kind, b) = kb
            val q = listOf(Pt(b.x1, b.y1), Pt(b.x2, b.y1), Pt(b.x2, b.y2), Pt(b.x1, b.y2)).map { toFlat.apply(it) }
            val x1 = q.minOf { it.x } * lb.scale + lb.padX
            val x2 = q.maxOf { it.x } * lb.scale + lb.padX
            val y1 = q.minOf { it.y } * lb.scale + lb.padY
            val y2 = q.maxOf { it.y } * lb.scale + lb.padY
            out[i] = ((x1 + x2) / 2).toFloat(); out[n + i] = ((y1 + y2) / 2).toFloat()
            out[2 * n + i] = (x2 - x1).toFloat(); out[3 * n + i] = (y2 - y1).toFloat()
            out[(4 + names.indexOf(kind)) * n + i] = 0.9f
        }
        return out
    }
}

class FakeHands : HandSource {
    override fun detect(frame: FrameImage, tSeconds: Double): List<Hand> = (frame as SimFrame).hand?.let { listOf(Hand(it, 0.95)) } ?: emptyList()
}

/** 검지 끝이 tip(픽셀)에 오는 손 관절 21점 (파이썬 sim.draw_hand와 같은 모양) */
fun handPoints(tip: Pt): List<Pt> {
    val wrist = tip + Pt(60.0, 330.0)
    val p = MutableList(21) { wrist }
    for ((j, a) in (5..8).zip(listOf(0.45, 0.6333, 0.8167, 1.0))) p[j] = wrist + (tip - wrist) * a
    for ((f, off) in listOf(1, 9, 13, 17).zip(listOf(-70.0, 25.0, 50.0, 75.0))) for (k in 0 until 4) p[f + k] = wrist + Pt(off * 0.8, -80.0 - 15 * k)
    return p
}

val VEC = mapOf("right" to (1 to 0), "up_right" to (1 to -1), "up" to (0 to -1), "up_left" to (-1 to -1), "left" to (-1 to 0),
    "down_left" to (-1 to 1), "down" to (0 to 1), "down_right" to (1 to 1))

class Outcome(val pressed: Double?, val inside: Boolean?, val verdict: String?, val speaks: List<Double>, val recallOk: Boolean, val maxFrameMs: Double)

fun scenario(seed: Int): Outcome {
    val rng = Random(seed)
    val ops = SimOps()
    val pipe = VisionPipeline(FakeM1(ops, seed), FakeM2(ops), ops, FakeHands())
    val cam = Camera(seed)
    var screen = makeScreen(seed * 10, rng)
    val next = makeScreen(seed * 10 + 1, rng)
    var finger: Pt? = null
    var pressed: Double? = null
    var inside: Boolean? = null
    var verdict: String? = null
    var targetTrue: Box? = null
    var recallOk = true
    val speaks = ArrayList<Double>()
    var maxMs = 0.0
    for (i in 0 until 20 * 15) {
        val t = i / 15.0
        val cs = cam.corners(t)
        val scr2img = Mat3.perspective(UNIT_SQUARE, cs)
        val hand = finger?.let { handPoints(scr2img.apply(it)) }
        val r = pipe.process(SimFrame(screen, cs, hand), t)
        if (i > 3) maxMs = maxOf(maxMs, r.timings["total_ms"] ?: 0.0)
        if (r.keyframe && pipe.guide.target == null && pressed == null) {
            // 정답 요소와 검출 요소 개수 비교 (펼치기·margin·좌표 변환이 맞는지)
            recallOk = pipe.elements.size == screen.boxes.size
            val cands = pipe.elements.filter { it.kind == "menu" || it.kind == "button" }
            val tgt = cands[rng.nextInt(cands.size)]
            pipe.setTarget(tgt.id)
            // 추정 좌표의 목표 → 실제 화면 좌표의 정답 박스
            val c = r.plane!!.toImage(Pt(tgt.box.cx, tgt.box.cy)).let { scr2img.inverse().apply(it) }
            targetTrue = screen.boxes.filter { it.first == tgt.kind }.map { it.second }.minBy { hypot(it.cx - c.x, it.cy - c.y) }
            finger = Pt(rng.nextDouble(0.3, 0.7), 0.97)
        }
        val ev = r.event
        if (ev != null) {
            if (ev.speak != null) speaks += t
            if (ev.type == "direction" && ev.dir != null && finger != null) {
                val (vx, vy) = VEC.getValue(ev.dir!!)
                val ang = atan2(vy.toDouble(), vx.toDouble()) + Math.toRadians(rng.nextDouble(-15.0, 15.0))
                val s = (if (ev.distance == "far") 0.035 else 0.012) * rng.nextDouble(0.7, 1.3)
                finger = Pt(finger!!.x + cos(ang) * s / pipe.guide.aspect, finger!!.y + sin(ang) * s)
            }
            if (ev.type == "press" && pressed == null) {
                pressed = t
                inside = targetTrue!!.contains(finger!!.x, finger!!.y)
            }
        }
        if (pressed != null && finger != null && t - pressed!! >= 0.2) {
            screen = next; finger = null
        }
        r.verdict?.let { verdict = it.result; return Outcome(pressed, inside, verdict, speaks, recallOk, maxMs) }
    }
    return Outcome(pressed, inside, verdict, speaks, recallOk, maxMs)
}

fun main() {
    var ok = 0
    var wrong = 0
    val n = 40
    val reach = ArrayList<Double>()
    for (seed in 0 until n) {
        val o = scenario(seed)
        val good = o.pressed != null && o.verdict == "success" && o.recallOk
        if (good) ok++
        if (o.inside == false) wrong++
        o.pressed?.let { reach += it }
        val gaps = o.speaks.zipWithNext { a, b -> b - a }
        if (gaps.any { it < 0.8 - 1e-6 }) println("seed $seed: 음성 간격 0.8초 위반 $gaps")
        if (!good) println("seed $seed: 실패 pressed=${o.pressed} verdict=${o.verdict} recall=${o.recallOk}")
    }
    println("\nKotlin 파이프라인 시뮬레이션: 성공 $ok/$n, 잘못된 '누르세요' $wrong 회, 평균 도달 ${"%.1f".format(reach.average())}s")
    if (ok != n || wrong != 0) kotlin.system.exitProcess(1)
}
