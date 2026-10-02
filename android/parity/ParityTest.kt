// 파이썬 기준 구현과 Kotlin 앱 코드의 동등성 테스트 (JUnit 없이 main으로 실행)
// 실행: bash android/parity/run_tests.sh

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kr.sonkkeut.core.Box
import kr.sonkkeut.core.Detection
import kr.sonkkeut.core.Element
import kr.sonkkeut.core.ElementPost
import kr.sonkkeut.core.Expect
import kr.sonkkeut.core.Fingertip
import kr.sonkkeut.core.FingertipTracker
import kr.sonkkeut.core.GrayImage
import kr.sonkkeut.core.Guide
import kr.sonkkeut.core.KeyframeDetector
import kr.sonkkeut.core.Letterbox
import kr.sonkkeut.core.PressVerifier
import kr.sonkkeut.core.Pt
import kr.sonkkeut.core.ScreenPlaneEstimator
import kr.sonkkeut.core.ScreenStructure
import kr.sonkkeut.core.YoloDecoder
import kr.sonkkeut.core.homographyFromCorners
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

var DIR = File("android/parity/golden")
val results = LinkedHashMap<String, Pair<Int, Int>>() // 이름 → (통과, 전체)
val failures = ArrayList<String>()

fun load(name: String): JsonElement = Json.parseToJsonElement(File(DIR, "$name.json").readText())
val JsonElement.d get() = jsonPrimitive.double
val JsonElement.i get() = jsonPrimitive.int
val JsonElement.s: String? get() = if (this is JsonNull) null else jsonPrimitive.content
fun JsonElement.box() = jsonArray.let { Box(it[0].d, it[1].d, it[2].d, it[3].d) }
fun JsonElement.pts() = jsonArray.map { Pt(it.jsonArray[0].d, it.jsonArray[1].d) }

fun check(group: String, ok: Boolean, msg: () -> String) {
    val (p, n) = results[group] ?: (0 to 0)
    results[group] = (p + if (ok) 1 else 0) to (n + 1)
    if (!ok && failures.size < 40) failures += "[$group] ${msg()}"
}

fun testGuide() {
    for ((ci, c) in load("guide").jsonArray.withIndex()) {
        val o = c.jsonObject
        val g = Guide()
        g.setTarget(Element("e7", "menu", o["box"]!!.box(), o["conf"]!!.d), o["aspect"]!!.d)
        for ((si, st) in o["steps"]!!.jsonArray.withIndex()) {
            val sj = st.jsonObject
            val tip = sj["tip"]!!.jsonObject
            val pos = tip["pos"]!!.let { if (it is JsonNull) null else Pt(it.jsonArray[0].d, it.jsonArray[1].d) }
            val ft = Fingertip(pos, tip["conf"]!!.d, lostFor = tip["lost_for"]!!.d, pointing = tip["pointing"]!!.d)
            val tc = sj["target_conf"]!!.let { if (it is JsonNull) null else it.d }
            val ev = g.update(ft, sj["t"]!!.d, tc)
            val exp = sj["event"]!!
            val same = if (exp is JsonNull) ev == null else {
                val e = exp.jsonObject
                ev != null && ev.type == e["type"]!!.s && ev.dir == e["dir"]!!.s && ev.distance == e["distance"]!!.s &&
                    ev.speak == e["speak"]!!.s && abs(ev.vibeHz - e["vibe_hz"]!!.d) < 1e-6
            }
            check("guide", same) { "case $ci step $si: py=$exp kt=${ev?.toMap()}" }
        }
    }
}

fun testKeyframe() {
    for ((ci, seq) in load("keyframe").jsonArray.withIndex()) {
        val kf = KeyframeDetector()
        for ((fi, f) in seq.jsonArray.withIndex()) {
            val o = f.jsonObject
            val w = o["w"]!!.i
            val h = o["h"]!!.i
            val px = o["img"]!!.jsonArray
            val img = GrayImage(w, h, FloatArray(w * h) { px[it].d.toFloat() })
            val hands = o["hands"]!!.let { if (it is JsonNull) null else it.jsonArray.map { b -> b.box() } }
            val (isKf, change) = kf.update(img, hands, o["force"]!!.jsonPrimitive.boolean)
            val since = kf.changedSinceKey(img, hands)
            check("keyframe", isKf == o["is_kf"]!!.jsonPrimitive.boolean) { "seq $ci frame $fi kf py=${o["is_kf"]} kt=$isKf" }
            check("keyframe.change", abs(change - o["change"]!!.d) < 1.0) { "seq $ci frame $fi change py=${o["change"]} kt=$change" }
            check("keyframe.since", since == o["changed_since"]!!.jsonPrimitive.boolean) { "seq $ci frame $fi since" }
        }
    }
}

fun structure(j: JsonElement): ScreenStructure {
    val o = j.jsonObject
    return ScreenStructure(o["screen_type"]!!.s!!, 0, emptyList(), o["cart_count"]?.i, o["selected"]?.jsonArray?.map { it.s!! })
}

fun testVerify() {
    for ((ci, c) in load("verify").jsonArray.withIndex()) {
        val o = c.jsonObject
        val expMap = o["expect"]!!.let { e ->
            if (e is JsonNull) null else e.jsonObject.mapValues { (_, v) ->
                val p = v.jsonPrimitive
                when {
                    p.isString -> p.content
                    p.content == "true" || p.content == "false" -> p.boolean
                    else -> p.int
                }
            }
        }
        val v = PressVerifier()
        v.arm(0.0, structure(o["before"]!!), Expect.fromMap(expMap))
        val r = v.judge(structure(o["after"]!!))
        check("verify", r.result == o["result"]!!.s && r.reason == o["reason"]!!.s && r.speak == o["speak"]!!.s) {
            "case $ci py=${o["result"]}/${o["reason"]} kt=${r.result}/${r.reason}"
        }
    }
}

fun readF32(name: String): FloatArray {
    val b = ByteBuffer.wrap(File(DIR, name).readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(b.remaining() / 4) { b.float }
}

fun testDetect() {
    for ((ci, c) in load("detect").jsonArray.withIndex()) {
        val o = c.jsonObject
        val raw = readF32(o["raw"]!!.s!!)
        val n = o["shape"]!!.jsonArray[2].i
        val lb = Letterbox.of(o["w"]!!.i, o["h"]!!.i, 640)
        val dets = YoloDecoder.decodeDetect(raw, 5, n, lb, 0.3)
        val pyBoxes = o["boxes"]!!.jsonArray.map { it.box() }
        val pyCls = o["cls"]!!.jsonArray.map { it.i }
        check("detect.count", dets.size == pyBoxes.size) { "case $ci count py=${pyBoxes.size} kt=${dets.size}" }
        for ((k, pb) in pyBoxes.withIndex()) {
            val m = dets.minByOrNull { d -> abs(d.box.x1 - pb.x1) + abs(d.box.y1 - pb.y1) + abs(d.box.x2 - pb.x2) + abs(d.box.y2 - pb.y2) }
            val err = m?.let { maxOf(abs(it.box.x1 - pb.x1), abs(it.box.y1 - pb.y1), abs(it.box.x2 - pb.x2), abs(it.box.y2 - pb.y2)) } ?: 1e9
            check("detect.box", err < 0.5 && m!!.cls == pyCls[k]) { "case $ci box $k err ${"%.2f".format(err)}px" }
        }
    }
}

fun testPose() {
    for ((ci, c) in load("pose").jsonArray.withIndex()) {
        val o = c.jsonObject
        val raw = readF32(o["raw"]!!.s!!)
        val n = o["shape"]!!.jsonArray[2].i
        val lb = Letterbox.of(o["w"]!!.i, o["h"]!!.i, 640)
        val d = YoloDecoder.decodePoseBest(raw, 4, n, lb, 0.25)
        val kp = o["kp"]!!.pts()
        val kc = o["kc"]!!.jsonArray.map { it.d }
        val err = if (d == null) 1e9 else (0 until 4).maxOf { (d.keypoints!![it] - kp[it]).norm() }
        val cerr = if (d == null) 1.0 else (0 until 4).maxOf { abs(d.kpConf!![it] - kc[it]) }
        check("pose", err < 0.5 && cerr < 1e-3 && abs(d!!.score - o["score"]!!.d) < 1e-4) { "case $ci kp err ${"%.3f".format(err)} conf err $cerr" }
    }
}

fun testElements() {
    for ((ci, c) in load("elements").jsonArray.withIndex()) {
        val o = c.jsonObject
        val dets = o["dets"]!!.jsonArray.map { d ->
            val dj = d.jsonObject
            Detection(dj["box"]!!.box(), dj["cls"]!!.i, dj["score"]!!.d)
        }
        val els = ElementPost.toElements(dets, o["w"]!!.i, o["h"]!!.i, o["margin"]!!.d)
        val py = o["elements"]!!.jsonArray
        var ok = els.size == py.size
        if (ok) for ((k, e) in els.withIndex()) {
            val p = py[k].jsonObject
            val pb = p["box"]!!.box()
            if (e.id != p["id"]!!.s || e.kind != p["kind"]!!.s || e.parent != p["parent"]!!.s ||
                maxOf(abs(e.box.x1 - pb.x1), abs(e.box.y2 - pb.y2)) > 1e-9) ok = false
        }
        check("elements", ok) { "case $ci py=${py.map { it.jsonObject["id"]!!.s + ":" + it.jsonObject["kind"]!!.s }} kt=${els.map { it.id + ":" + it.kind }}" }
    }
}

fun testGeometry() {
    val g = load("geometry").jsonObject
    for ((ci, h) in g["homography"]!!.jsonArray.withIndex()) {
        val o = h.jsonObject
        val (H, aspect) = homographyFromCorners(o["corners"]!!.pts())
        val py = o["H"]!!.jsonArray.flatMap { r -> r.jsonArray.map { it.d } }
        val err = (0 until 9).maxOf { abs(H.m[it] - py[it]) / maxOf(1e-6, abs(py[it])) }
        check("geometry.homography", err < 1e-5 && abs(aspect - o["aspect"]!!.d) < 1e-6) { "case $ci rel err $err" }
    }
    for ((ci, h) in g["hints"]!!.jsonArray.withIndex()) {
        val o = h.jsonObject
        val hint = ScreenPlaneEstimator.missingHint(o["kp"]!!.pts(), o["kc"]!!.jsonArray.map { it.d }, o["w"]!!.i, o["h"]!!.i)
        check("geometry.hint", hint == o["hint"]!!.s) { "case $ci py=${o["hint"]} kt=$hint" }
    }
    for ((ci, h) in g["pointing"]!!.jsonArray.withIndex()) {
        val o = h.jsonObject
        val v = FingertipTracker.pointingScore(o["points"]!!.pts())
        check("geometry.pointing", abs(v - o["score"]!!.d) < 1e-5) { "case $ci py=${o["score"]} kt=$v" }
    }
}

fun main(args: Array<String>) {
    if (args.isNotEmpty()) DIR = File(args[0])
    val tests = listOf(::testGuide, ::testKeyframe, ::testVerify, ::testDetect, ::testPose, ::testElements, ::testGeometry)
    for (t in tests) {
        try {
            t()
        } catch (e: Throwable) {
            failures += "${t.name} 예외: $e"
            results[t.name] = 0 to 1
        }
    }
    println("\n파이썬 ↔ Kotlin 동등성 테스트")
    for ((k, v) in results) println("  %-22s %5d / %-5d %s".format(k, v.first, v.second, if (v.first == v.second) "통과" else "실패"))
    if (failures.isNotEmpty()) {
        println("\n실패 예시:")
        failures.take(15).forEach { println("  $it") }
    }
    val allOk = results.values.all { it.first == it.second } && failures.isEmpty()
    println(if (allOk) "\n모두 통과" else "\n실패 있음")
    if (!allOk) kotlin.system.exitProcess(1)
}
