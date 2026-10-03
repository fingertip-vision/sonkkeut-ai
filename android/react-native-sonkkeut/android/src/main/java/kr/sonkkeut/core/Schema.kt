package kr.sonkkeut.core

/**
 * 기능 사이를 오가는 데이터 (기능 명세서 6장, 파이썬 sonkkeut_vision/schema.py와 같은 형식)
 * 좌표는 모두 '화면 평면 기준 0~1 비율'이다.
 */

val KINDS = listOf("tab", "menu", "price", "button", "back")

class Element(
    val id: String,
    val kind: String,
    var box: Box,
    var conf: Double,
    var parent: String? = null,
    var text: String? = null,
    var price: Int? = null,
    var confOcr: Double? = null,
    var uncertain: Boolean? = null,
    var ocrSource: String? = null,
    var qty: Int? = null,
) {
    fun copy(conf: Double = this.conf) = Element(id, kind, box, conf, parent, text, price, confOcr, uncertain, ocrSource, qty)

    fun toMap(): Map<String, Any?> = buildMap {
        put("id", id)
        put("kind", kind)
        put("box", box.toList().map { round4(it) })
        put("conf", round3(conf))
        parent?.let { put("parent", it) }
        text?.let { put("text", it) }
        price?.let { put("price", it) }
        confOcr?.let { put("conf_ocr", round4(it)) }
        uncertain?.let { put("uncertain", it) }
        ocrSource?.let { put("ocr_source", it) }
        qty?.let { put("qty", it) }
    }
}

class PlaneState(
    val corners: List<Pt>, // 카메라 픽셀, 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래
    val H: Mat3, // 카메라 픽셀 → 화면 0~1
    val conf: Double,
    val aspect: Double,
    val source: String, // detect | track
    val frameIdx: Int = 0,
) {
    var reframed = false
    var refPrev: PlaneState? = null
    private val hInv by lazy { H.inverse() }

    fun toScreen(p: Pt) = H.apply(p)
    fun toImage(p: Pt) = hInv.apply(p)

    /** 펼친 화면 이미지 크기 (W, H) */
    fun flatSize(longSide: Int = 960): Pair<Int, Int> =
        if (aspect >= 1) longSide to maxOf(1, Math.round(longSide / aspect).toInt())
        else maxOf(1, Math.round(longSide * aspect).toInt()) to longSide

    companion object {
        fun of(corners: List<Pt>, conf: Double, source: String, frameIdx: Int): PlaneState {
            val (h, a) = homographyFromCorners(corners)
            return PlaneState(corners, h, conf, a, source, frameIdx)
        }
    }
}

class Fingertip(
    val pos: Pt?, // 화면 0~1, 손을 못 찾으면 null
    val conf: Double = 0.0,
    val imagePx: Pt? = null,
    val lostFor: Double = 0.0,
    val insideScreen: Boolean = false,
    val pointing: Double = 1.0,
)

/**
 * 안내 이벤트 (임현승의 음성·진동 출력이 받는다)
 * type: direction | press | hold | reset | no_hand | point
 */
class GuidanceEvent(
    val type: String,
    val targetId: String?,
    val dir: String? = null,
    val distance: String? = null, // far | near | reach
    val speak: String? = null,
    val vibeHz: Double = 0.0,
    val error: Pt? = null,
) {
    fun toMap(): Map<String, Any?> = buildMap {
        put("type", type)
        targetId?.let { put("target_id", it) }
        dir?.let { put("dir", it) }
        distance?.let { put("distance", it) }
        speak?.let { put("speak", it) }
        put("vibe_hz", vibeHz)
        error?.let { put("error", listOf(round4(it.x), round4(it.y))) }
    }
}

class Verdict(val result: String, val reason: String, val speak: String, val extra: Map<String, Any?> = emptyMap()) {
    fun toMap(): Map<String, Any?> = mapOf("result" to result, "reason" to reason, "speak" to speak, "extra" to extra)
}

/** 화면 구조 (6장). 언어 쪽(F-04·F-05)이 screenType·text·price·cartCount·selected를 채운다. */
class ScreenStructure(
    val screenType: String,
    val keyframeId: Int,
    val elements: List<Element>,
    val cartCount: Int? = null,
    val selected: List<String>? = null,
    val totalPrice: Int? = null,
) {
    fun toMap(): Map<String, Any?> = buildMap {
        put("screen_type", screenType)
        put("keyframe_id", keyframeId)
        put("elements", elements.map { it.toMap() })
        cartCount?.let { put("cart_count", it) }
        selected?.let { put("selected", it) }
        totalPrice?.let { put("total_price", it) }
    }

    companion object {
        /** F-05가 없을 때의 기본 구조 (요소만, 화면 종류 unknown) */
        fun basic(elements: List<Element>, keyframeId: Int) = ScreenStructure("unknown", keyframeId, elements)
    }
}

/** 누른 뒤 기대 결과 (verify.py 머리말 참고) */
class Expect(
    val screenType: String? = null,
    val screenTypeNot: String? = null,
    val cartDelta: Int? = null,
    val selected: String? = null,
    val changed: Boolean = false,
    val successSpeak: String? = null,
) {
    companion object {
        val CHANGED = Expect(changed = true)

        fun fromMap(m: Map<String, Any?>?): Expect {
            if (m == null || m.isEmpty()) return CHANGED
            return Expect(
                screenType = m["screen_type"] as String?,
                screenTypeNot = m["screen_type_not"] as String?,
                cartDelta = (m["cart_delta"] as Number?)?.toInt(),
                selected = m["selected"] as String?,
                changed = (m["changed"] as Boolean?) ?: false,
                successSpeak = m["success_speak"] as String?,
            )
        }
    }
}

internal fun round4(v: Double) = Math.round(v * 10000.0) / 10000.0
internal fun round3(v: Double) = Math.round(v * 1000.0) / 1000.0
