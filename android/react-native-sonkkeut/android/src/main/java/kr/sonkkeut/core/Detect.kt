package kr.sonkkeut.core

import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8 ONNX 출력 해석 (Ultralytics의 전처리·후처리를 그대로 옮김)
 *
 * 입력: 640×640 레터박스(회색 114), RGB, 0~1, CHW
 * M2 출력: [1, 4+5, 8400]  (cx, cy, w, h, 클래스 점수 5개)
 * M1 출력: [1, 4+1+12, 8400] (cx, cy, w, h, 점수, 꼭짓점 4×(x, y, conf))
 */
class Letterbox(val scale: Double, val padX: Double, val padY: Double, val srcW: Int, val srcH: Int, val size: Int) {
    /** 이미지를 줄인 크기 (이만큼 리사이즈해 (padX, padY) 위치에 놓는다) */
    val newW get() = Math.rint(srcW * scale).toInt()
    val newH get() = Math.rint(srcH * scale).toInt()

    /** 모델 입력 픽셀 → 원본 픽셀 */
    fun toSrc(x: Double, y: Double) = Pt((x - padX) / scale, (y - padY) / scale)

    companion object {
        /** Ultralytics LetterBox(auto=False, center=True)와 같은 배치 */
        fun of(srcW: Int, srcH: Int, size: Int = 640): Letterbox {
            val r = min(size.toDouble() / srcH, size.toDouble() / srcW)
            val nw = Math.rint(srcW * r).toInt() // 파이썬 round(반올림 짝수)와 같게
            val nh = Math.rint(srcH * r).toInt()
            val dw = (size - nw) / 2.0
            val dh = (size - nh) / 2.0
            // Ultralytics: top = round(dh - 0.1), left = round(dw - 0.1)
            return Letterbox(r, Math.rint(dw - 0.1), Math.rint(dh - 0.1), srcW, srcH, size)
        }
    }
}

class Detection(val box: Box /* 원본 픽셀 */, val cls: Int, val score: Double, val keypoints: List<Pt>? = null,
                val kpConf: List<Double>? = null)

object YoloDecoder {
    /** 클래스별 NMS (Ultralytics 기본: iou 0.7, agnostic=False) */
    fun nms(dets: List<Detection>, iouThr: Double = 0.7, maxDet: Int = 300): List<Detection> {
        val sorted = dets.sortedByDescending { it.score }
        val keep = ArrayList<Detection>()
        for (d in sorted) {
            if (keep.any { it.cls == d.cls && iou(it.box, d.box) > iouThr }) continue
            keep += d
            if (keep.size >= maxDet) break
        }
        return keep
    }

    /** M2: out = [4+nc][n] (행 우선 평탄화) */
    fun decodeDetect(out: FloatArray, nc: Int, n: Int, lb: Letterbox, conf: Double, iouThr: Double = 0.7): List<Detection> {
        val dets = ArrayList<Detection>()
        for (i in 0 until n) {
            var best = -1
            var bs = 0f
            for (c in 0 until nc) {
                val s = out[(4 + c) * n + i]
                if (s > bs) { bs = s; best = c }
            }
            if (bs <= conf) continue
            dets += Detection(xywhToBox(out, n, i, lb), best, bs.toDouble())
        }
        return nms(dets, iouThr)
    }

    /** M1: out = [5+3k][n], 점수가 가장 높은 1개 (pose 머리는 꼭짓점 conf에 시그모이드가 이미 적용됨) */
    fun decodePoseBest(out: FloatArray, nk: Int, n: Int, lb: Letterbox, conf: Double): Detection? {
        var bi = -1
        var bs = conf.toFloat()
        for (i in 0 until n) {
            val s = out[4 * n + i]
            if (s > bs) { bs = s; bi = i }
        }
        if (bi < 0) return null
        val kps = ArrayList<Pt>()
        val kc = ArrayList<Double>()
        for (k in 0 until nk) {
            val x = out[(5 + 3 * k) * n + bi].toDouble()
            val y = out[(6 + 3 * k) * n + bi].toDouble()
            kps += lb.toSrc(x, y)
            kc += out[(7 + 3 * k) * n + bi].toDouble()
        }
        return Detection(xywhToBox(out, n, bi, lb), 0, bs.toDouble(), kps, kc)
    }

    private fun xywhToBox(out: FloatArray, n: Int, i: Int, lb: Letterbox): Box {
        val cx = out[i].toDouble()
        val cy = out[n + i].toDouble()
        val w = out[2 * n + i].toDouble()
        val h = out[3 * n + i].toDouble()
        val a = lb.toSrc(cx - w / 2, cy - h / 2)
        val b = lb.toSrc(cx + w / 2, cy + h / 2)
        return Box(max(0.0, a.x), max(0.0, a.y), min(lb.srcW.toDouble(), b.x), min(lb.srcH.toDouble(), b.y))
    }
}

/** F-03 후처리 (파이썬 elements.py: clean_overlaps, reading_order, parent) */
object ElementPost {
    val NAMES = listOf("tab", "menu", "price", "button", "back")

    data class Raw(val box: Box, val kind: String, val conf: Double)

    fun cleanOverlaps(dets: List<Raw>, sameIou: Double = 0.5, crossIou: Double = 0.8): List<Raw> {
        val keep = ArrayList<Raw>()
        for (d in dets.sortedByDescending { it.conf }) {
            val dup = keep.any { k ->
                val v = iou(d.box, k.box)
                (k.kind == d.kind && v >= sameIou) || (k.kind != d.kind && v >= crossIou)
            }
            if (!dup) keep += d
        }
        return keep
    }

    fun readingOrder(dets: List<Raw>, rowTol: Double = 0.5): List<Raw> {
        val sorted = dets.sortedBy { it.box.cy }
        val rows = ArrayList<List<Raw>>()
        var cur = ArrayList<Raw>()
        for (d in sorted) {
            if (cur.isNotEmpty()) {
                val refH = cur.map { it.box.h }.average()
                val refY = cur.map { it.box.cy }.average()
                if (kotlin.math.abs(d.box.cy - refY) > rowTol * refH) {
                    rows += cur; cur = ArrayList()
                }
            }
            cur += d
        }
        if (cur.isNotEmpty()) rows += cur
        return rows.flatMap { r -> r.sortedBy { it.box.x1 } }
    }

    /**
     * 펼친 화면 이미지(margin 포함, 크기 w×h)의 픽셀 검출 → 화면 0~1 좌표 Element 목록
     */
    fun toElements(dets: List<Detection>, w: Int, h: Int, margin: Double): List<Element> {
        val k = 1.0 + 2 * margin
        val raws = dets.mapNotNull { d ->
            val b = Box(d.box.x1 / w * k - margin, d.box.y1 / h * k - margin, d.box.x2 / w * k - margin, d.box.y2 / h * k - margin)
            if (b.x2 <= 0 || b.y2 <= 0 || b.x1 >= 1 || b.y1 >= 1) null // 화면 바깥(베젤·배경)
            else Raw(b, NAMES.getOrElse(d.cls) { d.cls.toString() }, d.score)
        }
        val els = readingOrder(cleanOverlaps(raws)).mapIndexed { i, r -> Element("e${i + 1}", r.kind, r.box, r.conf) }
        val menus = els.filter { it.kind == "menu" }
        for (e in els) {
            if (e.kind != "price" || menus.isEmpty()) continue
            val best = menus.maxBy { containment(e.box, it.box) }
            if (containment(e.box, best.box) > 0.6) e.parent = best.id
        }
        return els
    }
}
