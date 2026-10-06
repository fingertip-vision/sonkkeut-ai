package kr.sonkkeut.android

import android.graphics.Bitmap
import android.content.Context
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import kr.sonkkeut.core.*
import org.opencv.android.Utils
import org.opencv.core.Rect
import java.util.concurrent.TimeUnit

/** M3 reads line crops; ML Kit supplies full-frame line boxes and structural evidence. */
class KoreanStructure(context: Context) : StructureProvider, AutoCloseable {
    private val recognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    private val custom = try { KioskCtcRecognizer(context) } catch (error: Throwable) {
        recognizer.close()
        throw error
    }
    @Volatile var aliases: Map<String, String> = emptyMap()

    override fun build(elements: List<Element>, crops: Map<String, FrameImage>, flat: FrameImage, keyframeId: Int): ScreenStructure {
        val mat = (flat as MatFrame).mat
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        var handedOff = false
        try {
            Utils.matToBitmap(mat, bitmap)
            // A bounded wait on the frame worker, never the UI thread. Fail closed on timeout.
            val task = recognizer.process(InputImage.fromBitmap(bitmap, 0))
            // Keep the bitmap alive until ML Kit has finished, even if await times out.
            task.addOnCompleteListener { bitmap.recycle() }
            handedOff = true
            val text = runCatching { Tasks.await(task, 2, TimeUnit.SECONDS) }.getOrNull()
            val lines = text?.textBlocks?.flatMap { it.lines }.orEmpty()
            val margin = 0.04
            val scale = 1 + 2 * margin
            val selected = mutableListOf<String>()
            val buckets = elements.associate { it.id to mutableListOf<Text.Line>() }
            // A line belongs to the smallest containing M2 box, avoiding duplicate nested reads.
            for (line in lines) {
                val b = line.boundingBox ?: continue
                val x = b.exactCenterX() / flat.width * scale - margin
                val y = b.exactCenterY() / flat.height * scale - margin
                elements.filter { it.box.contains(x.toDouble(), y.toDouble()) }
                    .minByOrNull { (it.box.x2 - it.box.x1) * (it.box.y2 - it.box.y1) }
                    ?.let { buckets[it.id]!!.add(line) }
            }
            var remainingReads = 80
            for (el in elements) {
                val matched = buckets[el.id].orEmpty().sortedWith(compareBy({ it.boundingBox?.top }, { it.boundingBox?.left }))
                val readings = matched.map { line ->
                    val box = line.boundingBox!!
                    val x1 = (box.left - 2).coerceIn(0, mat.cols() - 1)
                    val y1 = (box.top - 2).coerceIn(0, mat.rows() - 1)
                    val x2 = (box.right + 2).coerceIn(x1 + 1, mat.cols())
                    val y2 = (box.bottom + 2).coerceIn(y1 + 1, mat.rows())
                    val crop = mat.submat(Rect(x1, y1, x2 - x1, y2 - y1))
                    try {
                        val reading = if (remainingReads-- > 0) runCatching { custom.recognize(crop) }.getOrNull() else null
                        if (reading != null && !reading.uncertain) Evidence(reading.text, reading.confidence, false, true)
                        else Evidence(line.text, reading?.confidence ?: 0.0, true, false)
                    } finally { crop.release() }
                }.toMutableList()
                // M2 crops already reverse the flat-image 0.04 margin and add 0.008 crop padding.
                if (readings.isEmpty()) (crops[el.id] as? MatFrame)?.let { crop ->
                    if (remainingReads-- > 0) runCatching { custom.recognize(crop.mat) }.getOrNull()?.let {
                        readings.add(Evidence(it.text, it.confidence, it.uncertain, true))
                    }
                }
                val raw = readings.joinToString(" ") { it.label }.trim()
                if (matched.any { it.text.contains("선택됨") } || raw.contains("선택됨")) selected.add(el.id)
                val priceLines = readings.filter { OcrText.price(it.label) != null }
                var textLines = readings.filter { OcrText.price(it.label) == null }
                if (textLines.isEmpty() && el.kind != "price") textLines = readings
                val label = textLines.joinToString(" ") { it.label.replace("선택됨", "").trim() }.trim()
                val quantity = OcrText.quantity(label)
                val name = if (quantity == null) label else label.replace(Regex("[xX×]\\s*[0-9]{1,2}\\s*$"), "").trim()
                // Keep menu/option OCR evidence intact. Product identity and ambiguous aliases
                // are resolved against the whole catalog by ScreenMenuResolver in the app.
                // Quantity-bearing cart rows retain legacy canonicalization for cart auditing.
                val canonicalName = if (quantity != null) aliases[name.replace(" ", "")] ?: name else name
                el.text = if (quantity == null) canonicalName else "$canonicalName x$quantity"
                el.price = priceLines.firstOrNull()?.let { OcrText.price(it.label) } ?: OcrText.findPrice(raw)
                el.qty = quantity
                el.confOcr = readings.minOfOrNull { it.score } ?: 0.0
                el.uncertain = readings.isEmpty() || readings.any { it.uncertain } || (label.isBlank() && el.price == null)
                el.ocrSource = if (readings.isNotEmpty() && readings.all { it.custom }) "m3_kiosk_rec_v2" else "mlkit_fallback"
            }
            val structuralText = text?.text ?: elements.joinToString(" ") { it.text.orEmpty() }
            val all = structuralText.replace(" ", "")
            val methodButtons = elements.filter {
                it.kind == "button" && it.conf >= 0.5 && it.uncertain == false && (it.confOcr ?: 0.0) >= 0.80
            }.mapNotNull { it.text }
            val type = when {
                Regex("카드를?(넣|삽입|대)|결제진행중|결제완료|카드결제화면").containsMatchIn(all) -> "payment"
                Regex("장바구니|주문내역").containsMatchIn(all) && Regex("총.?금액|합계|총액").containsMatchIn(all) -> "cart"
                Regex("담기|추가하기").containsMatchIn(all) && Regex("온도|HOT|ICE|따뜻|차갑|사이즈|크기").containsMatchIn(all) -> "option"
                OcrText.hasMethodButtons(methodButtons) -> "method"
                elements.any { it.kind == "menu" && !it.text.isNullOrBlank() && it.uncertain != true } -> "menu"
                Regex("주문시작|시작하기").containsMatchIn(all) -> "start"
                else -> "unknown"
            }
            val count = Regex("(?:장바구니|총|수량)\\s*(\\d+)\\s*(?:개|잔)").find(structuralText)?.groupValues?.get(1)?.toIntOrNull()
            val total = Regex("(?:총\\s*금액|합계|총액)\\s*([0-9][0-9,]*)\\s*원").find(structuralText)?.groupValues?.get(1)?.replace(",", "")?.toIntOrNull()
            return ScreenStructure(type, keyframeId, elements, count, selected, total)
        } catch (_: Exception) {
            elements.forEach { it.confOcr = 0.0; it.uncertain = true; it.ocrSource = "mlkit_fallback" }
            return ScreenStructure.basic(elements, keyframeId)
        } finally {
            if (!handedOff) bitmap.recycle()
        }
    }
    private data class Evidence(val label: String, val score: Double, val uncertain: Boolean, val custom: Boolean)
    override fun close() { try { recognizer.close() } finally { custom.close() } }
}
