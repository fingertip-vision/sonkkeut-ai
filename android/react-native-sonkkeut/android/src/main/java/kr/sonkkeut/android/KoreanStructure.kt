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
            var remainingReads = 80
            val buckets = elements.associate { it.id to mutableListOf<LineEvidence>() }
            val unowned = mutableListOf<LineEvidence>()
            // Read each full-frame line once, including lines missed by the synthetic M2 detector.
            for (line in lines.take(160)) {
                val b = line.boundingBox ?: continue
                val x1 = (b.left - 2).coerceIn(0, mat.cols() - 1)
                val y1 = (b.top - 2).coerceIn(0, mat.rows() - 1)
                val x2 = (b.right + 2).coerceIn(x1 + 1, mat.cols())
                val y2 = (b.bottom + 2).coerceIn(y1 + 1, mat.rows())
                val crop = mat.submat(Rect(x1, y1, x2 - x1, y2 - y1))
                val reading = try {
                    if (remainingReads-- > 0) runCatching { custom.recognize(crop) }.getOrNull() else null
                } finally { crop.release() }
                val box = Box(b.left.toDouble()/flat.width*scale-margin, b.top.toDouble()/flat.height*scale-margin,
                    b.right.toDouble()/flat.width*scale-margin, b.bottom.toDouble()/flat.height*scale-margin)
                if (!OcrEvidence.valid(box)) continue
                val evidence = OcrEvidence.line(line.text, box, reading)
                val owner = OcrEvidence.owner(box, elements)
                if (owner == null) unowned.add(evidence) else buckets.getValue(owner.id).add(evidence)
            }
            for (el in elements) {
                val matched = buckets[el.id].orEmpty().sortedWith(compareBy({ it.box.y1 }, { it.box.x1 }))
                val readings = matched.map { Evidence(it.text, it.confidence, it.uncertain, !it.uncertain) }.toMutableList()
                // M2 crops already reverse the flat-image 0.04 margin and add 0.008 crop padding.
                if (readings.isEmpty()) (crops[el.id] as? MatFrame)?.let { crop ->
                    if (remainingReads-- > 0) runCatching { custom.recognize(crop.mat) }.getOrNull()?.let {
                        // A whole-card crop is not an independent line-level agreement.
                        readings.add(Evidence(it.text, it.confidence, true, true))
                    }
                }
                if (matched.any { !it.uncertain && it.text.contains("선택됨") }) selected.add(el.id)
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
                el.price = priceLines.firstOrNull { !it.uncertain }?.let { OcrText.price(it.label) }
                    ?: readings.filter { !it.uncertain }.firstNotNullOfOrNull { OcrText.findPrice(it.label) }
                el.qty = quantity
                val identityEvidence = if(el.kind == "price") readings else textLines
                el.confOcr = identityEvidence.minOfOrNull { it.score } ?: 0.0
                el.uncertain = identityEvidence.isEmpty() || identityEvidence.any { it.uncertain } || (label.isBlank() && el.price == null)
                el.ocrSource = if (identityEvidence.isNotEmpty() && identityEvidence.all { !it.uncertain }) "mlkit_m3_consensus" else "ocr_unconfirmed"
            }
            val navigationBlocked = OcrEvidence.blockCategoryNavigation(elements, unowned)
            if (navigationBlocked) elements.filter { it.kind == "tab" ||
                (it.kind == "button" && OcrEvidence.normalized(it.text.orEmpty()) in listOf("다음", "다음페이지", "더보기")) }
                .forEach { it.uncertain = true }
            val enriched = elements + OcrEvidence.readOnlyRegions(unowned, keyframeId)
            val structuralText = text?.text ?: elements.joinToString(" ") { it.text.orEmpty() }
            val all = structuralText.replace(" ", "")
            val methodButtons = elements.filter {
                it.kind == "button" && it.conf >= 0.5 && it.uncertain == false && (it.confOcr ?: 0.0) >= 0.80
            }.mapNotNull { it.text }
            val type = when {
                Regex("카드를?(넣|삽입|대)|결제진행중|결제완료|카드결제화면").containsMatchIn(all) -> "payment"
                Regex("장바구니|주문내역").containsMatchIn(all) && Regex("총.?금액|합계|총액").containsMatchIn(all) -> "cart"
                Regex("담기|추가하기").containsMatchIn(all) && Regex("온도|HOT|ICE|따뜻|차갑|사이즈|크기|옵션선택|맵기|당도|토핑|굽기|추가선택").containsMatchIn(all) -> "option"
                OcrText.hasMethodButtons(methodButtons) -> "method"
                elements.any { it.kind == "menu" && !it.text.isNullOrBlank() && it.uncertain != true } -> "menu"
                Regex("주문시작|시작하기").containsMatchIn(all) -> "start"
                else -> "unknown"
            }
            val count = Regex("(?:장바구니|총|수량)\\s*(\\d+)\\s*(?:개|잔)").find(structuralText)?.groupValues?.get(1)?.toIntOrNull()
            val total = Regex("(?:총\\s*금액|합계|총액)\\s*([0-9][0-9,]*)\\s*원").find(structuralText)?.groupValues?.get(1)?.replace(",", "")?.toIntOrNull()
            return ScreenStructure(type, keyframeId, enriched, count, selected, total,
                if(navigationBlocked) "menu_ocr_incomplete" else if(lines.isEmpty()) "no_text" else "read",
                mapOf("ocr_lines" to lines.size, "unassigned_lines" to unowned.size,
                    "unconfirmed_menu_regions" to elements.count { it.kind == "menu" && it.uncertain != false },
                    "category_navigation_blocked" to navigationBlocked))
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
