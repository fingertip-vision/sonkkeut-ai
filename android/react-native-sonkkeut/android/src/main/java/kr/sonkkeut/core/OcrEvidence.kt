package kr.sonkkeut.core

import java.text.Normalizer
import java.util.Locale

/** M3 scores are not calibrated probabilities. A confident disagreement cannot identify a button. */
data class LineEvidence(val text: String, val box: Box, val confidence: Double, val uncertain: Boolean)

object OcrEvidence {
    fun normalized(text: String) = Normalizer.normalize(text, Normalizer.Form.NFKC)
        .lowercase(Locale.ROOT).replace(Regex("[\\s.,·₩]"), "")

    fun line(primary: String, box: Box, custom: CtcReading?): LineEvidence {
        val agrees = custom != null && !custom.uncertain && custom.confidence.isFinite() &&
            custom.confidence >= .8 && normalized(primary).isNotBlank() &&
            normalized(primary) == normalized(custom.text)
        return LineEvidence(primary.trim(), box, if (agrees) custom!!.confidence else 0.0, !agrees)
    }

    /** Prefer an actionable enclosing card to a nested price box. Require substantial overlap. */
    fun owner(line: Box, elements: List<Element>): Element? {
        val candidates = elements.filter { e -> e.box.w > 0 && e.box.h > 0 && containment(line, e.box) >= .6 }
        val cards = candidates.filter { it.kind in listOf("menu", "button", "tab", "back") }
        return (cards.ifEmpty { candidates }).minByOrNull { it.box.w * it.box.h }
    }

    fun valid(box: Box) = box.toList().all { it.isFinite() && it in 0.0..1.0 } && box.w > 0 && box.h > 0

    /** Fail closed instead of walking arbitrary categories when product OCR is incomplete.
     * Extra text regions are read-only evidence, never invented physical buttons. */
    fun blockCategoryNavigation(elements: List<Element>, unowned: List<LineEvidence>): Boolean {
        if (elements.any { it.kind == "menu" && (it.uncertain != false || (it.confOcr ?: 0.0) < .8 || it.text.isNullOrBlank()) }) return true
        val textLines = unowned.filter { valid(it.box) && OcrText.price(it.text) == null && Regex("[가-힣A-Za-z]").containsMatchIn(it.text) }
        // A price near an unassigned product label is evidence of a missed card.
        val prices = unowned.filter { OcrText.findPrice(it.text) != null } + elements.filter { it.kind == "price" }
            .map { LineEvidence(it.text.orEmpty(), it.box, it.confOcr ?: 0.0, it.uncertain != false) }
        return textLines.any { line ->
            OcrText.findPrice(line.text) != null || prices.any { price ->
                price.box.cy >= line.box.cy && price.box.y1 - line.box.y2 <= .10 &&
                    kotlin.math.abs(price.box.cx - line.box.cx) <= maxOf(.06, line.box.w * .5) }
        }
    }

    fun readOnlyRegions(unowned: List<LineEvidence>, keyframeId: Int): List<Element> = unowned.take(80).mapIndexedNotNull { i, line ->
        if (!valid(line.box) || line.text.isBlank()) null else Element(
            "ocr_${keyframeId}_$i", "text", line.box, 1.0, text=line.text,
            confOcr=line.confidence, uncertain=line.uncertain, ocrSource="mlkit_m3_consensus")
    }
}
