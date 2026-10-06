package kr.sonkkeut.core

import java.text.Normalizer
import kotlin.math.exp
import kotlin.math.max

data class CtcReading(val text: String, val confidence: Double, val uncertain: Boolean)

/** Paddle CTC: blank is 0, dictionary[0] is class 1; collapse repeats before blank removal. */
class CtcDecoder(private val dictionary: List<String>, private val threshold: Double = 0.80) {
    val classes: Int get() = dictionary.size + 1

    fun decode(scores: FloatArray, timesteps: Int, probabilities: Boolean = true): CtcReading {
        require(timesteps > 0 && scores.size == timesteps * classes) { "OCR output and dictionary shapes differ" }
        val text = StringBuilder()
        var previous = -1
        var confidence = 0.0
        var emitted = 0
        for (t in 0 until timesteps) {
            val offset = t * classes
            var best = 0
            for (c in 0 until classes) {
                require(scores[offset + c].isFinite()) { "OCR output contains a non-finite value" }
                if (scores[offset + c] > scores[offset + best]) best = c
            }
            if (best != 0 && best != previous) {
                text.append(dictionary[best - 1])
                val value = if (probabilities) scores[offset + best].toDouble().also {
                    require(it in 0.0..1.0) { "OCR probability is outside [0,1]" }
                } else {
                    val peak = scores[offset + best].toDouble()
                    var sum = 0.0
                    for (c in 0 until classes) sum += exp(scores[offset + c].toDouble() - peak)
                    1.0 / max(sum, 1.0)
                }
                confidence += value
                emitted++
            }
            previous = best
        }
        val label = Normalizer.normalize(text.toString(), Normalizer.Form.NFC).trim()
        val score = if (emitted == 0) 0.0 else confidence / emitted
        return CtcReading(label, score, label.isBlank() || score < threshold)
    }
}
