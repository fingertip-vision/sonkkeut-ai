package kr.sonkkeut.android

import java.text.Normalizer
import java.util.Locale
import kotlin.math.abs

data class MenuDocument(val name: String, val aliases: List<String> = emptyList(), val soldOut: Boolean = false,
                        val category: String = "", val price: Int? = null,
                        val relatedTerms: List<String> = emptyList(), val description: String = "",
                        val id: String = "")
data class MenuCandidate(val menu: MenuDocument, val score: Double)
data class SpeechCorrection(val start: Int, val end: Int, val replacement: String)
data class SpeechAmbiguity(val start: Int, val end: Int, val original: String, val candidates: List<MenuCandidate>)
data class MenuSpeechResult(val original: String, val text: String, val corrections: List<SpeechCorrection>, val ambiguities: List<SpeechAmbiguity>)

/** Pure Kotlin counterpart of the scoped menu retrieval used by SDK 0.1.4. */
class MenuSpeechIndex(documents: List<MenuDocument>) {
    private data class Entry(val document: MenuDocument, val term: String, val canonical: Boolean, val phonetic: String)
    private data class Hit(val start: Int, val end: Int, val priority: Double, val candidates: List<MenuCandidate>)
    private val entries = documents.flatMap { d -> (listOf(d.name) + d.aliases).distinct().map { name ->
        val term = normalize(name); Entry(d, term, name == d.name, jamo(term))
    } }.filter { it.term.length in 1..80 }
    init { require(documents.size <= 1000) }
    fun search(query: String): List<MenuCandidate> {
        val term = normalize(query)
        if (term.isEmpty()) return emptyList()
        val phonetic = jamo(term)
        return entries.mapNotNull { e ->
            if (abs(term.length - e.term.length) > 1) return@mapNotNull null
            val exact = term == e.term
            if (!exact && (term.length < 3 || e.term.length < 3 || !e.canonical)) return@mapNotNull null
            val score = if (exact) 1.0 else maxOf(
                1.0 - distance(phonetic, e.phonetic).toDouble() / maxOf(phonetic.length, e.phonetic.length),
                if (term.length >= 4 && e.term.length >= 4 && distance(term, e.term) == 1) .90 else 0.0)
            if (score < .78) null else MenuCandidate(e.document, score)
        }.groupBy { it.menu.name }.map { (_, values) -> values.maxBy { it.score } }
            .sortedWith(compareByDescending<MenuCandidate> { it.score }.thenBy { it.menu.name }).take(3)
    }
    fun correct(original: String): MenuSpeechResult {
        require(original.length <= 500) { "주문은 500자 이내로 말씀해 주세요." }
        val positions = original.indices.filter { !original[it].isWhitespace() && original[it] !in ".,!?·" }
        val compact = positions.joinToString("") { original[it].lowercaseChar().toString() }
        val lengths = entries.flatMap { listOf(it.term.length - 1, it.term.length, it.term.length + 1) }.filter { it >= 2 }.toSet()
        val hits = mutableListOf<Hit>()
        for (start in compact.indices) for (length in lengths) {
            if (start + length > compact.length) continue
            val fragment = compact.substring(start, start + length)
            val candidates = search(fragment)
            val top = candidates.firstOrNull() ?: continue
            val canonical = entries.any { it.term == fragment && it.document.name == top.menu.name && it.canonical }
            val prefix = compact.substring(0, start)
            val boundary = start == 0 || Regex("(?:아이스|따뜻한|차가운|그리고|하고|이랑|잔|개)$").containsMatchIn(prefix)
                || original.getOrNull(positions[start] - 1)?.let { it.isWhitespace() || it == ',' } == true
            if (top.score == 1.0 && !canonical && !boundary || top.score < .86) continue
            if (top.score < 1.0 && !boundary) continue
            hits += Hit(start, start + length, if (canonical) 2 + length / 100.0 else top.score + length / 200.0, candidates)
        }
        val selected = mutableListOf<Hit>()
        for (hit in hits.sortedByDescending { it.priority }) if (selected.none { it.start < hit.end && hit.start < it.end }) selected += hit
        val corrections = mutableListOf<SpeechCorrection>()
        val ambiguities = mutableListOf<SpeechAmbiguity>()
        for (hit in selected.sortedBy { it.start }) {
            val start = positions[hit.start]; val end = positions[hit.end - 1] + 1
            val top = hit.candidates[0]
            if (top.score - (hit.candidates.getOrNull(1)?.score ?: 0.0) < .06 || top.score < .88) {
                ambiguities += SpeechAmbiguity(start, end, original.substring(start, end), hit.candidates)
            } else if (normalize(original.substring(start, end)) != normalize(top.menu.name)
                && entries.none { !it.canonical && it.term == normalize(original.substring(start, end)) && it.document.name == top.menu.name }) {
                corrections += SpeechCorrection(start, end, top.menu.name)
            }
        }
        for (match in Regex("(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|하나|둘|셋|넷|\\d+)\\s*(찬|쟌|젠)(?=\\s|포장|주세요|$)").findAll(original)) {
            val end = match.range.last + 1
            if (selected.none { positions[it.start] < end && match.range.first < positions[it.end - 1] + 1 })
                corrections += SpeechCorrection(match.range.first, end, "${match.groupValues[1]} 잔")
        }
        var text = original
        for (change in corrections.sortedByDescending { it.start }) text = text.substring(0, change.start) + change.replacement + text.substring(change.end)
        return MenuSpeechResult(original, text, corrections.sortedBy { it.start }, ambiguities)
    }
    companion object {
        fun normalize(text: String) = Normalizer.normalize(text, Normalizer.Form.NFKC).lowercase(Locale.ROOT).replace(Regex("[\\s.,!?·]"), "")
        private fun jamo(text: String): String = buildString {
            val cho = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"; val jung = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"; val jong = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
            for (ch in text) { val n = ch.code - 0xac00; if (n in 0..11171) { append(cho[n / 588]); append(jung[n % 588 / 28]); if (n % 28 != 0) append(jong[n % 28]) } else append(ch) }
        }
        private fun distance(a: String, b: String): Int {
            var row = IntArray(b.length + 1) { it }
            for (i in 1..a.length) { val next = IntArray(b.length + 1); next[0] = i
                for (j in 1..b.length) next[j] = minOf(next[j - 1] + 1, row[j] + 1, row[j - 1] + if (a[i - 1] == b[j - 1]) 0 else 1)
                row = next
            }; return row[b.length]
        }
    }
}
