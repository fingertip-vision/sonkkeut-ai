package kr.sonkkeut.core

/** Keep numeric option/quantity buttons as text; normalize only plausible won amounts. */
object OcrText {
    private val standalone = Regex("^[+＋]?\\s*[₩W#\\\\]?\\s*[\\dOoIl,.\\s]{2,}\\s*(?:원|won)?$", RegexOption.IGNORE_CASE)
    private val embedded = Regex("[₩W#\\\\]?\\s*\\d[\\d,.]{2,}\\s*원?")
    fun price(text: String): Int? {
        val compact = text.trim().replace(" ", "").trimEnd(',', '.')
        if (!standalone.matches(compact) || compact.none { it.isDigit() }) return null
        val digits = compact.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').filter { it in '0'..'9' }
        return digits.toIntOrNull()?.takeIf { it in 100..1_000_000 }
    }
    fun findPrice(text: String): Int? = embedded.findAll(text).mapNotNull { price(it.value) }.firstOrNull()
    fun quantity(text: String): Int? = Regex("[xX×]\\s*([0-9]{1,2})\\s*$").find(text)?.groupValues?.get(1)?.toIntOrNull()
    /** Pass only independently trusted actionable button labels, never the page body. */
    fun hasMethodButtons(labels: List<String>): Boolean {
        val exact = labels.map { it.replace(Regex("\\s+"), "") }.toSet()
        return exact.any { it in setOf("매장", "매장에서먹기") } && exact.any { it in setOf("포장", "포장하기") }
    }
}
