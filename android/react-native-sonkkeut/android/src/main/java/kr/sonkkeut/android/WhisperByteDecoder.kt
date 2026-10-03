package kr.sonkkeut.android

import org.json.JSONArray
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.regex.Pattern

/** Whisper's GPT-2 byte alphabet, decoded across the complete token sequence. */
class WhisperByteDecoder(vocabularyFile: File) {
    private val vocabulary = JSONArray(vocabularyFile.readText(Charsets.UTF_8))
    private val byteDecoder = buildByteDecoder()
    private val byteEncoder = byteDecoder.entries.associate { it.value to it.key.toChar() }
    private val ranks = (0 until vocabulary.length()).associateBy { vocabulary.getString(it) }
    private val pattern = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?[\\p{L}]+| ?[\\p{N}]+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")
    private fun buildByteDecoder(): Map<Int, Int> {
        val visible = ((33..126).toList() + (161..172).toList() + (174..255).toList()).toSet()
        val result = mutableMapOf<Int, Int>()
        var extended = 256
        for (byte in 0..255) result[if (byte in visible) byte else extended++] = byte
        return result
    }
    fun decode(tokenIds: IntArray): String {
        val bytes = ByteArrayOutputStream()
        for (id in tokenIds) {
            require(id in 0 until vocabulary.length()) { "Invalid model token ID" }
            val piece = vocabulary.getString(id)
            // Special/language/timestamp tokens are entries of the model's vocabulary.
            if (piece.startsWith("<|") && piece.endsWith("|>")) continue
            var offset = 0
            while (offset < piece.length) {
                val codepoint = piece.codePointAt(offset)
                bytes.write(byteDecoder[codepoint] ?: error("Invalid Whisper byte-BPE token"))
                offset += Character.charCount(codepoint)
            }
        }
        return bytes.toByteArray().toString(Charsets.UTF_8).trim()
    }
    /** Whisper/tiktoken BPE ranks are vocabulary IDs; fixed prompt hints need no remote tokenizer. */
    fun encode(text: String): IntArray {
        val ids = mutableListOf<Int>()
        val matches = pattern.matcher(text)
        while (matches.find()) {
            val pieces = matches.group().toByteArray(Charsets.UTF_8)
                .map { byteEncoder[it.toInt() and 255]!!.toString() }.toMutableList()
            while (pieces.size > 1) {
                var best = -1
                var bestRank = Int.MAX_VALUE
                for (index in 0 until pieces.size - 1) {
                    val rank = ranks[pieces[index] + pieces[index + 1]] ?: continue
                    if (rank < bestRank) { best = index; bestRank = rank }
                }
                if (best < 0) break
                pieces[best] += pieces[best + 1]
                pieces.removeAt(best + 1)
            }
            for (piece in pieces) ids.add(ranks[piece] ?: error("Invalid Whisper byte alphabet"))
        }
        return ids.toIntArray()
    }
    fun tokenPieces(ids: IntArray): Array<String> = Array(ids.size) { vocabulary.getString(ids[it]) }
}
