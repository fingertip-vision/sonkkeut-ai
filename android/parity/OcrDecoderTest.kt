import kr.sonkkeut.core.CtcDecoder
import kr.sonkkeut.core.OcrText
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

fun main(args: Array<String>) {
    val decoder = CtcDecoder(listOf("가", "나", " "))
    fun frame(index: Int, probability: Float = 0.95f) = FloatArray(4) { if (it == index) probability else (1f - probability) / 3 }
    val repeated = decoder.decode(intArrayOf(1,1,0,1,2,2).flatMap { frame(it).toList() }.toFloatArray(),6)
    check(repeated.text == "가가나" && !repeated.uncertain)
    check(decoder.decode(frame(0),1).text.isEmpty())
    check(decoder.decode(frame(1,0.5f),1).uncertain)
    check(runCatching { decoder.decode(floatArrayOf(Float.NaN,0f,0f,0f),1) }.isFailure)
    check(OcrText.price("4,500원") == 4500)
    check(OcrText.price("+500") == 500)
    check(OcrText.price("2") == null)
    check(OcrText.price("가격 4500원") == null)
    check(OcrText.findPrice("합계 28,500원") == 28500)
    check(OcrText.quantity("초코라떼 ×3") == 3)
    check(OcrText.hasMethodButtons(listOf("매장", "포장")))
    check(OcrText.hasMethodButtons(listOf("매장에서 먹기", "포장하기")))
    check(!OcrText.hasMethodButtons(listOf("포장")))
    check(!OcrText.hasMethodButtons(listOf("매장 또는 포장을 선택하세요")))
    check(!OcrText.hasMethodButtons(listOf("매장", "포장 안내")))
    println("PASS CTC repeats/blank/confidence/invalid-output and price/quantity guards")
    // Real exported model outputs can be decoded without the Android or ONNX runtime.
    if (args.size == 4) {
        val dictionary = File(args[0]).readLines(Charsets.UTF_8)
        val bytes = File(args[1]).readBytes()
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val scores = FloatArray(buffer.remaining())
        buffer.get(scores)
        val reading = CtcDecoder(dictionary).decode(scores,args[2].toInt())
        check(reading.text == args[3]) { "Kotlin CTC differs from real ONNX reference decoding" }
        println("PASS real ONNX -> Kotlin CTC; text=${reading.text}, confidence=${reading.confidence}, uncertain=${reading.uncertain}")
    }
}
