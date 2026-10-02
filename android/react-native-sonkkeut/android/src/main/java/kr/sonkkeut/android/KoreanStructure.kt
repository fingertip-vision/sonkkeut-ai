package kr.sonkkeut.android

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import kr.sonkkeut.core.*
import org.opencv.android.Utils
import java.util.concurrent.TimeUnit

/** Bundled OCR: images never leave the device. Run once per keyframe on the camera worker. */
class KoreanStructure : StructureProvider {
    private val recognizer = TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    @Volatile var aliases: Map<String, String> = emptyMap()

    override fun build(elements: List<Element>, crops: Map<String, FrameImage>, flat: FrameImage, keyframeId: Int): ScreenStructure {
        val mat = (flat as MatFrame).mat
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        try {
            Utils.matToBitmap(mat, bitmap)
            // A bounded wait on the frame worker, never the UI thread. Fail closed on timeout.
            val task = recognizer.process(InputImage.fromBitmap(bitmap, 0))
            // Keep the bitmap alive until ML Kit has finished, even if await times out.
            task.addOnCompleteListener { bitmap.recycle() }
            val text = Tasks.await(task, 2, TimeUnit.SECONDS)
            val lines = text.textBlocks.flatMap { it.lines }
            val margin = 0.04
            val scale = 1 + 2 * margin
            val selected = mutableListOf<String>()
            for (el in elements) {
                val matched = lines.filter { line ->
                    val b = line.boundingBox ?: return@filter false
                    val x = b.exactCenterX() / flat.width * scale - margin
                    val y = b.exactCenterY() / flat.height * scale - margin
                    el.box.contains(x.toDouble(), y.toDouble())
                }.sortedWith(compareBy({ it.boundingBox?.top }, { it.boundingBox?.left }))
                val raw = matched.joinToString(" ") { it.text }.trim()
                if (raw.contains("선택됨")) selected.add(el.id)
                val label = raw.replace("선택됨", "").trim()
                el.text = aliases[label.replace(" ", "")] ?: label
                el.price = Regex("([0-9][0-9,]*)\\s*원").find(raw)?.groupValues?.get(1)?.replace(",", "")?.toIntOrNull()
                if (raw.isEmpty() && el.kind != "back") el.conf = 0.0
            }
            val all = text.text.replace(" ", "")
            val type = when {
                Regex("카드를?(넣|삽입|대)|결제진행중|결제완료|카드결제화면").containsMatchIn(all) -> "payment"
                Regex("장바구니|주문내역").containsMatchIn(all) && Regex("총.?금액|합계|총액").containsMatchIn(all) -> "cart"
                Regex("담기|추가하기").containsMatchIn(all) && Regex("온도|HOT|ICE|따뜻|차갑|사이즈|크기").containsMatchIn(all) -> "option"
                elements.any { it.kind == "menu" && !it.text.isNullOrBlank() } -> "menu"
                Regex("주문시작|시작하기").containsMatchIn(all) -> "start"
                else -> "unknown"
            }
            val count = Regex("(?:장바구니|총|수량)\\s*(\\d+)\\s*(?:개|잔)").find(text.text)?.groupValues?.get(1)?.toIntOrNull()
            val total = Regex("(?:총\\s*금액|합계|총액)\\s*([0-9][0-9,]*)\\s*원").find(text.text)?.groupValues?.get(1)?.replace(",", "")?.toIntOrNull()
            return ScreenStructure(type, keyframeId, elements, count, selected, total)
        } catch (_: Exception) {
            elements.forEach { it.conf = 0.0 }
            return ScreenStructure.basic(elements, keyframeId)
        }
    }
    fun close() = recognizer.close()
}
