package kr.sonkkeut.android

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import kr.sonkkeut.core.CtcDecoder
import kr.sonkkeut.core.CtcReading
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.max

/** Fine-tuned Paddle recognizer, exported to ONNX. All crops and tensors remain on the device. */
class KioskCtcRecognizer(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val decoder: CtcDecoder
    private val session: OrtSession
    private val inputName: String

    init {
        val assets = context.assets
        val dictionary = assets.open("sonkkeut/m3_kiosk_rec_v2_dictionary.txt").bufferedReader(Charsets.UTF_8).use { it.readLines() }
        require(dictionary.isNotEmpty() && dictionary.none { it.isEmpty() }) { "OCR dictionary is missing characters" }
        decoder = CtcDecoder(dictionary)
        val bytes = assets.open("sonkkeut/m3_kiosk_rec_v2.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions()
        try {
            options.setIntraOpNumThreads(2)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            session = environment.createSession(bytes, options)
        } finally { options.close() }
        try {
            inputName = session.inputNames.single()
            val shape = (session.inputInfo[inputName]!!.info as TensorInfo).shape
            require(shape.size == 4 && shape[1] == 3L && shape[2] == 48L) { "Expected OCR BGR NCHW input [1,3,48,W]" }
        } catch (error: Throwable) { session.close(); throw error }
    }

    fun recognize(rgb: Mat): CtcReading {
        require(!rgb.empty() && rgb.channels() == 3) { "OCR expects an RGB crop" }
        val height = 48
        val resizedWidth = max(1, ceil(height * rgb.cols().toDouble() / rgb.rows()).toInt()).coerceAtMost(3200)
        val width = max(320, resizedWidth)
        val resized = Mat()
        val bgr = Mat()
        try {
            Imgproc.resize(rgb, resized, Size(resizedWidth.toDouble(), height.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
            Imgproc.cvtColor(resized, bgr, Imgproc.COLOR_RGB2BGR)
            val bytes = ByteArray(height * resizedWidth * 3)
            bgr.get(0, 0, bytes)
            // Padding is 0 AFTER normalization, exactly as Paddle RecResizeImg.
            val input = FloatArray(3 * height * width)
            for (y in 0 until height) for (x in 0 until resizedWidth) for (c in 0 until 3) {
                val pixel = bytes[(y * resizedWidth + x) * 3 + c].toInt() and 255
                input[c * height * width + y * width + x] = pixel / 127.5f - 1f
            }
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), longArrayOf(1, 3, 48, width.toLong())).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { output ->
                    val values = output[0] as OnnxTensor
                    val shape = values.info.shape
                    require(shape.size == 3 && shape[0] == 1L && shape[2] == decoder.classes.toLong()) { "Expected OCR output [1,T,dictionary+blank]" }
                    val buffer = values.floatBuffer
                    val scores = FloatArray(buffer.remaining())
                    buffer.get(scores)
                    return decoder.decode(scores, shape[1].toInt(), probabilities = true)
                }
            }
        } finally { bgr.release(); resized.release() }
    }

    override fun close() = session.close()
}
