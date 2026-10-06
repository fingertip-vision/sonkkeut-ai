package kr.sonkkeut.android

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import kr.sonkkeut.core.TensorModel
import java.nio.FloatBuffer

/** ONNX Runtime Mobile로 모델 한 개 실행 (core.TensorModel 구현) */
class OnnxModel(bytes: ByteArray, threads: Int = 2) : TensorModel, AutoCloseable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(bytes, OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    })
    private val inputName = session.inputNames.first()

    override fun run(input: FloatArray, shape: LongArray): FloatArray {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { t ->
            session.run(mapOf(inputName to t)).use { out ->
                val v = out[0] as OnnxTensor
                val fb = v.floatBuffer
                val arr = FloatArray(fb.remaining())
                fb.get(arr)
                return arr
            }
        }
    }

    override fun close() = session.close()
}
