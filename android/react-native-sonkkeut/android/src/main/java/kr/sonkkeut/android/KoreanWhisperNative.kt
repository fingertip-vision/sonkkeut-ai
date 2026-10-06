package kr.sonkkeut.android

/** Synchronous JNI: invoke only on KoreanWhisper's single IO executor. */
internal object KoreanWhisperNative {
    init { System.loadLibrary("ctranslate2"); System.loadLibrary("sonkkeut_whisper") }
    external fun create(modelPath: String, threads: Int, quantized: Boolean): Long
    external fun transcribe(handle: Long, pcm: ShortArray, hotwordTokens: Array<String> = emptyArray()): String
    external fun transcribeFeatures(handle: Long, features: FloatArray): String
    external fun logMel(pcm: ShortArray): FloatArray
    external fun close(handle: Long)
}
