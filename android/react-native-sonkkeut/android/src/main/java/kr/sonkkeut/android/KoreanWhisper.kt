package kr.sonkkeut.android

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.sqrt

/**
 * User's CT2 Whisper v3 on CPU. Recording/preprocessing/inference/model installation
 * share one worker. Cancel suppresses stale callbacks and stops recording immediately;
 * an already-started CT2 inference finishes on the worker before the next job starts.
 */
class KoreanWhisper(context: Context) : AutoCloseable {
    data class Result(val text: String, val sequenceScore: Double?, val noSpeechProbability: Double,
                      val inferenceMs: Long, val audioSeconds: Double, val provider: String = "ct2-whisper-v3")
    private val ctx = context.applicationContext
    private val io = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val generation = AtomicLong(0)
    private val busy = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private val lifecycle = Any()
    private val recording = AtomicReference<AudioRecord?>(null)
    private val installer = WhisperModelInstaller(ctx)
    @Volatile private var handle = 0L
    @Volatile private var decoder: WhisperByteDecoder? = null
    @Volatile private var downloading = false
    @Volatile private var menus: List<String> = listOf("아메리카노", "카페라떼")

    /** Menu names only, never an unrestricted user-authored model prompt. */
    fun setMenuContext(names: List<String>) {
        val selected = mutableListOf<String>()
        var characters = 0
        for (value in names) {
            if (selected.size == 24) break
            val name = value.replace(Regex("[\\p{Cc}\\p{Cf}]"), " ").trim()
            if (name.isEmpty() || name in selected || name.length + characters + selected.size > 500) continue
            selected.add(name); characters += name.length
        }
        menus = selected.toList()
    }

    fun status(): Map<String, Any?> = mapOf("provider" to "ct2-whisper-v3", "model_version" to "v3",
        "installed" to installer.isInstalled(), "ready" to (handle != 0L),
        "downloading" to downloading, "busy" to busy.get(), "requires_download" to !installer.isInstalled())

    fun install(downloadUrl: String = WhisperModelInstaller.DOWNLOAD_URL,
                onProgress: (Long, Long, String) -> Unit, onReady: () -> Unit, onError: (Throwable) -> Unit) {
        begin(onError) { token ->
            downloading = true
            try {
                installer.install(downloadUrl, { closed.get() || generation.get() != token }) { done, total, stage ->
                    deliver(token) { onProgress(done, total, stage) }
                }
                checkCurrent(token)
                prepareModel()
                deliver(token, onReady)
            } catch (error: Throwable) { deliver(token) { onError(error) } }
            finally { downloading = false }
        }
    }

    fun prepare(onReady: () -> Unit, onError: (Throwable) -> Unit) = begin(onError) { token ->
        try { prepareModel(); deliver(token, onReady) }
        catch (error: Throwable) { deliver(token) { onError(error) } }
    }

    fun listen(onResult: (Result) -> Unit, onError: (Throwable) -> Unit) = begin(onError) { token ->
        try {
            check(ctx.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                "마이크 권한을 허용해 주세요."
            }
            prepareModel()
            checkCurrent(token)
            val pcm = capture(token)
            checkCurrent(token)
            infer(pcm, token, onResult)
        } catch (error: Throwable) { deliver(token) { onError(error) } }
    }

    /** Deterministic real-inference fixture; accepts 16 kHz mono PCM16, never mock output. */
    fun transcribePcm(pcm: ShortArray, onResult: (Result) -> Unit, onError: (Throwable) -> Unit) = begin(onError) { token ->
        try { prepareModel(); checkCurrent(token); infer(pcm, token, onResult) }
        catch (error: Throwable) { deliver(token) { onError(error) } }
    }

    private fun prepareModel() {
        if (handle != 0L) return
        val path: File = installer.modelDirectory() ?: error("음성 모델을 먼저 내려받아 주세요.")
        val nextDecoder = WhisperByteDecoder(File(path, "vocabulary.json"))
        val nextHandle = KoreanWhisperNative.create(path.absolutePath, 2, true)
        check(nextHandle != 0L) { "음성 모델을 불러오지 못했습니다." }
        decoder = nextDecoder
        handle = nextHandle
    }

    private fun infer(pcm: ShortArray, token: Long, onResult: (Result) -> Unit) {
        require(pcm.isNotEmpty() && pcm.size <= MAX_SAMPLES) { "음성은 30초 이내여야 합니다." }
        val hints = (menus + listOf("아이스", "따뜻한", "톨", "그란데", "벤티", "포장", "매장")).joinToString(" ")
        val hintIds = decoder!!.encode(" " + hints).takeLast(223).toIntArray()
        val raw = JSONObject(KoreanWhisperNative.transcribe(handle, pcm, decoder!!.tokenPieces(hintIds)))
        checkCurrent(token)
        val ids = raw.getJSONArray("token_ids")
        val text = decoder!!.decode(IntArray(ids.length()) { ids.getInt(it) })
        val result = Result(text, if (raw.has("sequence_score")) raw.getDouble("sequence_score") else null,
            raw.getDouble("no_speech_probability"), raw.getLong("inference_ms"), pcm.size / SAMPLE_RATE.toDouble())
        // Empty/silence results remain empty; the caller chooses the re-prompt policy.
        deliver(token) { onResult(result) }
    }

    private fun capture(token: Long): ShortArray {
        val minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        check(minimum > 0) { "이 기기에서 16kHz 음성을 녹음할 수 없습니다." }
        val audio = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minimum * 2, 6400))
        check(audio.state == AudioRecord.STATE_INITIALIZED) { audio.release(); "마이크를 준비하지 못했습니다." }
        recording.set(audio)
        val result = ShortArray(MAX_SAMPLES)
        val chunk = ShortArray(1600)
        var length = 0
        var speech = false
        var trailingSilence = 0
        try {
            checkCurrent(token)
            audio.startRecording()
            while (length < MAX_SAMPLES) {
                checkCurrent(token)
                val count = audio.read(chunk, 0, minOf(chunk.size, MAX_SAMPLES - length), AudioRecord.READ_BLOCKING)
                check(count > 0) { "마이크 녹음이 중단됐습니다." }
                chunk.copyInto(result, length, 0, count)
                length += count
                var energy = 0.0
                for (i in 0 until count) { val sample = chunk[i] / 32768.0; energy += sample * sample }
                // Simple energy endpointing; this is not a trained voice-activity model.
                if (sqrt(energy / count) >= 0.012) { speech = true; trailingSilence = 0 }
                else trailingSilence += count
                if (speech && length >= SAMPLE_RATE && trailingSilence >= SAMPLE_RATE * 12 / 10) break
                if (!speech && length >= SAMPLE_RATE * 6) break
            }
            return result.copyOf(length)
        } finally {
            recording.compareAndSet(audio, null)
            runCatching { audio.stop() }
            audio.release()
        }
    }

    private fun begin(onError: (Throwable) -> Unit, operation: (Long) -> Unit) {
        synchronized(lifecycle) {
            if (closed.get() || !busy.compareAndSet(false, true)) {
                main.post { onError(IllegalStateException("음성 작업이 진행 중이거나 종료됐습니다.")) }; return
            }
            val token = generation.incrementAndGet()
            io.execute {
                try { checkCurrent(token); operation(token) }
                catch (error: Throwable) { deliver(token) { onError(error) } }
                finally { busy.set(false) }
            }
        }
    }
    private fun checkCurrent(token: Long) {
        check(!closed.get() && generation.get() == token) { "음성 입력을 중지했습니다." }
    }
    private fun deliver(token: Long, block: () -> Unit) { main.post { if (!closed.get() && generation.get() == token) block() } }
    fun cancel() { generation.incrementAndGet(); recording.get()?.let { runCatching { it.stop() } } }
    override fun close() {
        synchronized(lifecycle) {
            if (!closed.compareAndSet(false, true)) return
            cancel()
            io.execute { if (handle != 0L) KoreanWhisperNative.close(handle); handle = 0L; decoder = null }
            io.shutdown()
        }
    }
    companion object { const val SAMPLE_RATE = 16000; const val MAX_SAMPLES = 480000 }
}
