package kr.sonkkeut.rn

import android.os.SystemClock
import android.os.Handler
import android.os.Looper
import android.os.Build
import android.content.Intent
import android.os.Bundle
import android.speech.SpeechRecognizer
import android.speech.RecognizerIntent
import android.speech.RecognitionListener
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableArray
import com.facebook.react.bridge.WritableMap
import com.facebook.react.modules.core.DeviceEventManagerModule
import kr.sonkkeut.android.SonkkeutEngine
import kr.sonkkeut.android.KoreanWhisper
import java.util.concurrent.Executors

/**
 * JS에서 부르는 제어 API (NativeModules.Sonkkeut)
 *   init({nativeFeedback}) → Promise, start(), stop(), setTarget(id, expect) → Promise<boolean>, setTargetAt(x, y, expect) → Promise<id|null>,
 *   clearTarget(), requestKeyframe()
 * 결과는 "SonkkeutResult" 이벤트로 온다 (안내·판정·화면 구조가 있을 때 즉시, 그 밖에는 0.2초마다).
 */
class SonkkeutModule(private val ctx: ReactApplicationContext) : ReactContextBaseJavaModule(ctx) {
    private val io = Executors.newSingleThreadExecutor()
    private var lastEmit = 0L
    private val main = Handler(Looper.getMainLooper())
    private var speech: SpeechRecognizer? = null
    private var speechPromise: Promise? = null
    private var speechGeneration = 0L
    private val whisper by lazy { KoreanWhisper(ctx.applicationContext) }
    private var modelSpeechPromise: Promise? = null
    private var modelDownloadPromise: Promise? = null
    private var modelPreparePromise: Promise? = null
    @Volatile private var disposed = false

    @ReactMethod
    fun say(text: String) { main.post { SonkkeutEngine.feedback?.say(text, true) } }

    @ReactMethod
    fun silence() { main.post { SonkkeutEngine.feedback?.silence() } }

    @ReactMethod
    fun setMenuAliases(aliases: ReadableMap) {
        val values = aliases.toHashMap().mapValues { it.value.toString() }
        SonkkeutEngine.setMenuAliases(values)
        whisper.setMenuContext(values.values.distinct())
    }

    @ReactMethod
    fun getSpeechModelStatus(promise: Promise) {
        if (disposed) { promise.reject("CLOSED", "앱이 종료됐습니다"); return }
        io.execute {
            try { promise.resolve(toWritable(whisper.status() + mapOf("download_size_bytes" to 484672869L))) }
            catch (error: Throwable) { promise.reject("ASR_STATUS", "음성 모델 상태를 확인하지 못했습니다", error) }
        }
    }

    @ReactMethod
    fun prepareSpeechModel(promise: Promise) { main.post {
        if (disposed || modelPreparePromise != null || modelSpeechPromise != null || modelDownloadPromise != null || speechPromise != null) { promise.reject("BUSY", "음성 작업이 진행 중입니다"); return@post }
        modelPreparePromise = promise
        whisper.prepare(onReady = {
            if (modelPreparePromise === promise) { modelPreparePromise = null; promise.resolve(toWritable(whisper.status())) }
        }, onError = { error ->
            if (modelPreparePromise === promise) { modelPreparePromise = null; promise.reject("ASR_INIT", "음성 모델을 준비하지 못했습니다. 다시 시도해 주세요.", error) }
        })
    } }

    @ReactMethod
    fun downloadSpeechModel(promise: Promise) { main.post {
        if (disposed || modelDownloadPromise != null || modelSpeechPromise != null || speechPromise != null || modelPreparePromise != null) {
            promise.reject("BUSY", "음성 작업이 진행 중입니다"); return@post
        }
        modelDownloadPromise = promise
        whisper.install(onProgress = { done, total, stage ->
            if (modelDownloadPromise === promise && ctx.hasActiveReactInstance()) {
                ctx.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java).emit("SonkkeutModelDownload",
                    toWritable(mapOf("bytes" to done, "total_bytes" to total, "stage" to stage)))
            }
        }, onReady = {
            if (modelDownloadPromise === promise) { modelDownloadPromise = null; promise.resolve(toWritable(whisper.status())) }
        }, onError = { error ->
            if (modelDownloadPromise === promise) { modelDownloadPromise = null; promise.reject("ASR_DOWNLOAD", "음성 모델을 내려받지 못했습니다. 인터넷 연결과 저장 공간을 확인해 주세요.", error) }
        })
    } }

    @ReactMethod
    fun cancelSpeechModelDownload() { main.post {
        modelDownloadPromise?.let { whisper.cancel(); it.reject("CANCELLED", "모델 다운로드를 중지했습니다") }
        modelDownloadPromise = null
    } }

    @ReactMethod
    fun listenModel(promise: Promise) { main.post {
        if (disposed || speechPromise != null || modelSpeechPromise != null || modelDownloadPromise != null || modelPreparePromise != null) {
            promise.reject("BUSY", "이미 음성 작업이 진행 중입니다"); return@post
        }
        SonkkeutEngine.feedback?.silence()
        modelSpeechPromise = promise
        whisper.listen(onResult = resultCallback@{ result ->
            if (modelSpeechPromise !== promise) return@resultCallback
            modelSpeechPromise = null
            if (result.text.isBlank() || result.noSpeechProbability >= 0.8) {
                promise.reject("EMPTY", "주문을 다시 말씀해 주세요"); return@resultCallback
            }
            promise.resolve(result.text)
            if (ctx.hasActiveReactInstance()) {
                ctx.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java).emit("SonkkeutSpeechResult",
                    toWritable(mapOf("provider" to result.provider, "model_version" to "v3", "sequence_score" to result.sequenceScore,
                        "no_speech_probability" to result.noSpeechProbability, "inference_ms" to result.inferenceMs,
                        "audio_seconds" to result.audioSeconds)))
            }
        }, onError = { error ->
            if (modelSpeechPromise === promise) { modelSpeechPromise = null; promise.reject("ASR", error.message ?: "음성 주문을 다시 말씀해 주세요", error) }
        })
    } }

    @ReactMethod
    fun listen(promise: Promise) {
        main.post {
            if (disposed || speechPromise != null || modelSpeechPromise != null || modelDownloadPromise != null || modelPreparePromise != null) { promise.reject("BUSY", "이미 듣고 있습니다"); return@post }
            if (Build.VERSION.SDK_INT < 31 || !SpeechRecognizer.isOnDeviceRecognitionAvailable(ctx)) {
                promise.reject("OFFLINE_UNAVAILABLE", "이 기기의 오프라인 음성 인식이 준비되지 않았습니다. 주문을 입력해 주세요.")
                return@post
            }
            SonkkeutEngine.feedback?.silence()
            speech?.destroy()
            val token = ++speechGeneration
            try {
            val recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(ctx)
            speech = recognizer
            speechPromise = promise
            recognizer.setRecognitionListener(object : RecognitionListener {
                override fun onReadyForSpeech(p: Bundle?) {}
                override fun onBeginningOfSpeech() {}
                override fun onRmsChanged(v: Float) {}
                override fun onBufferReceived(b: ByteArray?) {}
                override fun onEndOfSpeech() {}
                override fun onPartialResults(b: Bundle?) {}
                override fun onEvent(t: Int, b: Bundle?) {}
                override fun onError(e: Int) {
                    if (token != speechGeneration || speechPromise !== promise) return
                    speechPromise = null; speech = null; recognizer.destroy()
                    promise.reject("SPEECH_$e", "다시 말씀해 주세요. 한국어 오프라인 언어팩도 확인해 주세요.")
                }
                override fun onResults(b: Bundle?) {
                    if (token != speechGeneration || speechPromise !== promise) return
                    val value = b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()
                    speechPromise = null; speech = null; recognizer.destroy()
                    if (value.isNullOrBlank()) promise.reject("EMPTY", "다시 말씀해 주세요")
                    else promise.resolve(value)
                }
            })
            recognizer.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            })
            } catch (error: Throwable) {
                speechGeneration++; speech?.destroy(); speech = null; speechPromise = null
                promise.reject("SPEECH_START", "기기 음성 인식을 시작하지 못했습니다. 마이크 권한과 한국어 언어팩을 확인해 주세요.", error)
            }
        }
    }

    @ReactMethod
    fun cancelListening() { main.post {
        speechGeneration++
        speech?.cancel(); speech?.destroy(); speech = null
        speechPromise?.reject("CANCELLED", "음성 입력을 중지했습니다")
        speechPromise = null
        modelSpeechPromise?.let { whisper.cancel(); it.reject("CANCELLED", "음성 입력을 중지했습니다") }
        modelSpeechPromise = null
    } }

    override fun getName() = "Sonkkeut"

    @ReactMethod
    fun init(options: ReadableMap?, promise: Promise) {
        io.execute {
            try {
                check(!disposed) { "앱이 종료됐습니다" }
                val fb = options?.takeIf { it.hasKey("nativeFeedback") }?.getBoolean("nativeFeedback") ?: true
                SonkkeutEngine.init(ctx.applicationContext, fb)
                check(!disposed) { "앱이 종료됐습니다" }
                SonkkeutEngine.listener = { emit(it) }
                promise.resolve(true)
            } catch (e: Throwable) {
                promise.reject("SONKKEUT_INIT", e.message, e)
            }
        }
    }

    @ReactMethod
    fun start() {
        SonkkeutEngine.running = true
    }

    @ReactMethod
    fun stop() {
        SonkkeutEngine.running = false
        main.post { SonkkeutEngine.feedback?.silence() }
    }

    @ReactMethod
    fun setTarget(id: String, expect: ReadableMap?, promise: Promise) {
        promise.resolve(SonkkeutEngine.setTarget(id, expect?.toHashMap()))
    }

    @ReactMethod
    fun setTargetAt(x: Double, y: Double, expect: ReadableMap?, promise: Promise) {
        promise.resolve(SonkkeutEngine.setTargetAt(x, y, expect?.toHashMap()))
    }

    @ReactMethod
    fun setTargetAtImage(px: Double, py: Double, expect: ReadableMap?, promise: Promise) {
        promise.resolve(SonkkeutEngine.setTargetAtImage(px, py, expect?.toHashMap()))
    }

    @ReactMethod
    fun clearTarget() {
        SonkkeutEngine.clearTarget()
    }

    @ReactMethod
    fun requestKeyframe() {
        SonkkeutEngine.requestKeyframe()
    }

    // NativeEventEmitter 경고 방지용
    @ReactMethod
    fun addListener(eventName: String) {}

    @ReactMethod
    fun removeListeners(count: Int) {}

    private fun emit(m: Map<String, Any?>) {
        val important = m.containsKey("verdict") || m.containsKey("structure") || m.containsKey("hint") ||
            (m["event"] as Map<*, *>?)?.get("speak") != null || m["target_missing"] == true
        val now = SystemClock.uptimeMillis()
        if (!important && now - lastEmit < 200) return // 화면 갱신용 결과는 초당 5번이면 충분
        lastEmit = now
        if (!ctx.hasActiveReactInstance()) return
        ctx.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter::class.java).emit("SonkkeutResult", toWritable(m))
    }

    override fun invalidate() {
        disposed = true
        SonkkeutEngine.listener = null
        SonkkeutEngine.running = false
        main.post {
            speechGeneration++; speech?.destroy(); speech = null
            listOf(speechPromise, modelSpeechPromise, modelDownloadPromise, modelPreparePromise).forEach { it?.reject("CLOSED", "앱이 종료됐습니다") }
            speechPromise = null; modelSpeechPromise = null; modelDownloadPromise = null; modelPreparePromise = null
            whisper.close()
        }
        io.execute { SonkkeutEngine.release() }
        io.shutdown()
        super.invalidate()
    }

    companion object {
        fun toWritable(m: Map<*, *>): WritableMap {
            val w = Arguments.createMap()
            for ((k, v) in m) put(w, k.toString(), v)
            return w
        }

        private fun put(w: WritableMap, k: String, v: Any?) {
            when (v) {
                null -> w.putNull(k)
                is Boolean -> w.putBoolean(k, v)
                is Int -> w.putInt(k, v)
                is Number -> w.putDouble(k, v.toDouble())
                is String -> w.putString(k, v)
                is Map<*, *> -> w.putMap(k, toWritable(v))
                is List<*> -> w.putArray(k, toArray(v))
                else -> w.putString(k, v.toString())
            }
        }

        fun toArray(l: List<*>): WritableArray {
            val a = Arguments.createArray()
            for (v in l) when (v) {
                null -> a.pushNull()
                is Boolean -> a.pushBoolean(v)
                is Int -> a.pushInt(v)
                is Number -> a.pushDouble(v.toDouble())
                is String -> a.pushString(v)
                is Map<*, *> -> a.pushMap(toWritable(v))
                is List<*> -> a.pushArray(toArray(v))
                else -> a.pushString(v.toString())
            }
            return a
        }
    }
}
