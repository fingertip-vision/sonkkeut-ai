package kr.sonkkeut.android

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.view.accessibility.AccessibilityManager
import kr.sonkkeut.core.FrameResult
import java.util.Locale

/**
 * 기본 음성·진동 출력 (시연용).
 * 앱(임현승)이 React Native에서 직접 출력할 거라면 SonkkeutEngine.init(context, nativeFeedback = false)로 끈다.
 *
 * 음성: event.speak / verdict.speak / hint 를 읽는다. 엔진이 이미 0.8초 간격으로 걸러서 보낸다.
 *       "지금 누르세요"·판정은 진행 중인 말을 끊고(QUEUE_FLUSH), 나머지는 말하는 중이면 건너뛴다.
 * 진동: vibe_hz 주기로 짧게(25 ms) 두드린다. 멀면 느리게(2Hz), 가까우면 빠르게(4~8Hz), 버튼 위 10Hz.
 */
class NativeFeedback(context: Context) : TextToSpeech.OnInitListener {
    private val storage = context.getSharedPreferences("sonkkeut_feedback", Context.MODE_PRIVATE)
    private val policy = FeedbackPolicy(storage.getBoolean("voice", true), storage.getBoolean("vibration", true), storage.getFloat("rate", 1f))
    private val accessibility = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager?
    private val tts = TextToSpeech(context.applicationContext, this)
    @Volatile private var ttsReady = false
    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= 31) {
        (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager?)?.defaultVibrator
    } else {
        @Suppress("DEPRECATION") context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator?
    }
    private val thread = HandlerThread("sonkkeut-vibe").apply { start() }
    private val handler = Handler(thread.looper)
    @Volatile private var hz = 0.0
    @Volatile private var hzUpdated = 0L
    private var lastHint: String? = null
    private data class SpeechRequest(val text: String, val explicit: Boolean)
    private var pendingSpeech: SpeechRequest? = null

    private val pulse = object : Runnable {
        override fun run() {
            val h = hz
            val fresh = SystemClock.uptimeMillis() - hzUpdated < 400 // 결과가 끊기면 진동도 멈춘다
            if (policy.vibrationEnabled && h > 0 && fresh) {
                buzz(25)
                handler.postDelayed(this, (1000.0 / h).toLong().coerceAtLeast(60))
            } else {
                handler.postDelayed(this, 100)
            }
        }
    }

    init {
        handler.post(pulse)
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            tts.language = Locale.KOREAN
            tts.setSpeechRate(policy.speechRate)
            ttsReady = true
            pendingSpeech?.let { request -> pendingSpeech = null; say(request.text, true, request.explicit) }
        }
    }

    fun onResult(r: FrameResult) {
        val ev = r.event
        hz = if (policy.vibrationEnabled) ev?.vibeHz ?: 0.0 else 0.0
        hzUpdated = SystemClock.uptimeMillis()
        r.verdict?.let { say(it.speak, urgent = true) }
        ev?.speak?.let { say(it, urgent = ev.type == "press") }
        val hint = r.hint
        if (hint != null && hint != lastHint) say(hint, urgent = false)
        lastHint = hint
    }

    fun configure(voice: Boolean, vibration: Boolean, rate: Float) {
        val stopVoice = policy.voiceEnabled && !voice
        policy.configure(voice, vibration, rate)
        if (stopVoice) { pendingSpeech = null; tts.stop() }
        if (!vibration) { hz = 0.0; vibrator?.cancel() }
        if (ttsReady) tts.setSpeechRate(policy.speechRate)
    }

    fun say(text: String, urgent: Boolean, explicit: Boolean = false) {
        if (!policy.allowsSpeech(explicit, accessibility?.isTouchExplorationEnabled == true)) return
        if (!ttsReady) { pendingSpeech = SpeechRequest(text, explicit); return }
        if (!urgent && tts.isSpeaking) return
        tts.speak(text, if (urgent) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD, null, text)
    }

    private fun buzz(ms: Long) {
        val v = vibrator ?: return
        if (Build.VERSION.SDK_INT >= 26) v.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
        else @Suppress("DEPRECATION") v.vibrate(ms)
    }

    fun shutdown() {
        handler.removeCallbacksAndMessages(null)
        thread.quitSafely()
        tts.shutdown()
    }

    fun silence() {
        hz = 0.0
        pendingSpeech = null
        tts.stop()
        vibrator?.cancel()
    }
}
