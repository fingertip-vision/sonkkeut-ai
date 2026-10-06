package kr.sonkkeut.android

/** Automatic feedback preferences; an explicit repeat remains available with speech disabled. */
class FeedbackPolicy(voice: Boolean = true, vibration: Boolean = true, rate: Float = 1f) {
    @Volatile var voiceEnabled = voice; private set
    @Volatile var vibrationEnabled = vibration; private set
    @Volatile var speechRate = safeRate(rate); private set

    fun configure(voice: Boolean, vibration: Boolean, rate: Float) {
        voiceEnabled = voice; vibrationEnabled = vibration; speechRate = safeRate(rate)
    }
    fun allowsSpeech(explicit: Boolean, touchExploration: Boolean): Boolean =
        explicit || (voiceEnabled && !touchExploration)

    companion object {
        fun safeRate(rate: Float): Float = if (rate.isFinite()) rate.coerceIn(0.75f, 1.25f) else 1f
    }
}
