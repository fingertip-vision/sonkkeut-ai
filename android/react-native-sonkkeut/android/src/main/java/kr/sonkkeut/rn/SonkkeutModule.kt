package kr.sonkkeut.rn

import android.os.SystemClock
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

    override fun getName() = "Sonkkeut"

    @ReactMethod
    fun init(options: ReadableMap?, promise: Promise) {
        io.execute {
            try {
                val fb = options?.takeIf { it.hasKey("nativeFeedback") }?.getBoolean("nativeFeedback") ?: true
                SonkkeutEngine.init(ctx.applicationContext, fb)
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
        SonkkeutEngine.listener = null
        SonkkeutEngine.running = false
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
