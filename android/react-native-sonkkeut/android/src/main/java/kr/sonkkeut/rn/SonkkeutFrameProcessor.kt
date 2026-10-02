package kr.sonkkeut.rn

import com.mrousavy.camera.frameprocessors.Frame
import com.mrousavy.camera.frameprocessors.FrameProcessorPlugin
import com.mrousavy.camera.frameprocessors.VisionCameraProxy
import kr.sonkkeut.android.SonkkeutEngine

/**
 * VisionCamera Frame Processor 플러그인 "sonkkeut"
 * JS 워크릿에서 sonkkeutProcess(frame)으로 부른다. 결과는 작은 요약만 돌려주고
 * 전체 결과는 SonkkeutModule이 이벤트로 보낸다(워크릿 안에서 TTS를 부를 수 없기 때문).
 */
class SonkkeutFrameProcessor(proxy: VisionCameraProxy, options: Map<String, Any>?) : FrameProcessorPlugin() {
    override fun callback(frame: Frame, params: Map<String, Any>?): Any? {
        val image = frame.image
        // CameraX가 알려 주는 회전 각도(시계 방향으로 이만큼 돌려야 똑바로 선다). 필요하면 JS에서 덮어쓴다.
        val rot = (params?.get("rotation") as Number?)?.toInt() ?: frame.imageProxy.imageInfo.rotationDegrees
        val r = SonkkeutEngine.processYuv(image, rot) ?: return null
        val ev = r["event"] as Map<*, *>?
        return hashMapOf<String, Any?>(
            "found" to r["found"],
            "type" to ev?.get("type"),
            "speak" to ev?.get("speak"),
            "vibe_hz" to ev?.get("vibe_hz"),
            "total_ms" to (r["timings"] as Map<*, *>?)?.get("total_ms"),
        )
    }
}
