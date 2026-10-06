package kr.sonkkeut.android

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import kr.sonkkeut.core.FrameImage
import kr.sonkkeut.core.Hand
import kr.sonkkeut.core.HandSource
import kr.sonkkeut.core.Pt
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * M4 손 관절: MediaPipe Hand Landmarker (안드로이드 SDK)
 * 모델 파일은 assets/sonkkeut/hand_landmarker.task (빌드할 때 gradle이 내려받는다)
 */
class MediaPipeHands(context: Context, numHands: Int = 2, minConf: Float = 0.5f) : HandSource {
    private val landmarker: HandLandmarker = HandLandmarker.createFromOptions(
        context,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("sonkkeut/hand_landmarker.task").build())
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(numHands)
            .setMinHandDetectionConfidence(minConf)
            .setMinHandPresenceConfidence(minConf)
            .setMinTrackingConfidence(minConf)
            .build(),
    )
    private var lastTs = -1L
    private var bitmap: Bitmap? = null
    private val rgba = Mat()

    override fun detect(frame: FrameImage, tSeconds: Double): List<Hand> {
        val m = (frame as MatFrame).mat
        val w = m.cols()
        val h = m.rows()
        val bmp = bitmap?.takeIf { it.width == w && it.height == h } ?: Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also { bitmap = it }
        Imgproc.cvtColor(m, rgba, Imgproc.COLOR_RGB2RGBA)
        Utils.matToBitmap(rgba, bmp)
        val ts = maxOf((tSeconds * 1000).toLong(), lastTs + 1) // VIDEO 모드는 시각이 계속 커져야 한다
        lastTs = ts
        val res = landmarker.detectForVideo(BitmapImageBuilder(bmp).build(), ts)
        return res.landmarks().mapIndexed { i, lms ->
            val score = res.handedness().getOrNull(i)?.firstOrNull()?.score()?.toDouble() ?: 1.0
            Hand(lms.map { Pt(it.x() * w.toDouble(), it.y() * h.toDouble()) }, score)
        }
    }

    override fun close() {
        landmarker.close()
        rgba.release()
    }
}
