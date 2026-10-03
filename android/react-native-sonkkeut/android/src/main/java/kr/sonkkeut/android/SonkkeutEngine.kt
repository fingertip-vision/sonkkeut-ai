package kr.sonkkeut.android

import android.content.Context
import android.media.Image
import android.os.Build
import android.util.Log
import kr.sonkkeut.core.Expect
import kr.sonkkeut.core.FrameResult
import kr.sonkkeut.core.ScreenStructure
import kr.sonkkeut.core.StructureProvider
import kr.sonkkeut.core.VisionPipeline
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * 손끝길 영상 AI 엔진 (앱 전체에서 하나)
 *
 *   SonkkeutEngine.init(context)              // 모델 불러오기 (앱 시작 시 한 번, 1~2초)
 *   SonkkeutEngine.processYuv(image, 90)      // 카메라 프레임마다 (VisionCamera 플러그인이 부름)
 *   SonkkeutEngine.setTarget("e7", expect)    // 언어 쪽(F-07) 또는 화면 터치로 목표 지정
 *   SonkkeutEngine.listener = { map -> ... }  // 결과 (React Native 이벤트로 전달)
 *
 * 처리는 카메라 프레임 스레드에서 동기로 한다. 처리 중에 들어온 프레임은 VisionCamera가 버린다.
 */
object SonkkeutEngine {
    private const val TAG = "Sonkkeut"
    private const val MAX_LONG_SIDE = 1280 // 학습 영상(960×1280)과 같은 크기로 맞춘다

    private val lock = Any()
    private var pipeline: VisionPipeline? = null
    private var hands: MediaPipeHands? = null
    private val models = mutableListOf<OnnxModel>()
    private val ops = OpenCvImageOps()
    private val yuvBuf = YuvToRgb()

    @Volatile var running = false
    @Volatile var listener: ((Map<String, Any?>) -> Unit)? = null
    @Volatile var lastResult: Map<String, Any?> = emptyMap()
        private set
    var feedback: NativeFeedback? = null

    /** 언어 쪽(노현석: F-04·F-05)이 꽂는 자리. 기본값은 요소만 담은 구조 */
    private var koreanStructure: KoreanStructure? = null
    private var menuAliases: Map<String, String> = emptyMap()
    @Volatile var structureProvider: StructureProvider = StructureProvider { els, crops, flat, kid ->
        koreanStructure?.build(els, crops, flat, kid) ?: ScreenStructure.basic(els, kid)
    }

    fun setMenuAliases(aliases: Map<String, String>) = synchronized(lock) {
        menuAliases = aliases.toMap()
        koreanStructure?.aliases = menuAliases
    }

    val isReady get() = pipeline != null

    fun init(context: Context, nativeFeedback: Boolean = true) {
        synchronized(lock) {
            if (pipeline != null) return
            check(Build.SUPPORTED_ABIS.any { it.startsWith("arm") }) {
                "손 인식은 ARM Android 휴대폰에서 사용할 수 있습니다. 실제 휴대폰에 앱을 설치해 주세요."
            }
            check(OpenCVLoader.initLocal()) { "OpenCV를 불러오지 못했습니다" }
            val am = context.assets
            fun asset(name: String) = am.open("sonkkeut/$name").use { it.readBytes() }
            try {
                koreanStructure = KoreanStructure(context.applicationContext).also { it.aliases = menuAliases }
                val m1 = OnnxModel(asset("m1_screen_corners_int8.onnx")).also { models.add(it) }
                val m2 = OnnxModel(asset("m2_screen_elements_int8.onnx")).also { models.add(it) }
                val refiner = OnnxModel(asset("m1r_corner_refiner.onnx"), threads = 1).also { models.add(it) }
                val h = MediaPipeHands(context)
                hands = h
                if (nativeFeedback) feedback = NativeFeedback(context)
                pipeline = VisionPipeline(m1, m2, ops, h, refiner,
                    structureProvider = StructureProvider { els, crops, flat, kid -> structureProvider.build(els, crops, flat, kid) })
                Log.i(TAG, "모델 준비 완료 (첨부 모델 M1, M2, M1-R 사용)")
            } catch (e: Throwable) {
                release()
                throw e
            }
        }
    }

    fun release() {
        synchronized(lock) {
            running = false
            val ocr = koreanStructure
            koreanStructure = null
            runCatching { ocr?.close() }.onFailure { Log.w(TAG, "OCR 자원 해제 실패") }
            hands?.close()
            feedback?.shutdown()
            models.forEach { it.close() }
            models.clear()
            pipeline = null
            hands = null
            feedback = null
        }
    }

    // ---------- 목표 지정 (F-07 / 화면 터치) ----------
    /** 현재 화면에 그 id가 없으면 false (화면이 바뀌었으면 언어 쪽이 다시 계획) */
    fun setTarget(id: String, expect: Map<String, Any?>? = null): Boolean = synchronized(lock) {
        runCatching { pipeline?.setTarget(id, Expect.fromMap(expect)) }.isSuccess && pipeline != null
    }

    /** 화면 좌표(0~1)로 목표 지정. 지정된 요소 id (없으면 null) */
    fun setTargetAt(x: Double, y: Double, expect: Map<String, Any?>? = null): String? = synchronized(lock) {
        pipeline?.setTargetAt(x, y, Expect.fromMap(expect))?.id
    }

    /** 카메라 영상 픽셀(회전 보정 후, processYuv 결과의 frame_size 기준)로 목표 지정 — 미리보기 터치용 */
    fun setTargetAtImage(px: Double, py: Double, expect: Map<String, Any?>? = null): String? = synchronized(lock) {
        val p = pipeline ?: return null
        val st = p.plane.state ?: return null
        val s = st.toScreen(kr.sonkkeut.core.Pt(px, py))
        p.setTargetAt(s.x, s.y, Expect.fromMap(expect))?.id
    }

    fun clearTarget() {
        synchronized(lock) { pipeline?.clearTarget() }
    }

    fun requestKeyframe() {
        synchronized(lock) { pipeline?.requestKeyframe() }
    }

    // ---------- 프레임 처리 ----------
    /** rotationDegrees: 이미지를 시계 방향으로 이만큼 돌려야 똑바로 선다 (CameraX ImageInfo.rotationDegrees) */
    fun processYuv(image: Image, rotationDegrees: Int): Map<String, Any?>? {
        if (!running) return null
        val p = pipeline ?: return null
        val rgb = yuvBuf.convert(image)
        val upright = rotate(rgb, rotationDegrees)
        val frame = MatFrame(resizeLongSide(upright))
        return try {
            val res: FrameResult = synchronized(lock) { p.process(frame, System.nanoTime() / 1e9) }
            val target = p.guide.target
            val plane = res.plane
            val targetBox = if (target != null && plane != null) {
                val b = target.box
                val points = listOf(kr.sonkkeut.core.Pt(b.x1, b.y1), kr.sonkkeut.core.Pt(b.x2, b.y1),
                    kr.sonkkeut.core.Pt(b.x2, b.y2), kr.sonkkeut.core.Pt(b.x1, b.y2)).map { plane.toImage(it) }
                listOf(points.minOf { it.x }, points.minOf { it.y }, points.maxOf { it.x }, points.maxOf { it.y })
            } else null
            val map = res.toMap() + ("frame_size" to listOf(frame.width, frame.height)) + ("target_image_box" to targetBox)
            lastResult = map
            feedback?.onResult(res)
            listener?.invoke(map)
            map
        } catch (e: Throwable) {
            Log.e(TAG, "프레임 처리 실패", e)
            null
        } finally {
            frame.mat.release()
        }
    }

    private fun rotate(src: Mat, deg: Int): Mat {
        val code = when (((deg % 360) + 360) % 360) {
            90 -> Core.ROTATE_90_CLOCKWISE
            180 -> Core.ROTATE_180
            270 -> Core.ROTATE_90_COUNTERCLOCKWISE
            else -> return src
        }
        val dst = Mat()
        Core.rotate(src, dst, code)
        src.release()
        return dst
    }

    private fun resizeLongSide(src: Mat): Mat {
        val long = maxOf(src.cols(), src.rows())
        if (long <= MAX_LONG_SIDE) return src
        val s = MAX_LONG_SIDE.toDouble() / long
        val dst = Mat()
        Imgproc.resize(src, dst, Size(src.cols() * s, src.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        src.release()
        return dst
    }
}

/** android.media.Image(YUV_420_888) → RGB Mat. 줄 간격(rowStride)·화소 간격(pixelStride)을 처리한다. */
class YuvToRgb {
    private var nv21: ByteArray = ByteArray(0)

    fun convert(image: Image): Mat {
        val w = image.width
        val h = image.height
        val need = w * h * 3 / 2
        if (nv21.size != need) nv21 = ByteArray(need)
        val (yP, uP, vP) = image.planes.let { Triple(it[0], it[1], it[2]) }
        // Y
        val yb = yP.buffer
        var pos = 0
        for (row in 0 until h) {
            yb.position(row * yP.rowStride)
            yb.get(nv21, pos, w)
            pos += w
        }
        // VU 교대 (NV21)
        val vb = vP.buffer
        val ub = uP.buffer
        val ps = uP.pixelStride
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                val vi = row * vP.rowStride + col * vP.pixelStride
                val ui = row * uP.rowStride + col * ps
                nv21[pos++] = vb.get(vi)
                nv21[pos++] = ub.get(ui)
            }
        }
        val yuv = Mat(h * 3 / 2, w, CvType.CV_8UC1)
        yuv.put(0, 0, nv21)
        val rgb = Mat()
        Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2RGB_NV21)
        yuv.release()
        return rgb
    }
}
