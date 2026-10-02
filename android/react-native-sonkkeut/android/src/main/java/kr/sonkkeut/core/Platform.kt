package kr.sonkkeut.core

/**
 * 플랫폼(안드로이드)이 구현하는 부분.
 * core 패키지는 안드로이드·OpenCV·ONNX Runtime에 의존하지 않아 PC(JVM)에서 테스트할 수 있다.
 * 안드로이드 구현: kr.sonkkeut.android.OpenCvImageOps / OnnxModel / MediaPipeHands
 */

/** 카메라 프레임 (정방향으로 회전된 컬러 이미지). 실제 내용은 구현체가 들고 있다. */
interface FrameImage {
    val width: Int
    val height: Int
}

/** 흑백 프레임 (광류 추적용). 구현체가 들고 있다. */
interface GrayFrame

/** 추적할 특징점 묶음. 구현체가 들고 있다. */
interface TrackPoints {
    val count: Int
}

class TrackResult(val homography: Mat3 /* 직전 → 지금 */, val inliers: TrackPoints, val inlierRatio: Double)

interface ImageOps {
    fun toGray(frame: FrameImage): GrayFrame

    /** 레터박스 640×640, RGB 0~1, CHW float (Ultralytics와 같은 배치) */
    fun letterboxTensor(frame: FrameImage, lb: Letterbox): FloatArray

    /** 카메라 프레임을 H(카메라 픽셀 → 출력 픽셀)로 펼친다 */
    fun warp(frame: FrameImage, H: Mat3, outW: Int, outH: Int): FrameImage

    /** 펼친 작은 화면 → 흑백 (키프레임 비교용) */
    fun grayImage(frame: FrameImage): GrayImage

    /** 사각형 안쪽에서 추적할 특징점 고르기 (goodFeaturesToTrack) */
    fun seedPoints(gray: GrayFrame, quad: List<Pt>): TrackPoints?

    /** 직전 흑백 → 지금 흑백으로 특징점을 따라가고 RANSAC 호모그래피 (calcOpticalFlowPyrLK + findHomography) */
    fun track(prev: GrayFrame, cur: GrayFrame, pts: TrackPoints, minInliers: Int): TrackResult?

    /** 꼭짓점 보정망 입력 패치: center 주변 size×size를 64×64로, 꼭짓점 k를 '왼쪽 위' 모양으로 뒤집어 */
    fun cropPatch(gray: GrayFrame, center: Pt, size: Double, k: Int): FloatArray

    /** 요소 잘라내기 (문자 인식 입력) — 펼친 이미지 픽셀 박스 */
    fun crop(frame: FrameImage, x1: Int, y1: Int, x2: Int, y2: Int): FrameImage

    fun release(obj: Any?) {}
}

/** ONNX 모델 한 개 */
interface TensorModel {
    /** input: 평탄화된 텐서, shape: 입력 모양 → 평탄화된 첫 번째 출력 */
    fun run(input: FloatArray, shape: LongArray): FloatArray
}

/** 손 관절 21점 (카메라 픽셀) + 점수 */
class Hand(val points: List<Pt>, val score: Double)

interface HandSource {
    fun detect(frame: FrameImage, tSeconds: Double): List<Hand>
    fun close() {}
}

/**
 * 언어 쪽(노현석: F-04 문자 인식, F-05 화면 구조화)이 구현한다.
 * 키프레임마다 한 번 불린다. 기본값은 요소만 담고 screen_type=unknown.
 */
fun interface StructureProvider {
    fun build(elements: List<Element>, crops: Map<String, FrameImage>, flat: FrameImage, keyframeId: Int): ScreenStructure
}
