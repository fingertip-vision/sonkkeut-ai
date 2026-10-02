package kr.sonkkeut.android

import kr.sonkkeut.core.CornerRefiner
import kr.sonkkeut.core.FrameImage
import kr.sonkkeut.core.GrayFrame
import kr.sonkkeut.core.GrayImage
import kr.sonkkeut.core.ImageOps
import kr.sonkkeut.core.Letterbox
import kr.sonkkeut.core.Mat3
import kr.sonkkeut.core.Pt
import kr.sonkkeut.core.TrackPoints
import kr.sonkkeut.core.TrackResult
import org.opencv.calib3d.Calib3d
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video

/** RGB(8UC3) 프레임 */
class MatFrame(val mat: Mat) : FrameImage {
    override val width get() = mat.cols()
    override val height get() = mat.rows()
}

class MatGray(val mat: Mat) : GrayFrame

class MatPoints(val pts: MatOfPoint2f) : TrackPoints {
    override val count get() = pts.rows()
}

/** core.ImageOps의 OpenCV 구현 (파이썬 cv2 호출과 같은 인자) */
class OpenCvImageOps : ImageOps {

    override fun toGray(frame: FrameImage): GrayFrame {
        val g = Mat()
        Imgproc.cvtColor((frame as MatFrame).mat, g, Imgproc.COLOR_RGB2GRAY)
        return MatGray(g)
    }

    override fun letterboxTensor(frame: FrameImage, lb: Letterbox): FloatArray {
        val src = (frame as MatFrame).mat
        val rs = Mat()
        Imgproc.resize(src, rs, Size(lb.newW.toDouble(), lb.newH.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR)
        val top = lb.padY.toInt()
        val left = lb.padX.toInt()
        val padded = Mat()
        Core.copyMakeBorder(rs, padded, top, lb.size - lb.newH - top, left, lb.size - lb.newW - left,
            Core.BORDER_CONSTANT, Scalar(114.0, 114.0, 114.0))
        rs.release()
        val n = lb.size * lb.size
        val bytes = ByteArray(n * 3)
        padded.get(0, 0, bytes)
        padded.release()
        val out = FloatArray(n * 3)
        for (i in 0 until n) { // HWC(RGB) → CHW, 0~1
            out[i] = (bytes[3 * i].toInt() and 0xFF) / 255f
            out[n + i] = (bytes[3 * i + 1].toInt() and 0xFF) / 255f
            out[2 * n + i] = (bytes[3 * i + 2].toInt() and 0xFF) / 255f
        }
        return out
    }

    private fun Mat3.toMat(): Mat {
        val m = Mat(3, 3, CvType.CV_64F)
        m.put(0, 0, *this.m)
        return m
    }

    override fun warp(frame: FrameImage, H: Mat3, outW: Int, outH: Int): FrameImage {
        val dst = Mat()
        val hm = H.toMat()
        Imgproc.warpPerspective((frame as MatFrame).mat, dst, hm, Size(outW.toDouble(), outH.toDouble()), Imgproc.INTER_LINEAR)
        hm.release()
        return MatFrame(dst)
    }

    override fun grayImage(frame: FrameImage): GrayImage {
        val g = Mat()
        Imgproc.cvtColor((frame as MatFrame).mat, g, Imgproc.COLOR_RGB2GRAY)
        val b = ByteArray(g.cols() * g.rows())
        g.get(0, 0, b)
        val out = GrayImage.fromBytes(g.cols(), g.rows(), b)
        g.release()
        return out
    }

    override fun seedPoints(gray: GrayFrame, quad: List<Pt>): TrackPoints? {
        val g = (gray as MatGray).mat
        val mask = Mat.zeros(g.size(), CvType.CV_8UC1)
        Imgproc.fillConvexPoly(mask, MatOfPoint(*quad.map { Point(it.x, it.y) }.toTypedArray()), Scalar(255.0))
        // 화면 테두리 바로 안쪽은 베젤과 섞이므로 조금 줄인다
        Imgproc.erode(mask, mask, Mat.ones(9, 9, CvType.CV_8UC1))
        val corners = MatOfPoint()
        Imgproc.goodFeaturesToTrack(g, corners, 200, 0.01, 8.0, mask)
        mask.release()
        if (corners.rows() == 0) return null
        val f = MatOfPoint2f(*corners.toArray().map { Point(it.x, it.y) }.toTypedArray())
        corners.release()
        return MatPoints(f)
    }

    override fun track(prev: GrayFrame, cur: GrayFrame, pts: TrackPoints, minInliers: Int): TrackResult? {
        val p0 = (pts as MatPoints).pts
        val p1 = MatOfPoint2f()
        val status = MatOfByte()
        val err = MatOfFloat()
        Video.calcOpticalFlowPyrLK((prev as MatGray).mat, (cur as MatGray).mat, p0, p1, status, err, Size(21.0, 21.0), 3)
        val st = status.toArray()
        val a = p0.toArray()
        val b = p1.toArray()
        val okA = ArrayList<Point>()
        val okB = ArrayList<Point>()
        for (i in st.indices) if (st[i].toInt() == 1) { okA += a[i]; okB += b[i] }
        status.release(); err.release(); p1.release()
        if (okA.size < minInliers) return null
        val mask = Mat()
        val src = MatOfPoint2f(*okA.toTypedArray())
        val dst = MatOfPoint2f(*okB.toTypedArray())
        val h = Calib3d.findHomography(src, dst, Calib3d.RANSAC, 3.0, mask)
        src.release(); dst.release()
        if (h.empty()) { mask.release(); return null }
        val hv = DoubleArray(9)
        h.get(0, 0, hv)
        h.release()
        val inl = ArrayList<Point>()
        val mb = ByteArray(mask.rows())
        mask.get(0, 0, mb)
        mask.release()
        for (i in mb.indices) if (mb[i].toInt() != 0) inl += okB[i]
        val ratio = inl.size.toDouble() / a.size
        return TrackResult(Mat3(hv), MatPoints(MatOfPoint2f(*inl.toTypedArray())), ratio)
    }

    override fun cropPatch(gray: GrayFrame, center: Pt, size: Double, k: Int): FloatArray {
        val p = CornerRefiner.PATCH
        val (fx, fy) = CornerRefiner.FLIPS[k]
        val s = p / size
        val ax = if (fx) -s else s
        val ay = if (fy) -s else s
        val m = Mat(2, 3, CvType.CV_64F)
        m.put(0, 0, ax, 0.0, p / 2.0 - ax * center.x, 0.0, ay, p / 2.0 - ay * center.y)
        val out = Mat()
        Imgproc.warpAffine((gray as MatGray).mat, out, m, Size(p.toDouble(), p.toDouble()), Imgproc.INTER_AREA, Core.BORDER_REPLICATE)
        m.release()
        val b = ByteArray(p * p)
        out.get(0, 0, b)
        out.release()
        return FloatArray(p * p) { (b[it].toInt() and 0xFF) / 255f }
    }

    override fun crop(frame: FrameImage, x1: Int, y1: Int, x2: Int, y2: Int): FrameImage {
        val m = (frame as MatFrame).mat
        val r = Rect(x1, y1, maxOf(1, x2 - x1), maxOf(1, y2 - y1)).let {
            Rect(it.x.coerceIn(0, m.cols() - 1), it.y.coerceIn(0, m.rows() - 1),
                minOf(it.width, m.cols() - it.x.coerceIn(0, m.cols() - 1)), minOf(it.height, m.rows() - it.y.coerceIn(0, m.rows() - 1)))
        }
        return MatFrame(m.submat(r).clone())
    }

    override fun release(obj: Any?) {
        when (obj) {
            is MatFrame -> obj.mat.release()
            is MatGray -> obj.mat.release()
            is MatPoints -> obj.pts.release()
        }
    }
}
