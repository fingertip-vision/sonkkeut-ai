package kr.sonkkeut.android

import android.content.Context
import android.graphics.Bitmap
import android.media.Image
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.concurrent.TimeUnit

data class DetailLine(val text: String, val confidence: Double, val box: List<Double>, val tile: Int)
data class DetailScan(val lines: List<DetailLine>,val tilesCompleted: Int,val elapsedMs: Long,val timedOut: Boolean)

/** Bounded worker-only OCR. Partial-image coordinates must never be used as kiosk targets. */
class DetailScanner(context: Context): AutoCloseable {
    private val yuv=YuvToRgb()
    private val ml=TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
    private val m3=try { KioskCtcRecognizer(context) } catch(e: Throwable) { ml.close(); throw e }
    @Synchronized fun scan(image: Image,rotation: Int,cancelled: ()->Boolean): DetailScan {
        val rgb=yuv.convert(image); val upright=Mat()
        try {
            when(rotation) { 90 -> Core.rotate(rgb,upright,Core.ROTATE_90_CLOCKWISE); 180 -> Core.rotate(rgb,upright,Core.ROTATE_180); 270 -> Core.rotate(rgb,upright,Core.ROTATE_90_COUNTERCLOCKWISE); else -> rgb.copyTo(upright) }
            return scanRgb(upright,cancelled)
        } finally { rgb.release(); upright.release() }
    }
    @Synchronized fun scanRgb(input: Mat,cancelled: ()->Boolean = {false}): DetailScan {
        val start=android.os.SystemClock.elapsedRealtime(); val lines=mutableListOf<DetailLine>(); var completed=0
        val frame=Mat()
        try {
            if(maxOf(input.cols(),input.rows())>2400) { val ratio=2400.0/maxOf(input.cols(),input.rows()); Imgproc.resize(input,frame,Size(input.cols()*ratio,input.rows()*ratio)) } else input.copyTo(frame)
            for(y in 0..2) for(x in 0..1) {
                if(cancelled() || android.os.SystemClock.elapsedRealtime()-start>=8000) return DetailScan(lines,completed,android.os.SystemClock.elapsedRealtime()-start,true)
                val x1=(frame.cols()*x*.4).toInt(); val y1=(frame.rows()*y*.3).toInt()
                val tile=frame.submat(Rect(x1,y1,(frame.cols()*.6).toInt().coerceAtMost(frame.cols()-x1),(frame.rows()*.4).toInt().coerceAtMost(frame.rows()-y1)))
                val bitmap=Bitmap.createBitmap(tile.cols(),tile.rows(),Bitmap.Config.ARGB_8888)
                var handedOff=false
                try {
                    Utils.matToBitmap(tile,bitmap)
                    val task=ml.process(InputImage.fromBitmap(bitmap,0)); task.addOnCompleteListener { bitmap.recycle() }; handedOff=true
                    val read=runCatching { Tasks.await(task,1500,TimeUnit.MILLISECONDS) }.getOrNull() ?: continue
                    completed++
                    for(line in read.textBlocks.flatMap { it.lines }.take(12)) {
                        if(cancelled() || android.os.SystemClock.elapsedRealtime()-start>=8000) break
                        val b=line.boundingBox ?: continue
                        val left=(b.left-2).coerceIn(0,tile.cols()-1); val top=(b.top-2).coerceIn(0,tile.rows()-1)
                        val right=(b.right+2).coerceIn(left+1,tile.cols()); val bottom=(b.bottom+2).coerceIn(top+1,tile.rows())
                        val crop=tile.submat(Rect(left,top,right-left,bottom-top))
                        try {
                            val custom=runCatching { m3.recognize(crop) }.getOrNull() ?: continue
                            if(custom.uncertain || custom.confidence<.8 || custom.text.isBlank()) continue
                            val box=listOf((x1+left).toDouble()/frame.cols(),(y1+top).toDouble()/frame.rows(),(x1+right).toDouble()/frame.cols(),(y1+bottom).toDouble()/frame.rows())
                            val next=DetailLine(custom.text,custom.confidence,box,y*2+x)
                            if(lines.none { sameLine(it,next) }) lines+=next
                        } finally { crop.release() }
                    }
                } finally { if(!handedOff) bitmap.recycle(); tile.release() }
            }
            return DetailScan(lines,completed,android.os.SystemClock.elapsedRealtime()-start,false)
        } finally { frame.release() }
    }
    private fun sameLine(a: DetailLine,b: DetailLine): Boolean {
        if(a.text.replace(" ","")!=b.text.replace(" ","")) return false
        val intersection=(minOf(a.box[2],b.box[2])-maxOf(a.box[0],b.box[0])).coerceAtLeast(0.0)*(minOf(a.box[3],b.box[3])-maxOf(a.box[1],b.box[1])).coerceAtLeast(0.0)
        val area=minOf((a.box[2]-a.box[0])*(a.box[3]-a.box[1]),(b.box[2]-b.box[0])*(b.box[3]-b.box[1]))
        return area>0 && intersection/area>.5
    }
    @Synchronized override fun close() { try { ml.close() } finally { m3.close() } }
}
