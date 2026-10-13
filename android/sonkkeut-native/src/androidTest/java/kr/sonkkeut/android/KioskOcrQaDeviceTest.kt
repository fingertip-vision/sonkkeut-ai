package kr.sonkkeut.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import kr.sonkkeut.core.*
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

@RunWith(AndroidJUnit4::class)
class KioskOcrQaDeviceTest {
    private fun screenFrame(): MatFrame {
        check(OpenCVLoader.initLocal())
        val bitmap=Bitmap.createBitmap(1280,960,Bitmap.Config.ARGB_8888)
        val canvas=Canvas(bitmap); canvas.drawColor(Color.WHITE)
        val paint=Paint(Paint.ANTI_ALIAS_FLAG).apply { color=Color.BLACK; textSize=48f }
        canvas.drawText("아메리카노",240f,360f,paint)
        canvas.drawText("3,000원",260f,420f,paint)
        canvas.drawText("라테",80f,120f,paint)
        canvas.drawText("스무디",420f,120f,paint)
        val rgba=Mat(); val rgb=Mat()
        Utils.bitmapToMat(bitmap,rgba); bitmap.recycle()
        Imgproc.cvtColor(rgba,rgb,Imgproc.COLOR_RGBA2RGB); rgba.release()
        return MatFrame(rgb)
    }
    @Test fun missedCardTextIsRetainedAndCategoryNavigationIsBlocked() {
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val flat=screenFrame()
        KoreanStructure(context).use { reader ->
            val result=reader.build(listOf(
                Element("latte","tab",Box(.01,.03,.3,.18),.9),
                Element("smoothie","tab",Box(.30,.03,.6,.18),.9)), emptyMap(), flat, 4)
            assertTrue(result.elements.any { it.kind=="text" && it.text.orEmpty().contains("아메리카노") })
            assertTrue(result.elements.filter { it.kind=="tab" }.all { it.uncertain==true })
            assertEquals("menu_ocr_incomplete",result.recognitionStatus)
            assertFalse(result.elements.any { it.kind=="menu" }) // No fabricated physical button.
        }
        flat.mat.release()
    }
    @Test fun enclosingMenuKeepsAmericanoInsteadOfNestedPriceOwnership() {
        val context=InstrumentationRegistry.getInstrumentation().targetContext
        val flat=screenFrame()
        KoreanStructure(context).use { reader ->
            val result=reader.build(listOf(
                Element("coffee-card","menu",Box(.15,.25,.65,.6),.9),
                Element("bad-price-box","price",Box(.16,.25,.65,.44),.95)), emptyMap(),flat,5)
            val card=result.elements.single { it.id=="coffee-card" }
            assertTrue(card.text.orEmpty().contains("아메리카노"))
            assertFalse(card.text.orEmpty().contains("스무디"))
            assertFalse("The clean synthetic menu must be confirmed by both real OCR engines", card.uncertain != false)
            assertTrue((card.confOcr ?: 0.0)>=.8)
        }
        flat.mat.release()
    }
}
