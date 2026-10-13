package kr.sonkkeut.core

import org.junit.Assert.*
import org.junit.Test
import kr.sonkkeut.android.*

class KioskQaTest {
    private val box=Box(.2,.3,.6,.4)
    private fun menu(uncertain: Boolean=false) = Element("americano","menu",box,.9,text="아메리카노",confOcr=.95,uncertain=uncertain)
    @Test fun agreeingKoreanEnginesKeepObservedLabel() {
        val e=OcrEvidence.line("아메리카노",box,CtcReading("아메리카노",.96,false))
        assertFalse(e.uncertain); assertEquals("아메리카노",e.text)
    }
    @Test fun confidentWrongLatteIsNotAccepted() {
        val e=OcrEvidence.line("아메리카노",box,CtcReading("라테",.999,false))
        assertTrue(e.uncertain); assertEquals(0.0,e.confidence,0.0); assertEquals("아메리카노",e.text)
    }
    @Test fun confidentWrongSmoothieIsNotAccepted() {
        assertTrue(OcrEvidence.line("아메리카노",box,CtcReading("스무디",.99,false)).uncertain)
    }
    @Test fun benignSpacingAndPricePunctuationCanAgree() {
        assertFalse(OcrEvidence.line("아메리카노 3,000원",box,CtcReading("아메리카노3000원",.9,false)).uncertain)
    }
    @Test fun uncertainAndMissingReadingsCannotBecomeConfirmed() {
        assertTrue(OcrEvidence.line("아메리카노",box,null).uncertain)
        assertTrue(OcrEvidence.line("아메리카노",box,CtcReading("아메리카노",.7,true)).uncertain)
        assertTrue(OcrEvidence.line("아메리카노",box,CtcReading("아메리카노",Double.NaN,false)).uncertain)
    }
    @Test fun cardOwnsNameInsteadOfOverlappingNestedPrice() {
        val card=Element("card","menu",Box(.1,.2,.8,.7),.8)
        val price=Element("price","price",box,.9)
        assertEquals("card",OcrEvidence.owner(box,listOf(price,card))!!.id)
    }
    @Test fun largeMisalignedTextCannotBeAssignedByItsCenterAlone() {
        val small=Element("tiny","menu",Box(.39,.33,.41,.35),.9)
        assertNull(OcrEvidence.owner(box,listOf(small)))
    }
    @Test fun unconfirmedProductStopsUnrelatedCategoryNavigation() {
        assertTrue(OcrEvidence.blockCategoryNavigation(listOf(menu(true)),emptyList()))
    }
    @Test fun reliableProductDoesNotDisableCategoriesWithoutMissingEvidence() {
        assertFalse(OcrEvidence.blockCategoryNavigation(listOf(menu()),emptyList()))
    }
    @Test fun missedAmericanoWithNearbyPriceStopsCategories() {
        val label=LineEvidence("아메리카노",box,.95,false)
        val price=LineEvidence("3,000원",Box(.25,.42,.5,.46),.9,false)
        val tabs=listOf(Element("latte","tab",Box(.0,.1,.2,.2),.9,text="라테",confOcr=.9,uncertain=false),
            Element("smoothie","tab",Box(.2,.1,.4,.2),.9,text="스무디",confOcr=.9,uncertain=false))
        assertTrue(OcrEvidence.blockCategoryNavigation(tabs,listOf(label,price)))
    }
    @Test fun priceInAnotherColumnIsNotEvidenceForTitle() {
        assertFalse(OcrEvidence.blockCategoryNavigation(emptyList(),listOf(LineEvidence("음료",box,.9,false),
            LineEvidence("3,000원",Box(.85,.42,.99,.46),.9,false))))
    }
    @Test fun recoveredTextRemainsReadOnlyAndKeepsCoordinates() {
        val region=OcrEvidence.readOnlyRegions(listOf(LineEvidence("아메리카노",box,.95,false)),7).single()
        assertEquals("text",region.kind); assertEquals(box,region.box); assertEquals("ocr_7_0",region.id)
        val resolver=ScreenMenuResolver(listOf(MenuDocument("아메리카노",id="coffee")))
        val match=resolver.resolve("coffee",listOf(MenuTextRegion(region.id,region.kind,region.text!!,region.box.toList(),true)))
        assertEquals(ScreenMenuStatus.FOUND,match.status); assertEquals(region.id,match.candidate!!.regionId)
    }
    @Test fun categoryCannotBeResolvedAsAmericano() {
        val resolver=ScreenMenuResolver(listOf(MenuDocument("아메리카노",id="coffee")))
        for(name in listOf("라테","스무디","주스 및 에이드")) assertEquals(ScreenMenuStatus.MISSING,
            resolver.resolve("coffee",listOf(MenuTextRegion("category","tab",name,box.toList(),true))).status)
    }
    private fun pose(points: List<List<Pt>>, scores: List<Float>, confidences: List<Float>): FloatArray {
        val n=points.size; val out=FloatArray(17*n)
        for(i in 0 until n) {
            out[i]=320f; out[n+i]=320f; out[2*n+i]=400f; out[3*n+i]=500f; out[4*n+i]=scores[i]
            for(k in 0 until 4) { out[(5+3*k)*n+i]=points[i][k].x.toFloat(); out[(6+3*k)*n+i]=points[i][k].y.toFloat(); out[(7+3*k)*n+i]=confidences[i] }
        }
        return out
    }
    private val complete=listOf(Pt(100.0,80.0),Pt(540.0,80.0),Pt(540.0,560.0),Pt(100.0,560.0))
    @Test fun clippedTopPoseCannotHideCompleteScreen() {
        val clipped=complete.map { Pt(it.x-200,it.y) }
        val frame=object:FrameImage { override val width=640; override val height=640 }
        val estimator=ScreenPlaneEstimator(object:TensorModel { override fun run(input:FloatArray,shape:LongArray)=pose(listOf(clipped,complete),listOf(.99f,.8f),listOf(.9f,.9f)) },NoImageOps)
        val detection=estimator.detect(frame)!!
        assertEquals(complete,detection.corners)
    }
    @Test fun lowKeypointTopPoseCannotHideCompleteScreen() {
        val frame=object:FrameImage { override val width=640; override val height=640 }
        val estimator=ScreenPlaneEstimator(object:TensorModel { override fun run(input:FloatArray,shape:LongArray)=pose(listOf(complete,complete),listOf(.99f,.8f),listOf(.2f,.9f)) },NoImageOps)
        assertTrue(estimator.detect(frame)!!.kpConf.all { it>.8 })
    }
    @Test fun decoderRejectsNonFiniteCandidatesAndMalformedShape() {
        assertNull(YoloDecoder.decodePoseBest(pose(listOf(complete),listOf(Float.NaN),listOf(.9f)),4,1,Letterbox.of(640,640),.25))
        try { YoloDecoder.decodePoseBest(FloatArray(9),4,1,Letterbox.of(640,640),.25); fail("invalid pose accepted") } catch(_:IllegalArgumentException) {}
    }
    private object NoImageOps: ImageOps {
        override fun letterboxTensor(frame:FrameImage,lb:Letterbox)=FloatArray(0)
        override fun toGray(frame:FrameImage):GrayFrame=error("unused")
        override fun warp(frame:FrameImage,H:Mat3,outW:Int,outH:Int):FrameImage=error("unused")
        override fun grayImage(frame:FrameImage):GrayImage=error("unused")
        override fun seedPoints(gray:GrayFrame,quad:List<Pt>):TrackPoints?=null
        override fun track(prev:GrayFrame,cur:GrayFrame,pts:TrackPoints,minInliers:Int):TrackResult?=null
        override fun cropPatch(gray:GrayFrame,center:Pt,size:Double,k:Int):FloatArray=error("unused")
        override fun crop(frame:FrameImage,x1:Int,y1:Int,x2:Int,y2:Int):FrameImage=error("unused")
    }
}
