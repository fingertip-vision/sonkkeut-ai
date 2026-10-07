// 잘못된 "지금 누르세요"·거짓 성공을 막는 Kotlin 파이프라인 연결부 테스트 (파이썬 tests/test_pipeline_safety.py와 같은 상황)
//
//   A1 다시 읽은 화면에서 같은 자리·같은 종류의 '다른 글자' 버튼으로 목표가 바뀜
//   A2 강제로 다시 읽은 키프레임을 누름 성공으로 판정
//   A3 다시 읽기 전에 고른 요소 id로 목표 지정
//   A5 손끝 신뢰도가 낮은 프레임까지 머무름 시간에 넣음
// PipelineSimTest.kt의 가짜 영상 처리(SimOps·FakeM1·FakeM2·FakeHands)를 그대로 쓰고, 글자는 StructureProvider로 붙인다.
// 같은 패키지의 다른 테스트 main과 겹치지 않게 진입점은 object 안에 둔다: java ... PipelineSafetyTest

import kr.sonkkeut.core.Box
import kr.sonkkeut.core.Element
import kr.sonkkeut.core.FrameResult
import kr.sonkkeut.core.Fingertip
import kr.sonkkeut.core.Guide
import kr.sonkkeut.core.Mat3
import kr.sonkkeut.core.Pt
import kr.sonkkeut.core.ScreenStructure
import kr.sonkkeut.core.StructureProvider
import kr.sonkkeut.core.UNIT_SQUARE
import kr.sonkkeut.core.VisionPipeline

object PipelineSafetyTest {
    private val MENU = Box(0.10, 0.10, 0.90, 0.40)
    private val CORNERS = Camera(0).corners(0.0) // 휴대폰을 움직이지 않는 상황
    private val SCR2IMG = Mat3.perspective(UNIT_SQUARE, CORNERS)
    private val INSIDE = Pt(MENU.cx, MENU.cy)

    private fun screen(id: Int) = Screen(id, listOf("menu" to MENU))

    /** 화면 id별 버튼 글자 (F-04 문자 인식 결과 역할) */
    private class Labels(val byScreen: Map<Int, String>) : StructureProvider {
        override fun build(elements: List<Element>, crops: Map<String, kr.sonkkeut.core.FrameImage>, flat: kr.sonkkeut.core.FrameImage, keyframeId: Int): ScreenStructure {
            val sid = (flat as SimFlat).src.screen.id
            for (e in elements) { e.text = byScreen[sid]; e.confOcr = 0.95; e.uncertain = false }
            return ScreenStructure.basic(elements, keyframeId)
        }
    }

    private fun pipe(labels: Map<Int, String>): VisionPipeline {
        val ops = SimOps()
        return VisionPipeline(FakeM1(ops, 0, noise = 0.0), FakeM2(ops), ops, FakeHands(), structureProvider = Labels(labels))
    }

    private fun frame(id: Int, finger: Pt?) = SimFrame(screen(id), CORNERS, finger?.let { handPoints(SCR2IMG.apply(it)) })

    private fun run(p: VisionPipeline, id: Int, finger: Pt?, t0: Double, seconds: Double): List<FrameResult> =
        (0 until (seconds * 15).toInt()).map { p.process(frame(id, finger), t0 + it / 15.0) }

    private fun presses(rs: List<FrameResult>) = rs.count { it.event?.type == "press" }

    private fun check(name: String, ok: Boolean) {
        println((if (ok) "PASS " else "FAIL ") + name)
        if (!ok) failures++
    }

    private var failures = 0

    private fun a1DifferentTextIsNotTheTarget() {
        val p = pipe(mapOf(1 to "아메리카노", 2 to "카페라떼"))
        p.process(frame(1, null), 0.0)
        p.setTarget(p.elements.first { it.kind == "menu" }.id)
        // 탭이 넘어가 같은 자리에 카페라떼가 나타남 (손은 아직 화면 밖)
        val changed = run(p, 2, null, 1.0, 1.0)
        val after = run(p, 2, INSIDE, 2.0, 2.0)
        check("A1 같은 자리 다른 글자 → 목표 사라짐 처리", changed.any { it.targetMissing } && p.guide.target?.conf == 0.0)
        check("A1 다른 글자 버튼 위에서 '누르세요' 없음", presses(after) == 0)
    }

    private fun a1SameTextKeepsGuiding() {
        val p = pipe(mapOf(1 to "아메리카노", 3 to "아메 리카노"))
        p.process(frame(1, null), 0.0)
        p.setTarget(p.elements.first { it.kind == "menu" }.id)
        val rs = run(p, 3, null, 1.0, 1.0)
        check("A1 같은 글자로 다시 읽으면 목표 유지", rs.any { it.keyframe } && rs.none { it.targetMissing } && (p.guide.target?.conf ?: 0.0) > 0.5)
    }

    private fun a2ForcedKeyframeIsNotSuccess() {
        val p = pipe(mapOf(1 to "아메리카노"))
        p.process(frame(1, null), 0.0)
        p.setTarget(p.elements.first { it.kind == "menu" }.id)
        val pressed = run(p, 1, INSIDE, 0.1, 1.5)
        p.requestKeyframe() // 화면은 그대로인데 다시 읽기를 강제
        val rs = run(p, 1, INSIDE, 1.7, 2.0)
        val verdicts = rs.mapNotNull { it.verdict }
        check("A2 '누르세요' 1회 후 판정 대기", presses(pressed) == 1)
        check("A2 강제 키프레임은 판정하지 않음", rs.first().keyframe && rs.first().verdict == null)
        // 1.5초 뒤 '눌리지 않음' → 손끝이 그대로면 다시 누르라고 안내하므로 판정이 더 나올 수 있다. 성공만 없으면 된다.
        check("A2 화면이 그대로면 '눌리지 않음'", verdicts.isNotEmpty() && verdicts[0].reason == "no_change" && verdicts.none { it.result == "success" })
    }

    private fun a2RealChangeIsJudged() {
        val p = pipe(mapOf(1 to "아메리카노", 4 to "옵션"))
        p.process(frame(1, null), 0.0)
        p.setTarget(p.elements.first { it.kind == "menu" }.id)
        run(p, 1, INSIDE, 0.1, 1.5)
        val rs = run(p, 4, null, 1.7, 1.0) // 실제로 화면이 바뀜
        check("A2 실제 화면 변화는 성공으로 판정", rs.mapNotNull { it.verdict }.map { it.result } == listOf("success"))
    }

    private fun a3RejectsOldKeyframeId() {
        val p = pipe(mapOf(1 to "아메리카노", 2 to "카페라떼"))
        val first = p.process(frame(1, null), 0.0).keyframeId
        val id = p.elements.first { it.kind == "menu" }.id
        run(p, 2, null, 1.0, 1.0) // 확인을 기다리는 사이 화면을 다시 읽음
        val rejected = runCatching { p.setTarget(id, keyframeId = first) }.isFailure
        p.setTarget(id, keyframeId = p.elementsKeyframeId)
        check("A3 이전 키프레임의 id는 거부", p.elementsKeyframeId != first && rejected)
        check("A3 현재 키프레임의 id는 허용", p.guide.target?.text == "카페라떼")
    }

    private fun a5LowConfidenceFramesDoNotCount() {
        val g = Guide()
        g.setTarget(Element("e1", "menu", MENU, 0.9, text = "아메리카노"), 1.0)
        var t = 0.0
        var early = false
        repeat(8) { early = early || g.update(Fingertip(INSIDE, 0.3), t)?.type == "press"; t += 1 / 15.0 }
        val firstConfident = t
        early = early || g.update(Fingertip(INSIDE, 0.9), t)?.type == "press"
        var pressAt: Double? = null
        repeat(8) {
            t += 1 / 15.0
            if (g.update(Fingertip(INSIDE, 0.9), t)?.type == "press" && pressAt == null) pressAt = t
        }
        check("A5 신뢰도 낮은 프레임은 머무름에 넣지 않음", !early && pressAt != null && pressAt!! - firstConfident >= 0.3)
    }

    @JvmStatic
    fun main(args: Array<String>) {
        a1DifferentTextIsNotTheTarget()
        a1SameTextKeepsGuiding()
        a2ForcedKeyframeIsNotSuccess()
        a2RealChangeIsJudged()
        a3RejectsOldKeyframeId()
        a5LowConfidenceFramesDoNotCount()
        println(if (failures == 0) "\n모두 통과" else "\n실패 $failures 건")
        if (failures != 0) kotlin.system.exitProcess(1)
    }
}
