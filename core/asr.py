"""M5 음성인식 (F-06 ①) — 오프라인 Whisper(faster-whisper, CPU int8).
현재 화면 메뉴명을 hotwords 로 넘겨 메뉴명 인식을 돕는다(F-06 ② 앞단).
통합 Android는 같은 팀 가중치를 CTranslate2 ARM64 RUY JNI로 실행한다.
"""
from faster_whisper import WhisperModel
import os


class ASR:
    def __init__(self, size=None, device="cpu", compute_type="int8"):
        # Benchmarks may explicitly request stock small/base. Application defaults
        # must name the team's model rather than silently downloading a baseline.
        size = size or os.environ.get("SONKKEUT_ASR_MODEL")
        if not size:
            raise ValueError("Set SONKKEUT_ASR_MODEL to the verified whisper-elder-v3 directory")
        self.m = WhisperModel(size, device=device, compute_type=compute_type)

    def __call__(self, wav: str, menus: list[str] | None = None) -> str:
        kw = {"hotwords": " ".join(menus)} if menus else {}
        segs, _ = self.m.transcribe(wav, language="ko", beam_size=1, vad_filter=False,
                                    condition_on_previous_text=False, **kw)
        return " ".join(s.text.strip() for s in segs).strip()
