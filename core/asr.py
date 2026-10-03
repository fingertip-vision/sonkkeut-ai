"""M5 음성인식 (F-06 ①) — 오프라인 Whisper(faster-whisper, CPU int8).
현재 화면 메뉴명을 hotwords 로 넘겨 메뉴명 인식을 돕는다(F-06 ② 앞단). 안드로이드는 같은 모델을 whisper.cpp 로.
"""
from faster_whisper import WhisperModel


class ASR:
    def __init__(self, size="small", device="cpu", compute_type="int8"):
        self.m = WhisperModel(size, device=device, compute_type=compute_type)

    def __call__(self, wav: str, menus: list[str] | None = None) -> str:
        kw = {"hotwords": " ".join(menus)} if menus else {}
        segs, _ = self.m.transcribe(wav, language="ko", beam_size=1, vad_filter=False,
                                    condition_on_previous_text=False, **kw)
        return " ".join(s.text.strip() for s in segs).strip()
