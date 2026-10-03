"""Actual team fine-tuned CTranslate2 Whisper v3; never substitute stock weights."""
from __future__ import annotations

from pathlib import Path


class WhisperSpeech:
    model_id = "asr-whisper-elder-v3-ct2"

    def __init__(self, model_dir: str | Path, threads: int = 4):
        from faster_whisper import WhisperModel
        path = Path(model_dir)
        if not (path / "model.bin").is_file():
            raise FileNotFoundError("Download and verify the team's asr-whisper-elder-v3 release first")
        self.model = WhisperModel(str(path), device="cpu", compute_type="int8",
                                  cpu_threads=threads, num_workers=1, local_files_only=True)

    def __call__(self, audio_path, menus=None):
        hints = " ".join((menus or []) + ["아이스", "따뜻한", "톨", "그란데", "벤티", "포장", "매장"])
        segments, _ = self.model.transcribe(audio_path, language="ko", beam_size=1,
                                           condition_on_previous_text=False,
                                           vad_filter=False, hotwords=hints)
        return " ".join(segment.text.strip() for segment in segments).strip()
