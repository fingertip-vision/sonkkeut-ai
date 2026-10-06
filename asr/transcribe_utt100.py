"""주문 발화 100개 세트(TTS) 받아쓰기 → core/runs/asr_<tag>_<cond>.json (core/eval_f06 형식).
화면 메뉴명·옵션 단어를 hotwords 로 넘긴다. GPU(faster-whisper + torch 의 cuBLAS).
  python transcribe_utt100.py --model ct2/whisper-elder-v1 --tag elder_v1
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch  # noqa: F401
os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
from faster_whisper import WhisperModel  # noqa: E402

CORE = Path("D:/sonkkeutgil/core")
MENUS = ["아메리카노", "카페라떼", "카푸치노", "바닐라라떼", "카라멜마끼아또", "카페모카", "콜드브루", "에스프레소", "헤이즐넛라떼",
         "돌체라떼", "딸기라떼", "초코라떼", "녹차라떼", "레몬에이드", "자몽에이드", "청포도에이드", "딸기스무디", "망고스무디",
         "캐모마일", "페퍼민트", "얼그레이", "유자차", "레몬차", "복숭아아이스티", "치즈케이크", "티라미수", "크루아상", "베이글",
         "마카롱", "소금빵"]
HOT = MENUS + ["톨", "그란데", "벤티", "아이스", "따뜻한", "샷 추가", "시럽 추가", "포장", "매장", "잔", "개"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-hot", action="store_true")
    a = ap.parse_args()
    m = WhisperModel(a.model, device="cuda", compute_type="float16")
    utts = json.load(open(CORE / "data/utt100.json", encoding="utf-8"))
    for cond in ("clean", "noisy15", "noisy5"):
        out, lat = [], []
        for u in utts:
            t0 = time.perf_counter()
            segs, _ = m.transcribe(str(CORE / f"data/utt100_wav/{cond}/{u['id']:03d}.wav"), language="ko", beam_size=1,
                                   condition_on_previous_text=False, hotwords=None if a.no_hot else " ".join(HOT))
            out.append({"id": u["id"], "ref": u["text"], "hyp": " ".join(s.text.strip() for s in segs)})
            lat.append(time.perf_counter() - t0)
        json.dump(out, open(CORE / f"runs/asr_{a.tag}_{'nohw' if a.no_hot else 'hw'}_{cond}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(cond, f"평균 지연 {sum(lat) / len(lat):.3f}s")


if __name__ == "__main__":
    main()
