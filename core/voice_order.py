"""F-06 음성 주문 처리 (M5 + M6 결합) — 후보 여러 개 중 메뉴판에 맞는 의도 고르기 + 확신 없으면 다시 묻기
+ 시끄러우면 '휴대폰을 입 가까이' 안내.

  vo = VoiceOrder("ct2/whisper-elder-v3")
  r = vo.understand(audio_16k_float32, screen_menus, store_menus)
  r["decision"]: "confirm" (확인 질문) | "reask" (다시 말해 달라) | "suggest" (없는 메뉴 → 비슷한 메뉴 제안)
  r["say"]: 사용자에게 들려줄 문장,  r["intent"]: 주문 의도 JSON(명세 6장)

후보 고르기: Whisper 빔 탐색 상위 n개 문장을 모두 F-06 규칙으로 풀고, 같은 의도로 풀리는 후보들의 확률을 합친다.
메뉴명이 메뉴판과 정확히 맞을수록 가산. 합친 확률(=확신)이 기준 미만이면 추측하지 않고 다시 묻는다.
(시각장애인에게는 틀린 주문을 확인받는 것보다 한 번 더 묻는 편이 안전하다)
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from order_nlu import parse

PROMPT_START = "휴대폰을 입 가까이 대고 주문을 말씀해 주세요"
SAY_NOISY = "주변이 시끄러워요. 휴대폰을 입에 더 가까이 대고 다시 말씀해 주세요"
SAY_QUIET = "잘 들리지 않았어요. 휴대폰을 입 가까이 대고 조금 더 크게 말씀해 주세요"
SAY_AGAIN = "다시 한 번 말씀해 주세요"
OPTION_WORDS = ["톨", "그란데", "벤티", "아이스", "따뜻한", "샷 추가", "시럽 추가", "포장", "매장", "잔", "개"]


def snr_db(x: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """간이 신호대잡음비와 말소리 크기(dBFS). 25ms 프레임 에너지의 상위 10%(말) 대 하위 15%(배경)."""
    f = int(0.025 * sr)
    n = len(x) // f
    if n < 8:
        return 0.0, -90.0
    e = np.mean(x[: n * f].reshape(n, f) ** 2, axis=1) + 1e-12
    e.sort()
    noise = np.mean(e[: max(1, int(n * 0.15))])
    speech = np.mean(e[int(n * 0.9):])
    return float(10 * math.log10(speech / noise)), float(10 * math.log10(speech))


def intent_key(intent):
    items = sorted((it["menu"], it["qty"], (it["options"] or {}).get("temp") or "", (it["options"] or {}).get("size") or "",
                    tuple((it["options"] or {}).get("extras") or [])) for it in intent["items"])
    return tuple(items), intent.get("dine")


class VoiceOrder:
    def __init__(self, model="ct2/whisper-elder-v3", device="cuda", compute_type="float16", n_best=5,
                 threshold=0.7, noisy_db=22.0, quiet_dbfs=-38.0):
        """threshold: 확신이 이보다 낮으면 다시 묻는다 (주문 100개 시험에서 0.7 이 틀린 확인을 줄이면서 다시 묻기는 적었다).
        noisy_db: snr_db() 추정값 기준. 추정값은 실제 SNR 보다 약 14dB 높게 나온다(합성 시험: 5dB→중앙값 19, 15dB→28)
                  → 22 미만이면 '시끄러우니 가까이' 안내. 실제 폰 녹음으로 다시 맞춰야 한다."""
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        self.m = WhisperModel(model, device=device, compute_type=compute_type)
        self.tok = Tokenizer(self.m.hf_tokenizer, self.m.model.is_multilingual, task="transcribe", language="ko")
        self.n, self.threshold, self.noisy_db, self.quiet_dbfs = n_best, threshold, noisy_db, quiet_dbfs

    def nbest(self, audio: np.ndarray, hotwords: list[str]):
        """빔 탐색 상위 n개 (문장, 누적 로그확률, 평균 로그확률), 무음 확률."""
        from faster_whisper.audio import pad_or_trim
        feats = pad_or_trim(self.m.feature_extractor(audio)[:, :3000])
        enc = self.m.encode(feats)
        prompt = self.m.get_prompt(self.tok, [], without_timestamps=True, hotwords=" ".join(hotwords) if hotwords else None)
        r = self.m.model.generate(enc, [prompt], beam_size=self.n, num_hypotheses=self.n, return_scores=True,
                                  return_no_speech_prob=True, max_length=224, suppress_blank=True,
                                  suppress_tokens=[-1])[0]
        out = []
        for ids, sc in zip(r.sequences_ids, r.scores):
            L = len(ids)
            cum = sc * L                      # length_penalty=1 → score = cum / L
            out.append((self.tok.decode(ids).strip(), cum, cum / (L + 1)))
        return out, float(r.no_speech_prob)

    def understand(self, audio: np.ndarray, screen_menus: list[str], store_menus: list[str] | None = None,
                   use_nbest: bool = True, cached=None) -> dict:
        """cached=(hyps, no_speech_prob) 를 주면 음성인식을 다시 돌리지 않는다(같은 소리로 방식 비교용)."""
        snr, level = snr_db(audio)
        hyps, nsp = cached or self.nbest(audio, screen_menus + OPTION_WORDS)
        if not use_nbest:
            hyps = hyps[:1]
        mx = max(h[1] for h in hyps)
        w = [math.exp(h[1] - mx) for h in hyps]
        z = sum(w)
        groups, best_parse, suggest_say = defaultdict(float), {}, None
        for (text, cum, avg), wi in zip(hyps, w):
            p = parse(text, screen_menus, store_menus)
            if not p["items"]:
                if "없습니다" in p.get("say", "") and suggest_say is None:
                    suggest_say = p["say"]               # 주문은 했는데 메뉴판에 없는 메뉴
                continue
            q = min(it.get("score", 100) for it in p["items"]) / 100      # 메뉴명이 정확히 맞을수록 1
            k = intent_key(p)
            groups[k] += wi / z * q
            best_parse.setdefault(k, (p, text))
        res = {"snr_db": round(snr, 1), "level_dbfs": round(level, 1), "no_speech": round(nsp, 3),
               "hypotheses": [h[0] for h in hyps]}
        noisy, quiet = snr < self.noisy_db, level < self.quiet_dbfs
        if not groups:
            if suggest_say:
                return {**res, "decision": "suggest", "say": suggest_say, "intent": None}
            return {**res, "decision": "reask", "say": SAY_NOISY if noisy else SAY_QUIET if quiet else SAY_AGAIN,
                    "intent": None}
        k, conf = max(groups.items(), key=lambda kv: kv[1])
        p, text = best_parse[k]
        res.update(confidence=round(conf, 3), text=text, intent={x: p[x] for x in ("items", "dine", "source") if x in p})
        if conf < self.threshold:            # 추측하지 않는다
            res.update(decision="reask", say=SAY_NOISY if noisy else SAY_QUIET if quiet else SAY_AGAIN)
        else:
            res.update(decision="confirm", say=p["say"])
        return res
