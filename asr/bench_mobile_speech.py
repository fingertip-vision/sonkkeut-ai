"""Reproducible CT2 model test. Input labels are never passed as decoder hints.

Report raw CER and hypotheses; order acceptance is scored separately by the app.
Synthetic noise is not a substitute for real microphone/cafe evaluation.
"""
import argparse
import hashlib
import json
import re
import time
import wave
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.audio import pad_or_trim
from faster_whisper.tokenizer import Tokenizer
from rapidfuzz.distance import Levenshtein


def normalize(text):
    return re.sub(r'[^a-z0-9가-힣]', '', text.lower())


def signal(case, folder):
    if case.get('signal'):
        x = np.zeros(16000 * 3, dtype=np.float32)
    else:
        with wave.open(str(folder / f"{case['id']:03d}.wav")) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
            x = np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float32) / 32768
    rng = np.random.default_rng(53000 + case['id'])
    white = rng.normal(size=len(x))
    # Reproducible coloured noise plus appliance hum, not real cafe recordings.
    noise = np.convolve(white, np.ones(8) / 8, mode='same')
    noise += .15 * np.sin(np.arange(len(x)) * 2 * np.pi * 120 / 16000)
    noise /= np.sqrt(np.mean(noise ** 2)) + 1e-8
    if case.get('signal') == 'noise':
        x = (noise * .025).astype(np.float32)
    return x, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--manifest', type=Path, default=Path(__file__).parent / 'data/kiosk_speech_cases.json')
    ap.add_argument('--audio', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--beam', type=int, default=5)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--decoder', default='android', choices=['android', 'python'], help='Android uses one direct CT2 generation, without Python retries/VAD')
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    t = time.perf_counter()
    model = WhisperModel(args.model, device=args.device, compute_type='int8' if args.device == 'cpu' else 'float16',
                         cpu_threads=4, num_workers=1, local_files_only=True)
    load = time.perf_counter() - t
    hints = ' '.join([m['name'] for m in manifest['menus']] + ['아이스', '따뜻한', '톨', '그란데', '벤티', '포장', '매장'])
    tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task='transcribe', language='ko')
    prompt = model.get_prompt(tokenizer, previous_tokens=[], without_timestamps=True, hotwords=hints)
    rows = []
    for condition, snr in [('clean', None), ('synthetic15', 15), ('synthetic5', 5)]:
        for case in manifest['cases']:
            x, noise = signal(case, args.audio)
            if snr is not None and not case.get('signal'):
                x = np.clip(x + noise * np.sqrt(np.mean(x ** 2)) / 10 ** (snr / 20), -1, 1).astype(np.float32)
            t = time.perf_counter()
            if args.decoder == 'android':
                features = pad_or_trim(model.feature_extractor(x))
                encoded = model.encode(features)
                result = model.model.generate(encoded, [prompt], beam_size=args.beam, max_length=448,
                                              return_scores=True, return_no_speech_prob=True)[0]
                hyp = tokenizer.decode(result.sequences_ids[0]).strip()
                no_speech = result.no_speech_prob
            else:
                segments, _ = model.transcribe(x, language='ko', beam_size=args.beam, condition_on_previous_text=False,
                                              vad_filter=False, hotwords=hints)
                segments = list(segments)
                hyp = ' '.join(s.text.strip() for s in segments).strip()
                no_speech = max((s.no_speech_prob for s in segments), default=1.0)
            ref, norm_hyp = normalize(case['text']), normalize(hyp)
            rows.append({'id': case['id'], 'split': case['split'], 'condition': condition, 'reference': case['text'],
                         'hypothesis': hyp, 'latency_s': time.perf_counter() - t, 'audio_s': len(x) / 16000,
                         'cer_edits': Levenshtein.distance(ref, norm_hyp), 'reference_characters': len(ref),
                         'no_speech_probability': no_speech, 'accepted_asr': bool(hyp) and no_speech < .8,
                         'negative': case.get('negative', False), 'expected': {'items': case['items'], 'dine': case['dine']}})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            report = {'scope': manifest['scope'], 'model_sha256': hashlib.sha256(Path(args.model, 'model.bin').read_bytes()).hexdigest(),
                      'model_load_s': load, 'device': args.device, 'beam_size': args.beam, 'decoder': args.decoder, 'rows': rows}
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            if len(rows) % 10 == 0:
                print(f'{condition}: {len(rows)} hypotheses saved', flush=True)
        subset = [r for r in rows if r['condition'] == condition and not r['negative']]
        print(condition, 'CER', round(sum(r['cer_edits'] for r in subset) / sum(r['reference_characters'] for r in subset), 3), flush=True)


if __name__ == '__main__':
    main()
