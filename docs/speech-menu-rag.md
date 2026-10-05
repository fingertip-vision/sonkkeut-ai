# 음성 주문 성능 평가와 메뉴 검색 보정 · 2026-10-05

현재 모델만으로 시끄러운 매장 실사용을 보장하기 어렵다. 받아쓰기 외에 앱의 정확 문자열 매칭이 주문을 거부하는 문제를 확인해 **음성 → 원문 텍스트 → 매장 메뉴 DB 검색 → 이름/표현 보정 → 주문 파싱 → 사용자 확인**을 적용했다. ASR 가중치는 기존 Whisper elder v3 그대로이며 디코딩 후보 수는 1에서 5로 개선했다. GPU 재학습은 수행하지 않았다.

## 측정 결과

공개 시연 매장과 같은 3개 메뉴, 새로운 주문 24문장과 주문 아님/미등록 메뉴/무음/잡음 10사례를 3조건으로 추론했다. 총 102회다. Microsoft Heami 한 목소리와 3개 말하기 속도, 재현 가능한 유색 합성 잡음·기계음이다. **실제 사람, 실제 카페 소음, S26 Ultra의 측정값이 아니다.**

주문 정확도는 메뉴·수량·온도·매장/포장의 모든 항목이 정답과 같은 비율이다. 거부한 주문은 실패로 센다. 원문 CER은 공백·문장부호를 제거한 글자 편집 거리이며 보정 뒤 텍스트의 CER로 바꾸지 않는다.

| 조건 | 원문 글자 오류율 전→후 | v0.1.7 주문 정확도 | 디코딩·보정 후 정확도 | 재입력 요청 |
|---|---:|---:|---:|---:|
| 조용함 | 3.23%→2.42% | 15/24 · 62.5% | 23/24 · 95.8% | 1/24 |
| 합성 잡음 15dB | 5.65%→4.84% | 10/24 · 41.7% | 19/24 · 79.2% | 5/24 |
| 합성 잡음 5dB | 12.90%→11.56% | 5/24 · 20.8% | 8/24 · 33.3% | 15/24 |

각 조건의 음성 주문이 아닌 10사례를 주문으로 받아들인 경우는 0개였다. 강한 잡음의 주문 1개는 잘못된 주문 의도를 확인 단계에 제시했다. 이것도 정확도 실패로 센다. 앱은 항상 주문 확인을 요구하며 확인 전 자동 진행하지 않는다.

위 표는 **PC에서 Android와 같은 CT2 단일 디코딩 조건을 실행하고 실제 앱 파서를 적용한 결과**다. 물리적 휴대폰 102회 측정이 아니다. beam1+보정만 적용하면 21/24, 18/24, 7/24였다. 1차 Python transcribe()의 재시도 때문에 더 높았던 17→23/24, 14→22/24, 6→8/24는 최종 앱 지표로 사용하지 않았다. 같은 모델이라도 디코딩 조건을 구별해야 한다.

짝수 ID 17개는 개발, 홀수 ID 17개는 holdout이다. 개발 사례의 오류 문장으로 보정 규칙을 만들었고 holdout은 집계 점수를 확인했다. 독립적인 사람 음성 시험은 아니다. 원문·정답·조건·개별 결과·모델 SHA-256·분리 집계는 [`../asr/reports/mobile-speech-20261005.json`](../asr/reports/mobile-speech-20261005.json)에 있다. 조용함 95%/15dB 90%는 이번 개발에서 정한 임시 통과 기준이며 공인 기준이 아니다. 두 소음 조건은 기준 미달이다.

PC CPU int8 4스레드 디코딩 평균은 조건별 약 0.87초다. 녹음 종료 대기·모델 로딩·휴대폰 추론을 포함하지 않는다. Android API35 ARM64 변환 환경에서는 실제 Whisper JNI 합성 주문, 메뉴 DB, OCR·기존 영상 기능 등 13개 테스트가 통과했다. 물리적 휴대폰의 녹음 종료·음성 정확도·진동/음성 감각은 미검증이다. 초기 실행의 차가운 전체 화면 OCR은 한 번 unknown으로 반환했고, 모델 fixture를 설치한 후 전체 13개가 통과했다.

## 앱에 들어간 메뉴 DB

- `MenuRagDatabase.kt`: 앱 내부 `sonkkeut_menu_rag.db`, `catalog(scope PRIMARY KEY, payload)` SQLite 테이블. scope는 서버 주소/매장 코드/메뉴 버전이며 서로 다른 매장이나 버전을 섞지 않는다. 입력 메뉴가 바뀌면 해당 scope를 원자적으로 갱신한다.
- 데이터는 배포 백엔드의 메뉴명·관리자 등록 별칭·카테고리·품절 여부다. 오프라인일 때는 기존 매장 메뉴 캐시를 사용해 DB를 준비한다. 음성 원본과 발화 텍스트는 DB에 저장하지 않는다.
- `MenuRagIndex`: SQLite에서 읽은 문서로 한글 자모 bigram 검색 인덱스를 만들고 정확 이름/별칭/발음 유사도/긴 이름의 한 글자 누락을 비교한다. 가격·주문 가능 옵션은 기존 서버 메뉴와 주문 파서에서 처리한다.
- 이름뿐 아니라 ‘바닐라 커피’ 같은 의미 표현을 연결하려면 **해당 매장 메뉴의 aliases에 등록**한다. 임의의 의미 연관성을 근거로 상품을 생성하지 않는다. 짧은 별칭은 정확 일치만 허용한다.
- 유사도가 충분하고 2위 후보와 차이가 있을 때만 메뉴 이름을 바꾼다. 점수는 휴리스틱 유사도이며 확률이 아니다. 후보가 비슷하면 선택 화면으로 보낸다. 품절 후보는 선택할 수 없다.
- 수량은 새로 추측하지 않는다. 명시된 수량 다음의 제한된 단위 오인식(한 찬 → 한 잔)만 정규화한다. 인식한 메뉴명 내부는 바꾸지 않는다. 미등록 메뉴·디카페인 같은 미지원 옵션·이해하지 못한 수량은 재입력한다.
- 선택한 후보도 주문 확인 단계를 거친다. 확인 전에 키오스크 목표를 설정하거나 주문을 진행하지 않는다. 페이지를 벗어나면 늦은 음성 결과도 폐기한다.

이 방식은 **메뉴 검색에 근거한 보정(RAG 형태)**이며 벡터 DB나 생성 LLM을 추가한 구성은 아니다. 소수 상품의 외래어 오인식을 다루므로 메뉴 사전과 발음 검색을 사용했다. 음성 자체가 크게 훼손되면 RAG로 복구할 수 없어 다시 말하기·기기 음성 인식·직접 입력을 제공한다.

## 프론트 연결

SDK 0.1.4 / 앱 0.1.8:

```ts
const text = await Sonkkeut.listen('custom'); // 또는 명시적으로 'system'
const scope = JSON.stringify([server, storeCode, menuVersion]);
const result = await Sonkkeut.correctMenuSpeech(text, scope, storeMenus);
// result.original, result.text, result.corrections, result.ambiguities
// ambiguities가 있으면 사용자 선택; 그렇지 않으면 기존 parseOrder → 주문 확인.
```

`Sonkkeut.finishListening()`은 녹음을 끝내고 담긴 음성을 추론한다. 취소와 다르다. 앱의 ‘말하기 완료’로 소음 때문에 30초까지 기다리는 일을 줄인다. 자동 종료의 기존 에너지 감지는 훈련된 VAD가 아니며 실제 매장 녹음으로 검증해야 한다.

## 재현

AI와 프론트를 이 workspace처럼 `outputs/sonkkeut-ai`, `outputs/sonkkeut-frontend`에 두고 프론트 의존성을 설치한다. Python은 `requirements-unified.txt`, Windows 합성 음성은 Heami가 필요하다.

```powershell
& outputs/sonkkeut-ai/asr/synthesize_mobile_benchmark.ps1
python outputs/sonkkeut-ai/asr/bench_mobile_speech.py --model work/ai-model-downloads/whisper-elder-v3 --audio work/speech-benchmark/audio --out work/speech-benchmark/new.json --decoder android --beam 5
node outputs/sonkkeut-ai/asr/score_mobile_orders.cjs work/speech-benchmark/new.json
python outputs/sonkkeut-ai/asr/build_menu_rag_db.py --menu outputs/speech-rag-menu.json --out outputs/menu-rag-demo.sqlite --server https://amazing-manually-transcript-est.trycloudflare.com
```

기존 v0.1.7 파서는 `asr/fixtures/mobile-parser-v0.1.7.ts`에 고정했다. 모델 힌트는 메뉴 이름과 고정 옵션 어휘만 쓰고 평가 정답 문장을 넣지 않는다. CLI는 재현 가능한 합성 평가이며 현장 실사용 판정에는 여러 화자·실제 휴대폰·실제 소음의 녹음과 별도 test split이 필요하다. 현장 데이터가 확보되면 기존 `asr/train_whisper.py`로 GPU 재학습한 모델을 같은 기준으로 비교할 수 있다. 이번 합성 음성을 학습시키지 않았다.

참고: [faster-whisper 공식 추론 구현](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), [SQLite 공식 문서](https://www.sqlite.org/docs.html).
