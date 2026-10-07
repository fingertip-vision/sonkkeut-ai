/**
 * react-native-sonkkeut — 손끝길 온디바이스 영상 AI
 *
 * 사용 예 (자세한 예시는 README.md)
 *
 *   const { frameProcessor, result, ready, setTargetAt } = useSonkkeut()
 *   <Camera device={device} isActive format={format} frameProcessor={frameProcessor} pixelFormat="yuv" />
 *
 * AI 계산은 전부 휴대폰 안(Kotlin + ONNX Runtime + MediaPipe)에서 하고,
 * JS는 결과(안내 문장·진동 세기·화면 구조)만 받는다.
 */
import { useEffect, useRef, useState } from 'react'
import { NativeEventEmitter, NativeModules, Platform } from 'react-native'
import { type Frame, useFrameProcessor, VisionCameraProxy } from 'react-native-vision-camera'
import { MenuRagIndex, type MenuDocument } from './menuRag'
export { MenuRagIndex } from './menuRag'
export type { MenuDocument, MenuRagResult, MenuAmbiguity, MenuCorrection } from './menuRag'

// ---------- 결과 형식 (sonkkeut-ai/docs/interface.md와 같음) ----------
export type Kind = 'tab' | 'menu' | 'price' | 'button' | 'back' | 'cart_item' | 'text' | 'title'

export interface ScreenElement {
  id: string
  kind: Kind
  /** 화면 기준 0~1 좌표 [x1, y1, x2, y2] */
  box: [number, number, number, number]
  conf: number
  parent?: string
  text?: string
  price?: number
  conf_ocr?: number
  uncertain?: boolean
  ocr_source?: 'm3_kiosk_rec_v2' | 'mlkit_fallback' | string
  qty?: number
}

export interface ScreenStructure {
  screen_type: 'menu' | 'option' | 'cart' | 'payment' | 'start' | 'unknown' | string
  keyframe_id: number
  elements: ScreenElement[]
  cart_count?: number
  total_price?: number
  selected?: string[]
}

export interface GuidanceEvent {
  /** direction 방향 안내 · press 지금 누르세요 · hold 잠시 멈춤 · reset 다시 시작 · no_hand 손 없음 · point 검지만 펴기 */
  type: 'direction' | 'press' | 'hold' | 'reset' | 'no_hand' | 'point'
  target_id?: string
  dir?: 'right' | 'up_right' | 'up' | 'up_left' | 'left' | 'down_left' | 'down' | 'down_right'
  distance?: 'far' | 'near' | 'reach'
  /** 읽어 줄 문장. 0.8초에 한 번 이하로 이미 걸러져 있다. 없으면 이번에는 말하지 않는다 */
  speak?: string
  /** 진동 주기(Hz): 멀리 2 → 가까이 4~8 → 버튼 위 10, 0이면 진동 없음 */
  vibe_hz: number
}

export interface Verdict {
  result: 'success' | 'fail' | 'uncertain' | 'restarted'
  reason: string
  speak: string
}

export interface SonkkeutResult {
  /** 화면(키오스크)을 찾았는가 */
  found: boolean
  /** 화면을 못 찾았을 때 휴대폰 위치 안내 ("휴대폰을 조금 왼쪽으로") */
  hint?: string
  keyframe: boolean
  keyframe_id: number
  /** 화면이 바뀌어 새로 읽었을 때만 */
  structure?: ScreenStructure
  event?: GuidanceEvent
  /** 누른 뒤 판정 */
  verdict?: Verdict
  /** 새 화면에 목표 버튼이 없음 → 다시 계획 */
  target_missing?: boolean
  /** 손끝 위치 (화면 0~1) */
  tip?: [number, number]
  /** 카메라 영상 속 화면 네 꼭짓점 [x0,y0,…,x3,y3] (회전 보정된 영상 픽셀) */
  corners?: number[]
  /** 처리한 영상 크기 [폭, 높이] (회전 보정 후) */
  frame_size?: [number, number]
  target_image_box?: [number, number, number, number] | null
  timings: Record<string, number>
}

/** 누른 뒤 기대 결과 (sonkkeut-ai/sonkkeut_vision/verify.py 참고) */
export interface Expect {
  screen_type?: string
  screen_type_not?: string
  cart_delta?: number
  selected?: string
  changed?: boolean
  success_speak?: string
}

export interface SpeechModelStatus {
  provider: 'ct2-whisper-v3' | string
  model_version: string
  installed: boolean
  ready: boolean
  downloading: boolean
  busy: boolean
  requires_download: boolean
  download_size_bytes?: number
}

export interface ModelDownloadProgress { bytes: number; total_bytes: number; stage: string }
export interface SpeechInferenceInfo {
  provider: string; model_version: string; sequence_score?: number
  no_speech_probability: number; inference_ms: number; audio_seconds: number
}

// ---------- 네이티브 연결 ----------
const Native = NativeModules.Sonkkeut as
  | {
      init(options: { nativeFeedback?: boolean }): Promise<boolean>
      start(): void
      stop(): void
      setTarget(id: string, expect: Expect | null): Promise<boolean>
      setTargetOnKeyframe(id: string, keyframeId: number, expect: Expect | null): Promise<boolean>
      setTargetAt(x: number, y: number, expect: Expect | null): Promise<string | null>
      setTargetAtImage(px: number, py: number, expect: Expect | null): Promise<string | null>
      clearTarget(): void
      requestKeyframe(): void
      say(text: string): void
      announce(text: string): void
      configureFeedback(voice: boolean, vibration: boolean, rate: number): void
      silence(): void
      listen(): Promise<string>
      listenModel(): Promise<string>
      finishListening(): void
      menuRagCatalog(scope: string, payload: string): Promise<string>
      cancelListening(): void
      getSpeechModelStatus(): Promise<SpeechModelStatus>
      prepareSpeechModel(): Promise<SpeechModelStatus>
      downloadSpeechModel(): Promise<SpeechModelStatus>
      cancelSpeechModelDownload(): void
      setMenuAliases(aliases: Record<string, string>): void
    }
  | undefined

if (Native == null && Platform.OS === 'android') {
  console.warn('[sonkkeut] 네이티브 모듈을 찾지 못했습니다. 앱을 다시 빌드하세요 (npx react-native run-android).')
}

const plugin = VisionCameraProxy.initFrameProcessorPlugin('sonkkeut', {})

/** 프레임 처리 (워크릿 안에서 호출). rotation을 주면 자동 회전 대신 그 각도를 쓴다 */
export function sonkkeutProcess(frame: Frame, rotation?: number) {
  'worklet'
  if (plugin == null) return null
  // VisionCamera 4.6.4 converts a supplied second argument to a native Map.
  // Passing explicit undefined crashes the frame processor; omit the argument instead.
  return (rotation == null ? plugin.call(frame) : plugin.call(frame, { rotation })) as {
    found?: boolean
    type?: string
    speak?: string
    vibe_hz?: number
    total_ms?: number
  } | null
}

export const Sonkkeut = {
  /** Explicit speech, including repeat and screen reading, bypasses the automatic voice preference. */
  say: (text: string) => Native?.say(text),
  announce: (text: string) => Native?.announce(text),
  configureFeedback: (voice: boolean, vibration: boolean, rate: number) => Native?.configureFeedback(voice, vibration, rate),
  silence: () => Native?.silence(),
  listen: (provider: 'custom' | 'system' = 'custom') =>
    (provider === 'custom' ? Native?.listenModel() : Native?.listen()) ?? Promise.reject(new Error('네이티브 모듈이 없습니다')),
  cancelListening: () => Native?.cancelListening(),
  finishListening: () => Native?.finishListening(),
  async correctMenuSpeech(text: string, scope: string, menus: MenuDocument[]) {
    if (!Native) throw new Error('네이티브 모듈이 없습니다')
    const snapshot = await Native.menuRagCatalog(scope, JSON.stringify(menus))
    return new MenuRagIndex(JSON.parse(snapshot)).correct(text)
  },
  getSpeechModelStatus: () => Native?.getSpeechModelStatus() ?? Promise.reject(new Error('네이티브 모듈이 없습니다')),
  prepareSpeechModel: () => Native?.prepareSpeechModel() ?? Promise.reject(new Error('네이티브 모듈이 없습니다')),
  downloadSpeechModel: () => Native?.downloadSpeechModel() ?? Promise.reject(new Error('네이티브 모듈이 없습니다')),
  cancelSpeechModelDownload: () => Native?.cancelSpeechModelDownload(),
  addModelDownloadListener(cb: (progress: ModelDownloadProgress) => void) {
    const sub = new NativeEventEmitter(NativeModules.Sonkkeut).addListener('SonkkeutModelDownload', cb)
    return () => sub.remove()
  },
  addSpeechListener(cb: (info: SpeechInferenceInfo) => void) {
    const sub = new NativeEventEmitter(NativeModules.Sonkkeut).addListener('SonkkeutSpeechResult', cb)
    return () => sub.remove()
  },
  setMenuAliases: (aliases: Record<string, string>) => Native?.setMenuAliases(aliases),
  init: (options: { nativeFeedback?: boolean } = {}) => Native?.init(options) ?? Promise.resolve(false),
  start: () => Native?.start(),
  stop: () => Native?.stop(),
  /**
   * 목표 버튼 지정 (언어 쪽 F-07이 정한 요소 id). 지금 화면에 없으면 false.
   * keyframeId: id를 고른 screen의 keyframe_id. 요소 id는 키프레임마다 새로 매겨지므로, 넘기면 그 사이 화면을
   * 다시 읽었을 때 다른 버튼을 목표로 잡지 않고 false를 돌려준다. 음성 확인처럼 고른 뒤 시간이 걸리면 꼭 넘긴다.
   */
  setTarget: (id: string, expect?: Expect, keyframeId?: number) =>
    (keyframeId == null
      ? Native?.setTarget(id, expect ?? null)
      : Native?.setTargetOnKeyframe(id, keyframeId, expect ?? null)) ?? Promise.resolve(false),
  /** 화면 좌표(0~1)를 눌러 목표 지정 (시연·저시력 모드). 지정된 요소 id */
  setTargetAt: (x: number, y: number, expect?: Expect) => Native?.setTargetAt(x, y, expect ?? null) ?? Promise.resolve(null),
  /** 카메라 영상 픽셀(result.frame_size 기준)로 목표 지정 */
  setTargetAtImage: (px: number, py: number, expect?: Expect) =>
    Native?.setTargetAtImage(px, py, expect ?? null) ?? Promise.resolve(null),
  /**
   * 미리보기를 탭한 위치로 목표 지정. <Camera resizeMode="cover">(기본)에서 미리보기 크기와 탭 좌표를 주면
   * 영상 픽셀로 바꿔 넘긴다. 미리보기와 영상이 같은 방향(세로)이라고 가정한다.
   */
  setTargetAtPreview(tapX: number, tapY: number, viewW: number, viewH: number, frameSize?: [number, number], expect?: Expect) {
    if (!frameSize) return Promise.resolve(null)
    const [fw, fh] = frameSize
    const s = Math.max(viewW / fw, viewH / fh)
    const px = (tapX - (viewW - fw * s) / 2) / s
    const py = (tapY - (viewH - fh * s) / 2) / s
    return Sonkkeut.setTargetAtImage(px, py, expect)
  },
  clearTarget: () => Native?.clearTarget(),
  requestKeyframe: () => Native?.requestKeyframe(),
  addListener(cb: (r: SonkkeutResult) => void) {
    const em = new NativeEventEmitter(NativeModules.Sonkkeut)
    const sub = em.addListener('SonkkeutResult', (r) => cb(r as SonkkeutResult))
    return () => sub.remove()
  },
}

export interface UseSonkkeutOptions {
  /** 기본 음성·진동을 네이티브에서 낼지 (기본 true). 앱에서 직접 TTS·진동을 하려면 false */
  nativeFeedback?: boolean
  /** false면 처리를 멈춘다 (화면을 떠날 때 등) */
  active?: boolean
  onEvent?: (e: GuidanceEvent) => void
  onVerdict?: (v: Verdict) => void
  onScreen?: (s: ScreenStructure) => void
  onResult?: (r: SonkkeutResult) => void
}

/**
 * 카메라 화면 컴포넌트에서 쓰는 훅.
 * frameProcessor를 <Camera frameProcessor={...}>에 넘기면 된다.
 */
export function useSonkkeut(options: UseSonkkeutOptions = {}) {
  const { nativeFeedback = true, active = true } = options
  const [ready, setReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SonkkeutResult | null>(null)
  const [screen, setScreen] = useState<ScreenStructure | null>(null)
  const cbs = useRef(options)
  cbs.current = options

  useEffect(() => {
    let alive = true
    Sonkkeut.init({ nativeFeedback })
      .then((ok) => alive && setReady(!!ok))
      .catch((e) => alive && setError(String(e?.message ?? e)))
    const off = Sonkkeut.addListener((r) => {
      setResult(r)
      cbs.current.onResult?.(r)
      if (r.structure) {
        setScreen(r.structure)
        cbs.current.onScreen?.(r.structure)
      }
      if (r.event) cbs.current.onEvent?.(r.event)
      if (r.verdict) cbs.current.onVerdict?.(r.verdict)
    })
    return () => {
      alive = false
      off()
      Sonkkeut.stop()
    }
  }, [nativeFeedback])

  useEffect(() => {
    if (ready && active) Sonkkeut.start()
    else Sonkkeut.stop()
  }, [ready, active])

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'
    sonkkeutProcess(frame)
  }, [])

  return {
    ready,
    error,
    /** 가장 최근 결과 */
    result,
    /** 가장 최근에 읽은 화면 구조 (버튼 목록) */
    screen,
    frameProcessor,
    setTarget: Sonkkeut.setTarget,
    setTargetAt: Sonkkeut.setTargetAt,
    setTargetAtPreview: Sonkkeut.setTargetAtPreview,
    clearTarget: Sonkkeut.clearTarget,
    requestKeyframe: Sonkkeut.requestKeyframe,
  }
}
