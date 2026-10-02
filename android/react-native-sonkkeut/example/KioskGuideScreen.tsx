/**
 * 손끝길 카메라 화면 예시 (프론트 팀이 그대로 가져다 시작할 수 있게 만든 최소 화면)
 *
 * - 카메라를 켜고 영상 AI를 돌린다 (음성·진동은 기본으로 네이티브가 낸다)
 * - 화면을 탭하면 그 위치의 키오스크 버튼을 목표로 지정한다 (언어 쪽 F-07이 붙기 전 시연용)
 * - 화면 위에 인식 상태와 마지막 안내 문장을 크게 보여 준다 (저시력 사용자·시연용)
 *
 * 필요한 패키지: react-native-vision-camera, react-native-worklets-core, react-native-sonkkeut
 */
import React, { useEffect, useState } from 'react'
import { Pressable, StyleSheet, Text, View } from 'react-native'
import { Camera, useCameraDevice, useCameraFormat, useCameraPermission } from 'react-native-vision-camera'
import { useSonkkeut } from 'react-native-sonkkeut'

export default function KioskGuideScreen() {
  const { hasPermission, requestPermission } = useCameraPermission()
  const device = useCameraDevice('back')
  // 학습 영상과 비슷한 4:3, 1280×960 근처 해상도. 너무 크면 느려진다
  const format = useCameraFormat(device, [{ videoAspectRatio: 4 / 3 }, { videoResolution: { width: 1280, height: 960 } }, { fps: 30 }])
  const [lastSpeak, setLastSpeak] = useState('카메라를 키오스크 화면 쪽으로 들어 주세요')
  const [size, setSize] = useState({ w: 1, h: 1 })

  const { ready, error, result, screen, frameProcessor, setTargetAtPreview } = useSonkkeut({
    onEvent: (e) => e.speak && setLastSpeak(e.speak),
    onVerdict: (v) => setLastSpeak(v.speak),
  })

  useEffect(() => {
    if (!hasPermission) requestPermission()
  }, [hasPermission, requestPermission])

  useEffect(() => {
    if (result?.hint) setLastSpeak(result.hint)
  }, [result?.hint])

  if (!hasPermission || device == null) return <Text style={styles.big}>카메라 권한이 필요합니다</Text>
  if (error) return <Text style={styles.big}>AI를 불러오지 못했습니다: {error}</Text>

  // 미리보기 탭 → 영상 픽셀 → (네이티브에서) 키오스크 화면 좌표 → 그 위치의 버튼을 목표로
  const onTap = async (x: number, y: number) => {
    const id = await setTargetAtPreview(x, y, size.w, size.h, result?.frame_size)
    setLastSpeak(id ? `목표를 정했습니다 (${id})` : '그 위치에서 버튼을 찾지 못했습니다')
  }

  return (
    <View style={styles.root} onLayout={(e) => setSize({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}>
      <Camera style={StyleSheet.absoluteFill} device={device} format={format} isActive={ready} pixelFormat="yuv"
              frameProcessor={frameProcessor} />
      <Pressable style={StyleSheet.absoluteFill} onPress={(e) => onTap(e.nativeEvent.locationX, e.nativeEvent.locationY)}
                 accessibilityLabel="화면을 눌러 목표 버튼 지정" />
      <View style={styles.panel} accessibilityLiveRegion="polite">
        <Text style={styles.big}>{lastSpeak}</Text>
        <Text style={styles.small}>
          {ready ? (result?.found ? `화면 인식됨 · 버튼 ${screen?.elements.length ?? 0}개` : '화면을 찾는 중') : 'AI 준비 중'}
          {result?.timings?.total_ms != null ? ` · ${result.timings.total_ms} ms` : ''}
        </Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: 'black' },
  panel: { position: 'absolute', left: 16, right: 16, bottom: 32, padding: 16, borderRadius: 16, backgroundColor: 'rgba(0,0,0,0.75)' },
  big: { color: 'white', fontSize: 28, fontWeight: '700' },
  small: { color: '#ddd', fontSize: 16, marginTop: 6 },
})
