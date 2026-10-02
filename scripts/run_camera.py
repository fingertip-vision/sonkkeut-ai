"""
웹캠(또는 동영상)으로 영상 AI 전체 흐름을 실시간 확인한다.

  python scripts/run_camera.py                 # 기본 웹캠
  python scripts/run_camera.py --source 1      # 두 번째 카메라 (휴대폰을 USB 웹캠으로 쓸 때 등)
  python scripts/run_camera.py --source kiosk.mp4 --save out.mp4
  python scripts/run_camera.py --tts           # 안내 문장을 PC 스피커로 읽기 (pyttsx3 필요)

조작
  왼쪽 클릭  : 그 위치의 화면 요소를 목표 버튼으로 지정
  r         : 화면 다시 읽기(키프레임 강제)
  c         : 목표 해제
  q / ESC   : 종료
"""
import argparse
import json
import os
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sonkkeut_vision import VisionPipeline  # noqa: E402
from sonkkeut_vision.viz import draw_frame  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Speaker:
    """안내 문장을 별도 스레드에서 읽는다 (앱에서는 임현승의 TTS가 이 역할)"""

    def __init__(self, enabled):
        self.engine = None
        if enabled:
            try:
                import pyttsx3
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 190)
            except Exception as e:  # noqa: BLE001
                print("TTS를 쓸 수 없습니다:", e)
        self.busy = False

    def say(self, text):
        if self.engine is None or self.busy:
            return
        self.busy = True

        def run():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            finally:
                self.busy = False

        threading.Thread(target=run, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--m1", default=os.path.join(ROOT, "models", "m1_screen_corners.pt"))
    ap.add_argument("--m2", default=os.path.join(ROOT, "models", "m2_screen_elements.pt"))
    ap.add_argument("--device", default=None, help="0 = GPU, cpu = CPU")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--save", default=None, help="결과 영상 저장 경로(.mp4)")
    ap.add_argument("--log", default=None, help="안내 이벤트를 JSON Lines로 저장")
    ap.add_argument("--tts", action="store_true")
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"카메라/영상을 열 수 없습니다: {a.source}")

    pipe = VisionPipeline(a.m1, a.m2, device=a.device)
    speaker = Speaker(a.tts)
    state = {"click": None, "last_speak": None}
    writer, logf = None, open(a.log, "w", encoding="utf-8") if a.log else None

    def on_mouse(evt, x, y, *_):
        if evt == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    win = "sonkkeut-vision"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    fps_t, n = time.monotonic(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.monotonic()
        res = pipe.process(frame, t)

        # 클릭한 위치의 요소를 목표로 (앱에서는 F-07 버튼 순서 계획이 정한다)
        if state["click"] and res.plane is not None:
            sx, sy = res.plane.to_screen([state["click"]])[0]
            hit = [e for e in pipe.elements if e.contains(sx, sy)]
            if hit:
                tgt = min(hit, key=lambda e: (e.box[2] - e.box[0]) * (e.box[3] - e.box[1]))
                pipe.set_target(tgt.id)
                print(f"목표: {tgt.id} ({tgt.kind})")
            state["click"] = None

        for msg in [res.hint, res.event.speak if res.event else None, res.verdict.speak if res.verdict else None]:
            if msg:
                state["last_speak"] = msg
                speaker.say(msg)
                print(f"[{t:8.2f}] {msg}")
        if logf and (res.event or res.verdict):
            logf.write(json.dumps({"t": round(t, 3), "event": res.event.to_dict() if res.event else None,
                                   "verdict": res.verdict.to_dict() if res.verdict else None},
                                  ensure_ascii=False) + "\n")

        tid = pipe.guide.target.id if pipe.guide.target else None
        vis = draw_frame(frame, res, pipe.elements, tid, state["last_speak"])
        n += 1
        if time.monotonic() - fps_t > 1:
            cv2.setWindowTitle(win, f"sonkkeut-vision  {n / (time.monotonic() - fps_t):.1f} FPS")
            fps_t, n = time.monotonic(), 0
        cv2.imshow(win, vis)
        if a.save:
            if writer is None:
                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), 15, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            pipe.request_keyframe()
        if k == ord("c"):
            pipe.clear_target()
    cap.release()
    if writer:
        writer.release()
    if logf:
        logf.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
