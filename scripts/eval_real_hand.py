"""
실제 손 사진 + 실제 MediaPipe 손 관절 모델로 F-08(손끝 추적)과 전체 유도를 확인한다.

카메라로 직접 찍기 전 단계의 검증이다. 실제 손 사진을 합성 키오스크 위에 붙이고(sim.RealHandCompositor),
손 위치를 정답으로 알고 있으므로 손끝 오차를 잴 수 있다.

  python scripts/eval_real_hand.py --hands D:/data/hand-keypoints [--hand-model models/hand_landmarker.task]

손 사진: Ultralytics hand-keypoints 데이터셋
  https://github.com/ultralytics/assets/releases/download/v0.0.0/hand-keypoints.zip (약 390MB, 레포에 포함하지 않음)

주의: 이 데이터셋의 관절 라벨도 MediaPipe로 만든 것이라, 손 사진 자체에 대한 MediaPipe 정확도 평가로는 쓸 수 없다.
여기서 보는 것은 '큰 프레임 안에서 손을 찾고 → 검지 끝을 고르고 → 화면 좌표로 옮기는' 우리 쪽 처리다.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO  # noqa: E402

from sonkkeut_vision import FingertipTracker, MediaPipeHandSource, ScreenPlaneEstimator, VisionPipeline  # noqa: E402
from sonkkeut_vision.sim import FakeKioskCamera, RealHandCompositor, render, run_scenario  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tip_accuracy(m1, hands, hand_model, n=150, seed=0):
    """정지 프레임에서 손끝 위치 오차 (화면 0~1 좌표 기준)"""
    rng = np.random.default_rng(seed)
    errs, lost, times, bw = [], 0, [], []
    for i in range(n):
        screen, labels = render(["menu", "option", "cart"][i % 3], seed=1000 + i)
        hands.pick()
        cam = FakeKioskCamera(screen, seed=1000 + i, hand_drawer=hands)
        plane_est = ScreenPlaneEstimator(m1)
        src = MediaPipeHandSource(hand_model)
        tr = FingertipTracker(src, smooth=1)
        gt = (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.15, 0.9)))
        tip = None
        for k in range(3):  # VIDEO 모드라 몇 프레임 연속으로 준다
            img, info = cam.frame(k / 15, gt)
            plane, _ = plane_est.update(img, k / 15)
            t0 = time.perf_counter()
            tip = tr.update(img, plane, k / 15)
            times.append((time.perf_counter() - t0) * 1000)
        src.close()
        bw += [b[3] - b[1] for b in labels if b[0] in ("button", "menu")]
        if tip is None or tip.pos is None:
            lost += 1
            continue
        # 오차는 '추정 화면 좌표'가 아니라 카메라 영상에서 비교한 뒤 실제 화면 좌표로 환산한다
        est_img = plane.to_image([tip.pos])[0]
        Hinv = np.linalg.inv(info["H_scr2img"])
        q = Hinv @ np.array([est_img[0], est_img[1], 1.0])
        errs.append(np.hypot(q[0] / q[2] - gt[0], q[1] / q[2] - gt[1]))
    return np.array(errs), lost, np.array(times), float(np.median(bw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", required=True)
    ap.add_argument("--hand-model", default=None)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--scenarios", type=int, default=20)
    ap.add_argument("--mode", default="mask", choices=["mask", "rect"], help="손 사진 붙이는 방식")
    ap.add_argument("--scale", type=float, default=0.25, help="손목~검지 끝 길이 / 화면 높이")
    a = ap.parse_args()
    m1 = YOLO(os.path.join(ROOT, "models", "m1_screen_corners.pt"))
    m2 = YOLO(os.path.join(ROOT, "models", "m2_screen_elements.pt"))
    hands = RealHandCompositor(a.hands, seed=0, mode=a.mode, scale=a.scale)
    print(f"검지를 편 손 사진 {len(hands.items)}장")

    errs, lost, times, bw = tip_accuracy(m1, hands, a.hand_model, n=a.n)
    print("\n[F-08 손끝 위치]")
    print(f"손 검출률            {(a.n - lost)}/{a.n}")
    print(f"손끝 오차            중앙값 {np.median(errs) * 100:.2f}%  90% {np.percentile(errs, 90) * 100:.2f}% (화면 크기 대비)")
    print(f"버튼 폭 대비          중앙값 {np.median(errs) / bw:.2f}  90% {np.percentile(errs, 90) / bw:.2f}"
          f"  (버튼 폭 중앙값 {bw:.3f}, 기준: 1/4 = 0.25 이내)")
    print(f"손끝 처리 시간(CPU)   중앙값 {np.median(times):.1f} ms  95% {np.percentile(times, 95):.1f} ms (기준 25 ms)")

    print("\n[실제 손 사진 + MediaPipe로 누르기 시나리오]")
    ok = wrong = 0
    reach = []
    for seed in range(a.scenarios):
        kind = ["menu", "menu", "option", "cart"][seed % 4]
        screen, labels = render(kind, seed=seed)
        nxt, _ = render("payment", seed=seed + 1000)
        hands.pick()
        cam = FakeKioskCamera(screen, seed=seed, hand_drawer=hands)
        src = MediaPipeHandSource(a.hand_model)
        pipe = VisionPipeline(m1, m2, hand_source=src)
        log = run_scenario(pipe, cam, None, nxt, labels, seed=seed)
        src.close()
        good = bool(log["pressed_at"] is not None and log["verdict"] and log["verdict"][1] == "success")
        ok += good
        wrong += log["press_inside_gt"] is False
        if log["pressed_at"] is not None:
            reach.append(log["pressed_at"])
        print(f"seed {seed:3d} {kind:7s} {'성공' if good else '실패'} 목표 {log['target']} 도달 {log['pressed_at']}"
              f" 결과 {log['verdict']}" + ("  ※버튼 밖에서 누르라고 함" if log["press_inside_gt"] is False else ""))
    print(f"\n누름 성공률 {ok}/{a.scenarios}, 잘못된 '누르세요' {wrong}회, 평균 도달 {np.mean(reach):.1f}s")


if __name__ == "__main__":
    main()
