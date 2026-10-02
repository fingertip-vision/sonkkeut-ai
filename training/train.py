"""
손끝길 모델 학습 스크립트

  M1 화면 꼭짓점(YOLOv8n-pose):  python train.py --task m1
  M2 화면 요소 탐지(YOLOv8n):     python train.py --task m2

GPU(RTX 등)가 있으면 자동으로 쓴다. 학습이 끝나면 best.pt와 함께
모바일 탑재용 TFLite(INT8) 변환도 시도한다(--export).
"""
import argparse

import torch
from ultralytics import YOLO

CFG = {
    "m1": dict(model="yolov8n-pose.pt", data="data/m1/data.yaml", imgsz=640,
               # 꼭짓점 순서가 의미를 가지므로 회전·좌우반전 증강은 flip_idx로 처리된다
               fliplr=0.5, degrees=5.0, mosaic=0.5),
    "m2": dict(model="yolov8n.pt", data="data/m2/data.yaml", imgsz=640,
               # 화면 글자가 뒤집히면 실제와 달라지므로 좌우반전은 끈다
               fliplr=0.0, degrees=2.0, mosaic=1.0),
}


def set_kpt_sigma(data_yaml, sigma):
    """꼭짓점 손실·평가의 OKS sigma를 data.yaml에 적는다.

    Ultralytics는 사람 관절(17점)이 아닌 키포인트에 sigma 1/키포인트 수(4점이면 0.25)를 쓴다.
    0.25는 매우 느슨해서, 꼭짓점이 화면 폭의 5% 어긋나도 pose mAP50-95가 0.99로 나온다.
    (그래서 mAP 대신 scripts/bench.py의 '꼭짓점 오차 %'로 판단해야 한다)
    sigma를 줄이면 정밀도 압력이 커지지만, 너무 줄이면(0.025) 손실이 포화되어 학습이 무너진다.
    """
    lines = [ln for ln in open(data_yaml, encoding="utf-8").read().splitlines() if not ln.startswith("kpt_oks_sigmas")]
    lines.append(f"kpt_oks_sigmas: [{sigma}, {sigma}, {sigma}, {sigma}]")
    open(data_yaml, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"kpt_oks_sigmas = {sigma}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["m1", "m2"], required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--weights", default=None, help="이어서 학습할 가중치(예: models/m1_screen_corners.pt)")
    ap.add_argument("--kpt-sigma", type=float, default=None, help="M1 꼭짓점 OKS sigma (실험용, set_kpt_sigma 참고)")
    ap.add_argument("--export", action="store_true", help="학습 후 TFLite INT8로 변환")
    a = ap.parse_args()

    c = CFG[a.task]
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if a.task == "m1" and a.kpt_sigma:
        set_kpt_sigma(a.data or c["data"], a.kpt_sigma)
    model = YOLO(a.weights or c["model"])
    model.train(
        data=a.data or c["data"], epochs=a.epochs, imgsz=a.imgsz or c["imgsz"], batch=a.batch,
        device=device, workers=a.workers, name=a.name or a.task, patience=20,
        fliplr=c["fliplr"], degrees=c["degrees"], mosaic=c["mosaic"],
        hsv_v=0.5, perspective=0.0005, plots=True,
    )
    if a.export:
        best = YOLO(model.trainer.best)
        best.export(format="tflite", int8=True, data=a.data or c["data"], imgsz=a.imgsz or c["imgsz"])


if __name__ == "__main__":
    main()
