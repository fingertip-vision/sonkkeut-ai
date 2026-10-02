"""
모바일(안드로이드) 탑재용 모델 내보내기: ONNX(FP32) + ONNX INT8(정적 양자화)

  python scripts/export_mobile.py --calib-m2 ../path/data/m2/images/train --calib-m1 ../path/data/m1/images/train

왜 ONNX인가: 안드로이드 앱(Kotlin)에서 ONNX Runtime Mobile로 바로 돌릴 수 있고,
Ultralytics 8.4의 TFLite(LiteRT) 내보내기는 의존성이 무거워 환경마다 실패가 잦다.
TFLite가 꼭 필요하면 PC에서 `yolo export model=... format=litert`를 따로 시도한다.

INT8 정적 양자화는 실제와 비슷한 입력(합성 화면) 200장으로 값의 범위를 재서 8비트로 바꾼다.
모델 크기가 1/4로 줄고 휴대폰 CPU에서 빨라지지만 정확도가 떨어질 수 있으므로, 같은 검증 세트로 꼭 다시 잰다.
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def letterbox(img, size=640):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
    out = np.full((size, size, 3), 114, np.uint8)
    y0, x0 = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


class Reader:
    def __init__(self, files, input_name, size=640):
        self.it = iter(files)
        self.name, self.size = input_name, size

    def get_next(self):
        f = next(self.it, None)
        if f is None:
            return None
        x = letterbox(cv2.imread(f), self.size)[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return {self.name: x}


def export_one(pt, calib_dir, n=200, size=640):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import onnxruntime as ort
    from ultralytics import YOLO

    onnx_path = YOLO(pt).export(format="onnx", imgsz=size, simplify=True, dynamic=False)
    pre = onnx_path.replace(".onnx", "_pre.onnx")
    quant_pre_process(onnx_path, pre)
    name = ort.InferenceSession(pre, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    files = sorted(glob.glob(os.path.join(calib_dir, "*.jpg")))
    random.Random(0).shuffle(files)
    out = onnx_path.replace(".onnx", "_int8.onnx")
    # Detect 머리(마지막 출력부)는 양자화하면 좌표 정밀도가 크게 떨어지므로 FP32로 남긴다
    import onnx

    m = onnx.load(pre)
    keep_fp32 = [nd.name for nd in m.graph.node if "/model.22/" in nd.name and nd.op_type in ("Conv", "Mul", "Add", "Sub", "Div", "Concat", "Sigmoid", "Softmax", "MatMul")]
    quantize_static(pre, out, Reader(files[:n], name, size), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=keep_fp32)
    os.remove(pre)
    return onnx_path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-m1", required=True)
    ap.add_argument("--calib-m2", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    for pt, calib in ((os.path.join(ROOT, "models", "m1_screen_corners.pt"), a.calib_m1),
                      (os.path.join(ROOT, "models", "m2_screen_elements.pt"), a.calib_m2)):
        fp32, int8 = export_one(pt, calib, a.n)
        print(f"{os.path.basename(fp32)} {os.path.getsize(fp32) / 1e6:.1f}MB -> "
              f"{os.path.basename(int8)} {os.path.getsize(int8) / 1e6:.1f}MB")


if __name__ == "__main__":
    sys.exit(main())
