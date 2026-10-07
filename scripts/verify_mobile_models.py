"""Validate the Android model manifest and run every bundled ONNX model on CPU.

Optional verification dependencies: pip install onnx onnxruntime
Run from any directory: python scripts/verify_mobile_models.py
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def same_shape(actual, documented):
    """축 수가 같고, 양쪽 모두 숫자인 축은 값이 같은가"""
    return len(actual) == len(documented) and all(
        not (isinstance(a, int) and isinstance(d, int)) or a == d for a, d in zip(actual, documented))


def main():
    assets = Path(__file__).resolve().parents[1] / 'android/react-native-sonkkeut/android/src/main/assets/sonkkeut'
    manifest = json.loads((assets / 'model-manifest.json').read_text(encoding='utf-8'))
    expected = {'m1_screen_corners_int8.onnx', 'm2_screen_elements_int8.onnx', 'm1r_corner_refiner.onnx',
                'm3_kiosk_rec_v2.onnx', 'whisper-elder-v3-ct2.zip'}
    if {record['name'] for record in manifest['models']} != expected:
        raise ValueError('Unexpected model set')
    for record in manifest['models']:
        if record.get('bundled') is False:
            # 첫 사용 때 내려받는 모델(Whisper)은 APK에 없으므로, 앱 설치기가 검증에 쓰는 정보만 확인한다.
            if not (str(record.get('url', '')).startswith('https://') and len(record.get('sha256', '')) == 64
                    and record.get('size_bytes', 0) > 0):
                raise ValueError(f'Download record incomplete: {record["name"]}')
            print(f'PASS {record["name"]}: download on first use ({record["size_bytes"]} bytes)')
            continue
        path = assets / record['name']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256'] or len(data) != record['size_bytes']:
            raise ValueError(f'Model identity mismatch: {path.name}')
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        spec = session.get_inputs()[0]
        # 가변 축은 이름이 문서(dynamic_width)와 ONNX(DynamicDimension.1)에서 다르므로, 양쪽 모두 숫자인 축만 비교한다
        if spec.type != 'tensor(float)' or not same_shape(spec.shape, record['input']['shape']):
            raise ValueError(f'Input contract mismatch: {path.name}')
        # 가변 축 크기: 문서가 폭(width)이라고 적은 축은 실제 줄 이미지 비율에 가깝게 320, 나머지(배치 등)는 1
        shape = [want if isinstance(want, int) else (got if isinstance(got, int) else (320 if 'width' in str(want) else 1))
                 for got, want in zip(spec.shape, record['input']['shape'])]
        outputs = session.run(None, {spec.name: np.zeros(shape, dtype=np.float32)})
        if not all(same_shape(list(value.shape), output['shape']) for value, output in zip(outputs, record['outputs'])):
            raise ValueError(f'Output contract mismatch: {path.name}')
        if not all(np.isfinite(value).all() for value in outputs):
            raise ValueError(f'Nonfinite inference: {path.name}')
        if path.name.startswith('m2_'):
            classes = ast.literal_eval(session.get_modelmeta().custom_metadata_map['names'])
            if classes != {0: 'tab', 1: 'menu', 2: 'price', 3: 'button', 4: 'back'}:
                raise ValueError('Element decoder class order mismatch')
        print(f'PASS {path.name}: {spec.shape} -> {[list(value.shape) for value in outputs]}')


if __name__ == '__main__':
    main()
