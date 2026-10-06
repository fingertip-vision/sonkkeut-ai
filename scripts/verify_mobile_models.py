"""Validate the Android model manifest and run all three ONNX models on CPU.

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


def main():
    assets = Path(__file__).resolve().parents[1] / 'android/react-native-sonkkeut/android/src/main/assets/sonkkeut'
    manifest = json.loads((assets / 'model-manifest.json').read_text(encoding='utf-8'))
    expected = {'m1_screen_corners_int8.onnx', 'm2_screen_elements_int8.onnx', 'm1r_corner_refiner.onnx'}
    if {record['name'] for record in manifest['models']} != expected:
        raise ValueError('Unexpected model set')
    for record in manifest['models']:
        path = assets / record['name']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256'] or len(data) != record['size_bytes']:
            raise ValueError(f'Model identity mismatch: {path.name}')
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        spec = session.get_inputs()[0]
        if spec.type != 'tensor(float)' or spec.shape != record['input']['shape']:
            raise ValueError(f'Input contract mismatch: {path.name}')
        shape = [value if isinstance(value, int) else 1 for value in spec.shape]
        outputs = session.run(None, {spec.name: np.zeros(shape, dtype=np.float32)})
        if [list(value.shape) for value in outputs] != [output['shape'] for output in record['outputs']]:
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
