import numpy as np

from sonkkeut_vision.plane import ScreenPlaneEstimator, homography_from_corners, is_valid_quad
from sonkkeut_vision.schema import PlaneState


def test_homography_roundtrip():
    c = np.float32([[100, 120], [500, 90], [540, 800], [80, 760]])
    H, aspect = homography_from_corners(c)
    st = PlaneState(c, H, 0.9, aspect, "detect")
    unit = st.to_screen(c)
    assert np.allclose(unit, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-6)
    p = np.array([[0.3, 0.7]])
    assert np.allclose(st.to_screen(st.to_image(p)), p, atol=1e-9)
    assert 0.5 < aspect < 0.7


def test_invalid_quad():
    assert not is_valid_quad(np.float32([[0, 0], [100, 100], [100, 0], [0, 100]]), (480, 640))  # 꼬인 사각형
    assert not is_valid_quad(np.float32([[0, 0], [10, 0], [10, 10], [0, 10]]), (480, 640))  # 너무 작음


def test_missing_corner_hint():
    shape = (1280, 960)
    kp = np.float32([[-30, 100], [800, 100], [800, 1000], [-20, 1000]])
    kc = np.ones(4)
    assert ScreenPlaneEstimator.missing_hint(kp, kc, shape) == "휴대폰을 조금 왼쪽으로"
    kp = np.float32([[100, -5], [800, -5], [800, 1000], [100, 1000]])
    assert ScreenPlaneEstimator.missing_hint(kp, kc, shape) == "휴대폰을 조금 위로"
    kp = np.float32([[100, 100], [800, 100], [800, 1000], [100, 1000]])
    assert ScreenPlaneEstimator.missing_hint(kp, np.array([1, 1, 0.1, 1.0]), shape) == "휴대폰을 조금 오른쪽 아래로"
    assert ScreenPlaneEstimator.missing_hint(kp, kc, shape) is None
