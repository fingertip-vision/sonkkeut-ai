import math
import unittest

from sonkkeut_vision.contracts import Point
from sonkkeut_vision.geometry import Homography, PlaneTracker


class HomographyTests(unittest.TestCase):
    def assertPointClose(self, actual, expected, places=8):
        self.assertAlmostEqual(actual.x, expected.x, places=places)
        self.assertAlmostEqual(actual.y, expected.y, places=places)

    def test_pixel_rectangle_maps_to_unit_square_without_clipping(self):
        transform = Homography.from_corners((Point(100, 200), Point(900, 200),
                                             Point(900, 600), Point(100, 600)))
        self.assertPointClose(transform.camera_to_screen(Point(500, 400)), Point(0.5, 0.5))
        self.assertPointClose(transform.screen_to_camera(Point(0.25, 0.75)), Point(300, 500))
        self.assertPointClose(transform.project(Point(1300, 400)), Point(1.5, 0.5))

    def test_perspective_forward_inverse_matches_independent_transform(self):
        # Independent screen->camera fixture with a nonconstant denominator.
        def camera(point):
            denominator = 0.3 * point.x + 0.1 * point.y + 1
            return Point((900 * point.x + 100 * point.y + 100) / denominator,
                         (80 * point.x + 650 * point.y + 70) / denominator)

        unit_corners = (Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1))
        transform = Homography.from_corners(tuple(camera(point) for point in unit_corners))
        for screen in (*unit_corners, Point(0.31, 0.62), Point(0.8, 0.2), Point(-0.2, 1.3)):
            with self.subTest(screen=screen):
                self.assertPointClose(transform.camera_to_screen(camera(screen)), screen)
                self.assertPointClose(transform.screen_to_camera(screen), camera(screen))

    def test_malformed_geometry_rejected(self):
        bad_sets = (
            (Point(0, 0), Point(1, 1), Point(1, 0), Point(0, 1)),  # crossing
            (Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0)),  # mirrored
            (Point(0, 0), Point(2, 0), Point(0.5, 0.5), Point(0, 2)),  # concave
            (Point(0, 0), Point(1, 0), Point(1, 0), Point(0, 1)),  # repeated
            (Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)),  # collinear
            (Point(0, 0), Point(1, 0), Point(1, 1e-14), Point(0, 1e-14)),
            (Point(0, 0), Point(1, 0), Point(1, 1)),
        )
        for corners in bad_sets:
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                Homography.from_corners(corners)

    def test_finite_matrix_and_invertibility_validation(self):
        for matrix in ((1, 0, 0, 0, 1, 0, 0, 0, math.nan),
                       (1, 0, 0, 0, 1, 0, 0, 0, math.inf),
                       (1, 0, 0, 0, 0, 0, 0, 0, 1),
                       (1, 0, 0, 0, 1, 0, 0, 1)):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                Homography(matrix)
        for coordinate in (math.nan, math.inf, -math.inf):
            with self.subTest(coordinate=coordinate), self.assertRaises(ValueError):
                Point(coordinate, 0)

    def test_projective_horizon_rejected_in_forward_and_inverse_mapping(self):
        transform = Homography((1, 0, 0, 0, 1, 0, 1, 0, 1))
        with self.assertRaises(ValueError):
            transform.camera_to_screen(Point(-1, 0))
        with self.assertRaises(ValueError):
            transform.camera_to_screen(Point(-1 + 1e-14, 0))
        with self.assertRaises(ValueError):
            transform.screen_to_camera(Point(1, 0))
        with self.assertRaises(ValueError):
            Homography((2, 0, 0, 0, 1, 0, 0, 0, 1)).camera_to_screen(Point(1e308, 0))

    def test_large_offset_coordinates_preserve_pixel_accuracy(self):
        transform = Homography.from_corners((Point(1e6, 1e6), Point(1e6 + 800, 1e6),
                                             Point(1e6 + 800, 1e6 + 400), Point(1e6, 1e6 + 400)))
        self.assertPointClose(transform.camera_to_screen(Point(1e6 + 400, 1e6 + 200)), Point(0.5, 0.5))


class PlaneTrackerTests(unittest.TestCase):
    corners = (Point(0, 0), Point(800, 0), Point(800, 600), Point(0, 600))

    def test_invalid_corners_invalidate_old_transform_immediately(self):
        tracker = PlaneTracker()
        self.assertTrue(tracker.update(self.corners, 1).valid)
        result = tracker.update(tuple(reversed(self.corners)), 1.1)
        self.assertFalse(result.valid)
        self.assertIsNone(result.homography)
        self.assertTrue(result.requires_reacquisition)
        self.assertFalse(result.loss_timeout)
        self.assertIsNotNone(result.reason)

    def test_low_or_nonfinite_confidence_invalidates_plane(self):
        tracker = PlaneTracker(min_confidence=0.7)
        for value in (0.69, math.nan, math.inf, -1, True):
            tracker.update(self.corners, 1, 1)
            with self.subTest(value=value):
                self.assertFalse(tracker.update(self.corners, 1, value).valid)
        self.assertTrue(tracker.update(self.corners, 1, 0.7).valid)

    def test_three_second_loss_timeout_and_recovery_reset(self):
        tracker = PlaneTracker()
        self.assertFalse(tracker.update(None, 5).loss_timeout)
        self.assertFalse(tracker.update(None, 7.99).loss_timeout)
        result = tracker.update(None, 8)
        self.assertEqual(result.lost_for_s, 3)
        self.assertTrue(result.loss_timeout)
        recovered = tracker.update(self.corners, 9)
        self.assertTrue(recovered.valid)
        self.assertEqual(recovered.lost_for_s, 0)
        self.assertFalse(tracker.update(None, 10).loss_timeout)

    def test_fresh_corners_change_mapping_on_each_update(self):
        tracker = PlaneTracker()
        initial = tracker.update(self.corners, 0).homography
        shifted = tuple(Point(point.x + 100, point.y) for point in self.corners)
        updated = tracker.update(shifted, 0.1).homography
        self.assertAlmostEqual(initial.camera_to_screen(Point(400, 300)).x, 0.5)
        self.assertAlmostEqual(updated.camera_to_screen(Point(400, 300)).x, 0.375)

    def test_time_and_settings_validation(self):
        for settings in ({"min_confidence": -0.1}, {"loss_timeout_s": 0}, {"loss_timeout_s": math.inf}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PlaneTracker(**settings)
        tracker = PlaneTracker()
        tracker.update(None, 10)
        for timestamp in (9, -1, math.nan, math.inf):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                tracker.update(self.corners, timestamp)


if __name__ == "__main__":
    unittest.main()
