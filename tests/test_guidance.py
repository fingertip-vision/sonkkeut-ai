import unittest

from sonkkeut_vision.contracts import Box, Element, Point
from sonkkeut_vision.guidance import GuidanceEngine, direction_for_delta


def target(*, element_id="button-1", box=Box(0.45, 0.45, 0.55, 0.55), conf=0.95):
    return Element(element_id, "button", box, conf)


class GuidanceTests(unittest.TestCase):
    def setUp(self):
        self.engine = GuidanceEngine()
        self.target = target()

    def update(self, point, timestamp, *, raw_point=None, **kwargs):
        return self.engine.update(point, self.target, timestamp,
                                  raw_point=point if raw_point is None else raw_point, **kwargs)

    def test_eight_directions_use_screen_positive_y_down(self):
        cases = ((Point(0.2, 0.5), "right"), (Point(0.2, 0.2), "down_right"),
                 (Point(0.5, 0.2), "down"), (Point(0.8, 0.2), "down_left"),
                 (Point(0.8, 0.5), "left"), (Point(0.8, 0.8), "up_left"),
                 (Point(0.5, 0.8), "up"), (Point(0.2, 0.8), "up_right"))
        for i, (point, expected) in enumerate(cases):
            with self.subTest(expected=expected):
                result = self.update(point, i * 0.05)
                self.assertEqual(result.action, "move")
                self.assertEqual(result.direction, expected)
                self.assertAlmostEqual(result.dx, 0.5 - point.x)
                self.assertAlmostEqual(result.dy, 0.5 - point.y)

    def test_center_has_no_direction(self):
        self.assertIsNone(direction_for_delta(0, 0))

    def test_sector_boundary_and_adjacent_values(self):
        import math
        angle = math.pi / 8
        self.assertEqual(direction_for_delta(1, math.tan(angle - 1e-6)), "right")
        self.assertEqual(direction_for_delta(1, math.tan(angle + 1e-6)), "down_right")
        self.assertEqual(direction_for_delta(-1, 0), "left")

    def test_near_distance_is_configurable_and_inclusive(self):
        narrow = target(box=Box(0.499, 0.499, 0.501, 0.501))
        self.engine = GuidanceEngine(near_distance=0.06)
        self.target = narrow
        self.assertEqual(self.update(Point(0.44, 0.5), 0).action, "near")
        self.assertEqual(self.update(Point(0.439, 0.5), 0.05).action, "move")

    def test_press_requires_three_tenths_second_and_emits_once(self):
        point = Point(0.5, 0.5)
        for t in (0, 0.1, 0.2, 0.299):
            self.assertEqual(self.update(point, t).action, "hold")
        self.assertEqual(self.update(point, 0.3).action, "press")
        for t in (0.4, 0.5, 0.6):
            result = self.update(point, t)
            self.assertEqual(result.action, "hold")
            self.assertEqual(result.reason, "awaiting_result")

    def test_button_edges_count_as_inside(self):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.45, 0.55), t)
        self.assertEqual(self.update(Point(0.45, 0.55), 0.3).action, "press")

    def test_raw_outside_jitter_resets_dwell_even_when_smooth_inside(self):
        center = Point(0.5, 0.5)
        for t in (0, 0.1, 0.2):
            self.update(center, t)
        result = self.update(center, 0.3, raw_point=Point(0.550001, 0.5))
        self.assertNotEqual(result.action, "press")
        self.assertEqual(result.reason, "raw_outside_target")
        for t in (0.4, 0.5, 0.6):
            self.assertEqual(self.update(center, t).action, "hold")
        self.assertEqual(self.update(center, 0.7).action, "press")

    def test_smooth_outside_also_resets_dwell(self):
        center = Point(0.5, 0.5)
        for t in (0, 0.1, 0.2):
            self.update(center, t)
        result = self.update(Point(0.56, 0.5), 0.3, raw_point=center)
        self.assertEqual(result.reason, "smoothed_outside_target")
        self.assertNotEqual(self.update(center, 0.4).action, "press")

    def test_missing_raw_point_never_allows_press(self):
        for t in (0, 0.1, 0.2, 0.3, 0.4):
            result = self.engine.update(Point(0.5, 0.5), self.target, t)
            self.assertEqual(result.action, "stop")
            self.assertEqual(result.reason, "missing_finger")

    def test_missing_finger_resets_dwell(self):
        self.update(Point(0.5, 0.5), 0)
        self.update(Point(0.5, 0.5), 0.1)
        self.assertEqual(self.update(None, 0.2).action, "stop")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3).dwell_seconds, 0)

    def test_unobserved_gap_does_not_satisfy_dwell(self):
        self.update(Point(0.5, 0.5), 0)
        result = self.update(Point(0.5, 0.5), 0.3)
        self.assertEqual(result.action, "hold")
        self.assertEqual(result.dwell_seconds, 0)

    def test_frame_gap_limit_is_configurable(self):
        self.engine = GuidanceEngine(max_frame_gap=0.05)
        self.update(Point(0.5, 0.5), 0)
        self.assertEqual(self.update(Point(0.5, 0.5), 0.06).dwell_seconds, 0)

    def test_target_identity_change_resets_dwell(self):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.5, 0.5), t)
        self.target = target(element_id="button-2")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3).dwell_seconds, 0)

    def test_target_box_change_with_same_id_resets_dwell(self):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.5, 0.5), t)
        self.target = target(box=Box(0.4, 0.4, 0.6, 0.6))
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3).dwell_seconds, 0)

    def test_keyframe_change_resets_dwell(self):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.5, 0.5), t, keyframe_id=1)
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3, keyframe_id=2).dwell_seconds, 0)

    def test_tracked_hand_identity_change_resets_dwell(self):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.5, 0.5), t, hand_id="a")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3, hand_id="b").dwell_seconds, 0)

    def test_low_target_confidence_stops_and_requests_reacquisition(self):
        self.update(Point(0.5, 0.5), 0)
        self.target = target(conf=0.699)
        result = self.update(Point(0.5, 0.5), 0.1)
        self.assertEqual(result.action, "stop")
        self.assertTrue(result.reacquire)
        self.assertEqual(result.reason, "low_target_confidence")
        self.target = target()
        self.assertEqual(self.update(Point(0.5, 0.5), 0.2).dwell_seconds, 0)

    def test_low_finger_confidence_resets_dwell(self):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.5, 0.5), t)
        result = self.update(Point(0.5, 0.5), 0.3, finger_confidence=0.69)
        self.assertEqual(result.reason, "low_finger_confidence")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.4).dwell_seconds, 0)

    def test_invalid_confidence_fails_closed(self):
        for i, value in enumerate((-1, 1.1, float("nan"), True)):
            with self.subTest(value=value):
                result = self.update(Point(0.5, 0.5), i * 0.05, finger_confidence=value)
                self.assertEqual(result.action, "stop")
                self.assertEqual(result.reason, "invalid_finger_confidence")

    def test_missing_or_invalid_target_requests_reacquisition(self):
        for i, bad in enumerate((None, "target")):
            result = self.engine.update(Point(0.5, 0.5), bad, i * 0.05, raw_point=Point(0.5, 0.5))
            self.assertEqual(result.action, "stop")
            self.assertTrue(result.reacquire)

    def test_offscreen_raw_is_stopped_without_clamping(self):
        for i, point in enumerate((Point(-0.01, 0.5), Point(1.01, 0.5), Point(0.5, -0.01), Point(0.5, 1.01))):
            result = self.update(Point(0.5, 0.5), i * 0.05, raw_point=point)
            self.assertEqual(result.reason, "outside_plane")
            self.assertEqual(result.action, "stop")

    def test_offscreen_smooth_point_and_invalid_point_types_stop(self):
        self.assertEqual(self.update(Point(1.1, 0.5), 0).reason, "outside_plane")
        result = self.engine.update((0.5, 0.5), self.target, 0.05, raw_point=Point(0.5, 0.5))
        self.assertEqual(result.reason, "invalid_finger")

    def test_replayed_or_nonfinite_timestamps_reset_progress_and_raise(self):
        self.update(Point(0.5, 0.5), 1)
        self.update(Point(0.5, 0.5), 1.1)
        for timestamp in (1.1, 1, float("nan"), float("inf"), True):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.update(Point(0.5, 0.5), timestamp)
        self.assertEqual(self.update(Point(0.5, 0.5), 1.2).dwell_seconds, 0)

    def test_leaving_and_reentering_does_not_emit_second_press(self):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.5, 0.5), t)
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3).action, "press")
        outside = self.update(Point(0.6, 0.5), 0.4)
        self.assertEqual(outside.reason, "awaiting_result")
        self.assertIsNone(outside.distance_band)
        for t in (0.5, 0.6, 0.7):
            self.assertEqual(self.update(Point(0.5, 0.5), t).action, "hold")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.8).reason, "awaiting_result")

    def press_once(self, **kwargs):
        for t in (0, 0.1, 0.2):
            self.update(Point(0.5, 0.5), t, **kwargs)
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3, **kwargs).action, "press")

    def test_missing_finger_after_press_does_not_unlock_another_cue(self):
        self.press_once()
        self.assertEqual(self.update(None, 0.4).action, "stop")
        for t in (0.5, 0.6, 0.7, 0.8):
            self.assertEqual(self.update(Point(0.5, 0.5), t).reason, "awaiting_result")

    def test_missing_target_after_press_does_not_unlock_same_target(self):
        self.press_once()
        self.assertEqual(self.engine.update(Point(0.5, 0.5), None, 0.4,
                                            raw_point=Point(0.5, 0.5)).action, "stop")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.5).reason, "awaiting_result")

    def test_box_and_hand_changes_do_not_unlock_same_target(self):
        self.press_once(hand_id="a")
        self.target = target(box=Box(0.4, 0.4, 0.6, 0.6))
        for t in (0.4, 0.5, 0.6, 0.7):
            self.assertEqual(self.update(Point(0.5, 0.5), t, hand_id="b").reason, "awaiting_result")

    def test_confidence_drop_and_frame_gap_do_not_unlock_same_target(self):
        self.press_once()
        self.assertEqual(self.update(Point(0.5, 0.5), 0.4, finger_confidence=0.1).action, "stop")
        self.assertEqual(self.update(Point(0.5, 0.5), 1.5).reason, "awaiting_result")

    def test_bad_timestamp_after_press_does_not_unlock_same_target(self):
        self.press_once()
        with self.assertRaises(ValueError):
            self.update(Point(0.5, 0.5), 0.3)
        self.assertEqual(self.update(Point(0.5, 0.5), 0.4).reason, "awaiting_result")

    def test_explicit_reset_attempt_permits_fresh_dwell(self):
        self.press_once()
        self.engine.reset_attempt()
        for t in (0.4, 0.5, 0.6):
            self.assertEqual(self.update(Point(0.5, 0.5), t).action, "hold")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.7).action, "press")

    def test_new_target_or_keyframe_cannot_unlock_pending_attempt(self):
        for context in ("target", "keyframe", "both"):
            with self.subTest(context=context):
                self.engine = GuidanceEngine()
                self.target = target()
                self.press_once(keyframe_id=1)
                if context in ("target", "both"):
                    self.target = target(element_id="button-2")
                keyframe_id = 2 if context in ("keyframe", "both") else 1
                for t in (0.4, 0.5, 0.6, 0.7):
                    result = self.update(Point(0.5, 0.5), t, keyframe_id=keyframe_id)
                    self.assertEqual(result.action, "hold")
                    self.assertEqual(result.reason, "awaiting_result")
                    self.assertIsNone(result.distance_band)

    def test_explicit_reset_allows_new_target_and_keyframe_after_pending_attempt(self):
        self.press_once(keyframe_id=1)
        self.target = target(element_id="button-2")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.4, keyframe_id=2).reason, "awaiting_result")
        self.engine.reset_attempt()
        for t in (0.5, 0.6, 0.7):
            self.assertEqual(self.update(Point(0.5, 0.5), t, keyframe_id=2).action, "hold")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.8, keyframe_id=2).action, "press")

    def test_price_is_noninteractive_and_cannot_receive_press_cue(self):
        self.target = Element("price-1", "price", Box(0.45, 0.45, 0.55, 0.55), 0.95, price=2000)
        for t in (0, 0.1, 0.2, 0.3, 0.4):
            result = self.update(Point(0.5, 0.5), t)
            self.assertEqual(result.action, "stop")
            self.assertEqual(result.reason, "noninteractive_target")
            self.assertTrue(result.reacquire)

    def test_distance_band_maps_specification_values(self):
        self.assertEqual(self.update(Point(0.1, 0.5), 0).distance_band, "far")
        self.target = target(box=Box(0.499, 0.499, 0.501, 0.501))
        self.assertEqual(self.update(Point(0.49, 0.5), 0.1).distance_band, "near")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.2).distance_band, "reached")
        self.assertIsNone(self.update(None, 0.3).distance_band)

    def test_three_seconds_of_increasing_error_emits_one_recenter(self):
        for i in range(30):
            self.assertFalse(self.update(Point(0.4 - i * 0.005, 0.5), i * 0.1).recenter)
        self.assertTrue(self.update(Point(0.25, 0.5), 3.0).recenter)
        self.assertFalse(self.update(Point(0.245, 0.5), 3.1).recenter)

    def test_error_decrease_breaks_continuous_increase(self):
        for i in range(29):
            self.update(Point(0.4 - i * 0.005, 0.5), i * 0.1)
        self.assertFalse(self.update(Point(0.4, 0.5), 2.9).recenter)
        self.assertFalse(self.update(Point(0.39, 0.5), 3.0).recenter)

    def test_missing_sample_resets_increasing_error_timer(self):
        for i in range(29):
            self.update(Point(0.4 - i * 0.005, 0.5), i * 0.1)
        self.update(None, 2.9)
        self.assertFalse(self.update(Point(0.25, 0.5), 3.0).recenter)

    def test_invalid_context_resets_dwell(self):
        self.update(Point(0.5, 0.5), 0)
        self.assertEqual(self.update(Point(0.5, 0.5), 0.1, keyframe_id=-1).reason, "invalid_keyframe")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.2, hand_id="").reason, "invalid_hand_id")
        self.assertEqual(self.update(Point(0.5, 0.5), 0.3).dwell_seconds, 0)

    def test_configuration_rejects_invalid_values(self):
        for kwargs in ({"near_distance": 0}, {"dwell_duration": -1}, {"max_frame_gap": 0},
                       {"recenter_duration": float("nan")}, {"min_confidence": 2}, {"increasing_epsilon": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GuidanceEngine(**kwargs)


if __name__ == "__main__":
    unittest.main()
