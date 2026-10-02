import unittest
from dataclasses import dataclass

from sonkkeut_vision.contracts import Point
from sonkkeut_vision.tracking import FingertipTracker, HandObservation


@dataclass(frozen=True)
class PixelProjector:
    scale: float = 100.0

    def camera_to_screen(self, point):
        return Point(point.x / self.scale, point.y / self.scale)


class InvalidProjector:
    def camera_to_screen(self, point):
        raise ValueError("no valid plane")


def hand(x=50, y=50, *, hand_id="left", conf=0.95, distance=None):
    landmarks = [Point(99, 99)] * 21
    landmarks[8] = Point(x, y)
    return HandObservation(hand_id, tuple(landmarks), conf, distance)


class FingertipTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = FingertipTracker()
        self.plane = PixelProjector()

    def update(self, observation, timestamp, **kwargs):
        hands = [] if observation is None else [observation]
        return self.tracker.update(hands, self.plane, timestamp, **kwargs)

    def test_joint_eight_is_projected_from_camera_pixels(self):
        state = self.update(hand(25, 75), 0)
        self.assertTrue(state.valid)
        self.assertEqual(state.raw_point, Point(0.25, 0.75))
        self.assertEqual(state.point, state.raw_point)

    def test_recent_five_mean_discards_older_points(self):
        for i in range(7):
            state = self.update(hand(i * 10, 50), i * 0.05)
        self.assertAlmostEqual(state.point.x, 0.4)
        self.assertAlmostEqual(state.raw_point.x, 0.6)

    def test_confidence_threshold_is_inclusive(self):
        self.assertTrue(self.update(hand(conf=0.7), 0).valid)
        self.assertEqual(self.update(hand(conf=0.699), 0.05).reason, "low_hand_confidence")

    def test_low_confidence_clears_old_coordinates(self):
        self.update(hand(10), 0)
        rejected = self.update(hand(10, conf=0.1), 0.05)
        self.assertIsNone(rejected.point)
        self.assertEqual(self.update(hand(90), 0.1).point, Point(0.9, 0.5))

    def test_offscreen_projection_is_rejected_without_clamping(self):
        self.update(hand(100), 0)
        for i, (x, y) in enumerate(((-1, 50), (101, 50), (50, -1), (50, 101)), 1):
            state = self.update(hand(x, y), i * 0.05)
            self.assertEqual(state.reason, "outside_plane")
            self.assertIsNone(state.raw_point)
            self.assertIsNone(state.point)
        self.assertEqual(self.update(hand(50), 0.3).point, Point(0.5, 0.5))

    def test_plane_boundary_remains_valid(self):
        for i, (x, y) in enumerate(((0, 0), (100, 100), (100, 0), (0, 100))):
            self.assertTrue(self.update(hand(x, y), i * 0.05).valid)

    def test_missing_hand_clears_history_and_becomes_lost_after_one_second(self):
        self.update(hand(10), 0)
        state = self.update(None, 0.1)
        self.assertEqual(state.reason, "missing_hand")
        self.assertIsNone(state.point)
        self.assertEqual(self.update(None, 0.99).reason, "missing_hand")
        self.assertEqual(self.update(None, 1.0).reason, "lost_hand")
        self.assertEqual(self.update(hand(90), 1.05).point, Point(0.9, 0.5))

    def test_missing_from_first_frame_has_its_own_timer(self):
        self.assertEqual(self.update(None, 10).reason, "missing_hand")
        self.assertEqual(self.update(None, 10.999).reason, "missing_hand")
        self.assertEqual(self.update(None, 11).reason, "lost_hand")

    def test_valid_observation_resets_missing_timer(self):
        self.update(None, 0)
        self.update(hand(), 0.9)
        self.assertEqual(self.update(None, 1.1).reason, "missing_hand")
        self.assertEqual(self.update(None, 1.9).reason, "lost_hand")

    def test_identity_change_clears_smoothing(self):
        self.update(hand(10, hand_id="a"), 0)
        state = self.update(hand(90, hand_id="b"), 0.05)
        self.assertEqual(state.point, Point(0.9, 0.5))
        self.assertEqual(state.hand_id, "b")

    def test_large_frame_gap_clears_smoothing(self):
        self.update(hand(10), 0)
        self.assertEqual(self.update(hand(90), 0.201).point, Point(0.9, 0.5))

    def test_maximum_frame_gap_is_configurable(self):
        tracker = FingertipTracker(max_frame_gap=0.05)
        tracker.update([hand(10)], self.plane, 0)
        self.assertEqual(tracker.update([hand(90)], self.plane, 0.06).point, Point(0.9, 0.5))

    def test_equal_and_reversed_timestamps_are_rejected_and_reset(self):
        self.update(hand(10), 1)
        for timestamp in (1, 0.5, float("nan"), float("inf"), True):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                self.update(hand(90), timestamp)
        self.assertEqual(self.update(hand(90), 1.1).point, Point(0.9, 0.5))

    def test_multiple_unmeasured_hands_are_ambiguous(self):
        state = self.tracker.update([hand(hand_id="a"), hand(hand_id="b")], self.plane, 0)
        self.assertEqual(state.reason, "ambiguous_hand")
        self.assertIsNone(state.point)

    def test_measured_distance_selects_closest_to_plane(self):
        hands = [hand(10, hand_id="a", distance=5), hand(90, hand_id="b", distance=2)]
        state = self.tracker.update(hands, self.plane, 0)
        self.assertEqual(state.hand_id, "b")
        self.assertEqual(state.raw_point, Point(0.9, 0.5))
        self.assertEqual(state.selection_basis, "measured_plane_distance")

    def test_unmeasured_distance_retains_identity_without_nearest_claim(self):
        self.update(hand(10, hand_id="a"), 0)
        hands = [hand(20, hand_id="a"), hand(90, hand_id="b", distance=0)]
        state = self.tracker.update(hands, self.plane, 0.05)
        self.assertEqual(state.hand_id, "a")
        self.assertEqual(state.selection_basis, "retained_identity")

    def test_partial_measured_distances_do_not_establish_nearest(self):
        state = self.tracker.update([hand(hand_id="a", distance=0), hand(hand_id="b")], self.plane, 0)
        self.assertEqual(state.reason, "ambiguous_hand")

    def test_equal_measured_distances_require_identity_or_ambiguity(self):
        hands = [hand(hand_id="a", distance=1), hand(hand_id="b", distance=1)]
        self.assertEqual(self.tracker.update(hands, self.plane, 0).reason, "ambiguous_hand")
        self.update(hand(hand_id="a"), 0.05)
        state = self.tracker.update(hands, self.plane, 0.1)
        self.assertEqual(state.hand_id, "a")
        self.assertEqual(state.selection_basis, "retained_identity")

    def test_duplicate_hand_identity_is_ambiguous(self):
        state = self.tracker.update([hand(), hand(90)], self.plane, 0)
        self.assertEqual(state.reason, "ambiguous_hand")

    def test_closest_low_confidence_hand_is_not_silently_replaced(self):
        state = self.tracker.update([hand(hand_id="near", conf=0.1, distance=1),
                                     hand(hand_id="far", distance=3)], self.plane, 0)
        self.assertEqual(state.reason, "low_hand_confidence")
        self.assertEqual(state.hand_id, "near")

    def test_missing_or_invalid_plane_clears_smoothing(self):
        self.update(hand(10), 0)
        self.assertEqual(self.tracker.update([hand()], None, 0.05).reason, "invalid_plane")
        self.assertEqual(self.tracker.update([hand()], InvalidProjector(), 0.1).reason, "invalid_plane")
        self.assertEqual(self.update(hand(90), 0.15).point, Point(0.9, 0.5))

    def test_changed_keyframe_clears_smoothing(self):
        self.update(hand(10), 0, keyframe_id=1)
        state = self.update(hand(90), 0.05, keyframe_id=2)
        self.assertEqual(state.point, Point(0.9, 0.5))

    def test_valid_camera_pose_updates_preserve_screen_space_history(self):
        self.update(hand(20, 20), 0, keyframe_id=1)
        state = self.tracker.update([hand(40, 40)], PixelProjector(200), 0.05, keyframe_id=1)
        self.assertEqual(state.raw_point, Point(0.2, 0.2))
        self.assertEqual(state.point, Point(0.2, 0.2))
        state = self.tracker.update([hand(80, 80)], PixelProjector(200), 0.1, keyframe_id=1)
        self.assertAlmostEqual(state.point.x, (0.2 + 0.2 + 0.4) / 3)
        self.assertAlmostEqual(state.point.y, (0.2 + 0.2 + 0.4) / 3)

    def test_invalid_keyframe_resets_history(self):
        self.update(hand(10), 0)
        self.assertEqual(self.update(hand(90), 0.05, keyframe_id=-1).reason, "invalid_keyframe")
        self.assertEqual(self.update(hand(90), 0.1).point, Point(0.9, 0.5))

    def test_invalid_observation_fails_closed(self):
        state = self.tracker.update([None], self.plane, 0)
        self.assertEqual(state.reason, "invalid_hand_observation")

    def test_hand_contract_rejects_invalid_landmarks_and_distance(self):
        for landmarks in ([], [Point(1, 2)] * 20, [None] * 21):
            with self.subTest(landmarks=landmarks), self.assertRaises(ValueError):
                HandObservation("a", landmarks, 0.9)
        for distance in (-1, float("nan"), float("inf"), True):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                hand(distance=distance)

    def test_configuration_rejects_invalid_values(self):
        for kwargs in ({"smoothing_frames": 0}, {"smoothing_frames": True}, {"max_frame_gap": 0},
                       {"missing_timeout": -1}, {"min_confidence": 2}, {"max_frame_gap": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FingertipTracker(**kwargs)


if __name__ == "__main__":
    unittest.main()
