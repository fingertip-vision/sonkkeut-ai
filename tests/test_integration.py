import unittest

from sonkkeut_vision.contracts import Box, Element, Point
from sonkkeut_vision.demo import hand_at, run_demo
from sonkkeut_vision.geometry import PlaneTracker
from sonkkeut_vision.guidance import GuidanceEngine
from sonkkeut_vision.tracking import FingertipTracker


class VisualCoreIntegrationTests(unittest.TestCase):
    def test_three_synthetic_transition_fixtures_repeat_twenty_times(self):
        # Offline model-output fixtures, not the spec's actual kiosk/user trials.
        for repeat in range(20):
            with self.subTest(repeat=repeat):
                demo = run_demo()
                self.assertFalse(demo["models_executed"])
                for flow in demo["flows"]:
                    self.assertEqual(flow["verification"]["verdict"], "success")
                    presses = [event for event in flow["events"] if event["action"] == "press"]
                    self.assertEqual(len(presses), 1)
                    self.assertGreaterEqual(presses[0]["dwell_seconds"] + 1e-9, 0.3)

    def test_plane_loss_stops_guidance_even_with_old_hand_observation(self):
        corners = (Point(0, 0), Point(1280, 0), Point(1280, 720), Point(0, 720))
        plane_tracker = PlaneTracker()
        fingertip = FingertipTracker()
        guidance = GuidanceEngine()
        target = Element("e1", "button", Box(0.4, 0.4, 0.6, 0.6), 0.95)
        plane = plane_tracker.update(corners, 0)
        observed = hand_at(plane.homography, Point(0.5, 0.5))
        for index in range(3):
            timestamp = index / 10
            state = fingertip.update((observed,), plane.homography, timestamp, keyframe_id=1)
            decision = guidance.update(state.point, target, timestamp, raw_point=state.raw_point,
                                       finger_confidence=state.confidence, keyframe_id=1, hand_id=state.hand_id)
            self.assertNotEqual(decision.action, "press")
        invalid_plane = plane_tracker.update(None, 0.3)
        state = fingertip.update((observed,), invalid_plane.homography, 0.3, keyframe_id=1)
        decision = guidance.update(state.point, target, 0.3, raw_point=state.raw_point,
                                   finger_confidence=state.confidence, keyframe_id=1, hand_id=state.hand_id)
        self.assertEqual(decision.action, "stop")
        recovered = plane_tracker.update(corners, 0.4)
        state = fingertip.update((observed,), recovered.homography, 0.4, keyframe_id=1)
        decision = guidance.update(state.point, target, 0.4, raw_point=state.raw_point,
                                   finger_confidence=state.confidence, keyframe_id=1, hand_id=state.hand_id)
        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.dwell_seconds, 0)


if __name__ == "__main__":
    unittest.main()
