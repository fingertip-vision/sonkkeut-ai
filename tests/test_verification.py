import unittest

from sonkkeut_vision.contracts import Box, Element, ScreenSnapshot
from sonkkeut_vision.verification import ExpectedEffect, PressVerifier, screen_changed


def screen(kind="menu", frame=1, text="아메리카노", conf=0.95, offset=0):
    return ScreenSnapshot(kind, frame, (Element(f"e{frame}", "menu", Box(0.1 + offset, 0.1, 0.3 + offset, 0.3), conf, text),))


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.verifier = PressVerifier()
        self.before = screen()
        self.verifier.begin(self.before, ExpectedEffect(screen_type="option"), 0)

    def observe(self, after, timestamp, **kwargs):
        kwargs.setdefault("captured_at", timestamp)
        if after is not None and "cart_quantities" in kwargs:
            kwargs.setdefault("cart_keyframe_id", after.keyframe_id)
        return self.verifier.observe(after, timestamp, **kwargs)

    def test_expected_screen_is_success(self):
        result = self.observe(screen("option", 2), 0.5)
        self.assertEqual((result.verdict, result.reason, result.terminal), ("success", "expected_screen", True))

    def test_unrelated_text_change_does_not_count_as_success(self):
        result = self.observe(screen(frame=2, text="광고"), 0.5)
        self.assertEqual(result.verdict, "uncertain")
        result = self.observe(screen(frame=3, text="광고"), 1.5)
        self.assertEqual(result.verdict, "failure")

    def test_ids_confidence_and_small_jitter_are_not_changes(self):
        self.assertFalse(screen_changed(self.before, screen(frame=2, conf=0.9, offset=0.005)))

    def test_unchanged_screen_at_timeout_is_failure(self):
        result = self.observe(screen(frame=2), 1.5)
        self.assertEqual((result.verdict, result.reason), ("failure", "no_screen_change"))

    def test_normal_frame_cadence_does_not_need_exact_deadline(self):
        for index in range(1, 8):
            result = self.observe(screen(frame=index + 1), index * 0.2)
            self.assertFalse(result.terminal)
        self.assertEqual(self.observe(screen(frame=9), 1.6).reason, "no_screen_change")

    def test_cart_unchanged_with_no_exact_deadline_frame(self):
        self.verifier.begin(screen("cart"), ExpectedEffect(cart_item_key="x|hot", before_quantity=0), 0)
        for index in range(1, 8):
            result = self.observe(screen("cart", index + 1), index * 0.2, cart_quantities={"x|hot": 0})
            self.assertFalse(result.terminal)
        self.assertEqual(self.observe(screen("cart", 9), 1.6, cart_quantities={"x|hot": 0}).reason, "cart_unchanged")

    def test_late_restored_screen_after_a_change_is_not_unpressed(self):
        self.observe(screen(frame=2, text="변경"), 0.5)
        self.assertEqual(self.observe(screen(frame=3), 1.6).reason, "late_observation")

    def test_expected_screen_at_exact_deadline_is_success(self):
        self.assertEqual(self.observe(screen("option", 2), 1.5).verdict, "success")

    def test_late_expected_screen_is_uncertain(self):
        self.assertEqual(self.observe(screen("option", 2), 1.6).reason, "late_observation")

    def test_missing_screen_at_timeout_is_uncertain(self):
        result = self.observe(None, 1.5)
        self.assertEqual((result.verdict, result.reason, result.terminal), ("uncertain", "missing_screen", True))

    def test_buffered_pre_press_frame_does_not_succeed(self):
        self.verifier.begin(screen(), ExpectedEffect(screen_type="option"), 10)
        result = self.observe(screen("option", 2), 10.2, captured_at=9.8)
        self.assertEqual((result.verdict, result.reason), ("uncertain", "stale_capture"))

    def test_missing_or_future_capture_time_is_uncertain(self):
        self.assertEqual(self.observe(screen("option", 2), 0.5, captured_at=None).reason, "missing_capture_time")
        self.assertEqual(self.observe(screen("option", 2), 0.5, captured_at=0.6).reason, "stale_capture")

    def test_processing_delay_preserves_in_window_capture_evidence(self):
        self.assertEqual(self.observe(screen("option", 2), 1.7, captured_at=1.4).verdict, "success")

    def test_early_unchanged_capture_delivered_late_is_not_no_press(self):
        self.assertEqual(self.observe(screen(frame=2), 1.7, captured_at=1.4).reason, "incomplete_window")

    def test_cart_evidence_must_be_bound_to_same_keyframe(self):
        self.verifier.begin(screen("cart"), ExpectedEffect(cart_item_key="x|hot", before_quantity=0), 0)
        result = self.observe(screen("cart", 2), 0.5, cart_quantities={"x|hot": 1}, cart_keyframe_id=1)
        self.assertEqual((result.verdict, result.reason), ("uncertain", "unbound_cart_evidence"))

    def test_invalid_cart_input_does_not_consume_corrected_frame(self):
        self.verifier.begin(screen("cart"), ExpectedEffect(cart_item_key="x|hot", before_quantity=0), 0)
        with self.assertRaises(ValueError):
            self.observe(screen("cart", 2), 0.5, cart_quantities={"x|hot": -1})
        self.assertEqual(self.observe(screen("cart", 2), 0.5, cart_quantities={"x|hot": 1}).verdict, "success")

    def test_changed_then_restored_is_not_unpressed(self):
        self.observe(screen(frame=2, text="변경"), 0.5)
        self.assertEqual(self.observe(screen(frame=3), 1.5).reason, "unexpected_change")

    def test_restart_flag_and_observation_types_are_strict(self):
        with self.assertRaises(ValueError):
            self.observe(screen("option", 2), 0.5, returned_to_start="false")
        with self.assertRaises(ValueError):
            self.verifier.observe({}, 0.5)
        self.assertEqual(self.observe(screen("option", 2), 0.5).verdict, "success")

    def test_old_or_out_of_order_frame_does_not_pass(self):
        self.assertEqual(self.observe(screen("option", 1), 0.3).reason, "stale_keyframe")
        self.assertEqual(self.observe(screen(frame=3), 0.4).verdict, "uncertain")
        self.assertEqual(self.observe(screen("option", 2), 0.5).reason, "stale_keyframe")

    def test_unknown_and_low_confidence_are_uncertain(self):
        self.assertEqual(self.observe(screen("other", 2), 0.5).verdict, "uncertain")
        self.assertEqual(self.observe(screen("option", 3, conf=0.2), 1.5).verdict, "uncertain")

    def test_empty_observation_does_not_imply_press_failure(self):
        result = self.observe(ScreenSnapshot("option", 2, ()), 1.5)
        self.assertEqual(result.verdict, "uncertain")

    def test_explicit_restart_retains_progress(self):
        result = self.observe(screen(frame=2), 0.5, returned_to_start=True)
        self.assertEqual((result.verdict, result.reason, result.retain_progress), ("failure", "returned_to_start", True))

    def test_unexpected_screen_transition_is_failure(self):
        self.assertEqual(self.observe(screen("payment", 2), 0.5).reason, "unexpected_screen")

    def test_before_already_matches_expectation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.verifier.begin(screen("option"), ExpectedEffect(screen_type="option"), 0)

    def test_terminal_result_is_cached_until_next_attempt(self):
        first = self.observe(screen("option", 2), 0.5)
        self.assertEqual(self.observe(screen("payment", 3), 0.6), first)
        self.verifier.begin(screen("option", 3), ExpectedEffect(screen_type="cart"), 1)
        self.assertEqual(self.observe(screen("cart", 4), 1.2).verdict, "success")

    def test_cart_only_exact_item_and_delta_succeed(self):
        self.verifier.begin(screen("option"), ExpectedEffect(cart_item_key="아메리카노|hot", before_quantity=0, quantity_delta=2), 0)
        self.assertEqual(self.observe(screen("cart", 2), 0.5, cart_quantities={"아메리카노|ice": 2}).verdict, "uncertain")
        self.assertEqual(self.observe(screen("cart", 3), 0.6, cart_quantities={"아메리카노|hot": 2}).verdict, "success")

    def test_missing_cart_evidence_is_not_zero(self):
        self.verifier.begin(screen("cart"), ExpectedEffect(cart_item_key="아메리카노|hot", before_quantity=0), 0)
        self.assertEqual(self.observe(screen("cart", 2), 1.5).verdict, "uncertain")

    def test_cart_unchanged_and_double_add_are_failure(self):
        for count, reason in [(0, "cart_unchanged"), (2, "unexpected_cart_quantity")]:
            with self.subTest(count=count):
                self.verifier.begin(screen("cart"), ExpectedEffect(cart_item_key="아메리카노|hot", before_quantity=0), 0)
                self.assertEqual(self.observe(screen("cart", 2), 1.5, cart_quantities={"아메리카노|hot": count}).reason, reason)

    def test_invalid_cart_evidence_and_timestamps(self):
        self.verifier.begin(screen("cart"), ExpectedEffect(cart_item_key="아메리카노|hot", before_quantity=0), 1)
        with self.assertRaises(ValueError):
            self.observe(screen("cart", 2), 0.5)
        with self.assertRaises(ValueError):
            self.observe(screen("cart", 2), 1.2, cart_quantities={"아메리카노|hot": -1})

    def test_no_expectation_or_unknown_expectation_rejected(self):
        for kwargs in [{}, {"screen_type": "other"}, {"screen_type": "cart", "cart_item_key": "x"},
                       {"cart_item_key": "x", "before_quantity": None}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExpectedEffect(**kwargs)


if __name__ == "__main__":
    unittest.main()
