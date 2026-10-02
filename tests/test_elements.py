import math
import unittest

from sonkkeut_vision.contracts import Box
from sonkkeut_vision.elements import Detection, process_detections


class ElementPostprocessingTests(unittest.TestCase):
    def test_confidence_filter_and_same_class_nms(self):
        detections = (
            Detection("button", Box(0.1, 0.1, 0.5, 0.5), 0.9),
            Detection("button", Box(0.12, 0.12, 0.52, 0.52), 0.8),
            Detection("menu", Box(0.6, 0.6, 0.9, 0.9), 0.49),
        )
        result = process_detections(detections, 3)
        self.assertEqual(len(result.elements), 1)
        self.assertEqual(result.elements[0].box, detections[0].box)
        self.assertEqual(result.elements[0].id, "e3_1")
        self.assertFalse(result.requires_plane_reacquisition)

    def test_different_classes_survive_identical_boxes(self):
        box = Box(0.1, 0.1, 0.4, 0.4)
        result = process_detections((Detection("button", box, 0.9), Detection("price", box, 0.8)), 0)
        self.assertEqual({element.kind for element in result.elements}, {"button", "price"})
        self.assertEqual(len(result.elements), 2)

    def test_reading_order_and_ids_are_stable_under_input_permutation(self):
        detections = (
            Detection("menu", Box(0.6, 0.1, 0.9, 0.3), 0.99),
            Detection("button", Box(0.1, 0.1, 0.4, 0.3), 0.7),
            Detection("back", Box(0.01, 0.8, 0.1, 0.9), 0.8),
            Detection("tab", Box(0.1, 0.4, 0.4, 0.5), 0.5),
        )
        result = process_detections(detections, 6)
        reversed_result = process_detections(reversed(detections), 6)
        self.assertEqual(result, reversed_result)
        self.assertEqual([element.kind for element in result.elements], ["button", "menu", "tab", "back"])
        self.assertEqual([element.id for element in result.elements], ["e6_1", "e6_2", "e6_3", "e6_4"])
        next_frame = process_detections(detections, 7)
        self.assertEqual(next_frame.elements[0].id, "e7_1")

    def test_equal_confidence_nms_tie_is_deterministic(self):
        first = Detection("button", Box(0.1, 0.1, 0.5, 0.5), 0.9)
        second = Detection("button", Box(0.12, 0.12, 0.52, 0.52), 0.9)
        self.assertEqual(process_detections((first, second), 0), process_detections((second, first), 0))
        self.assertEqual(process_detections((second, first), 0).elements[0].box, first.box)

    def test_threshold_boundaries_are_explicit(self):
        duplicate = Detection("button", Box(0.1, 0.1, 0.5, 0.5), 0.5)
        result = process_detections((duplicate, duplicate), 0, min_confidence=0.5, nms_iou_threshold=1)
        self.assertEqual(len(result.elements), 2)  # Suppression uses strictly greater IoU.
        result = process_detections((duplicate, duplicate), 0, nms_iou_threshold=0.99)
        self.assertEqual(len(result.elements), 1)

    def test_empty_or_fully_filtered_result_requires_plane_reacquisition(self):
        self.assertTrue(process_detections((), 0).requires_plane_reacquisition)
        detection = Detection("menu", Box(0, 0, 1, 1), 0.4)
        result = process_detections((detection,), 1)
        self.assertEqual(result.elements, ())
        self.assertTrue(result.requires_plane_reacquisition)

    def test_no_ocr_or_screen_classification_is_inferred(self):
        result = process_detections((Detection("price", Box(0.1, 0.1, 0.3, 0.3), 0.9),), 1)
        self.assertIsNone(result.elements[0].text)
        self.assertIsNone(result.elements[0].price)

    def test_invalid_metadata_and_tuning_are_rejected(self):
        for conf in (math.nan, math.inf, -0.1, 1.1, True):
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                Detection("button", Box(0, 0, 1, 1), conf)
        with self.assertRaises(ValueError):
            Detection("unknown", Box(0, 0, 1, 1), 0.8)
        with self.assertRaises(ValueError):
            Detection("button", (0, 0, 1, 1), 0.8)
        for keyframe in (-1, 1.2, True):
            with self.subTest(keyframe=keyframe), self.assertRaises(ValueError):
                process_detections((), keyframe)
        for setting in ({"min_confidence": math.nan}, {"nms_iou_threshold": 1.1}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                process_detections((), 0, **setting)
        with self.assertRaises(ValueError):
            process_detections((object(),), 0)


if __name__ == "__main__":
    unittest.main()
