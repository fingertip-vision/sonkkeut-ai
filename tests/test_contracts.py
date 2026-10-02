import unittest

from sonkkeut_vision.contracts import Box, Element, Point, ScreenSnapshot


class ContractTests(unittest.TestCase):
    def test_chapter_six_screen_round_trip(self):
        data = {"screen_type": "menu", "keyframe_id": 12, "elements": [
            {"id": "e1", "kind": "tab", "text": "커피", "box": [0.02, 0.10, 0.20, 0.16], "conf": 0.94},
            {"id": "e7", "kind": "menu", "text": "아메리카노", "price": 4500,
             "box": [0.05, 0.25, 0.30, 0.45], "conf": 0.91},
            {"id": "e20", "kind": "button", "text": "장바구니", "box": [0.70, 0.88, 0.98, 0.97], "conf": 0.97}]}
        self.assertEqual(ScreenSnapshot.from_dict(data).to_dict(), data)

    def test_reject_bad_box_and_nan(self):
        for values in [(0.3, 0.2, 0.1, 0.4), (0, 0, 0, 1), (-0.1, 0, 1, 1), (0, 0, float("nan"), 1)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Box(*values)
        with self.assertRaises(ValueError):
            Point(float("inf"), 1)

    def test_offscreen_point_is_preserved_internally(self):
        self.assertEqual(Point(-2, 500).x, -2)

    def test_duplicate_id_is_rejected(self):
        element = Element("e1", "menu", Box(0, 0, 0.5, 0.5), 0.9)
        with self.assertRaises(ValueError):
            ScreenSnapshot("menu", 1, (element, element))

    def test_invalid_confidence_type_and_price(self):
        for conf in [-0.1, 1.1, True, float("nan")]:
            with self.subTest(conf=conf), self.assertRaises(ValueError):
                Element("e1", "menu", Box(0, 0, 1, 1), conf)
        for price in [-1, 4.5, True]:
            with self.subTest(price=price), self.assertRaises(ValueError):
                Element("e1", "price", Box(0, 0, 1, 1), 0.9, price=price)


if __name__ == "__main__":
    unittest.main()
