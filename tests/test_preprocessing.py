import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.preprocessing import preprocess_pipeline, simple_grayscale, _estimate_rotation_correction
import cv2


def _make_blank_receipt(w=300, h=400, angle=0.0):
    """A plain white rectangle on a gray background, optionally rotated --
    enough structure to exercise the contour-detection code path without
    needing a real receipt image."""
    canvas = np.full((h + 100, w + 100, 3), 180, dtype=np.uint8)
    canvas[50:50 + h, 50:50 + w] = 255
    cv2.rectangle(canvas, (70, 70), (w - 20, h - 20), (0, 0, 0), 2)
    if angle != 0.0:
        center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
        mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        canvas = cv2.warpAffine(canvas, mat, (canvas.shape[1], canvas.shape[0]), borderValue=(180, 180, 180))
    return Image.fromarray(canvas)


class TestPreprocessing(unittest.TestCase):
    def test_simple_grayscale_converts_mode(self):
        img = Image.new("RGB", (50, 50), color=(10, 20, 30))
        gray = simple_grayscale(img)
        self.assertEqual(gray.mode, "L")

    def test_pipeline_runs_end_to_end_without_error(self):
        img = _make_blank_receipt(angle=7.0)
        result = preprocess_pipeline(img)
        self.assertIsInstance(result.final, Image.Image)
        self.assertGreater(result.final.size[0], 0)
        self.assertGreater(result.final.size[1], 0)

    def test_rotation_estimate_recovers_known_angle(self):
        img = _make_blank_receipt(angle=8.0)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        angle = _estimate_rotation_correction(bgr)
        # We rotated the content by +8 degrees, so the correction should be
        # close to -8 degrees. Allow some tolerance since this is a coarse
        # rectangle, not real receipt text.
        self.assertAlmostEqual(angle, -8.0, delta=2.0)

    def test_pipeline_handles_already_straight_image(self):
        img = _make_blank_receipt(angle=0.0)
        result = preprocess_pipeline(img)
        self.assertLess(abs(result.angle_corrected_deg), 2.0)


if __name__ == "__main__":
    unittest.main()
