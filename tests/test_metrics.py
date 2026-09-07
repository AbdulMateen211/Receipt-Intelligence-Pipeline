import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import levenshtein, character_error_rate, money_matches, text_matches


class TestMetrics(unittest.TestCase):
    def test_levenshtein_identical(self):
        self.assertEqual(levenshtein("hello", "hello"), 0)

    def test_levenshtein_known_distance(self):
        # kitten -> sitting is the textbook edit-distance-3 example.
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_levenshtein_empty_strings(self):
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("", ""), 0)

    def test_cer_perfect_match(self):
        self.assertEqual(character_error_rate("hello world", "hello world"), 0.0)

    def test_cer_ignores_whitespace_differences(self):
        # Collapsing whitespace means layout/line-break noise shouldn't
        # inflate the error rate.
        self.assertEqual(character_error_rate("hello   world", "hello\nworld"), 0.0)

    def test_cer_empty_reference(self):
        self.assertEqual(character_error_rate("", ""), 0.0)
        self.assertEqual(character_error_rate("", "x"), 1.0)

    def test_money_matches_within_tolerance(self):
        self.assertTrue(money_matches(10.00, 10.005, tol=0.01))
        self.assertFalse(money_matches(10.00, 10.50, tol=0.01))

    def test_money_matches_none_is_never_a_match(self):
        self.assertFalse(money_matches(None, 10.0))
        self.assertFalse(money_matches(10.0, None))

    def test_text_matches_case_and_whitespace_insensitive(self):
        self.assertTrue(text_matches("Greenleaf   Market", "greenleaf market"))
        self.assertFalse(text_matches("Greenleaf Market", "Corner Mart"))


if __name__ == "__main__":
    unittest.main()
