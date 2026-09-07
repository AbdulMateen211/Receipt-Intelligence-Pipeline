import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction import extract_fields


class TestExtraction(unittest.TestCase):
    def test_basic_receipt_fields(self):
        text = (
            "GREENLEAF MARKET\n"
            "Receipt #R0003_H9SD\n"
            "2026-08-21 08:44\n"
            "----------------------------------------\n"
            "Pasta 500g                   x3   4.59\n"
            "Free Range Eggs 12ct         x2   8.74\n"
            "Olive Oil 500ml                  13.22\n"
            "----------------------------------------\n"
            "SUBTOTAL                        26.55\n"
            "TAX (10.00%)                     2.66\n"
            "TOTAL                           29.21\n"
            "VISA **** 4471\n"
            "FOLLOW US @STORE ONLINE\n"
        )
        fields = extract_fields(text)
        self.assertEqual(fields.merchant, "GREENLEAF MARKET")
        self.assertEqual(fields.date, "2026-08-21")
        self.assertAlmostEqual(fields.subtotal, 26.55)
        self.assertAlmostEqual(fields.tax, 2.66)
        self.assertAlmostEqual(fields.total, 29.21)
        self.assertEqual(len(fields.items), 3)

    def test_tax_line_does_not_pick_up_the_percentage(self):
        # Regression test: "TAX (8.00%)   3.44" contains two number-like
        # tokens. We must extract the trailing amount (3.44), not the rate.
        text = "TAX (8.00%)                     3.44\n"
        fields = extract_fields(text)
        self.assertAlmostEqual(fields.tax, 3.44)

    def test_missing_fields_return_none(self):
        fields = extract_fields("just some noise\nwith no structure\n")
        self.assertIsNone(fields.total)
        self.assertIsNone(fields.subtotal)
        self.assertEqual(fields.items, [])

    def test_subtotal_not_confused_with_total(self):
        text = "SUBTOTAL   10.00\nTOTAL   10.80\n"
        fields = extract_fields(text)
        self.assertAlmostEqual(fields.subtotal, 10.00)
        self.assertAlmostEqual(fields.total, 10.80)

    def test_item_lines_are_parsed(self):
        text = "Bananas 1kg                          1.35\nCoffee                                3.50\n"
        fields = extract_fields(text)
        names = [it.name for it in fields.items]
        prices = [it.price for it in fields.items]
        self.assertIn("Bananas 1kg", names)
        self.assertIn(3.50, prices)


if __name__ == "__main__":
    unittest.main()
