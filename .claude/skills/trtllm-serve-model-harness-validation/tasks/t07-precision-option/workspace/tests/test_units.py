# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

from convert.units import UnknownUnitError, convert, format_result


class ConvertTest(unittest.TestCase):
    def test_metric(self):
        self.assertAlmostEqual(convert(2.5, "km", "m"), 2500.0)

    def test_miles_to_kilometres(self):
        self.assertAlmostEqual(convert(1, "mi", "km"), 1.609344)

    def test_unknown_unit(self):
        with self.assertRaises(UnknownUnitError):
            convert(1, "m", "league")


class FormatResultTest(unittest.TestCase):
    def test_two_decimal_places(self):
        self.assertEqual(format_result(6.2137119, "mi"), "6.21 mi")


if __name__ == "__main__":
    unittest.main()
