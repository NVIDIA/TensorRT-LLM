# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

from reports import daily, monthly, weekly
from reports.util import round_half_up


class RoundHalfUpTest(unittest.TestCase):
    def test_halves_round_up(self):
        self.assertEqual(round_half_up(2.675), 2.68)
        self.assertEqual(round_half_up(0.125), 0.13)

    def test_places(self):
        self.assertEqual(round_half_up(1.25, places=1), 1.3)
        self.assertEqual(round_half_up(7.5, places=0), 8.0)

    def test_negative_halves_round_away_from_zero(self):
        self.assertEqual(round_half_up(-0.125), -0.13)

    def test_reports_share_one_helper(self):
        for module in (daily, weekly, monthly):
            self.assertFalse(hasattr(module, "_round_half_up"), module.__name__)


if __name__ == "__main__":
    unittest.main()
