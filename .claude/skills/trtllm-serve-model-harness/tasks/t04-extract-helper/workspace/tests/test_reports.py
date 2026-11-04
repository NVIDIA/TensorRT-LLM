# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

from reports.daily import daily_summary
from reports.monthly import monthly_summary
from reports.weekly import weekly_summary


class ReportsTest(unittest.TestCase):
    def test_daily_summary_rounds_halves_up(self):
        self.assertEqual(daily_summary([1.25, 2.5, 0.375]), {"total": 4.13, "mean": 1.38})

    def test_daily_summary_without_readings(self):
        self.assertEqual(daily_summary([]), {"total": 0.0, "mean": 0.0})

    def test_weekly_summary(self):
        week = [3.5, 4.125, 2.0, 0.5, 1.0, 0.0, 0.0]
        self.assertEqual(weekly_summary(week), {"total": 11.13, "busiest_day": 1})

    def test_monthly_summary(self):
        month = monthly_summary([10.5, 12.25, 9.125, 11.0], rate=0.5)
        self.assertEqual(month, {"total": 42.88, "cost": 21.44})


if __name__ == "__main__":
    unittest.main()
