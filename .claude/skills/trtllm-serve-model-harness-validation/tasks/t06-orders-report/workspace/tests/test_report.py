# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import unittest

from report import format_report, load_orders, revenue_by_category

SAMPLE = (
    "order_id,category,quantity,unit_price\n"
    "A00001,tools,2,10.00\n"
    "A00002,garden,1,4.50\n"
    "A00003,tools,n/a,3.00\n"
    "A00004,garden,3,\n"
    "A00005,tools,1\n"
    'A00006,toys,2,"4,50"\n'
    "A00007,tools,1,2.50\n"
)


class LoadOrdersTest(unittest.TestCase):
    def test_parsable_rows_are_kept_in_order(self):
        orders, _ = load_orders(io.StringIO(SAMPLE))
        self.assertEqual(
            [(order["category"], order["quantity"], order["unit_price"]) for order in orders],
            [("tools", 2, 10.0), ("garden", 1, 4.5), ("tools", 1, 2.5)],
        )

    def test_unparsable_rows_are_counted(self):
        _, skipped = load_orders(io.StringIO(SAMPLE))
        self.assertEqual(skipped, 4)


class ReportTest(unittest.TestCase):
    def test_revenue_by_category(self):
        orders = [
            {"category": "tools", "quantity": 2, "unit_price": 10.0},
            {"category": "garden", "quantity": 1, "unit_price": 4.5},
            {"category": "tools", "quantity": 1, "unit_price": 2.5},
        ]
        self.assertEqual(revenue_by_category(orders), {"tools": 22.5, "garden": 4.5})

    def test_report_lists_categories_then_the_total(self):
        lines = format_report({"tools": 22.5, "garden": 4.5}).splitlines()
        self.assertEqual(
            [line.split() for line in lines[:3]],
            [["garden", "4.50"], ["tools", "22.50"], ["total", "27.00"]],
        )

    def test_report_ends_with_the_skipped_rows(self):
        report = format_report({"tools": 22.5}, skipped=4)
        self.assertEqual(report.splitlines()[-1], "skipped rows: 4")


if __name__ == "__main__":
    unittest.main()
