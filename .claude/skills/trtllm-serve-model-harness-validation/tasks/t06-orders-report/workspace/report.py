# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Revenue per category from an orders CSV file.

usage: python3 report.py [orders.csv]
"""

import csv
import sys


def load_orders(stream):
    """Read orders from a CSV stream with the columns order_id, category, quantity, unit_price."""
    orders = []
    for row in csv.DictReader(stream):
        orders.append(
            {
                "category": row["category"],
                "quantity": int(row["quantity"]),
                "unit_price": float(row["unit_price"]),
            }
        )
    return orders


def revenue_by_category(orders):
    """Return {category: revenue}, the revenue being the sum of quantity times unit_price."""
    totals = {}
    for order in orders:
        revenue = order["quantity"] * order["unit_price"]
        totals[order["category"]] = totals.get(order["category"], 0.0) + revenue
    return totals


def format_report(totals):
    """Return the report: one line per category in name order, then the grand total."""
    lines = [f"{name:<10}{totals[name]:>12.2f}" for name in sorted(totals)]
    lines.append(f"{'total':<10}{sum(totals.values()):>12.2f}")
    return "\n".join(lines)


def main(argv):
    path = argv[1] if len(argv) > 1 else "orders.csv"
    with open(path, newline="") as fh:
        orders = load_orders(fh)
    print(format_report(revenue_by_category(orders)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
