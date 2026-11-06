# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write orders.csv: 2,500 synthetic orders, a few of them malformed on purpose."""

import csv

ROWS = 2500
OUTPUT = "orders.csv"
CATEGORIES = ["books", "garden", "kitchen", "tools", "toys"]
PRICES = {"books": 12.5, "garden": 7.25, "kitchen": 18.0, "tools": 23.75, "toys": 9.5}
# Data rows (the first row after the header is 1) that are broken, and how.
BROKEN = {
    317: "quantity n/a",
    804: "empty unit_price",
    1288: "quantity in words",
    1950: "decimal comma",
    2391: "missing column",
}


def row(number):
    """Return the fields of data row number."""
    category = CATEGORIES[(number * 7 + number // 5) % len(CATEGORIES)]
    quantity = 1 + (number * 13) % 9
    unit_price = PRICES[category] + (number % 4) * 0.25
    fields = [f"A{number:05d}", category, str(quantity), f"{unit_price:.2f}"]
    broken = BROKEN.get(number)
    if broken == "quantity n/a":
        fields[2] = "n/a"
    elif broken == "empty unit_price":
        fields[3] = ""
    elif broken == "quantity in words":
        fields[2] = "three"
    elif broken == "decimal comma":
        fields[3] = fields[3].replace(".", ",")
    elif broken == "missing column":
        fields = fields[:3]
    return fields


def main():
    with open(OUTPUT, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["order_id", "category", "quantity", "unit_price"])
        for number in range(1, ROWS + 1):
            writer.writerow(row(number))
    print(f"wrote {OUTPUT}: {ROWS} orders")


if __name__ == "__main__":
    main()
