# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Daily summary of meter readings."""

from decimal import ROUND_HALF_UP, Decimal


def _round_half_up(value, places=2):
    """Round value to the given number of decimal places, halves away from zero."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def daily_summary(readings):
    """Return the total and the mean of one day's readings, rounded to two places."""
    total = sum(readings)
    mean = total / len(readings) if readings else 0.0
    return {"total": _round_half_up(total), "mean": _round_half_up(mean)}
