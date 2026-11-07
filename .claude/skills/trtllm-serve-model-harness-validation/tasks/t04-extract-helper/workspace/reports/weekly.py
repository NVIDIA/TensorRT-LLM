# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weekly summary built from daily totals."""

from decimal import ROUND_HALF_UP, Decimal


def _round_half_up(value, places=2):
    """Round value to the given number of decimal places, halves away from zero."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def weekly_summary(daily_totals):
    """Return the week's total, rounded to two places, and the index of its busiest day."""
    total = sum(daily_totals)
    busiest = None
    for index, value in enumerate(daily_totals):
        if busiest is None or value > daily_totals[busiest]:
            busiest = index
    return {"total": _round_half_up(total), "busiest_day": busiest}
