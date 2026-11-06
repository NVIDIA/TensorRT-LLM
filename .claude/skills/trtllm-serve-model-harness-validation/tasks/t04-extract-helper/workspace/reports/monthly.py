# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monthly summary built from weekly totals."""

from decimal import ROUND_HALF_UP, Decimal


def _round_half_up(value, places=2):
    """Round value to the given number of decimal places, halves away from zero."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def monthly_summary(weekly_totals, rate):
    """Return the month's total and its cost at rate per unit, both rounded to two places."""
    total = sum(weekly_totals)
    return {"total": _round_half_up(total), "cost": _round_half_up(total * rate)}
