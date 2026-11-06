# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Length units and conversions between them."""

# Metres per unit.
FACTORS = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
    "mi": 1609.344,
}


class UnknownUnitError(ValueError):
    """A unit that is not in FACTORS."""


def convert(value, from_unit, to_unit):
    """Return value, given in from_unit, converted to to_unit."""
    for unit in (from_unit, to_unit):
        if unit not in FACTORS:
            raise UnknownUnitError(f"unknown unit: {unit}")
    return value * FACTORS[from_unit] / FACTORS[to_unit]


def format_result(value, unit):
    """Return value and its unit as printed, with two decimal places."""
    return f"{value:.2f} {unit}"
