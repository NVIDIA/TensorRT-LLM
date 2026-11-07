# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command line of the length converter."""

import argparse
import sys

from convert.units import UnknownUnitError, convert, format_result


def build_parser():
    """Return the argument parser."""
    parser = argparse.ArgumentParser(prog="convert", description="Convert a length between units.")
    parser.add_argument("value", type=float, help="the length to convert")
    parser.add_argument("from_unit", help="the unit VALUE is given in")
    parser.add_argument("to_unit", help="the unit to convert VALUE to")
    return parser


def main(argv=None):
    """Run the converter; return the exit status."""
    args = build_parser().parse_args(argv)
    try:
        result = convert(args.value, args.from_unit, args.to_unit)
    except UnknownUnitError as error:
        print(f"convert: {error}", file=sys.stderr)
        return 1
    print(format_result(result, args.to_unit))
    return 0
