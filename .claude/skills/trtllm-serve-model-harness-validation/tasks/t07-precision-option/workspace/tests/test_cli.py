# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import unittest

from convert.cli import main


def run_cli(*args):
    """Run the command line with args; return its exit status, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            status = main(list(args))
        except SystemExit as stop:
            status = stop.code
    return status, out.getvalue(), err.getvalue()


class CliTest(unittest.TestCase):
    def test_prints_two_decimal_places(self):
        self.assertEqual(run_cli("10", "km", "m"), (0, "10000.00 m\n", ""))

    def test_unknown_unit(self):
        status, out, err = run_cli("1", "km", "league")
        self.assertEqual((status, out), (1, ""))
        self.assertIn("unknown unit: league", err)

    def test_missing_argument_is_a_usage_error(self):
        status, _, err = run_cli("1", "km")
        self.assertEqual(status, 2)
        self.assertIn("usage:", err)


if __name__ == "__main__":
    unittest.main()
