#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the original PrimTS promotion command."""

import sys

if __package__:
    from .vendor.promote import main
else:
    from vendor.promote import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] == "promote":
        arguments[1:1] = [
            "--vendor",
            "flashinfer-prims-ts",
            "--upstream-repo",
            "flashinfer-ai/flashinfer",
            "--canonical-branch",
            "trtllm-prims-ts-dev",
            "--legacy-prims-ts",
        ]
    raise SystemExit(main(arguments))
