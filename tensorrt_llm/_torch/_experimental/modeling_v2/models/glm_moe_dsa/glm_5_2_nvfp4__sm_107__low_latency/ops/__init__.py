# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entries only this target calls, one module per op under a category package."""

# Register before a target-local wrapper invokes a C++ operator.
from . import _cpp_fakes  # noqa: F401
