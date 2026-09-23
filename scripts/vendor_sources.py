#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point; use scripts/vendor/manage.py for new invocations."""

if __package__:
    from .vendor.manage import main
else:
    from vendor.manage import main

if __name__ == "__main__":
    raise SystemExit(main())
