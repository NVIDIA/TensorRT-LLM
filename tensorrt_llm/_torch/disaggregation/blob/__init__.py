# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The blob backend: KV rows shared through a byte store.

Only the private driver boundary lives here so far. Nothing in this package imports the public
backend contract or an optional store SDK at import time.
"""
