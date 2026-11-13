# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tp4c1: M3 low-latency kernels, runtime, presets and micro-net harness (no vLLM import outside harness)."""

__version__ = "0.1.0"

from . import public_build as _public_build

_public_build.install()  # no-op unless this is an exported public tree
