# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Layout of the live-row word (no CuTe DSL import, so the engine can read it on any platform).

Bits 0-15 hold the real requests R of a forward, bits 16-31 its requests P.
"""

REQUEST_BITS = 16
REQUEST_MASK = (1 << REQUEST_BITS) - 1
