# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels vendored from the low-latency reference TP4 layer chain (the whole-chain port), family level.

``moe_engine_kernel.py`` / ``moe_engine_host.py``: the fused MoE engine (``kernels/moe/engine/fused``);
``tp4_moe_engine``: how a target builds it on its own m3_moe plan: width 768 at TP4, 1536 at TP2 (TP2WC_MOE_ENGINE;
``host_module`` gives the host per width)."""
