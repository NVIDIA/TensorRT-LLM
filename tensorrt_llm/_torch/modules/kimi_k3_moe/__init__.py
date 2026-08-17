# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 sparse MoE runtime pieces.

Ships the routing gate (``KimiK3MoEGate``) and the shared MLP /
RMSNorm building blocks (``_mlp``) used by the serving runtime
(``KimiK3MoERuntime`` in ``modeling_kimi_linear.py``). The test-only
HF-parity reference block (``KimiK3SparseMoeBlock`` and its MXFP4 /
kernel helpers) lives with its test at
``tests/unittest/_torch/modules/moe/kimi_k3_ref_moe/``.
"""

from .kimi_k3_moe_gate import KimiK3MoEGate, copy_hf_moe_gate_weights

__all__ = [
    "KimiK3MoEGate",
    "copy_hf_moe_gate_weights",
]
