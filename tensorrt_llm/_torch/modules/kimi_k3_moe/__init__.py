# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 sparse MoE in-tree module.

Ships the promoted ``KimiK3SparseMoeBlock`` and its supporting pieces
(``KimiK3MoEGate``, latent projections, shared experts, MXFP4-packed
routed expert bank, native TRTLLM-Gen SiTU dispatch). Structurally
mirrors HF ``KimiSparseMoeBlock`` at ``modeling_kimi.py:806-918``.

Two kernel paths coexist under one module class:

* ``use_fused_cubin=False`` — Python fallback with MXFP4 group-32
  routed expert weights, dequantized to canonical fp32 on demand.
  Byte-exact HF parity under random weights.
* ``use_fused_cubin=True`` — native in-tree
  ``torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner`` invocation
  (``act_type=SiTu``) on checkpoint-derived MXFP4 weights shared with
  the fallback bank. Routing goes through the op's
  ``topk_weights``/``topk_ids`` bypass fed by the real K3 gate.
"""

from .kimi_k3_moe_block import (
    KimiK3RoutedExpertBank,
    KimiK3SparseMoeBlock,
    MoEBlockProvenance,
    copy_hf_moe_block_weights,
)
from .kimi_k3_moe_gate import KimiK3MoEGate, copy_hf_moe_gate_weights

__all__ = [
    "KimiK3MoEGate",
    "KimiK3RoutedExpertBank",
    "KimiK3SparseMoeBlock",
    "MoEBlockProvenance",
    "copy_hf_moe_gate_weights",
    "copy_hf_moe_block_weights",
]
