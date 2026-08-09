# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 Attention Residual fused decoder-layer op.

Wraps the sm_100 ``attn_res_fwd`` (packed forward) kernel from
``exisiting_optimization_work/Attention_residual`` as an in-tree
decoder-layer fused op. K3 sets ``attn_res_block_size=12`` and uses this
op in two positions per decoder layer (before self-attention and after
the sub-block that produces the running ``prefix_sum``), plus once at
the top of the model.

The kernel is Blackwell (sm_100a) only. On other devices the module
routes to a pure-torch reference implementation identical to
``exisiting_optimization_work/Attention_residual/tests/util/attn_res_ref.py``
and to HF ``modeling_kimi._apply_attn_res``.
"""

from .kimi_k3_attn_res import (
    KimiK3AttnResidualKernelPath,
    KimiK3AttnResidualOp,
    apply_attn_res_reference,
    attn_res_fwd_chunked_reference,
)

__all__ = [
    "KimiK3AttnResidualKernelPath",
    "KimiK3AttnResidualOp",
    "apply_attn_res_reference",
    "attn_res_fwd_chunked_reference",
]
