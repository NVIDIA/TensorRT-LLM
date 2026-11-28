# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ._key_padding import KeyPadding
from .prims_routed_attention import prims_routed_attention
from .sol_predictor import SOL_BLOCK_SIZE, sol_predictor


def sol_attention_ineligible_reason(q, k, v) -> str | None:
    if not q.is_cuda:
        return "q is not a CUDA tensor"
    if q.ndim != 4:
        return f"q must be 4-D [B, S, H, D], got ndim={q.ndim}"
    if q.shape[-1] != 128:
        return f"head_dim must be 128, got {q.shape[-1]}"
    if q.dtype != torch.bfloat16:
        return f"dtype must be bfloat16, got {q.dtype}"
    major, minor = torch.cuda.get_device_capability(q.device)
    if f"sm_{major}{minor}" not in SolAttention.ARCHS:
        supported = ", ".join(arch.name.replace("_", "") for arch in sorted(SolAttention.ARCHS))
        return f"no Sol-Attn kernel for SM{major * 10 + minor}; supported: {supported}"
    if k.shape != q.shape or v.shape != q.shape:
        return f"k/v shape must match q {tuple(q.shape)}"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return f"k/v dtype must match q {q.dtype}"
    return None


class SolAttention(OpWrapper):
    """Noncausal attention with exact 64-token tiles and pooled K/V for the other tiles."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell("diagonal threshold and partial final tile", dict(tokens=321, thresh_type="diag")),
        Cell("exact score variance", dict(tokens=384, thresh_type="exact")),
    )

    def key_padding(self) -> KeyPadding:
        return KeyPadding.NONE

    @torch.compiler.disable
    def __call__(self, q, k, v, *, tau: float, thresh_type: str, sm_scale: float):
        from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.sol_attn import sol_attn

        return sol_attn(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            tau=tau,
            thresh_type=thresh_type,
            scale=sm_scale,
        )

    def reference(self, q, k, v, *, tau: float, thresh_type: str, sm_scale: float):
        bits, keys, values = sol_predictor.reference(
            q, k, v, tau=tau, sm_scale=sm_scale, thresh_type=thresh_type
        )
        return prims_routed_attention.reference(
            q,
            k,
            v,
            plans={},
            q_block_size=SOL_BLOCK_SIZE,
            kv_block_size=SOL_BLOCK_SIZE,
            exact_block_bits=bits,
            k_summary=keys,
            v_summary=values,
            sm_scale=sm_scale,
        )

    def is_valid(self, q, k, v, *, tau: float, thresh_type: str, sm_scale: float):
        assert sol_attention_ineligible_reason(q, k, v) is None
        assert q.shape[1] > 0 and k.device == q.device == v.device
        assert thresh_type in ("exact", "diag")
        assert math.isfinite(tau) and math.isfinite(sm_scale)


sol_attention = SolAttention()
