# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense layers' per-head Gemma RMSNorm + partial NeoX RoPE of the packed [Q | K | V], into a fresh FP8 copy."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)


class M3QkNormRopeFp8(OpWrapper):
    """``fused_qk_norm_rope`` written out of place in FP8 E4M3: Q and K heads normed and roped in fp32 and rounded
    once to E4M3 (saturating at +/-448), V cast, ``qkv`` left untouched. Same arguments as the catalog's in-place
    entry.

    The dense layers (0-2) feed the FP8 main KV cache with it: on every step of an FP8 cache, and on the decode steps
    of a hybrid NVFP4 cache (whose dense layers sit in its FP8 pool).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token, tp4 heads (16 q, 1 kv)", spec=dict(m=1, q=16, kv=1)),
        Cell(why="a 48-row decode verify", spec=dict(m=48, q=16, kv=1)),
        Cell(why="a prefill chunk on an FP8 cache", spec=dict(m=2048, q=16, kv=1)),
    )

    note = """
    `reference` is the in-place kernel followed by a cast to E4M3. V matches it bit for bit; Q and K round once from
    fp32 where the composition rounds through bf16 first, so they are within one E4M3 step of it.
    """

    def __call__(
        self,
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        is_qk_norm,
        use_gemma,
        use_mrope,
        mrope_section1,
        mrope_section2,
    ) -> torch.Tensor:
        return torch.ops.trtllm.fused_qk_norm_rope_to_fp8(
            qkv,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            rotary_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            factor,
            low,
            high,
            attention_factor,
            is_qk_norm,
            use_gemma,
            use_mrope,
            mrope_section1,
            mrope_section2,
        )

    def reference(
        self,
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        rotary_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        is_qk_norm,
        use_gemma,
        use_mrope,
        mrope_section1,
        mrope_section2,
    ) -> torch.Tensor:
        """The catalog's in-place kernel on a copy, then the E4M3 cast (V exact; Q and K within one E4M3 step)."""
        out = fused_qk_norm_rope(
            qkv.clone(),
            num_heads_q,
            num_heads_k,
            num_heads_v,
            head_dim,
            rotary_dim,
            eps,
            q_weight,
            k_weight,
            base,
            is_neox,
            position_ids,
            factor,
            low,
            high,
            attention_factor,
            is_qk_norm,
            use_gemma,
            use_mrope,
            mrope_section1,
            mrope_section2,
        )
        return out.to(torch.float8_e4m3fn)


m3_qk_norm_rope_fp8 = M3QkNormRopeFp8()
