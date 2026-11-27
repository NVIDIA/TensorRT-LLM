# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`ctx_k_b`: the context rows' absorbed K-B, on the prefill kernel whose epilogue writes the
e4m3 q-base straight into the attention's fp8 query where the rope kernel takes the layer
(it then skips those groups), else the bf16 K-B (`absorbed_bmm`) into the bf16 fused query.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count
from ..attention.ctx_rope import CtxQuery
from . import cute_dsl_absorbed_bmm as _absorbed_bmm_op
from .absorbed_bmm import AbsorbedBmmHandle, absorbed_bmm
from .cute_dsl_absorbed_bmm import cute_dsl_absorbed_bmm


@dataclass(frozen=True)
class CtxKbHandle:
    """`fp8` the prefill K-B writing the e4m3 q-base (None: the bf16 K-B on every row),
    `bf16` the plain K-B's handle (its decode cells keep the row counts they serve on the
    bf16 K-B), and the model's MLA dims."""

    fp8: Any
    bf16: AbsorbedBmmHandle
    heads: int
    nope: int
    rope_dim: int
    kv_lora: int
    count: Count


class CtxKb(OpWrapper):
    """The context rows' K-B of `q` (`[tc, heads * (nope + rope)]`) into a new `CtxQuery`.
    `quant_q` is the step's fp8 query buffer where the plan holds one (None: a new one);
    `on_kernel` whether the layer's rope takes the context kernel. On the fp8 q-base kernel
    (counted `ctx_kb_fp8`) where `on_kernel`, it built and the row count is neither a decode
    cell's nor above the prefill kernel's: the bf16 latent half of `fused_q` (129 MB at 8k
    rows) is then neither written nor re-read; same bytes into the attention (FP32 -> BF16 ->
    E4M3, nearest, on both paths). Else the bf16 K-B into `fused_q` (`absorbed_bmm` at the
    Target's `ph`)."""

    def __call__(
        self,
        ph,
        q: torch.Tensor,
        k_b: torch.Tensor,
        k_b_nk: torch.Tensor,
        quant_q: torch.Tensor | None,
        on_kernel: bool,
        handle: CtxKbHandle,
    ) -> CtxQuery:
        h = handle
        tc, dev, dt = int(q.shape[0]), q.device, q.dtype
        lat_dim = h.kv_lora + h.rope_dim
        q_nope, q_pe = torch.split(
            torch.reshape(q, [tc, h.heads, h.nope + h.rope_dim]), [h.nope, h.rope_dim], -1
        )
        fused_q = torch.empty([tc, h.heads, lat_dim], dtype=dt, device=dev)
        if quant_q is None:
            quant_q = torch.empty([tc, h.heads, lat_dim], dtype=torch.float8_e4m3fn, device=dev)
        from_kb = (
            on_kernel
            and h.fp8 is not None
            and tc not in h.bf16.cells
            and tc <= _absorbed_bmm_op.PREFILL_MAX_ROWS
        )
        if from_kb:
            h.count("ctx_kb_fp8", True)
            q_base = torch.split(quant_q.view(torch.uint8), [h.kv_lora, h.rope_dim], -1)[0]
            cute_dsl_absorbed_bmm(
                torch.transpose(q_nope, 0, 1), k_b_nk, torch.transpose(q_base, 0, 1), h.fp8
            )
        else:
            fq_nope = fused_q[..., : h.kv_lora]
            absorbed_bmm(
                ph,
                torch.transpose(q_nope, 0, 1),
                k_b,
                k_b_nk,
                torch.transpose(fq_nope, 0, 1),
                h.bf16,
            )
        return CtxQuery(fused_q, q_pe, quant_q, on_kernel, from_kb)

    def reference(
        self,
        ph,
        q: torch.Tensor,
        k_b: torch.Tensor,
        k_b_nk: torch.Tensor,
        quant_q: torch.Tensor | None,
        on_kernel: bool,
        handle: CtxKbHandle,
    ) -> torch.Tensor:
        """The q-base `q_nope @ k_b` per head, `[tc, heads, kv_lora]` in fp32 (both kernels
        round it once: to bf16, or to bf16 then e4m3); `quant_q` and `on_kernel` only pick
        where it is written."""
        h = handle
        tc = int(q.shape[0])
        q_nope = torch.reshape(q, [tc, h.heads, h.nope + h.rope_dim])[..., : h.nope]
        return torch.einsum("rhn,hnk->rhk", q_nope.float(), k_b.float())


ctx_k_b = CtxKb()
