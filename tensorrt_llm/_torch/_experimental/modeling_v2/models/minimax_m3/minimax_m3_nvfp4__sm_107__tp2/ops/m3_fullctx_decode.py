# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 full-context FP8 decode attention on Rubin: ``fullctx_decode``.

The dense layers (0-2) attend every token of their FP8 E4M3 (unscaled) paged KV cache. On SM 10.7 the stock call
(flashinfer trtllm-gen, ``minimax_m3_trtllm_gen_dense_decode``) caps its split at 16 cluster CTAs per (request, KV
head): at TP2 (2 KV heads) one request runs on 32 CTAs of 212 and takes 28-30 us for 85 MB at 165k tokens.

This runs the same arithmetic -- FP8 Q (the same E4M3 rounding of q), FP8 K/V, P quantized to E4M3 per split -- on
the vendored PrimTS decode kernel (``attention/backends/prims_ts``, SM 10.7 through the sm_100f family) at a pinned
split-KV recipe, which fans one request out over 96-144 CTAs. The split partials are merged by
``torch.ops.trtllm.minimax_m3_sm107_tp2_fullctx_reduce`` (``cpp/tensorrt_llm/kernels/minimaxM3Sm107Tp2/
fullCtxReduce.cu``): unnormalized fp32 accumulation over the active splits, one final division, one BF16 rounding,
the merge form of stock trtllm-gen. Only the split changes against stock (modeling's ``FULLCTX_DECODE_RECIPES``).

A plan is compiled per (device, query rows per request, requests) by ``prebuild`` before the executor's warmup; the
forward only reads plans (``plan_for``). The plan is built from PrimTS's own config, compile and workspace helpers at
an explicit recipe; the PrimTS wrapper's planning policy is not used, so no PrimTS state is replaced. With
``own_reducer`` off the PrimTS reducer merges into FP16 and the result is cast to BF16.
"""

from __future__ import annotations

import threading
from typing import Dict, Mapping, Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.fullctx import (
    FullCtxGeometry,
    FullCtxPlanBase,
    check_layout,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.fullctx import (
    device_index as _index,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.fullctx import (
    kv_page_views as kv_page_views,
)

HEADS = 32
KV_HEADS = 2  # capacity: fixed: the tp2 KV heads per rank
HEAD_DIM = 128  # capacity: fixed: the checkpoint's head dim
PAGE = 128
# The per-request K/V length PrimTS's config selection sizes its policy for (Q/KV tiles, launch mode); the pinned
# recipes were measured with it. The compiled capacity (``max_kv_len``) may be larger.
POLICY_KV_LEN = (
    1 << 20
)  # capacity: fixed: the PrimTS policy's K/V length bound (the plans' compile shape)
HEADS_PER_KV = HEADS // KV_HEADS  # capacity: fixed: the tp2 query heads per KV head

_plans: Dict[Tuple[int, int, int, bool], "FullCtxPlan"] = {}
_lock = threading.Lock()


class FullCtxPlan(FullCtxPlanBase):
    """One compiled PrimTS split-KV decode at a pinned recipe, its workspace and its FP16 staging output."""

    GEOMETRY = FullCtxGeometry(HEADS, KV_HEADS, HEAD_DIM, PAGE, POLICY_KV_LEN, HEADS_PER_KV)

    def run(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        sm_scale: float,
        output: torch.Tensor,
    ) -> None:
        """Run the pinned decode and write its reduced attention output."""
        self._run_primts(q, k_pages, v_pages, block_table, seq_lens, sm_scale)
        if self.own_reducer:
            torch.ops.trtllm.minimax_m3_sm107_tp2_fullctx_reduce(
                self.workspace.partial_o,
                self.workspace.partial_stats,
                seq_lens,
                output,
                self.qlen,
                self.q_tokens_per_cta,
                self.tile_kv,
                self.insts_kv,
                self.causal,
                self.pdl,
            )
            return
        output.view(self.staging.shape).copy_(self.staging)


def _check_layout(cfg, qlen: int, splits: int) -> None:
    return check_layout(cfg, qlen, splits, heads_per_kv=HEADS_PER_KV)


def has_own_reducer() -> bool:
    """Require the BF16 split-KV merge op supplied by this target build."""
    torch.ops.trtllm.minimax_m3_sm107_tp2_fullctx_reduce
    return True


def plan_for(
    device: torch.device, qlen: int, nreq: int, own_reducer: bool = True
) -> Optional[FullCtxPlan]:
    """The prebuilt plan for this shape, or None."""
    return _plans.get((_index(device), qlen, nreq, own_reducer))


def prebuild(
    device: torch.device,
    recipes: Mapping[Tuple[int, int], Mapping],
    max_kv_len: int,
    own_reducer: bool,
    pdl: bool,
) -> int:
    """Compile the plan of every (query rows per request, requests) in ``recipes``. Call before warmup."""
    built = 0
    for (qlen, nreq), recipe in recipes.items():
        key = (_index(device), qlen, nreq, own_reducer)
        with _lock:
            if key not in _plans:
                _plans[key] = FullCtxPlan(device, qlen, nreq, max_kv_len, recipe, own_reducer, pdl)
                built += 1
    return built
