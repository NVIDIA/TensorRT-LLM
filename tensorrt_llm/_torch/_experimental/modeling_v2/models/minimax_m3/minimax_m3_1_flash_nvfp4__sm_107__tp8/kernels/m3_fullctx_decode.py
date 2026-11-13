# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 full-context FP8 decode attention on SM107: ``fullctx_decode``.

The dense layers (0-2) attend every token of their FP8 E4M3 (unscaled) paged KV cache. On SM 10.7 the stock call
(flashinfer trtllm-gen, ``minimax_m3_trtllm_gen_dense_decode``) caps its split at 16 cluster CTAs per (request, KV
head): at TP8 (8 query heads and 1 KV head per rank) one request runs on 16 CTAs of 212.

This runs the same arithmetic -- FP8 Q (the same E4M3 rounding of q), FP8 K/V, P quantized to E4M3 per split -- on
the vendored PrimTS decode kernel (``attention/backends/prims_ts``, SM 10.7 through the sm_100f family) at a pinned
split-KV recipe, which fans one request out over many more CTAs. The split partials are merged by
``torch.ops.trtllm.minimax_m3_sm107_tp2_fullctx_reduce`` (``cpp/tensorrt_llm/kernels/minimaxM3Sm107Tp2/
fullCtxReduce.cu``, head-count generic, used as is): unnormalized fp32 accumulation over the active splits, one final
division, one BF16 rounding, the merge form of stock trtllm-gen. Only the split changes against stock
(modeling's ``FULLCTX_DECODE_RECIPES``).

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
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.fullctx import (
    device_index as _index,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.attention.fullctx import (
    kv_page_views as kv_page_views,
)

# The TP8 per-rank geometry: 8 query heads over 1 KV head.
HEADS = 8
KV_HEADS = 1  # capacity: fixed: the tp8 KV heads per rank
HEAD_DIM = 128  # capacity: fixed: the checkpoint's head dim
PAGE = 128
# The per-request K/V length PrimTS's config selection sizes its policy for (Q/KV tiles, launch mode); the pinned
# recipes assume it. The compiled capacity (``max_kv_len``) may be larger.
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
        mxfp8: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> bool:
        """Decodes into output; with mxfp8 (E4M3 rows, UE8M0 scales) and the TP8 merge, the merge also writes
        the MXFP8 copy of output. Returns whether it did."""
        self._run_primts(q, k_pages, v_pages, block_table, seq_lens, sm_scale)
        ws = self.workspace
        if self.own_reducer:
            reduce = _reduce_op()
            args = (
                ws.partial_o,
                ws.partial_stats,
                seq_lens,
                output,
                self.qlen,
                self.q_tokens_per_cta,
                self.tile_kv,
                self.insts_kv,
                self.causal,
                self.pdl,
            )
            if mxfp8 is not None and reduce is _tp8_reduce_op():
                reduce(*args, mxfp8[0], mxfp8[1])
                return True
            reduce(*args)
            return False
        output.view(self.staging.shape).copy_(self.staging)
        return False


# The split-KV merge runs the TP8 copy of the BF16 reducer when the op library carries it (the same bits).
TP8_REDUCER = True


def _reduce_op():
    """The BF16 split-KV merge: the TP8 copy (one row per CTA, its dependent released right after its wait) when the
    TP8 op library carries it, else the image's TP2 op; the same per-row arithmetic, so the same bits."""
    ops = torch.ops.trtllm
    if TP8_REDUCER and hasattr(ops, "minimax_m3_sm107tp8_fullctx_reduce"):
        return ops.minimax_m3_sm107tp8_fullctx_reduce
    return ops.minimax_m3_sm107_tp2_fullctx_reduce


def _tp8_reduce_op():
    return getattr(torch.ops.trtllm, "minimax_m3_sm107tp8_fullctx_reduce", None)


def emits_mxfp8() -> bool:
    """Whether the merge run uses can also write the o_proj MXFP8 hand-off (the TP8 copy of the merge)."""
    return _tp8_reduce_op() is not None and _reduce_op() is _tp8_reduce_op()


def has_own_reducer() -> bool:
    """Whether this build carries a BF16 split-KV merge op (the TP8 copy or the image's TP2 op)."""
    ops = torch.ops.trtllm
    return hasattr(ops, "minimax_m3_sm107tp8_fullctx_reduce") or hasattr(
        ops, "minimax_m3_sm107_tp2_fullctx_reduce"
    )


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
