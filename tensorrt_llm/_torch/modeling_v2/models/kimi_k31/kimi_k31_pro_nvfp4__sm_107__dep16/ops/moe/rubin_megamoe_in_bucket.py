# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts through MegaMoE on SM107, read from the bucket its producers filled.

``rubin_megamoe_in_bucket`` launches the MegaMoE kernel on a bucket (``megamoe_bucket``) whose
rows the routing and the quantization wrote in place (``rubin_route_topk_out``,
``rubin_megamoe_quantize``): the launch is the kernel alone, where ``rubin_megamoe`` first copies
the rows in and writes the bucket's tail. Same kernel, same bucket, same bits.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper

from ...kernels.moe.rubin_megamoe import SM107_RUNGS, MegaMoeBucket, MegaMoeWorkspace
from ...kernels.moe.rubin_megamoe import run_in_bucket as _launch
from .rubin_megamoe import rubin_megamoe


def _cell(why: str, **spec) -> Cell:
    return Cell(why=why, spec=dict(ranks=4, local=224, rungs=SM107_RUNGS, **spec))


class RubinMegamoeInBucket(OpWrapper):
    """This rank's rows of one chunk through the experts of every rank, read from ``bucket``.

    ``bucket``: ``megamoe_bucket(ws, chunk_max_tokens, num_tokens)``, filled for this chunk: rows
    ``[0, num_tokens)`` of its ``activation`` / ``activation_sf`` hold the NVFP4 values and their
    linear scales, the same rows of its ``topk_idx`` / ``topk_weights`` the routing, and every row
    past them ids -1 and weights 0. ``fc1_w`` ... ``fc1_norm_const``, ``ws`` and ``num_tokens`` as
    ``rubin_megamoe`` takes them. Returns BF16 ``[num_tokens, hidden]``, a view of the bucket's
    output that the next launch on the bucket overwrites: the bits ``rubin_megamoe`` returns for
    the same rows.
    """

    # Multi-rank: no architecture until its cells pass across ranks on SM107. EP4 over one tray
    # with all 896 experts (224 per rank), so the routing kernel serves the rows.
    CELLS: tuple[Cell, ...] = (
        _cell(
            "DEP16 generation b12 k7 on one tray: 96 rows per rank written into bucket 128",
            rows=(96,),
        ),  # fmt: skip
        _cell(
            "96 rows after 120 in bucket 128: the quantization empties the longer tail",
            rows=(120, 96),
        ),  # fmt: skip
        _cell("a full bucket: 128 rows per rank", rows=(128,)),
        _cell(
            "uneven ranks, one without rows (it stages its empty chunk, the others do not)",
            counts=(96, 0, 40, 1),
        ),  # fmt: skip
    )

    note = """
    The kernel reads the bucket's buffers as they are and stages nothing: rows past
    ``num_tokens`` must route nowhere (ids -1, weights 0), which ``rubin_megamoe_quantize``
    writes on every call, so a CUDA graph captured at one row count replays right after any
    other. Every rank of the group launches every chunk on the same bucket in the same order,
    whichever of the two entries it uses.
    """

    def __call__(
        self,
        bucket: MegaMoeBucket,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc31_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        ws: MegaMoeWorkspace,
        num_tokens: int,
    ) -> torch.Tensor:
        return _launch(
            ws,
            bucket,
            (fc1_w, fc1_s, fc2_w, fc2_s, fc31_alpha, fc2_alpha, fc1_norm_const),
            num_tokens,
        )

    def reference(
        self,
        bucket: MegaMoeBucket,
        fc1_w: Sequence[torch.Tensor],
        fc1_s: Sequence[torch.Tensor],
        fc2_w: Sequence[torch.Tensor],
        fc2_s: Sequence[torch.Tensor],
        fc31_alpha: Sequence[torch.Tensor],
        fc2_alpha: Sequence[torch.Tensor],
        fc1_norm_const: Sequence[torch.Tensor],
        ws: MegaMoeWorkspace,
        num_tokens: int,
    ) -> torch.Tensor:
        """This rank's rows, FP32: ``rubin_megamoe``'s reference over the rows ``bucket`` holds (a
        collective: ``bucket`` is this rank's, the experts every rank's in rank order)."""
        r, n = ws.ep_rank, num_tokens
        return rubin_megamoe.reference(
            {r: bucket.activation[:n]},
            {r: bucket.activation_sf[:n]},
            {r: bucket.topk_idx[:n]},
            {r: bucket.topk_weights[:n, : ws.top_k]},
            fc1_w,
            fc1_s,
            fc2_w,
            fc2_s,
            fc31_alpha,
            fc2_alpha,
            fc1_norm_const,
            ws,
            n,
            n,
        )

    def is_valid(
        self,
        bucket: MegaMoeBucket,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc31_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        ws: MegaMoeWorkspace,
        num_tokens: int,
    ) -> None:
        """The kernel reads the bucket as it is: a bucket of another workspace, more rows than it
        holds, or a routing tail that still routes somewhere is read wrong."""
        assert any(b is bucket for b in ws.buckets), "a bucket of this workspace (megamoe_bucket)"
        assert 0 <= num_tokens <= bucket.max_tokens, (num_tokens, bucket.max_tokens)
        assert bool((bucket.topk_idx[num_tokens:] == -1).all()), "ids past the rows must be -1"
        assert not bucket.topk_weights[num_tokens:, : ws.top_k].any(), "weights past the rows: 0"
        e, i, h = ws.num_local_experts, ws.intermediate, ws.hidden
        assert tuple(fc1_w.shape) == (e, 2 * i, h // 2), tuple(fc1_w.shape)
        assert tuple(fc2_w.shape) == (e, h, i // 2), tuple(fc2_w.shape)
        assert fc1_s.shape[0] == fc2_s.shape[0] == e, (fc1_s.shape, fc2_s.shape)
        for name, t in (("fc31_alpha", fc31_alpha), ("fc2_alpha", fc2_alpha),
                        ("fc1_norm_const", fc1_norm_const)):  # fmt: skip
            assert tuple(t.shape) == (e,) and t.dtype == torch.float32, name
            assert t.is_contiguous(), f"{name}: a dense vector, not an expanded view"


rubin_megamoe_in_bucket = RubinMegamoeInBucket()
