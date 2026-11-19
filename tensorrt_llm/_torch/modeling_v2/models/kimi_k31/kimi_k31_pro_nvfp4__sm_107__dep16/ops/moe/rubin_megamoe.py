# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed experts through MegaMoE on SM107: dispatch, FC1, SiTU, FC2 and combine fused.

``rubin_megamoe`` runs one chunk of this rank's rows through the MegaMoE CuteDSL NVFP4 kernel
over the expert-parallel group and returns them complete; ``__call__`` calls the family launcher
(``kernels/moe/rubin_megamoe.py``), whose staging is stateless. Its buffers come from the builder
:func:`rubin_megamoe_workspace`, a runtime helper a target calls once in ``post_load_weights``
over the group :func:`megamoe_ep_group` returns. Nothing is registered as a torch op.

:func:`megamoe_bucket` names the bucket a chunk launches over, for producers that write the rows
into it themselves (``rubin_route_topk_out``, ``rubin_megamoe_quantize``); the entry
``rubin_megamoe_in_bucket`` then launches on it without staging.

The launch runs the engine MegaMoE backend's defaults: a BF16 combine, no tactic sweep, every
bucket primed in the tuning pass. Also here, for the targets' construction (not an entry):
``megamoe_exchange_ctas``, the SMs the generation kernel's token exchange takes at its launch.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper, true_fp32_matmul

from ....ops.moe._rubin_moe_common import SF_VEC, dequantize_nvfp4, quantize_nvfp4
from ...kernels.moe.rubin_megamoe import (
    SM107_RUNGS,
    TOPK_DTYPE,
    MegaMoeBucket,
    MegaMoeWorkspace,
    bucket_for,
    build_workspace,
    exchange_grid_ctas,
    ladder,
)
from ...kernels.moe.rubin_megamoe import run as _launch

#: Kimi K3.1: experts per token, the SiTU soft caps.
TOP_K = 16
SITU = (4.0, 25.0)


def megamoe_ep_group(mapping: Any) -> Any:
    """The torch.distributed group of ``mapping``'s expert-parallel ranks, as the engine MegaMoE
    backend resolves it: the runtime's torch.distributed bootstrap under MPI
    (``ensure_cutedsl_torch_distributed``), then ``mapping.moe_ep_group_pg``, else the WORLD group
    when the expert-parallel group is the whole world.

    The kernel resets its barrier over the WORLD group, so the group must be the whole world (pure
    DEP). Collective: every rank calls it at the same point of model construction.
    """
    import torch.distributed as dist

    from tensorrt_llm._torch.distributed.initialization import ensure_cutedsl_torch_distributed

    ensure_cutedsl_torch_distributed(mapping.moe_ep_size)
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "MegaMoE needs torch.distributed initialized (mpirun or Ray) for multi-rank execution"
        )
    if dist.get_world_size() != mapping.moe_ep_size:
        raise ValueError(
            f"MegaMoE needs the expert-parallel group to be the whole world: world "
            f"{dist.get_world_size()}, EP {mapping.moe_ep_size}"
        )
    try:
        return mapping.moe_ep_group_pg
    except (NotImplementedError, AttributeError):
        return dist.group.WORLD


def rubin_megamoe_workspace(
    group: Any,
    ep_rank: int,
    ep_size: int,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    max_num_tokens: int,
    chunk_tokens: int,
    device: torch.device,
    *,
    rungs: Sequence[int] = SM107_RUNGS,
    pusher_ctas: int | None = None,
) -> MegaMoeWorkspace:
    """MegaMoE's buffers for every bucket of the ladder, rendezvoused over ``group``.

    ``max_num_tokens``: the engine's rows per rank of one step; ``chunk_tokens``: the rows per rank
    one chunk may hold (the scheduler splits a step at it); the top bucket holds the smaller.
    ``rungs``: the ladder below the top (the engine's SM107 rungs). ``pusher_ctas``: the
    generation kernel's token-exchange pusher CTAs, None for its default (``build_workspace``).
    Collective over ``group`` when ``ep_size`` > 1, bucket by bucket in ladder order.
    """
    return build_workspace(
        group,
        ep_rank,
        ep_size,
        num_local_experts,
        hidden,
        intermediate,
        TOP_K,
        max_num_tokens,
        chunk_tokens,
        device,
        situ=SITU,
        rungs=rungs,
        pusher_ctas=pusher_ctas,
    )


def megamoe_bucket(ws: MegaMoeWorkspace, chunk_max_tokens: int, num_tokens: int) -> MegaMoeBucket:
    """The bucket one chunk launches over, the same on every rank, for its producers to fill.

    ``chunk_max_tokens``: the chunk's largest row count over the EP ranks; ``num_tokens``: this
    rank's rows of it. The routing writes rows ``[0, num_tokens)`` of the bucket's ``topk_idx`` /
    ``topk_weights``, the quantization those rows of ``activation`` / ``activation_sf`` and the
    routing past them (``rubin_megamoe_quantize``); ``rubin_megamoe_in_bucket`` launches on it. A
    runtime helper, not an entry: no kernel runs.
    """
    return bucket_for(ws, chunk_max_tokens, num_tokens)


def megamoe_exchange_ctas(ws: MegaMoeWorkspace) -> int:
    """The SMs the generation kernel's token exchange takes at its launch over ``ws``'s EP group:
    its grid, padded to whole clusters, with the levers ``ws`` bound. A runtime helper, not an
    entry: no kernel runs."""
    return exchange_grid_ctas(ws.ep_size)


def _unswizzle(s: torch.Tensor) -> torch.Tensor:
    """The row-major ``[M, C]`` block scales of a 128x4-swizzled ``[M, C]`` matrix."""
    m, c = s.shape
    return s.reshape(m // 128, c // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(m, c)


def _expert_weights(fc1_w, fc1_s, fc2_w, fc2_s, j: int) -> tuple[torch.Tensor, ...]:
    """Expert ``j``'s gate, up and down projections as FP32 values without global scales."""
    w1 = dequantize_nvfp4(fc1_w[j], _unswizzle(fc1_s[j].view(torch.uint8)))
    rows = w1.shape[0]
    pairs = w1.view(rows // 32, 2, 16, -1)  # 16 gate rows, then 16 up rows
    gate = pairs[:, 0].reshape(rows // 2, -1)
    up = pairs[:, 1].reshape(rows // 2, -1)
    down = dequantize_nvfp4(fc2_w[j], _unswizzle(fc2_s[j].view(torch.uint8)))
    return gate, up, down


class RubinMegamoe(OpWrapper):
    """This rank's rows of one chunk through the experts of every rank: ``[num_tokens, hidden]``.

    ``x4`` / ``sf``: NVFP4 values ``[num_tokens, hidden / 2]`` and their linear E4M3 scales
    ``[num_tokens, hidden / 16]``; ``ids`` int32 / ``weights`` FP32 ``[num_tokens, 16]``, the
    routing; ``fc1_w`` ... ``fc1_norm_const``: this rank's experts as the weight table stores them
    (FC1 rows 16 gate then 16 up, block scales 128x4 swizzled; per-expert alphas and FC2 input
    scale); ``ws``: the workspace of :func:`rubin_megamoe_workspace`; ``chunk_max_tokens``: the
    chunk's largest row count over the EP ranks (picks the bucket, so every rank picks the same);
    ``num_tokens``: this rank's rows. Returns BF16 rows, a view of the bucket's output that the
    next call on the bucket overwrites.
    """

    # Multi-rank: no architecture until its cells pass across ranks on SM107.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DEP16 generation b12 k7 on one tray: 96 rows per rank in bucket 128, staged",
            spec=dict(ranks=4, local=56, rows=(96,), rungs=SM107_RUNGS),
        ),
        Cell(
            why="96 rows in a bucket of 96: the kernel reads the rows where they are",
            spec=dict(ranks=4, local=56, rows=(96,), rungs=(*SM107_RUNGS, 96)),
        ),
        Cell(
            why="40 rows after 96 in the same bucket: the tail rows are routed nowhere",
            spec=dict(ranks=4, local=56, rows=(96, 40), rungs=SM107_RUNGS),
        ),
        Cell(
            why="one rank without rows: it still launches, the others get theirs",
            spec=dict(ranks=4, local=56, counts=(8, 0, 8, 8), rungs=SM107_RUNGS),
        ),
    )

    note = """
    The kernel exchanges over the workspace's symmetric buffers: every rank of the group must
    launch every chunk, the same bucket, in the same order. The staging writes the bucket's whole
    tail on every call (ids -1, weights 0), so a CUDA graph captured at one row count replays
    right after any other. The routing weights apply after FC2; FC1's output is quantized to NVFP4
    under ``fc1_norm_const``.
    """

    def __call__(
        self,
        x4: torch.Tensor,
        sf: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc31_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        ws: MegaMoeWorkspace,
        chunk_max_tokens: int,
        num_tokens: int,
    ) -> torch.Tensor:
        return _launch(
            ws,
            x4,
            sf,
            ids,
            weights,
            (fc1_w, fc1_s, fc2_w, fc2_s, fc31_alpha, fc2_alpha, fc1_norm_const),
            chunk_max_tokens,
            num_tokens,
        )

    def reference(
        self,
        x4: Sequence[torch.Tensor],
        sf: Sequence[torch.Tensor],
        ids: Sequence[torch.Tensor],
        weights: Sequence[torch.Tensor],
        fc1_w: Sequence[torch.Tensor],
        fc1_s: Sequence[torch.Tensor],
        fc2_w: Sequence[torch.Tensor],
        fc2_s: Sequence[torch.Tensor],
        fc31_alpha: Sequence[torch.Tensor],
        fc2_alpha: Sequence[torch.Tensor],
        fc1_norm_const: Sequence[torch.Tensor],
        ws: MegaMoeWorkspace,
        chunk_max_tokens: int,
        num_tokens: int,
    ) -> torch.Tensor:
        """This rank's rows, FP32, from every rank's arguments in rank order (a collective).

        Rank ``r`` holds experts ``[r E, (r + 1) E)``. Per expert: FC1 of the dequantized input with
        the expert scaled by ``fc31_alpha``, SiTU, NVFP4 quantize-dequantize under
        ``fc1_norm_const``, FC2 with the expert scaled by ``fc2_alpha * fc1_norm_const``, times the
        routing weight, summed into the row. ``ws`` gives this rank; ``chunk_max_tokens`` is not
        read.
        """
        rank = ws.ep_rank
        local = fc1_w[0].shape[0]
        rows = x4[rank][:num_tokens].view(torch.uint8)
        x = dequantize_nvfp4(rows, sf[rank][:num_tokens].view(torch.uint8))
        ids_r = ids[rank][:num_tokens].long()
        out = torch.zeros(num_tokens, x.shape[1], dtype=torch.float32, device=x.device)
        with true_fp32_matmul():
            for e in ids_r.unique().tolist():
                if e < 0:
                    continue
                owner, j = divmod(e, local)
                tok, slot = (ids_r == e).nonzero(as_tuple=True)
                gate, up, down = _expert_weights(
                    fc1_w[owner], fc1_s[owner], fc2_w[owner], fc2_s[owner], j
                )
                a1 = fc31_alpha[owner][j].float()
                g = x[tok] @ (gate * a1).t()
                u = x[tok] @ (up * a1).t()
                act = SITU[0] * torch.tanh(g / SITU[0]) * torch.sigmoid(g)
                act = act * SITU[1] * torch.tanh(u / SITU[1])
                g2 = float(fc1_norm_const[owner][j])
                y = quantize_nvfp4(act, g2) @ (down * (fc2_alpha[owner][j].float() * g2)).t()
                w = weights[rank][:num_tokens][tok, slot].float().unsqueeze(1)
                out.index_add_(0, tok, y * w)
        return out

    def is_valid(
        self,
        x4: torch.Tensor,
        sf: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc31_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
        ws: MegaMoeWorkspace,
        chunk_max_tokens: int,
        num_tokens: int,
    ) -> None:
        """The kernel reads raw buffers: shapes, dtypes and the bucket are not checked there."""
        n, h = num_tokens, ws.hidden
        assert tuple(x4.shape) == (n, h // 2) and x4.dtype == torch.uint8, tuple(x4.shape)
        assert tuple(sf.shape) == (n, h // SF_VEC), f"linear scales [{n}, {h // SF_VEC}]"
        assert tuple(ids.shape) == tuple(weights.shape) == (n, ws.top_k), tuple(ids.shape)
        assert ids.dtype == TOPK_DTYPE and weights.dtype == torch.float32, (
            ids.dtype,
            weights.dtype,
        )
        assert ids.is_contiguous() and weights.is_contiguous(), "dense routing"
        e, i = ws.num_local_experts, ws.intermediate
        assert tuple(fc1_w.shape) == (e, 2 * i, h // 2), tuple(fc1_w.shape)
        assert tuple(fc1_s.shape) == (e, 2 * i, h // SF_VEC), tuple(fc1_s.shape)
        assert tuple(fc2_w.shape) == (e, h, i // 2), tuple(fc2_w.shape)
        assert tuple(fc2_s.shape) == (e, h, i // SF_VEC), tuple(fc2_s.shape)
        for name, t in (("fc31_alpha", fc31_alpha), ("fc2_alpha", fc2_alpha),
                        ("fc1_norm_const", fc1_norm_const)):  # fmt: skip
            assert tuple(t.shape) == (e,) and t.dtype == torch.float32, name
            assert t.is_contiguous(), f"{name}: a dense vector, not an expanded view"
        assert 0 <= n <= chunk_max_tokens or chunk_max_tokens <= 0, (n, chunk_max_tokens)
        assert chunk_max_tokens <= ws.buckets[-1].max_tokens, "the chunk exceeds the top bucket"


rubin_megamoe = RubinMegamoe()

__all__ = [
    "MegaMoeBucket",
    "MegaMoeWorkspace",
    "ladder",
    "megamoe_bucket",
    "megamoe_ep_group",
    "rubin_megamoe",
    "rubin_megamoe_workspace",
]
