# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 Helix generation tail in one launch: split-KV fold, V up-projection, exchange, combine."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_helix import k3_helix_tail as _tail

_LOG2_E = math.log2(math.e)


def packed_reduce_reference(
    fold: _tail.FoldInputs, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 fold of the packed decode's split state: ``(latent BF16 [requests * valid_m, 512],
    stats FP32 [requests * valid_m, 2])``, natural-log ``(max, sum)``.

    A request planned on one task keeps the row the decode stored in ``rows`` and that task's
    statistics. The FP32 partials are max-relative numerators (``output_scale`` converts them)."""
    plan = fold.plan.tolist()
    begins = plan[2 : 3 + fold.num_requests]
    valid_m = fold.valid_m
    scale = fold.sm_scale * _LOG2_E / (fold.q_scale.double().item() * fold.kv_scale.double().item())
    latent, maxima, sums = [], [], []
    for r, (begin, end) in enumerate(zip(begins, begins[1:])):
        row_sum = fold.sums[begin:end, :valid_m].double()
        active = row_sum != 0
        row_max = fold.maxima[begin:end, :valid_m].double() * scale
        global_max = row_max.masked_fill(~active, -torch.inf).amax(0)
        weights = torch.where(active, torch.exp2(row_max - global_max), 0.0)
        values = fold.partial[begin:end, :valid_m].double()
        numerator = (torch.where(active.unsqueeze(-1), values, 0.0) * weights.unsqueeze(-1)).sum(0)
        denominator = (row_sum * weights).sum(0)
        out = torch.where(
            denominator.unsqueeze(-1) != 0,
            numerator / denominator.unsqueeze(-1) * fold.output_scale,
            0.0,
        )
        if end - begin == 1:
            out = rows[r * valid_m : (r + 1) * valid_m].double()
        latent.append(out)
        maxima.append(torch.where(denominator != 0, global_max / _LOG2_E, -torch.inf))
        sums.append(denominator)
    stats = torch.stack((torch.cat(maxima), torch.cat(sums)), -1).float()
    return torch.cat(latent).to(rows.dtype), stats


def _combine(values: Sequence[torch.Tensor], stats: Sequence[torch.Tensor]) -> torch.Tensor:
    """``helix_post_process_native`` in FP64: the stats-weighted mean of the sources' rows."""
    maxima = torch.stack([s[..., 0].double() for s in stats])
    sums = torch.stack([s[..., 1].double() for s in stats])
    global_max = maxima.amax(0)
    weights = torch.where(sums != 0, sums * (maxima - global_max).exp(), 0.0)
    num = sum(w.unsqueeze(-1) * v.double() for w, v in zip(weights, values))
    return num / weights.sum(0).unsqueeze(-1)


def _mxfp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from ...kernels.k3r_gemm.k3_fp8w_lowm_mxfp8 import mxfp8_quantize_reference

    return mxfp8_quantize_reference(x)


class K3HelixTailOp(OpWrapper):
    """``k3_helix_tail``: this rank's BF16 ``out`` ``[T, H / P * 128]`` from the decode's latent
    rows of all ``H`` heads (``rows`` ``[T * H, 512]`` with ``stats`` ``[T * H, 2]``, or the split
    state ``fold``), the per-head V up-projection ``w_uv`` ``[H, 128, 512]``, the exchange over the
    ``P`` CP ranks and the stats-weighted combine. Writes ``out`` (and ``mx_out``'s ``xq`` /
    ``xs``) in place and returns None."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # Spec: `cp_size` (P; > 1 runs on one GPU with `loopback`), the decode's split state
    # (`kv_tiles` per request, `splits` task slots, `wave_tasks`, `query_len`; T = requests *
    # query_len), `partial` ("fp32", or None: rows stored directly), `mx_out` (o_proj's MXFP8
    # operand). H = 96 heads per rank, latent 512, V head 128.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 decode: P = 8 (12 heads out), FP32 split partials folded, 8 requests x 4",
            spec=dict(
                cp_size=8,
                kv_tiles=[40] * 8,
                splits=8,
                wave_tasks=106,
                query_len=4,
                partial="fp32",
                mx_out=False,
            ),
        ),  # fmt: skip
        Cell(
            why="helix8 decode: one 21.9k-token request split over 8 tasks",
            spec=dict(
                cp_size=8,
                kv_tiles=[86],
                splits=8,
                wave_tasks=106,
                query_len=4,
                partial="fp32",
                mx_out=False,
            ),
        ),  # fmt: skip
        Cell(
            why="helix8 side gate: the gate joins before the tail, o_proj's MXFP8 operand stored",
            spec=dict(
                cp_size=8,
                kv_tiles=[60, 9, 33, 1, 86, 20, 5, 40],
                splits=8,
                wave_tasks=106,
                query_len=4,
                partial="fp32",
                mx_out=True,
            ),
        ),  # fmt: skip
        Cell(
            why="helix8 DSpark drafter: rows the attention stored directly, no fold",
            spec=dict(
                cp_size=8,
                kv_tiles=[1] * 2,
                splits=1,
                wave_tasks=106,
                query_len=8,
                partial=None,
                mx_out=False,
            ),
        ),  # fmt: skip
        Cell(
            why="one rank (P = 1): single-task requests between multi-task ones, rows copied",
            spec=dict(
                cp_size=1,
                kv_tiles=[1, 79, 3, 176],
                splits=6,
                wave_tasks=17,
                query_len=8,
                partial="fp32",
                mx_out=False,
            ),
        ),  # fmt: skip
    )

    note = """
    Collective over the CP group whose receive buffers are the MNNVL workspace at `ws_addr`
    (rank r's at `ws_addr + r * rank_stride`, two halves of `half_bytes`, armed with the sentinel);
    `state` is the launch counter of that workspace. Launches sharing a workspace must run in the
    same order on one stream of every rank. `loopback`: one GPU stands for the group (records of
    head group d are sent to this rank as source d), for single-GPU tests.

    Numerics: the fold and the combine are bitwise the packed split reduction and
    `helix_post_process_native`; the V up-projection rounds FP32 dot products in another summation
    order than the BMM (within one BF16 ulp of the FP64 product). Rows of tokens a rank owns no KV
    for (`zero_kv_mask`) contribute nothing. `mx_out = (gate, xq, xs)` also stores the MXFP8 form
    of `bf16(out * gate)` (o_proj's operand, bitwise the gated low-M GEMM's quantization).

    A process cache holds compiled kernels per feature combination; one CTA per SM at most
    (`max_ctas`). `timestamps` (microbenchmarks) stores %globaltimer phase stamps. The receive buffer and `state`
    are internal: the reference does not model them. The cells run one GPU (`loopback`), so no
    cell masks a token (every source of a looped-back token is the same rank).
    """

    def __call__(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor | None,
        w_uv: torch.Tensor,
        out: torch.Tensor,
        fold: _tail.FoldInputs | None,
        zero_kv_mask: torch.Tensor | None,
        cp_rank: int,
        cp_size: int,
        ws_addr: int,
        rank_stride: int,
        half_bytes: int,
        state: torch.Tensor,
        max_ctas: int,
        use_pdl: bool,
        mx_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        timestamps: torch.Tensor | None,
        loopback: bool,
        mx_atom: bool,
    ) -> None:
        _tail.k3_helix_tail(
            rows,
            stats,
            w_uv,
            out,
            fold=fold,
            zero_kv_mask=zero_kv_mask,
            cp_rank=cp_rank,
            cp_size=cp_size,
            ws_addr=ws_addr,
            rank_stride=rank_stride,
            half_bytes=half_bytes,
            state=state,
            max_ctas=max_ctas,
            use_pdl=use_pdl,
            mx_out=mx_out,
            timestamps=timestamps,
            loopback=loopback,
            mx_atom=mx_atom,
        )

    def reference(
        self,
        rows: torch.Tensor | Sequence[torch.Tensor],
        stats: torch.Tensor | Sequence[torch.Tensor] | None,
        w_uv: torch.Tensor,
        out: torch.Tensor,
        fold: _tail.FoldInputs | Sequence[_tail.FoldInputs] | None,
        zero_kv_mask: torch.Tensor | Sequence[torch.Tensor] | None,
        cp_rank: int,
        cp_size: int,
        ws_addr: int,
        rank_stride: int,
        half_bytes: int,
        state: torch.Tensor,
        max_ctas: int,
        use_pdl: bool,
        mx_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        timestamps: torch.Tensor | None,
        loopback: bool,
        mx_atom: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The post-state ``out`` (and ``(out, xq, xs)`` with ``mx_out``), FP64-accumulated.

        A collective: with ``cp_size > 1`` and not ``loopback``, ``rows``, ``stats``, ``fold`` and
        ``zero_kv_mask`` are the group's inputs in rank order (``w_uv`` is replicated). With
        ``loopback`` (or ``P = 1``) they are this rank's, and source ``d`` is this rank's head
        group ``d``. The workspace arguments, ``state``, ``max_ctas``, ``use_pdl`` and
        ``timestamps`` do not change the result. ``mx_atom`` (another scale layout) is not
        modelled: its cells are none."""
        heads = w_uv.shape[0]
        group = heads // cp_size
        per_rank = cp_size > 1 and not loopback

        def as_list(value):
            if per_rank:
                return list(value) if value is not None else [None] * cp_size
            return [value]

        sources = []
        for r_rows, r_stats, r_fold, r_mask in zip(
            as_list(rows), as_list(stats), as_list(fold), as_list(zero_kv_mask)
        ):
            if r_fold is not None:
                r_rows, r_stats = packed_reduce_reference(r_fold, r_rows)
            tokens = r_rows.shape[0] // heads
            latent = r_rows.view(tokens, heads, -1).double()
            values = torch.einsum("tjk,jnk->tjn", latent, w_uv.double()).to(torch.bfloat16)
            r_stats = r_stats.view(tokens, heads, 2).clone()
            if r_mask is not None:
                dead = r_mask[:tokens].bool()
                values[dead] = 0
                r_stats[dead] = torch.tensor([-torch.inf, 0.0], device=r_stats.device)
            sources.append((values, r_stats))
        if per_rank:
            heads_of = slice(cp_rank * group, (cp_rank + 1) * group)
            values = [v[:, heads_of] for v, _ in sources]
            stats_by_source = [s[:, heads_of] for _, s in sources]
        else:
            v, s = sources[0]
            values = [v[:, d * group : (d + 1) * group] for d in range(cp_size)]
            stats_by_source = [s[:, d * group : (d + 1) * group] for d in range(cp_size)]
        combined = _combine(values, stats_by_source)
        result = combined.reshape(out.shape).to(out.dtype)
        if mx_out is None:
            return result
        gate = mx_out[0]
        xq, xs = _mxfp8_reference(result * gate)
        return result, xq, xs

    def is_valid(
        self,
        rows: torch.Tensor,
        stats: torch.Tensor | None,
        w_uv: torch.Tensor,
        out: torch.Tensor,
        fold: _tail.FoldInputs | None,
        zero_kv_mask: torch.Tensor | None,
        cp_rank: int,
        cp_size: int,
        ws_addr: int,
        rank_stride: int,
        half_bytes: int,
        state: torch.Tensor,
        max_ctas: int,
        use_pdl: bool,
        mx_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        timestamps: torch.Tensor | None,
        loopback: bool,
        mx_atom: bool,
    ) -> None:
        # Every CTA polls after its own pushes: more CTAs than SMs can leave a sender unscheduled
        # behind a spinning receiver.
        sms = torch.cuda.get_device_properties(out.device).multi_processor_count
        assert max_ctas <= sms, f"max_ctas {max_ctas} exceeds the {sms} SMs (all CTAs co-resident)"
        # Records that overrun a receive half land in the other half or past the workspace.
        tokens = out.shape[0]
        need = tokens * cp_size * _tail.record_bytes(w_uv.shape[0] // cp_size, w_uv.shape[1])
        assert need <= half_bytes, f"{need} bytes of records exceed a {half_bytes}-byte half"
        # A mask shorter than the batch is read past its end.
        if zero_kv_mask is not None:
            assert zero_kv_mask.numel() >= tokens and zero_kv_mask.is_contiguous()


k3_helix_tail = K3HelixTailOp()
