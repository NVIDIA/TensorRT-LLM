# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 attention output projection + TP all-reduce + residual add + attention-residual mix +
RMSNorm as two kernels: the push o_proj GEMM and a PDL attn_res mix that finishes the all-reduce."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.distributed import ops as _dist_ops
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.norm import attn_res as _k3_attn_res
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.norm import _attn_res_common as _c

from ...kernels.comm import bf16_ar_push_gemm as _push_gemm
from ..comm.oproj_allreduce_push import (  # noqa: E501
    DEFAULT_ONE_SHOT_MAX_BYTES,
    lamport_workspace,
    oproj_allreduce_push,
    push_serves,
)

#: Largest row count whose all-reduce the finishing mix finishes, by TP world. Above it the separate
#: finish kernel + mix is faster (GB300, W = 16: 96 rows 15.76 us push + finish + mix vs 16.33 us
#: with the finishing mix, 128 rows 17.35 vs 18.31). W = 32: the finishing mix up to 48 rows, the in-place
#: finish below from 49 rows (us per site, 32 ranks, finishing mix -> in place: 49 rows 13.91 -> 11.76, 64 rows
#: 14.85 -> 13.01, 80 rows 17.14 -> 15.05, 128 rows 21.78 -> 18.93; cuBLAS + engine AR + mix 14.02 / 14.72 /
#: 16.59 / 22.30).
FUSED_FINISH_MAX_ROWS = {4: 95, 8: 95, 16: 95, 32: 48}
#: Largest row count above the finishing mix's cap whose two-shot all-reduce the finish kernel completes in place
#: (no gather) for the mix to read from the workspace's broadcast stage; above it the finish kernel gathers into a
#: tensor (GB300, W = 16, us per site push + finish + mix -> in place: 96 rows 15.39 -> 14.67, 120 rows 16.67 ->
#: 15.56; 160 rows 22.33 -> 23.23, 190 rows 24.69 -> 25.76). W = 32: in place up to the PDL mix's last row count
#: (32 ranks, gather -> in place: 144 rows 22.34 -> 17.32, 160 rows 23.88 -> 18.47, 190 rows 28.57 -> 22.99).
IN_PLACE_FINISH_MAX_ROWS = {4: 128, 8: 128, 16: 128, 32: 191}
#: Largest row count whose all-reduce the finishing mix takes on the one-shot transport, by TP world, where the
#: engine's rule would take the two-shot one; the mix then rounds the summed addend as the two-shot owner does, so
#: the bits are the two-shot transport's. Two-shot puts a second NVLink hop (the row owner's reduce and broadcast) on
#: the mix's critical path; one-shot moves W times the rows over NVLink. GB300, W = 8, K = 1536, push GEMM +
#: finishing mix, us per call one-shot vs two-shot over N = 2..9: 10 rows -0.84..-1.20, 12 rows -0.83..-1.05,
#: 15 rows -0.22..-0.46, 16 rows -0.07..+0.12, 20 rows +0.14..+0.87.
FUSED_ONE_SHOT_MAX_ROWS = {8: 15}


def _cell(rows: int, k: int, candidates: int, why: str) -> Cell:
    return Cell(why=why, spec=dict(world=4, rows=rows, k=k, candidates=candidates))


class K31OprojPushAttnResAddRmsnorm(OpWrapper):
    """``(updated, out)``: ``updated = bf16(prefix + all_reduce(input @ weight^T))``, ``out`` the
    attention-residual mix of ``[block..., updated]`` and its RMSNorm (``k31_attn_res_rmsnorm``).

    ``input [M, K]`` / ``weight [7168, K]``: this rank's slice of the attention heads and of the
    row-parallel o_proj (BF16, contiguous); ``all_reduce``: the engine ``AllReduce`` module of the
    attention TP group; ``prefix [M, 7168]``: the running residual; ``block [N - 1, M, 7168]``: the
    snapshot rows; the weights / epsilons as ``k31_attn_res_add_rmsnorm``.

    Where the group runs MNNVL, the push GEMM has a tactic for (M, 7168, K) and M is at most
    ``FUSED_FINISH_MAX_ROWS[tp]`` (95, 48 at TP 32: above it the separate finish kernel wins), two kernels run:
    the push GEMM stores its BF16 partial rows into the Lamport slots of the group's MNNVL
    workspace, and the PDL attn_res mix polls them, sums them in the engine all-reduce's order
    (rank order, the engine kernel's chunking), adds the sum to ``prefix`` (round-to-nearest),
    mixes and normalizes. The transport is the engine's choice, except that up to
    ``FUSED_ONE_SHOT_MAX_ROWS[tp]`` rows it is one-shot with the two-shot transport's bits. Above
    ``FUSED_FINISH_MAX_ROWS[tp]``, up to ``IN_PLACE_FINISH_MAX_ROWS[tp]`` rows on the two-shot
    transport, the finish kernel completes the all-reduce in place and the mix reads the result rows
    from the workspace (``k31_attn_res_lamport_add_rmsnorm``, the add of ``k31_attn_res_add_rmsnorm``).
    Otherwise ``oproj_allreduce_push`` (push GEMM + finish, or cuBLAS + the engine all-reduce) and
    ``k31_attn_res_add_rmsnorm`` run.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        _cell(1, 768, 3, "TEP16 shard, one decode row"),
        _cell(6, 768, 5, "TEP16 shard, mid B1 verify rows"),
        _cell(8, 384, 9, "TEP32 shard, ll B1 verify rows"),
        _cell(24, 768, 9, "TEP16 shard, mid B4 verify rows (two-shot transport at W=16)"),
        _cell(72, 768, 5, "TEP16 shard, M2 B12 verify rows"),
        _cell(128, 768, 5, "TEP16 shard past the fused-finish cap: push + finish in place + mix"),
        _cell(160, 768, 5, "TEP16 shard past the in-place cap: push + finish kernel + mix"),
        _cell(48, 384, 5, "TEP32 shard, B9-B10 verify rows"),
        _cell(64, 384, 5, "TEP32 shard, B12-B13 verify rows (4 ranks: the finishing mix)"),
        _cell(120, 384, 9, "TEP32 shard, 120 rows (4 ranks: push + finish in place + mix)"),
        _cell(160, 384, 5, "TEP32 shard, 160 rows (4 ranks: push + finish kernel + mix)"),
    )
    note = """
    The pre-MoE site of a decoder layer whose attention output is added to the running residual
    (every layer that does not append a snapshot), on generation servers at attention TP. The
    all-reduce half is bitwise against the engine MNNVL all-reduce of the push GEMM's partials;
    the push GEMM half is not bitwise against cuBLAS (its own K tile order: one BF16 step per
    rank partial); the mix + RMSNorm output is one BF16 step from the engine attn_res ops in a
    small fraction of the elements; `updated` has the bits of the torch add of the all-reduce
    result. Workspace and ordering rules as `oproj_allreduce_push` (the group's engine MNNVL
    workspace, grown before CUDA-graph capture; every rank makes the same calls). The finishing
    mix must be the next kernel on the workspace after the push GEMM; it polls and sums the slots
    before its griddepcontrol.wait and stores after it.
    """

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        all_reduce: AllReduce,
        prefix: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows, hidden = input.shape[0], weight.shape[0]
        world = all_reduce.mapping.tp_size
        if not push_serves(input, weight, all_reduce):
            addend = oproj_allreduce_push(input, weight, all_reduce)
            return _k3_attn_res.k31_attn_res_add_rmsnorm(
                prefix, addend, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
            )
        one_shot = _lam.one_shot_for(rows, hidden, world, DEFAULT_ONE_SHOT_MAX_BYTES)
        cpp_one_shot = _lam.one_shot_for(
            rows, hidden, world, _dist_ops._MNNVL_ONE_SHOT_THRESHOLD_BYTES
        )
        if rows > FUSED_FINISH_MAX_ROWS.get(world, 0):
            if one_shot or rows > IN_PLACE_FINISH_MAX_ROWS.get(world, 0):
                addend = oproj_allreduce_push(input, weight, all_reduce)
                return _k3_attn_res.k31_attn_res_add_rmsnorm(
                    prefix, addend, block, res_weight, rms_weight, out_weight, rms_eps, out_eps
                )
            # Push GEMM, the two-shot finish without its gather, and the mix reading the result rows from the
            # workspace's broadcast stage (the bits of the finish kernel's output and k31_attn_res_add_rmsnorm's add).
            ws = lamport_workspace(all_reduce, rows, hidden, False)
            use_pdl = get_env_enable_pdl()
            _push_gemm.run_bf16_gemm_ar_push(input, weight, ws, False, prefix, use_pdl=use_pdl)
            _lam.run_lamport_finish_in_place(
                ws, rows, hidden, use_pdl=use_pdl, cpp_one_shot=cpp_one_shot
            )
            return _k3_attn_res.k31_attn_res_lamport_add_rmsnorm(
                prefix,
                ws.uc_local,
                ws.flags.data_ptr(),
                block,
                res_weight,
                rms_weight,
                out_weight,
                rms_eps,
                out_eps,
                0,
                0,
                False,
                0,
                0,
                False,
            )
        # One-shot below the per-world cap even where the engine takes two-shot; the mix keeps the bits of the
        # engine's transport (two_shot_bits below) and its reduction order (the chunk).
        two_shot_bits = not one_shot and rows <= FUSED_ONE_SHOT_MAX_ROWS.get(world, 0)
        one_shot = one_shot or two_shot_bits
        ws = lamport_workspace(all_reduce, rows, hidden, one_shot)
        # The push GEMM only reads the all-reduce output's shape ([rows, 7168], not written).
        _push_gemm.run_bf16_gemm_ar_push(
            input, weight, ws, one_shot, prefix, use_pdl=get_env_enable_pdl()
        )
        return _k3_attn_res.k31_attn_res_lamport_add_rmsnorm(
            prefix,
            ws.uc_local,
            ws.flags.data_ptr(),
            block,
            res_weight,
            rms_weight,
            out_weight,
            rms_eps,
            out_eps,
            ws.world,
            _lam.reduce_chunk(ws.world, cpp_one_shot),
            not one_shot,
            ws.mc_base,
            ws.rank,
            True,
            two_shot_bits,
        )

    def reference(
        self,
        inputs_in_rank_order: Sequence[torch.Tensor],
        weights_in_rank_order: Sequence[torch.Tensor],
        prefix: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collective: every rank's ``input`` and ``weight`` in rank order. The FP32 sum of the
        partial products rounded once, the torch add, then the mix + RMSNorm reference."""
        addend = oproj_allreduce_push.reference(inputs_in_rank_order, weights_in_rank_order, None)
        updated = prefix + addend
        out = _c.reference(updated, block, res_weight, rms_weight, out_weight, rms_eps, out_eps)
        return updated, out

    def is_valid(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        all_reduce: AllReduce,
        prefix: torch.Tensor,
        block: torch.Tensor,
        res_weight: torch.Tensor,
        rms_weight: torch.Tensor,
        out_weight: torch.Tensor,
        rms_eps: float,
        out_eps: float,
    ) -> None:
        oproj_allreduce_push.is_valid(input, weight, all_reduce)
        assert weight.shape[0] == _c.HIDDEN, "o_proj output features must be 7168"
        assert input.shape[0] == prefix.shape[0], "one input row per residual row"
        _c.check_operands(prefix, block, res_weight, rms_weight, out_weight)


k31_oproj_push_attn_res_add_rmsnorm = K31OprojPushAttnResAddRmsnorm()
