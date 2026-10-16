# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the GLM-5.2 target's mnnvl_moe_finalize_hop entry.

Its own launcher, like ``_mnnvl_hop_op_matrix.py`` (see ``_mnnvl_ranks``):

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_moe_finalize_hop_op_matrix.py

The fused kernel must equal the two steps it replaces bit for bit: the TRTLLM-Gen finalize with
the shared experts' output as addend (computed here exactly in torch: a BF16 weight times a BF16
row is exact in FP32, so the separate multiply and add are the FMA), followed by ``mnnvl_hop`` on
its rows. Beyond the cells: empty (-1) slots, FC2 rows wider than the hidden size, interleaving with
the plain hops on one workspace, CUDA graph replays, and the index guard. The collected entry
point is ``test_modeling_v2_mnnvl_moe_finalize_hop_op_matrix.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Tuple

import _mnnvl_ranks as ranks
import torch
import torch.distributed as dist

assert torch.cuda.is_available(), "mnnvl_moe_finalize_hop requires CUDA devices"

DEADLINE_S = 1200
HIDDEN = 6144
EPS = 1e-5
TOP_K = 8

# Bound in the rank body: the entries and the workspace helpers import tensorrt_llm.
entry: Any = None
hop: Any = None
ws: Any = None
validating: Any = None
RANK = 0
WORLD = 1
MAPPING: Any = None

MoeInputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _moe_inputs(
    rows: int, seed: int, rank: int, routing: str, pitch: int = HIDDEN, empty_slots: bool = False
) -> MoeInputs:
    """One rank's ``(gemm2_output, expert_weights, expanded_idx_to_permuted_idx, addend)``.

    ``top8``: rows * 8 FC2 rows plus 5 padding rows, a random permutation of them over the
    (token, k) slots, positive routing weights; with ``empty_slots`` some slots are -1.
    ``routed_row``: the MTP layer's add in the hop, one FC2 row per token (its routed output) in
    slot 0 with weight 1, slots 1-7 empty. Scaled so that the finalized rows summed over the ranks
    have unit variance.
    """
    scale = WORLD**-0.5
    base = seed * 7919 + rank * 131
    addend = ranks.normal((rows, HIDDEN), base + 1, scale=0.5 * scale)
    if routing == "routed_row":
        gemm2_output = ranks.normal((rows, pitch), base + 2, scale=0.5 * scale)
        weights = torch.zeros((rows, TOP_K), dtype=torch.bfloat16, device="cuda")
        weights[:, 0] = 1.0
        index = torch.full((rows, TOP_K), -1, dtype=torch.int32, device="cuda")
        index[:, 0] = torch.arange(rows, dtype=torch.int32, device="cuda")
        return gemm2_output, weights, index, addend
    assert routing == "top8", routing
    fc2_rows = rows * TOP_K + 5
    gemm2_output = ranks.normal((fc2_rows, pitch), base + 3, scale=scale)
    gen = torch.Generator(device="cuda").manual_seed(base + 4)
    index = torch.randperm(fc2_rows, generator=gen, device="cuda")[: rows * TOP_K]
    index = index.to(torch.int32).view(rows, TOP_K).contiguous()
    if empty_slots:
        drop = torch.rand((rows, TOP_K), generator=gen, device="cuda") < 0.3
        index = torch.where(drop, torch.full_like(index, -1), index)
    weights = (torch.rand((rows, TOP_K), generator=gen, device="cuda") * 0.25).to(torch.bfloat16)
    return gemm2_output, weights, index, addend


def _finalized_rows(inputs: MoeInputs) -> torch.Tensor:
    """The TRTLLM-Gen finalize with an addend, bit for bit: the routed sum in FP32 in k order
    (each weight times row exact in FP32, so the add is the FMA's rounding), rounded to BF16, the
    addend added in FP32 and rounded once more."""
    gemm2_output, weights, index, addend = inputs
    routed = torch.zeros(addend.shape, dtype=torch.float32, device="cuda")
    for k in range(TOP_K):
        slot = index[:, k]
        taken = (slot >= 0).view(-1, 1)
        rows = gemm2_output[slot.clamp(min=0).long(), :HIDDEN].float()
        product = weights[:, k].float().view(-1, 1) * rows
        routed = torch.where(taken, routed + product, routed)
    return (addend.float() + routed.to(torch.bfloat16).float()).to(torch.bfloat16)


def _residual(rows: int, seed: int) -> torch.Tensor:
    """The residual stream: the same on every rank, as on the target."""
    return ranks.normal((rows, HIDDEN), seed * 1009 + 500)


def _gamma(seed: int) -> torch.Tensor:
    return (1.0 + ranks.normal(HIDDEN, seed, scale=0.1, dtype=torch.float32)).to(torch.bfloat16)


def _compare(norm, residual_out, ref_norm, ref_residual, where: str) -> None:
    """Bands of the fused hop against the FP32 reference.

    On top of mnnvl_hop's two roundings (the sum, the residual add) each rank rounds its routed
    sum and then the addend's sum to BF16 before the exchange: two more half-ulp roundings of a
    rank's row, whose sum over the ranks (independent signs) stays near one ulp of the summed
    row's scale. residual_out: 4 ulp per element and 2 ulp of relative RMS allowed; norm_out: 6
    and 2. Derived, not fitted; the fused-vs-two-step gate in the checks is exact.
    """
    from _validating import assert_within_ulp

    try:
        assert_within_ulp(residual_out, ref_residual, element_ulp=4.0, rms_ulp=2.0)
        assert_within_ulp(norm, ref_norm, element_ulp=6.0, rms_ulp=2.0)
    except AssertionError as exc:
        raise AssertionError(f"{where}: {exc}") from exc


def _fused(inputs: MoeInputs, residual, gamma) -> Tuple[torch.Tensor, torch.Tensor]:
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, residual.shape[0], HIDDEN)
    gemm2_output, weights, index, addend = inputs
    return entry.mnnvl_moe_finalize_hop(
        gemm2_output, weights, index, addend, residual, gamma, EPS, comm_buffer, buffer_flags
    )


def _two_step(inputs: MoeInputs, residual, gamma) -> Tuple[torch.Tensor, torch.Tensor]:
    """The finalize, then the plain hop on its rows."""
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, residual.shape[0], HIDDEN)
    rows = _finalized_rows(inputs)
    return hop.mnnvl_hop(rows, residual, gamma, EPS, comm_buffer, buffer_flags, False)


def check_certified_cells() -> None:
    """Every cell: equal to finalize + hop bit for bit, within the band of the reference, and the
    same on every rank."""
    gamma = _gamma(7)
    for i, cell in enumerate(entry.mnnvl_moe_finalize_hop.CELLS):
        spec = cell.spec
        rows = spec["rows"]
        assert spec["hidden"] == HIDDEN, cell.why
        seed = 100 + 13 * i
        per_rank = [_moe_inputs(rows, seed, r, spec["routing"]) for r in range(WORLD)]
        residual = _residual(rows, seed)
        comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, rows, HIDDEN)
        gemm2_output, weights, index, addend = per_rank[RANK]
        with validating(entry.mnnvl_moe_finalize_hop):
            norm, residual_out = entry.mnnvl_moe_finalize_hop(
                gemm2_output,
                weights,
                index,
                addend,
                residual,
                gamma,
                EPS,
                comm_buffer,
                buffer_flags,
            )
        ref_norm, ref_residual = entry.mnnvl_moe_finalize_hop.reference(
            per_rank, residual, gamma, EPS
        )
        _compare(norm, residual_out, ref_norm, ref_residual, cell.why)
        two_step_norm, two_step_residual = _two_step(per_rank[RANK], residual, gamma)
        ranks.assert_bitwise(norm, two_step_norm, f"{cell.why}: norm_out vs finalize + hop")
        ranks.assert_bitwise(residual_out, two_step_residual, f"{cell.why}: residual_out")
        ranks.assert_same_on_every_rank(norm, f"{cell.why}: norm_out")
        ranks.assert_same_on_every_rank(residual_out, f"{cell.why}: residual_out")
        dist.barrier()


def check_empty_slots_and_wide_rows() -> None:
    """-1 slots are skipped and FC2 rows wider than the hidden size are read at their pitch."""
    gamma = _gamma(11)
    for rows, pitch, empty in ((8, HIDDEN, True), (8, HIDDEN + 128, False), (3, HIDDEN + 64, True)):
        seed = 2000 + rows + pitch
        per_rank = [
            _moe_inputs(rows, seed, r, "top8", pitch=pitch, empty_slots=empty) for r in range(WORLD)
        ]
        residual = _residual(rows, seed)
        norm, residual_out = _fused(per_rank[RANK], residual, gamma)
        two_step = _two_step(per_rank[RANK], residual, gamma)
        where = f"rows={rows} pitch={pitch} empty slots={empty}"
        ranks.assert_bitwise(norm, two_step[0], f"{where}: norm_out vs finalize + hop")
        ranks.assert_bitwise(residual_out, two_step[1], f"{where}: residual_out")
        ref = entry.mnnvl_moe_finalize_hop.reference(per_rank, residual, gamma, EPS)
        _compare(norm, residual_out, *ref, where)
        dist.barrier()


def check_lamport_flags_rotate() -> None:
    """The fused hop rotates the workspace's flags as the plain hop does."""
    gamma = _gamma(13)
    for step, rows in enumerate((8, 1, 8)):
        _, buffer_flags = ws.hop_workspace(MAPPING, rows, HIDDEN)
        before = ranks.lamport_flags(buffer_flags)
        routing = "top8" if rows > 1 else "routed_row"
        _fused(_moe_inputs(rows, 3000 + step, RANK, routing), _residual(rows, step), gamma)
        after = ranks.lamport_flags(buffer_flags)
        bytes_to_clear = rows * HIDDEN * WORLD * torch.bfloat16.itemsize
        ranks.assert_rotated(before, after, bytes_to_clear, f"step {step}")
        dist.barrier()


def check_interleaved_with_the_hops() -> None:
    """Fused hops, plain hops and the built-in all-reduce follow each other on one workspace."""
    from tensorrt_llm.functional import AllReduceFusionOp

    gamma = _gamma(17)
    issued: List[Tuple[Any, ...]] = []
    for step in range(9):
        rows = (8, 1, 8)[step % 3]
        seed = 4000 + step
        residual = _residual(rows, seed)
        comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, rows, HIDDEN)
        if step % 3 == 2:
            per_rank_rows = [ranks.normal((rows, HIDDEN), seed * 3 + r) for r in range(WORLD)]
            outputs = hop.mnnvl_hop(
                per_rank_rows[RANK], residual, gamma, EPS, comm_buffer, buffer_flags, False
            )
            ref = hop.mnnvl_hop.reference(per_rank_rows, residual, gamma, EPS)
        elif step == 4:
            per_rank_rows = [ranks.normal((rows, HIDDEN), seed * 3 + r) for r in range(WORLD)]
            outputs = torch.ops.trtllm.mnnvl_fusion_allreduce(
                per_rank_rows[RANK],
                gamma,
                residual,
                EPS,
                comm_buffer,
                buffer_flags,
                True,
                None,
                int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
            )
            ref = hop.mnnvl_hop.reference(per_rank_rows, residual, gamma, EPS)
        else:
            routing = "top8" if rows > 1 else "routed_row"
            per_rank = [_moe_inputs(rows, seed, r, routing) for r in range(WORLD)]
            outputs = _fused(per_rank[RANK], residual, gamma)
            ref = entry.mnnvl_moe_finalize_hop.reference(per_rank, residual, gamma, EPS)
        issued.append((step, *outputs, *ref))
    torch.cuda.synchronize()
    for step, norm, residual_out, ref_norm, ref_residual in issued:
        _compare(norm, residual_out, ref_norm, ref_residual, f"interleaved step {step}")


def check_cuda_graph_replays() -> None:
    """A captured verify-layer pair and a draft-step hop replay right with fresh routing."""
    gamma = _gamma(19)
    static8 = [t.clone() for t in _moe_inputs(8, 5000, RANK, "top8")]
    static1 = [t.clone() for t in _moe_inputs(1, 5001, RANK, "routed_row")]
    residual8 = _residual(8, 5000).clone()

    def chain():
        norm8, res8 = _fused(static8, residual8, gamma)
        norm1, res1 = _fused(static1, res8[:1].contiguous(), gamma)
        return norm8, res8, norm1, res1

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        chain()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = chain()
    for replay in range(5):
        seed = 5100 + 10 * replay
        per_rank8 = [_moe_inputs(8, seed, r, "top8") for r in range(WORLD)]
        per_rank1 = [_moe_inputs(1, seed + 1, r, "routed_row") for r in range(WORLD)]
        for dst, src in zip(static8, per_rank8[RANK]):
            dst.copy_(src)
        for dst, src in zip(static1, per_rank1[RANK]):
            dst.copy_(src)
        residual8.copy_(_residual(8, seed))
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        norm8, res8, norm1, res1 = outputs
        ref = entry.mnnvl_moe_finalize_hop.reference(per_rank8, residual8, gamma, EPS)
        _compare(norm8, res8, *ref, f"replay {replay} verify layer")
        ref = entry.mnnvl_moe_finalize_hop.reference(per_rank1, res8[:1], gamma, EPS)
        _compare(norm1, res1, *ref, f"replay {replay} draft step")
        ranks.assert_same_on_every_rank(norm1, f"replay {replay} draft step")
        dist.barrier()
    del graph
    torch.cuda.synchronize()


def check_the_guard_refuses_an_index_out_of_range() -> None:
    """An index past the FC2 rows reads other memory as an expert row: is_valid refuses it."""
    gemm2_output, weights, index, addend = _moe_inputs(8, 6000, RANK, "top8")
    index = index.clone()
    index[3, 2] = gemm2_output.shape[0]
    comm_buffer, buffer_flags = ws.hop_workspace(MAPPING, 8, HIDDEN)
    residual = _residual(8, 6000)
    try:
        with validating(entry.mnnvl_moe_finalize_hop):
            entry.mnnvl_moe_finalize_hop(
                gemm2_output,
                weights,
                index,
                addend,
                residual,
                _gamma(23),
                EPS,
                comm_buffer,
                buffer_flags,
            )
    except AssertionError as exc:
        assert "must index gemm2_output's rows" in str(exc), exc
    else:
        raise AssertionError("is_valid admitted an index past the FC2 rows")


CHECKS = (
    check_certified_cells,
    check_empty_slots_and_wide_rows,
    check_lamport_flags_rotate,
    check_interleaved_with_the_hops,
    check_cuda_graph_replays,
    check_the_guard_refuses_an_index_out_of_range,
)


def _run_one_rank() -> None:
    global entry, hop, ws, validating, RANK, WORLD, MAPPING
    RANK, WORLD, MAPPING = ranks.init_rank()
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.comm import (  # noqa: E501
        _mnnvl_workspace,
        mnnvl_hop,
        mnnvl_moe_finalize_hop,
    )

    entry, hop, ws = mnnvl_moe_finalize_hop, mnnvl_hop, _mnnvl_workspace
    # Run as a script, not collected: the shared _validating module is reached the plain way.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from _validating import validating as _validating_cm

    validating = _validating_cm
    ranks.run_checks(CHECKS, RANK)


if __name__ == "__main__":
    if ranks.WORKER_FLAG in sys.argv:
        _run_one_rank()
    ranks.spawn_ranks(Path(__file__).resolve(), DEADLINE_S)
    print("OK")
