# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused finalize + all-reduce parity with the unfused finalize and NCCL, on four GB300 ranks.

Each test runs its body on four MPI workers spawned by the test process. The
reference on every rank is the unfused finalize of the same inputs followed by
an NCCL all-reduce; every rank must also end with the same bits.
"""

import contextlib
import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenMoE,
    post_expert_norm_combine,
)
from tensorrt_llm._torch.models.musespark_open_finalize_ar import HAS_CUTEDSL, FinalizeIpcWorkspace
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

RANKS = 4

pytestmark = [
    # The worker pool is reused across tests, and NCCL starts threads that outlive each test.
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(
        torch.cuda.device_count() < RANKS
        or not HAS_CUTEDSL
        or torch.cuda.get_device_capability() != (10, 3),
        reason="four GB300 GPUs and CuTe DSL required",
    ),
    pytest.mark.parametrize("mpi_pool_executor", [RANKS], indirect=True),
]


def _run_rank(body, *args):
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank)
    mapping = Mapping(world_size=RANKS, tp_size=RANKS, rank=rank, gpus_per_node=RANKS)
    with torch.inference_mode():
        return body(rank, mapping, *args)


def _on_every_rank(executor, body, *args) -> list:
    """`body(rank, mapping, *args)` on each worker; the per-rank results in rank order."""
    return list(executor.map(_run_rank, [body] * RANKS, *[[arg] * RANKS for arg in args]))


def _reference(nccl, gemm, indices, weights, norm, eps, tokens):
    partial = post_expert_norm_combine(
        gemm, indices, weights, norm, eps, num_tokens=tokens, top_k=8, hidden_size=4096
    )
    return nccl(partial.to(torch.bfloat16))


def _within_gates(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    actual, expected = actual.float(), expected.float()
    if not torch.allclose(actual, expected, rtol=0.04, atol=0.02):
        return False
    # Every expert can be masked in a one-token case, which makes both sides zero.
    return bool((actual - expected).norm() <= 0.01 * expected.norm())


def _same_on_all_ranks(tensor: torch.Tensor) -> bool:
    gathered = MPI.COMM_WORLD.allgather(tensor.view(torch.int16).cpu().numpy())
    return all((peer == gathered[0]).all() for peer in gathered)


def _all_ranks(ok: bool) -> bool:
    """Combine per-rank verdicts, so a failure on one rank never strands the others."""
    return all(MPI.COMM_WORLD.allgather(bool(ok)))


def _inputs(tokens: int, rank: int, seed: int, rows=None, width=4096):
    """Routing shared by all ranks; every fourth slot is local to `rank`."""
    shared = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.rand(tokens, 8, device="cuda", generator=shared)
    weights /= weights.sum(-1, keepdim=True)
    norm = (torch.randn(4096, device="cuda", generator=shared) * 0.03).to(torch.bfloat16)
    local = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    rows = max(64, tokens * 8) if rows is None else rows
    gemm = torch.randn(rows, width, device="cuda", generator=local).to(torch.bfloat16)
    indices = torch.arange(tokens * 8, device="cuda", dtype=torch.int32).reshape(tokens, 8)
    indices = torch.where(indices % RANKS == rank, indices, -1).contiguous()
    return gemm, indices, weights, norm


def _graph_parity(rank, mapping, tokens, case):
    workspace = FinalizeIpcWorkspace(mapping)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    last = rank == RANKS - 1
    gemm, indices, weights, norm = _inputs(tokens, rank, 1270 + tokens)
    source = gemm.clone()
    if case == "upper":
        indices[0, rank] = 10000
    if case == "empty_rank" and last:
        indices.fill_(-1)
        gemm.fill_(float("nan"))
    if case == "zero_rows" and last:
        indices.zero_()
        gemm = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)

    for _ in range(3):
        workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
    ok = True
    for repeat in range(3):
        if not (case in ("empty_rank", "zero_rows") and last):
            gemm.copy_(source * (-1 if repeat % 2 else 1) + 0.1 * repeat)
        expected = _reference(nccl, gemm, indices, weights, norm, 1e-8, tokens)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        ok &= _within_gates(actual, expected)
        ok &= _same_on_all_ranks(actual)
    del graph
    return _all_ranks(ok)


@pytest.mark.parametrize("tokens", [1, 2, 8, 16, 32, 80, 115, 122, 123, 128])
@pytest.mark.parametrize("case", ["normal", "upper", "empty_rank", "zero_rows"])
def test_finalize_allreduce_graph_parity(mpi_pool_executor, tokens: int, case: str) -> None:
    """Replays of a captured graph track mutated inputs, including a rank with no work."""
    assert all(_on_every_rank(mpi_pool_executor, _graph_parity, tokens, case))


def _compiles_nothing(rank, mapping):
    import cutlass.cute as cute

    workspace = FinalizeIpcWorkspace(mapping)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()

    def no_compilation(*args, **kwargs):
        raise AssertionError("a kernel was compiled after workspace construction")

    ok = True
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cute, "compile", no_compilation)
        for step, tokens in enumerate(
            (1, 2, 7, 8, 9, 16, 31, 32, 33, 64, 65, 100, 127, 128, 1, 128)
        ):
            rows = (0, 1, 3, max(64, tokens * 8))[(step + rank) % 4]
            gemm, indices, weights, norm = _inputs(tokens, rank, 300 + step, rows=rows)
            eps = (1e-8, 1e-6, 1e-5)[step % 3]
            actual = workspace.finalize(gemm, indices, weights, norm, eps, tokens)
            ok &= _within_gates(actual, _reference(nccl, gemm, indices, weights, norm, eps, tokens))
            ok &= _same_on_all_ranks(actual)
        # Global capture mode turns any compilation or module load during capture into an error.
        gemm, indices, weights, norm = _inputs(16, rank, 999)
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="global"):
            actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, 16)
        graph.replay()
        expected = _reference(nccl, gemm, indices, weights, norm, 1e-8, 16)
        torch.cuda.synchronize()
        ok &= _within_gates(actual, expected)
        ok &= _same_on_all_ranks(actual)
        del graph
    return _all_ranks(ok)


def test_finalize_compiles_nothing_after_construction(mpi_pool_executor) -> None:
    """Token counts, row counts, eps and routing changes reuse the constructed kernel."""
    assert all(_on_every_rank(mpi_pool_executor, _compiles_nothing))


def _replays_track_inputs(rank, mapping, tokens):
    workspace = FinalizeIpcWorkspace(mapping)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    calls = [_inputs(tokens, rank, 500 + i) for i in range(5)]
    for gemm, indices, weights, norm in calls:
        workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
    torch.cuda.synchronize()
    MPI.COMM_WORLD.Barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [workspace.finalize(g, i, w, n, 1e-8, tokens) for g, i, w, n in calls]
    ok = True
    for repeat in range(4):
        for index, (gemm, indices, weights, norm) in enumerate(calls):
            fresh = _inputs(tokens, rank, 700 + 10 * repeat + index)
            gemm.copy_(fresh[0])
            indices.copy_(torch.roll(fresh[1], shifts=repeat, dims=1))
            weights.copy_(fresh[2])
            norm.copy_(fresh[3])
        expected = [_reference(nccl, g, i, w, n, 1e-8, tokens) for g, i, w, n in calls]
        torch.cuda.synchronize()
        MPI.COMM_WORLD.Barrier()
        graph.replay()
        torch.cuda.synchronize()
        for actual, want in zip(outputs, expected):
            ok &= _within_gates(actual, want)
            ok &= _same_on_all_ranks(actual)
    del graph
    return _all_ranks(ok)


@pytest.mark.parametrize("tokens", [1, 9, 64, 128])
def test_finalize_graph_replays_track_every_input(mpi_pool_executor, tokens: int) -> None:
    """Each captured call reads its own inputs; stale data would surface as a mismatch."""
    assert all(_on_every_rank(mpi_pool_executor, _replays_track_inputs, tokens))


def _contract_case(rank, mapping, case, tokens):
    workspace = FinalizeIpcWorkspace(mapping)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    gemm, indices, weights, norm = _inputs(tokens, rank, 1100 + tokens)
    eps_values = (1e-8,)
    if case == "negative_zero_partials":
        # Alternating-sign rows times subnormal weights round to BF16 (+0, -0)
        # pairs: the sentinel bit pattern, unless negative zeros are canonicalized.
        signs = torch.where(torch.arange(4096, device="cuda") % 2 == 0, 1.0, -1.0)
        gemm = signs.expand(gemm.shape[0], -1).to(torch.bfloat16).contiguous()
        weights = torch.full_like(weights, 1e-41)
    elif case == "all_invalid":
        indices = torch.full_like(indices, -1)
    elif case == "duplicate_rows":
        indices[:, 1] = indices[:, 0]
    elif case == "extreme_indices":
        rows = gemm.shape[0]
        extremes = torch.tensor([rows - 1, rows, -1, -(2**31), 2**31 - 1], dtype=torch.int32)
        indices[:, :5] = extremes.to(indices.device)
    elif case == "padded_rows_with_nan":
        wide = torch.full(
            (gemm.shape[0], 4096 + 64), float("nan"), device="cuda", dtype=torch.bfloat16
        )
        wide[:, :4096] = gemm
        gemm = wide[:, :4096]
    elif case == "eps_sweep":
        gemm = (gemm.float() * 1e-2).to(torch.bfloat16)
        eps_values = (1e-6, 1e-3, 0.1, 1.0)
    elif case == "single_contributor" and rank != 0:
        indices = torch.full_like(indices, -1)
    ok = True
    for eps in eps_values:
        actual = workspace.finalize(gemm, indices, weights, norm, eps, tokens)
        ok &= _within_gates(actual, _reference(nccl, gemm, indices, weights, norm, eps, tokens))
        ok &= _same_on_all_ranks(actual)
        if case == "all_invalid":
            ok &= bool((actual.float() == 0).all())
    return _all_ranks(ok)


@pytest.mark.parametrize(
    "case",
    [
        "negative_zero_partials",
        "all_invalid",
        "duplicate_rows",
        "extreme_indices",
        "padded_rows_with_nan",
        "eps_sweep",
        "single_contributor",
    ],
)
@pytest.mark.parametrize("tokens", [1, 33, 128])
def test_finalize_contract_cases(mpi_pool_executor, case: str, tokens: int) -> None:
    assert all(_on_every_rank(mpi_pool_executor, _contract_case, case, tokens))


def _malformed_calls(rank, mapping):
    workspace = FinalizeIpcWorkspace(mapping)
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    tokens = 4
    gemm, indices, weights, norm = _inputs(tokens, rank, 77)
    padded = torch.zeros(gemm.numel() + 8, device="cuda", dtype=torch.bfloat16)
    misaligned = padded[1 : 1 + gemm.numel()].view(gemm.shape)
    odd_stride = torch.zeros(gemm.shape[0], 4100, device="cuda", dtype=torch.bfloat16)[:, :4096]
    malformed = [
        (gemm.float(), indices, weights, norm, 1e-8, tokens),
        (gemm[:, :2048], indices, weights, norm, 1e-8, tokens),
        (gemm.t(), indices, weights, norm, 1e-8, tokens),
        (misaligned, indices, weights, norm, 1e-8, tokens),
        (odd_stride, indices, weights, norm, 1e-8, tokens),
        (gemm, indices.long(), weights, norm, 1e-8, tokens),
        (gemm, indices[:, :4].contiguous(), weights, norm, 1e-8, tokens),
        (gemm, indices, weights.half(), norm, 1e-8, tokens),
        (gemm, indices, weights, norm[:2048], 1e-8, tokens),
        (gemm, indices, weights, norm, 0.0, tokens),
        (gemm, indices, weights, norm, float("nan"), tokens),
        (gemm, indices, weights, norm, 1e-8, 0),
        (gemm, indices, weights, norm, 1e-8, 129),
    ]
    rejected = []
    for call in malformed:
        try:
            workspace.finalize(*call)
        except ValueError:
            rejected.append(True)
        else:
            rejected.append(False)
    actual = workspace.finalize(gemm, indices, weights, norm, 1e-8, tokens)
    expected = _reference(nccl, gemm, indices, weights, norm, 1e-8, tokens)
    ok = all(rejected) and _within_gates(actual, expected)
    ok &= _same_on_all_ranks(actual)
    return _all_ranks(ok)


def test_finalize_rejects_malformed_calls_before_launch(mpi_pool_executor) -> None:
    """Every rank rejects the same malformed call; the next valid call still works."""
    assert all(_on_every_rank(mpi_pool_executor, _malformed_calls))


def _fp32_accumulation(rank, mapping):
    workspace = FinalizeIpcWorkspace(mapping)
    tokens = 3
    rows = torch.ones(8, 4096, dtype=torch.bfloat16, device="cuda")
    norm = torch.zeros(4096, dtype=torch.bfloat16, device="cuda")
    ok = True
    # Rank partials: rank 0 contributes 1, every other rank 2^-8.
    indices = torch.full((tokens, 8), -1, dtype=torch.int32, device="cuda")
    indices[:, 0] = 0
    weights = torch.zeros(tokens, 8, device="cuda")
    weights[:, 0] = 1.0 if rank == 0 else 2.0**-8
    actual = workspace.finalize(rows, indices, weights, norm, 1e-8, tokens)
    expected = float(torch.tensor(1.0 + (RANKS - 1) * 2.0**-8).to(torch.bfloat16))
    ok &= bool((actual.float() == expected).all())
    # Slot partials on rank 0: eight slots on one row with weights [1, 2^-9 x 7] give 1.015625.
    indices = torch.zeros(tokens, 8, dtype=torch.int32, device="cuda")
    if rank != 0:
        indices.fill_(-1)
    weights = torch.full((tokens, 8), 2.0**-9, device="cuda")
    weights[:, 0] = 1.0
    actual = workspace.finalize(rows, indices, weights, norm, 1e-8, tokens)
    ok &= bool((actual.float() == 1.015625).all())
    return _all_ranks(ok)


def test_finalize_accumulates_in_fp32(mpi_pool_executor) -> None:
    """Exact sums that BF16 accumulation would round away; tolerance gates cannot see this."""
    assert all(_on_every_rank(mpi_pool_executor, _fp32_accumulation))


def _model_hook(rank, mapping, tokens):
    """The MoE's registered finalize, with forward having chosen the fused path."""
    moe = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(moe)
    moe.mapping = mapping
    moe.nvtx = lambda name: contextlib.nullcontext()
    moe.finalize_allreduce_workspace = FinalizeIpcWorkspace(mapping)
    moe._reduce_in_finalize = True
    gemm, indices, weights, norm = _inputs(tokens, rank, 4400 + tokens)
    moe.post_expert_norm_weight = norm
    moe.post_expert_norm_eps = 1e-6
    # TRTLLM-Gen dispatches BF16 routing weights.
    weights = weights.to(torch.bfloat16)
    actual = moe.finalize(
        gemm2_output=gemm,
        expanded_idx_to_permuted_idx=indices,
        token_selected_slots=None,
        routing_weights=weights,
        num_tokens=tokens,
    )
    nccl = AllReduce(mapping, strategy=AllReduceStrategy.NCCL).cuda()
    expected = _reference(nccl, gemm, indices, weights, norm, 1e-6, tokens)
    ok = actual.dtype == torch.bfloat16 and actual.shape == (tokens, 4096)
    ok &= _within_gates(actual, expected)
    ok &= _same_on_all_ranks(actual)
    return _all_ranks(ok)


@pytest.mark.parametrize("tokens", [1, 128])
def test_the_moe_finalize_hook_reduces_across_the_group(mpi_pool_executor, tokens: int) -> None:
    """The hook the MoE scheduler calls returns the group sum when forward chose the fusion."""
    assert all(_on_every_rank(mpi_pool_executor, _model_hook, tokens))
