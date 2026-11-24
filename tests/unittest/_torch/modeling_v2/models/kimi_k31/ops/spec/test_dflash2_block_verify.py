# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 block-verification entry (CuTe DSL).

Every certified cell is gated bitwise against the entry's reference (the family's torch
``dflash2_block_verify``): the same accept counts and the same bonus tokens for the same
uniforms, over several draws per cell. The family's own verification tests (the dense float64
Algorithm 2 and the chi-square losslessness of selector walk + verification) are rerun with the
kernel in place of the torch function.
"""

import importlib.util
import pathlib
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
    dflash2_block_verify as verify_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_block_verify import (  # noqa: E501
    dflash2_block_verify as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_block_verify import (  # noqa: E501
    dflash2_block_verify_seeded as seeded_op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)

#: Draws per certified cell (fresh target rows, proposals and uniforms each); single-request
#: cells take more so their accept counts cover more than tau = 0.
_DRAWS = 4
_DRAWS_SINGLE = 24


def _build(spec, seed):
    """Target rows, a selector-like proposal over each slot's top-k candidates and its draws."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n, g, vocab, k = spec["batch"], spec["drafts"], spec["vocab"], spec["top_k"]
    temp = spec["temperature"]
    logits = 3.0 * torch.randn(n, g + 1, vocab, device="cuda", generator=gen)
    probs = torch.softmax(logits / temp, dim=-1)
    if spec["top_p_floor"] > 0:
        probs = torch.where(probs < spec["top_p_floor"], torch.zeros_like(probs), probs)
        probs = probs / probs.sum(-1, keepdim=True)
    if spec["padding_rows"]:
        probs[: spec["padding_rows"]] = 1.0 / vocab
    if spec["degenerate_rows"]:
        # Requests 0, 2, ... get all-zero target rows; 1, 3, ... a NaN in one row past the
        # first. Neither leaves mass to draw the bonus from.
        for b in range(spec["degenerate_rows"]):
            if b % 2 == 0:
                probs[b] = 0.0
            else:
                probs[b, 1 + b % g, 1000 + 37 * b] = float("nan")
    draft_logits = logits[:, :g] + spec["closeness"] * torch.randn(
        n, g, vocab, device="cuda", generator=gen
    )
    cand = draft_logits.topk(k, dim=-1).indices
    q = torch.softmax(draft_logits.gather(-1, cand) / temp, dim=-1)
    pick = torch.multinomial(q.reshape(-1, k), 1, generator=gen).view(n, g)
    drafts = cand.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
    if spec["stale"] > 0:
        # DFlash2ProposalStore.load's point mass for a request the walk did not draft.
        stale = torch.rand(n, device="cuda", generator=gen) < spec["stale"]
        cand = torch.where(stale[:, None, None], drafts[..., None].expand_as(cand), cand)
        lane0 = (torch.arange(k, device="cuda") == 0).float().expand_as(q)
        q = torch.where(stale[:, None, None], lane0, q)
    if spec["outside"] > 0:
        outside = torch.rand(n, g, device="cuda", generator=gen) < spec["outside"]
        drafts = torch.where(outside, (drafts + 7) % vocab, drafts)
    # The worker slices one [n, K + 1] uniform block: strided coins and final coins.
    uniforms = torch.rand(n, g + 1, device="cuda", generator=gen)
    return (
        probs.contiguous(),
        drafts.to(spec["draft_dtype"]),
        cand.to(spec["cand_dtype"]),
        q,
        uniforms[:, :g],
        uniforms[:, g],
    )


def _single_request_reference(entry, args, coin_args):
    """A single request: ``tau`` from the reference on it (torch's one-row sums are
    deterministic), the bonus from the reference on the request duplicated to two rows, where
    torch.cumsum takes its deterministic scan (the one the kernel reproduces); the duplicated
    batch's own sums may round its ``h`` differently, so its bonus counts only where its ``tau``
    agrees (``None`` otherwise)."""
    probs, drafts, cand, q, *_ = args
    tau, _ = entry.reference(*args)
    dup = [a.expand(2, *a.shape[1:]) for a in coin_args]
    dup_tau, dup_bonus = entry.reference(
        probs.expand(2, -1, -1).contiguous(),
        drafts.expand(2, -1),
        cand.expand(2, -1, -1),
        q.expand(2, -1, -1),
        *dup,
    )
    return tau, (dup_bonus[:1] if int(dup_tau[0]) == int(tau[0]) else None)


def _reference(args):
    """The entry's reference (a single request: :func:`_single_request_reference`)."""
    if args[0].shape[0] != 1:
        return op.reference(*args)
    return _single_request_reference(op, args, args[4:6])


def _assert_equal(got, want, what="") -> None:
    for name, a, b in zip(("tau", "bonus"), got, want):
        if b is None:
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape, a.dtype, b.dtype)
        bad = (a != b).nonzero().flatten().tolist()
        assert not bad, (
            f"{what} {name}: requests {bad[:8]} differ: {a[bad[:8]].tolist()} vs {b[bad[:8]].tolist()}"
        )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    base = zlib.crc32(cell.why.encode()) % 2**31
    taus = []
    for draw in range(_DRAWS if cell.spec["batch"] > 1 else _DRAWS_SINGLE):
        args = _build(cell.spec, seed=base + draw)
        got = op(*args)
        _assert_equal(got, _reference(args), f"draw {draw}")
        taus.append(got[0])
    print(
        f"accept counts: {torch.bincount(torch.cat(taus), minlength=cell.spec['drafts'] + 1).tolist()}"
    )


def _seeded_args(args, seed):
    """The cell's tensors with full-range int64 seeds and offsets in place of the coins."""
    gen = torch.Generator().manual_seed(seed)
    n = args[0].shape[0]
    seeds = torch.randint(-(2**63), 2**63 - 1, (n,), generator=gen, dtype=torch.int64)
    offsets = torch.randint(0, 2**40, (n,), generator=gen, dtype=torch.int64)
    return (*args[:4], seeds.cuda(), offsets.cuda())


def _seeded_reference(args):
    if args[0].shape[0] != 1:
        return seeded_op.reference(*args)
    return _single_request_reference(seeded_op, args, args[4:6])


@pytest.mark.parametrize("cell", seeded_op.CELLS, ids=[c.why[:48] for c in seeded_op.CELLS])
def test_seeded_cells(cell, certifying) -> None:  # noqa: F811
    certifying(seeded_op)
    base = zlib.crc32(cell.why.encode()) % 2**31
    for draw in range(_DRAWS if cell.spec["batch"] > 1 else _DRAWS_SINGLE):
        args = _seeded_args(_build(cell.spec, seed=base + draw), seed=base + draw)
        _assert_equal(seeded_op(*args), _seeded_reference(args), f"draw {draw}")


def test_seeded_one_launch_per_call() -> None:
    args = _seeded_args(_build(seeded_op.CELLS[2].spec, seed=1), seed=1)
    seeded_op(*args), seeded_op.reference(*args)
    torch.cuda.synchronize()
    counts = {}
    for name, fn in (("op", seeded_op), ("reference", seeded_op.reference)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        counts[name] = sum(
            1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        )
    print(f"device launches per call (seeded): {counts}")
    assert counts["op"] == 1, counts


def _reference_h(target_probs, draft_tokens, candidate_ids, q):
    """``h_i`` of the interior rows exactly as the reference computes it (its own torch ops)."""
    n, rows, vocab = target_probs.shape
    g = rows - 1
    probs = target_probs.float()
    drafts = draft_tokens.long().clamp(0, vocab - 1)
    cand = candidate_ids.long().clamp(0, vocab - 1)
    q = torch.where((q >= 0) & (q <= 1), q.float(), 0.0)
    p_draft = probs[:, :g].gather(-1, drafts.unsqueeze(-1)).squeeze(-1)
    q_draft = torch.where(cand == drafts.unsqueeze(-1), q, 0.0).sum(dim=-1)
    q_ok = (q_draft > 0) & (q_draft <= 1)
    prefix = torch.ones(n, dtype=torch.float32, device=probs.device)
    ratios = []
    for i in range(g):
        prefix = torch.where(
            q_ok[:, i], torch.clamp(prefix * p_draft[:, i] / q_draft[:, i], max=1.0), 0.0
        )
        ratios.append(prefix)
    r_mid = torch.stack(ratios, dim=1)[:, : g - 1]
    p_rows = probs[:, 1:g]
    overlap = torch.minimum(r_mid.unsqueeze(-1) * p_rows.gather(-1, cand[:, 1:g]), q[:, 1:g])
    residual = (r_mid * p_rows.sum(dim=-1) - overlap.sum(dim=-1)).clamp(min=0.0)
    interior = (r_mid > 0) & (r_mid < 1)
    return torch.where(interior, residual / (residual + 1.0 - r_mid), r_mid), interior


@pytest.mark.parametrize("batch,drafts", [(1, 4), (1, 7), (3, 7), (8, 4), (64, 7)])
def test_knife_edge_coins(batch, drafts) -> None:
    """Acceptance coins at the reference's own ``h_i`` (rejects) or one float32 ulp below it
    (accepts): ``tau`` matches only if every ``h_i`` is bitwise the reference's, including the
    vocabulary sum of each interior row (torch's reduction order)."""
    spec = dict(op.CELLS[0].spec, batch=batch, drafts=drafts, closeness=0.3, temperature=0.9)
    edges = 0
    for draw in range(6):
        probs, d, cand, q, coins, finals = _build(spec, seed=1000 + draw)
        h, interior = _reference_h(probs, d, cand, q)
        gen = torch.Generator(device="cuda").manual_seed(draw)
        at = torch.rand(h.shape, device="cuda", generator=gen) < 0.5
        edge = torch.where(at, h, torch.nextafter(h, torch.zeros_like(h)))
        use = interior & (edge < 1)
        coins = coins.clone()
        coins[:, : drafts - 1] = torch.where(use, edge, coins[:, : drafts - 1])
        edges += int(use.sum())
        args = (probs, d, cand, q, coins, finals)
        _assert_equal(op(*args), _reference(args), f"draw {draw}")
    assert edges > 0


def test_one_launch_per_call() -> None:
    """The op is one kernel; the reference's launch count is printed."""
    args = _build(op.CELLS[5].spec, seed=1)
    op(*args), op.reference(*args)
    torch.cuda.synchronize()
    counts = {}
    for name, fn in (("op", op), ("reference", op.reference)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        counts[name] = sum(
            1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        )
    print(f"device launches per call: {counts}")
    assert counts["op"] == 1, counts


def test_cuda_graph_replay() -> None:
    """Captured after one eager call, replayed on new rows, proposals and uniforms."""
    spec = op.CELLS[3].spec
    args = _build(spec, seed=2)
    op(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    for seed in (3, 4, 5):
        fresh = _build(spec, seed=seed)
        for dst, src in zip(args[:4], fresh[:4]):
            dst.copy_(src)
        args[4].copy_(fresh[4])
        args[5].copy_(fresh[5])
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal(out, op.reference(*args), f"replay {seed}")


def test_workspace_bounded_over_batch_sizes() -> None:
    """Every batch size slices the workspace of its power-of-two request capacity, so the cache stays bounded
    however the batch varies; each call keeps the reference's bits, and a graph captured at a batch size whose
    workspace a larger eager call made replays correctly."""
    spec = op.CELLS[2].spec
    before = set(verify_kernel._WORKSPACES)
    for n in (7, 3, 8, 1, 5, 2, 6):
        args = _build(dict(spec, batch=n), seed=40 + n)
        _assert_equal(op(*args), _reference(args), f"batch {n}")
    added = set(verify_kernel._WORKSPACES) - before
    assert {key[1] for key in added} <= {1, 2, 4, 8}, added
    assert all(key[1] & (key[1] - 1) == 0 for key in verify_kernel._WORKSPACES)
    g, vocab = spec["drafts"], spec["vocab"]
    nchunks = -(-vocab // (2 << verify_kernel.torch_scan_log_threads_x(8, vocab)))
    device = torch.device("cuda", torch.cuda.current_device())
    shared = [verify_kernel._workspace(device, n, g, nchunks, 16) for n in (5, 8)]
    assert shared[0][0].data_ptr() == shared[1][0].data_ptr()
    assert shared[0][0].numel() == 5 * (g + 1) * nchunks * 16 and shared[0][3].numel() == 5

    args = _build(dict(spec, batch=5), seed=50)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    fresh = _build(dict(spec, batch=5), seed=51)
    for dst, src in zip(args[:6], fresh[:6]):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    _assert_equal(out, op.reference(*args), "replay at 5 requests")


def test_guard_refuses_half_probs() -> None:
    args = list(_build(op.CELLS[2].spec, seed=6))
    args[0] = args[0].half()
    with validating(op), pytest.raises(AssertionError, match="float32"):
        op(*args)


def test_guard_refuses_misaligned_rows() -> None:
    args = list(_build(op.CELLS[2].spec, seed=7))
    wide = torch.zeros(args[0].shape[0], args[0].shape[1], args[0].shape[2] + 1, device="cuda")
    args[0] = wide[..., 1:]
    with validating(op), pytest.raises(AssertionError, match="16-byte aligned"):
        op(*args)


# ---- the family's verification tests, rerun on the kernel -------------------------------------

_FAMILY_TEST = pathlib.Path(__file__).resolve().with_name("test_dflash2_verify.py")


def _family_tests():
    spec = importlib.util.spec_from_file_location("_k31_family_dflash2_verify", _FAMILY_TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernel_verify(target_probs, draft_tokens, candidate_ids, q, coins, final_coins):
    return op(target_probs, draft_tokens, candidate_ids, q, coins, final_coins)


def test_family_lossless_chi_square_on_kernel(monkeypatch) -> None:
    """Selector walk + kernel verification reproduce the target's joint law (400000 rows)."""
    family = _family_tests()
    monkeypatch.setattr(family, "dflash2_block_verify", _kernel_verify)
    family.test_verify_is_lossless()


@pytest.mark.parametrize("g", [4, 7])
def test_family_dense_reference_on_kernel(g, monkeypatch) -> None:
    """The dense float64 Algorithm 2 of every request (vocab 200, 32 candidates)."""
    family = _family_tests()
    monkeypatch.setattr(family, "dflash2_block_verify", _kernel_verify)
    family.test_verify_matches_dense_reference(g)
