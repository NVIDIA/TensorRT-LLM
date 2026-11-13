# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared DRAFT_PROB1 on 2 sm_107 GPUs: ``mpirun -n 2 python -m pytest -q <this file>`` (one
rank per GPU; every
test is collective, run the whole file on every rank). Skipped unless the MPI world is 2.

- the workspace builds on MNNVL (collective) and the draft runs: the mirrored draft_logits are
bitwise identical on
  every rank, the tokens are identical on every rank, the err flag stays 0;
- the verify contract: tokens int32 [G, K]; draft_probs[slot][s][:V] = softmax(lg / T) of the
mirrored bf16 logits
  (fp32 reference), rows sum to 1; T == 0 rows one-hot at their token; padding rows uniform; only
the gen slots are
  written;
- q-contract: N draws of step 0 (fresh seeds) against the stored q (chi-square over the top 32 +
remainder, p >= 1e-3);
- a CUDA graph of the draft replays the eager bits;
- a rank 3 s late does not fault (the peer's bounded waits are ~2^35 clocks);
- an injected err on one rank: every rank faults the same draft -- token 0 and a one-hot q at 0 for
every position of
  every row (each stored q equals its deterministic draw: rejection stays exact) -- fault_count 1 on
every rank, and
  recovery requires collective workspace and graph rebuild;
- the served class (dspark_markov_worker.S4WorkerMixin): _prebuild_prob1 engages (kill switch off) / does not
(kill switch on);
  the S4-logits step returns the stashed tokens with draft_probs_last_dim = V; never engaged without
rejection sampling."""

import math
import os
import time
from types import SimpleNamespace

import pytest
import torch

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:  # pragma: no cover
    COMM = None

FAM = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
WORLD, K, V, R = int(os.environ.get("M31_PROB1_TEST_WORLD", "2")), 7, 32768, 256
assert WORLD == 2, "this TP2 transfer fixture requires world 2"
VS = V // WORLD
SLOTS = 16
pytestmark = pytest.mark.skipif(
    COMM is None or COMM.Get_size() != WORLD,
    reason=f"{WORLD} MPI ranks required",
)
_S = {}


def _env():
    if "p1" in _S:
        return _S
    import importlib

    rank = COMM.Get_rank()
    local = COMM.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank, local_size = local.Get_rank(), local.Get_size()
    local.Free()
    available = torch.cuda.is_available() and torch.cuda.device_count() >= local_size
    if not COMM.allreduce(available, op=MPI.LAND):
        pytest.skip("requires one visible CUDA GPU per local MPI rank on every host")
    torch.cuda.set_device(local_rank)
    supported = torch.cuda.get_device_capability() == (10, 7)
    if not COMM.allreduce(supported, op=MPI.LAND):
        pytest.skip("every selected MPI rank GPU must be sm_107")
    from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=WORLD, rank=rank, gpus_per_node=local_size, tp_size=WORLD)
    prob1 = importlib.import_module(f"{FAM}.kernels.m3ll.prob1.draft_prob1")
    g = torch.Generator(device="cuda").manual_seed(1234)  # same weights on every rank
    w1 = (0.05 * torch.randn(V, R, device="cuda", generator=g)).to(torch.bfloat16)
    w2 = (0.05 * torch.randn(V, R, device="cuda", generator=g)).to(torch.bfloat16)
    w2s = w2[rank * VS : (rank + 1) * VS].contiguous()
    comm = _get_mnnvl_workspace_comm(mapping)
    p1 = prob1.DraftProb1.create(mapping, comm, w1, w2s, rank * VS, V, K)
    _S.update(p1=p1, prob1=prob1, mapping=mapping, rank=rank, w1=w1, w2=w2, w2s=w2s, comm=comm)
    return _S


def _inputs(rows, seed=7, temps=None, pad_row=None, base_scale=3.0):
    """Same full-vocab base logits on every rank (this rank's shard is its slice), anchors, sampling state."""
    s = _env()
    g = torch.Generator(device="cuda").manual_seed(seed)
    full = (base_scale * torch.randn(rows * K, V, device="cuda", generator=g)).to(torch.bfloat16)
    base = full[:, s["rank"] * VS : (s["rank"] + 1) * VS].contiguous()
    prev = torch.randint(0, V, (rows,), device="cuda", generator=g, dtype=torch.int64)
    slot = torch.arange(rows, dtype=torch.int64, device="cuda") + 3
    dummy = 0
    if pad_row is not None:
        dummy = SLOTS - 1
        slot[pad_row] = dummy
    offsets = 1000 * torch.arange(rows, dtype=torch.int64, device="cuda")
    t = torch.ones(rows, dtype=torch.float32, device="cuda") if temps is None else temps
    seeds = 17 + torch.arange(rows, dtype=torch.int64, device="cuda")
    dp = torch.zeros(SLOTS, K, V, dtype=torch.float32, device="cuda")
    return SimpleNamespace(
        base=base, prev=prev, slot=slot, dummy=dummy, offsets=offsets, t=t, seeds=seeds, dp=dp
    )


def _run(x):
    return _env()["p1"].run(x.base, x.prev, x.slot, x.dummy, x.offsets, 0, 32, x.t, x.seeds, x.dp)


def _gather_eq(t: torch.Tensor) -> bool:
    """Bitwise equality of a tensor across the ranks (md5 of its bytes)."""
    import hashlib

    h = hashlib.md5(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return len(set(COMM.allgather(h))) == 1


def _assert_on_every_rank(**checks: bool) -> None:
    """Assert ``checks`` on every rank together: the ranks' failed check names are gathered first, so a rank whose
    local check fails never leaves its peer blocked in the test's next collective."""
    failed = COMM.allgather(sorted(name for name, ok in checks.items() if not ok))
    by_rank = {rank: names for rank, names in enumerate(failed) if names}
    assert not by_rank, f"failed checks by rank: {by_rank}"


def test_draft_mirror_tokens_and_contract():
    s = _env()
    rows = 5
    temps = torch.tensor([1.0, 0.7, 0.0, 1.3, 1.0], device="cuda")
    x = _inputs(rows, temps=temps, pad_row=4)
    tok = _run(x).clone()
    torch.cuda.synchronize()
    p1 = s["p1"]
    cache = p1.buf["cache"][:rows]
    _assert_on_every_rank(
        tokens_int32=tok.dtype == torch.int32, tokens_shape=tuple(tok.shape) == (rows, K)
    )
    assert _gather_eq(cache) and _gather_eq(tok) and _gather_eq(x.dp)
    assert int(p1.buf["err"].item()) == 0 and int(p1.buf["fault"].item()) == 0
    for r in range(rows):
        slot = int(x.slot[r])
        for st in range(K):
            q = x.dp[slot, st]
            if r == 4:  # padding
                assert torch.allclose(q, torch.full_like(q, 1.0 / V))
                continue
            if float(temps[r]) == 0.0:  # greedy row: the draw is the argmax, q one-hot at it
                assert float(q[int(tok[r, st])]) == 1.0 and float(q.sum()) == 1.0
                assert int(tok[r, st]) == int(cache[r, st].float().argmax())
                continue
            ref = torch.softmax(cache[r, st].double() / float(temps[r]), -1)
            assert torch.allclose(q.double(), ref, rtol=1e-4, atol=1e-7), (r, st)
            assert abs(float(q.sum()) - 1.0) < 1e-4 and float(q[int(tok[r, st])]) > 0.0
    written = {int(v) for v in x.slot}
    for slot in range(SLOTS):
        if slot not in written:
            assert float(x.dp[slot].abs().sum()) == 0.0


def _chi2_p(stat, dof):
    """Wilson-Hilferty upper tail."""
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def test_q_contract_step0_draws():
    n = 2048
    x = _inputs(1, seed=11, base_scale=4.0)
    counts = torch.zeros(V, dtype=torch.float64)
    q0 = None
    for i in range(n):
        x.seeds.fill_(100003 + i)
        tok = _run(x)
        if q0 is None:
            torch.cuda.synchronize()
            q0 = x.dp[int(x.slot[0]), 0].double().cpu()
        counts[int(tok[0, 0])] += 1
    assert _gather_eq(counts)
    top = torch.topk(q0, 32).indices
    exp = torch.cat([q0[top], (1 - q0[top].sum()).clamp(min=0).view(1)]) * n
    obs = torch.cat([counts[top], (n - counts[top].sum()).view(1)])
    keep = exp > 5
    stat = float((((obs - exp) ** 2) / exp.clamp(min=1e-12))[keep].sum())
    dof = int(keep.sum()) - 1
    assert _chi2_p(stat, dof) >= 1e-3, (stat, dof)


def test_graph_replay_matches_eager():
    x = _inputs(3, seed=21)
    eager_tok = _run(x).clone()
    eager_dp = x.dp.clone()
    torch.cuda.synchronize()
    x.dp.zero_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(x)  # warm on the side stream
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(x)
    for _ in range(32):
        x.dp.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager_tok) and torch.equal(x.dp, eager_dp)
    assert int(_env()["p1"].buf["fault_count"].item()) == 0


def test_late_rank_does_not_fault():
    s = _env()
    x = _inputs(2, seed=31)
    before = int(s["p1"].buf["fault_count"].item())
    COMM.Barrier()
    if s["rank"] == WORLD - 1:
        time.sleep(3.0)
    _run(x)
    torch.cuda.synchronize()
    assert int(s["p1"].buf["fault_count"].item()) == before
    assert int(s["p1"].buf["err"].item()) == 0


def test_injected_fault_is_sticky_exact_until_collective_rebuild():
    s = _env()
    p1 = s["p1"]
    rows = 4
    x = _inputs(rows, seed=41, temps=torch.tensor([1.0, 0.0, 0.8, 1.0], device="cuda"), pad_row=3)
    before = int(p1.buf["fault_count"].item())
    COMM.Barrier()
    if s["rank"] == 0:
        p1.buf["err"].fill_(1)  # as the peer kernel's bounded wait would leave it
    torch.cuda.synchronize()
    tok = _run(x).clone()
    torch.cuda.synchronize()
    # Every position of every row: the stored q equals its deterministic draw (token 0).
    draws_exact = all(
        float(x.dp[int(x.slot[r]), st][0]) == 1.0
        and float(x.dp[int(x.slot[r]), st].sum()) == 1.0
        and int(tok[r, st]) == 0
        for r in range(rows)
        for st in range(K)
    )
    _assert_on_every_rank(
        fault_count=int(p1.buf["fault_count"].item()) == before + 1,  # every rank
        fault=int(p1.buf["fault"].item()) == 1,
        tokens_zero=int(tok.abs().sum()) == 0,
        draws_exact=draws_exact,
    )
    # A failed workspace stays failed; never reset peer buffers while a late rank may still use them.
    x.dp.zero_()
    again = _run(x).clone()
    torch.cuda.synchronize()
    _assert_on_every_rank(
        fault_count_again=int(p1.buf["fault_count"].item()) == before + 2,
        fault_again=int(p1.buf["fault"].item()) == 1,
        tokens_zero_again=int(again.abs().sum()) == 0,
    )
    assert _gather_eq(x.dp)
    line = p1.fault_line()
    _assert_on_every_rank(
        fault_line=line is not None and f"faults {before + 2}" in line and "rebuild" in line
    )
    # A collective workspace rebuild is the only recovery; captured graphs must also be rebuilt by serving.
    COMM.Barrier()
    fresh = s["prob1"].DraftProb1.create(
        s["mapping"], s["comm"], s["w1"], s["w2s"], s["rank"] * VS, V, K
    )
    s["p1"] = fresh
    x.dp.zero_()
    _run(x)
    torch.cuda.synchronize()
    _assert_on_every_rank(
        fresh_fault_count=int(fresh.buf["fault_count"].item()) == 0,
        fresh_fault=int(fresh.buf["fault"].item()) == 0,
    )
    assert _gather_eq(x.dp)
    q = x.dp[int(x.slot[0]), 0]
    assert abs(float(q.sum()) - 1.0) < 1e-4 and float(q.max()) < 1.0


def test_served_class_engagement_and_kill_switch(monkeypatch):
    import importlib

    s = _env()
    s4 = importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp2.dspark_markov_worker"
    )
    prob1 = s["prob1"]

    class Worker(s4.S4WorkerMixin):
        pass

    w = Worker()
    w.mapping = s["mapping"]
    w._s4_logits_on = True
    w._prebuild_prob1(s["mapping"], s["w1"], s["w2s"], s["rank"] * VS, V, K)
    _assert_on_every_rank(
        engaged=w._p1 is not None and w._p1.describe().startswith(f"{prob1.TAG} engaged")
    )
    rows = 2
    x = _inputs(rows, seed=51)
    buf = torch.full((rows, K, VS + 64), float("-inf"), dtype=torch.bfloat16, device="cuda")
    buf[:, :, :VS].copy_(x.base.view(rows, K, VS))
    state = SimpleNamespace(shard=VS, block=K, buffer=lambda b: buf)
    md = SimpleNamespace(
        use_rejection_sampling=True,
        wants_advanced_draft_sampling=True,
        draft_probs=x.dp,
        batch_slot_ids=x.slot,
        request_seeds=x.seeds,
        request_offsets=x.offsets,
        request_temperatures=x.t,
        dummy_slot_row=0,
        rng_draft_slot=lambda i: 0,
        draft_probs_last_dim=0,
    )
    w._s4_step, w._s4_step_mode, w._s4_step_meta = state, "logits", (0, rows, md)
    gen_logits = buf[:, :, :VS]
    out = w._refine_block_logits(None, gen_logits, {"first_prev_tokens": x.prev}, md)
    torch.cuda.synchronize()
    _assert_on_every_rank(
        logits_returned=out is gen_logits,
        draft_probs_last_dim=md.draft_probs_last_dim == V,
        tokens=w._s4_tokens is not None
        and tuple(w._s4_tokens.shape) == (rows, K)
        and w._s4_tokens.dtype == torch.int32,
    )
    assert _gather_eq(w._s4_tokens)
    assert (
        w.sample_draft_tokens(gen_logits, md, rows) is not None
    )  # returns the stashed tokens, no stock sampler
    # never under strict / exact-match acceptance
    md_strict = SimpleNamespace(**{**vars(md), "use_rejection_sampling": False})
    assert w._p1_engages(md_strict, rows, state) is False
    # kill switch (read at import; patched alike on every rank): nothing set up
    monkeypatch.setattr(prob1, "_DISABLED", True)
    w2 = Worker()
    w2.mapping = s["mapping"]
    w2._s4_logits_on = True
    w2._prebuild_prob1(s["mapping"], s["w1"], s["w2s"], s["rank"] * VS, V, K)
    assert w2._p1 is None


@pytest.mark.parametrize("boundary", [16384, 32768])
def test_full_epoch_tags_reject_unused_rows_at_old_wrap(boundary):
    s = _env()
    COMM.Barrier()
    fresh = s["prob1"].DraftProb1.create(
        s["mapping"], s["comm"], s["w1"], s["w2s"], s["rank"] * VS, V, K
    )
    s["p1"] = fresh
    assert fresh.buf["xs"].numel() == 256
    fresh.buf["epoch"].fill_(boundary - 1)
    torch.cuda.synchronize()
    COMM.Barrier()
    # Row1 was never used and its full tag is still zero. Low14-bit payload epochs alias here.
    x = _inputs(2, seed=boundary + 3)
    tokens = _run(x).clone()
    torch.cuda.synchronize()
    _assert_on_every_rank(
        no_fault=int(fresh.buf["fault"].item()) == 0,
        tokens_in_vocab=bool(((tokens >= 0) & (tokens < V)).all()),
    )
    assert _gather_eq(tokens) and _gather_eq(x.dp)


def test_repeated_fresh_calls_preserve_slow_reader_caches():
    """Clone each call locally; no host barrier may protect the async producer/reader sequence."""
    s = _env()
    cases = [_inputs(rows, seed=101 + i) for i, rows in enumerate([1, 2, 4, 2])]
    expected = []
    for x in cases:
        COMM.Barrier()
        tokens = _run(x).clone()
        probs = x.dp.clone()
        torch.cuda.synchronize()
        expected.append((tokens, probs))
    COMM.Barrier()
    observed = []
    for repeat in range(4):
        for i, x in enumerate(cases):
            if s["rank"] == 0 and (repeat + i) % 2 == 0:
                torch.cuda._sleep(500000)  # skew rank submission without a host-side collective
            observed.append((_run(x).clone(), x.dp.clone()))
    torch.cuda.synchronize()
    assert int(s["p1"].buf["fault"].item()) == 0
    for i, (tokens, probs) in enumerate(observed):
        want_tokens, want_probs = expected[i % len(cases)]
        assert torch.equal(tokens, want_tokens) and torch.equal(probs, want_probs)


@pytest.mark.parametrize("bad_slot", [-1, SLOTS])
@pytest.mark.parametrize("all_ranks_invalid", [True, False])
def test_invalid_slots_fault_without_writing_outside_draft_probs(bad_slot, all_ranks_invalid):
    s = _env()
    x = _inputs(2, seed=81)
    backing = torch.full((SLOTS + 2, K, V), -7.0, device="cuda")
    x.dp = backing[1:-1]
    x.dp.zero_()
    if all_ranks_invalid or s["rank"] == 0:
        x.slot[0] = bad_slot
    before = int(s["p1"].buf["fault_count"].item())
    try:
        tok = _run(x).clone()
        torch.cuda.synchronize()
        _assert_on_every_rank(
            fault=int(s["p1"].buf["fault"].item()) == 1,
            fault_count=int(s["p1"].buf["fault_count"].item()) == before + 1,
            tokens_zero=int(tok.abs().sum()) == 0,
        )
        assert _gather_eq(tok)
        assert bool((backing[0] == -7).all()) and bool((backing[-1] == -7).all())
        q = x.dp[int(x.slot[1])]
        assert bool((q[:, 0] == 1).all()) and bool((q.sum(-1) == 1).all())
    finally:
        COMM.Barrier()
        s["p1"] = s["prob1"].DraftProb1.create(
            s["mapping"], s["comm"], s["w1"], s["w2s"], s["rank"] * VS, V, K
        )


@pytest.mark.parametrize("dummy", [-1, SLOTS])
def test_invalid_dummy_slot_is_rejected_before_launch(dummy):
    x = _inputs(1, seed=91)
    x.dummy = dummy
    with pytest.raises(RuntimeError, match="dummy_slot_row"):
        _run(x)
