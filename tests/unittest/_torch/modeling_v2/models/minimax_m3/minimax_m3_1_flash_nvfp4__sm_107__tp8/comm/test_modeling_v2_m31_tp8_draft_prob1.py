# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_PROB1 (TP8) on 8 sm_107 GPUs over two 4-GPU nodes (MNNVL), ``mpirun -n 8 python -m pytest -q <this
file>`` (one rank per GPU; every test is collective, run the whole file on every rank), or on one 4-GPU node with
``-n 4`` (the w8 translation unit at exchange world 4: the TP8 op is forced for world 4 too). Skipped unless the MPI
world is 8 or 4. The TP8 op takes up to 4 rows (dkmkpw::Lim<8>::kMaxC).

- the workspace builds on MNNVL (collective) and the draft runs: the mirrored draft_logits are bitwise identical on
  every rank, the tokens are identical on every rank, the err flag stays 0;
- the verify contract: tokens int32 [G, K]; draft_probs[slot][s][:V] = softmax(lg / T) of the mirrored bf16 logits
  (fp32 reference), rows sum to 1; T == 0 rows one-hot at their token; padding rows uniform; only the gen slots are
  written;
- q-contract: N draws of step 0 (fresh seeds) against the stored q (chi-square over the top 32 + remainder, p >= 1e-3);
- a CUDA graph of the draft replays the eager bits;
- a rank 3 s late does not fault (the peer's bounded waits are ~2^35 clocks);
- an injected err on one rank: every rank faults the same draft -- token 0 and a one-hot q at 0 for every position of
  every row (each stored q equals its deterministic draw: rejection stays exact) -- fault_count 1 on every rank, and
  the next draft is clean;
- the worker class (dspark_markov_worker.S4WorkerMixin): _prebuild_prob1 engages (kill switch off) / does not (kill
switch on);
  the S4-logits step returns the stashed tokens with draft_probs_last_dim = V; never engaged without rejection sampling.
"""

import math
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
WORLD = COMM.Get_size() if COMM is not None else 0
K, V, R = 7, 32768, 256
VS = V // max(WORLD, 1)
SLOTS = 16
pytestmark = pytest.mark.skipif(
    COMM is None
    or WORLD not in (4, 8)
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7),
    reason="8 (or 4) MPI ranks on sm_107",
)
_S = {}


def _env():
    if "p1" in _S:
        return _S
    import importlib

    from tensorrt_llm._torch.distributed.ops import _get_mnnvl_workspace_comm
    from tensorrt_llm.mapping import Mapping

    rank = COMM.Get_rank()
    gpn = min(torch.cuda.device_count(), WORLD)
    torch.cuda.set_device(rank % gpn)
    mapping = Mapping(world_size=WORLD, rank=rank, gpus_per_node=gpn, tp_size=WORLD)
    prob1 = importlib.import_module(f"{FAM}.kernels.m3ll.prob1.draft_prob1")
    prob1.TOPOLOGIES[WORLD] = prob1.TOPOLOGIES[
        8
    ]  # the w8 unit at world 4 too (its runtime world <= 8)
    g = torch.Generator(device="cuda").manual_seed(1234)  # same weights on every rank
    w1 = (0.05 * torch.randn(V, R, device="cuda", generator=g)).to(torch.bfloat16)
    w2 = (0.05 * torch.randn(V, R, device="cuda", generator=g)).to(torch.bfloat16)
    w2s = w2[rank * VS : (rank + 1) * VS].contiguous()
    comm = _get_mnnvl_workspace_comm(mapping)
    p1 = prob1.DraftProb1.create(mapping, comm, w1, w2s, rank * VS, V, K)
    assert p1.op == prob1.OP_TP8 and p1.max_rows == 4 and tuple(p1.buf["cache"].shape) == (4, K, V)
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


def test_draft_mirror_tokens_and_contract():
    s = _env()
    rows = 4
    temps = torch.tensor([1.0, 0.0, 1.3, 1.0], device="cuda")
    x = _inputs(rows, temps=temps, pad_row=3)
    tok = _run(x).clone()
    torch.cuda.synchronize()
    p1 = s["p1"]
    cache = p1.buf["cache"][:rows]
    assert tok.dtype == torch.int32 and tuple(tok.shape) == (rows, K)
    assert _gather_eq(cache) and _gather_eq(tok) and _gather_eq(x.dp)
    assert int(p1.buf["err"].item()) == 0 and int(p1.buf["fault"].item()) == 0
    for r in range(rows):
        slot = int(x.slot[r])
        for st in range(K):
            q = x.dp[slot, st]
            if r == 3:  # padding
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


def test_injected_fault_is_agreed_exact_and_recovers():
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
    assert int(p1.buf["fault_count"].item()) == before + 1  # every rank
    assert int(p1.buf["fault"].item()) == 1 and int(p1.buf["err"].item()) == 0
    assert int(tok.abs().sum()) == 0
    for r in range(
        rows
    ):  # every position of every row: the stored q equals its deterministic draw (token 0)
        for st in range(K):
            q = x.dp[int(x.slot[r]), st]
            assert float(q[0]) == 1.0 and float(q.sum()) == 1.0 and int(tok[r, st]) == 0
    # the next draft is clean (state reset on every rank alike)
    x.dp.zero_()
    _run(x)
    torch.cuda.synchronize()
    assert int(p1.buf["fault_count"].item()) == before + 1 and int(p1.buf["fault"].item()) == 0
    assert _gather_eq(x.dp)
    q = x.dp[int(x.slot[0]), 0]
    assert abs(float(q.sum()) - 1.0) < 1e-4 and float(q.max()) < 1.0
    line = p1.fault_line()  # host read at a safe point
    assert line is not None and f"faults {before + 1}" in line


def test_worker_class_engagement_and_kill_switch(monkeypatch):
    import importlib

    s = _env()
    s4 = importlib.import_module(
        f"{FAM}.minimax_m3_1_flash_nvfp4__sm_107__tp8.dspark_markov_worker"
    )
    prob1 = s["prob1"]

    class Worker(s4.S4WorkerMixin):
        pass

    w = Worker()
    w.mapping = s["mapping"]
    w._s4_logits_on = True
    w._prebuild_prob1(s["mapping"], s["w1"], s["w2s"], s["rank"] * VS, V, K)
    assert w._p1 is not None and w._p1.describe().startswith(f"{prob1.TAG} engaged: tp8")
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
    assert out is gen_logits and md.draft_probs_last_dim == V
    assert (
        w._s4_tokens is not None
        and tuple(w._s4_tokens.shape) == (rows, K)
        and w._s4_tokens.dtype == torch.int32
    )
    assert _gather_eq(w._s4_tokens)
    assert (
        w.sample_draft_tokens(gen_logits, md, rows) is not None
    )  # returns the stashed tokens, no stock sampler
    # never under strict / exact-match acceptance
    md_strict = SimpleNamespace(**{**vars(md), "use_rejection_sampling": False})
    assert w._p1_engages(md_strict, rows, state) is False
    assert w._p1_engages(md, 5, state) is False  # 4 rows at TP8
    # kill switch (read at import; patched alike on every rank): nothing set up
    monkeypatch.setattr(prob1, "_DISABLED", True)
    w2 = Worker()
    w2.mapping = s["mapping"]
    w2._s4_logits_on = True
    w2._prebuild_prob1(s["mapping"], s["w1"], s["w2s"], s["rank"] * VS, V, K)
    assert w2._p1 is None
