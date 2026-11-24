# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the DFlash2 step-bookkeeping entries (k31_dflash2_accept_commit, k31_dflash2_accept_record,
k31_dflash2_draft_tail): every cell bitwise against its torch reference, in-place state included; the accept commit
against the engine's ``_apply_force_accepted_tokens`` over several steps of the forced pool; the record against
``KDAReplayState.record_acceptance``; CUDA-graph replay."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec import dflash2_verify
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_step_bookkeeping import (  # noqa: E501
    FORCE_COUNTER_STRIDE,
    FORCE_OFF,
    FORCE_POOL,
    FORCE_SLOT_STRIDE,
    k31_dflash2_accept_commit,
    k31_dflash2_accept_record,
    k31_dflash2_draft_tail,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KDAReplayState
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD as THR
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the K3.1 DFlash2 bookkeeping entries are certified on sm_103"
)

V = 163840


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


class _ForceWorker:
    """The engine's forced-acceptance methods and state (what the K3.1 workers inherit)."""

    _apply_force_accepted_tokens = SpecWorkerBase._apply_force_accepted_tokens
    _ensure_force_accept_rng_state = SpecWorkerBase._ensure_force_accept_rng_state
    _real_draft_rows = staticmethod(SpecWorkerBase._real_draft_rows)

    def __init__(self, forced: float):
        self.force_num_accepted_tokens = forced
        self._force_accept_rng_pool = None
        self._force_accept_rng_counter = None


def _temperatures(kind, n, g):
    if kind == "hot":
        return 0.3 + torch.rand(n, generator=g, device="cuda")
    if kind == "edges":
        t = torch.tensor(THR, dtype=torch.float32)
        values = [
            t,
            torch.nextafter(t, torch.tensor(1.0)),
            torch.tensor(0.0),
            torch.tensor(float("nan")),
        ]
        return torch.stack(values).repeat(n)[:n].cuda()
    temps = torch.rand(n, generator=g, device="cuda")
    temps[::3] = 0.0
    return temps


def _commit_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, K = spec["n"], spec["K"]
    tau = torch.randint(0, K + 1, (n,), generator=g, device="cuda")
    bonus = torch.randint(0, V, (n,), generator=g, device="cuda")
    gtau = torch.randint(0, K + 1, (n,), generator=g, device="cuda")
    gbonus = torch.randint(0, V, (n,), generator=g, device="cuda")
    temps = _temperatures(spec["temps"], n, g)
    drafts = torch.randint(0, V, (n, K), generator=g, device="cuda").to(spec["draft_dtype"])
    worker = _ForceWorker(spec["forced"])
    meta = types.SimpleNamespace(is_cuda_graph=False)
    mode, base, mx, frac, counter, pool = dflash2_verify._force_args(
        worker, K, meta, torch.device("cuda")
    )
    assert mode == spec["force"]
    if counter is None:
        counter = torch.zeros(1, dtype=torch.int64, device="cuda")
        pool = torch.zeros(1, device="cuda")
    else:
        counter.fill_(int(torch.randint(0, 1 << 40, (1,), generator=g, device="cuda")))
    return [
        tau,
        bonus,
        gtau,
        gbonus,
        temps,
        THR,
        drafts,
        mode,
        base,
        mx,
        frac,
        counter,
        pool,
        FORCE_COUNTER_STRIDE,
        FORCE_SLOT_STRIDE,
    ], worker


@pytest.mark.parametrize(
    "cell",
    k31_dflash2_accept_commit.CELLS,
    ids=[c.why[:48] for c in k31_dflash2_accept_commit.CELLS],
)
def test_accept_commit_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args, _ = _commit_args(cell.spec, _seed(cell))
    ref_args = list(args)
    ref_args[11] = args[11].clone()
    certifying(k31_dflash2_accept_commit)
    acc, num = k31_dflash2_accept_commit(*args)
    acc_ref, num_ref = k31_dflash2_accept_commit.reference(*ref_args)
    assert torch.equal(acc, acc_ref) and torch.equal(num, num_ref)
    assert torch.equal(args[11], ref_args[11])  # the counter bump


@pytest.mark.parametrize("forced", [0.0, 2.21, 2.6, 3.0, 5.0])
def test_accept_commit_is_the_verify_tail(forced) -> None:
    """Five steps of one batch: the op's (tokens, counts, counter) equal the torch tail of
    ``dflash2_sample_and_accept`` followed by the engine's ``_apply_force_accepted_tokens``."""
    n, K = 7, 4
    spec = dict(n=n, K=K, force=FORCE_OFF, forced=forced, temps="mixed", draft_dtype=torch.int32)
    spec["force"] = dflash2_verify._force_args(
        _ForceWorker(forced), K, types.SimpleNamespace(is_cuda_graph=False), torch.device("cuda")
    )[0]
    args, worker = _commit_args(spec, seed=int(forced * 100))
    engine = _ForceWorker(forced)
    meta = types.SimpleNamespace(is_cuda_graph=False)
    for _ in range(5):
        if engine._force_accept_rng_pool is None and worker._force_accept_rng_pool is not None:
            engine._ensure_force_accept_rng_state(torch.device("cuda"))
            engine._force_accept_rng_counter.copy_(worker._force_accept_rng_counter)
        tau, bonus, gtau, gbonus, temps, _, drafts = args[:7]
        greedy = temps <= THR
        t = torch.where(greedy, gtau, tau)
        b = torch.where(greedy, gbonus, bonus)
        tokens = torch.cat((drafts.long(), drafts[:, -1:].long()), dim=1).scatter(
            1, t[:, None], b[:, None]
        )
        acc_ref = torch.zeros((n, K + 1), dtype=torch.int32, device="cuda")
        num_ref = torch.ones(n, dtype=torch.int32, device="cuda")
        acc_ref[0:] = tokens.to(torch.int32)
        num_ref[0:] = (t + 1).to(torch.int32)
        num_ref = engine._apply_force_accepted_tokens(num_ref, 0, K, spec_metadata=meta)
        acc, num = k31_dflash2_accept_commit(*args)
        assert torch.equal(acc, acc_ref) and torch.equal(num, num_ref)
        if engine._force_accept_rng_counter is not None:
            assert torch.equal(engine._force_accept_rng_counter, args[11])


def _record_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots = spec["n"], spec["slots"]
    num = torch.randint(1, 6, (n,), generator=g, device="cuda", dtype=torch.int32)
    if spec["dummies"] == "zero":
        num[::2] = 0
    state = torch.randperm(slots, generator=g, device="cuda")[:n].to(spec["state_dtype"])
    dummy = torch.zeros(n, dtype=torch.bool, device="cuda")
    if spec["dummies"] == "some":
        dummy[1::7] = True
    elif spec["dummies"] == "all":
        dummy[:] = True
    pitch = spec.get("prev_stride", 1)
    # The V2 manager's replay counts: one int32 per slot of a pool at the slot pitch.
    prev = torch.randint(0, 6, (slots, pitch), generator=g, device="cuda", dtype=torch.int32)[:, 0]
    kv = torch.randint(0, 300000, (n,), generator=g, device="cuda", dtype=torch.int32)
    return [num, state, dummy, prev, kv]


@pytest.mark.parametrize(
    "cell",
    k31_dflash2_accept_record.CELLS,
    ids=[c.why[:48] for c in k31_dflash2_accept_record.CELLS],
)
def test_accept_record_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _record_args(cell.spec, _seed(cell))
    ref_args = [a.clone() for a in args]
    certifying(k31_dflash2_accept_record)
    rewind = k31_dflash2_accept_record(*args)
    rewind_ref = k31_dflash2_accept_record.reference(*ref_args)
    assert torch.equal(rewind, rewind_ref)
    assert torch.equal(args[3], ref_args[3]) and torch.equal(args[4], ref_args[4])


def test_accept_record_is_the_replay_state_record() -> None:
    """The replay counts equal ``KDAReplayState.record_acceptance(slots, num - 1, is_dummy)``."""
    args = _record_args(k31_dflash2_accept_record.CELLS[1].spec, seed=5)
    num, state, dummy, prev, kv = args
    replay = types.SimpleNamespace(prev_num_accepted_tokens=prev.clone())
    KDAReplayState.record_acceptance(replay, state.to(torch.int32), num - 1, dummy)
    k31_dflash2_accept_record(*args)
    assert torch.equal(prev, replay.prev_num_accepted_tokens)


def _tail_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, K = spec["b"], spec["K"]
    rows = b + 3
    acc = torch.randint(0, V, (rows, K + 1), generator=g, device="cuda", dtype=torch.int32)
    batch = torch.randperm(rows, generator=g, device="cuda")[:b].to(spec["batch_dtype"])
    num = torch.randint(1, K + 2, (b,), generator=g, device="cuda", dtype=torch.int32)
    kv = torch.randint(0, 300000, (b,), generator=g, device="cuda", dtype=torch.int32)
    rewind = 1 - num
    if spec["counts"] == "edge":
        num[0] = 0
        kv[1] = 0
        rewind[1] = 3
    drafts = torch.randint(0, V, (b, K), generator=g, device="cuda", dtype=torch.int32)
    return [kv, rewind, acc, batch, num, drafts, spec["rewind"]]


@pytest.mark.parametrize(
    "cell", k31_dflash2_draft_tail.CELLS, ids=[c.why[:48] for c in k31_dflash2_draft_tail.CELLS]
)
def test_draft_tail_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _tail_args(cell.spec, _seed(cell))
    ref_args = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    certifying(k31_dflash2_draft_tail)
    out = k31_dflash2_draft_tail(*args)
    out_ref = k31_dflash2_draft_tail.reference(*ref_args)
    assert out.dtype == out_ref.dtype and torch.equal(out, out_ref)
    assert torch.equal(args[0], ref_args[0])


def test_cuda_graph_replay() -> None:
    """The three launches captured once and replayed over fresh inputs, the pool counter advancing per replay."""
    spec = dict(n=16, K=4, force=FORCE_POOL, forced=2.21, temps="mixed", draft_dtype=torch.int32)
    args, _ = _commit_args(spec, seed=9)
    rec = _record_args(k31_dflash2_accept_record.CELLS[1].spec, seed=10)
    tail = _tail_args(k31_dflash2_draft_tail.CELLS[0].spec, seed=11)
    k31_dflash2_accept_commit(*args)
    k31_dflash2_accept_record(*rec)
    k31_dflash2_draft_tail(*tail)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            acc, num = k31_dflash2_accept_commit(*args)
            rewind = k31_dflash2_accept_record(*rec)
            out = k31_dflash2_draft_tail(*tail)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.Generator(device="cuda").manual_seed(12)
    for _ in range(3):
        args[0].copy_(torch.randint(0, 5, args[0].shape, generator=g, device="cuda"))
        ref = list(args)
        ref[11] = args[11].clone()
        rec_ref = [a.clone() for a in rec]
        tail_ref = [a.clone() if isinstance(a, torch.Tensor) else a for a in tail]
        graph.replay()
        torch.cuda.synchronize()
        acc_ref, num_ref = k31_dflash2_accept_commit.reference(*ref)
        assert (
            torch.equal(acc, acc_ref)
            and torch.equal(num, num_ref)
            and torch.equal(args[11], ref[11])
        )
        assert torch.equal(rewind, k31_dflash2_accept_record.reference(*rec_ref))
        assert torch.equal(rec[3], rec_ref[3]) and torch.equal(rec[4], rec_ref[4])
        assert torch.equal(out, k31_dflash2_draft_tail.reference(*tail_ref))
        assert torch.equal(tail[0], tail_ref[0])


def test_guard_refuses_int32_picks() -> None:
    args, _ = _commit_args(k31_dflash2_accept_commit.CELLS[0].spec, seed=1)
    args[0] = args[0].to(torch.int32)
    with pytest.raises(AssertionError, match="int64 verification picks"):
        with validating(k31_dflash2_accept_commit):
            k31_dflash2_accept_commit(*args)


class _VerifyWorker(_ForceWorker):
    """The hooks ``dflash2_sample_and_accept`` calls, the engine's forced count included."""

    max_draft_len = 4
    _zero_padding_rows = staticmethod(SpecWorkerBase._zero_padding_rows)
    _commit_occurrence_counts = SpecWorkerBase._commit_occurrence_counts

    def __init__(self, forced, top_k):
        super().__init__(forced)
        self._draft_model_ref = lambda: types.SimpleNamespace(
            has_candidate_selector=True, candidate_selector=types.SimpleNamespace(top_k=top_k)
        )

    def _reshape_draft_tokens_for_accept(self, spec_metadata, num_gens, device):
        return spec_metadata.draft_tokens.reshape(num_gens, spec_metadata.runtime_draft_len)

    def _reshape_logits_for_accept(self, logits, num_contexts, num_gens, spec_metadata):
        return logits

    def _apply_occurrence_penalties(self, logits, *args):
        return logits

    @staticmethod
    def _compute_probs_from_logits(mode, logits, temperatures, top_ks, top_ps, min_ps):
        return torch.softmax(logits / temperatures.unsqueeze(-1), dim=-1)

    def _rng_state_per_request(self, spec_metadata, start, end, slot):
        return spec_metadata.request_seeds[start:end], spec_metadata.request_offsets[
            start:end
        ] + slot


def _verify_step(k=32, vocab=4096, n=7, K=4):
    """A generation-only sampled verify step: ``(meta, attn, logits, (candidates, q, stored drafts))``."""
    gen = torch.Generator().manual_seed(3)
    req_t = torch.tensor([1.0, 1e-5, 1.0, 0.7, 1e-5, 1.3, 1.0], device="cuda")
    tok_t = req_t.repeat_interleave(K + 1)
    meta = types.SimpleNamespace(
        is_ragged_verify=False,
        is_cuda_graph=False,
        runtime_draft_len=K,
        use_rejection_sampling=True,
        is_all_greedy_sample=False,
        draft_probs=None,
        num_seq_slots=16,
        max_num_requests=8,
        dummy_slot_row=16,
        batch_slot_ids=torch.tensor([3, 9, 0, 4, 12, 6, 16], device="cuda"),
        request_temperatures=req_t,
        temperatures=tok_t,
        top_ks=torch.zeros_like(tok_t, dtype=torch.int32),
        top_ps=torch.ones_like(tok_t),
        min_ps=torch.zeros_like(tok_t),
        request_seeds=torch.full((n,), 42, dtype=torch.int64, device="cuda"),
        request_offsets=torch.arange(n, dtype=torch.int64, device="cuda") * 1000,
        rng_accept_slot=10,
        advanced_sampling_mode=None,
    )
    cand = torch.stack([torch.randperm(vocab, generator=gen)[:k] for _ in range(n * K)]).view(
        n, K, k
    )
    q = torch.softmax(torch.randn(n, K, k, generator=gen), dim=-1)
    pick = torch.multinomial(q.view(-1, k), 1, generator=gen).view(n, K)
    drafts = cand.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
    stored = drafts.clone()
    stored[4, 0] = (stored[4, 0] + 1) % vocab
    meta.draft_tokens = drafts.cuda().flatten().to(torch.int32)
    logits = torch.randn(n * (K + 1), vocab, generator=gen).cuda()
    logits[: K + 1][torch.arange(K), drafts[0].cuda()] += 8.0
    attn = types.SimpleNamespace(num_seqs=n, num_contexts=0)
    return meta, attn, logits, (cand.cuda(), q.cuda(), stored.cuda())


def _verify_run(worker, meta, attn, logits, proposals):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_sampling import (
        dflash2_proposal_store,
    )

    store = dflash2_proposal_store(worker, meta)
    store.store(meta.batch_slot_ids, *proposals)
    return dflash2_verify.dflash2_sample_and_accept(worker, logits.clone(), attn, meta)


@pytest.mark.parametrize("forced", [0.0, 2.21, 3.0])
def test_sample_and_accept_fused_equals_torch(forced, monkeypatch) -> None:
    """``dflash2_sample_and_accept`` of a generation-only step (sampled, greedy and padding requests, stale and
    walked proposals): the fused tail returns the torch tail's tokens and counts, the pool counter in step."""
    k = 32
    meta, attn, logits, proposals = _verify_step(k)
    applies = dflash2_verify._fused_commit_applies
    results = []
    for fused in (False, True):
        # The torch tail: the path of the steps the fused tail does not serve.
        monkeypatch.setattr(
            dflash2_verify, "_fused_commit_applies", applies if fused else lambda *args: False
        )
        worker = _VerifyWorker(forced, k)
        out = _verify_run(worker, meta, attn, logits, proposals)
        counter = worker._force_accept_rng_counter
        results.append((out, None if counter is None else counter.clone()))
    (acc_t, num_t), ctr_t = results[0]
    (acc_f, num_f), ctr_f = results[1]
    assert acc_f.dtype == torch.int32 and num_f.dtype == torch.int32
    assert torch.equal(acc_t, acc_f) and torch.equal(num_t, num_f)
    assert (ctr_t is None) == (ctr_f is None)
    if ctr_t is not None:
        assert torch.equal(ctr_t, ctr_f)


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("forced", [0.0, 2.21])
def test_sample_and_accept_overlap_equals_serial(forced, graph, monkeypatch) -> None:
    """The fused verify with its proposal load, seeds and greedy verification on the side stream (multi-stream
    execution) returns the serial order's tokens and counts and leaves the same pool counter, eager and replayed
    from a CUDA graph."""
    from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream

    k = 32
    meta, attn, logits, proposals = _verify_step(k)
    monkeypatch.setenv("TLLM_K31_DFLASH2_VERIFY_OVERLAP", "0")
    serial = _VerifyWorker(forced, k)
    serial._ensure_force_accept_rng_state(torch.device("cuda"))
    acc_s, num_s = _verify_run(serial, meta, attn, logits, proposals)
    monkeypatch.setenv("TLLM_K31_DFLASH2_VERIFY_OVERLAP", "1")
    worker = _VerifyWorker(forced, k)
    worker._ensure_force_accept_rng_state(torch.device("cuda"))
    counter0 = worker._force_accept_rng_counter.clone()
    with with_multi_stream(True):
        if not graph:
            acc, num = _verify_run(worker, meta, attn, logits, proposals)
            assert getattr(worker, "_k31_verify_overlap", None) is not None, (
                "the side stream did not serve"
            )
        else:
            _verify_run(
                worker, meta, attn, logits, proposals
            )  # compiles the kernels outside the capture
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            cuda_graph = torch.cuda.CUDAGraph()
            static_logits = logits.clone()
            with torch.cuda.stream(stream), torch.cuda.graph(cuda_graph, stream=stream):
                acc, num = dflash2_verify.dflash2_sample_and_accept(
                    worker, static_logits, attn, meta
                )
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(2):
                worker._force_accept_rng_counter.copy_(counter0)
                static_logits.copy_(logits)
                cuda_graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(acc, acc_s) and torch.equal(num, num_s)
                assert torch.equal(
                    worker._force_accept_rng_counter, serial._force_accept_rng_counter
                )
    torch.cuda.synchronize()
    assert torch.equal(acc, acc_s) and torch.equal(num, num_s)
    assert torch.equal(worker._force_accept_rng_counter, serial._force_accept_rng_counter)
