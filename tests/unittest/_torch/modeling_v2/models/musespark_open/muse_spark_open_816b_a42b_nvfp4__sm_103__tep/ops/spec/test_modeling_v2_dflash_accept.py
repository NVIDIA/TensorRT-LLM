# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Muse Spark Open tep8 ``dflash_accept`` entry and fast path ``fused_accept``.

The entry's claim is bitwise equality with the built-in acceptance chain, so
every cell is gated with ``torch.equal`` against the entry's ``reference``
(the built-in chain unrolled), and the installed worker is gated step by step
against the built-in ``DFlashWorker`` -- same logits, same drafts, same
forced-acceptance pool -- over hundreds of steps, eagerly and under CUDA-graph
replay. The fractional override's one float comparison is swept over every
pool entry.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_accept import (  # noqa: E501
    FORCE_FRACTIONAL,
    MAX_REQUESTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec.dflash_accept import (  # noqa: E501
    dflash_accept as op,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

__extra_import_path__ = ["../../../../.."]
from _validating import certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "dflash_accept requires a CUDA device"

VOCAB = 512


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _pool_and_counter(seed: int = 0xACCE9D, size: int = 1 << 16):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pool = torch.rand(size, dtype=torch.float32, generator=gen).cuda()
    return pool, torch.zeros(1, dtype=torch.int64, device="cuda")


def _tokens(contexts, gens, k, g, int64=False):
    """Target tokens and drafts with planted leading matches of every length."""
    dtype = torch.int64 if int64 else torch.int32
    target = torch.randint(0, VOCAB, (contexts + gens * (k + 1),), generator=g, device="cuda")
    target = target.to(dtype)
    gen_targets = target[contexts:].view(gens, k + 1)
    draft = torch.randint(0, VOCAB, (gens, k), generator=g, device="cuda").to(torch.int32)
    lead = torch.randint(0, k + 1, (gens,), generator=g, device="cuda")
    cols = torch.arange(k, device="cuda")
    keep = cols.unsqueeze(0) < lead.unsqueeze(1)
    draft = torch.where(keep, gen_targets[:, :k].to(torch.int32), draft)
    return target, draft


def _force_args(force, rate, k):
    int_part = int(rate)
    frac = rate - int_part
    max_total = k + 1
    base_total = min(int_part + 1, max_total)
    return base_total, max_total, frac


def _check(out, ref, contexts):
    (acc, num), (ref_acc, ref_num) = out, ref
    assert acc.dtype == num.dtype == torch.int32
    assert torch.equal(num, ref_num), (num, ref_num)
    assert torch.equal(acc[contexts:], ref_acc[contexts:])
    assert torch.equal(acc[:contexts, 0], ref_acc[:contexts, 0])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    k, contexts, gens = spec["k"], spec["contexts"], spec["gens"]
    pool, counter = _pool_and_counter()
    ref_counter = counter.clone()
    base_total, max_total, frac = _force_args(spec["force"], spec["rate"], k)
    for _ in range(20):
        target, draft = _tokens(contexts, gens, k, g, spec.get("int64", False))
        before = (target.clone(), draft.clone())
        out = op(target, draft, contexts, spec["force"], base_total, max_total, frac, pool, counter)
        ref = op.reference(
            target, draft, contexts, spec["force"], base_total, max_total, frac, pool, ref_counter
        )
        _check(out, ref, contexts)
        assert torch.equal(counter, ref_counter)
        assert torch.equal(before[0], target) and torch.equal(before[1], draft), "inputs mutated"


@pytest.mark.parametrize("rate", [3.05, 2.7, 2.06, 1.6, 0.5, 3.0000001, 0.999])
def test_fractional_compare_over_the_whole_pool(rate):
    """Every pool entry, compared against the rate's fractional part as torch compares it."""
    k = 7
    pool, counter = _pool_and_counter()
    ref_counter = counter.clone()
    base_total, max_total, frac = _force_args(FORCE_FRACTIONAL, rate, k)
    gens = MAX_REQUESTS
    target = torch.zeros(gens * (k + 1), dtype=torch.int32, device="cuda")
    draft = torch.zeros(gens, k, dtype=torch.int32, device="cuda")
    seen = torch.zeros(pool.numel(), dtype=torch.bool, device="cuda")
    for _ in range(pool.numel() // gens + 64):
        out = op(target, draft, 0, FORCE_FRACTIONAL, base_total, max_total, frac, pool, counter)
        ref = op.reference(
            target, draft, 0, FORCE_FRACTIONAL, base_total, max_total, frac, pool, ref_counter
        )
        _check(out, ref, 0)
        slots = torch.arange(gens, device="cuda", dtype=torch.int64)
        seen[(ref_counter * 6007 + slots * 1009) & (pool.numel() - 1)] = True
    assert seen.float().mean().item() > 0.6, "the sweep should reach most pool entries"


# --------------------------------------------------------------------------
# The installed worker against the built-in worker
# --------------------------------------------------------------------------
def _worker(cls, rate):
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    worker.force_num_accepted_tokens = rate
    worker._force_accept_rng_pool = None
    worker._force_accept_rng_counter = None
    if cls is not DFlashWorker:
        assert drafter_worker.install_fused_accept(worker) is None
        assert type(worker) is cls
    return worker


def _metadata(is_cuda_graph=False):
    return SimpleNamespace(
        is_ragged_verify=False, is_cuda_graph=is_cuda_graph, is_all_greedy_sample=True
    )


def _logits_for(target, g):
    """Greedy logits whose argmax is ``target``."""
    logits = torch.randn(target.numel(), VOCAB, generator=g, device="cuda")
    logits.scatter_(1, target.long().unsqueeze(1), 100.0)
    return logits


@pytest.mark.parametrize(
    "rate, k, contexts, gens",
    [
        (3.05, 7, 0, 1),  # the target's decode step
        (0.0, 7, 0, 1),  # strict acceptance
        (3.0, 7, 0, 1),  # integer override
        (2.7, 5, 0, 2),
        (6.5, 7, 0, 1),  # base_total 7 < max 8: fractional
        (7.5, 7, 0, 1),  # base_total == max: integer branch
        (3.05, 7, 1, 2),  # mixed batch
        (0.0, 3, 2, 3),
    ],
)
def test_installed_worker_matches_builtin(rate, k, contexts, gens):
    g = torch.Generator(device="cuda").manual_seed(_seed(f"{rate}-{k}-{contexts}-{gens}"))
    builtin, fused = (
        _worker(DFlashWorker, rate),
        _worker(drafter_worker.FusedAcceptDFlashWorker, rate),
    )
    batch = contexts + gens
    for step in range(300):
        target, draft = _tokens(contexts, gens, k, g)
        logits = _logits_for(target, g)
        md = _metadata(is_cuda_graph=(step % 50 == 7))  # some eager-warmup steps: no override
        ref = DFlashWorker._sample_and_accept_draft_tokens_base(
            builtin, logits, draft, contexts, batch, md
        )
        out = fused._sample_and_accept_draft_tokens_base(logits, draft, contexts, batch, md)
        _check(out, ref, contexts)
        if builtin._force_accept_rng_counter is not None:
            assert torch.equal(builtin._force_accept_rng_counter, fused._force_accept_rng_counter)
        else:
            assert fused._force_accept_rng_counter is None


def test_installed_worker_falls_back_outside_its_shapes(monkeypatch):
    """Ragged verify, more than MAX_REQUESTS requests, or no metadata: the built-in method runs."""
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    fused = _worker(drafter_worker.FusedAcceptDFlashWorker, 3.05)
    fused_calls, builtin_calls = [], []
    monkeypatch.setattr(drafter_worker, "dflash_accept", lambda *a: fused_calls.append(a))
    builtin = SpecWorkerBase._sample_and_accept_draft_tokens_base

    def spy(self, *args):
        builtin_calls.append(args)
        return builtin(self, *args)

    monkeypatch.setattr(SpecWorkerBase, "_sample_and_accept_draft_tokens_base", spy)
    g = torch.Generator(device="cuda").manual_seed(5)
    gens = MAX_REQUESTS + 1
    target, draft = _tokens(0, gens, 1, g)
    logits = _logits_for(target, g)
    acc, num = fused._sample_and_accept_draft_tokens_base(logits, draft, 0, gens, _metadata())
    assert len(builtin_calls) == 1 and not fused_calls
    assert acc.shape == (gens, 2) and num.shape == (gens,)
    with pytest.raises(AttributeError):
        # Ragged verify goes to the built-in, which reads the ragged metadata fields.
        ragged = SimpleNamespace(**{**vars(_metadata()), "is_ragged_verify": True})
        fused._sample_and_accept_draft_tokens_base(logits[:16], draft[:8], 0, 8, ragged)
    assert len(builtin_calls) == 2 and not fused_calls


def test_cuda_graph_replay_matches_builtin():
    """Capture the installed worker's acceptance once; every replay matches the built-in eagerly."""
    rate, k = 3.05, 7
    g = torch.Generator(device="cuda").manual_seed(17)
    builtin, fused = (
        _worker(DFlashWorker, rate),
        _worker(drafter_worker.FusedAcceptDFlashWorker, rate),
    )
    target, draft = _tokens(0, 1, k, g)
    logits = _logits_for(target, g)
    # The pool is built on the first non-warmup call, outside the capture (as in serving).
    DFlashWorker._sample_and_accept_draft_tokens_base(builtin, logits, draft, 0, 1, _metadata())
    fused._sample_and_accept_draft_tokens_base(logits, draft, 0, 1, _metadata())
    md = _metadata(is_cuda_graph=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        acc, num = fused._sample_and_accept_draft_tokens_base(logits, draft, 0, 1, md)
    torch.cuda.current_stream().wait_stream(stream)
    # The capture recorded the counter bump without running it; the built-in
    # replays it on the first replay, so advance the eager side in lock-step.
    for _ in range(200):
        new_target, new_draft = _tokens(0, 1, k, g)
        logits.copy_(_logits_for(new_target, g))
        draft.copy_(new_draft)
        graph.replay()
        ref = DFlashWorker._sample_and_accept_draft_tokens_base(
            builtin, logits, draft, 0, 1, _metadata()
        )
        torch.cuda.synchronize()
        _check((acc, num), ref, 0)
        assert torch.equal(builtin._force_accept_rng_counter, fused._force_accept_rng_counter)
