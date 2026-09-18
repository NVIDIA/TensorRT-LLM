# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the sampler JIT warmup (TorchSampler.warmup and its op helpers).

The core property under test: the standalone fake-logits warmup fully
pre-compiles every dynamo variant that real sample_async-style calls hit, so
no torch.compile happens after warmup (tripwire for torch guard-model drift,
see the warmup docstrings). Besides the synthetic edge battery, a
runtime-faithful replay reconstructs the Fusions call sites exactly as
TorchSampler._sample_batched_by_strategy / _process_logprobs do (int64
cumsum-derived indexer output, storage-offset slice-view sources, pinned-host
``.to(non_blocking=True)`` index transfers, per-step ``torch.empty`` logits)
so warmup/runtime guard mismatches cannot regress silently again.
"""

import time
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import Fusions, warmup_compile_fusions
from tensorrt_llm._torch.pyexecutor.sampler.sampler_features import _PackedStepIndexer
from tensorrt_llm._utils import prefer_pinned

VOCAB_SIZE = 1024
DEVICE = torch.device("cuda")

# Number of dynamo graphs the warmup set compiles: none -- both logprob ops
# are Triton kernels (see warmup_compile_fusions); the warmup only pays their
# Triton JIT.
EXPECTED_WARMUP_GRAPHS = 0

# Generous per-call budget for a previously-compiled op (execution is sub-ms;
# a recompile is >100 ms even for the cheapest variant).
MAX_COMPILE_FREE_CALL_MS = 50.0


def _unique_graphs() -> int:
    from torch._dynamo.utils import counters

    return counters["stats"].get("unique_graphs", 0)


def _timed_ms(fn, *args, **kwargs) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def _fake_logits(num_rows: int, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    return torch.randn(num_rows, vocab_size, dtype=torch.float32, device=DEVICE)


def _indices(num_indices: int) -> torch.Tensor:
    # int64, like the cumsum-derived _PackedStepIndexer output at runtime (see
    # the guard-fidelity contract in warmup_compile_fusions).
    return torch.arange(num_indices, dtype=torch.int64, device=DEVICE)


def _gather_log_softmax(num_rows: int, num_gathered: int) -> float:
    # Mirror _process_logprobs: the output is a row slice of a larger buffer.
    out = _fake_logits(num_rows)[:num_gathered]
    return _timed_ms(
        Fusions.gather_log_softmax_with_output,
        _fake_logits(num_rows),
        _indices(num_gathered),
        out=out,
    )


def _determine_sampled_rank(num_rows: int) -> float:
    group_logprobs = _fake_logits(num_rows)
    # Mirror _process_logprobs: the sampled values come from torch.gather, i.e. a
    # freshly-allocated (num_rows, 1) tensor NOT aliasing group_logprobs
    # (dynamo guards on input aliasing, so this matters).
    index = torch.zeros((num_rows, 1), dtype=torch.int64, device=DEVICE)
    sampled = torch.gather(group_logprobs, dim=-1, index=index).squeeze(-1).unsqueeze(-1)
    return _timed_ms(Fusions.determine_sampled_rank, group_logprobs, sampled)


@pytest.fixture
def fresh_dynamo():
    """Isolate dynamo compile caches/counters so graph counting is deterministic."""
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()
    counters.clear()


# Real-style edge battery: shapes mimicking runtime sample_async calls,
# covering all 0/1-specialization combos of the mark_dynamic'ed dims.
EDGE_BATTERY = [
    ("gather_log_softmax_with_output", _gather_log_softmax, (64, 64)),
    ("gather_log_softmax_with_output", _gather_log_softmax, (64, 7)),
    ("gather_log_softmax_with_output", _gather_log_softmax, (64, 1)),
    ("gather_log_softmax_with_output", _gather_log_softmax, (200, 37)),
    ("gather_log_softmax_with_output", _gather_log_softmax, (2, 1)),
    ("gather_log_softmax_with_output", _gather_log_softmax, (1, 1)),
    ("determine_sampled_rank", _determine_sampled_rank, (64,)),
    ("determine_sampled_rank", _determine_sampled_rank, (2,)),
    ("determine_sampled_rank", _determine_sampled_rank, (1,)),
]


@pytest.mark.usefixtures("fresh_dynamo")
def test_warmup_compile_fusions_precompiles_all_variants():
    """The warmup set covers every variant real-style calls can hit.

    Both logprob ops are Triton kernels whose row count enters the launch grid
    only, so the warmup compiles no dynamo graph; the battery below checks that
    no call shape compiles anything (dynamo or Triton) after warmup.
    """
    graphs_before = _unique_graphs()
    warmup_compile_fusions(VOCAB_SIZE, DEVICE)
    warmup_graphs = _unique_graphs() - graphs_before
    assert warmup_graphs == EXPECTED_WARMUP_GRAPHS, (
        f"warmup compiled {warmup_graphs} graphs, expected {EXPECTED_WARMUP_GRAPHS}"
    )

    # The battery must trigger neither new dynamo graphs nor recompiles
    # (a recompile of an existing frame would not bump unique_graphs, so the
    # wall-time bound is the second tripwire).
    graphs_before_battery = _unique_graphs()
    with torch.inference_mode():
        for name, fn, args in EDGE_BATTERY:
            elapsed_ms = fn(*args)
            assert _unique_graphs() == graphs_before_battery, (
                f"{name}{args} compiled a new graph after warmup"
            )
            assert elapsed_ms < MAX_COMPILE_FREE_CALL_MS, (
                f"{name}{args} took {elapsed_ms:.1f} ms after warmup "
                f"(>{MAX_COMPILE_FREE_CALL_MS} ms; likely a recompile)"
            )


@pytest.mark.usefixtures("fresh_dynamo")
def test_warmup_compile_fusions_is_idempotent():
    warmup_compile_fusions(VOCAB_SIZE, DEVICE)
    graphs_before = _unique_graphs()
    warmup_compile_fusions(VOCAB_SIZE, DEVICE)
    assert _unique_graphs() == graphs_before


# --- Runtime-faithful replay -------------------------------------------------
# Reconstructs the tensor metadata the Fusions ops see in production, by
# mimicking TorchSampler._sample_batched_by_strategy and
# LogProbsHandler._process_logprobs: per-step logits allocation,
# _PackedStepIndexer-built index tensors (int64 via torch.cumsum offsets) moved
# with pinned-host .to(device, non_blocking=True), the raw-logprobs output
# written into a row slice of the over-provisioned logprobs buffer, and the
# sampled values gathered as a fresh (n, 1) tensor. This guards the warmup's
# guard-fidelity contract against silent drift of either the sampler code or
# the dynamo guard model (an 8-rank trace once showed ~4.4 s of mid-run
# recompile stalls from an int32-vs-int64 index dtype mismatch that synthetic
# warmup-style calls could not detect).


@dataclass
class _ReqSpec:
    """One request in a replayed sampling wave."""

    num_steps: int = 1  # 1 + draft_len
    raw: bool = False  # needs raw (temperature-1) logprobs -> gather_log_softmax_with_output
    logprobs: bool = True  # returns logprobs (partakes in _process_logprobs)


@torch.inference_mode()
def _replay_wave(
    reqs: list[_ReqSpec],
    call_times_ms: list[tuple[str, float]],
    *,
    tag: str,
    vocab_size: int = VOCAB_SIZE,
    max_beam_width: int = 1,
) -> None:
    """Replay one sample_async wave's Fusions calls with runtime metadata."""
    device = DEVICE
    max_tokens = max(r.num_steps for r in reqs)

    def timed(name, fn, *args, **kwargs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.cuda.synchronize()
        call_times_ms.append((f"{tag}:{name}", (time.perf_counter() - t0) * 1e3))
        return out

    # ---- SamplingRequestsMetadata / _select_generated_logits ----
    req_num_steps = torch.tensor(
        [r.num_steps for r in reqs], dtype=torch.int32, pin_memory=prefer_pinned()
    )
    req_num_beams = torch.tensor([1] * len(reqs), dtype=torch.int32, pin_memory=prefer_pinned())
    req_num_generated_tokens = req_num_steps * req_num_beams
    req_offsets, sum_steps = _PackedStepIndexer.calculate_request_offsets(
        req_num_generated_tokens, pin_memory=prefer_pinned()
    )
    # Model-output logits: per-step allocation.
    logits_cuda = torch.randn(sum_steps, vocab_size, dtype=torch.float32, device=device)
    logits_cuda_indexer = _PackedStepIndexer(
        num_steps=req_num_generated_tokens,
        max_steps=max_tokens * max_beam_width,
        req_offsets=req_offsets,
    )

    # ---- _sample_batched_by_strategy: logprobs buffer + raw logit indices ----
    # Over-provisioned buffer; slice(0, processed_end) holds the processed
    # logprobs the strategy samplers wrote (not a Fusions op), the raw ones
    # follow it.
    logprobs_cuda = torch.randn(
        (logits_cuda.size(0), logits_cuda.size(1)), device=device, dtype=torch.float32
    )
    processed_end = sum(r.num_steps for r in reqs if r.logprobs and not r.raw)
    need_raw_logprobs = torch.tensor([r.raw for r in reqs], dtype=torch.bool)
    raw_req_indices = torch.nonzero(need_raw_logprobs).view(-1)
    if raw_req_indices.numel() > 0:
        raw_logit_indices_cuda = logits_cuda_indexer[raw_req_indices].to(
            device=device, non_blocking=True
        )
        raw_start = processed_end
        raw_end = raw_start + raw_logit_indices_cuda.size(0)
        timed(
            "gather_log_softmax_with_output",
            Fusions.gather_log_softmax_with_output,
            logits_cuda,
            raw_logit_indices_cuda,
            out=logprobs_cuda[raw_start:raw_end],
        )
        logprobs_end = raw_end
    else:
        logprobs_end = processed_end
    if logprobs_end == 0:
        return
    logprobs_cuda = logprobs_cuda[:logprobs_end]

    # ---- _process_logprobs (single beam) ----
    sampled_indices_cuda = torch.randint(
        0, vocab_size, (logprobs_end,), device=device, dtype=torch.int64
    )
    sampled_vals_cuda = torch.gather(
        logprobs_cuda, dim=1, index=sampled_indices_cuda.unsqueeze(-1)
    ).squeeze(-1)
    timed(
        "determine_sampled_rank",
        Fusions.determine_sampled_rank,
        logprobs_cuda,
        sampled_vals_cuda.unsqueeze(-1),
    )


# Mixed raw/processed compositions plus the 0/1-specialization and multi-step
# (draft-token) sweeps.
_REPLAY_BATTERY: list[tuple[str, list[_ReqSpec]]] = [
    ("rows48_all_raw", [_ReqSpec(raw=True) for _ in range(48)]),
    ("rows1_raw", [_ReqSpec(raw=True)]),
    ("rows48_processed_only", [_ReqSpec() for _ in range(48)]),
    ("rows1_processed_only", [_ReqSpec()]),
    ("mixed_raw_subset", [_ReqSpec(raw=(i % 3 == 0)) for i in range(48)]),
    ("raw_m1_of_48", [_ReqSpec(raw=(i == 47)) for i in range(48)]),
    ("no_logprobs_then_raw", [_ReqSpec(logprobs=False) for _ in range(16)] + [_ReqSpec(raw=True) for _ in range(32)]),
    ("steps2_subset_raw", [_ReqSpec(num_steps=2, raw=(i < 3)) for i in range(24)]),
]


@pytest.mark.usefixtures("fresh_dynamo")
def test_warmup_covers_runtime_faithful_replay():
    """After warmup, runtime-faithful sample_async replays never (re)compile."""
    warmup_compile_fusions(VOCAB_SIZE, DEVICE)
    graphs_after_warmup = _unique_graphs()

    call_times_ms: list[tuple[str, float]] = []
    for tag, reqs in _REPLAY_BATTERY:
        _replay_wave(reqs, call_times_ms, tag=tag)
        assert _unique_graphs() == graphs_after_warmup, (
            f"replay wave '{tag}' compiled a new dynamo graph after warmup"
        )
    for name, elapsed_ms in call_times_ms:
        assert elapsed_ms < MAX_COMPILE_FREE_CALL_MS, (
            f"{name} took {elapsed_ms:.1f} ms after warmup "
            f"(>{MAX_COMPILE_FREE_CALL_MS} ms; likely a recompile)"
        )
def _make_sampler() -> TorchSampler:
    return TorchSampler(
        TorchSampler.Args(
            max_seq_len=32,
            max_draft_len=0,
            max_num_sequences=8,
            max_beam_width=1,
            max_total_draft_tokens=0,
        )
    )


def _mock_engine(vocab_size: int = VOCAB_SIZE) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(lm_head=SimpleNamespace(vocab_size_padded=vocab_size))
    )


@pytest.mark.usefixtures("fresh_dynamo")
def test_torch_sampler_warmup(monkeypatch):
    monkeypatch.delenv("TLLM_SAMPLER_JIT_WARMUP", raising=False)
    sampler = _make_sampler()
    graphs_before = _unique_graphs()
    sampler.warmup(_mock_engine())
    assert _unique_graphs() - graphs_before == EXPECTED_WARMUP_GRAPHS
    # RNG determinism: the warmup must not create/advance the sampler's
    # deterministic generator (cross-rank consistent sampling stream).
    assert sampler._generator is None


@pytest.mark.usefixtures("fresh_dynamo")
def test_torch_sampler_warmup_kill_switch(monkeypatch):
    monkeypatch.setenv("TLLM_SAMPLER_JIT_WARMUP", "0")
    sampler = _make_sampler()
    graphs_before = _unique_graphs()
    sampler.warmup(_mock_engine())
    assert _unique_graphs() == graphs_before


@pytest.mark.usefixtures("fresh_dynamo")
def test_torch_sampler_warmup_unresolvable_vocab_is_a_noop(monkeypatch):
    monkeypatch.delenv("TLLM_SAMPLER_JIT_WARMUP", raising=False)
    sampler = _make_sampler()
    graphs_before = _unique_graphs()
    # Neither lm_head.vocab_size_padded nor config.vocab_size resolvable.
    sampler.warmup(SimpleNamespace(model=SimpleNamespace()))
    assert _unique_graphs() == graphs_before


def test_resolve_warmup_vocab_size_fallbacks():
    resolve = TorchSampler._resolve_warmup_vocab_size
    assert resolve(_mock_engine(2048)) == 2048
    # Falls back to model.config.vocab_size.
    engine = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vocab_size=333)))
    assert resolve(engine) == 333
    # lm_head wins over config.
    engine = SimpleNamespace(
        model=SimpleNamespace(
            lm_head=SimpleNamespace(vocab_size_padded=2048),
            config=SimpleNamespace(vocab_size=333),
        )
    )
    assert resolve(engine) == 2048
    # Unresolvable / invalid -> None.
    assert resolve(SimpleNamespace()) is None
    assert resolve(SimpleNamespace(model=SimpleNamespace())) is None
    engine = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vocab_size=0)))
    assert resolve(engine) is None
    engine = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(vocab_size="x")))
    assert resolve(engine) is None
