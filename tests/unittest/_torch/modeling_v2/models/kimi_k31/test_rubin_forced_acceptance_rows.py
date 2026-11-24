# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rubin targets' forced acceptance touches only the rows that verify real drafts.

Each target's DFlash2 worker overrides ``_apply_force_accepted_tokens`` to count what the engine
forced; both verify paths call it. The engine forces only the generation rows that
``SpecMetadata.real_draft_rows`` marks: a request's first generation step verifies the zero
placeholders the worker gave its context step, so it keeps its natural count and is not counted.

CPU (``cpu_only``): a mixed batch on each target; graph warmup forces and counts nothing.
One sm_107 GPU: the override captured in a CUDA graph and replayed with other natural counts and
other marked rows equals the eager calls, counts and totals alike.

The default target (``kimi_k31_pro_nvfp4__sm_107__default``) runs one worker for its three
layouts. It has cases of its own of each kind, with their own batches and expected counts.
"""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_TARGETS = (
    "kimi_k31_pro_nvfp4__sm_107__dep16",
    "kimi_k31_pro_nvfp4__sm_107__helix8",
    "kimi_k31_pro_nvfp4__sm_107__tep8",
)
_FORCE_ENV = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"
_K = 7

# The targets import the family kernels, which need the CuTe DSL.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="this environment lacks cutlass"
)
_ON_SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the Rubin targets run on sm_107",
)


def _worker(target: str, monkeypatch, value: str, device: str = "cpu"):
    """The target's DFlash2 worker with only the state forced acceptance reads."""
    from tensorrt_llm._torch.speculative.interface import get_force_num_accepted_tokens_float

    mod = importlib.import_module(f"{_FAMILY}.{target}.modeling")
    monkeypatch.setenv(_FORCE_ENV, value)
    worker = mod.DFlashWorker.__new__(mod.DFlashWorker)
    torch.nn.Module.__init__(worker)
    worker.force_num_accepted_tokens = get_force_num_accepted_tokens_float()
    worker._force_accept_rng_pool = worker._force_accept_rng_counter = None
    worker._ensure_force_accept_rng_state(torch.device(device))
    worker._forced, worker._forced_calls = None, 0
    return worker


def _spec(real_rows, *, is_cuda_graph: bool = False, device: str = "cpu"):
    rows = torch.tensor(real_rows, dtype=torch.bool, device=device)
    return SimpleNamespace(is_cuda_graph=is_cuda_graph, real_draft_rows=rows)


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", _TARGETS)
def test_only_rows_with_real_drafts_are_forced_and_counted(target, monkeypatch):
    worker = _worker(target, monkeypatch, "2")
    # One context row, then generation rows: a first step after the context step (placeholder
    # drafts, natural 1), a placeholder row the target happened to match (natural 2), and two
    # rows with the drafter's drafts.
    natural = torch.tensor([1, 1, 2, 1, 6], dtype=torch.int32)
    out = worker._apply_force_accepted_tokens(
        natural.clone(), 1, _K, spec_metadata=_spec([False, False, True, True])
    )
    assert out.tolist() == [1, 1, 2, 3, 3]
    # Two forced rows of 3 tokens each; the placeholder rows are not counted.
    assert worker._forced.tolist() == [2, 6]
    assert "rows=2 accepted=6 mean=3.0000" in worker.forced_line()


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", _TARGETS)
def test_graph_warmup_forces_and_counts_nothing(target, monkeypatch):
    # An eager pass of a CUDA graph step: no capture is underway.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _worker(target, monkeypatch, "2.68")
    natural = torch.tensor([1, 1, 2], dtype=torch.int32)
    out = worker._apply_force_accepted_tokens(
        natural.clone(), 1, _K, spec_metadata=_spec([True, True], is_cuda_graph=True)
    )
    assert out.tolist() == natural.tolist()
    assert worker._forced.tolist() == [0, 0] and worker._forced_calls == 0


# The rows each replay marks: none, all, and two mixes.
_ROW_PATTERNS = (
    (False,) * 8,
    (True,) * 8,
    (False, True) * 4,
    (True, True, False, True, False, False, True, False),
)


@_ON_SM107
@pytest.mark.parametrize("target", _TARGETS)
def test_a_captured_override_forces_the_rows_marked_at_replay(target, monkeypatch):
    num_contexts, num_gens = 1, 8
    graph_worker = _worker(target, monkeypatch, "2.68", "cuda")
    eager_worker = _worker(target, monkeypatch, "2.68", "cuda")
    # The step's fixed tensors: the natural counts and the metadata's marked rows.
    counts = torch.ones(num_contexts + num_gens, dtype=torch.int32, device="cuda")
    rows = torch.ones(num_gens, dtype=torch.bool, device="cuda")
    graph_spec = SimpleNamespace(is_cuda_graph=True, real_draft_rows=rows)
    # The engine's warmup pass comes first: it forces and counts nothing.
    graph_worker._apply_force_accepted_tokens(counts, num_contexts, _K, spec_metadata=graph_spec)
    assert counts.tolist() == [1] * (num_contexts + num_gens)
    snapshot = torch.empty_like(counts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_worker._apply_force_accepted_tokens(
            counts, num_contexts, _K, spec_metadata=graph_spec
        )
        snapshot.copy_(counts)
    # Capture records the forced draws and the totals without running them.
    assert graph_worker._force_accept_rng_counter.item() == 0
    assert graph_worker._forced.tolist() == [0, 0]

    generator = torch.Generator().manual_seed(0)
    for step, pattern in enumerate(_ROW_PATTERNS * 2):
        natural = torch.randint(1, _K + 2, (num_contexts + num_gens,), generator=generator)
        natural = natural.to(device="cuda", dtype=torch.int32)
        marked = torch.tensor(pattern, dtype=torch.bool, device="cuda")
        counts.copy_(natural)
        rows.copy_(marked)
        graph.replay()
        expected = eager_worker._apply_force_accepted_tokens(
            natural.clone(),
            num_contexts,
            _K,
            spec_metadata=SimpleNamespace(is_cuda_graph=False, real_draft_rows=marked),
        )
        torch.cuda.synchronize()
        assert torch.equal(snapshot, expected), f"replay {step}"
        kept = ~marked
        assert torch.equal(snapshot[num_contexts:][kept], natural[num_contexts:][kept])
        forced = snapshot[num_contexts:][marked]
        assert bool(((forced == 3) | (forced == 4)).all())
    assert graph_worker._forced.tolist() == eager_worker._forced.tolist()
    assert graph_worker._forced[0].item() == sum(map(sum, _ROW_PATTERNS)) * 2


# The default target: one DFlash2 worker for its three layouts, with cases of its own.
_DEFAULT = "kimi_k31_pro_nvfp4__sm_107__default"


@pytest.mark.cpu_only
def test_the_default_target_forces_and_counts_only_rows_with_real_drafts(monkeypatch):
    worker = _worker(_DEFAULT, monkeypatch, "3")
    # Two context rows, then generation rows that alternate between the drafter's drafts and a
    # first step's placeholders (natural 1, or 2 where the target matched a placeholder).
    natural = torch.tensor([1, 1, 2, 1, 8, 2], dtype=torch.int32)
    out = worker._apply_force_accepted_tokens(
        natural.clone(), 2, _K, spec_metadata=_spec([True, False, True, False])
    )
    assert out.tolist() == [1, 1, 4, 1, 4, 2]
    # Two forced rows of 4 tokens each; the placeholder rows are not counted.
    assert worker._forced.tolist() == [2, 8]
    assert "rows=2 accepted=8 mean=4.0000" in worker.forced_line()


@pytest.mark.cpu_only
def test_the_default_target_forces_and_counts_nothing_in_graph_warmup(monkeypatch):
    # An eager pass of a CUDA graph step: no capture is underway.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = _worker(_DEFAULT, monkeypatch, "2.68")
    natural = torch.tensor([1, 2, 1, 5], dtype=torch.int32)
    out = worker._apply_force_accepted_tokens(
        natural.clone(), 1, _K, spec_metadata=_spec([True, False, True], is_cuda_graph=True)
    )
    assert out.tolist() == natural.tolist()
    assert worker._forced.tolist() == [0, 0] and worker._forced_calls == 0


# The rows the default target's replays mark: all, two mixes and none.
_DEFAULT_ROW_PATTERNS = (
    (True,) * 6,
    (False, False, True, True, False, True),
    (True, False) * 3,
    (False,) * 6,
)


@_ON_SM107
def test_the_default_target_captured_override_forces_the_rows_marked_at_replay(monkeypatch):
    num_contexts, num_gens = 2, 6
    graph_worker = _worker(_DEFAULT, monkeypatch, "2.68", "cuda")
    eager_worker = _worker(_DEFAULT, monkeypatch, "2.68", "cuda")
    counts = torch.ones(num_contexts + num_gens, dtype=torch.int32, device="cuda")
    rows = torch.ones(num_gens, dtype=torch.bool, device="cuda")
    graph_spec = SimpleNamespace(is_cuda_graph=True, real_draft_rows=rows)
    # The engine's warmup pass, then the capture: neither forces nor counts anything.
    graph_worker._apply_force_accepted_tokens(counts, num_contexts, _K, spec_metadata=graph_spec)
    assert counts.tolist() == [1] * (num_contexts + num_gens)
    snapshot = torch.empty_like(counts)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_worker._apply_force_accepted_tokens(
            counts, num_contexts, _K, spec_metadata=graph_spec
        )
        snapshot.copy_(counts)
    assert graph_worker._force_accept_rng_counter.item() == 0
    assert graph_worker._forced.tolist() == [0, 0]

    generator = torch.Generator().manual_seed(1)
    for step, pattern in enumerate(_DEFAULT_ROW_PATTERNS * 2):
        natural = torch.randint(1, _K + 2, (num_contexts + num_gens,), generator=generator)
        natural = natural.to(device="cuda", dtype=torch.int32)
        marked = torch.tensor(pattern, dtype=torch.bool, device="cuda")
        counts.copy_(natural)
        rows.copy_(marked)
        graph.replay()
        expected = eager_worker._apply_force_accepted_tokens(
            natural.clone(),
            num_contexts,
            _K,
            spec_metadata=SimpleNamespace(is_cuda_graph=False, real_draft_rows=marked),
        )
        torch.cuda.synchronize()
        assert torch.equal(snapshot, expected), f"replay {step}"
        # Context rows and unmarked rows keep their natural counts; marked rows are forced.
        assert torch.equal(snapshot[:num_contexts], natural[:num_contexts])
        assert torch.equal(snapshot[num_contexts:][~marked], natural[num_contexts:][~marked])
        forced = snapshot[num_contexts:][marked]
        assert bool(((forced == 3) | (forced == 4)).all())
    assert graph_worker._forced.tolist() == eager_worker._forced.tolist()
    assert graph_worker._forced[0].item() == sum(map(sum, _DEFAULT_ROW_PATTERNS)) * 2
