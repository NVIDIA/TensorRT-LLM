# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tail bookkeeping (RuntimeHooks.spec_tail_bookkeep -> SpecMetadata.tail_bookkeep): the shared items against the
in-tree path, bitwise.

* B1 (``advanced_sample_draft_block``): at one decode request the draft block's sampling parameters are views of the
  per-token buffers instead of four repeats of the per-request ones.
* S1 (``sample_draft_tokens``): a draft block with no d2t map returns its int32 tokens without the int64 round trip.
* A1 (``_sample_and_accept_draft_tokens_rejection``): a decode-only accept returns the acceptance kernel's outputs
  instead of copying them into a ones-filled and an empty buffer.

Each test runs the same step twice, on a metadata with the flag off (the in-tree path) and one with it on, and compares
every output (and every tensor the samplers were handed) bit for bit. The samplers are deterministic stand-ins that
record their inputs, so the comparison covers the parameter and Philox tensors, not only the tokens.

The CPU tests fill the per-token buffers with a mirror of ``populate_sampling_params_for_one_model``'s layout and also
check that the flag off takes the in-tree lines. The CUDA tests fill them with the real populate, eagerly and inside
``torch.cuda.graph`` (captured on one request's values, replayed on another's). R1 and Z1 are the M3.1 target's
overrides and are tested with it (test_modeling_v2_m31_tailbk.py).
"""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import runner as runner_module
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.runtime_hooks import DEFAULT_RUNTIME_HOOKS, RuntimeHooks
from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import (
    _RNG_SLOT_SPAN,
    SpecMetadata,
    SpeculativeDecodingMode,
    SpecWorkerBase,
    tail_bookkeep_draft_len,
)
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="the real populate allocates CUDA buffers")

V = 64  # vocab
MAX_REQUESTS = 8
DRAFT_LENS = list(range(1, 8))  # K = 1..7
# (temperature, top_k, top_p, min_p, seed), all sampled: unseeded, a seed at the int64 edge, a uint64 seed
# (reinterpreted as negative), seed 0, and top-k 2 with a large min-p (top-k 1 would be greedy).
REQUESTS = [
    (0.7, 0, 0.9, 0.0, None),
    (1.3, 17, 0.5, 0.05, (1 << 63) - 1),
    (0.25, 3, 1.0, 0.2, (1 << 64) - 1),
    (1.0, 0, 0.95, 0.0, 0),
    (2.0, 2, 0.1, 0.5, 1234),
]


class _Worker(SpecWorkerBase):
    """SpecWorkerBase without its constructor (which needs flashinfer): the state the sampler tail reads."""

    def _forward_impl(self, *args, **kwargs):  # pragma: no cover - never run
        raise NotImplementedError

    @property
    def max_draft_len(self) -> int:
        return max(DRAFT_LENS)


def _worker(d2t: Optional[torch.Tensor] = None, force_num_accepted_tokens: float = 0.0) -> _Worker:
    worker = object.__new__(_Worker)
    worker._d2t = d2t
    worker.force_num_accepted_tokens = force_num_accepted_tokens
    worker.mapping = None
    return worker


def _request(params, slot: int = 3, request_id: int = 7):
    temperature, top_k, top_p, min_p, seed = params
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, seed=seed
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
        py_request_id=request_id,
        py_decoding_iter=0,
    )


def _meta(k: int, on: bool, mode=AdvancedSamplingMode.NO_TOPK) -> SpecMetadata:
    """A one-model DSpark metadata with rejection sampling (the advanced draft path) at draft length ``k``."""
    meta = SpecMetadata(
        max_num_requests=MAX_REQUESTS,
        max_draft_len=k,
        max_total_draft_tokens=k,
        spec_dec_mode=SpeculativeDecodingMode.DSPARK,
    )
    assert meta.spec_dec_mode.use_one_engine()
    meta.runtime_draft_len = k
    meta.advanced_sampling_mode = mode
    meta.use_rejection_sampling = True
    meta.vocab_size = V
    meta.draft_vocab_size = V
    meta.tail_bookkeep = on
    return meta


def _fill_like_populate(meta: SpecMetadata, params, offset: int, device: str = "cpu") -> None:
    """CPU mirror of populate_sampling_params_for_one_model + _populate_request_rng_state for one generation request
    (the in-tree code allocates on CUDA): request_* hold its values, its K + 1 per-token rows hold the same values and
    the offset window ``offset + r * SPAN``; every other row holds junk, so a wrong slice shows."""
    temperature, top_k, top_p, min_p, seed = params
    k = meta.max_draft_len
    rows = (k + 1) * MAX_REQUESTS
    junk = torch.arange(rows, device=device)

    def per_token(value, dtype):
        out = (junk + 1000).to(dtype)
        out[: k + 1] = value
        return out

    def per_request(value, dtype):
        out = (junk[:MAX_REQUESTS] + 2000).to(dtype)
        out[0] = value
        return out

    seed = interface.DEFAULT_SAMPLING_SEED if seed is None else seed
    seed = seed - (1 << 64) if seed >= (1 << 63) else seed
    meta.temperatures = per_token(temperature, torch.float32)
    meta.top_ks = per_token(top_k, torch.int32)
    meta.top_ps = per_token(top_p, torch.float32)
    meta.min_ps = per_token(min_p, torch.float32)
    meta.request_temperatures = per_request(temperature, torch.float32)
    meta.request_top_ks = per_request(top_k, torch.int32)
    meta.request_top_ps = per_request(top_p, torch.float32)
    meta.request_min_ps = per_request(min_p, torch.float32)
    meta.request_seeds = per_request(seed, torch.int64)
    meta.request_offsets = per_request(offset, torch.int64)
    meta.seeds = per_token(seed, torch.int64)
    meta.offsets = (junk + 3000).to(torch.int64)
    meta.offsets[: k + 1] = torch.tensor([offset + r * _RNG_SLOT_SPAN for r in range(k + 1)])
    meta.is_all_greedy_sample = False


def _populate(meta: SpecMetadata, params, slot: int = 3, request_id: int = 7) -> None:
    """The real populate for a one-request decode batch (CUDA)."""
    meta.populate_sampling_params_for_one_model([_request(params, slot, request_id)])
    assert not meta.is_all_greedy_sample


class _Sampler:
    """A deterministic stand-in for the advanced samplers: records (a copy of) every tensor it is handed and returns
    the argmax as int32 tokens with softmax probs."""

    def __init__(self):
        self.seen = []

    def __call__(
        self, mode, logits, temperatures, top_ks, top_ps, min_ps, *, seed=None, offset=None
    ):
        self.seen.append(
            tuple(t.clone() for t in (temperatures, top_ks, top_ps, min_ps, seed, offset))
        )
        return logits.argmax(-1).to(torch.int32), torch.softmax(logits, dim=-1)


def _fake_rejection(draft_probs, draft_token_ids, target_probs, seed, offset):
    """A stand-in acceptance kernel: fresh int32 outputs that depend on every input, as the in-tree op's."""
    bonus = target_probs[:, -1].argmax(-1, keepdim=True).to(torch.int32)
    accepted = torch.cat([draft_token_ids, bonus], dim=1)
    num = ((seed ^ offset) % (draft_token_ids.shape[1] + 1) + 1).to(torch.int32)
    num = num + (draft_probs.sum(dim=(1, 2)) < 0).to(torch.int32)  # reads the draft probs (adds 0)
    return accepted, num


def _patch_samplers(monkeypatch) -> _Sampler:
    sampler = _Sampler()
    monkeypatch.setattr(SpecWorkerBase, "_sample_from_logits_with_probs", staticmethod(sampler))
    monkeypatch.setattr(
        SpecWorkerBase,
        "_compute_probs_from_logits",
        staticmethod(lambda mode, logits, t, k, p, m: torch.softmax(logits / t.unsqueeze(-1), -1)),
    )
    monkeypatch.setattr(interface, "rejection_sampling_one_model", _fake_rejection)
    return sampler


def _assert_same(a, b, what: str) -> None:
    if isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list)) and len(a) == len(b), what
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{what}[{i}]")
        return
    if a is None or b is None:
        assert a is None and b is None, what
        return
    assert a.dtype == b.dtype and a.shape == b.shape, (
        f"{what}: {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}"
    )
    if a.dtype == torch.float32:
        a, b = a.view(torch.int32), b.view(torch.int32)  # bits, so -0.0 / NaN payloads count too
    assert torch.equal(a.cpu(), b.cpu()), what


# ---------------------------------------------------------------------------------------------------------------------
# The switch: off by default everywhere, copied to the spec metadata.
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_the_hook_and_the_metadata_flag_are_off_by_default():
    assert RuntimeHooks().spec_tail_bookkeep is False
    assert DEFAULT_RUNTIME_HOOKS.spec_tail_bookkeep is False
    assert SpecMetadata.tail_bookkeep is False
    assert (
        SpecMetadata(max_num_requests=1, max_draft_len=1, max_total_draft_tokens=1).tail_bookkeep
        is False
    )
    assert not RuntimeHooks(spec_tail_bookkeep=True).skips_unchanged_metadata_uploads


@pytest.mark.cpu_only
@pytest.mark.parametrize("on", [False, True])
def test_setup_speculative_copies_the_hook_and_logs_nothing(monkeypatch, on):
    created = types.SimpleNamespace()
    lines = []
    monkeypatch.setattr(runner_module, "create_spec_metadata", lambda *args: created)
    monkeypatch.setattr(runner_module.logger, "info", lines.append)
    metadata = object.__new__(DecoderRunner)
    metadata.runtime_hooks = RuntimeHooks(spec_tail_bookkeep=on)
    metadata.spec_metadata = None
    metadata._config = None
    metadata.model = types.SimpleNamespace(config=None)

    assert metadata._set_up_spec_metadata(None) is created
    assert created.tail_bookkeep is on
    assert lines == []  # the arm line is the target's (it knows which items its worker takes)


# ---------------------------------------------------------------------------------------------------------------------
# The shared precondition (host state and shapes only).
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize("k", DRAFT_LENS)
def test_the_precondition_holds_for_one_decode_request(k):
    meta = _meta(k, on=True)
    _fill_like_populate(meta, REQUESTS[0], offset=0)
    assert tail_bookkeep_draft_len(meta, 0, 1) == k


def _breaks():
    def flag_off(meta):
        meta.tail_bookkeep = False

    def tree(meta):
        meta.is_spec_dec_tree = True

    def dynamic_tree(meta):
        meta.is_spec_dec_dynamic_tree = True

    def ragged(meta):
        meta.verify_lens = torch.ones(1, dtype=torch.int32)

    def dynamic_draft_len(meta):
        meta.runtime_draft_len = meta.max_draft_len - 1

    def short_buffer(meta):
        meta.seeds = meta.seeds[: meta.max_draft_len]

    def missing_buffer(meta):
        meta.min_ps = None

    def wrong_dtype(meta):
        meta.offsets = meta.offsets.to(torch.int32)

    def strided_buffer(meta):
        meta.top_ps = torch.stack([meta.top_ps, meta.top_ps], dim=1)[:, 0]

    return [flag_off, tree, dynamic_tree, ragged, dynamic_draft_len, short_buffer, missing_buffer, wrong_dtype,
            strided_buffer]  # fmt: skip


@pytest.mark.cpu_only
@pytest.mark.parametrize("brk", _breaks(), ids=lambda f: f.__name__)
def test_the_precondition_fails_closed(brk):
    meta = _meta(3, on=True)
    _fill_like_populate(meta, REQUESTS[0], offset=0)
    brk(meta)
    assert tail_bookkeep_draft_len(meta, 0, 1) == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_contexts,batch_size", [(0, 2), (1, 2), (0, 0), (1, 1), (0, None)])
def test_the_precondition_needs_exactly_one_generation_request(num_contexts, batch_size):
    meta = _meta(3, on=True)
    _fill_like_populate(meta, REQUESTS[0], offset=0)
    assert tail_bookkeep_draft_len(meta, num_contexts, batch_size) == 0


# ---------------------------------------------------------------------------------------------------------------------
# B1 + S1: the draft block (sample_draft_tokens, block form).
# ---------------------------------------------------------------------------------------------------------------------


def _draft_block(meta, logits, d2t=None):
    worker = _worker(d2t=d2t)
    tokens = worker.sample_draft_tokens(logits, meta, 1, num_contexts=0)
    return tokens


@pytest.mark.cpu_only
@pytest.mark.parametrize("k", DRAFT_LENS)
@pytest.mark.parametrize("params", REQUESTS)
def test_draft_block_cpu_on_matches_off(monkeypatch, k, params):
    sampler = _patch_samplers(monkeypatch)
    logits = torch.randn(1, k, V, generator=torch.Generator().manual_seed(k))
    outs = {}
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, params, offset=5 * k)
        tokens = _draft_block(meta, logits)
        outs[on] = (tokens, sampler.seen[-1])
    _assert_same(outs[False], outs[True], "tokens + sampler inputs")
    assert outs[True][0].dtype == torch.int32


@pytest.mark.cpu_only
def test_draft_block_cpu_flag_off_takes_the_in_tree_lines(monkeypatch):
    """Off: the parameters are fresh repeats (not views of the per-token buffers) and the tokens a fresh int32 copy
    (not the sampler's tensor)."""
    k = 4
    seen_tokens = []

    def sampler(mode, logits, temperatures, top_ks, top_ps, min_ps, *, seed=None, offset=None):
        tokens = logits.argmax(-1).to(torch.int32)
        seen_tokens.append(tokens)
        sampler.params = (temperatures, top_ks, top_ps, min_ps)
        return tokens, torch.softmax(logits, -1)

    monkeypatch.setattr(SpecWorkerBase, "_sample_from_logits_with_probs", staticmethod(sampler))
    logits = torch.randn(1, k, V)
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[1], offset=0)
        tokens = _draft_block(meta, logits)
        views = [p.untyped_storage().data_ptr() for p in sampler.params]
        bufs = [meta.temperatures, meta.top_ks, meta.top_ps, meta.min_ps]
        shared = [v == b.untyped_storage().data_ptr() for v, b in zip(views, bufs)]
        aliases_sampler = (
            tokens.untyped_storage().data_ptr() == seen_tokens[-1].untyped_storage().data_ptr()
        )
        assert shared == [on] * 4
        assert aliases_sampler is on


@pytest.mark.cpu_only
def test_s1_keeps_the_round_trip_with_a_d2t_map(monkeypatch):
    _patch_samplers(monkeypatch)
    k = 3
    d2t = torch.arange(V, dtype=torch.int64) % 5
    logits = torch.randn(1, k, V)
    outs = []
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[0], offset=0)
        outs.append(_draft_block(meta, logits, d2t=d2t))
    _assert_same(outs[0], outs[1], "d2t-mapped tokens")


@pytest.mark.cpu_only
def test_b1_is_not_taken_for_a_padded_batch(monkeypatch):
    """Two generation requests (a padded graph): the per-request repeats run with the flag on too."""
    sampler = _patch_samplers(monkeypatch)
    k = 3
    meta = _meta(k, on=True)
    _fill_like_populate(meta, REQUESTS[0], offset=0)
    worker = _worker()
    worker.advanced_sample_draft_block(torch.randn(2, k, V), meta, 0, 2)
    temps = sampler.seen[-1][0]
    assert torch.equal(temps, meta.request_temperatures[:2].repeat_interleave(k))


# ---------------------------------------------------------------------------------------------------------------------
# A1: the rejection accept.
# ---------------------------------------------------------------------------------------------------------------------


def _accept(meta, logits, draft_tokens, draft_probs, num_contexts, batch_size, force=0.0):
    worker = _worker(force_num_accepted_tokens=force)
    return worker._sample_and_accept_draft_tokens_rejection(
        logits, draft_tokens, draft_probs, num_contexts, batch_size, meta
    )


def _accept_inputs(k, num_gens, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(num_gens * (k + 1), V, generator=g, device=device)
    draft_tokens = torch.randint(0, V, (num_gens, k), generator=g, device=device, dtype=torch.int32)
    draft_probs = torch.softmax(torch.randn(num_gens, k, V, generator=g, device=device), -1)
    return logits, draft_tokens, draft_probs


@pytest.mark.cpu_only
@pytest.mark.parametrize("k", DRAFT_LENS)
@pytest.mark.parametrize("force", [0.0, 2.0])
def test_accept_cpu_on_matches_off(monkeypatch, k, force):
    _patch_samplers(monkeypatch)
    logits, draft_tokens, draft_probs = _accept_inputs(k, 1, seed=k)
    outs = []
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[2], offset=96 * k)
        outs.append(_accept(meta, logits, draft_tokens, draft_probs, 0, 1, force=force))
    _assert_same(outs[0], outs[1], "accepted tokens + counts")


@pytest.mark.cpu_only
def test_accept_flag_on_returns_the_kernel_outputs(monkeypatch):
    returned = []

    def rejection(*args, **kwargs):
        returned.append(_fake_rejection(*args, **kwargs))
        return returned[-1]

    _patch_samplers(monkeypatch)
    monkeypatch.setattr(interface, "rejection_sampling_one_model", rejection)
    k = 3
    logits, draft_tokens, draft_probs = _accept_inputs(k, 1)
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[0], offset=0)
        accepted, num = _accept(meta, logits, draft_tokens, draft_probs, 0, 1)
        assert (accepted is returned[-1][0]) is on
        assert (num is returned[-1][1]) is on


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "bad",
    ["int64_tokens", "int64_counts", "strided_tokens"],
)
def test_accept_falls_back_when_the_kernel_outputs_differ_in_layout(monkeypatch, bad):
    def rejection(*args, **kwargs):
        accepted, num = _fake_rejection(*args, **kwargs)
        if bad == "int64_tokens":
            accepted = accepted.long()
        elif bad == "int64_counts":
            num = num.long()
        else:
            strided = torch.zeros(accepted.shape[0], 2 * accepted.shape[1], dtype=accepted.dtype)
            strided[:, ::2] = accepted
            accepted = strided[:, ::2]
        return accepted, num

    _patch_samplers(monkeypatch)
    monkeypatch.setattr(interface, "rejection_sampling_one_model", rejection)
    k = 3
    logits, draft_tokens, draft_probs = _accept_inputs(k, 1)
    outs = []
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[0], offset=0)
        outs.append(_accept(meta, logits, draft_tokens, draft_probs, 0, 1))
    _assert_same(outs[0], outs[1], "fallback")
    assert outs[1][0].dtype == torch.int32 and outs[1][0].is_contiguous()


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_gens", [2, 4])
def test_accept_with_several_requests_matches_off(monkeypatch, num_gens):
    """A1 is not limited to one request: any decode-only accept returns the kernel's outputs."""
    _patch_samplers(monkeypatch)
    k = 3
    logits, draft_tokens, draft_probs = _accept_inputs(k, num_gens)
    outs = []
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[0], offset=0)
        outs.append(_accept(meta, logits, draft_tokens, draft_probs, 0, num_gens))
    _assert_same(outs[0], outs[1], "accept")


@pytest.mark.cpu_only
def test_accept_with_a_context_row_takes_the_in_tree_lines(monkeypatch):
    """num_contexts > 0: the outputs are assembled into fresh buffers whatever the flag."""
    _patch_samplers(monkeypatch)
    monkeypatch.setattr(
        _Worker,
        "_sample_tokens_for_batch",
        lambda self, logits, meta, n, b: logits.argmax(-1).to(torch.int32),
    )
    returned = []

    def rejection(*args, **kwargs):
        returned.append(_fake_rejection(*args, **kwargs))
        return returned[-1]

    monkeypatch.setattr(interface, "rejection_sampling_one_model", rejection)
    k = 3
    gen_logits, draft_tokens, draft_probs = _accept_inputs(k, 1)
    logits = torch.cat([torch.randn(1, V), gen_logits])
    outs = []
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[0], offset=0)
        accepted, num = _accept(meta, logits, draft_tokens, draft_probs, 1, 2)
        assert accepted is not returned[-1][0] and num is not returned[-1][1]
        # The context row defines only its first token (the rest of its row is the in-tree torch.empty), so compare
        # the defined entries: the context row's first token, the generation rows and every count.
        outs.append((accepted[:1, :1], accepted[1:], num))
    _assert_same(outs[0], outs[1], "mixed batch")


# ---------------------------------------------------------------------------------------------------------------------
# CUDA: the real populate, eager and inside torch.cuda.graph.
# ---------------------------------------------------------------------------------------------------------------------


def _graph(fn):
    """Capture ``fn`` (after a side-stream warmup, as torch.cuda.graphs asks); returns (graph, captured outputs)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _step(meta, worker, logits, accept_inputs, sampler):
    """One sampled step's tail: the rejection accept, then the draft block; the outputs and the draft-prob row."""
    v_logits, draft_tokens, draft_probs = accept_inputs
    accepted = worker._sample_and_accept_draft_tokens_rejection(
        v_logits, draft_tokens, draft_probs, 0, 1, meta
    )
    tokens = worker.sample_draft_tokens(logits, meta, 1, num_contexts=0)
    return accepted, tokens, sampler.seen[-1]


@needs_cuda
@pytest.mark.parametrize("k", DRAFT_LENS)
@pytest.mark.parametrize("mode", [AdvancedSamplingMode.NO_TOPK, AdvancedSamplingMode.FULL])
def test_cuda_eager_on_matches_off(monkeypatch, k, mode):
    sampler = _patch_samplers(monkeypatch)
    worker = _worker()
    metas = {on: _meta(k, on, mode=mode) for on in (False, True)}
    logits = torch.randn(1, k, V, device="cuda")
    inputs = _accept_inputs(k, 1, device="cuda", seed=k)
    for step, params in enumerate(REQUESTS):
        outs = {}
        for on, meta in metas.items():
            _populate(meta, params)
            outs[on] = _step(meta, worker, logits + step, inputs, sampler)
            outs[on] += (meta.draft_probs[3, :k].clone(),)
        torch.cuda.synchronize()
        _assert_same(outs[False], outs[True], f"step {step}")


@needs_cuda
@pytest.mark.parametrize("k", [1, 3, 7])
def test_cuda_graph_on_matches_off(monkeypatch, k):
    """The ON step captured on the first request's values, then replayed after each later populate (same buffers,
    new values), against the in-tree step run eagerly on the same values."""
    sampler = _patch_samplers(monkeypatch)
    worker = _worker()
    on_meta, off_meta = _meta(k, True), _meta(k, False)
    logits = torch.randn(1, k, V, device="cuda")
    inputs = _accept_inputs(k, 1, device="cuda", seed=k)
    _populate(on_meta, REQUESTS[0])
    _populate(off_meta, REQUESTS[0])
    graph, captured = _graph(lambda: _step(on_meta, worker, logits, inputs, sampler))
    for step, params in enumerate(REQUESTS[1:], start=1):
        _populate(on_meta, params)
        graph.replay()
        _populate(off_meta, params)
        expected = _step(off_meta, worker, logits, inputs, sampler)
        torch.cuda.synchronize()
        _assert_same(captured, expected, f"replay {step}")
        _assert_same(on_meta.draft_probs[3, :k], off_meta.draft_probs[3, :k], f"draft probs {step}")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
