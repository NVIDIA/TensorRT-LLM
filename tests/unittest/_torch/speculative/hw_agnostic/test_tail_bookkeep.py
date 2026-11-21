# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SpecMetadata.tail_bookkeep shortcuts (draft-block parameter views, int32 draft tokens without a d2t map,
decode-only rejection accept returning the kernel outputs) must match the in-tree path bit for bit, eagerly and under
CUDA graph replay. The samplers are deterministic stand-ins that record their inputs."""

import types
from typing import Optional

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
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
DRAFT_LENS = [1, 4, 7]
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
        py_spec_sampling_params=None,
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
    """CPU mirror of populate_sampling_params_for_one_model for one generation request; other rows hold junk."""
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


def _fake_rejection(draft_probs, draft_token_ids, target_probs, seed, offset, deterministic=False):
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


_BREAKS = {
    "flag_off": lambda m: setattr(m, "tail_bookkeep", False),
    "tree": lambda m: setattr(m, "is_spec_dec_tree", True),
    "dynamic_tree": lambda m: setattr(m, "is_spec_dec_dynamic_tree", True),
    "ragged": lambda m: setattr(m, "verify_lens", torch.ones(1, dtype=torch.int32)),
    "dynamic_draft_len": lambda m: setattr(m, "runtime_draft_len", m.max_draft_len - 1),
    "short_buffer": lambda m: setattr(m, "seeds", m.seeds[: m.max_draft_len]),
    "missing_buffer": lambda m: setattr(m, "min_ps", None),
    "wrong_dtype": lambda m: setattr(m, "offsets", m.offsets.to(torch.int32)),
    "strided_buffer": lambda m: setattr(
        m, "top_ps", torch.stack([m.top_ps, m.top_ps], dim=1)[:, 0]
    ),
}


@pytest.mark.cpu_only
@pytest.mark.parametrize("brk", list(_BREAKS))
def test_the_precondition_fails_closed(brk):
    meta = _meta(3, on=True)
    _fill_like_populate(meta, REQUESTS[0], offset=0)
    _BREAKS[brk](meta)
    assert tail_bookkeep_draft_len(meta, 0, 1) == 0


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_contexts,batch_size,k,expected",
    [
        (0, 1, 1, 1),
        (0, 1, 7, 7),
        (0, 2, 3, 0),
        (1, 2, 3, 0),
        (0, 0, 3, 0),
        (1, 1, 3, 0),
        (0, None, 3, 0),
    ],
)
def test_the_precondition_needs_exactly_one_generation_request(
    num_contexts, batch_size, k, expected
):
    meta = _meta(k, on=True)
    _fill_like_populate(meta, REQUESTS[0], offset=0)
    assert tail_bookkeep_draft_len(meta, num_contexts, batch_size) == expected


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
@pytest.mark.parametrize("num_gens", [1, 3])
def test_accept_cpu_on_matches_off(monkeypatch, k, force, num_gens):
    _patch_samplers(monkeypatch)
    logits, draft_tokens, draft_probs = _accept_inputs(k, num_gens, seed=k)
    outs = []
    for on in (False, True):
        meta = _meta(k, on)
        _fill_like_populate(meta, REQUESTS[2], offset=96 * k)
        outs.append(_accept(meta, logits, draft_tokens, draft_probs, 0, num_gens, force=force))
    _assert_same(outs[0], outs[1], "accepted tokens + counts")


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
@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parametrize("mode", [AdvancedSamplingMode.NO_TOPK, AdvancedSamplingMode.FULL])
def test_cuda_graph_on_matches_off(monkeypatch, k, mode):
    """ON captured on the first request, replayed after each later populate, vs OFF run eagerly."""
    sampler = _patch_samplers(monkeypatch)
    worker = _worker()
    on_meta, off_meta = _meta(k, True, mode=mode), _meta(k, False, mode=mode)
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
