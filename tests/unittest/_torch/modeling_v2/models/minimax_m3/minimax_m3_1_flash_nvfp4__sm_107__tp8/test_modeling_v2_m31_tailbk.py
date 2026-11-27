# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAILBK's worker items on this target, R1 (``M3DSparkWorker._rng_state_per_request``) and Z1
(``M3DSparkWorker._zero_padding_rows``), against the engine's in-tree methods bit for bit, eagerly and inside
``torch.cuda.graph``. They import the target's dspark module and need CUDA.
"""

import importlib
import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp8"

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="imports the sm_107 target's worker; CUDA buffers",
)

V = 64
SLOT = 3
# (temperature, top_k, top_p, min_p, seed): unseeded, int64-edge seed, a uint64 seed (negative as int64), seed 0.
REQUESTS = [
    (0.7, 0, 0.9, 0.0, None),
    (1.3, 17, 0.5, 0.05, (1 << 63) - 1),
    (0.25, 3, 1.0, 0.2, (1 << 64) - 1),
    (1.0, 0, 0.95, 0.0, 0),
]


def _import(module: str):
    try:
        return importlib.import_module(
            f"tensorrt_llm._torch.modeling_v2.models.minimax_m3.{TARGET}.{module}"
        )
    except NoTarget as exc:
        pytest.skip(str(exc))


def _dspark():
    return _import("dspark")


def _interface():
    return importlib.import_module("tensorrt_llm._torch.speculative.interface")


@pytest.fixture(autouse=True)
def _zeroed_padding_rows(monkeypatch):
    # The sm_107 targets keep padding rows zeroed (the engine's default).
    monkeypatch.setattr(_interface(), "_CHEAP_PADDING_ROWS", False)


def _worker(attention_dp: bool = False):
    worker = object.__new__(_dspark().M3DSparkWorker)  # the overrides read only these fields
    worker.mapping = types.SimpleNamespace(enable_attention_dp=attention_dp, tp_size=1, tp_rank=0)
    worker._d2t = None
    worker.force_num_accepted_tokens = 0.0
    return worker


def _request(params, request_id: int = 7):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

    temperature, top_k, top_p, min_p, seed = params
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p, seed=seed
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=SLOT,
        py_request_id=request_id,
        py_decoding_iter=0,
        py_spec_sampling_params=None,
    )


def _meta(k: int, on: bool):
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    interface = _interface()
    meta = interface.SpecMetadata(
        max_num_requests=8,
        max_draft_len=k,
        max_total_draft_tokens=k,
        spec_dec_mode=interface.SpeculativeDecodingMode.DSPARK,
    )
    meta.runtime_draft_len = k
    meta.advanced_sampling_mode = AdvancedSamplingMode.NO_TOPK
    meta.use_rejection_sampling = True
    meta.vocab_size = V
    meta.draft_vocab_size = V
    meta.tail_bookkeep = on
    return meta


def _populate(meta, params) -> None:
    meta.populate_sampling_params_for_one_model([_request(params)])
    assert not meta.is_all_greedy_sample


def _stock_rng(worker, meta, k):
    base = _interface().SpecWorkerBase._rng_state_per_request
    accept = base(worker, meta, 0, 1, slot=meta.rng_accept_slot)
    block = base(worker, meta, 0, 1, repeat=k, slot=meta.rng_draft_slot(0))
    return accept, block


def _r1_rng(worker, meta, k, stash: bool = True):
    accept = worker._rng_state_per_request(meta, 0, 1, slot=meta.rng_accept_slot)
    if not stash:
        worker._tailbk_win = None
    block = worker._rng_state_per_request(meta, 0, 1, repeat=k, slot=meta.rng_draft_slot(0))
    return accept, block


def _assert_rng_equal(got, want, what):
    for (g_seed, g_off), (w_seed, w_off), part in zip(got, want, ("accept", "block")):
        for g, w, name in ((g_seed, w_seed, "seed"), (g_off, w_off, "offset")):
            assert g.dtype == w.dtype == torch.int64, f"{what} {part} {name} dtype"
            assert g.shape == w.shape and g.is_contiguous(), f"{what} {part} {name} shape"
            assert torch.equal(g, w), f"{what} {part} {name}: {g.tolist()} vs {w.tolist()}"


@needs_cuda
@pytest.mark.parametrize("k", [1, 7])
@pytest.mark.parametrize("stash", [True, False], ids=["stash", "no_stash"])
def test_r1_eager_matches_the_in_tree_rows(k, stash):
    worker = _worker()
    on, off = _meta(k, True), _meta(k, False)
    for params in REQUESTS:
        _populate(on, params)
        _populate(off, params)
        _assert_rng_equal(
            _r1_rng(worker, on, k, stash), _stock_rng(worker, off, k), f"K={k} {params}"
        )
        assert worker._tailbk_win is None  # the block consumed (or never had) the window


@needs_cuda
@pytest.mark.parametrize("k", [7])
def test_r1_offsets_that_wrap_int64(k):
    """The largest offset a host upload allows (o + K * SPAN = 2^63 - 1): the accept and draft rows wrap past 2^63
    in both, to the same bits (int64 addition is associative mod 2^64)."""
    span = _interface()._RNG_SLOT_SPAN
    worker = _worker()
    on, off = _meta(k, True), _meta(k, False)
    o = (1 << 63) - 1 - k * span
    for meta in (on, off):
        _populate(meta, REQUESTS[1])
        meta.request_offsets[0] = o
        meta.offsets[: k + 1] = torch.tensor(
            [o + r * span for r in range(k + 1)], dtype=torch.int64, device="cuda"
        )
    got, want = _r1_rng(worker, on, k), _stock_rng(worker, off, k)
    _assert_rng_equal(got, want, "wrap")
    assert int(want[0][1][0]) < 0  # the accept row did wrap


@needs_cuda
@pytest.mark.parametrize(
    "case", ["flag_off", "two_requests", "context_row", "step_form_slot", "dynamic_draft_len"]
)
def test_r1_falls_back_to_the_in_tree_rows(case):
    k = 3
    worker = _worker()
    meta = _meta(k, case != "flag_off")
    _populate(meta, REQUESTS[1])
    base = _interface().SpecWorkerBase._rng_state_per_request
    args = {
        "flag_off": ((0, 1), {"slot": meta.rng_accept_slot}),
        "two_requests": ((0, 2), {"slot": meta.rng_accept_slot}),
        "context_row": ((1, 2), {"slot": meta.rng_accept_slot}),
        "step_form_slot": ((0, 1), {"slot": meta.rng_draft_slot(1)}),
        "dynamic_draft_len": ((0, 1), {"repeat": k - 1, "slot": meta.rng_draft_slot(0)}),
    }[case]
    if case == "dynamic_draft_len":
        meta.runtime_draft_len = k - 1
    got = worker._rng_state_per_request(meta, *args[0], **args[1])
    want = base(worker, meta, *args[0], **args[1])
    assert worker._tailbk_win is None
    for g, w in zip(got, want):
        assert torch.equal(g, w)


@needs_cuda
@pytest.mark.parametrize("k", [1, 7])
def test_r1_in_a_cuda_graph(k):
    """Captured on the first request's populate, replayed after each later one (same buffers, new values)."""
    worker = _worker()
    on, off = _meta(k, True), _meta(k, False)
    _populate(on, REQUESTS[0])
    _populate(off, REQUESTS[0])

    def step():
        worker._tailbk_win = None  # the forward's reset
        return _r1_rng(worker, on, k)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for params in REQUESTS[1:]:
        _populate(on, params)
        _populate(off, params)
        graph.replay()
        want = _stock_rng(worker, off, k)
        torch.cuda.synchronize()
        _assert_rng_equal(captured, want, f"replay {params}")


DUMMY_ROW = 16  # spec_metadata.dummy_slot_row


def _z1_meta(slots, on: bool = True, ragged: bool = False, fused: bool = False):
    from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

    return types.SimpleNamespace(
        tail_bookkeep=on,
        batch_slot_ids=torch.tensor(slots, dtype=torch.long, device="cuda"),
        dummy_slot_row=DUMMY_ROW,
        is_ragged_verify=ragged,
        advanced_sampling_mode=AdvancedSamplingMode.FULL if fused else AdvancedSamplingMode.NO_TOPK,
    )


def _z1_logits(rows: int) -> torch.Tensor:
    logits = torch.randn(rows, V, device="cuda")
    logits[:, 0] = float("inf")
    logits[:, 1] = -float("inf")
    logits[:, 2] = -0.0
    return logits


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _stock_zero(*args, **kwargs):
    return _interface().SpecWorkerBase._zero_padding_rows(*args, **kwargs)


@needs_cuda
@pytest.mark.parametrize("rows_per_request", [1, 8])
def test_z1_skips_only_while_capturing_batch_one(rows_per_request):
    worker = _worker()
    meta = _z1_meta([SLOT])
    logits = _z1_logits(rows_per_request)
    eager = worker._zero_padding_rows(logits, meta, 0, 1, rows_per_request)
    assert (
        eager is not logits
    )  # eager steps (the graph warmup's dummy included) keep the in-tree zeroing
    graph, captured = _capture(
        lambda: worker._zero_padding_rows(logits, meta, 0, 1, rows_per_request)
    )
    assert captured is logits
    graph.replay()
    # The in-tree result at the one real request is a bitwise copy of the logits.
    stock = _stock_zero(logits, meta, 0, 1, rows_per_request)
    torch.cuda.synchronize()
    assert torch.equal(stock.view(torch.int32), logits.view(torch.int32))


@needs_cuda
@pytest.mark.parametrize("batch, rows_per_request", [(2, 1), (8, 8)])
def test_z1_keeps_the_zeroing_in_padded_graphs(batch, rows_per_request):
    """Batches 2..8 can hold dummies: the captured graph zeroes their rows exactly as the in-tree method."""
    worker = _worker()
    real = batch // 2
    meta = _z1_meta([SLOT + i for i in range(real)] + [DUMMY_ROW] * (batch - real))
    logits = _z1_logits(batch * rows_per_request)
    graph, captured = _capture(
        lambda: worker._zero_padding_rows(logits, meta, 0, batch, rows_per_request)
    )
    assert captured is not logits
    graph.replay()
    stock = _stock_zero(logits, meta, 0, batch, rows_per_request)
    torch.cuda.synchronize()
    assert torch.equal(captured.view(torch.int32), stock.view(torch.int32))
    assert bool((captured[real * rows_per_request :] == 0).all())
