# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for TRTLLM_Q35_TEMPERATURE_GRAPHS.

The one-engine advanced-sampling CUDA graphs are captured with synthetic
top-k / top-p values (0.7 / 50 / 0.9), so a capture-time check such as the
vocabulary-sharded target sampling's "temperature-only batch" predicate sees
top-k / top-p enabled and the stock sampler is what gets recorded; steady
temperature-only decode then replays the stock sampler. With the flag the model
engine captures a third, temperature-only variant under its own graph key bit
(``KeyType.q35_temperature_only``), and the per-batch scan sets that bit for
non-greedy batches without top-k / top-p (and without rejection sampling).

Same style as test_capture_override_leak.py: a real (base) ``SpecMetadata``, an
unbound ``CUDAGraphRunner`` method on a stand-in, no GPU.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.interface import (
    Q35_TEMPERATURE_GRAPHS_ENV,
    SpecMetadata,
    SpeculativeDecodingMode,
    q35_temperature_graphs_enabled,
)

DISABLE_TOPK_VAL = torch.iinfo(torch.int32).max


def _meta(captured=True, rejection=False):
    meta = SpecMetadata(max_num_requests=8, max_draft_len=6, max_total_draft_tokens=6)
    meta.q35_temperature_graphs = captured
    meta.use_rejection_sampling = rejection
    return meta


def _request(temperature=None, top_k=None, top_p=None, slot=0):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=[temperature] if temperature is not None else None,
            top_k=[top_k] if top_k is not None else None,
            top_p=[top_p] if top_p is not None else None,
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
        py_draft_tokens=[0] * 6,
    )


def _scan(meta, requests):
    normalized, _ = SpecMetadata._scan_one_model_sampling(meta, requests)
    return [entry[:3] for entry in normalized]


def _key(meta, requests):
    """CUDAGraphRunner.get_graph_key, unbound, for a generation-only batch."""
    runner = types.SimpleNamespace(
        _get_seq_len_mode=lambda *args: False,
        config=types.SimpleNamespace(is_draft_model=False),
    )
    batch = types.SimpleNamespace(
        batch_size=len(requests), generation_requests=requests, context_requests=[]
    )
    return CUDAGraphRunner.get_graph_key(runner, batch, spec_metadata=meta)


@pytest.mark.parametrize(
    "params",
    [
        dict(temperature=1.0),  # temperature only, the other fields left unset
        dict(temperature=0.6),
        dict(temperature=1.0, top_k=0, top_p=1.0),  # explicit "disabled" values
    ],
)
def test_temperature_only_batch_selects_the_variant(params):
    meta = _meta()
    _scan(meta, [_request(**params), _request(**params, slot=1)])
    assert meta.is_all_greedy_sample is False
    assert meta.q35_temperature_only is True
    key = _key(meta, [_request(**params), _request(**params, slot=1)])
    assert key.is_all_greedy_sample is False and key.q35_temperature_only is True


def test_mixed_greedy_and_temperature_rows_select_the_variant():
    # The Gumbel kernel keeps rows below the greedy threshold greedy.
    meta = _meta()
    _scan(meta, [_request(), _request(temperature=1.0, slot=1)])
    assert meta.q35_temperature_only is True


@pytest.mark.parametrize(
    "params",
    [
        dict(temperature=1.0, top_k=20),
        dict(temperature=1.0, top_p=0.95),
        dict(temperature=0.6, top_k=20, top_p=0.95),
    ],
)
def test_top_k_or_top_p_keeps_the_stock_advanced_graph(params):
    meta = _meta()
    reqs = [_request(temperature=1.0, slot=1), _request(**params)]
    _scan(meta, reqs)
    assert meta.is_all_greedy_sample is False
    assert meta.q35_temperature_only is False
    assert _key(meta, reqs).q35_temperature_only is False


def test_greedy_batch_keeps_the_greedy_graph():
    meta = _meta()
    reqs = [_request(), _request(temperature=0.0, slot=1)]
    _scan(meta, reqs)
    assert meta.is_all_greedy_sample is True
    assert meta.q35_temperature_only is False
    key = _key(meta, reqs)
    assert key.is_all_greedy_sample is True and key.q35_temperature_only is False


def test_negative_controls_variant_not_captured_or_rejection_sampling():
    # Not captured (flag off): the bit must stay False, or temperature-only
    # batches would look up a graph that does not exist and run eager.
    meta = _meta(captured=False)
    reqs = [_request(temperature=1.0)]
    _scan(meta, reqs)
    assert meta.q35_temperature_only is False
    assert _key(meta, reqs) == KeyType(
        batch_size=1, draft_len=6, is_first_draft=False, is_all_greedy_sample=False
    )
    # Rejection sampling takes its own path; never the sharded sampler.
    meta = _meta(rejection=True)
    _scan(meta, reqs)
    assert meta.q35_temperature_only is False


def test_capture_override_records_the_temperature_only_variant():
    meta = _meta()
    meta._force_non_greedy_for_capture = True
    meta._q35_temperature_only_for_capture = True
    graph_meta = meta.create_cuda_graph_metadata(4)
    warmup = [_request(slot=None), _request(slot=None)]
    assert _scan(graph_meta, warmup) == [(0.7, DISABLE_TOPK_VAL, 1.0)] * 2
    assert graph_meta.is_all_greedy_sample is False
    assert graph_meta.skip_top_k and graph_meta.skip_top_p
    assert not graph_meta.skip_temperature
    assert graph_meta.q35_temperature_only is True
    assert _key(graph_meta, warmup).q35_temperature_only is True


def test_stock_capture_override_is_unchanged():
    # Pass 2 (stock advanced variant) with the variant enabled: synthetic
    # top-k / top-p as before, and its key bit stays False.
    meta = _meta()
    meta._force_non_greedy_for_capture = True
    warmup = [_request(slot=None)]
    assert _scan(meta, warmup) == [(0.7, 50, 0.9)]
    assert not meta.skip_top_k and not meta.skip_top_p
    assert meta.q35_temperature_only is False
    assert _key(meta, warmup).q35_temperature_only is False


def test_teardown_clears_the_capture_flag_on_cached_copies():
    meta = _meta()
    meta._force_non_greedy_for_capture = True
    meta._q35_temperature_only_for_capture = True
    copies = [meta.create_cuda_graph_metadata(bs) for bs in (1, 2, 4)]
    runner = types.SimpleNamespace(
        graph_metadata={("t", i): {"spec_metadata": c} for i, c in enumerate(copies)}
    )
    assert CUDAGraphRunner.clear_capture_only_spec_state(runner) == 3
    for c in copies:
        assert not c._force_non_greedy_for_capture
        assert not c._q35_temperature_only_for_capture
        # A serving scan on a reseated copy honors the client's params again.
        assert _scan(c, [_request(temperature=1.0, top_p=0.9)]) == [(1.0, DISABLE_TOPK_VAL, 0.9)]
        assert c.q35_temperature_only is False


def test_legacy_three_field_keys_default_to_the_stock_variant():
    assert KeyType(*(4, 6, False)).q35_temperature_only is False
    assert KeyType(4, 6, False) != KeyType(4, 6, False, q35_temperature_only=True)


def test_enable_gate(monkeypatch):
    mtp = types.SimpleNamespace(spec_dec_mode=SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL)
    eagle3 = types.SimpleNamespace(spec_dec_mode=SpeculativeDecodingMode.EAGLE3_ONE_MODEL)
    tp8 = types.SimpleNamespace(tp_size=8, enable_attention_dp=False)
    adp = types.SimpleNamespace(tp_size=8, enable_attention_dp=True)
    tp1 = types.SimpleNamespace(tp_size=1, enable_attention_dp=False)
    monkeypatch.delenv(Q35_TEMPERATURE_GRAPHS_ENV, raising=False)
    monkeypatch.setenv("TRTLLM_Q35_SHARDED_TARGET_SAMPLING", "1")
    assert not q35_temperature_graphs_enabled(mtp, tp8)  # default off
    monkeypatch.setenv(Q35_TEMPERATURE_GRAPHS_ENV, "1")
    assert q35_temperature_graphs_enabled(mtp, tp8)
    assert not q35_temperature_graphs_enabled(mtp, adp)
    assert not q35_temperature_graphs_enabled(mtp, tp1)
    assert not q35_temperature_graphs_enabled(eagle3, tp8)
    assert not q35_temperature_graphs_enabled(None, tp8)
    monkeypatch.setenv("TRTLLM_Q35_SHARDED_TARGET_SAMPLING", "0")
    assert not q35_temperature_graphs_enabled(mtp, tp8)
