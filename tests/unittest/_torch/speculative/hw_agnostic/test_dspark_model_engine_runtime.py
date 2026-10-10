# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused CPU tests for DSpark decoder-runner ragged runtime state."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine


def _runner(*, secondary=True):
    return SimpleNamespace(
        enabled=True,
        agreed_ragged_bucket=None,
        ragged_pad_verify_len=0,
        ragged_zero_real_high_rows=0,
        supported_batch_sizes=[4, 16],
        secondary_padding_dummy_requests={5: object()} if secondary else {},
        _round_up_batch_size=lambda rows: 4 if rows <= 4 else 16,
        will_pad_to=lambda *_args: True,
    )


def _engine(runner, buckets):
    engine = object.__new__(DecoderRunner)
    engine.cuda_graph_runner = runner
    engine._config = SimpleNamespace(
        spec_config=SimpleNamespace(max_draft_len=5),
        max_batch_size=8,
        cuda_graph_batch_sizes=[1, 4, 8],
        use_mrope=False,
    )
    engine._dspark_confidence_enabled = True
    engine._dspark_trims_submitted_tokens = True
    engine._dspark_device_windows = False
    engine._dspark_last_padded_bs = None
    engine.ragged_verify_token_buckets = lambda _rows: list(buckets)
    return engine


def test_pinned_host_ping_pongs_without_private_cuda_events() -> None:
    engine = SimpleNamespace(
        _pinned_host_cache={},
        _pinned_host_active={},
    )

    first = DecoderRunner._pinned_host(engine, "rows", [1, 2], torch.long)
    second = DecoderRunner._pinned_host(engine, "rows", [3, 4], torch.long)
    third = DecoderRunner._pinned_host(engine, "rows", [5, 6], torch.long)

    assert first.data_ptr() == third.data_ptr()
    assert first.data_ptr() != second.data_ptr()
    assert third.tolist() == [5, 6]


@pytest.mark.parametrize(
    ("verifier_budget", "scheduled_window", "high_rows"),
    [(80, 4, 0), (81, 5, 1), (88, 5, 8)],
)
def test_zero_real_exact_fit_uses_one_scheduled_dummy(verifier_budget, scheduled_window, high_rows):
    runner = _runner()
    engine = _engine(runner, [48, 64, 80, 81, 88, 96])
    dummy = SimpleNamespace(is_attention_dp_dummy=True, py_verify_len=None)

    bucket = DecoderRunner.fit_ragged_verify_lens(
        engine,
        [dummy],
        [scheduled_window],
        peer_stats=[[16, 88, 1], [0, 0, 1]],
        exact_shape=(16, verifier_budget, 5),
        exact_zero_real=True,
    )

    assert bucket == verifier_budget
    assert dummy.py_verify_len == scheduled_window
    assert engine._dspark_last_num_real == 0
    assert runner.ragged_pad_verify_len == 4
    assert runner.ragged_zero_real_high_rows == high_rows


def test_zero_real_exact_fit_requires_the_secondary_dummy():
    runner = _runner(secondary=False)
    engine = _engine(runner, [88])
    dummy = SimpleNamespace(is_attention_dp_dummy=True, py_verify_len=None)

    with pytest.raises(RuntimeError, match="disappeared after"):
        DecoderRunner.fit_ragged_verify_lens(
            engine,
            [dummy],
            [5],
            peer_stats=[[16, 88, 1], [0, 0, 1]],
            exact_shape=(16, 88, 5),
            exact_zero_real=True,
        )


def test_exact_fit_publishes_the_measured_bucket_and_pad_window():
    runner = _runner()
    engine = _engine(runner, [8, 24])
    requests = [SimpleNamespace(py_verify_len=None) for _ in range(2)]

    bucket = DecoderRunner.fit_ragged_verify_lens(
        engine,
        requests,
        [2, 2],
        peer_stats=[[2, 6, 1]],
        exact_shape=(4, 8, 1),
    )

    assert bucket == 8
    assert [request.py_verify_len for request in requests] == [2, 2]
    assert runner.agreed_ragged_bucket == 8
    assert runner.ragged_pad_verify_len == 0


def test_full_k_bucket_preserves_the_native_static_graph():
    runner = _runner()
    engine = _engine(runner, [24])
    requests = [SimpleNamespace(py_verify_len=None) for _ in range(2)]

    bucket = DecoderRunner.fit_ragged_verify_lens(
        engine,
        requests,
        [5, 5],
        peer_stats=[[2, 12, 1]],
        exact_shape=(4, 24, 6),
    )

    assert bucket is None
    assert runner.agreed_ragged_bucket is None
    assert all(request.py_verify_len is None for request in requests)


def _warmup_metadata(position_helper, mask_helper):
    ratios = [1, 4]
    return SimpleNamespace(
        _compress_ratios_sorted=ratios,
        max_draft_tokens=5,
        past_kv_lens_cuda={ratio: torch.empty(8, dtype=torch.int32) for ratio in ratios},
        cu_new_comp_kv_cuda={ratio: torch.empty(9, dtype=torch.int32) for ratio in ratios},
        new_comp_kv_lens_cuda={ratio: torch.empty(8, dtype=torch.int32) for ratio in ratios},
        compressed_position_ids_cuda={
            ratio: torch.empty(48, dtype=torch.int32) for ratio in ratios
        },
        compressed_mask_cuda={ratio: torch.empty(48, dtype=torch.bool) for ratio in ratios},
        _compute_gen_compressed_position_ids=position_helper,
        _compute_compressed_mask=mask_helper,
    )


def test_ragged_compressor_warmup_primes_and_cleans_intermediate_shape(monkeypatch):
    position_calls = []
    mask_calls = []

    def record_positions(*args):
        position_calls.append((torch.is_inference_mode_enabled(), args[3:]))

    def record_mask(*args):
        mask_calls.append((torch.is_inference_mode_enabled(), args[3:]))

    metadata = _warmup_metadata(record_positions, record_mask)
    engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        _dspark_trims_submitted_tokens=True,
        attn_metadata=metadata,
        _config=SimpleNamespace(max_batch_size=8, cuda_graph_batch_sizes=[1, 4, 8]),
    )
    sync_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))

    DecoderRunner._warmup_dspark_ragged_compressor_metadata(engine)

    assert position_calls == [(True, (0, 7, 6, [1, 4], {1: 0, 4: 0}))]
    assert mask_calls == [(True, (7, {1: 42, 4: 14}, [1, 4]))]
    for ratio, token_count in ((1, 42), (4, 14)):
        assert not metadata.past_kv_lens_cuda[ratio][:7].any()
        assert not metadata.new_comp_kv_lens_cuda[ratio][:7].any()
        assert not metadata.cu_new_comp_kv_cuda[ratio][:8].any()
        assert not metadata.compressed_position_ids_cuda[ratio][:token_count].any()
        assert not metadata.compressed_mask_cuda[ratio][:token_count].any()
    assert sync_calls == [True, True]


def test_ragged_compressor_warmup_cleans_after_helper_failure(monkeypatch):
    def fail_positions(*args):
        args[2][1][:18].fill_(7)
        raise RuntimeError("compile boom")

    metadata = _warmup_metadata(fail_positions, lambda *_args: None)
    engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        _dspark_trims_submitted_tokens=True,
        attn_metadata=metadata,
        _config=SimpleNamespace(max_batch_size=4, cuda_graph_batch_sizes=[1, 4]),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="compile boom"):
        DecoderRunner._warmup_dspark_ragged_compressor_metadata(engine)

    assert not metadata.past_kv_lens_cuda[1][:3].any()
    assert not metadata.new_comp_kv_lens_cuda[1][:3].any()
    assert not metadata.cu_new_comp_kv_cuda[1][:4].any()
    assert not metadata.compressed_position_ids_cuda[1][:18].any()
    assert not metadata.compressed_mask_cuda[1][:18].any()


def test_ragged_compressor_warmup_requires_trimmed_tokens():
    engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        _dspark_trims_submitted_tokens=False,
    )

    DecoderRunner._warmup_dspark_ragged_compressor_metadata(engine)


def test_ragged_compressor_warmup_rejects_missing_metadata_contract():
    engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        _dspark_trims_submitted_tokens=True,
        attn_metadata=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="missing .*max_draft_tokens"):
        DecoderRunner._warmup_dspark_ragged_compressor_metadata(engine)


@pytest.mark.parametrize("runtime_draft_len", [0, 2, 5])
def test_publish_gen_token_layout_uses_call_draft_length(runtime_draft_len: int) -> None:
    engine = _engine(_runner(), [])
    engine.get_runtime_tokens_per_gen_step = lambda draft_len: draft_len + 1
    requests = [
        SimpleNamespace(py_verify_len=min(1, runtime_draft_len)),
        SimpleNamespace(py_verify_len=min(3, runtime_draft_len)),
    ]
    metadata = SimpleNamespace(
        runtime_tokens_per_gen_step=0,
        ragged_verify_lens=None,
        device_windows_mode=False,
    )
    assert not hasattr(engine, "runtime_draft_len")

    engine._publish_gen_token_layout(metadata, requests, runtime_draft_len=runtime_draft_len)

    assert metadata.runtime_tokens_per_gen_step == runtime_draft_len + 1
    assert metadata.ragged_verify_lens == [request.py_verify_len + 1 for request in requests]
    assert not metadata.device_windows_mode
    assert not hasattr(engine, "runtime_draft_len")


def test_attach_ragged_layout_refreshes_persistent_buffers_in_place() -> None:
    runner = _engine(_runner(), [])
    runner._pinned_host_cache = {}
    runner._pinned_host_active = {}
    runner.ragged_verify_lens_cuda = torch.empty(9, dtype=torch.int32)
    runner.ragged_qo_indptr_cuda = torch.empty(10, dtype=torch.int32)
    attention = SimpleNamespace(ragged_verify_lens=None)
    speculation = SimpleNamespace()
    requests = [SimpleNamespace(py_verify_len=1), SimpleNamespace(py_verify_len=3)]

    runner._attach_ragged_verify_layout(speculation, attention, requests)
    first_lens_ptr = speculation.verify_lens.data_ptr()
    first_prefix_ptr = speculation.qo_indptr.data_ptr()

    assert speculation.verify_lens.tolist() == [2, 4]
    assert speculation.qo_indptr.tolist() == [0, 2, 6]
    assert speculation.total_verify_tokens == 6
    assert attention.ragged_verify_lens == [2, 4]

    requests[0].py_verify_len = 0
    requests[1].py_verify_len = 2
    runner._attach_ragged_verify_layout(speculation, attention, requests)

    assert speculation.verify_lens.tolist() == [1, 3]
    assert speculation.qo_indptr.tolist() == [0, 1, 4]
    assert speculation.total_verify_tokens == 4
    assert attention.ragged_verify_lens == [1, 3]
    assert speculation.verify_lens.data_ptr() == first_lens_ptr
    assert speculation.qo_indptr.data_ptr() == first_prefix_ptr


@pytest.mark.parametrize("trim_enabled", [False, True])
def test_missing_or_disabled_windows_clear_stale_ragged_views(trim_enabled: bool) -> None:
    runner = _engine(_runner(), [])
    runner._dspark_trims_submitted_tokens = trim_enabled
    attention = SimpleNamespace(ragged_verify_lens=[2, 4])
    speculation = SimpleNamespace(
        verify_lens=torch.tensor([2, 4]),
        qo_indptr=torch.tensor([0, 2, 6]),
        total_verify_tokens=6,
    )

    runner._attach_ragged_verify_layout(
        speculation,
        attention,
        [SimpleNamespace(py_verify_len=1), SimpleNamespace(py_verify_len=None)],
    )

    assert speculation.verify_lens is None
    assert speculation.qo_indptr is None
    assert speculation.total_verify_tokens is None
    assert attention.ragged_verify_lens is None


def test_model_engine_delegates_ragged_state_to_decoder_owner() -> None:
    runner = _engine(_runner(), [])
    runner._dspark_sps_cost_table = object()
    runner._dspark_exact_candidate_cells = ((4, 12),)
    runner._dspark_exact_identity_words = tuple(range(8))
    runner.fit_ragged_verify_lens = lambda *args, **kwargs: (args, kwargs)
    worker = object()
    runner._get_spec_worker = lambda: worker
    engine = object.__new__(PyTorchModelEngine)
    engine._cleanup_done = True
    engine._runner = runner

    assert engine._dspark_confidence_enabled
    assert engine._dspark_trims_submitted_tokens
    assert engine._dspark_sps_cost_table is runner._dspark_sps_cost_table
    assert engine._dspark_exact_candidate_cells == ((4, 12),)
    assert engine._dspark_exact_identity_words == tuple(range(8))
    assert engine._dspark_device_budget is None
    assert engine._get_spec_worker() is worker
    engine._dspark_device_budget = 7
    assert runner._dspark_device_budget == 7
    assert engine.fit_ragged_verify_lens([1], exact_shape=(4, 12, 3)) == (
        ([1],),
        {"exact_shape": (4, 12, 3)},
    )


def test_model_engine_non_decoder_keeps_confidence_inactive() -> None:
    engine = object.__new__(PyTorchModelEngine)
    engine._cleanup_done = True
    engine._runner = SimpleNamespace()

    assert not engine._dspark_confidence_enabled
    assert not engine._dspark_trims_submitted_tokens
    assert engine._dspark_sps_cost_table is None
    assert engine._dspark_exact_candidate_cells == ()
    assert engine._dspark_exact_identity_words == (0,) * 8
    assert engine._dspark_device_budget is None
    assert engine._get_spec_worker() is None
