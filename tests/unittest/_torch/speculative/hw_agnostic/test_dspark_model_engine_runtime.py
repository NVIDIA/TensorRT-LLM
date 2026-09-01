# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused CPU tests for DSpark model-engine ragged runtime state."""

from types import SimpleNamespace

import pytest
import torch

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
    return SimpleNamespace(
        cuda_graph_runner=runner,
        spec_config=SimpleNamespace(verify_len_tiers=[1, 3, 5], max_draft_len=5),
        _dspark_last_padded_bs=None,
        ragged_verify_token_buckets=lambda _rows: list(buckets),
    )


@pytest.mark.parametrize(
    ("verifier_budget", "scheduled_window", "high_rows"),
    [(80, 4, 0), (81, 5, 1), (88, 5, 8)],
)
def test_zero_real_exact_fit_uses_one_scheduled_dummy(verifier_budget, scheduled_window, high_rows):
    runner = _runner()
    engine = _engine(runner, [48, 64, 80, 81, 88, 96])
    dummy = SimpleNamespace(is_attention_dp_dummy=True, py_verify_len=None)

    bucket = PyTorchModelEngine.fit_ragged_verify_lens(
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
        PyTorchModelEngine.fit_ragged_verify_lens(
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

    bucket = PyTorchModelEngine.fit_ragged_verify_lens(
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

    bucket = PyTorchModelEngine.fit_ragged_verify_lens(
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
        batch_size=8,
        _cuda_graph_batch_sizes=[1, 4, 8],
    )
    sync_calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))

    PyTorchModelEngine._warmup_dspark_ragged_compressor_metadata(engine)

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
        batch_size=4,
        _cuda_graph_batch_sizes=[1, 4],
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    with pytest.raises(RuntimeError, match="compile boom"):
        PyTorchModelEngine._warmup_dspark_ragged_compressor_metadata(engine)

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

    PyTorchModelEngine._warmup_dspark_ragged_compressor_metadata(engine)


def test_ragged_compressor_warmup_rejects_missing_metadata_contract():
    engine = SimpleNamespace(
        _dspark_confidence_enabled=True,
        _dspark_trims_submitted_tokens=True,
        attn_metadata=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="missing .*max_draft_tokens"):
        PyTorchModelEngine._warmup_dspark_ragged_compressor_metadata(engine)
