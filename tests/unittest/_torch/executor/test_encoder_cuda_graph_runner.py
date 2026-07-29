# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    EncoderCUDAGraphRunner,
    EncoderCUDAGraphRunnerConfig,
)
from tensorrt_llm._torch.pyexecutor.model_engine import _build_encoder_decoder_cuda_graph_keys


def _dynamic_layout_runner(
    capture_keys: list[tuple[int, int, int]] | None = None,
    enable_padding: bool = False,
) -> EncoderCUDAGraphRunner:
    return EncoderCUDAGraphRunner(
        EncoderCUDAGraphRunnerConfig(
            use_cuda_graph=False,
            cuda_graph_padding_enabled=enable_padding,
            cuda_graph_batch_sizes=[1, 2, 4, 8],
            cuda_graph_num_tokens=[],
            cuda_graph_seq_lens=list(range(64, 513, 64)),
            max_cuda_graph_batch_size=8,
            max_cuda_graph_num_tokens=4096,
            max_num_tokens=4096,
            max_seq_len=512,
            cuda_graph_mem_pool=None,
            encoder_decoder_capture_keys=capture_keys or [],
        )
    )


def test_encoder_decoder_capture_keys_select_layout_mode():
    encoder_decoder_runner = _dynamic_layout_runner()
    encoder_only_runner = EncoderCUDAGraphRunner(
        EncoderCUDAGraphRunnerConfig(
            use_cuda_graph=False,
            cuda_graph_padding_enabled=False,
            cuda_graph_batch_sizes=[1],
            cuda_graph_num_tokens=[8],
            cuda_graph_seq_lens=[8],
            max_cuda_graph_batch_size=1,
            max_cuda_graph_num_tokens=8,
            max_num_tokens=8,
            max_seq_len=8,
            cuda_graph_mem_pool=None,
        )
    )

    assert encoder_decoder_runner.is_encoder_decoder
    assert not encoder_only_runner.is_encoder_decoder


def test_build_encoder_decoder_cuda_graph_keys():
    keys = _build_encoder_decoder_cuda_graph_keys(
        batch_sizes=[1, 2],
        num_tokens=[96, 576, 1056],
        seq_lens=[512],
    )

    assert keys == [
        (1, 96, 512),
        (2, 96, 512),
        (2, 576, 512),
    ]


def test_bart_encoder_graph_config_builds_feasible_key_grid():
    num_tokens = list(range(96, 4801, 96))
    keys = _build_encoder_decoder_cuda_graph_keys(
        batch_sizes=[1, 2, 4, 8],
        num_tokens=num_tokens,
        seq_lens=[512, 1024],
    )

    assert len(keys) == 201
    assert {total_tokens for batch_size, total_tokens, _ in keys if batch_size == 8} == set(
        num_tokens
    )


def test_encoder_graph_builds_reachable_startup_warmup_layouts():
    capture_keys = _build_encoder_decoder_cuda_graph_keys(
        batch_sizes=[1, 2],
        num_tokens=[96, 320],
        seq_lens=[256, 512],
    )
    runner = _dynamic_layout_runner(
        capture_keys=capture_keys,
        enable_padding=True,
    )

    for key in capture_keys:
        sequence_lengths = runner.get_capture_warmup_sequence_lengths(key)
        if sequence_lengths is None:
            continue

        selected_key, _, is_valid = runner.get_graph_key({"seq_lens": sequence_lengths})
        assert is_valid
        assert selected_key == key
        assert len(sequence_lengths) == key[0]
        assert sum(sequence_lengths) == key[1]

    assert runner.get_capture_warmup_sequence_lengths((1, 96, 256)) == [96]
    assert runner.get_capture_warmup_sequence_lengths((1, 96, 512)) is None
    assert runner.get_capture_warmup_sequence_lengths((2, 320, 512)) == [257, 63]


def test_encoder_graph_key_reuses_total_tokens_and_max_bucket():
    runner = _dynamic_layout_runner()

    key, is_padding_performed, is_valid = runner.get_graph_key(
        {
            "input_ids": list(range(580)),
            "seq_lens": [260, 320],
        }
    )
    other_layout_key, _, _ = runner.get_graph_key(
        {
            "input_ids": list(range(580)),
            "seq_lens": [284, 296],
        }
    )

    assert key == (2, 580, 320)
    assert other_layout_key == key
    assert not is_padding_performed
    assert is_valid


def test_encoder_graph_key_distinguishes_max_buckets():
    runner = _dynamic_layout_runner()

    key, _, _ = runner.get_graph_key(
        {
            "input_ids": [0] * 1400,
            "seq_lens": [332, 356, 356, 356],
        }
    )
    larger_bucket_key, _, _ = runner.get_graph_key(
        {
            "input_ids": [0] * 1400,
            "seq_lens": [260, 260, 440, 440],
        }
    )

    assert key == (4, 1400, 384)
    assert larger_bucket_key == (4, 1400, 448)


def test_encoder_graph_key_pads_tokens_and_max_sequence_length():
    runner = _dynamic_layout_runner(
        capture_keys=[
            (2, 640, 320),
            (2, 640, 384),
            (2, 704, 384),
        ],
        enable_padding=True,
    )

    key, is_padding_performed, is_valid = runner.get_graph_key(
        {
            "input_ids": [0] * 556,
            "seq_lens": [260, 296],
        }
    )

    assert key == (2, 640, 320)
    assert is_padding_performed
    assert is_valid


def test_encoder_graph_pad_batch_selects_compatible_capture_key():
    runner = _dynamic_layout_runner(
        capture_keys=[
            (4, 350, 192),
            (4, 384, 192),
            (8, 768, 192),
        ],
        enable_padding=True,
    )
    runner.enabled = True
    inputs = {
        "input_ids": [0] * 350,
        "seq_lens": [100, 120, 130],
    }

    with runner.pad_batch(inputs, batch_size=3) as padded_inputs:
        assert padded_inputs["seq_lens"] == [100, 120, 130, 1]
        assert padded_inputs["input_ids"] is inputs["input_ids"]
        key, is_padding_performed, is_valid = runner.get_graph_key(padded_inputs)

    assert key == (4, 384, 192)
    assert is_padding_performed
    assert is_valid


def test_encoder_graph_padding_rejects_incompatible_capture_keys():
    runner = _dynamic_layout_runner(
        capture_keys=[
            (4, 320, 128),
            (8, 512, 128),
        ],
        enable_padding=True,
    )
    runner.enabled = True
    inputs = {
        "input_ids": [0] * 350,
        "seq_lens": [100, 120, 130],
    }

    with runner.pad_batch(inputs, batch_size=3) as padded_inputs:
        assert padded_inputs is inputs
        key, is_padding_performed, is_valid = runner.get_graph_key(padded_inputs)

    assert key == (3, 0, 0)
    assert not is_padding_performed
    assert not is_valid


def test_encoder_graph_key_rejects_oversized_inputs():
    runner = _dynamic_layout_runner()

    _, _, is_valid = runner.get_graph_key(
        {
            "input_ids": [0] * 4097,
            "seq_lens": [4097],
        }
    )

    assert not is_valid


def test_encoder_graph_only_captures_during_warmup():
    key = (1, 8, 64)
    runner = _dynamic_layout_runner(capture_keys=[key])

    assert not runner.needs_capture(key)
    with runner.allow_capture():
        assert runner.needs_capture(key)
    assert not runner.needs_capture(key)


def test_encoder_graph_reuses_same_key_for_different_sequence_layouts():
    runner = _dynamic_layout_runner()
    runner.enabled = True
    graph_metadata = object.__new__(TrtllmAttentionMetadata)
    key = (2, 580, 320)
    runner.graph_metadata[key] = {
        "attn_metadata": graph_metadata,
    }

    matched_metadata, matched_key = runner.maybe_get_cuda_graph(
        {
            "input_ids": [0] * 580,
            "seq_lens": [260, 320],
        },
        graph_metadata,
    )
    reused_metadata, reused_key = runner.maybe_get_cuda_graph(
        {
            "input_ids": [0] * 580,
            "seq_lens": [284, 296],
        },
        graph_metadata,
    )

    assert matched_metadata is graph_metadata
    assert matched_key == key
    assert reused_metadata is graph_metadata
    assert reused_key == key


def test_encoder_graph_replay_uses_plain_graph_mapping(monkeypatch):
    runner = _dynamic_layout_runner()
    key = (1, 8, 64)
    attn_metadata = object()
    expected_output = object()
    replay_calls = []
    recorded_streams = []
    current_stream = object()

    runner.graphs[key] = SimpleNamespace(replay=lambda: replay_calls.append(key))
    runner.graph_metadata[key] = {"attn_metadata": attn_metadata}
    runner.graph_outputs[key] = expected_output
    runner._capture_h2d_copy = True
    monkeypatch.setattr(runner, "retire_staging", lambda: None)
    monkeypatch.setattr(runner, "_stage_inputs", lambda _key, _inputs: None)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda: SimpleNamespace(record=lambda stream: recorded_streams.append(stream)),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: current_stream)

    output = runner.replay(key, {"attn_metadata": attn_metadata})

    assert output is expected_output
    assert replay_calls == [key]
    assert recorded_streams == [current_stream]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_encoder_graph_capture_stages_warmup_and_replays_new_layout():
    runner = _dynamic_layout_runner()
    runner.enabled = True
    runner._create_shared_static_tensors()

    key = (2, 8, 64)
    seq_lens_host = runner.shared_static_tensors_cpu["seq_lens"][:2]
    seq_lens_host.copy_(torch.tensor([2, 3], dtype=torch.int32))
    attn_metadata = SimpleNamespace(
        _seq_lens=seq_lens_host,
        _seq_lens_cuda=torch.ones(2, device="cuda", dtype=torch.int32),
    )
    inputs = {
        "input_ids": [10, 11, 12, 13, 14],
        "position_ids": [0, 1, 0, 1, 2],
        "seq_lens": [2, 3],
        "attn_metadata": attn_metadata,
    }
    warmup_seq_lens = []

    def forward_fn(capture_inputs):
        if not torch.cuda.is_current_stream_capturing():
            warmup_seq_lens.append(capture_inputs["attn_metadata"]._seq_lens_cuda.cpu().tolist())
        return capture_inputs["input_ids"] + capture_inputs["attn_metadata"]._seq_lens_cuda[0]

    runner.capture(key, forward_fn, inputs)

    assert warmup_seq_lens == [[2, 3]]

    first_output = runner.replay(key, inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        first_output,
        torch.tensor([12, 13, 14, 15, 16, 2, 2, 2], device="cuda", dtype=torch.int32),
    )

    seq_lens_host.copy_(torch.tensor([1, 4], dtype=torch.int32))
    reused_inputs = {
        **inputs,
        "seq_lens": [1, 4],
    }
    reused_output = runner.replay(key, reused_inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        reused_output,
        torch.tensor([11, 12, 13, 14, 15, 1, 1, 1], device="cuda", dtype=torch.int32),
    )
