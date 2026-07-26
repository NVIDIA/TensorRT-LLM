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


def _dynamic_layout_runner(
    max_cuda_graphs: int = 0, capture_keys: list[tuple[int, int, int]] | None = None
) -> EncoderCUDAGraphRunner:
    return EncoderCUDAGraphRunner(
        EncoderCUDAGraphRunnerConfig(
            use_cuda_graph=False,
            cuda_graph_padding_enabled=False,
            cuda_graph_batch_sizes=[1, 2, 4, 8],
            cuda_graph_num_tokens=[],
            cuda_graph_seq_lens=list(range(64, 513, 64)),
            max_cuda_graph_batch_size=8,
            max_cuda_graph_num_tokens=4096,
            max_num_tokens=4096,
            max_seq_len=512,
            cuda_graph_mem_pool=None,
            dynamic_sequence_layout=True,
            allow_runtime_capture=True,
            max_cuda_graphs=max_cuda_graphs,
            capture_keys=capture_keys or [],
        )
    )


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


def test_bart_microbatch_key_set_fits_graph_cache():
    runner = _dynamic_layout_runner(max_cuda_graphs=64)
    sequence_length_cycle = list(range(260, 441, 12))
    keys = set()

    for batch_size in (1, 2, 4, 8):
        for start in range(len(sequence_length_cycle)):
            sequence_lengths = [
                sequence_length_cycle[(start + offset) % len(sequence_length_cycle)]
                for offset in range(batch_size)
            ]
            key, _, is_valid = runner.get_graph_key(
                {
                    "input_ids": [0] * sum(sequence_lengths),
                    "seq_lens": sequence_lengths,
                }
            )
            assert is_valid
            keys.add(key)

    assert len(keys) == 59
    assert len(keys) <= runner.max_cuda_graphs


def test_encoder_graph_key_rejects_oversized_inputs():
    runner = _dynamic_layout_runner()

    _, _, is_valid = runner.get_graph_key(
        {
            "input_ids": [0] * 4097,
            "seq_lens": [4097],
        }
    )

    assert not is_valid


def test_encoder_graph_capture_allowlist_must_fit_cache():
    capture_keys = [(1, num_tokens, 64) for num_tokens in range(1, 66)]

    with pytest.raises(ValueError, match="capture key count"):
        _dynamic_layout_runner(max_cuda_graphs=64, capture_keys=capture_keys)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_encoder_graph_capture_stages_warmup_and_replays_new_layout():
    runner = _dynamic_layout_runner()
    runner.enabled = True
    runner._create_shared_static_tensors()

    key = (2, 5, 64)
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
        torch.tensor([12, 13, 14, 15, 16], device="cuda", dtype=torch.int32),
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
        torch.tensor([11, 12, 13, 14, 15], device="cuda", dtype=torch.int32),
    )


def test_encoder_graph_lru_evicts_oldest_graph():
    class _Graph:
        def __init__(self):
            self.was_reset = False

        def reset(self):
            self.was_reset = True

    runner = _dynamic_layout_runner(max_cuda_graphs=1)
    key = (1, 8, 64)
    graph = _Graph()
    runner.graphs[key] = graph
    runner.graph_outputs[key] = object()
    runner.graph_metadata[key] = object()

    runner._evict_graph_if_needed()

    assert graph.was_reset
    assert not runner.graphs
    assert not runner.graph_outputs
    assert not runner.graph_metadata
