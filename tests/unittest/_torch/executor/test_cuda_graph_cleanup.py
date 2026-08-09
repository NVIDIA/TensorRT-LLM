# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner,
    CUDAGraphRunnerConfig,
    EncoderCUDAGraphRunner,
    EncoderCUDAGraphRunnerConfig,
    KeyType,
)


class _ResetFailureGraph:
    def reset(self) -> None:
        raise RuntimeError("reset failed")


class _RecordingGraph:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


class _SpecDecMode:
    def needs_kv_cache_recompute(self) -> bool:
        return False


class _SpecConfig:
    spec_dec_mode = _SpecDecMode()


class _AttentionMetadata:
    def __init__(self) -> None:
        self.kv_lens_cuda = torch.tensor([3])
        self._seq_lens = torch.tensor([1])
        self._seq_lens_cuda = torch.tensor([1])
        self.num_seqs = 1
        self.update_count = 0

    def on_update_kv_lens(self) -> None:
        self.update_count += 1


def _make_decoder_runner() -> CUDAGraphRunner:
    runner = CUDAGraphRunner(
        CUDAGraphRunnerConfig(
            use_cuda_graph=False,
            cuda_graph_padding_enabled=False,
            cuda_graph_batch_sizes=[],
            max_cuda_graph_batch_size=0,
            max_beam_width=1,
            max_num_tokens=1,
            spec_config=None,
            cuda_graph_mem_pool=None,
            use_mrope=False,
            original_max_draft_len=0,
            original_max_total_draft_tokens=0,
            is_draft_model=False,
            enable_attention_dp=False,
            is_encoder_decoder=False,
            batch_size=0,
            mapping=None,
            dist=None,
            kv_cache_manager_key=None,
        )
    )
    runner.shared_static_tensors = {
        "input_ids": torch.empty(1),
        "position_ids": torch.empty((1, 1)),
    }
    return runner


def _make_encoder_runner() -> EncoderCUDAGraphRunner:
    return EncoderCUDAGraphRunner(
        EncoderCUDAGraphRunnerConfig(
            use_cuda_graph=False,
            cuda_graph_padding_enabled=False,
            cuda_graph_batch_sizes=[],
            cuda_graph_num_tokens=[],
            cuda_graph_seq_lens=[],
            max_cuda_graph_batch_size=0,
            max_cuda_graph_num_tokens=0,
            max_num_tokens=1,
            max_seq_len=1,
            cuda_graph_mem_pool=None,
        )
    )


def test_decoder_clear_continues_after_reset_failure() -> None:
    runner = _make_decoder_runner()
    remaining_graph = _RecordingGraph()
    runner.graphs = {"broken": _ResetFailureGraph(), "remaining": remaining_graph}
    runner.graph_outputs = {"output": object()}
    runner.graph_metadata = {"metadata": object()}
    runner.padding_dummy_requests = {"request": object()}

    with (
        patch("torch.cuda.empty_cache"),
        patch("tensorrt_llm._torch.utils.logger.warning") as warning,
    ):
        runner.clear()

    assert remaining_graph.reset_count == 1
    warning.assert_called_once()
    assert warning.call_args.args == (
        "Failed to reset CUDA graph during decoder cleanup: reset failed",
    )
    assert runner.graphs == {}
    assert runner.graph_outputs == {}
    assert runner.graph_metadata == {}
    assert runner.padding_dummy_requests == {}
    assert runner.memory_pool is None


def test_decoder_warmup_failure_restores_state_without_publishing_metadata() -> None:
    runner = _make_decoder_runner()
    runner.config.spec_config = _SpecConfig()
    metadata = _AttentionMetadata()

    def fail_after_mutating_state(inputs: dict[str, object]) -> None:
        inputs["attn_metadata"].kv_lens_cuda.add_(2)
        raise ValueError("warmup failed")

    with pytest.raises(ValueError, match="warmup failed"):
        runner.capture(
            (1, 0, False, False, False),
            fail_after_mutating_state,
            {"attn_metadata": metadata},
            enable_spec_decode=True,
        )

    torch.testing.assert_close(metadata.kv_lens_cuda, torch.tensor([3]))
    assert metadata.update_count == 1
    assert runner.graphs == {}
    assert runner.graph_outputs == {}
    assert runner.graph_metadata == {}


def test_decoder_warmup_publishes_metadata_for_capture() -> None:
    runner = _make_decoder_runner()
    runner.is_warmup_only = True
    metadata = _AttentionMetadata()
    spec_metadata = object()
    output = torch.tensor([1])
    key = KeyType(1, 0, False, False, False)

    result = runner.capture(
        key,
        lambda _inputs: output,
        {
            "attn_metadata": metadata,
            "spec_metadata": spec_metadata,
        },
    )

    assert result is output
    assert runner.graph_metadata[key] == {
        "attn_metadata": metadata,
        "spec_metadata": spec_metadata,
    }
    assert runner.graphs == {}
    assert runner.graph_outputs == {}


def test_decoder_capture_failure_resets_graph_without_publishing_metadata() -> None:
    runner = _make_decoder_runner()
    runner.WARMUP_STEPS = 0
    graph = _RecordingGraph()

    def fail_capture(_inputs: dict[str, object]) -> None:
        raise ValueError("capture failed")

    with (
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()),
        pytest.raises(ValueError, match="capture failed"),
    ):
        runner.capture(
            (1, 0, False, False, False),
            fail_capture,
            {"attn_metadata": object()},
        )

    assert graph.reset_count == 1
    assert runner.graphs == {}
    assert runner.graph_outputs == {}
    assert runner.graph_metadata == {}


def test_encoder_rejects_nested_output_without_orphaning_graph() -> None:
    runner = _make_encoder_runner()
    runner.WARMUP_STEPS = 0
    runner.shared_static_tensors = {
        "input_ids": torch.empty(1),
        "position_ids": torch.empty((1, 1)),
    }
    runner.shared_static_tensors_cpu = runner.shared_static_tensors
    runner._arange_max = torch.arange(1, dtype=torch.int32)
    runner._capture_h2d_copy = False
    graph = _RecordingGraph()
    nested_output = torch.nested.nested_tensor([torch.tensor([1.0]), torch.tensor([1.0, 2.0])])

    with (
        patch("torch.cuda.CUDAGraph", return_value=graph),
        patch("torch.cuda.graph", return_value=nullcontext()),
        pytest.raises(TypeError, match="nested tensor outputs"),
    ):
        runner.capture(
            (1, 1, 1),
            lambda _inputs: nested_output,
            {
                "attn_metadata": _AttentionMetadata(),
                "input_ids": [1],
                "seq_lens": [1],
            },
        )

    assert graph.reset_count == 1
    assert runner.graphs == {}
    assert runner.graph_outputs == {}
    assert runner.graph_metadata == {}


def test_encoder_warmup_publishes_metadata_for_capture() -> None:
    runner = _make_encoder_runner()
    runner.is_warmup_only = True
    runner.shared_static_tensors = {
        "input_ids": torch.empty(1),
        "position_ids": torch.empty((1, 1)),
    }
    runner.shared_static_tensors_cpu = runner.shared_static_tensors
    runner._arange_max = torch.arange(1, dtype=torch.int32)
    runner._capture_h2d_copy = False
    metadata = _AttentionMetadata()
    output = torch.tensor([1])
    key = (1, 1, 1)

    with patch("torch.cuda.current_stream"):
        result = runner.capture(
            key,
            lambda _inputs: output,
            {
                "attn_metadata": metadata,
                "input_ids": [1],
                "seq_lens": [1],
            },
        )

    assert result is output
    assert runner.graph_metadata[key] == {"attn_metadata": metadata}
    assert runner.graphs == {}
    assert runner.graph_outputs == {}
