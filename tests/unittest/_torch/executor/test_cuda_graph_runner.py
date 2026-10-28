# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from unittest.mock import MagicMock, Mock

import pytest
import torch

import tensorrt_llm._torch.pyexecutor.cuda_graph_runner as cuda_graph_runner_module
from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner,
    CUDAGraphRunnerConfig,
    EncoderCUDAGraphRunner,
    EncoderCUDAGraphRunnerConfig,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.llmapi.llm_args import BaseSparseAttentionConfig, DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def test_encoder_capture_discards_warmup_and_capture_outputs(monkeypatch) -> None:
    discarded = []

    @contextlib.contextmanager
    def discard(inputs):
        discarded.append(inputs)
        yield

    capture_inputs = {"attn_metadata": object()}
    graph = MagicMock()
    graph.pool.return_value = (1, 2)
    monkeypatch.setattr(cuda_graph_runner_module, "discard_nccl_window_tensor_outputs", discard)
    monkeypatch.setattr(cuda_graph_runner_module.torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(
        cuda_graph_runner_module,
        "nccl_window_graph_capture",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    runner = EncoderCUDAGraphRunner.__new__(EncoderCUDAGraphRunner)
    runner.feature_mode = False
    runner.is_warmup_only = False
    runner.memory_pool = (1, 2)
    runner.graph_metadata = {}
    runner.graphs = {}
    runner.graph_outputs = {}
    runner._prepare_token_capture = Mock(return_value=(capture_inputs, None))
    runner._get_capture_stream = Mock(return_value=object())
    runner._contains_nested_tensor = Mock(return_value=False)

    runner.capture((1, 1, 1), Mock(return_value=torch.tensor(1)), {})

    assert discarded == [capture_inputs] * (runner.WARMUP_STEPS + 1)


def _feature_encoder_runner(
    batch_sizes: list[int], fixed_seq_len: int = 1_500
) -> EncoderCUDAGraphRunner:
    """Build a feature-mode backend without enabling CUDA capture."""
    config = EncoderCUDAGraphRunnerConfig(
        use_cuda_graph=False,
        cuda_graph_padding_enabled=True,
        cuda_graph_batch_sizes=batch_sizes,
        cuda_graph_num_tokens=[],
        cuda_graph_seq_lens=[],
        max_cuda_graph_batch_size=max(batch_sizes),
        max_cuda_graph_num_tokens=max(batch_sizes) * fixed_seq_len,
        max_num_tokens=max(batch_sizes) * fixed_seq_len,
        max_seq_len=fixed_seq_len,
        is_encoder_decoder=True,
        use_fixed_sequence_slots=False,
        feature_shape=(480_000,),
        feature_dtype=torch.float32,
        fixed_seq_len=fixed_seq_len,
    )
    return EncoderCUDAGraphRunner(config)


def test_feature_encoder_capture_keys_are_reachable() -> None:
    fixed_seq_len = 1_500
    batch_sizes = [1, 2, 4, 8]
    runner = _feature_encoder_runner(batch_sizes, fixed_seq_len)

    assert runner._capture_sequence_lengths == {
        (batch_size, batch_size * fixed_seq_len, fixed_seq_len): [fixed_seq_len] * batch_size
        for batch_size in batch_sizes
    }
    assert runner.capture_keys == frozenset(runner._capture_sequence_lengths)


def test_feature_encoder_padding_rejects_excessive_extra_work() -> None:
    fixed_seq_len = 1_500
    runner = _feature_encoder_runner([1, 2, 4, 9], fixed_seq_len)
    runner.enabled = True

    cases = [
        (4, [fixed_seq_len] * 4),
        (8, [fixed_seq_len] * 9),
        (3, [fixed_seq_len] * 3),
        (5, [fixed_seq_len] * 5),
    ]
    for batch_size, expected_sequence_lengths in cases:
        inputs = {"seq_lens": [fixed_seq_len] * batch_size}
        with runner.pad_batch(inputs, batch_size) as padded:
            assert padded["seq_lens"] == expected_sequence_lengths
        assert padded["seq_lens"] == expected_sequence_lengths
        assert inputs["seq_lens"] == [fixed_seq_len] * batch_size


def test_token_encoder_padding_survives_context_exit_without_mutating_inputs() -> None:
    runner = EncoderCUDAGraphRunner.__new__(EncoderCUDAGraphRunner)
    runner.enabled = True
    runner.padding_enabled = True
    runner.is_encoder_decoder = False
    runner.feature_mode = False
    runner.capture_keys = frozenset()
    runner.supported_batch_sizes = [2]
    runner.max_supported_num_tokens = 4
    inputs = {"input_ids": [11, 12], "seq_lens": [2]}

    with runner.pad_batch(inputs, 1) as padded:
        assert padded["seq_lens"] == [2, 1]

    assert padded["seq_lens"] == [2, 1]
    assert inputs["seq_lens"] == [2]


def test_captured_feature_metadata_avoids_eager_metadata_build() -> None:
    fixed_seq_len = 1_500
    runner = _feature_encoder_runner([1, 2], fixed_seq_len)
    runner.enabled = True
    runner.retire_staging = Mock()
    metadata = object()
    key = (2, 2 * fixed_seq_len, fixed_seq_len)
    runner.graph_metadata[key] = {"attn_metadata": metadata}

    actual_metadata, actual_key = runner.captured_graph_metadata({"seq_lens": [fixed_seq_len] * 2})

    assert actual_metadata is metadata
    assert actual_key == key
    runner.retire_staging.assert_called_once_with()

    runner.retire_staging.reset_mock()
    assert runner.captured_graph_metadata({"seq_lens": [fixed_seq_len]}) == (None, None)
    runner.retire_staging.assert_not_called()


def test_encoder_decoder_fixed_slots_restore_source_order() -> None:
    runner = EncoderCUDAGraphRunner.__new__(EncoderCUDAGraphRunner)
    runner.is_encoder_decoder = True
    runner.use_fixed_sequence_slots = True
    runner.supported_batch_sizes = [2]
    runner.supported_seq_lens = [512]
    runner.max_supported_num_tokens = 1_024
    small_key = (2, 512, 512)
    compatible_key = (2, 1_024, 512)
    runner._capture_sequence_lengths = {
        small_key: [511, 1],
        compatible_key: [512, 512],
    }
    runner._capture_keys_by_batch_size = {2: [small_key, compatible_key]}
    runner._arange_max = torch.arange(1_024, dtype=torch.int32)

    assert runner._get_dynamic_capture_key([200, 300], allow_batch_padding=False) == compatible_key

    source_sequence_lengths = [1, 400]
    key = runner._get_dynamic_capture_key(source_sequence_lengths, allow_batch_padding=False)
    assert key == small_key
    assert runner._get_capture_sequence_offsets(key) == [0, 511, 512]

    input_ids = torch.arange(401, dtype=torch.int32)
    inputs = runner.prepare_encoder_decoder_inputs(
        {
            "input_ids": input_ids,
            "position_ids": input_ids,
            "seq_lens": source_sequence_lengths,
        },
        key,
        source_sequence_lengths,
    )
    assert inputs["seq_lens"] == [400, 1]
    assert inputs["_encoder_source_to_slot"] == [1, 0]

    static_tensors = {
        "input_ids": torch.empty(512, dtype=torch.int32),
        "position_ids": torch.empty((1, 512), dtype=torch.int32),
    }
    runner._stage_encoder_decoder_inputs(key, inputs, static_tensors)
    expected_staged_ids = torch.zeros(512, dtype=torch.int32)
    expected_staged_ids[:400] = input_ids[1:]
    expected_staged_ids[511] = input_ids[0]
    torch.testing.assert_close(static_tensors["input_ids"], expected_staged_ids)
    torch.testing.assert_close(static_tensors["position_ids"][0], expected_staged_ids)

    fixed_slot_output = torch.arange(512).unsqueeze(1)
    restored_output = runner.restore_encoder_decoder_output(key, fixed_slot_output, inputs)
    expected_output = torch.cat((fixed_slot_output[511:512], fixed_slot_output[:400]))
    torch.testing.assert_close(restored_output, expected_output)


def _decoder_runner(
    *,
    short_long_seq_len_threshold: int | None = None,
    sparse_attention_config: BaseSparseAttentionConfig | None = None,
    private_metadata_buffers: bool = False,
) -> CUDAGraphRunner:
    """A generation-graph runner with capture disabled, so it allocates no device buffers."""
    config = CUDAGraphRunnerConfig(
        use_cuda_graph=False,
        cuda_graph_padding_enabled=False,
        cuda_graph_batch_sizes=[1, 2],
        max_cuda_graph_batch_size=2,
        batch_size=2,
        max_beam_width=1,
        max_num_tokens=2,
        use_mrope=False,
        spec_config=None,
        cuda_graph_mem_pool=None,
        enable_attention_dp=False,
        original_max_total_draft_tokens=0,
        is_encoder_decoder=False,
        mapping=Mapping(),
        dist=None,
        kv_cache_manager_key=ResourceManagerType.KV_CACHE_MANAGER,
        sparse_attention_config=sparse_attention_config,
        short_long_seq_len_threshold=short_long_seq_len_threshold,
        private_metadata_buffers=private_metadata_buffers,
    )
    return CUDAGraphRunner(config)


def _dsa_config(skip_indexer_for_short_seqs: bool) -> DeepSeekSparseAttentionConfig:
    return DeepSeekSparseAttentionConfig(
        index_topk=512, skip_indexer_for_short_seqs=skip_indexer_for_short_seqs
    )


def test_short_long_threshold_from_the_runner_config_wins() -> None:
    runner = _decoder_runner(
        short_long_seq_len_threshold=2_048, sparse_attention_config=_dsa_config(True)
    )

    assert runner.short_long_seq_len_threshold == 2_048


def test_short_long_threshold_is_derived_from_a_sparse_config_that_asks_for_the_split() -> None:
    runner = _decoder_runner(sparse_attention_config=_dsa_config(True))

    assert runner.short_long_seq_len_threshold == 512


@pytest.mark.parametrize(
    "sparse_attention_config",
    [None, _dsa_config(False)],
    ids=["dense", "dsa_without_split"],
)
def test_short_long_threshold_is_none_for_a_single_graph_family(sparse_attention_config) -> None:
    runner = _decoder_runner(sparse_attention_config=sparse_attention_config)

    assert runner.short_long_seq_len_threshold is None


def test_graph_metadata_uses_the_process_buffer_pool_by_default() -> None:
    assert _decoder_runner().cuda_graph_meta_buffers is get_memory_buffers()


def test_private_metadata_buffers_give_each_runner_its_own_pool() -> None:
    first = _decoder_runner(private_metadata_buffers=True)
    second = _decoder_runner(private_metadata_buffers=True)

    assert first.cuda_graph_meta_buffers is not get_memory_buffers()
    assert first.cuda_graph_meta_buffers is not second.cuda_graph_meta_buffers


def _generation_batch(*num_tokens: int) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.generation_requests = [
        LlmRequest(
            request_id=request_id,
            max_new_tokens=4,
            input_tokens=list(range(tokens)),
            sampling_config=SamplingConfig(),
            seq_slot=request_id,
            is_streaming=False,
        )
        for request_id, tokens in enumerate(num_tokens)
    ]
    return batch


@pytest.mark.parametrize(
    ("num_tokens", "short"),
    [((100,), True), ((100, 2_048), True), ((100, 2_049), False), ((4_000,), False)],
)
def test_seq_len_mode_splits_at_the_configured_threshold(num_tokens, short) -> None:
    runner = _decoder_runner(short_long_seq_len_threshold=2_048)

    assert runner._get_seq_len_mode(_generation_batch(*num_tokens)) is short


def test_seq_len_mode_is_long_without_a_threshold() -> None:
    runner = _decoder_runner()

    assert runner._get_seq_len_mode(_generation_batch(1)) is False
