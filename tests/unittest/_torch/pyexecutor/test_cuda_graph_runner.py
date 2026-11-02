# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from _torch.pyexecutor.engine_test_utils import _make_request_stub

import tensorrt_llm._torch.pyexecutor.cuda_graph_runner as cuda_graph_runner_module
from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner,
    CUDAGraphRunnerConfig,
    EncoderCUDAGraphRunner,
    EncoderCUDAGraphRunnerConfig,
    KeyType,
    SampleType,
    _restore_spec_decode_capture_state,
    _save_spec_decode_capture_state,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.llmapi.llm_args import (
    BaseSparseAttentionConfig,
    DeepSeekSparseAttentionConfig,
    SeqLenAwareSparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.cpu_only


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


class GraphOwnerTeardownTestCase(unittest.TestCase):
    def test_cuda_graph_runner_destructor_does_not_release_window_owner(self) -> None:
        runner = Mock()

        CUDAGraphRunner.__del__(runner)

        runner.clear.assert_called_once_with(release_nccl_window_owners=False)


class GraphKeyTestCase(unittest.TestCase):
    def test_graph_key_forwards_promoted_context_ids(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
        runner._get_seq_len_mode.return_value = True
        runner._ragged_verify_bucket.return_value = None
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]
        promoted_ids = frozenset({request.py_request_id})

        key = CUDAGraphRunner.get_graph_key(
            runner,
            batch,
            new_tensors_device=None,
            promoted_context_request_ids=promoted_ids,
        )

        runner._get_seq_len_mode.assert_called_once_with(batch, None, promoted_ids)
        self.assertEqual(
            key, KeyType(batch_size=1, draft_len=0, short_seq_len_mode=True, num_encoder_tokens=0)
        )

    def test_graph_key_aggregates_encoder_tokens(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
        runner.max_beam_width = 1
        runner._get_seq_len_mode.return_value = False
        runner._ragged_verify_bucket.return_value = None
        context = _make_request_stub(1)
        context.encoder_output_len = 7
        context.py_skip_cross_kv_projection = False
        skipped_context = _make_request_stub(2)
        skipped_context.encoder_output_len = 11
        skipped_context.py_skip_cross_kv_projection = True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context, skipped_context]
        batch.generation_requests = [_make_request_stub(3)]

        key = CUDAGraphRunner.get_graph_key(runner, batch)

        assert key is not None
        self.assertEqual(key.num_contexts, 2)
        self.assertEqual(key.context_query_len, 1)
        self.assertEqual(key.num_encoder_tokens, 7)
        self.assertEqual(CUDAGraphRunner._get_num_tokens_for_key(runner, key), 3)

    def test_graph_key_rejects_nonuniform_context_query_lengths(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
        runner._get_seq_len_mode.return_value = False
        runner._ragged_verify_bucket.return_value = None
        first_context = _make_request_stub(1)
        first_context.encoder_output_len = 7
        first_context.py_skip_cross_kv_projection = False
        second_context = _make_request_stub(2)
        second_context.context_chunk_size = 2
        second_context.encoder_output_len = 11
        second_context.py_skip_cross_kv_projection = False
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [first_context, second_context]
        batch.generation_requests = [_make_request_stub(3)]

        key = CUDAGraphRunner.get_graph_key(runner, batch)

        self.assertIsNone(key)

    def test_graph_key_rounds_encoder_tokens_up_to_captured_extent(self) -> None:
        key = KeyType(
            batch_size=2, draft_len=0, num_contexts=1, context_query_len=1, num_encoder_tokens=7
        )
        smaller_key = key._replace(num_encoder_tokens=6)
        compatible_key = key._replace(num_encoder_tokens=8)
        larger_key = key._replace(num_encoder_tokens=16)
        runner = SimpleNamespace(
            padding_enabled=True,
            _capture_allowed=False,
            graph_metadata={},
            graph_outputs={
                smaller_key: object(),
                compatible_key: object(),
                larger_key: object(),
            },
        )

        actual_key = CUDAGraphRunner._get_compatible_mixed_encoder_decoder_key(runner, key)

        self.assertEqual(actual_key, compatible_key)

    def test_graph_key_includes_peft_cache_dtype(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
        runner._get_seq_len_mode.return_value = False
        runner._ragged_verify_bucket.return_value = None
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]

        model_dtype_key = CUDAGraphRunner.get_graph_key(
            runner, batch, peft_cache_data_type=torch.bfloat16
        )
        fp8_key = CUDAGraphRunner.get_graph_key(
            runner, batch, peft_cache_data_type=torch.float8_e4m3fn
        )

        self.assertNotEqual(model_dtype_key, fp8_key)
        self.assertEqual(
            model_dtype_key._replace(peft_cache_data_type=None),
            fp8_key._replace(peft_cache_data_type=None),
        )

    def test_graph_dtype_change_falls_back_to_eager(self) -> None:
        runner = Mock()
        runner.enabled = True
        runner._agree_graph_shape = CUDAGraphRunner._agree_graph_shape.__get__(runner)
        runner._dspark_confidence_enabled = False
        runner.config = SimpleNamespace(
            enable_attention_dp=False,
            use_mrope=False,
        )
        model_dtype_key = KeyType(batch_size=1, draft_len=0, peft_cache_data_type=torch.bfloat16)
        fp8_key = KeyType(batch_size=1, draft_len=0, peft_cache_data_type=torch.float8_e4m3fn)
        runner.get_graph_key.return_value = fp8_key
        runner.graph_metadata = {model_dtype_key: object()}
        runner._capture_allowed = False
        runner._is_mixed_encoder_decoder_batch.return_value = False
        runner._can_run_cuda_graph_batch.return_value = True

        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]

        with patch(
            "tensorrt_llm._torch.pyexecutor.cuda_graph_runner.ExpertStatistic.should_record",
            return_value=False,
        ):
            result = CUDAGraphRunner.maybe_get_cuda_graph(
                runner,
                batch,
                enable_spec_decode=False,
                attn_metadata=object(),
                peft_cache_data_type=torch.float8_e4m3fn,
            )

        self.assertEqual(result, (None, None, None))
        runner.get_graph_key.assert_called_once_with(
            batch, None, None, None, frozenset(), torch.float8_e4m3fn, False, None
        )

    def test_graph_key_includes_lora_variant(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
        runner._get_seq_len_mode.return_value = False
        runner._ragged_verify_bucket.return_value = None
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]

        key = CUDAGraphRunner.get_graph_key(
            runner,
            batch,
            use_lora_graph=True,
        )

        self.assertEqual(
            key,
            KeyType(batch_size=1, draft_len=0, use_lora_graph=True),
        )

    def test_graph_lookup_forwards_promoted_context_ids(self) -> None:
        runner = Mock()
        runner.enabled = True
        runner._agree_graph_shape = CUDAGraphRunner._agree_graph_shape.__get__(runner)
        runner._dspark_confidence_enabled = False
        runner.config = SimpleNamespace(
            enable_attention_dp=False,
            use_mrope=False,
        )
        key = KeyType(batch_size=1, draft_len=0, short_seq_len_mode=True, num_encoder_tokens=0)
        graph_attn_metadata = object()
        graph_spec_metadata = object()
        runner.get_graph_key.return_value = key
        runner.graphs = {key: object()}
        runner.graph_metadata = {
            key: {
                "attn_metadata": graph_attn_metadata,
                "spec_metadata": graph_spec_metadata,
            }
        }
        runner._is_mixed_encoder_decoder_batch.return_value = False
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]
        promoted_ids = frozenset({request.py_request_id})

        with patch(
            "tensorrt_llm._torch.pyexecutor.cuda_graph_runner.ExpertStatistic.should_record",
            return_value=False,
        ):
            result = CUDAGraphRunner.maybe_get_cuda_graph(
                runner,
                batch,
                enable_spec_decode=False,
                attn_metadata=object(),
                promoted_context_request_ids=promoted_ids,
            )

        runner.get_graph_key.assert_called_once_with(
            batch, None, None, None, promoted_ids, None, False, None
        )
        self.assertEqual(result, (graph_attn_metadata, graph_spec_metadata, key))


class CaptureStateTestCase(unittest.TestCase):
    def test_spec_decode_capture_restores_kv_lens_between_warmups(self) -> None:
        attn_metadata = Mock()
        attn_metadata.num_seqs = 1
        attn_metadata.kv_lens_cuda = torch.tensor([4095], dtype=torch.int32)

        saved_kv_lens_cuda = _save_spec_decode_capture_state(attn_metadata, enable_spec_decode=True)

        # CUDA graph capture performs two eager warmup forwards. A speculative
        # draft loop may advance the static attention metadata during each
        # forward, but the next warmup must start from the original input.
        for _ in range(2):
            attn_metadata.kv_lens_cuda.add_(1)
            _restore_spec_decode_capture_state(attn_metadata, saved_kv_lens_cuda)
            self.assertEqual(attn_metadata.kv_lens_cuda.tolist(), [4095])

        self.assertEqual(attn_metadata.on_update_kv_lens.call_count, 2)


def test_encoder_capture_discards_warmup_and_capture_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discarded = []

    @contextlib.contextmanager
    def discard(inputs: dict[str, object]) -> Iterator[None]:
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
        enable_attention_dp=False,
        original_max_total_draft_tokens=0,
        is_encoder_decoder=False,
        mapping=Mapping(),
        dist=None,
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


class SparseGraphKeyTestCase(unittest.TestCase):
    def test_sparse_sequence_mode_uses_promoted_context_cursor(self) -> None:
        sparse_config = Mock(spec=SeqLenAwareSparseAttentionConfig)
        sparse_config.needs_separate_short_long_cuda_graphs.return_value = True
        sparse_config.seq_len_threshold = 16
        runner = object.__new__(CUDAGraphRunner)
        runner.sparse_config = sparse_config
        runner.spec_config = None
        runner.graphs = {}
        runner.graph_outputs = {}
        runner.graph_metadata = {}
        runner.padding_dummy_requests = {}
        runner.memory_pool = None

        request = _make_request_stub(7, prompt_len=8)
        request.py_batch_idx = 0
        request.max_beam_num_tokens = 64
        batch = ScheduledRequests()
        batch.generation_requests = [request]
        overlap_state = SimpleNamespace(new_tokens=object())

        self.assertTrue(
            runner._get_seq_len_mode(batch, overlap_state, frozenset({request.py_request_id}))
        )
        self.assertFalse(runner._get_seq_len_mode(batch, overlap_state, frozenset()))
