# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import Mock, patch

import torch

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_encoder import \
    MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_multimodal_mixin import \
    MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import \
    KvCacheConnectorWorker
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner, EncoderCUDAGraphRunner, EncoderCUDAGraphRunnerConfig,
    KeyType, _restore_spec_decode_capture_state,
    _save_spec_decode_capture_state)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import \
    setup_mm_encoder_attn_metadata
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine, _build_request_multimodal_input,
    _filter_cuda_graph_batch_sizes, _get_context_prompt_lookahead_token,
    _make_single_token_context_graph_batch)
from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig,
                                          EncodeCudaGraphConfig,
                                          PrefillCudaGraphBackend,
                                          SeqLenAwareSparseAttentionConfig,
                                          TorchLlmArgs)

# isort: off
from tensorrt_llm._torch.pyexecutor.resource_manager import (KVCacheManager,
                                                             ResourceManager,
                                                             ResourceManagerType
                                                             )
# isort: on
from utils.util import skip_ray

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.interface import \
    INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_sampler_base import \
    SampleStateTensorsSpec
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.inputs.registry import (BaseMultimodalDummyInputsBuilder,
                                          BaseMultimodalInputProcessor)
from tensorrt_llm.llmapi import (CudaGraphConfig, SADecodingConfig,
                                 SamplingParams)
from tensorrt_llm.mapping import CpType, Mapping


@dataclass
class Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256
    architectures: list[str] = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class DummyKvCacheConnectorWorker(KvCacheConnectorWorker):

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        pass


class DummyModel(torch.nn.Module):

    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=Config(
            torch_dtype=dtype))
        self.recorded_position_ids = None

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        self.recorded_position_ids = kwargs["position_ids"]
        batch_size = input_ids.size(0)
        return {"logits": torch.randn((batch_size, 10), device='cuda')}


class DummyMultimodalIndexModel(torch.nn.Module):

    class Config:
        vocab_size = 100

    config = Config()

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        return torch.tensor([90, 91], dtype=torch.int32)


class DummyLegacyMultimodalIndexModel(MultimodalModelMixin, torch.nn.Module):

    class Config:
        vocab_size = 100

    config = Config()

    @property
    def mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([90, 91], dtype=torch.int32)


class DummyModelEngine(PyTorchModelEngine):

    def __init__(
        self,
        llm_args: TorchLlmArgs,
        dtype: torch.dtype,
        spec_config: DecodingBaseConfig | None = None,
    ) -> None:
        self.dtype = dtype
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          rank=tensorrt_llm.mpi_rank())
        model = DummyModel(self.dtype)
        super().__init__(model_path="dummy",
                         mapping=mapping,
                         model=model,
                         llm_args=llm_args,
                         spec_config=spec_config)


def _create_request(num_tokens, req_id: int):
    sampling_params = SamplingParams()
    kwargs = {
        "request_id":
        req_id,
        "max_new_tokens":
        1,
        "input_tokens": [0] * num_tokens,
        "sampling_config":
        tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        "is_streaming":
        False,
    }
    result = LlmRequest(**kwargs)
    result.paged_kv_block_ids = []
    return result


def _create_request_with_tokens(tokens: list[int], req_id: int) -> LlmRequest:
    sampling_params = SamplingParams()
    request = LlmRequest(
        request_id=req_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
    )
    request.paged_kv_block_ids = []
    return request


def test_context_prompt_lookahead_stops_at_prompt_boundary() -> None:
    request = _create_request_with_tokens([10, 11, 12, 13, 14], 1)

    assert _get_context_prompt_lookahead_token(request, 2) == 12
    assert _get_context_prompt_lookahead_token(request, 4) == 14
    assert (_get_context_prompt_lookahead_token(
        request, 5) == INVALID_PROMPT_LOOKAHEAD_TOKEN)


def _make_request_stub(req_id: int, prompt_len: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=req_id,
        context_chunk_size=1,
        context_remaining_length=1,
        context_current_position=prompt_len - 1,
        py_prompt_len=prompt_len,
        py_beam_width=1,
        py_draft_tokens=[],
        py_is_first_draft=False,
        is_context_only_request=False,
        is_generation_only_request=lambda: False,
        py_disaggregated_params=None,
        py_multimodal_data=None,
        py_mm_encoder_event=None,
        py_mrope_position_delta=None,
        py_return_context_logits=False,
        py_batch_idx=None,
        lora_task_id=None,
        is_dummy=False,
        max_beam_num_tokens=prompt_len,
        state="context",
        py_llm_request_type="context_and_generation",
    )


def _make_forward_only_engine(
    graph_key: KeyType | None,
    runner_enabled: bool = True,
) -> tuple[PyTorchModelEngine, Mock, Mock, Mock, dict[str, object]]:
    engine = object.__new__(PyTorchModelEngine)
    engine.model = SimpleNamespace(
        extra_attrs={},
        model_config=SimpleNamespace(pretrained_config=SimpleNamespace(
            rope_scaling=None)))
    engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
    engine.enable_spec_decode = False
    engine.is_spec_decode = False
    engine.is_draft_model = False
    engine.guided_decoder = None
    engine.max_beam_width = 1
    engine._is_encode_only = False
    engine.llm_args = SimpleNamespace(mm_encoder_only=False)
    engine.mapping = SimpleNamespace(
        cp_size=1,
        enable_lm_head_tp_in_adp=False,
    )
    engine.runtime_draft_len = 0
    engine.attn_backend = None
    engine.original_max_draft_len = 0
    engine.original_max_total_draft_tokens = 0
    engine._spec_dec_max_total_draft_tokens = 0
    engine.spec_config = None
    engine.get_runtime_tokens_per_gen_step = Mock(return_value=1)
    engine.iter_states = {}
    engine.forward_pass_callable = None
    engine.moe_load_balancer = None
    engine._is_encoder_decoder_model = Mock(return_value=False)
    engine._get_draft_kv_cache_manager = Mock(return_value=None)
    engine.cuda_graph_lora_manager = None
    engine._force_lora_graph_for_capture = None

    semantic_attn_metadata = Mock()
    graph_attn_metadata = Mock()
    engine.attn_metadata = semantic_attn_metadata
    engine._set_up_attn_metadata = Mock(return_value=semantic_attn_metadata)
    spec_dec_mode = Mock()
    spec_dec_mode.attention_need_spec_dec_mode.return_value = False
    spec_dec_mode.is_parallel_draft.return_value = False
    spec_metadata = Mock(
        spec_dec_mode=spec_dec_mode,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
    )
    engine.spec_metadata = spec_metadata
    engine._set_up_spec_metadata = Mock(return_value=spec_metadata)
    prepared_inputs = {
        "prepared": True,
        "input_ids": torch.zeros(2, dtype=torch.int32),
    }
    engine._prepare_inputs = Mock(return_value=(prepared_inputs, None))
    outputs = {"logits": object()}
    engine._forward_step = Mock(return_value=outputs)
    engine._execute_logit_post_processors = Mock()
    engine.breakable_cuda_graph_runner = None

    runner = Mock()
    runner.enabled = runner_enabled
    runner.pad_batch.side_effect = lambda batch, *_args: nullcontext(batch)
    runner.maybe_get_cuda_graph.return_value = ((graph_attn_metadata, None,
                                                 graph_key)
                                                if graph_key is not None else
                                                (None, None, None))
    runner.get_graph_pool.return_value = None
    runner.needs_capture.return_value = False
    runner.is_warmup_only = False
    runner.replay.return_value = outputs
    engine.cuda_graph_runner = runner

    resource_manager = Mock()
    peft_cache_manager = Mock(data_type=torch.bfloat16)

    def get_resource_manager(resource_type):
        if resource_type == ResourceManagerType.PEFT_CACHE_MANAGER:
            return peft_cache_manager
        return object()

    resource_manager.get_resource_manager.side_effect = get_resource_manager
    return engine, runner, resource_manager, semantic_attn_metadata, outputs


def create_model_engine_and_kvcache(
    llm_args: TorchLlmArgs | None = None,
    execution_stream: torch.cuda.Stream | None = None,
    spec_config: DecodingBaseConfig | None = None,
) -> tuple[PyTorchModelEngine, KVCacheManager]:
    tokens_per_block = 1
    max_tokens = 258  # Atleast 1 more than the max seq len
    num_layers = 1
    batch_size = 13

    if llm_args is None:
        llm_args = TorchLlmArgs(model="dummy",
                                max_batch_size=batch_size,
                                max_num_tokens=max_tokens,
                                cuda_graph_config=CudaGraphConfig(
                                    enable_padding=True,
                                    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]))
        # The padding tests below rely on these properties of the default
        # config: batches of 5 and 13 must round up to 8 and 16 respectively.
        test_batches = (5, 13)
        for test_batch_size in test_batches:
            assert test_batch_size not in llm_args.cuda_graph_config.batch_sizes

        assert (8 in llm_args.cuda_graph_config.batch_sizes
                and 16 in llm_args.cuda_graph_config.batch_sizes)

    model_engine = DummyModelEngine(llm_args, torch.half, spec_config)

    kv_cache_config = KvCacheConfig(max_tokens=max_tokens)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=num_layers,
        num_kv_heads=model_engine.model.config.num_key_value_heads,
        head_dim=model_engine.model.config.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_tokens,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.HALF,
        execution_stream=execution_stream,
    )

    return model_engine, kv_cache_manager


class SingleTokenContextGraphBatchTestCase(unittest.TestCase):

    def test_generation_only_is_identity(self) -> None:
        generation = _make_request_stub(1)
        batch = ScheduledRequests()
        batch.generation_requests = [generation]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch)

        self.assertIs(graph_batch, batch)
        self.assertEqual(promoted_ids, frozenset())

    def test_eligible_batch_has_independent_lists_and_stable_order(
            self) -> None:
        context_0 = _make_request_stub(10, prompt_len=1)
        context_1 = _make_request_stub(11, prompt_len=8)
        generation = _make_request_stub(12, prompt_len=16)
        paused = object()
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context_0, context_1]
        batch.generation_requests = [generation]
        batch.paused_requests = [paused]
        semantic_lists = (
            batch.encoder_requests,
            batch.context_requests_chunking,
            batch.context_requests_last_chunk,
            batch.generation_requests,
            batch.paused_requests,
        )
        semantic_snapshot = vars(context_1).copy()

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch)

        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.context_requests, [])
        self.assertEqual(graph_batch.generation_requests,
                         [context_0, context_1, generation])
        self.assertEqual(graph_batch.paused_requests, [paused])
        graph_lists = (
            graph_batch.encoder_requests,
            graph_batch.context_requests_chunking,
            graph_batch.context_requests_last_chunk,
            graph_batch.generation_requests,
            graph_batch.paused_requests,
        )
        for semantic_list, graph_list in zip(semantic_lists, graph_lists):
            self.assertIsNot(semantic_list, graph_list)
        self.assertEqual(promoted_ids, frozenset({10, 11}))
        self.assertEqual(vars(context_1), semantic_snapshot)

        graph_batch.generation_requests.append(object())
        self.assertEqual(batch.context_requests_last_chunk,
                         [context_0, context_1])
        self.assertEqual(batch.generation_requests, [generation])

    def test_structural_fallbacks_return_semantic_batch(self) -> None:
        context = _make_request_stub(1)

        encoder_batch = ScheduledRequests()
        encoder_batch.encoder_requests = [object()]
        encoder_batch.context_requests_last_chunk = [context]
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            encoder_batch)
        self.assertIs(graph_batch, encoder_batch)
        self.assertFalse(promoted_ids)

        chunking_batch = ScheduledRequests()
        chunking_batch.context_requests_chunking = [context]
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            chunking_batch)
        self.assertIs(graph_batch, chunking_batch)
        self.assertFalse(promoted_ids)

    def test_context_shape_and_mode_fallback_matrix(self) -> None:
        cases = (
            ("multi_token_chunk", "context_chunk_size", 2),
            ("more_context_remaining", "context_remaining_length", 2),
            ("cursor_prompt_mismatch", "py_prompt_len", 5),
            ("beam", "py_beam_width", 2),
            ("draft", "py_draft_tokens", [9]),
            ("first_draft", "py_is_first_draft", True),
            ("context_only", "is_context_only_request", True),
            ("disaggregated", "py_disaggregated_params", object()),
            ("multimodal", "py_multimodal_data", {}),
            ("multimodal_event", "py_mm_encoder_event", object()),
        )
        for name, attribute, value in cases:
            with self.subTest(name=name):
                context = _make_request_stub(1)
                setattr(context, attribute, value)
                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [context]

                graph_batch, promoted_ids = \
                    _make_single_token_context_graph_batch(batch)

                self.assertIs(graph_batch, batch)
                self.assertFalse(promoted_ids)

    def test_context_logits_use_final_token_graph_candidate(self) -> None:
        context = _make_request_stub(1)
        context.py_return_context_logits = True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch)

        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))

    def test_generation_only_request_in_context_list_falls_back(self) -> None:
        context = _make_request_stub(1)
        context.is_generation_only_request = lambda: True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)

    def test_generation_shape_fallback_matrix(self) -> None:
        cases = (
            ("beam", "py_beam_width", 2),
            ("draft", "py_draft_tokens", [9]),
            ("first_draft", "py_is_first_draft", True),
            ("disaggregated", "py_disaggregated_params", object()),
        )
        for name, attribute, value in cases:
            with self.subTest(name=name):
                context = _make_request_stub(1)
                generation = _make_request_stub(2)
                setattr(generation, attribute, value)
                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [context]
                batch.generation_requests = [generation]

                graph_batch, promoted_ids = \
                    _make_single_token_context_graph_batch(batch)

                self.assertIs(graph_batch, batch)
                self.assertFalse(promoted_ids)

    def test_mixed_one_and_two_token_contexts_fall_back_together(self) -> None:
        one_token = _make_request_stub(1)
        two_tokens = _make_request_stub(2)
        two_tokens.context_current_position -= 1
        two_tokens.context_remaining_length = 2
        two_tokens.context_chunk_size = 2
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [one_token, two_tokens]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch)

        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)
        self.assertEqual(batch.context_requests_last_chunk,
                         [one_token, two_tokens])

    def test_mrope_delta_is_supported_by_decode_provider(self) -> None:
        context = _make_request_stub(1)
        context.py_mrope_position_delta = object()
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch)

        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))

    def test_multimodal_context_requires_compatible_decode_token(self) -> None:
        context = _make_request_stub(1)
        context.py_multimodal_data = {}
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)

        incompatible = Mock(return_value=False)
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch, incompatible)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)
        incompatible.assert_called_once_with(context)

        compatible = Mock(return_value=True)
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch, compatible)
        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))
        compatible.assert_called_once_with(context)

    def test_multimodal_pending_event_is_rechecked(self) -> None:
        context = _make_request_stub(1)
        context.py_multimodal_data = {}
        context.py_mm_encoder_event = object()
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        compatible = Mock(return_value=True)

        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch, compatible)
        self.assertIs(graph_batch, batch)
        self.assertFalse(promoted_ids)
        compatible.assert_not_called()

        context.py_mm_encoder_event = None
        graph_batch, promoted_ids = _make_single_token_context_graph_batch(
            batch, compatible)
        self.assertIsNot(graph_batch, batch)
        self.assertEqual(promoted_ids, frozenset({context.py_request_id}))
        compatible.assert_called_once_with(context)

    def test_multimodal_decode_compatibility_uses_final_prompt_token(
            self) -> None:
        engine = object.__new__(PyTorchModelEngine)
        engine.model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=100),
            mm_token_ids=torch.tensor([99], dtype=torch.int32),
        )
        request = _create_request_with_tokens([11, 99, 22], 1)

        request.context_current_position = 2
        self.assertTrue(
            engine._is_final_multimodal_context_decode_compatible(request))

        request.context_current_position = 1
        self.assertFalse(
            engine._is_final_multimodal_context_decode_compatible(request))

        engine.model.mm_token_ids = None
        request = _create_request_with_tokens([11, 100], 2)
        request.context_current_position = 1
        self.assertFalse(
            engine._is_final_multimodal_context_decode_compatible(request))

        engine.model.mm_token_ids = torch.tensor([99], dtype=torch.int32)
        engine.model.model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(rope_scaling={"type": "mrope"}))
        request = _create_request_with_tokens([11, 22], 3)
        request.context_current_position = 1
        request.py_multimodal_data = {"mrope_config": {}}
        self.assertTrue(
            engine._is_final_multimodal_context_decode_compatible(request))

        request.py_multimodal_data["multimodal_embedding"] = object()
        self.assertFalse(
            engine._is_final_multimodal_context_decode_compatible(request))

        request.py_multimodal_data["mrope_config"][
            "mrope_position_deltas"] = object()
        self.assertTrue(
            engine._is_final_multimodal_context_decode_compatible(request))

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
            runner._get_seq_len_mode(batch, overlap_state,
                                     frozenset({request.py_request_id})))
        self.assertFalse(
            runner._get_seq_len_mode(batch, overlap_state, frozenset()))

    def test_graph_key_forwards_promoted_context_ids(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace(is_draft_model=False)
        runner._get_seq_len_mode.return_value = True
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

        runner._get_seq_len_mode.assert_called_once_with(
            batch, None, promoted_ids)
        self.assertEqual(
            key,
            KeyType(batch_size=1,
                    draft_len=0,
                    is_first_draft=False,
                    short_seq_len_mode=True,
                    num_encoder_tokens=0))

    def test_graph_key_aggregates_encoder_tokens(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace(is_draft_model=False)
        runner.max_beam_width = 1
        runner._get_seq_len_mode.return_value = False
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
        self.assertEqual(CUDAGraphRunner._get_num_tokens_for_key(runner, key),
                         3)

    def test_graph_key_rejects_nonuniform_context_query_lengths(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace(is_draft_model=False)
        runner._get_seq_len_mode.return_value = False
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

    def test_graph_key_rounds_encoder_tokens_up_to_captured_extent(
            self) -> None:
        key = KeyType(batch_size=2,
                      draft_len=0,
                      is_first_draft=False,
                      num_contexts=1,
                      context_query_len=1,
                      num_encoder_tokens=7)
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

        actual_key = CUDAGraphRunner._get_compatible_mixed_encoder_decoder_key(
            runner, key)

        self.assertEqual(actual_key, compatible_key)

    def test_graph_key_includes_peft_cache_dtype(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace(is_draft_model=False)
        runner._get_seq_len_mode.return_value = False
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]

        model_dtype_key = CUDAGraphRunner.get_graph_key(
            runner, batch, peft_cache_data_type=torch.bfloat16)
        fp8_key = CUDAGraphRunner.get_graph_key(
            runner, batch, peft_cache_data_type=torch.float8_e4m3fn)

        self.assertNotEqual(model_dtype_key, fp8_key)
        self.assertEqual(
            model_dtype_key._replace(peft_cache_data_type=None),
            fp8_key._replace(peft_cache_data_type=None),
        )

    def test_graph_dtype_change_falls_back_to_eager(self) -> None:
        runner = Mock()
        runner.enabled = True
        runner.config = SimpleNamespace(
            enable_attention_dp=False,
            use_mrope=False,
        )
        model_dtype_key = KeyType(batch_size=1,
                                  draft_len=0,
                                  is_first_draft=False,
                                  peft_cache_data_type=torch.bfloat16)
        fp8_key = KeyType(batch_size=1,
                          draft_len=0,
                          is_first_draft=False,
                          peft_cache_data_type=torch.float8_e4m3fn)
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
                return_value=False):
            result = CUDAGraphRunner.maybe_get_cuda_graph(
                runner,
                batch,
                enable_spec_decode=False,
                attn_metadata=object(),
                peft_cache_data_type=torch.float8_e4m3fn,
            )

        self.assertEqual(result, (None, None, None))

    def test_graph_key_includes_lora_variant(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace(is_draft_model=False)
        runner._get_seq_len_mode.return_value = False
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
            KeyType(batch_size=1,
                    draft_len=0,
                    is_first_draft=False,
                    use_lora_graph=True),
        )

    def test_lora_graph_variant_selection(self) -> None:
        engine = object.__new__(PyTorchModelEngine)
        engine.cuda_graph_lora_manager = object()
        engine._force_lora_graph_for_capture = None
        lora_config = SimpleNamespace(cuda_graph_specialize_lora=True)
        engine.llm_args = SimpleNamespace(lora_config=lora_config)
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]

        self.assertFalse(engine._use_lora_cuda_graph(batch))

        request.lora_task_id = 42
        self.assertTrue(engine._use_lora_cuda_graph(batch))

        request.lora_task_id = None
        lora_config.cuda_graph_specialize_lora = False
        self.assertTrue(engine._use_lora_cuda_graph(batch))

        engine._force_lora_graph_for_capture = False
        self.assertFalse(engine._use_lora_cuda_graph(batch))

    def test_graph_lookup_forwards_promoted_context_ids(self) -> None:
        runner = Mock()
        runner.enabled = True
        runner.config = SimpleNamespace(
            enable_attention_dp=False,
            use_mrope=False,
        )
        key = KeyType(batch_size=1,
                      draft_len=0,
                      is_first_draft=False,
                      short_seq_len_mode=True,
                      num_encoder_tokens=0)
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
                return_value=False):
            result = CUDAGraphRunner.maybe_get_cuda_graph(
                runner,
                batch,
                enable_spec_decode=False,
                attn_metadata=object(),
                promoted_context_request_ids=promoted_ids,
            )

        runner.get_graph_key.assert_called_once_with(batch, None, None, None,
                                                     promoted_ids, None, False)
        self.assertEqual(result,
                         (graph_attn_metadata, graph_spec_metadata, key))

    def test_forward_commits_candidate_only_on_graph_hit(self) -> None:
        key = KeyType(batch_size=2, draft_len=0, is_first_draft=False)
        engine, runner, resource_manager, _, outputs = \
            _make_forward_only_engine(key)
        context = _make_request_stub(1)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]
        event = Mock()

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=event):
            actual_outputs = engine.forward(batch, resource_manager)

        self.assertIs(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context, generation])
        prepare_args = engine._prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], graph_batch)
        self.assertEqual(prepare_args[-1], frozenset({1}))
        prepared_inputs = engine._prepare_inputs.return_value[0]
        runner.replay.assert_called_once_with(key, prepared_inputs)
        engine._forward_step.assert_not_called()
        engine._execute_logit_post_processors.assert_called_once_with(
            batch, outputs)
        self.assertEqual(engine.iter_states['num_ctx_requests'], 1)
        self.assertEqual(engine.iter_states['num_ctx_tokens'], 1)
        self.assertEqual(engine.iter_states['num_generation_tokens'], 1)
        event.record.assert_called_once()

    def test_forward_graph_miss_uses_semantic_eager_batch(self) -> None:
        engine, runner, resource_manager, semantic_attn_metadata, outputs = \
            _make_forward_only_engine(None)
        context = _make_request_stub(1)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch, resource_manager)

        self.assertIs(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertIsNot(graph_batch, batch)
        prepare_args = engine._prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], batch)
        self.assertIs(prepare_args[2], semantic_attn_metadata)
        self.assertEqual(prepare_args[-1], frozenset())
        engine._forward_step.assert_called_once()
        runner.replay.assert_not_called()
        engine._execute_logit_post_processors.assert_called_once_with(
            batch, outputs)

    def test_zero_runtime_draft_speculation_commits_graph_candidate(
            self) -> None:
        key = KeyType(batch_size=2, draft_len=0, is_first_draft=False)
        engine, runner, resource_manager, semantic_attn_metadata, outputs = \
            _make_forward_only_engine(key)
        engine.enable_spec_decode = True
        engine.spec_config = SimpleNamespace(is_linear_tree=True)
        graph_attn_metadata = runner.maybe_get_cuda_graph.return_value[0]
        runner.maybe_get_cuda_graph.return_value = (
            graph_attn_metadata,
            engine.spec_metadata,
            key,
        )
        context = _make_request_stub(1)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch, resource_manager)

        self.assertIs(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertEqual(graph_batch.generation_requests, [context, generation])
        self.assertTrue(
            runner.maybe_get_cuda_graph.call_args.kwargs["enable_spec_decode"])
        engine.spec_metadata.update_is_all_greedy_sample.assert_called_once_with(
            graph_batch.all_requests())
        prepare_args = engine._prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], graph_batch)
        self.assertIs(prepare_args[3], engine.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset({context.py_request_id}))
        semantic_attn_metadata.update_spec_dec_param.assert_called_once()
        self.assertEqual(
            semantic_attn_metadata.update_spec_dec_param.call_args.
            kwargs["num_contexts"], 1)
        prepared_inputs = engine._prepare_inputs.return_value[0]
        runner.replay.assert_called_once_with(key, prepared_inputs)

    def test_zero_runtime_draft_speculation_graph_miss_is_semantic_eager(
            self) -> None:
        engine, runner, resource_manager, semantic_attn_metadata, outputs = \
            _make_forward_only_engine(None)
        engine.enable_spec_decode = True
        engine.spec_config = SimpleNamespace(is_linear_tree=True)
        context = _make_request_stub(1)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch, resource_manager)

        self.assertIs(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertEqual(graph_batch.generation_requests, [context])
        prepare_args = engine._prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], batch)
        self.assertIs(prepare_args[2], semantic_attn_metadata)
        self.assertIs(prepare_args[3], engine.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset())
        engine._forward_step.assert_called_once()
        runner.replay.assert_not_called()

    def test_zero_runtime_non_linear_tree_speculation_uses_semantic_eager_batch(
            self) -> None:
        engine, runner, resource_manager, semantic_attn_metadata, outputs = \
            _make_forward_only_engine(None)
        engine.enable_spec_decode = True
        engine.spec_config = SimpleNamespace(is_linear_tree=False)
        context = _make_request_stub(1)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine._make_single_token_context_graph_batch"
        ) as selector, patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch, resource_manager)

        self.assertIs(actual_outputs, outputs)
        selector.assert_not_called()
        self.assertIs(runner.maybe_get_cuda_graph.call_args.args[0], batch)
        prepare_args = engine._prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], batch)
        self.assertIs(prepare_args[2], semantic_attn_metadata)
        self.assertIs(prepare_args[3], engine.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset())
        engine._forward_step.assert_called_once()
        runner.replay.assert_not_called()

    def test_forward_allows_guided_context_logits_on_graph_hit(self) -> None:
        key = KeyType(batch_size=1, draft_len=0, is_first_draft=False)
        engine, runner, resource_manager, _, outputs = \
            _make_forward_only_engine(key)
        engine.guided_decoder = Mock()
        context = _make_request_stub(1)
        context.py_return_context_logits = True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch,
                                            resource_manager,
                                            gather_context_logits=True)

        self.assertIs(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertEqual(graph_batch.generation_requests, [context])
        prepare_args = engine._prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], graph_batch)
        self.assertEqual(prepare_args[-1], frozenset({context.py_request_id}))
        prepared_inputs = engine._prepare_inputs.return_value[0]
        runner.replay.assert_called_once_with(key, prepared_inputs)

    def test_multimodal_graph_miss_preserves_semantic_payload(self) -> None:
        engine, runner, resource_manager, _, _ = _make_forward_only_engine(None)
        engine.model.config = SimpleNamespace(vocab_size=100)
        engine.model.mm_token_ids = torch.tensor([99], dtype=torch.int32)
        context = _make_request_stub(1, prompt_len=3)
        context.get_tokens = Mock(return_value=[99, 11, 22])
        multimodal_data = {
            "multimodal_embedding": object(),
            "mrope_config": {
                "mrope_position_deltas": object()
            },
        }
        context.py_multimodal_data = multimodal_data
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            engine.forward(batch, resource_manager)

        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertIsNot(graph_batch, batch)
        self.assertEqual(graph_batch.generation_requests, [context])
        self.assertIs(engine._prepare_inputs.call_args.args[0], batch)
        self.assertIs(context.py_multimodal_data, multimodal_data)
        self.assertIn("multimodal_embedding", multimodal_data)

    def test_breakable_graph_falls_back_for_context_logits(self) -> None:
        engine, _, resource_manager, _, outputs = \
            _make_forward_only_engine(None)
        breakable_runner = Mock()
        breakable_runner.is_capturing = False
        breakable_runner.is_warming_up = False
        breakable_runner.has_graph.return_value = True
        breakable_runner.execute.return_value = outputs
        engine.breakable_cuda_graph_runner = breakable_runner

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [_make_request_stub(1)]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()
        ), patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.get_per_request_prefill_cuda_graph_flag",
                return_value=True):
            actual_outputs = engine.forward(
                batch,
                resource_manager,
                gather_context_logits=True,
            )

        self.assertIs(actual_outputs, outputs)
        breakable_runner.execute.assert_not_called()
        engine._forward_step.assert_called_once()

    def test_generation_only_forward_does_not_call_new_selector(self) -> None:
        key = KeyType(batch_size=1, draft_len=0, is_first_draft=False)
        engine, runner, resource_manager, _, _ = _make_forward_only_engine(key)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.generation_requests = [generation]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine._make_single_token_context_graph_batch"
        ) as selector, patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            engine.forward(batch, resource_manager)

        selector.assert_not_called()
        self.assertIs(runner.maybe_get_cuda_graph.call_args.args[0], batch)
        self.assertFalse(
            runner.maybe_get_cuda_graph.call_args.kwargs["use_lora_graph"])
        self.assertIs(engine._prepare_inputs.call_args.args[0], batch)
        self.assertEqual(engine._prepare_inputs.call_args.args[-1], frozenset())

    def test_generation_lora_request_selects_lora_graph(self) -> None:
        key = (1, 0, False, False, True, True)
        engine, runner, resource_manager, _, _ = _make_forward_only_engine(key)
        engine.cuda_graph_lora_manager = object()
        engine.llm_args.lora_config = SimpleNamespace(
            cuda_graph_specialize_lora=True)
        generation = _make_request_stub(2)
        generation.lora_task_id = 42
        batch = ScheduledRequests()
        batch.generation_requests = [generation]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            engine.forward(batch, resource_manager)

        self.assertTrue(
            runner.maybe_get_cuda_graph.call_args.kwargs["use_lora_graph"])
        self.assertEqual(
            runner.maybe_get_cuda_graph.call_args.
            kwargs["peft_cache_data_type"], torch.bfloat16)
        self.assertTrue(
            engine._prepare_inputs.call_args.kwargs["use_lora_graph"])

    def test_global_incompatibilities_bypass_candidate_selection(self) -> None:
        cases = (
            "graphs_disabled",
            "speculative_nonzero_runtime_draft",
            "speculative_draft_model",
            "beam",
            "encoder_decoder",
            "encode_only",
            "mm_encoder_only",
            "ple_recurrent_state",
            "nested_ple_recurrent_state",
            "context_parallel",
        )
        for case in cases:
            with self.subTest(case=case):
                engine, runner, resource_manager, _, _ = \
                    _make_forward_only_engine(None)
                gather_context_logits = False
                if case == "graphs_disabled":
                    runner.enabled = False
                elif case == "speculative_nonzero_runtime_draft":
                    engine.enable_spec_decode = True
                    engine.runtime_draft_len = 1
                elif case == "speculative_draft_model":
                    engine.enable_spec_decode = True
                    engine.is_draft_model = True
                elif case == "beam":
                    engine.max_beam_width = 2
                elif case == "encoder_decoder":
                    engine._is_encoder_decoder_model.return_value = True
                elif case == "encode_only":
                    engine._is_encode_only = True
                elif case == "mm_encoder_only":
                    engine.llm_args.mm_encoder_only = True
                elif case == "ple_recurrent_state":
                    engine.model.has_ple = True
                elif case == "nested_ple_recurrent_state":
                    engine.model.model = SimpleNamespace(llm=SimpleNamespace(
                        model=SimpleNamespace(has_ple=True)))
                elif case == "context_parallel":
                    engine.mapping.cp_size = 2

                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [_make_request_stub(1)]
                with patch(
                        "tensorrt_llm._torch.pyexecutor.model_engine._make_single_token_context_graph_batch"
                ) as selector, patch(
                        "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                        return_value=Mock()):
                    engine.forward(
                        batch,
                        resource_manager,
                        gather_context_logits=gather_context_logits,
                    )

                selector.assert_not_called()
                self.assertIs(engine._prepare_inputs.call_args.args[0], batch)


class PyTorchModelEngineTestCase(unittest.TestCase):

    @staticmethod
    def _feature_encoder_runner(
            batch_sizes: List[int],
            fixed_seq_len: int = 1500) -> EncoderCUDAGraphRunner:
        """A feature-mode runner with capture disabled, so no CUDA is touched."""
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
            cuda_graph_mem_pool=None,
            is_encoder_decoder=True,
            use_fixed_sequence_slots=False,
            feature_shape=(480000, ),
            feature_dtype=torch.float32,
            fixed_seq_len=fixed_seq_len,
        )
        return EncoderCUDAGraphRunner(config)

    def test_feature_encoder_capture_keys_are_all_reachable(self) -> None:
        """Every feature capture key matches a batch shape the runtime can actually produce."""
        # Every request contributes exactly fixed_seq_len positions, so the
        # only reachable key per batch size is (bs, bs * fixed, fixed), and
        # every slot in that layout is a full fixed_seq_len sequence. The token
        # path's cross product would also emit keys whose token count no batch
        # can produce, and capture_keys drives mixed encoder/decoder
        # decoder-graph warmup.
        fixed = 1500
        batch_sizes = [1, 2, 4, 8]
        runner = self._feature_encoder_runner(batch_sizes, fixed)

        self.assertEqual(
            runner._capture_sequence_lengths,
            {(bs, bs * fixed, fixed): [fixed] * bs
             for bs in batch_sizes},
        )
        self.assertEqual(runner.capture_keys,
                         frozenset(runner._capture_sequence_lengths))

    @staticmethod
    def _encoder_spec_engine(
        encoder_cuda_graph_config: Optional[EncodeCudaGraphConfig],
        declares_spec: bool,
        tp_size: int = 1,
        is_encode_only: bool = False
    ) -> Tuple[PyTorchModelEngine, Tuple[Tuple[int, ...], torch.dtype, int]]:
        """A bare engine carrying only what `_encoder_graph_spec` reads."""
        spec = ((480000, ), torch.float32, 1500)

        class _Model:
            model_config = SimpleNamespace(is_encoder_decoder=True)

            if declares_spec:

                def encoder_graph_spec(
                        self) -> Tuple[Tuple[int, ...], torch.dtype, int]:
                    """Stand-in fixed-shape encoder contract for the test model."""
                    return spec

        engine = PyTorchModelEngine.__new__(PyTorchModelEngine)
        engine.encoder_cuda_graph_config = encoder_cuda_graph_config
        engine.is_draft_model = False
        engine._is_encode_only = is_encode_only
        engine.model = _Model()
        engine.mapping = SimpleNamespace(tp_size=tp_size)
        return engine, spec

    def test_encoder_graph_spec_selection(self) -> None:
        """The model, not the config, selects feature mode; TP > 1 stays eager."""
        # The model selects feature mode, not the config: an encoder either
        # takes fixed-shape features or it does not. TP > 1 is gated off
        # because allreduce inside encoder capture is unverified.
        declined = (None, None, None)
        cases = [
            ("feature model", EncodeCudaGraphConfig(batch_sizes=[1, 2]), True,
             1, None),
            ("token model",
             EncodeCudaGraphConfig(batch_sizes=[1],
                                   num_tokens=[1500],
                                   seq_lens=[1500]), False, 1, declined),
            ("no config", None, True, 1, declined),
            ("tensor parallel", EncodeCudaGraphConfig(batch_sizes=[1]), True, 2,
             declined),
        ]

        for name, config, declares_spec, tp_size, expected in cases:
            with self.subTest(name):
                engine, spec = self._encoder_spec_engine(
                    config, declares_spec=declares_spec, tp_size=tp_size)
                self.assertEqual(engine._encoder_graph_spec(),
                                 expected if expected is not None else spec)

    def test_encoder_graph_bucket_config_is_required_for_token_encoders(
            self) -> None:
        """A token encoder missing its bucket lists fails loudly instead of silently running eager."""
        # A token encoder's num_tokens/seq_lens buckets are the whole key
        # space, so a config missing them can only run eager — a loud failure,
        # not a silent perf regression. A feature encoder derives both from the
        # model, so the same config is complete there.
        engine, _ = self._encoder_spec_engine(
            EncodeCudaGraphConfig(batch_sizes=[1, 2]), declares_spec=False)
        with self.assertRaisesRegex(
                ValueError, "num_tokens/max_num_token and "
                "seq_lens/max_seq_len"):
            engine._check_encoder_graph_bucket_config([], [])

        engine, _ = self._encoder_spec_engine(EncodeCudaGraphConfig(
            batch_sizes=[1, 2], num_tokens=[1500]),
                                              declares_spec=False)
        with self.assertRaisesRegex(ValueError, "seq_lens/max_seq_len unset"):
            engine._check_encoder_graph_bucket_config([1500], [])

        for name, config, declares_spec in [
            ("token model with both buckets",
             EncodeCudaGraphConfig(batch_sizes=[1],
                                   num_tokens=[1500],
                                   seq_lens=[1500]), False),
            ("feature model derives both",
             EncodeCudaGraphConfig(batch_sizes=[1, 2]), True),
            ("no config", None, False),
        ]:
            with self.subTest(name):
                engine, _ = self._encoder_spec_engine(
                    config, declares_spec=declares_spec)
                num_tokens = config.num_tokens if config else []
                seq_lens = config.seq_lens if config else []
                engine._check_encoder_graph_bucket_config(
                    num_tokens or [], seq_lens or [])

    def test_encoder_graph_bucket_config_warns_for_encode_only(self) -> None:
        """An encode-only model warns and stays eager rather than raising."""
        # An encode-only model receives its buckets through `cuda_graph_config`,
        # a slot that has always accepted a batch-sizes-only
        # EncodeCudaGraphConfig and run eager. Raising there would break
        # deployments that predate feature mode, so warn and stay eager.
        engine, _ = self._encoder_spec_engine(
            EncodeCudaGraphConfig(batch_sizes=[1, 2]),
            declares_spec=False,
            is_encode_only=True)
        with patch("tensorrt_llm._torch.pyexecutor.model_engine.logger.warning"
                   ) as warning:
            engine._check_encoder_graph_bucket_config([], [])
        warning.assert_called_once()
        self.assertIn("stays eager", warning.call_args.args[0])

    def test_feature_encoder_batch_sizes_drop_past_the_token_budget(
            self) -> None:
        """The encoder token budget caps feature bucket sizes; an empty list means stay eager."""
        # A feature request costs a whole fixed_seq_len against the encoder
        # token budget, so the budget caps the bucket list far below
        # encoder_max_batch_size. An empty result is the signal to stay eager;
        # a floor of 1 here would capture a graph larger than the metadata
        # budget the encoder step actually builds.
        fixed = 1500
        for name, max_batch_size, max_num_tokens, expected in [
            ("budget allows every bucket", 8, 8 * fixed, [1, 2, 4, 8]),
            ("budget truncates the tail", 8, 2 * fixed, [1, 2]),
            ("budget below one request", 8, fixed - 1, []),
            ("batch size caps below the budget", 2, 8 * fixed, [1, 2]),
        ]:
            with self.subTest(name):
                self.assertEqual(
                    _filter_cuda_graph_batch_sizes([1, 2, 4, 8],
                                                   max_batch_size,
                                                   max_num_tokens,
                                                   fixed,
                                                   enable_padding=False),
                    expected)

    def test_feature_pad_batch_refuses_wide_bucket_gaps(self) -> None:
        """Feature padding is refused once it would cost more than 12.5% extra work."""
        # A feature pad slot is a full fixed_seq_len encoder forward, unlike the
        # 1-token pads of the token path, so padding is bounded at 12.5% extra
        # work. Consecutive powers of two never clear that bound; a batch of 8
        # padding to a configured bucket of 9 is the first case that does.
        fixed = 1500
        runner = self._feature_encoder_runner([1, 2, 4, 9], fixed)
        runner.enabled = True

        for name, batch_size, expected_seq_lens in [
            ("exact bucket yields unchanged", 4, [fixed] * 4),
            ("8 -> 9 is within 12.5%", 8, [fixed] * 9),
            ("3 -> 4 exceeds 12.5%", 3, [fixed] * 3),
            ("5 -> 9 exceeds 12.5%", 5, [fixed] * 5),
        ]:
            with self.subTest(name):
                inputs = {'seq_lens': [fixed] * batch_size}
                with runner.pad_batch(inputs, batch_size) as padded:
                    self.assertEqual(padded['seq_lens'], expected_seq_lens)

    def test_captured_graph_metadata_skips_the_eager_metadata_build(
            self) -> None:
        """A graph hit resolves captured metadata without an eager attn_metadata build."""
        # The runtime path asks for captured metadata before building any, so a
        # graph hit must not need an attn_metadata argument at all: on a hit
        # `maybe_get_cuda_graph` only reads it for a backend check.
        fixed = 1500
        runner = self._feature_encoder_runner([1, 2], fixed)
        runner.enabled = True
        runner.retire_staging = Mock()
        sentinel = object()
        key = (2, 2 * fixed, fixed)
        runner.graph_metadata[key] = {"attn_metadata": sentinel}

        metadata, hit_key = runner.captured_graph_metadata(
            {'seq_lens': [fixed] * 2})
        self.assertIs(metadata, sentinel)
        self.assertEqual(hit_key, key)
        runner.retire_staging.assert_called_once()

        # An uncaptured bucket must miss, leaving the caller to build metadata
        # and take the full path rather than silently reusing another key's.
        runner.retire_staging.reset_mock()
        self.assertEqual(runner.captured_graph_metadata({'seq_lens': [fixed]}),
                         (None, None))
        runner.retire_staging.assert_not_called()

    def test_encoder_cuda_graph_stages_and_restores_fixed_sequence_slots(
            self) -> None:
        runner = EncoderCUDAGraphRunner.__new__(EncoderCUDAGraphRunner)
        runner.is_encoder_decoder = True
        runner.use_fixed_sequence_slots = True
        runner.supported_batch_sizes = [2]
        runner.supported_seq_lens = [512]
        runner.max_supported_num_tokens = 1024
        small_key = (2, 512, 512)
        compatible_key = (2, 1024, 512)
        runner._capture_sequence_lengths = {
            small_key: [511, 1],
            compatible_key: [512, 512],
        }
        runner._capture_keys_by_batch_size = {
            2: [small_key, compatible_key],
        }
        runner._arange_max = torch.arange(1024, dtype=torch.int32)

        self.assertEqual(
            runner._get_dynamic_capture_key([200, 300],
                                            allow_batch_padding=False),
            compatible_key)

        source_sequence_lengths = [1, 400]
        key = runner._get_dynamic_capture_key(source_sequence_lengths,
                                              allow_batch_padding=False)
        self.assertEqual(key, small_key)
        self.assertEqual(runner._get_capture_sequence_offsets(key),
                         [0, 511, 512])

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
        self.assertEqual(inputs["seq_lens"], [400, 1])
        self.assertEqual(inputs["_encoder_source_to_slot"], [1, 0])

        static_tensors = {
            "input_ids": torch.empty(512, dtype=torch.int32),
            "position_ids": torch.empty((1, 512), dtype=torch.int32),
        }
        runner._stage_encoder_decoder_inputs(key, inputs, static_tensors)
        expected_staged_ids = torch.zeros(512, dtype=torch.int32)
        expected_staged_ids[:400] = input_ids[1:]
        expected_staged_ids[511] = input_ids[0]
        torch.testing.assert_close(static_tensors["input_ids"],
                                   expected_staged_ids)
        torch.testing.assert_close(static_tensors["position_ids"][0],
                                   expected_staged_ids)

        fixed_slot_output = torch.arange(512).unsqueeze(1)
        restored_output = runner.restore_encoder_decoder_output(
            key, fixed_slot_output, inputs)
        expected_output = torch.cat(
            (fixed_slot_output[511:512], fixed_slot_output[:400]))
        torch.testing.assert_close(restored_output, expected_output)

    def test_breakable_rejects_multimodal_models(self) -> None:
        engine = object.__new__(PyTorchModelEngine)
        engine.model = DummyLegacyMultimodalIndexModel()
        engine.input_processor = None
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            disable_mm_encoder=False)

        self.assertTrue(engine.is_multimodal)
        with self.assertRaisesRegex(ValueError, "multimodal models"):
            engine._validate_breakable_cuda_graph_compatibility()

    def test_breakable_allows_text_decoder_with_multimodal_processor(
            self) -> None:
        engine = object.__new__(PyTorchModelEngine)
        engine.model = Mock(spec=DecoderModelForCausalLM)
        engine.input_processor = Mock(spec=BaseMultimodalInputProcessor)
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            disable_mm_encoder=False)

        self.assertTrue(engine.is_multimodal)
        engine._validate_breakable_cuda_graph_compatibility()

    def test_breakable_allows_multimodal_wrapper_in_text_only_mode(
            self) -> None:
        engine = object.__new__(PyTorchModelEngine)
        engine.model = DummyLegacyMultimodalIndexModel()
        engine.model.llm = Mock(spec=DecoderModelForCausalLM)
        engine.model.mm_encoder = None
        engine.input_processor = Mock(spec=BaseMultimodalInputProcessor)
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            disable_mm_encoder=True)

        self.assertTrue(engine.is_multimodal)
        engine._validate_breakable_cuda_graph_compatibility()

    def test_prepare_multimodal_indices_uses_mixin_token_ids(self) -> None:
        engine = object.__new__(PyTorchModelEngine)
        engine.model = DummyMultimodalIndexModel()

        text_indices, multimodal_indices = engine._prepare_multimodal_indices(
            [1, 90, 2, 91, 3])

        torch.testing.assert_close(text_indices, torch.tensor([0, 2, 4]))
        torch.testing.assert_close(multimodal_indices, torch.tensor([1, 3]))

    def test_prepare_multimodal_indices_uses_legacy_token_ids(self) -> None:
        engine = object.__new__(PyTorchModelEngine)
        engine.model = DummyLegacyMultimodalIndexModel()

        text_indices, multimodal_indices = engine._prepare_multimodal_indices(
            [1, 90, 2, 91, 3])

        torch.testing.assert_close(text_indices, torch.tensor([0, 2, 4]))
        torch.testing.assert_close(multimodal_indices, torch.tensor([1, 3]))

    def test_build_request_multimodal_input_skips_when_cache_disabled(
            self) -> None:
        request = LlmRequest(
            request_id=1,
            max_new_tokens=1,
            input_tokens=[0, 1, 2],
            sampling_config=tensorrt_llm.bindings.SamplingConfig(1),
            is_streaming=False,
            multimodal_hashes=[[1, 2, 3, 4, 5, 6, 7, 8]],
            multimodal_positions=[1],
            multimodal_lengths=[1],
            multimodal_uuids=["image-0"],
        )

        # With the encoder cache disabled, nothing consumes `multimodal_input`,
        # so it should not be built at all.
        self.assertIsNone(
            _build_request_multimodal_input(request, cache_enabled=False))

    def test_spec_decode_capture_restores_kv_lens_between_warmups(self) -> None:
        attn_metadata = Mock()
        attn_metadata.num_seqs = 1
        attn_metadata.kv_lens_cuda = torch.tensor([4095], dtype=torch.int32)

        saved_kv_lens_cuda = _save_spec_decode_capture_state(
            attn_metadata, enable_spec_decode=True)

        # CUDA graph capture performs two eager warmup forwards. A speculative
        # draft loop may advance the static attention metadata during each
        # forward, but the next warmup must start from the original input.
        for _ in range(2):
            attn_metadata.kv_lens_cuda.add_(1)
            _restore_spec_decode_capture_state(attn_metadata,
                                               saved_kv_lens_cuda)
            self.assertEqual(attn_metadata.kv_lens_cuda.tolist(), [4095])

        self.assertEqual(attn_metadata.on_update_kv_lens.call_count, 2)

    def test_promoted_context_uses_prompt_token_during_overlap(self) -> None:
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=32,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        context = _create_request_with_tokens([11, 22, 33, 44], 1)
        context.context_current_position = 3
        context.context_chunk_size = 1
        context.py_seq_slot = 0
        context.py_batch_idx = 3
        context.py_num_compressed_tokens = 1

        generation = _create_request_with_tokens([50, 51, 52, 53, 54], 2)
        generation.py_seq_slot = 1
        generation.py_batch_idx = 1

        graph_batch = ScheduledRequests()
        graph_batch.generation_requests = [context, generation]
        new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")
        new_tokens[0, 0, 0] = 999
        new_tokens[0, 1, 0] = 777
        overlap_state = SimpleNamespace(new_tokens=new_tokens)
        model_engine._can_use_incremental_update = Mock(return_value=True)
        model_engine._can_use_steady_gen_fast_prepare = Mock(return_value=True)

        inputs, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=graph_batch,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            new_tensors_device=overlap_state,
            resource_manager=resource_manager,
            promoted_context_request_ids=frozenset({context.py_request_id}),
        )

        self.assertEqual(inputs["input_ids"][:2].cpu().tolist(), [44, 777])
        self.assertEqual(inputs["position_ids"][0, :2].cpu().tolist(), [3, 5])
        self.assertEqual(
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [2, 5])
        self.assertEqual(context.cached_tokens, 3)
        model_engine._can_use_incremental_update.assert_not_called()
        model_engine._can_use_steady_gen_fast_prepare.assert_not_called()
        self.assertEqual(
            model_engine.previous_batch_indices_cuda[:1].cpu().tolist(), [1])
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(model_engine.previous_request_ids,
                         [generation.py_request_id])
        kv_cache_manager.shutdown()

    def test_promoted_context_precedes_speculative_overlap_generation(
            self) -> None:
        spec_config = SADecodingConfig(
            max_draft_len=1,
            draft_len_schedule={1: 1},
        )
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            spec_config=spec_config)
        model_engine.runtime_draft_len = 0
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=32,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False
        # A bare Mock auto-vivifies every attribute, so optional metadata
        # fields read by _prepare_tp_inputs must be pinned off explicitly.
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )

        context = _create_request_with_tokens([11, 22, 33, 44], 1)
        context.context_current_position = 3
        context.context_chunk_size = 1
        context.py_seq_slot = 0
        # A promoted context row must ignore any stale overlap slot.
        context.py_batch_idx = 3

        generation = _create_request_with_tokens([50, 51, 52, 53, 54], 2)
        generation.py_seq_slot = 1
        generation.py_batch_idx = 1
        generation.py_needs_onehot_draft_probs = True

        graph_batch = ScheduledRequests()
        graph_batch.generation_requests = [context, generation]
        new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")
        new_tokens[0, 0, 0] = 999
        new_tokens[0, 1, 0] = 777
        overlap_state = SampleStateTensorsSpec(
            new_tokens=new_tokens,
            new_tokens_lens=torch.ones(4, dtype=torch.int32, device="cuda"),
            next_draft_tokens=torch.zeros((4, 1),
                                          dtype=torch.int32,
                                          device="cuda"),
        )

        inputs, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=graph_batch,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            new_tensors_device=overlap_state,
            resource_manager=resource_manager,
            promoted_context_request_ids=frozenset({context.py_request_id}),
        )

        self.assertEqual(inputs["input_ids"][:2].cpu().tolist(), [44, 777])
        self.assertEqual(inputs["position_ids"][0, :2].cpu().tolist(), [3, 4])
        self.assertEqual(attn_metadata.request_ids,
                         [context.py_request_id, generation.py_request_id])
        self.assertEqual(
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [3, 5])
        self.assertEqual(
            model_engine.previous_batch_indices_cuda[:1].cpu().tolist(), [1])
        self.assertEqual(
            model_engine.previous_pos_id_offsets_cuda[:2].cpu().tolist(),
            [0, 1])
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(model_engine.previous_request_ids,
                         [generation.py_request_id])
        self.assertEqual(spec_metadata.request_ids,
                         [context.py_request_id, generation.py_request_id])
        self.assertFalse(generation.py_needs_onehot_draft_probs)
        spec_metadata.write_padding_onehot_draft_probs.assert_called_once_with(
            [generation.py_seq_slot], 0)
        kv_cache_manager.shutdown()

    def test_multimodal_encoder_max_seq_len(self) -> None:

        class CapturingEncoder(torch.nn.Module, MultimodalEncoderMixin):

            def __init__(self) -> None:
                super().__init__()
                self.setup_args = None
                self.max_seq_len = None

            def setup_attn_metadata(self, max_num_tokens: int) -> None:
                self.setup_args = max_num_tokens

            def set_attn_max_seq_len(self, max_seq_len: int) -> None:
                self.max_seq_len = max_seq_len

        encoder_max_num_tokens = 16384
        processor_max_num_tokens = 65536
        cases = [
            ({
                "image": processor_max_num_tokens
            }, processor_max_num_tokens),
            ({
                "image": 4096
            }, encoder_max_num_tokens),
            ({}, encoder_max_num_tokens),
            (None, encoder_max_num_tokens),
        ]
        for max_tokens_per_item, expected_max_seq_len in cases:
            with self.subTest(max_tokens_per_item=max_tokens_per_item):
                encoder = CapturingEncoder()
                if max_tokens_per_item is None:
                    input_processor = Mock()
                else:
                    input_processor = Mock(
                        spec=BaseMultimodalDummyInputsBuilder)
                    input_processor.get_mm_max_tokens_per_item.return_value = max_tokens_per_item

                setup_mm_encoder_attn_metadata(torch.nn.Sequential(encoder),
                                               input_processor,
                                               encoder_max_num_tokens, None)

                self.assertEqual(encoder.setup_args, encoder_max_num_tokens)
                self.assertEqual(encoder.max_seq_len, expected_max_seq_len)

    def test_pad_generation_requests(self) -> None:
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        batch_sizes_and_seqlens = [
            (5, 1),
            (13, 1),
            (5, 25),
        ]
        for (batch_size, max_seq_len) in batch_sizes_and_seqlens:
            requests = [
                _create_request(max_seq_len, i) for i in range(batch_size)
            ]
            batch = ScheduledRequests()
            batch.context_requests_last_chunk = requests

            pages_before = kv_cache_manager.get_num_free_blocks()
            with model_engine.cuda_graph_runner.pad_batch(
                    batch, resource_manager) as padded_batch:
                # No padding for prefill
                self.assertIs(batch, padded_batch)
            self.assertEqual(kv_cache_manager.get_num_free_blocks(),
                             pages_before)

            batch = ScheduledRequests()
            batch.generation_requests = requests
            pages_before = kv_cache_manager.get_num_free_blocks()
            new_dummy_block = 1 if not model_engine.cuda_graph_runner.padding_dummy_requests else 0
            with model_engine.cuda_graph_runner.pad_batch(
                    batch, resource_manager) as padded_batch:
                if batch_size < 8 and max_seq_len < 25:
                    self.assertEqual(
                        len(padded_batch.generation_requests) % 8, 0)
                else:
                    # No padding if it would create too many concurrent requests.
                    # This requirement is not strictly required, but we should probably
                    # respect the requirement?
                    # The seqlen check makes sure we don't exceed the KV cache memory
                    # budget.
                    self.assertIs(batch, padded_batch)
            self.assertEqual(
                kv_cache_manager.get_num_free_blocks() + new_dummy_block,
                pages_before)

        kv_cache_manager.shutdown()

    def test_pad_batch_strips_cudagraph_dummies_on_clean_exit(self) -> None:
        # Regression guard for the invariant that CUDAGraphRunner.pad_batch's
        # `finally` strips every is_cuda_graph_dummy=True entry from
        # scheduled_requests.generation_requests before the `with` block
        # exits. Downstream consumers of scheduled_batch.generation_requests
        # — including the per-iteration stats populate block in
        # PyExecutor._update_iter_stats — rely on never observing
        # cudagraph dummies.
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        # batch_size=5 rounds up to 8 (nearest captured graph size in the
        # fixture config) -> padding_size=3, deterministically.
        real_batch_size = 5
        max_seq_len = 1
        real_requests = [
            _create_request(max_seq_len, i) for i in range(real_batch_size)
        ]
        real_ids = [id(r) for r in real_requests]

        batch = ScheduledRequests()
        batch.generation_requests = list(real_requests)

        with model_engine.cuda_graph_runner.pad_batch(
                batch, resource_manager) as padded_batch:
            # Positive assertion that padding actually fired — guards
            # against a vacuous pass where padding was a no-op.
            self.assertGreater(
                len(padded_batch.generation_requests), real_batch_size,
                "padding did not fire; fixture config may have drifted "
                "so that 5 no longer rounds up to 8")
            # Every appended entry past the original count is a
            # cudagraph-flagged dummy.
            for req in padded_batch.generation_requests[real_batch_size:]:
                self.assertTrue(
                    getattr(req, "is_cuda_graph_dummy", False),
                    "pad_batch appended a request without "
                    "is_cuda_graph_dummy=True")
            # Real requests' identities and order are untouched.
            self.assertEqual([
                id(r)
                for r in padded_batch.generation_requests[:real_batch_size]
            ], real_ids)

        # After the with-block: finally must have sliced off the padding.
        self.assertEqual(
            len(batch.generation_requests), real_batch_size,
            "pad_batch.finally did not strip cudagraph dummies — "
            "downstream consumers of scheduled_batch.generation_requests "
            "would observe the leaked dummies")
        for req in batch.generation_requests:
            self.assertFalse(
                getattr(req, "is_cuda_graph_dummy", False),
                "cudagraph dummy leaked out of pad_batch's finally")

        kv_cache_manager.shutdown()

    def test_pad_batch_strips_cudagraph_dummies_on_exception(self) -> None:
        # The strip must fire even when the body raises. This is the
        # critical property of `finally` vs. a plain trailing statement —
        # it guards the invariant on the error path. A refactor that
        # accidentally dropped the `finally` would be caught here but not
        # by the clean-exit variant.
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        real_batch_size = 5
        real_requests = [_create_request(1, i) for i in range(real_batch_size)]

        batch = ScheduledRequests()
        batch.generation_requests = list(real_requests)

        class _ForwardBoom(Exception):
            pass

        with self.assertRaises(_ForwardBoom):
            with model_engine.cuda_graph_runner.pad_batch(
                    batch, resource_manager) as padded_batch:
                self.assertGreater(len(padded_batch.generation_requests),
                                   real_batch_size)
                raise _ForwardBoom()

        self.assertEqual(len(batch.generation_requests), real_batch_size)
        for req in batch.generation_requests:
            self.assertFalse(getattr(req, "is_cuda_graph_dummy", False))

        kv_cache_manager.shutdown()

    def test_position_id_preparation(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 256
        requests = [_create_request(prompt_len, 0)]

        # Prefill run
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

        expected_prefill_pos_ids = torch.arange(0,
                                                prompt_len,
                                                dtype=torch.int32,
                                                device='cuda').unsqueeze(0)
        torch.testing.assert_close(model_engine.model.recorded_position_ids,
                                   expected_prefill_pos_ids,
                                   atol=0,
                                   rtol=0)

        # Simulate decoding one token after prefill
        requests[-1].add_new_token(42, 0)

        # Generation run
        batch = ScheduledRequests()
        batch.generation_requests = requests
        kv_cache_manager.prepare_resources(batch)

        model_engine.forward(batch, resource_manager)
        expected_gen_pos_id = torch.tensor([prompt_len],
                                           dtype=torch.int32,
                                           device='cuda').unsqueeze(0)
        torch.testing.assert_close(model_engine.model.recorded_position_ids,
                                   expected_gen_pos_id,
                                   atol=0,
                                   rtol=0)

        kv_cache_manager.shutdown()

    def test_warmup(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        # Test with a huge batch size. The warmup run should bail out of
        # warmup instead of crashing (there's not enough KV cache space for this).
        model_engine._cuda_graph_batch_sizes.append(1000000000)

        num_free_before = kv_cache_manager.get_num_free_blocks()
        model_engine.warmup(resource_manager)

        # Warmup pre-allocates the CUDA graph padding dummy for each captured
        # draft length (only draft_len 0 here, no speculation), which
        # intentionally keeps holding its KV blocks so padding cannot fall
        # back to eager once the cache saturates.
        padding_dummies = model_engine.cuda_graph_runner.padding_dummy_requests
        self.assertEqual([0], sorted(padding_dummies.keys()))

        # Make sure we don't leak any blocks beyond that dummy: freeing it
        # must restore the exact pre-warmup free-block count.
        kv_cache_manager.free_resources(padding_dummies[0])
        self.assertEqual(num_free_before,
                         kv_cache_manager.get_num_free_blocks())

        kv_cache_manager.shutdown()

    def test_warmup_skips_padding_dummy_when_padding_impossible(self):
        # With max_batch_size=1 and a graph for batch size 1, every batch
        # already matches a graph size, so no padded batch can ever occur and
        # warmup must not retain a padding dummy (it would permanently hold
        # KV blocks and spec/hybrid resources that are never used).
        llm_args = TorchLlmArgs(model="dummy",
                                max_batch_size=1,
                                max_num_tokens=258,
                                cuda_graph_config=CudaGraphConfig(
                                    enable_padding=True, batch_sizes=[1, 2, 4]))
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            llm_args)
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        num_free_before = kv_cache_manager.get_num_free_blocks()
        model_engine.warmup(resource_manager)

        self.assertEqual({},
                         model_engine.cuda_graph_runner.padding_dummy_requests)
        self.assertEqual(num_free_before,
                         kv_cache_manager.get_num_free_blocks())

        kv_cache_manager.shutdown()

    def test_warmup_skips_padding_dummy_during_estimation(self):
        # The estimation-phase KV cache is sized with no headroom for retained
        # dummies; holding blocks there can leave the estimation requests
        # unschedulable. Warmup must skip the preallocation for managers
        # created for KV cache capacity estimation.
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        kv_cache_manager.is_estimating_kv_cache = True
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        num_free_before = kv_cache_manager.get_num_free_blocks()
        model_engine.warmup(resource_manager)

        self.assertEqual({},
                         model_engine.cuda_graph_runner.padding_dummy_requests)
        self.assertEqual(num_free_before,
                         kv_cache_manager.get_num_free_blocks())

        kv_cache_manager.shutdown()

    def test_preallocate_padding_dummies_uses_captured_draft_lens(self):
        # Speculative engines pad batches with the runtime draft length, so
        # the preallocation must create dummies for the draft lengths of the
        # captured graphs — not draft_len 0 unconditionally (an extra
        # draft_len-0 dummy would permanently hold KV blocks that runtime
        # padding never uses).
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        runner = model_engine.cuda_graph_runner
        # Simulate a spec-decode capture: graphs keyed (batch_size, draft_len,
        # is_first_draft, short_seq_len_mode, is_all_greedy_sample) exist only
        # for draft_len 2.
        runner.graphs[(8, 2, False, False, True)] = Mock()
        try:
            runner.preallocate_padding_dummies(resource_manager)

            self.assertEqual([2], sorted(runner.padding_dummy_requests.keys()))
        finally:
            runner.graphs.clear()
            for dummy in runner.padding_dummy_requests.values():
                kv_cache_manager.free_resources(dummy)
            kv_cache_manager.shutdown()

    def test_release_padding_dummy_covers_every_manager(self):
        # A padding dummy's request ID is spread across several managers, so
        # releasing only the main KV cache manager leaves the others holding
        # it — and re-creation reuses the same ID.
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        spec_manager = Mock()
        cross_manager = Mock()
        resource_manager = ResourceManager({
            ResourceManagerType.KV_CACHE_MANAGER:
            kv_cache_manager,
            ResourceManagerType.SPEC_RESOURCE_MANAGER:
            spec_manager,
            ResourceManagerType.CROSS_KV_CACHE_MANAGER:
            cross_manager,
        })

        runner = model_engine.cuda_graph_runner
        try:
            self.assertIsNotNone(
                runner._get_or_create_padding_dummy(resource_manager, 0))
            dummy = runner.padding_dummy_requests[0]

            self.assertTrue(runner.release_padding_dummy(resource_manager, 0))

            # Dropped from the runner so the lazy path re-creates it...
            self.assertEqual({}, runner.padding_dummy_requests)
            # ...and the spec resource manager slot is released too, not just
            # the main KV cache manager.
            spec_manager.free_resources.assert_called_once_with(dummy)
            # The cross-KV manager is only involved for encoder-decoder, which
            # this engine is not.
            self.assertFalse(runner.is_encoder_decoder)
            cross_manager.free_resources.assert_not_called()

            # Releasing again is a no-op rather than a double free.
            self.assertFalse(runner.release_padding_dummy(resource_manager, 0))
            spec_manager.free_resources.assert_called_once()
        finally:
            for dummy in runner.padding_dummy_requests.values():
                kv_cache_manager.free_resources(dummy)
            runner.padding_dummy_requests.clear()
            kv_cache_manager.shutdown()

    def test_layerwise_nvtx_marker(self):
        llm_args = TorchLlmArgs(
            model="dummy",
            enable_layerwise_nvtx_marker=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True))
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            llm_args)
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

        expected_prefill_pos_ids = torch.arange(0,
                                                prompt_len,
                                                dtype=torch.int32,
                                                device='cuda').unsqueeze(0)
        torch.testing.assert_close(model_engine.model.recorded_position_ids,
                                   expected_prefill_pos_ids,
                                   atol=0,
                                   rtol=0)

        kv_cache_manager.shutdown()

    def test_cuda_graph_padding_filters_huge_batch_size(self):
        llm_args = TorchLlmArgs(
            model="dummy",
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True,
                batch_sizes=[1, 2, 3, 1000000000000000000000000]))
        model_engine = DummyModelEngine(llm_args, torch.half)

        self.assertEqual(model_engine._cuda_graph_batch_sizes,
                         [1, 2, 3, model_engine.max_seq_len])

    def test_forward_pass_callable_on_cuda_graph_on(self):
        llm_args = TorchLlmArgs(model="dummy",
                                cuda_graph_config=CudaGraphConfig(
                                    enable_padding=True, ))
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            llm_args)

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_forward_pass_callable_on_cuda_graph_off(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_foward_pass_callable_off(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        self.assertTrue(model_engine.forward_pass_callable is None,
                        "forward_pass_callback should be None by default")

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

    def test_foward_pass_callable_backward_compat(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        self.assertTrue(model_engine.forward_pass_callable is None,
                        "forward_pass_callback should be None by default")

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        kv_cache_manager.prepare_resources(batch)
        model_engine.forward(batch, resource_manager)

    @skip_ray
    def test_prepare_tp_inputs_with_helix_parallelism(self) -> None:
        """Test _prepare_tp_inputs function with helix parallelism."""

        # Create model engine with helix parallelism.
        llm_args = TorchLlmArgs(model="dummy")
        model_engine = DummyModelEngine(llm_args, dtype=torch.half)

        # Provide mapping for model engine.
        cp_size = 2
        cp_rank = 0
        cp_config = {"cp_type": CpType.HELIX, "tokens_per_block": 4}
        mapping = Mapping(world_size=cp_size,
                          tp_size=1,
                          pp_size=1,
                          cp_size=cp_size,
                          cp_config=cp_config,
                          rank=cp_rank)
        model_engine.mapping = mapping

        # Create scheduled requests with two generation requests.
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        prompt_lens = [20, 15]
        gen_requests = []
        for idx in range(len(prompt_lens)):
            req = _create_request(num_tokens=prompt_lens[idx], req_id=idx + 1)
            req.py_prompt_len = prompt_lens[idx]
            req.py_batch_idx = None
            req.is_dummy_request = False
            req.py_seq_slot = idx
            req.sampling_config.beam_width = 1
            req.py_multimodal_data = {}
            req.total_input_len_cp = prompt_lens[idx] * 2
            req.seqlen_this_rank_cp = prompt_lens[idx]
            req.py_decoding_iter = 1
            gen_requests.append(req)
        scheduled_requests.generation_requests = gen_requests

        # Create KV cache manager for attention metadata.
        kv_cache_config = KvCacheConfig(max_tokens=512)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=16,
            head_dim=16,
            tokens_per_block=1,
            max_seq_len=512,
            max_batch_size=4,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.HALF,
        )
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=512,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        # Initialize model engine buffers.
        max_num_tokens = 512
        model_engine.max_num_tokens = max_num_tokens
        model_engine.input_ids_cuda = torch.zeros(max_num_tokens,
                                                  dtype=torch.int32,
                                                  device='cuda')
        model_engine.position_ids_cuda = torch.zeros(max_num_tokens,
                                                     dtype=torch.int32,
                                                     device='cuda')
        model_engine.previous_batch_indices_cuda = torch.zeros(
            max_num_tokens, dtype=torch.int32, device='cuda')

        result, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata)

        # Verify expected keys are present.
        self.assertIsNotNone(result)
        self.assertIn('input_ids', result)
        self.assertIn('position_ids', result)
        self.assertIn('attn_metadata', result)

        # Also, verify that position_ids are properly calculated.
        position_ids = result['position_ids']
        self.assertIsInstance(position_ids, torch.Tensor)
        expected_positions = [40, 30]
        actual_positions = position_ids.squeeze(0).cpu().tolist()[:2]
        self.assertEqual(
            actual_positions, expected_positions,
            f"Position IDs should reflect CP allgather results. Expected: {expected_positions}, Got: {actual_positions}"
        )

        # Verify attention metadata is properly configured.
        self.assertEqual(attn_metadata.request_ids, [1, 2])
        self.assertEqual(attn_metadata.prompt_lens, [20, 15])
        self.assertEqual(attn_metadata.num_contexts, 0)

        # Verify KV cache parameters
        self.assertIsNotNone(attn_metadata.kv_cache_params)
        self.assertTrue(attn_metadata.kv_cache_params.use_cache)

        # Verify sequence lengths are correct.
        expected_seq_lens = [1, 1]
        if hasattr(attn_metadata,
                   'seq_lens') and attn_metadata.seq_lens is not None:
            actual_seq_lens = attn_metadata.seq_lens.cpu().tolist()
            self.assertEqual(actual_seq_lens, expected_seq_lens)

    def test_prepare_tp_inputs_with_partial_mrope_segments(self) -> None:
        """Test generation-only MRoPE assembly with a real multimodal span and a dummy padded request."""
        llm_args = TorchLlmArgs(model="dummy")
        model_engine = DummyModelEngine(llm_args, dtype=torch.half)
        model_engine.model.model_config.pretrained_config.rope_scaling = {
            "type": "mrope"
        }

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=32)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=16,
            head_dim=16,
            tokens_per_block=1,
            max_seq_len=32,
            max_batch_size=4,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.HALF,
        )
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=32,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        model_engine.max_num_tokens = 32
        model_engine.input_ids_cuda = torch.zeros(32,
                                                  dtype=torch.int32,
                                                  device='cuda')
        model_engine.position_ids_cuda = torch.zeros(32,
                                                     dtype=torch.int32,
                                                     device='cuda')
        model_engine.mrope_position_ids_cuda = torch.zeros((3, 1, 32),
                                                           dtype=torch.int32,
                                                           device='cuda')
        model_engine.previous_batch_indices_cuda = torch.zeros(
            32, dtype=torch.int32, device='cuda')

        multimodal_request = _create_request(4, 1)
        multimodal_request.py_prompt_len = 4
        multimodal_request.py_batch_idx = None
        multimodal_request.py_seq_slot = 0
        multimodal_request.sampling_config.beam_width = 1
        multimodal_request.py_multimodal_data = {
            "mrope_config": {
                "mrope_position_deltas": torch.tensor([[10]], dtype=torch.int32)
            },
            "multimodal_embedding": torch.ones((1, 1), dtype=torch.float16),
        }

        dummy_request = _create_request(6, 2)
        dummy_request.py_prompt_len = 6
        dummy_request.py_batch_idx = None
        dummy_request.py_seq_slot = 1
        dummy_request.sampling_config.beam_width = 1
        dummy_request.py_multimodal_data = {}
        dummy_request.is_cuda_graph_dummy = True

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [
            multimodal_request, dummy_request
        ]

        result, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata)

        position_ids = result["position_ids"]
        self.assertEqual(tuple(position_ids.shape), (3, 1, 2))
        expected = torch.tensor([[[13, 5]], [[13, 5]], [[13, 5]]],
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(position_ids, expected, atol=0, rtol=0)
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(),
                         [0])
        # Read slots are dense w.r.t. the generation batch: the padded dummy
        # has no MRoPE metadata, so it resolves to the reserved zero slot
        # (max_num_tokens * pp_size) rather than being dropped, which would
        # shift every later request onto another request's delta.
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [0, 32])
        self.assertNotIn("multimodal_embedding",
                         multimodal_request.py_multimodal_data)
        kv_cache_manager.shutdown()

    def _setup_mrope_engine(self, max_num_tokens: int = 32):
        """Build a DummyModelEngine that takes the MRoPE path, plus its KV cache
        manager and attention metadata."""
        llm_args = TorchLlmArgs(model="dummy")
        model_engine = DummyModelEngine(llm_args, dtype=torch.half)
        model_engine.model.model_config.pretrained_config.rope_scaling = {
            "type": "mrope"
        }

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=max_num_tokens)
        kv_cache_manager = KVCacheManager(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=16,
            head_dim=16,
            tokens_per_block=1,
            max_seq_len=max_num_tokens,
            max_batch_size=4,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.HALF,
        )
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=max_num_tokens,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        model_engine.max_num_tokens = max_num_tokens
        model_engine.input_ids_cuda = torch.zeros(max_num_tokens,
                                                  dtype=torch.int32,
                                                  device='cuda')
        model_engine.position_ids_cuda = torch.zeros(max_num_tokens,
                                                     dtype=torch.int32,
                                                     device='cuda')
        model_engine.mrope_position_ids_cuda = torch.zeros(
            (3, 1, max_num_tokens), dtype=torch.int32, device='cuda')
        model_engine.previous_batch_indices_cuda = torch.zeros(
            max_num_tokens, dtype=torch.int32, device='cuda')
        return model_engine, kv_cache_manager, attn_metadata

    @staticmethod
    def _make_mrope_gen_request(num_tokens: int, req_id: int, seq_slot: int,
                                delta):
        """Generation request carrying an MRoPE delta when `delta` is not None,
        and no multimodal data at all otherwise (a text-only prompt)."""
        request = _create_request(num_tokens, req_id)
        request.py_prompt_len = num_tokens
        request.py_batch_idx = None
        request.py_seq_slot = seq_slot
        request.sampling_config.beam_width = 1
        if delta is None:
            request.py_multimodal_data = {}
        else:
            request.py_multimodal_data = {
                "mrope_config": {
                    "mrope_position_deltas":
                    torch.tensor([[delta]], dtype=torch.int32)
                },
            }
        return request

    def test_prepare_tp_inputs_mixed_text_only_keeps_mrope_deltas_dense(
            self) -> None:
        """A text-only request between two multimodal ones must not compact the
        MRoPE delta read slots.

        The attention kernel indexes `mrope_position_deltas` by generation batch
        index, so a list that skips the text-only request would hand request 2's
        delta to the text-only request and read out of bounds for request 2.
        """
        model_engine, kv_cache_manager, attn_metadata = self._setup_mrope_engine(
        )

        # (num_tokens, seq_slot, delta); the middle request is text-only.
        requests = [
            self._make_mrope_gen_request(4, 1, 0, 10),
            self._make_mrope_gen_request(5, 2, 1, None),
            self._make_mrope_gen_request(6, 3, 2, 20),
        ]

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = requests

        result, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata)

        # One entry per generation request, in batch order. Slot 32 is the
        # reserved zero slot (max_num_tokens * pp_size) standing in for the
        # text-only request's zero delta.
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [0, 32, 2])
        # Only the two multimodal requests seed the seq-slot delta cache.
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(),
                         [0, 2])

        # past_seen_token_num is num_tokens - 1, offset by the request's delta;
        # the text-only request keeps the plain scalar position on all 3 axes.
        position_ids = result["position_ids"]
        self.assertEqual(tuple(position_ids.shape), (3, 1, 3))
        expected = torch.tensor([[[13, 4, 25]]] * 3,
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(position_ids, expected, atol=0, rtol=0)
        kv_cache_manager.shutdown()

    def test_prepare_tp_inputs_all_text_only_drops_mrope_deltas(self) -> None:
        """A generation batch with no MRoPE metadata at all emits no delta
        tensors, so the steady-state generation fast path stays reachable."""
        model_engine, kv_cache_manager, attn_metadata = self._setup_mrope_engine(
        )

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [
            self._make_mrope_gen_request(4, 1, 0, None),
            self._make_mrope_gen_request(6, 2, 1, None),
        ]

        result, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata)

        # An all-zero delta vector is identical to passing no deltas at all.
        self.assertNotIn("mrope_delta_read_seq_slots", result)
        self.assertNotIn("mrope_delta_write_seq_slots", result)

        # MRoPE models keep the (3,1,N) layout even with no mrope work.
        position_ids = result["position_ids"]
        self.assertEqual(tuple(position_ids.shape), (3, 1, 2))
        expected = torch.tensor([[[3, 5]]] * 3,
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(position_ids, expected, atol=0, rtol=0)
        kv_cache_manager.shutdown()

    def test_promoted_mrope_context_uses_decode_state_contract(self) -> None:
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        model_engine.model.model_config.pretrained_config.rope_scaling = {
            "type": "mrope"
        }
        model_engine.mrope_position_ids_cuda = torch.zeros(
            (3, 1, model_engine.max_num_tokens),
            dtype=torch.int32,
            device="cuda",
        )
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=32,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        context = _create_request_with_tokens([11, 22, 33, 44], 1)
        context.context_current_position = 3
        context.context_chunk_size = 1
        context.py_seq_slot = 0
        context.py_batch_idx = 3
        mrope_delta = torch.tensor([[10]], dtype=torch.int32)
        context.py_mrope_position_delta = mrope_delta
        context.py_mrope_delta_cache_slot = context.py_seq_slot
        context.py_multimodal_data = {
            "mrope_config": {
                "mrope_position_deltas": mrope_delta,
            },
            "multimodal_embedding": torch.ones((1, 1), dtype=torch.float16),
        }
        graph_batch = ScheduledRequests()
        graph_batch.generation_requests = [context]

        inputs, _ = model_engine._prepare_tp_inputs(
            scheduled_requests=graph_batch,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            resource_manager=resource_manager,
            promoted_context_request_ids=frozenset({context.py_request_id}),
        )

        self.assertEqual(inputs["input_ids"][:1].cpu().tolist(), [44])
        expected_positions = torch.full((3, 1, 1),
                                        13,
                                        dtype=torch.int32,
                                        device="cuda")
        torch.testing.assert_close(inputs["position_ids"],
                                   expected_positions,
                                   atol=0,
                                   rtol=0)
        self.assertEqual(
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [3])
        self.assertEqual(inputs["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [0])
        self.assertNotIn("mrope_delta_write_seq_slots", inputs)
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(model_engine.previous_request_ids, [])
        kv_cache_manager.shutdown()

    def test_kv_cache_manager_with_execution_stream(self) -> None:
        """Test that KVCacheManager uses the provided execution_stream.
        """
        # Create a dedicated execution stream
        execution_stream = torch.cuda.Stream()

        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            execution_stream=execution_stream)

        # Verify the KVCacheManager uses the provided execution stream
        self.assertEqual(
            kv_cache_manager._stream.cuda_stream, execution_stream.cuda_stream,
            "KVCacheManager should use the provided execution_stream")

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        kv_cache_manager.prepare_resources(batch)
        with torch.cuda.stream(execution_stream):
            model_engine.forward(batch, resource_manager)

        # Verify the stream is still the same after forward pass
        self.assertEqual(
            kv_cache_manager._stream.cuda_stream, execution_stream.cuda_stream,
            "KVCacheManager should still use the provided execution_stream after forward"
        )

        kv_cache_manager.shutdown()

    def test_cuda_graph_replay_observes_execution_stream_dependency(
            self) -> None:
        """A graph replay on the KV manager stream waits for restored KV data."""
        execution_stream = torch.cuda.Stream()
        transfer_stream = torch.cuda.Stream()
        _, kv_cache_manager = create_model_engine_and_kvcache(
            execution_stream=execution_stream)

        source = torch.zeros(1, dtype=torch.int32, device="cuda")
        observed = torch.zeros_like(source)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=execution_stream):
            observed.copy_(source)

        ready = torch.cuda.Event()
        with torch.cuda.stream(transfer_stream):
            # Model the async host-to-device restore completed by the local
            # offload manager. Recording the event after the write is the same
            # dependency shape that refreshBlocks/resume installs.
            source.fill_(7)
            ready.record()

        manager_stream = torch.cuda.ExternalStream(
            kv_cache_manager._stream.cuda_stream)
        with torch.cuda.stream(manager_stream):
            manager_stream.wait_event(ready)
            graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(observed.item(), 7)
        kv_cache_manager.shutdown()


if __name__ == "__main__":
    unittest.main()
