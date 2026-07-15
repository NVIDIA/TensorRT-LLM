# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import \
    KvCacheConnectorWorker
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import (
    PyTorchModelEngine, _make_single_token_context_graph_batch)
from tensorrt_llm.llmapi.llm_args import (SeqLenAwareSparseAttentionConfig,
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
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.llmapi import CudaGraphConfig, SamplingParams
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


class DummyModelEngine(PyTorchModelEngine):

    def __init__(self, llm_args: TorchLlmArgs, dtype: torch.dtype) -> None:
        self.dtype = dtype
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          rank=tensorrt_llm.mpi_rank())
        model = DummyModel(self.dtype)
        super().__init__(model_path="dummy",
                         mapping=mapping,
                         model=model,
                         llm_args=llm_args)


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
        is_dummy=False,
        max_beam_num_tokens=prompt_len,
        state="context",
        py_llm_request_type="context_and_generation",
    )


def _make_forward_only_engine(
    graph_key: tuple[int, int, bool, bool, bool] | None,
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
    engine.mapping = SimpleNamespace(cp_size=1)
    engine.runtime_draft_len = 0
    engine.attn_backend = None
    engine.model_is_wrapped = False
    engine.original_max_draft_len = 0
    engine.original_max_total_draft_tokens = 0
    engine._spec_dec_max_total_draft_tokens = 0
    engine.get_runtime_tokens_per_gen_step = Mock(return_value=1)
    engine.iter_states = {}
    engine.forward_pass_callable = None
    engine._is_encoder_decoder_model = Mock(return_value=False)
    engine._get_draft_kv_cache_manager = Mock(return_value=None)

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
    engine._prepare_inputs = Mock(return_value=({"prepared": True}, None))
    outputs = {"logits": object()}
    engine._forward_step = Mock(return_value=outputs)
    engine._execute_logit_post_processors = Mock()

    runner = Mock()
    runner.enabled = runner_enabled
    runner.pad_batch.side_effect = lambda batch, *_args: nullcontext(batch)
    runner.maybe_get_cuda_graph.return_value = ((graph_attn_metadata, None,
                                                 graph_key)
                                                if graph_key is not None else
                                                (None, None, None))
    runner.get_graph_pool.return_value = None
    runner.needs_capture.return_value = False
    runner.replay.return_value = outputs
    engine.cuda_graph_runner = runner

    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = object()
    return engine, runner, resource_manager, semantic_attn_metadata, outputs


def create_model_engine_and_kvcache(llm_args: TorchLlmArgs = None,
                                    execution_stream: torch.cuda.Stream = None):
    tokens_per_block = 1
    max_tokens = 258  # Atleast 1 more than the max seq len
    num_layers = 1
    batch_size = 13

    llm_args = llm_args if llm_args else TorchLlmArgs(
        model="dummy",
        max_batch_size=batch_size,
        max_num_tokens=max_tokens,
        cuda_graph_config=CudaGraphConfig(
            enable_padding=True, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]))
    test_batches = (5, 13)
    for batch_size in test_batches:
        assert batch_size not in llm_args.cuda_graph_config.batch_sizes

    assert (8 in llm_args.cuda_graph_config.batch_sizes
            and 16 in llm_args.cuda_graph_config.batch_sizes)

    model_engine = DummyModelEngine(llm_args, torch.half)

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
            ("context_logits", "py_return_context_logits", True),
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
        self.assertEqual(key, (1, 0, False, True, True))

    def test_graph_lookup_forwards_promoted_context_ids(self) -> None:
        runner = Mock()
        runner.enabled = True
        runner.config = SimpleNamespace(
            enable_attention_dp=False,
            use_mrope=False,
        )
        key = (1, 0, False, True, True)
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
                                                     promoted_ids)
        self.assertEqual(result,
                         (graph_attn_metadata, graph_spec_metadata, key))

    def test_forward_commits_candidate_only_on_graph_hit(self) -> None:
        key = (2, 0, False, False, True)
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
        runner.replay.assert_called_once_with(key, {"prepared": True})
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

    def test_generation_only_forward_does_not_call_new_selector(self) -> None:
        key = (1, 0, False, False, True)
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
        self.assertIs(engine._prepare_inputs.call_args.args[0], batch)
        self.assertEqual(engine._prepare_inputs.call_args.args[-1], frozenset())

    def test_global_incompatibilities_bypass_candidate_selection(self) -> None:
        cases = (
            "graphs_disabled",
            "speculative",
            "guided",
            "beam",
            "context_logits",
            "encoder_decoder",
            "encode_only",
            "mm_encoder_only",
            "context_parallel",
        )
        for case in cases:
            with self.subTest(case=case):
                engine, runner, resource_manager, _, _ = \
                    _make_forward_only_engine(None)
                gather_context_logits = False
                if case == "graphs_disabled":
                    runner.enabled = False
                elif case == "speculative":
                    engine.enable_spec_decode = True
                elif case == "guided":
                    engine.guided_decoder = object()
                elif case == "beam":
                    engine.max_beam_width = 2
                elif case == "context_logits":
                    gather_context_logits = True
                elif case == "encoder_decoder":
                    engine._is_encoder_decoder_model.return_value = True
                elif case == "encode_only":
                    engine._is_encode_only = True
                elif case == "mm_encoder_only":
                    engine.llm_args.mm_encoder_only = True
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

        generation = _create_request_with_tokens([50, 51, 52, 53, 54], 2)
        generation.py_seq_slot = 1
        generation.py_batch_idx = 1

        graph_batch = ScheduledRequests()
        graph_batch.generation_requests = [context, generation]
        new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")
        new_tokens[0, 0, 0] = 999
        new_tokens[0, 1, 0] = 777
        overlap_state = SimpleNamespace(new_tokens=new_tokens)

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
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [3, 5])
        self.assertEqual(
            model_engine.previous_batch_indices_cuda[:1].cpu().tolist(), [1])
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(model_engine.previous_request_ids,
                         [generation.py_request_id])
        kv_cache_manager.shutdown()

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
        # Make sure we don't leak any blocks.
        self.assertEqual(num_free_before,
                         kv_cache_manager.get_num_free_blocks())

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
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [0])
        self.assertNotIn("multimodal_embedding",
                         multimodal_request.py_multimodal_data)
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
