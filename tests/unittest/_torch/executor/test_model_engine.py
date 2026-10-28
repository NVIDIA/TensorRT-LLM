# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch
from _torch.executor.engine_test_utils import (
    DummyModel,
    DummyModelEngine,
    _AttentionBackend,
    _clone_tensors,
    _create_request,
    _create_request_with_tokens,
    _model_engine_with_runner,
    create_model_engine_and_kvcache,
)

import tensorrt_llm._torch.pyexecutor.model_engine as model_engine_module
from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.multimodal import is_multimodal
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import (
    DecoderInputBuffers,
    DecoderInputPreparer,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import PrefillCompiledModel
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder import EncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ModelRunner, ScheduledInputs
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import torch_compiling
from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SADecodingConfig
from tensorrt_llm.llmapi.llm_args import (
    EncodeCudaGraphConfig,
    MultimodalConfig,
    PrefillCudaGraphBackend,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping


class DummyLegacyMultimodalIndexModel(MultimodalModelMixin, torch.nn.Module):
    class Config:
        vocab_size = 100

    config = Config()

    @property
    def mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([90, 91], dtype=torch.int32)


def _prepare_kv_cache_resources(manager: KVCacheManagerV2, batch: ScheduledRequests) -> None:
    # These model-engine tests bypass the scheduler, which owns V2 allocation.
    for request in batch.context_requests:
        assert manager.prepare_context(request)
        assert manager.resize_context(
            request, request.prompt_len - request.context_current_position
        )
    for request in batch.generation_requests:
        assert manager.try_allocate_generation(request)
    manager.prepare_resources(batch)


def _install_graph_runner_stub(runner: DecoderRunner, key: KeyType | None = None) -> Mock:
    """Replace the runner's CUDA graph runner with one that never pads.

    Every lookup hits `stub.key`, or misses while it is None. A hit replays by
    recording a copy of the prepared inputs.
    """
    stub = Mock()
    stub.enabled = True
    stub.is_warmup_only = False
    stub.key = key
    stub.pad_batch.side_effect = lambda batch, *_args: nullcontext(batch)
    stub.get_graph_pool.return_value = None
    stub.needs_capture.return_value = False
    stub.maybe_get_cuda_graph.side_effect = lambda batch, **kwargs: (
        (kwargs["attn_metadata"], kwargs["spec_metadata"], stub.key)
        if stub.key is not None
        else (None, None, None)
    )
    stub.replayed_inputs = []

    def replay(key, inputs, position_id_offsets=None):
        stub.replayed_inputs.append(_clone_tensors(inputs))
        return {"logits": torch.zeros((inputs["input_ids"].shape[0], 10), device="cuda")}

    stub.replay.side_effect = replay
    runner._graphs.cuda_graph_runner = stub
    return stub


@pytest.mark.cpu_only
def test_cleanup_releases_runner_model_before_release_gc() -> None:
    class Model:
        pass

    engine = object.__new__(ModelEngine)
    model = Model()
    model_ref = weakref.ref(model)
    engine._cleanup_done = False
    engine.model_loader = None
    engine._runner = SimpleNamespace(_model=model)
    engine._model_caller = ModelCaller(model)
    engine._mm_item_scheduler = None
    engine._lora_model = model
    engine._lora_owner = None
    engine.model = model
    engine._release_cuda_graphs = Mock()
    engine.input_processor = None
    del model

    def assert_model_released() -> None:
        assert model_ref() is None

    with patch(
        "tensorrt_llm._torch.pyexecutor.model_engine.release_gc", side_effect=assert_model_released
    ) as release_gc:
        engine.cleanup()

    release_gc.assert_called_once_with()


class GraphOwnerTeardownTestCase(unittest.TestCase):
    def test_engine_graph_ownership_is_split(self) -> None:
        clear_blocks_graph = Mock()
        model = Mock()
        model.modules.return_value = [SimpleNamespace(clear_blocks_cuda_graph=clear_blocks_graph)]
        backend = Mock()
        cuda_runner = Mock()
        breakable_runner = Mock()
        decoder_runner = object.__new__(DecoderRunner)
        decoder_runner._graphs = object.__new__(DecoderCudaGraphs)
        decoder_runner._graphs.cuda_graph_runner = cuda_runner
        decoder_runner._graphs.breakable_cuda_graph_runner = breakable_runner
        engine = SimpleNamespace(
            model=model,
            _torch_compile_backend=backend,
            _runner=decoder_runner,
        )

        ModelEngine._release_cuda_graphs(engine, release_nccl_window_owners=False)

        clear_blocks_graph.assert_not_called()
        backend.clear_piecewise_cuda_graphs.assert_called_once_with(
            release_nccl_window_owners=False
        )
        for runner in (cuda_runner, breakable_runner):
            runner.clear.assert_called_once_with(release_nccl_window_owners=False)

        ModelEngine._release_model_owned_cuda_graphs(engine, release_nccl_window_owners=False)

        clear_blocks_graph.assert_called_once_with(release_nccl_window_owners=False)

    def test_executor_release_clears_model_owned_graphs_first(self) -> None:
        for release_model_owned_graphs in (True, False):
            with self.subTest(release_model_owned_graphs=release_model_owned_graphs):
                engine = Mock()

                ModelEngine.release_executor_resources(
                    engine, release_model_owned_graphs=release_model_owned_graphs
                )

                expected = [call._release_cuda_graphs()]
                if release_model_owned_graphs:
                    expected.insert(0, call._release_model_owned_cuda_graphs())
                self.assertEqual(engine.method_calls, expected)


class DecoderForwardTestCase(unittest.TestCase):
    def _make_engine(
        self,
        model: DummyModel | None = None,
    ) -> tuple[ModelEngine, KVCacheManagerV2, ResourceManager, Mock]:
        (engine, kv_cache_manager) = self.enterContext(create_model_engine_and_kvcache(model=model))
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        graph_runner = _install_graph_runner_stub(engine._runner)
        return engine, kv_cache_manager, resource_manager, graph_runner

    @staticmethod
    def _forward(
        engine: ModelEngine,
        kv_cache_manager: KVCacheManagerV2,
        resource_manager: ResourceManager,
        batch: ScheduledRequests,
        new_tensors_device=None,
    ):
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        return engine.forward(batch, resource_manager, new_tensors_device)

    @staticmethod
    def _request(tokens: list[int], req_id: int, seq_slot: int) -> LlmRequest:
        request = _create_request_with_tokens(tokens, req_id)
        request.py_seq_slot = seq_slot
        return request

    def _prefill(self, engine, kv_cache_manager, resource_manager, *requests: LlmRequest) -> None:
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = list(requests)
        self._forward(engine, kv_cache_manager, resource_manager, batch)

    def _prefill_all_but_last_token(
        self, engine, kv_cache_manager, resource_manager, request: LlmRequest
    ) -> None:
        """Leave `request` with a final one-token context chunk."""
        batch = ScheduledRequests()
        batch.context_requests_chunking = [request]
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        # Like the scheduler, narrow the chunk after V2 spans it to the prompt end.
        request.context_chunk_size = request.py_prompt_len - 1
        engine.forward(batch, resource_manager)
        request.move_to_next_context_chunk()
        request.context_chunk_size = 1

    def _final_context_and_generation(self, engine, kv_cache_manager, resource_manager):
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        generation = self._request([50, 51, 52, 53, 54], 2, seq_slot=1)
        self._prefill_all_but_last_token(engine, kv_cache_manager, resource_manager, context)
        self._prefill(engine, kv_cache_manager, resource_manager, generation)
        generation.add_new_token(60, 0)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]
        return context, generation, batch

    def test_forward_commits_candidate_only_on_graph_hit(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = self._make_engine()
        context, generation, batch = self._final_context_and_generation(
            engine, kv_cache_manager, resource_manager
        )
        num_model_calls = len(engine.model.recorded_kwargs)
        graph_runner.key = KeyType(batch_size=2, draft_len=0)

        outputs = self._forward(engine, kv_cache_manager, resource_manager, batch)

        lookup = graph_runner.maybe_get_cuda_graph.call_args
        self.assertIsNot(lookup.args[0], batch)
        self.assertEqual(lookup.args[0].generation_requests, [context, generation])
        self.assertEqual(
            lookup.kwargs["promoted_context_request_ids"], frozenset({context.py_request_id})
        )
        graph_runner.replay.assert_called_once()
        self.assertEqual(graph_runner.replay.call_args.args[0], graph_runner.key)
        replayed = graph_runner.replayed_inputs[0]
        self.assertEqual(replayed["input_ids"].tolist(), [44, 60])
        self.assertEqual(replayed["position_ids"].flatten().tolist(), [3, 5])
        self.assertEqual(len(engine.model.recorded_kwargs), num_model_calls)
        self.assertEqual(set(outputs), {"logits", "runtime_draft_len"})
        self.assertEqual(engine.attn_metadata.num_contexts, 0)
        self.assertEqual(engine.iter_states["num_ctx_requests"], 1)
        self.assertEqual(engine.iter_states["num_ctx_tokens"], 1)
        self.assertEqual(engine.iter_states["num_generation_tokens"], 1)

    def test_forward_graph_miss_uses_semantic_eager_batch(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = self._make_engine()
        context, generation, batch = self._final_context_and_generation(
            engine, kv_cache_manager, resource_manager
        )

        self._forward(engine, kv_cache_manager, resource_manager, batch)

        lookup = graph_runner.maybe_get_cuda_graph.call_args
        self.assertEqual(lookup.args[0].generation_requests, [context, generation])
        graph_runner.replay.assert_not_called()
        model_kwargs = engine.model.recorded_kwargs[-1]
        self.assertEqual(model_kwargs["input_ids"].tolist(), [44, 60])
        self.assertEqual(model_kwargs["position_ids"].flatten().tolist(), [3, 5])
        self.assertEqual(engine.attn_metadata.num_contexts, 1)

    def test_forward_allows_guided_context_logits_on_graph_hit(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = self._make_engine()
        engine._runner.guided_decoder = Mock()
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        self._prefill_all_but_last_token(engine, kv_cache_manager, resource_manager, context)
        context.py_return_context_logits = True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        graph_runner.key = KeyType(batch_size=1, draft_len=0)

        self._forward(engine, kv_cache_manager, resource_manager, batch)

        lookup = graph_runner.maybe_get_cuda_graph.call_args
        self.assertEqual(lookup.args[0].generation_requests, [context])
        self.assertEqual(
            lookup.kwargs["promoted_context_request_ids"], frozenset({context.py_request_id})
        )
        graph_runner.replay.assert_called_once()
        self.assertEqual(graph_runner.replayed_inputs[0]["input_ids"].tolist(), [44])

    def test_promoted_context_uses_prompt_token_during_overlap(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = self._make_engine()
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        generation = self._request([50, 51, 52, 53, 54], 2, seq_slot=1)
        self._prefill_all_but_last_token(engine, kv_cache_manager, resource_manager, context)
        self._prefill(engine, kv_cache_manager, resource_manager, generation)
        # The chunked prefill stands in for a reused prefix, so the promoted
        # row is the context's first forward and latches its cached tokens.
        context._cached_tokens_set = False
        # A promoted context row must ignore any stale overlap slot.
        context.py_batch_idx = 3
        context.py_num_compressed_tokens = 1
        generation.py_batch_idx = 1
        new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")
        new_tokens[0, 0, 0] = 999
        new_tokens[0, 1, 0] = 777
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]
        graph_runner.key = KeyType(batch_size=2, draft_len=0)

        self._forward(
            engine,
            kv_cache_manager,
            resource_manager,
            batch,
            SimpleNamespace(new_tokens=new_tokens),
        )

        replayed = graph_runner.replayed_inputs[0]
        self.assertEqual(replayed["input_ids"].tolist(), [44, 777])
        self.assertEqual(replayed["position_ids"].flatten().tolist(), [3, 5])
        attn_metadata = engine.attn_metadata
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [2, 5])
        self.assertEqual(context.cached_tokens, 3)

    def test_steady_overlap_generation_advances_positions(self) -> None:
        engine, kv_cache_manager, resource_manager, _ = self._make_engine()
        requests = [
            self._request([10, 11, 12, 13], 1, seq_slot=0),
            self._request([20, 21, 22, 23, 24, 25], 2, seq_slot=1),
        ]
        self._prefill(engine, kv_cache_manager, resource_manager, *requests)
        for batch_idx, request in enumerate(requests):
            request.py_batch_idx = batch_idx
        batch = ScheduledRequests()
        batch.generation_requests = requests
        steps = ((100, 200), (101, 201), (102, 202))

        for step, tokens in enumerate(steps):
            new_tokens = torch.tensor(tokens, dtype=torch.int32, device="cuda").view(1, 2, 1)
            self._forward(
                engine,
                kv_cache_manager,
                resource_manager,
                batch,
                SimpleNamespace(new_tokens=new_tokens),
            )
            model_kwargs = engine.model.recorded_kwargs[-1]
            with self.subTest(step=step):
                self.assertEqual(model_kwargs["input_ids"].tolist(), list(tokens))
                self.assertEqual(
                    model_kwargs["position_ids"].flatten().tolist(), [4 + step, 6 + step]
                )
            # The sampler appends the previous step's token after the next
            # step has been scheduled.
            for request, token in zip(requests, tokens):
                request.add_new_token(token, 0)

    def _make_mrope_engine(self):
        return self._make_engine(DummyModel(torch.half, rope_scaling={"type": "mrope"}))

    def _mrope_generation(
        self,
        engine,
        kv_cache_manager,
        resource_manager,
        num_tokens: int,
        req_id: int,
        seq_slot: int,
        delta: int | None,
    ) -> LlmRequest:
        """A generation request at position `num_tokens - 1`.

        It carries an MRoPE delta when `delta` is not None, and no
        multimodal data at all otherwise (a text-only prompt).
        """
        request = self._request(list(range(1, num_tokens)), req_id, seq_slot)
        self._prefill(engine, kv_cache_manager, resource_manager, request)
        request.add_new_token(num_tokens, 0)
        if delta is None:
            request.py_multimodal_data = {}
        else:
            request.py_multimodal_data = {
                "mrope_config": {
                    "mrope_position_deltas": torch.tensor([[delta]], dtype=torch.int32)
                },
            }
        return request

    def _forward_generation(
        self, engine, kv_cache_manager, resource_manager, requests: list[LlmRequest]
    ) -> dict:
        batch = ScheduledRequests()
        batch.generation_requests = requests
        self._forward(engine, kv_cache_manager, resource_manager, batch)
        return engine.model.recorded_kwargs[-1]

    def test_mrope_generation_with_partial_segments(self) -> None:
        """A real multimodal span next to a padding dummy without MRoPE data."""
        engine, kv_cache_manager, resource_manager, _ = self._make_mrope_engine()
        multimodal = self._mrope_generation(engine, kv_cache_manager, resource_manager, 4, 1, 0, 10)
        multimodal.py_multimodal_data["multimodal_embedding"] = torch.ones(
            (1, 1), dtype=torch.float16
        )
        # A generation-only admission has no batch slot yet, which is when
        # the context-phase payload is stripped.
        multimodal.py_batch_idx = None
        dummy = self._mrope_generation(engine, kv_cache_manager, resource_manager, 6, 2, 1, None)
        dummy.is_cuda_graph_dummy = True

        model_kwargs = self._forward_generation(
            engine, kv_cache_manager, resource_manager, [multimodal, dummy]
        )

        expected = torch.tensor([[[13, 5]]] * 3, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(model_kwargs["position_ids"], expected, atol=0, rtol=0)
        self.assertEqual(model_kwargs["mrope_delta_write_seq_slots"].cpu().tolist(), [0])
        # Read slots are dense w.r.t. the generation batch: the padded dummy
        # has no MRoPE metadata, so it resolves to the reserved zero slot
        # (max_num_tokens * pp_size) rather than being dropped, which would
        # shift every later request onto another request's delta.
        self.assertEqual(
            model_kwargs["mrope_delta_read_seq_slots"].cpu().tolist(), [0, engine.max_num_tokens]
        )
        self.assertNotIn("multimodal_embedding", multimodal.py_multimodal_data)

    def test_mrope_mixed_text_only_keeps_deltas_dense(self) -> None:
        """A text-only request between two multimodal ones must not compact the
        MRoPE delta read slots.

        The attention kernel indexes `mrope_position_deltas` by generation batch
        index, so a list that skips the text-only request would hand request 2's
        delta to the text-only request and read out of bounds for request 2.
        """
        engine, kv_cache_manager, resource_manager, _ = self._make_mrope_engine()
        # (num_tokens, req_id, seq_slot, delta); the middle one is text-only.
        requests = [
            self._mrope_generation(engine, kv_cache_manager, resource_manager, *args)
            for args in ((4, 1, 0, 10), (5, 2, 1, None), (6, 3, 2, 20))
        ]

        model_kwargs = self._forward_generation(
            engine, kv_cache_manager, resource_manager, requests
        )

        # One entry per generation request, in batch order; the reserved zero
        # slot stands in for the text-only request's zero delta.
        self.assertEqual(
            model_kwargs["mrope_delta_read_seq_slots"].cpu().tolist(), [0, engine.max_num_tokens, 2]
        )
        # Only the two multimodal requests seed the seq-slot delta cache.
        self.assertEqual(model_kwargs["mrope_delta_write_seq_slots"].cpu().tolist(), [0, 2])
        # The position is num_tokens - 1, offset by the request's delta; the
        # text-only request keeps the plain scalar position on all 3 axes.
        expected = torch.tensor([[[13, 4, 25]]] * 3, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(model_kwargs["position_ids"], expected, atol=0, rtol=0)

    def test_mrope_all_text_only_drops_deltas(self) -> None:
        """A generation batch with no MRoPE metadata at all emits no delta
        tensors, so the steady-state generation fast path stays reachable."""
        engine, kv_cache_manager, resource_manager, _ = self._make_mrope_engine()
        requests = [
            self._mrope_generation(engine, kv_cache_manager, resource_manager, 4, 1, 0, None),
            self._mrope_generation(engine, kv_cache_manager, resource_manager, 6, 2, 1, None),
        ]

        model_kwargs = self._forward_generation(
            engine, kv_cache_manager, resource_manager, requests
        )

        # An all-zero delta vector is identical to passing no deltas at all.
        self.assertNotIn("mrope_delta_read_seq_slots", model_kwargs)
        self.assertNotIn("mrope_delta_write_seq_slots", model_kwargs)
        # MRoPE models keep the (3,1,N) layout even with no mrope work.
        expected = torch.tensor([[[3, 5]]] * 3, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(model_kwargs["position_ids"], expected, atol=0, rtol=0)

    def test_promoted_mrope_context_uses_decode_state_contract(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = self._make_mrope_engine()
        engine.model.config.vocab_size = 100
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        self._prefill_all_but_last_token(engine, kv_cache_manager, resource_manager, context)
        mrope_delta = torch.tensor([[10]], dtype=torch.int32)
        context.py_mrope_position_delta = mrope_delta
        context.py_mrope_delta_cache_slot = context.py_seq_slot
        context.py_multimodal_data = {
            "mrope_config": {
                "mrope_position_deltas": mrope_delta,
            },
            "multimodal_embedding": torch.ones((1, 1), dtype=torch.float16),
        }
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        graph_runner.key = KeyType(batch_size=1, draft_len=0)

        self._forward(engine, kv_cache_manager, resource_manager, batch)

        self.assertEqual(
            graph_runner.maybe_get_cuda_graph.call_args.kwargs["promoted_context_request_ids"],
            frozenset({context.py_request_id}),
        )
        replayed = graph_runner.replayed_inputs[0]
        self.assertEqual(replayed["input_ids"].tolist(), [44])
        expected_positions = torch.full((3, 1, 1), 13, dtype=torch.int32, device="cuda")
        torch.testing.assert_close(replayed["position_ids"], expected_positions, atol=0, rtol=0)
        self.assertEqual(replayed["mrope_delta_read_seq_slots"].cpu().tolist(), [0])
        self.assertNotIn("mrope_delta_write_seq_slots", replayed)
        attn_metadata = engine.attn_metadata
        self.assertEqual(attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [3])
        self.assertEqual(attn_metadata.num_contexts, 0)


class ModelEngineTestCase(unittest.TestCase):
    def test_breakable_rejects_multimodal_models(self) -> None:
        engine = object.__new__(ModelEngine)
        engine.model = DummyLegacyMultimodalIndexModel()
        engine.input_processor = None
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE, disable_mm_encoder=False
        )

        self.assertTrue(is_multimodal(engine.model, engine.input_processor))
        with self.assertRaisesRegex(ValueError, "multimodal models"):
            engine._validate_breakable_cuda_graph_compatibility()

    def test_breakable_allows_text_decoder_with_multimodal_processor(self) -> None:
        engine = object.__new__(ModelEngine)
        engine.model = Mock(spec=DecoderModelForCausalLM)
        engine.input_processor = Mock(spec=BaseMultimodalInputProcessor)
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE, disable_mm_encoder=False
        )

        self.assertTrue(is_multimodal(engine.model, engine.input_processor))
        engine._validate_breakable_cuda_graph_compatibility()

    def test_breakable_allows_multimodal_wrapper_in_text_only_mode(self) -> None:
        engine = object.__new__(ModelEngine)
        engine.model = DummyLegacyMultimodalIndexModel()
        engine.model.llm = Mock(spec=DecoderModelForCausalLM)
        engine.model.mm_encoder = None
        engine.input_processor = Mock(spec=BaseMultimodalInputProcessor)
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE, disable_mm_encoder=True
        )

        self.assertTrue(is_multimodal(engine.model, engine.input_processor))
        engine._validate_breakable_cuda_graph_compatibility()

    def test_restore_after_refit_rewraps_without_recapture(self) -> None:
        """Restore re-wraps torch.compile and does nothing else.

        Refit moves no tensor, so the piecewise captures stay valid across it;
        recapturing them costs a full re-capture per refit and leaves PWCG cold
        in between. This pins the cheap path -- reintroducing the recapture,
        the torch.compiler.reset() or the gc.collect() fails here.

        Replaces test_piecewise_refit_recapture_preserves_padding_dummy_requests,
        which asserted the refit path recaptures. That path no longer exists.
        """
        cuda_graph_runner = SimpleNamespace(
            padding_dummy_requests={0: object()},
            allow_capture=lambda: nullcontext(),
        )
        model_engine = SimpleNamespace(
            warmup_scope=nullcontext,
            _torch_compile_backend=object(),
            llm_args=SimpleNamespace(torch_compile_config=SimpleNamespace(enable_fullgraph=False)),
            cuda_graph_runner=cuda_graph_runner,
            _apply_torch_compile=Mock(),
            _capture_piecewise_cuda_graphs=Mock(),
        )

        with patch("torch.compiler.reset") as reset, patch("gc.collect") as collect:
            ModelEngine.restore_compiled_model_after_refit(model_engine, resource_manager=object())

        model_engine._apply_torch_compile.assert_called_once()
        model_engine._capture_piecewise_cuda_graphs.assert_not_called()
        reset.assert_not_called()
        # No full collection either: it ran for seconds inside the refit
        # bubble and the replaced wrapper is freed by reference count.
        collect.assert_not_called()

    def test_restore_after_refit_noop_when_torch_compile_disabled(self) -> None:
        """An eager engine must not be silently handed a compiled wrapper."""
        model_engine = SimpleNamespace(
            _torch_compile_backend=None,
            _apply_torch_compile=Mock(),
        )
        ModelEngine.restore_compiled_model_after_refit(model_engine, resource_manager=object())
        model_engine._apply_torch_compile.assert_not_called()

    def test_restore_after_refit_never_touches_the_kv_cache(self) -> None:
        """The re-wrap hook must not run the warm-up KV-cache cleanup.

        ``warmup_with_kv_cache_cleanup`` ends with
        ``check_invalid_values_in_kv_cache(fill_with_zero=True)``, which
        zero-fills every attention KV pool whether or not it found a NaN. That
        is right after a capture warm-up that ran forwards on placeholder
        inputs; restore never runs a forward, and under an in-flight weight
        update the zero-fill would wipe the KV of the requests kept across the
        refit. So the fully decorated hook must never even look up the KV cache
        manager -- with torch.compile on or off.
        """
        for compile_enabled in (True, False):
            with self.subTest(compile_enabled=compile_enabled):
                resource_manager = Mock()
                model_engine = SimpleNamespace(
                    warmup_scope=nullcontext,
                    _torch_compile_backend=Mock() if compile_enabled else None,
                    llm_args=SimpleNamespace(
                        torch_compile_config=SimpleNamespace(enable_fullgraph=False)
                    ),
                    _apply_torch_compile=Mock(),
                )
                with patch("gc.collect"):
                    ModelEngine.restore_compiled_model_after_refit(model_engine, resource_manager)
                resource_manager.get_resource_manager.assert_not_called()
                self.assertEqual(
                    model_engine._apply_torch_compile.call_count, 1 if compile_enabled else 0
                )

    def test_unwrap_for_refit_gates_on_torch_compile_not_piecewise(self) -> None:
        """Compile-on/piecewise-off must still unwrap.

        This is the original defect: the hook was gated on the piecewise flag,
        so TorchCompileConfig() -- compile on, piecewise off -- got no refit
        lifecycle and refit loaded into a wrapped model, matching nothing.
        """
        for piecewise in (True, False):
            with self.subTest(piecewise=piecewise):
                model_engine = SimpleNamespace(
                    llm_args=SimpleNamespace(
                        prefill_cuda_graph_backend=(
                            PrefillCudaGraphBackend.PIECEWISE if piecewise else None
                        )
                    ),
                    _torch_compile_backend=Mock(),
                    _remove_torch_compile=Mock(),
                )
                unwrap_core = ModelEngine.unwrap_compiled_model_for_refit
                with patch("torch.cuda.synchronize"):
                    unwrap_core(model_engine)
                model_engine._remove_torch_compile.assert_called_once()
                # The captures are kept, not cleared: refit moves no tensor, so
                # clearing them only forces PWCG cold for no benefit.
                model_engine._torch_compile_backend.clear_piecewise_cuda_graphs.assert_not_called()

        # Eager engine: the hook must be a no-op.
        eager = SimpleNamespace(
            _torch_compile_backend=None,
            _remove_torch_compile=Mock(),
        )
        ModelEngine.unwrap_compiled_model_for_refit(eager)
        eager._remove_torch_compile.assert_not_called()

    def test_refit_lifecycle_aliases_point_at_renamed_hooks(self) -> None:
        """NeMo-RL resolves these with getattr() and falls back to doing nothing.

        Dropping an alias would therefore not raise -- it would silently skip
        the unwrap and reintroduce the refit corruption.
        """
        self.assertIs(
            ModelEngine.release_piecewise_cuda_graphs_for_refit,
            ModelEngine.unwrap_compiled_model_for_refit,
        )
        self.assertIs(
            ModelEngine.recapture_piecewise_cuda_graphs_after_refit,
            ModelEngine.restore_compiled_model_after_refit,
        )

    def test_decoder_buffers_are_allocated_only_for_decoder_runner(self) -> None:
        llm_args = TorchLlmArgs(
            model="dummy",
            max_batch_size=4,
            max_num_tokens=64,
            max_beam_width=2,
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4]),
        )
        for is_generation in (True, False):
            with self.subTest(is_generation=is_generation):
                engine = DummyModelEngine(llm_args, torch.half, is_generation=is_generation)

                runner = engine._runner
                if is_generation:
                    self.assertIsInstance(runner, DecoderRunner)
                    self.assertIsNotNone(engine.cuda_graph_runner)
                    self.assertEqual(
                        tuple(runner._buffers.cache_indirection.shape), (4, 2, engine.max_seq_len)
                    )
                else:
                    self.assertIsInstance(runner, PoolingRunner)
                    self.assertIsNone(engine.cuda_graph_runner)
                self.assertEqual(isinstance(runner._buffers, DecoderInputBuffers), is_generation)

    def test_no_kv_cache_runners_reject_speculative_config(self) -> None:
        for mm_encoder_only, runner_name in (
            (False, "PoolingRunner"),
            (True, "MultimodalEncoderRunner"),
        ):
            with self.subTest(runner=runner_name):
                llm_args = TorchLlmArgs(
                    model="dummy",
                    max_batch_size=4,
                    max_num_tokens=64,
                    mm_encoder_only=mm_encoder_only,
                )
                with self.assertRaisesRegex(
                    ValueError, f"{runner_name} does not support speculative decoding"
                ):
                    DummyModelEngine(
                        llm_args,
                        torch.half,
                        spec_config=SADecodingConfig(max_draft_len=1),
                        is_generation=False,
                    )

    def test_runner_engine_shared_entries_skip_decoder_state(self) -> None:
        runner = Mock()
        engine = object.__new__(ModelEngine)
        engine._runner = runner
        engine._torch_compile_backend = None

        engine.wait_for_input_copy()
        engine.release_executor_resources(release_model_owned_graphs=False)
        engine.init_cuda_graph_lora_manager(Mock())

        runner.wait_for_input_copy.assert_called_once_with()
        runner.release_graphs.assert_called_once_with(release_nccl_window_owners=True)
        runner.init_cuda_graph_lora_manager.assert_not_called()

    def test_decoder_engine_shared_entries_use_decoder_state(self) -> None:
        engine = object.__new__(ModelEngine)
        decoder = object.__new__(DecoderRunner)
        engine._runner = decoder
        engine._torch_compile_backend = None
        decoder._inputs = object.__new__(DecoderInputPreparer)
        decoder._inputs._prepare_inputs_event = Mock()
        decoder._graphs = DecoderCudaGraphs(
            cuda_graph_runner=Mock(),
            breakable_cuda_graph_runner=Mock(),
            config=None,
            mapping=None,
            model=None,
            sparse_attention_config=None,
            metrics={},
            dummy_requests=None,
            agreement=None,
            metadata=None,
            get_runtime_tokens_per_gen_step=None,
            dynamic_draft_len_mapping=None,
            lora=None,
            forward_warmup=None,
            torch_compile_bypass=nullcontext,
            context_promotion_supported=True,
        )
        decoder._metadata = SimpleNamespace(attn_metadata=Mock())

        engine.wait_for_input_copy()
        engine.release_executor_resources(release_model_owned_graphs=False)

        decoder._inputs._prepare_inputs_event.synchronize.assert_called_once_with()
        decoder.cuda_graph_runner.clear.assert_called_once_with(release_nccl_window_owners=True)
        decoder._graphs.breakable_cuda_graph_runner.clear.assert_called_once_with(
            release_nccl_window_owners=True
        )
        self.assertIsNone(decoder.attn_metadata)

    def test_position_id_preparation(self):
        (model_engine, kv_cache_manager) = self.enterContext(create_model_engine_and_kvcache())
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 256
        requests = [_create_request(prompt_len, 0)]

        # Prefill run
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

        expected_prefill_pos_ids = torch.arange(
            0, prompt_len, dtype=torch.int32, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(
            model_engine.model.recorded_position_ids, expected_prefill_pos_ids, atol=0, rtol=0
        )

        # Simulate decoding one token after prefill
        requests[-1].add_new_token(42, 0)

        # Generation run
        batch = ScheduledRequests()
        batch.generation_requests = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)

        model_engine.forward(batch, resource_manager)
        expected_gen_pos_id = torch.tensor(
            [prompt_len], dtype=torch.int32, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(
            model_engine.model.recorded_position_ids, expected_gen_pos_id, atol=0, rtol=0
        )

    def test_warmup(self):
        (model_engine, kv_cache_manager) = self.enterContext(create_model_engine_and_kvcache())
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        # Test with a huge batch size. The warmup run should bail out of
        # warmup instead of crashing (there's not enough KV cache space for this).
        model_engine._runner._config.cuda_graph_batch_sizes.append(1000000000)

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
        self.assertEqual(num_free_before, kv_cache_manager.get_num_free_blocks())

    def test_warmup_skips_padding_dummy_when_padding_impossible(self):
        # With max_batch_size=1 and a graph for batch size 1, every batch
        # already matches a graph size, so no padded batch can ever occur and
        # warmup must not retain a padding dummy (it would permanently hold
        # KV blocks and spec/hybrid resources that are never used).
        llm_args = TorchLlmArgs(
            model="dummy",
            max_batch_size=1,
            max_num_tokens=258,
            cuda_graph_config=CudaGraphConfig(enable_padding=True, batch_sizes=[1, 2, 4]),
        )
        (model_engine, kv_cache_manager) = self.enterContext(
            create_model_engine_and_kvcache(llm_args)
        )
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        num_free_before = kv_cache_manager.get_num_free_blocks()
        model_engine.warmup(resource_manager)

        self.assertEqual({}, model_engine.cuda_graph_runner.padding_dummy_requests)
        self.assertEqual(num_free_before, kv_cache_manager.get_num_free_blocks())

    def test_warmup_skips_padding_dummy_during_estimation(self):
        # The estimation-phase KV cache is sized with no headroom for retained
        # dummies; holding blocks there can leave the estimation requests
        # unschedulable. Warmup must skip the preallocation for managers
        # created for KV cache capacity estimation.
        (model_engine, kv_cache_manager) = self.enterContext(create_model_engine_and_kvcache())
        kv_cache_manager.is_estimating_kv_cache = True
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        num_free_before = kv_cache_manager.get_num_free_blocks()
        model_engine.warmup(resource_manager)

        self.assertEqual({}, model_engine.cuda_graph_runner.padding_dummy_requests)
        self.assertEqual(num_free_before, kv_cache_manager.get_num_free_blocks())

    def test_layerwise_nvtx_marker(self):
        llm_args = TorchLlmArgs(
            model="dummy",
            enable_layerwise_nvtx_marker=True,
            cuda_graph_config=CudaGraphConfig(enable_padding=True),
        )
        (model_engine, kv_cache_manager) = self.enterContext(
            create_model_engine_and_kvcache(llm_args)
        )
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

        expected_prefill_pos_ids = torch.arange(
            0, prompt_len, dtype=torch.int32, device="cuda"
        ).unsqueeze(0)
        torch.testing.assert_close(
            model_engine.model.recorded_position_ids, expected_prefill_pos_ids, atol=0, rtol=0
        )

    def test_cuda_graph_padding_filters_huge_batch_size(self):
        llm_args = TorchLlmArgs(
            model="dummy",
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True, batch_sizes=[1, 2, 3, 1000000000000000000000000]
            ),
        )
        model_engine = DummyModelEngine(llm_args, torch.half)

        self.assertEqual(model_engine.cuda_graph_batch_sizes, [1, 2, 3, model_engine.max_seq_len])
        self.assertTrue(model_engine.cuda_graph_padding_enabled)

    def test_forward_pass_callable_on_cuda_graph_on(self):
        llm_args = TorchLlmArgs(
            model="dummy",
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True,
            ),
        )
        (model_engine, kv_cache_manager) = self.enterContext(
            create_model_engine_and_kvcache(llm_args)
        )

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_forward_pass_callable_on_cuda_graph_off(self):
        (model_engine, kv_cache_manager) = self.enterContext(create_model_engine_and_kvcache())

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_foward_pass_callable_off(self):
        (model_engine, kv_cache_manager) = self.enterContext(create_model_engine_and_kvcache())
        self.assertTrue(
            model_engine._runner.forward_pass_callable is None,
            "forward_pass_callback should be None by default",
        )

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

    def test_foward_pass_callable_backward_compat(self):
        (model_engine, kv_cache_manager) = self.enterContext(create_model_engine_and_kvcache())
        self.assertTrue(
            model_engine._runner.forward_pass_callable is None,
            "forward_pass_callback should be None by default",
        )

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

    def test_kv_cache_manager_with_execution_stream(self) -> None:
        """Test that KVCacheManagerV2 uses the provided execution_stream."""
        # Create a dedicated execution stream
        execution_stream = torch.cuda.Stream()

        (model_engine, kv_cache_manager) = self.enterContext(
            create_model_engine_and_kvcache(execution_stream=execution_stream)
        )

        # Verify the KVCacheManagerV2 uses the provided execution stream
        self.assertEqual(
            kv_cache_manager._stream.cuda_stream,
            execution_stream.cuda_stream,
            "KVCacheManagerV2 should use the provided execution_stream",
        )

        resource_manager = ResourceManager({ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        with torch.cuda.stream(execution_stream):
            model_engine.forward(batch, resource_manager)

        # Verify the stream is still the same after forward pass
        self.assertEqual(
            kv_cache_manager._stream.cuda_stream,
            execution_stream.cuda_stream,
            "KVCacheManagerV2 should still use the provided execution_stream after forward",
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("model_kind", ["default", "m3", "m3_vl", "gdn", "mamba"])
@pytest.mark.parametrize("compile_enabled,piecewise", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("compile_only_piecewise_graphs", [False, True])
def test_pcg_fx_fallback_policy_is_model_specific(
    monkeypatch: pytest.MonkeyPatch,
    model_kind: str,
    compile_enabled: bool,
    piecewise: bool,
    compile_only_piecewise_graphs: bool,
) -> None:
    """Exercise the real constructor's routing gate without loading weights or CUDA."""
    from tensorrt_llm._torch.models.modeling_minimaxm3 import (
        MiniMaxM3ForCausalLM,
        MiniMaxM3VLForConditionalGeneration,
    )
    from tensorrt_llm._torch.models.modeling_nemotron_h import NemotronHForCausalLM
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextForCausalLM
    from tensorrt_llm._torch.pyexecutor import _util

    model_cls = {
        "default": DecoderModelForCausalLM,
        "m3": MiniMaxM3ForCausalLM,
        "m3_vl": MiniMaxM3VLForConditionalGeneration,
        "gdn": Qwen3NextForCausalLM,
        "mamba": NemotronHForCausalLM,
    }[model_kind]
    assert model_cls.use_fx_for_pcg_fallback is (model_kind not in ("m3", "m3_vl"))
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    mapping = Mapping()
    model.model_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(torch_dtype=torch.float32, hidden_size=4),
        mapping=mapping,
        sparse_attention_config=None,
    )
    eager = torch.nn.Linear(4, 4)
    model.model = eager
    compile_config = SimpleNamespace(
        enable_fullgraph=True,
        compile_only_piecewise_graphs=compile_only_piecewise_graphs,
        enable_inductor=False,
        enable_userbuffers=False,
        max_num_streams=1,
    )
    llm_args = SimpleNamespace(
        encode_only=False,
        mm_encoder_only=False,
        get_runtime_sizes=lambda: (1, 16, 64, 4),
        encoder_max_batch_size=None,
        encoder_max_num_tokens=None,
        enable_in_graph_sampling=False,
        enable_return_routed_experts=False,
        multimodal_config=MultimodalConfig(),
        checkpoint_format="HF",
        trust_remote_code=False,
        disable_overlap_scheduler=True,
        kv_cache_config=KvCacheConfig(),
        sleep_config=None,
        enable_layerwise_nvtx_marker=False,
        cuda_graph_config=None,
        torch_compile_config=compile_config if compile_enabled else None,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.PIECEWISE if piecewise else None,
        prefill_capture_num_tokens=[4],
        allreduce_strategy="AUTO",
        attn_backend="TRTLLM",
        sparse_attention_config=None,
    )
    for name in (
        "should_enable_overlap_headroom",
        "resolved_kv_cache_manager_is_v2",
    ):
        monkeypatch.setattr(_util, name, Mock(return_value=False))
    monkeypatch.setattr(_util, "compute_max_num_sequences", Mock(return_value=4))
    for name in (
        "_configure_deep_gemm_pdl",
        "create_input_processor",
        "setup_mm_encoder_attn_metadata",
    ):
        monkeypatch.setattr(model_engine_module, name, Mock())
    monkeypatch.setattr(
        model_engine_module, "resolve_mrope_position_deltas_cache", lambda model: None
    )
    monkeypatch.setattr(
        model_engine_module, "is_hybrid_linear", lambda config: model_kind in ("gdn", "mamba")
    )
    monkeypatch.setattr(
        model_engine_module.MultimodalItemScheduler, "maybe_create", Mock(return_value=None)
    )
    monkeypatch.setattr(ModelEngine, "_validate_breakable_cuda_graph_compatibility", Mock())
    monkeypatch.setattr(ModelEngine, "_init_model_capacity", Mock())
    monkeypatch.setattr(ModelEngine, "__del__", lambda self: None)
    backend_factory = Mock()
    backend_factory.Streams = list
    monkeypatch.setattr(model_engine_module, "Backend", backend_factory)
    compiled = torch.nn.Identity()
    compile_model = Mock(return_value=compiled)
    monkeypatch.setattr(torch, "compile", compile_model)
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 16)
    # Stop after the complete compilation block, before runtime cache allocation.
    monkeypatch.setattr(
        model_engine_module,
        "get_attention_backend",
        Mock(side_effect=RuntimeError("compile setup complete")),
    )
    engine = ModelEngine.__new__(ModelEngine)
    with torch_compiling(False), pytest.raises(RuntimeError, match="compile setup complete"):
        ModelEngine.__init__(
            engine,
            model_path="dummy",
            mapping=mapping,
            model=model,
            llm_args=llm_args,
            checkpoint_loader=Mock(),
        )

    expected_prefill_only = (
        compile_enabled
        and piecewise
        and (compile_only_piecewise_graphs or model_kind in ("m3", "m3_vl"))
    )
    assert engine._torch_compile_prefill_only is expected_prefill_only
    if not compile_enabled:
        compile_model.assert_not_called()
        assert model.model is eager
    else:
        compile_model.assert_called_once_with(
            eager, backend=engine._torch_compile_backend, fullgraph=True
        )
        assert backend_factory.call_args.args[0] is False  # Inductor stays disabled.
        if expected_prefill_only:
            assert isinstance(model.model, PrefillCompiledModel)
            assert model.model.eager_model is eager
            assert model.model.compiled_model is compiled
        else:
            assert model.model is compiled


class TestWarmupCleanup(unittest.TestCase):
    @pytest.mark.cpu_only
    def test_no_kv_cache_warmup_delegates_runner_lifecycle(self):
        """Verify no-KV-cache warmup delegates warmup and graph capture to the runner."""
        model_engine = object.__new__(ModelEngine)
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine._metrics = {}
        model_engine._warmup_depth = 0
        model_engine._runner = Mock(spec=NoKVCacheRunner)
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = None

        model_engine.warmup(resource_manager)

        self.assertEqual(
            model_engine._runner.method_calls,
            [call.warmup(resource_manager)],
        )

    @pytest.mark.cpu_only
    def test_no_kv_cache_warmup_rejects_allocated_kv_cache(self):
        """Verify a no-KV-cache runner rejects an allocated KV cache before warmup."""
        model_engine = object.__new__(ModelEngine)
        model_engine._metrics = {}
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
        model_engine.moe_load_balancer = None
        model_engine._warmup_depth = 0
        model_engine._runner = Mock(spec=NoKVCacheRunner)
        runner = model_engine._runner
        runner._validate_resources = NoKVCacheRunner._validate_resources.__get__(runner)
        runner.warmup.side_effect = NoKVCacheRunner.warmup.__get__(runner)
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = object()

        with self.assertRaisesRegex(
            AssertionError,
            "no-KV-cache runner was initialized, but a KV cache manager was allocated",
        ):
            model_engine.warmup(resource_manager)

        model_engine._runner.warmup.assert_called_once_with(resource_manager)

    @pytest.mark.cpu_only
    def test_warmup_total_timing_and_kv_cache_cleanup(self) -> None:
        model_engine = object.__new__(ModelEngine)
        model_engine._metrics = {}
        model_engine._warmup_depth = 0
        model_engine.moe_load_balancer = None
        model_engine._runner = Mock(spec=DecoderRunner)
        kv_cache_manager = Mock()
        kv_cache_manager.check_invalid_values_in_kv_cache.return_value = False
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = kv_cache_manager
        with (
            patch.object(model_engine, "_warmup_impl") as warmup_impl,
            patch("time.perf_counter", side_effect=[0.0, 2.0, 5.0, 7.0]),
        ):
            model_engine.warmup(resource_manager)
        warmup_impl.assert_called_once_with(resource_manager)
        kv_cache_manager.check_invalid_values_in_kv_cache.assert_called_once_with(
            fill_with_zero=True
        )
        self.assertEqual(
            model_engine.metrics, {"kv_cache_cleanup_seconds": 3.0, "total_warmup_seconds": 7.0}
        )

    def test_encoder_decoder_encoder_warmup_delegates_runner_lifecycle(self):
        model_engine = object.__new__(ModelEngine)
        model_engine.model = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=True))
        model_engine.moe_load_balancer = None
        model_engine._warmup_depth = 0
        model_engine._runner = Mock(spec=EncoderDecoderRunner)
        warmup_depths = []
        model_engine._runner.warmup_encoder.side_effect = lambda _: warmup_depths.append(
            model_engine._warmup_depth
        )
        resource_manager = object()

        model_engine.warmup_encoder(resource_manager)

        self.assertEqual(
            model_engine._runner.method_calls,
            [call.warmup_encoder(resource_manager)],
        )
        self.assertEqual(warmup_depths, [1])
        self.assertEqual(model_engine._warmup_depth, 0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("runner_type", "initializer_name"),
    [
        (EncoderRunner, "_initialize_encoder_runner"),
        (EncoderDecoderRunner, "_initialize_encoder_decoder_runner"),
        (NoKVCacheRunner, "_initialize_no_kv_cache_runner"),
        (DecoderRunner, "_initialize_decoder_runner"),
    ],
)
def test_model_engine_initializes_runner_by_family(
    runner_type: type[ModelRunner], initializer_name: str
) -> None:
    engine = object.__new__(ModelEngine)
    engine._model_caller = Mock()
    expected = Mock(spec=runner_type)
    initializer = Mock(return_value=expected)
    setattr(engine, initializer_name, initializer)

    assert engine._initialize_runner(runner_type) is expected
    initializer.assert_called_once_with(runner_type)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("cuda_graph_config", "expected_encode_config"),
    [
        (None, False),
        (CudaGraphConfig(), False),
        (EncodeCudaGraphConfig(batch_sizes=[1], num_tokens=[16], seq_lens=[16]), True),
    ],
)
def test_encoder_runner_graph_config_comes_only_from_cuda_graph_config(
    cuda_graph_config: CudaGraphConfig | None,
    expected_encode_config: bool,
) -> None:
    """Encode-only takes its buckets from `cuda_graph_config`, never from llm_args."""
    engine = object.__new__(ModelEngine)
    engine._model_caller = Mock()
    engine.model = object()
    engine.mapping = object()
    engine.cuda_graph_config = cuda_graph_config
    engine.llm_args = SimpleNamespace(
        encoder_cuda_graph_config=EncodeCudaGraphConfig(
            batch_sizes=[8],
            num_tokens=[64],
            seq_lens=[64],
        ),
        enable_autotuner=False,
    )
    engine.batch_size = 4
    engine.max_num_tokens = 16
    engine.max_seq_len = 16
    engine.max_beam_width = 1
    engine.without_logits = False
    engine.attn_backend = _AttentionBackend
    engine.attn_runtime_features = AttentionRuntimeFeatures()
    engine.dist = object()
    engine.moe_load_balancer = None
    runner_config = object()
    expected_runner = object()
    runner_type = Mock(return_value=expected_runner)

    with patch(
        "tensorrt_llm._torch.pyexecutor.model_engine.EncoderRunnerConfig.create",
        return_value=runner_config,
    ) as create_config:
        actual = engine._initialize_encoder_runner(runner_type)

    assert actual is expected_runner
    resolved = create_config.call_args.kwargs["graph_config"]
    assert resolved is (cuda_graph_config if expected_encode_config else None)
    runner_type.assert_called_once_with(
        engine.model,
        runner_config,
        mapping=engine.mapping,
        dist=engine.dist,
        moe_load_balancer=engine.moe_load_balancer,
        model_caller=engine._model_caller,
        attention_runtime_features=engine.attn_runtime_features,
    )


@pytest.mark.cpu_only
def test_model_engine_rejects_unregistered_runner_family() -> None:
    class UnregisteredRunner:
        pass

    engine = object.__new__(ModelEngine)
    engine._model_caller = Mock()

    with pytest.raises(TypeError, match="No runner initializer registered"):
        engine._initialize_runner(UnregisteredRunner)


@pytest.mark.cpu_only
def test_model_engine_forward_delegates_to_resolved_runner() -> None:
    runner = Mock(spec=NoKVCacheRunner)
    expected_outputs = {"logits": object()}
    runner.forward.return_value = expected_outputs
    engine, resource_manager = _model_engine_with_runner(
        runner,
        kv_cache_manager=None,
    )
    batch = ScheduledRequests()

    actual_outputs = engine.forward(batch, resource_manager)

    assert actual_outputs is expected_outputs
    runner.forward.assert_called_once_with(
        ScheduledInputs(batch=batch),
        resource_manager=resource_manager,
        is_dummy=False,
    )


@pytest.mark.cpu_only
def test_model_engine_forward_encoder_delegates_scheduled_encoder_batch() -> None:
    runner = Mock(spec=EncoderDecoderRunner)
    hidden_states = object()
    runner.forward_encoder.return_value = {
        "encoder_hidden_states": hidden_states,
        "encoder_seq_lens": [2, 3],
    }
    engine = object.__new__(ModelEngine)
    engine._model_caller = Mock()
    engine.model = SimpleNamespace(
        model_config=SimpleNamespace(is_encoder_decoder=True),
    )
    engine._runner = runner
    engine._warmup_depth = 0
    requests = [object(), object()]
    resource_manager = object()

    outputs = engine.forward_encoder(requests, resource_manager)

    assert outputs == (hidden_states, [2, 3])
    inputs = runner.forward_encoder.call_args.args[0]
    assert inputs.batch.encoder_requests == requests
    runner.forward_encoder.assert_called_once_with(
        inputs,
        resource_manager=resource_manager,
        is_dummy=False,
    )


@pytest.mark.cpu_only
def test_model_engine_releases_runner_owned_graphs() -> None:
    engine = object.__new__(ModelEngine)
    engine._model_caller = Mock()
    engine._runner = Mock(spec=EncoderRunner)
    engine._torch_compile_backend = None

    engine.release_executor_resources(release_model_owned_graphs=False)

    engine._runner.release_graphs.assert_called_once_with(release_nccl_window_owners=True)


@pytest.mark.cpu_only
@pytest.mark.parametrize("effective_length", [None, 0, 3])
@pytest.mark.parametrize("is_dummy", [False, True])
def test_engine_passes_call_state_and_returns_length_update(effective_length, is_dummy):
    runner = Mock(spec=NoKVCacheRunner)
    outputs = {"logits": object()}
    if effective_length is not None:
        outputs["runtime_draft_len"] = effective_length
    runner.forward.return_value = outputs
    engine, resources = _model_engine_with_runner(runner, kv_cache_manager=None)
    engine._warmup_depth = int(is_dummy)
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [SimpleNamespace(py_return_context_logits=True)]

    assert engine.forward(batch, resources, enable_spec_decode=True, runtime_draft_len=2) is outputs

    inputs = runner.forward.call_args.args[0]
    assert inputs.batch is batch
    assert inputs.gather_context_logits
    assert inputs.enable_spec_decode
    assert inputs.runtime_draft_len == 2
    assert runner.forward.call_args.kwargs["is_dummy"] is is_dummy
    assert outputs.get("runtime_draft_len") == effective_length


@pytest.mark.cpu_only
@pytest.mark.parametrize("raw_output", [torch.tensor([1.0]), None], ids=["tensor", "none"])
@pytest.mark.parametrize("is_dummy", [False, True])
def test_engine_forward_preserves_raw_decoder_outputs(raw_output, is_dummy):
    engine, resources = _model_engine_with_runner(None, kv_cache_manager=object())
    engine._warmup_depth = int(is_dummy)
    engine._runner._forward_decoder = Mock(return_value=raw_output)

    assert engine.forward(ScheduledRequests(), resources, runtime_draft_len=2) is raw_output
    assert engine._runner._forward_decoder.call_args.args[0].runtime_draft_len == 2
    assert engine._runner._forward_decoder.call_args.kwargs == {"is_dummy": is_dummy}
