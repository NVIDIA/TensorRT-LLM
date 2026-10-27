# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_encoder import \
    MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalModelMixin, _build_request_multimodal_input)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import \
    KvCacheConnectorWorker
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    CUDAGraphRunner, KeyType, SampleType, _restore_spec_decode_capture_state,
    _save_spec_decode_capture_state)
from tensorrt_llm._torch.pyexecutor.engine.input_buffers import InputBuffers
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
    is_multimodal, setup_mm_encoder_attn_metadata)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import (
    DecoderCudaGraphs, _make_single_token_context_graph_batch)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import (
    DecoderInputBuffers, DecoderInputPreparer,
    _get_context_prompt_lookahead_token, _zero_fill_row_spans)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.metadata import \
    DecoderMetadata
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import \
    PrefillCompiledModel
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import \
    DecoderWarmup
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import \
    PreparedInputs
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import \
    KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm.llmapi.llm_args import (DecodingBaseConfig,
                                          PrefillCudaGraphBackend,
                                          SeqLenAwareSparseAttentionConfig,
                                          TorchLlmArgs)

# isort: off
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
# isort: on
from utils.util import skip_pre_blackwell_unittest, skip_ray

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import \
    TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttention
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative.interface import \
    INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_sampler_base import \
    SampleStateTensorsSpec
from tensorrt_llm._torch.utils import set_per_request_prefill_cuda_graph_flag
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalItemMetadata)
from tensorrt_llm.inputs.registry import (BaseMultimodalDummyInputsBuilder,
                                          BaseMultimodalInputProcessor)
from tensorrt_llm.llmapi import (CudaGraphConfig, MTPDecodingConfig,
                                 NGramDecodingConfig, SADecodingConfig,
                                 SamplingParams)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import CpType, Mapping

# One-engine checks see a two-model mode, so no draft KV cache is looked up.
_TWO_MODEL_SPEC_MODE = SimpleNamespace(use_one_engine=lambda: False)


def _input_preparer() -> DecoderInputPreparer:
    return DecoderInputPreparer(buffers=None,
                                config=SimpleNamespace(is_spec_decode=False),
                                mapping=None,
                                dist=None,
                                model=None,
                                get_runtime_tokens_per_gen_step=None,
                                iter_states={},
                                lora=None,
                                use_beam_search=False,
                                steady_gen_cache_supported=True)


class TestCachedKvTokenLogging(unittest.TestCase):

    def setUp(self) -> None:
        self.inputs = _input_preparer()
        self.real = SimpleNamespace(is_cuda_graph_dummy=False)
        self.padding = SimpleNamespace(is_cuda_graph_dummy=True)

    def check_counts(
        self,
        counts: list[int],
        groups: tuple[tuple[list[SimpleNamespace], int], ...],
        expected: list[int],
        padding: int,
    ) -> None:
        self.inputs.iter_states['cached_kv_tokens'] = sum(counts)
        self.inputs.record_cached_kv_tokens_per_req(counts, groups)
        self.assertEqual(
            self.inputs.iter_states, {
                'cached_kv_tokens': sum(counts),
                'cached_kv_tokens_per_req': expected,
                'cached_kv_tokens_cuda_graph_padding': padding,
            })
        counts[:] = [999]
        self.assertEqual(self.inputs.iter_states['cached_kv_tokens_per_req'],
                         expected)

    def test_mtp_padding_before_generation(self) -> None:
        # Input packing is context, extend (including padding), gen.
        self.check_counts([10, 100, 7, 200],
                          (([self.real], 1), ([self.real, self.padding], 1),
                           ([self.real], 1)), [10, 100, 200], 7)

    def test_beams_preserved_after_context_and_padding(self) -> None:
        self.check_counts([10, 100, 101, 7, 7, 200, 201],
                          (([self.real], 1),
                           ([self.real, self.padding, self.real], 2)),
                          [10, 100, 101, 200, 201], 14)

    def test_attention_dp_dummy_is_not_graph_padding(self) -> None:
        adp_dummy = SimpleNamespace(is_dummy=True,
                                    is_attention_dp_dummy=True,
                                    is_cuda_graph_dummy=False)
        self.check_counts([10, 1, 7],
                          (([self.real, adp_dummy, self.padding], 1), ),
                          [10, 1], 7)

    def test_padding_only(self) -> None:
        self.check_counts([7, 7], (([self.padding, self.padding], 1), ), [], 14)

    def test_group_row_count_must_match_counts(self) -> None:
        for counts, groups in (
            ([100, 200], (([self.real], 1), )),
            ([100, 101, 200], (([self.real, self.real], 2), )),
        ):
            with self.subTest(counts=counts, groups=groups):
                with self.assertRaises(AssertionError):
                    self.inputs.record_cached_kv_tokens_per_req(counts, groups)

    def test_steady_batch_clears_previous_padding(self) -> None:
        self.inputs.iter_states['cached_kv_tokens_cuda_graph_padding'] = 7
        self.check_counts([100, 200], (), [100, 200], 0)

    def test_empty_batch_clears_previous_values(self) -> None:
        self.inputs.iter_states['cached_kv_tokens_per_req'] = [100]
        self.inputs.iter_states['cached_kv_tokens_cuda_graph_padding'] = 7
        self.check_counts([], (([], 1), ), [], 0)


@dataclass
class Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256
    architectures: list[str] = None
    rope_scaling: dict | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class DummyKvCacheConnectorWorker(KvCacheConnectorWorker):

    def register_kv_caches(self, kv_cache_tensor: torch.Tensor):
        pass


class DummyModel(torch.nn.Module):

    def __init__(self, dtype: torch.dtype, rope_scaling: dict | None = None):
        super().__init__()
        self.model_config = ModelConfig(pretrained_config=Config(
            torch_dtype=dtype, rope_scaling=rope_scaling))
        self.recorded_position_ids = None
        self.recorded_kwargs = []

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(self, *args, **kwargs) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        self.recorded_position_ids = kwargs["position_ids"]
        # Input buffers are reused across steps, so keep copies.
        if not torch.cuda.is_current_stream_capturing():
            self.recorded_kwargs.append(_clone_tensors(kwargs))
        batch_size = input_ids.size(0)
        return {"logits": torch.randn((batch_size, 10), device='cuda')}


class DummyLegacyMultimodalIndexModel(MultimodalModelMixin, torch.nn.Module):

    class Config:
        vocab_size = 100

    config = Config()

    @property
    def mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([90, 91], dtype=torch.int32)


class DummyModelEngine(ModelEngine):

    def __init__(
        self,
        llm_args: TorchLlmArgs,
        dtype: torch.dtype,
        spec_config: DecodingBaseConfig | None = None,
        is_generation: bool = True,
        model: DummyModel | None = None,
    ) -> None:
        self.dtype = dtype
        mapping = Mapping(world_size=tensorrt_llm.mpi_world_size(),
                          tp_size=tensorrt_llm.mpi_world_size(),
                          rank=tensorrt_llm.mpi_rank())
        if model is None:
            model = DummyModel(self.dtype)
        model.model_config.is_generation = is_generation
        super().__init__(model_path="dummy",
                         mapping=mapping,
                         model=model,
                         llm_args=llm_args,
                         spec_config=spec_config)


def _prepare_kv_cache_resources(manager: KVCacheManagerV2,
                                batch: ScheduledRequests) -> None:
    # These model-engine tests bypass the scheduler, which owns V2 allocation.
    for request in batch.context_requests:
        assert manager.prepare_context(request)
        assert manager.resize_context(
            request, request.prompt_len - request.context_current_position)
    for request in batch.generation_requests:
        assert manager.try_allocate_generation(request)
    manager.prepare_resources(batch)


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
    """An internal chunk hands back the next prompt token; the last one does not.

    The boundary is read off the request's live chunking state rather than a
    Python-side prompt length, which a preemption leaves stale.
    """
    request = _create_request_with_tokens([10, 11, 12, 13, 14], 1)

    request.context_chunk_size = 2
    assert _get_context_prompt_lookahead_token(request, 2) == 12

    # A reused prefix advances the live cursor while py_prompt_len remains the
    # original prompt length. Internal chunk [2, 4) still looks up token 4.
    request.context_current_position = 2
    request.context_chunk_size = 2
    assert _get_context_prompt_lookahead_token(request, 4) == 14

    # Final chunk [2, 5): nothing follows in the prompt, so the drafter falls
    # back to the token the target just sampled.
    request.context_chunk_size = 3
    assert (_get_context_prompt_lookahead_token(
        request, 5) == INVALID_PROMPT_LOOKAHEAD_TOKEN)


def test_zero_fill_row_spans_merges_unsorted_and_adjacent_spans() -> None:
    buffer = torch.arange(1, 13, dtype=torch.int32)
    _zero_fill_row_spans(buffer, [(9, 11), (2, 4), (4, 5), (7, 8)])
    assert buffer.tolist() == [1, 2, 0, 0, 0, 6, 7, 0, 9, 0, 0, 12]


def test_zero_fill_row_spans_without_spans_is_noop() -> None:
    buffer = torch.arange(4, dtype=torch.int32)
    _zero_fill_row_spans(buffer, [])
    assert buffer.tolist() == [0, 1, 2, 3]


def _make_request_stub(req_id: int, prompt_len: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=req_id,
        context_chunk_size=1,
        context_remaining_length=1,
        context_current_position=prompt_len - 1,
        py_prompt_len=prompt_len,
        py_beam_width=1,
        py_draft_tokens=[],
        is_context_only_request=False,
        is_generation_only_request=False,
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
    private_decoder: bool = False,
) -> tuple[ModelEngine, Mock, Mock, Mock, dict[str, object]]:
    engine = object.__new__(ModelEngine)
    decoder = object.__new__(DecoderRunner)
    engine.model = decoder._model = SimpleNamespace(
        extra_attrs={},
        model_config=SimpleNamespace(pretrained_config=SimpleNamespace(
            rope_scaling=None)))
    engine._warmup_depth = 0
    decoder._config = SimpleNamespace(
        compile_only_piecewise_graphs=False,
        disable_overlap_scheduler=True,
        is_spec_decode=False,
        max_beam_width=1,
        is_encode_only=False,
        attention_backend=None,
        original_max_draft_len=0,
        original_max_total_draft_tokens=0,
        spec_dec_max_total_draft_tokens=0,
        use_mrope=False,
        spec_config=None,
    )
    engine.llm_args = SimpleNamespace(mm_encoder_only=False)
    engine.mapping = decoder._mapping = SimpleNamespace(
        cp_size=1,
        enable_lm_head_tp_in_adp=False,
    )
    decoder._dist = None
    get_runtime_tokens_per_gen_step = Mock(return_value=1)
    decoder._get_runtime_tokens_per_gen_step = get_runtime_tokens_per_gen_step
    engine.iter_states = decoder._iter_states = {}
    decoder.forward_pass_callable = None
    decoder.start_beam_histories_callable = None
    decoder._stage_in_graph_sampling = None
    decoder._moe_load_balancer = None
    engine._runner = decoder
    decoder._lora = SimpleNamespace(cuda_graph_manager=None)
    decoder._trtllm_gen_jit_warmup = False

    semantic_attn_metadata = Mock()
    graph_attn_metadata = Mock()
    decoder._metadata = SimpleNamespace(
        attn_metadata=semantic_attn_metadata,
        setup_attention=Mock(return_value=semantic_attn_metadata))
    spec_dec_mode = Mock()
    spec_dec_mode.attention_need_spec_dec_mode.return_value = False
    spec_dec_mode.is_parallel_draft.return_value = False
    spec_metadata = Mock(
        spec_dec_mode=spec_dec_mode,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
    )
    decoder._metadata.spec_metadata = spec_metadata
    decoder._metadata.setup_speculative = Mock(return_value=spec_metadata)
    prepared_inputs = {
        "prepared": True,
        "input_ids": torch.zeros(2, dtype=torch.int32),
    }
    decoder._inputs = SimpleNamespace(
        guided_decoder=None,
        prepare_inputs=Mock(return_value=PreparedInputs(prepared_inputs)),
        record_promoted_iter_states=Mock(),
        record_input_copy=Mock())
    outputs = {"logits": object()}
    decoder._forward_step = Mock(return_value=outputs)
    decoder._execute_logit_post_processors = Mock()

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
    decoder._graphs = DecoderCudaGraphs(
        cuda_graph_runner=runner,
        breakable_cuda_graph_runner=None,
        config=decoder._config,
        mapping=decoder._mapping,
        model=decoder._model,
        sparse_attention_config=None,
        metrics={},
        dummy_requests=None,
        agreement=None,
        metadata=decoder._metadata,
        get_runtime_tokens_per_gen_step=get_runtime_tokens_per_gen_step,
        dynamic_draft_len_mapping=None,
        lora=decoder._lora,
        forward_warmup=decoder._forward_warmup,
        torch_compile_bypass=decoder._bypass_torch_compile,
        context_promotion_supported=True,
    )

    resource_manager = Mock()
    peft_cache_manager = Mock(data_type=torch.bfloat16)

    def get_resource_manager(resource_type):
        if resource_type == ResourceManagerType.PEFT_CACHE_MANAGER:
            return peft_cache_manager
        if resource_type == ResourceManagerType.KV_CACHE_MANAGER:
            return SimpleNamespace(has_private_swa_suffix=Mock(
                return_value=private_decoder))
        return object()

    resource_manager.get_resource_manager.side_effect = get_resource_manager
    return engine, runner, resource_manager, semantic_attn_metadata, outputs


@pytest.mark.cpu_only
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "ordinary_capture,eligible",
    [(False, True), (False, False), (True, True), (True, False)],
)
def test_decoder_forward_routes_compile_bypass(enabled: bool,
                                               ordinary_capture: bool,
                                               eligible: bool) -> None:
    """Exercise both the regular forward and ordinary-graph capture callbacks."""
    key = KeyType(batch_size=1, draft_len=0) if ordinary_capture else None
    engine, graph_runner, resources, _, outputs = _make_forward_only_engine(key)
    eager = torch.nn.Identity()
    compiled = torch.nn.Identity()
    eager.forward = Mock(return_value=outputs)
    compiled.forward = Mock(return_value=outputs)
    wrapper = PrefillCompiledModel(eager, compiled)
    engine._runner._config.compile_only_piecewise_graphs = enabled
    engine._runner._model.model = wrapper
    engine._runner._forward_step.side_effect = lambda *args, **kwargs: wrapper()
    graph_runner.needs_capture.return_value = ordinary_capture
    graph_runner.is_warmup_only = ordinary_capture
    graph_runner.capture.side_effect = lambda key, forward, inputs, **kwargs: forward(
        inputs)
    batch = ScheduledRequests()
    batch.generation_requests = [_make_request_stub(1)]

    with patch(
            "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
            return_value=Mock()
    ), patch(
            "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner.get_per_request_prefill_cuda_graph_flag",
            return_value=eligible,
    ):
        result = engine.forward(batch, resources)

    assert result["logits"] is outputs["logits"]
    expected_compiled = eligible and not (enabled and ordinary_capture)
    selected = compiled.forward if expected_compiled else eager.forward
    other = eager.forward if expected_compiled else compiled.forward
    selected.assert_called_once_with()
    other.assert_not_called()
    assert not wrapper._bypass_active


def create_model_engine_and_kvcache(
    llm_args: TorchLlmArgs | None = None,
    execution_stream: torch.cuda.Stream | None = None,
    spec_config: DecodingBaseConfig | None = None,
    model: DummyModel | None = None,
) -> tuple[ModelEngine, KVCacheManagerV2]:
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

    model_engine = DummyModelEngine(llm_args,
                                    torch.half,
                                    spec_config,
                                    model=model)

    kv_cache_config = KvCacheConfig(max_tokens=max_tokens)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    kv_cache_manager = KVCacheManagerV2(
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


def _resize_input_buffers(runner: DecoderRunner, max_num_tokens: int) -> None:
    """Reallocate the runner's input buffers for ``max_num_tokens`` tokens."""
    buffers = DecoderInputBuffers.for_config(
        replace(runner._config, max_num_tokens=max_num_tokens),
        runner._use_beam_search)
    runner._buffers = runner._inputs._buffers = runner._metadata._buffers = buffers


def _set_runner_config(runner: DecoderRunner, **changes) -> None:
    """Replace the config shared by the runner and its input components."""
    config = replace(runner._config, **changes)
    runner._config = runner._inputs._config = runner._metadata._config = config


def _clone_tensors(values: dict) -> dict:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }


def _install_graph_runner_stub(runner: DecoderRunner,
                               key: KeyType | None = None) -> Mock:
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
        if stub.key is not None else (None, None, None))
    stub.replayed_inputs = []

    def replay(key, inputs, position_id_offsets=None, static_inputs_staged=0):
        stub.replayed_inputs.append(_clone_tensors(inputs))
        return {
            "logits": torch.zeros((inputs["input_ids"].shape[0], 10),
                                  device="cuda")
        }

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

    with patch("tensorrt_llm._torch.pyexecutor.model_engine.release_gc",
               side_effect=assert_model_released) as release_gc:
        engine.cleanup()

    release_gc.assert_called_once_with()


class GraphOwnerTeardownTestCase(unittest.TestCase):

    def test_cuda_graph_runner_destructor_does_not_release_window_owner(
            self) -> None:
        runner = Mock()

        CUDAGraphRunner.__del__(runner)

        runner.clear.assert_called_once_with(release_nccl_window_owners=False)

    def test_engine_graph_ownership_is_split(self) -> None:
        clear_blocks_graph = Mock()
        model = Mock()
        model.modules.return_value = [
            SimpleNamespace(clear_blocks_cuda_graph=clear_blocks_graph)
        ]
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

        ModelEngine._release_cuda_graphs(engine,
                                         release_nccl_window_owners=False)

        clear_blocks_graph.assert_not_called()
        backend.clear_piecewise_cuda_graphs.assert_called_once_with(
            release_nccl_window_owners=False)
        for runner in (cuda_runner, breakable_runner):
            runner.clear.assert_called_once_with(
                release_nccl_window_owners=False)

        ModelEngine._release_model_owned_cuda_graphs(
            engine, release_nccl_window_owners=False)

        clear_blocks_graph.assert_called_once_with(
            release_nccl_window_owners=False)

    def test_executor_release_clears_model_owned_graphs_first(self) -> None:
        for release_model_owned_graphs in (True, False):
            with self.subTest(
                    release_model_owned_graphs=release_model_owned_graphs):
                engine = Mock()

                ModelEngine.release_executor_resources(
                    engine,
                    release_model_owned_graphs=release_model_owned_graphs)

                expected = [call._release_cuda_graphs()]
                if release_model_owned_graphs:
                    expected.insert(0, call._release_model_owned_cuda_graphs())
                self.assertEqual(engine.method_calls, expected)


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
        context.is_generation_only_request = True
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
        graphs = object.__new__(DecoderCudaGraphs)
        graphs.model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=100),
            mm_token_ids=torch.tensor([99], dtype=torch.int32),
        )
        graphs._config = SimpleNamespace(use_mrope=False)
        request = _create_request_with_tokens([11, 99, 22], 1)

        request.context_current_position = 2
        self.assertTrue(
            graphs._is_final_multimodal_context_decode_compatible(request))

        request.context_current_position = 1
        self.assertFalse(
            graphs._is_final_multimodal_context_decode_compatible(request))

        graphs.model.mm_token_ids = None
        request = _create_request_with_tokens([11, 100], 2)
        request.context_current_position = 1
        self.assertFalse(
            graphs._is_final_multimodal_context_decode_compatible(request))

        graphs.model.mm_token_ids = torch.tensor([99], dtype=torch.int32)
        graphs.model.model_config = SimpleNamespace(
            pretrained_config=SimpleNamespace(rope_scaling={"type": "mrope"}))
        graphs._config.use_mrope = True
        request = _create_request_with_tokens([11, 22], 3)
        request.context_current_position = 1
        request.py_multimodal_data = {"mrope_config": {}}
        self.assertTrue(
            graphs._is_final_multimodal_context_decode_compatible(request))

        request.py_multimodal_data["multimodal_embedding"] = object()
        self.assertFalse(
            graphs._is_final_multimodal_context_decode_compatible(request))

        request.py_multimodal_data["mrope_config"][
            "mrope_position_deltas"] = object()
        self.assertTrue(
            graphs._is_final_multimodal_context_decode_compatible(request))

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
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
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
                    short_seq_len_mode=True,
                    num_encoder_tokens=0))

    def test_graph_key_aggregates_encoder_tokens(self) -> None:
        runner = Mock()
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
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
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
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
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
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
                                  peft_cache_data_type=torch.bfloat16)
        fp8_key = KeyType(batch_size=1,
                          draft_len=0,
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
        runner.config = SimpleNamespace()
        runner._resolve_sample_type.return_value = SampleType.FULL
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
            KeyType(batch_size=1, draft_len=0, use_lora_graph=True),
        )

    def test_lora_graph_variant_selection(self) -> None:
        graphs = object.__new__(DecoderCudaGraphs)
        graphs._lora = SimpleNamespace(cuda_graph_manager=object())
        graphs._force_lora_graph_for_capture = None
        graphs._config = SimpleNamespace(cuda_graph_specialize_lora=True)
        request = _make_request_stub(7)
        batch = ScheduledRequests()
        batch.generation_requests = [request]

        self.assertFalse(graphs.use_lora_cuda_graph(batch))

        request.lora_task_id = 42
        self.assertTrue(graphs.use_lora_cuda_graph(batch))

        request.lora_task_id = None
        graphs._config.cuda_graph_specialize_lora = False
        self.assertTrue(graphs.use_lora_cuda_graph(batch))

        graphs._force_lora_graph_for_capture = False
        self.assertFalse(graphs.use_lora_cuda_graph(batch))

    def test_graph_lookup_forwards_promoted_context_ids(self) -> None:
        runner = Mock()
        runner.enabled = True
        runner.config = SimpleNamespace(
            enable_attention_dp=False,
            use_mrope=False,
        )
        key = KeyType(batch_size=1,
                      draft_len=0,
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
                                                     promoted_ids, None, False,
                                                     None)
        self.assertEqual(result,
                         (graph_attn_metadata, graph_spec_metadata, key))

    def test_ced_final_context_cannot_promote_to_decode_graph(self) -> None:
        key = KeyType(batch_size=1, draft_len=0)
        engine, runner, resource_manager, metadata, outputs = \
            _make_forward_only_engine(key, private_decoder=True)
        engine.model.model = SimpleNamespace(ced_kv_precompute=True,
                                             decoder_replay_split=20)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [_make_request_stub(1)]
        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event"
        ), patch(
                "tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs._make_single_token_context_graph_batch"
        ) as promote:
            actual_outputs = engine.forward(batch, resource_manager)
            self.assertEqual(actual_outputs, {
                **outputs, "runtime_draft_len": 0
            })
            self.assertIs(actual_outputs["logits"], outputs["logits"])
        promote.assert_not_called()
        runner.replay.assert_not_called()
        engine._runner._forward_step.assert_called_once()
        self.assertIs(engine._runner._inputs.prepare_inputs.call_args.args[2],
                      metadata)

    def test_zero_runtime_draft_speculation_commits_graph_candidate(
            self) -> None:
        key = KeyType(batch_size=2, draft_len=0)
        engine, runner, resource_manager, semantic_attn_metadata, outputs = \
            _make_forward_only_engine(key)
        engine._runner._config.spec_config = SimpleNamespace(
            spec_dec_mode=_TWO_MODEL_SPEC_MODE,
            is_linear_tree=True,
            get_runtime_tokens_per_gen_step=Mock(return_value=1))
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
            actual_outputs = engine.forward(batch,
                                            resource_manager,
                                            enable_spec_decode=True)

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
        self.assertIsNot(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertEqual(graph_batch.generation_requests, [context, generation])
        self.assertTrue(
            runner.maybe_get_cuda_graph.call_args.kwargs["enable_spec_decode"])
        engine.spec_metadata.update_is_all_greedy_sample.assert_called_once_with(
            graph_batch.all_requests())
        prepare_args = engine._runner._inputs.prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], graph_batch)
        self.assertIs(prepare_args[3], engine.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset({context.py_request_id}))
        semantic_attn_metadata.update_spec_dec_param.assert_called_once()
        self.assertEqual(
            semantic_attn_metadata.update_spec_dec_param.call_args.
            kwargs["num_contexts"], 1)
        prepared_inputs = engine._runner._inputs.prepare_inputs.return_value.kwargs
        runner.replay.assert_called_once_with(key,
                                              prepared_inputs,
                                              position_id_offsets=None,
                                              static_inputs_staged=0)

    def test_zero_runtime_draft_speculation_graph_miss_is_semantic_eager(
            self) -> None:
        engine, runner, resource_manager, semantic_attn_metadata, outputs = \
            _make_forward_only_engine(None)
        engine._runner._config.spec_config = SimpleNamespace(
            spec_dec_mode=_TWO_MODEL_SPEC_MODE,
            is_linear_tree=True,
            get_runtime_tokens_per_gen_step=Mock(return_value=1))
        context = _make_request_stub(1)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch,
                                            resource_manager,
                                            enable_spec_decode=True)

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
        self.assertIsNot(actual_outputs, outputs)
        graph_batch = runner.maybe_get_cuda_graph.call_args.args[0]
        self.assertEqual(graph_batch.generation_requests, [context])
        prepare_args = engine._runner._inputs.prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], batch)
        self.assertIs(prepare_args[2], semantic_attn_metadata)
        self.assertIs(prepare_args[3], engine.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset())
        engine._runner._forward_step.assert_called_once()
        runner.replay.assert_not_called()

    def test_zero_runtime_non_linear_tree_speculation_uses_semantic_eager_batch(
            self) -> None:
        engine, runner, resource_manager, semantic_attn_metadata, outputs = \
            _make_forward_only_engine(None)
        engine._runner._config.spec_config = SimpleNamespace(
            spec_dec_mode=_TWO_MODEL_SPEC_MODE,
            is_linear_tree=False,
            tokens_per_gen_step=8,
            get_runtime_tokens_per_gen_step=Mock(return_value=1))
        context = _make_request_stub(1)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch,
                                            resource_manager,
                                            enable_spec_decode=True)

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 0})
        self.assertIsNot(actual_outputs, outputs)
        lookup = runner.maybe_get_cuda_graph.call_args
        self.assertIs(lookup.args[0], batch)
        self.assertEqual(lookup.kwargs["promoted_context_request_ids"],
                         frozenset())
        prepare_args = engine._runner._inputs.prepare_inputs.call_args.args
        self.assertIs(prepare_args[0], batch)
        self.assertIs(prepare_args[2], semantic_attn_metadata)
        self.assertIs(prepare_args[3], engine.spec_metadata)
        self.assertEqual(prepare_args[-1], frozenset())
        engine._runner._forward_step.assert_called_once()
        runner.replay.assert_not_called()

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
        self.assertIs(engine._runner._inputs.prepare_inputs.call_args.args[0],
                      batch)
        self.assertIs(context.py_multimodal_data, multimodal_data)
        self.assertIn("multimodal_embedding", multimodal_data)

    def test_breakable_graph_falls_back_for_context_logits(self) -> None:
        for return_context_logits in (True, False):
            with self.subTest(return_context_logits=return_context_logits):
                engine, _, resource_manager, _, outputs = \
                    _make_forward_only_engine(None)
                breakable_runner = Mock()
                breakable_runner.is_capturing = False
                breakable_runner.is_warming_up = False
                breakable_runner.has_graph.return_value = True
                breakable_runner.execute.return_value = outputs
                engine._runner._graphs.breakable_cuda_graph_runner = breakable_runner
                prepared = engine._runner._inputs.prepare_inputs.return_value

                def prepare_prefill_graph_batch(*args, **kwargs):
                    set_per_request_prefill_cuda_graph_flag(True)
                    return prepared

                engine._runner._inputs.prepare_inputs.side_effect = \
                    prepare_prefill_graph_batch
                self.addCleanup(set_per_request_prefill_cuda_graph_flag, False)

                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [_make_request_stub(1)]
                batch.context_requests_last_chunk[
                    0].py_return_context_logits = return_context_logits

                with patch(
                        "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                        return_value=Mock()):
                    actual_outputs = engine.forward(
                        batch,
                        resource_manager,
                    )

                self.assertEqual(actual_outputs, {
                    **outputs, "runtime_draft_len": 0
                })
                self.assertIsNot(actual_outputs, outputs)
                if return_context_logits:
                    breakable_runner.execute.assert_not_called()
                    engine._runner._forward_step.assert_called_once()
                else:
                    breakable_runner.execute.assert_called_once()
                    engine._runner._forward_step.assert_not_called()

    def test_generation_only_forward_does_not_call_new_selector(self) -> None:
        key = KeyType(batch_size=1, draft_len=0)
        engine, runner, resource_manager, _, _ = _make_forward_only_engine(key)
        generation = _make_request_stub(2)
        batch = ScheduledRequests()
        batch.generation_requests = [generation]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            engine.forward(batch, resource_manager)

        lookup = runner.maybe_get_cuda_graph.call_args
        self.assertIs(lookup.args[0], batch)
        self.assertEqual(lookup.kwargs["promoted_context_request_ids"],
                         frozenset())
        self.assertFalse(lookup.kwargs["use_lora_graph"])
        self.assertIs(engine._runner._inputs.prepare_inputs.call_args.args[0],
                      batch)
        self.assertEqual(
            engine._runner._inputs.prepare_inputs.call_args.args[-1],
            frozenset())

    def test_generation_lora_request_selects_lora_graph(self) -> None:
        key = KeyType(batch_size=1, draft_len=0, use_lora_graph=True)
        engine, runner, resource_manager, _, _ = _make_forward_only_engine(key)
        engine._runner._lora = engine._runner._graphs._lora = SimpleNamespace(
            cuda_graph_manager=object())
        engine._runner._config.cuda_graph_specialize_lora = True
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
        self.assertTrue(engine._runner._inputs.prepare_inputs.call_args.
                        kwargs["use_lora_graph"])

    def test_global_incompatibilities_bypass_candidate_selection(self) -> None:
        cases = (
            "graphs_disabled",
            "speculative_nonzero_runtime_draft",
            "beam",
            "extra_inputs",
            "ple_recurrent_state",
            "nested_ple_recurrent_state",
            "context_parallel",
            "context_only_cache",
        )
        for case in cases:
            with self.subTest(case=case):
                engine, runner, resource_manager, _, _ = \
                    _make_forward_only_engine(None)
                spec_state = {}
                if case == "graphs_disabled":
                    runner.enabled = False
                elif case == "speculative_nonzero_runtime_draft":
                    spec_state = dict(enable_spec_decode=True,
                                      runtime_draft_len=1)
                    engine._runner._config.spec_config = SimpleNamespace(
                        spec_dec_mode=_TWO_MODEL_SPEC_MODE,
                        is_linear_tree=True,
                        get_runtime_tokens_per_gen_step=Mock(return_value=1))
                elif case == "beam":
                    engine._runner._config.max_beam_width = 2
                elif case == "extra_inputs":
                    engine._runner._graphs._context_promotion_supported = False
                elif case == "ple_recurrent_state":
                    engine.model.has_ple = True
                elif case == "nested_ple_recurrent_state":
                    engine.model.model = SimpleNamespace(llm=SimpleNamespace(
                        model=SimpleNamespace(has_ple=True)))
                elif case == "context_parallel":
                    engine.mapping.cp_size = 2
                elif case == "context_only_cache":
                    resource_manager.get_resource_manager.side_effect = (
                        lambda _: SimpleNamespace(context_swa_layer_limit=20))

                batch = ScheduledRequests()
                batch.context_requests_last_chunk = [_make_request_stub(1)]
                with patch(
                        "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                        return_value=Mock()):
                    engine.forward(batch, resource_manager, **spec_state)

                lookup = runner.maybe_get_cuda_graph.call_args
                self.assertIs(lookup.args[0], batch)
                self.assertEqual(lookup.kwargs["promoted_context_request_ids"],
                                 frozenset())
                self.assertIs(
                    engine._runner._inputs.prepare_inputs.call_args.args[0],
                    batch)

    def test_forward_threads_call_state_and_returns_prepared_length(
            self) -> None:
        engine, runner, resource_manager, _, outputs = \
            _make_forward_only_engine(None)
        engine._runner._config.spec_config = SimpleNamespace(
            spec_dec_mode=_TWO_MODEL_SPEC_MODE,
            is_linear_tree=False,
            tokens_per_gen_step=8,
            get_runtime_tokens_per_gen_step=Mock(return_value=1))
        # A non-linear tree widens the length to its total draft tokens during
        # preparation, after graph selection.
        prepared = engine._runner._inputs.prepare_inputs.return_value
        engine._runner._inputs.prepare_inputs.return_value = replace(
            prepared, runtime_draft_len=7)
        batch = ScheduledRequests()
        batch.generation_requests = [_make_request_stub(2)]

        with patch(
                "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                return_value=Mock()):
            actual_outputs = engine.forward(batch,
                                            resource_manager,
                                            enable_spec_decode=True,
                                            runtime_draft_len=2)

        self.assertEqual(actual_outputs, {**outputs, "runtime_draft_len": 7})
        self.assertEqual(runner.pad_batch.call_args.args[2], 2)
        self.assertTrue(
            runner.maybe_get_cuda_graph.call_args.kwargs["enable_spec_decode"])
        prepare_kwargs = engine._runner._inputs.prepare_inputs.call_args.kwargs
        self.assertEqual(
            (prepare_kwargs["enable_spec_decode"],
             prepare_kwargs["runtime_draft_len"], prepare_kwargs["is_dummy"]),
            (True, 2, False))
        step_kwargs = engine._runner._forward_step.call_args.kwargs
        self.assertEqual(
            (step_kwargs["enable_spec_decode"],
             step_kwargs["runtime_draft_len"], step_kwargs["is_dummy"]),
            (True, 7, False))


class DecoderForwardTestCase(unittest.TestCase):
    """Drive the decoder runner through `engine.forward`.

    The runner, KV cache manager and requests are real; only the CUDA graph
    runner is stubbed. Assertions read what the model or graph replay received.
    """

    def _make_engine(
        self,
        model: DummyModel | None = None,
    ) -> tuple[ModelEngine, KVCacheManagerV2, ResourceManager, Mock]:
        engine, kv_cache_manager = create_model_engine_and_kvcache(model=model)
        self.addCleanup(kv_cache_manager.shutdown)
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        graph_runner = _install_graph_runner_stub(engine._runner)
        return engine, kv_cache_manager, resource_manager, graph_runner

    @staticmethod
    def _forward(engine: ModelEngine,
                 kv_cache_manager: KVCacheManagerV2,
                 resource_manager: ResourceManager,
                 batch: ScheduledRequests,
                 new_tensors_device=None):
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        return engine.forward(batch, resource_manager, new_tensors_device)

    @staticmethod
    def _request(tokens: list[int], req_id: int, seq_slot: int) -> LlmRequest:
        request = _create_request_with_tokens(tokens, req_id)
        request.py_seq_slot = seq_slot
        return request

    def _prefill(self, engine, kv_cache_manager, resource_manager, *requests:
                 LlmRequest) -> None:
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = list(requests)
        self._forward(engine, kv_cache_manager, resource_manager, batch)

    def _prefill_all_but_last_token(self, engine, kv_cache_manager,
                                    resource_manager,
                                    request: LlmRequest) -> None:
        """Leave `request` with a final one-token context chunk."""
        batch = ScheduledRequests()
        batch.context_requests_chunking = [request]
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        # Like the scheduler, narrow the chunk after V2 spans it to the prompt end.
        request.context_chunk_size = request.py_prompt_len - 1
        engine.forward(batch, resource_manager)
        request.move_to_next_context_chunk()
        request.context_chunk_size = 1

    def _final_context_and_generation(self, engine, kv_cache_manager,
                                      resource_manager):
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        generation = self._request([50, 51, 52, 53, 54], 2, seq_slot=1)
        self._prefill_all_but_last_token(engine, kv_cache_manager,
                                         resource_manager, context)
        self._prefill(engine, kv_cache_manager, resource_manager, generation)
        generation.add_new_token(60, 0)
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        batch.generation_requests = [generation]
        return context, generation, batch

    def test_forward_commits_candidate_only_on_graph_hit(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = \
            self._make_engine()
        context, generation, batch = self._final_context_and_generation(
            engine, kv_cache_manager, resource_manager)
        num_model_calls = len(engine.model.recorded_kwargs)
        graph_runner.key = KeyType(batch_size=2, draft_len=0)

        outputs = self._forward(engine, kv_cache_manager, resource_manager,
                                batch)

        lookup = graph_runner.maybe_get_cuda_graph.call_args
        self.assertIsNot(lookup.args[0], batch)
        self.assertEqual(lookup.args[0].generation_requests,
                         [context, generation])
        self.assertEqual(lookup.kwargs["promoted_context_request_ids"],
                         frozenset({context.py_request_id}))
        graph_runner.replay.assert_called_once()
        self.assertEqual(graph_runner.replay.call_args.args[0],
                         graph_runner.key)
        replayed = graph_runner.replayed_inputs[0]
        self.assertEqual(replayed["input_ids"].tolist(), [44, 60])
        self.assertEqual(replayed["position_ids"].flatten().tolist(), [3, 5])
        self.assertEqual(len(engine.model.recorded_kwargs), num_model_calls)
        self.assertEqual(set(outputs), {"logits", "runtime_draft_len"})
        self.assertEqual(engine.attn_metadata.num_contexts, 0)
        self.assertEqual(engine.iter_states['num_ctx_requests'], 1)
        self.assertEqual(engine.iter_states['num_ctx_tokens'], 1)
        self.assertEqual(engine.iter_states['num_generation_tokens'], 1)

    def test_forward_graph_miss_uses_semantic_eager_batch(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = \
            self._make_engine()
        context, generation, batch = self._final_context_and_generation(
            engine, kv_cache_manager, resource_manager)

        self._forward(engine, kv_cache_manager, resource_manager, batch)

        lookup = graph_runner.maybe_get_cuda_graph.call_args
        self.assertEqual(lookup.args[0].generation_requests,
                         [context, generation])
        graph_runner.replay.assert_not_called()
        model_kwargs = engine.model.recorded_kwargs[-1]
        self.assertEqual(model_kwargs["input_ids"].tolist(), [44, 60])
        self.assertEqual(model_kwargs["position_ids"].flatten().tolist(),
                         [3, 5])
        self.assertEqual(engine.attn_metadata.num_contexts, 1)

    def test_forward_allows_guided_context_logits_on_graph_hit(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = \
            self._make_engine()
        engine._runner.guided_decoder = Mock()
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        self._prefill_all_but_last_token(engine, kv_cache_manager,
                                         resource_manager, context)
        context.py_return_context_logits = True
        batch = ScheduledRequests()
        batch.context_requests_last_chunk = [context]
        graph_runner.key = KeyType(batch_size=1, draft_len=0)

        self._forward(engine, kv_cache_manager, resource_manager, batch)

        lookup = graph_runner.maybe_get_cuda_graph.call_args
        self.assertEqual(lookup.args[0].generation_requests, [context])
        self.assertEqual(lookup.kwargs["promoted_context_request_ids"],
                         frozenset({context.py_request_id}))
        graph_runner.replay.assert_called_once()
        self.assertEqual(graph_runner.replayed_inputs[0]["input_ids"].tolist(),
                         [44])

    def test_promoted_context_uses_prompt_token_during_overlap(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = \
            self._make_engine()
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        generation = self._request([50, 51, 52, 53, 54], 2, seq_slot=1)
        self._prefill_all_but_last_token(engine, kv_cache_manager,
                                         resource_manager, context)
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

        self._forward(engine, kv_cache_manager, resource_manager, batch,
                      SimpleNamespace(new_tokens=new_tokens))

        replayed = graph_runner.replayed_inputs[0]
        self.assertEqual(replayed["input_ids"].tolist(), [44, 777])
        self.assertEqual(replayed["position_ids"].flatten().tolist(), [3, 5])
        attn_metadata = engine.attn_metadata
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [2, 5])
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
            new_tokens = torch.tensor(tokens, dtype=torch.int32,
                                      device="cuda").view(1, 2, 1)
            self._forward(engine, kv_cache_manager, resource_manager, batch,
                          SimpleNamespace(new_tokens=new_tokens))
            model_kwargs = engine.model.recorded_kwargs[-1]
            with self.subTest(step=step):
                self.assertEqual(model_kwargs["input_ids"].tolist(),
                                 list(tokens))
                self.assertEqual(
                    model_kwargs["position_ids"].flatten().tolist(),
                    [4 + step, 6 + step])
            # The sampler appends the previous step's token after the next
            # step has been scheduled.
            for request, token in zip(requests, tokens):
                request.add_new_token(token, 0)

    def _make_mrope_engine(self):
        return self._make_engine(
            DummyModel(torch.half, rope_scaling={"type": "mrope"}))

    def _mrope_generation(self, engine, kv_cache_manager, resource_manager,
                          num_tokens: int, req_id: int, seq_slot: int,
                          delta: int | None) -> LlmRequest:
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
                    "mrope_position_deltas":
                    torch.tensor([[delta]], dtype=torch.int32)
                },
            }
        return request

    def _forward_generation(self, engine, kv_cache_manager, resource_manager,
                            requests: list[LlmRequest]) -> dict:
        batch = ScheduledRequests()
        batch.generation_requests = requests
        self._forward(engine, kv_cache_manager, resource_manager, batch)
        return engine.model.recorded_kwargs[-1]

    def test_mrope_generation_with_partial_segments(self) -> None:
        """A real multimodal span next to a padding dummy without MRoPE data."""
        engine, kv_cache_manager, resource_manager, _ = \
            self._make_mrope_engine()
        multimodal = self._mrope_generation(engine, kv_cache_manager,
                                            resource_manager, 4, 1, 0, 10)
        multimodal.py_multimodal_data["multimodal_embedding"] = torch.ones(
            (1, 1), dtype=torch.float16)
        # A generation-only admission has no batch slot yet, which is when
        # the context-phase payload is stripped.
        multimodal.py_batch_idx = None
        dummy = self._mrope_generation(engine, kv_cache_manager,
                                       resource_manager, 6, 2, 1, None)
        dummy.is_cuda_graph_dummy = True

        model_kwargs = self._forward_generation(engine, kv_cache_manager,
                                                resource_manager,
                                                [multimodal, dummy])

        expected = torch.tensor([[[13, 5]]] * 3,
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(model_kwargs["position_ids"],
                                   expected,
                                   atol=0,
                                   rtol=0)
        self.assertEqual(
            model_kwargs["mrope_delta_write_seq_slots"].cpu().tolist(), [0])
        # Read slots are dense w.r.t. the generation batch: the padded dummy
        # has no MRoPE metadata, so it resolves to the reserved zero slot
        # (max_num_tokens * pp_size) rather than being dropped, which would
        # shift every later request onto another request's delta.
        self.assertEqual(
            model_kwargs["mrope_delta_read_seq_slots"].cpu().tolist(),
            [0, engine.max_num_tokens])
        self.assertNotIn("multimodal_embedding", multimodal.py_multimodal_data)

    def test_mrope_mixed_text_only_keeps_deltas_dense(self) -> None:
        """A text-only request between two multimodal ones must not compact the
        MRoPE delta read slots.

        The attention kernel indexes `mrope_position_deltas` by generation batch
        index, so a list that skips the text-only request would hand request 2's
        delta to the text-only request and read out of bounds for request 2.
        """
        engine, kv_cache_manager, resource_manager, _ = \
            self._make_mrope_engine()
        # (num_tokens, req_id, seq_slot, delta); the middle one is text-only.
        requests = [
            self._mrope_generation(engine, kv_cache_manager, resource_manager,
                                   *args)
            for args in ((4, 1, 0, 10), (5, 2, 1, None), (6, 3, 2, 20))
        ]

        model_kwargs = self._forward_generation(engine, kv_cache_manager,
                                                resource_manager, requests)

        # One entry per generation request, in batch order; the reserved zero
        # slot stands in for the text-only request's zero delta.
        self.assertEqual(
            model_kwargs["mrope_delta_read_seq_slots"].cpu().tolist(),
            [0, engine.max_num_tokens, 2])
        # Only the two multimodal requests seed the seq-slot delta cache.
        self.assertEqual(
            model_kwargs["mrope_delta_write_seq_slots"].cpu().tolist(), [0, 2])
        # The position is num_tokens - 1, offset by the request's delta; the
        # text-only request keeps the plain scalar position on all 3 axes.
        expected = torch.tensor([[[13, 4, 25]]] * 3,
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(model_kwargs["position_ids"],
                                   expected,
                                   atol=0,
                                   rtol=0)

    def test_mrope_all_text_only_drops_deltas(self) -> None:
        """A generation batch with no MRoPE metadata at all emits no delta
        tensors, so the steady-state generation fast path stays reachable."""
        engine, kv_cache_manager, resource_manager, _ = \
            self._make_mrope_engine()
        requests = [
            self._mrope_generation(engine, kv_cache_manager, resource_manager,
                                   4, 1, 0, None),
            self._mrope_generation(engine, kv_cache_manager, resource_manager,
                                   6, 2, 1, None),
        ]

        model_kwargs = self._forward_generation(engine, kv_cache_manager,
                                                resource_manager, requests)

        # An all-zero delta vector is identical to passing no deltas at all.
        self.assertNotIn("mrope_delta_read_seq_slots", model_kwargs)
        self.assertNotIn("mrope_delta_write_seq_slots", model_kwargs)
        # MRoPE models keep the (3,1,N) layout even with no mrope work.
        expected = torch.tensor([[[3, 5]]] * 3,
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(model_kwargs["position_ids"],
                                   expected,
                                   atol=0,
                                   rtol=0)

    def test_promoted_mrope_context_uses_decode_state_contract(self) -> None:
        engine, kv_cache_manager, resource_manager, graph_runner = \
            self._make_mrope_engine()
        engine.model.config.vocab_size = 100
        context = self._request([11, 22, 33, 44], 1, seq_slot=0)
        self._prefill_all_but_last_token(engine, kv_cache_manager,
                                         resource_manager, context)
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
            graph_runner.maybe_get_cuda_graph.call_args.
            kwargs["promoted_context_request_ids"],
            frozenset({context.py_request_id}))
        replayed = graph_runner.replayed_inputs[0]
        self.assertEqual(replayed["input_ids"].tolist(), [44])
        expected_positions = torch.full((3, 1, 1),
                                        13,
                                        dtype=torch.int32,
                                        device="cuda")
        torch.testing.assert_close(replayed["position_ids"],
                                   expected_positions,
                                   atol=0,
                                   rtol=0)
        self.assertEqual(replayed["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [0])
        self.assertNotIn("mrope_delta_write_seq_slots", replayed)
        attn_metadata = engine.attn_metadata
        self.assertEqual(
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [3])
        self.assertEqual(attn_metadata.num_contexts, 0)


class PyTorchModelEngineTestCase(unittest.TestCase):

    def test_warmup_builds_dsa_attn_metadata_when_no_forward_ran(self) -> None:
        """The DSA top-K pre-compile hooks read attn_metadata; when every
        warmup forward was skipped it is built on demand so the engines still
        compile before serving."""
        from tensorrt_llm._torch.attention.backends.sparse.dsa import \
            DSAtrtllmAttentionMetadata

        engine = object.__new__(DecoderWarmup)
        engine._config = SimpleNamespace(
            attention_backend=SimpleNamespace(
                Metadata=DSAtrtllmAttentionMetadata),
            original_max_draft_len=2,
            cuda_graph_batch_sizes=[8],
            spec_config=None,
        )
        engine._metadata = DecoderMetadata(config=engine._config,
                                           mapping=None,
                                           model=None,
                                           buffers=None,
                                           attention_runtime_features=None)
        metadata = Mock(spec=DSAtrtllmAttentionMetadata)

        def build(kv_cache_manager, draft_kv_cache_manager):
            engine._metadata.attn_metadata = metadata
            return metadata

        engine._metadata.setup_attention = Mock(side_effect=build)
        kv_cache_manager = Mock()
        resource_manager = Mock()
        resource_manager.get_resource_manager.return_value = kv_cache_manager

        engine._ensure_dsa_attn_metadata_for_warmup(resource_manager)
        engine._warmup_cute_dsl_radix_topk()

        engine._metadata.setup_attention.assert_called_once_with(
            kv_cache_manager, None)
        metadata.warmup_cute_dsl_radix_topk.assert_called_once_with(3)
        metadata.warmup_selfsampling_topk.assert_called_once_with(
            3, batch_sizes=[8])

    def test_warmup_does_not_build_metadata_for_non_dsa_backend(self) -> None:
        from tensorrt_llm._torch.attention.backends.trtllm import \
            TrtllmAttentionMetadata

        engine = object.__new__(DecoderWarmup)
        engine._config = SimpleNamespace(attention_backend=SimpleNamespace(
            Metadata=TrtllmAttentionMetadata))
        engine._metadata = DecoderMetadata(config=engine._config,
                                           mapping=None,
                                           model=None,
                                           buffers=None,
                                           attention_runtime_features=None)
        engine._metadata.setup_attention = Mock()

        engine._ensure_dsa_attn_metadata_for_warmup(Mock())
        engine._warmup_cute_dsl_radix_topk()

        engine._metadata.setup_attention.assert_not_called()

    def test_breakable_rejects_multimodal_models(self) -> None:
        engine = object.__new__(ModelEngine)
        engine.model = DummyLegacyMultimodalIndexModel()
        engine.input_processor = None
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            disable_mm_encoder=False)

        self.assertTrue(is_multimodal(engine.model, engine.input_processor))
        with self.assertRaisesRegex(ValueError, "multimodal models"):
            engine._validate_breakable_cuda_graph_compatibility()

    def test_breakable_allows_text_decoder_with_multimodal_processor(
            self) -> None:
        engine = object.__new__(ModelEngine)
        engine.model = Mock(spec=DecoderModelForCausalLM)
        engine.input_processor = Mock(spec=BaseMultimodalInputProcessor)
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            disable_mm_encoder=False)

        self.assertTrue(is_multimodal(engine.model, engine.input_processor))
        engine._validate_breakable_cuda_graph_compatibility()

    def test_breakable_allows_multimodal_wrapper_in_text_only_mode(
            self) -> None:
        engine = object.__new__(ModelEngine)
        engine.model = DummyLegacyMultimodalIndexModel()
        engine.model.llm = Mock(spec=DecoderModelForCausalLM)
        engine.model.mm_encoder = None
        engine.input_processor = Mock(spec=BaseMultimodalInputProcessor)
        engine.llm_args = SimpleNamespace(
            prefill_cuda_graph_backend=PrefillCudaGraphBackend.BREAKABLE,
            disable_mm_encoder=True)

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
            llm_args=SimpleNamespace(torch_compile_config=SimpleNamespace(
                enable_fullgraph=False)),
            cuda_graph_runner=cuda_graph_runner,
            _apply_torch_compile=Mock(),
            _capture_piecewise_cuda_graphs=Mock(),
        )

        with patch("torch.compiler.reset") as reset, \
                patch("gc.collect") as collect:
            ModelEngine.restore_compiled_model_after_refit(
                model_engine, resource_manager=object())

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
        ModelEngine.restore_compiled_model_after_refit(
            model_engine, resource_manager=object())
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
                        torch_compile_config=SimpleNamespace(
                            enable_fullgraph=False)),
                    _apply_torch_compile=Mock(),
                )
                with patch("gc.collect"):
                    ModelEngine.restore_compiled_model_after_refit(
                        model_engine, resource_manager)
                resource_manager.get_resource_manager.assert_not_called()
                self.assertEqual(model_engine._apply_torch_compile.call_count,
                                 1 if compile_enabled else 0)

    def test_unwrap_for_refit_gates_on_torch_compile_not_piecewise(
            self) -> None:
        """Compile-on/piecewise-off must still unwrap.

        This is the original defect: the hook was gated on the piecewise flag,
        so TorchCompileConfig() -- compile on, piecewise off -- got no refit
        lifecycle and refit loaded into a wrapped model, matching nothing.
        """
        for piecewise in (True, False):
            with self.subTest(piecewise=piecewise):
                model_engine = SimpleNamespace(
                    llm_args=SimpleNamespace(prefill_cuda_graph_backend=(
                        PrefillCudaGraphBackend.PIECEWISE if piecewise else None
                    )),
                    _torch_compile_backend=Mock(),
                    _remove_torch_compile=Mock(),
                )
                unwrap_core = (ModelEngine.unwrap_compiled_model_for_refit)
                with patch("torch.cuda.synchronize"):
                    unwrap_core(model_engine)
                model_engine._remove_torch_compile.assert_called_once()
                # The captures are kept, not cleared: refit moves no tensor, so
                # clearing them only forces PWCG cold for no benefit.
                model_engine._torch_compile_backend \
                    .clear_piecewise_cuda_graphs.assert_not_called()

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
        self.assertIs(ModelEngine.release_piecewise_cuda_graphs_for_refit,
                      ModelEngine.unwrap_compiled_model_for_refit)
        self.assertIs(ModelEngine.recapture_piecewise_cuda_graphs_after_refit,
                      ModelEngine.restore_compiled_model_after_refit)

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

    def test_decoder_buffers_are_allocated_only_for_decoder_runner(
            self) -> None:
        llm_args = TorchLlmArgs(
            model="dummy",
            max_batch_size=4,
            max_num_tokens=64,
            max_beam_width=2,
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4]))
        for is_generation in (True, False):
            with self.subTest(is_generation=is_generation):
                engine = DummyModelEngine(llm_args,
                                          torch.half,
                                          is_generation=is_generation)

                runner = engine._runner
                if is_generation:
                    self.assertIsInstance(runner, DecoderRunner)
                    self.assertIsNotNone(engine.cuda_graph_runner)
                    self.assertEqual(
                        tuple(runner._buffers.cache_indirection.shape),
                        (4, 2, engine.max_seq_len))
                else:
                    self.assertIsInstance(runner, PoolingRunner)
                    self.assertIsNone(engine.cuda_graph_runner)
                self.assertEqual(
                    isinstance(runner._buffers, DecoderInputBuffers),
                    is_generation)

    def test_no_kv_cache_runners_reject_speculative_config(self) -> None:
        for mm_encoder_only, runner_name in ((False, "PoolingRunner"),
                                             (True, "MultimodalEncoderRunner")):
            with self.subTest(runner=runner_name):
                llm_args = TorchLlmArgs(model="dummy",
                                        max_batch_size=4,
                                        max_num_tokens=64,
                                        mm_encoder_only=mm_encoder_only)
                with self.assertRaisesRegex(
                        ValueError,
                        f"{runner_name} does not support speculative decoding"):
                    DummyModelEngine(
                        llm_args,
                        torch.half,
                        spec_config=SADecodingConfig(max_draft_len=1),
                        is_generation=False)

    def test_runner_engine_shared_entries_skip_decoder_state(self) -> None:
        runner = Mock()
        engine = object.__new__(ModelEngine)
        engine._runner = runner
        engine._torch_compile_backend = None

        engine.wait_for_input_copy()
        engine.release_executor_resources(release_model_owned_graphs=False)
        engine.init_cuda_graph_lora_manager(Mock())

        runner.wait_for_input_copy.assert_called_once_with()
        runner.release_graphs.assert_called_once_with(
            release_nccl_window_owners=True)
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

        decoder._inputs._prepare_inputs_event.synchronize.assert_called_once_with(
        )
        decoder.cuda_graph_runner.clear.assert_called_once_with(
            release_nccl_window_owners=True)
        decoder._graphs.breakable_cuda_graph_runner.clear.assert_called_once_with(
            release_nccl_window_owners=True)
        self.assertIsNone(decoder.attn_metadata)

    def test_prepare_inputs_returns_steady_gen_fast_inputs(self) -> None:
        inputs = _input_preparer()
        inputs._config = SimpleNamespace(spec_config=None)
        inputs.guided_decoder = Mock()
        inputs._can_use_steady_gen_fast_prepare = Mock(return_value=True)
        fast_inputs = PreparedInputs({"prepared": True})
        inputs._apply_steady_gen_fast_prepare = Mock(return_value=fast_inputs)
        batch = ScheduledRequests()
        new_tokens = torch.zeros(1)

        result = inputs.prepare_inputs(
            batch,
            Mock(),
            Mock(),
            new_tensors_device=SimpleNamespace(new_tokens=new_tokens),
            enable_spec_decode=False,
            runtime_draft_len=3,
            is_dummy=True)

        self.assertIs(result.kwargs, fast_inputs.kwargs)
        self.assertEqual(result.runtime_draft_len, 3)
        inputs.guided_decoder.add_batch.assert_called_once_with(
            batch, new_tokens=new_tokens, runtime_draft_len=3)
        inputs._can_use_steady_gen_fast_prepare.assert_called_once_with(
            batch, new_tokens, None, None, True)

    def test_prepare_inputs_reports_tree_draft_length(self) -> None:
        inputs = _input_preparer()
        inputs._config = SimpleNamespace(spec_config=SimpleNamespace(
            is_linear_tree=False, tokens_per_gen_step=8))
        inputs._can_use_steady_gen_fast_prepare = Mock(return_value=True)
        inputs._apply_steady_gen_fast_prepare = Mock(
            return_value=PreparedInputs({}))

        for enable_spec_decode, expected in ((True, 7), (False, 2)):
            with self.subTest(enable_spec_decode=enable_spec_decode):
                result = inputs.prepare_inputs(
                    ScheduledRequests(),
                    Mock(),
                    Mock(),
                    enable_spec_decode=enable_spec_decode,
                    runtime_draft_len=2,
                    is_dummy=True)
                self.assertEqual(result.runtime_draft_len, expected)

    def test_steady_gen_fast_prepare_skips_dummy_passes(self) -> None:
        inputs = _input_preparer()
        request = _make_request_stub(7)
        inputs._steady_gen_cache = {
            "num_requests": 1,
            "request_ids": [request.py_request_id],
        }
        batch = ScheduledRequests()
        batch.generation_requests = [request]
        new_tokens = torch.zeros(1)

        self.assertTrue(
            inputs._can_use_steady_gen_fast_prepare(batch, new_tokens, None,
                                                    None, False))
        self.assertFalse(
            inputs._can_use_steady_gen_fast_prepare(batch, new_tokens, None,
                                                    None, True))

    def test_padding_request_input_id_rows_are_zeroed(self) -> None:
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=32,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        generation = _create_request_with_tokens([50, 51, 52, 53, 54], 1)
        generation.py_seq_slot = 0
        generation.py_batch_idx = 0
        padding = _create_request_with_tokens([60, 61, 62], 2)
        padding.is_cuda_graph_dummy = True
        padding.py_seq_slot = 1

        graph_batch = ScheduledRequests()
        graph_batch.generation_requests = [generation, padding]
        new_tokens = torch.zeros((1, 4, 1), dtype=torch.int32, device="cuda")
        new_tokens[0, 0, 0] = 777
        overlap_state = SimpleNamespace(new_tokens=new_tokens)
        inputs = model_engine._runner._inputs
        inputs._can_use_steady_gen_fast_prepare = Mock(return_value=False)
        # The padding row is written by neither input-id copy, so poison the
        # buffer with an out-of-vocabulary id to catch a stale read.
        model_engine._runner._buffers.input_ids_cuda.fill_(1 << 30)

        prepared = inputs.prepare_inputs(
            scheduled_requests=graph_batch,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            new_tensors_device=overlap_state,
            resource_manager=resource_manager,
            enable_spec_decode=False,
            runtime_draft_len=0,
            is_dummy=False,
        )

        self.assertEqual(prepared.kwargs["input_ids"][:2].cpu().tolist(),
                         [777, 0])
        kv_cache_manager.shutdown()

    def test_promoted_context_precedes_speculative_overlap_generation(
            self) -> None:
        spec_config = SADecodingConfig(
            max_draft_len=1,
            draft_len_schedule={1: 1},
        )
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            spec_config=spec_config)
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=32,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False
        # A bare Mock auto-vivifies every attribute, so optional metadata
        # fields read by prepare_inputs must be pinned off explicitly.
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

        prepared = model_engine._runner._inputs.prepare_inputs(
            scheduled_requests=graph_batch,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            new_tensors_device=overlap_state,
            resource_manager=resource_manager,
            promoted_context_request_ids=frozenset({context.py_request_id}),
            enable_spec_decode=True,
            runtime_draft_len=0,
            is_dummy=False,
        )

        inputs = prepared.kwargs
        self.assertEqual(inputs["input_ids"][:2].cpu().tolist(), [44, 777])
        self.assertEqual(inputs["position_ids"][0, :2].cpu().tolist(), [3, 4])
        self.assertEqual(attn_metadata.request_ids,
                         [context.py_request_id, generation.py_request_id])
        self.assertEqual(
            attn_metadata.kv_cache_params.num_cached_tokens_per_seq, [3, 5])
        self.assertEqual(
            model_engine._runner._buffers.previous_batch_indices_cuda[:1].cpu().
            tolist(), [1])
        self.assertEqual(
            model_engine._runner._buffers.previous_pos_id_offsets_cuda[:2].cpu(
            ).tolist(), [0, 1])
        self.assertEqual(attn_metadata.num_contexts, 0)
        self.assertEqual(spec_metadata.request_ids,
                         [context.py_request_id, generation.py_request_id])
        self.assertFalse(generation.py_needs_onehot_draft_probs)
        spec_metadata.write_padding_onehot_draft_probs.assert_called_once_with(
            [generation.py_seq_slot], 0)
        kv_cache_manager.shutdown()

    def test_spec_decode_graph_step_gather_kernel_matches_torch_path(
            self) -> None:
        """A CUDA graph decode step of a speculative engine whose overlap
        gathers run as one StepInputGather launch writes what the torch path
        writes."""
        from tensorrt_llm._torch.attention.backends.trtllm import \
            TrtllmAttentionMetadata
        from tensorrt_llm._torch.kernels.spec_step_copies import \
            op as spec_step_copies
        if not spec_step_copies.is_supported():
            self.skipTest("the step-copy kernels run on SM 100")
        max_draft_len = 3
        tokens_per_step = max_draft_len + 1
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            spec_config=SADecodingConfig(max_draft_len=max_draft_len))
        inputs = model_engine._runner._inputs
        buffers = model_engine._runner._buffers
        step_gather = inputs._step_input_gather
        self.assertIsNotNone(step_gather)
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        # Three requests that ran in the previous step, in slots out of order.
        slots = [5, 2, 7]
        requests = kv_cache_manager.add_dummy_requests(
            [1, 2, 3],
            token_nums=[10, 17, 33],
            is_gen=True,
            max_num_draft_tokens=max_draft_len)
        for request, slot in zip(requests, slots):
            request.is_dummy_request = False
            request.py_seq_slot = slot
        batch = ScheduledRequests()
        batch.generation_requests = requests
        attn_metadata = model_engine._runner._metadata.setup_attention(
            kv_cache_manager)
        self.assertIs(type(attn_metadata), TrtllmAttentionMetadata)
        graph_metadata = attn_metadata.create_cuda_graph_metadata(
            len(requests), False, max_draft_len)
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
            supports_steady_prepare=False,
        )

        # The sampler's slot stores as the previous step left them.
        num_slots = 8
        generator = torch.Generator(device="cuda").manual_seed(0)
        previous = SampleStateTensorsSpec(
            new_tokens=torch.randint(0,
                                     1000, (tokens_per_step, num_slots, 1),
                                     generator=generator,
                                     dtype=torch.int32,
                                     device="cuda"),
            new_tokens_lens=torch.randint(1,
                                          tokens_per_step + 1, (num_slots, ),
                                          generator=generator,
                                          dtype=torch.int32,
                                          device="cuda"),
            next_draft_tokens=torch.randint(0,
                                            1000, (num_slots, max_draft_len),
                                            generator=generator,
                                            dtype=torch.int32,
                                            device="cuda"),
        )

        def step_inputs(step_input_gather):
            """The buffers one step writes, from a filler they all start at."""
            inputs._step_input_gather = step_input_gather
            for request, slot in zip(requests, slots):
                request.py_batch_idx = slot
            written = (buffers.input_ids_cuda, buffers.position_ids_cuda,
                       buffers.draft_tokens_cuda,
                       buffers.previous_pos_id_offsets_cuda,
                       buffers.previous_kv_lens_offsets_cuda)
            for buffer in written:
                buffer.fill_(-3)
            inputs.prepare_inputs(scheduled_requests=batch,
                                  kv_cache_manager=kv_cache_manager,
                                  attn_metadata=graph_metadata,
                                  spec_metadata=spec_metadata,
                                  new_tensors_device=previous,
                                  resource_manager=resource_manager,
                                  enable_spec_decode=True,
                                  runtime_draft_len=max_draft_len,
                                  is_dummy=False)
            torch.cuda.synchronize()
            return [buffer.clone() for buffer in written]

        torch_path = step_inputs(None)
        launched = []
        launch = step_gather.gather

        def recorded_launch(*args, **kwargs):
            launched.append(launch(*args, **kwargs))
            return launched[-1]

        with patch.object(step_gather, "gather", side_effect=recorded_launch):
            kernel_path = step_inputs(step_gather)
        self.assertEqual(launched, [True])
        for expected, actual in zip(torch_path, kernel_path):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The previous rows' input ids came from the stores, by slot.
        num_tokens = len(requests) * tokens_per_step
        expected_input_ids = previous.new_tokens[:, slots, 0].t().reshape(-1)
        self.assertEqual(kernel_path[0][:num_tokens].tolist(),
                         expected_input_ids.tolist())
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

        def free_blocks() -> int:
            return kv_cache_manager.get_kv_cache_stats().free_num_blocks

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

            pages_before = free_blocks()
            with model_engine.cuda_graph_runner.pad_batch(
                    batch, resource_manager) as padded_batch:
                # No padding for prefill
                self.assertIs(batch, padded_batch)
            self.assertEqual(free_blocks(), pages_before)

            batch = ScheduledRequests()
            batch.generation_requests = requests
            pages_before = free_blocks()
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
            self.assertEqual(free_blocks() + new_dummy_block, pages_before)

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
        _prepare_kv_cache_resources(kv_cache_manager, batch)
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
        _prepare_kv_cache_resources(kv_cache_manager, batch)

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
        # short_seq_len_mode, is_all_greedy_sample) exist only for draft_len 2.
        runner.graphs[(8, 2, False, True)] = Mock()
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
        _prepare_kv_cache_resources(kv_cache_manager, batch)
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

        self.assertEqual(model_engine.cuda_graph_batch_sizes,
                         [1, 2, 3, model_engine.max_seq_len])
        self.assertTrue(model_engine.cuda_graph_padding_enabled)

    def test_forward_pass_callable_on_cuda_graph_on(self):
        llm_args = TorchLlmArgs(model="dummy",
                                cuda_graph_config=CudaGraphConfig(
                                    enable_padding=True, ))
        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            llm_args)
        self.addCleanup(kv_cache_manager.shutdown)

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_forward_pass_callable_on_cuda_graph_off(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        self.addCleanup(kv_cache_manager.shutdown)

        mock_callable = Mock()
        model_engine.register_forward_pass_callable(mock_callable)

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

        mock_callable.assert_called_once()

    def test_foward_pass_callable_off(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        self.addCleanup(kv_cache_manager.shutdown)
        self.assertTrue(model_engine._runner.forward_pass_callable is None,
                        "forward_pass_callback should be None by default")

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

    def test_foward_pass_callable_backward_compat(self):
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        self.addCleanup(kv_cache_manager.shutdown)
        self.assertTrue(model_engine._runner.forward_pass_callable is None,
                        "forward_pass_callback should be None by default")

        # Assert we can run `forward` without a forward_pass_callback without error
        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        model_engine.forward(batch, resource_manager)

    @skip_ray
    def test_prepare_inputs_with_helix_parallelism(self) -> None:
        """Test prepare_inputs with helix parallelism."""

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
        model_engine._runner._inputs.mapping = mapping

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
        kv_cache_config = KvCacheConfig(max_tokens=512,
                                        enable_block_reuse=False)
        kv_cache_manager = KVCacheManagerV2(
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
        kv_cache_manager.add_dummy_requests([1, 2], prompt_lens, is_gen=True)
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=4,
            max_num_tokens=512,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping)
        attn_metadata.is_cuda_graph = False

        # Initialize model engine buffers.
        max_num_tokens = 512
        model_engine._runner._inputs._config = replace(
            model_engine._runner._inputs._config, max_num_tokens=max_num_tokens)
        _resize_input_buffers(model_engine._runner, max_num_tokens)

        result = model_engine._runner._inputs.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            enable_spec_decode=False,
            runtime_draft_len=0,
            is_dummy=False).kwargs

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

    def _setup_mrope_engine(self,
                            max_num_tokens: int = 32,
                            spec_config: DecodingBaseConfig | None = None):
        """Build a DummyModelEngine that takes the MRoPE path, plus its KV cache
        manager and attention metadata."""
        llm_args = TorchLlmArgs(model="dummy")
        model_engine = DummyModelEngine(llm_args,
                                        dtype=torch.half,
                                        spec_config=spec_config)
        model_engine.model.model_config.pretrained_config.rope_scaling = {
            "type": "mrope"
        }
        _set_runner_config(model_engine._runner, use_mrope=True)

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        kv_cache_config = KvCacheConfig(max_tokens=max_num_tokens)
        kv_cache_manager = KVCacheManagerV2(
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

        _set_runner_config(model_engine._runner, max_num_tokens=max_num_tokens)
        _resize_input_buffers(model_engine._runner, max_num_tokens)
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

    def _make_mrope_spec_batch(self, previous_batch: bool):
        """Two multimodal verify groups around a text-only one, plus a CUDA graph
        padding dummy, each carrying one draft token.

        Returns the scheduled batch and the multimodal requests."""
        # (num_tokens, req_id, seq_slot, delta); the middle request is text-only.
        requests = [
            self._make_mrope_gen_request(4, 1, 0, 10),
            self._make_mrope_gen_request(5, 2, 1, None),
            self._make_mrope_gen_request(6, 3, 2, 20),
        ]
        for request in requests:
            request.py_draft_tokens = [7]
            if previous_batch:
                # The overlap scheduler's previous step ran these requests at
                # their current slots, which also seeded the delta cache.
                request.py_batch_idx = request.py_seq_slot
                if request.py_multimodal_data:
                    request.py_mrope_delta_cache_slot = request.py_seq_slot
        # The padding dummy carries the zero delta that `add_dummy_requests`
        # attaches on MRoPE models.
        dummy = self._make_mrope_gen_request(2, 4, 3, 0)
        dummy.is_cuda_graph_dummy = True
        dummy.py_draft_tokens = [0]

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = requests + [dummy]
        return scheduled_requests, [requests[0], requests[2]]

    def test_prepare_inputs_spec_decode_adds_mrope_delta(self) -> None:
        """One-model speculative verify groups get the request's MRoPE delta.

        Every token a generation step feeds is text after the prompt's vision
        spans, so its three coordinates are `seq_index + delta`, as for an
        ordinary generation request. Read slots stay dense, one per verify
        group, and a generation-only request seeds the delta cache.
        """
        model_engine, kv_cache_manager, attn_metadata = self._setup_mrope_engine(
            spec_config=SADecodingConfig(max_draft_len=1))
        scheduled_requests, multimodal_requests = self._make_mrope_spec_batch(
            previous_batch=False)
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )

        result = model_engine._runner._inputs.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            enable_spec_decode=True,
            runtime_draft_len=1,
            is_dummy=False).kwargs

        # Each verify group is [last token, draft token] at past_seen + 0/1;
        # the text-only request and the zero-delta dummy keep the scalar
        # positions.
        position_ids = result["position_ids"]
        self.assertEqual(tuple(position_ids.shape), (3, 1, 8))
        expected = torch.tensor([[[13, 14, 4, 5, 25, 26, 1, 2]]] * 3,
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(position_ids, expected, atol=0, rtol=0)
        # Slot 32 is the reserved zero slot (max_num_tokens * pp_size).
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [0, 32, 2, 32])
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(),
                         [0, 2])
        self.assertEqual([
            request.py_mrope_delta_cache_slot for request in multimodal_requests
        ], [0, 2])
        kv_cache_manager.shutdown()

    def test_prepare_inputs_spec_decode_overlap_keeps_mrope_delta(self) -> None:
        """Under the overlap scheduler the accepted-token correction applied in
        `preprocess_inputs` lands on top of the delta, on all three axes."""
        model_engine, kv_cache_manager, attn_metadata = self._setup_mrope_engine(
            spec_config=SADecodingConfig(max_draft_len=1))
        scheduled_requests, _ = self._make_mrope_spec_batch(previous_batch=True)
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )
        # The previous step accepted 2, 1 and 2 tokens in slots 0, 1 and 2.
        overlap_state = SampleStateTensorsSpec(
            new_tokens=torch.zeros((2, 4, 1), dtype=torch.int32, device="cuda"),
            new_tokens_lens=torch.tensor([2, 1, 2, 1],
                                         dtype=torch.int32,
                                         device="cuda"),
            next_draft_tokens=torch.zeros((4, 1),
                                          dtype=torch.int32,
                                          device="cuda"),
        )

        inputs = model_engine._runner._inputs.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            new_tensors_device=overlap_state,
            enable_spec_decode=True,
            runtime_draft_len=1,
            is_dummy=False).kwargs

        # The host packs positions from the stale base (past_seen + delta).
        stale = torch.tensor([[[13, 14, 4, 5, 25, 26, 1, 2]]] * 3,
                             dtype=torch.int32,
                             device='cuda')
        torch.testing.assert_close(inputs["position_ids"],
                                   stale,
                                   atol=0,
                                   rtol=0)
        self.assertEqual(inputs["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [0, 32, 2, 32])
        # The delta cache already holds both multimodal requests' deltas.
        self.assertNotIn("mrope_delta_write_seq_slots", inputs)

        inputs = model_engine._runner._inputs.preprocess_inputs(
            inputs, enable_spec_decode=True, runtime_draft_len=1)
        corrected = torch.tensor([[[15, 16, 5, 6, 27, 28, 1, 2]]] * 3,
                                 dtype=torch.int32,
                                 device='cuda')
        torch.testing.assert_close(inputs["position_ids"],
                                   corrected,
                                   atol=0,
                                   rtol=0)
        kv_cache_manager.shutdown()

    @staticmethod
    def _make_mrope_ctx_request(num_tokens: int, req_id: int, seq_slot: int,
                                delta: int):
        """Single-chunk context request carrying MRoPE metadata.

        Its position on axis `a` of token `i` is `100 + 1000 * a + i`, so a
        spliced span says which axis and token it came from."""
        request = _create_request(num_tokens, req_id)
        request.py_prompt_len = num_tokens
        request.py_batch_idx = None
        request.py_seq_slot = seq_slot
        request.context_current_position = 0
        request.context_chunk_size = num_tokens
        request.sampling_config.beam_width = 1
        positions = (100 + 1000 * torch.arange(3).view(3, 1, 1) +
                     torch.arange(num_tokens).view(1, 1, -1)).to(torch.int32)
        request.py_multimodal_data = {
            "mrope_config": {
                "mrope_position_ids": positions,
                "mrope_position_deltas": torch.tensor([[delta]],
                                                      dtype=torch.int32),
            },
        }
        return request

    @staticmethod
    def _prepare_mrope_spec_inputs(model_engine, kv_cache_manager,
                                   attn_metadata, scheduled_requests):
        spec_metadata = Mock(
            _force_non_greedy_for_capture=False,
            context_prompt_lookahead_tokens=None,
        )
        result = model_engine._runner._inputs.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            enable_spec_decode=True,
            runtime_draft_len=1,
            is_dummy=False).kwargs
        return result

    def _assert_mrope_positions(self, result, rows) -> None:
        """`rows` is one list for all three axes, or one list per axis."""
        if not isinstance(rows[0], list):
            rows = [rows] * 3
        position_ids = result["position_ids"]
        self.assertEqual(tuple(position_ids.shape), (3, 1, len(rows[0])))
        expected = torch.tensor([rows[0:1], rows[1:2], rows[2:3]],
                                dtype=torch.int32,
                                device='cuda')
        torch.testing.assert_close(position_ids, expected, atol=0, rtol=0)

    def test_prepare_inputs_spec_decode_mrope_plain_rows_follow_verify_groups(
            self) -> None:
        """Verify groups are packed ahead of plain generation requests, and the
        MRoPE delta read slots follow that packing rather than scheduler order.

        The attention kernel indexes the deltas by generation batch index: row
        i of the packed generation batch must read the delta of the request
        packed i-th, so a verify group's slot comes before every plain
        request's, whatever order the scheduler listed them in.
        """
        model_engine, kv_cache_manager, attn_metadata = self._setup_mrope_engine(
            spec_config=MTPDecodingConfig(max_draft_len=1))
        # `verify_*` carry a draft token, `plain_*` do not; `*_mm` have an
        # MRoPE delta, `*_text` none.
        plain_mm = self._make_mrope_gen_request(6, 1, 0, 20)
        verify_mm = self._make_mrope_gen_request(4, 2, 1, 10)
        plain_text = self._make_mrope_gen_request(5, 3, 2, None)
        verify_text = self._make_mrope_gen_request(3, 4, 3, None)
        verify_mm.py_draft_tokens = [7]
        verify_text.py_draft_tokens = [7]

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [
            plain_mm, verify_mm, plain_text, verify_text
        ]
        result = self._prepare_mrope_spec_inputs(model_engine, kv_cache_manager,
                                                 attn_metadata,
                                                 scheduled_requests)

        # Packed order: verify_mm, verify_text, plain_mm, plain_text. The
        # verify groups hold two rows each, the plain requests one.
        self._assert_mrope_positions(result, [13, 14, 2, 3, 25, 4])
        # Slot 32 is the reserved zero slot (max_num_tokens * pp_size) read by
        # the text-only requests.
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [1, 32, 0, 32])
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(),
                         [1, 0])
        kv_cache_manager.shutdown()

    def test_prepare_inputs_spec_decode_mrope_context_takes_no_read_slot(
            self) -> None:
        """A context request packs ahead of the verify groups with its own
        MRoPE positions and takes no generation read slot, so the verify
        groups' slots start the list."""
        model_engine, kv_cache_manager, attn_metadata = self._setup_mrope_engine(
            spec_config=MTPDecodingConfig(max_draft_len=1))
        # A context request with multimodal content takes the placeholder-token
        # split, which reads the vocabulary size.
        model_engine.model.config.vocab_size = 1000
        context = self._make_mrope_ctx_request(5, 1, 0, 30)
        verify_mm = self._make_mrope_gen_request(4, 2, 1, 10)
        verify_mm.py_draft_tokens = [7]
        plain_mm = self._make_mrope_gen_request(6, 3, 2, 20)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = [context]
        scheduled_requests.generation_requests = [plain_mm, verify_mm]
        result = self._prepare_mrope_spec_inputs(model_engine, kv_cache_manager,
                                                 attn_metadata,
                                                 scheduled_requests)

        # The context span keeps the processor's per-axis positions; the
        # generation rows after it are text at position + delta on all axes.
        self._assert_mrope_positions(result, [
            [100, 101, 102, 103, 104, 13, 14, 25],
            [1100, 1101, 1102, 1103, 1104, 13, 14, 25],
            [2100, 2101, 2102, 2103, 2104, 13, 14, 25],
        ])
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [1, 2])
        # The context request seeds the delta cache first, then the verify
        # group and the plain request seed theirs.
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(),
                         [0, 1, 2])
        self.assertEqual(context.py_mrope_delta_cache_slot, 0)
        kv_cache_manager.shutdown()

    def _setup_extend_as_context_engine(self):
        """A two-model draft mode on an attention backend that runs verify groups
        as chunked context rows.

        `extend_ctx` is False for every one-model mode and for the TRTLLM
        backend below sm_120, so this takes a drafter-based mode and swaps in
        the vanilla backend.
        """
        model_engine, kv_cache_manager, attn_metadata = self._setup_mrope_engine(
            spec_config=NGramDecodingConfig(max_draft_len=1))
        _set_runner_config(model_engine._runner,
                           attention_backend=VanillaAttention,
                           disable_overlap_scheduler=True)
        spec_config = model_engine._runner._config.spec_config
        self.assertTrue(spec_config.spec_dec_mode.extend_ctx(VanillaAttention))
        self.assertTrue(spec_config.is_linear_tree)
        return model_engine, kv_cache_manager, attn_metadata

    def test_prepare_inputs_mrope_verify_as_context_takes_no_read_slot(
            self) -> None:
        """When the kernel runs verify groups as chunked context rows, their
        rotary comes from `position_ids` alone and only the plain generation
        requests take read slots."""
        model_engine, kv_cache_manager, attn_metadata = (
            self._setup_extend_as_context_engine())
        verify_mm = self._make_mrope_gen_request(4, 1, 0, 10)
        verify_text = self._make_mrope_gen_request(3, 2, 1, None)
        plain_mm = self._make_mrope_gen_request(6, 3, 2, 20)
        plain_text = self._make_mrope_gen_request(5, 4, 3, None)
        verify_mm.py_draft_tokens = [7]
        verify_text.py_draft_tokens = [7]

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [
            verify_mm, verify_text, plain_mm, plain_text
        ]
        result = self._prepare_mrope_spec_inputs(model_engine, kv_cache_manager,
                                                 attn_metadata,
                                                 scheduled_requests)

        self._assert_mrope_positions(result, [13, 14, 2, 3, 25, 4])
        self.assertEqual(result["mrope_delta_read_seq_slots"].cpu().tolist(),
                         [2, 32])
        # The verify group still seeds the delta cache for the plain steps
        # that follow it.
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(),
                         [0, 2])
        kv_cache_manager.shutdown()

    def test_prepare_inputs_mrope_verify_as_context_alone_emits_no_read_slots(
            self) -> None:
        """A batch whose only delta carriers are verify groups run as context
        rows has nothing for the generation-indexed read."""
        model_engine, kv_cache_manager, attn_metadata = (
            self._setup_extend_as_context_engine())
        verify_mm = self._make_mrope_gen_request(4, 1, 0, 10)
        verify_mm.py_draft_tokens = [7]
        plain_text = self._make_mrope_gen_request(5, 2, 1, None)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_last_chunk = []
        scheduled_requests.generation_requests = [verify_mm, plain_text]
        result = self._prepare_mrope_spec_inputs(model_engine, kv_cache_manager,
                                                 attn_metadata,
                                                 scheduled_requests)

        self._assert_mrope_positions(result, [13, 14, 4])
        self.assertNotIn("mrope_delta_read_seq_slots", result)
        self.assertEqual(result["mrope_delta_write_seq_slots"].cpu().tolist(),
                         [0])
        kv_cache_manager.shutdown()

    @skip_pre_blackwell_unittest
    def test_ced_warmup_passes_dummy_context_requests(self) -> None:
        for warmup in (False, True):
            with self.subTest(warmup=warmup):
                engine, manager, metadata = self._setup_mrope_engine(1024)
                manager.has_private_swa_suffix = Mock(return_value=True)
                engine.model.model_config.pretrained_config.rope_scaling = None
                engine.model.model = SimpleNamespace(ced_kv_precompute=True,
                                                     decoder_replay_split=20)
                _set_runner_config(engine._runner,
                                   use_mrope=False,
                                   disable_overlap_scheduler=True)
                engine._runner.is_warmup = warmup
                engine._runner._graphs.cuda_graph_runner = SimpleNamespace(
                    enabled=False)
                scheduled = ScheduledRequests()
                scheduled.reset_context_requests(
                    manager.add_dummy_requests([0],
                                               token_nums=[257],
                                               is_gen=False))
                try:
                    prepared = engine._runner._inputs.prepare_inputs(
                        scheduled,
                        manager,
                        metadata,
                        enable_spec_decode=False,
                        runtime_draft_len=0,
                        is_dummy=warmup)
                    inputs = prepared.kwargs
                    self.assertEqual(inputs['context_requests'],
                                     scheduled.context_requests)
                    self.assertTrue(inputs['context_requests'][0].is_dummy)
                    self.assertEqual(metadata.seq_lens.tolist(), [257])
                    self.assertEqual(inputs['input_ids'].numel(), 257)
                finally:
                    for req in scheduled.context_requests:
                        manager.free_resources(req)
                    manager.shutdown()

    @skip_pre_blackwell_unittest
    def test_ced_inputs_preserve_physical_rows_and_token_sources(self) -> None:
        from tensorrt_llm._torch.pyexecutor.ced_replay import EncoderReplay

        for overlap, mixed in ((False, True), (True, True), (True, False)):
            with self.subTest(overlap=overlap, mixed=mixed):
                engine, manager, metadata = self._setup_mrope_engine(5001)
                manager.has_private_swa_suffix = Mock(return_value=True)
                engine.model.model_config.pretrained_config.rope_scaling = None
                engine.model.model = SimpleNamespace(ced_kv_precompute=True,
                                                     decoder_replay_split=20)
                _set_runner_config(engine._runner,
                                   use_mrope=False,
                                   disable_overlap_scheduler=not overlap)
                engine._runner.is_warmup = False
                engine._runner._graphs.cuda_graph_runner = SimpleNamespace(
                    enabled=not mixed)
                scheduled = ScheduledRequests()
                if mixed:
                    context = _create_request_with_tokens(list(range(5000)), 7)
                    context.context_current_position = 4000
                    context.context_chunk_size = 1000
                    context.py_seq_slot = 0
                    context.py_ced_replay = EncoderReplay(7, 123, 4000, 3872)
                    scheduled.context_requests_last_chunk = [context]
                for index, (length, previous_index) in enumerate(
                    ((20, 2), (30, 0))):
                    req = _create_request(length, 8 + index)
                    req.py_seq_slot = index + 1
                    req.py_batch_idx = previous_index
                    scheduled.generation_requests.append(req)
                new_tokens = torch.tensor([[[222], [999], [111]]],
                                          dtype=torch.int32,
                                          device='cuda')
                try:
                    prepared = engine._runner._inputs.prepare_inputs(
                        scheduled,
                        manager,
                        metadata,
                        new_tensors_device=SimpleNamespace(
                            new_tokens=new_tokens) if overlap else None,
                        enable_spec_decode=False,
                        runtime_draft_len=0,
                        is_dummy=False)
                    inputs, gather_ids = prepared.kwargs, prepared.gather_ids
                    expected_context = list(range(3872, 5000)) if mixed else []
                    self.assertEqual(
                        inputs['input_ids'].cpu().tolist(),
                        expected_context + ([111, 222] if overlap else [0, 0]))
                    self.assertEqual(
                        inputs['position_ids'].view(-1).cpu().tolist(),
                        expected_context + ([20, 30] if overlap else [19, 29]))
                    self.assertIsNone(gather_ids)
                    if mixed:
                        self.assertEqual(inputs['context_requests'], [context])
                        self.assertEqual(metadata.seq_lens.tolist(),
                                         [1128, 1, 1])
                        self.assertEqual(context.context_current_position, 4000)
                        self.assertEqual(context.context_chunk_size, 1000)
                        self.assertEqual(context.cached_tokens, 4000)
                        if not overlap:
                            self.assertEqual(
                                list(metadata.kv_cache_params.
                                     num_cached_tokens_per_seq), [3872, 19, 29])
                    else:
                        self.assertNotIn('context_requests', inputs)
                    # GPU sampling is one step ahead of CPU token append.
                    self.assertEqual([
                        r.max_beam_num_tokens
                        for r in scheduled.generation_requests
                    ], [20, 30])
                finally:
                    manager.shutdown()

    def test_item_scheduled_chunk_embedding_is_not_retained(self) -> None:
        """The per-forward item-scheduler payload must not replace the
        request's own multimodal data, or each chunked prefill would keep its
        last chunk's joined embedding rows until prefill completes."""
        model_engine, kv_cache_manager = create_model_engine_and_kvcache()
        model_engine.model.config.vocab_size = 100
        attn_metadata = AttentionMetadata(max_num_requests=4,
                                          max_num_tokens=32,
                                          kv_cache_manager=kv_cache_manager)
        attn_metadata.is_cuda_graph = False

        context = _create_request_with_tokens([11, 22, 33, 44], 1)
        context.context_chunk_size = 2
        context.py_seq_slot = 0
        context.py_mm_encoder_state = object()
        multimodal_data = {
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY:
            MultimodalItemMetadata([('image', 0)], output_embedding_lengths=[2])
        }
        context.py_multimodal_data = multimodal_data
        model_engine._mm_item_scheduler = Mock()
        model_engine._runner._inputs._mm_item_scheduler = model_engine._mm_item_scheduler
        model_engine._mm_item_scheduler.build_multimodal_data_for_llm.return_value = {
            "multimodal_embedding": torch.ones((2, 1), dtype=torch.float16),
            "multimodal_embedding_is_chunk": True,
        }
        scheduled_requests = ScheduledRequests()
        scheduled_requests.context_requests_chunking = [context]

        model_engine._runner._inputs.prepare_inputs(
            scheduled_requests=scheduled_requests,
            kv_cache_manager=kv_cache_manager,
            attn_metadata=attn_metadata,
            enable_spec_decode=False,
            runtime_draft_len=0,
            is_dummy=False,
        )

        model_engine._mm_item_scheduler.build_multimodal_data_for_llm.assert_called_once(
        )
        self.assertIs(context.py_multimodal_data, multimodal_data)
        self.assertNotIn("multimodal_embedding", multimodal_data)
        kv_cache_manager.shutdown()

    def test_kv_cache_manager_with_execution_stream(self) -> None:
        """Test that KVCacheManagerV2 uses the provided execution_stream.
        """
        # Create a dedicated execution stream
        execution_stream = torch.cuda.Stream()

        model_engine, kv_cache_manager = create_model_engine_and_kvcache(
            execution_stream=execution_stream)

        # Verify the KVCacheManagerV2 uses the provided execution stream
        self.assertEqual(
            kv_cache_manager._stream.cuda_stream, execution_stream.cuda_stream,
            "KVCacheManagerV2 should use the provided execution_stream")

        resource_manager = ResourceManager(
            {ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager})

        prompt_len = 32
        requests = [_create_request(prompt_len, 0)]

        batch = ScheduledRequests()
        batch.context_requests_last_chunk = requests
        _prepare_kv_cache_resources(kv_cache_manager, batch)
        with torch.cuda.stream(execution_stream):
            model_engine.forward(batch, resource_manager)

        # Verify the stream is still the same after forward pass
        self.assertEqual(
            kv_cache_manager._stream.cuda_stream, execution_stream.cuda_stream,
            "KVCacheManagerV2 should still use the provided execution_stream after forward"
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


@pytest.mark.parametrize("draft_capacity", [None, 0, 8])
def test_runner_input_buffers_support_repeated_cuda_graph_replay(
        draft_capacity):
    buffers = InputBuffers.allocate(
        max_num_tokens=16,
        max_batch_size=2,
        max_beam_width=2,
        max_seq_len=8,
        use_cache_indirection=True,
        max_num_draft_tokens=draft_capacity,
    )
    assert buffers.cache_indirection.shape == (2, 2, 8)
    if draft_capacity is None:
        assert buffers.gather_ids_cuda is None
        assert buffers.draft_tokens_cuda is None
    else:
        assert buffers.gather_ids_cuda.numel() == 16
        assert buffers.draft_tokens_cuda.numel() == draft_capacity
    input_ptr = buffers.input_ids_cuda.data_ptr()
    buffers.input_ids_cuda.fill_(1)
    buffers.position_ids_cuda.fill_(2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = buffers.input_ids_cuda + buffers.position_ids_cuda
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = buffers.input_ids_cuda + buffers.position_ids_cuda
    output_ptr = output.data_ptr()
    for token in (3, 7):
        buffers.input_ids_cuda.fill_(token)
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, token + 2))
        assert output.data_ptr() == output_ptr
        assert buffers.input_ids_cuda.data_ptr() == input_ptr
    graph.reset()


class TestDecodeGraphMetadataPreparation(unittest.TestCase):

    @staticmethod
    def _metadata():
        from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import \
            CSA2TrtllmMetadata
        metadata = object.__new__(CSA2TrtllmMetadata)
        metadata.is_cuda_graph = True
        return metadata

    def test_scope_requires_selected_ordinary_decode_graph(self):
        key = KeyType(batch_size=1, draft_len=0)
        cases = (
            "ordinary",
            "ordinary_overlap",
            "graph_miss",
            "eager_metadata",
            "context",
            "promoted_context",
            "speculative",
            "beam",
            "encoder_decoder",
            "context_parallel",
            "draft_key",
        )
        for case in cases:
            with self.subTest(case=case):
                engine, _, _, _, _ = _make_forward_only_engine(key)
                enable_spec_decode = False
                metadata = self._metadata()
                batch = ScheduledRequests()
                batch.generation_requests = [_make_request_stub(1)]
                selected = key
                promoted = frozenset()
                engine._runner._config.disable_overlap_scheduler = case != "ordinary_overlap"
                if case == "graph_miss":
                    selected = None
                elif case == "eager_metadata":
                    metadata.is_cuda_graph = False
                elif case == "context":
                    batch.context_requests_last_chunk = [_make_request_stub(2)]
                elif case == "promoted_context":
                    promoted = frozenset({1})
                elif case == "speculative":
                    enable_spec_decode = True
                elif case == "beam":
                    engine._runner._config.max_beam_width = 2
                elif case == "encoder_decoder":
                    engine._runner._deferred_decode_outputs_supported = False
                elif case == "context_parallel":
                    engine.mapping.cp_size = 2
                elif case == "draft_key":
                    selected = key._replace(draft_len=1)
                with engine._runner._cuda_graph_metadata_prepare_scope(
                        metadata,
                        selected,
                        batch,
                        promoted,
                        enable_spec_decode=enable_spec_decode):
                    self.assertEqual(metadata._csa2_defer_decode_outputs, case
                                     in ("ordinary", "ordinary_overlap"))
                self.assertFalse(metadata._csa2_defer_decode_outputs)

    def test_forward_scopes_proof_to_prepare_for_capture_replay_and_fallback(
            self):
        key = KeyType(batch_size=1, draft_len=0)
        for mode in ("warmup", "capture", "replay", "fallback",
                     "prepare_error"):
            with self.subTest(mode=mode):
                selected = None if mode == "fallback" else key
                engine, runner, resource_manager, _, outputs = _make_forward_only_engine(
                    selected)
                metadata = self._metadata()
                engine._runner.attn_metadata = metadata
                engine._runner._metadata.setup_attention.return_value = metadata
                runner.maybe_get_cuda_graph.return_value = (metadata, None,
                                                            selected)
                runner.needs_capture.return_value = mode in ("warmup",
                                                             "capture")
                runner.is_warmup_only = mode == "warmup"
                runner.capture.return_value = outputs
                batch = ScheduledRequests()
                batch.generation_requests = [_make_request_stub(1)]
                prepared = engine._runner._inputs.prepare_inputs.return_value

                def prepare(*args, **kwargs):
                    self.assertIs(metadata.trtllm_gen_jit_warmup, False)
                    self.assertEqual(metadata._csa2_defer_decode_outputs, mode
                                     != "fallback")
                    if mode == "prepare_error":
                        raise RuntimeError("prepare failed")
                    return prepared

                engine._runner._inputs.prepare_inputs.side_effect = prepare
                with patch(
                        "tensorrt_llm._torch.pyexecutor.model_engine.torch.cuda.Event",
                        return_value=Mock()):
                    if mode == "prepare_error":
                        with self.assertRaisesRegex(RuntimeError,
                                                    "prepare failed"):
                            engine.forward(batch, resource_manager)
                    else:
                        actual_outputs = engine.forward(batch, resource_manager)
                        self.assertEqual(actual_outputs, {
                            **outputs, "runtime_draft_len": 0
                        })
                        self.assertIs(actual_outputs["logits"],
                                      outputs["logits"])
                self.assertFalse(metadata._csa2_defer_decode_outputs)
                if mode in ("warmup", "capture"):
                    runner.capture.assert_called_once()
                if mode in ("capture", "replay"):
                    runner.replay.assert_called_once()
                if mode == "fallback":
                    engine._runner._forward_step.assert_called_once()

    def test_jit_warmup_flag_marks_only_the_warmup_forward(self):
        engine, _, resource_manager, _, _ = _make_forward_only_engine(None)
        runner = engine._runner
        metadata = self._metadata()
        runner.attn_metadata = metadata
        runner._metadata.setup_attention.return_value = metadata
        prepared = runner._inputs.prepare_inputs.return_value
        observed = []

        def prepare(*args, **kwargs):
            observed.append(metadata.trtllm_gen_jit_warmup)
            return prepared

        runner._inputs.prepare_inputs.side_effect = prepare
        batch = ScheduledRequests()
        batch.generation_requests = [_make_request_stub(1)]
        for jit_warmup in (True, False):
            runner._forward_warmup(batch,
                                   resource_manager,
                                   enable_spec_decode=False,
                                   runtime_draft_len=0,
                                   trtllm_gen_jit_warmup=jit_warmup)
        self.assertEqual(observed, [True, False])
        self.assertFalse(runner._trtllm_gen_jit_warmup)


if __name__ == "__main__":
    unittest.main()
