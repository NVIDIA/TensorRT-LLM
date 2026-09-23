# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.compilation.backend import Backend
from tensorrt_llm._torch.pyexecutor.engine.input_buffers import InputBuffers
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners import no_kv_cache as no_kv_cache_module
from tensorrt_llm._torch.pyexecutor.engine.runners import resolve_runner_type
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder import EncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.interface import (
    ModelRunner,
    PreparedInputs,
    RunnerConfig,
    ScheduledInputs,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import (
    NoKVCacheRunner,
    NoKVCacheRunnerConfig,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import get_model_extra_attrs, model_extra_attrs
from tensorrt_llm.llmapi.llm_args import (
    CudaGraphConfig,
    EncodeCudaGraphConfig,
    PrefillCudaGraphBackend,
)

pytestmark = pytest.mark.cpu_only


class _AttentionMetadata:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.block_ids_per_seq = object()
        self.kv_block_ids_per_seq = object()
        self.on_update_kv_lens = Mock()


class _AttentionBackend:
    Metadata = _AttentionMetadata


def _make_runner(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
    model: Any,
    config: NoKVCacheRunnerConfig | None = None,
    *,
    model_caller: Mock | None = None,
) -> PoolingRunner | MultimodalEncoderRunner:
    config = config or _config()
    buffers = InputBuffers(
        input_ids_cuda=torch.empty(config.max_num_tokens, dtype=torch.int),
        position_ids_cuda=torch.empty(config.max_num_tokens, dtype=torch.int),
        gather_ids_cuda=torch.empty(config.max_num_tokens, dtype=torch.int)
        if config.spec_config is not None
        else None,
        draft_tokens_cuda=torch.empty(
            config.max_draft_loop_tokens * config.max_batch_size, dtype=torch.int
        )
        if config.spec_config is not None
        else None,
    )
    kwargs = dict(mapping=SimpleNamespace(), dist=SimpleNamespace(), moe_load_balancer=None)
    if issubclass(runner_type, PoolingRunner):
        kwargs["model_caller"] = model_caller or Mock(return_value={"logits": torch.empty(0)})
    with patch.object(InputBuffers, "allocate", return_value=buffers):
        return runner_type(model, config, **kwargs)


def _config() -> NoKVCacheRunnerConfig:
    return NoKVCacheRunnerConfig(
        max_batch_size=4,
        max_num_tokens=16,
        max_seq_len=16,
        max_beam_width=1,
        without_logits=False,
        attention_backend=_AttentionBackend,
        attention_runtime_features=AttentionRuntimeFeatures(),
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=[],
        mm_encoder_cache_enabled=False,
        spec_config=None,
        is_draft_model=False,
        num_seq_slots=None,
        original_max_draft_len=0,
        original_max_total_draft_tokens=0,
        spec_dec_max_total_draft_tokens=0,
        max_draft_loop_tokens=0,
    )


def _model(*, is_generation: bool, is_encoder_decoder: bool = False) -> SimpleNamespace:
    pretrained_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_generation=is_generation,
            is_encoder_decoder=is_encoder_decoder,
            pretrained_config=pretrained_config,
            sparse_attention_config=None,
            enable_flash_mla=False,
        ),
        config=pretrained_config,
    )


@pytest.mark.parametrize(
    ("encode_only", "mm_encoder_only", "is_generation", "is_encoder_decoder", "runner_type"),
    [
        (False, True, False, False, MultimodalEncoderRunner),
        (False, False, False, False, PoolingRunner),
        (True, False, False, False, EncoderRunner),
        (False, False, True, True, EncoderDecoderRunner),
        (False, False, True, False, None),
    ],
)
def test_resolve_runner_dispatches_startup_families(
    encode_only: bool,
    mm_encoder_only: bool,
    is_generation: bool,
    is_encoder_decoder: bool,
    runner_type: type[ModelRunner] | None,
) -> None:
    args = SimpleNamespace(encode_only=encode_only, mm_encoder_only=mm_encoder_only)

    actual_type = resolve_runner_type(
        _model(is_generation=is_generation, is_encoder_decoder=is_encoder_decoder),
        args,
    )

    assert actual_type is runner_type


def test_resolve_runner_type_does_not_require_runtime_dependencies() -> None:
    args = SimpleNamespace(encode_only=False, mm_encoder_only=False)

    assert resolve_runner_type(_model(is_generation=False), args) is PoolingRunner


def test_resolve_runner_checks_encode_only_before_pooling() -> None:
    args = SimpleNamespace(encode_only=True, mm_encoder_only=False)

    assert resolve_runner_type(_model(is_generation=False), args) is EncoderRunner


def test_resolve_runner_checks_mm_encoder_before_non_generation() -> None:
    args = SimpleNamespace(encode_only=False, mm_encoder_only=True)

    assert resolve_runner_type(_model(is_generation=False), args) is MultimodalEncoderRunner


@pytest.mark.parametrize(
    ("runner_type", "initializer_name"),
    [
        (EncoderRunner, "_initialize_encoder_runner"),
        (EncoderDecoderRunner, "_initialize_encoder_decoder_runner"),
        (NoKVCacheRunner, "_initialize_no_kv_cache_runner"),
    ],
)
def test_model_engine_initializes_runner_by_family(
    runner_type: type[ModelRunner], initializer_name: str
) -> None:
    engine = object.__new__(PyTorchModelEngine)
    engine._model_caller = Mock()
    expected = Mock(spec=runner_type)
    initializer = Mock(return_value=expected)
    setattr(engine, initializer_name, initializer)

    assert engine._initialize_runner(runner_type) is expected
    initializer.assert_called_once_with(runner_type)


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
    engine = object.__new__(PyTorchModelEngine)
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
    engine.is_draft_model = False
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
    )


def test_model_engine_rejects_unregistered_runner_family() -> None:
    class UnregisteredRunner:
        pass

    engine = object.__new__(PyTorchModelEngine)
    engine._model_caller = Mock()

    with pytest.raises(TypeError, match="No runner initializer registered"):
        engine._initialize_runner(UnregisteredRunner)


def _model_engine_with_runner(
    runner: Mock | None,
    *,
    kv_cache_manager: object | None,
) -> tuple[PyTorchModelEngine, Mock]:
    engine = object.__new__(PyTorchModelEngine)
    engine._model_caller = Mock()
    engine.model = SimpleNamespace(extra_attrs={})
    engine.kv_cache_manager_key = ResourceManagerType.KV_CACHE_MANAGER
    engine._runner = runner
    engine._fallback_to_engine = False
    engine.enable_spec_decode = False
    engine.runtime_draft_len = 0
    engine.moe_load_balancer = None
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = kv_cache_manager
    return engine, resource_manager


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


def test_no_kv_cache_runner_rejects_kv_manager_before_preparation() -> None:
    runner = _make_runner(PoolingRunner, _model(is_generation=False))
    runner.prepare_inputs = Mock()
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = object()

    with pytest.raises(
        AssertionError,
        match="no-KV-cache runner was initialized, but a KV cache manager was allocated",
    ):
        runner.forward(
            ScheduledInputs(batch=ScheduledRequests()), resource_manager=resource_manager
        )

    runner.prepare_inputs.assert_not_called()


def test_model_engine_forward_encoder_delegates_scheduled_encoder_batch() -> None:
    runner = Mock(spec=EncoderDecoderRunner)
    hidden_states = object()
    runner.forward.return_value = {
        "encoder_hidden_states": hidden_states,
        "encoder_seq_lens": [2, 3],
    }
    engine = object.__new__(PyTorchModelEngine)
    engine._model_caller = Mock()
    engine.model = SimpleNamespace(
        model_config=SimpleNamespace(is_encoder_decoder=True),
    )
    engine._runner = runner
    requests = [object(), object()]
    resource_manager = object()

    outputs = engine.forward_encoder(requests, resource_manager)

    assert outputs == (hidden_states, [2, 3])
    inputs = runner.forward.call_args.args[0]
    assert inputs.batch.encoder_requests == requests
    runner.forward.assert_called_once_with(
        inputs,
        resource_manager=resource_manager,
        is_dummy=False,
    )


def test_model_engine_releases_runner_owned_graphs() -> None:
    engine = object.__new__(PyTorchModelEngine)
    engine._model_caller = Mock()
    engine._runner = Mock(spec=EncoderRunner)
    engine._torch_compile_backend = None
    engine.cuda_graph_runner = None
    engine.breakable_cuda_graph_runner = None

    engine._release_cuda_graphs()

    engine._runner.release_graphs.assert_called_once_with()


def test_prepared_inputs_is_frozen_and_preserves_kwargs_identity() -> None:
    kwargs = {"input_ids": torch.tensor([1])}
    prepared = PreparedInputs(kwargs)

    assert prepared.kwargs is kwargs
    with pytest.raises(FrozenInstanceError):
        prepared.gather_ids = torch.tensor([0])


def test_no_kv_cache_config_extends_common_runner_config() -> None:
    assert isinstance(_config(), RunnerConfig)


def test_no_kv_cache_runner_exposes_only_forward_step_as_abstract() -> None:
    assert NoKVCacheRunner.__abstractmethods__ == frozenset({"_forward_step"})


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_concrete_runners_implement_no_kv_cache_forward_step(
    runner_type: type[NoKVCacheRunner],
) -> None:
    assert issubclass(runner_type, NoKVCacheRunner)
    assert runner_type._forward_step is not NoKVCacheRunner._forward_step


def test_no_kv_cache_runner_owns_and_reuses_attention_metadata() -> None:
    runner = _make_runner(PoolingRunner, _model(is_generation=False))

    first = runner.setup_attn_metadata()
    second = runner.setup_attn_metadata()

    assert first is second
    assert first.kv_cache_manager is None
    assert first.max_num_requests == 4
    assert first.max_num_sequences == 4
    assert first.num_heads_per_kv == 4
    assert first.block_ids_per_seq is None
    assert first.kv_block_ids_per_seq is None


def test_no_kv_cache_runner_owns_spec_metadata_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_mode = SimpleNamespace(
        attention_need_spec_dec_mode=Mock(return_value=True),
        is_parallel_draft=Mock(return_value=False),
    )
    spec_metadata = SimpleNamespace(
        spec_dec_mode=spec_mode,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=False,
    )
    get_spec_metadata = Mock(return_value=spec_metadata)
    monkeypatch.setattr(no_kv_cache_module, "get_spec_metadata", get_spec_metadata)
    runner = _make_runner(
        PoolingRunner,
        _model(is_generation=False),
        replace(
            _config(),
            spec_config=SimpleNamespace(
                get_runtime_tokens_per_gen_step=lambda runtime_draft_len: runtime_draft_len + 1,
            ),
            original_max_draft_len=2,
            spec_dec_max_total_draft_tokens=3,
        ),
    )
    attn_metadata = SimpleNamespace(update_spec_dec_param=Mock())
    scheduled_requests = SimpleNamespace(
        batch_size=2,
        num_context_requests=2,
        context_requests=[object(), object()],
        generation_requests=[],
    )
    resource_manager = SimpleNamespace(get_resource_manager=Mock(return_value=None))

    result = runner.setup_spec_metadata(
        scheduled_requests,
        resource_manager,
        attn_metadata,
        runtime_draft_len=1,
    )

    assert result is spec_metadata
    assert spec_metadata.runtime_draft_len == 1
    assert spec_metadata.runtime_tokens_per_gen_step == 2
    attn_metadata.update_spec_dec_param.assert_called_once_with(
        batch_size=2,
        is_spec_decoding_enabled=True,
        is_spec_dec_tree=True,
        is_spec_dec_dynamic_tree=False,
        max_draft_len=2,
        max_total_draft_tokens=3,
        spec_metadata=spec_metadata,
        spec_tree_manager=None,
        num_contexts=2,
    )


def test_pooling_runner_owns_forward_output_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.arange(12).reshape(3, 4)
    model_forward = Mock(return_value=logits)
    runner = _make_runner(
        PoolingRunner,
        _model(is_generation=False),
        model_caller=model_forward,
    )
    attn_metadata = SimpleNamespace(on_update_kv_lens=Mock())
    prepared = PreparedInputs(
        {"attn_metadata": attn_metadata, "input_ids": torch.tensor([1])},
        gather_ids=torch.tensor([2, 0]),
    )
    monkeypatch.setattr(runner, "prepare_inputs", Mock(return_value=prepared))

    outputs = runner.forward(
        ScheduledInputs(batch=ScheduledRequests()),
        resource_manager=SimpleNamespace(get_resource_manager=Mock(return_value=None)),
    )

    torch.testing.assert_close(outputs["logits"], logits[[2, 0]])
    attn_metadata.on_update_kv_lens.assert_called_once_with()
    model_forward.assert_called_once_with(
        attn_metadata=attn_metadata,
        input_ids=prepared.kwargs["input_ids"],
        return_context_logits=True,
    )


def test_pooling_runner_returns_raw_model_outputs_when_logits_are_disabled() -> None:
    model_outputs = {"hidden_states": object()}
    model_forward = Mock(return_value=model_outputs)
    runner = _make_runner(
        PoolingRunner,
        _model(is_generation=False),
        replace(_config(), without_logits=True),
        model_caller=model_forward,
    )

    outputs = runner._forward_step({}, SimpleNamespace())

    assert outputs is model_outputs


@pytest.mark.parametrize("runner_type", [PoolingRunner, MultimodalEncoderRunner])
def test_no_kv_cache_runner_default_lifecycle_with_no_kv_resources(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
) -> None:
    runner = _make_runner(runner_type, _model(is_generation=False))
    resource_manager = SimpleNamespace(get_resource_manager=Mock(return_value=None))

    assert runner.warmup(resource_manager) is None
    assert runner.release_graphs() is None
    assert runner.wait_for_input_copy() is None


def test_mm_encoder_runner_forward_step_returns_empty_result_without_multimodal_params() -> None:
    runner = _make_runner(MultimodalEncoderRunner, SimpleNamespace())

    result = runner._forward_step({}, SimpleNamespace(context_requests=[]))

    assert result == {
        "mm_embeddings": [],
        "mm_embedding_request_indices": [],
        "mm_embedding_lengths": [],
    }


def test_mm_encoder_runner_forward_step_rejects_payload_request_count_mismatch() -> None:
    request = SimpleNamespace(py_multimodal_data={"image": object()})
    runner = _make_runner(MultimodalEncoderRunner, SimpleNamespace())

    with pytest.raises(ValueError, match="one multimodal payload per context"):
        runner._forward_step(
            {"multimodal_params": [SimpleNamespace(), SimpleNamespace()]},
            SimpleNamespace(context_requests=[request]),
        )


def test_mm_encoder_runner_forward_step_skips_metadata_only_and_missing_length_requests() -> None:
    requests = [
        SimpleNamespace(py_multimodal_data={"mrope_config": {}}),
        SimpleNamespace(py_multimodal_data={"image": object()}),
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                "multimodal_embedding_lengths": [2],
            },
            multimodal_lengths=[2],
        ),
    ]
    params = [
        SimpleNamespace(multimodal_data={}),
        SimpleNamespace(multimodal_data={}),
        SimpleNamespace(multimodal_data={}),
    ]
    model = SimpleNamespace(forward=Mock(return_value=[torch.arange(4).reshape(2, 2)]))
    runner = _make_runner(MultimodalEncoderRunner, model)

    result = runner._forward_step(
        {"multimodal_params": params},
        SimpleNamespace(context_requests=requests),
    )

    model.forward.assert_called_once_with([params[2]])
    assert result["logits"] is None
    assert result["mm_embedding_request_indices"] == [2]
    assert result["mm_embedding_lengths"] == [[2]]
    torch.testing.assert_close(result["mm_embeddings"][0], torch.arange(4).reshape(2, 2))


def test_mm_encoder_runner_splits_embeddings_and_returns_mrope_metadata() -> None:
    first_param = SimpleNamespace(
        multimodal_data={
            "image": object(),
            "mrope_config": {
                "mrope_position_ids": "first-ids",
                "mrope_position_deltas": "first-deltas",
            },
        }
    )
    second_param = SimpleNamespace(
        multimodal_data={
            "image": object(),
            "mrope_config": {
                "mrope_position_ids": "second-ids",
                "mrope_position_deltas": "second-deltas",
            },
        }
    )
    requests = [
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                "multimodal_embedding_lengths": [1],
            },
            multimodal_lengths=[1],
        ),
        SimpleNamespace(
            py_multimodal_data={
                "image": object(),
                "multimodal_embedding_lengths": [2],
            },
            multimodal_lengths=[2],
        ),
    ]
    embeddings = torch.arange(6).reshape(3, 2)
    model = SimpleNamespace(forward=Mock(return_value=[embeddings]))
    runner = _make_runner(MultimodalEncoderRunner, model)

    result = runner._forward_step(
        {"multimodal_params": [first_param, second_param]},
        SimpleNamespace(context_requests=requests),
    )

    model.forward.assert_called_once_with([first_param, second_param])
    assert result["logits"] is None
    assert result["mm_embedding_request_indices"] == [0, 1]
    assert result["mm_embedding_lengths"] == [[1], [2]]
    torch.testing.assert_close(result["mm_embeddings"][0], embeddings[:1])
    torch.testing.assert_close(result["mm_embeddings"][1], embeddings[1:])
    assert result["mrope_position_ids"] == ["first-ids", "second-ids"]
    assert result["mrope_position_deltas"] == ["first-deltas", "second-deltas"]


@pytest.mark.parametrize("effective_length", [None, 0, 3])
@pytest.mark.parametrize("is_dummy", [False, True])
def test_engine_consumes_optional_length_update_and_passes_call_state(effective_length, is_dummy):
    runner = Mock(spec=NoKVCacheRunner)
    outputs = {"logits": object()}
    if effective_length is not None:
        outputs["runtime_draft_len"] = effective_length
    runner.forward.return_value = outputs
    engine, resources = _model_engine_with_runner(runner, kv_cache_manager=None)
    engine.runtime_draft_len = 2
    engine.enable_spec_decode = True
    engine._is_warmup = is_dummy
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = [SimpleNamespace(py_return_context_logits=True)]

    assert engine.forward(batch, resources) is outputs

    inputs = runner.forward.call_args.args[0]
    assert inputs.batch is batch
    assert inputs.gather_context_logits
    assert inputs.enable_spec_decode
    assert inputs.runtime_draft_len == 2
    assert runner.forward.call_args.kwargs["is_dummy"] is is_dummy
    assert engine.runtime_draft_len == (2 if effective_length is None else effective_length)
    assert "runtime_draft_len" not in outputs


def test_decoder_fallback_length_update_does_not_modify_graph_output_dictionary():
    engine, resources = _model_engine_with_runner(None, kv_cache_manager=object())
    engine._fallback_to_engine = True
    cached_outputs = {"logits": torch.tensor([1.0])}
    inputs = ScheduledInputs(batch=ScheduledRequests(), runtime_draft_len=2)

    def decoder_forward(*args):
        engine.runtime_draft_len = 4
        return cached_outputs

    engine._forward_decoder = Mock(side_effect=decoder_forward)
    outputs = engine._forward_scheduled(inputs, resource_manager=resources)

    assert outputs is not cached_outputs
    assert outputs["logits"] is cached_outputs["logits"]
    assert outputs.pop("runtime_draft_len") == 4
    assert set(cached_outputs) == {"logits"}
    assert inputs.runtime_draft_len == 2


def test_no_kv_forward_preserves_model_output_dictionary_without_length_update():
    outputs = {"hidden_states": object()}
    caller = Mock(return_value=outputs)
    runner = _make_runner(PoolingRunner, _model(is_generation=False), model_caller=caller)
    runner.prepare_inputs = Mock(return_value=PreparedInputs({}))
    inputs = ScheduledInputs(batch=ScheduledRequests(), runtime_draft_len=3)
    resources = SimpleNamespace(get_resource_manager=Mock(return_value=None))

    actual = runner.forward(inputs, resource_manager=resources)

    assert actual is outputs
    assert "runtime_draft_len" not in outputs
    assert inputs.runtime_draft_len == 3


@pytest.mark.parametrize("raw_output", [torch.tensor([1.0]), None], ids=["tensor", "none"])
@pytest.mark.parametrize("is_dummy", [False, True])
def test_engine_forward_preserves_raw_decoder_outputs_and_length(raw_output, is_dummy):
    engine, resources = _model_engine_with_runner(None, kv_cache_manager=object())
    engine._fallback_to_engine = True
    engine._is_warmup = is_dummy
    engine.runtime_draft_len = 2

    def decoder_forward(inputs, resource_manager):
        assert inputs.runtime_draft_len == 2
        engine.runtime_draft_len = 4
        return raw_output

    engine._forward_decoder = Mock(side_effect=decoder_forward)

    assert engine.forward(ScheduledRequests(), resources) is raw_output
    assert engine.runtime_draft_len == 4


@pytest.mark.parametrize("previous_slots", [None, {}, {7: 3, 2: 9}])
def test_engine_forwards_previous_request_slots_to_decoder(previous_slots):
    engine, resources = _model_engine_with_runner(None, kv_cache_manager=object())
    engine._fallback_to_engine = True
    engine._forward_decoder = Mock(return_value={"logits": None})
    previous_requests = (
        {req_id: SimpleNamespace(py_seq_slot=slot) for req_id, slot in previous_slots.items()}
        if previous_slots is not None
        else None
    )
    batch = ScheduledRequests()

    engine.forward(batch, resources, req_id_to_old_request=previous_requests)

    inputs, actual_resources = engine._forward_decoder.call_args.args
    assert inputs.batch is batch
    assert actual_resources is resources
    assert inputs.previous_request_slots == previous_slots
    if previous_requests:
        previous_requests[7].py_seq_slot = 5
        assert inputs.previous_request_slots[7] == 3


def test_model_caller_uses_current_forward_and_restores_outer_attribute_context():
    metadata = _AttentionMetadata()
    spec_metadata = object()
    attrs = {}
    outer_attrs = {"outer": True}
    model = SimpleNamespace(model_config=SimpleNamespace(extra_attrs={"model_flag": True}))

    def first_forward(**kwargs):
        assert get_model_extra_attrs() is attrs
        assert attrs["attention_metadata"]() is metadata
        assert attrs["spec_metadata"] is spec_metadata
        assert attrs["model_flag"] is True
        return "first"

    model.forward = Mock(side_effect=first_forward)
    caller = ModelCaller(model)
    with model_extra_attrs(outer_attrs):
        with model_extra_attrs(attrs):
            assert caller(attn_metadata=metadata, spec_metadata=spec_metadata) == "first"
            model.forward = Mock(return_value="replacement")
            assert caller(attn_metadata=metadata) == "replacement"
            assert attrs["spec_metadata"] is None
            model.forward.side_effect = RuntimeError("model failure")
            with pytest.raises(RuntimeError, match="model failure"):
                caller(attn_metadata=metadata)
        assert get_model_extra_attrs() is outer_attrs


def test_model_caller_borrows_live_compile_streams_and_events():
    streams = Backend.Streams()
    backend = SimpleNamespace(events=Backend.Events())
    metadata = _AttentionMetadata()
    current_stream = object()
    attrs = {}

    def forward(**kwargs):
        assert get_model_extra_attrs() is attrs
        assert attrs["aux_streams"]() is streams
        assert attrs["events"]() is backend.events
        assert attrs["global_stream"] is current_stream
        return len(attrs["aux_streams"]()), len(attrs["events"]())

    model = SimpleNamespace(model_config=SimpleNamespace(extra_attrs={}), forward=forward)
    caller = ModelCaller(model, compile_backend=backend, aux_streams=streams)
    with patch("torch.cuda.current_stream", return_value=current_stream), model_extra_attrs(attrs):
        assert caller(attn_metadata=metadata) == (0, 0)
        streams.append(object())
        backend.events = Backend.Events([object(), object()])
        assert caller(attn_metadata=metadata) == (1, 2)


def test_model_caller_requires_streams_with_compile_backend():
    with pytest.raises(ValueError, match="requires its auxiliary stream container"):
        ModelCaller(SimpleNamespace(), compile_backend=SimpleNamespace(events=Backend.Events()))
