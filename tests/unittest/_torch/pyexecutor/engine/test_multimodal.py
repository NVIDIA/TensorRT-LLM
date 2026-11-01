# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
import torch
from _torch.pyexecutor.multimodal_utils import (
    bare_mm_item_scheduler,
    make_llm_request,
    make_mm_request,
)

import tensorrt_llm
from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    MultimodalEncoderContractError,
    MultimodalModelMixin,
    _build_request_multimodal_input,
)
from tensorrt_llm._torch.pyexecutor.engine.multimodal import (
    MultimodalItemScheduler,
    resolve_bytes_per_mm_encoder_embedding,
    resolve_mm_encoder_output_budget,
    setup_mm_encoder_attn_metadata,
    setup_mm_encoder_cuda_graphs,
    validate_mm_encoder_scheduling_compatibility,
)
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    MultimodalEncoderRequestError,
    initialize_multimodal_encoder_request,
    is_multimodal_encoder_ready,
    make_mm_encoder_transient_cache_key,
    release_consumed_mm_items,
)
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalParams,
    MultimodalRuntimeData,
)
from tensorrt_llm.inputs.registry import BaseMultimodalDummyInputsBuilder, MultimodalItemMetadata
from tensorrt_llm.llmapi.llm_args import (
    MultimodalConfig,
    MultimodalEncoderCudaGraphConfig,
    MultimodalEncoderSchedulingPolicy,
)
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

# The item-scheduling surface is pure logic: no kernels, no device transfers.
pytestmark = pytest.mark.cpu_only


def test_disabled_encoder_does_not_initialize_processor() -> None:
    processor = Mock(spec=BaseMultimodalDummyInputsBuilder)
    processor.get_mm_max_tokens_per_item.side_effect = AssertionError("encoder is disabled")
    model = torch.nn.Module()
    model.mm_encoder = None
    setup_mm_encoder_attn_metadata(model, processor, 1024, None)
    processor.get_mm_max_tokens_per_item.assert_not_called()


@pytest.mark.parametrize("release_nccl_window_owners", [False, True])
def test_encoder_graph_capture_follows_metadata_setup_and_runs_once(
    release_nccl_window_owners: bool,
) -> None:
    events = []

    class Encoder(torch.nn.Module, MultimodalEncoderMixin):
        def __init__(self, name: str, config: MultimodalEncoderCudaGraphConfig | None) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1))
            self.name = name
            self.encoder_cuda_graph_config = config
            self._blocks_graph_runner = None
            self.runner = Mock()
            self.runner.capture_all.side_effect = lambda device: events.append(
                (self.name, "capture", device)
            )

        def setup_attn_metadata(
            self,
            max_num_tokens: int,
            attention_metadata_capacity: dict[str, int] | None = None,
        ) -> None:
            events.append((self.name, "metadata", max_num_tokens))

        def _build_blocks_graph_runner(self, config: MultimodalEncoderCudaGraphConfig) -> Mock:
            assert events.index((self.name, "metadata", 1024)) >= 0
            assert config is self.encoder_cuda_graph_config
            return self.runner

    config = MultimodalEncoderCudaGraphConfig(buckets=[(256, 1)])
    model = torch.nn.Module()
    model.first = Encoder("first", config)
    model.second = Encoder("second", config)
    model.disabled = Encoder("disabled", None)
    model.shared = model.first
    model.unrelated = torch.nn.Linear(1, 1)

    setup_mm_encoder_attn_metadata(model, object(), 1024, None)
    setup_mm_encoder_cuda_graphs(model)
    setup_mm_encoder_cuda_graphs(model)

    assert events == [
        ("first", "metadata", 1024),
        ("second", "metadata", 1024),
        ("disabled", "metadata", 1024),
        ("first", "capture", torch.device("cpu")),
        ("second", "capture", torch.device("cpu")),
    ]
    assert model.first._blocks_graph_runner is model.first.runner
    assert model.second._blocks_graph_runner is model.second.runner
    model.disabled.runner.capture_all.assert_not_called()

    for encoder in (model.first, model.second, model.disabled):
        encoder.clear_blocks_cuda_graph(release_nccl_window_owners=release_nccl_window_owners)
        encoder.clear_blocks_cuda_graph(release_nccl_window_owners=release_nccl_window_owners)
        assert encoder._blocks_graph_runner is None
    for encoder in (model.first, model.second):
        encoder.runner.clear.assert_called_once_with(
            release_nccl_window_owners=release_nccl_window_owners
        )
    model.disabled.runner.clear.assert_not_called()

    setup_mm_encoder_cuda_graphs(model)
    assert model.first.runner.capture_all.call_count == 2
    assert model.second.runner.capture_all.call_count == 2
    model.disabled.runner.capture_all.assert_not_called()


def test_encoder_graph_configs_are_selected_per_tower_and_validated_together() -> None:
    class Encoder(torch.nn.Module, MultimodalEncoderMixin):
        def __init__(self, modality: str, mm_config: MultimodalConfig) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.encoder_cuda_graph_modality = modality
            self.configure_encoder_cuda_graph(mm_config)
            self.runner = Mock()

        def _build_blocks_graph_runner(self, config: MultimodalEncoderCudaGraphConfig) -> Mock:
            assert config is self.encoder_cuda_graph_config
            return self.runner

    vision_config = MultimodalEncoderCudaGraphConfig(buckets=[(256, 1)])
    audio_config = MultimodalEncoderCudaGraphConfig(buckets=[(1024, 1)])
    mm_config = MultimodalConfig(
        encoder_cuda_graph={"vision": vision_config, "audio": audio_config}
    )
    model = torch.nn.Module()
    model.vision = Encoder("vision", mm_config)
    model.audio = Encoder("audio", mm_config)
    assert model.vision.encoder_cuda_graph_config is vision_config
    assert model.audio.encoder_cuda_graph_config is audio_config

    setup_mm_encoder_cuda_graphs(model, mm_config.encoder_cuda_graph)
    setup_mm_encoder_cuda_graphs(model, mm_config.encoder_cuda_graph)
    model.vision.runner.capture_all.assert_called_once_with(torch.device("cpu"))
    model.audio.runner.capture_all.assert_called_once_with(torch.device("cpu"))


def test_unsupported_encoder_graph_modality_fails_before_any_capture() -> None:
    class VisionEncoder(torch.nn.Module, MultimodalEncoderMixin):
        encoder_cuda_graph_modality = "vision"

        def enable_blocks_cuda_graph(self) -> None:
            raise AssertionError("Validation must happen before graph capture")

    model = torch.nn.Module()
    model.encoder = VisionEncoder()
    graph_configs = {"audio": MultimodalEncoderCudaGraphConfig(buckets=[(1024, 1)])}
    with pytest.raises(ValueError, match="Unsupported.*audio.*Supported.*vision"):
        setup_mm_encoder_cuda_graphs(model, graph_configs)


def test_automatic_encoder_graph_budgets_are_resolved_per_tower_without_mutating_config() -> None:
    class Encoder(torch.nn.Module, MultimodalEncoderMixin):
        def __init__(self, modality: str, mm_config: MultimodalConfig) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.encoder_cuda_graph_modality = modality
            self.configure_encoder_cuda_graph(mm_config)
            self.build_runner = Mock(return_value=Mock())

        def _build_blocks_graph_runner(self, config: MultimodalEncoderCudaGraphConfig) -> Mock:
            return self.build_runner(config)

    mm_config = MultimodalConfig(
        encoder_cuda_graph={
            "vision": {"max_batch_size": 8},
            "audio": {"max_batch_size": 4, "max_num_tokens": 512},
        }
    )
    model = torch.nn.Module()
    model.vision = Encoder("vision", mm_config)
    model.audio = Encoder("audio", mm_config)
    setup_mm_encoder_cuda_graphs(model, mm_config.encoder_cuda_graph, max_num_tokens=2048)
    setup_mm_encoder_cuda_graphs(model, mm_config.encoder_cuda_graph, max_num_tokens=2048)

    vision_config = model.vision.build_runner.call_args.args[0]
    audio_config = model.audio.build_runner.call_args.args[0]
    assert vision_config.max_batch_size == 8
    assert vision_config.max_num_tokens == 2048
    assert audio_config.max_batch_size == 4
    assert audio_config.max_num_tokens == 512
    assert mm_config.encoder_cuda_graph["vision"].max_num_tokens is None
    assert mm_config.encoder_cuda_graph["audio"].max_num_tokens == 512
    model.vision.build_runner.assert_called_once()
    model.audio.build_runner.assert_called_once()
    model.vision._blocks_graph_runner.capture_all.assert_called_once_with(torch.device("cpu"))
    model.audio._blocks_graph_runner.capture_all.assert_called_once_with(torch.device("cpu"))


def test_automatic_encoder_graphs_need_a_token_budget_before_capture() -> None:
    encoder = MultimodalEncoderMixin()
    encoder.encoder_cuda_graph_config = MultimodalEncoderCudaGraphConfig(max_batch_size=8)
    with pytest.raises(ValueError, match="from the config or engine"):
        encoder.enable_blocks_cuda_graph()


@pytest.mark.parametrize("encoder_max_num_tokens", [None, 1024])
@pytest.mark.parametrize("item_scheduling", [False, True])
def test_engine_captures_encoder_graphs_with_original_token_budget(
    monkeypatch: pytest.MonkeyPatch,
    encoder_max_num_tokens: int | None,
    item_scheduling: bool,
) -> None:
    from tensorrt_llm._torch.pyexecutor import _util, model_engine

    class Encoder(torch.nn.Module, MultimodalEncoderMixin):
        encoder_cuda_graph_modality = "vision"

        def __init__(self, mm_config: MultimodalConfig) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.configure_encoder_cuda_graph(mm_config)
            self.setup_attn_metadata = Mock()
            self.set_attn_max_seq_len = Mock()
            self.build_runner = Mock(return_value=Mock())

        def _build_blocks_graph_runner(self, config: MultimodalEncoderCudaGraphConfig) -> Mock:
            return self.build_runner(config)

    mm_config = MultimodalConfig(encoder_cuda_graph={"vision": {"max_batch_size": 8}})
    mapping = Mapping()
    model = torch.nn.Module()
    model.encoder = Encoder(mm_config)
    model.config = SimpleNamespace(torch_dtype=torch.float32)
    model.model_config = SimpleNamespace(
        pretrained_config=model.config, mapping=mapping, sparse_attention_config=None
    )
    llm_args = SimpleNamespace(
        encode_only=False,
        mm_encoder_only=False,
        get_runtime_sizes=lambda: (1, 4096, 128, 4),
        encoder_max_batch_size=None,
        encoder_max_num_tokens=encoder_max_num_tokens,
        enable_in_graph_sampling=False,
        enable_return_routed_experts=False,
        multimodal_config=mm_config,
        checkpoint_format="HF",
        trust_remote_code=False,
        disable_overlap_scheduler=True,
        kv_cache_config=None,
        sleep_config=None,
        enable_layerwise_nvtx_marker=False,
        cuda_graph_config=None,
        torch_compile_config=None,
        prefill_cuda_graph_backend=None,
        prefill_capture_num_tokens=None,
        allreduce_strategy="AUTO",
        attn_backend="TRTLLM",
        sparse_attention_config=None,
    )
    for name in (
        "should_enable_adp_dummy_fixes",
        "should_enable_scheduler_aware_adp_dummy",
        "should_enable_non_overlap_adp_forward_intent",
        "should_enable_overlap_headroom",
        "resolved_kv_cache_manager_is_v2",
    ):
        monkeypatch.setattr(_util, name, Mock(return_value=False))
    monkeypatch.setattr(_util, "compute_max_num_sequences", Mock(return_value=4))
    monkeypatch.setattr(model_engine, "_configure_deep_gemm_pdl", Mock())
    monkeypatch.setattr(model_engine, "create_input_processor", Mock(return_value=object()))
    monkeypatch.setattr(
        model_engine, "resolve_mrope_position_deltas_cache", Mock(return_value=None)
    )
    monkeypatch.setattr(model_engine, "is_hybrid_linear", Mock(return_value=False))
    scheduler = (
        SimpleNamespace(
            encoder_max_num_tokens=65536,
            output_budget_bytes=1024,
            bytes_per_embedding=4,
            attention_metadata_capacity=None,
        )
        if item_scheduling
        else None
    )
    monkeypatch.setattr(
        model_engine.MultimodalItemScheduler, "maybe_create", Mock(return_value=scheduler)
    )
    engine_cls = model_engine.ModelEngine
    monkeypatch.setattr(engine_cls, "_validate_breakable_cuda_graph_compatibility", Mock())
    monkeypatch.setattr(engine_cls, "_init_model_capacity", engine_cls._init_max_num_tokens)
    monkeypatch.setattr(engine_cls, "__del__", lambda self: None)
    monkeypatch.setattr(
        model_engine,
        "get_attention_backend",
        Mock(side_effect=RuntimeError("encoder graph setup complete")),
    )
    engine = engine_cls.__new__(engine_cls)
    with pytest.raises(RuntimeError, match="encoder graph setup complete"):
        engine_cls.__init__(
            engine,
            model_path="dummy",
            mapping=mapping,
            model=model,
            llm_args=llm_args,
            checkpoint_loader=Mock(),
        )

    configured_budget = encoder_max_num_tokens if encoder_max_num_tokens is not None else 4096
    assert engine.max_num_tokens == 512
    assert engine.encoder_max_num_tokens == (65536 if item_scheduling else configured_budget)
    model.encoder.set_attn_max_seq_len.assert_called_once_with(engine.encoder_max_num_tokens)
    capture_config = model.encoder.build_runner.call_args.args[0]
    assert capture_config.max_num_tokens == configured_budget
    assert mm_config.encoder_cuda_graph["vision"].max_num_tokens is None
    model.encoder.build_runner.assert_called_once()
    model.encoder._blocks_graph_runner.capture_all.assert_called_once_with(torch.device("cpu"))


@pytest.mark.parametrize("enable_padding", [False, True])
@pytest.mark.parametrize("automatic", [False, True], ids=["explicit", "automatic"])
def test_encoder_graphs_preserve_eager_metadata_capacity(
    enable_padding: bool, automatic: bool
) -> None:
    class Encoder(torch.nn.Module, MultimodalEncoderMixin):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_cuda_graph_config = (
                MultimodalEncoderCudaGraphConfig(
                    max_batch_size=8, max_num_tokens=64, enable_padding=enable_padding
                )
                if automatic
                else MultimodalEncoderCudaGraphConfig(
                    buckets=[(64, 8)], enable_padding=enable_padding
                )
            )
            self.setup_attn_metadata = Mock()
            self.set_attn_max_seq_len = Mock()

    model = torch.nn.Module()
    model.encoder = Encoder()
    capacities = {"attention": 4}
    setup_mm_encoder_attn_metadata(model, object(), 32, capacities)
    model.encoder.setup_attn_metadata.assert_called_once_with(
        max_num_tokens=32,
        attention_metadata_capacity={"attention": 4},
    )
    model.encoder.set_attn_max_seq_len.assert_called_once_with(64)
    assert capacities == {"attention": 4}


def test_encoder_graph_modality_validation_allows_delegated_encoders() -> None:
    model = torch.nn.Module()
    model.mm_encoder = None
    setup_mm_encoder_cuda_graphs(
        model, {"audio": MultimodalEncoderCudaGraphConfig(buckets=[(1024, 1)])}
    )


def _bind_items(
    mm_item_scheduler: MultimodalItemScheduler, request: LlmRequest, *, row_bytes: int = 8
) -> None:
    state = request.py_mm_encoder_state
    for item_idx, rows in enumerate(
        request.py_multimodal_params.item_metadata.output_embedding_lengths
    ):
        cache_key = make_mm_encoder_transient_cache_key(request.request_id, item_idx)
        assert mm_item_scheduler.encoder_cache.acquire(
            cache_key, rows * row_bytes, retain_after_release=False
        )
        state.set_item_cache_key(item_idx, cache_key, ready=False)


def test_qwen3_output_budget_uses_post_merge_embedding_capacity() -> None:
    from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase

    processor = object.__new__(Qwen3VLInputProcessorBase)
    processor._config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    model = SimpleNamespace(embedding_dim=16384, embedding_dtype=torch.float16)

    budget, bytes_per_embedding = resolve_mm_encoder_output_budget(processor, 65536, model)

    assert bytes_per_embedding == 32768
    assert budget == 512 * 1024**2


def test_output_row_bytes_use_config_dtype_without_embedding_weight() -> None:
    model = SimpleNamespace(
        embedding_dim=16384,
        model_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    )

    assert resolve_bytes_per_mm_encoder_embedding(model) == 32768


def test_output_budget_requires_processor_embedding_capacity() -> None:
    processor = SimpleNamespace(get_max_mm_encoder_output_embeddings=lambda *_: None)

    with pytest.raises(ValueError, match="get_max_mm_encoder_output_embeddings"):
        resolve_mm_encoder_output_budget(processor, 65536, None)


def test_eager_compatibility_is_checked_only_for_item_scheduled_models() -> None:
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.EAGER,
            encoder_side_stream_max_ahead=0,
        ),
        pipeline_parallel_size=1,
        enable_attention_dp=True,
        cache_transceiver_config=SimpleNamespace(backend="NIXL"),
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="attention DP"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)

    args.enable_attention_dp = False
    with pytest.raises(ValueError, match="disaggregated"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


def test_side_stream_compatibility_is_checked_only_for_item_scheduled_models() -> None:
    args = SimpleNamespace(
        multimodal_config=SimpleNamespace(
            encoder_scheduling_policy=MultimodalEncoderSchedulingPolicy.DEFAULT,
            encoder_side_stream_max_ahead=1,
        ),
        pipeline_parallel_size=1,
        enable_attention_dp=False,
        cache_transceiver_config=None,
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="side-stream prefetch"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)


@pytest.mark.parametrize(
    ("mm_config_kwargs", "falls_back"),
    [
        ({}, True),
        ({"encoder_scheduling_policy": "DEFAULT"}, True),
        ({"encoder_scheduling_policy": "EAGER"}, False),
    ],
    ids=["unset", "explicit-default", "explicit-eager"],
)
def test_pipeline_parallel_compatibility_is_checked_only_for_item_scheduled_models(
    mm_config_kwargs: dict, falls_back: bool
) -> None:
    args = SimpleNamespace(
        multimodal_config=MultimodalConfig(**mm_config_kwargs),
        pipeline_parallel_size=2,
        disable_mm_encoder=False,
        enable_attention_dp=False,
        cache_transceiver_config=None,
    )

    validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=False)

    with pytest.raises(ValueError, match="pipeline parallelism"):
        validate_mm_encoder_scheduling_compatibility(args, item_scheduling_enabled=True)

    class _Model(MultimodalModelMixin):
        supports_mm_encoder_item_scheduling = True

    create_kwargs = dict(
        llm_args=args, model=_Model(), input_processor=None, encoder_max_num_tokens=None
    )
    if falls_back:
        # DEFAULT runs inline encode even when set (model defaults mark every field
        # set); only EAGER fails startup.
        with patch(
            "tensorrt_llm._torch.pyexecutor.engine.multimodal.logger.warning_once"
        ) as warning_once:
            assert MultimodalItemScheduler.maybe_create(**create_kwargs) is None
        warning_once.assert_called_once()
    else:
        with pytest.raises(ValueError, match="pipeline parallelism"):
            MultimodalItemScheduler.maybe_create(**create_kwargs)


def test_item_encoder_classifies_request_state_contract_errors() -> None:
    mm_item_scheduler = bare_mm_item_scheduler(MultimodalModelMixin())
    request = make_llm_request(1)

    with pytest.raises(MultimodalEncoderRequestError, match="no longer active"):
        mm_item_scheduler.forward_items([], {request.request_id: [0]})

    with pytest.raises(MultimodalEncoderRequestError, match="no encoder item state"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


def test_item_encoder_classifies_output_count_contract_error() -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(
            self, _: Sequence[tuple[MultimodalParams, int]]
        ) -> list[tuple[SimpleNamespace, list[int], str]]:
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(
            self, _: Sequence[tuple[MultimodalParams, Sequence[int], str]]
        ) -> list[torch.Tensor]:
            return []

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])
    _bind_items(mm_item_scheduler, request)

    with pytest.raises(MultimodalEncoderRequestError, match="one output per item"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_translates_model_contract_errors(failure_stage: str) -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(
            self, _: Sequence[tuple[MultimodalParams, int]]
        ) -> list[tuple[SimpleNamespace, list[int], str]]:
            if failure_stage == "prepare":
                raise MultimodalEncoderContractError("bad request metadata")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(
            self, _: Sequence[tuple[MultimodalParams, Sequence[int], str]]
        ) -> list[torch.Tensor]:
            raise MultimodalEncoderContractError("bad encoder output rows")

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])
    _bind_items(mm_item_scheduler, request)
    # A follower of the scheduled entry keeps its reference to produce it later.
    follower = make_mm_request(2, [4])
    shared_key = request.py_mm_encoder_state.item_cache_keys[0]
    assert mm_item_scheduler.encoder_cache.acquire(shared_key, 8, retain_after_release=False)
    follower.py_mm_encoder_state.set_item_cache_key(0, shared_key, ready=False)

    expected = "bad request metadata" if failure_stage == "prepare" else "bad encoder output rows"
    with pytest.raises(MultimodalEncoderRequestError, match=expected) as error:
        mm_item_scheduler.forward_items([request, follower], {request.request_id: [0]})

    assert error.value.request_ids == {request.request_id}


@pytest.mark.parametrize("failure_stage", ["prepare", "forward"])
def test_item_encoder_does_not_translate_system_errors(failure_stage: str) -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(
            self, _: Sequence[tuple[MultimodalParams, int]]
        ) -> list[tuple[SimpleNamespace, list[int], str]]:
            if failure_stage == "prepare":
                raise torch.cuda.OutOfMemoryError("encoder OOM")
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(
            self, _: Sequence[tuple[MultimodalParams, Sequence[int], str]]
        ) -> list[torch.Tensor]:
            raise torch.cuda.OutOfMemoryError("encoder OOM")

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])
    _bind_items(mm_item_scheduler, request)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="encoder OOM"):
        mm_item_scheduler.forward_items([request], {request.request_id: [0]})


@pytest.mark.parametrize(
    "forward_failure,expected_error,scope_failed",
    [
        ("output_count", MultimodalEncoderRequestError, False),
        ("contract", MultimodalEncoderRequestError, False),
        ("system", torch.cuda.OutOfMemoryError, True),
    ],
)
def test_item_encoder_closes_window_scope_before_request_errors(
    forward_failure: str, expected_error: type, scope_failed: bool
) -> None:
    class _Model(MultimodalModelMixin):
        def prepare_multimodal_encoder_inputs(
            self, _: Sequence[tuple[MultimodalParams, int]]
        ) -> list[tuple[SimpleNamespace, list[int], str]]:
            encoder_input = SimpleNamespace(to_device=lambda *_args, **_kwargs: None)
            return [(encoder_input, [1], "image")]

        def forward_multimodal_encoder_items(
            self, _: Sequence[tuple[MultimodalParams, Sequence[int], str]]
        ) -> list[torch.Tensor]:
            if forward_failure == "contract":
                raise MultimodalEncoderContractError("bad encoder output rows")
            if forward_failure == "system":
                raise torch.cuda.OutOfMemoryError("encoder OOM")
            return []

    class _RecordingScope:
        exits = []

        def __init__(self, _inputs: object) -> None:
            pass

        def __enter__(self) -> _RecordingScope:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: TracebackType | None,
        ) -> bool:
            self.exits.append(exc_type is not None)
            return False

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    request = make_mm_request(1, [4])
    _bind_items(mm_item_scheduler, request)

    with patch(
        "tensorrt_llm._torch.pyexecutor.engine.multimodal.discard_nccl_window_tensor_outputs",
        _RecordingScope,
    ):
        with pytest.raises(expected_error):
            mm_item_scheduler.forward_items([request], {request.request_id: [0]})

    # Request-level failures leave the scope normally so its window leases are
    # released; anything else still fails the scope and quarantines them.
    assert _RecordingScope.exits == [scope_failed]


@pytest.mark.parametrize("invalid_output", ["rows", "width", "dtype", "rank", "type"])
def test_item_encoder_validates_all_outputs_before_commit(invalid_output: str) -> None:
    class _Model(MultimodalModelMixin):
        embedding_dtype = torch.bfloat16

        def prepare_multimodal_encoder_inputs(
            self, _: Sequence[tuple[MultimodalParams, int]]
        ) -> list[tuple[SimpleNamespace, list[int], str]]:
            return []

        def forward_multimodal_encoder_items(
            self, _: Sequence[tuple[MultimodalParams, Sequence[int], str]]
        ) -> list[torch.Tensor | None]:
            return outputs

    outputs = [torch.ones(1, 4, dtype=torch.bfloat16) for _ in range(3)]
    outputs[1] = {
        "rows": torch.ones(2, 4, dtype=torch.bfloat16),
        "width": torch.ones(1, 3, dtype=torch.bfloat16),
        # Same byte size as bfloat16: byte accounting alone cannot catch this.
        "dtype": torch.ones(1, 4, dtype=torch.float16),
        "rank": torch.ones(4, dtype=torch.bfloat16),
        "type": None,
    }[invalid_output]
    actual = {
        "rows": "a torch.bfloat16 tensor with shape (2, 4)",
        "width": "a torch.bfloat16 tensor with shape (1, 3)",
        "dtype": "a torch.float16 tensor with shape (1, 4)",
        "rank": "a torch.bfloat16 tensor with shape (4,)",
        "type": "NoneType",
    }[invalid_output]
    scheduler = bare_mm_item_scheduler(_Model())
    scheduler.bytes_per_embedding = 8
    producers = [make_mm_request(request_id, [4]) for request_id in (1, 2, 3)]
    for request in producers:
        _bind_items(scheduler, request)
    follower = make_mm_request(4, [4])
    shared_key = producers[1].py_mm_encoder_state.item_cache_keys[0]
    assert scheduler.encoder_cache.acquire(shared_key, 8, retain_after_release=False)
    follower.py_mm_encoder_state.set_item_cache_key(0, shared_key, ready=False)
    requests = [*producers, follower]
    selected = {request.request_id: [0] for request in producers}

    with pytest.raises(MultimodalEncoderRequestError, match="must produce") as error:
        scheduler.forward_items(requests, selected)

    assert str(error.value).endswith(f"got {actual}")
    assert error.value.request_ids == {2, 4}
    assert scheduler.encoder_cache.current_bytes == 0
    for request in requests:
        state = request.py_mm_encoder_state
        assert state.item_ready == [False]
        assert state.item_cache_keys[0] is not None
        assert "image" in request.py_multimodal_data

    outputs[1] = torch.ones(1, 4, dtype=torch.bfloat16)
    scheduler.forward_items(requests, selected)
    assert scheduler.encoder_cache.current_bytes == 24
    assert all(is_multimodal_encoder_ready(request) for request in requests)
    # Raw inputs stay until prefill completes, in case a released item is needed again.
    assert all("image" in request.py_multimodal_data for request in requests)


@pytest.mark.parametrize("window", [None, (0, 3), (2, 7), (6, 8), (3, 5)])
def test_item_outputs_commit_to_prompt_ordered_cache_keys(
    monkeypatch: pytest.MonkeyPatch,
    window: tuple[int, int] | None,
) -> None:
    class _Model(MultimodalModelMixin):
        embedding_dtype = torch.float32

        def forward_multimodal_encoder_items(
            self, encoder_inputs: Sequence[tuple[MultimodalParams, Sequence[int], str]]
        ) -> list[torch.Tensor]:
            return [
                torch.full((embedding_length, 2), float(embedding_length))
                for _, embedding_lengths, _ in encoder_inputs
                for embedding_length in embedding_lengths
            ]

    monkeypatch.setattr(MultimodalParams, "to_device", lambda self, *args, **kwargs: self)
    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    mm_item_scheduler.bytes_per_embedding = 8
    # Two image-A rows, a text-only gap, then three image-B rows.
    embed_mask_cumsum = torch.tensor([0, 1, 2, 2, 2, 3, 4, 5])
    request = make_llm_request(
        1,
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(5).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 3]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                item_refs=[("image", 0), ("image", 1)],
                encoder_token_lengths=[2, 3],
                output_embedding_lengths=[2, 3],
            ),
            "multimodal_embed_mask_cumsum": embed_mask_cumsum,
        },
        input_tokens=list(range(8)),
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=8)
    _bind_items(mm_item_scheduler, request)
    state = request.py_mm_encoder_state

    mm_item_scheduler.forward_items([request], {request.request_id: [0]})

    assert state.item_ready == [True, False]
    first = mm_item_scheduler.encoder_cache.get(state.item_cache_keys[0])
    torch.testing.assert_close(first, torch.full((2, 2), 2.0))
    assert "image" in request.py_multimodal_data

    mm_item_scheduler.forward_items([request], {request.request_id: [1]})

    second = mm_item_scheduler.encoder_cache.get(state.item_cache_keys[1])
    torch.testing.assert_close(second, torch.full((3, 2), 3.0))
    assert "multimodal_embedding" not in request.py_multimodal_data
    assert "image" in request.py_multimodal_data
    assert is_multimodal_encoder_ready(request)

    runtime = None
    if window is not None:
        runtime = MultimodalRuntimeData(
            embed_mask_cumsum=embed_mask_cumsum,
            past_seen_token_num=window[0],
            chunk_end_pos=window[1],
        )
        # Progress past image A releases it; later chunks read only image B.
        request.context_current_position = window[0]
        release_consumed_mm_items(request, mm_item_scheduler.encoder_cache)
    start = runtime.num_cached_mm_tokens if runtime is not None else 0
    end = start + runtime.num_mm_tokens_in_chunk if runtime is not None else 5
    expected = torch.cat([first, second])[start:end]
    with patch("torch.cat", wraps=torch.cat) as join:
        multimodal_data = mm_item_scheduler.build_multimodal_data_for_llm(request, runtime)
    output = multimodal_data["multimodal_embedding"]
    # A chunk without image rows gets an empty embedding on the default device.
    torch.testing.assert_close(output, expected, check_device=False)
    assert join.call_count == int(start < 2 < end)
    if start < end and join.call_count == 0:
        source = first if end <= 2 else second
        assert output.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert multimodal_data["multimodal_embedding_is_chunk"]
    assert "multimodal_embedding_is_chunk" not in request.py_multimodal_data
    assert "multimodal_embedding" not in request.py_multimodal_data
    assert mm_item_scheduler.encoder_cache.current_bytes == (24 if start >= 2 else 40)


def test_admission_requires_item_rows_to_fill_prompt_embedding_slots() -> None:
    def make_request(embed_mask_cumsum: list[int]) -> LlmRequest:
        return make_llm_request(
            1,
            multimodal_data={
                "image": {"pixel_values": torch.empty(5, 1)},
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    item_refs=[("image", 0), ("image", 1)],
                    encoder_token_lengths=[2, 3],
                    output_embedding_lengths=[2, 3],
                ),
                "multimodal_embed_mask_cumsum": torch.tensor(embed_mask_cumsum),
            },
        )

    # Two image-A slots, a text-only gap, then three image-B slots.
    initialize_multimodal_encoder_request(make_request([0, 1, 2, 2, 2, 3, 4, 5]), max_num_tokens=8)
    for embed_mask_cumsum in ([0, 1, 2, 2, 2, 3, 4], [0, 1, 2, 2, 2, 3, 4, 5, 6]):
        with pytest.raises(ValueError, match="declare 5 embedding rows"):
            initialize_multimodal_encoder_request(make_request(embed_mask_cumsum), max_num_tokens=8)


def test_cache_hit_only_request_omits_raw_inputs_from_llm_payload() -> None:
    class _Model(MultimodalModelMixin):
        embedding_dtype = torch.float32

    mm_item_scheduler = bare_mm_item_scheduler(_Model())
    mm_item_scheduler.bytes_per_embedding = 8
    request = make_llm_request(
        1,
        multimodal_data={
            "image": {
                "pixel_values": torch.arange(2).unsqueeze(1),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                item_refs=[("image", 0)],
                encoder_token_lengths=[2],
                output_embedding_lengths=[2],
            ),
        },
    )
    initialize_multimodal_encoder_request(request, max_num_tokens=8)
    # Bind the item to an already committed entry, as a scheduler cache hit
    # does, without running `forward_items`.
    encoder_cache = mm_item_scheduler.encoder_cache
    cache_key = make_mm_encoder_transient_cache_key(request.request_id, 0)
    assert encoder_cache.acquire(cache_key, 16, retain_after_release=False)
    encoder_cache.commit(cache_key, torch.ones(2, 2))
    request.py_mm_encoder_state.set_item_cache_key(0, cache_key, ready=True)
    assert is_multimodal_encoder_ready(request)

    multimodal_data = mm_item_scheduler.build_multimodal_data_for_llm(request)

    assert "image" not in multimodal_data
    torch.testing.assert_close(multimodal_data["multimodal_embedding"], torch.ones(2, 2))
    # Only the per-forward payload drops the raw inputs.
    assert "image" in request.py_multimodal_data


class MultimodalSetupTestCase(unittest.TestCase):
    def test_build_request_multimodal_input_skips_when_cache_disabled(self) -> None:
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
        self.assertIsNone(_build_request_multimodal_input(request, cache_enabled=False))

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
            ({"image": processor_max_num_tokens}, processor_max_num_tokens),
            ({"image": 4096}, encoder_max_num_tokens),
            ({}, encoder_max_num_tokens),
            (None, encoder_max_num_tokens),
        ]
        for max_tokens_per_item, expected_max_seq_len in cases:
            with self.subTest(max_tokens_per_item=max_tokens_per_item):
                encoder = CapturingEncoder()
                if max_tokens_per_item is None:
                    input_processor = Mock()
                else:
                    input_processor = Mock(spec=BaseMultimodalDummyInputsBuilder)
                    input_processor.get_mm_max_tokens_per_item.return_value = max_tokens_per_item

                setup_mm_encoder_attn_metadata(
                    torch.nn.Sequential(encoder), input_processor, encoder_max_num_tokens, None
                )

                self.assertEqual(encoder.setup_args, encoder_max_num_tokens)
                self.assertEqual(encoder.max_seq_len, expected_max_seq_len)
