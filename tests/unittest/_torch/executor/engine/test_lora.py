# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LoRA parameter construction and CUDA-graph model scope."""

from collections.abc import Sequence
from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import CudaGraphLoraParams
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.pyexecutor.engine.lora import (
    LoraParamBuilder,
    make_cuda_graph_lora_manager,
)
from tensorrt_llm._torch.pyexecutor.engine.model_call import ModelCaller
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.encoder_decoder import EncoderDecoderRunner
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import model_extra_attrs

pytestmark = pytest.mark.cpu_only

# kCONTEXT / kGENERATION as the LoRA op's C++ kernel sees them.
K_CONTEXT = 0
K_GENERATION = 1


def _request(request_id: int, lora_task_id: int | None) -> SimpleNamespace:
    return SimpleNamespace(py_request_id=request_id, lora_task_id=lora_task_id)


def _module(
    *,
    layer_id: int,
    module_id: int,
    adapter_size: int,
    weights_in: int,
    weights_out: int,
    scaling_vec_pointer: int | None = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=layer_id,
        module_id=module_id,
        adapter_size=adapter_size,
        weights_in_pointer=weights_in,
        weights_out_pointer=weights_out,
        scaling_vec_pointer=scaling_vec_pointer,
    )


def _batch(
    context: Sequence[SimpleNamespace] = (), generation: Sequence[SimpleNamespace] = ()
) -> ScheduledRequests:
    batch = ScheduledRequests()
    batch.context_requests_last_chunk = list(context)
    batch.generation_requests = list(generation)
    return batch


def _attn_metadata(
    *, num_contexts: int = 1, num_generations: int = 0, prompt_lens: list[int] | None = None
) -> SimpleNamespace:
    num_seqs = num_contexts + num_generations
    request_types = [K_CONTEXT] * num_contexts + [K_GENERATION] * num_generations
    if prompt_lens is None:
        prompt_lens = [5] * num_contexts + [1] * num_generations
    return SimpleNamespace(
        host_request_types=torch.tensor(request_types, dtype=torch.int32),
        prompt_lens_cpu=torch.tensor(prompt_lens, dtype=torch.int32),
        num_seqs=num_seqs,
        num_contexts=num_contexts,
        num_generations=num_generations,
    )


def _peft_cache_manager(
    peft_table: dict[int, list[SimpleNamespace]] | None, data_type: torch.dtype = torch.float16
) -> SimpleNamespace:
    return SimpleNamespace(
        get_and_reset_batch_peft_table=lambda: peft_table,
        data_type=data_type,
    )


def _builder(
    *, is_linear_tree: bool = True, extend_ctx: bool = False, cuda_graph_manager=None
) -> LoraParamBuilder:
    spec_config = SimpleNamespace(
        is_linear_tree=is_linear_tree,
        spec_dec_mode=SimpleNamespace(extend_ctx=lambda backend: extend_ctx),
    )
    return LoraParamBuilder(
        spec_config=spec_config, attn_backend=object(), cuda_graph_manager=cuda_graph_manager
    )


def _build(
    builder: LoraParamBuilder,
    batch: ScheduledRequests,
    attn_metadata: SimpleNamespace,
    peft_table: dict[int, list[SimpleNamespace]] | None,
    **kwargs: Any,
) -> dict | None:
    """Drive the eager path: no CUDA-graph manager, so ``maybe_graph`` is moot."""
    kwargs.setdefault("enable_spec_decode", False)
    kwargs.setdefault("runtime_draft_len", 0)
    return builder.build(
        batch,
        attn_metadata,
        peft_cache_manager=_peft_cache_manager(peft_table),
        **kwargs,
    )


def test_single_request_produces_tensors_of_the_declared_dtypes() -> None:
    batch = _batch(context=[_request(0, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=3, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(_builder(), batch, _attn_metadata(), peft_table)

    entry = params[3][1]
    assert entry["adapter_size"].dtype == torch.int32
    assert entry["weight_pointers"].dtype == torch.int64
    assert entry["adapter_size"].tolist() == [4]
    assert entry["weight_pointers"].tolist() == [11, 22, 0]
    assert params["data_type"] is torch.float16


def test_request_without_an_adapter_is_padded_in_batch_order() -> None:
    """The zero rows must land at the non-LoRA request's position, not at the end."""
    batch = _batch(
        context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=None)]
    )
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(_builder(), batch, _attn_metadata(num_generations=1), peft_table)

    entry = params[0][1]
    assert entry["adapter_size"].tolist() == [4, 0]
    assert entry["weight_pointers"].tolist() == [11, 22, 0, 0, 0, 0]


def test_requests_touching_different_modules_take_the_union() -> None:
    """Each request must get a row in every module the batch touches."""
    batch = _batch(context=[_request(0, lora_task_id=7), _request(1, lora_task_id=8)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)],
        8: [_module(layer_id=0, module_id=2, adapter_size=6, weights_in=33, weights_out=44)],
    }

    params = _build(_builder(), batch, _attn_metadata(num_contexts=2), peft_table)

    assert sorted(params[0]) == [1, 2]
    assert params[0][1]["adapter_size"].tolist() == [4, 0]
    assert params[0][1]["weight_pointers"].tolist() == [11, 22, 0, 0, 0, 0]
    assert params[0][2]["adapter_size"].tolist() == [0, 6]
    assert params[0][2]["weight_pointers"].tolist() == [0, 0, 0, 33, 44, 0]


def test_absent_scaling_vector_becomes_a_null_pointer() -> None:
    batch = _batch(context=[_request(0, lora_task_id=7)])
    peft_table = {
        7: [
            _module(
                layer_id=0,
                module_id=1,
                adapter_size=4,
                weights_in=11,
                weights_out=22,
                scaling_vec_pointer=None,
            )
        ]
    }

    params = _build(_builder(), batch, _attn_metadata(), peft_table)

    assert params[0][1]["weight_pointers"].tolist() == [11, 22, 0]


def test_spec_decode_relabels_generation_requests_as_context() -> None:
    """One generation request covers runtime_draft_len + 1 tokens, so it is fed as context."""
    batch = _batch(context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }
    attn_metadata = _attn_metadata(num_generations=1)

    params = _build(
        _builder(),
        batch,
        attn_metadata,
        peft_table,
        enable_spec_decode=True,
        runtime_draft_len=3,
    )

    assert params["host_request_types"].tolist() == [K_CONTEXT, K_CONTEXT]
    assert params["prompt_lens_cpu"].tolist() == [5, 4]
    assert params["num_seqs"] == 2
    # The caller's metadata is shared with the rest of input preparation.
    assert attn_metadata.host_request_types.tolist() == [K_CONTEXT, K_GENERATION]
    assert attn_metadata.prompt_lens_cpu.tolist() == [5, 1]


def test_spec_decode_state_is_read_per_call_not_snapshotted() -> None:
    """The same builder must relabel or not purely on this call's arguments.

    Warmup and CUDA-graph capture rewrite ``enable_spec_decode`` and
    ``runtime_draft_len`` between forwards; a builder that captured them at
    construction would keep using stale token counts without erroring.
    """
    builder = _builder()
    batch = _batch(context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    spec_on = _build(
        builder,
        batch,
        _attn_metadata(num_generations=1),
        peft_table,
        enable_spec_decode=True,
        runtime_draft_len=3,
    )
    spec_off = _build(
        builder,
        batch,
        _attn_metadata(num_generations=1),
        peft_table,
        enable_spec_decode=False,
        runtime_draft_len=3,
    )

    assert spec_on["host_request_types"].tolist() == [K_CONTEXT, K_CONTEXT]
    assert spec_off["host_request_types"].tolist() == [K_CONTEXT, K_GENERATION]
    assert spec_off["prompt_lens_cpu"].tolist() == [5, 1]


@pytest.mark.parametrize(
    "builder_kwargs",
    [{"is_linear_tree": False}, {"extend_ctx": True}],
    ids=["not-linear-tree", "extend-ctx"],
)
def test_spec_decode_relabelling_needs_the_whole_predicate(builder_kwargs: dict[str, bool]) -> None:
    """Either construction-time condition alone suppresses the re-labelling."""
    batch = _batch(context=[_request(0, lora_task_id=7)], generation=[_request(1, lora_task_id=7)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(
        _builder(**builder_kwargs),
        batch,
        _attn_metadata(num_generations=1),
        peft_table,
        enable_spec_decode=True,
        runtime_draft_len=3,
    )

    assert params["host_request_types"].tolist() == [K_CONTEXT, K_GENERATION]


def test_missing_peft_table_yields_none_and_an_empty_one_yields_an_empty_dict() -> None:
    """``None`` and ``{}`` are distinct results; callers pass both to the model."""
    batch = _batch(context=[_request(0, lora_task_id=7)])
    builder = _builder()

    no_manager = builder.build(
        batch,
        _attn_metadata(),
        enable_spec_decode=False,
        runtime_draft_len=0,
        peft_cache_manager=None,
    )
    empty_table = _build(builder, batch, _attn_metadata(), {})

    assert no_manager is None
    assert empty_table == {}


def test_batch_without_any_adapter_gets_no_metadata_keys() -> None:
    """No LoRA request means no parameters at all -- not zero-filled ones."""
    batch = _batch(context=[_request(0, lora_task_id=None)])
    peft_table = {
        7: [_module(layer_id=0, module_id=1, adapter_size=4, weights_in=11, weights_out=22)]
    }

    params = _build(_builder(), batch, _attn_metadata(), peft_table)

    assert params == {}


class _FakeCudaGraphLoraManager:
    """Records the calls `build` makes on a CUDA-graph LoRA manager."""

    def __init__(self, graph_params: dict | None = None) -> None:
        self._graph_params = graph_params
        self.base_only_calls: list[object] = []
        self.tokens_per_seq_calls: list[int] = []
        self.evicted_for: list[object] = []
        self.adapter_slot_manager = SimpleNamespace(
            remove_evicted_slots_in_cpp=self.evicted_for.append
        )

    def prepare_base_only_batch(self, peft_cache_manager: object) -> None:
        self.base_only_calls.append(peft_cache_manager)

    def prepare_cuda_graph_lora_params(
        self,
        scheduled_requests: ScheduledRequests,
        attn_metadata: SimpleNamespace,
        peft_cache_manager: object,
        tokens_per_seq: int,
    ) -> dict | None:
        self.tokens_per_seq_calls.append(tokens_per_seq)
        return self._graph_params


def test_base_only_batch_short_circuits_before_any_parameter_build() -> None:
    """A graph batch with no LoRA request reserves base-only slots and returns nothing."""
    manager = _FakeCudaGraphLoraManager()
    peft_cache_manager = _peft_cache_manager({})

    params = _builder(cuda_graph_manager=manager).build(
        _batch(context=[_request(0, lora_task_id=None)]),
        _attn_metadata(),
        enable_spec_decode=False,
        runtime_draft_len=0,
        peft_cache_manager=peft_cache_manager,
        maybe_graph=True,
        use_lora_graph=False,
    )

    assert params is None
    assert manager.base_only_calls == [peft_cache_manager]
    assert manager.tokens_per_seq_calls == []


@pytest.mark.parametrize(
    ("enable_spec_decode", "runtime_draft_len", "expected_tokens_per_seq"),
    [(False, 3, 1), (True, 0, 1), (True, 3, 4)],
    ids=["spec-decode-off", "no-draft-tokens", "spec-decode-on"],
)
def test_cuda_graph_path_sizes_tokens_per_seq(
    enable_spec_decode: bool, runtime_draft_len: int, expected_tokens_per_seq: int
) -> None:
    """Only a spec-decode batch with draft tokens covers more than one token per sequence."""
    graph_params = {"source": "cuda-graph"}
    manager = _FakeCudaGraphLoraManager(graph_params=graph_params)

    params = _builder(cuda_graph_manager=manager).build(
        _batch(generation=[_request(0, lora_task_id=7)]),
        _attn_metadata(num_contexts=0, num_generations=1),
        enable_spec_decode=enable_spec_decode,
        runtime_draft_len=runtime_draft_len,
        peft_cache_manager=_peft_cache_manager({}),
        maybe_graph=True,
        use_lora_graph=True,
    )

    assert params is graph_params
    assert manager.tokens_per_seq_calls == [expected_tokens_per_seq]


def test_eager_fallback_releases_evicted_adapter_slots() -> None:
    """Falling back to eager with a live manager must still drop slots the cache evicted."""
    manager = _FakeCudaGraphLoraManager()
    peft_cache_manager = _peft_cache_manager({})

    _builder(cuda_graph_manager=manager).build(
        _batch(context=[_request(0, lora_task_id=None)]),
        _attn_metadata(),
        enable_spec_decode=False,
        runtime_draft_len=0,
        peft_cache_manager=peft_cache_manager,
        maybe_graph=False,
    )

    assert manager.evicted_for == [peft_cache_manager]
    assert manager.base_only_calls == []
    assert manager.tokens_per_seq_calls == []


class _LoraBlock(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int | None,
        output_sizes: list[int],
        module_types: Sequence[LoraModuleType] = (LoraModuleType.ATTENTION_Q,),
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.lora = LoraLayer(list(module_types), output_sizes)
        self.workspace_reservations: list[tuple[int, int, int]] = []
        self.forward_calls = 0

    def reserve_moe_lora_cuda_graph_workspace(
        self, max_tokens: int, max_rank: int, max_adapters: int
    ) -> None:
        self.workspace_reservations.append((max_tokens, max_rank, max_adapters))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return value + 1


class _MultimodalLoraModel(torch.nn.Module):
    def __init__(self, layer_idx: int | None = 7) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(_LoraBlock(layer_idx, [24]))
        self.language_model = torch.nn.Sequential(_LoraBlock(layer_idx, [96]))
        self.model_config = SimpleNamespace(extra_attrs={})

    def forward(self, value: torch.Tensor, attn_metadata: object) -> torch.Tensor:
        return self.language_model(self.encoder(value))


@pytest.fixture
def graph_lora_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[LoraConfig, LoraModelConfig]:
    # Keep the real layer-key equality and discovery; only tensor allocation needs CUDA.
    monkeypatch.setattr(CudaGraphLoraParams, "__init__", lambda self, **kwargs: None)
    return (
        LoraConfig(max_loras=4, max_lora_rank=8, lora_target_modules=["attn_q"]),
        LoraModelConfig(
            lora_target_modules=["attn_q"],
            trtllm_modules_to_hf_modules={},
            hidden_size=96,
            dtype="float16",
        ),
    )


def _graph_manager(
    model: torch.nn.Module,
    settings: tuple[LoraConfig, LoraModelConfig],
    max_num_tokens: int = 17,
) -> CudaGraphLoraManager:
    return make_cuda_graph_lora_manager(
        model,
        *settings,
        max_batch_size=3,
        max_tokens_per_seq=2,
        max_num_tokens=max_num_tokens,
    )


@pytest.mark.parametrize("layer_idx", [7, None])
def test_cuda_graph_lora_excludes_encoder_with_same_local_key(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig], layer_idx: int | None
) -> None:
    model = _MultimodalLoraModel(layer_idx)
    encoder_lora = model.encoder[0].lora

    manager = _graph_manager(model, graph_lora_settings)

    key = CudaGraphLoraParams.LoraLayerKey(layer_idx, (LoraModuleType.ATTENTION_Q,))
    assert manager.layer_info == {
        key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[96])
    }
    assert model.encoder[0].lora is encoder_lora
    assert encoder_lora.output_hidden_sizes == [24]


def test_cuda_graph_lora_excludes_encoder_without_layer_indices(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig],
) -> None:
    model = _MultimodalLoraModel()
    del model.encoder[0].layer_idx
    del model.language_model[0].layer_idx
    model.encoder[0].lora = LoraLayer([LoraModuleType.MLP_H_TO_4H], [24])
    model.language_model[0].lora = LoraLayer([LoraModuleType.MLP_H_TO_4H], [96])
    graph_lora_settings[0].lora_target_modules = ["mlp_h_to_4h"]
    graph_lora_settings[1].lora_target_modules = ["mlp_h_to_4h"]

    manager = _graph_manager(model, graph_lora_settings)

    key = CudaGraphLoraParams.LoraLayerKey(None, (LoraModuleType.MLP_H_TO_4H,))
    assert manager.layer_info == {
        key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[96])
    }


@pytest.mark.parametrize("layer_idx", [7, None])
def test_cuda_graph_lora_still_rejects_duplicate_decoder_keys(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig], layer_idx: int | None
) -> None:
    model = _MultimodalLoraModel(layer_idx)
    model.encoder = torch.nn.Identity()
    model.language_model.append(_LoraBlock(layer_idx, [112]))

    with pytest.raises(AssertionError, match="already exists"):
        _graph_manager(model, graph_lora_settings)


def test_cuda_graph_lora_filters_targets_before_duplicate_detection(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig],
) -> None:
    model = _MultimodalLoraModel()
    fused_types = (
        LoraModuleType.ATTENTION_Q,
        LoraModuleType.ATTENTION_K,
        LoraModuleType.ATTENTION_V,
    )
    model.language_model = torch.nn.Sequential(
        _LoraBlock(7, [96, 32, 32], fused_types),
        _LoraBlock(7, [96], (LoraModuleType.ATTENTION_DENSE,)),
        _LoraBlock(7, [96], (LoraModuleType.ATTENTION_DENSE,)),
    )
    graph_lora_settings[0].lora_target_modules = ["attn_k"]
    graph_lora_settings[1].lora_target_modules = ["attn_k"]

    manager = _graph_manager(model, graph_lora_settings)

    key = CudaGraphLoraParams.LoraLayerKey(7, fused_types)
    assert manager.layer_info == {
        key: CudaGraphLoraParams.LoraLayerInfo(module_num=3, output_sizes=[96, 32, 32])
    }


def test_cuda_graph_lora_keeps_text_only_model_scope(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig],
) -> None:
    model = torch.nn.Sequential(_LoraBlock(7, [96]))

    manager = _graph_manager(model, graph_lora_settings)

    key = CudaGraphLoraParams.LoraLayerKey(7, (LoraModuleType.ATTENTION_Q,))
    assert manager.layer_info == {
        key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[96])
    }


@pytest.mark.parametrize(("max_num_tokens", "reserved_tokens"), [(4, 6), (17, 17)])
def test_cuda_graph_lora_reserves_workspace_only_on_decoder(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig],
    max_num_tokens: int,
    reserved_tokens: int,
) -> None:
    model = _MultimodalLoraModel()
    model.encoder[0].layer_idx = 11

    _graph_manager(model, graph_lora_settings, max_num_tokens)

    assert model.encoder[0].workspace_reservations == []
    assert model.language_model[0].workspace_reservations == [(reserved_tokens, 8, 4)]


def _runner_for_lora(
    runner_type: type[DecoderRunner],
    model: torch.nn.Module,
    lora_model_config: LoraModelConfig,
) -> DecoderRunner:
    runner = runner_type.__new__(runner_type)
    runner.model = model
    runner._model_caller = ModelCaller(model)
    runner.lora_model_config = lora_model_config
    runner.cuda_graph_runner = SimpleNamespace(enabled=True)
    runner._config = SimpleNamespace(
        is_spec_decode=True,
        original_max_draft_len=1,
        max_batch_size=3,
        max_num_tokens=17,
        spec_config=None,
        attention_backend=object(),
    )
    return runner


def test_runner_lora_initialization_preserves_outer_multimodal_forward(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig],
) -> None:
    model = _MultimodalLoraModel()
    runner = _runner_for_lora(DecoderRunner, model, graph_lora_settings[1])

    runner.init_cuda_graph_lora_manager(graph_lora_settings[0])

    assert runner.model is model
    assert runner._lora.cuda_graph_manager is not None
    with model_extra_attrs({}):
        output = runner._model_caller(value=torch.tensor([1]), attn_metadata=model)
    assert output.tolist() == [3]
    assert model.encoder[0].forward_calls == 1
    assert model.language_model[0].forward_calls == 1


def test_encoder_decoder_runner_inherits_lora_model_fallback(
    graph_lora_settings: tuple[LoraConfig, LoraModelConfig],
) -> None:
    model = torch.nn.Sequential(_LoraBlock(7, [96]))
    runner = _runner_for_lora(EncoderDecoderRunner, model, graph_lora_settings[1])

    runner.init_cuda_graph_lora_manager(graph_lora_settings[0])

    key = CudaGraphLoraParams.LoraLayerKey(7, (LoraModuleType.ATTENTION_Q,))
    assert runner._lora.cuda_graph_manager.layer_info == {
        key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[96])
    }
    assert runner.model is model


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("gemma3vl", "Gemma3VLM"),
        ("phi4mm", "Phi4MMForCausalLM"),
        ("qwen2vl", "Qwen2VLModelBase"),
        ("step3p7vl", "Step3p7VLForConditionalGeneration"),
        ("nemotron_h_multimodal", "NemotronHMultimodalModel"),
        ("mistral", "Mistral3VLM"),
    ],
)
def test_multimodal_language_model_property_returns_llm(module_name: str, class_name: str) -> None:
    model_type = getattr(
        import_module(f"tensorrt_llm._torch.models.modeling_{module_name}"), class_name
    )
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.llm = torch.nn.Module()

    assert model.language_model is model.llm
