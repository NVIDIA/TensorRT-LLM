# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_params import CudaGraphLoraParams
from tensorrt_llm._torch.peft.lora.layer import LoraLayer, LoraModuleType
from tensorrt_llm._torch.peft.lora.loaders import HfLoraLoader, get_all_hf_lora_weights
from tensorrt_llm._torch.peft.lora.manager import LoraManager, LoraModelConfig, load_torch_lora
from tensorrt_llm._torch.peft.lora.ownership import DecoderLoraOwner
from tensorrt_llm._torch.pyexecutor.engine.lora import (
    LoraParamBuilder,
    make_lora_model_config,
    resolve_lora_owner,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.resource_manager import PeftCacheManager
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _Decoder(torch.nn.Module):
    def __init__(self, module_type: LoraModuleType, output_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, torch_dtype=torch.float16)
        self.block = torch.nn.Module()
        self.block.layer_idx = 0
        self.block.lora = LoraLayer([module_type], [output_size])


class _MultimodalModel(torch.nn.Module, MultimodalModelMixin):
    lora_owner = DecoderLoraOwner(module_path="decoder")

    def __init__(self, module_type: LoraModuleType, output_size: int) -> None:
        super().__init__()
        self.encoder = _Decoder(module_type, 20)
        self.decoder = _Decoder(module_type, output_size)


@dataclass
class _Runtime:
    config: LoraConfig
    model_config: LoraModelConfig
    manager: LoraManager
    worker: BaseWorker
    cache: PeftCacheManager
    transfer: Mock
    request: SimpleNamespace | None = None


@pytest.fixture(params=[False, True], ids=["eager", "graphs"])
def graph_enabled(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=["hf", "nemo"])
def checkpoint_source(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=["worker", "lazy"])
def entrypoint(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def cpu_allocations(monkeypatch: pytest.MonkeyPatch) -> Mock:
    transfer = Mock(side_effect=lambda tensor: tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: transfer(tensor))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 0))
    # Registration and key equality remain real; graph buffers require CUDA.
    monkeypatch.setattr(CudaGraphLoraParams, "__init__", lambda self, **kwargs: None)
    return transfer


def _make_runtime(
    source: str,
    graphs: bool,
    transfer: Mock,
    *,
    module_name: str | None = None,
    hf_modules: str | None = None,
) -> _Runtime:
    module_name = module_name or ("attn_dense" if source == "hf" else "attn_qkv")
    hf_modules = hf_modules or ("o_proj" if source == "hf" else "attn_qkv")
    module_type = LoraModuleType(LoraManager.LORA_MODULE_IDS[module_name])
    output_size = 4 if source == "hf" else 12
    config = LoraConfig(
        lora_ckpt_source=source,
        lora_target_modules=[module_name],
        trtllm_modules_to_hf_modules={module_name: hf_modules},
        max_loras=2,
        max_lora_rank=2,
    )
    model = _MultimodalModel(module_type, output_size)
    owner, decoder = resolve_lora_owner(model, None)
    model_config = make_lora_model_config(
        decoder,
        config.lora_target_modules,
        config.trtllm_modules_to_hf_modules,
        owner=owner,
    )
    runner = object.__new__(DecoderRunner)
    runner._model = model
    runner._lora = LoraParamBuilder(attn_backend=object(), cuda_graph_manager=None)
    runner.lora_model_config = model_config
    runner._graphs = SimpleNamespace(enabled=graphs)
    runner._config = SimpleNamespace(
        is_spec_decode=False,
        max_batch_size=2,
        max_num_tokens=2,
        spec_config=None,
        attention_backend=object(),
    )
    runner.init_cuda_graph_lora_manager(config, decoder)
    graph_manager = runner._lora.cuda_graph_manager
    if graphs:
        key = CudaGraphLoraParams.LoraLayerKey(0, (module_type,))
        assert graph_manager.layer_info == {
            key: CudaGraphLoraParams.LoraLayerInfo(module_num=1, output_sizes=[output_size])
        }
        assert graph_manager.lora_model_config is model_config
    else:
        assert graph_manager is None

    cache_impl = Mock()
    cache_impl.is_task_cached.return_value = False
    manager = LoraManager(
        mapping=Mapping(), model_config=model_config, cpp_peft_cache_manager=cache_impl
    )
    cache = object.__new__(PeftCacheManager)
    cache.impl = cache_impl
    cache._lora_manager = manager
    cache._lora_model_config = model_config
    cache._lora_config = config
    worker = object.__new__(BaseWorker)
    worker.doing_shutdown = True
    worker._runtime_model_config = None
    worker._lora_model_config = model_config
    worker._lora_manager = manager
    return _Runtime(config, model_config, manager, worker, cache, transfer)


@pytest.fixture
def runtime(checkpoint_source: str, graph_enabled: bool, cpu_allocations: Mock) -> _Runtime:
    return _make_runtime(checkpoint_source, graph_enabled, cpu_allocations)


@pytest.fixture
def hf_runtime(graph_enabled: bool, cpu_allocations: Mock) -> _Runtime:
    return _make_runtime("hf", graph_enabled, cpu_allocations)


def _weights(source: str, prefix: str) -> dict[str, torch.Tensor]:
    if source == "hf":
        stem = f"{prefix}.0.self_attn.o_proj"
        return {
            f"{stem}.lora_A.weight": torch.full((2, 4), 1.0, dtype=torch.float16),
            f"{stem}.lora_B.weight": torch.full((4, 2), 2.0, dtype=torch.float16),
        }
    stem = f"{prefix}.0.self_attention.adapter_layer.lora_kqv_adapter"
    return {
        f"{stem}.linear_in.weight": torch.full((2, 4), 1.0, dtype=torch.float16),
        f"{stem}.linear_out.weight": torch.full((12, 2), 2.0, dtype=torch.float16),
    }


def _write_adapter(
    directory: Path, source: str, weights: dict[str, torch.Tensor], *, use_bin: bool = False
) -> Path:
    directory.mkdir()
    if source == "hf":
        if use_bin:
            torch.save(weights, directory / "adapter_model.bin")
        else:
            save_file(weights, directory / "adapter_model.safetensors")
        (directory / "adapter_config.json").write_text(
            json.dumps({"r": 2, "lora_alpha": 2, "target_modules": ["o_proj"]})
        )
        return directory
    checkpoint = io.BytesIO()
    torch.save(weights, checkpoint)
    path = directory / "adapter.nemo"
    with tarfile.open(path, "w") as archive:
        for name, contents in (
            ("model_weights.ckpt", checkpoint.getvalue()),
            ("model_config.yaml", b"model:\n  hidden_size: 4\n  num_layers: 1\n"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            archive.addfile(member, io.BytesIO(contents))
    return path


def _decoder_prefix(source: str, prefixed: bool = False) -> str:
    if source == "hf":
        return (
            "base_model.model.language_model.model.layers"
            if prefixed
            else "base_model.model.model.layers"
        )
    return "model.language_model.encoder.layers" if prefixed else "model.layers"


def _load(runtime: _Runtime, entrypoint: str, path: Path) -> None:
    if entrypoint == "worker":
        runtime.worker._load_lora_adapter(
            LoRARequest("adapter", 7, str(path), runtime.config.lora_ckpt_source)
        )
    else:
        runtime.request = SimpleNamespace(
            lora_task_id="7",
            py_lora_path=str(path),
            lora_weights=None,
            lora_config=None,
            py_lora_ckpt_source=runtime.config.lora_ckpt_source,
        )
        runtime.cache.add_request_peft(runtime.request)


def _assert_not_cached(runtime: _Runtime) -> None:
    assert runtime.manager.cpp_lora_weights == {}
    assert runtime.manager.cpp_lora_config == {}
    assert runtime.manager._lora_uid_to_low_ranks == {}
    assert runtime.manager._lora_weights_pointers_list == {}
    assert runtime.manager.lora_weights == []
    runtime.transfer.assert_not_called()
    runtime.cache.impl.add_request_peft.assert_not_called()
    if runtime.request is not None:
        assert runtime.request.lora_weights is None
        assert runtime.request.lora_config is None


def _assert_loaded(runtime: _Runtime, entrypoint: str) -> None:
    source = runtime.config.lora_ckpt_source
    expected_id = 4 if source == "hf" else 0
    assert runtime.manager.cpp_lora_config["7"][:, :3].tolist() == [[expected_id, 0, 2]]
    expected = torch.cat(
        (
            torch.ones(8, dtype=torch.float16),
            torch.full((8 if source == "hf" else 24,), 2.0, dtype=torch.float16),
        )
    )
    torch.testing.assert_close(runtime.manager.cpp_lora_weights["7"], expected.unsqueeze(0))
    assert runtime.transfer.call_count == 2
    if entrypoint == "lazy":
        runtime.cache.impl.add_request_peft.assert_called_once_with(runtime.request, True)
        torch.testing.assert_close(runtime.request.lora_weights, expected.reshape(1, 1, -1))
    else:
        runtime.cache.impl.add_request_peft.assert_not_called()


@pytest.mark.parametrize("prefixed", [False, True], ids=["flat", "language-model"])
def test_decoder_aliases_load_through_worker_and_lazy_cache(
    runtime: _Runtime, entrypoint: str, prefixed: bool, tmp_path: Path
) -> None:
    source = runtime.config.lora_ckpt_source
    path = _write_adapter(
        tmp_path / "adapter", source, _weights(source, _decoder_prefix(source, prefixed))
    )
    _load(runtime, entrypoint, path)
    _assert_loaded(runtime, entrypoint)


@pytest.mark.parametrize(
    "component", ["vision", "audio", "connector", "vision.language_model", "language_model_extra"]
)
def test_out_of_scope_adapter_is_rejected_before_cache_and_uid_can_be_reused(
    runtime: _Runtime, entrypoint: str, component: str, tmp_path: Path
) -> None:
    source = runtime.config.lora_ckpt_source
    suffix = "layers" if "language_model" in component else "encoder.layers"
    prefix = (
        f"base_model.model.{component}.{suffix}"
        if source == "hf"
        else f"model.{component}.{suffix}"
    )
    invalid_weights = _weights(source, prefix)
    path = _write_adapter(tmp_path / "invalid", source, invalid_weights)
    with pytest.raises(ValueError, match="outside selected owner 'decoder'") as exc:
        _load(runtime, entrypoint, path)
    assert any(key in str(exc.value) for key in invalid_weights)
    _assert_not_cached(runtime)

    valid = _write_adapter(tmp_path / "valid", source, _weights(source, _decoder_prefix(source)))
    _load(runtime, entrypoint, valid)
    _assert_loaded(runtime, entrypoint)


def test_duplicate_decoder_aliases_are_rejected_before_cache(
    runtime: _Runtime, entrypoint: str, tmp_path: Path
) -> None:
    source = runtime.config.lora_ckpt_source
    flat = _weights(source, _decoder_prefix(source))
    prefixed = _weights(source, _decoder_prefix(source, True))
    path = _write_adapter(tmp_path / "duplicate", source, flat | prefixed)
    with pytest.raises(ValueError, match="duplicate destination") as exc:
        _load(runtime, entrypoint, path)
    assert any(key in str(exc.value) for key in flat)
    assert any(key in str(exc.value) for key in prefixed)
    _assert_not_cached(runtime)


@pytest.mark.parametrize(
    "module",
    [
        "multi_modal_projector.linear_1",
        "model.embed_tokens_extend.image_embed.img_projection",
        "vision.embed_tokens_projector",
        "audio.lm_head_projector",
    ],
)
def test_non_layered_projector_rejected_at_startup_and_runtime(
    runtime: _Runtime, entrypoint: str, module: str, tmp_path: Path
) -> None:
    source = runtime.config.lora_ckpt_source
    weights = _weights(source, _decoder_prefix(source))
    key = (
        f"base_model.model.{module}.lora_A.weight"
        if source == "hf"
        else f"model.{module}.adapter_layer.lora_kqv_adapter.linear_in.weight"
    )
    weights[key] = torch.ones(2, 4, dtype=torch.float16)
    path = _write_adapter(tmp_path / "mixed", source, weights)
    config = runtime.config.model_copy(update={"lora_dir": [str(path)]})

    for load in (
        lambda: load_torch_lora(config, owner=runtime.model_config.owner),
        lambda: _load(runtime, entrypoint, path),
    ):
        with pytest.raises(ValueError, match="Encoder/projector LoRA is not supported") as exc:
            load()
        assert key in str(exc.value)
        assert "decoder-only adapter checkpoint" in str(exc.value)
        _assert_not_cached(runtime)


def test_mixed_checkpoint_rejects_unmapped_encoder_weights(
    hf_runtime: _Runtime, entrypoint: str, tmp_path: Path
) -> None:
    weights = _weights("hf", _decoder_prefix("hf"))
    invalid = _weights("hf", "base_model.model.vision.encoder.layers")
    invalid = {key.replace("o_proj", "unmapped_proj"): value for key, value in invalid.items()}
    path = _write_adapter(tmp_path / "mixed", "hf", weights | invalid)
    config = hf_runtime.config.model_copy(update={"lora_dir": [str(path)]})

    for load in (
        lambda: load_torch_lora(config, owner=hf_runtime.model_config.owner),
        lambda: _load(hf_runtime, entrypoint, path),
    ):
        with pytest.raises(ValueError, match="outside selected owner 'decoder'") as exc:
            load()
        assert any(key in str(exc.value) for key in invalid)
        _assert_not_cached(hf_runtime)


@pytest.fixture(
    params=[
        ("gemma3vl", "Gemma3VLM"),
        ("mistral", "Mistral3VLM"),
        ("step3p7vl", "Step3p7VLForConditionalGeneration"),
        ("gemma4mm", "Gemma4ForConditionalGeneration"),
        ("step3p7", "Step3p7ForCausalLM"),
        ("qwen4_exp", "Qwen4ExpForCausalLM"),
    ],
    ids=["gemma3", "mistral3-dense", "step3p7", "gemma4", "step3p7-text", "qwen4-exp-text"],
)
def nested_language_model_owner(request: pytest.FixtureRequest) -> DecoderLoraOwner:
    module_name, class_name = request.param
    module = import_module(f"tensorrt_llm._torch.models.modeling_{module_name}")
    model_type = getattr(module, class_name)
    if module_name in {"step3p7", "qwen4_exp"}:
        model = model_type.__new__(model_type)
        torch.nn.Module.__init__(model)
        owner, selected_model = resolve_lora_owner(model, None)
        assert selected_model is model
        assert owner == DecoderLoraOwner()
        return owner
    if module_name == "mistral":
        model = model_type.__new__(model_type)
        torch.nn.Module.__init__(model)
        model.llm = module.MistralForCausalLM.__new__(module.MistralForCausalLM)
        torch.nn.Module.__init__(model.llm)
        return model.lora_owner
    return model_type.lora_owner


def test_nested_language_model_alias_parses_decoder_weights(
    nested_language_model_owner: DecoderLoraOwner,
) -> None:
    weights = _weights("hf", "base_model.model.model.language_model.layers")

    parsed = get_all_hf_lora_weights(
        weights,
        {"o_proj"},
        owner=nested_language_model_owner,
        hf_modules_to_trtllm_modules={"o_proj": "attn_dense"},
    )

    assert set(parsed) == {0}
    assert set(parsed[0]) == {"o_proj"}
    assert set(parsed[0]["o_proj"]) == {"in", "out"}
    assert torch.all(parsed[0]["o_proj"]["in"] == 1)
    assert torch.all(parsed[0]["o_proj"]["out"] == 2)


def test_nested_language_model_alias_rejects_duplicate_flat_destination(
    nested_language_model_owner: DecoderLoraOwner,
) -> None:
    nested = _weights("hf", "base_model.model.model.language_model.layers")
    flat = _weights("hf", "base_model.model.model.layers")

    with pytest.raises(ValueError, match="duplicate destination") as exc:
        get_all_hf_lora_weights(
            nested | flat,
            {"o_proj"},
            owner=nested_language_model_owner,
            hf_modules_to_trtllm_modules={"o_proj": "attn_dense"},
        )

    assert next(iter(nested)) in str(exc.value)
    assert next(iter(flat)) in str(exc.value)
    assert f"module path '{nested_language_model_owner.module_path or '<root>'}'" in str(exc.value)


@pytest.mark.parametrize("explicit_targets", [False, True])
def test_initial_adapter_accepts_nested_language_model_alias(
    nested_language_model_owner: DecoderLoraOwner, explicit_targets: bool, tmp_path: Path
) -> None:
    path = _write_adapter(
        tmp_path / "adapter", "hf", _weights("hf", "base_model.model.model.language_model.layers")
    )
    config = LoraConfig(
        lora_dir=[str(path)],
        lora_target_modules=["attn_dense"] if explicit_targets else [],
        trtllm_modules_to_hf_modules={"attn_dense": "o_proj"},
    )

    dtype = load_torch_lora(config, owner=nested_language_model_owner)

    assert config.lora_target_modules == ["attn_dense"]
    assert dtype == torch.float16


@pytest.mark.parametrize("explicit_targets", [False, True])
@pytest.mark.parametrize("invalid_kind", ["component", "duplicate"])
def test_initial_adapter_validation_cannot_be_bypassed_by_explicit_targets(
    runtime: _Runtime, explicit_targets: bool, invalid_kind: str, tmp_path: Path
) -> None:
    source = runtime.config.lora_ckpt_source
    weights = _weights(source, _decoder_prefix(source))
    if invalid_kind == "component":
        prefix = (
            "base_model.model.vision.encoder.layers"
            if source == "hf"
            else "model.vision.encoder.layers"
        )
        weights.update(_weights(source, prefix))
    else:
        weights.update(_weights(source, _decoder_prefix(source, True)))
    path = _write_adapter(tmp_path / "invalid", source, weights)
    config = runtime.config.model_copy(
        update={
            "lora_dir": [str(path)],
            "lora_target_modules": runtime.config.lora_target_modules if explicit_targets else [],
        }
    )
    with pytest.raises(ValueError, match="selected owner 'decoder'"):
        load_torch_lora(config, owner=runtime.model_config.owner)
    _assert_not_cached(runtime)


@pytest.mark.parametrize("prefixed", [False, True])
def test_initial_adapter_inference_accepts_decoder_aliases(
    runtime: _Runtime, prefixed: bool, tmp_path: Path
) -> None:
    source = runtime.config.lora_ckpt_source
    path = _write_adapter(
        tmp_path / "adapter", source, _weights(source, _decoder_prefix(source, prefixed))
    )
    config = runtime.config.model_copy(update={"lora_dir": [str(path)], "lora_target_modules": []})
    dtype = load_torch_lora(config, owner=runtime.model_config.owner)
    assert config.lora_target_modules == runtime.config.lora_target_modules
    assert dtype == (torch.float16 if source == "hf" else None)
    _assert_not_cached(runtime)


def test_hf_module_aliases_cannot_overwrite_one_canonical_destination(
    hf_runtime: _Runtime, entrypoint: str, tmp_path: Path
) -> None:
    hf_runtime.model_config.trtllm_modules_to_hf_modules["attn_dense"] = ["o_proj", "out_proj"]
    weights = _weights("hf", _decoder_prefix("hf"))
    duplicate = {key.replace("o_proj", "out_proj"): value.clone() for key, value in weights.items()}
    path = _write_adapter(tmp_path / "duplicate", "hf", weights | duplicate)
    with pytest.raises(ValueError, match="duplicate destination") as exc:
        _load(hf_runtime, entrypoint, path)
    assert "o_proj" in str(exc.value) and "out_proj" in str(exc.value)
    _assert_not_cached(hf_runtime)


def test_hf_experts_and_matrix_directions_have_distinct_destinations(tmp_path: Path) -> None:
    weights = {}
    for expert in (0, 1):
        stem = f"base_model.model.model.layers.0.mlp.experts.{expert}.gate_proj"
        weights[f"{stem}.lora_A.weight"] = torch.full(
            (2, 4), float(expert + 1), dtype=torch.float16
        )
        weights[f"{stem}.lora_B.weight"] = torch.full(
            (4, 2), float(expert + 3), dtype=torch.float16
        )
    path = _write_adapter(tmp_path / "experts", "hf", weights)
    loader = HfLoraLoader([str(path)])
    mapping = {"moe_gate": "mlp.experts.gate_proj"}
    assert loader.get_target_modules(mapping) == ["moe_gate"]
    parsed = get_all_hf_lora_weights(
        loader.lora_weight,
        {"mlp.experts.gate_proj"},
        hf_modules_to_trtllm_modules={"mlp.experts.gate_proj": "moe_gate"},
    )
    experts = parsed[0]["mlp.experts.gate_proj"]
    assert set(experts) == {0, 1}
    for expert in (0, 1):
        assert set(experts[expert]) == {"in", "out"}
        assert torch.all(experts[expert]["in"] == expert + 1)
        assert torch.all(experts[expert]["out"] == expert + 3)


def _moe_weights(indexed: bool, expert: int = 0) -> dict[str, torch.Tensor]:
    module = f"mlp.experts.{expert}.w1" if indexed else "mlp.w1"
    stem = f"base_model.model.model.layers.0.{module}"
    return {
        f"{stem}.lora_A.weight": torch.full((2, 4), float(2 * expert + 1), dtype=torch.float16),
        f"{stem}.lora_B.weight": torch.full((4, 2), float(2 * expert + 2), dtype=torch.float16),
    }


@pytest.mark.parametrize("flat_first", [False, True], ids=["indexed-first", "flat-first"])
def test_mixed_moe_representations_are_rejected_at_every_loading_entrypoint(
    graph_enabled: bool,
    cpu_allocations: Mock,
    entrypoint: str,
    flat_first: bool,
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(
        "hf",
        graph_enabled,
        cpu_allocations,
        module_name="moe_h_to_4h",
        hf_modules="w1",
    )
    flat, indexed = _moe_weights(False), _moe_weights(True)
    weights = (flat | indexed) if flat_first else (indexed | flat)
    # Torch checkpoints retain insertion order, so both parser orders reach runtime loading.
    path = _write_adapter(tmp_path / "mixed", "hf", weights, use_bin=True)
    loader = HfLoraLoader([str(path)], owner=runtime.model_config.owner)
    assert list(loader.lora_weight) == list(weights)
    with pytest.raises(ValueError, match="mix flat and expert-indexed weights"):
        get_all_hf_lora_weights(
            loader.lora_weight,
            {"w1"},
            owner=runtime.model_config.owner,
            hf_modules_to_trtllm_modules={"w1": "moe_h_to_4h"},
        )
    config = runtime.config.model_copy(update={"lora_dir": [str(path)]})
    with pytest.raises(ValueError, match="mix flat and expert-indexed weights"):
        load_torch_lora(config, owner=runtime.model_config.owner)
    with pytest.raises(ValueError, match="mix flat and expert-indexed weights") as exc:
        _load(runtime, entrypoint, path)
    assert next(iter(flat)) in str(exc.value)
    assert next(iter(indexed)) in str(exc.value)
    assert "selected owner 'decoder'" in str(exc.value)
    _assert_not_cached(runtime)


@pytest.mark.parametrize("indexed", [False, True], ids=["packed", "two-experts"])
def test_moe_representations_load_when_only_one_is_present(
    graph_enabled: bool, cpu_allocations: Mock, entrypoint: str, indexed: bool, tmp_path: Path
) -> None:
    runtime = _make_runtime(
        "hf",
        graph_enabled,
        cpu_allocations,
        module_name="moe_h_to_4h",
        hf_modules="w1",
    )
    weights = _moe_weights(indexed)
    if indexed:
        weights.update(_moe_weights(True, 1))
    path = _write_adapter(tmp_path / "moe", "hf", weights)
    config = runtime.config.model_copy(update={"lora_dir": [str(path)]})
    load_torch_lora(config, owner=runtime.model_config.owner)
    _load(runtime, entrypoint, path)
    assert runtime.manager.cpp_lora_config["7"][:, :3].tolist() == [[13, 0, 2]]
    expected = torch.cat(
        [
            weight.flatten()
            for direction in ("A", "B")
            for key, weight in weights.items()
            if f"lora_{direction}.weight" in key
        ]
    )
    torch.testing.assert_close(runtime.manager.cpp_lora_weights["7"], expected.unsqueeze(0))
    assert runtime.transfer.call_count == 2
    if entrypoint == "lazy":
        runtime.cache.impl.add_request_peft.assert_called_once_with(runtime.request, True)


def test_decoder_target_filtering_remains_distinct_from_owner_validation(
    hf_runtime: _Runtime, entrypoint: str, tmp_path: Path
) -> None:
    hf_runtime.model_config.trtllm_modules_to_hf_modules["attn_q"] = "q_proj"
    weights = _weights("hf", _decoder_prefix("hf"))
    weights.update(
        {key.replace("o_proj", "q_proj"): value.clone() for key, value in weights.items()}
    )
    path = _write_adapter(tmp_path / "adapter", "hf", weights)
    with pytest.warns(UserWarning, match="not in target modules"):
        _load(hf_runtime, entrypoint, path)
    _assert_loaded(hf_runtime, entrypoint)


def test_saved_decoder_embeddings_and_head_remain_supported(
    hf_runtime: _Runtime, entrypoint: str, tmp_path: Path
) -> None:
    weights = _weights("hf", _decoder_prefix("hf"))
    weights["base_model.model.lm_head.weight"] = torch.full((8, 4), 3.0, dtype=torch.float16)
    weights["base_model.model.model.embed_tokens.weight"] = torch.full(
        (8, 4), 4.0, dtype=torch.float16
    )
    path = _write_adapter(tmp_path / "adapter", "hf", weights)
    config_file = path / "adapter_config.json"
    config = json.loads(config_file.read_text())
    config["modules_to_save"] = ["lm_head", "embed_tokens"]
    config_file.write_text(json.dumps(config))
    loader = HfLoraLoader([str(path)], owner=hf_runtime.model_config.owner)
    torch.testing.assert_close(loader.lm_head, weights["base_model.model.lm_head.weight"])
    torch.testing.assert_close(
        loader.embed_tokens, weights["base_model.model.model.embed_tokens.weight"]
    )
    startup_config = hf_runtime.config.model_copy(update={"lora_dir": [str(path)]})
    assert load_torch_lora(startup_config, owner=hf_runtime.model_config.owner) == torch.float16
    _load(hf_runtime, entrypoint, path)
    _assert_loaded(hf_runtime, entrypoint)


@pytest.mark.parametrize("saved_module", ["lm_head", "embed_tokens"])
def test_saved_encoder_weights_cannot_bypass_owner_validation(
    hf_runtime: _Runtime, entrypoint: str, saved_module: str, tmp_path: Path
) -> None:
    weights = _weights("hf", _decoder_prefix("hf"))
    invalid_key = f"base_model.model.vision.{saved_module}.weight"
    weights[invalid_key] = torch.ones(4, 4, dtype=torch.float16)
    path = _write_adapter(tmp_path / "invalid", "hf", weights)
    with pytest.raises(ValueError, match="outside selected owner") as exc:
        _load(hf_runtime, entrypoint, path)
    assert invalid_key in str(exc.value)
    _assert_not_cached(hf_runtime)


def test_nemo_unsupported_targets_keep_the_format_error(tmp_path: Path) -> None:
    path = _write_adapter(tmp_path / "adapter", "nemo", _weights("nemo", "model.layers"))
    config = LoraConfig(
        lora_dir=[str(path)],
        lora_ckpt_source="nemo",
        lora_target_modules=["attn_qkv", "mlp_h_to_4h"],
    )
    with pytest.raises(ValueError, match="NeMo LoRA only supports"):
        load_torch_lora(config)


def test_nemo_unsupported_key_keeps_key_error(tmp_path: Path) -> None:
    path = _write_adapter(
        tmp_path / "adapter",
        "nemo",
        {"model.layers.0.unsupported.weight": torch.ones(1, dtype=torch.float16)},
    )
    with pytest.raises(KeyError, match="unsupported key"):
        load_torch_lora(LoraConfig(lora_dir=[str(path)], lora_ckpt_source="nemo"))


def test_nemo_invalid_file_type_keeps_path_error(tmp_path: Path) -> None:
    path = tmp_path / "adapter.pt"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="not a .nemo file"):
        load_torch_lora(LoraConfig(lora_dir=[str(path)], lora_ckpt_source="nemo"))


def test_unknown_checkpoint_source_keeps_dispatch_error() -> None:
    with pytest.raises(ValueError, match="Unsupported lora_ckpt_source"):
        load_torch_lora(SimpleNamespace(lora_ckpt_source="unknown"))
