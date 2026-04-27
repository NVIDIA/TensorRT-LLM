# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import types
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from tensorrt_llm._torch.auto_deploy.export.export import (
    _build_aliasing_load_pre_hook,
    _load_hook_for_deduplication,
)
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactory
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    BaseTransform,
    MemStats,
    SharedConfig,
    Stages,
    TransformInfo,
    TransformRegistry,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fusion import _insert_fused_gemm
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    ParameterUpdateInfo,
    ShardingTransformConfig,
    ShardingTransformContainer,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.transform.pipeline_cache import (
    HOOKS_FILE_NAME,
    MODULE_FILE_NAME,
    PipelineCacheConfig,
    _load_graphmodule_structural,
    _save_graphmodule_structural,
    _use_cached_graphmodule_code_for_pickling,
    collect_hook_specs,
    reattach_hooks,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_weight_name, is_linear_op

_COUNTERS = {
    "build": 0,
    "boundary": 0,
    "after": 0,
}
_MUTATING_CONFIG_RUNS = 0


class _ToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.weight


class _RequiresArgTracer(torch.fx.Tracer):
    def __init__(self, scope_root):
        super().__init__()
        self.scope_root = scope_root


class _CallModuleToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


class _EmbeddingToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class _WrapperWithGraphModule(nn.Module):
    def __init__(self, graph_module):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = graph_module
        self.vision_tower = nn.Linear(1, 1, bias=False, device="meta")

    def forward(self, x):
        return self.language_model.model(x) + self.vision_tower(x)


class _DummyFactory(ModelFactory):
    @property
    def max_seq_len(self) -> int:
        return 8

    def _build_model(self, device: str) -> nn.Module:
        return _ToyModule().to(device)

    def _load_checkpoint(self, model: nn.Module, device, disable_preload: bool = False):
        return None

    def get_export_infos(self, model: nn.Module):
        return []


class _SourceHookBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(2, 3)
        self._register_load_state_dict_pre_hook(_fake_embedding_rename_hook)


class _SourceHookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.backbone = _SourceHookBackbone()

    def forward(self, x):
        return x + self.weight


class _SourceHookGraphModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, device="meta"))
        self.backbone = nn.Module()
        self.backbone.add_module("embeddings", nn.Embedding(2, 3, device="meta"))
        self.backbone._register_load_state_dict_pre_hook(_fake_embedding_rename_hook)

    def forward(self, x):
        token_ids = torch.tensor([0], dtype=torch.long)
        return x + self.weight + self.backbone.embeddings(token_ids).sum()


class _SourceHookFactory(ModelFactory):
    @property
    def max_seq_len(self) -> int:
        return 8

    def _build_model(self, device: str) -> nn.Module:
        return _SourceHookModel().to(device)

    def _load_checkpoint(self, model: nn.Module, device, disable_preload: bool = False):
        return None

    def get_export_infos(self, model: nn.Module):
        return []


def _fake_embedding_rename_hook(state_dict, prefix, *args, **kwargs):
    del args, kwargs
    for key in list(state_dict.keys()):
        if key == f"{prefix}embedding.weight":
            state_dict[f"{prefix}embeddings.weight"] = state_dict.pop(key)


@TransformRegistry.register("_unit_test_build_graph_for_pipeline_cache")
class _BuildGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_ToyModule().to("meta"))
        gm.meta["build_counter"] = _COUNTERS["build"]
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_call_module_graph_for_pipeline_cache")
class _BuildCallModuleGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        return symbolic_trace(_CallModuleToyModule().to("meta")), TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_wrapper_for_pipeline_cache")
class _BuildWrapperForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_ToyModule().to("meta"))
        return _WrapperWithGraphModule(gm), TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_mutating_config_before_pipeline_cache")
class _MutatingConfigBeforePipelineCache(BaseTransform):
    def _apply(self, gm, cm, factory, shared_config: SharedConfig):
        global _MUTATING_CONFIG_RUNS

        _MUTATING_CONFIG_RUNS += 1
        self.config.mutated_cache_key_field = _MUTATING_CONFIG_RUNS
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_forward_hook_graph_for_pipeline_cache")
class _BuildForwardHookGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_ToyModule().to("meta"))

        def local_forward_pre_hook(*args, **kwargs):
            del args, kwargs

        gm.register_forward_pre_hook(local_forward_pre_hook)
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_source_hook_graph_for_pipeline_cache")
class _BuildSourceHookGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_SourceHookGraphModule())
        gm.backbone._register_load_state_dict_pre_hook(_fake_embedding_rename_hook)
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_unknown_ad_hook_graph_for_pipeline_cache")
class _BuildUnknownADHookGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_ToyModule().to("meta"))

        def unsupported_hook(state_dict, prefix, *args, **kwargs):
            del state_dict, prefix, args, kwargs

        unsupported_hook.__module__ = "tensorrt_llm._torch.auto_deploy.transform.unit_test"
        gm._register_load_state_dict_pre_hook(unsupported_hook)
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_boundary_for_pipeline_cache")
class _BoundaryForPipelineCache(BaseTransform):
    def _apply(self, gm, cm, factory, shared_config: SharedConfig):
        _COUNTERS["boundary"] += 1
        gm.register_buffer("boundary_marker", torch.tensor([_COUNTERS["boundary"]]))
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_after_boundary_for_pipeline_cache")
class _AfterBoundaryForPipelineCache(BaseTransform):
    def _apply(self, gm, cm, factory, shared_config: SharedConfig):
        _COUNTERS["after"] += 1
        gm.register_buffer("after_marker", torch.tensor([_COUNTERS["after"]]))
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


def _reset_counters():
    global _MUTATING_CONFIG_RUNS

    for key in _COUNTERS:
        _COUNTERS[key] = 0
    _MUTATING_CONFIG_RUNS = 0


def _patch_mem_stats(monkeypatch):
    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )


def _cache_seq_interface():
    return CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=16,
        device="cpu",
    )


def _optimizer_config(tmp_path, build_transform="_unit_test_build_graph_for_pipeline_cache"):
    return {
        build_transform: {
            "stage": "export",
            "run_per_gm": False,
        },
        "_unit_test_boundary_for_pipeline_cache": {
            "stage": "sharding",
        },
        "pipeline_cache": {
            "stage": "sharding",
            "enabled": True,
            "root": str(tmp_path),
            "trust_cache_root": True,
        },
        "_unit_test_after_boundary_for_pipeline_cache": {
            "stage": "weight_load",
        },
    }


def test_pipeline_cache_config_defaults_root_for_enabled_cache():
    config = PipelineCacheConfig(stage=Stages.SHARDING, enabled=True)
    unsupported_transform_options = {
        "debug_visualize_dir",
        "expect_mem_change",
        "requires_clean_graph",
        "requires_shape_prop",
        "run_graph_cleanup",
        "run_per_gm",
        "run_shape_prop",
        "skip_on_error",
    }

    assert unsupported_transform_options.isdisjoint(PipelineCacheConfig.model_fields)
    assert config.run_per_gm is False
    assert config.requires_clean_graph is False
    assert config.requires_shape_prop is False
    assert config.run_graph_cleanup is False
    assert config.run_shape_prop is False
    assert config.root == str(
        Path.home() / ".cache" / "tensorrt_llm" / "auto_deploy" / "pipeline_cache"
    )
    assert config.trust_cache_root


def test_pipeline_cache_config_accepts_user_root_by_default(tmp_path):
    config = PipelineCacheConfig(
        stage=Stages.SHARDING,
        enabled=True,
        root=str(tmp_path),
    )

    assert config.root == str(tmp_path)
    assert config.trust_cache_root


def test_pipeline_cache_config_rejects_explicit_untrusted_root(tmp_path):
    with pytest.raises(ValueError, match="trust_cache_root=true"):
        PipelineCacheConfig(
            stage=Stages.SHARDING,
            enabled=True,
            root=str(tmp_path),
            trust_cache_root=False,
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("debug_visualize_dir", "/tmp/debug"),
        ("expect_mem_change", True),
        ("requires_clean_graph", True),
        ("requires_shape_prop", True),
        ("run_graph_cleanup", True),
        ("run_per_gm", False),
        ("run_per_gm", True),
        ("run_shape_prop", True),
        ("skip_on_error", True),
    ],
)
def test_pipeline_cache_config_rejects_unsupported_transform_options(
    tmp_path, field_name, field_value
):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PipelineCacheConfig(
            stage=Stages.SHARDING,
            enabled=True,
            root=str(tmp_path),
            **{field_name: field_value},
        )


def test_pipeline_cache_transform_restores_and_skips_prefix(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()

    optimizer_first = InferenceOptimizer(factory=factory, config=_optimizer_config(tmp_path))
    model_first = optimizer_first(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert hasattr(model_first, "boundary_marker")
    assert hasattr(model_first, "after_marker")

    optimizer_second = InferenceOptimizer(factory=factory, config=_optimizer_config(tmp_path))
    model_second = optimizer_second(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert hasattr(model_second, "boundary_marker")
    assert hasattr(model_second, "after_marker")
    assert list(tmp_path.rglob(MODULE_FILE_NAME))
    assert list(tmp_path.rglob(HOOKS_FILE_NAME))


def test_pipeline_cache_ignores_post_boundary_runtime_transform_config(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config_first = _optimizer_config(tmp_path)
    config_first["_unit_test_after_boundary_for_pipeline_cache"].update(
        {
            "compile_model": {"piecewise_enabled": True},
            "fuse_moe": {"enabled": True},
            "max_batch_size": 8,
        }
    )
    config_second = _optimizer_config(tmp_path)
    config_second["_unit_test_after_boundary_for_pipeline_cache"].update(
        {
            "compile_model": {"piecewise_enabled": False},
            "fuse_moe": {"enabled": False},
            "max_batch_size": 64,
        }
    )

    InferenceOptimizer(factory=factory, config=config_first)(cm)
    InferenceOptimizer(factory=factory, config=config_second)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}


def test_pipeline_cache_ignores_non_prefix_transform_factory_model_kwargs(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    cm = _cache_seq_interface()
    model_kwargs_first = {
        "pipeline_cache": {"root": str(tmp_path / "cache_a")},
        "_unit_test_after_boundary_for_pipeline_cache": {
            "fused_moe": {"enabled": True},
        },
        "transforms": {
            "pipeline_cache": {"root": str(tmp_path / "cache_a")},
            "_unit_test_after_boundary_for_pipeline_cache": {
                "fused_moe": {"enabled": True},
            },
        },
    }
    model_kwargs_second = {
        "pipeline_cache": {"root": str(tmp_path / "cache_b")},
        "_unit_test_after_boundary_for_pipeline_cache": {
            "fused_moe": {"enabled": False},
        },
        "transforms": {
            "pipeline_cache": {"root": str(tmp_path / "cache_b")},
            "_unit_test_after_boundary_for_pipeline_cache": {
                "fused_moe": {"enabled": False},
            },
        },
    }

    InferenceOptimizer(
        factory=_DummyFactory(model="dummy-model", model_kwargs=model_kwargs_first),
        config=_optimizer_config(tmp_path),
    )(cm)
    InferenceOptimizer(
        factory=_DummyFactory(model="dummy-model", model_kwargs=model_kwargs_second),
        config=_optimizer_config(tmp_path),
    )(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}


def test_pipeline_cache_keeps_world_size_factory_model_kwarg_in_key(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    cm = _cache_seq_interface()

    InferenceOptimizer(
        factory=_DummyFactory(model="dummy-model", model_kwargs={"world_size": 1}),
        config=_optimizer_config(tmp_path),
    )(cm)
    InferenceOptimizer(
        factory=_DummyFactory(model="dummy-model", model_kwargs={"world_size": 2}),
        config=_optimizer_config(tmp_path),
    )(cm)

    assert _COUNTERS == {"build": 2, "boundary": 2, "after": 2}


def test_pipeline_cache_keeps_structural_factory_model_kwargs_in_key(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    cm = _cache_seq_interface()

    InferenceOptimizer(
        factory=_DummyFactory(model="dummy-model", model_kwargs={"hidden_size": 1}),
        config=_optimizer_config(tmp_path),
    )(cm)
    InferenceOptimizer(
        factory=_DummyFactory(model="dummy-model", model_kwargs={"hidden_size": 2}),
        config=_optimizer_config(tmp_path),
    )(cm)

    assert _COUNTERS == {"build": 2, "boundary": 2, "after": 2}


def test_pipeline_cache_restore_uses_pre_mutation_config(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(tmp_path)
    config["_unit_test_mutating_config_before_pipeline_cache"] = {
        "stage": "pattern_matcher",
    }

    InferenceOptimizer(factory=factory, config=config)(cm)
    InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert _MUTATING_CONFIG_RUNS == 1


def test_pipeline_cache_transform_allows_call_module(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(tmp_path, "_unit_test_build_call_module_graph_for_pipeline_cache")

    InferenceOptimizer(factory=factory, config=config)(cm)
    restored = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert any(node.op == "call_module" for node in restored.graph.nodes)
    assert "linear" in dict(restored.named_modules())


def test_pipeline_cache_transform_restores_wrapper_with_graphmodule(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(tmp_path, "_unit_test_build_wrapper_for_pipeline_cache")

    first = InferenceOptimizer(factory=factory, config=config)(cm)
    restored = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert isinstance(first.language_model.model, torch.fx.GraphModule)
    assert isinstance(restored.language_model.model, torch.fx.GraphModule)
    assert isinstance(restored.vision_tower, nn.Linear)
    assert hasattr(restored.language_model.model, "boundary_marker")
    assert hasattr(restored.language_model.model, "after_marker")


def test_pipeline_cache_strips_unpickleable_forward_hooks(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(tmp_path, "_unit_test_build_forward_hook_graph_for_pipeline_cache")

    first = InferenceOptimizer(factory=factory, config=config)(cm)
    restored = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert first._forward_pre_hooks
    assert not restored._forward_pre_hooks


def test_pipeline_cache_pickles_existing_graphmodule_code(monkeypatch):
    gm = symbolic_trace(_ToyModule())
    assert hasattr(gm, "_code")
    gm._tracer_cls = _RequiresArgTracer

    def fail_recompile(self):
        raise AssertionError("pipeline cache save should not regenerate FX code")

    monkeypatch.setattr(torch.fx.GraphModule, "recompile", fail_recompile)
    buffer = io.BytesIO()
    with _use_cached_graphmodule_code_for_pickling():
        torch.save(gm, buffer)

    buffer.seek(0)
    monkeypatch.undo()
    restored = torch.load(buffer, map_location="cpu", weights_only=False)

    assert isinstance(restored, torch.fx.GraphModule)
    assert torch.equal(restored(torch.ones(1)), torch.tensor([2.0]))


def test_pipeline_cache_saves_structural_graphmodule_without_graphmodule_reduce(monkeypatch):
    gm = symbolic_trace(_ToyModule())
    gm._tracer_cls = _RequiresArgTracer
    gm.graph._tracer_cls = _RequiresArgTracer
    gm.meta["preserved"] = True

    def fail_reduce(self):
        raise AssertionError("pipeline cache structural save must not pickle GraphModule itself")

    monkeypatch.setattr(torch.fx.GraphModule, "__reduce__", fail_reduce)
    buffer = io.BytesIO()
    _save_graphmodule_structural(gm, buffer)

    assert gm.graph.owning_module is gm

    buffer.seek(0)
    restored = _load_graphmodule_structural(buffer)

    assert isinstance(restored, torch.fx.GraphModule)
    assert restored.graph.owning_module is restored
    assert restored.meta["preserved"]
    assert torch.equal(restored(torch.ones(1)), torch.tensor([2.0]))


def test_pipeline_cache_restored_graphmodule_rebuilds_weight_node_mapping():
    root = nn.Module()
    root.register_parameter("w0", nn.Parameter(torch.ones(2, 1, device="meta")))
    root.register_parameter("w1", nn.Parameter(torch.ones(3, 1, device="meta")))
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    w0 = graph.get_attr("w0")
    w0.meta["val"] = root.w0
    linear0 = graph.call_function(torch.ops.aten.linear.default, args=(x, w0, None))
    w1 = graph.get_attr("w1")
    w1.meta["val"] = root.w1
    linear1 = graph.call_function(torch.ops.aten.linear.default, args=(x, w1, None))
    graph.output((linear0, linear1))
    gm = torch.fx.GraphModule(root, graph)

    linear_nodes = [node for node in gm.graph.nodes if is_linear_op(node)]
    assert extract_weight_name(linear_nodes[0]) == "w0"
    assert gm.meta["_weight_mapping_computed"]

    buffer = io.BytesIO()
    _save_graphmodule_structural(gm, buffer)

    buffer.seek(0)
    restored = _load_graphmodule_structural(buffer)
    restored_input = next(node for node in restored.graph.nodes if node.op == "placeholder")
    restored_linears = [node for node in restored.graph.nodes if is_linear_op(node)]

    assert "_weight_mapping_computed" not in restored.meta
    assert _insert_fused_gemm(restored, 0, restored_input, restored_linears)


def test_pipeline_cache_drops_consumed_sharding_transforms_from_graphmodule_body():
    gm = symbolic_trace(_ToyModule())
    call_node = next(node for node in gm.graph.nodes if node.op == "call_function")
    config = ShardingTransformConfig(stage=Stages.SHARDING, allreduce_strategy="NCCL")
    container = ShardingTransformContainer(config=config)
    container.parameter_update_transforms.append(
        ParameterUpdateInfo(
            target_node=call_node.name,
            config=config,
            args=(call_node,),
        )
    )
    gm._sharding_transform_container = container

    buffer = io.BytesIO()
    _save_graphmodule_structural(gm, buffer)

    buffer.seek(0)
    restored = _load_graphmodule_structural(buffer)
    restored_container = restored._sharding_transform_container

    assert restored_container.config.allreduce_strategy.name == "NCCL"
    assert restored_container.weight_sharding_transforms == []
    assert restored_container.parameter_update_transforms == []


def test_pipeline_cache_restores_graphmodule_bound_methods():
    gm = symbolic_trace(_EmbeddingToyModule())
    gm.get_input_embeddings = types.MethodType(_EmbeddingToyModule.get_input_embeddings, gm)

    buffer = io.BytesIO()
    _save_graphmodule_structural(gm, buffer)

    buffer.seek(0)
    restored = _load_graphmodule_structural(buffer)

    assert restored.get_input_embeddings() is restored.embed_tokens


def test_pipeline_cache_saves_structural_graphmodule_with_op_overload_target():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
    graph.output(size)
    gm = torch.fx.GraphModule({}, graph)

    buffer = io.BytesIO()
    _save_graphmodule_structural(gm, buffer)

    buffer.seek(0)
    restored = _load_graphmodule_structural(buffer)

    assert restored(torch.ones(3, 2)) == 3


def test_pipeline_cache_replays_source_model_hooks(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _SourceHookFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(tmp_path, "_unit_test_build_source_hook_graph_for_pipeline_cache")

    InferenceOptimizer(factory=factory, config=config)(cm)
    restored = InferenceOptimizer(factory=factory, config=config)(cm)

    expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    restored.load_state_dict(
        {"backbone.embedding.weight": expected_weight},
        strict=False,
        assign=True,
    )

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert torch.equal(restored.backbone.embeddings.weight, expected_weight)


def test_pipeline_cache_rejects_unknown_ad_managed_hook(monkeypatch, tmp_path):
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(
        tmp_path, "_unit_test_build_unknown_ad_hook_graph_for_pipeline_cache"
    )

    model = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert hasattr(model, "after_marker")
    assert list(tmp_path.rglob("manifest.json")) == []


def _make_gm_with_hooks():
    gm = symbolic_trace(_ToyModule())
    gm._register_load_state_dict_pre_hook(
        partial(
            _load_hook_for_deduplication,
            param_key_remaining="weight",
            param_key_removed="weight_alias",
        )
    )
    gm._register_load_state_dict_pre_hook(
        _build_aliasing_load_pre_hook([["weight", "weight_copy"]])
    )
    return gm


def test_hook_spec_round_trip_dedup_and_alias():
    gm = _make_gm_with_hooks()

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert {spec["type"] for spec in specs} == {"dedup", "alias"}
    assert json.loads(json.dumps(specs)) == specs

    fresh = symbolic_trace(_ToyModule())
    assert len(fresh._load_state_dict_pre_hooks) == 0
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 2


def test_hook_spec_round_trip_shard_tp():
    from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
        _load_hook,
        _split_tensor_for_tp,
    )

    gm = symbolic_trace(_ToyModule())
    f_split = partial(_split_tensor_for_tp, dim=0, rank=0, world_size=2, min_local_shape=1)
    gm._register_load_state_dict_pre_hook(
        partial(_load_hook, f_split=f_split, param_key="weight", param_shape=torch.Size([1]))
    )

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "shard_tp"
    assert specs[0]["world_size"] == 2

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1
