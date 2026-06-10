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
import pickle
import types
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.fx import symbolic_trace

from tensorrt_llm._torch.auto_deploy.export.export import (
    _build_aliasing_load_pre_hook,
    _clean_up_assertions_and_guards,
    _clean_up_export_forward_hooks,
    _load_hook_for_deduplication,
)
from tensorrt_llm._torch.auto_deploy.models.custom.mla_rope_utils import (
    _kv_b_proj_dequant_load_hook,
    _rope_deinterleave_load_hook,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4Model,
    Gemma4TextDecoderLayer,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_nemotron_h import NemotronHModel
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextModel,
)
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactory
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    BaseTransform,
    DistConfig,
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
    _load_hook,
    _split_tensor_for_tp,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import (
    _fp8_block_scale_pipeline_cache_spec,
    _shard_scale_and_hook,
    _split_fp8_block_scale,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.transform.pipeline_cache.hooks import (
    collect_hook_specs,
    reattach_hooks,
)
from tensorrt_llm._torch.auto_deploy.transform.pipeline_cache.pipeline_cache import (
    HOOKS_FILE_NAME,
    MODULE_FILE_NAME,
    PipelineCacheConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.pipeline_cache.structural import (
    load_module_structural,
    save_module_structural,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import (
    WeightNode,
    extract_weight_name,
    is_linear_op,
)
from tensorrt_llm._torch.auto_deploy.utils.pipeline_cache_hooks import mark_pipeline_cache_hook

_COUNTERS = {
    "build": 0,
    "boundary": 0,
    "after": 0,
}
_MUTATING_CONFIG_RUNS = 0


def _unit_test_forward_pre_hook(*args, **kwargs):
    del args, kwargs


def _unit_test_load_state_dict_post_hook(module, incompatible_keys):
    del incompatible_keys
    with torch.no_grad():
        module.weight.add_(2.0)


def _unit_test_load_state_dict_pre_hook(module, state_dict, prefix, *args):
    del args
    state_dict[prefix + "weight"] = torch.full_like(module.weight, 3.0)


class _ToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.weight


class _PartialBoundHookOwner:
    def __init__(self, payload_value):
        self.payload_value = payload_value

    def load_hook(self, state_dict, prefix, *args, weight_name):
        del args
        state_dict[prefix + weight_name] = self._payload_value()

    def _payload_value(self):
        return self.payload_value


class _ReadOnlyPropertyHookModule(nn.Module):
    def __init__(self, payload_value):
        super().__init__()
        self.payload_value = payload_value
        self._register_load_state_dict_pre_hook(self.load_hook)

    @property
    def can_record_outputs(self):
        return {}

    def load_hook(self, state_dict, prefix, *args):
        del args
        state_dict[prefix + "weight"] = self.payload_value


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
        gm.register_forward_pre_hook(_unit_test_forward_pre_hook)
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_unknown_ad_pipeline_hook_graph_for_pipeline_cache")
class _BuildUnknownADPipelineHookGraphForPipelineCache(BaseTransform):
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


@TransformRegistry.register("_unit_test_build_materialized_buffer_graph_for_pipeline_cache")
class _BuildMaterializedBufferGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_ToyModule().to("meta"))
        gm.register_buffer("materialized_buffer", torch.ones(1))
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_materialized_parameter_graph_for_pipeline_cache")
class _BuildMaterializedParameterGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_ToyModule().to("meta"))
        gm.weight = nn.Parameter(torch.ones(1))
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
        gm.register_buffer("boundary_marker", torch.empty(1, device="meta"))
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
        },
        "_unit_test_after_boundary_for_pipeline_cache": {
            "stage": "weight_load",
        },
    }


def _pre_sharding_optimizer_config(
    tmp_path, build_transform="_unit_test_build_graph_for_pipeline_cache"
):
    return {
        build_transform: {
            "stage": "export",
            "run_per_gm": False,
        },
        "pipeline_cache": {
            "stage": "pattern_matcher",
            "enabled": True,
            "root": str(tmp_path),
        },
        "_unit_test_boundary_for_pipeline_cache": {
            "stage": "sharding",
        },
        "_unit_test_after_boundary_for_pipeline_cache": {
            "stage": "weight_load",
        },
    }


def test_pipeline_cache_config_defaults_root_for_enabled_cache():
    """An enabled cache config defaults to the standard cache root and disables runtime transform knobs."""
    config = PipelineCacheConfig(stage=Stages.SHARDING, enabled=True)
    unsupported_transform_options = {
        "cache_key_extra",
        "debug_visualize_dir",
        "expect_mem_change",
        "requires_clean_graph",
        "requires_shape_prop",
        "run_graph_cleanup",
        "run_per_gm",
        "run_shape_prop",
        "skip_on_error",
        "strict_root_permissions",
        "trust_cache_root",
    }

    assert unsupported_transform_options.isdisjoint(PipelineCacheConfig.model_fields)
    assert config.run_per_gm is False
    assert config.requires_clean_graph is False
    assert config.requires_shape_prop is False
    assert config.run_graph_cleanup is False
    assert config.run_shape_prop is False
    assert config.skip_on_error is True
    assert config.root == str(
        Path.home() / ".cache" / "tensorrt_llm" / "auto_deploy" / "pipeline_cache"
    )


def test_pipeline_cache_config_accepts_user_root_by_default(tmp_path):
    """A user-supplied cache root is preserved on the config."""
    config = PipelineCacheConfig(
        stage=Stages.SHARDING,
        enabled=True,
        root=str(tmp_path),
    )

    assert config.root == str(tmp_path)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("debug_visualize_dir", "/tmp/debug"),
        ("cache_key_extra", {"manual": "token"}),
        ("expect_mem_change", True),
        ("requires_clean_graph", True),
        ("requires_shape_prop", True),
        ("run_graph_cleanup", True),
        ("run_per_gm", False),
        ("run_per_gm", True),
        ("run_shape_prop", True),
        ("skip_on_error", True),
        ("strict_root_permissions", True),
        ("trust_cache_root", True),
    ],
)
def test_pipeline_cache_config_rejects_unsupported_transform_options(
    tmp_path, field_name, field_value
):
    """Generic transform options that don't apply to the cache are rejected at config construction."""
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PipelineCacheConfig(
            stage=Stages.SHARDING,
            enabled=True,
            root=str(tmp_path),
            **{field_name: field_value},
        )


def test_pipeline_cache_transform_restores_and_skips_prefix(monkeypatch, tmp_path):
    """Second run restores from cache and skips re-running the pre-boundary build transform."""
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
    assert list(tmp_path.glob("*/rank_0"))
    assert not list(tmp_path.glob("*/pipeline_cache/rank_0"))


def test_pipeline_cache_ignores_post_boundary_runtime_transform_config(monkeypatch, tmp_path):
    """Post-boundary runtime transform settings don't affect the cache key, so a second run still hits."""
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


def test_pipeline_cache_ignores_dist_config_before_sharding_boundary(monkeypatch, tmp_path):
    """A cache boundary before sharding excludes dist config from the key, so differing TP still hits."""
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _pre_sharding_optimizer_config(tmp_path)

    InferenceOptimizer(
        factory=factory,
        config=config,
        dist_config=DistConfig(world_size=2, tp_size=1),
    )(cm)
    InferenceOptimizer(
        factory=factory,
        config=config,
        dist_config=DistConfig(world_size=2, tp_size=2, moe_tp_size=2),
    )(cm)

    assert _COUNTERS == {"build": 1, "boundary": 2, "after": 2}


def test_pipeline_cache_keeps_dist_config_after_sharding_boundary(monkeypatch, tmp_path):
    """A cache boundary after sharding keeps dist config in the key, so differing TP misses and rebuilds."""
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(tmp_path)

    InferenceOptimizer(
        factory=factory,
        config=config,
        dist_config=DistConfig(world_size=2, tp_size=1),
    )(cm)
    InferenceOptimizer(
        factory=factory,
        config=config,
        dist_config=DistConfig(world_size=2, tp_size=2, moe_tp_size=2),
    )(cm)

    assert _COUNTERS == {"build": 2, "boundary": 2, "after": 2}


def test_pipeline_cache_keeps_world_size_factory_model_kwarg_in_key(monkeypatch, tmp_path):
    """A world_size factory model kwarg participates in the cache key, so differing values miss."""
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
    """Structural factory model kwargs (e.g. hidden_size) participate in the key, so differing values miss."""
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
    """Cache key uses the config as it was before any pre-boundary transform mutated it, so the rerun hits."""
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
    """A graph containing a call_module node round-trips through the cache and rebuilds the submodule."""
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
    """A wrapper module nesting a GraphModule restores with its GraphModule and sibling modules intact."""
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


def test_pipeline_cache_rejects_forward_hooks(monkeypatch, tmp_path):
    """A graph carrying forward hooks is not cached (warns and skips publish), since hooks aren't serializable."""
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.auto_deploy.transform.pipeline_cache.pipeline_cache.ad_logger.warning",
        warnings.append,
    )
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(tmp_path, "_unit_test_build_forward_hook_graph_for_pipeline_cache")

    model = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert model._forward_pre_hooks
    assert list(tmp_path.rglob("manifest.json")) == []
    assert any("forward hooks" in msg for msg in warnings)


def test_pipeline_cache_cleans_tmp_dir_when_publish_fails(monkeypatch, tmp_path):
    """A failed atomic publish leaves no manifest and removes the temp staging dir, so optimization proceeds."""
    _reset_counters()
    _patch_mem_stats(monkeypatch)

    def fail_publish(tmp_rank_dir, rank_dir):
        del rank_dir
        assert tmp_rank_dir.exists()
        raise OSError("publish failed")

    monkeypatch.setattr(
        "tensorrt_llm._torch.auto_deploy.transform.pipeline_cache.pipeline_cache."
        "atomic_publish_rank_dir",
        fail_publish,
    )
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()

    model = InferenceOptimizer(factory=factory, config=_optimizer_config(tmp_path))(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert hasattr(model, "after_marker")
    assert list(tmp_path.rglob("manifest.json")) == []
    assert list(tmp_path.glob(".*.tmp.*")) == []


def test_pipeline_cache_saves_graphmodule_snapshot_without_graphmodule_reduce(monkeypatch):
    """Structural save/load reconstructs a GraphModule (preserving meta) without pickling the GraphModule itself."""
    gm = symbolic_trace(_ToyModule())
    gm._tracer_cls = _RequiresArgTracer
    gm.graph._tracer_cls = _RequiresArgTracer
    gm.meta["preserved"] = True

    def fail_reduce(self):
        raise AssertionError("pipeline cache snapshot must not pickle GraphModule itself")

    monkeypatch.setattr(torch.fx.GraphModule, "__reduce__", fail_reduce)
    buffer = io.BytesIO()
    save_module_structural(gm, buffer)

    assert gm.graph.owning_module is gm

    buffer.seek(0)
    restored = load_module_structural(buffer)

    assert isinstance(restored, torch.fx.GraphModule)
    assert restored.graph.owning_module is restored
    assert restored.meta["preserved"]
    assert torch.equal(restored(torch.ones(1)), torch.tensor([2.0]))


def test_pipeline_cache_rejects_native_graphmodule_pickle():
    """A natively-pickled GraphModule payload is rejected by the structural loader as an unsupported shape."""
    gm = symbolic_trace(_ToyModule())
    buffer = io.BytesIO()
    torch.save(gm, buffer)
    buffer.seek(0)

    with pytest.raises(ValueError, match="unsupported payload shape"):
        load_module_structural(buffer)


def test_pipeline_cache_restored_graphmodule_rebuilds_weight_node_mapping():
    """A restored GraphModule drops the cached weight-mapping meta and recomputes it lazily for GEMM fusion."""
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
    save_module_structural(gm, buffer)

    buffer.seek(0)
    restored = load_module_structural(buffer)
    restored_input = next(node for node in restored.graph.nodes if node.op == "placeholder")
    restored_linears = [node for node in restored.graph.nodes if is_linear_op(node)]

    assert "_weight_mapping_computed" not in restored.meta
    assert _insert_fused_gemm(restored, 0, restored_input, restored_linears)


def test_pipeline_cache_drops_consumed_sharding_transforms_from_graphmodule_body():
    """Saving drops already-applied sharding transforms from the container while keeping its config."""
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
    save_module_structural(gm, buffer)

    buffer.seek(0)
    restored = load_module_structural(buffer)
    restored_container = restored._sharding_transform_container

    assert restored_container.config.allreduce_strategy.name == "NCCL"
    assert restored_container.weight_sharding_transforms == []
    assert restored_container.parameter_update_transforms == []


def test_pipeline_cache_restores_graphmodule_bound_methods():
    """Custom bound methods attached to a GraphModule (e.g. get_input_embeddings) survive save/load."""
    gm = symbolic_trace(_EmbeddingToyModule())
    gm.get_input_embeddings = types.MethodType(_EmbeddingToyModule.get_input_embeddings, gm)

    buffer = io.BytesIO()
    save_module_structural(gm, buffer)

    buffer.seek(0)
    restored = load_module_structural(buffer)

    assert restored.get_input_embeddings() is restored.embed_tokens


def test_pipeline_cache_saves_exported_program_module_train_eval_methods():
    """An ExportedProgram.module()'s synthesized train/eval methods survive save/load and toggle training."""
    ep = torch.export.export(_ToyModule(), (torch.ones(1),))
    gm = ep.module()

    assert "ExportedProgram.module.<locals>._train" in gm.train.__func__.__qualname__
    assert "ExportedProgram.module.<locals>._eval" in gm.eval.__func__.__qualname__

    _clean_up_assertions_and_guards(gm)
    _clean_up_export_forward_hooks(gm)

    buffer = io.BytesIO()
    save_module_structural(gm, buffer)

    buffer.seek(0)
    restored = load_module_structural(buffer)

    assert torch.equal(restored(torch.ones(1)), torch.tensor([2.0]))
    restored.eval()
    assert not restored.training
    assert restored.train(True) is restored
    assert restored.training


def test_pipeline_cache_saves_graphmodule_snapshot_with_op_overload_target():
    """A graph node targeting an OpOverload (aten.sym_size.int) round-trips and stays executable."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    size = graph.call_function(torch.ops.aten.sym_size.int, args=(x, 0))
    graph.output(size)
    gm = torch.fx.GraphModule({}, graph)

    buffer = io.BytesIO()
    save_module_structural(gm, buffer)

    buffer.seek(0)
    restored = load_module_structural(buffer)

    assert restored(torch.ones(3, 2)) == 3


def test_pipeline_cache_rejects_unknown_ad_pipeline_hook(monkeypatch, tmp_path):
    """A graph with an unrecognized AutoDeploy pipeline hook is not cached (warns and skips publish)."""
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.auto_deploy.transform.pipeline_cache.hooks.ad_logger.warning",
        warnings.append,
    )
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(
        tmp_path, "_unit_test_build_unknown_ad_pipeline_hook_graph_for_pipeline_cache"
    )

    model = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert hasattr(model, "after_marker")
    assert list(tmp_path.rglob("manifest.json")) == []
    assert any("unrecognized hook" in msg for msg in warnings)


def test_pipeline_cache_allows_materialized_buffers(monkeypatch, tmp_path):
    """A graph with materialized (non-meta) buffers is cached and the buffer values are restored."""
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(
        tmp_path, "_unit_test_build_materialized_buffer_graph_for_pipeline_cache"
    )

    model = InferenceOptimizer(factory=factory, config=config)(cm)
    restored = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert torch.equal(model.materialized_buffer, torch.ones(1))
    assert torch.equal(restored.materialized_buffer, torch.ones(1))
    assert list(tmp_path.rglob("manifest.json"))


def test_pipeline_cache_rejects_materialized_parameters(monkeypatch, tmp_path):
    """A graph with materialized (non-meta) parameters is not cached (warns and skips publish)."""
    _reset_counters()
    _patch_mem_stats(monkeypatch)
    warnings = []
    monkeypatch.setattr(
        "tensorrt_llm._torch.auto_deploy.transform.pipeline_cache.pipeline_cache.ad_logger.warning",
        warnings.append,
    )
    factory = _DummyFactory(model="dummy-model")
    cm = _cache_seq_interface()
    config = _optimizer_config(
        tmp_path, "_unit_test_build_materialized_parameter_graph_for_pipeline_cache"
    )

    model = InferenceOptimizer(factory=factory, config=config)(cm)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert hasattr(model, "after_marker")
    assert list(tmp_path.rglob("manifest.json")) == []
    assert any("parameters found: ['weight']" in msg for msg in warnings)


def _make_gm_with_hooks():
    gm = symbolic_trace(_ToyModule())
    hook = partial(
        _load_hook_for_deduplication,
        param_key_remaining="weight",
        param_key_removed="weight_alias",
    )
    gm._register_load_state_dict_pre_hook(
        mark_pipeline_cache_hook(
            hook,
            {
                "type": "dedup",
                "param_key_remaining": "weight",
                "param_key_removed": "weight_alias",
            },
        )
    )
    gm._register_load_state_dict_pre_hook(
        _build_aliasing_load_pre_hook([["weight", "weight_copy"]])
    )
    return gm


def test_hook_spec_round_trip_dedup_and_alias():
    """Dedup and aliasing pre-hooks serialize to JSON specs and reattach to a fresh module."""
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
    """A marked TP-shard load hook serializes to a JSON spec and reattaches to a fresh module."""
    gm = symbolic_trace(_ToyModule())
    f_split = partial(_split_tensor_for_tp, dim=0, rank=0, world_size=2, min_local_shape=1)
    hook = partial(_load_hook, f_split=f_split, param_key="weight", param_shape=torch.Size([1]))
    gm._register_load_state_dict_pre_hook(
        mark_pipeline_cache_hook(
            hook,
            {
                "type": "shard_tp",
                "param_key": "weight",
                "param_shape": [1],
                "dim": 0,
                "rank": 0,
                "world_size": 2,
                "min_local_shape": 1,
                "fused_weight_dims": None,
            },
        )
    )

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "shard_tp"
    assert specs[0]["world_size"] == 2

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1


def test_hook_spec_round_trip_sharding_ir_fp8_block_scale():
    """An FP8 block-scale shard hook round-trips through a spec and re-shards the scale on reload."""
    gm = symbolic_trace(_ToyModule())
    scale = torch.arange(24 * 12, dtype=torch.float32).reshape(24, 12)
    rank = 3
    world_size = 8
    dim = 0
    f_split = partial(_split_fp8_block_scale, dim=dim, rank=rank, world_size=world_size)
    sharded = f_split(scale)
    sn = WeightNode(
        node=next(iter(gm.graph.nodes)),
        tensor=scale,
        node_key="weight_scale_inv",
        submod=gm,
    )
    _shard_scale_and_hook(
        gm,
        sn,
        sharded,
        f_split,
        _fp8_block_scale_pipeline_cache_spec(sn, sharded, dim, rank, world_size),
    )

    specs, has_unknown = collect_hook_specs(gm)

    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "shard_fp8_block_scale"
    assert json.loads(json.dumps(specs)) == specs

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1

    hook = next(iter(fresh._load_state_dict_pre_hooks.values()))
    hook_fn = hook.hook if hasattr(hook, "hook") else hook
    state_dict = {"weight_scale_inv": scale.clone()}
    hook_fn(state_dict, "", {}, True, [], [], [])
    assert torch.equal(state_dict["weight_scale_inv"], sharded)


def test_hook_spec_round_trip_post_load_hooks():
    """An importable post-load hook round-trips as a 'post' phase spec and runs on reload."""
    gm = symbolic_trace(_ToyModule())
    gm.register_load_state_dict_post_hook(_unit_test_load_state_dict_post_hook)

    specs, has_unknown = collect_hook_specs(gm)

    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "importable_load_hook"
    assert specs[0]["phase"] == "post"
    assert json.loads(json.dumps(specs)) == specs

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_post_hooks) == 1

    fresh.load_state_dict({"weight": torch.zeros(1)})
    assert torch.equal(fresh.weight, torch.full((1,), 2.0))


def test_hook_spec_round_trip_with_module_load_hooks():
    """A with_module pre-hook round-trips and reattaches preserving its with_module binding."""
    gm = symbolic_trace(_ToyModule())
    gm.register_load_state_dict_pre_hook(_unit_test_load_state_dict_pre_hook)

    specs, has_unknown = collect_hook_specs(gm)

    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "importable_load_hook"
    assert specs[0]["phase"] == "pre"
    assert specs[0]["with_module"] is True
    assert json.loads(json.dumps(specs)) == specs

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1
    hook = next(iter(fresh._load_state_dict_pre_hooks.values()))
    assert hook.with_module is True

    fresh.load_state_dict({"weight": torch.zeros(1)})
    assert torch.equal(fresh.weight, torch.full((1,), 3.0))


def test_hook_specs_reject_unsupported_marked_sharding_closures():
    """A marked hook wrapping an unpickleable local closure is reported via has_unknown, not cached."""
    gm = symbolic_trace(_ToyModule())
    start_idx = 0
    end_idx = 1

    def slice_tensor(t: torch.Tensor) -> torch.Tensor:
        return t[start_idx:end_idx]

    shard_slice_hook = partial(
        _load_hook,
        f_split=slice_tensor,
        param_key="weight",
        param_shape=torch.Size([1]),
    )
    with pytest.raises((AttributeError, pickle.PicklingError), match="local object"):
        pickle.dumps(shard_slice_hook)

    gm._register_load_state_dict_pre_hook(
        mark_pipeline_cache_hook(
            shard_slice_hook,
            {
                "type": "shard_slice",
                "param_key": "weight",
                "param_shape": [1],
                "start_idx": start_idx,
                "end_idx": end_idx,
            },
        )
    )
    specs, has_unknown = collect_hook_specs(gm)

    assert specs == []
    assert has_unknown


def test_hook_specs_cover_unpickleable_fused_tp_closures():
    """A fused-TP hook (unpickleable closure) is still captured by its marker spec and stays JSON-safe."""
    fused_gm = symbolic_trace(_ToyModule())
    rank = 0
    world_size = 2
    min_local_shape = 1
    fused_weight_dims = [1]
    dim = 0

    def f_split(
        t: torch.Tensor,
        fused_dims: list[int] = fused_weight_dims,
        d: int = dim,
    ) -> torch.Tensor:
        return torch.cat(
            [
                _split_tensor_for_tp(w, d, rank, world_size, min_local_shape)
                for w in torch.split(t, fused_dims, dim=d)
            ],
            dim=d,
        )

    fused_tp_hook = partial(
        _load_hook,
        f_split=f_split,
        param_key="weight",
        param_shape=torch.Size([1]),
    )
    with pytest.raises((AttributeError, pickle.PicklingError), match="local object"):
        pickle.dumps(fused_tp_hook)

    fused_gm._register_load_state_dict_pre_hook(
        mark_pipeline_cache_hook(
            fused_tp_hook,
            {
                "type": "shard_tp",
                "param_key": "weight",
                "param_shape": [1],
                "dim": dim,
                "rank": rank,
                "world_size": world_size,
                "min_local_shape": min_local_shape,
                "fused_weight_dims": fused_weight_dims,
            },
        )
    )
    specs, has_unknown = collect_hook_specs(fused_gm)

    assert not has_unknown
    assert specs[0]["type"] == "shard_tp"
    assert specs[0]["fused_weight_dims"] == fused_weight_dims
    assert specs[0]["rank"] == rank
    assert specs[0]["world_size"] == world_size
    assert specs[0]["min_local_shape"] == min_local_shape
    assert json.loads(json.dumps(specs)) == specs


def test_hook_spec_round_trip_mla_rope_utils():
    """MLA RoPE-deinterleave and kv_b dequant load hooks round-trip as importable-hook specs."""
    gm = symbolic_trace(_ToyModule())
    gm._register_load_state_dict_pre_hook(
        partial(
            _rope_deinterleave_load_hook,
            qk_rope_head_dim=4,
            qk_nope_head_dim=6,
            num_heads=2,
            kv_lora_rank=3,
            num_layers=1,
        )
    )
    gm._register_load_state_dict_pre_hook(partial(_kv_b_proj_dequant_load_hook, num_layers=1))

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert {spec["type"] for spec in specs} == {"importable_load_hook"}
    assert {spec["callable"]["qualname"] for spec in specs} == {
        "_rope_deinterleave_load_hook",
        "_kv_b_proj_dequant_load_hook",
    }
    assert (
        next(
            spec for spec in specs if spec["callable"]["qualname"] == "_rope_deinterleave_load_hook"
        )["keywords"]["qk_rope_head_dim"]
        == 4
    )
    assert json.loads(json.dumps(specs)) == specs

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 2


def test_hook_spec_round_trip_partial_bound_method():
    """A partial over a bound method serializes its owner class/payload and rebuilds a working hook."""
    gm = symbolic_trace(_ToyModule())
    owner = _PartialBoundHookOwner(payload_value=7)
    gm._register_load_state_dict_pre_hook(partial(owner.load_hook, weight_name="weight"))

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "importable_load_hook"
    assert specs[0]["callable"]["qualname"] == "_PartialBoundHookOwner.load_hook"
    assert specs[0]["owner_class"]["qualname"] == "_PartialBoundHookOwner"
    assert specs[0]["owner_payload"]["payload_value"] == 7
    assert specs[0]["keywords"]["weight_name"] == "weight"
    assert json.loads(json.dumps(specs)) == specs

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1

    hook = next(iter(fresh._load_state_dict_pre_hooks.values()))
    hook_fn = hook.hook if hasattr(hook, "hook") else hook
    state_dict = {}
    hook_fn(state_dict, "prefix.", {}, True, [], [], [])
    assert state_dict["prefix.weight"] == 7


def test_hook_spec_round_trip_module_bound_method_with_read_only_property():
    """A module-bound hook reattaches by rebinding to the live module, and legacy owner-payload specs still load."""
    module = _ReadOnlyPropertyHookModule(payload_value=7)

    specs, has_unknown = collect_hook_specs(module)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "importable_load_hook"
    assert specs[0]["callable"]["qualname"] == "_ReadOnlyPropertyHookModule.load_hook"
    assert specs[0]["bind_to_module"]
    assert "can_record_outputs" not in specs[0]
    assert "owner_payload" not in specs[0]
    assert json.loads(json.dumps(specs)) == specs

    fresh = _ReadOnlyPropertyHookModule(payload_value=11)
    fresh._load_state_dict_pre_hooks.clear()
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1

    hook = next(iter(fresh._load_state_dict_pre_hooks.values()))
    hook_fn = hook.hook if hasattr(hook, "hook") else hook
    state_dict = {}
    hook_fn(state_dict, "prefix.", {}, True, [], [], [])
    assert state_dict["prefix.weight"] == 11

    legacy_spec = {
        "type": "importable_load_hook",
        "scope": "root",
        "callable": {
            "module": _ReadOnlyPropertyHookModule.load_hook.__module__,
            "qualname": _ReadOnlyPropertyHookModule.load_hook.__qualname__,
        },
        "owner_class": {
            "module": _ReadOnlyPropertyHookModule.__module__,
            "qualname": _ReadOnlyPropertyHookModule.__qualname__,
        },
        "owner_payload": {
            "payload_value": 7,
            "can_record_outputs": {},
        },
    }
    fresh._load_state_dict_pre_hooks.clear()
    reattach_hooks(fresh, [legacy_spec])
    hook = next(iter(fresh._load_state_dict_pre_hooks.values()))
    hook_fn = hook.hook if hasattr(hook, "hook") else hook
    state_dict = {}
    hook_fn(state_dict, "prefix.", {}, True, [], [], [])
    assert state_dict["prefix.weight"] == 11


def test_hook_spec_round_trip_custom_model_checkpoint_hooks():
    """Checkpoint load hooks from Gemma4/NemotronH/Qwen3.5-MoE custom models round-trip and reapply."""
    gm = symbolic_trace(_ToyModule())
    qwen_norm = nn.Module()
    qwen_norm._register_load_state_dict_pre_hook(Qwen3_5MoeRMSNorm._offset_weight)
    gm.add_module("qwen_norm", qwen_norm)

    qwen_moe = nn.Module()
    qwen_moe._register_load_state_dict_pre_hook(
        Qwen3_5MoeSparseMoeBlock._load_experts_from_fused_checkpoint
    )
    gm.add_module("qwen_moe", qwen_moe)

    gemma4_model = nn.Module()
    gemma4_model._register_load_state_dict_pre_hook(Gemma4Model._remap_and_drop_weights)
    gm.add_module("gemma4_model", gemma4_model)

    gemma4_layer = nn.Module()
    gemma4_layer_owner = types.SimpleNamespace(num_experts=2, expert_intermediate_size=3)
    gemma4_layer._register_load_state_dict_pre_hook(
        types.MethodType(Gemma4TextDecoderLayer._unfuse_moe_weights, gemma4_layer_owner)
    )
    gm.add_module("gemma4_layer", gemma4_layer)

    nemotron_h = nn.Module()
    nemotron_h._register_load_state_dict_pre_hook(
        types.MethodType(NemotronHModel.load_hook, types.SimpleNamespace())
    )
    gm.add_module("nemotron_h", nemotron_h)

    qwen_text = nn.Module()
    qwen_text._register_load_state_dict_pre_hook(
        Qwen3_5MoeTextModel._remap_checkpoint_hierarchy_for_exported_text_model,
        with_module=True,
    )
    gm.add_module("qwen_text", qwen_text)

    qwen_conditional = nn.Module()
    qwen_conditional._register_load_state_dict_pre_hook(
        Qwen3_5MoeForConditionalGeneration._mirror_lm_head_weight_into_text_alias,
        with_module=True,
    )
    gm.add_module("qwen_conditional", qwen_conditional)

    gemma4_lm = nn.Module()
    gemma4_lm.config = types.SimpleNamespace(tie_word_embeddings=True)
    gemma4_lm.embed_tokens = nn.Embedding(1, 1)
    gemma4_lm.lm_head = nn.Linear(1, 1, bias=False)
    gemma4_lm.register_load_state_dict_post_hook(Gemma4ForCausalLM._retie_lm_head_weight)
    gm.add_module("gemma4_lm", gemma4_lm)

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert {spec["type"] for spec in specs} == {"importable_load_hook"}
    assert {spec["callable"]["qualname"] for spec in specs} == {
        "Qwen3_5MoeRMSNorm._offset_weight",
        "Qwen3_5MoeSparseMoeBlock._load_experts_from_fused_checkpoint",
        "Gemma4Model._remap_and_drop_weights",
        "Gemma4TextDecoderLayer._unfuse_moe_weights",
        "NemotronHModel.load_hook",
        "Qwen3_5MoeTextModel._remap_checkpoint_hierarchy_for_exported_text_model",
        "Qwen3_5MoeForConditionalGeneration._mirror_lm_head_weight_into_text_alias",
        "Gemma4ForCausalLM._retie_lm_head_weight",
    }
    assert (
        next(
            spec
            for spec in specs
            if spec["callable"]["qualname"]
            == "Qwen3_5MoeTextModel._remap_checkpoint_hierarchy_for_exported_text_model"
        )["with_module"]
        is True
    )
    assert (
        next(
            spec
            for spec in specs
            if spec["callable"]["qualname"] == "Gemma4ForCausalLM._retie_lm_head_weight"
        )["phase"]
        == "post"
    )
    assert json.loads(json.dumps(specs)) == specs
    assert (
        next(
            spec
            for spec in specs
            if spec["callable"]["qualname"] == "Gemma4TextDecoderLayer._unfuse_moe_weights"
        )["owner_payload"]["expert_intermediate_size"]
        == 3
    )
    fresh = symbolic_trace(_ToyModule())
    fresh.add_module("qwen_norm", nn.Module())
    fresh.add_module("qwen_moe", nn.Module())
    fresh.add_module("gemma4_model", nn.Module())
    fresh.add_module("gemma4_layer", nn.Module())
    fresh.add_module("nemotron_h", nn.Module())
    fresh.add_module("qwen_text", nn.Module())
    fresh.add_module("qwen_conditional", nn.Module())
    fresh.gemma4_lm = nn.Module()
    fresh.gemma4_lm.config = types.SimpleNamespace(tie_word_embeddings=True)
    fresh.gemma4_lm.embed_tokens = nn.Embedding(1, 1)
    fresh.gemma4_lm.lm_head = nn.Linear(1, 1, bias=False)

    reattach_hooks(fresh, specs)

    def run_first_pre_hook(module, state_dict, prefix=""):
        hook = next(iter(module._load_state_dict_pre_hooks.values()))
        hook_fn = hook.hook if hasattr(hook, "hook") else hook
        if bool(getattr(hook, "with_module", False)):
            hook_fn(module, state_dict, prefix, {}, True, [], [], [])
        else:
            hook_fn(state_dict, prefix, {}, True, [], [], [])

    qwen_norm_state = {"weight": torch.zeros(1)}
    run_first_pre_hook(fresh.qwen_norm, qwen_norm_state)
    assert torch.equal(qwen_norm_state["weight"], torch.ones(1))

    qwen_moe_state = {
        "experts.gate_up_proj": torch.arange(12, dtype=torch.float32).reshape(2, 6, 1),
        "experts.down_proj": torch.arange(6, dtype=torch.float32).reshape(2, 1, 3),
    }
    run_first_pre_hook(fresh.qwen_moe, qwen_moe_state)
    assert "experts.1.up_proj.weight" in qwen_moe_state
    assert "experts.down_proj" not in qwen_moe_state

    gemma4_model_state = {"audio_tower.weight": torch.ones(1)}
    run_first_pre_hook(fresh.gemma4_model, gemma4_model_state)
    assert gemma4_model_state == {}

    gemma4_layer_state = {
        "moe.gate_up_proj": torch.arange(12, dtype=torch.float32).reshape(2, 6, 1),
        "moe.down_proj": torch.ones(2, 1, 3),
        "moe.per_expert_scale": torch.tensor([2.0, 3.0]),
    }
    run_first_pre_hook(fresh.gemma4_layer, gemma4_layer_state)
    assert "moe.experts.1.up_proj.weight" in gemma4_layer_state
    assert torch.equal(
        gemma4_layer_state["moe.experts.1.down_proj.weight"], torch.full((1, 3), 3.0)
    )

    nemotron_h_state = {"embedding.weight": torch.ones(1)}
    run_first_pre_hook(fresh.nemotron_h, nemotron_h_state)
    assert "embeddings.weight" in nemotron_h_state

    qwen_text_state = {
        "model.language_model.layers.0.weight": torch.ones(1),
        "lm_head.weight": torch.ones(1),
    }
    run_first_pre_hook(fresh.qwen_text, qwen_text_state, prefix="text.")
    assert "text.layers.0.weight" in qwen_text_state
    assert "model.language_model.layers.0.weight" not in qwen_text_state
    assert "text.lm_head.weight" in qwen_text_state

    qwen_conditional_state = {"lm_head.weight": torch.ones(1)}
    run_first_pre_hook(fresh.qwen_conditional, qwen_conditional_state)
    assert "model.language_model.lm_head.weight" in qwen_conditional_state

    post_hook = next(iter(fresh.gemma4_lm._load_state_dict_post_hooks.values()))
    assert fresh.gemma4_lm.lm_head.weight is not fresh.gemma4_lm.embed_tokens.weight
    post_hook(fresh.gemma4_lm, types.SimpleNamespace())
    assert fresh.gemma4_lm.lm_head.weight is fresh.gemma4_lm.embed_tokens.weight

    assert len(fresh.qwen_norm._load_state_dict_pre_hooks) == 1
    assert len(fresh.qwen_moe._load_state_dict_pre_hooks) == 1
    assert len(fresh.gemma4_model._load_state_dict_pre_hooks) == 1
    assert len(fresh.gemma4_layer._load_state_dict_pre_hooks) == 1
    assert len(fresh.nemotron_h._load_state_dict_pre_hooks) == 1
    assert len(fresh.qwen_text._load_state_dict_pre_hooks) == 1
    assert len(fresh.qwen_conditional._load_state_dict_pre_hooks) == 1
    assert len(fresh.gemma4_lm._load_state_dict_post_hooks) == 1
