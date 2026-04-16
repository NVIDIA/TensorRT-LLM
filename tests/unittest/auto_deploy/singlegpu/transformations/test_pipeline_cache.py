import gzip
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import ConfigDict, Field
from torch.fx import symbolic_trace

from tensorrt_llm._torch.auto_deploy.export.export import (
    _build_aliasing_load_pre_hook,
    _load_hook_for_deduplication,
)
from tensorrt_llm._torch.auto_deploy.llm_args import PipelineCacheConfig
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactory
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.ad_ir import (
    build_graph_module,
    extract_ir,
    load_ir,
    resolve_arg,
    resolve_target,
    save_ir,
    serialize_arg,
    serialize_target,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    BaseTransform,
    MemStats,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.transform.pipeline_cache import (
    collect_hook_specs,
    reattach_hooks,
)
from tensorrt_llm.mapping import Mapping, MpiTopology
from tensorrt_llm.version import __version__ as TRTLLM_VERSION

_COUNTERS = {
    "build": 0,
    "boundary": 0,
    "after": 0,
}


class _ToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.weight


class _DummyFactory(ModelFactory):
    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len or 512

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
        self._register_load_state_dict_pre_hook(_fake_nemotron_embedding_rename_hook)


class _SourceHookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.backbone = _SourceHookBackbone()

    def forward(self, x):
        return x + self.weight


class _SourceHookFactory(ModelFactory):
    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len or 512

    def _build_model(self, device: str) -> nn.Module:
        return _SourceHookModel().to(device)

    def _load_checkpoint(self, model: nn.Module, device, disable_preload: bool = False):
        return None

    def get_export_infos(self, model: nn.Module):
        return []


class _CallModuleToyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def _fake_nemotron_embedding_rename_hook(state_dict, prefix, *args, **kwargs):
    for key in list(state_dict.keys()):
        if "embedding." in key:
            state_dict[key.replace("embedding.", "embeddings.")] = state_dict.pop(key)
            break


def _fake_module_aware_rename_hook(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    for key in list(state_dict.keys()):
        if "embedding." in key:
            state_dict[key.replace("embedding.", "embeddings.")] = state_dict.pop(key)
            break


@TransformRegistry.register("_unit_test_build_graph_for_pipeline_cache")
class _BuildGraphForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_ToyModule())
        gm.meta["build_counter"] = _COUNTERS["build"]
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


@TransformRegistry.register("_unit_test_build_graph_with_call_module_for_pipeline_cache")
class _BuildGraphWithCallModuleForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = symbolic_trace(_CallModuleToyModule())
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


@TransformRegistry.register("_unit_test_build_graph_with_supported_source_hook_for_pipeline_cache")
class _BuildGraphWithSupportedSourceHookForPipelineCache(BaseTransform):
    def _apply_to_full_model(self, model, cm, factory, shared_config: SharedConfig):
        _COUNTERS["build"] += 1
        gm = _make_gm_with_supported_source_model_hook()
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


class _BoundaryWithMappingConfig(TransformConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mapping: Mapping = Field(default_factory=Mapping)


@TransformRegistry.register("_unit_test_boundary_with_mapping_for_pipeline_cache")
class _BoundaryWithMappingForPipelineCache(BaseTransform):
    @classmethod
    def get_config_class(cls):
        return _BoundaryWithMappingConfig

    def _apply(self, gm, cm, factory, shared_config: SharedConfig):
        return gm, TransformInfo(
            skipped=False,
            num_matches=1,
            is_clean=True,
            has_valid_shapes=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimizer_config():
    return {
        "_unit_test_build_graph_for_pipeline_cache": {
            "stage": "export",
            "run_per_gm": False,
        },
        "_unit_test_boundary_for_pipeline_cache": {
            "stage": "sharding",
        },
        "_unit_test_after_boundary_for_pipeline_cache": {
            "stage": "weight_load",
        },
    }


def _make_optimizer_config_with_call_module():
    return {
        "_unit_test_build_graph_with_call_module_for_pipeline_cache": {
            "stage": "export",
            "run_per_gm": False,
        },
        "_unit_test_boundary_for_pipeline_cache": {
            "stage": "sharding",
        },
        "_unit_test_after_boundary_for_pipeline_cache": {
            "stage": "weight_load",
        },
    }


def _make_optimizer_config_with_supported_source_hook():
    return {
        "_unit_test_build_graph_with_supported_source_hook_for_pipeline_cache": {
            "stage": "export",
            "run_per_gm": False,
        },
        "_unit_test_boundary_for_pipeline_cache": {
            "stage": "sharding",
        },
        "_unit_test_after_boundary_for_pipeline_cache": {
            "stage": "weight_load",
        },
    }


def _make_gm_with_hooks():
    """Build a traced GraphModule and attach representative load hooks."""
    gm = symbolic_trace(_ToyModule())

    # Dedup hook
    gm._register_load_state_dict_pre_hook(
        partial(
            _load_hook_for_deduplication,
            param_key_remaining="weight",
            param_key_removed="weight_alias",
        )
    )

    # Alias hook
    gm._register_load_state_dict_pre_hook(
        _build_aliasing_load_pre_hook([["weight", "weight_copy"]])
    )

    return gm


def _make_gm_with_supported_source_model_hook():
    gm = symbolic_trace(_ToyModule())
    backbone = nn.Module()
    backbone.add_module("embeddings", nn.Embedding(2, 3))
    backbone._register_load_state_dict_pre_hook(_fake_nemotron_embedding_rename_hook)
    gm.add_module("backbone", backbone)
    return gm


def _make_gm_with_module_aware_source_model_hook():
    gm = symbolic_trace(_ToyModule())
    backbone = nn.Module()
    backbone.add_module("embeddings", nn.Embedding(2, 3))
    backbone.register_load_state_dict_pre_hook(_fake_module_aware_rename_hook)
    gm.add_module("backbone", backbone)
    return gm


# ---------------------------------------------------------------------------
# Tests: pre-weight-loading save / restore round trip
# ---------------------------------------------------------------------------


def test_pipeline_cache_restores_graph_only_boundary(monkeypatch, tmp_path):
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer_config = _make_optimizer_config()

    optimizer_first = InferenceOptimizer(
        factory=factory,
        config=optimizer_config,
        pipeline_cache_config=cache_config,
    )
    model_first = optimizer_first(cache_seq_interface)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert hasattr(model_first, "boundary_marker")
    assert hasattr(model_first, "after_marker")

    optimizer_second = InferenceOptimizer(
        factory=factory,
        config=optimizer_config,
        pipeline_cache_config=cache_config,
    )
    model_second = optimizer_second(cache_seq_interface)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert hasattr(model_second, "boundary_marker")
    assert hasattr(model_second, "after_marker")

    history = model_second.meta["_autodeploy"]["transform_history"]
    assert "_unit_test_after_boundary_for_pipeline_cache" in history
    assert all(isinstance(value, TransformInfo) for value in history.values())

    manifest_paths = list(tmp_path.rglob("manifest.json"))
    assert len(manifest_paths) == 1


def test_pipeline_cache_hashes_mapping_objects_in_transform_config(tmp_path):
    factory = _DummyFactory(model="dummy-model")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_with_mapping_for_pipeline_cache",
    )
    optimizer_config = {
        "_unit_test_boundary_with_mapping_for_pipeline_cache": {
            "stage": "sharding",
            "mapping": MpiTopology(world_size=1, rank=0),
        },
    }

    optimizer = InferenceOptimizer(
        factory=factory,
        config=optimizer_config,
        pipeline_cache_config=cache_config,
    )

    assert optimizer.pipeline_cache.enabled


def test_pipeline_cache_save_failure_does_not_abort_pipeline(monkeypatch, tmp_path):
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    def _raise_pickle_error(*args, **kwargs):
        raise AttributeError(
            "Can't pickle local object '_create_stateful_graph_module.<locals>.<lambda>'"
        )

    import tensorrt_llm._torch.auto_deploy.transform.pipeline_cache as _pc_mod

    monkeypatch.setattr(_pc_mod, "save_ir", _raise_pickle_error)

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer_config = _make_optimizer_config()

    optimizer = InferenceOptimizer(
        factory=factory,
        config=optimizer_config,
        pipeline_cache_config=cache_config,
    )
    model = optimizer(cache_seq_interface)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert hasattr(model, "boundary_marker")
    assert hasattr(model, "after_marker")
    assert list(tmp_path.rglob("manifest.json")) == []
    assert list(tmp_path.iterdir()) == []


def test_pipeline_cache_skips_graphs_with_call_module(monkeypatch, tmp_path):
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )

    optimizer = InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config_with_call_module(),
        pipeline_cache_config=cache_config,
    )
    optimizer(cache_seq_interface)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}
    assert list(tmp_path.rglob("manifest.json")) == []


def test_pipeline_cache_allows_supported_source_model_hooks(monkeypatch, tmp_path):
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _SourceHookFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )

    optimizer_first = InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config_with_supported_source_hook(),
        pipeline_cache_config=cache_config,
    )
    optimizer_first(cache_seq_interface)

    optimizer_second = InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config_with_supported_source_hook(),
        pipeline_cache_config=cache_config,
    )
    restored_model, start_idx = optimizer_second.pipeline_cache.maybe_restore(cache_seq_interface)
    assert restored_model is not None
    assert start_idx == 2

    expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    restored_model.load_state_dict(
        {"backbone.embedding.weight": expected_weight},
        strict=False,
        assign=True,
    )
    assert torch.equal(restored_model.backbone.embeddings.weight, expected_weight)

    model_second = optimizer_second(cache_seq_interface)

    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 2}
    assert list(tmp_path.rglob("manifest.json"))
    assert hasattr(model_second, "after_marker")


# ---------------------------------------------------------------------------
# Tests: HookSpec round-trip
# ---------------------------------------------------------------------------


def test_hook_spec_round_trip_dedup_and_alias():
    """Collect specs from a module with dedup + alias hooks, then rebuild them."""
    gm = _make_gm_with_hooks()

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 2

    types = {s["type"] for s in specs}
    assert "dedup" in types
    assert "alias" in types

    # Build a fresh module and reattach
    fresh = symbolic_trace(_ToyModule())
    assert len(fresh._load_state_dict_pre_hooks) == 0

    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 2

    state = {"weight": torch.ones(1) * 42.0}
    fresh.load_state_dict(state, strict=False)
    assert torch.equal(fresh.weight, torch.ones(1) * 42.0)


def test_hook_spec_round_trip_remove():
    """Collect and rebuild a ``remove`` hook."""
    from tensorrt_llm._torch.auto_deploy.transform.library.sharding import _load_hook_remove

    gm = symbolic_trace(_ToyModule())
    gm._register_load_state_dict_pre_hook(partial(_load_hook_remove, param_key="bias"))

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "remove"
    assert specs[0]["param_key"] == "bias"

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1


def test_hook_spec_round_trip_shard_tp():
    """Collect and rebuild a ``shard_tp`` hook."""
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
    assert specs[0]["dim"] == 0
    assert specs[0]["world_size"] == 2

    fresh = symbolic_trace(_ToyModule())
    reattach_hooks(fresh, specs)
    assert len(fresh._load_state_dict_pre_hooks) == 1


def test_collect_hook_specs_for_source_model_hook():
    gm = _make_gm_with_supported_source_model_hook()

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "source_model"
    assert specs[0]["scope"] == "backbone"
    assert "hook_identity" in specs[0]
    assert json.loads(json.dumps(specs)) == specs


def test_collect_hook_specs_for_module_aware_source_model_hook():
    gm = _make_gm_with_module_aware_source_model_hook()

    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "source_model"
    assert specs[0]["with_module"] is True


def test_hook_spec_serializes_to_json():
    """All HookSpec dicts must be JSON-serializable."""
    gm = _make_gm_with_hooks()
    specs, _ = collect_hook_specs(gm)
    serialized = json.dumps(specs)
    deserialized = json.loads(serialized)
    assert deserialized == specs


def test_target_serialization_round_trip():
    """torch.ops targets and builtins survive serialize/resolve."""
    import operator

    targets = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mm.default,
        operator.getitem,
    ]
    for target in targets:
        key = serialize_target(target)
        restored = resolve_target(key)
        assert restored is target, f"Round-trip failed for {target}: {key} -> {restored}"


def test_arg_serialization_round_trip():
    """Arg serialization covers common types."""
    from torch.fx import Graph

    graph = Graph()
    n1 = graph.placeholder("x")
    n2 = graph.placeholder("y")
    node_map = {"x": n1, "y": n2}

    test_cases = [
        (42, 42),
        (3.14, 3.14),
        ("hello", "hello"),
        (None, None),
        (True, True),
        (torch.float16, torch.float16),
        (torch.device("cuda:0"), torch.device("cuda:0")),
    ]

    for original, expected in test_cases:
        serialized = serialize_arg(original)
        resolved = resolve_arg(serialized, node_map)
        assert resolved == expected, f"Round-trip failed for {original}"

    serialized_ref = serialize_arg(n1)
    assert resolve_arg(serialized_ref, node_map) is n1

    serialized_tuple = serialize_arg((n1, 42, n2))
    resolved_tuple = resolve_arg(serialized_tuple, node_map)
    assert isinstance(resolved_tuple, tuple)
    assert resolved_tuple[0] is n1
    assert resolved_tuple[1] == 42
    assert resolved_tuple[2] is n2


def test_ad_ir_save_produces_expected_files(tmp_path):
    """AD IR save creates ``ad_ir.json.gz`` and ``real_buffers.pt``."""
    gm = symbolic_trace(_ToyModule())
    gm.register_buffer("scale", torch.tensor(1.5))

    ir, real_buffers = extract_ir(gm)
    save_ir(ir, real_buffers, tmp_path)

    assert (tmp_path / "ad_ir.json.gz").exists()
    assert not (tmp_path / "ad_ir.json").exists()
    assert (tmp_path / "real_buffers.pt").exists()

    with gzip.open(tmp_path / "ad_ir.json.gz", "rt", encoding="utf-8") as f:
        ir_data = json.load(f)
    assert any(n["op"] == "placeholder" for n in ir_data["nodes"])
    assert "weight" in ir_data["params"]
    assert ir_data["params"]["weight"]["dtype"] == "float32"
    assert "scale" in ir_data["buffers"]
    assert ir_data["buffers"]["scale"]["is_meta"] is False


def test_ad_ir_round_trip(tmp_path):
    """Save and restore a GraphModule via AD IR."""
    gm = symbolic_trace(_ToyModule())
    gm.register_buffer("scale", torch.tensor(2.5))

    ir, real_buffers = extract_ir(gm)
    save_ir(ir, real_buffers, tmp_path)

    result = load_ir(tmp_path)
    assert result is not None
    loaded_ir, loaded_bufs = result
    loaded = build_graph_module(loaded_ir, loaded_bufs)

    assert hasattr(loaded, "weight")
    assert hasattr(loaded, "scale")
    assert loaded.weight.shape == torch.Size([1])
    assert loaded.weight.device.type == "meta"
    assert torch.equal(loaded.scale, torch.tensor(2.5))


def test_ad_ir_round_trip_with_hooks(tmp_path):
    """AD IR round-trip with hook specs."""
    gm = _make_gm_with_hooks()
    specs, has_unknown = collect_hook_specs(gm)
    assert not has_unknown

    ir, real_buffers = extract_ir(gm, hook_specs=specs)
    save_ir(ir, real_buffers, tmp_path)

    result = load_ir(tmp_path)
    assert result is not None
    loaded_ir, loaded_bufs = result
    loaded = build_graph_module(loaded_ir, loaded_bufs)

    assert len(loaded_ir.hook_specs) == 2
    reattach_hooks(loaded, loaded_ir.hook_specs)
    assert len(loaded._load_state_dict_pre_hooks) == 2

    state = {"weight": torch.ones(1) * 7.0}
    loaded.load_state_dict(state, strict=False, assign=True)
    assert torch.equal(loaded.weight, torch.ones(1) * 7.0)


# ---------------------------------------------------------------------------
# Tests: manifest metadata
# ---------------------------------------------------------------------------


def test_manifest_metadata(monkeypatch, tmp_path):
    """Saved manifests contain the expected metadata fields."""
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer_config = _make_optimizer_config()

    optimizer = InferenceOptimizer(
        factory=factory,
        config=optimizer_config,
        pipeline_cache_config=cache_config,
    )
    optimizer(cache_seq_interface)

    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["boundary_name"] == "_unit_test_boundary_for_pipeline_cache"
    assert manifest["transform_index"] == 1
    assert manifest["boundary_stage"] == "sharding"
    assert manifest["cache_key"]
    assert manifest["transform_prefix_hash"]
    assert isinstance(manifest["producer_hash"], str)
    assert manifest["producer_hash"]
    assert manifest["model_identifier"] == "dummy-model"
    assert manifest["checkpoint_fingerprint"] == "dummy-model"
    assert manifest["trtllm_version"] == TRTLLM_VERSION
    assert manifest["mapping"] == {"world_size": 1}
    assert manifest["weights_materialized"] is False
    assert "hook_spec_count" in manifest
    assert "has_unserializable_hooks" in manifest
    assert "source_model_hooks_required" in manifest
    assert manifest["has_unserializable_hooks"] is False
    assert manifest["source_model_hooks_required"] is False
    assert manifest["rank"] == 0
    assert manifest["world_size"] == 1


def test_manifest_ad_ir_files_written(monkeypatch, tmp_path):
    """Saving writes compressed AD IR files and no legacy lean files."""
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer = InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )
    optimizer(cache_seq_interface)

    assert list(tmp_path.rglob("ad_ir.json.gz"))
    assert not list(tmp_path.rglob("ad_ir.json"))
    assert not list(tmp_path.rglob("graph_code.json"))
    assert not list(tmp_path.rglob("module.pt"))


def test_manifest_ignores_extra_fields(monkeypatch, tmp_path):
    """Unknown manifest fields should not invalidate an otherwise valid cache."""
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer_config = _make_optimizer_config()

    opt1 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt1(cache_seq_interface)
    assert _COUNTERS["build"] == 1

    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    data = json.loads(manifests[0].read_text())
    data["legacy_format_version"] = 3
    manifests[0].write_text(json.dumps(data))

    opt2 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt2(cache_seq_interface)
    assert _COUNTERS["build"] == 1


def test_cache_miss_when_producer_hash_changes(monkeypatch, tmp_path):
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        device="cpu",
    )
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer_config = _make_optimizer_config()

    opt1 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt1(cache_seq_interface)
    assert _COUNTERS["build"] == 1

    import tensorrt_llm._torch.auto_deploy.transform.pipeline_cache as _pc_mod

    original = _pc_mod._build_producer_hash
    monkeypatch.setattr(
        _pc_mod,
        "_build_producer_hash",
        lambda prefix_transform_names: original(prefix_transform_names) + "-changed",
    )

    opt2 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt2(cache_seq_interface)
    assert _COUNTERS["build"] == 2


def test_producer_hash_only_uses_prefix_transforms(monkeypatch, tmp_path):
    factory = _DummyFactory(model="dummy-model")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )

    import tensorrt_llm._torch.auto_deploy.transform.pipeline_cache as _pc_mod

    seen_qualnames = []
    original = _pc_mod._describe_source_object

    def _record(obj):
        seen_qualnames.append(getattr(obj, "__qualname__", type(obj).__qualname__))
        return original(obj)

    monkeypatch.setattr(_pc_mod, "_describe_source_object", _record)

    InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )

    assert any("BuildGraphForPipelineCache" in name for name in seen_qualnames)
    assert any("BoundaryForPipelineCache" in name for name in seen_qualnames)
    assert not any("AfterBoundaryForPipelineCache" in name for name in seen_qualnames)


# ---------------------------------------------------------------------------
# Tests: pre-weight-loading dimension tolerance
# ---------------------------------------------------------------------------


def test_pipeline_cache_tolerates_different_max_batch_size(monkeypatch, tmp_path):
    """Changing max_batch_size between runs should hit the cache (not rebuild)."""
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer_config = _make_optimizer_config()

    cm_first = CachedSequenceInterface(max_seq_len=8, max_batch_size=2, device="cpu")
    opt1 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt1(cm_first)
    assert _COUNTERS == {"build": 1, "boundary": 1, "after": 1}

    cm_second = CachedSequenceInterface(max_seq_len=8, max_batch_size=64, device="cpu")
    opt2 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt2(cm_second)

    assert _COUNTERS["build"] == 1, "Should reuse cache, not rebuild"
    assert _COUNTERS["boundary"] == 1, "Should reuse cache, not re-run sharding"
    assert _COUNTERS["after"] == 2, "Post-boundary transform should run again"
    assert cm_second.info.max_batch_size == 64, "Current run's max_batch_size should be kept"


def test_pipeline_cache_tolerates_different_kv_cache_config(monkeypatch, tmp_path):
    """Changing KV cache config between runs should hit the cache and use current config."""
    from tensorrt_llm._torch.auto_deploy.shim.interface import KvCacheConfig

    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )
    optimizer_config = _make_optimizer_config()

    kv1 = KvCacheConfig(max_tokens=1000)
    cm_first = CachedSequenceInterface(
        max_seq_len=8, max_batch_size=2, device="cpu", kv_cache_config=kv1
    )
    opt1 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt1(cm_first)
    assert _COUNTERS["build"] == 1

    kv2 = KvCacheConfig(max_tokens=5000)
    cm_second = CachedSequenceInterface(
        max_seq_len=8, max_batch_size=2, device="cpu", kv_cache_config=kv2
    )
    opt2 = InferenceOptimizer(
        factory=factory, config=optimizer_config, pipeline_cache_config=cache_config
    )
    opt2(cm_second)

    assert _COUNTERS["build"] == 1, "Should reuse cache"
    assert cm_second.kv_cache_config.max_tokens == 5000, "Current run's KV config should be kept"


# ---------------------------------------------------------------------------
# Tests: hardening — schema versions, checksums, fail-closed, hook
# classification, pickle-free TreeSpec, atomic publish, content-based
# fingerprint, call_module rejection on restore.
# ---------------------------------------------------------------------------


def test_manifest_includes_schema_versions(monkeypatch, tmp_path):
    """Schema versions appear in the manifest and round-trip through save/restore."""
    import tensorrt_llm._torch.auto_deploy.transform.pipeline_cache as _pc_mod
    from tensorrt_llm._torch.auto_deploy.transform.ad_ir import (
        GRAPH_SCHEMA_VERSION,
        HOOK_SCHEMA_VERSION,
    )

    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cache_seq_interface = CachedSequenceInterface(max_seq_len=8, max_batch_size=2, device="cpu")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )

    optimizer = InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )
    optimizer(cache_seq_interface)

    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["cache_contract_version"] == _pc_mod.CACHE_CONTRACT_VERSION
    assert manifest["graph_schema_version"] == GRAPH_SCHEMA_VERSION
    assert manifest["hook_schema_version"] == HOOK_SCHEMA_VERSION
    assert manifest["sidecar_checksums"]
    assert "ad_ir.json.gz" in manifest["sidecar_checksums"]


def test_cache_miss_on_schema_version_bump(monkeypatch, tmp_path):
    """A graph_schema_version change causes the restore to miss."""
    import tensorrt_llm._torch.auto_deploy.transform.ad_ir as _ir_mod

    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cm = CachedSequenceInterface(max_seq_len=8, max_batch_size=2, device="cpu")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )

    InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )(cm)
    assert _COUNTERS["build"] == 1

    monkeypatch.setattr(_ir_mod, "GRAPH_SCHEMA_VERSION", _ir_mod.GRAPH_SCHEMA_VERSION + 99)
    # The pipeline_cache module imported the old version, so patch it there too.
    import tensorrt_llm._torch.auto_deploy.transform.pipeline_cache as _pc_mod

    monkeypatch.setattr(_pc_mod, "GRAPH_SCHEMA_VERSION", _ir_mod.GRAPH_SCHEMA_VERSION)

    InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )(cm)
    assert _COUNTERS["build"] == 2, "Schema version bump must invalidate the cache"


def test_cache_miss_on_sidecar_checksum_mismatch(monkeypatch, tmp_path):
    """Tampering with ad_ir.json.gz after save invalidates the restore."""
    for key in _COUNTERS:
        _COUNTERS[key] = 0

    monkeypatch.setattr(
        BaseTransform,
        "_get_mem_stats",
        lambda self, empty_cache=True: MemStats(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    factory = _DummyFactory(model="dummy-model")
    cm = CachedSequenceInterface(max_seq_len=8, max_batch_size=2, device="cpu")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="_unit_test_boundary_for_pipeline_cache",
    )

    InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )(cm)
    assert _COUNTERS["build"] == 1

    ir_files = list(tmp_path.rglob("ad_ir.json.gz"))
    assert len(ir_files) == 1
    # Append a spurious byte to corrupt the gzip stream's checksum match.
    with open(ir_files[0], "ab") as f:
        f.write(b"\x00")

    InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )(cm)
    assert _COUNTERS["build"] == 2, "Checksum mismatch must force a rebuild"


def test_build_graph_module_rejects_call_module(tmp_path):
    """build_graph_module must refuse to reconstruct call_module graphs."""
    import pytest

    gm = symbolic_trace(_CallModuleToyModule())
    ir, bufs = extract_ir(gm)

    with pytest.raises(ValueError, match="call_module"):
        build_graph_module(ir, bufs)


def test_ad_ir_uses_no_pickle_for_treespec(tmp_path):
    """TreeSpec round-trip uses JSON-based treespec_dumps, not pickle."""
    gm = symbolic_trace(_ToyModule())
    ir, _bufs = extract_ir(gm)
    # in_spec should either be None (no pytree info) or a JSON string.
    if ir.in_spec is not None:
        # treespec_dumps emits a JSON-parseable payload.
        json.loads(ir.in_spec)


def test_torch_load_fallback_removed(monkeypatch, tmp_path):
    """load_ir must strictly use weights_only=True (no unrestricted fallback)."""
    import tensorrt_llm._torch.auto_deploy.transform.ad_ir as _ir_mod

    # Write a buffer we can reload; then monkeypatch torch.load to fail if
    # weights_only is absent or False.
    gm = symbolic_trace(_ToyModule())
    gm.register_buffer("scale", torch.tensor(1.0))
    ir, real_buffers = extract_ir(gm)
    _ir_mod.save_ir(ir, real_buffers, tmp_path)

    original_load = torch.load
    seen_flags = []

    def _checked_load(path, *args, **kwargs):
        seen_flags.append(kwargs.get("weights_only", False))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", _checked_load)
    result = _ir_mod.load_ir(tmp_path)
    assert result is not None
    assert seen_flags == [True], f"expected weights_only=True, saw {seen_flags}"


def test_unknown_ad_pipeline_hook_blocks_save(monkeypatch, tmp_path):
    """Block save when an unknown callable comes from an AD pipeline sub-package.

    A callable attached from inside an AD pipeline sub-package (transform/
    or export/) but not matching any known pattern must block the save
    (has_unknown=True).
    """
    from tensorrt_llm._torch.auto_deploy.transform import pipeline_cache as _pc_mod

    # A function whose __module__ lives under an AD pipeline prefix but isn't a
    # known hook pattern.
    def _bogus_ad_hook(state_dict, prefix, *args, **kwargs):
        del state_dict, prefix, args, kwargs

    _bogus_ad_hook.__module__ = "tensorrt_llm._torch.auto_deploy.transform._bogus"

    gm = symbolic_trace(_ToyModule())
    gm._register_load_state_dict_pre_hook(_bogus_ad_hook)

    specs, has_unknown = _pc_mod.collect_hook_specs(gm)
    assert has_unknown is True
    # Unknown hooks from the AD pipeline must NOT be classified as source_model.
    assert all(s.get("type") != "source_model" for s in specs)


def test_custom_model_hook_stays_source_model():
    """Treat ``auto_deploy/models/`` hooks as source-model hooks.

    Hooks defined under ``auto_deploy/models/`` (e.g. custom-model load
    hooks like the Nemotron-H one) are source-model hooks even though they
    live inside the ``auto_deploy`` package.
    """
    from tensorrt_llm._torch.auto_deploy.transform import pipeline_cache as _pc_mod

    def _custom_model_load_hook(state_dict, prefix, *args, **kwargs):
        del state_dict, prefix, args, kwargs

    _custom_model_load_hook.__module__ = (
        "tensorrt_llm._torch.auto_deploy.models.custom.modeling_nemotron_h"
    )

    gm = symbolic_trace(_ToyModule())
    gm._register_load_state_dict_pre_hook(_custom_model_load_hook)

    specs, has_unknown = _pc_mod.collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "source_model"


def test_external_callable_stays_source_model():
    """Callables defined outside the AD package keep the source_model path."""
    from tensorrt_llm._torch.auto_deploy.transform import pipeline_cache as _pc_mod

    gm = symbolic_trace(_ToyModule())
    gm._register_load_state_dict_pre_hook(_fake_nemotron_embedding_rename_hook)

    specs, has_unknown = _pc_mod.collect_hook_specs(gm)
    assert not has_unknown
    assert len(specs) == 1
    assert specs[0]["type"] == "source_model"


def test_atomic_publish_survives_concurrent_reader(monkeypatch, tmp_path):
    """Publishing a new snapshot never leaves the rank dir missing."""
    from tensorrt_llm._torch.auto_deploy.transform import pipeline_cache as _pc_mod

    target = tmp_path / "rank_0"
    target.mkdir()
    (target / "manifest.json").write_text('{"x": 1}')

    src = tmp_path / "src"
    src.mkdir()
    (src / "manifest.json").write_text('{"x": 2}')

    # Simulate a rename that fails, and verify rollback restores the original.
    original_rename = Path.rename
    call_count = [0]

    def _failing_rename(self, target):
        call_count[0] += 1
        if call_count[0] == 2:
            raise OSError("simulated publish failure")
        return original_rename(self, target)

    monkeypatch.setattr(Path, "rename", _failing_rename)

    import pytest

    with pytest.raises(OSError):
        _pc_mod._atomic_publish_rank_dir(src, target)

    assert target.exists()
    assert (target / "manifest.json").read_text() == '{"x": 1}'


def test_hf_checkpoint_fingerprint_changes_with_config(tmp_path):
    """Editing any metadata file in a checkpoint dir changes the fingerprint."""
    from tensorrt_llm._torch.auto_deploy.models.hf import _hash_checkpoint_metadata

    (tmp_path / "config.json").write_text('{"model_type": "toy"}')
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 128)

    fingerprint_v1 = _hash_checkpoint_metadata(str(tmp_path))

    (tmp_path / "config.json").write_text('{"model_type": "toy", "hidden_size": 16}')
    fingerprint_v2 = _hash_checkpoint_metadata(str(tmp_path))
    assert fingerprint_v1 != fingerprint_v2, "config.json edit must change fingerprint"

    # Shard size change should also invalidate.
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 256)
    fingerprint_v3 = _hash_checkpoint_metadata(str(tmp_path))
    assert fingerprint_v3 != fingerprint_v2

    # Any new non-weight file must also shift the fingerprint — no allowlist.
    (tmp_path / "some_future_metadata.json").write_text('{"x": 1}')
    fingerprint_v4 = _hash_checkpoint_metadata(str(tmp_path))
    assert fingerprint_v4 != fingerprint_v3

    # Editing shard contents without changing size must NOT change the
    # fingerprint — weights are layout-only by design.
    (tmp_path / "model.safetensors").write_bytes(b"\xff" * 256)
    fingerprint_v5 = _hash_checkpoint_metadata(str(tmp_path))
    assert fingerprint_v5 == fingerprint_v4


def test_hf_checkpoint_fingerprint_uses_snapshot_sha_fast_path(tmp_path):
    """A path shaped like an HF snapshot directory returns the commit SHA."""
    from tensorrt_llm._torch.auto_deploy.models.hf import _hash_checkpoint_metadata

    commit_sha = "a" * 40
    snapshot_dir = tmp_path / "models--org--repo" / "snapshots" / commit_sha
    snapshot_dir.mkdir(parents=True)
    # Populate the snapshot with files that would otherwise contribute to a
    # content hash — the fast path must ignore them.
    (snapshot_dir / "config.json").write_text('{"x": 1}')
    (snapshot_dir / "model.safetensors").write_bytes(b"\x00" * 32)

    fingerprint = _hash_checkpoint_metadata(str(snapshot_dir))
    assert fingerprint == f"hf_snapshot:{commit_sha}"

    # A path under a different commit SHA produces a different fingerprint.
    other = tmp_path / "models--org--repo" / "snapshots" / ("b" * 40)
    other.mkdir(parents=True)
    assert _hash_checkpoint_metadata(str(other)) == f"hf_snapshot:{'b' * 40}"

    # A "snapshots" directory with a non-SHA child falls back to content hash.
    odd_dir = tmp_path / "snapshots" / "not-a-sha"
    odd_dir.mkdir(parents=True)
    (odd_dir / "config.json").write_text("{}")
    fingerprint = _hash_checkpoint_metadata(str(odd_dir))
    assert not fingerprint.startswith("hf_snapshot:")


def test_cache_disabled_when_boundary_name_invalid(tmp_path):
    """A misconfigured boundary name disables the cache rather than crashing."""
    factory = _DummyFactory(model="dummy-model")
    cache_config = PipelineCacheConfig(
        enabled=True,
        root=tmp_path,
        boundary="this_transform_does_not_exist",
    )
    optimizer = InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )
    assert not optimizer.pipeline_cache.enabled
