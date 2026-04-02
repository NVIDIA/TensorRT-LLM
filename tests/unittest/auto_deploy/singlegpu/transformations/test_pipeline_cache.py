import json
from functools import partial

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
    _BOUNDARY_CLASS_PRE_WEIGHT_LOAD,
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
        boundaries=["_unit_test_boundary_with_mapping_for_pipeline_cache"],
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
    """AD IR save creates ad_ir.json and real_buffers.pt."""
    gm = symbolic_trace(_ToyModule())
    gm.register_buffer("scale", torch.tensor(1.5))

    ir, real_buffers = extract_ir(gm)
    save_ir(ir, real_buffers, tmp_path)

    assert (tmp_path / "ad_ir.json").exists()
    assert (tmp_path / "real_buffers.pt").exists()

    ir_data = json.loads((tmp_path / "ad_ir.json").read_text())
    assert ir_data["format_version"] == 4
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
    assert "format_version" not in manifest
    assert manifest["boundary_class"] == _BOUNDARY_CLASS_PRE_WEIGHT_LOAD
    assert isinstance(manifest["producer_hash"], str)
    assert manifest["producer_hash"]
    assert manifest["trtllm_version"] == TRTLLM_VERSION
    assert "hook_spec_count" in manifest
    assert "has_unserializable_hooks" in manifest
    assert "source_model_hooks_required" in manifest
    assert manifest["has_unserializable_hooks"] is False


def test_manifest_ad_ir_files_written(monkeypatch, tmp_path):
    """Saving writes AD IR files (ad_ir.json, no legacy lean files)."""
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
    )
    optimizer = InferenceOptimizer(
        factory=factory,
        config=_make_optimizer_config(),
        pipeline_cache_config=cache_config,
    )
    optimizer(cache_seq_interface)

    assert list(tmp_path.rglob("ad_ir.json"))
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
        boundaries=["_unit_test_boundary_for_pipeline_cache"],
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
