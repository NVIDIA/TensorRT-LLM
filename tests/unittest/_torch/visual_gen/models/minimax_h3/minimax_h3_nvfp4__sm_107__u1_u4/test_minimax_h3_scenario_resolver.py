# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing and settings of the MiniMax-H3 scenario ``minimax_h3_nvfp4__sm_107__u1_u4`` (CPU only)."""

import ast
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tensorrt_llm._torch.visual_gen.models.minimax_h3 import scenarios
from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
    config as scenario_config,
)
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_PACKAGE = scenario_config.__name__.rsplit(".", 1)[0]


@pytest.fixture(autouse=True)
def _no_selection_yet(monkeypatch):
    """Each test starts as a process that has not selected the scenario."""
    monkeypatch.setattr(scenario_config, "SCENARIO", scenario_config.ScenarioConfig())
    monkeypatch.setattr(scenario_config, "_SELECTED", None)


def _snapshot():
    return scenario_config.SCENARIO, scenario_config._SELECTED


class _Family:
    """Stands in for MiniMaxH3Pipeline: the resolver only hands the class back."""


def _config(
    *,
    quant_algo=QuantAlgo.NVFP4,
    dynamic=False,
    backend="CUDNN",
    fp8=True,
    ulysses=1,
    tp=1,
    async_ulysses=False,
    mode="auto",
    overrides=None,
    layer_quant=None,
    attention_precision=None,
    runtime_attention_precision=None,
):
    qac = SimpleNamespace(qk_dtype="fp8", v_dtype="fp8") if fp8 else None
    extra = {"visual_gen_scenario": mode}
    if overrides is not None:
        extra["minimax_h3_scenario"] = overrides
    if runtime_attention_precision is not None:
        extra["minimax_h3_attention_precision"] = runtime_attention_precision
    quantization_config = {"quant_algo": "NVFP4"}
    if attention_precision is not None:
        quantization_config["attention_precision"] = attention_precision
    return SimpleNamespace(
        model_configs={
            "transformer": SimpleNamespace(
                quant_config=SimpleNamespace(quant_algo=quant_algo),
                quant_config_dict=layer_quant,
                pretrained_config=SimpleNamespace(quantization_config=quantization_config),
            )
        },
        quant_config=SimpleNamespace(quant_algo=quant_algo),
        quant_config_dict=layer_quant,
        dynamic_weight_quant=dynamic,
        attention=SimpleNamespace(backend=backend, quant_attention_config=qac),
        visual_gen_mapping=SimpleNamespace(ulysses_size=ulysses),
        mapping=SimpleNamespace(tp_size=tp),
        parallel=SimpleNamespace(async_ulysses=async_ulysses),
        extra_attrs=extra,
    )


def test_off_keeps_the_family_and_imports_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(scenarios, "ROUTES", (lambda cfg: called.append(cfg),))
    assert scenarios.resolve_scenario(_Family, _config(mode="off")) is _Family
    assert called == []


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_route_declines_without_sm_107(monkeypatch, mode):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 3))
    if mode == "auto":
        assert scenarios.resolve_scenario(_Family, _config(mode="auto")) is _Family
    else:
        with pytest.raises(RuntimeError, match="SM 107"):
            scenarios.resolve_scenario(_Family, _config(mode="require"))


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (dict(quant_algo=QuantAlgo.FP8), "NVFP4"),
        (dict(dynamic=True), "dynamic_weight_quant"),
        (dict(backend="FA4"), "CUDNN"),
        (dict(fp8=False), "fp8"),
        (dict(ulysses=2), "Ulysses 1 or 4"),
        (dict(tp=2), "tp_size"),
        (dict(async_ulysses=True), "async_ulysses"),
    ],
)
def test_route_names_the_failed_requirement(monkeypatch, kwargs, reason):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    with pytest.raises(scenarios.NoScenario, match=reason):
        scenarios.route_nvfp4_sm_107_u1_u4(_config(**kwargs))


def _layers(*algos):
    """A per-layer table as the MIXED_PRECISION parser returns it: name -> config with a quant_algo."""
    return {
        f"transformer_blocks.{i}.ff.net.2": SimpleNamespace(quant_algo=a)
        for i, a in enumerate(algos)
    }


def _stub_scenario_package(monkeypatch):
    """Keep the route from importing the scenario's SM 107 modules on a CPU host."""
    pipeline = ModuleType(f"{_PACKAGE}.pipeline")
    pipeline.MiniMaxH3Nvfp4Sm107U1U4Pipeline = type("MiniMaxH3Nvfp4Sm107U1U4Pipeline", (), {})
    monkeypatch.setitem(sys.modules, f"{_PACKAGE}.pipeline", pipeline)
    return pipeline.MiniMaxH3Nvfp4Sm107U1U4Pipeline


@pytest.mark.parametrize("global_algo", [QuantAlgo.NVFP4, QuantAlgo.FP8])
@pytest.mark.parametrize("ulysses", [1, 4])
def test_route_admits_a_mixed_nvfp4_fp8_checkpoint(monkeypatch, global_algo, ulysses):
    """Admission reads the per-layer table, not the global algorithm (the largest group)."""
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    cls = _stub_scenario_package(monkeypatch)
    config = _config(
        quant_algo=global_algo,
        layer_quant=_layers(QuantAlgo.NVFP4, QuantAlgo.FP8),
        ulysses=ulysses,
    )
    assert scenarios.route_nvfp4_sm_107_u1_u4(config) is cls


def test_route_keeps_the_uniform_nvfp4_checkpoint_at_ulysses_4(monkeypatch):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    cls = _stub_scenario_package(monkeypatch)
    assert scenarios.route_nvfp4_sm_107_u1_u4(_config(ulysses=4)) is cls


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (dict(layer_quant=_layers(QuantAlgo.NVFP4, QuantAlgo.FP8), ulysses=2), "Ulysses 1 or 4"),
        (
            dict(layer_quant=_layers(QuantAlgo.NVFP4, QuantAlgo.FP8), dynamic=True),
            "dynamic_weight_quant",
        ),
        (
            dict(layer_quant=_layers(QuantAlgo.NVFP4, QuantAlgo.FP8_BLOCK_SCALES)),
            "FP8_BLOCK_SCALES",
        ),
        (dict(layer_quant=_layers(QuantAlgo.NVFP4, None)), "None"),
        (dict(layer_quant={}), "no layers"),
        (dict(layer_quant=_layers(QuantAlgo.NVFP4, QuantAlgo.FP8), ulysses=8), "Ulysses 1 or 4"),
        (
            dict(
                layer_quant=_layers(QuantAlgo.NVFP4, QuantAlgo.FP8), ulysses=4, async_ulysses=True
            ),
            "async_ulysses",
        ),
    ],
)
def test_route_names_why_a_mixed_checkpoint_is_declined(monkeypatch, kwargs, reason):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    with pytest.raises(scenarios.NoScenario, match=reason):
        scenarios.route_nvfp4_sm_107_u1_u4(_config(**kwargs))


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_declined_mixed_checkpoint_falls_to_the_family_or_raises(monkeypatch, mode):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    config = _config(mode=mode, layer_quant=_layers(QuantAlgo.NVFP4, QuantAlgo.FP8), ulysses=8)
    if mode == "auto":
        assert scenarios.resolve_scenario(_Family, config) is _Family
    else:
        with pytest.raises(RuntimeError, match="Ulysses 1 or 4"):
            scenarios.resolve_scenario(_Family, config)


_ATTENTION_PRECISION = {"schema_version": 1, "layers": {"transformer_blocks.45.attn": "bf16"}}


@pytest.mark.parametrize("source", ["attention_precision", "runtime_attention_precision"])
@pytest.mark.parametrize("ulysses", [1, 4])
def test_route_admits_attention_precision(monkeypatch, source, ulysses):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    cls = _stub_scenario_package(monkeypatch)
    config = _config(ulysses=ulysses, **{source: _ATTENTION_PRECISION})
    assert scenarios.route_nvfp4_sm_107_u1_u4(config) is cls


@pytest.mark.parametrize("source", ["attention_precision", "runtime_attention_precision"])
@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (dict(ulysses=2), "Ulysses 1 or 4"),
        (dict(ulysses=8), "Ulysses 1 or 4"),
        (dict(ulysses=4, async_ulysses=True), "async_ulysses"),
    ],
)
def test_route_declines_attention_precision_outside_ulysses_1_and_4(
    monkeypatch, source, kwargs, reason
):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    with pytest.raises(scenarios.NoScenario, match=reason):
        scenarios.route_nvfp4_sm_107_u1_u4(_config(**kwargs, **{source: _ATTENTION_PRECISION}))


def test_route_keeps_ulysses_4_for_an_empty_attention_precision(monkeypatch):
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    cls = _stub_scenario_package(monkeypatch)
    empty = {"schema_version": 1, "layers": {}}
    assert scenarios.route_nvfp4_sm_107_u1_u4(_config(ulysses=4, attention_precision=empty)) is cls


def test_route_leaves_the_attention_precision_entry_to_the_transformer(monkeypatch):
    """The route does not parse the entry; the transformer it selects validates it once."""
    monkeypatch.setattr(scenarios, "_device_capability", lambda: (10, 7))
    cls = _stub_scenario_package(monkeypatch)
    bad = {"schema_version": 1, "layers": {"transformer_blocks.45.attn": "fp8"}}
    assert scenarios.route_nvfp4_sm_107_u1_u4(_config(attention_precision=bad)) is cls
    assert not hasattr(scenarios, "resolve_attention_precision")


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="visual_gen_scenario"):
        scenarios.resolve_scenario(_Family, _config(mode="sometimes"))


def test_config_defaults_are_the_recipe():
    cfg = scenario_config.ScenarioConfig()
    cfg.validate()
    assert cfg.attention_fmha == "trtllm-gen-fp16sm"
    assert cfg.fp8_scale_fusion == "qkstatic"
    assert cfg.skip_corr_threshold == 1.0
    assert cfg.grouped_cast and cfg.vamax_fold and cfg.fused_quant and cfg.out_quant
    assert cfg.preloop_device_plan and cfg.preloop_async_copies and cfg.preloop_noise_early
    assert cfg.resolved_vae_fp8_qtile(1) == (32, 8, 8)
    assert cfg.resolved_vae_fp8_qtile(4) == (64, 4, 8)


def test_first_selection_applies_overrides():
    cfg = scenario_config.apply({"skip_corr_threshold": 0.0, "vae_fp8_qtile": [16, 4, 4]})
    assert cfg.skip_corr_threshold == 0.0 and cfg.vae_fp8_qtile == (16, 4, 4)
    assert scenario_config.SCENARIO is cfg


def test_equal_second_selection_returns_the_same_object():
    first = scenario_config.apply({"skip_corr_threshold": 0.0, "out_quant": False})
    held = scenario_config.SCENARIO  # what a module importing SCENARIO now would keep
    again = scenario_config.apply({"out_quant": False, "skip_corr_threshold": 0.0})
    assert again is first and scenario_config.SCENARIO is held


def test_differing_second_selection_raises_and_names_the_fields():
    first = scenario_config.apply({"skip_corr_threshold": 0.0, "out_quant": False})
    with pytest.raises(RuntimeError, match="one scenario configuration per process") as err:
        scenario_config.apply({"skip_corr_threshold": 0.0, "amax_allreduce": "nccl"})
    message = str(err.value)
    assert (
        "'out_quant': (False, True)" in message and "'amax_allreduce': ('symm', 'nccl')" in message
    )
    assert "skip_corr_threshold" not in message
    assert _snapshot() == (first, first) and scenario_config.SCENARIO is first


def test_defaults_after_an_override_selection_raise():
    scenario_config.apply({"out_quant": False})
    with pytest.raises(RuntimeError, match="out_quant"):
        scenario_config.apply(None)


@pytest.mark.parametrize("selected_first", [False, True])
@pytest.mark.parametrize(
    "overrides, reason",
    [({"no_such_switch": 1}, "unknown settings"), ({"fp8_scale_fusion": "dynamic"}, "qkstatic")],
)
def test_invalid_mapping_raises_and_changes_nothing(selected_first, overrides, reason):
    if selected_first:
        scenario_config.apply({"skip_corr_threshold": 0.0})
    before = _snapshot()
    with pytest.raises(ValueError, match=reason):
        scenario_config.apply(overrides)
    after = _snapshot()
    assert after[0] is before[0] and after[1] is before[1]


def test_config_is_frozen():
    cfg = scenario_config.apply(None)
    with pytest.raises(FrozenInstanceError):
        cfg.out_quant = False


def test_module_imported_before_a_selection_locks_the_defaults(monkeypatch):
    consumer = ModuleType(f"{_PACKAGE}.kernels._probe_consumer")
    consumer.SCENARIO = scenario_config.SCENARIO  # as `from ..config import SCENARIO` leaves it
    monkeypatch.setitem(sys.modules, consumer.__name__, consumer)
    assert scenario_config.apply(None) is consumer.SCENARIO
    with pytest.raises(RuntimeError, match="out_quant"):
        scenario_config.apply({"out_quant": False})
    assert scenario_config.SCENARIO is consumer.SCENARIO


def test_module_without_a_binding_does_not_lock_the_defaults(monkeypatch):
    namespace = ModuleType(f"{_PACKAGE}.kernels._probe_namespace")
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    cfg = scenario_config.apply({"skip_corr_threshold": 0.0})
    assert scenario_config.SCENARIO is cfg and cfg.skip_corr_threshold == 0.0


def _scenario_reads(tree: ast.AST) -> tuple[set[str], int]:
    names, writes = set(), 0
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "SCENARIO"
        ):
            names.add(node.attr)
            writes += not isinstance(node.ctx, ast.Load)
    return names, writes


def test_every_setting_reaches_a_consumer_through_the_shared_object():
    """Static: the modules that read the settings bind ``SCENARIO`` by name and only read it, and together
    with the config's own methods they read every field, so one object carries all settings to all of them."""
    root = Path(scenario_config.__file__).parent
    field_names = {f.name for f in fields(scenario_config.ScenarioConfig)}
    config_tree = ast.parse(Path(scenario_config.__file__).read_text())
    cls = next(
        n for n in config_tree.body if isinstance(n, ast.ClassDef) and n.name == "ScenarioConfig"
    )
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    read = {
        n.attr
        for n in ast.walk(cls)
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"
    } & field_names
    consumers = []
    for path in sorted(root.rglob("*.py")):
        if path.name in ("config.py", "__init__.py"):
            continue
        tree = ast.parse(path.read_text())
        names, writes = _scenario_reads(tree)
        if not names:
            continue
        consumers.append(path.name)
        imported = [
            alias.name
            for n in tree.body
            if isinstance(n, ast.ImportFrom) and n.level >= 1 and n.module == "config"
            for alias in n.names
        ]
        assert imported == ["SCENARIO"], f"{path.name} imports {imported} from the scenario config"
        assert writes == 0, f"{path.name} writes a setting"
        assert names <= field_names | methods, (
            f"{path.name} reads unknown settings {sorted(names - field_names - methods)}"
        )
        read |= names & field_names
    assert consumers, "no module of the scenario reads its settings"
    assert read == field_names, f"settings no module reads: {sorted(field_names - read)}"
