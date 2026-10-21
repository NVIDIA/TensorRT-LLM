# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One KV-cache configuration per unit, whichever gate builds the engine.

The accuracy gate's ``trtllm-eval`` reads the unit's options file, and
``quickstart_advanced.py`` (boot, acceptance, the witness legs) builds its
``KvCacheConfig`` from flags that have no ``mamba_state_config``. Without a
snapshot policy the engine turns block reuse off for a hybrid model, so every
quickstart gate ran without the reuse its unit asked for. These tests hold the
resolver to the keys the quickstart has no flag for, the gates to handing them
over, the harness to merging them, the gates to failing on the engine's
warning, and preflight to saying where reuse will be lost.
"""

from __future__ import annotations

import ast
import json
import shlex
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates, witness
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import engine_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import (
    RunEnvironment,
    resolve_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run import preflight
from agent_flow.workflows.trtllm_modeling_v2.spec.variants import variant_spec

REPO_ROOT = Path(__file__).resolve().parents[4]
SHIM = witness.QUICKSTART_SHIM

#: The live run's snapshot policy (``configs/core.llm_api.yaml``).
_SNAPSHOTS = {
    "periodic_snapshot_interval": 2048,
    "additional_snapshot_offsets_from_start": [256, 512, 768],
}
_UNIT_KV = {
    "free_gpu_memory_fraction": 0.6,
    "enable_block_reuse": True,
    "use_kv_cache_manager_v2": True,
    "mamba_state_config": _SNAPSHOTS,
}
_HANDED = {"mamba_state_config": _SNAPSHOTS}
_WARNING = (
    "[TensorRT-LLM][WARNING] Disabling KV cache block reuse for the hybrid Mamba model "
    "because no Mamba state snapshot policy is configured. Set "
    "kv_cache_config.mamba_state_config.periodic_snapshot_interval to a positive value "
    "or provide additional snapshot offsets to enable block reuse."
)


def _options(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / f"{name}.llm_api.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _task(tmp_path: Path, *, options: dict | None = None, **extra) -> dict:
    task: dict = {
        "checkpoint_path": "/ckpt/model",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "features": ["disagg"],
        "runtime": {"cuda_graph": False, "extra_flags": ["--max_batch_size=8"]},
        "accuracy": {"benchmark": "gsm8k"},
    }
    if options is not None:
        task["accuracy"]["llm_api_options"] = {
            unit: str(_options(tmp_path, unit, data)) for unit, data in options.items()
        }
    task.update(extra)
    return task


def _env() -> RunEnvironment:
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _assignment(command: str) -> dict | None:
    """The handed-over keys a launch line exports, or ``None`` when it exports none."""
    prefix = f"{gates.KV_CACHE_CONFIG_ENV}="
    for word in shlex.split(command):
        if word.startswith(prefix):
            return json.loads(word[len(prefix) :])
    return None


def _records(*texts: str) -> str:
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
        for n, t in enumerate(texts)
    )


class FakeKvCacheConfig:
    """``KvCacheConfig`` as the merge sees it: keyword construction and ``model_fields_set``."""

    def __init__(self, **fields):
        self.fields = dict(fields)

    @property
    def model_fields_set(self) -> set[str]:
        return set(self.fields)

    def __getattr__(self, name):
        try:
            return self.__dict__["fields"][name]
        except KeyError:
            raise AttributeError(name) from None


# ------------------------------------------------------------- the resolver


def test_the_quickstart_gets_the_keys_it_has_no_flag_for():
    options = {
        "kv_cache_config": {
            **_UNIT_KV,
            "dtype": "fp8",
            "tokens_per_block": 64,
            "mamba_ssm_cache_dtype": "float32",
            "max_tokens": 4096,
            "enable_partial_reuse": False,
        },
        "max_num_tokens": 8192,
    }
    assert gates.quickstart_kv_cache_config(options) == {
        "mamba_state_config": _SNAPSHOTS,
        "max_tokens": 4096,
        "enable_partial_reuse": False,
    }
    assert gates.quickstart_kv_cache_config({}) == {}
    assert gates.quickstart_kv_cache_config({"kv_cache_config": None}) == {}


def test_the_flagged_keys_are_the_ones_the_quickstart_sets():
    """Read from the tree's own quickstart, so a new flag fails here and not in a gate."""
    script = REPO_ROOT / "examples" / "llm-api" / "quickstart_advanced.py"
    if not script.is_file():
        pytest.skip(f"{script} is not in this checkout")
    calls = [
        node
        for node in ast.walk(ast.parse(script.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "KvCacheConfig"
    ]
    assert calls, "the quickstart builds no KvCacheConfig"
    assert {k.arg for k in calls[0].keywords} == gates.QUICKSTART_KV_CACHE_KEYS


def test_the_engine_warning_the_gates_read_is_the_engines():
    loader = REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "model_loader.py"
    if not loader.is_file():
        pytest.skip(f"{loader} is not in this checkout")
    source = loader.read_text(encoding="utf-8")
    assert '"Disabling KV cache block reuse for the hybrid Mamba model "' in source
    assert (
        gates.REUSE_DISABLED_MARKER == "Disabling KV cache block reuse for the hybrid Mamba model"
    )


def test_a_unit_resolves_its_own_options_file(tmp_path):
    task = _task(
        tmp_path,
        options={
            "core": {"kv_cache_config": _UNIT_KV},
            "mtp": {"kv_cache_config": {"free_gpu_memory_fraction": 0.5}},
        },
    )
    assert gates.resolve_kv_cache_config(task, "core") == _HANDED
    assert gates.resolve_kv_cache_config(task, "mtp") == {}
    assert gates.resolve_kv_cache_config(_task(tmp_path), "core") == {}


def test_a_variant_hands_over_its_merged_options(tmp_path):
    task = _task(
        tmp_path,
        options={"core": {"kv_cache_config": _UNIT_KV}},
        variants={
            "snap512": {
                "llm_api_options": {
                    "kv_cache_config": {"mamba_state_config": {"periodic_snapshot_interval": 512}}
                }
            },
            "kv_fp8": {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}},
        },
    )
    merged = {"mamba_state_config": {**_SNAPSHOTS, "periodic_snapshot_interval": 512}}
    spec = variant_spec(task, "snap512", world_size=1, options_dir=tmp_path / "variants")
    assert gates.resolve_kv_cache_config(spec, "snap512") == merged
    launches = {launch.name: launch for launch in gates.launch_kv_caches(task)}
    assert launches["snap512"].handed == merged and launches["snap512"].unit == "core"
    assert launches["kv_fp8"].handed == _HANDED
    assert launches["core"].snapshots and launches["core"].quickstart_reuse


# ---------------------------------------------------------- the hand-off


def test_the_keys_travel_as_json_in_the_launch_prefix(tmp_path):
    task = _task(tmp_path, options={"core": {"kv_cache_config": _UNIT_KV}})
    env = gates.with_kv_cache_config(_env(), task, "core")
    assert json.loads(env.env[gates.KV_CACHE_CONFIG_ENV]) == _HANDED
    command = gates.engine_driver_command(env, tmp_path / "s.json")
    assert _assignment(command) == _HANDED
    # A unit with nothing to hand over launches without the variable, and an
    # environment that carried another unit's keys loses them.
    assert gates.KV_CACHE_CONFIG_ENV not in gates.with_kv_cache_config(env, task, "mtp").env
    assert gates.with_kv_cache_config(_env(), task, "mtp").env == _env().env


def test_the_orchestrator_and_the_harness_name_one_variable():
    assert engine_driver.KV_CACHE_CONFIG_ENV == gates.KV_CACHE_CONFIG_ENV
    assert engine_driver.MODELING_V2_ENV == gates.MODELING_V2_ENV


def test_the_boot_gate_hands_the_keys_to_its_engine(tmp_path):
    task = _task(tmp_path, options={"core": {"kv_cache_config": _UNIT_KV}})
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=_records("Paris", "x", "y"))

    result = gates.run_boot_gate(_env(), task_data=task, runner=run, unit="core", workdir=tmp_path)
    assert result.passed, result.detail
    assert _assignment(seen[0]) == _HANDED
    seen.clear()
    gates.run_boot_gate(
        _env(), task_data=_task(tmp_path), runner=run, unit="core", workdir=tmp_path
    )
    assert _assignment(seen[0]) is None


def test_both_acceptance_legs_get_the_same_keys(tmp_path):
    task = _task(tmp_path, options={"mtp": {"kv_cache_config": _UNIT_KV}})
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout="")

    gates.run_acceptance_gate(_env(), task_data=task, runner=run, unit="mtp", workdir=tmp_path)
    assert len(seen) == 2
    assert "TRTLLM_MODELING_V2=require" in seen[0] and "TRTLLM_MODELING_V2=off" in seen[1]
    assert _assignment(seen[0]) == _assignment(seen[1]) == _HANDED


def test_every_witness_leg_and_leg_b_get_the_keys(tmp_path):
    """Leg A is the shim over the quickstart; leg B is the run's own launcher."""
    block = {
        "contrast_env": {"MV2FLOW_DISAGG_LEG": "B"},
        "contrast_command": "/run/disagg_leg_b.py",
        "min_generated_tokens": 1,
    }
    task = _task(tmp_path, options={"disagg": {"kv_cache_config": _UNIT_KV}})
    task["accuracy"]["witness"] = {"disagg": block}
    seen: list[str] = []
    out = "[0] Prompt: 'p'..., Generated text: ' A'...\n[0] Witness token ids: [1, 2]\n"

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=_ablated(command) or out)

    result = witness.run_declared_witness(
        RunEnvironment(kind="local", python="/py", world_size=1),
        unit="disagg",
        member="tp1",
        task_data=task,
        runner=run,
        workdir=tmp_path / "w",
    )
    assert result.passed, result.detail
    assert len(seen) == 3 and str(SHIM) in seen[0] and "disagg_leg_b.py" in seen[1]
    # The third run is leg B's self-test, which gets them too.
    assert _assignment(seen[0]) == _assignment(seen[1]) == _assignment(seen[2]) == _HANDED


def _ablated(command: str) -> str:
    """Leg B's self-test run destroys the state it received, and its continuation moves."""
    if f"{witness.LEG_B_ABLATE_ENV}={witness.ABLATE_STATE}" not in command:
        return ""
    return "[0] Prompt: 'p'..., Generated text: ' B'...\n[0] Witness token ids: [3]\n" + (
        "Witness ablated: conv, ssm\n"
    )


# ------------------------------------------------------------ the harness


def test_kv_cache_overrides_read_the_variable():
    variable = gates.KV_CACHE_CONFIG_ENV
    assert engine_driver.kv_cache_overrides({}) == {}
    assert engine_driver.kv_cache_overrides({variable: "  "}) == {}
    assert engine_driver.kv_cache_overrides({variable: json.dumps(_HANDED)}) == _HANDED
    with pytest.raises(ValueError, match="not a JSON object"):
        engine_driver.kv_cache_overrides({variable: "[1, 2]"})


def test_the_merge_keeps_what_the_quickstart_set():
    """The quickstart's flags are the gate argv's settings; only the rest is added."""
    stated = FakeKvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.6)
    merged = engine_driver.merge_kv_cache_config(
        stated, {**_HANDED, "free_gpu_memory_fraction": 0.1}
    )
    assert isinstance(merged, FakeKvCacheConfig) and merged is not stated
    assert merged.fields == {
        "enable_block_reuse": True,
        "free_gpu_memory_fraction": 0.6,
        "mamba_state_config": _SNAPSHOTS,
    }
    assert engine_driver.merge_kv_cache_config(stated, {}) is stated
    assert engine_driver.merge_kv_cache_config(None, _HANDED) == _HANDED
    assert engine_driver.merge_kv_cache_config({"dtype": "fp8"}, _HANDED) == {
        **_HANDED,
        "dtype": "fp8",
    }
    with pytest.raises(TypeError, match="not a pydantic model"):
        engine_driver.merge_kv_cache_config(types.SimpleNamespace(dtype="auto"), _HANDED)


def test_the_merge_validates_nested_settings_as_the_llm_api_does():
    """A real pydantic model: the nested mapping becomes its model, an unknown key is refused."""
    pydantic = pytest.importorskip("pydantic")

    class MambaStateConfig(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        periodic_snapshot_interval: int = 0
        additional_snapshot_offsets_from_start: list[int] = []

    class KvCacheConfig(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid")
        enable_block_reuse: bool = True
        free_gpu_memory_fraction: float = 0.9
        mamba_state_config: MambaStateConfig = MambaStateConfig()

    stated = KvCacheConfig(enable_block_reuse=True, free_gpu_memory_fraction=0.6)
    merged = engine_driver.merge_kv_cache_config(stated, _HANDED)
    assert isinstance(merged.mamba_state_config, MambaStateConfig)
    assert merged.mamba_state_config.periodic_snapshot_interval == 2048
    assert merged.free_gpu_memory_fraction == 0.6 and merged.enable_block_reuse
    with pytest.raises(pydantic.ValidationError):
        engine_driver.merge_kv_cache_config(stated, {"no_such_key": 1})


def test_engine_kwargs_add_the_target_and_the_keys():
    kwargs = {"model": "m", "kv_cache_config": FakeKvCacheConfig(enable_block_reuse=True)}
    out = engine_driver.engine_kwargs(kwargs, mode="require", kv_cache=_HANDED)
    assert out["modeling_v2"] == "require" and out["model"] == "m"
    assert out["kv_cache_config"].fields == {"enable_block_reuse": True, **_HANDED}
    assert "modeling_v2" not in kwargs, "the caller's mapping is not changed"
    explicit = engine_driver.engine_kwargs({"modeling_v2": "off"}, mode="require", kv_cache=None)
    assert explicit == {"modeling_v2": "off"}
    assert engine_driver.engine_kwargs(kwargs, mode=None, kv_cache=None) == kwargs


def test_the_driver_builds_its_engine_with_the_keys(monkeypatch):
    built: list[dict] = []

    class LLM:
        def __init__(self, **kwargs):
            built.append(kwargs)

    quickstart = types.ModuleType("fake_quickstart")
    quickstart.LLM = LLM
    quickstart.parse_arguments = lambda: types.SimpleNamespace()

    def setup_llm(args):
        return quickstart.LLM(model="m", kv_cache_config=FakeKvCacheConfig(dtype="auto")), None

    quickstart.setup_llm = setup_llm
    monkeypatch.setenv(gates.KV_CACHE_CONFIG_ENV, json.dumps(_HANDED))
    monkeypatch.setenv(gates.MODELING_V2_ENV, "require")
    engine_driver.build_llm(quickstart, [])
    assert built[0]["modeling_v2"] == "require"
    assert built[0]["kv_cache_config"].fields == {"dtype": "auto", **_HANDED}
    assert quickstart.LLM is LLM, "the quickstart's class is restored"
    monkeypatch.delenv(gates.KV_CACHE_CONFIG_ENV)
    monkeypatch.delenv(gates.MODELING_V2_ENV)
    engine_driver.build_llm(quickstart, [])
    assert built[1]["kv_cache_config"].fields == {"dtype": "auto"}
    assert "modeling_v2" not in built[1]


_FAKE_QUICKSTART = textwrap.dedent(
    """
    import argparse
    import json


    class KvCacheConfig:
        def __init__(self, **fields):
            self.fields = fields

        @property
        def model_fields_set(self):
            return set(self.fields)

        def __getattr__(self, name):
            try:
                return self.__dict__["fields"][name]
            except KeyError:
                raise AttributeError(name) from None


    class LLM:
        def __init__(self, model, **kwargs):
            kv = kwargs["kv_cache_config"]
            print("BUILT " + json.dumps(
                {"modeling_v2": kwargs.get("modeling_v2"), "kv": kv.fields}, sort_keys=True
            ))


    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir")
        parser.add_argument("--disable_kv_cache_reuse", action="store_true")
        return parser.parse_args()


    def setup_llm(args):
        kv = KvCacheConfig(
            enable_block_reuse=not args.disable_kv_cache_reuse, free_gpu_memory_fraction=0.9
        )
        return LLM(model=args.model_dir, kv_cache_config=kv), None


    def main():
        setup_llm(parse_arguments())
    """
)


def test_the_shim_merges_the_keys_into_the_quickstarts_config(tmp_path):
    """End to end, as a witness leg runs: the shim, the tree's quickstart, the variable."""
    script = tmp_path / "quickstart_advanced.py"
    script.write_text(_FAKE_QUICKSTART, encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(SHIM), str(script), "--model_dir=/ckpt"],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            gates.KV_CACHE_CONFIG_ENV: json.dumps(_HANDED),
            gates.MODELING_V2_ENV: "require",
        },
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    [line] = [x for x in done.stdout.splitlines() if x.startswith("BUILT ")]
    assert json.loads(line[len("BUILT ") :]) == {
        "modeling_v2": "require",
        "kv": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.9, **_HANDED},
    }


# ---------------------------------------------------------- the log check


def test_the_warning_fails_with_the_missing_setting_named():
    assert gates.reuse_disabled_detail("all fine\n", options="/o.yaml") is None
    detail = gates.reuse_disabled_detail(f"loading\n{_WARNING}\nready\n", options="/o.yaml")
    assert detail is not None
    assert "asks for KV-cache block reuse and the engine turned it off" in detail
    assert "Disabling KV cache block reuse for the hybrid Mamba model" in detail
    assert "`kv_cache_config.mamba_state_config`" in detail and "/o.yaml" in detail
    assert "kv_cache_reuse" in detail
    assert "an options file for the unit" in gates.reuse_disabled_detail(_WARNING, options=None)
    # Another reason the engine turns reuse off is not this one.
    mla = "KV cache reuse for MLA can only be enabled on SM90, disable enable_block_reuse"
    assert gates.reuse_disabled_detail(mla, options=None) is None


def test_a_boot_gate_whose_engine_lost_reuse_fails(tmp_path):
    task = _task(tmp_path, options={"core": {"kv_cache_config": {"enable_block_reuse": True}}})

    def run(command, env=None, timeout=0):
        return CommandResult(returncode=0, stdout=_records("Paris", "x", "y"), stderr=_WARNING)

    result = gates.run_boot_gate(_env(), task_data=task, runner=run, unit="core", workdir=tmp_path)
    assert not result.passed
    assert "mamba_state_config" in result.detail and "core.llm_api.yaml" in result.detail


def test_an_accuracy_gate_whose_engine_lost_reuse_fails(tmp_path):
    task = _task(tmp_path, options={"core": {"kv_cache_config": {"enable_block_reuse": True}}})
    task["accuracy"]["floor"] = {"rule": "tolerance"}
    task["accuracy"]["tolerance"] = 5.0
    score = "lm-eval gsm8k average accuracy: 90.00\n"

    def run(command, env=None, timeout=0):
        return CommandResult(returncode=0, stdout=f"{_WARNING}\n{score}")

    result = gates.run_accuracy_gate(
        _env(), task_data=task, reference=90.0, runner=run, unit="core"
    )
    assert not result.passed and "mamba_state_config" in result.detail

    def clean(command, env=None, timeout=0):
        return CommandResult(returncode=0, stdout=score)

    assert gates.run_accuracy_gate(
        _env(), task_data=task, reference=90.0, runner=clean, unit="core"
    ).passed


def test_an_acceptance_gate_stops_at_the_leg_that_lost_reuse(tmp_path):
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=_WARNING)

    result = gates.run_acceptance_gate(
        _env(), task_data=_task(tmp_path), runner=run, unit="mtp", workdir=tmp_path
    )
    assert not result.passed and len(seen) == 1
    assert result.detail.startswith("the TRTLLM_MODELING_V2=require leg: the unit asks for")


# --------------------------------------------------------------- preflight


#: A composite checkpoint of a family the engine does not name: hybrid only through its layer list.
_HYBRID = {
    "architectures": ["NewModelForConditionalGeneration"],
    "model_type": "newmodel",
    "text_config": {
        "model_type": "newmodel_text",
        "layer_types": ["linear_attention", "mlp", "moe_attention", "moe_mamba"],
    },
}
_HYBRID_MARKER = "`text_config.layer_types` with linear_attention, moe_mamba"


@pytest.mark.parametrize(
    "config,marker",
    [
        (_HYBRID, _HYBRID_MARKER),
        (
            {"model_type": "glm5_next", "text_config": {"model_type": "glm5_next_text"}},
            "`model_type: glm5_next`",
        ),
        (
            {"architectures": ["Qwen4ExpForConditionalGeneration"]},
            "`architectures: Qwen4ExpForConditionalGeneration`",
        ),
        ({"hybrid_override_pattern": "M-M-M*-"}, "`hybrid_override_pattern`"),
        ({"architectures": ["Qwen3NextForCausalLM"]}, "`architectures: Qwen3NextForCausalLM`"),
        (
            {
                "model_type": "kimi_k3",
                "text_config": {"model_type": "kimi_linear", "linear_attn_config": {}},
            },
            "`text_config.model_type: kimi_linear`",
        ),
        (
            {"model_type": "newmodel", "layer_types": ["full_attention", "linear_attention"]},
            "`layer_types` with linear_attention",
        ),
        (
            {"text_config": {"layers_block_type": ["mamba", "attention", "mamba"]}},
            "`text_config.layers_block_type` with mamba",
        ),
        (
            {"model_type": "newmodel", "layer_types": ["attn", "moe_mamba"]},
            "`layer_types` with moe_mamba",
        ),
        ({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, None),
        ({"layer_types": ["sliding_attention", "full_attention"]}, None),
        ({"model_type": "kimi_linear"}, None),
        ({"hybrid_override_pattern": ""}, None),
    ],
)
def test_hybrid_marker(config, marker):
    assert preflight.hybrid_marker(config) == marker


@pytest.mark.parametrize(
    "kv_cache,policy",
    [
        ({}, False),
        ({"mamba_state_config": {"periodic_snapshot_interval": 0}}, False),
        ({"mamba_state_config": {"enable_branch_snapshot": True}}, False),
        ({"mamba_state_config": {"periodic_snapshot_interval": 2048}}, True),
        ({"mamba_state_config": {"additional_snapshot_offsets_from_end": [0]}}, True),
        ({"mamba_state_cache_interval": 128}, True),
        (
            {
                "mamba_state_config": {"periodic_snapshot_interval": 2048},
                "block_reuse_config": {"policy": "per_conversation"},
            },
            False,
        ),
        (
            {
                "mamba_state_config": {"additional_snapshot_offsets_from_start": [512]},
                "block_reuse_config": {"policy": "per_conversation"},
            },
            True,
        ),
    ],
)
def test_the_snapshot_policy_is_the_engines(kv_cache, policy):
    assert gates.has_snapshot_policy(kv_cache) is policy


def test_a_hybrid_checkpoint_without_snapshots_gets_a_note(tmp_path):
    task = _task(
        tmp_path,
        options={
            "core": {"kv_cache_config": _UNIT_KV},
            "mtp": {"kv_cache_config": {"enable_block_reuse": True}},
        },
        features=["mtp"],
    )
    notes = preflight.check_kv_cache_configs(task, _HYBRID)
    [handed] = [n for n in notes if n.startswith("kv cache:")]
    assert handed.startswith("kv cache: core: ") and json.dumps(_HANDED, sort_keys=True) in handed
    [reuse] = [n for n in notes if n.startswith("block reuse:")]
    assert reuse.startswith("block reuse: mtp: ask for KV-cache block reuse")
    assert _HYBRID_MARKER in reuse and "mamba_state_config" in reuse
    # The same spec on a checkpoint the engine does not treat as hybrid.
    assert not [n for n in preflight.check_kv_cache_configs(task, {}) if "block reuse" in n]


def test_a_unit_that_turns_reuse_off_on_both_paths_gets_no_note(tmp_path):
    task = _task(
        tmp_path,
        options={"core": {"kv_cache_config": {"enable_block_reuse": False}}},
        runtime={"kv_cache_reuse": False},
        features=[],
    )
    assert preflight.check_kv_cache_configs(task, _HYBRID) == []
    # Off on the quickstart path alone: accuracy still asks for reuse.
    task["accuracy"]["llm_api_options"] = {
        "core": str(_options(tmp_path, "core", {"kv_cache_config": {"dtype": "fp8"}}))
    }
    [note] = preflight.check_kv_cache_configs(task, _HYBRID)
    assert note.startswith("block reuse: core:")
    flagged = _task(tmp_path, runtime={"extra_flags": ["--disable_kv_cache_reuse"]}, features=[])
    flagged["accuracy"]["llm_api_options"] = {
        "core": str(_options(tmp_path, "off", {"kv_cache_config": {"enable_block_reuse": False}}))
    }
    assert preflight.check_kv_cache_configs(flagged, _HYBRID) == []


def test_run_preflight_reports_the_note(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_HYBRID), encoding="utf-8")
    task = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path),
        "accuracy": {"llm_api_options": str(_options(tmp_path, "run", {"max_num_tokens": 512}))},
        "environment": {
            "kind": "local",
            "python": "/py",
            "env": {"TMPDIR": "/scratch/mv2flow-tmp", "CLAUDE_CODE_TMPDIR": "/scratch/mv2flow-tmp"},
        },
    }

    def runner(argv, env=None, timeout=0):  # pragma: no cover - must not run
        raise AssertionError(f"probed the host: {argv}")

    result = preflight.run_preflight(task, runner=runner, skip_device_claim=True)
    assert any(n.startswith("block reuse: core:") for n in result.notes)
    assert "block reuse: core:" in preflight.format_preflight_report(result)


def test_a_witness_whose_leg_turned_block_reuse_off_fails(tmp_path):
    """Equal continuations prove nothing about reuse when an engine ran without it."""
    block = {
        "contrast_env": {"MV2FLOW_DISAGG_LEG": "B"},
        "contrast_command": "/run/disagg_leg_b.py",
        "min_generated_tokens": 1,
    }
    task = _task(tmp_path, options={"disagg": {"kv_cache_config": _UNIT_KV}})
    task["accuracy"]["witness"] = {"disagg": block}
    out = "[0] Prompt: 'p'..., Generated text: ' A'...\n[0] Witness token ids: [1, 2]\n"
    warning = (
        "[W] Disabling KV cache block reuse for the hybrid Mamba model because no Mamba "
        "state snapshot policy is configured.\n"
    )
    legs: list[str] = []

    def run(command, env=None, timeout=0):
        legs.append(command)
        # Leg B's engines ran without the snapshot policy.
        stdout = _ablated(command) or out
        return CommandResult(returncode=0, stdout=stdout, stderr=warning if len(legs) == 2 else "")

    result = witness.run_declared_witness(
        RunEnvironment(kind="local", python="/py", world_size=1),
        unit="disagg",
        member="tp1",
        task_data=task,
        runner=run,
        workdir=tmp_path / "w",
    )
    assert not result.passed
    assert result.detail.startswith("a witness leg: ") and "mamba_state_config" in result.detail
