# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The spec's deployments: stated natively, or expanded from the shorthand keys.

The first part is the design's own cases (FINAL-v3 sections 3, 5 and 10, DECISIONS), written
against the spec alone; the second is what the delivered specs, the messages a user sees
and the derivation come to.
"""

from __future__ import annotations

import copy
import json
import shlex
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.spec import advice, deployments, derive, task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

from ._helpers import MOVES, _workflow


@pytest.fixture
def spec(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"architectures": ["ToyForCausalLM"]}))
    tree = tmp_path / "tree"
    tree.mkdir()
    return {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": str(tree),
        "arch": "sm_103",
        "agents": {"defaults": {"backend": "claude-code", "model": "test-model"}},
        "environment": {"python": "python3"},
    }


def load(tmp_path, spec, notices=None):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return load_and_validate_task_yaml(path, notices)


def resolve(tmp_path, spec, notices=None):
    return deployments.resolve(load(tmp_path, spec, notices))


def native(spec, **extra):
    return {**spec, "deployments": [{"name": "first", "options": {}, **extra}]}


def shorthand(spec, **extra):
    accuracy = {
        "model_key": "org/toy",
        "benchmark": "gsm8k",
        "reference_yaml": "references.yaml",
    }
    accuracy.update(extra.pop("accuracy", {}))
    return {**spec, "parallel": "tp1", "accuracy": accuracy, **extra}


def test_native_aggregate_preserves_all_declared_fields(tmp_path, spec):
    """D1/D6; v3 §§3,10: arbitrary native options and multiple benchmarks."""
    options = {
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 2,
        "kv_cache_config": {"dtype": "nvfp4"},
        "future_native_key": 17,
    }
    (tmp_path / "engine.yaml").write_text(yaml.safe_dump(options))
    value = native(
        spec,
        options="engine.yaml",
        launch="launcher --nodes 2",
        stop="stop-owned",
        env={"UCX_TLS": "tcp,self", "OMP_NUM_THREADS": 2},
        timeout=5400,
        inputs=["text", "image"],
        tests=["tests/test_native.py::test_acceptance"],
        benchmarks=[
            {"name": "gsm8k"},
            {
                "name": "aa_lcr",
                "anchor": 0.62,
                "source": "measured baseline",
                "env": {"JUDGE": "local"},
                "args": ["--limit", "8"],
            },
        ],
    )
    [dep] = resolve(tmp_path, value)
    assert dep.core and dep.kind == "aggregated"
    assert dep.server.options == options
    assert dep.server.options_path == str(tmp_path / "engine.yaml")
    assert dep.server.launch == "launcher --nodes 2"
    assert dep.server.stop == "stop-owned" and dep.timeout == 5400
    assert dep.env == {"UCX_TLS": "tcp,self", "OMP_NUM_THREADS": "2"}
    assert dep.inputs == ("text", "image")
    assert dep.tests == ("tests/test_native.py::test_acceptance",)
    assert [(b.name, b.anchor) for b in dep.benchmarks] == [
        ("gsm8k", None),
        ("aa_lcr", 0.62),
    ]
    assert dep.benchmarks[1].source == "measured baseline"
    assert dep.benchmarks[1].env == {"JUDGE": "local"}


def test_disaggregated_sides_are_independent_and_relative(tmp_path, spec):
    """D1: asymmetric attention-DP context and TEP generation remain distinct."""
    ctx = {"tensor_parallel_size": 4, "enable_attention_dp": True, **MOVES}
    gen = {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2, **MOVES}
    value = native(spec)
    value["deployments"].append(
        {
            "name": "pd",
            "ctx": {"options": ctx, "launch": "ctx-launch"},
            "gen": {"options": gen, "launch": "gen-launch"},
            "relative_to": "first",
            "benchmarks": [{"name": "gsm8k"}],
            "env": {"UCX_TLS": "tcp,self"},
            "timeout": 100,
        }
    )
    first, pd = resolve(tmp_path, value)
    assert first.core and not pd.core and pd.relative_to == "first"
    assert pd.ctx.options == ctx and pd.gen.options == gen
    assert pd.ctx.launch == "ctx-launch" and pd.gen.launch == "gen-launch"
    assert pd.env == {"UCX_TLS": "tcp,self"} and pd.timeout == 100


def test_disaggregated_stop_is_per_server(tmp_path, spec):
    """DECISIONS #5: stop sits beside launch on each server."""
    value = {
        **spec,
        "deployments": [
            {
                "name": "pd",
                "ctx": {"options": MOVES, "launch": "ctx-launch", "stop": "ctx-stop"},
                "gen": {"options": MOVES, "launch": "gen-launch", "stop": "gen-stop"},
            }
        ],
    }
    [pd] = resolve(tmp_path, value)
    assert pd.ctx.stop == "ctx-stop" and pd.gen.stop == "gen-stop"


def test_first_disaggregated_deployment_is_core(tmp_path, spec):
    value = {
        **spec,
        "deployments": [
            {"name": "pd", "ctx": {"options": MOVES}, "gen": {"options": MOVES}},
            {"name": "later", "options": {}},
        ],
    }
    found = resolve(tmp_path, value)
    assert [d.core for d in found] == [True, False]


def test_native_benchmark_arguments_use_shell_quoting(tmp_path, spec):
    """DECISIONS #6: a shell string and equivalent argv are identical."""
    args = '--stop "END OF ANSWER" --path "/data/a b.json" --empty ""'
    [dep] = resolve(tmp_path, native(spec, benchmarks=[{"name": "gsm8k", "args": args}]))
    assert dep.benchmarks[0].args == tuple(shlex.split(args))


def test_accuracy_model_key_supplies_default_gsm8k(tmp_path, spec):
    """DECISIONS #7: model_key-only accuracy defaults to gsm8k, not boot-only."""
    value = native(spec)
    value["accuracy"] = {"model_key": "org/toy"}
    [dep] = resolve(tmp_path, value)
    assert [b.name for b in dep.benchmarks] == ["gsm8k"]


def test_native_accepts_accuracy_defaults_and_per_benchmark_override(tmp_path, spec):
    """DECISIONS #7: top-level defaults and benchmark entries coexist."""
    value = native(spec, benchmarks=[{"name": "aa_lcr", "anchor": 0.62, "source": "measured"}])
    value["accuracy"] = {
        "model_key": "org/toy",
        "benchmark": "gsm8k",
        "reference_yaml": "references.yaml",
        "discriminators": {"quant_algo": "NVFP4"},
        "dataset_path": "/offline/data",
    }
    data = load(tmp_path, value)
    [dep] = deployments.resolve(data)
    assert data["accuracy"]["benchmark"] == "gsm8k"
    assert [b.name for b in dep.benchmarks] == ["aa_lcr"]
    assert dep.benchmarks[0].anchor == 0.62


def test_benchmark_without_reference_is_allowed_and_unjudged(tmp_path, spec):
    [dep] = resolve(tmp_path, native(spec, benchmarks=[{"name": "aa_lcr"}]))
    assert dep.benchmarks[0].anchor is None
    assert any(
        "no accuracy anchor" in line
        for line in advice.spec_warnings(
            load(tmp_path, native(spec, benchmarks=[{"name": "aa_lcr"}]))
        )
    )


def test_no_accuracy_is_boot_only_for_native_spec(tmp_path, spec):
    [dep] = resolve(tmp_path, native(spec))
    assert dep.benchmarks == ()


def test_parallel_tp1_shorthand_preserves_known_engine_defaults(tmp_path, spec):
    value = shorthand(
        spec,
        runtime={
            "cuda_graph": True,
            "chunked_prefill": False,
            "kv_cache_reuse": False,
            "extra_flags": ["--max_batch_size=64", "--max_seq_len=5120"],
        },
    )
    [dep] = resolve(tmp_path, value)
    options = dep.server.options
    assert dep.core and options["tensor_parallel_size"] == 1
    assert options["pipeline_parallel_size"] == options["moe_expert_parallel_size"] == 1
    assert options["max_batch_size"] == 64 and options["max_seq_len"] == 5120
    assert options["max_num_tokens"] == 8192
    assert options["kv_cache_config"]["enable_block_reuse"] is False
    assert options["kv_cache_config"]["free_gpu_memory_fraction"] == 0.9
    assert options["enable_chunked_prefill"] is False
    assert options["disable_overlap_scheduler"] is False
    assert options["cuda_graph_config"] == {}


def test_members_preserve_order_and_first_is_core(tmp_path, spec):
    value = shorthand(spec, parallel="unified", members=["dep4", "tp1", "tep2"])
    found = resolve(tmp_path, value)
    assert [d.server.options["tensor_parallel_size"] for d in found] == [4, 1, 2]
    assert [d.core for d in found] == [True, False, False]
    assert found[0].server.options["enable_attention_dp"] is True
    assert found[2].server.options["moe_expert_parallel_size"] == 2


def test_dep_batch_division_is_once_and_native_is_not_redivided(tmp_path, spec):
    options = {
        "max_batch_size": 128,
        "cuda_graph_config": {
            "batch_sizes": [1, 4, 32, 64, 128],
            "max_batch_size": 128,
        },
    }
    (tmp_path / "core.yaml").write_text(yaml.safe_dump(options))
    value = shorthand(
        spec,
        parallel="dep4",
        accuracy={"model_key": "org/toy", "llm_api_options": "core.yaml"},
    )
    [expanded] = resolve(tmp_path, value)
    assert expanded.server.options["max_batch_size"] == 32
    assert expanded.server.options["cuda_graph_config"]["max_batch_size"] == 32
    [stated] = resolve(tmp_path, native(spec, options=expanded.server.options))
    assert stated.server.options == expanded.server.options


def test_kv_variant_overlay_isolated_and_relative(tmp_path, spec):
    value = shorthand(
        spec,
        variants={
            "kv_fp8": {
                "gates": ["accuracy"],
                "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
            }
        },
    )
    core, variant = resolve(tmp_path, value)
    assert core.server.options["kv_cache_config"].get("dtype") is None
    assert variant.server.options["kv_cache_config"]["dtype"] == "fp8"
    assert variant.relative_to == core.name
    assert [b.name for b in variant.benchmarks] == ["gsm8k"]


def test_feature_expands_with_its_native_options(tmp_path, spec):
    (tmp_path / "mtp.yaml").write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}})
    )
    value = shorthand(
        spec,
        features=["mtp"],
        accuracy={"model_key": "org/toy", "llm_api_options": {"mtp": "mtp.yaml"}},
    )
    core, feature = resolve(tmp_path, value)
    assert core.server.options["speculative_config"] is None
    assert feature.server.options["speculative_config"]["max_draft_len"] == 3
    assert feature.server.options["speculative_config"]["decoding_type"] == "MTP"
    assert not feature.core


def test_a_variant_and_a_disaggregated_request_are_one_deployment_each_on_the_first_member(
    tmp_path, spec
):
    """The ruled table: nothing of the old contract's choice of member is carried."""
    fp8 = {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}
    value = shorthand(
        spec,
        parallel="unified",
        members=["tp1", "tep4", "dep4"],
        features=["disagg"],
        accuracy={"benchmark_members": "dep4", "disagg": {"test": "tests/test_pd.py::test_hop"}},
        variants={"kv_fp8": {**fp8, "members": ["tp1", "dep4"], "gates": ["accuracy"]}},
    )
    notices = []
    found = resolve(tmp_path, value, notices)
    assert [d.name for d in found] == ["tp1", "tep4", "dep4", "disagg", "kv_fp8"]
    by_name = {d.name: d for d in found}
    # One variant, on the first member, judged against it; what chose members before is ignored.
    assert by_name["kv_fp8"].server.options["tensor_parallel_size"] == 1
    assert all(d.relative_to == "tp1" for d in found[1:]) and found[0].relative_to is None
    # Each side is core's options, and says how the cache moves, as the tree's own test does.
    moved = {"disable_overlap_scheduler": True, "cache_transceiver_config": {"backend": "DEFAULT"}}
    assert found[0].server.options["disable_overlap_scheduler"] is False
    assert (
        by_name["disagg"].ctx.options
        == by_name["disagg"].gen.options
        == {**found[0].server.options, **moved}
    )
    assert by_name["disagg"].tests == ("tests/test_pd.py::test_hop",)
    assert all([b.name for b in d.benchmarks] == ["gsm8k"] for d in found)
    assert sorted(n.split("'")[1] for n in notices) == [
        "accuracy.benchmark_members",
        "variants.kv_fp8.gates",
        "variants.kv_fp8.members",
    ]
    # The spec names the member a disaggregated deployment is served at.
    value["members"] = {"default": ["tp1", "dep4"], "disagg": ["dep4"]}
    pd = next(d for d in resolve(tmp_path, value) if d.kind == "disaggregated")
    assert pd.ctx.options["tensor_parallel_size"] == pd.gen.options["tensor_parallel_size"] == 4
    assert pd.ctx.options["enable_attention_dp"] is True
    # A backend the spec's own options name stands.
    named = {"cache_transceiver_config": {"backend": "NIXL"}}
    value["accuracy"]["llm_api_options"] = {"default": named}
    pd = next(d for d in resolve(tmp_path, value) if d.kind == "disaggregated")
    assert pd.gen.options["cache_transceiver_config"] == {"backend": "NIXL"}


def test_a_stated_disaggregated_side_that_cannot_move_the_cache_is_one_line_at_load(tmp_path, spec):
    sides = {"ctx": {"options": {"cache_transceiver_config": {}}}, "gen": {"options": MOVES}}
    with pytest.raises(TaskSchemaError) as raised:
        load(tmp_path, {**spec, "deployments": [{"name": "pd", **sides}]})
    lines = [line for line in str(raised.value).splitlines() if "cache_transceiver" in line]
    assert lines == [
        "  - missing deployments.pd.ctx options cache_transceiver_config: a disaggregated "
        "server refuses its first request without a backend to move the cache over; fix: "
        "`cache_transceiver_config: {backend: DEFAULT}`"
    ]
    # An aggregated deployment moves no cache and owes no such key.
    load(tmp_path, {**spec, "deployments": [{"name": "tp1", "options": {}}]})


def test_accuracy_specific_engine_limits_survive_shorthand(tmp_path, spec):
    """D2/DECISIONS #11: legacy accuracy limits are measured engine inputs."""
    value = shorthand(
        spec,
        accuracy={"model_key": "org/toy", "max_batch_size": 64, "max_seq_len": 4096},
    )
    [dep] = resolve(tmp_path, value)
    assert dep.server.options["max_batch_size"] == 64
    assert dep.server.options["max_seq_len"] == 4096


def test_normalization_roundtrip_is_stable(tmp_path, spec):
    value = shorthand(
        spec,
        parallel="unified",
        members=["tp2", "dep4"],
        variants={
            "kv_fp8": {
                "gates": ["accuracy"],
                "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
            }
        },
    )
    first = load(tmp_path, value)
    before = [d.as_dict() for d in deployments.resolve(first)]
    second = load(tmp_path, first)
    assert [d.as_dict() for d in deployments.resolve(second)] == before


def test_options_files_are_written_without_mutation(tmp_path, spec):
    [dep] = resolve(
        tmp_path,
        native(spec, options={"tensor_parallel_size": 2, "future_key": {"nested": 9}}),
    )
    before = copy.deepcopy(dep.server.options)
    paths = deployments.write_options(dep, tmp_path / "engines")
    assert yaml.safe_load(paths["server"].read_text()) == before
    assert dep.server.options == before


@pytest.mark.parametrize(
    "change",
    [
        {"deployments": []},
        {"deployments": [{"name": "only_ctx", "ctx": {"options": {}}}]},
        {
            "deployments": [
                {
                    "name": "mixed",
                    "options": {},
                    "ctx": {"options": {}},
                    "gen": {"options": {}},
                }
            ]
        },
        {
            "deployments": [
                {"name": "duplicate", "options": {}},
                {"name": "duplicate", "options": {}},
            ]
        },
        {"deployments": [{"name": "badref", "options": {}, "relative_to": "missing"}]},
        {"deployments": [{"name": "first", "options": {}}, {"name": "core", "options": {}}]},
        # The run's own tasks are called `survey`, `config` and `fix-<n>`.
        {"deployments": [{"name": "first", "options": {}}, {"name": "survey", "options": {}}]},
        {"deployments": [{"name": "config", "options": {}}]},
        {"deployments": [{"name": "first", "options": {}}, {"name": "fix", "options": {}}]},
        {
            "deployments": [
                {
                    "name": "badanchor",
                    "options": {},
                    "benchmarks": [{"name": "gsm8k", "anchor": 80}],
                }
            ]
        },
        {"deployments": [{"name": "badopts", "options": "absent.yaml"}]},
        {"deployments": [{"name": "typo", "options": {}, "benchmakrs": []}]},
    ],
)
def test_invalid_native_structure_is_refused(tmp_path, spec, change):
    with pytest.raises(TaskSchemaError):
        load(tmp_path, {**spec, **change})


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_nonfinite_anchor_is_refused(tmp_path, spec, score):
    value = native(
        spec,
        benchmarks=[{"name": "gsm8k", "anchor": score, "source": "not a measurement"}],
    )
    with pytest.raises(TaskSchemaError):
        load(tmp_path, value)


def test_bad_input_type_is_schema_error_not_crash(tmp_path, spec):
    with pytest.raises(TaskSchemaError):
        load(tmp_path, native(spec, inputs=["text", {"image": "file"}]))


def test_only_dropped_accuracy_keys_are_noticed(tmp_path, spec):
    notices = []
    value = shorthand(
        spec,
        accuracy={
            "model_key": "org/toy",
            "envelope": {"enabled": True},
            "reference_logits": {"path": "old.json"},
        },
    )
    data = load(tmp_path, value, notices)
    assert "envelope" not in data["accuracy"] and "reference_logits" not in data["accuracy"]
    assert len(notices) == 2 and all("ignored" in n for n in notices)
    assert all(any(key in n for n in notices) for key in ("envelope", "reference_logits"))
    repeated = []
    load(tmp_path, data, repeated)
    assert repeated == []


def test_unknown_key_is_not_silently_ignored(tmp_path, spec):
    value = shorthand(spec)
    value["accuracy"]["refernece_logits"] = {}
    with pytest.raises(TaskSchemaError):
        load(tmp_path, value)


def test_policy_keys_of_the_old_contract_are_ignored_by_one_table_with_a_notice_each(
    tmp_path, spec
):
    notices = []
    value = shorthand(
        spec,
        profile="smoke",
        coverage={"accept": ["kv_fp8@tp1"]},
        accuracy={"floor": {"k": 3}, "tolerance": 5.0, "acceptance": {"min": 2}},
    )
    data = load(tmp_path, value, notices)
    assert "profile" not in data and "coverage" not in data
    assert sorted(n.split("'")[1] for n in notices) == [
        "accuracy.acceptance",
        "accuracy.floor",
        "accuracy.tolerance",
        "coverage",
        "profile",
    ]
    assert all(" is ignored: " in n and n.endswith("; drop the key") for n in notices)
    assert set(task_schema.IGNORED_KEYS) >= {"profile", "coverage", "accuracy.envelope"}


def test_three_warnings_are_advice_not_validation_failures(tmp_path, spec):
    checkpoint = Path(spec["checkpoint_path"])
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 300 * 2**30}})
    )
    spec["environment"]["gpu_memory_gib"] = 80
    value = native(
        spec,
        options={"tensor_parallel_size": 1, "cuda_graph_config": None},
        benchmarks=[{"name": "gsm8k"}],
    )
    data = load(tmp_path, value)
    warnings = advice.spec_warnings(data)
    assert len(warnings) == 3
    assert any("too narrow" in w for w in warnings)
    assert any("no accuracy anchor" in w for w in warnings)
    assert any("without CUDA graphs" in w for w in warnings)


def test_missing_matching_anchor_is_warned_even_if_other_format_has_row(tmp_path, spec):
    """P6/D6: a BF16 row is not an NVFP4 measurement."""
    reference = Path(spec["trtllm_repo_path"]) / "references.yaml"
    reference.write_text("org/toy:\n  - accuracy: 90.0\n")
    value = shorthand(spec, accuracy={"discriminators": {"quant_algo": "NVFP4"}})
    warnings = advice.spec_warnings(load(tmp_path, value))
    assert any("no accuracy anchor" in line for line in warnings)


def test_warning_resolves_per_unit_model_key_without_crashing(tmp_path, spec):
    """Existing per-unit shorthand stays loadable and warnings cannot terminate preflight."""
    reference = Path(spec["trtllm_repo_path"]) / "references.yaml"
    reference.write_text("org/toy:\n  - accuracy: 41.77\n")
    value = shorthand(spec, accuracy={"model_key": {"default": "org/toy"}})
    warnings = advice.spec_warnings(load(tmp_path, value))
    assert not any("no accuracy anchor" in line for line in warnings)


# ---------------------------------------------------------------- the delivered specs

DISAGG_NODE = (
    "tests/integration/defs/accuracy/test_disaggregated_serving.py::TestQwen3::test_auto_dtype"
)


def _dirs(tmp_path) -> tuple[str, str]:
    for name in ("ckpt", "repo"):
        (tmp_path / name).mkdir(exist_ok=True)
    return str(tmp_path / "ckpt"), str(tmp_path / "repo")


def _load(tmp_path, spec: dict) -> dict:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return load_and_validate_task_yaml(path)


def _delivered(tmp_path, *, parallel: str, model_key: str, agent: str, disagg: bool) -> dict:
    """A spec as the weak-model GPU runs wrote it: derived, then the operator's keys."""
    checkpoint, repo = _dirs(tmp_path)
    options = tmp_path / "llm_api" / "core.yaml"
    options.parent.mkdir(exist_ok=True)
    options.write_text(
        yaml.safe_dump(
            {
                "kv_cache_config": {"enable_block_reuse": False, "host_cache_size": 8589934592},
                "enable_chunked_prefill": False,
                "cuda_graph_config": {"enable_padding": True, "max_batch_size": 64},
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "checkpoint_path": checkpoint,
        "arch": "sm_103",
        "trtllm_repo_path": repo,
        "parallel": parallel,
        "features": ["disagg"] if disagg else [],
        "scope": "full",
        "runtime": {
            "cuda_graph": True,
            "overlap_scheduler": True,
            "chunked_prefill": False,
            "kv_cache_reuse": False,
            "extra_flags": ["--max_seq_len=5120", "--max_batch_size=64", "--kv_cache_fraction=0.3"],
        },
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": "/var/tmp/mv2w", "CLAUDE_CODE_TMPDIR": "/var/tmp/mv2w-cc"},
            "gpus_per_node": 4 if disagg else 1,
            "gpu_memory_gib": 279,
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": model_key,
            "max_seq_len": 5120,
            "max_batch_size": 64,
            "boot": {
                "prompts": [
                    {"prompt": "The capital of France is", "keywords_any": ["Paris"]},
                    "In one sentence, photosynthesis is",
                ],
                "max_tokens": 32,
            },
            "llm_api_options": {"core": str(options)},
        },
        "variants": {
            "kv_fp8": {
                "unit": "core",
                "gates": ["accuracy"],
                "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
                "note": "KV cache stored as fp8, judged against the unit's own score",
            }
        },
        "agents": {
            "defaults": {"backend": "claude-code", "model": agent, "reasoning_effort": "max"}
        },
        "commit_policy": {"base_ref": "weak-base"},
        "notes": "Architecture: a dense decoder. Prior art in this tree worth surveying: none.",
    }
    if disagg:
        spec["accuracy"]["disagg"] = {"test": DISAGG_NODE, "env": {"LLM_MODELS_ROOT": "/models"}}
    return spec


def test_a_delivered_spec_is_its_member_its_disaggregated_deployment_and_its_kv_variant(tmp_path):
    spec = _load(
        tmp_path,
        _delivered(tmp_path, parallel="tp2", model_key="Qwen/Qwen3-0.6B", agent="m", disagg=True),
    )
    core, pd, kv = deployments.resolve(spec)
    assert [d.name for d in (core, pd, kv)] == ["tp2", "disagg", "kv_fp8"]
    options = core.server.options
    assert core.core and "model" not in options and options["tensor_parallel_size"] == 2
    assert (options["max_seq_len"], options["max_batch_size"]) == (5120, 64)
    assert options["kv_cache_config"] == {
        "enable_block_reuse": False,
        "host_cache_size": 8589934592,
        "free_gpu_memory_fraction": 0.3,
    }
    assert options["cuda_graph_config"] == {"enable_padding": True, "max_batch_size": 64}
    # Disaggregation is a deployment of two servers on core's options, with the spec's test.
    assert pd.kind == deployments.DISAGGREGATED and list(pd.servers) == ["ctx", "gen"]
    # Each side says how the cache moves and runs without the overlap scheduler.
    assert (
        pd.ctx.options
        == pd.gen.options
        == {
            **options,
            "disable_overlap_scheduler": True,
            "cache_transceiver_config": {"backend": "DEFAULT"},
        }
    )
    assert pd.tests == (DISAGG_NODE,) and pd.env == {"LLM_MODELS_ROOT": "/models"}
    # The variant is core's options with its own setting, judged against core's score.
    assert kv.relative_to == "tp2" and kv.server.options["kv_cache_config"]["dtype"] == "fp8"
    assert {k: v for k, v in kv.server.options.items() if k != "kv_cache_config"} == {
        k: v for k, v in options.items() if k != "kv_cache_config"
    }
    assert deployments.named(spec).name == deployments.named(spec, "core").name == "tp2"
    with pytest.raises(KeyError, match="'tp4' names no deployment of this spec"):
        deployments.named(spec, "tp4")


def _native(tmp_path) -> dict:
    """Kimi K3 as the design states it: one engine on four nodes, then context and generation."""
    checkpoint, repo = _dirs(tmp_path)
    engines = tmp_path / "engines"
    engines.mkdir(exist_ok=True)
    dep16 = {
        "tensor_parallel_size": 16,
        "moe_expert_parallel_size": 16,
        "enable_attention_dp": True,
        "kv_cache_config": {"dtype": "nvfp4"},
    }
    (engines / "dep16.yaml").write_text(yaml.safe_dump(dep16), encoding="utf-8")
    (engines / "ctx.yaml").write_text(yaml.safe_dump({**dep16, **MOVES}), encoding="utf-8")
    (engines / "gen.yaml").write_text(
        yaml.safe_dump({**dep16, **MOVES, "enable_attention_dp": False}), encoding="utf-8"
    )
    return {
        "checkpoint_path": checkpoint,
        "trtllm_repo_path": repo,
        "arch": "sm_103",
        "accuracy": {"model_key": "moonshotai/Kimi-K3"},
        "agents": {"defaults": {"backend": "claude-code", "model": "claude-test-model"}},
        "environment": {"python": "/opt/venv/bin/python", "launch": "srun -N1 --gpus=1"},
        "deployments": [
            {
                "name": "dep16",
                "options": "engines/dep16.yaml",
                "launch": "srun -N4 --ntasks=16 --mpi=pmix",
                "timeout": 5400,
                "inputs": ["text", "image"],
                "benchmarks": [
                    {"name": "gsm8k"},
                    {"name": "mmmu", "args": "--post_process_fn 'strip reasoning'"},
                    {
                        "name": "aa_lcr",
                        "anchor": 0.62,
                        "source": "trtllm-eval aa_lcr at DEP16 on 16 GB300, 2026-11-02",
                        "env": {"NS_JUDGE_MODEL": "judge"},
                    },
                ],
            },
            {
                "name": "pd",
                "env": {"UCX_TLS": "tcp,self,sm,cuda_copy,cuda_ipc"},
                "ctx": {"options": "engines/ctx.yaml", "launch": "srun --nodelist=n[1-4]"},
                "gen": {
                    "options": "engines/gen.yaml",
                    "launch": "srun --nodelist=n[5-8]",
                    "stop": "scancel --name gen",
                },
                "relative_to": "dep16",
                "tests": [DISAGG_NODE],
            },
        ],
    }


def test_what_the_table_cannot_state_is_one_line_and_a_missing_key_says_its_fix(tmp_path, spec):
    def refused(value) -> str:
        with pytest.raises(TaskSchemaError) as raised:
            load(tmp_path, value)
        return str(raised.value)

    # A deployment is not called what a task of the run is called.
    fp8 = {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}
    for taken in ("survey", "config", "fix"):
        said = f"'variants.{taken}': a variant needs a name of its own"
        assert said in refused(shorthand(spec, variants={taken: fp8}))
        stated = {
            **spec,
            "deployments": [{"name": "tp1", "options": {}}, {"name": taken, "options": {}}],
        }
        assert f"`{taken}` is a task of the run; name the deployment otherwise" in refused(stated)
    # A boot prompt rendered by hand cannot go through an endpoint: refused, never replaced.
    by_hand = {"prompt": "<image>What is it", "images": ["a.png"], "render": "raw"}
    raw = {**spec, "accuracy": {**(spec.get("accuracy") or {}), "boot": {"prompts": [by_hand]}}}
    said = "'accuracy.boot.prompts[0]': `render: raw` cannot be sent: a boot asks the deployment's"
    assert said in refused(raw)
    by_unit = shorthand(spec, runtime={"default": {"cuda_graph": True}, "mtp": {}})
    assert "state deployments: 'runtime' differs by unit" in refused(by_unit)
    per_feature = shorthand(spec, parallel="unified", members={"default": ["tp1"], "mtp": ["tep4"]})
    assert "state deployments: 'members' lists members for ['mtp']" in refused(per_feature)
    flag = shorthand(spec, runtime={"extra_flags": ["--some_new_flag=1"]})
    assert "state deployments: runtime.extra_flags has --some_new_flag" in refused(flag)
    # How a check samples is its protocol, not a setting of the spec.
    sampled = shorthand(spec, runtime={"extra_flags": ["--temperature=0.7"]})
    assert "runtime.extra_flags has --temperature" in refused(sampled)
    twice = shorthand(
        spec,
        runtime={"extra_flags": ["--kv_cache_fraction=0.7"]},
        accuracy={
            "llm_api_options": {"core": {"kv_cache_config": {"free_gpu_memory_fraction": 0.5}}}
        },
    )
    assert "two values for `kv_cache_config.free_gpu_memory_fraction`" in refused(twice)
    both = {**native(spec), "features": ["mtp"]}
    assert "'features' beside 'deployments'" in refused(both)
    # A missing or wrong key is one line: what it is, and what to write.
    bare = {k: v for k, v in native(spec).items() if k not in ("arch", "checkpoint_path")}
    bare["environment"] = {}
    said = refused(bare)
    for line in (
        "missing arch: the GPU architecture, never inferred (sm_100 is B200, sm_103 is GB300); "
        "fix: `arch: sm_103`",
        "missing checkpoint_path: the HuggingFace checkpoint directory; fix: `checkpoint_path: ",
        "missing environment.python: the interpreter every Python command runs under; fix: ",
    ):
        assert line in said
    no_options = {**spec, "deployments": [{"name": "first"}]}
    assert "missing deployments.first.options: a native LLM-API options file; fix: " in refused(
        no_options
    )
    quoting = native(spec, benchmarks=[{"name": "gsm8k", "args": '--stop "unclosed'}])
    assert "args' cannot be split as a shell would: No closing quotation" in refused(quoting)
    # The smallest spec loads: one member and the model's key.
    smallest = {**spec, "parallel": "tp1", "accuracy": {"model_key": "org/toy"}}
    [only] = resolve(tmp_path, smallest)
    assert (only.name, [b.name for b in only.benchmarks]) == ("tp1", ["gsm8k"])


def _routing(repo, family: str, body: str) -> None:
    path = repo / "tensorrt_llm/_torch/_experimental/modeling_v2/models" / family / "routing.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_the_family_is_the_one_whose_routing_declares_the_architecture(tmp_path):
    repo = tmp_path / "repo"
    _routing(
        repo,
        "kimi_k3_vl",
        'from ..._router_index import routes\n\n\n@routes("KimiK3ForConditionalGeneration")\n'
        "def route(*args): ...\n",
    )
    _routing(
        repo,
        "minimax_m3",
        "from ... import _router_index as index\n\n\n"
        '@index.routes("MiniMaxM3SparseForConditionalGeneration", "MiniMaxM3ForCausalLM")\n'
        "def route(*args): ...\n",
    )
    _routing(repo, "broken", "def route(:\n")
    routed = task_schema.routed_family
    assert routed(repo, "KimiK3ForConditionalGeneration") == "kimi_k3_vl"
    assert routed(repo, "MiniMaxM3ForCausalLM") == "minimax_m3"
    assert routed(repo, "Qwen3ForCausalLM") is None
    # An architecture nothing routes yet is named after its class, whatever its head.
    assert task_schema.derive_family("Qwen3ForCausalLM") == "qwen3"
    assert task_schema.derive_family("Qwen4ExpForConditionalGeneration") == "qwen4_exp"
    assert task_schema.derive_family("GptOssForCausalLM") == "gpt_oss"

    def family(architecture, target=None):
        spec = {
            "trtllm_repo_path": str(repo),
            "checkpoint_path": "/models/Kimi-K3",
            "arch": "sm_103",
            "parallel": "dep16",
            "target": target or {},
        }
        return preflight.resolve_target_identity(spec, {"architectures": [architecture]})

    assert family("KimiK3ForConditionalGeneration")[0] == "kimi_k3_vl"
    assert family("KimiK3ForConditionalGeneration")[1].endswith(
        "models/kimi_k3_vl/kimi_k3__sm_103__dep16"
    )
    assert family("KimiK3ForConditionalGeneration", {"family": "mine"})[0] == "mine"
    assert family("NewModelForCausalLM")[0] == "new_model"


def test_a_run_journals_the_deployments_its_shorthand_comes_to(tmp_path, task_file):
    workflow = _workflow(tmp_path, dry_run=True)
    with workflow:
        assert workflow.run(task_file)
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "deployments: 1" in journal
    assert "- `tp1` (core): aggregated, 1 rank(s); judged by its boot, gsm8k" in journal


def test_a_dry_run_says_what_the_spec_comes_to_one_line_for_each_deployment(
    tmp_path, task_file, capsys
):
    workflow = _workflow(tmp_path, dry_run=True)
    with workflow:
        assert workflow.run(task_file)
    line = "- `tp1` (core): aggregated, 1 rank(s); judged by its boot, gsm8k (reference 85.6: "
    assert "deployments: 1" in (tmp_path / "ws" / "journal.md").read_text()
    assert line in (tmp_path / "ws" / "journal.md").read_text()
    assert line in capsys.readouterr().out.replace("\n", "")


def test_a_run_reads_a_spec_that_states_its_deployments(tmp_path):
    path = tmp_path / "task.yaml"
    stated = _native(tmp_path)
    stated["environment"]["env"] = {
        "TMPDIR": "/var/tmp/mv2w",
        "CLAUDE_CODE_TMPDIR": "/var/tmp/mv2w",
    }
    path.write_text(yaml.safe_dump(stated), encoding="utf-8")
    config = {"architectures": ["KimiK3"], "vision_config": {}}
    (tmp_path / "ckpt" / "config.json").write_text(json.dumps(config))
    workflow = _workflow(tmp_path, dry_run=True)
    with workflow:
        assert workflow.run(path)
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert (
        "- `dep16` (core): aggregated, 16 rank(s), attention DP, behind a launch prefix" in journal
    )
    assert "- `pd`: disaggregated, ctx 16 rank(s), attention DP" in journal
    assert "devices:     (none claimed here)" in journal
    assert [check.key for check in workflow._contract.checks] == [
        "dep16:boot",
        "dep16:accuracy[gsm8k]",
        "dep16:accuracy[mmmu]",
        "dep16:accuracy[aa_lcr]",
        "pd:boot",
        "pd:accuracy[gsm8k]",
        "pd:tests",
    ]


def test_intake_derive_writes_the_smallest_spec_and_prints_its_warnings(tmp_path, capsys):
    checkpoint, repo = _dirs(tmp_path)
    (tmp_path / "ckpt" / "config.json").write_text(
        json.dumps({"architectures": ["ToyForCausalLM"], "_name_or_path": "org/toy"})
    )
    out = tmp_path / "out"
    argv = ["derive", "--checkpoint", checkpoint, "--repo", repo, "--out", str(out)]
    assert derive.main([*argv, "--goal", "tp1 dep4 disagg@dep4 kv=bf16/fp8"]) == 0
    said = capsys.readouterr()
    # A goal is accepted and ignored, with one notice that says what to do instead.
    assert said.err.count("\n") == 1 and said.err.startswith("note: --goal is ignored: ")
    assert "by editing the spec's `deployments`" in said.err
    printed = said.out
    written = yaml.safe_load((out / derive.SPEC_FILENAME).read_text())
    assert "parallel" not in written and "members" not in written and "variants" not in written
    assert written["deployments"] == [{"name": "tp1", "options": "engines/tp1.yaml"}]
    assert sorted(p.name for p in (out / "engines").iterdir()) == ["tp1.yaml"]
    # The options are the engine's own, stated rather than left to its defaults.
    tp1 = yaml.safe_load((out / "engines" / "tp1.yaml").read_text())
    assert (tp1["max_seq_len"], tp1["max_batch_size"]) == (5120, 64)
    # What it left for the operator, one `open:` line each.
    opened = [line.strip() for line in printed.splitlines() if line.strip().startswith("open:")]
    assert [line.split(":")[1].strip() for line in opened] == [
        "arch",
        "agents.defaults.model",
        "environment",
    ]
    # Once those are answered, what was written loads as it is.
    written["environment"]["python"] = "/opt/venv/bin/python"
    written["agents"] = {"defaults": {"backend": "claude-code", "model": "m"}}
    written["arch"] = "sm_103"
    (out / derive.SPEC_FILENAME).write_text(yaml.safe_dump(written, sort_keys=False))
    loaded = load_and_validate_task_yaml(out / derive.SPEC_FILENAME)
    assert [d.name for d in deployments.resolve(loaded)] == ["tp1"]
    assert "warning: no accuracy anchor: `tp1` on `gsm8k` has no declared anchor" in printed
    assert derive.main(["--help"]) == 0
    assert derive.main(["lint", "--task", "x.yaml"]) == 2
    said = capsys.readouterr().err
    assert said.count("\n") == 1 and "`intake lint` is replaced by" in said and "--dry-run" in said
    assert derive.main(["other"]) == 2
    assert "usage: trtllm-modeling-v2 intake derive" in capsys.readouterr().err
