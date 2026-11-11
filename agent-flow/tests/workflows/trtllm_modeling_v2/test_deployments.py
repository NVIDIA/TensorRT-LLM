# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spec's deployments: stated as native options, or expanded from the shorthand keys."""

from __future__ import annotations

import copy

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import check_config
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments, task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

from ._helpers import _workflow

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


def _check_options(spec: dict, unit: str, member: str) -> dict:
    """The options today's checks send the engine for ``unit`` at ``member``, less the model."""
    environment = resolve_environment(spec, devices=(), world_size=1)
    config = check_config.check_configuration(
        spec, environment, unit=unit, features=None, max_tokens=32, member=member
    )
    return {key: value for key, value in config["llm"].items() if key != "model"}


# ------------------------------------------------------------- the delivered specs


@pytest.mark.parametrize("agent", ["devin/kimi-k3-1-pro", "devin/deepseek-v4-1-flash-1017"])
@pytest.mark.parametrize("model_key", ["Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-1.7B"])
def test_a_delivered_tp1_spec_is_core_and_its_kv_variant(tmp_path, agent, model_key):
    spec = _load(
        tmp_path,
        _delivered(tmp_path, parallel="tp1", model_key=model_key, agent=agent, disagg=False),
    )
    core, kv = deployments.resolve(spec)
    assert (core.name, core.core, core.kind) == ("tp1", True, deployments.AGGREGATED)
    assert (kv.name, kv.core, kv.kind) == ("kv_fp8@tp1", False, deployments.AGGREGATED)
    options = core.server.options
    assert "model" not in options
    assert (options["tensor_parallel_size"], options["enable_attention_dp"]) == (1, False)
    assert (options["max_seq_len"], options["max_batch_size"]) == (5120, 64)
    assert options["kv_cache_config"] == {
        "enable_block_reuse": False,
        "host_cache_size": 8589934592,
        "free_gpu_memory_fraction": 0.3,
    }
    assert options["cuda_graph_config"] == {"enable_padding": True, "max_batch_size": 64}
    assert options["speculative_config"] is None
    # The variant is its unit's options with its own setting, judged against the unit's score.
    assert kv.server.options["kv_cache_config"]["dtype"] == "fp8"
    assert {k: v for k, v in kv.server.options.items() if k != "kv_cache_config"} == {
        k: v for k, v in options.items() if k != "kv_cache_config"
    }
    assert kv.relative_to == "tp1" and core.relative_to is None
    assert [b.name for b in core.benchmarks] == [b.name for b in kv.benchmarks] == ["gsm8k"]
    assert core.inputs == ("text",) and not core.tests


def test_a_delivered_tp2_spec_with_disagg_is_three_deployments(tmp_path):
    spec = _load(
        tmp_path,
        _delivered(
            tmp_path,
            parallel="tp2",
            model_key="Qwen/Qwen3-0.6B",
            agent="devin/kimi-k3-1-pro",
            disagg=True,
        ),
    )
    core, pd, kv = deployments.resolve(spec)
    assert [d.name for d in (core, pd, kv)] == ["tp2", "disagg@tp2", "kv_fp8@tp2"]
    assert core.core and core.server.options["tensor_parallel_size"] == 2
    # Disaggregation is a deployment of two servers, judged by the test the spec names.
    assert pd.kind == deployments.DISAGGREGATED and list(pd.servers) == ["ctx", "gen"]
    assert pd.ctx.options == pd.gen.options == core.server.options
    assert pd.tests == (DISAGG_NODE,) and not pd.benchmarks
    assert pd.env == {"LLM_MODELS_ROOT": "/models"}
    assert kv.relative_to == "tp2" and kv.server.options["kv_cache_config"]["dtype"] == "fp8"


def test_a_shorthand_deployment_carries_the_options_the_checks_send_today(tmp_path):
    spec = _load(
        tmp_path,
        _delivered(
            tmp_path,
            parallel="tp2",
            model_key="Qwen/Qwen3-0.6B",
            agent="devin/kimi-k3-1-pro",
            disagg=True,
        ),
    )
    core, _, kv = deployments.resolve(spec)
    assert core.server.options == _check_options(spec, "core", "tp2")
    # A variant's checks run under the spec `variant_spec` writes, its options file included.
    variant = check_config.variant_spec(spec, "kv_fp8", world_size=2, options_dir=tmp_path / "v")
    assert kv.server.options == _check_options(variant, "kv_fp8", "tp2")


# ------------------------------------------------------------------ shorthand, in general


def test_parallel_m_and_members_m_state_the_same_deployments(tmp_path):
    base = _delivered(
        tmp_path,
        parallel="tp1",
        model_key="Qwen/Qwen3-0.6B",
        agent="claude-test-model",
        disagg=False,
    )
    listed = copy.deepcopy(base)
    del listed["parallel"]
    listed["members"] = ["tp1"]
    one = deployments.resolve(_load(tmp_path, base))
    other = deployments.resolve(_load(tmp_path, listed))
    assert one == other and [d.name for d in one] == ["tp1", "kv_fp8@tp1"]


def test_units_members_and_variants_expand_in_the_spec_order(tmp_path):
    spec = _delivered(
        tmp_path, parallel="unified", model_key="org/model", agent="claude-test-model", disagg=False
    )
    spec["members"] = {"default": ["tep4", "dep4", "tp4"], "multimodal": ["tep4"]}
    spec["features"] = ["mtp", "multimodal"]
    spec["accuracy"]["benchmark_members"] = "dep4"
    spec["accuracy"]["benchmark_args"] = ["--apply_chat_template"]
    spec["variants"]["chunked"] = {
        "unit": "mtp",
        "members": ["tep4", "tp4"],
        "gates": ["boot"],
        "runtime": {"chunked_prefill": True},
        "env": {"MY_SWITCH": 1},
    }
    found = deployments.resolve(_load(tmp_path, spec))
    assert [d.name for d in found] == [
        "tep4", "dep4", "tp4",
        "mtp@tep4", "mtp@dep4", "mtp@tp4",
        "multimodal@tep4",
        "kv_fp8@dep4",
        "chunked@tep4", "chunked@tp4",
    ]  # fmt: skip
    by_name = {d.name: d for d in found}
    assert [d.name for d in found if d.core] == ["tep4"]  # the first member listed is core's
    # A member name is a topology, stated as the engine's own options.
    shapes = {
        name: (
            by_name[name].server.options["tensor_parallel_size"],
            by_name[name].server.options["moe_expert_parallel_size"],
            by_name[name].server.options["enable_attention_dp"],
        )
        for name in ("tp4", "tep4", "dep4")
    }
    assert shapes == {"tp4": (4, 1, False), "tep4": (4, 4, False), "dep4": (4, 4, True)}
    # A global batch is each attention-DP rank's share.
    assert by_name["dep4"].server.options["max_batch_size"] == 16
    assert by_name["tep4"].server.options["max_batch_size"] == 64
    # A unit carries the speculator of every feature up to it; core carries none.
    assert by_name["tep4"].server.options["speculative_config"] is None
    for name in ("mtp@tep4", "multimodal@tep4", "chunked@tep4"):
        assert by_name[name].server.options["speculative_config"]["decoding_type"] == "MTP"
    assert by_name["multimodal@tep4"].inputs == ("text", "image")
    assert by_name["mtp@tep4"].inputs == ("text",)
    # The benchmark runs where `benchmark_members` says, or at every member of a unit that
    # has not got that one; a boot-only variant has none.
    scored = [d.name for d in found if d.benchmarks]
    assert scored == ["dep4", "mtp@dep4", "multimodal@tep4", "kv_fp8@dep4"]
    assert by_name["dep4"].benchmarks[0].args == ("--apply_chat_template",)
    assert by_name["kv_fp8@dep4"].relative_to == "dep4"
    chunked = by_name["chunked@tp4"]
    assert chunked.server.options["enable_chunked_prefill"] is True
    assert chunked.env == {"MY_SWITCH": "1"} and chunked.relative_to is None


def test_shorthand_that_asks_two_values_of_one_setting_is_refused_when_expanded(tmp_path):
    spec = _delivered(
        tmp_path, parallel="tp1", model_key="org/model", agent="claude-test-model", disagg=False
    )
    spec["runtime"]["extra_flags"].append("--kv_cache_dtype=nvfp4")
    spec["variants"] = {}
    loaded = _load(tmp_path, spec)
    (tmp_path / "llm_api" / "core.yaml").write_text("kv_cache_config: {dtype: fp8}\n")
    with pytest.raises(ValueError, match="kv_cache_config.dtype: runtime.extra_flags asks 'nvfp4'"):
        deployments.resolve(loaded)


# ---------------------------------------------------------------- deployments, stated


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
    (engines / "ctx.yaml").write_text(yaml.safe_dump(dep16), encoding="utf-8")
    (engines / "gen.yaml").write_text(
        yaml.safe_dump({**dep16, "enable_attention_dp": False}), encoding="utf-8"
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
                    {"name": "mmmu", "args": "--post_process_fn strip_reasoning"},
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
                "gen": {"options": "engines/gen.yaml", "launch": "srun --nodelist=n[5-8]"},
                "relative_to": "dep16",
                "tests": [DISAGG_NODE],
            },
        ],
    }


def test_a_spec_states_its_deployments_as_native_options(tmp_path):
    spec = _load(tmp_path, _native(tmp_path))
    assert deployments.is_native(spec)
    assert spec["environment"] == {
        "python": "/opt/venv/bin/python",
        "launch": "srun -N1 --gpus=1",
        "kind": "local",
        "env": {},
    }
    core, pd = deployments.resolve(spec)
    assert (core.name, core.core, core.kind) == ("dep16", True, deployments.AGGREGATED)
    assert core.server.options_path == str(tmp_path / "engines" / "dep16.yaml")
    assert core.server.options["tensor_parallel_size"] == 16
    assert core.server.launch == "srun -N4 --ntasks=16 --mpi=pmix" and core.timeout == 5400
    assert core.inputs == ("text", "image")
    gsm8k, mmmu, aa_lcr = core.benchmarks
    assert (gsm8k.name, gsm8k.anchor, gsm8k.args) == ("gsm8k", None, ())
    assert mmmu.args == ("--post_process_fn", "strip_reasoning")
    assert (aa_lcr.anchor, aa_lcr.env) == (0.62, {"NS_JUDGE_MODEL": "judge"})
    assert aa_lcr.source.startswith("trtllm-eval aa_lcr")
    # Context and generation are two servers with their own options and prefixes.
    assert (pd.kind, pd.core, pd.relative_to) == (deployments.DISAGGREGATED, False, "dep16")
    assert pd.ctx.options["enable_attention_dp"] and not pd.gen.options["enable_attention_dp"]
    assert pd.gen.launch == "srun --nodelist=n[5-8]"
    assert pd.env == {"UCX_TLS": "tcp,self,sm,cuda_copy,cuda_ipc"} and pd.tests == (DISAGG_NODE,)
    # What it loads is what it states: the resolved spec loads again to the same deployments.
    assert deployments.resolve(_load(tmp_path, spec)) == (core, pd)
    assert [d.as_dict()["name"] for d in (core, pd)] == ["dep16", "pd"]


def test_a_small_model_on_one_host_states_one_inline_deployment_and_no_accuracy(tmp_path):
    checkpoint, repo = _dirs(tmp_path)
    spec = {
        "checkpoint_path": checkpoint,
        "trtllm_repo_path": repo,
        "arch": "sm_103",
        "environment": {"python": "/opt/venv/bin/python"},
        "deployments": [{"name": "tp1", "options": {"max_seq_len": 5120}}],
    }
    [only] = deployments.resolve(_load(tmp_path, spec))
    assert only.core and only.server.options == {"max_seq_len": 5120}
    assert (only.server.launch, only.server.options_path, only.benchmarks) == ("", "", ())


def test_every_problem_with_stated_deployments_is_batched(tmp_path):
    spec = _native(tmp_path)
    core, pd = spec["deployments"]
    core["options"] = "engines/absent.yaml"
    core["benchmarks"][2].pop("source")
    core["benchmarks"].append({"name": "gsm8k"})
    core["inputs"] = ["image"]
    core["relative_to"] = "pd"
    pd["name"] = "dep16"
    pd.pop("gen")
    pd["launch"] = "srun"
    pd["timeout"] = 0
    spec["deployments"].append({"name": "Bad Name", "options": {}, "lanch": "x"})
    spec["features"] = ["mtp"]
    spec["accuracy"]["benchmark"] = "mmlu"
    with pytest.raises(TaskSchemaError) as raised:
        _load(tmp_path, spec)
    message = str(raised.value)
    for expected in (
        "'deployments.dep16.options' does not exist",
        "'deployments.dep16.benchmarks[2].source' is required with an anchor",
        "'deployments.dep16.benchmarks[3]' repeats the benchmark 'gsm8k'",
        "'deployments.dep16.inputs' must be `text` and any of ['image']",
        "'deployments.dep16.relative_to' is 'pd', which names no deployment before this one",
        "'deployments.dep16' is named twice",
        "has `ctx` alone; disaggregated needs `ctx` and `gen`",
        "'deployments.dep16.launch' sits beside `ctx` and `gen`",
        "'deployments.dep16.timeout' must be an integer number of seconds >= 1",
        "'deployments[2].name' must be lowercase letters, digits and '_'",
        "unknown key 'deployments[2].lanch' — did you mean 'deployments[2].launch'?",
        "'features' beside 'deployments': these keys are shorthand for deployments",
        "'accuracy' carries ['benchmark'] beside 'deployments'",
    ):
        assert expected in message, expected


def test_options_are_written_where_a_server_reads_them(tmp_path):
    core, pd = deployments.resolve(_load(tmp_path, _native(tmp_path)))
    out = tmp_path / "ws" / "deployments"
    assert deployments.write_options(core, out) == {"server": out / "dep16.yaml"}
    assert yaml.safe_load((out / "dep16.yaml").read_text()) == dict(core.server.options)
    written = deployments.write_options(pd, out)
    assert written == {"ctx": out / "pd.ctx.yaml", "gen": out / "pd.gen.yaml"}
    assert yaml.safe_load(written["gen"].read_text())["enable_attention_dp"] is False


def test_a_run_on_stated_deployments_says_what_it_read_and_that_it_does_not_run_them_yet(tmp_path):
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(_native(tmp_path)), encoding="utf-8")
    workflow = _workflow(tmp_path)
    with workflow, pytest.raises(TaskSchemaError) as raised:
        workflow.run(path)
    message = str(raised.value)
    assert "states 2 deployment(s)" in message
    assert (
        "- `dep16` (core): aggregated, 16 rank(s), attention DP, behind a launch prefix" in message
    )
    assert "- `pd`: disaggregated, ctx 16 rank(s), attention DP" in message
    assert "validated here and not yet run" in message


# ------------------------------------------------------------------------- the family


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
