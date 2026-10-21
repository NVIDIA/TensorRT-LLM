# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the run-scoped declarations: runtime switches, scope, embargo.

Each of the three is a place where the obvious implementation is wrong in the
same way: it trusts that saying a thing is the same as its being true. A
runtime switch is not a flag (two of them are spelled as negations), an
embargo on a path that does not exist protects nothing, and a text-only
scope that nobody tells the weight loader about reads as a defect.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import build_trtllm_modeling_v2_prompts
from agent_flow.workflows.trtllm_modeling_v2.agents.prompts._common import (
    scope_block,
    source_embargo_block,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.gates import runtime_flags, speculation_flags
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    RUNTIME_SWITCHES,
    TaskSchemaError,
    load_and_validate_task_yaml,
)

# Never created. The TMPDIR check reads the declared value rather than
# stat-ing it, and pytest's own tmp_path lives under /tmp, which the check
# rejects by design.
_SCRATCH = "/scratch/mv2flow-run-tmp"


# ----------------------------------------------------------- runtime flags


def test_enabling_a_positive_switch_adds_its_flag():
    flags = runtime_flags({"cuda_graph": True, "chunked_prefill": True})
    assert "--use_cuda_graph" in flags
    assert "--enable_chunked_prefill" in flags


def test_wanting_a_negative_switch_on_emits_nothing():
    """The trap this function exists for.

    Overlap scheduling and block reuse are on by default in the engine, so
    the only flags that exist for them turn them *off*. Asked to "enable
    block reuse", the obvious move adds ``--disable_kv_cache_reuse`` and
    silently gates the opposite of what was requested.
    """
    flags = runtime_flags({"overlap_scheduler": True, "kv_cache_reuse": True})
    assert flags == []


def test_turning_a_negative_switch_off_emits_its_disable_flag():
    flags = runtime_flags({"overlap_scheduler": False, "kv_cache_reuse": False})
    assert flags == ["--disable_overlap_scheduler", "--disable_kv_cache_reuse"]


def test_an_absent_switch_takes_the_engine_default_not_false():
    """``{}`` must mean "the engine's defaults", not "everything off"."""
    assert runtime_flags({}) == []


def test_extra_flags_pass_through_in_order():
    flags = runtime_flags({"extra_flags": ["--mamba_ssm_cache_dtype=float32", "--x=1"]})
    assert flags == ["--mamba_ssm_cache_dtype=float32", "--x=1"]


@pytest.mark.parametrize(
    ("features", "expected_algo"),
    [(["mtp"], "MTP"), (["eagle3"], "Eagle3"), (["MTP"], "MTP"), (["dspark"], "DSPARK")],
)
def test_a_speculator_feature_becomes_spec_decode_flags(features, expected_algo):
    flags = speculation_flags(features)
    assert flags[0] == f"--spec_decode_algo={expected_algo}"
    assert flags[1] == "--spec_decode_max_draft_len=1"


def test_a_non_speculative_feature_produces_no_flags():
    """`host_offload` changes the target, not the command that runs it."""
    assert speculation_flags(["host_offload", "kv_quant"]) == []
    assert speculation_flags(None) == []


# --------------------------------------------------------- boot gate mode


def _boot_task(tmp_path):
    return {
        "checkpoint_path": str(tmp_path),
        "trtllm_repo_path": str(tmp_path),
        "parallel": "tp1",
        "runtime": {"cuda_graph": True},
        "features": [],
    }


def _capture_runner(sink):
    def run(command, env=None, timeout=0):
        sink.append(command)

        return CommandResult(returncode=0, stdout="Generated text: Paris")

    return run


def test_the_boot_gate_runs_under_require_by_default(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import run_boot_gate

    commands = []
    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/py"}}, world_size=1
    )
    result = run_boot_gate(
        environment, task_data=_boot_task(tmp_path), runner=_capture_runner(commands)
    )
    assert "TRTLLM_MODELING_V2=require" in commands[0]
    assert result.name == "boot"


def test_a_control_run_is_named_so_it_cannot_be_mistaken_for_the_gate(tmp_path):
    """Two rows both called "boot" would be a journal that lies.

    The control exists to answer "is this the target or the host?", and it
    measures the *stock* implementation. Recording it under the real gate's
    name would report the stock implementation's result as the target's.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import MODE_OFF, run_boot_gate

    commands = []
    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/py"}}, world_size=1
    )
    result = run_boot_gate(
        environment,
        task_data=_boot_task(tmp_path),
        runner=_capture_runner(commands),
        mode=MODE_OFF,
    )
    assert "TRTLLM_MODELING_V2=off" in commands[0]
    assert result.name == "boot[off]"


def test_the_control_run_is_otherwise_the_same_command(tmp_path):
    """One variable, or it is not a control."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import MODE_OFF, run_boot_gate

    require_cmds, off_cmds = [], []
    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/py"}}, world_size=1
    )
    task = _boot_task(tmp_path)
    run_boot_gate(environment, task_data=task, runner=_capture_runner(require_cmds))
    run_boot_gate(environment, task_data=task, runner=_capture_runner(off_cmds), mode=MODE_OFF)

    normalize = lambda c: c.replace("TRTLLM_MODELING_V2=off", "TRTLLM_MODELING_V2=require")  # noqa: E731
    assert normalize(off_cmds[0]) == require_cmds[0]


# ------------------------------------------------------------------- scope


def test_text_only_scope_tells_the_loader_to_ignore_vision_tensors():
    block = scope_block("text_only")
    assert "ignore the vision tensors" in block
    # The specific failure worth naming: a strict loader on a correct target.
    assert "asserts every tensor was consumed" in block


def test_full_scope_adds_nothing():
    assert scope_block("full") == ""


# ---------------------------------------------------------------- embargo


def test_an_embargo_names_every_path_and_permits_running_them():
    block = source_embargo_block(["tensorrt_llm/_torch/models/modeling_qwen3_5.py"])
    assert "`tensorrt_llm/_torch/models/modeling_qwen3_5.py`" in block
    # Reading is forbidden; running is the point. A block that forbade both
    # would throw away the run's only accuracy oracle.
    assert "off limits to read" in block
    assert "`modeling_v2=off`" in block


def test_no_embargo_adds_nothing():
    assert source_embargo_block([]) == ""


def test_an_out_of_repo_entry_does_not_claim_this_repository_implements_it():
    """The opening sentence has to split on where the paths point.

    "This checkpoint already has an implementation in this repository" is
    false for a sibling vLLM checkout, and a rule whose premise is visibly
    wrong is a rule an agent reasons its way around.
    """
    block = source_embargo_block(["/lustre/users/x/vllm"])
    assert "in this repository" not in block
    assert "another engine's checkout on this filesystem" in block
    # Running it is still the point — as a server rather than under
    # TRTLLM_MODELING_V2=off, which means nothing to another engine.
    assert "black-box oracle is expected" in block
    assert "TRTLLM_MODELING_V2=off" not in block


def test_both_kinds_of_entry_each_get_their_own_premise_and_permission():
    block = source_embargo_block(
        ["tensorrt_llm/_torch/models/modeling_qwen3_5.py", "/lustre/users/x/sglang"]
    )
    assert "already has an implementation in this repository" in block
    assert "another engine's checkout on this filesystem" in block
    assert "`modeling_v2=off`" in block
    assert "black-box oracle is expected" in block


def test_a_directory_entry_is_said_to_embargo_the_tree_beneath_it():
    """A sibling engine checkout is a tree.

    Enumerating its model files in a spec is asking the spec to go stale.
    """
    assert "the whole tree beneath it" in source_embargo_block(["/lustre/users/x/vllm"])


def test_every_role_gets_the_embargo_and_the_scope():
    """A role not told the rule is the role that breaks it.

    The reviewer is the specific worry: its whole job is comparison, so an
    un-briefed reviewer will helpfully open the embargoed file to check the
    target against it.
    """
    bundle = build_trtllm_modeling_v2_prompts(
        source_embargo=["tensorrt_llm/_torch/models/modeling_qwen3_5.py"],
        scope="text_only",
    )
    for role in ("plan_drafter", "plan_reviewer", "smith", "assembler", "reviewer"):
        prompt = getattr(bundle, role)
        assert "Embargoed sources" in prompt, role
        assert "Scope: text only" in prompt, role


# --------------------------------------------- preflight under a spool session


def test_preflight_claims_no_devices_under_a_spool_session(tmp_path):
    """The orchestrator's host may have no GPUs at all, and that is not a failure.

    The session's scheduler hands devices out per job, so probing this host
    with `nvidia-smi` would fail a run that is perfectly fine.
    """
    import json

    from agent_flow.workflows.trtllm_modeling_v2.run import preflight

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}), encoding="utf-8"
    )

    def _explode(*args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("preflight probed devices under a scheduler-owned allocation")

    result = preflight.run_preflight(
        {
            "checkpoint_path": str(checkpoint),
            "arch": "sm_103",
            "parallel": "tp1",
            "trtllm_repo_path": str(tmp_path),
            "environment": {
                "kind": "session_spool",
                "spool": str(tmp_path),
                "python": "/opt/venv/bin/python",
                "env": {"TMPDIR": _SCRATCH, "CLAUDE_CODE_TMPDIR": _SCRATCH},
            },
        },
        runner=_explode,
    )
    assert result.devices == ()
    # The trust being extended has to be visible in the report, because
    # nothing local verified it.
    assert any("taken on trust" in note for note in result.notes)


# ----------------------------------------------------------- schema wiring


def _task_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "task.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _base(repo: Path, checkpoint: Path) -> str:
    return f"""
        checkpoint_path: {checkpoint}
        trtllm_repo_path: {repo}
        arch: sm_103
        parallel: tp1
        accuracy:
          benchmark: gsm8k
          reference_yaml: tests/x/gsm8k.yaml
        environment:
          kind: local
          python: /usr/bin/python3
          env:
            TMPDIR: {_SCRATCH}
        """


def test_runtime_defaults_are_filled_in_when_the_block_is_absent(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = _task_yaml(tmp_path, _base(repo, repo))
    data = load_and_validate_task_yaml(task)
    assert data["runtime"] == dict(RUNTIME_SWITCHES) | {"extra_flags": []}
    assert data["scope"] == "full"
    assert data["source_embargo"] == []


def test_a_misspelled_runtime_switch_is_an_error_not_a_shrug(tmp_path):
    """Silently ignoring it gates a feature set nobody asked for."""
    repo = tmp_path / "repo"
    repo.mkdir()
    task = _task_yaml(
        tmp_path,
        _base(repo, repo)
        + """
        runtime:
          cuda_graphs: true
        """,
    )
    with pytest.raises(TaskSchemaError, match="unknown switch"):
        load_and_validate_task_yaml(task)


def test_an_embargo_on_a_missing_path_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = _task_yaml(
        tmp_path,
        _base(repo, repo)
        + """
        source_embargo:
          - tensorrt_llm/_torch/models/modeling_nothing.py
        """,
    )
    with pytest.raises(TaskSchemaError, match="protects nothing"):
        load_and_validate_task_yaml(task)


def test_an_embargo_on_a_real_path_is_accepted(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "modeling_x.py").write_text("", encoding="utf-8")
    task = _task_yaml(
        tmp_path,
        _base(repo, repo)
        + """
        source_embargo:
          - pkg/modeling_x.py
        """,
    )
    assert load_and_validate_task_yaml(task)["source_embargo"] == ["pkg/modeling_x.py"]


def test_an_embargo_may_name_a_tree_outside_the_repo(tmp_path):
    """The trees most likely to be copied from are not in this repository.

    TensorRT-LLM ships no implementation of a genuinely new checkpoint — that
    is the premise of the run — while a sibling vLLM or SGLang checkout on the
    same filesystem does, is one `Read` away from every agent, and is in the
    run's working set anyway because the accuracy anchor comes from it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    sibling = tmp_path / "vllm"
    (sibling / "model_executor").mkdir(parents=True)
    task = _task_yaml(
        tmp_path,
        _base(repo, repo)
        + f"""
        source_embargo:
          - {sibling}
        """,
    )
    assert load_and_validate_task_yaml(task)["source_embargo"] == [str(sibling)]


def test_an_absolute_path_inside_the_repo_is_still_rejected(tmp_path):
    """An in-repo path must be spelled repo-relative.

    A repo-relative list means the same thing from any working directory; an
    absolute one that happens to point inside the repo does not.
    """
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "modeling_x.py").write_text("", encoding="utf-8")
    task = _task_yaml(
        tmp_path,
        _base(repo, repo)
        + f"""
        source_embargo:
          - {repo}/pkg/modeling_x.py
        """,
    )
    with pytest.raises(TaskSchemaError, match="Spell an in-repo path repo-relative"):
        load_and_validate_task_yaml(task)


def test_an_out_of_repo_embargo_must_still_exist(tmp_path):
    """The must-exist argument is path-shape-independent.

    An embargo naming a path that is not there reads as protection while
    protecting nothing.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    task = _task_yaml(
        tmp_path,
        _base(repo, repo)
        + """
        source_embargo:
          - /definitely/not/here/vllm
        """,
    )
    with pytest.raises(TaskSchemaError, match="protects nothing"):
        load_and_validate_task_yaml(task)


# ------------------------------------------- per-unit gating of features


def test_core_is_gated_without_the_speculator_flags(tmp_path):
    """Core is green *before* the first feature increment — so must its gate be.

    A core target may refuse `--spec_decode_algo=MTP` through its own
    fail-fast assert. Deleting the assert to make the gate pass would be the
    silent wrong — one-engine MTP is drafted by the model, so a base model
    under that config emits no drafts and the run would gate a speculator
    that is not there.
    """
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(tmp_path / "ws")
    workflow.task_data = {"features": ["mtp", "eagle3"]}
    assert workflow._features_through("core") == []
    assert workflow._features_through("mtp") == ["mtp"]
    assert workflow._features_through("eagle3") == ["mtp", "eagle3"]


def test_an_unknown_unit_falls_back_to_the_whole_feature_list(tmp_path):
    """Conservative: gate against everything rather than silently against nothing."""
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(tmp_path / "ws")
    workflow.task_data = {"features": ["mtp"]}
    assert workflow._features_through("not-a-unit") == ["mtp"]


def test_the_boot_gate_honours_the_features_override(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import run_boot_gate

    commands = []
    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/py"}}, world_size=1
    )
    task = _boot_task(tmp_path) | {"features": ["mtp"]}

    run_boot_gate(environment, task_data=task, runner=_capture_runner(commands))
    assert "--spec_decode_algo=MTP" in commands[0]

    run_boot_gate(environment, task_data=task, runner=_capture_runner(commands), features=[])
    assert "--spec_decode_algo=MTP" not in commands[1]


# ------------------------------------------- the eval gates' configuration


def test_the_eval_gates_carry_the_declared_runtime_configuration(tmp_path):
    """`trtllm-eval` takes none of the quickstart flags, so it needs the YAML.

    Without it the accuracy gate measures the engine's defaults — including a
    `max_seq_len` at the checkpoint's own maximum — while the anchor was
    measured under the requested configuration. That is not a loose
    comparison; it is a different question.
    """
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import eval_command

    options = tmp_path / "llm_api.yaml"
    options.write_text("kv_cache_config: {dtype: fp8}\n", encoding="utf-8")
    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/py"}}, world_size=1
    )
    task = {
        "checkpoint_path": "/ckpt",
        "parallel": "tp1",
        "accuracy": {
            "llm_api_options": str(options),
            "max_seq_len": 8192,
            "max_batch_size": 32,
        },
    }
    parts = eval_command(task, environment, "gsm8k")
    joined = " ".join(parts)
    assert f'--extra_llm_api_options="{options}"' in joined
    assert "--max_seq_len=8192" in joined
    assert "--max_batch_size=32" in joined
    # The benchmark is a positional and must stay last of the eval args.
    assert parts[-1] == "gsm8k"


def test_the_accuracy_command_selects_the_target_through_the_llm_api(tmp_path):
    """The accuracy command hands the mode to the LLM API, not just to the environment.

    Nothing in the engine reads TRTLLM_MODELING_V2: target selection is the LLM-API
    argument `modeling_v2`, default "off". A gate that only exported the switch would
    serve the built-in model and record the verdict as the target's.
    """
    from dataclasses import replace

    import yaml

    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import MODELING_V2_ENV, eval_command

    options = tmp_path / "llm_api.yaml"
    options.write_text("kv_cache_config: {dtype: fp8}\n", encoding="utf-8")
    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/py"}}, world_size=1
    )
    environment = replace(environment, env={**environment.env, MODELING_V2_ENV: "require"})
    task = {
        "checkpoint_path": "/ckpt",
        "parallel": "tp1",
        "accuracy": {"llm_api_options": str(options)},
    }
    joined = " ".join(eval_command(task, environment, "gsm8k"))
    used = joined.split('--extra_llm_api_options="', 1)[1].split('"', 1)[0]
    merged = yaml.safe_load(open(used, encoding="utf-8"))
    assert merged["modeling_v2"] == "require"
    assert merged["kv_cache_config"] == {"dtype": "fp8"}


def test_the_benchmark_stays_last_when_a_dataset_path_is_present():
    """MMLU needs --dataset_path, and it follows the positional benchmark."""
    from agent_flow.workflows.trtllm_modeling_v2.checks.gates import eval_command

    environment = resolve_environment(
        {"environment": {"kind": "local", "python": "/py"}}, world_size=1
    )
    task = {
        "checkpoint_path": "/ckpt",
        "parallel": "tp1",
        "accuracy": {"dataset_path": "/data/mmlu"},
    }
    parts = eval_command(task, environment, "mmlu")
    assert parts[-2] == "mmlu"
    assert parts[-1] == '--dataset_path="/data/mmlu"'


def test_an_llm_api_options_file_that_does_not_exist_is_refused(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    task = tmp_path / "task.yaml"
    task.write_text(
        textwrap.dedent(f"""
        checkpoint_path: {repo}
        trtllm_repo_path: {repo}
        arch: sm_103
        parallel: tp1
        accuracy:
          benchmark: gsm8k
          reference_yaml: tests/x/gsm8k.yaml
          llm_api_options: {tmp_path}/nope.yaml
        environment:
          kind: local
          python: /usr/bin/python3
          env:
            TMPDIR: {_SCRATCH}
        """),
        encoding="utf-8",
    )
    with pytest.raises(TaskSchemaError, match="measures the engine's defaults"):
        load_and_validate_task_yaml(task)
