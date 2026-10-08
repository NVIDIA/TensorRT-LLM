"""Tests for the modeling-bringup ModelExpress (MX) qualification step."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.agent_team.state import STATE_FILENAME
from agent_flow.workflows.modeling_bringup import cli as _cli
from agent_flow.workflows.modeling_bringup import task_schema as _task_schema
from agent_flow.workflows.modeling_bringup.prompts import _common, build_modeling_bringup_prompts

ROLES = ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa")
POLICY_HEADING = "## ModelExpress (MX) qualification step"
DISABLED_HEADING = "## ModelExpress (MX) step disabled for this run"
EXISTING_HEADING = "## Existing-model MX qualification (task override)"
MX_BLOCKS = (
    _common.MODEL_EXPRESS_POLICY,
    _common.MODEL_EXPRESS_DISABLED,
    _common.MODEL_EXPRESS_EXISTING_MODEL,
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _valid_task_payload(tmp_path: Path) -> dict:
    """Return a payload whose three required paths exist on disk under `tmp_path`."""
    ref = tmp_path / "modeling.py"
    ref.write_text("# stub\n", encoding="utf-8")
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    repo = tmp_path / "trtllm-repo"
    repo.mkdir()
    return {
        "reference_code_path": str(ref),
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
    }


def _write_task_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _run_cli(monkeypatch, argv: list[str]) -> dict:
    captured = {}

    def fake_team_main(forwarded_argv, *, prompts):
        captured["argv"] = forwarded_argv
        captured["prompts"] = prompts

    monkeypatch.setattr(_cli, "_team_main", fake_team_main)
    _cli.main(argv)
    return captured


def test_model_express_policy_reaches_every_role_by_default():
    """The MX step is on by default for every role, and no override is appended."""
    bundle = build_modeling_bringup_prompts()
    for role in ROLES:
        prompt = getattr(bundle, role)
        flat = _flat(prompt)
        assert POLICY_HEADING in prompt, role
        for phrase in (
            "model_express.enabled: false",
            "docs/source/features/model-express.md",
            "_MX_BF16_DENSE_RUNTIME_CONSTRAINTS",
            "MX: not eligible —",
            "Never reuse or reinterpret an existing ABI ID",
            "ModelLoader._qualify_post_transform_profile",
            "not a REJECT trigger",
            "neither leaked prescriptions nor scope creep",
        ):
            assert phrase in flat, (role, phrase)
        assert DISABLED_HEADING not in prompt, role
        assert EXISTING_HEADING not in prompt, role


@pytest.mark.parametrize("block", [*MX_BLOCKS, _common.DOMAIN_PRIMING])
def test_model_express_blocks_avoid_role_scoped_and_environment_text(block):
    """The MX text goes to all five roles, so it must stay transport- and role-neutral."""
    for banned in (
        "trtllm-test-specialist",
        "trtllm-agent-toolkit",
        "slurm-environment",
        "Slurm container bootstrap",
        "test_command.md",
        "## Stage/Goal",
        "## Hand-written HF reference",
        "Done / TODO",
        "AutoModelForCausalLM.from_pretrained",
        "update_status",
        "read_status",
        "ask_human",
        "append_",
        "``",
    ):
        assert banned not in block, banned
    assert "\n\n\n" not in block


@pytest.mark.parametrize(
    "specialist_block", ["", "## Running tests — stub\n\nUse the stub runner.\n"]
)
def test_model_express_policy_joins_cleanly_with_or_without_test_specialist_block(
    monkeypatch, specialist_block
):
    """CI has no skill to probe, so the specialist block is empty there.

    The MX block sits before it, so neither case leaves a blank-line run.
    """
    extras = [
        importlib.import_module(f"agent_flow.workflows.modeling_bringup.prompts.{role}_extra")
        for role in ROLES
    ]
    prompts_pkg = importlib.import_module("agent_flow.workflows.modeling_bringup.prompts")
    monkeypatch.setattr(_common, "TRTLLM_TEST_SPECIALIST_INVOCATION", specialist_block)
    try:
        for module in extras:
            importlib.reload(module)
        bundle = importlib.reload(prompts_pkg).build_modeling_bringup_prompts()
        for role in ROLES:
            prompt = getattr(bundle, role)
            assert "\n\n\n" not in prompt, role
            if specialist_block and role != "coder":
                assert prompt.index(POLICY_HEADING) < prompt.index("## Running tests — stub")
    finally:
        # Never reload `_common`: importing it probes for skills with real sessions.
        monkeypatch.undo()
        for module in extras:
            importlib.reload(module)
        importlib.reload(prompts_pkg)


@pytest.mark.parametrize("include_model_express", [True, False])
def test_domain_priming_names_staged_post_load_hooks(include_model_express):
    """The post-load hook rule applies to every bring-up, even with the MX step off."""
    bundle = build_modeling_bringup_prompts(include_model_express=include_model_express)
    for role in ROLES:
        prompt = getattr(bundle, role)
        for hook in (
            "setup_aliases()",
            "transform_weights()",
            "_weights_transformed",
            "cache_derived_state()",
            "post_load_weights()",
        ):
            assert hook in prompt, (role, hook)


@pytest.mark.parametrize("include_slurm_environment", [False, True])
@pytest.mark.parametrize("replan_on_qa", [False, True])
def test_model_express_disabled_appends_explicit_override(include_slurm_environment, replan_on_qa):
    """Turning the step off appends an override after every other block."""
    bundle = build_modeling_bringup_prompts(
        include_slurm_environment=include_slurm_environment,
        replan_on_qa=replan_on_qa,
        include_model_express=False,
    )
    for role in ROLES:
        prompt = getattr(bundle, role)
        assert prompt.rstrip().endswith(_common.MODEL_EXPRESS_DISABLED.rstrip()), role
        assert "even if an earlier section asks for it" in _flat(prompt), role
        assert prompt.index(POLICY_HEADING) < prompt.index(DISABLED_HEADING), role
        assert EXISTING_HEADING not in prompt, role


def test_model_express_existing_model_override_neutralizes_new_model_mandates():
    """Existing-model mode overrides the new-model mandates, including Stage/Goal ones."""
    bundle = build_modeling_bringup_prompts(replan_on_qa=True, model_express_existing_model=True)
    for role in ROLES:
        prompt = getattr(bundle, role)
        assert prompt.rstrip().endswith(_common.MODEL_EXPRESS_EXISTING_MODEL.rstrip()), role
        assert DISABLED_HEADING not in prompt, role

    planner = bundle.plan_drafter
    override_index = planner.index(EXISTING_HEADING)
    assert planner.index("Bring-up project-level required mechanisms") < override_index
    assert planner.index("## Stage/Goal plan schema") < override_index

    flat = _flat(_common.MODEL_EXPRESS_EXISTING_MODEL)
    for phrase in (
        "regardless of how `task.yaml` is phrased",
        "single Stage",
        "Not measured — existing-model MX qualification",
        "QA may APPROVE",
    ):
        assert phrase in flat, phrase


def test_build_prompts_rejects_existing_model_when_model_express_disabled():
    """Existing-model mode is the MX step itself, so it cannot run with the step off."""
    with pytest.raises(ValueError, match="model_express_existing_model"):
        build_modeling_bringup_prompts(
            include_model_express=False, model_express_existing_model=True
        )


@pytest.mark.parametrize(
    ("model_express", "expected_heading"),
    [
        (None, None),
        ({"enabled": False}, DISABLED_HEADING),
        ({"existing_model": True}, EXISTING_HEADING),
    ],
)
def test_modeling_bringup_cli_gates_model_express_overrides_on_task_spec(
    tmp_path, monkeypatch, model_express, expected_heading
):
    """The `model_express` block in `task.yaml` picks which override, if any, is appended."""
    payload = _valid_task_payload(tmp_path)
    if model_express is not None:
        payload["model_express"] = model_express
    task_path = _write_task_yaml(tmp_path / "task.yaml", payload)
    argv = ["--task", str(task_path), "--workspace", str(tmp_path / "work")]

    captured = _run_cli(monkeypatch, argv)

    assert captured["argv"] == argv
    for role in ROLES:
        prompt = getattr(captured["prompts"], role)
        assert POLICY_HEADING in prompt, role
        for heading in (DISABLED_HEADING, EXISTING_HEADING):
            assert (heading in prompt) == (heading == expected_heading), (role, heading)


def test_modeling_bringup_cli_rejects_invalid_model_express_block(tmp_path, monkeypatch, capsys):
    """A typo in the block fails at the CLI boundary instead of silently using defaults."""
    payload = _valid_task_payload(tmp_path)
    payload["model_express"] = {"enable": False}
    task_path = _write_task_yaml(tmp_path / "task.yaml", payload)
    monkeypatch.setattr(_cli, "_team_main", lambda *args, **kwargs: pytest.fail("launched"))

    with pytest.raises(SystemExit) as excinfo:
        _cli.main(["--task", str(task_path), "--workspace", str(tmp_path / "work")])

    assert excinfo.value.code == 2
    assert "model_express" in capsys.readouterr().err


@pytest.mark.parametrize("clean", [False, True])
def test_modeling_bringup_cli_reads_checkpointed_task_on_resume(tmp_path, monkeypatch, clean):
    """A resumed run follows the checkpointed task copy that the agents read."""
    payload = _valid_task_payload(tmp_path)
    task_path = _write_task_yaml(tmp_path / "task.yaml", payload)
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / STATE_FILENAME).write_text("{}", encoding="utf-8")
    _write_task_yaml(workspace / "task.yaml", {**payload, "model_express": {"enabled": False}})
    argv = ["--task", str(task_path), "--workspace", str(workspace)]
    if clean:
        argv.append("--clean")

    captured = _run_cli(monkeypatch, argv)

    assert captured["argv"] == argv
    assert (DISABLED_HEADING in captured["prompts"].coder) is (not clean)


@pytest.mark.parametrize("block", ["absent", None])
def test_task_schema_defaults_model_express(tmp_path, block):
    """An absent or empty block resolves to the defaults: step on, not existing-model."""
    payload = _valid_task_payload(tmp_path)
    if block != "absent":
        payload["model_express"] = block

    data = _task_schema.load_and_validate_task_yaml(_write_task_yaml(tmp_path / "t.yaml", payload))

    assert data["model_express"] == {"enabled": True, "existing_model": False}
    assert _task_schema.model_express_enabled(data) is True
    assert _task_schema.model_express_existing_model(data) is False


@pytest.mark.parametrize(
    ("block", "enabled", "existing_model"),
    [
        ({"enabled": False}, False, False),
        ({"existing_model": True}, True, True),
        ({"enabled": True, "existing_model": False}, True, False),
    ],
)
def test_task_schema_accepts_model_express_modes(tmp_path, block, enabled, existing_model):
    """Each supported mode round-trips through validation and the accessors."""
    payload = _valid_task_payload(tmp_path)
    payload["model_express"] = block

    data = _task_schema.load_and_validate_task_yaml(_write_task_yaml(tmp_path / "t.yaml", payload))

    assert _task_schema.model_express_enabled(data) is enabled
    assert _task_schema.model_express_existing_model(data) is existing_model


@pytest.mark.parametrize(
    ("block", "message"),
    [
        (False, "must be a mapping"),
        ("yes", "must be a mapping"),
        ({"enabled": "no"}, "must be a boolean"),
        ({"existing_model": 1}, "must be a boolean"),
        ({"enable": False}, "unknown field"),
        ({"enabled": False, "existing_model": True}, "requires"),
    ],
)
def test_task_schema_rejects_invalid_model_express(tmp_path, block, message):
    """Malformed blocks are rejected with an actionable message."""
    payload = _valid_task_payload(tmp_path)
    payload["model_express"] = block

    with pytest.raises(_task_schema.TaskSchemaError, match=message):
        _task_schema.load_and_validate_task_yaml(_write_task_yaml(tmp_path / "t.yaml", payload))


@pytest.mark.parametrize("spec", [{}, {"model_express": None}, {"model_express": "junk"}])
def test_model_express_accessors_tolerate_raw_specs(spec):
    """The accessors fall back to the defaults for specs that skipped validation."""
    assert _task_schema.model_express_enabled(spec) is True
    assert _task_schema.model_express_existing_model(spec) is False
