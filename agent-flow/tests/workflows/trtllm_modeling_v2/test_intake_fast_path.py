# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fast path: a request derived from the checkpoint, the site and a goal, checked in phase 0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.logger import get_logger
from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.run.preflight import check_intake_request
from agent_flow.workflows.trtllm_modeling_v2.run.stages.preflight import Phase0Probe
from agent_flow.workflows.trtllm_modeling_v2.run.state import (
    STAGE_PLANNED,
    STATE_FILENAME,
    WorkflowState,
    save_state,
)
from agent_flow.workflows.trtllm_modeling_v2.spec.intake import derive
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.checks import blocking, run_checks
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.cli import main as intake_main
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.compile import compile_files
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.schema import IntakeError
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

from .conftest import SCRATCH
from .test_workflow import _workflow

REFERENCES = "tests/integration/defs/accuracy/references/gsm8k.yaml"


def _checkpoint(root: Path, **config) -> Path:
    path = root / "newmodel"
    path.mkdir(parents=True, exist_ok=True)
    base = {
        "architectures": ["NewModelForCausalLM"],
        "_name_or_path": "org/newmodel",
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_experts": 16,
        "max_position_embeddings": 32768,
        "num_nextn_predict_layers": 1,
        "vision_config": {"model_type": "vit"},
    }
    (path / "config.json").write_text(json.dumps({**base, **config}))
    return path


def _repo(root: Path, rows: dict | None = None) -> Path:
    repo = root / "repo"
    references = repo / REFERENCES
    references.parent.mkdir(parents=True, exist_ok=True)
    rows = rows if rows is not None else {"org/newmodel": [{"accuracy": 80.0}]}
    rows["org/newmodel"] = [
        *rows.get("org/newmodel", []),
        {"spec_dec_algo": "MTP", "accuracy": 80.0},
    ]
    references.write_text(yaml.safe_dump(rows))
    return repo


def _site(root: Path, repo: Path, **extra) -> dict:
    return {
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(root / "spool"),
            "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        },
        "device": {"arch": "sm_100", "gpus_per_node": 4},
        "defaults": {"members": ["tp1", "tep4", "dep4"]},
        **extra,
    }


def _site_file(root: Path, site: dict) -> Path:
    (root / "spool").mkdir(exist_ok=True)
    path = root / "site.yaml"
    path.write_text(yaml.safe_dump(site))
    return path


# ------------------------------------------------------------------ the goal


def test_the_goal_names_features_members_and_the_workload():
    goal = derive.parse_goal(
        "mtp, disagg@tp1/tep4 -multimodal tp1 tep4; isl=8192 concurrency=1/16 tune"
    )
    assert goal.features == {
        "speculative": "required",
        "disaggregated": "required",
        "multimodal": "out_of_scope",
    }
    assert goal.algo == "mtp"
    assert goal.feature_members == {"disaggregated": ["tp1", "tep4"]}
    assert goal.members == ["tp1", "tep4"]
    assert goal.workload == {"isl": 8192, "concurrency": [1, 16]}
    assert goal.tune


def test_every_word_the_goal_does_not_know_is_named_at_once():
    with pytest.raises(IntakeError) as raised:
        derive.parse_goal("mtpp tp1 isl=lots colour=blue kv=int3")
    message = str(raised.value)
    for word in ("'mtpp'", "isl=lots", "'colour'", "kv=int3"):
        assert word in message, message


# ------------------------------------------------------------ the derivation


def test_features_the_checkpoint_carries_are_built_with_their_provenance(tmp_path):
    repo = _repo(tmp_path)
    derivation = derive.derive_task(
        _checkpoint(tmp_path), _site(tmp_path, repo), derive.parse_goal("")
    )
    task = derivation.task
    assert task["features"]["speculative"] == {
        "status": "required",
        "params": {"algo": "mtp", "draft_len": 1},
    }
    assert task["features"]["multimodal"] == {"status": "required"}
    assert task["features"]["cuda_graph"] == {"status": "required"}
    assert [f"{p['kind']}{p['world']}" for p in task["parallelism"]] == ["tp1", "tep4", "dep4"]
    assert task["workload"] == derive.DEFAULT_WORKLOAD
    assert task["accuracy"]["model_key"] == "org/newmodel"
    assert task["accuracy"]["boot"] == derive.DEFAULT_BOOT
    assert task["provenance"]["features.speculative.params.algo"] == "inferred"
    assert task["provenance"]["workload.isl"] == "default"
    assert task["provenance"]["parallelism"] == "inferred"
    assert task["open_questions"] == []
    assert task["derived"]["goal"] == ""
    assert blocking(run_checks(task)) == []


def test_the_goal_overrides_what_the_checkpoint_and_the_site_say(tmp_path):
    repo = _repo(tmp_path)
    goal = derive.parse_goal("-mtp text_only tep4 isl=1024 osl=256 arch=sm_103")
    task = derive.derive_task(_checkpoint(tmp_path), _site(tmp_path, repo), goal).task
    assert task["features"]["speculative"] == {"status": "out_of_scope"}
    assert task["features"]["multimodal"] == {"status": "out_of_scope"}
    assert task["parallelism"] == [{"kind": "tep", "world": 4}]
    assert task["workload"]["isl"] == 1024 and task["provenance"]["workload.isl"] == "user"
    assert task["device"]["arch"] == "sm_103"


def test_what_cannot_be_derived_is_an_open_question_and_blocks(tmp_path):
    repo = _repo(tmp_path)
    site = _site(tmp_path, repo)
    site.pop("device")
    task = derive.derive_task(_checkpoint(tmp_path), site, derive.parse_goal("disagg")).task
    questions = "\n".join(task["open_questions"])
    assert "device.arch" in questions
    assert "accuracy.witness.disagg" not in questions
    assert len(blocking(run_checks(task))) >= 1


def test_a_disagg_witness_without_a_launcher_runs_the_flows_leg_b(tmp_path):
    repo = _repo(tmp_path)
    site = _site(tmp_path, repo)
    task = derive.derive_task(_checkpoint(tmp_path), site, derive.parse_goal("disagg")).task
    assert task["accuracy"]["witness"] == {"disagg": {}}
    assert task["open_questions"] == []
    site = _site(tmp_path, repo)
    site["defaults"]["witness"] = {"disagg": {"contrast_command": "leg_b.py"}}
    task = derive.derive_task(_checkpoint(tmp_path), site, derive.parse_goal("disagg")).task
    assert any("accuracy.witness.disagg: declare contrast_env" in q for q in task["open_questions"])


def test_the_sites_witness_declaration_is_carried_to_the_run(tmp_path):
    repo = _repo(tmp_path)
    witness = {"contrast_env": {"LEG": "B"}, "contrast_command": "leg_b.py"}
    site = _site(tmp_path, repo)
    site["defaults"]["witness"] = {"disagg": witness}
    task = derive.derive_task(_checkpoint(tmp_path), site, derive.parse_goal("disagg")).task
    assert task["accuracy"]["witness"] == {"disagg": witness}
    assert task["open_questions"] == []


def test_a_hybrid_checkpoint_gets_a_snapshot_policy_for_block_reuse(tmp_path):
    repo = _repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, layer_types=["linear_attention", "full_attention"])
    task = derive.derive_task(checkpoint, _site(tmp_path, repo), derive.parse_goal("")).task
    params = task["features"]["prefix_reuse"]["params"]
    assert params == {"snapshot_interval": derive.DEFAULT_SNAPSHOT_INTERVAL}
    assert blocking(run_checks(task)) == []
    # The check this default answers still fails without it.
    task["features"]["prefix_reuse"] = {"status": "required"}
    assert any("snapshot policy" in f.render() for f in blocking(run_checks(task)))


def test_a_default_workload_fits_the_model(tmp_path):
    repo = _repo(tmp_path)
    checkpoint = _checkpoint(tmp_path, max_position_embeddings=2048)
    task = derive.derive_task(checkpoint, _site(tmp_path, repo), derive.parse_goal("")).task
    assert task["workload"]["isl"] + task["workload"]["osl"] <= 2048
    stated = derive.parse_goal("isl=4096 osl=1024")
    task = derive.derive_task(checkpoint, _site(tmp_path, repo), stated).task
    assert any("exceeds the model" in f.render() for f in blocking(run_checks(task)))


# ---------------------------------------------- compiled, and checked in phase 0


def _compiled(tmp_path, task: dict, site: dict) -> tuple[Path, dict]:
    request = tmp_path / "ws" / derive.REQUEST_FILENAME
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(yaml.safe_dump(task))
    resolved = tmp_path / "ws" / "task.resolved.yaml"
    compile_files(request, _site_file(tmp_path, site), resolved)
    return resolved, load_and_validate_task_yaml(resolved)


def test_a_derived_request_compiles_to_a_spec_the_phase_0_checks_pass(tmp_path):
    repo = _repo(tmp_path)
    site = _site(tmp_path, repo)
    task = derive.derive_task(_checkpoint(tmp_path), site, derive.parse_goal("-multimodal")).task
    resolved, spec = _compiled(tmp_path, task, site)
    assert spec["features"] == ["mtp"]
    assert spec["accuracy"]["discriminators"] == {"default": {}, "mtp": {"spec_dec_algo": "MTP"}}
    mtp = yaml.safe_load(Path(spec["accuracy"]["llm_api_options"]["mtp"]).read_text())
    assert mtp["speculative_config"] == {"decoding_type": "MTP", "max_draft_len": 1}
    assert spec["accuracy"]["boot"]["prompts"][0]["keywords_any"] == ["Paris"]
    findings = check_intake_request(task, spec)
    assert blocking(findings) == [], [f.render() for f in findings]


def test_a_quantized_checkpoint_without_its_row_is_a_phase_0_block(tmp_path):
    repo = _repo(tmp_path)
    site = _site(tmp_path, repo)
    checkpoint = _checkpoint(tmp_path, quantization_config={"quant_algo": "NVFP4"})
    task = derive.derive_task(checkpoint, site, derive.parse_goal("-multimodal")).task
    _, spec = _compiled(tmp_path, task, site)
    blocks = [f.render() for f in blocking(check_intake_request(task, spec))]
    # Both units miss their NVFP4 row; neither falls back to the unquantized one.
    assert sum("accuracy anchor" in b for b in blocks) == 2, blocks


# --------------------------------------------------------------- the command


def _fast(tmp_path, site_path, goal=None, confirm_by=None, clean=False):
    return cli._derive_intake_task(
        _checkpoint(tmp_path),
        site_path,
        tmp_path / "ws",
        goal=goal,
        confirm_by=confirm_by,
        clean=clean,
    )


def _start(workspace: Path) -> None:
    save_state(
        workspace / STATE_FILENAME,
        WorkflowState(task_path=str(workspace / "task.yaml"), stage=STAGE_PLANNED),
    )


def test_the_command_derives_writes_and_compiles_in_one_step(tmp_path):
    site_path = _site_file(tmp_path, _site(tmp_path, _repo(tmp_path)))
    resolved, request = _fast(tmp_path, site_path, goal="-multimodal")
    assert request == tmp_path / "ws" / derive.REQUEST_FILENAME
    assert (tmp_path / "ws" / derive.REPORT_FILENAME).is_file()
    assert load_and_validate_task_yaml(resolved)["features"] == ["mtp"]


def test_an_edit_before_the_run_starts_is_taken_and_after_it_is_refused(tmp_path):
    site_path = _site_file(tmp_path, _site(tmp_path, _repo(tmp_path)))
    _, request = _fast(tmp_path, site_path, goal="-multimodal")
    text = request.read_text()
    request.write_text(text.replace("osl: 1024", "osl: 512"))
    resolved, _ = _fast(tmp_path, site_path)
    assert "--max_seq_len=4608" in load_and_validate_task_yaml(resolved)["runtime"]["extra_flags"]

    _start(tmp_path / "ws")
    _fast(tmp_path, site_path)  # unchanged since the run started
    request.write_text(request.read_text().replace("osl: 512", "osl: 256"))
    with pytest.raises(IntakeError, match="changed after its run started"):
        _fast(tmp_path, site_path)
    _fast(tmp_path, site_path, confirm_by="reviewer")
    request.write_text(request.read_text().replace("osl: 256", "osl: 128"))
    with pytest.raises(IntakeError, match="changed after it was confirmed"):
        _fast(tmp_path, site_path)


def test_another_goal_or_checkpoint_needs_clean(tmp_path):
    site_path = _site_file(tmp_path, _site(tmp_path, _repo(tmp_path)))
    _fast(tmp_path, site_path, goal="-multimodal")
    with pytest.raises(IntakeError, match="--clean"):
        _fast(tmp_path, site_path, goal="-multimodal -mtp")
    _, request = _fast(tmp_path, site_path, goal="-multimodal -mtp", clean=True)
    assert yaml.safe_load(request.read_text())["features"]["speculative"]["status"] == (
        "out_of_scope"
    )


def test_confirming_a_request_with_a_block_is_refused_with_every_block(tmp_path):
    site = _site(tmp_path, _repo(tmp_path))
    site.pop("device")
    site_path = _site_file(tmp_path, site)
    with pytest.raises(IntakeError) as raised:
        _fast(tmp_path, site_path, goal="disagg", confirm_by="reviewer")
    assert "device.arch" in str(raised.value) and "contrast_command" not in str(raised.value)


def test_the_command_line_takes_a_checkpoint_or_a_task(tmp_path):
    with pytest.raises(SystemExit):
        cli._parse_args(["--checkpoint", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli._parse_args(["--task", "t.yaml", "--goal", "mtp"])
    args = cli._parse_args(
        ["--checkpoint", str(tmp_path), "--site", "s.yaml", "--goal", "mtp", "--confirm"]
    )
    assert args.confirm == "" and args.goal == "mtp"


def test_a_reload_does_not_confirm_again(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.os, "execve", lambda path, argv, env: seen.update(argv=argv))
    argv = ["python", "--checkpoint", "c", "--confirm", "--site", "s.yaml", "--confirm", "me"]
    cli._relauncher([*argv, "--confirm=me", "--workspace", "ws"], {})()
    assert seen["argv"][1:] == ["--checkpoint", "c", "--site", "s.yaml", "--workspace", "ws"]


def test_intake_derive_writes_the_request_and_its_report(tmp_path, capsys):
    site_path = _site_file(tmp_path, _site(tmp_path, _repo(tmp_path)))
    out = tmp_path / "out" / "task.yaml"
    argv = ["derive", "--checkpoint", str(_checkpoint(tmp_path)), "--site", str(site_path)]
    assert intake_main([*argv, "--goal=-multimodal", "-o", str(out)]) == 0
    assert (out.parent / derive.REPORT_FILENAME).is_file()
    assert intake_main(["check", str(out)]) == 0


# --------------------------------------------------------------------- phase 0


def _request(tmp_path, checkpoint: Path, **changes) -> Path:
    task = {
        "version": 1,
        "goals": ["onboard"],
        "model": {"checkpoint": str(checkpoint)},
        "device": {"arch": "sm_103"},
        "parallelism": [{"kind": "tp", "world": 1}],
        "workload": {"isl": 1024, "osl": 256},
        **changes,
    }
    path = tmp_path / derive.REQUEST_FILENAME
    path.write_text(yaml.safe_dump(task))
    return path


def test_phase_0_stops_on_every_intake_block_before_the_host_is_touched(
    tmp_path, task_file, checkpoint
):
    checkpoint.joinpath("config.json").write_text(
        json.dumps({"architectures": ["GptOssForCausalLM"], "num_attention_heads": 8})
    )
    workflow = _workflow(tmp_path)
    log = get_logger().console
    state = workflow._init_state(task_file, log)
    workflow.intake_request = _request(
        tmp_path,
        checkpoint,
        parallelism=[{"kind": "tp", "world": 3}],
        open_questions=["which benchmark?"],
    )
    assert workflow._run_preflight(state, log) is False
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "intake checks failed: 3 block(s)" in journal
    assert "8 heads do not split 3" in journal and "which benchmark?" in journal
    assert workflow.environment is None


def test_phase_0_records_the_confirmation_and_goes_on(tmp_path, task_file, checkpoint):
    workflow = _workflow(tmp_path)
    log = get_logger().console
    state = workflow._init_state(task_file, log)
    workflow.intake_request = _request(tmp_path, checkpoint)
    assert workflow._run_preflight(state, log) is True
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "intake: not confirmed; the run does not wait for it" in journal


def test_phase_0_probes_run_beside_and_a_failing_one_does_not_stop_the_run(tmp_path, task_file):
    seen = []

    def measure(workflow, result):
        seen.append(result.world_size)

    def broken(workflow, result):
        raise RuntimeError("no engine")

    workflow = _workflow(tmp_path)
    workflow.phase0_probes = (Phase0Probe("measure", measure), Phase0Probe("broken", broken))
    log = get_logger().console
    state = workflow._init_state(task_file, log)
    assert workflow._run_preflight(state, log) is True
    assert workflow._await_phase0_probes(timeout=10)
    assert seen == [1]
    assert "phase-0 probe `broken` failed" in (tmp_path / "ws" / "journal.md").read_text()
