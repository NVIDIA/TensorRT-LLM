# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One setting, two launchers: the quickstart's flags and the LLM-API options file agree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates, serve_gate
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.run import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.preflight import PreflightError


def _task(tmp_path: Path, *, flags=(), options=None, auto_map=False) -> dict:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir(exist_ok=True)
    config = {"architectures": ["NewModelForCausalLM"]}
    if auto_map:
        config["auto_map"] = {"AutoConfig": "configuration_new.NewConfig"}
    (checkpoint / "config.json").write_text(json.dumps(config))
    accuracy = {"benchmark": "gsm8k"}
    if options is not None:
        path = tmp_path / "core.llm_api.yaml"
        path.write_text(yaml.safe_dump(options))
        accuracy["llm_api_options"] = {"core": str(path)}
    return {
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": str(tmp_path / "repo"),
        "parallel": "tp1",
        "runtime": {"extra_flags": list(flags)},
        "accuracy": accuracy,
        "environment": {"kind": "local", "python": "/venv/bin/python"},
    }


def _quickstart(task: dict) -> list[str]:
    env = resolve_environment(task, devices=[0], world_size=1)
    return gates.quickstart_words(task, env, unit="core", features=[], max_tokens=8, member="tp1")


def test_a_fraction_stated_as_a_runtime_flag_reaches_the_options_file_launchers(tmp_path):
    """The run-5 OOM: --kv_cache_fraction reached the boot and not trtllm-eval."""
    task = _task(tmp_path, flags=["--kv_cache_fraction=0.5", "--max_seq_len=9216"])
    assert gates.launch_overlay(task, "core") == {
        "kv_cache_config": {"free_gpu_memory_fraction": 0.5}
    }
    env = resolve_environment(task, devices=[0], world_size=1)
    parts = gates.eval_command(task, env, "gsm8k", unit="core", member="tp1", workdir=tmp_path)
    [options] = [p for p in parts if p.startswith("--extra_llm_api_options=")]
    written = yaml.safe_load(Path(options.split("=", 1)[1].strip('"')).read_text())
    assert written["kv_cache_config"]["free_gpu_memory_fraction"] == 0.5
    served = serve_gate.write_serve_options(
        tmp_path / "serve.yaml", task, env, unit="core", member="tp1"
    )
    assert yaml.safe_load(served.read_text())["kv_cache_config"]["free_gpu_memory_fraction"] == 0.5


def test_a_setting_only_the_options_file_states_becomes_a_quickstart_flag(tmp_path):
    task = _task(
        tmp_path,
        options={"kv_cache_config": {"free_gpu_memory_fraction": 0.6, "enable_block_reuse": True}},
    )
    assert "--kv_cache_fraction=0.6" in _quickstart(task)
    # One stated as a flag already is not repeated.
    task["runtime"]["extra_flags"] = ["--kv_cache_fraction", "0.6"]
    assert sum(w.startswith("--kv_cache_fraction") for w in _quickstart(task)) == 1


def test_the_options_file_keeps_its_other_keys_beneath_a_handed_setting(tmp_path):
    task = _task(
        tmp_path,
        flags=["--kv_cache_fraction=0.5"],
        options={"kv_cache_config": {"enable_block_reuse": True}},
    )
    env = resolve_environment(task, devices=[0], world_size=1)
    served = serve_gate.write_serve_options(
        tmp_path / "serve.yaml", task, env, unit="core", member="tp1"
    )
    block = yaml.safe_load(served.read_text())["kv_cache_config"]
    assert block == {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5}


def test_a_checkpoint_that_ships_its_code_is_trusted_by_every_launcher(tmp_path):
    task = _task(tmp_path, auto_map=True)
    assert gates.checkpoint_remote_code(task["checkpoint_path"])
    assert "--trust_remote_code" in _quickstart(task)
    assert gates.launch_overlay(task, "core") == {"trust_remote_code": True}
    (tmp_path / "plain").mkdir()
    plain = _task(tmp_path / "plain")
    assert not gates.checkpoint_remote_code(plain["checkpoint_path"])
    assert "--trust_remote_code" not in _quickstart(plain)


def test_two_values_for_one_setting_are_refused_at_preflight(tmp_path):
    task = _task(
        tmp_path,
        flags=["--kv_cache_fraction=0.5"],
        options={"kv_cache_config": {"free_gpu_memory_fraction": 0.7}},
    )
    assert gates.launch_conflicts(task, "core") == [
        "--kv_cache_fraction=0.5 in runtime.extra_flags, but "
        "kv_cache_config.free_gpu_memory_fraction: 0.7 in the options file"
    ]
    with pytest.raises(PreflightError, match="different values for one setting"):
        preflight.check_launch_settings(task)
    task["runtime"]["extra_flags"] = ["--kv_cache_fraction=0.7"]
    assert preflight.check_launch_settings(task) == []


def test_the_logit_reference_trusts_a_checkpoint_that_ships_its_code_unless_told_not_to(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.onboard import logit_reference as lr

    task = _task(tmp_path, auto_map=True)
    assert lr.ships_code(task["checkpoint_path"]) == gates.checkpoint_remote_code(
        task["checkpoint_path"]
    )
    args = lr._parse(["--model", task["checkpoint_path"], "--out", "o.json"])
    assert args.trust_remote_code is None  # decided from the checkpoint in main()
    args = lr._parse(["--model", "m", "--out", "o.json", "--no-trust-remote-code"])
    assert args.trust_remote_code is False
