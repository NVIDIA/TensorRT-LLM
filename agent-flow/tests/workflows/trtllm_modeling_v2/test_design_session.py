# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design obligations through real session API and the suite's stand-in servers."""

import copy

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import session
from agent_flow.workflows.trtllm_modeling_v2.checks.check_runtime import (
    default_shell_runner,
    in_tree,
)
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

MOVES = {"cache_transceiver_config": {"backend": "DEFAULT"}}
SCORED = {
    "name": "gsm8k",
    "anchor": 41.77,
    "source": "declared protocol",
    "samples": 1319,
}
PD = {"name": "pd", "ctx": {"options": dict(MOVES)}, "gen": {"options": dict(MOVES)}}


def lines(result):
    return {line.name: line for line in result.lines}


def test_pd_chunk_uses_prefill_side_limits(task, tmp_path):
    block = copy.deepcopy(PD)
    block["ctx"]["options"].update(enable_chunked_prefill=True, max_num_tokens=64, max_seq_len=512)
    block["gen"]["options"].update(
        enable_chunked_prefill=False, max_num_tokens=8192, max_seq_len=512
    )
    task["deployments"] = [block]
    dep = deployments.named(task)
    planned = session.plan(task, dep, directory=tmp_path)
    check = next(p for p in planned if p.name == "chunk")
    assert not check.blocked and check.steps, "prefill is chunked on ctx, not gen"
    assert len(check.steps[0]["requests"][0]["body"]["prompt"]) > 64


def test_missing_acceptance_dataset_is_blocked_before_load(run_session, task, tmp_path):
    task["accuracy"]["acceptance"] = {"dataset": str(tmp_path / "absent-acceptance.jsonl")}
    result, _ = run_session(
        {
            "name": "draft",
            "options": {
                "speculative_config": {"decoding_type": "MTP"},
                "kv_cache_config": {"enable_block_reuse": True},
            },
            "probes": ["reuse"],
        }
    )
    checked = lines(result)
    assert checked["boot"].passed
    assert checked["acceptance"].blocked and checked["reuse"].passed, {
        name: (row.blocked, row.detail) for name, row in checked.items()
    }


def test_hop_proxy_request_failure_is_failed(run_session, site):
    server = site / "trtllm-serve"
    text = server.read_text().replace(
        "    def do_POST(self):\n",
        '    def do_POST(self):\n        if proxy: return self._send({"error":"proxy down"}, 503)\n',
    )
    server.write_text(text)
    result, _ = run_session(PD)
    hop = lines(result)["hop"]
    assert not hop.passed and not hop.unverified and "503" in hop.detail


def test_nonzero_session_exit_cannot_pass(run_session, task):
    runner = in_tree(default_shell_runner, task["trtllm_repo_path"])

    def bad_exit(command, env, timeout):
        # A lead/launcher wrapper fails after its child wrote a complete report.
        done = runner(command + "; exit 17", env, timeout)
        assert done.returncode == 17
        return done

    result, _ = run_session({"name": "core", "options": {}}, runner=bad_exit)
    assert not result.result.passed, "final records do not erase an unsuccessful process exit"


@pytest.mark.parametrize("fault", ["keywords", "chunk", "unfinished"])
def test_boot_probes_still_detect_bad_answers(run_session, task, tmp_path, monkeypatch, fault):
    block = {"name": "core", "options": {}}
    if fault == "keywords":
        task["accuracy"]["boot"]["prompts"][0]["keywords_any"] = ["not-present"]
    if fault == "chunk":
        block["options"].update(enable_chunked_prefill=False, max_num_tokens=64, max_seq_len=256)
        block["probes"] = ["chunk"]
    if fault == "unfinished":
        monkeypatch.setenv("FAKE_FINISH", "cancelled")
    result, _ = run_session(block)
    assert not result.result.passed, result.render()
    assert result.loads == 1


def test_explicit_native_tests_do_not_disappear_from_contract(task):
    from agent_flow.workflows.trtllm_modeling_v2.run.validation.contract import compile_contract

    task["deployments"] = [
        {"name": "core", "options": {}, "tests": ["tests/test_state.py::test_transfer"]}
    ]
    gates = [check.gate for check in compile_contract(task).required()]
    assert "deployment" in gates and "tests" in gates
