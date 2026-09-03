"""Tests for the `bench-disagg` boundary.

Everything here pins a property of the CLI's protocol that this workflow
*depends on*, so that a change upstream fails here rather than three
stages into a campaign. The envelope shapes are taken from the installed
0.2.0 CLI, not paraphrased from its source.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from agent_flow.workflows.perf_optimize import bench_cli


class _Completed:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _envelope(kind: str, data: dict, ok: bool = True, error: dict | None = None) -> str:
    payload = {"api_version": "v1", "kind": kind, "ok": ok}
    if ok:
        payload["data"] = data
    else:
        payload["error"] = error or {}
    return json.dumps(payload, indent=2)


# ------------------------------------------------------------------ the envelope


def test_a_failed_envelope_becomes_a_branchable_error(monkeypatch):
    """In the CLI's own words, agents branch on the codes; messages are for humans.

    Losing the code and keeping only the message would make every failure
    look alike to the caller, which is what makes WORKSPACE_BUSY (retry)
    indistinguishable from CONTEXT_CHANGED (stop, the workspace is void).
    """
    monkeypatch.setattr(
        bench_cli.subprocess,
        "run",
        lambda *a, **k: _Completed(
            _envelope(
                "workspace.show",
                {},
                ok=False,
                error={
                    "code": "WORKSPACE_NOT_FOUND",
                    "message": "no such workspace",
                    "details": {},
                },
            ),
            returncode=2,
        ),
    )
    with pytest.raises(bench_cli.BenchCliError) as caught:
        bench_cli.run(["workspace", "show", "--workspace", "nope"])
    assert caught.value.code == "WORKSPACE_NOT_FOUND"
    assert "no such workspace" in str(caught.value)


def test_a_non_json_stdout_is_named_as_a_protocol_break(monkeypatch):
    """The CLI redirects subordinate prints to stderr precisely to prevent this.

    So a non-JSON stdout is not "no results" — it means something escaped
    that discipline, and retrying cannot help. Surfacing it as a
    JSONDecodeError from inside would hide that.
    """
    monkeypatch.setattr(
        bench_cli.subprocess,
        "run",
        lambda *a, **k: _Completed("Traceback (most recent call last):"),
    )
    with pytest.raises(bench_cli.BenchCliError, match="did not print a JSON envelope"):
        bench_cli.run(["sweep", "plan"])


def test_an_empty_stdout_points_at_the_install_not_the_data(monkeypatch):
    monkeypatch.setattr(bench_cli.subprocess, "run", lambda *a, **k: _Completed(""))
    with pytest.raises(bench_cli.BenchCliError, match="wrong PATH / venv"):
        bench_cli.run(["sweep", "plan"])


def test_a_missing_cli_says_what_to_install(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError(bench_cli.BENCH_DISAGG)

    monkeypatch.setattr(bench_cli.subprocess, "run", _boom)
    with pytest.raises(bench_cli.BenchCliError, match="pip install trtllm-disagg-bench"):
        bench_cli.run(["sweep", "plan"])


def test_a_timeout_is_not_mistaken_for_a_failed_command(monkeypatch):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="bench-disagg", timeout=1)

    monkeypatch.setattr(bench_cli.subprocess, "run", _boom)
    with pytest.raises(bench_cli.BenchCliError, match="timed out") as caught:
        bench_cli.run(["sweep", "status"], timeout=1)
    assert caught.value.code is None  # no envelope, so no taxonomy value


# ------------------------------------------------------------------ command shapes


def _capture(monkeypatch) -> list:
    seen: list = []

    def _run(command, **kwargs):
        seen.append(command)
        return _Completed(_envelope("x", {}))

    monkeypatch.setattr(bench_cli.subprocess, "run", _run)
    return seen


def test_plan_asks_for_cases_because_that_is_the_whole_point(monkeypatch):
    seen = _capture(monkeypatch)
    bench_cli.plan("sweep.yaml", "w")
    assert "--cases" in seen[0]


# ------------------------------------------------------------------ the readers


PLAN = {
    "workload": {"isl": 200000, "osl": 1024},
    "cases": [
        {
            "case": "gen-ctx1_gen1_tep4_b32_mnt128_gmf0.9_eplb0_mtp3_conc1",
            "stage": "gen",
            "status": "fresh",
            "config": {"gen_num": 1, "concurrency": 1, "tp_size": 4},
        },
        {
            "case": "gen-ctx1_gen2_dep32_b64_mnt256_gmf0.7_eplb0_mtp3_conc128",
            "stage": "gen",
            "status": "fresh",
            "config": {"gen_num": 2, "concurrency": 128, "tp_size": 32},
        },
        {"case": "ctx-isl200000_osl1_b1_tp4_mtp3_ratio0.8", "stage": "ctx", "config": {}},
    ],
}


def test_the_operating_point_is_the_product_not_the_listed_value():
    """The trap driving the CLI does NOT remove — it only moves the inputs.

    A sweep row's concurrency is per generation server; the client is
    driven at `concurrency * gen_num` and the harness names its result
    directory after that product. The case name keeps the listed value
    (it is the address), but `task.yaml`'s `concurrency` means what an
    aggregate campaign means by it, so the product is what belongs there.
    """
    assert bench_cli.operating_points(PLAN) == [1, 256]  # 1x1 and 128x2


def test_a_case_without_a_usable_concurrency_is_skipped_not_guessed():
    assert bench_cli.operating_points({"cases": [{"case": "ctx-x", "config": {}}]}) == []
    assert bench_cli.operating_points({}) == []


def test_the_cli_is_found_beside_the_interpreter_when_not_on_path(monkeypatch, tmp_path):
    """The console script and this package install into the same bin/.

    Only an *activated* venv puts that directory on PATH, and failing on
    that difference would be failing on an environment variable rather
    than on a missing dependency.
    """
    binned = tmp_path / "bin"
    binned.mkdir()
    (binned / bench_cli.BENCH_DISAGG).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(bench_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(bench_cli.sys, "executable", str(binned / "python"))

    assert bench_cli.executable() == str(binned / bench_cli.BENCH_DISAGG)

    seen = _capture(monkeypatch)
    bench_cli.plan("s.yaml", "w")
    assert seen[0][0] == str(binned / bench_cli.BENCH_DISAGG)


def test_a_cli_that_is_nowhere_still_names_the_install(monkeypatch):
    monkeypatch.setattr(bench_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(bench_cli.sys, "executable", "/no/such/python")
    assert bench_cli.executable() == bench_cli.BENCH_DISAGG
