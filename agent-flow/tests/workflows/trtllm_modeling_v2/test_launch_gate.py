# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launch-path smoke: a unit's own deployment entry, driven over HTTP."""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import launch_gate, workflow
from agent_flow.workflows.trtllm_modeling_v2.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.gates import ShellResult
from agent_flow.workflows.trtllm_modeling_v2.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "unified",
    "trtllm_repo_path": "/repo",
    "features": ["dspark", "multimodal", "disagg"],
    "accuracy": {
        "benchmark": {"default": "gsm8k", "multimodal": "mmmu"},
        "reference_yaml": "r.yaml",
        "max_seq_len": 16384,
    },
}


def _env(world_size=1):
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}},
        devices=tuple(range(world_size)),
        world_size=world_size,
    )


def _smoke(statuses, *, accuracy=100.0, started=True, leftovers=(), texts=None):
    lines = []
    for n, status in enumerate(statuses):
        record = {
            "index": n,
            "id": f"r{n}",
            "status": status,
            "text": (texts or {}).get(n, "#### 1"),
            "keywords_missing": [],
        }
        if status != 200:
            record["error"] = '{"error": "cannot pickle \'coroutine\' object"}'
        lines.append(f"{workload_io.SERVE_MARKER} {json.dumps(record)}")
    summary = {
        "requests": len(statuses),
        "started": started,
        "scored": len(statuses),
        "accuracy": accuracy,
        "leftover_pids": list(leftovers),
        "server_tail": ["RuntimeError: CUDA out of memory"],
    }
    lines.append(f"{workload_io.SERVE_SUMMARY_MARKER} {json.dumps(summary)}")
    return "\n".join(lines) + "\n"


def _runner(output):
    seen = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return ShellResult(returncode=0, stdout=output)

    run.seen = seen
    return run


def _spec_of(command):
    path = re.search(r"--spec (\S+?\.json)", command).group(1)
    return json.loads(Path(path).read_text())


def _gate(tmp_path, output, *, unit="dspark", member="tep4", task=None, **kwargs):
    run = _runner(output)
    result = launch_gate.run_launch_gate(
        _env(),
        task_data=task or _TASK,
        reference=92.3,
        runner=run,
        unit=unit,
        member=member,
        features=kwargs.pop("features", ["dspark"]),
        workdir=tmp_path,
        **kwargs,
    )
    return result, run


# --------------------------------------------------------------- the entry


def test_the_entry_is_trtllm_serve_with_the_units_configuration(tmp_path):
    options = tmp_path / "dspark.yaml"
    options.write_text("speculative_config: {decoding_type: DSpark}\n", encoding="utf-8")
    task = {
        **_TASK,
        "accuracy": {**_TASK["accuracy"], "llm_api_options": {"default": str(options)}},
    }
    command = launch_gate.serve_command(task, _env(4), unit="dspark", member="dep4")
    assert "trtllm-serve" in command and "--port={port}" in command
    assert "--tp_size=4 --ep_size=4" in command and "--max_seq_len=16384" in command
    assert "mpirun -n 4" in command and "trtllm-llmapi-launch" in command
    assert "TRTLLM_MODELING_V2=require" in command
    # Attention DP has no server flag either: it travels in the member's overlay.
    overlay = tmp_path / "dspark.dep4.llm_api.yaml"
    assert f'--extra_llm_api_options="{overlay}"' in command
    assert yaml.safe_load(overlay.read_text())["enable_attention_dp"] is True


def test_a_clean_smoke_passes_and_says_what_it_sent(tmp_path):
    result, run = _gate(tmp_path, _smoke([200] * 32, accuracy=93.75))
    assert result.passed, result.detail
    assert "32 request(s) through trtllm-serve" in result.detail
    assert result.floor == pytest.approx(92.3 - 2 * 4.71, abs=0.05)
    spec = _spec_of(run.seen[0])
    assert spec["server_command"].count("{port}") == 1
    assert spec["dataset"]["source"] == "openai/gsm8k" and spec["scorer"] == "gsm8k"
    assert str(launch_gate.SERVE_SMOKE) in run.seen[0]


def test_http_400_through_the_server_fails_the_unit(tmp_path):
    """The defect the accuracy gate's in-process LLM API could not see."""
    statuses = [200] * 32 + [400]
    result, _ = _gate(
        tmp_path, _smoke(statuses), unit="multimodal", features=["dspark", "multimodal"]
    )
    assert not result.passed
    assert "1 of 33 request(s) through trtllm-serve failed with HTTP [400]" in result.detail
    assert "coroutine" in result.detail
    assert "HTTP [400]" in result.summary()


def test_the_multimodal_unit_sends_the_image_with_its_keywords(tmp_path):
    _, run = _gate(
        tmp_path, _smoke([200] * 33), unit="multimodal", features=["dspark", "multimodal"]
    )
    [image] = _spec_of(run.seen[0])["requests"]
    assert image["images"] == ["/repo/tests/integration/test_input_files/merlion.png"]
    assert "merlion" in image["keywords_any"]


def test_an_entry_that_never_serves_fails_with_its_log(tmp_path):
    result, _ = _gate(tmp_path, _smoke([], started=False))
    assert not result.passed and "never became healthy" in result.detail
    assert "CUDA out of memory" in result.stdout_tail


def test_garbage_through_the_server_fails_the_floor(tmp_path):
    result, _ = _gate(tmp_path, _smoke([200] * 32, accuracy=12.5))
    assert not result.passed and "served path answers worse" in result.detail


def test_a_declared_floor_and_a_disabled_scorer(tmp_path):
    config = launch_gate.LaunchConfig(dataset={"source": "x"}, min_accuracy=95.0, scorer="gsm8k")
    result, _ = _gate(tmp_path, _smoke([200] * 32, accuracy=93.75), config=config)
    assert not result.passed and "declared min_accuracy 95.00" in result.detail


def test_leftover_processes_are_reported_not_killed(tmp_path):
    result, _ = _gate(tmp_path, _smoke([200] * 32, accuracy=93.75, leftovers=[4242]))
    assert (
        result.passed and "still alive after a graceful stop, not killed: [4242]" in result.detail
    )


def test_the_disagg_unit_needs_its_launcher(tmp_path):
    result, run = _gate(tmp_path, _smoke([200] * 32), unit="disagg")
    assert not result.passed and "accuracy.launch.disagg.command" in result.detail
    assert run.seen == []
    task = {**_TASK, "accuracy": {**_TASK["accuracy"], "launch": {"disagg": {"command": "/l.sh"}}}}
    result, run = _gate(tmp_path, _smoke([200] * 32, accuracy=93.75), unit="disagg", task=task)
    assert result.passed and "through the declared launcher" in result.detail
    spec = _spec_of(run.seen[0])
    assert spec["launcher_command"] == "/l.sh" and "server_command" not in spec
    assert f"{launch_gate.LAUNCH_MEMBER_ENV}=tep4" in run.seen[0]


# ------------------------------------------------------------ configuration


def test_the_multimodal_unit_is_judged_against_the_text_anchor(tmp_path):
    asked = []

    def key(scope):
        asked.append(scope)
        return "M"

    refs = tmp_path / "r.yaml"
    refs.write_text(yaml.safe_dump({"M": [{"accuracy": 92.3}]}), encoding="utf-8")
    task = {**_TASK, "trtllm_repo_path": str(tmp_path)}
    assert launch_gate.smoke_reference(task, "multimodal", key) == 92.3
    assert launch_gate.smoke_reference(task, "dspark", key) == 92.3
    assert asked == ["default", "dspark"]


def test_units_members_and_entries_layer_per_unit():
    accuracy = {
        **_TASK["accuracy"],
        "launch": {"members": "first", "num_samples": 16, "core": {"enabled": False}},
    }
    config = launch_gate.resolve_launch_config(accuracy, "dspark")
    assert config.num_samples == 16 and config.dataset["num"] == 16
    assert launch_gate.select_members(config, ["tep4", "dep4", "tp4"]) == ["tep4"]
    assert not launch_gate.applies(accuracy, "core") and launch_gate.applies(accuracy, "dspark")


def _spec(tmp_path, features, launch):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "parallel": "tp1",
        "features": features,
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml", "model_key": "M"},
    }
    if launch is not None:
        data["accuracy"]["launch"] = launch
    if "disagg" in features:
        data["accuracy"]["witness"] = {
            "disagg": {"contrast_env": {"X": "1"}, "contrast_command": "/b.py"}
        }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_schema_requires_the_disagg_launcher(tmp_path):
    with pytest.raises(TaskSchemaError, match="accuracy.launch.disagg.command"):
        load_and_validate_task_yaml(_spec(tmp_path, ["disagg"], None))
    load_and_validate_task_yaml(_spec(tmp_path, ["disagg"], {"disagg": {"command": "/l.sh"}}))
    load_and_validate_task_yaml(_spec(tmp_path, ["disagg"], {"disagg": {"enabled": False}}))


@pytest.mark.parametrize(
    "launch,message",
    [
        ({"endpoint": "grpc"}, "endpoint"),
        ({"members": "some"}, "members"),
        ({"scorer": "bleu"}, "scorer"),
        ({"min_accuracy": 120}, "min_accuracy"),
        ({"core": {"port": 1}}, "port"),
    ],
)
def test_the_schema_refuses_a_malformed_launch_block(tmp_path, launch, message):
    with pytest.raises(TaskSchemaError, match=message):
        load_and_validate_task_yaml(_spec(tmp_path, [], launch))


def test_the_unit_loop_runs_it_last():
    source = inspect.getsource(workflow.TrtllmModelingV2Workflow._assemble_one_unit)
    assert source.index("run_acceptance_gate") < source.index("run_launch_gate")
    block = source[source.index("launch_gate.applies") :][:1400]
    assert "select_members" in block and "smoke_reference" in block


# ------------------------------------------- the smoke driver, end to end

_FAKE_SERVER = textwrap.dedent(
    """
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {})
            elif self.path == "/v1/models":
                self._send(200, {"data": [{"id": "served-model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = body["messages"][0]["content"]
            if isinstance(content, list):
                self._send(400, {"error": "TypeError: cannot pickle 'coroutine' object"})
                return
            self._send(200, {"choices": [{"message": {"content": "so the answer is 3."}}]})

    print("MV2FLOW_ENDPOINT=http://127.0.0.1:%d" % int(sys.argv[1]), flush=True)
    HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
    """
)


def test_the_smoke_driver_serves_scores_and_stops(tmp_path):
    server = tmp_path / "server.py"
    server.write_text(_FAKE_SERVER, encoding="utf-8")
    data = tmp_path / "q.jsonl"
    data.write_text(
        "".join(json.dumps({"question": f"q{i}", "answer": f"#### {i}"}) + "\n" for i in range(5)),
        "utf-8",
    )
    image = tmp_path / "img.png"
    image.write_bytes(b"\x89PNG\r\n")
    spec = {
        "server_command": f"{sys.executable} {server} {{port}}",
        "requests": [{"id": "image0", "prompt": "describe", "images": [str(image)]}],
        "dataset": {"source": str(data), "num": 5},
        "scorer": "gsm8k",
        "startup_timeout_s": 60,
        "log_dir": str(tmp_path),
    }
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(launch_gate.SERVE_SMOKE), "--spec", str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    records = workload_io.parse_marked(done.stdout, workload_io.SERVE_MARKER)
    assert [r["status"] for r in records] == [400, 200, 200, 200, 200, 200]
    assert "coroutine" in records[0]["error"]
    assert [r["correct"] for r in records[1:]] == [False, False, False, True, False]
    [summary] = workload_io.parse_marked(done.stdout, workload_io.SERVE_SUMMARY_MARKER)
    assert summary["model"] == "served-model" and summary["accuracy"] == pytest.approx(20.0)
    assert summary["http_errors"] == 1 and summary["leftover_pids"] == []


def test_a_launcher_names_its_endpoint(tmp_path):
    server = tmp_path / "server.py"
    server.write_text(_FAKE_SERVER, encoding="utf-8")
    spec = {
        "launcher_command": f"exec {sys.executable} {server} 0 || true",
        "requests": [{"prompt": "q", "answer": "#### 3"}],
        "scorer": "gsm8k",
        "startup_timeout_s": 20,
        "log_dir": str(tmp_path),
    }
    # Port 0 is not a reachable endpoint: the launcher must say where it serves.
    spec["launcher_command"] = (
        f"exec {sys.executable} {server} "
        f'$({sys.executable} -c \'import socket;s=socket.socket();s.bind(("127.0.0.1",0));'
        f"print(s.getsockname()[1])')"
    )
    path = tmp_path / "launch.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(launch_gate.SERVE_SMOKE), "--spec", str(path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    [record] = workload_io.parse_marked(done.stdout, workload_io.SERVE_MARKER)
    assert record["status"] == 200 and record["correct"] is True
