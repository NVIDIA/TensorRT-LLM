# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The serve gate: `trtllm-serve` starts the unit's target and answers over HTTP."""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli, gates, serve_driver
from agent_flow.workflows.trtllm_modeling_v2.checks import serve_gate as sg
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run import contract as c
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    ACCURACY_KEYS,
    SERVE_KEYS,
    _validate_accuracy,
)

_MEMBERS = ["tp1", "tp2", "tep2", "dep2", "tp4", "tep4", "dep4"]


def _task(tmp_path, **accuracy):
    options = tmp_path / "core.llm_api.yaml"
    options.write_text("kv_cache_config: {free_gpu_memory_fraction: 0.6}\n", encoding="utf-8")
    return {
        "checkpoint_path": "/ckpt/model",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "parallel": "unified",
        "members": list(_MEMBERS),
        "features": ["mtp", "multimodal"],
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "llm_api_options": str(options),
            "max_seq_len": 9216,
            "max_batch_size": 128,
            **accuracy,
        },
    }


def _env(world_size, *, kind="local", devices=()):
    block = {"kind": kind, "python": "/venv/bin/python", "env": {"HF_HUB_OFFLINE": "1"}}
    return resolve_environment({"environment": block}, devices=devices, world_size=world_size)


def _server(healthy=True, **extra):
    return f"{serve_driver.SERVER_MARKER} {json.dumps({'healthy': healthy, **extra})}\n"


def _answer(index, text="Paris", status=200, **extra):
    record = {"index": index, "status": status, "text": text, **extra}
    return f"{workload_io.RECORD_MARKER} {json.dumps(record)}\n"


def _recorder(output=""):
    seen = []

    def run(command, env=None, timeout=0):
        seen.append((command, timeout))
        return CommandResult(returncode=0, stdout=output)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def _driver_spec(command):
    return json.loads(Path(re.search(r"--spec (\S+)'?", command).group(1).rstrip("'")).read_text())


# ------------------------------------------------------------------ the member


def test_a_unit_serves_at_its_widest_member_cheapest_first():
    assert sg.serve_member(_MEMBERS) == "tep4"
    assert sg.serve_member(["tp1", "tp2", "tep2", "dep2"]) == "tep2"
    assert sg.serve_member(["tp1"]) == "tp1"


def test_delivery_also_serves_at_one_rank_when_the_unit_has_one():
    assert sg.delivery_member(_MEMBERS) == "tp1"
    # Every member is multi-rank, or the unit already serves at one rank.
    assert sg.delivery_member(["tp4", "tep4", "dep4"]) is None
    assert sg.delivery_member(["tp1"]) is None


# ------------------------------------------------------------------ the command line


def test_the_server_runs_the_accuracy_options_at_the_members_topology(tmp_path):
    run = _recorder()
    sg.run_serve_gate(
        _env(4), task_data=_task(tmp_path), runner=run, unit="core", member="dep4", workdir=tmp_path
    )
    command, timeout = run.seen[0]
    spec = _driver_spec(command)
    options = tmp_path / "serve.core.dep4.llm_api.yaml"
    assert spec["serve_args"] == [
        "--config",
        str(options),
        "--tp_size=4",
        "--ep_size=4",
        "--max_seq_len=9216",
        "--max_batch_size=128",
    ]
    assert spec["world_size"] == 4 and spec["model"] == "/ckpt/model"
    written = yaml.safe_load(options.read_text(encoding="utf-8"))
    # The unit's own file, attention DP for a dep member, and the target selected.
    assert written == {
        "kv_cache_config": {"free_gpu_memory_fraction": 0.6},
        "enable_attention_dp": True,
        "modeling_v2": "require",
    }
    # One process: the driver creates the server's ranks itself.
    assert "mpirun" not in command and "unset" in command
    assert str(sg.SERVE_DRIVER) in command
    assert "TRTLLM_MODELING_V2=require" in command and "HF_HUB_OFFLINE=1" in command
    assert "CUDA_VISIBLE_DEVICES" not in command
    assert timeout == spec["boot_timeout"] + 2 * sg.REQUEST_TIMEOUT_S + sg.TEARDOWN_S


def test_a_one_rank_member_serves_without_attention_dp_on_the_claimed_devices(tmp_path):
    run = _recorder()
    sg.run_serve_gate(
        _env(1, devices=(3,)),
        task_data=_task(tmp_path),
        runner=run,
        unit="core",
        member="tp1",
        workdir=tmp_path,
    )
    command = run.seen[0][0]
    spec = _driver_spec(command)
    assert spec["world_size"] == 1
    assert "--tp_size=1" in spec["serve_args"] and "--ep_size=1" not in spec["serve_args"]
    written = yaml.safe_load((tmp_path / "serve.core.tp1.llm_api.yaml").read_text())
    assert "enable_attention_dp" not in written
    assert command.startswith("CUDA_VISIBLE_DEVICES=3 ")


def test_the_server_command_creates_every_rank_before_the_engine(monkeypatch):
    monkeypatch.setattr(serve_driver, "entry_point", lambda name: f"/venv/bin/{name}")
    spec = {"model": "/ckpt", "world_size": 4, "serve_args": ["--config", "o.yaml"]}
    argv = serve_driver.server_argv(spec, 8123)
    mpirun = argv.index("mpirun")
    assert argv[0] == "env" and ["-u", "SLURM_PROCID"] == argv[1:3]
    assert argv[mpirun : mpirun + 4] == ["mpirun", "-n", "4", "--oversubscribe"]
    serve = argv.index("/venv/bin/trtllm-serve")
    assert argv[serve - 1] == "/venv/bin/trtllm-llmapi-launch"
    assert argv[serve:] == [
        "/venv/bin/trtllm-serve",
        "/ckpt",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--config",
        "o.yaml",
    ]
    assert serve_driver.server_argv({**spec, "world_size": 1}, 8123)[0] == "/venv/bin/trtllm-serve"


# ------------------------------------------------------------------ the requests


def test_a_text_unit_asks_one_completion(tmp_path):
    task = _task(tmp_path, serve={"prompt": "Two plus two is", "max_tokens": 8})
    requests = sg.serve_requests(task, sg.ServeSpec.from_task(task), gates.BootSpec())
    assert requests == [
        {
            "id": "text",
            "endpoint": "/v1/completions",
            "body": {
                "model": "/ckpt/model",
                "prompt": "Two plus two is",
                "max_tokens": 8,
                "temperature": 0.0,
            },
        }
    ]


def test_a_multimodal_unit_also_asks_a_chat_with_the_boot_image(tmp_path):
    task = _task(tmp_path)
    boot = gates.resolve_boot_spec(task, "multimodal", ["mtp", "multimodal"])
    requests = sg.serve_requests(task, sg.ServeSpec(), boot)
    assert [r["endpoint"] for r in requests] == ["/v1/completions", "/v1/chat/completions"]
    content = requests[1]["body"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": gates.DEFAULT_BOOT_IMAGE_PROMPT}
    assert content[1]["image_url"]["url"] == str(tmp_path / "repo" / gates.DEFAULT_BOOT_IMAGE)


def test_a_pre_rendered_image_prompt_is_not_sent_to_the_chat_template(tmp_path):
    task = _task(tmp_path)
    raw = gates.BootPrompt(prompt="<image>\nThis is a photo of", images=("a.png",), render="raw")
    plain = gates.BootPrompt(prompt="What is shown?", images=("b.png",))
    texts = [
        sg.serve_requests(task, sg.ServeSpec(), gates.BootSpec(items=(item,)))[1]["body"][
            "messages"
        ][0]["content"][0]["text"]
        for item in (raw, plain)
    ]
    assert texts == [gates.DEFAULT_BOOT_IMAGE_PROMPT, "What is shown?"]


def test_a_local_image_is_sent_inline(tmp_path):
    image = tmp_path / "merlion.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": str(image)}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                ],
            }
        ]
    }
    sent = serve_driver.inline_images(body)
    url = sent["messages"][0]["content"][1]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    assert sent["messages"][0]["content"][2] == body["messages"][0]["content"][2]
    assert body["messages"][0]["content"][1]["image_url"]["url"] == str(image)


def test_an_lfs_pointer_is_named_rather_than_sent(tmp_path):
    pointer = tmp_path / "merlion.png"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:x\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        serve_driver.data_url(str(pointer))
    with pytest.raises(FileNotFoundError):
        serve_driver.data_url(str(tmp_path / "missing.png"))


# ------------------------------------------------------------------ the verdict

_REQUESTS = [
    {"id": "text", "endpoint": "/v1/completions"},
    {"id": "image", "endpoint": "/v1/chat/completions"},
]


def test_a_healthy_server_that_answers_every_request_passes():
    output = _answer(0, " Paris.") + _answer(1, "A lion statue.") + _server(boot_s=42.0)
    passed, detail, warnings = sg.judge(output, _REQUESTS)
    assert passed and warnings == ()
    assert "healthy after 42.0s" in detail and "A lion statue." in detail


def test_a_server_that_never_came_up_fails_with_its_cause():
    output = _server(
        healthy=False,
        error="server exited with 1 during boot; tail of /w/serve.log:",
        cause="AttributeError: 'NewModelInputProcessor' object has no attribute "
        "'get_preferred_media_io_kwargs'",
        log="/w/serve.log",
    )
    passed, detail, _ = sg.judge(output, _REQUESTS)
    assert not passed
    assert "never became healthy" in detail and "get_preferred_media_io_kwargs" in detail


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (_answer(0) + _answer(1, "", finish_reason="stop"), "image (/v1/chat/completions): empty"),
        (_answer(0) + _answer(1, status=500, error="boom"), "HTTP 500: boom"),
        (_answer(0) + _answer(1, status=None, error="timed out"), "no response: timed out"),
        (_answer(0), "image (/v1/chat/completions): no answer recorded"),
    ],
)
def test_every_answer_must_be_a_200_with_text(answers, expected):
    passed, detail, _ = sg.judge(answers + _server(), _REQUESTS)
    assert not passed and expected in detail


def test_no_server_record_is_a_failure_naming_the_exit():
    passed, detail, _ = sg.judge("Traceback ...\n", _REQUESTS, returncode=1)
    assert not passed and "exited 1 before it reported the server" in detail


def test_a_rank_left_holding_a_device_is_reported():
    output = _answer(0) + _answer(1) + _server(left=[4242])
    passed, _, warnings = sg.judge(output, _REQUESTS)
    assert passed and "4242" in warnings[0]


def test_a_driver_that_exits_non_zero_after_its_answers_fails():
    passed, detail, _ = sg.judge(_answer(0) + _server(), _REQUESTS[:1], returncode=143)
    assert not passed and "exited 143" in detail


def test_the_gate_reads_the_driver_and_names_the_member(tmp_path):
    run = _recorder(_answer(0, " Paris.") + _server(boot_s=3.0))
    result = sg.run_serve_gate(
        _env(1),
        task_data=_task(tmp_path),
        runner=run,
        features=[],
        unit="core",
        member="tp1",
        workdir=tmp_path,
    )
    assert result.passed, result.detail
    assert result.summary().startswith("[PASS] serve@tp1: trtllm-serve healthy after 3.0s")


def test_a_lost_session_is_not_a_verdict(tmp_path):
    def run(command, env=None, timeout=0):
        return CommandResult(returncode=1, infrastructure="the session went away")

    result = sg.run_serve_gate(
        _env(1), task_data=_task(tmp_path), runner=run, unit="core", member="tp1", workdir=tmp_path
    )
    assert not result.passed and result.infrastructure == "the session went away"


def test_a_job_past_its_wall_fails_as_timed_out(tmp_path):
    def run(command, env=None, timeout=0):
        return CommandResult(returncode=-9, timed_out=True)

    task = _task(tmp_path, serve={"timeout": 60})
    result = sg.run_serve_gate(
        _env(1),
        task_data=task,
        runner=run,
        features=[],
        unit="core",
        member="tp1",
        workdir=tmp_path,
    )
    assert not result.passed
    assert f"timed out after {60 + sg.REQUEST_TIMEOUT_S + sg.TEARDOWN_S}s" in result.detail


# ------------------------------------------------------------------ the driver, end to end

_FAKE_SERVER = """
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
mode = os.environ.get("FAKE_SERVE_MODE", "ok")
seen = os.environ["FAKE_SERVE_SEEN"]
if mode == "crash":
    print("Traceback (most recent call last):")
    print("AttributeError: 'NewModelInputProcessor' object has no attribute "
          "'get_preferred_media_io_kwargs'", flush=True)
    sys.exit(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def answer(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.answer(200 if self.path == "/health" else 404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(seen, "a") as handle:
            handle.write(json.dumps({"path": self.path, "body": body}) + "\\n")
        if self.path == "/v1/completions":
            choice = {"text": " Paris.", "finish_reason": "length"}
        else:
            choice = {"message": {"role": "assistant", "content": "A lion statue."}}
        self.answer(200, {"choices": [choice], "usage": {"completion_tokens": 3}})


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def fake_serve(tmp_path, monkeypatch):
    """A `trtllm-serve` that answers like one, launched by the real driver."""
    server = tmp_path / "fake_serve.py"
    server.write_text(_FAKE_SERVER, encoding="utf-8")
    script = tmp_path / "trtllm-serve"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{server}" "$@"\n', encoding="utf-8")
    script.chmod(0o755)
    seen = tmp_path / "seen.jsonl"
    monkeypatch.setenv("FAKE_SERVE_SEEN", str(seen))
    monkeypatch.setattr(serve_driver, "entry_point", lambda name: str(script))
    monkeypatch.setattr(serve_driver, "_holders", lambda world: set())
    return seen


def _requests_with_image(tmp_path):
    image = tmp_path / "merlion.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    boot = gates.BootSpec(items=(gates.BootPrompt(prompt="What is it?", images=(str(image),)),))
    return sg.serve_requests(_task(tmp_path), sg.ServeSpec(max_tokens=4), boot), image


def test_the_driver_serves_asks_and_stops(tmp_path, fake_serve, capsys):
    requests, image = _requests_with_image(tmp_path)
    spec = {
        "model": "/ckpt/model",
        "serve_args": ["--config", "o.yaml"],
        "world_size": 1,
        "server_log": str(tmp_path / "server.log"),
        "boot_timeout": 60,
        "request_timeout": 30,
        "requests": requests,
    }
    report = serve_driver.run(spec)
    passed, detail, _ = sg.judge(capsys.readouterr().out, requests)
    assert passed, detail
    assert report["healthy"] and report["returncode"] is not None
    received = [json.loads(line) for line in fake_serve.read_text().splitlines()]
    assert [r["path"] for r in received] == ["/v1/completions", "/v1/chat/completions"]
    url = received[1]["body"]["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(url.split(",", 1)[1]) == image.read_bytes()


def test_the_driver_reports_a_server_that_dies_while_starting(
    tmp_path, fake_serve, capsys, monkeypatch
):
    monkeypatch.setenv("FAKE_SERVE_MODE", "crash")
    spec = {
        "model": "/ckpt/model",
        "world_size": 1,
        "server_log": str(tmp_path / "server.log"),
        "boot_timeout": 60,
        "requests": [_REQUESTS[0]],
    }
    report = serve_driver.run(spec)
    assert not report["healthy"] and "get_preferred_media_io_kwargs" in report["cause"]
    passed, detail, _ = sg.judge(capsys.readouterr().out, [_REQUESTS[0]])
    assert not passed and "exited with 1 during boot" in detail


# ------------------------------------------------------------------ the contract


def test_the_contract_owes_no_serve_check_unless_the_spec_asks(tmp_path):
    task = _task(tmp_path)
    off = c.compile_contract(task, world_size=4)
    assert not [check for check in off.checks if check.gate == sg.GATE_NAME]
    disabled = _task(tmp_path, serve={"enabled": False})
    assert c.compile_contract(disabled, world_size=4) == off


def test_turned_on_each_unit_owes_it_once_and_delivery_at_one_rank(tmp_path):
    contract = c.compile_contract(_task(tmp_path, serve={"enabled": True}), world_size=4)
    serve = [(check.key, check.tier) for check in contract.checks if check.gate == sg.GATE_NAME]
    assert serve == [
        ("core:serve@tep4", c.TIER_INNER),
        ("mtp:serve@tep4", c.TIER_INNER),
        ("multimodal:serve@tep4", c.TIER_INNER),
        ("multimodal:serve@tp1", c.TIER_DELIVERY),
    ]


def test_a_resumed_spec_without_the_switch_keeps_every_check_and_its_equivalence(tmp_path):
    """Turning the gate on adds serve checks and changes no other check, sharing keys included."""
    before = c.compile_contract(_task(tmp_path), world_size=4)
    after = c.compile_contract(_task(tmp_path, serve={"enabled": True}), world_size=4)
    kept = [check for check in after.checks if check.gate != sg.GATE_NAME]
    assert [(k.key, k.tier, k.equivalence) for k in kept] == [
        (k.key, k.tier, k.equivalence) for k in before.checks
    ]


# ------------------------------------------------------------------ the schema and the CLI


def test_the_switch_is_a_closed_validated_block():
    assert "serve" in ACCURACY_KEYS and "enabled" in SERVE_KEYS
    errors: list[str] = []
    _validate_accuracy(
        {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "serve": {"enabled": "yes", "max_tokens": 0, "prompt": " ", "bogus": 1},
        },
        errors,
    )
    joined = "\n".join(errors)
    for key in ("serve.enabled", "serve.max_tokens", "serve.prompt", "serve.bogus"):
        assert f"accuracy.{key}" in joined
    errors = []
    _validate_accuracy(
        {"benchmark": "gsm8k", "reference_yaml": "x.yaml", "serve": {"enabled": True}}, errors
    )
    assert not [e for e in errors if "serve" in e]
    errors = []
    _validate_accuracy({"benchmark": "gsm8k", "reference_yaml": "x.yaml", "serve": True}, errors)
    assert "'accuracy.serve' must be a mapping" in errors


def _task_file(tmp_path):
    (tmp_path / "repo").mkdir(exist_ok=True)
    (tmp_path / "ckpt").mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(tmp_path / "ckpt"),
        "trtllm_repo_path": str(tmp_path / "repo"),
        "arch": "sm_100",
        "parallel": "unified",
        "members": ["tp2", "tep2"],
        "features": ["mtp"],
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml", "model_key": "M"},
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_worker_runs_it_with_the_gate_command(tmp_path, monkeypatch, capsys):
    seen = []

    def fake_runner_for(executor, **kwargs):
        def run(command, env=None, timeout=0):
            seen.append(command)
            return CommandResult(returncode=0, stdout=_answer(0, " Paris.") + _server(boot_s=1))

        return run

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    assert "serve" in gate_cli.GATE_NAMES
    argv = ["serve", "--task", str(_task_file(tmp_path)), "--unit", "mtp", "--member", "tep2"]
    code = gate_cli.main([*argv, "--workdir", str(tmp_path / "w"), "--timeout", "120"])
    assert code == 0, capsys.readouterr()
    assert capsys.readouterr().out.startswith("[PASS] serve@tep2")
    spec = _driver_spec(seen[0])
    assert spec["boot_timeout"] == 120 and spec["world_size"] == 2
    assert spec["serve_args"][2:4] == ["--tp_size=2", "--ep_size=2"]
    assert Path(spec["server_log"]).parent == (tmp_path / "w" / "mtp").resolve()


def test_the_orchestrator_runs_a_serve_check_through_the_serve_gate(tmp_path, monkeypatch):
    """A `serve` contract check reaches run_serve_gate, not the witness dispatch."""
    from types import SimpleNamespace

    from agent_flow.workflows.trtllm_modeling_v2.checks import serve_gate as sg
    from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.planned import PlannedStage

    calls: list[dict] = []
    verdict = SimpleNamespace(passed=True, summary=lambda: "[PASS] serve@tp1")

    def fake_serve(env, **kwargs):
        calls.append(kwargs)
        return verdict

    monkeypatch.setattr(sg, "run_serve_gate", fake_serve)
    monkeypatch.setattr(
        "agent_flow.workflows.trtllm_modeling_v2.checks.gates.resolve_boot_spec",
        lambda *args: "boot-spec",
    )

    class _Stage(PlannedStage):
        def __init__(self):
            self.workspace = tmp_path
            self.boot_spec_path = None
            self.gate_log = SimpleNamespace(record=lambda result: None)
            self.journal = SimpleNamespace(event=lambda kind, text: None)

        def _spec_for(self, unit):
            return {"accuracy": {}}, ["mtp"]

        def _member_env(self, member):
            return "env"

        def _member_runner(self, member):
            return "runner"

        def _gate_until_verdict(self, run, log):
            return run()

        def _run_witness(self, unit, member, log):
            raise AssertionError("a serve check went to the witness dispatch")

    check = contract_mod.Check(unit="core", gate=sg.GATE_NAME, member="tp1", tier="inner")
    assert _Stage()._run_contract_check(check, log=None) is verdict
    (call,) = calls
    assert call["unit"] == "core" and call["member"] == "tp1"
    assert call["boot"] == "boot-spec" and call["features"] == ["mtp"]
