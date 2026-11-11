# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the trtllm_modeling_v2 tests.

``repo`` is a minimal TensorRT-LLM checkout holding one merged target,
``checkpoint`` a HuggingFace checkpoint directory for it, and ``task_file`` a
task spec that points at both.
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
import sys

import pytest
import yaml

from ._helpers import _build_repo

SCRATCH = "/scratch/mv2flow-run-tmp"


@pytest.fixture
def repo(tmp_path):
    return _build_repo(tmp_path)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "gpt-oss-120b"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"architectures": ["GptOssForCausalLM"]}))
    return path


@pytest.fixture
def task_file(tmp_path, repo, checkpoint):
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "agents": {"defaults": {"backend": "claude-code", "model": "claude-test-model"}},
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _runs_start_past_the_survey(monkeypatch):
    """A scripted run here starts from the workspace's vocabulary; a survey test turns it on."""
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", False)


@pytest.fixture(autouse=True)
def _git_stays_under_the_base_temp(tmp_path_factory, monkeypatch):
    """Stop git's repository search at pytest's base temp directory.

    A fixture tree that is not a checkout has to read as not one, even when
    the base temp directory sits inside some other checkout.
    """
    monkeypatch.setenv("GIT_blocked_DIRECTORIES", str(tmp_path_factory.getbasetemp()))


@pytest.fixture(autouse=True)
def _no_parent_codex_session(monkeypatch, tmp_path):
    """Offline tests never touch the developer's Codex state or live session.

    ``$CODEX_HOME`` and ``$TMPDIR`` are the test's own.
    """
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CODEX_SQLITE_HOME", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "user-codex-home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _no_agent_probe(monkeypatch):
    """Keep preflight's agent probe from starting a real agent turn; its own tests unset this."""
    monkeypatch.setenv("MV2FLOW_SKIP_AGENT_PROBE", "1")


@pytest.fixture(autouse=True)
def _gsm8k_on_disk(monkeypatch, tmp_path_factory):
    """The one dataset the checks know is under the models root, as on a prepared host."""
    root = tmp_path_factory.mktemp("models")
    split = root / "datasets" / "openai" / "gsm8k" / "main"
    split.mkdir(parents=True)
    (split / "test-00000-of-00001.parquet").write_text("rows")
    monkeypatch.setenv("LLM_MODELS_ROOT", str(root))


# A stand-in for `trtllm-serve`: it binds a port, reports it, answers /health, completions
# and chat completions, and writes down the command line and environment it was started
# with. As a real server's ranks do under the rank hook, it prints what each rank runs: the
# options it was given (FAKE_RANKS=0: nothing; FAKE_DTYPE: another KV dtype than asked),
# from the directory it was started in. A prompt it has seen resumes from its cache
# (FAKE_REUSE=0: never). FAKE_SERVE_EXIT: it refuses to start, its error line far above
# the end of a backtrace; FAKE_SERVE_HANG: it never
# reports an address; FAKE_DIE_AFTER_BIND=<ctx|gen>: that side dies once the proxy is
# being started;
# FAKE_NO_IMAGES: it answers a chat completion with a 400 whose body says why;
# FAKE_UNFINISHED: no finish reason; FAKE_FINISH: that one. A chat completion's answer
# says how many images came with it when that is not one.
_FAKE_SERVE = """#!{python}
import http.server, json, os, signal, sys, threading, time

argv = sys.argv[1:]
report = argv[argv.index("--report_addr") + 1]
if os.environ.get("FAKE_SERVE_EXIT"):
    print("fake: refused to start")
    print("E1111 09:00:00.000000 77 nixl.cpp:9] Invalid UCX_TLS=tcp for NIXL UCX backend")
    print("RuntimeError: Failed to create NIXL backend")
    for frame in range(40):
        print("#%d 0x0000ffff in frame" % frame)
    sys.exit(3)
if os.environ.get("FAKE_SERVE_HANG"):
    signal.pause()
if "--config" in argv and os.environ.get("FAKE_RANKS", "1") == "1":
    import yaml

    options = yaml.safe_load(open(argv[argv.index("--config") + 1])) or {{}}
    if os.environ.get("FAKE_DTYPE"):
        options.setdefault("kv_cache_config", {{}})["dtype"] = os.environ["FAKE_DTYPE"]
    # With the switch off the engine builds its stock class, as a control run does.
    stock = os.environ.get("TRTLLM_MODELING_V2") == "off"
    package = "models.fake" if stock else "_experimental.modeling_v2.models.fake.model"
    target = "tensorrt_llm._torch." + package + ".Model"
    for rank in range(int(options.get("tensor_parallel_size") or 1)):
        record = {{
            "rank": rank,
            "options": options,
            "target": target,
            "origin": os.path.join(os.getcwd(), "tensorrt_llm", "__init__.py"),
        }}
        print("MV2FLOW_RANK_ENGINE " + json.dumps(record), flush=True)
seen = []


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({{"status": "ok"}})

    def do_POST(self):
        asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        finish = None if os.environ.get("FAKE_UNFINISHED") else os.environ.get("FAKE_FINISH", "stop")
        if self.path.endswith("/chat/completions"):
            if os.environ.get("FAKE_NO_IMAGES"):
                why = "this model takes no image; " + "set a vision tower; " * 20 + "END"
                return self._send({{"error": why}}, 400)
            text, *images = asked["messages"][0]["content"]
            for image in images:
                assert image["image_url"]["url"].startswith("data:image/png;base64,")
            said = "A lion statue, asked: " + text["text"]
            said += "" if len(images) == 1 else " (%d images)" % len(images)
            message = {{"role": "assistant", "content": said}}
            return self._send({{"choices": [{{"message": message, "finish_reason": finish}}]}})
        again = asked["prompt"] in seen and os.environ.get("FAKE_REUSE", "1") == "1"
        seen.append(asked["prompt"])
        usage = {{"prompt_tokens_details": {{"cached_tokens": 448 if again else 0}}}}
        text = " Paris, after: " + asked["prompt"]
        self._send({{"choices": [{{"text": text, "finish_reason": finish}}], "usage": usage}})


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(report + ".started.json", "w") as out:
    json.dump({{"argv": argv, "env": dict(os.environ)}}, out)
with open(report + ".tmp", "w") as out:
    out.write("127.0.0.1:%d\\n" % server.server_address[1])
os.replace(report + ".tmp", report)
died = {{"ctx": "context", "gen": "generation"}}.get(os.environ.get("FAKE_DIE_AFTER_BIND"))
if died and died in argv:
    while not os.path.exists(os.path.join(os.path.dirname(report), "proxy.yaml")):
        time.sleep(0.01)
    print("fake: %s worker died" % died)
    sys.exit(4)
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
server.serve_forever()
"""
_FAKE_SMI = '#!/bin/bash\n[ -f "$FAKE_SMI" ] && cat "$FAKE_SMI"\nexit 0\n'
# A stand-in for `mpirun`: one process, the command after the LLM API's launcher.
_FAKE_MPIRUN = (
    '#!/bin/bash\nwhile [ $# -gt 0 ] && [ "$1" != trtllm-llmapi-launch ]; do shift; done\n'
    'shift\nexec "$@"\n'
)


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A PATH with the stand-in `trtllm-serve` and an `nvidia-smi` that reads a file."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (
        ("trtllm-serve", _FAKE_SERVE.format(python=sys.executable)),
        ("nvidia-smi", _FAKE_SMI),
        ("mpirun", _FAKE_MPIRUN),
    ):
        path = bin_dir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SMI", str(tmp_path / "holders"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    from agent_flow.workflows.trtllm_modeling_v2.checks import servers

    monkeypatch.setattr(servers, "RELEASE_TIMEOUT_S", 1)
    monkeypatch.setattr(servers, "STOP_GRACE_S", 10)
    yield tmp_path
    # A test that failed with servers up leaves none behind on a shared host.
    for record in tmp_path.rglob(servers.OWNED_FILENAME):
        with contextlib.suppress(Exception):
            servers.clean_owned(record.parent, budget=5)
