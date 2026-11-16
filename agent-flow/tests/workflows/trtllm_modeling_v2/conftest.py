# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the trtllm_modeling_v2 tests.

``repo`` is a minimal TensorRT-LLM checkout holding one merged target,
``checkpoint`` a HuggingFace checkpoint directory for it, and ``task_file`` a
task spec that points at both.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from ._helpers import _build_repo

SCRATCH = "/scratch/mv2flow-run-tmp"
#: One GPU's native options, as `intake derive` writes them.
TP1_OPTIONS = {
    "cuda_graph_config": {"enable_padding": True},
    "enable_chunked_prefill": False,
    "kv_cache_config": {"enable_block_reuse": False},
    "speculative_config": None,
    "max_seq_len": 5120,
    "max_batch_size": 64,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "moe_expert_parallel_size": 1,
    "enable_attention_dp": False,
}


@pytest.fixture(autouse=True)
def _own_node_state(tmp_path, monkeypatch):
    """What the flow keeps per node is this test's own, under its directory.

    The digest cache, and the trees a run makes on "the node's own disk" when its spec states
    no place for them.
    """
    monkeypatch.setenv("MV2FLOW_DIGEST_CACHE", str(tmp_path / "node" / "digests.json"))
    monkeypatch.setenv("MV2FLOW_TREES_DIR", str(tmp_path / "node"))


@pytest.fixture(autouse=True)
def _no_stay_for_an_answer(monkeypatch):
    """A controller with everything left blocked leaves at once: no case waits out the stay."""
    from agent_flow.workflows.trtllm_modeling_v2.run import orchestrator

    monkeypatch.setattr(orchestrator, "ANSWER_WAIT_S", 0.0)


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
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b", "parallel": "tp1"},
        # One deployment, the first: its session is what the run's contract owes.
        "deployments": [{"name": "core", "options": dict(TP1_OPTIONS)}],
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
            "tolerance": 5.0,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _scripted_runs_start_without_a_survey(monkeypatch):
    """A scripted run here starts from the deployments' own tasks; a test that wants the survey says so."""
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", False)


@pytest.fixture(autouse=True)
def _git_stays_under_the_base_temp(tmp_path_factory, monkeypatch):
    """Stop git's repository search at pytest's base temp directory.

    A fixture tree that is not a checkout has to read as not one, even when
    the base temp directory sits inside some other checkout.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path_factory.getbasetemp()))


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
def _no_dataset_downloads(monkeypatch):
    """Keep the run's dataset staging off the network; ``test_datasets.py`` calls it itself."""
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import datasets

    monkeypatch.setattr(datasets, "start", lambda *args, **kwargs: None)


# A stand-in for `trtllm-serve`: it binds a port, reports it, answers /health, completions
# (streamed too) and chat completions, and writes down the command line and environment it
# was started with. As a real server's ranks do under the rank hook, it prints what each
# rank runs: the options it was given, the directory it was started in, its host and the
# devices it was shown. What a test turns on through the environment:
#   FAKE_RANKS=0            the ranks print nothing
#   FAKE_MERGED_RANKS=1     two ranks print at once: their records arrive as one broken line
#   FAKE_NO_RANK_FILES=1    the ranks keep no record file of their own (a hook that only prints)
#   FAKE_DTYPE=<dtype>      the ranks run another KV dtype than asked
#   FAKE_REUSE=0            a prompt seen before resumes nothing from the cache
#   FAKE_SERVE_EXIT=1       it refuses to start, its error line far above the end of a backtrace
#   FAKE_NO_IMAGES=1        a chat completion with an image is answered with a 400
#   FAKE_FINISH=<reason>    answers end with that finish reason
#   FAKE_NO_HOP=1           the proxy's answers name no state transfer on the generation side
#   FAKE_NO_METRICS=1       the proxy returns no metrics header, asked or not
#   FAKE_ACCEPTED=<n>       draft tokens accepted per request, of 20 verify steps
# As `trtllm-serve disaggregated` it is the proxy, and answers as a server does.
_FAKE_SERVE = """#!{python}
import http.server, json, os, signal, socket, sys, threading

argv = sys.argv[1:]
proxy = argv[:1] == ["disaggregated"]
report = argv[argv.index("--report_addr") + 1]
if os.environ.get("FAKE_SERVE_EXIT") and not proxy:
    print("fake: refused to start")
    print("E1111 09:00:00.000000 77 nixl.cpp:9] Invalid UCX_TLS=tcp for NIXL UCX backend")
    print("RuntimeError: Failed to create NIXL backend")
    for frame in range(40):
        print("#%d 0x0000ffff in frame" % frame)
    sys.exit(3)
options = {{}}
if "--config" in argv:
    import yaml

    options = yaml.safe_load(open(argv[argv.index("--config") + 1])) or {{}}
if not proxy and os.environ.get("FAKE_RANKS", "1") == "1":
    shown = dict(options)
    if os.environ.get("FAKE_DTYPE"):
        shown["kv_cache_config"] = dict(shown.get("kv_cache_config") or {{}}, dtype=os.environ["FAKE_DTYPE"])
    target = "tensorrt_llm._torch._experimental.modeling_v2.models.fake.model.Model"
    # Started by a step, this process is one rank of the world and speaks for itself alone;
    # started by this host's `mpirun` stand-in, it speaks for every rank.
    stepped = os.environ.get("SLURM_PROCID")
    world = [int(stepped)] if stepped else range(int(options.get("tensor_parallel_size") or 1))
    said = []
    for rank in world:
        record = {{
            "rank": rank,
            "options": shown,
            "target": target,
            "origin": os.path.join(os.getcwd(), "tensorrt_llm", "__init__.py"),
            "host": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
            "devices": [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d],
        }}
        said.append(record)
    kept = os.environ.get("MV2FLOW_RANK_RECORDS")
    if kept and not os.environ.get("FAKE_NO_RANK_FILES"):
        # As the rank hook: each rank's record in a file of its own.
        instance = os.environ.get("MV2FLOW_ROLE") or "server"
        for record in said:
            name = "rank-engine.%s.%s.jsonl" % (instance, record["rank"])
            with open(os.path.join(kept, name), "a") as out:
                out.write(json.dumps(record) + "\\n")
    lines = ["MV2FLOW_RANK_ENGINE " + json.dumps(record) for record in said]
    if os.environ.get("FAKE_MERGED_RANKS") and len(lines) > 1:
        # As two ranks that print at once to the stream they share: one broken line.
        half = len(lines[0]) // 2
        lines = [lines[0][:half] + lines[1] + lines[0][half:], *lines[2:]]
    for line in lines:
        print(line, flush=True)
judged = os.environ.get("SLURM_PROCID") or ("0" if os.environ.get("MV2FLOW_RESOLVE_LOG") else "")
if not proxy and judged and os.environ.get("FAKE_SILENT_RANK") != judged:
    try:
        # The origin of a rank: where it imported the package from. A rank of a step, or a
        # server of a pool's job on one node (its watcher judges where it came from).
        import tensorrt_llm
    except ImportError:
        pass
if not proxy and os.environ.get("SLURM_PROCID", "0") != "0":
    # A rank past the first serves nothing: it is a worker of the world until it is ended.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        signal.pause()
seen = []
crossed = []
lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload, status=200, headers=None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {{}}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _metrics(self):
        # As the engine's proxy answers a request sent with `X-TRTLLM-return-metrics: 1`:
        # the timestamps of each phase, the generation worker's transfer among them.
        if not proxy or self.headers.get("X-TRTLLM-return-metrics") != "1":
            return {{}}
        if os.environ.get("FAKE_NO_METRICS"):
            return {{}}
        items = ["disagg-start;ts=10.000000000", "ctx-start;ts=10.100000000"]
        items += ["ctx-end;ts=10.300000000", "gen-start;ts=10.400000000"]
        if not os.environ.get("FAKE_NO_HOP"):
            items += ["gen-kv-start;ts=10.200000000", "gen-kv-end;ts=10.350000000"]
        items += ["gen-end;ts=10.900000000"]
        return {{"X-TRTLLM-Start-End-Time": ", ".join(items)}}

    def do_GET(self):
        self._send({{"status": "ok"}})

    def do_POST(self):
        asked = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        finish = os.environ.get("FAKE_FINISH", "stop")
        with lock:
            crossed.append(self.path)
        if self.path.endswith("/chat/completions"):
            content = asked["messages"][0]["content"]
            if isinstance(content, str):
                text, images = content, []
            else:
                text, images = content[0]["text"], content[1:]
            if images and os.environ.get("FAKE_NO_IMAGES"):
                return self._send({{"error": "this model takes no image"}}, 400)
            for image in images:
                assert image["image_url"]["url"].startswith("data:image/png;base64,")
            said = ("A lion statue, asked: " if images else "Paris, asked: ") + text
            message = {{"role": "assistant", "content": said}}
            return self._send({{"choices": [{{"message": message, "finish_reason": finish}}]}})
        prompt = asked["prompt"]
        length = len(prompt) if isinstance(prompt, list) else len(str(prompt).split())
        key = json.dumps(prompt)
        with lock:
            repeated = key in seen
            again = repeated and os.environ.get("FAKE_REUSE", "1") == "1"
            seen.append(key)
        text = " Paris, after: " + (str(prompt) if isinstance(prompt, str) else "tokens")
        if asked.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for part, end in ((" Paris", None), (", again", finish)):
                event = {{"choices": [{{"text": part, "finish_reason": end}}]}}
                self.wfile.write(("data: " + json.dumps(event) + "\\n\\n").encode())
            self.wfile.write(b"data: [DONE]\\n\\n")
            return
        tokens = int(asked.get("max_tokens") or 1)
        choice = {{"text": text, "finish_reason": finish}}
        if os.environ.get("FAKE_ACCEPTED") is not None:
            accepted = int(os.environ["FAKE_ACCEPTED"])
            choice["speculative_decoding"] = {{
                "num_spec_steps": 20,
                "total_accepted_draft_tokens": accepted,
                "total_draft_tokens": 60,
            }}
        usage = {{
            "prompt_tokens": length,
            "completion_tokens": tokens,
            "prompt_tokens_details": {{"cached_tokens": 448 if again else 0}},
        }}
        self._send({{"choices": [choice], "usage": usage}}, headers=self._metrics())


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(report + ".started.json", "w") as out:
    json.dump({{"argv": argv, "env": dict(os.environ)}}, out)
with open(report + ".tmp", "w") as out:
    out.write("127.0.0.1:%d\\n" % server.server_address[1])
os.replace(report + ".tmp", report)
signal.signal(signal.SIGTERM, lambda *a: os._exit(0))
server.serve_forever()
"""
_FAKE_SMI = '#!/bin/bash\n[ -f "$FAKE_SMI" ] && cat "$FAKE_SMI"\nexit 0\n'
# A stand-in for `mpirun`: one process, the command after the LLM API's launcher.
_FAKE_MPIRUN = (
    '#!/bin/bash\nwhile [ $# -gt 0 ] && [ "${1##*/}" != trtllm-llmapi-launch ]; do shift; done\n'
    'shift\nexec "$@"\n'
)
#: What a step's rank runs the server under, past one rank: here it only hands on.
_FAKE_LAUNCH = '#!/bin/bash\nexec "$@"\n'
#: A stand-in for `srun`, as the flow's own step line calls it: the command once per rank,
#: each told its rank and its node as a step tells it; a line per call in $FAKE_STEP_LOG. As
#: a step into a container, its shell is a fresh one: nothing of what the pool's watcher gave
#: the job for its origin records, and no PYTHONPATH, reaches the ranks by inheritance.
_FAKE_STEP = """#!{python}
import os, signal, subprocess, sys

command, asked = sys.argv[1:], {{}}
while command and command[0].startswith("--"):
    key, _, value = command.pop(0)[2:].partition("=")
    asked[key] = value
ranks, nodes, hosts = int(asked["ntasks"]), int(asked["nodes"]), asked["nodelist"].split(",")
order = [host for host in hosts for _ in range(ranks // nodes)]
if os.environ.get("FAKE_STEP_LOG"):
    with open(os.environ["FAKE_STEP_LOG"], "a") as log:
        said = dict(ranks=ranks, nodes=nodes, hosts=hosts, order=order)
        log.write(repr((said, command[-2:])) + "\\n")
started = []
fresh = dict(os.environ)
for name in ("PYTHONPATH", "MV2FLOW_RESOLVE_LOG", "MV2FLOW_RESOLVE_MODULES", "MV2FLOW_TREE"):
    fresh.pop(name, None)
for name in ("MV2FLOW_SPOOL", "MV2FLOW_SPOOL_JOB"):
    fresh.pop(name, None)
for rank in range(ranks):
    env = dict(fresh, SLURM_PROCID=str(rank), PMIX_RANK=str(rank), SLURM_NTASKS=str(ranks))
    env["SLURMD_NODENAME"] = order[rank]
    started.append(subprocess.Popen(command, env=env))


def end(*_):
    for process in started:
        process.terminate()


signal.signal(signal.SIGTERM, end)
codes = [process.wait() for process in started]
sys.exit(next((code for code in codes if code), 0))
"""
# A stand-in for the endpoint evaluator: it asks the endpoint once and prints a score.
_FAKE_EVAL = """#!{python}
import json, os, sys, urllib.request

if "--dataset_only" in sys.argv:
    # As the evaluator's own loader, asked before any server: the data is there or it is not.
    if os.environ.get("FAKE_NO_DATA"):
        print("endpoint-eval: the evaluator cannot load its data: " + os.environ["FAKE_NO_DATA"])
        sys.exit(4)
    sys.exit(0)
url = sys.argv[sys.argv.index("--url") + 1]
body = json.dumps({{"prompt": "2 + 2 =", "max_tokens": 4}}).encode()
request = urllib.request.Request(url + "/v1/completions", data=body)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
with opener.open(request, timeout=30) as reply:
    json.loads(reply.read())
print("endpoint-eval: argv " + json.dumps(sys.argv[1:]))
if os.environ.get("FAKE_EVAL_EXIT"):
    sys.exit(int(os.environ["FAKE_EVAL_EXIT"]))
print("lm-eval gsm8k average accuracy: " + os.environ.get("FAKE_SCORE", "41.70"))
"""


#: What the engine's metrics module defines of a request's own metrics, as the tree has it.
ENGINE_METRICS_SOURCE = """
START_END_TIME_HEADER = "X-TRTLLM-Start-End-Time"
RETURN_METRICS_HEADER = "X-TRTLLM-return-metrics"


def _jsonl_perf_metrics(phase_record):
    # As the engine has it: the two fields listed side by side, which name no header item.
    for name in ("kv_cache_transfer_start", "kv_cache_transfer_end"):
        phase_record.pop(name, None)


def build_metrics_headers(records):
    ranges = (("kv_transfer", "kv_cache_transfer_start", "kv_cache_transfer_end"),)
    for name, field in (
        ("start", "arrival_time"),
        ("kv-start", "kv_cache_transfer_start"),
        ("kv-end", "kv_cache_transfer_end"),
    ):
        pass
"""
#: Where the stand-in tree keeps them: a place no code of the flow could name.
ENGINE_METRICS_FILE = "tensorrt_llm/web/request_times.py"


def engine_definitions(tree, means: str | None = "headers") -> None:
    """Write into ``tree`` what an engine defines of observing a state transfer, or nothing."""
    source = Path(tree) / ENGINE_METRICS_FILE
    source.parent.mkdir(parents=True, exist_ok=True)
    (Path(tree) / "tensorrt_llm" / "__init__.py").touch()
    source.unlink(missing_ok=True)
    if means == "headers":
        source.write_text(ENGINE_METRICS_SOURCE)


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A PATH with stand-ins for `trtllm-serve`, `mpirun` and `nvidia-smi`, and a fake evaluator.

    Returns the directory; ``site / "eval.py"`` is the evaluator a session's benchmark runs.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (
        ("trtllm-serve", _FAKE_SERVE.format(python=sys.executable)),
        ("nvidia-smi", _FAKE_SMI),
        ("mpirun", _FAKE_MPIRUN),
        ("trtllm-llmapi-launch", _FAKE_LAUNCH),
        ("srun", _FAKE_STEP.format(python=sys.executable)),
        ("eval.py", _FAKE_EVAL.format(python=sys.executable)),
    ):
        path = bin_dir / name
        path.write_text(text)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SMI", str(tmp_path / "holders"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    from agent_flow.workflows.trtllm_modeling_v2.checks import session

    monkeypatch.setattr(session, "ENDPOINT_EVAL", bin_dir / "eval.py")
    return bin_dir


# Fixtures kept in modules of their own. Imported last: those modules take this one's stand-ins
# when a fixture runs, never when they are imported.
from ._design import run_session, task  # noqa: E402,F401
from ._two_nodes import two_nodes  # noqa: E402,F401
