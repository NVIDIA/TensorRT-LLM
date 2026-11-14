# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The trace, replay, trial-build, session-runner and pi kit tools.

CPU only: no TensorRT-LLM import, no docker.
"""

import asyncio
import http.server
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import aiohttp
import pytest
import yaml
from aiohttp import web
from aiohttp.test_utils import TestServer

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
for _path in (SCRIPTS, os.path.join(SCRIPTS, "pi_kit", "bin"),
              os.path.join(SCRIPTS, "pi_kit", "measure")):  # fmt: skip
    sys.path.insert(0, _path)

import build_replay_corpus  # noqa: E402
import pikit  # noqa: E402
import probe_recall  # noqa: E402
import replay_requests  # noqa: E402
import summarize  # noqa: E402

# --------------------------------------------------------------------------- replay corpus


def _transform(body, reasoning):
    args = types.SimpleNamespace(model=None, reasoning=reasoning, max_output_tokens=0,
                                 route="/v1/chat/completions", stream="keep")  # fmt: skip
    return build_replay_corpus.transform(body, args)


def test_strip_encrypted_keeps_the_visible_reasoning():
    """Only `drop` removes reasoning text; `strip-encrypted` removes what a server issued."""
    chat = {"messages": [{"role": "user", "content": "q"},
                         {"role": "assistant", "content": "a", "reasoning_content": "r",
                          "reasoning": "r"}]}  # fmt: skip
    body, done = _transform(chat, "strip-encrypted")
    assert body == chat and done == []
    body, done = _transform(chat, "drop")
    assert body["messages"][1] == {"role": "assistant", "content": "a"}
    assert done == ["reasoning:drop"]

    messages = {"messages": [{"role": "assistant", "content": [
        {"type": "thinking", "thinking": "r", "signature": "s"},
        {"type": "redacted_thinking", "data": "d"}, {"type": "text", "text": "a"}]}]}  # fmt: skip
    body, done = _transform(messages, "strip-encrypted")
    assert body["messages"][0]["content"] == [{"type": "thinking", "thinking": "r"},
                                              {"type": "text", "text": "a"}]  # fmt: skip
    assert done == ["reasoning:strip-encrypted"]


def test_the_corpus_is_readable_by_its_owner_only(tmp_path):
    """The corpus holds whole request bodies: 0600, whether new or overwritten."""
    shard = tmp_path / "requests-1.jsonl"
    shard.write_text(json.dumps({"event": "request", "trace_id": "t1", "route": "/v1/responses",
                                 "body": {"model": "m", "input": "go"}}) + "\n")  # fmt: skip
    out = tmp_path / "corpus.jsonl"
    out.write_text("")
    os.chmod(out, 0o644)
    result = subprocess.run([sys.executable, os.path.join(SCRIPTS, "build_replay_corpus.py"),
                             "--out", str(out), "--manifest", str(tmp_path / "manifest.json"),
                             str(shard)], capture_output=True, text=True)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    assert len(out.read_text().splitlines()) == 1


def test_the_corpus_leaves_out_what_the_credential_scan_flags(tmp_path):
    """A request the scan flags, an environment dump included, never reaches the corpus."""
    dump = "HOME=/root\nOPENAI_API_KEY=" + "FAKE" * 6 + "\nPATH=/bin"  # obviously not a key
    tool_turn = [{"type": "message", "role": "user", "content": "go"},
                 {"type": "function_call_output", "call_id": "c1", "output": dump}]  # fmt: skip
    shard = tmp_path / "requests-1.jsonl"
    shard.write_text("".join(json.dumps({"event": "request", "trace_id": f"t{i}",
                                         "route": "/v1/responses",
                                         "body": {"model": "m", "input": body}}) + "\n"
                             for i, body in enumerate(["go", tool_turn])))  # fmt: skip
    out, manifest = tmp_path / "corpus.jsonl", tmp_path / "manifest.json"
    result = subprocess.run([sys.executable, os.path.join(SCRIPTS, "build_replay_corpus.py"),
                             "--out", str(out), "--manifest", str(manifest), str(shard)],
                            capture_output=True, text=True)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "FAKE" not in out.read_text() and len(out.read_text().splitlines()) == 1
    assert json.loads(manifest.read_text())["counts"]["drop:credential"] == 1


def test_a_corpus_cut_while_it_is_written_leaves_no_corpus(tmp_path, monkeypatch):
    """A shard that fails during the second pass leaves neither a corpus nor a manifest."""
    from tracelib import ShardError

    shard = tmp_path / "requests-1.jsonl"
    shard.write_text("".join(json.dumps({"event": "request", "trace_id": f"t{i}", "route": "/v1/responses",
                                         "body": {"model": "m", "input": f"go {i}"}}) + "\n"
                             for i in range(3)))  # fmt: skip
    real, seen = build_replay_corpus.iter_lines, []

    def cut_on_second_pass(path):
        seen.append(path)
        for number, (line_no, raw) in enumerate(real(path)):
            if len(seen) > 1 and number == 1:
                raise ShardError(f"{path}: cut")
            yield line_no, raw

    monkeypatch.setattr(build_replay_corpus, "iter_lines", cut_on_second_pass)
    out, manifest = tmp_path / "corpus.jsonl", tmp_path / "manifest.json"
    monkeypatch.setattr(sys, "argv", ["build_replay_corpus.py", "--out", str(out), "--manifest",
                                      str(manifest), str(shard)])  # fmt: skip
    with pytest.raises(ShardError):
        build_replay_corpus.main()
    assert len(seen) == 2
    assert not out.exists() and not manifest.exists()
    assert not os.path.exists(str(out) + ".partial")


@pytest.mark.parametrize(
    "script", ["build_replay_corpus.py", "summarize_capture.py", "validate/harness_signals.py"]
)
def test_a_cut_shard_stops_a_trace_tool_with_its_name(tmp_path, script):
    """A truncated .zst shard ends the run with a message naming it, and nothing is written."""
    import zstandard

    record = {"event": "request", "route": "/v1/responses", "body": {"model": "m", "input": "go"}}
    data = "".join(json.dumps(dict(record, trace_id=f"t{i}")) + "\n" for i in range(50))
    packed = zstandard.ZstdCompressor().compress(data.encode())
    shard = tmp_path / "requests-1.jsonl.zst"
    shard.write_bytes(packed[: len(packed) // 2])
    out = tmp_path / "out.json"
    argv = [sys.executable, os.path.join(SCRIPTS, script), "--out", str(out), str(shard)]
    if script == "build_replay_corpus.py":
        argv[4:4] = ["--manifest", str(tmp_path / "manifest.json")]

    result = subprocess.run(argv, capture_output=True, text=True)

    assert result.returncode != 0 and str(shard) in result.stderr, result.stdout + result.stderr
    assert "Traceback" not in result.stderr and not out.exists()


def test_a_count_tokens_corpus_cut_while_it_is_written_is_not_left(tmp_path):
    """Entries written before a shard's damage leave no corpus; a whole run's is owner-only."""
    import gzip

    def request(i):
        body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": f"go {i}"}]}
        return {"event": "request", "trace_id": f"t{i}", "route": "/v1/messages", "body": body}

    packed = gzip.compress("".join(json.dumps(request(i)) + "\n" for i in range(200)).encode())
    whole = tmp_path / "whole" / "requests-1.jsonl.gz"
    cut = tmp_path / "cut" / "requests-1.jsonl.gz"
    for shard, data in ((whole, packed), (cut, packed[: len(packed) // 2])):
        shard.parent.mkdir()
        shard.write_bytes(data)
    script = os.path.join(SCRIPTS, "count_tokens_corpus.py")
    out = tmp_path / "corpus.jsonl"

    result = subprocess.run([sys.executable, script, "--out", str(out), str(cut)],
                            capture_output=True, text=True)  # fmt: skip
    assert result.returncode != 0 and str(cut) in result.stderr, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists() and not os.path.exists(str(out) + ".partial")

    result = subprocess.run([sys.executable, script, "--out", str(out), str(whole)],
                            capture_output=True, text=True)  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert len(out.read_text().splitlines()) == 200
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600


# --------------------------------------------------------------------------- replay

EVENT = b'event: response.created\ndata: {"type": "response.created"}\n\n'


async def _replay(events, first_event_timeout, timeout, out_dir=None, event=EVENT, build_id=None):
    """Replay one request against a stream that sends ``events`` events, then stays silent."""
    release = asyncio.Event()

    async def handler(request):
        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        for _ in range(events):
            await resp.write(event)
        if events > 1:
            await resp.write_eof()
            return resp
        await release.wait()
        return resp

    app = web.Application()
    app.router.add_route("POST", "/v1/responses", handler)
    async with TestServer(app) as server:
        args = types.SimpleNamespace(
            base_url=str(server.make_url("")), first_event_timeout=first_event_timeout,
            timeout=timeout, out_dir=out_dir, corpus=None, limit=0, header=None,
            api_key_env=None, dry_run=False, concurrency=1, build_id=build_id)  # fmt: skip
        entry = {"route": "/v1/responses", "body": {"stream": True}}
        try:
            if out_dir is None:
                async with aiohttp.ClientSession() as session:
                    return (await replay_requests.replay_one(session, entry, args, {}))[1]
            args.corpus = os.path.join(os.path.dirname(out_dir), "corpus.jsonl")
            with open(args.corpus, "w") as fh:
                fh.write(json.dumps(entry) + "\n")
            await replay_requests.run(args)
        finally:
            release.set()


@pytest.mark.parametrize(
    "events, first_event_timeout, timeout, status",
    [
        (0, 0.3, 5.0, "client_first_event_timeout"),  # silent from the start
        (1, 5.0, 0.5, "client_timeout"),  # cut by --timeout after the first event
        (0, 5.0, 0.3, "client_timeout"),  # --timeout is the earlier limit
    ],
    ids=["first_event", "after_first_event", "total_first"],
)
def test_a_replay_names_the_limit_that_cut_it(events, first_event_timeout, timeout, status):
    """The status names the limit that cut the reply: a cut mid-stream is no first-event stall."""
    response = asyncio.run(_replay(events, first_event_timeout, timeout))

    assert response["status"] == status
    assert ("first_event_s" in response["timing"]) == (events > 0)


def test_a_replay_finds_the_first_event_of_a_crlf_stream():
    """An SSE stream framed with CRLF has its first event, as an LF one has."""
    response = asyncio.run(_replay(2, 5.0, 5.0, event=EVENT.replace(b"\n", b"\r\n")))

    assert response["status"] == "completed" and "first_event_s" in response["timing"]


def test_a_replay_names_the_build_that_answered(tmp_path):
    """With --build-id every record names the build, so the verdict can check it (I-05)."""
    out_dir = tmp_path / "replay"
    asyncio.run(_replay(2, 5.0, 5.0, out_dir=str(out_dir), build_id="1.3.0+abc"))

    for name in ("requests-replay.jsonl", "responses-replay.jsonl"):
        (record,) = [json.loads(line) for line in (out_dir / name).read_text().splitlines()]
        assert record["build_id"] == "1.3.0+abc", name


def test_the_replay_output_is_readable_by_its_owner_only(tmp_path):
    out_dir = tmp_path / "replay"
    asyncio.run(_replay(2, 5.0, 5.0, out_dir=str(out_dir)))

    assert stat.S_IMODE(os.stat(out_dir).st_mode) == 0o700
    for name in ("requests-replay.jsonl", "responses-replay.jsonl"):
        assert stat.S_IMODE(os.stat(out_dir / name).st_mode) == 0o600, name


# --------------------------------------------------------------------------- trial build

# What `git rev-parse --local-env-vars` lists: set by a caller (a hook that runs these tests, say),
# each points git at another repository than the temporary one the test runs it in.
GIT_LOCATION_VARS = ("GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS",
                     "GIT_CONFIG_COUNT", "GIT_OBJECT_DIRECTORY", "GIT_DIR", "GIT_WORK_TREE",
                     "GIT_IMPLICIT_WORK_TREE", "GIT_GRAFT_FILE", "GIT_INDEX_FILE",
                     "GIT_NO_REPLACE_OBJECTS", "GIT_REPLACE_REF_BASE", "GIT_PREFIX",
                     "GIT_SHALLOW_FILE", "GIT_COMMON_DIR")  # fmt: skip


def _git_env():
    """The caller's environment for git in a temporary repository: its own config and identity."""
    env = {k: v for k, v in os.environ.items() if k not in GIT_LOCATION_VARS}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")  # fmt: skip
    return env


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize(
    "branch, prov_args, prov",
    [("trial-br", ["-p", "prov.txt"], "prov.txt"),
     ("user/trial", [], "trial.user_trial.provenance.txt")],
    ids=["given", "default_for_a_branch_with_a_slash"],
)  # fmt: skip
def test_a_trial_build_takes_its_paths_from_the_caller(tmp_path, branch, prov_args, prov):
    """Relative worktree and provenance paths name what they name from where the script ran."""
    env = _git_env()
    repo = tmp_path / "repo"
    sub = repo / "sub"
    sub.mkdir(parents=True)

    def git(*args, cwd=repo):
        subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (sub / "a.txt").write_text("a\n")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "feat")
    (sub / "a.txt").write_text("b\n")
    git("commit", "-q", "-am", "feat")
    git("checkout", "-q", "main")

    result = subprocess.run(
        ["bash", os.path.join(SCRIPTS, "compose_trial_build.sh"), "-m", "main", *prov_args,
         "trial", branch, "main", "feat=feat"],
        cwd=sub, env=env, capture_output=True, text=True)  # fmt: skip

    assert result.returncode == 0, result.stdout + result.stderr
    assert (sub / "trial" / "sub" / "a.txt").read_text() == "b\n"
    assert not (repo / "trial").exists()
    provenance = (sub / prov).read_text()
    assert "item feat feat head=" in provenance and f"result {branch} head=" in provenance


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_a_trial_build_applies_a_binary_change(tmp_path):
    """An item that changes a binary file is applied, not reported as a conflict."""
    env = _git_env()
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("checkout", "-q", "-b", "feat")
    (repo / "blob.bin").write_bytes(b"\x00\xff\xfe\xfd")
    git("commit", "-q", "-am", "feat")
    git("checkout", "-q", "main")

    result = subprocess.run(
        ["bash", os.path.join(SCRIPTS, "compose_trial_build.sh"), "-m", "main", "-p", "prov.txt",
         "trial", "trial-br", "main", "feat=feat"],
        cwd=repo, env=env, capture_output=True, text=True)  # fmt: skip

    assert result.returncode == 0, result.stdout + result.stderr
    assert (repo / "trial" / "blob.bin").read_bytes() == b"\x00\xff\xfe\xfd"
    assert "CONFLICT" not in (repo / "prov.txt").read_text()


# --------------------------------------------------------------------------- mutation gate


def test_a_mutation_that_hangs_is_detected_and_restored(tmp_path):
    """A planted bug that loops is stopped at --timeout, counted DETECTED, and taken out."""
    # mutate.py resolves `file` against the repository four levels above its own directory.
    gate = tmp_path / ".claude" / "skills" / "s" / "scripts" / "mutate.py"
    gate.parent.mkdir(parents=True)
    shutil.copy(os.path.join(SCRIPTS, "mutate.py"), gate)
    target = tmp_path / "target.py"
    target.write_text("HANG = False\n")
    mutations = tmp_path / "t.mutations"
    mutations.write_text("[mutation]\nname = loops\nfile = target.py\n"
                         "find = HANG = False\nreplace = HANG = True\n")  # fmt: skip
    runner = tmp_path / "runner.py"
    runner.write_text(f"import time\nif 'True' in open({str(target)!r}).read():\n"
                      "    time.sleep(20)\n")  # fmt: skip

    result = subprocess.run([sys.executable, str(gate), str(mutations), "--runner", str(runner),
                             "--timeout", "2"], capture_output=True, text=True,
                            timeout=60)  # fmt: skip

    assert result.returncode == 0, result.stdout + result.stderr
    assert "DETECTED loops" in result.stdout
    assert target.read_text() == "HANG = False\n"


def test_a_gate_on_a_failing_baseline_does_not_pass(tmp_path):
    """A property that fails before any bug is planted proves nothing: the gate exits 1."""
    gate = tmp_path / ".claude" / "skills" / "s" / "scripts" / "mutate.py"
    gate.parent.mkdir(parents=True)
    shutil.copy(os.path.join(SCRIPTS, "mutate.py"), gate)
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n")
    mutations = tmp_path / "t.mutations"
    mutations.write_text("[mutation]\nname = other\nfile = target.py\n"
                         "find = VALUE = 1\nreplace = VALUE = 2\n")  # fmt: skip
    runner = tmp_path / "runner.py"
    runner.write_text(f"import sys\nprint(open({str(target)!r}).read())\nsys.exit(1)\n")

    result = subprocess.run([sys.executable, str(gate), str(mutations), "--runner", str(runner)],
                            capture_output=True, text=True, timeout=60,
                            env=dict(os.environ, TMPDIR=str(tmp_path)))  # fmt: skip

    assert "baseline: ALREADY FAILING" in result.stdout and "DETECTED other" in result.stdout
    assert result.returncode == 1, result.stdout + result.stderr
    assert target.read_text() == "VALUE = 1\n"


# --------------------------------------------------------------------------- pi kit


def _supervisor(run_dir):
    sessions = [{"id": "s0001", "task": "t"}]
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w") as fh:
        json.dump({"run_id": "r", "tasks": [{"id": "t"}], "sessions": sessions,
                   "timeout_s": 60, "concurrency": 1}, fh)  # fmt: skip
    return pikit.Supervisor(run_dir)


def test_a_run_dir_too_deep_for_the_sockets_is_refused(tmp_path, monkeypatch):
    """A session's socket path has at most 107 bytes: refuse the run, not each session."""
    deep = str(tmp_path / ("d" * 100))
    for variable in ("PIKIT_IMAGE", "PIKIT_NODE_DIR", "PIKIT_PI_DIR"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(SystemExit, match="unix socket"):
        pikit.main(["run", "--run-dir", deep, "--api", "openai-completions",
                    "--base-url", "http://127.0.0.1:1/v1", "--model", "m",
                    "--context-window", "1000", "--max-tokens", "10"])  # fmt: skip
    pikit.check_socket_paths("/r", [{"id": "s0001"}])


def test_the_request_count_reads_only_what_was_appended(tmp_path):
    sup = _supervisor(str(tmp_path))

    def count():
        return sup.write_table(write=False).splitlines()[-1].split("\t")[-1]

    assert count() == "-"
    trace = tmp_path / "sessions" / "s0001" / "trace"
    trace.mkdir(parents=True)
    capture = trace / "requests-capture.jsonl"
    capture.write_bytes(b"a\nb\n")
    assert count() == "2"
    with open(capture, "r+b") as fh:  # what was counted is not read again
        fh.write(b"a b ")
    with open(capture, "ab") as fh:
        fh.write(b"c")
    assert count() == "3"
    with open(capture, "ab") as fh:
        fh.write(b"\nd\n")
    assert count() == "4"
    replacement = trace / "next.jsonl"  # another attempt's capture is another file
    replacement.write_bytes(b"x\n")
    os.replace(replacement, capture)
    assert count() == "1"


def test_docker_runs_off_the_event_loop(tmp_path, monkeypatch):
    """While docker stops one session, the relays of the others keep moving bytes."""
    sup = _supervisor(str(tmp_path))
    session = sup.sessions[0]
    session.state["container_id"] = "c1"
    session.deadline = 0  # past its time limit: the supervisor stops it
    sup.running[session.id] = session
    ticks, calls, stalled = [], [], []

    def slow_docker(*args, **kwargs):
        before = len(ticks)
        time.sleep(0.2)
        calls.append(args[0])
        stalled.append(len(ticks) == before)
        out = json.dumps({"Status": "running"}) if args[0] == "inspect" else ""
        return subprocess.CompletedProcess(["docker", *args], 0, out, "")

    monkeypatch.setattr(pikit, "docker", slow_docker)

    async def scenario():
        async def ticker():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)

        task = asyncio.ensure_future(ticker())
        await asyncio.sleep(0)
        await sup.poll()
        task.cancel()

    asyncio.run(scenario())

    assert calls == ["inspect", "stop", "inspect", "rm"]
    assert not any(stalled), "a docker call held the event loop"
    assert session.state["status"] == "timeout"


def test_a_case_without_a_final_message_is_summarized(tmp_path, monkeypatch):
    """Pi that died before its first answer leaves a row, and the report is still written."""
    with open(os.path.join(SCRIPTS, "pi_kit", "measure", "cases.tsv")) as fh:
        case = next(line.split("\t")[0] for line in fh if line.strip() and line[0] != "#")
    (tmp_path / "runs" / case).mkdir(parents=True)
    monkeypatch.setattr(summarize, "WORK", str(tmp_path))

    summarize.main()

    rows = (tmp_path / "RESULTS.md").read_text().splitlines()
    assert next(r for r in rows if r.startswith(f"| {case} |")).endswith("| - |")


# --------------------------------------------------------------------------- session runners

# A stand-in harness: it records its arguments, working directory and environment as exec passed
# them (from /proc: Python adds variables of its own), then prints one event. A prompt with
# "no-namespaces" adds the event Codex writes when its sandbox cannot start a command; one with
# "panic" exits as Codex does when it panics; one with "exit-42" exits 42.
STUB = """#!{python}
import json, os, sys
with open("/proc/self/environ", "rb") as fh:
    env = dict(item.decode().split("=", 1) for item in fh.read().split(b"\\0") if item)
with open(os.path.join({records!r}, "%d.json" % os.getpid()), "w") as fh:
    json.dump({{"argv": sys.argv[1:], "cwd": os.getcwd(), "env": env}}, fh)
print(json.dumps({{"type": "stub"}}))
prompt = " ".join(sys.argv[1:])
if "no-namespaces" in prompt:
    print(json.dumps({{"type": "item.completed", "item": {{"type": "command_execution", "exit_code": 1,
                     "aggregated_output": "bwrap: No permissions to create a new namespace"}}}}))
if "panic" in prompt:
    sys.exit(101)
if "exit-42" in prompt:
    sys.exit(42)
"""
# runner: (script, the variable naming the harness executable, required settings)
RUNNERS = {
    "claude": ("run_claude_sessions.sh", "CLAUDE_BIN", {"BASE_URL": "http://127.0.0.1:9"}),
    "codex": ("run_codex_sessions.sh", "CODEX_BIN", {"BASE_URL": "http://127.0.0.1:9/v1"}),
    "pi": ("run_pi_sessions.sh", "NODE_BIN", {"BASE_URL": "http://127.0.0.1:9/v1", "MODEL": "m"}),
}
# What a session's environment holds besides the proxy variables.
ALLOWED = {
    "claude": {"PATH", "HOME", "CLAUDE_CONFIG_DIR", "TMPDIR", "ANTHROPIC_BASE_URL",
               "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL",
               "DISABLE_AUTOUPDATER", "CLAUDE_CODE_GATEWAY_HINT_HEADERS",
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"},
    "codex": {"PATH", "HOME", "CODEX_HOME", "TMPDIR", "TRTLLM_API_KEY", "TERM"},
    "pi": {"PATH", "HOME", "PI_CODING_AGENT_DIR", "TMPDIR", "PI_OFFLINE", "PI_SKIP_VERSION_CHECK",
           "PI_TELEMETRY"},
}  # fmt: skip
PROXIES = {"HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
           "NO_PROXY": "127.0.0.1,localhost", "ALL_PROXY": "http://127.0.0.1:9"}  # fmt: skip
PROXIES.update({name.lower(): value for name, value in PROXIES.items()})  # curl reads http_proxy
needs_proc = pytest.mark.skipif(
    not os.path.exists("/proc/self/environ") or shutil.which("timeout") is None,
    reason="needs /proc and coreutils timeout",
)


def _run_sessions(cwd, runner, relative=False, tasks="src\tsay hi\n", environment=None):
    """Run ``runner`` from ``cwd`` with the stub as its harness: ``(run, stub records)``.

    ``relative`` passes the tasks file, the output directory and the executables relative to
    ``cwd``; the source dir in ``tasks`` is relative in any case.
    """
    script, binary, settings = RUNNERS[runner]
    records = cwd / "records"
    for directory in (cwd / "bin", cwd / "src", records):
        directory.mkdir(parents=True)
    stub = cwd / "bin" / "harness"
    stub.write_text(STUB.format(python=sys.executable, records=str(records)))
    stub.chmod(0o755)
    (cwd / "bin" / "pi.js").write_text("")
    (cwd / "src" / "a.txt").write_text("a\n")
    (cwd / "tasks.tsv").write_text(tasks)

    def path(name):
        return name if relative else str(cwd / name)

    # The caller's environment, with a parent agent session's variables that no session may see.
    env = {"PATH": os.environ["PATH"], "HOME": str(cwd), "TERM": "xterm",
           "OPENAI_API_KEY": "sk-caller-only", "CLAUDE_CODE_SSE_PORT": "1",
           binary: path("bin/harness"), **settings, **(environment or {})}  # fmt: skip
    if runner == "pi":
        env["PI_BIN"] = path("bin/pi.js")
    result = subprocess.run(["bash", os.path.join(SCRIPTS, script), path("tasks.tsv"), path("out")],
                            cwd=cwd, env=env, capture_output=True, text=True, timeout=120)  # fmt: skip
    return result, [json.loads(p.read_text()) for p in sorted(records.iterdir())]


def _prompt(runner, argv):
    return argv[argv.index("-p") + 1] if runner == "claude" else argv[-1]


@needs_proc
@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_session_whose_source_cannot_be_copied_is_not_started(tmp_path, runner):
    """A task source that cannot be copied fails its session, before the harness starts."""
    result, records = _run_sessions(tmp_path, runner, tasks="no-such-dir\tsay hi\n")

    row = (tmp_path / "out" / "sessions.tsv").read_text().split("\t")
    assert records == [] and row[:2] == ["1", "1"], result.stdout + result.stderr
    assert "cannot copy the task source" in (tmp_path / "out" / "s0001" / "stderr.log").read_text()


@needs_proc
@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_runner_refuses_an_output_directory_in_use(tmp_path, runner):
    """A rerun into an earlier run's directory would carry its sessions' files over: exit 2."""
    stale = tmp_path / "out" / "s0001" / "workspace" / "left.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("from the earlier run\n")
    result, records = _run_sessions(tmp_path, runner)

    assert result.returncode == 2 and "out is not empty" in result.stderr, result.stderr
    assert records == [] and stale.read_text() == "from the earlier run\n"
    assert not (tmp_path / "out" / "sessions.tsv").exists()


@needs_proc
@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_runner_takes_relative_paths_from_where_it_started(tmp_path, runner):
    """Relative paths: the outputs land in <out_dir>, and the harness is given absolute paths."""
    result, records = _run_sessions(tmp_path, runner, relative=True)

    session = tmp_path / "out" / "s0001"
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "sessions.tsv").read_text().split("\t")[:2] == ["1", "0"]
    assert (session / "events.jsonl").read_text() == '{"type": "stub"}\n'
    assert (session / "workspace" / "a.txt").read_text() == "a\n"
    ((argv, cwd, env),) = [(r["argv"], r["cwd"], r["env"]) for r in records]
    given = [
        ("HOME", env.get("HOME"), session / "home"),
        ("TMPDIR", env.get("TMPDIR"), session / "tmp"),
    ]
    if runner == "claude":
        given += [("CLAUDE_CONFIG_DIR", env.get("CLAUDE_CONFIG_DIR"), session / "home" / ".claude"),
                  ("cwd", cwd, session / "workspace")]  # fmt: skip
    elif runner == "codex":
        given += [("CODEX_HOME", env.get("CODEX_HOME"), session / "home"),
                  ("-C", argv[argv.index("-C") + 1], session / "workspace"),
                  ("-o", argv[argv.index("-o") + 1], session / "last_message.txt")]  # fmt: skip
    else:
        given += [("PI_CODING_AGENT_DIR", env.get("PI_CODING_AGENT_DIR"), session / "agent"),
                  ("--session-dir", argv[argv.index("--session-dir") + 1], session / "sessions"),
                  ("PI_BIN", argv[0], tmp_path / "bin" / "pi.js"),
                  ("cwd", cwd, session / "workspace")]  # fmt: skip
    wrong = [(name, got) for name, got, want in given
             if not (got and os.path.isabs(got) and os.path.realpath(got) == os.path.realpath(want))]  # fmt: skip
    assert wrong == []


@needs_proc
@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_session_sees_its_allowlist_and_the_proxies_set(tmp_path, runner):
    """Of the caller's variables, only the proxy variables reach a session, and only when set."""
    for case, environment in (("unset", {}), ("set", PROXIES)):
        result, records = _run_sessions(tmp_path / case, runner, environment=environment)

        assert result.returncode == 0 and len(records) == 1, result.stderr
        env = records[0]["env"]
        assert set(env) - set(PROXIES) == ALLOWED[runner], (case, sorted(env))
        assert {k: v for k, v in env.items() if k in PROXIES} == environment, case


@needs_proc
@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_tasks_file_starts_exactly_its_tasks(tmp_path, runner):
    """A '#' header and blank lines start no session; a last task without a newline does."""
    tasks = ("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.\n"
             "\nsrc\tfirst task\n  \n-\tsecond task")  # fmt: skip
    result, records = _run_sessions(tmp_path, runner, tasks=tasks)

    assert result.returncode == 0, result.stderr
    rows = (tmp_path / "out" / "sessions.tsv").read_text().splitlines()
    assert sorted(row.split("\t")[:2] for row in rows) == [["1", "0"], ["2", "0"]]
    assert sorted(_prompt(runner, r["argv"]) for r in records) == ["first task", "second task"]


@needs_proc
@pytest.mark.parametrize("runner", sorted(RUNNERS))
def test_a_runner_fails_when_a_session_fails(tmp_path, runner):
    """A session that exits non-zero fails the run; every session still runs and is recorded."""
    result, records = _run_sessions(tmp_path, runner, tasks="src\tsay hi\nsrc\texit-42 now\n")

    rows = (tmp_path / "out" / "sessions.tsv").read_text().splitlines()
    assert sorted(row.split("\t")[:2] for row in rows) == [["1", "0"], ["2", "42"]]
    assert len(records) == 2
    assert result.returncode == 1 and "1 session(s) exited non-zero" in result.stderr


@needs_proc
def test_a_codex_run_whose_sandbox_did_not_start_fails(tmp_path):
    """A command Codex's sandbox could not start, or a panic, fails the run: no tool of it ran."""
    tasks = "src\tsay hi\nsrc\tno-namespaces\n-\tpanic\n"
    result, records = _run_sessions(tmp_path, "codex", tasks=tasks)

    rows = (tmp_path / "out" / "sessions.tsv").read_text().splitlines()
    # index, exit code and sandbox failures; the seconds vary
    got = sorted((r[0], r[1], r[3]) for r in (row.split("\t") for row in rows))
    assert got == [("1", "0", "0"), ("2", "0", "1"), ("3", "101", "0")]
    assert result.returncode == 3 and "did not start in 2 session(s)" in result.stderr
    assert "No permissions" not in result.stdout + result.stderr  # counted, never printed


# --------------------------------------------------------------------------- recall probe


class _Model(http.server.BaseHTTPRequestHandler):
    """A served model that recalls in short prompts, forgets in long ones, refuses past its window."""

    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).seen.append((self.path, self.headers.get("anthropic-version"), body))
        prompt = body["messages"][0]["content"]
        if len(prompt) > 20000:
            self.send_response(400)
            self.end_headers()
            return
        answer = re.search(r"TANGERINE-\d{4}", prompt)[0] if len(prompt) < 5000 else "Unknown."
        if self.path == "/v1/messages":
            reply = {"content": [{"type": "text", "text": answer}], "usage": {"input_tokens": 7}}
        else:
            reply = {"choices": [{"message": {"content": answer}}], "usage": {"prompt_tokens": 7}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_a_recall_probe_reports_each_probe_and_never_the_answer(capsys):
    """Each size at both depths: recalled, forgotten or refused, with the server's prompt count."""
    _Model.seen = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        chat = probe_recall.main(["--base-url", base, "--model", "m", "--sizes", "200,3000,6000",
                                  "--extra", '{"chat_template_kwargs": {"enable_thinking": false}}'])  # fmt: skip
        chat_lines = capsys.readouterr().out.splitlines()
        messages = probe_recall.main(["--base-url", base + "/", "--model", "m", "--sizes", "200",
                                      "--route", "messages"])  # fmt: skip
        messages_lines = capsys.readouterr().out.splitlines()
    finally:
        server.shutdown()
        server.server_close()

    assert (chat, messages) == (1, 0)
    assert [re.search(r"recalled (\w+)", line)[1] for line in chat_lines] == [
        "yes", "yes", "no", "no", "no", "no"]  # fmt: skip
    assert all("refused: HTTP 400" in line for line in chat_lines[4:])
    assert all("prompt_tokens        7" in line for line in chat_lines[:4] + messages_lines)
    assert "TANGERINE" not in "\n".join(chat_lines + messages_lines)
    routes = [(path, version, body.get("chat_template_kwargs"), body["temperature"])
              for path, version, body in _Model.seen]  # fmt: skip
    assert routes == [("/v1/chat/completions", None, {"enable_thinking": False}, 0)] * 6 + [
        ("/v1/messages", "2023-06-01", None, 0)] * 2  # fmt: skip


@pytest.mark.parametrize("sizes", ["", ",", " , ", "0", "-5,100", "abc"])
def test_a_recall_probe_without_a_size_is_refused(sizes):
    """A probe that would send nothing, or a size of no tokens, is a usage error and not a pass."""
    with pytest.raises(SystemExit) as stop:
        probe_recall.main(["--base-url", "http://127.0.0.1:9", "--model", "m", "--sizes", sizes])
    assert stop.value.code == 2


# --------------------------------------------------------------------------- the Codex pointer


def _frontmatter(path):
    return yaml.safe_load(path.read_text().split("---\n")[1])


def test_the_codex_pointer_carries_the_skills_name_and_description():
    """Codex finds the skill through .codex/skills, whose SKILL.md must stay the skill's own."""
    skill = Path(SCRIPTS).parent
    pointer = skill.parents[2] / ".codex" / "skills" / skill.name / "SKILL.md"
    if not pointer.exists():
        pytest.skip("no Codex pointer here: a copy of the skill alone")
    real, mirror = _frontmatter(skill / "SKILL.md"), _frontmatter(pointer)
    keys = ("name", "description")
    assert {k: mirror.get(k) for k in keys} == {k: real.get(k) for k in keys}
    assert f".claude/skills/{skill.name}/SKILL.md" in pointer.read_text()
