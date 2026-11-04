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
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import types

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
for _path in (SCRIPTS, os.path.join(SCRIPTS, "pi_kit", "bin"),
              os.path.join(SCRIPTS, "pi_kit", "measure")):  # fmt: skip
    sys.path.insert(0, _path)

import build_replay_corpus  # noqa: E402
import pikit  # noqa: E402
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


# --------------------------------------------------------------------------- replay

EVENT = b'event: response.created\ndata: {"type": "response.created"}\n\n'


async def _replay(events, first_event_timeout, timeout, out_dir=None):
    """Replay one request against a stream that sends ``events`` events, then stays silent."""
    release = asyncio.Event()

    async def handler(request):
        resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
        await resp.prepare(request)
        for _ in range(events):
            await resp.write(EVENT)
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
            api_key_env=None, dry_run=False, concurrency=1)  # fmt: skip
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


def test_the_replay_output_is_readable_by_its_owner_only(tmp_path):
    out_dir = tmp_path / "replay"
    asyncio.run(_replay(2, 5.0, 5.0, out_dir=str(out_dir)))

    assert stat.S_IMODE(os.stat(out_dir).st_mode) == 0o700
    for name in ("requests-replay.jsonl", "responses-replay.jsonl"):
        assert stat.S_IMODE(os.stat(out_dir / name).st_mode) == 0o600, name


# --------------------------------------------------------------------------- trial build


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize(
    "branch, prov_args, prov",
    [("trial-br", ["-p", "prov.txt"], "prov.txt"),
     ("user/trial", [], "trial.user_trial.provenance.txt")],
    ids=["given", "default_for_a_branch_with_a_slash"],
)  # fmt: skip
def test_a_trial_build_takes_its_paths_from_the_caller(tmp_path, branch, prov_args, prov):
    """Relative worktree and provenance paths name what they name from where the script ran."""
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")  # fmt: skip
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
# them (from /proc: Python adds variables of its own), then prints one event.
STUB = """#!{python}
import json, os, sys
with open("/proc/self/environ", "rb") as fh:
    env = dict(item.decode().split("=", 1) for item in fh.read().split(b"\\0") if item)
with open(os.path.join({records!r}, "%d.json" % os.getpid()), "w") as fh:
    json.dump({{"argv": sys.argv[1:], "cwd": os.getcwd(), "env": env}}, fh)
print(json.dumps({{"type": "stub"}}))
"""
# runner: (script, the variable naming the harness executable, required settings)
RUNNERS = {
    "claude": ("run_claude_sessions.sh", "CLAUDE_BIN", {"BASE_URL": "http://127.0.0.1:9"}),
    "codex": ("run_codex_sessions.sh", "CODEX_BIN", {"BASE_URL": "http://127.0.0.1:9/v1"}),
    "pi": ("run_pi_sessions.sh", "NODE_BIN", {"BASE_URL": "http://127.0.0.1:9/v1", "MODEL": "m"}),
}
# What a session's environment holds besides the proxy variables.
ALLOWED = {
    "claude": {"PATH", "HOME", "CLAUDE_CONFIG_DIR", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
               "ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL", "DISABLE_AUTOUPDATER",
               "CLAUDE_CODE_GATEWAY_HINT_HEADERS", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"},
    "codex": {"PATH", "HOME", "CODEX_HOME", "TRTLLM_API_KEY", "TERM"},
    "pi": {"PATH", "HOME", "PI_CODING_AGENT_DIR", "PI_OFFLINE", "PI_SKIP_VERSION_CHECK",
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
def test_a_runner_takes_relative_paths_from_where_it_started(tmp_path, runner):
    """Relative paths: the outputs land in <out_dir>, and the harness is given absolute paths."""
    result, records = _run_sessions(tmp_path, runner, relative=True)

    session = tmp_path / "out" / "s0001"
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "sessions.tsv").read_text().split("\t")[:2] == ["1", "0"]
    assert (session / "events.jsonl").read_text() == '{"type": "stub"}\n'
    assert (session / "workspace" / "a.txt").read_text() == "a\n"
    ((argv, cwd, env),) = [(r["argv"], r["cwd"], r["env"]) for r in records]
    given = [("HOME", env.get("HOME"), session / "home")]
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
