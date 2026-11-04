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
"""The replay, trial-build and pi kit tools (CPU only, no TensorRT-LLM import, no docker)."""

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
def test_a_trial_build_takes_its_paths_from_the_caller(tmp_path):
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
        ["bash", os.path.join(SCRIPTS, "compose_trial_build.sh"), "-m", "main", "-p", "prov.txt",
         "trial", "trial-br", "main", "feat=feat"],
        cwd=sub, env=env, capture_output=True, text=True)  # fmt: skip

    assert result.returncode == 0, result.stdout + result.stderr
    assert (sub / "trial" / "sub" / "a.txt").read_text() == "b\n"
    assert not (repo / "trial").exists()
    provenance = (sub / "prov.txt").read_text()
    assert "item feat feat head=" in provenance and "result trial-br head=" in provenance


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
