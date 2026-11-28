#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pi kit: run pi coding-agent sessions against one trtllm-serve endpoint.

Each session runs in its own throwaway docker container, with record_requests.py in front of the
endpoint. Run it on a docker host. Python 3 standard library only; the recorder runs with the validator's
environment (aiohttp). The image, node and pi come from options or PIKIT_IMAGE, PIKIT_NODE_DIR,
PIKIT_PI_DIR and PIKIT_PYTHON (README.md).

  pikit.py run    --run-dir DIR --tasks TASKS.tsv --api API --base-url URL --model NAME
                  --context-window N --max-tokens N [options]      start (detached) or resume a run
  pikit.py resume --run-dir DIR [--rerun timeout,done]              continue a run with its saved config
  pikit.py stop   --run-dir DIR                                     stop what this run started
  pikit.py status --run-dir DIR                                     one line per session

Each session is one task in one container:
  * docker --runtime runc, NVIDIA_VISIBLE_DEVICES=void (no GPU), --network none, --cap-drop ALL,
    no-new-privileges, the caller's uid:gid, a CPU/memory/pid cap;
  * mounts: the session's copy of the task workspace (/workspace), pi's agent dir (/pi), the
    session's io dir (/io), and node, pi and this kit's bin dir read-only. No home directory, no
    credentials (the API key pi sends is a dummy, and the recorder does not forward credentials);
  * the only way out of the container is a unix socket in /io, which this process relays to the
    session's recorder (record_requests.py --session RUN/sNNNN --forward ENDPOINT), or straight to
    the endpoint with --no-record.

State lives in the run dir, so a run can be stopped, resumed and re-run:
  run.json                     the run's configuration and provenance
  supervisor.pid, supervisor.log
  sessions.tsv                 one line per session, rewritten on every change
  sessions/sNNNN/state.json    status (pending, running, done, timeout, stopped, error), attempt,
                               container id, recorder pid, pi exit code, summary of pi's events
  sessions/sNNNN/{workspace,pi,io,trace}/, recorder.log, docker.log; attempts/aN/ keeps the
                               output of an attempt that did not finish (stopped or failed)
`done` and `timeout` are final; `resume` re-runs the others (and, with --rerun, those too).
`stop` signals the recorded supervisor pid, which stops the containers and recorders it started;
when no supervisor is alive it stops the recorded container ids and recorder pids itself. Nothing
is ever killed by name pattern.

The model entry pi gets: --compat-preset NAME (presets/NAME.json, or a path to such a file) sets
its `compat` and `thinkingLevelMap`, so pi's thinking level reaches the served model's own effort
setting; --compat is merged over the preset's compat (chatTemplateKwargs key by key, as pi merges
compat). A run without a preset writes the entry it always wrote.
"""

import argparse
import asyncio
import datetime
import errno
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(KIT, "bin")
PRESETS = os.path.join(KIT, "presets")
# The kit lives in the skill's scripts/ directory: the skill is two levels up.
DEFAULT_SKILL = os.path.dirname(os.path.dirname(KIT))
DEFAULT_IMAGE = os.environ.get("PIKIT_IMAGE", "")
DEFAULT_NODE = os.environ.get("PIKIT_NODE_DIR", "")
DEFAULT_PI = os.environ.get("PIKIT_PI_DIR", "")
DEFAULT_RECORDER_PY = os.environ.get("PIKIT_PYTHON", sys.executable)
APIS = ("openai-completions", "openai-responses", "anthropic-messages")
# pi's thinking levels (pi-ai 1.4.0 providers/model-schema.js MODEL_THINKING_LEVELS); pi's own
# default is medium (pi-coding-agent dist/core/defaults.js).
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
# compat keys whose objects pi merges key by key (pi-coding-agent provider-composer.js mergeCompat)
NESTED_COMPAT = (
    "openRouterRouting",
    "vercelGatewayRouting",
    "chatTemplateKwargs",
    "chatTemplateArgs",
)
FINAL = ("done", "timeout")
IN_PORT = 18080  # pi's endpoint inside the container (loopback only)
PROVIDER = "trtllm"
POLL_S = 2.0
STOP_GRACE_S = 20
# The longest path a unix socket binds to on Linux: sun_path's 108 bytes, less the closing NUL.
UNIX_PATH_MAX = 107


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: str, data) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def proc_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def pid_matches(pid, needle: str) -> bool:
    """Whether pid is alive and is the process we started: its command line holds ``needle``."""
    return isinstance(pid, int) and pid > 0 and needle in proc_cmdline(pid)


def docker(*args, check=False, timeout=120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check,
                          timeout=timeout)  # fmt: skip


async def docker_async(*args, **kwargs) -> subprocess.CompletedProcess:
    """``docker`` in a worker thread: the event loop also relays every running session's bytes."""
    return await asyncio.to_thread(docker, *args, **kwargs)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --------------------------------------------------------------------------- tasks and config


def load_tasks(path: str) -> list:
    """Tasks from a TSV: ``id<TAB>kind<TAB>size<TAB>dir``; dir holds prompt.md and workspace/."""
    base = os.path.dirname(os.path.abspath(path))
    tasks = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise SystemExit(f"{path}: want id, kind, size, dir: {line!r}")
            task_dir = os.path.join(base, fields[3])
            for need in ("prompt.md", "workspace"):
                if not os.path.exists(os.path.join(task_dir, need)):
                    raise SystemExit(f"{task_dir}: missing {need}")
            tasks.append({"id": fields[0], "kind": fields[1], "size": fields[2], "dir": task_dir})
    if not tasks:
        raise SystemExit(f"{path}: no tasks")
    return tasks


def check_socket_paths(run_dir: str, sessions: list) -> None:
    """Refuse a run dir so deep that a session's unix socket (io/ep.sock) cannot bind."""
    paths = [os.path.join(os.path.abspath(run_dir), "sessions", s["id"], "io", "ep.sock")
             for s in sessions]  # fmt: skip
    longest = max(paths, key=lambda p: len(os.fsencode(p)), default="")
    if len(os.fsencode(longest)) > UNIX_PATH_MAX:
        raise SystemExit(f"--run-dir is too deep for the sessions' unix sockets: {longest} is "
                         f"{len(os.fsencode(longest))} bytes, the limit is {UNIX_PATH_MAX}; "
                         "use a shorter --run-dir")  # fmt: skip


def endpoint_origin(base_url: str) -> str:
    """The endpoint without a trailing /v1: the recorder forwards each request's own path."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise SystemExit(f"--base-url {base_url!r}: want http://host:port[/v1]")
    return url


def image_info(image: str) -> dict:
    result = docker("image", "inspect", image, "--format", "{{json .}}")
    if result.returncode:
        raise SystemExit(f"docker image {image!r} not found on this host: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    env = dict(
        item.split("=", 1) for item in info.get("Config", {}).get("Env") or [] if "=" in item
    )
    return {"id": info.get("Id"), "path": env.get("PATH", "/usr/local/bin:/usr/bin:/bin")}


def load_preset(name: str) -> dict:
    """A model-entry preset: presets/<name>.json, or a path to a file of that shape."""
    path = (
        name if os.sep in name or name.endswith(".json") else os.path.join(PRESETS, f"{name}.json")
    )
    try:
        with open(path) as fh:
            preset = json.load(fh)
    except (OSError, json.JSONDecodeError) as error:
        known = sorted(n[:-5] for n in os.listdir(PRESETS) if n.endswith(".json"))
        raise SystemExit(f"--compat-preset {name!r}: {error} (presets: {', '.join(known)})")
    if preset.get("schema") != "pi-kit/compat-preset@1" or not isinstance(
        preset.get("compat"), dict
    ):
        raise SystemExit(f"{path}: not a pi-kit/compat-preset@1 file with a `compat` object")
    level_map = preset.get("thinkingLevelMap")
    if level_map is not None and (
        not isinstance(level_map, dict)
        or set(level_map) - set(THINKING_LEVELS)
        or any(v is not None and not isinstance(v, str) for v in level_map.values())
    ):
        raise SystemExit(f"{path}: thinkingLevelMap maps pi's levels {THINKING_LEVELS} to a "
                         "string or null")  # fmt: skip
    with open(path, "rb") as fh:
        preset["_provenance"] = {"name": preset.get("id") or name, "path": os.path.abspath(path),
                                 "sha256": hashlib.sha256(fh.read()).hexdigest()}  # fmt: skip
    return preset


def merge_compat(base: dict, override: dict) -> dict:
    """``override`` over ``base`` as pi merges a model's compat: nested objects key by key."""
    merged = {**base, **override}
    for key in NESTED_COMPAT:
        parts = [part for part in (base.get(key), override.get(key)) if isinstance(part, dict)]
        if parts:
            merged[key] = {k: v for part in parts for k, v in part.items()}
    return merged


def model_settings(args):
    """(compat, thinkingLevelMap or None, preset provenance or None) for the model entry."""
    compat = json.loads(args.compat)
    if not isinstance(compat, dict):
        raise SystemExit("--compat: a JSON object")
    if not args.compat_preset:
        return compat, None, None
    preset = load_preset(args.compat_preset)
    if preset.get("api") not in (None, args.api):
        raise SystemExit(f"--compat-preset {args.compat_preset}: written for api "
                         f"{preset['api']}, not {args.api}")  # fmt: skip
    return (
        merge_compat(preset["compat"], compat),
        preset.get("thinkingLevelMap"),
        preset["_provenance"],
    )


def model_entry(cfg: dict) -> dict:
    """The models.json entry of the served model.

    Without a thinkingLevelMap (every run before presets existed) it is exactly the entry those
    runs wrote.
    """
    entry = {"id": cfg["model"], "name": f"pikit-{cfg['model']}", "reasoning": cfg["reasoning"],
             "input": ["text"], "contextWindow": cfg["context_window"],
             "maxTokens": cfg["max_tokens"],
             "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
             "compat": cfg["compat"]}  # fmt: skip
    if cfg.get("thinking_level_map") is not None:
        entry["thinkingLevelMap"] = cfg["thinking_level_map"]
    return entry


def build_config(args) -> dict:
    if args.api not in APIS:
        raise SystemExit(f"--api: one of {', '.join(APIS)}")
    tasks = load_tasks(args.tasks)
    compat, thinking_level_map, preset = model_settings(args)
    origin = endpoint_origin(args.base_url)
    if args.no_record and urllib.parse.urlparse(origin).path not in ("", "/"):
        raise SystemExit("--no-record relays TCP bytes: the endpoint cannot have a path prefix")
    if args.no_record and urllib.parse.urlparse(origin).scheme != "http":
        raise SystemExit("--no-record relays TCP bytes: the endpoint must be plain http")
    count = args.sessions or len(tasks)
    sessions = [{"id": f"s{i + 1:04d}", "task": tasks[i % len(tasks)]["id"]} for i in range(count)]
    if args.only:
        wanted = set(args.only.split(","))
        sessions = [s for s in sessions if s["id"] in wanted]
    check_socket_paths(args.run_dir, sessions)
    for option, value, variable in (("--image", args.image, "PIKIT_IMAGE"),
                                    ("--node-dir", args.node_dir, "PIKIT_NODE_DIR"),
                                    ("--pi-dir", args.pi_dir, "PIKIT_PI_DIR")):  # fmt: skip
        if not value:
            raise SystemExit(f"{option} (or {variable}) is required")
    skill = os.path.abspath(args.skill_dir)
    recorder = os.path.join(skill, "scripts", "record_requests.py")
    if not args.no_record and not os.path.exists(recorder):
        raise SystemExit(f"recorder not found: {recorder} (--skill-dir)")
    node_dir = os.path.realpath(args.node_dir)
    pi_dir = os.path.realpath(args.pi_dir)
    with open(os.path.join(pi_dir, "node_modules", "@earendil-works", "pi-coding-agent",
                           "package.json")) as fh:  # fmt: skip
        pi_version = json.load(fh)["version"]
    image = image_info(args.image)
    # The image's PATH, without entries under a home directory (none is mounted).
    path = ":".join(p for p in image["path"].split(":") if p and not p.startswith("/home/"))
    return {
        "schema": "pi-kit/run@1",
        "created": now(),
        "run_id": os.path.basename(os.path.abspath(args.run_dir.rstrip("/"))),
        "api": args.api,
        "base_url": args.base_url,
        "origin": origin,
        "model": args.model,
        "context_window": args.context_window,
        "max_tokens": args.max_tokens,
        "reasoning": args.reasoning == "true",
        "thinking": args.thinking,
        "compat": compat,
        "thinking_level_map": thinking_level_map,
        "compat_preset": preset,
        "tools": args.tools,
        "timeout_s": args.timeout_s,
        "concurrency": args.concurrency,
        "record": not args.no_record,
        "build_id": args.build_id,
        "image": args.image,
        "image_id": image["id"],
        "container_path": f"{path}:/opt/node/bin",
        "cpus": args.cpus,
        "memory": args.memory,
        "pids_limit": args.pids_limit,
        "node_dir": node_dir,
        "pi_dir": pi_dir,
        "pi_version": pi_version,
        "skill_dir": skill,
        "recorder": recorder,
        "recorder_python": args.recorder_python,
        "tasks_file": os.path.abspath(args.tasks),
        "tasks": tasks,
        "sessions": sessions,
        "kit_sha256": {
            name: sha256_file(os.path.join(BIN, name))
            for name in ("pikit.py", "session-entry.sh", "inbridge.mjs")
        },  # fmt: skip
        "uid_gid": f"{os.getuid()}:{os.getgid()}",
    }


# Fields that change what a session does: a resumed run must keep them. A run.json written before
# presets has no thinking_level_map, which reads as None, the value of a run without a preset.
BINDING = ("api", "origin", "model", "context_window", "max_tokens", "reasoning", "thinking",
           "compat", "thinking_level_map", "tools", "timeout_s", "record", "build_id", "image_id",
           "pi_version", "tasks", "sessions")  # fmt: skip


# --------------------------------------------------------------------------- sessions


class Session:
    def __init__(self, run_dir: str, config: dict, entry: dict):
        self.config = config
        self.id = entry["id"]
        self.task = next(t for t in config["tasks"] if t["id"] == entry["task"])
        self.dir = os.path.join(run_dir, "sessions", self.id)
        self.state_path = os.path.join(self.dir, "state.json")
        self.state = read_json(self.state_path) or {
            "session": self.id, "task": self.task["id"], "status": "pending", "attempt": 0}  # fmt: skip
        self.recorder = None  # asyncio subprocess
        self.bridge = None  # asyncio server
        self.deadline = None
        self.capture_read = None  # ((st_dev, st_ino), bytes read, newlines, ends mid-line)

    # paths
    def path(self, *parts) -> str:
        return os.path.join(self.dir, *parts)

    @property
    def container_name(self) -> str:
        return f"pikit-{self.config['run_id']}-{self.id}-a{self.state['attempt']}"

    def save(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        write_json(self.state_path, self.state)

    def archive_previous(self) -> None:
        """Keep an unfinished attempt's output under attempts/aN/ before the next attempt."""
        produced = [n for n in ("workspace", "pi", "io", "trace", "recorder.log", "docker.log")
                    if os.path.exists(self.path(n))]  # fmt: skip
        if not produced:
            return
        if not self.state.get("attempt"):
            # Left over from a session that never started an attempt: nothing to keep.
            for name in produced:
                if os.path.isdir(self.path(name)):
                    shutil.rmtree(self.path(name))
                else:
                    os.remove(self.path(name))
            return
        dest = self.path("attempts", f"a{self.state['attempt']}")
        os.makedirs(dest, exist_ok=True)
        for name in produced:
            os.replace(self.path(name), os.path.join(dest, name))
        write_json(os.path.join(dest, "state.json"), self.state)

    def prepare(self) -> None:
        self.archive_previous()
        self.state.update(attempt=self.state.get("attempt", 0) + 1, status="starting",
                          started_at=now(), ended_at=None, container_id=None, recorder_pid=None,
                          pi_exit=None, container_exit=None, reason=None, summary=None)  # fmt: skip
        shutil.copytree(os.path.join(self.task["dir"], "workspace"), self.path("workspace"),
                        symlinks=True)  # fmt: skip
        for sub in ("pi/agent", "pi/sessions", "io", "trace"):
            os.makedirs(self.path(sub), exist_ok=True)
        cfg = self.config
        base = f"http://127.0.0.1:{IN_PORT}" + ("" if cfg["api"] == "anthropic-messages" else "/v1")
        write_json(self.path("pi", "agent", "models.json"), {"providers": {PROVIDER: {
            "baseUrl": base, "api": cfg["api"], "apiKey": "pikit-dummy-key",
            "models": [model_entry(cfg)]}}})  # fmt: skip
        shutil.copyfile(os.path.join(self.task["dir"], "prompt.md"), self.path("io", "prompt.txt"))
        self.save()

    # recorder and bridge
    async def start_recorder(self) -> int:
        cfg = self.config
        port = free_port()
        argv = [cfg["recorder_python"], cfg["recorder"], "--out-dir", self.path("trace"),
                "--host", "127.0.0.1", "--port", str(port), "--forward", cfg["origin"],
                "--session", f"{cfg['run_id']}/{self.id}"]  # fmt: skip
        if cfg.get("build_id"):
            argv += ["--build-id", cfg["build_id"]]
        log = open(self.path("recorder.log"), "ab")
        self.recorder = await asyncio.create_subprocess_exec(
            *argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True
        )
        log.close()
        self.state["recorder_pid"] = self.recorder.pid
        self.state["recorder_port"] = port
        self.save()
        for _ in range(100):
            if self.recorder.returncode is not None:
                raise RuntimeError(f"recorder exited with {self.recorder.returncode}")
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                return port
            except OSError:
                await asyncio.sleep(0.1)
        raise RuntimeError("recorder did not listen within 10 s")

    async def start_bridge(self, host: str, port: int) -> None:
        sock_path = self.path("io", "ep.sock")
        if os.path.exists(sock_path):
            os.remove(sock_path)

        async def pump(reader, writer, peer_writer):
            try:
                while True:
                    data = await reader.read(65536)
                    if not data:
                        break
                    writer.write(data)
                    await writer.drain()
                if writer.can_write_eof():
                    writer.write_eof()
            except (ConnectionError, OSError):
                # A reset on one side resets the other: a cut stays a cut.
                writer.transport.abort()
                peer_writer.transport.abort()

        async def relay(reader, writer):
            try:
                up_reader, up_writer = await asyncio.open_connection(host, port)
            except OSError:
                writer.transport.abort()
                return
            await asyncio.gather(
                pump(reader, up_writer, writer), pump(up_reader, writer, up_writer)
            )
            for w in (writer, up_writer):
                w.close()

        self.bridge = await asyncio.start_unix_server(relay, path=sock_path)
        os.chmod(sock_path, 0o660)

    async def stop_side(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        if self.recorder is not None and self.recorder.returncode is None:
            # Let the recorder write the records of the connections the bridge just closed.
            await asyncio.sleep(0.5)
            self.recorder.terminate()
            try:
                await asyncio.wait_for(self.recorder.wait(), 15)
            except asyncio.TimeoutError:
                self.recorder.kill()
                await self.recorder.wait()
        self.recorder = None

    # container
    def docker_run_args(self) -> list:
        cfg = self.config
        env = {
            "HOME": "/tmp/home", "PATH": cfg["container_path"], "TERM": "dumb",
            "NVIDIA_VISIBLE_DEVICES": "void", "PI_CODING_AGENT_DIR": "/pi/agent",
            "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0",
            "PIKIT_PORT": str(IN_PORT), "PIKIT_PROVIDER": PROVIDER, "PIKIT_MODEL": cfg["model"],
            "PIKIT_TOOLS": cfg["tools"], "PIKIT_TIMEOUT_S": str(cfg["timeout_s"]),
            "PIKIT_THINKING": cfg.get("thinking") or "",
        }  # fmt: skip
        argv = ["run", "-d", "--name", self.container_name,
                "--label", f"pikit.run={cfg['run_id']}", "--label", f"pikit.session={self.id}",
                "--label", f"pikit.attempt={self.state['attempt']}",
                "--runtime", "runc", "--network", "none", "--init",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--user", cfg["uid_gid"], "--cpus", str(cfg["cpus"]), "--memory", cfg["memory"],
                "--pids-limit", str(cfg["pids_limit"]),
                # The image's own /home (its build user's dotfiles) is hidden as well.
                "--tmpfs", "/home:rw,size=16m,mode=755",
                "-v", f"{self.path('workspace')}:/workspace", "-v", f"{self.path('pi')}:/pi",
                "-v", f"{self.path('io')}:/io", "-v", f"{cfg['node_dir']}:/opt/node:ro",
                "-v", f"{cfg['pi_dir']}:/opt/pi:ro", "-v", f"{BIN}:/opt/pikit:ro",
                "-w", "/workspace", "--entrypoint", "/bin/bash"]  # fmt: skip
        for key, value in env.items():
            argv += ["-e", f"{key}={value}"]
        return argv + [cfg["image_id"], "/opt/pikit/session-entry.sh"]

    def requests_recorded(self) -> str:
        """Lines of the session's request capture, or "-" before there is one.

        The capture holds whole request bodies and only grows, so each call reads only what was
        appended since the previous one. Another file (a new attempt's) is read from its start.
        """
        try:
            fh = open(self.path("trace", "requests-capture.jsonl"), "rb")
        except FileNotFoundError:
            self.capture_read = None
            return "-"
        with fh:
            st = os.fstat(fh.fileno())
            ident, offset, lines, mid_line = self.capture_read or (None, 0, 0, False)
            if ident != (st.st_dev, st.st_ino) or st.st_size < offset:
                offset, lines, mid_line = 0, 0, False
            fh.seek(offset)
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                lines += chunk.count(b"\n")
                mid_line = not chunk.endswith(b"\n")
                offset += len(chunk)
            self.capture_read = ((st.st_dev, st.st_ino), offset, lines, mid_line)
        return str(lines + mid_line)

    def inspect(self):
        result = docker("inspect", "--format", "{{json .State}}", self.state["container_id"])
        return json.loads(result.stdout) if result.returncode == 0 else None

    def summarize(self) -> dict:
        """Counts from pi's own event stream: how the session went, without its content."""
        out = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "auto_retries": 0,
               "compactions": 0, "final_stop_reason": None, "final_error_chars": 0,
               "events": 0}  # fmt: skip
        try:
            fh = open(self.path("io", "events.jsonl"))
        except FileNotFoundError:
            return out
        with fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out["events"] += 1
                kind = event.get("type")
                if kind == "turn_end":
                    out["turns"] += 1
                elif kind == "tool_execution_end":
                    out["tool_calls"] += 1
                    out["tool_errors"] += bool(event.get("isError"))
                elif kind == "auto_retry_start":
                    out["auto_retries"] += 1
                elif kind == "compaction_end":
                    out["compactions"] += 1
                elif (
                    kind == "message_end"
                    and (event.get("message") or {}).get("role") == "assistant"
                ):
                    message = event["message"]
                    out["final_stop_reason"] = message.get("stopReason")
                    out["final_error_chars"] = len(message.get("errorMessage") or "")
        return out


# --------------------------------------------------------------------------- supervisor


class Supervisor:
    def __init__(self, run_dir: str, rerun=()):
        self.run_dir = os.path.abspath(run_dir)
        self.config = read_json(os.path.join(self.run_dir, "run.json"))
        if not self.config:
            raise SystemExit(f"{self.run_dir}: no run.json")
        self.sessions = [Session(self.run_dir, self.config, s) for s in self.config["sessions"]]
        self.rerun = set(rerun)
        self.stopping = False
        self.running = {}

    def write_table(self, write: bool = True) -> str:
        lines = ["# session\ttask\tstatus\tattempt\tseconds\tpi_exit\tstop_reason\tturns\ttool_calls"
                 "\tauto_retries\tcompactions\trequests"]  # fmt: skip
        for s in self.sessions:
            st = s.state
            summary = st.get("summary") or {}
            requests = s.requests_recorded()
            lines.append("\t".join(str(x) for x in (
                s.id, st.get("task"), st.get("status"), st.get("attempt"), st.get("seconds", "-"),
                st.get("pi_exit"), summary.get("final_stop_reason"), summary.get("turns", "-"),
                summary.get("tool_calls", "-"), summary.get("auto_retries", "-"),
                summary.get("compactions", "-"), requests)))  # fmt: skip
        text = "\n".join(lines) + "\n"
        if not write:
            return text
        # A temporary file per process: `status` may render the table while the supervisor
        # rewrites it, and with a shared name one process could move the other's file away
        # (the supervisor would then die on FileNotFoundError).
        tmp = os.path.join(self.run_dir, f"sessions.tsv.tmp.{os.getpid()}")
        try:
            with open(tmp, "w") as fh:
                fh.write(text)
            os.replace(tmp, os.path.join(self.run_dir, "sessions.tsv"))
        except OSError as e:
            self.log(f"could not write sessions.tsv: {e.__class__.__name__}: {e}")
        return text

    def log(self, message: str) -> None:
        print(f"{now()} {message}", flush=True)

    def reap_orphans(self) -> None:
        """A session a dead supervisor left running: stop exactly what it recorded."""
        for s in self.sessions:
            if s.state.get("status") not in ("starting", "running", "stopping"):
                continue
            cid = s.state.get("container_id")
            if cid:
                docker("stop", "-t", str(STOP_GRACE_S), cid, timeout=STOP_GRACE_S + 60)
                docker("rm", "-f", cid)
            pid = s.state.get("recorder_pid")
            if pid_matches(pid, s.path("trace")):
                os.kill(pid, signal.SIGTERM)
            s.state.update(status="stopped", reason="left running by a supervisor that died",
                           ended_at=now())  # fmt: skip
            s.save()
            self.log(f"{s.id}: orphan of a dead supervisor stopped (container {cid})")

    async def start(self, s: Session) -> None:
        cfg = self.config
        try:
            s.prepare()
            if cfg["record"]:
                port = await s.start_recorder()
                await s.start_bridge("127.0.0.1", port)
            else:
                parsed = urllib.parse.urlparse(cfg["origin"])
                await s.start_bridge(parsed.hostname, parsed.port or 80)
            result = await docker_async(*s.docker_run_args())
            with open(s.path("docker.log"), "a") as fh:
                fh.write(f"{now()} docker run rc={result.returncode}\n{result.stderr}")
            if result.returncode:
                raise RuntimeError(f"docker run failed (rc {result.returncode}): see docker.log")
            s.state.update(status="running", container_id=result.stdout.strip(),
                           container_name=s.container_name)  # fmt: skip
            s.deadline = time.monotonic() + cfg["timeout_s"] + 60
            s.save()
            self.running[s.id] = s
            self.log(f"{s.id}: started task {s.task['id']} attempt {s.state['attempt']} "
                     f"container {s.state['container_id'][:12]}")  # fmt: skip
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            await s.stop_side()
            s.state.update(status="error", reason=str(error)[:500], ended_at=now())
            s.save()
            self.log(f"{s.id}: failed to start: {error}")

    async def finish(self, s: Session, state: dict, status: str, reason=None) -> None:
        await s.stop_side()
        try:
            with open(s.path("io", "pi_exit")) as fh:
                pi_exit = int(fh.read().strip())
        except (OSError, ValueError):
            pi_exit = None
        if status == "done" and pi_exit == 124:
            status, reason = "timeout", f"pi ran past {self.config['timeout_s']} s"
        if state and state.get("OOMKilled"):
            reason = "out of memory"
        started = s.state.get("started_at")
        seconds = None
        if started:
            begin = datetime.datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=datetime.timezone.utc
            )
            seconds = int((datetime.datetime.now(datetime.timezone.utc) - begin).total_seconds())
        s.state.update(status=status, reason=reason, ended_at=now(), seconds=seconds,
                       pi_exit=pi_exit, container_exit=(state or {}).get("ExitCode"),
                       summary=s.summarize())  # fmt: skip
        s.save()
        await docker_async("rm", "-f", s.state["container_id"])
        self.running.pop(s.id, None)
        self.log(f"{s.id}: {status} (pi exit {pi_exit}, {seconds} s"
                 + (f", {reason}" if reason else "") + ")")  # fmt: skip

    async def poll(self) -> None:
        for s in list(self.running.values()):
            state = await asyncio.to_thread(s.inspect)
            if state is None:
                await self.finish(s, None, "error", "container disappeared")
            elif state.get("Status") in ("exited", "dead"):
                await self.finish(s, state, "done")
            elif time.monotonic() > s.deadline:
                self.log(f"{s.id}: past its time limit, stopping container")
                await docker_async("stop", "-t", str(STOP_GRACE_S), s.state["container_id"],
                                   timeout=STOP_GRACE_S + 60)  # fmt: skip
                await self.finish(s, await asyncio.to_thread(s.inspect), "timeout",
                                  "stopped by the supervisor")  # fmt: skip

    async def shutdown(self) -> None:
        self.log("stop requested: stopping the sessions this run started")
        for s in list(self.running.values()):
            await docker_async("stop", "-t", str(STOP_GRACE_S), s.state["container_id"],
                               timeout=STOP_GRACE_S + 60)  # fmt: skip
            await self.finish(s, await asyncio.to_thread(s.inspect), "stopped",
                              "stopped by pikit stop")  # fmt: skip

    async def main(self) -> int:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            loop.add_signal_handler(sig, self.request_stop)
        self.reap_orphans()
        todo = [s for s in self.sessions
                if s.state.get("status") not in FINAL or s.state.get("status") in self.rerun]  # fmt: skip
        self.log(f"run {self.config['run_id']}: {len(todo)} of {len(self.sessions)} sessions to run, "
                 f"{self.config['concurrency']} at a time")  # fmt: skip
        self.write_table()
        while (todo or self.running) and not self.stopping:
            while todo and len(self.running) < self.config["concurrency"] and not self.stopping:
                await self.start(todo.pop(0))
                self.write_table()
            await asyncio.sleep(POLL_S)
            await self.poll()
            self.write_table()
        if self.stopping:
            await self.shutdown()
        self.write_table()
        counts = {}
        for s in self.sessions:
            counts[s.state.get("status")] = counts.get(s.state.get("status"), 0) + 1
        self.log(f"run {self.config['run_id']} ends: {counts}")
        return 0

    def request_stop(self) -> None:
        self.stopping = True


# --------------------------------------------------------------------------- commands


def lock_or_exit(run_dir: str):
    fh = open(os.path.join(run_dir, "supervisor.lock"), "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno in (errno.EAGAIN, errno.EACCES):
            raise SystemExit(f"{run_dir}: a supervisor is already running (supervisor.pid)")
        raise
    return fh


def supervisor_alive(run_dir: str):
    pid = read_json(os.path.join(run_dir, "supervisor.pid"), {}).get("pid")
    return (
        pid
        if pid_matches(pid, os.path.abspath(run_dir)) and pid_matches(pid, "supervise")
        else None
    )


def cmd_supervise(args) -> int:
    run_dir = os.path.abspath(args.run_dir)
    lock = lock_or_exit(run_dir)
    write_json(os.path.join(run_dir, "supervisor.pid"),
               {"pid": os.getpid(), "host": socket.gethostname(), "started": now()})  # fmt: skip
    try:
        return asyncio.run(Supervisor(run_dir, args.rerun.split(",") if args.rerun else ()).main())
    finally:
        lock.close()


def launch(run_dir: str, foreground: bool, rerun: str) -> int:
    argv = [sys.executable, os.path.abspath(__file__), "supervise", "--run-dir", run_dir]
    if rerun:
        argv += ["--rerun", rerun]
    if foreground:
        return subprocess.call(argv)
    if supervisor_alive(run_dir):
        raise SystemExit(f"{run_dir}: a supervisor is already running")
    with open(os.path.join(run_dir, "supervisor.log"), "ab") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                start_new_session=True, close_fds=True)  # fmt: skip
    print(f"supervisor pid {proc.pid}; follow {os.path.join(run_dir, 'supervisor.log')} or "
          f"`pikit.py status --run-dir {run_dir}`")  # fmt: skip
    return 0


def cmd_run(args) -> int:
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    config = build_config(args)
    saved = read_json(os.path.join(run_dir, "run.json"))
    if saved:
        changed = [k for k in BINDING if saved.get(k) != config.get(k)]
        if changed:
            raise SystemExit(f"{run_dir} was started with other settings ({', '.join(changed)}): "
                             "use a new --run-dir, or `resume` to continue this one")  # fmt: skip
        print(f"{run_dir}: same settings as before, resuming")
    else:
        write_json(os.path.join(run_dir, "run.json"), config)
    return launch(run_dir, args.foreground, "")


def cmd_resume(args) -> int:
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.exists(os.path.join(run_dir, "run.json")):
        raise SystemExit(f"{run_dir}: no run.json")
    return launch(run_dir, args.foreground, args.rerun)


def cmd_stop(args) -> int:
    run_dir = os.path.abspath(args.run_dir)
    pid = supervisor_alive(run_dir)
    if pid:
        os.kill(pid, signal.SIGTERM)
        print(f"sent TERM to supervisor {pid}; waiting for it to stop its sessions")
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline and supervisor_alive(run_dir):
            time.sleep(1)
        if not supervisor_alive(run_dir):
            print("supervisor stopped")
            return 0
        print("supervisor still running; stopping its recorded containers directly")
    config = read_json(os.path.join(run_dir, "run.json"), {})
    stopped = 0
    for entry in config.get("sessions", []):
        s = Session(run_dir, config, entry)
        if s.state.get("status") not in ("starting", "running", "stopping"):
            continue
        cid = s.state.get("container_id")
        if cid:
            docker("stop", "-t", str(STOP_GRACE_S), cid, timeout=STOP_GRACE_S + 60)
            docker("rm", "-f", cid)
        rpid = s.state.get("recorder_pid")
        if pid_matches(rpid, s.path("trace")):
            os.kill(rpid, signal.SIGTERM)
        s.state.update(status="stopped", reason="stopped by pikit stop", ended_at=now())
        s.save()
        stopped += 1
    print(f"stopped {stopped} sessions from their recorded container ids and recorder pids")
    return 0


def cmd_status(args) -> int:
    run_dir = os.path.abspath(args.run_dir)
    config = read_json(os.path.join(run_dir, "run.json"))
    if not config:
        raise SystemExit(f"{run_dir}: no run.json")
    pid = supervisor_alive(run_dir)
    preset = (config.get("compat_preset") or {}).get("name", "none")
    print(f"run {config['run_id']}: api {config['api']}, model {config['model']}, "
          f"endpoint {config['origin']}, compat preset {preset}, "
          f"thinking {config.get('thinking') or 'pi default'}, "
          f"supervisor {'pid ' + str(pid) if pid else 'not running'}")  # fmt: skip
    sup = Supervisor(run_dir)
    # Read-only: the table is rendered here, never written (the supervisor owns sessions.tsv).
    sys.stdout.write(sup.write_table(write=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="start (or resume) a run")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--tasks", default=os.path.join(KIT, "tasks", "tasks.tsv"))
    run.add_argument("--api", required=True, choices=APIS)
    run.add_argument("--base-url", required=True, help="the endpoint, e.g. http://host:8000/v1")
    run.add_argument("--model", required=True, help="the served model name")
    run.add_argument("--context-window", type=int, required=True, help="the served max_seq_len")
    run.add_argument("--max-tokens", type=int, required=True, help="pi's output cap per reply")
    run.add_argument("--reasoning", choices=("true", "false"), default="true")
    run.add_argument("--thinking", default="", choices=("",) + THINKING_LEVELS,
                     help="pi --thinking level (default: pi's own, medium)")  # fmt: skip
    run.add_argument("--compat-preset", default="",
                     help="the model entry's compat and thinkingLevelMap: a name in presets/ "
                          "(glm-5.3, kimi-k3, deepseek-v41) or a path to such a file")  # fmt: skip
    run.add_argument(
        "--compat",
        default="{}",
        help="the model entry's compat object (JSON), merged over the preset's",
    )
    run.add_argument("--tools", default="read,bash,edit,write")
    run.add_argument("--timeout-s", type=int, default=3600, help="per-session time limit")
    run.add_argument("--concurrency", type=int, default=8)
    run.add_argument(
        "--sessions", type=int, default=0, help="number of sessions (default: one per task)"
    )
    run.add_argument("--only", default="", help="comma-separated session ids to run")
    run.add_argument("--build-id", default="", help="the served build, for the recorder (I-05)")
    run.add_argument(
        "--no-record", action="store_true", help="relay to the endpoint without a recorder"
    )
    run.add_argument("--image", default=DEFAULT_IMAGE)
    run.add_argument("--cpus", default="2")
    run.add_argument("--memory", default="8g")
    run.add_argument("--pids-limit", type=int, default=4096)
    run.add_argument("--node-dir", default=DEFAULT_NODE)
    run.add_argument("--pi-dir", default=DEFAULT_PI)
    run.add_argument("--skill-dir", default=DEFAULT_SKILL)
    run.add_argument("--recorder-python", default=DEFAULT_RECORDER_PY)
    run.add_argument("--foreground", action="store_true")
    resume = sub.add_parser("resume", help="continue a run with its saved settings")
    resume.add_argument("--run-dir", required=True)
    resume.add_argument("--rerun", default="", help="also re-run sessions in these final states")
    resume.add_argument("--foreground", action="store_true")
    stop = sub.add_parser("stop", help="stop what this run started")
    stop.add_argument("--run-dir", required=True)
    stop.add_argument("--wait", type=int, default=120)
    status = sub.add_parser("status", help="one line per session")
    status.add_argument("--run-dir", required=True)
    sup = sub.add_parser("supervise", help=argparse.SUPPRESS)
    sup.add_argument("--run-dir", required=True)
    sup.add_argument("--rerun", default="")
    args = ap.parse_args(argv)
    return {"run": cmd_run, "resume": cmd_resume, "stop": cmd_stop, "status": cmd_status,
            "supervise": cmd_supervise}[args.cmd](args)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
