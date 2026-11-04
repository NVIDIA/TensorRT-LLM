#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving perf sweep for a modeling_v2 target: one label = one measurement session.

Boots ``trtllm-serve`` on a merged LLM-API config, warms it up, drives the upstream
``benchmark_serving`` client at each concurrency point (random token ids sent as ids, fixed
ISL/OSL, ``--ignore-eos``), and archives under ``<out-root>/<label>/`` one json per point,
``config.yaml`` (what the server ran), ``meta.json`` (tree, build, host, GPU, versions),
``server.log`` and ``summary.json``. It runs **by path inside the engine's container** (standard
library + PyYAML only).

Which forward serves is an LLM-API argument: ``modeling_v2: require`` serves the target (and
raises if no target claims the configuration), ``off`` the stock in-tree model, the reference
curve. ``auto`` requires ``--allow-auto``: a configuration that misses the target's routing
silently serves the stock model, and the number reads as the target's.

Configs compose left to right (``--config base.yaml --config variant.yaml``): mappings merge key
by key, lists and scalars replace; ``--set key.sub=value`` applies last.

A point is one closed loop: ``concurrency`` requests at once, then one more as each finishes,
until every prompt is served. ``tok_s_user`` is output tokens over end-to-end latency, both
summed over the measured run's requests (:data:`USER_RATE_BASIS`): Little's law ties it to the
point's throughput however the server admitted the waves. benchmark_serving's mean of
per-request rates grows with the spread of latencies; it is kept as ``tok_s_user_per_request``.

Guards: the staging dir is swapped in only on success; the warmup's seed is disjoint from the
points' (seed = concurrency, so KV reuse cannot serve one point from another's cache); the
server's liveness is polled during a point; teardown is ``killpg``; stdout carries one sentinel
line (``[perf] done: <dir>`` or ``[perf] FAILED: <reason>``): grep it, not the exit code, which
a pipe reports for its last stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Load only the runtime file: the engine environment need not install agent_flow.
_runtime_path = Path(__file__).resolve().parents[2] / "infra" / "compute" / "server_runtime.py"
_runtime_spec = importlib.util.spec_from_file_location("mv2flow_server_runtime", _runtime_path)
runtime = importlib.util.module_from_spec(_runtime_spec)
_runtime_spec.loader.exec_module(runtime)

PERCENTILE_METRICS = "ttft,tpot,itl,e2el"
METRIC_PERCENTILES = "50,90,99"
SERVER_BOOT_TIMEOUT_S = 1800
NSYS_BOOT_TIMEOUT_S = 5400
CLIENT_POINT_TIMEOUT_S = 5400
SERVER_LIVENESS_POLL_S = 10
WARMUP_SEED = 424242
#: How each summary row's ``tok_s_user`` aggregates the requests of its point (module docstring).
USER_RATE_BASIS = "output tokens / e2e latency, both summed over the measured run's requests"
#: The basis of a row whose point recorded no end-to-end latency: benchmark_serving's own.
PER_REQUEST_BASIS = "mean over requests of output tokens / e2e latency"


def point_file(concurrency: int) -> str:
    return f"con{concurrency:03d}.json"


def user_rate(output_tokens: float, completed: int, mean_e2el_ms: float) -> float | None:
    """Output tokens per second a user waited, over every request of a point; None without data.

    ``output_tokens`` is the point's total, ``completed`` its successful requests and
    ``mean_e2el_ms`` their mean end-to-end latency: the ratio of the two sums.
    """
    if not output_tokens or not completed or not mean_e2el_ms:
        return None
    return float(output_tokens) / (int(completed) * float(mean_e2el_ms) / 1000.0)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_scalar(text: str):
    return yaml.safe_load(text)


def apply_set(config: dict, assignment: str) -> None:
    key, _, raw = assignment.partition("=")
    if not _:
        raise SystemExit(f"--set expects key=value, got {assignment!r}")
    node = config
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise SystemExit(f"--set {assignment!r}: {part} is not a mapping")
    node[parts[-1]] = parse_scalar(raw)


def resolve_config(paths: list[Path], sets: list[str]) -> dict:
    config: dict = {}
    for path in paths:
        config = deep_merge(config, yaml.safe_load(path.read_text()) or {})
    for assignment in sets:
        apply_set(config, assignment)
    return config


def run_client(
    *,
    model: str,
    tokenizer: str,
    port: int,
    isl: int,
    osl: int,
    concurrency: int,
    num_prompts: int,
    seed: int,
    server: subprocess.Popen,
    log_path: Path,
    client_log: Path,
    save_to: Path | None = None,
    metadata: dict | None = None,
) -> float:
    argv = [
        sys.executable, "-m", "tensorrt_llm.serve.scripts.benchmark_serving",
        "--model", model,
        "--tokenizer", tokenizer,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dataset-name", "random",
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--random-range-ratio", "0",
        "--random-ids",
        "--tokenize-on-client",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--seed", str(seed),
        "--ignore-eos",
        "--percentile-metrics", PERCENTILE_METRICS,
        "--metric-percentiles", METRIC_PERCENTILES,
        "--disable-tqdm",
    ]  # fmt: skip
    if save_to is not None:
        argv += ["--save-result", "--result-dir", str(save_to.parent)]
        argv += ["--result-filename", save_to.name]
        argv += ["--metadata"] + [f"{k}={v}" for k, v in (metadata or {}).items()]
    started = time.monotonic()
    with client_log.open("ab") as log:
        log.write(f"\n=== client con={concurrency} n={num_prompts} seed={seed}\n".encode())
        log.flush()
        client = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    deadline = started + CLIENT_POINT_TIMEOUT_S
    while True:
        try:
            code = client.wait(timeout=SERVER_LIVENESS_POLL_S)
        except subprocess.TimeoutExpired:
            pass
        else:
            if code != 0:
                raise SystemExit(
                    f"benchmark_serving exited with {code} at concurrency {concurrency}; tail of "
                    f"{client_log}:\n{runtime.tail(client_log)}"
                )
            return time.monotonic() - started
        if server.poll() is not None:
            client.kill()
            client.wait()
            raise SystemExit(
                f"server exited with {server.returncode} during concurrency {concurrency}; "
                f"tail of {log_path}:\n{runtime.tail(log_path)}"
            )
        if time.monotonic() >= deadline:
            client.kill()
            client.wait()
            raise SystemExit(
                f"client exceeded {CLIENT_POINT_TIMEOUT_S}s at concurrency {concurrency}"
            )


class NsysWindow:
    """Collect ``duration`` s from ``delay`` s after ``start()``, through nsys session control."""

    def __init__(self, session: str, output: str, delay: float, duration: float) -> None:
        import threading

        self.session, self.output, self.delay, self.duration = session, output, delay, duration
        self.log: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _nsys(self, *args: str) -> None:
        out = subprocess.run(["nsys", *args], capture_output=True, text=True, timeout=1800)
        self.log.append(
            f"nsys {' '.join(args)} -> rc={out.returncode} {out.stdout[-400:]} {out.stderr[-400:]}"
        )

    def _run(self) -> None:
        time.sleep(self.delay)
        self._nsys("start", f"--session={self.session}", "-o", self.output, "-f", "true")
        time.sleep(self.duration)
        self._nsys("stop", f"--session={self.session}")

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)


def git(tree: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(tree), *args], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def tree_identity(tree: Path) -> dict:
    """What the measurement ran on: HEAD, and a digest of the uncommitted diff."""
    diff = git(tree, "diff", "HEAD")
    return {
        "tree": str(tree),
        "commit": git(tree, "rev-parse", "HEAD") or None,
        "branch": git(tree, "rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty_files": [
            line for line in git(tree, "status", "--porcelain", "-uno").splitlines() if line
        ],
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def gpu_identity() -> dict:
    device = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    query = "name,driver_version,memory.total,clocks.max.sm,power.limit"
    out = subprocess.run(
        ["nvidia-smi", "-i", device, f"--query-gpu={query}", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    return {"device": device, "query": query, "value": out or None}


def host_load() -> dict:
    try:
        load = os.getloadavg()
    except OSError:
        load = None
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    return {"loadavg": load, "compute_apps": apps.splitlines() if apps else []}


def summarize(out_dir: Path, concurrencies: list[int], world_size: int) -> list[dict]:
    rows = []
    for concurrency in concurrencies:
        record = json.loads((out_dir / point_file(concurrency)).read_text())
        tokens = record.get("total_output_tokens")
        if tokens is None:
            tokens = (record.get("output_throughput") or 0) * (record.get("duration") or 0)
        rate = user_rate(tokens, record.get("completed") or 0, record.get("mean_e2el_ms") or 0)
        rows.append(
            {
                "concurrency": concurrency,
                "completed": record.get("completed"),
                "duration_s": record.get("duration"),
                "output_tok_s": record.get("output_throughput"),
                "output_tok_s_gpu": (record.get("output_throughput") or 0) / world_size,
                "total_tok_s_gpu": (record.get("total_token_throughput") or 0) / world_size,
                "tok_s_user": record.get("user_throughput") if rate is None else rate,
                "tok_s_user_basis": PER_REQUEST_BASIS if rate is None else USER_RATE_BASIS,
                "tok_s_user_per_request": record.get("user_throughput"),
                "mean_ttft_ms": record.get("mean_ttft_ms"),
                "p99_ttft_ms": record.get("p99_ttft_ms"),
                "mean_tpot_ms": record.get("mean_tpot_ms"),
                "p99_tpot_ms": record.get("p99_tpot_ms"),
                "mean_itl_ms": record.get("mean_itl_ms"),
                "mean_e2el_ms": record.get("mean_e2el_ms"),
                "request_throughput": record.get("request_throughput"),
            }
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    header = (
        f"{'con':>4} {'tok/s/gpu':>10} {'tok/s/user':>11} {'ttft ms':>9} {'p99 ttft':>9} "
        f"{'tpot ms':>8} {'p99 tpot':>9} {'done':>5}"
    )
    print(f"\n{header}\n{'-' * len(header)}")
    for r in rows:
        print(
            f"{r['concurrency']:>4} {r['output_tok_s_gpu']:>10.1f} {r['tok_s_user'] or 0:>11.2f} "
            f"{r['mean_ttft_ms'] or 0:>9.1f} {r['p99_ttft_ms'] or 0:>9.1f} {r['mean_tpot_ms'] or 0:>8.2f} "
            f"{r['p99_tpot_ms'] or 0:>9.2f} {r['completed']:>5}"
        )


def warmup_shape(
    concurrencies: list[int], concurrency: int = 0, prompts: int = 0
) -> tuple[int, int]:
    """Warmup shape: 0 selects the largest concurrency and twice that many prompts."""
    con = concurrency or max(concurrencies or [16])
    return con, prompts or 2 * con


def num_prompts_for(concurrency: int, rounds: int | None, fixed: dict[int, int]) -> int:
    if concurrency in fixed:
        return fixed[concurrency]
    if rounds:
        return concurrency * rounds
    # Low-concurrency points get more rounds: so few requests otherwise make
    # ttft/tpot noisy. High-concurrency points need enough waves for the
    # steady state (prefill interleaved with decode) to dominate the ramp.
    return concurrency * (16 if concurrency <= 8 else 4)


def main() -> int:
    signal.signal(signal.SIGTERM, runtime._terminate)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="HF checkpoint dir served by trtllm-serve")
    parser.add_argument("--label", required=True, help="label; data lands in <out-root>/<label>/")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--config", action="append", type=Path, default=[], help="LLM-API yaml; repeat to layer"
    )
    parser.add_argument("--set", action="append", default=[], help="dotted key=value applied last")
    parser.add_argument("--isl", type=int, default=8192)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--concurrency", default="1,128", help="comma-separated points")
    parser.add_argument(
        "--rounds", type=int, default=None, help="requests per point = concurrency x rounds"
    )
    parser.add_argument("--num-prompts", default="", help="per-point override, e.g. 1:16,128:512")
    # 0 = the largest concurrency and twice that many prompts: a smaller warmup leaves its batch
    # shapes (JIT kernels, autotuner tactics, mixed-step caches) to be built in its measured window
    # on a fresh session, which then reads it low by more than the noise floor.
    parser.add_argument("--warmup-prompts", type=int, default=0)
    parser.add_argument("--warmup-concurrency", type=int, default=0)
    parser.add_argument("--warmup-osl", type=int, default=64)
    parser.add_argument(
        "--tree", type=Path, default=None, help="checkout the measurement runs on (identity only)"
    )
    parser.add_argument(
        "--nsys", default=None, metavar="PREFIX", help="profile the server under nsys"
    )
    parser.add_argument(
        "--nsys-extra", default="", help="extra nsys profile flags (space separated)"
    )
    parser.add_argument("--nsys-mode", choices=["session", "api"], default="session")
    parser.add_argument(
        "--nsys-delay", type=float, default=30.0, help="session mode: seconds into the first point"
    )
    parser.add_argument(
        "--nsys-duration", type=float, default=3.0, help="session mode: seconds collected"
    )
    parser.add_argument(
        "--allow-auto", action="store_true", help="permit modeling_v2: auto (not recommended)"
    )
    parser.add_argument("--force", action="store_true", help="overwrite a completed label")
    parser.add_argument(
        "--max-used-mib", type=int, default=2048, help="refuse a device with more memory in use"
    )
    args = parser.parse_args()

    concurrencies = [int(c) for c in args.concurrency.split(",") if c]
    args.warmup_concurrency, args.warmup_prompts = warmup_shape(
        concurrencies, args.warmup_concurrency, args.warmup_prompts
    )
    fixed = {}
    for item in filter(None, args.num_prompts.split(",")):
        con, _, n = item.partition(":")
        fixed[int(con)] = int(n)
    config = resolve_config(args.config, args.set)
    mode = config.get("modeling_v2", "off")
    if mode == "auto" and not args.allow_auto:
        raise SystemExit("modeling_v2: auto can silently serve the stock model; use require or off")
    world_size = (
        int(config.get("tensor_parallel_size", 1))
        * int(config.get("pipeline_parallel_size", 1))
        * int(config.get("context_parallel_size", 1))
    )
    env_overrides = config.get("env_overrides") or {}

    # A device another process still holds (e.g. a server stranded by an earlier
    # job) measures that process as much as this one.
    used = runtime.device_memory_used_mib(world_size)
    if used is not None and used > args.max_used_mib:
        raise SystemExit(
            f"device not clean: {used} MiB already in use (> {args.max_used_mib}); "
            f"compute apps: {host_load()['compute_apps']}"
        )

    final_dir = args.out_root / args.label
    if (final_dir / "meta.json").exists() and not args.force:
        raise SystemExit(f"{final_dir} already holds a completed run; use --force")
    out_dir = final_dir.with_name(final_dir.name + ".partial")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    serve = shutil.which("trtllm-serve") or str(Path(sys.executable).with_name("trtllm-serve"))
    port = runtime.free_port()
    launcher = shutil.which("trtllm-llmapi-launch") or str(
        Path(sys.executable).with_name("trtllm-llmapi-launch")
    )
    server_argv = [
        *runtime.rank_launch(world_size, launcher),
        serve, args.model, "--host", "127.0.0.1", "--port", str(port), "--config", str(config_path),
    ]  # fmt: skip
    nsys_session = ""
    if args.nsys and args.nsys_mode == "api":
        # TensorRT-LLM's documented capture: the iteration window comes from
        # TLLM_PROFILE_START_STOP, `-c cudaProfilerApi` collects only inside it,
        # and `--capture-range-end=stop` keeps the server serving after the
        # window closes. The in-process cuProfilerStop can hang a modeling_v2
        # server; prefer --nsys-mode session.
        server_argv = [
            "nsys", "profile", "-o", args.nsys, "-f", "true",
            "-t", "cuda,nvtx", "-c", "cudaProfilerApi",
            "--cuda-graph-trace", os.environ.get("MV2FLOW_NSYS_GRAPH_TRACE", "node"),
            "--capture-range-end=stop", "--gpu-metrics-devices=none",
            *filter(None, args.nsys_extra.split(" ")),
        ] + server_argv  # fmt: skip
    elif args.nsys:
        # Session mode: nsys injects at launch and collects only between an
        # external `nsys start` and `nsys stop`, timed from the measured point
        # (see NsysWindow). Nothing inside the server calls the profiler API,
        # so nothing in it can block on the flush.
        nsys_session = f"bench-{os.getpid()}"
        server_argv = [
            "nsys", "launch", f"--session-new={nsys_session}", "-t", "cuda,nvtx",
            "--cuda-graph-trace", os.environ.get("MV2FLOW_NSYS_GRAPH_TRACE", "node"),
            *filter(None, args.nsys_extra.split(" ")),
        ] + server_argv  # fmt: skip

    env = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"} | dict(os.environ)
    env.update({str(k): str(v) for k, v in env_overrides.items()})
    log_path = out_dir / "server.log"
    client_log = out_dir / "client.log"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    load_before = host_load()
    print(
        f"[perf] label={args.label} modeling_v2={mode} isl={args.isl} osl={args.osl} points={concurrencies}"
    )
    print(f"[perf] server log: {log_path}", flush=True)
    timings: dict = {}
    holders_before = runtime.device_holders(world_size)
    server = runtime.start_server(server_argv, log_path, env)
    try:
        timings["boot_s"] = runtime.wait_healthy(
            port, server, log_path, NSYS_BOOT_TIMEOUT_S if args.nsys else SERVER_BOOT_TIMEOUT_S
        )
        print(f"[perf] healthy after {timings['boot_s']:.0f}s; warmup", flush=True)
        timings["warmup_s"] = run_client(
            model=args.model, tokenizer=args.model, port=port, isl=args.isl, osl=args.warmup_osl,
            concurrency=args.warmup_concurrency, num_prompts=args.warmup_prompts, seed=WARMUP_SEED,
            server=server, log_path=log_path, client_log=client_log,
        )  # fmt: skip
        window = None
        for concurrency in concurrencies:
            n = num_prompts_for(concurrency, args.rounds, fixed)
            print(f"[perf] con={concurrency} num_prompts={n}", flush=True)
            if nsys_session and window is None:
                window = NsysWindow(nsys_session, args.nsys, args.nsys_delay, args.nsys_duration)
                window.start()
            timings[f"con{concurrency}_s"] = run_client(
                model=args.model, tokenizer=args.model, port=port, isl=args.isl, osl=args.osl,
                concurrency=concurrency, num_prompts=n, seed=concurrency,
                server=server, log_path=log_path, client_log=client_log,
                save_to=out_dir / point_file(concurrency),
                metadata={"label": args.label, "isl": args.isl, "osl": args.osl, "seed": concurrency},
            )  # fmt: skip
        if window is not None:
            window.join(timeout=1800)
            timings["nsys"] = window.log
            print("[perf] nsys: " + " | ".join(window.log), flush=True)
    finally:
        runtime.stop_server(
            server,
            world_size,
            holders_before,
            grace_s=900 if args.nsys else 90,
            log=lambda message: print(message, flush=True),
            report_remaining=False,
        )

    rows = summarize(out_dir, concurrencies, world_size)
    # The tree the scheduler put first on the path (and whose import it checked)
    # is the one measured; `--tree` names it only when the job ran without one.
    tree = (
        Path(os.environ["MV2FLOW_TREE"])
        if os.environ.get("MV2FLOW_TREE")
        else (args.tree or Path.cwd())
    )
    try:
        import importlib.metadata as md

        versions = {p: md.version(p) for p in ("tensorrt_llm", "flashinfer-python", "torch")}
    except Exception as exc:  # noqa: BLE001 - identity is best-effort, never fails a sweep
        versions = {"error": repr(exc)}
    meta = {
        "label": args.label,
        "modeling_v2": mode,
        "model": args.model,
        "config": config,
        "world_size": world_size,
        "isl": args.isl,
        "osl": args.osl,
        "concurrencies": concurrencies,
        "num_prompts": {c: num_prompts_for(c, args.rounds, fixed) for c in concurrencies},
        "warmup": {
            "prompts": args.warmup_prompts,
            "concurrency": args.warmup_concurrency,
            "osl": args.warmup_osl,
        },
        "hostname": socket.gethostname(),
        "gpu": gpu_identity(),
        "host_before": load_before,
        "host_after": host_load(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "versions": versions,
        "tree": tree_identity(tree),
        "timings": timings,
        "nsys": args.nsys,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if final_dir.exists():
        shutil.rmtree(final_dir)
    out_dir.rename(final_dir)
    print_summary(rows)
    print(f"\n[perf] done: {final_dir}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"[perf] FAILED: {exc.code}", flush=True)
        raise
    except Exception as exc:  # noqa: BLE001 - the sentinel line is the caller contract
        print(f"[perf] FAILED: {type(exc).__name__}: {exc}", flush=True)
        raise
