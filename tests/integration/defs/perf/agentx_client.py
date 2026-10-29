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
"""AgentX benchmark client for perf-sanity disaggregated lanes.

AgentX is an agentic, multi-turn *trace replay* workload. It is driven by
``aiperf`` from the SemiAnalysis ``agentx-harness`` fork, whose
``inferencex-agentx-mvp`` scenario replays a recorded conversation corpus for a
fixed **wall-clock duration** rather than a fixed prompt count. That single
difference is why it cannot reuse
``tensorrt_llm.serve.scripts.benchmark_serving``: there is no ISL/OSL/num-prompts
to give it, and the run length is time-bounded.

This module is the adapter between the perf-sanity harness and that client. It

1. installs the pinned ``agentx-harness`` build (see ``AGENTX_AIPERF_REF``),
2. verifies the endpoint really serves one token before spending GPU-hours,
3. runs ``aiperf profile``, and
4. translates ``profile_export_aiperf.json`` into the exact stdout lines that
   ``PERF_METRIC_LOG_QUERIES`` in ``test_perf_sanity.py`` already scans for.

Step 4 is the reason this is a separate process rather than harness code: by
emitting ``benchmark_serving``'s own report format verbatim, the perf-sanity
metric scanner, database upload and regression check all keep working unchanged.

Two deliberate deviations from the upstream ``run_benchmark_agentx.sh``:

* **Concurrency is passed through, not multiplied.** The reference script
  computes ``concurrency * num_gen_servers`` because its YAML states per-server
  concurrency. perf-sanity's ``concurrency_list`` is already the whole-cluster
  total (it feeds ``--max-concurrency`` directly in the default client), so
  scaling it here would silently over-drive the lane.
* **Readiness is re-verified, briefly.** The harness has already waited for the
  disagg server, so the long poll is redundant; the one-token probe is kept
  because it is the only cheap check that proves the ctx->gen KV path works.
  Without it a broken KV transfer surfaces an hour later as a client-side
  "failed request threshold" error, sending the reader to the wrong logs.

All tuning knobs are read from the environment with the same names and defaults
as the reference script, so a lane is configured purely through the YAML's
``environment.client_env_var``.
"""

import argparse
import http.client
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

# Pinned agentx-harness build. It installs *as* ``aiperf`` (shadowing any
# PyPI aiperf), so the version is pinned by commit rather than by release.
DEFAULT_AIPERF_REF = (
    "git+https://github.com/SemiAnalysisAI/agentx-harness.git"
    "@754356e9a39acc6cc6afb242d123bb57c3fb6f75"
)

# aiperf's idle-connection keep-alive, in seconds. Must stay below the server's
# own keep-alive timeout: trtllm-serve's is 5 s (TIMEOUT_KEEP_ALIVE in
# tensorrt_llm/serve/openai_server.py).
AIPERF_HTTP_KEEPALIVE_TIMEOUT_S = 4

# aiperf's CLI/serialization/UI dependencies. The TRT-LLM container already
# carries its ML stack (numpy/pandas/torch/huggingface-hub/aiohttp); installing
# only this set has been verified to add packages without reinstalling the
# pinned scientific stack.
AIPERF_EXTRA_DEPS = [
    "cyclopts",
    "ruamel.yaml",
    "msgspec",
    "orjson",
    "kaleido",
    "dash",
    "dash-bootstrap-components",
    "textual",
    "starlette-compress",
    "setproctitle",
    "jmespath",
    "aiofiles",
    "ffmpeg-python",
    "pydantic-settings",
    "prometheus_client",
    "fastapi",
    "soundfile",
    "plotly",
    "seaborn",
    "tiktoken",
    "tqdm",
    "uvicorn[standard]",
    "uvloop",
    "crick",
    "zstandard",
]

# The container ships datasets 3.x, which cannot parse the trace corpus schema
# ("Feature type 'Json' not found"). 4.x+ is required.
MIN_DATASETS_MAJOR = 4

# The aiperf scenario that defines the AgentX workload. Only the agentx-harness
# fork registers it, so its presence is also how the fork is told apart from the
# upstream PyPI aiperf of the same import name.
AGENTX_SCENARIO = "inferencex-agentx-mvp"

# PyExecutor's HangDetector is hard-coded at 300 s with no env override, so a
# multi-minute pip install with zero requests in flight idles the GEN worker
# into MPI_Abort -- which then presents as a *client* connection failure. A
# best-effort 1-token request on this period keeps the executor awake.
HEARTBEAT_PERIOD_S = 60

# Mapping from perf-sanity's report labels to aiperf metric fields.
#
# Two of these are easy to get wrong and are worth stating explicitly, because
# both names exist in the export and differ by ~47x:
#   * ``inter_chunk_latency`` is pooled over every streamed chunk (count == total
#     chunks) -- that is ITL.
#   * ``inter_token_latency`` is already averaged per request (count == request
#     count) -- that is TPOT.
# ``user_throughput`` likewise maps to ``e2e_output_token_throughput`` (which
# includes TTFT, matching benchmark_serving's definition) and NOT to the
# similarly-named ``output_token_throughput_per_user`` (decode-only).
#
# aiperf reports every latency in ms and every throughput in tokens/sec or
# requests/sec, which is what the labels below claim, so no unit conversion is
# applied. ``_read_metric`` asserts the unit to keep that assumption honest.
THROUGHPUT_METRICS = [
    ("Request throughput (req/s)", "request_throughput", "requests/sec"),
    ("Output token throughput (tok/s)", "output_token_throughput", "tokens/sec"),
    ("Total Token throughput (tok/s)", "total_token_throughput", "tokens/sec"),
    ("User throughput (tok/s)", "e2e_output_token_throughput", "tokens/sec/user"),
]

# (label stem, aiperf field). Each contributes Mean/Median/P99 lines.
LATENCY_METRICS = [
    ("TTFT", "time_to_first_token"),
    ("ITL", "inter_chunk_latency"),
    ("TPOT", "inter_token_latency"),
    ("E2EL", "request_latency"),
]

# benchmark_serving's report format. Reproduced exactly so the existing
# regex scanner needs no change; see _print_metric for why width matters.
_REPORT_FORMAT = "{:<40} {:<10.2f}"


def _log(msg: str) -> None:
    print(f"[agentx] {msg}", flush=True)


def _print_metric(label: str, value: float) -> None:
    """Emit one metric line in benchmark_serving's exact report format.

    ``PERF_METRIC_LOG_QUERIES`` requires at least one whitespace character
    between the colon and the number. The 40-char field supplies it only while
    ``len(label) + 1 < 40``, which holds for every label here (longest is 30);
    a longer label would silently stop matching, hence the assert.
    """
    stem = f"{label}:"
    assert len(stem) < 40, f"label {label!r} too long to leave a separator"
    print(_REPORT_FORMAT.format(stem, value), flush=True)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def aiperf_is_usable() -> bool:
    """Return whether the ambient interpreter already has a working agentx stack.

    Checks importability, that the ``aiperf`` found is the agentx-harness fork
    (it alone ships the ``inferencex-agentx-mvp`` scenario), *and* the datasets
    major version, since a container with aiperf but datasets 3.x fails only
    later, when loading the corpus. The fork check matters because
    requirements-dev.txt pins the upstream PyPI ``aiperf``, which imports fine
    but rejects ``--scenario`` at the start of the run.

    The scenario is looked up through ``aiperf.common.scenario.registry``, the
    module path in the pinned ``DEFAULT_AIPERF_REF`` build. The reason for a
    failed probe is logged, so a fork that moves that module shows up as such
    instead of as a reinstall on every run.
    """
    probe = (
        "import sys\n"
        "try:\n"
        "    import aiperf, cyclopts, datasets, transformers, pydantic, msgspec, orjson\n"
        "except ImportError as e:\n"
        "    sys.exit(f'missing module: {e}')\n"
        "try:\n"
        "    from aiperf.common.scenario.registry import get_scenario\n"
        "except ImportError:\n"
        "    sys.exit('aiperf has no aiperf.common.scenario.registry: not the agentx-harness fork, '\n"
        "             'or the fork moved its scenario registry')\n"
        "try:\n"
        f"    get_scenario({AGENTX_SCENARIO!r})\n"
        "except Exception as e:\n"
        f"    sys.exit(f'aiperf does not provide the {AGENTX_SCENARIO} scenario: {{e}}')\n"
        f"if int(datasets.__version__.split('.')[0]) < {MIN_DATASETS_MAJOR}:\n"
        f"    sys.exit(f'datasets {{datasets.__version__}} < {MIN_DATASETS_MAJOR}')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        _log(f"aiperf stack not usable: {lines[-1] if lines else f'exit {result.returncode}'}")
    return result.returncode == 0


def _http_error_text(error: urllib.error.HTTPError) -> str:
    """Describe an HTTP error as ``HTTP <code>: <first 2 KB of its body>`` and close it.

    A body that cannot be read still yields a description, so the best-effort
    callers never see an exception from here.
    """
    with error:
        try:
            body = error.read(2048).decode(errors="replace")
        except (OSError, ValueError, http.client.HTTPException) as read_error:
            body = f"<unreadable body: {read_error!r}>"
    return f"HTTP {error.code}: {body}"


def post_one_token(
    url: str, model: str, timeout: float, *, fail_on_client_error: bool = False
) -> "str | None":
    """POST one token and return the response body.

    An HTTP error comes back as ``HTTP <code>: <body>`` so callers can report the
    server's reason. With ``fail_on_client_error``, a permanent 4xx (anything but
    408/429) raises instead. Transport failures return None.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode(errors="replace")
    except urllib.error.HTTPError as error:
        reason = _http_error_text(error)
        if fail_on_client_error and 400 <= error.code < 500 and error.code not in (408, 429):
            raise RuntimeError(
                f"{req.full_url} rejected the 1-token readiness request: {reason}"
            ) from error
        return reason
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


class _Heartbeat:
    """Keeps the GEN worker's hang detector at bay during a long pip install."""

    def __init__(self, url: str, model: str):
        self._url = url
        self._model = model
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(HEARTBEAT_PERIOD_S):
            # Best-effort by construction: post_one_token swallows every
            # network error, so the heartbeat can never fail the run.
            post_one_token(self._url, self._model, timeout=60)

    def __enter__(self) -> "_Heartbeat":
        self._thread.start()
        _log(f"heartbeat armed ({HEARTBEAT_PERIOD_S}s) to hold off the 300s hang detector")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join(timeout=HEARTBEAT_PERIOD_S + 30)


def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *args])


def ensure_aiperf(url: str, model: str) -> None:
    """Install the pinned agentx build unless a usable one is already present."""
    if os.environ.get("AGENTX_FORCE_REINSTALL", "0") == "1":
        _log("AGENTX_FORCE_REINSTALL=1 -> reinstalling unconditionally")
    elif aiperf_is_usable():
        _log("aiperf + deps already usable; skipping install")
        return

    ref = os.environ.get("AGENTX_AIPERF_REF", DEFAULT_AIPERF_REF)
    with _Heartbeat(url, model):
        _log(f"installing aiperf (agentx) from {ref}")
        _pip("--force-reinstall", "--no-deps", f"aiperf @ {ref}")
        _pip(*AIPERF_EXTRA_DEPS)
        _pip("-U", "datasets")
        _pip("transformers==5.8.1")
        # aiperf needs a newer pydantic than the container pins, but its own
        # dependency resolution would drag in a fresh ML stack; pin the pair
        # directly with --no-deps instead.
        _pip("--force-reinstall", "--no-deps", "pydantic==2.12.5", "pydantic-core==2.41.5")

    if not aiperf_is_usable():
        raise RuntimeError(
            "aiperf is still not importable after install; refusing to start the "
            "benchmark (see the pip output above)"
        )


def _is_completion(response: "str | None") -> bool:
    """Whether a post_one_token result is a chat completion, not an error description."""
    try:
        payload = json.loads(response or "")
    except ValueError:
        return False
    return isinstance(payload, dict) and "choices" in payload


def verify_endpoint(url: str, model: str) -> None:
    """Prove the endpoint serves one real token before spending GPU-hours.

    A healthy process may still reject inference requests or fail to generate.
    This check applies to both aggregated and disaggregated servers. Permanent
    request errors fail immediately with the response body; transient failures
    retain the readiness retry budget.
    """
    timeout = _env_int("AGENTX_READY_TIMEOUT", 900)
    interval = _env_int("AGENTX_READY_INTERVAL", 15)
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = post_one_token(url, model, timeout=300, fail_on_client_error=True)
        if _is_completion(last):
            _log("1-token completion OK -- the inference endpoint is ready")
            return
        _log(f"not serving yet (response head: {str(last)[:200]})")
        time.sleep(interval)
    raise RuntimeError(
        f"{url} could not complete ONE token within {timeout}s. "
        f"Inspect the inference server logs and readiness request failures. "
        f"Last response: {str(last)[:500]}"
    )


def build_aiperf_cmd(args: argparse.Namespace, url: str, artifact_dir: str) -> "list[str]":
    """Assemble the ``aiperf profile`` argv.

    Note ``--url`` carries scheme+host+port only; the request path is a separate
    ``--endpoint``. That is why this client takes ``--host``/``--port`` and
    composes them itself instead of having them appended to the argv.

    aiperf runs as ``python -m aiperf`` under ``sys.executable``, the interpreter
    that ``aiperf_is_usable`` checked and ``ensure_aiperf`` installed into. A bare
    ``aiperf`` on PATH could be a different installation, such as the upstream
    PyPI build that has no AgentX scenario.
    """
    cmd = [
        sys.executable,
        "-m",
        "aiperf",
        "profile",
        "--scenario",
        AGENTX_SCENARIO,
        "-m",
        args.model,
        "--tokenizer-trust-remote-code",
        "--url",
        url,
        "--endpoint",
        "/v1/chat/completions",
        "--endpoint-type",
        "chat",
        "--streaming",
        "--public-dataset",
        args.dataset,
        "--max-context-length",
        str(_env_int("AGENTX_MAX_CTX", 1048576)),
        "--trajectory-start-min-ratio",
        str(_env_float("AGENTX_MIN_RATIO", 0.25)),
        "--trajectory-start-max-ratio",
        str(_env_float("AGENTX_MAX_RATIO", 0.75)),
        "--failed-request-threshold",
        str(_env_float("AGENTX_FAILED_THRESH", 0.1)),
        "--use-server-token-count",
        "--trace-idle-gap-cap-seconds",
        "300",
        "--no-gpu-telemetry",
        "--random-seed",
        str(_env_int("AGENTX_SEED", 42)),
        "--benchmark-duration",
        str(_env_int("AGENTX_DURATION", 3600)),
        "--concurrency",
        str(args.concurrency),
        "--artifact-dir",
        artifact_dir,
        "--ui",
        "simple",
    ]

    # --benchmark-duration is always set; adding --request-count makes the run
    # stop at whichever limit is hit first, which is only ever used for smoke
    # tests. Left unset in CI so the lane is purely duration-bounded.
    mult = os.environ.get("AGENTX_REQUEST_COUNT_MULT")
    if mult:
        count = max(args.concurrency, args.concurrency * int(mult))
        _log(f"request-count = max({args.concurrency}, {mult} x {args.concurrency}) = {count}")
        cmd += ["--request-count", str(count)]

    # aiperf runs warmup as a distinct phase reported under a separate
    # top-level `warmup_metrics` key, so these requests never pollute the
    # measured window. <= 0 omits the flag entirely.
    warmup = _env_int("AGENTX_WARMUP_PER_LANE", 10)
    if warmup > 0:
        cmd += ["--warmup-requests-per-lane", str(warmup)]

    # The scenario forces ignore_eos, so the model can emit its end-of-sequence
    # token as an ordinary output token. With the server's default
    # skip_special_tokens it decodes to no text, and a one-token response
    # (every warmup request, and every ctx_only request) then carries no
    # content, which aiperf rejects as InvalidInferenceResultError. Keeping
    # special tokens in the text makes every generated token visible. The token
    # counts the metrics are built from stay the same only because the command
    # also passes --use-server-token-count: without it aiperf would re-tokenize
    # this text, special-token strings included, and count differently.
    cmd += ["--extra-inputs", "skip_special_tokens:false"]

    # Prefill-only replay (the perf-sanity ctx_only mode). The trace sets
    # max_tokens per turn, so a fixed output length cannot be passed the way
    # benchmark_serving takes --random-output-len. aiperf merges --extra-inputs
    # into every request body after the turn's own max_completion_tokens, so
    # this overrides it for every turn, subagent turns included. With the
    # scenario's forced ignore_eos every request then decodes exactly one token.
    if args.max_output_tokens is not None:
        cmd += ["--extra-inputs", f"max_completion_tokens:{args.max_output_tokens}"]

    # Plain os.environ.get with a default, mirroring the reference script's
    # `${VAR-default}`: an explicitly empty value must disable the flag, which
    # is how a production (>=900 s) run drops --unsafe-override.
    extra = os.environ.get("AGENTX_EXTRA_ARGS", "--unsafe-override")
    if extra.strip():
        cmd += extra.split()
    return cmd


def _read_metric(export: dict, field: str, stat: str, expect_unit: "str | None" = None) -> float:
    """Read one statistic out of the aiperf export, failing loudly if absent.

    A silently-missing metric would leave the perf-sanity scanner short a line,
    which it reports as a vague "metrics are missing" much later; and a unit
    change would corrupt a baseline without any error at all. Both are checked
    here, at the point where the assumption is actually made.
    """
    entry = export.get(field)
    if not isinstance(entry, dict):
        raise RuntimeError(f"aiperf export has no metric {field!r}")
    if expect_unit is not None and entry.get("unit") != expect_unit:
        raise RuntimeError(
            f"aiperf metric {field!r} changed unit: expected {expect_unit!r}, "
            f"got {entry.get('unit')!r}. Refusing to upload a mis-scaled value."
        )
    if stat not in entry:
        raise RuntimeError(f"aiperf metric {field!r} has no {stat!r} (keys: {sorted(entry)})")
    return float(entry[stat])


def check_run_health(export: dict) -> None:
    """Fail unless the export shows a valid, complete, error-free measurement.

    Deliberately gates on *content*, never on aiperf's exit status: a run that
    served nothing but HTTP 500s can still exit 0, and conversely the artifacts
    are flushed before the process tears down. ``submission_valid`` is the
    scenario's own verdict and is the single most important field here -- it
    goes false when the measured window does not actually cover the requested
    duration, which is exactly the failure mode that makes a short run look
    like a fast one.
    """
    metadata = export.get("metadata") or {}

    if export.get("was_cancelled"):
        raise RuntimeError("aiperf reports was_cancelled=true; the measurement is incomplete")

    # Report coverage before the verdict: when submission_valid is false this
    # is the number that says why, so it must be in the log either way.
    for phase in metadata.get("metric_duration_coverage") or []:
        _log(
            "coverage[{}]: ttft={:.4f} itl={:.4f} required={} expected_duration={}s".format(
                phase.get("phase_name"),
                phase.get("ttft_ratio", float("nan")),
                phase.get("inter_token_latency_ratio", float("nan")),
                phase.get("required_ratio"),
                phase.get("expected_duration_seconds"),
            )
        )

    if "submission_valid" not in metadata:
        raise RuntimeError(
            "aiperf export has no metadata.submission_valid; the agentx scenario "
            "did not complete (schema_version="
            f"{export.get('schema_version')!r})"
        )
    if not metadata["submission_valid"]:
        raise RuntimeError(
            "aiperf reports submission_valid=false: the measured window does not "
            "cover the requested duration, so these numbers are not comparable to "
            "a baseline. See the coverage line above."
        )

    requests = _read_metric(export, "request_count", "avg")
    duration = _read_metric(export, "benchmark_duration", "avg", "sec")
    if requests <= 0:
        raise RuntimeError("aiperf completed 0 requests")
    check_request_errors(export.get("error_summary") or [], requests)
    _log(f"health OK: {int(requests)} requests over {duration:.1f}s, submission_valid=true")


# A streamed response that closed after the role chunk but before any content.
# It is a server/transport flake, not a slow or wrong result, and it lands on
# long-running requests: an e2e lane sees ~1 in 30k, a prefill-only (ctx_only)
# lane, whose every request is a long prefill before its single token, ~1 in 600.
EMPTY_CONTENT_ERROR_TYPE = "InvalidInferenceResultError"
# Largest share of measured requests that may be empty-content before the run
# fails. Well above the observed ctx_only rate (0.17%), far below anything that
# could move a throughput or latency statistic.
MAX_EMPTY_CONTENT_ERROR_RATIO = 0.005


def check_request_errors(errors: list, requests: float) -> None:
    """Fail on request errors, except a small share of empty-content responses.

    Args:
        errors: ``error_summary`` from the aiperf export: ``{"error_details":
            {"type": ...}, "count": N}`` entries for the profiling phase.
        requests: Successful requests in the profiling phase.

    Raises:
        RuntimeError: On any other error type, or empty-content errors above
            ``MAX_EMPTY_CONTENT_ERROR_RATIO`` of all measured requests.
    """
    if not errors:
        return
    other = [
        e for e in errors if (e.get("error_details") or {}).get("type") != EMPTY_CONTENT_ERROR_TYPE
    ]
    if other:
        raise RuntimeError(f"aiperf reported request errors: {json.dumps(errors)[:2000]}")
    empty = sum(int(e.get("count", 0)) for e in errors)
    ratio = empty / (requests + empty)
    if ratio > MAX_EMPTY_CONTENT_ERROR_RATIO:
        raise RuntimeError(
            f"{empty} empty-content responses ({ratio:.2%} of measured requests) exceed the "
            f"{MAX_EMPTY_CONTENT_ERROR_RATIO:.1%} tolerance: {json.dumps(errors)[:2000]}"
        )
    _log(
        f"tolerating {empty} empty-content responses ({ratio:.3%} of measured requests, "
        f"limit {MAX_EMPTY_CONTENT_ERROR_RATIO:.1%}); details: {json.dumps(errors)[:500]}"
    )


# Decode-interval metrics. A one-token response has no interval between tokens,
# so aiperf records none of these for a prefill-only (--max-output-tokens 1) run.
DECODE_INTERVAL_FIELDS = ("inter_chunk_latency", "inter_token_latency")


def report_metrics(export: dict, prefill_only: bool = False) -> None:
    """Print every PERF_METRIC_LOG_QUERIES line from the aiperf export.

    With ``prefill_only`` every response is one token, so the decode-interval
    metrics do not exist. They are reported as 0 rather than omitted, because
    the harness requires every line. The database uses 0 for the same
    single-token case in benchmark_serving's ctx_only runs, so the series stay
    comparable. Throughput, TTFT and E2EL are measured normally.
    """
    for label, field, unit in THROUGHPUT_METRICS:
        _print_metric(label, _read_metric(export, field, "avg", unit))
    for stem, field in LATENCY_METRICS:
        absent = prefill_only and field in DECODE_INTERVAL_FIELDS and field not in export
        for prefix, stat in (("Mean", "avg"), ("Median", "p50"), ("P99", "p99")):
            value = 0.0 if absent else _read_metric(export, field, stat, "ms")
            _print_metric(f"{prefix} {stem} (ms)", value)


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model path or HF id served by the endpoint")
    parser.add_argument(
        "--concurrency",
        required=True,
        type=int,
        help="Total in-flight requests. Passed through to aiperf unscaled.",
    )
    parser.add_argument("--dataset", required=True, help="aiperf --public-dataset loader name")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument(
        "--artifact-dir",
        # The harness exports this so artifacts land in the test output
        # directory and are collected with the rest of the lane's logs.
        default=os.environ.get("TRTLLM_AGENTX_ARTIFACT_DIR", "agentx_artifacts"),
        help="Parent directory for aiperf artifacts; a concurrency_<N> subdir is created in it.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Cap every request's output to this many tokens, overriding the trace's "
            "per-turn max_tokens. 1 makes the replay prefill-only (perf-sanity ctx_only)."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be >= 1")
    return args


def _run(args: argparse.Namespace, artifact_dir: str) -> int:
    url = f"http://{args.host}:{args.port}"
    # Custom tokenizers (e.g. deepseek_v4) are loaded in aiperf's worker
    # processes, which do not inherit a CLI trust flag.
    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    # trtllm-serve closes a connection that has been idle for its keep-alive
    # timeout (5 s). aiperf keeps idle pooled connections for 300 s, so it can
    # write a request into a connection the server is closing at that moment,
    # which fails as ClientOSError "Connection reset by peer". Dropping idle
    # connections on the client side first avoids that.
    os.environ.setdefault("AIPERF_HTTP_KEEPALIVE_TIMEOUT", str(AIPERF_HTTP_KEEPALIVE_TIMEOUT_S))

    ensure_aiperf(url, args.model)
    verify_endpoint(url, args.model)

    cmd = build_aiperf_cmd(args, url, artifact_dir)
    export_path = os.path.join(artifact_dir, "profile_export_aiperf.json")
    # Drop any export left by an earlier invocation that reused this directory
    # (a retried lane, or a local rerun pointed at the same --output-dir).
    # Without this, a crashed aiperf would be validated against -- and would
    # report -- the previous run's numbers, turning a failure into a green
    # result carrying stale metrics. Removing it first makes the existence
    # check below a statement about *this* run.
    if os.path.exists(export_path):
        _log(f"removing export from a previous run: {export_path}")
        os.remove(export_path)
    _log("running: " + " ".join(cmd))
    grace = _env_int("AGENTX_TIMEOUT_GRACE", 900)
    completed = subprocess.run(cmd, timeout=_env_int("AGENTX_DURATION", 3600) + grace)

    if not os.path.exists(export_path):
        # No export at all means aiperf died before writing results; its exit
        # status is the only signal left, so surface it.
        raise RuntimeError(
            f"aiperf exited {completed.returncode} and wrote no {export_path}. "
            f"See the aiperf output above and {artifact_dir}/logs."
        )
    if completed.returncode != 0:
        # An export exists, so the health checks below are more informative
        # than the status; log it and let them make the call.
        _log(f"warning: aiperf exited {completed.returncode} but wrote an export; validating it")

    with open(export_path) as f:
        export = json.load(f)

    check_run_health(export)
    report_metrics(export, prefill_only=args.max_output_tokens == 1)
    return 0


def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)
    # aiperf writes its files flat into --artifact-dir, so give each
    # concurrency its own directory to keep exports from overwriting.
    artifact_dir = os.path.abspath(
        os.path.join(args.artifact_dir, f"concurrency_{args.concurrency}")
    )
    os.makedirs(artifact_dir, exist_ok=True)
    try:
        return _run(args, artifact_dir)
    except Exception as exc:
        # The perf-sanity harness invokes this via subprocess.check_output,
        # which discards stdout when the exit status is non-zero -- so a
        # failure would otherwise lose every diagnostic printed above. Persist
        # the reason next to aiperf's own logs, which are collected with the
        # test output.
        reason = f"{type(exc).__name__}: {exc}"
        _log(f"FAILED -- {reason}")
        try:
            with open(os.path.join(artifact_dir, "agentx_client_error.txt"), "w") as f:
                f.write(reason + "\n")
        except OSError:
            pass
        raise


if __name__ == "__main__":
    sys.exit(main())
