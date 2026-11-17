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
"""The un-measured warmup pass the perf tests send before each measured client.

The server's own startup warmup cannot reach every prefill shape a benchmark
produces: how many prompts arrive together, how they are chunked against
``max_num_tokens`` and which decode tokens share a step with them are only known
once real traffic arrives. JIT kernels keyed on those shapes would otherwise
compile inside the first measured requests. The warmup pass sends that traffic
once beforehand: the measured client's own ``benchmark_serving`` command, cut to
one wave of requests with different prompts. Nothing it prints is parsed or
uploaded.

perf-sanity (``test_perf_sanity.py``) runs it for plain aggregated cases,
``ctx_only`` and disagg ``e2e``, whose measured window contains prefill. The
``gen_only`` modes never get one: they measure decode only, and their generation
server's benchmark fill gate opens once, so warmup requests would open it before
the measured burst arrives.

The single-node perf tests (``test_perf.py``) run it for both runtimes. A
``serve`` case sends it before each measured client. A ``bench`` case measures
inside the ``trtllm-bench throughput`` process that loads the model, so its pass
is that command's own ``--warmup`` requests, raised to one wave.

``TRTLLM_PERF_WARMUP_PASS=0`` turns it off in both, e.g. to reproduce a JIT stall
the pass would otherwise absorb.
"""

import os
import re
import subprocess
import time
from typing import Callable, List, Mapping, NamedTuple, Optional, Sequence

WARMUP_PASS_ENV = "TRTLLM_PERF_WARMUP_PASS"

# Benchmark modes that get a warmup pass. A plain aggregated case has no
# benchmark mode, which the runner carries as "".
WARMUP_PASS_MODES = ("", "ctx_only", "e2e")

_DISABLED_VALUES = ("0", "false", "no", "off")

_NUM_PROMPTS_FLAG = "--num-prompts"
_SEED_FLAG = "--seed"
_NO_TEST_INPUT_FLAG = "--no-test-input"
# Flags whose value the warmup pass replaces or drops: the request count, the seed,
# and where the measured client writes its time breakdown and result file.
_FLAGS_WITH_REPLACED_VALUE = (
    _NUM_PROMPTS_FLAG,
    _SEED_FLAG,
    "--save-request-time-breakdown",
    "--result-dir",
    "--result-filename",
)
# Flags without a value that the warmup pass drops; it adds --no-test-input back.
_DROPPED_FLAGS = (_NO_TEST_INPUT_FLAG, "--save-result", "--save-detailed")
# benchmark_serving's --seed default.
_BENCHMARK_SERVING_DEFAULT_SEED = 0
# benchmark_serving's "Failed requests:" summary line, its only failure signal:
# benchmark_serving exits 0 even when requests fail.
_FAILED_REQUESTS_RE = re.compile(r"^Failed requests:\s+(\d+)", re.MULTILINE)


def warmup_pass_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the warmup pass is on.

    It is on unless ``TRTLLM_PERF_WARMUP_PASS`` is ``0``, ``false``, ``no`` or
    ``off`` (any case).

    Args:
        environ: Environment to read; ``os.environ`` when None.
    """
    env = os.environ if environ is None else environ
    return env.get(WARMUP_PASS_ENV, "1").strip().lower() not in _DISABLED_VALUES


def warmup_pass_applies(benchmark_mode: Optional[str]) -> bool:
    """Return whether a case in this benchmark mode gets a warmup pass.

    Args:
        benchmark_mode: The case's benchmark mode; None or "" for a plain
            aggregated case.
    """
    return (benchmark_mode or "") in WARMUP_PASS_MODES


def warmup_pass_plan(
    eligible_requests: Sequence[int], measured: Sequence[bool], time_breakdown: bool
) -> List[int]:
    """Return each client's warmup pass size for one server, in run order (0: none).

    The time breakdown leaves out only the records written before the server's
    first measured client, and pools every later record. So while it is collected,
    only that first measured client gets a pass: a later pass's records would count
    as measured for every client of the server. Without it, each client keeps its
    own pass.

    Args:
        eligible_requests: Each client's pass size on its own; 0 if it gets none.
        measured: Whether each client is measured; False for accuracy-only clients.
        time_breakdown: Whether the time breakdown is collected for the server.
    """
    plan: List[int] = []
    after_first_measured = False
    for num_requests, is_measured in zip(eligible_requests, measured):
        plan.append(0 if time_breakdown and after_first_measured else num_requests)
        after_first_measured = after_first_measured or is_measured
    return plan


def one_wave_requests(concurrency: int, num_requests: int, max_batch_size: int) -> int:
    """Return the requests in one wave of a benchmark: those it keeps in flight at once.

    That is the client's concurrency limit. A client without one sends every
    request at once, so the server's batch size caps the wave. A wave never holds
    more requests than the benchmark sends.

    Args:
        concurrency: The client's concurrency limit; 0 or less for none.
        num_requests: Requests the benchmark sends; 0 or less if unknown.
        max_batch_size: The server's maximum batch size.
    """
    wave = concurrency if concurrency > 0 else max_batch_size
    return min(wave, num_requests) if num_requests > 0 else wave


class WarmupPass(NamedTuple):
    """The warmup pass to send before one measured client."""

    # The build_warmup_pass_cmd command, without host and port.
    cmd: List[str]
    num_requests: int


def build_warmup_pass_cmd(measured_cmd: Sequence[str], num_requests: int) -> List[str]:
    """Derive the warmup pass's ``benchmark_serving`` command from the measured one.

    Every flag that shapes the traffic is kept: model, tokenizer, dataset and
    sequence lengths, maximum concurrency, backend, streaming, chat template. Only
    these change:

    * ``--num-prompts`` becomes ``num_requests``.
    * ``--seed`` becomes the measured seed plus one, so the pass draws different
      prompts with the same length distribution: other random token ids, or
      another shuffle of the same dataset file.
    * ``--no-test-input`` is added, so no single test request goes first.
    * ``--save-request-time-breakdown``, ``--save-result``, ``--save-detailed``,
      ``--result-dir`` and ``--result-filename`` are dropped with their values,
      so the pass writes none of the measured client's output files.

    The measured command must pass each flag and its value as separate items, as
    the runners build it. The caller adds host and port.

    Args:
        measured_cmd: The measured client's ``benchmark_serving`` command.
        num_requests: Requests the pass sends; at least 1.

    Returns:
        The warmup pass command.

    Raises:
        ValueError: ``num_requests`` is below 1.
    """
    if num_requests < 1:
        raise ValueError(f"A warmup pass needs at least one request, got {num_requests}")
    warmup_cmd: List[str] = []
    measured_seed = _BENCHMARK_SERVING_DEFAULT_SEED
    args = iter(measured_cmd)
    for arg in args:
        if arg in _FLAGS_WITH_REPLACED_VALUE:
            value = next(args, None)
            if arg == _SEED_FLAG and value is not None:
                measured_seed = int(value)
            continue
        if arg in _DROPPED_FLAGS:
            continue
        warmup_cmd.append(arg)
    return warmup_cmd + [
        _NUM_PROMPTS_FLAG,
        str(num_requests),
        _SEED_FLAG,
        str(measured_seed + 1),
        _NO_TEST_INPUT_FLAG,
    ]


def count_failed_requests(output: str) -> int:
    """Return the failed requests a ``benchmark_serving`` output reports, or 0."""
    match = _FAILED_REQUESTS_RE.search(output)
    return int(match.group(1)) if match else 0


def send_warmup_pass(
    warmup_cmd: List[str],
    num_requests: int,
    run_benchmark: Callable[[List[str]], str],
    log_path: str,
    timeout: Optional[float] = None,
    log_info: Callable[[str], None] = print,
    log_warning: Callable[[str], None] = print,
) -> int:
    """Send the warmup pass; return how many of its requests succeeded.

    If only some requests fail, the pass still warmed the shapes it reached, so it
    logs a warning and the measured client decides whether the test passes.

    Args:
        warmup_cmd: The ``build_warmup_pass_cmd`` command, with the server's host
            and port.
        num_requests: Requests the command sends.
        run_benchmark: Runs a command and returns its output. Raises
            ``subprocess.CalledProcessError`` if the command exits non-zero, and
            ``subprocess.TimeoutExpired`` if it applies a time limit that runs out.
        log_path: Where ``run_benchmark`` writes the output; named in messages.
        timeout: The time limit ``run_benchmark`` applies, in seconds; None if it
            applies none.
        log_info: Receives progress messages.
        log_warning: Receives the message about a partial failure.

    Returns:
        The number of warmup requests that succeeded.

    Raises:
        RuntimeError: The command exited non-zero, timed out, or had every
            request fail.
    """
    log_info(f"Warmup pass: sending {num_requests} requests. cmd is {warmup_cmd}")
    start = time.monotonic()
    try:
        output = run_benchmark(warmup_cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Warmup pass: benchmark_serving exited with code {e.returncode}. See {log_path}"
        ) from e
    except subprocess.TimeoutExpired as e:
        limit = f" within {timeout:.0f}s" if timeout is not None else ""
        raise RuntimeError(
            f"Warmup pass: {num_requests} requests did not finish{limit}. See {log_path}"
        ) from e
    elapsed = time.monotonic() - start
    failed_count = count_failed_requests(output)
    if failed_count >= num_requests:
        raise RuntimeError(f"Warmup pass: all {num_requests} requests failed. See {log_path}")
    if failed_count:
        log_warning(
            f"Warmup pass: {failed_count} of {num_requests} requests failed ({elapsed:.0f}s). "
            f"See {log_path}"
        )
    else:
        log_info(f"Warmup pass: {num_requests} requests done in {elapsed:.0f}s")
    return num_requests - failed_count
