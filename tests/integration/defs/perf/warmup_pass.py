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
"""The un-measured warmup pass perf-sanity sends before each measured client.

The server's own startup warmup cannot reach every prefill shape a benchmark
produces: how many prompts arrive together, how they are chunked against
``max_num_tokens`` and which decode tokens share a step with them are only known
once real traffic arrives. JIT kernels keyed on those shapes would otherwise
compile inside the first measured requests. The warmup pass sends that traffic
once beforehand: the measured client's own ``benchmark_serving`` command, cut to
one wave of ``concurrency`` requests with different prompts. Nothing it prints is
parsed or uploaded.

It runs for plain aggregated cases, ``ctx_only`` and disagg ``e2e``, whose
measured window contains prefill. The ``gen_only`` modes never get one: they
measure decode only, and their generation server's benchmark fill gate opens
once, so warmup requests would open it before the measured burst arrives.

``TRTLLM_PERF_SANITY_WARMUP_PASS=0`` turns it off, e.g. to reproduce a JIT stall
the pass would otherwise absorb.

Stdlib-only, so unit tests can load it without torch.
"""

import os
from typing import List, Mapping, Optional, Sequence

WARMUP_PASS_ENV = "TRTLLM_PERF_SANITY_WARMUP_PASS"

# Benchmark modes that get a warmup pass. A plain aggregated case has no
# benchmark mode, which the runner carries as "".
WARMUP_PASS_MODES = ("", "ctx_only", "e2e")

_DISABLED_VALUES = ("0", "false", "no", "off")

_NUM_PROMPTS_FLAG = "--num-prompts"
_SEED_FLAG = "--seed"
_NO_TEST_INPUT_FLAG = "--no-test-input"
_SAVE_TIME_BREAKDOWN_FLAG = "--save-request-time-breakdown"
# Flags whose value the warmup pass replaces or drops.
_FLAGS_WITH_REPLACED_VALUE = (_NUM_PROMPTS_FLAG, _SEED_FLAG, _SAVE_TIME_BREAKDOWN_FLAG)
# benchmark_serving's --seed default.
_BENCHMARK_SERVING_DEFAULT_SEED = 0


def warmup_pass_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the warmup pass is on.

    It is on unless ``TRTLLM_PERF_SANITY_WARMUP_PASS`` is ``0``, ``false``, ``no``
    or ``off`` (any case).

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
    * ``--save-request-time-breakdown`` and its value are dropped.

    The measured command must pass each flag and its value as separate items, as
    the runner builds it. The caller adds host and port.

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
        if arg == _NO_TEST_INPUT_FLAG:
            continue
        warmup_cmd.append(arg)
    return warmup_cmd + [
        _NUM_PROMPTS_FLAG,
        str(num_requests),
        _SEED_FLAG,
        str(measured_seed + 1),
        _NO_TEST_INPUT_FLAG,
    ]
