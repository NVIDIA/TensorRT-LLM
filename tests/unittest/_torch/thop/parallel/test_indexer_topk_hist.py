# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Driver for the env-gated ported coarse-histogram indexer top-k
(``TRTLLM_DSA_TOPK_HIST``, kernel in cpp/tensorrt_llm/kernels/indexerTopKHist.cu).

The C++ gate in ``invokeIndexerTopKDecode`` reads the env once per process
(``std::call_once``), so the ported path cannot be toggled within a single
pytest process. This driver runs the actual correctness checks
(``indexer_topk_hist_tests.py``) in a fresh subprocess with the env set, then
asserts:

  1. the subprocess passed (every supported shape matched ``torch.topk``), and
  2. the one-time arming log fired -- proving the ported kernel actually engaged
     rather than silently falling back to the stock path (which would also match
     torch.topk and yield a misleading green).
"""

import os
import subprocess
import sys

from utils.util import skip_pre_hopper

# Emitted once by the C++ gate (indexerTopK.cu) when TRTLLM_DSA_TOPK_HIST=1 and
# the first fp32 decode top-k is launched. Keep in sync with that log line.
_ARMING_MARKER = "routing supported fp32 DSA indexer decode"


@skip_pre_hopper
def test_indexer_topk_hist_port_matches_torch():
    """Run the ported-kernel correctness suite in a subprocess with the gate on;
    require both a clean pass and evidence that the kernel armed."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(current_dir, "indexer_topk_hist_tests.py")

    process_env = os.environ.copy()
    process_env["TRTLLM_DSA_TOPK_HIST"] = "1"
    process_env["TLLM_LOG_LEVEL"] = "INFO"  # ensure the arming INFO log is emitted

    # -s (no capture) lets the C++ stderr arming log flow through to our captured
    # subprocess stderr; pytest would otherwise swallow it on success.
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", worker, "-v", "-s"],
            capture_output=True,
            text=True,
            env=process_env,
            # Hard backstop: the cluster-path deadlock-stress test would hang the whole
            # subprocess on a Fix-A/Fix-B regression, and a hung CUDA sync cannot be
            # interrupted by pytest-timeout's in-process signal. Kill + fail instead.
            timeout=1200,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        err = exc.stderr or b""

        def decode(b):
            return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")

        raise AssertionError(
            "ported histogram top-k worker DEADLOCKED (subprocess timed out after "
            "1200 s) -- likely a regression of the decode-hang fixes in "
            "indexerTopKHist.cu (Fix A: PDL-wait ordering; Fix B: cluster.sync at the "
            "overflow guard). Tail of output:\n" + decode(out)[-4000:] + "\n" + decode(err)[-4000:]
        )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    assert result.returncode == 0, (
        "ported histogram top-k correctness subprocess failed "
        f"(exit {result.returncode}); see captured output above"
    )

    combined = result.stdout + result.stderr
    assert _ARMING_MARKER in combined, (
        "arming log not found: the ported kernel never engaged (env gate did not "
        "fire), so the correctness pass above would be a false green against the "
        "stock path"
    )


@skip_pre_hopper
def test_indexer_topk_hist_cluster_sanitizer():
    """Deterministic detector for the two decode-hang bugs' shared signature.

    Runs the minimal cluster-path shape under compute-sanitizer:
      * ``synccheck`` -- flags divergent ``__syncthreads()``/``cluster.sync()``
        participation, the exact failure mode of both deadlock fixes.
      * ``racecheck`` -- flags the unsynchronized histogram/tie union hazard (a peer's
        remote tie-merge vs. rank 0's overflow-guard read) that Fix B's cluster.sync()
        serialises; unlike the behavioural stress test this is timing-independent.

    Skipped when compute-sanitizer is unavailable (e.g. CPU-only or minimal CI images).
    """
    import shutil

    sanitizer = shutil.which("compute-sanitizer")
    if sanitizer is None:
        import pytest

        pytest.skip("compute-sanitizer not found; skipping cluster sync/race check")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(current_dir, "indexer_topk_hist_tests.py")

    process_env = os.environ.copy()
    process_env["TRTLLM_DSA_TOPK_HIST"] = "1"
    process_env["TLLM_LOG_LEVEL"] = "INFO"

    for tool in ("synccheck", "racecheck"):
        result = subprocess.run(
            [
                sanitizer,
                "--tool",
                tool,
                "--error-exitcode",
                "1",
                sys.executable,
                "-m",
                "pytest",
                worker,
                "-v",
                "-s",
                "-k",
                "cluster_min_for_sanitizer",
            ],
            capture_output=True,
            text=True,
            env=process_env,
            timeout=1800,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        combined = result.stdout + result.stderr
        assert result.returncode == 0, (
            f"compute-sanitizer --tool {tool} reported errors on the cluster path "
            "(barrier divergence / shared-memory hazard) -- a regression of the "
            "decode-hang fixes in indexerTopKHist.cu. Tail:\n" + combined[-4000:]
        )
        # A clean exit alone does not prove the histogram kernel ran: if dispatch
        # fell back to the stock path, the sanitizer would pass without ever
        # checking the intended kernel. Require the arming log to prove it engaged.
        assert _ARMING_MARKER in combined, (
            f"arming log not found for compute-sanitizer --tool {tool}; the worker "
            "may have validated the stock fallback path instead of the ported kernel"
        )
