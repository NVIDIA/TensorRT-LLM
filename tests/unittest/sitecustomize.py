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
"""Per-process stack forensics for multi-GPU hang investigation (NVBUG 6801108).

WHY THIS FILE EXISTS
--------------------
The multi-GPU MoE tests drive rank 1..3 through ``mpi4py.futures.MPIPoolExecutor``,
which starts them with ``MPI_Comm_spawn`` -- they are independent MPI processes, not
forks of pytest. Consequences that motivate this hook:

* Their stdout/stderr have no collection path. The S3 log plugin captures the pytest
  process (file-descriptor capture), and a spawned child does not inherit those
  descriptors, so nothing a worker prints is archived.
* ``tests/unittest/conftest.py`` is imported by the pytest process only, so neither
  its SIGALRM stack dumper nor ``pytest-timeout`` can reach a worker.

The net effect is that when the suite hangs, the only stack ever recovered is the
pytest process waiting in ``MPIPoolExecutor``. The state actually holding the hang
lives in the workers and has never been observed.

``tests/unittest`` is placed on ``PYTHONPATH`` unconditionally for the inner pytest
run (see ``tests/integration/defs/test_unittests.py``), and spawned workers inherit
that environment, so Python imports this module automatically in every one of those
processes. That makes it the only hook point that reaches the workers.

ACTIVATION
----------
Inert unless ``TLLM_FORENSICS_DIR`` is set, so local runs are unaffected. Set it only
for the stages under investigation.

OUTPUT (two deliberately distinct markers -- never merge them)
--------------------------------------------------------------
* ``FORENSICS_ARMED``, written once at import into ``armed-*.txt``: proves this hook
  loaded in this process. It says nothing about whether a hang occurred.
* ``FORENSICS_STACK``, rewritten periodically into ``stack-*.txt``: the stacks of all
  threads, with a monotonically increasing ``dump_seq``.

Keeping the markers separate is what allows "the probe was installed" to be counted
independently of "a hang was captured". A shared marker would make the arming check
trivially true and hide a dead probe. ``dump_seq`` advancing across dumps is the
evidence that the dumper thread is still alive rather than merely once-armed.

Every operation is wrapped so that a forensics failure can never fail a test.
"""

import os
import sys

_ENV_DIR = "TLLM_FORENSICS_DIR"
_ENV_INTERVAL = "TLLM_FORENSICS_INTERVAL"
_DEFAULT_INTERVAL = 60.0


def _process_role(orig_argv):
    """Classify this process, or return None to stay inert.

    Mirrors the detection used by jenkins/scripts/cbts/coverage_utils/sitecustomize.py,
    which is already relied on in CI to recognize mpi4py pool workers.
    """
    # sys.orig_argv holds the launching command line; sys.argv has not yet gained
    # "pytest" at the time sitecustomize runs.
    if any("mpi4py.futures" in a for a in orig_argv):
        return "worker"
    if any("pytest" in a for a in orig_argv[:4]):
        return "pytest"
    return None


def _mpi_rank_hint():
    """Best-effort MPI rank, for labeling only.

    Spawned children form their own MPI world, so this is not a unique key; the pid
    is. Recorded because it still helps read the dumps.
    """
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"):
        value = os.environ.get(var)
        if value:
            return f"{var}={value}"
    return "rank=unknown"


def _install():
    output_dir = os.environ.get(_ENV_DIR, "").strip()
    if not output_dir:
        return

    orig_argv = getattr(sys, "orig_argv", sys.argv) or [""]
    role = _process_role(orig_argv)
    if role is None:
        return

    import faulthandler
    import threading
    import time

    os.makedirs(output_dir, exist_ok=True)
    pid = os.getpid()
    stem = f"{role}-{pid}"

    # FORENSICS_ARMED: this hook loaded here. Deliberately not the hang marker.
    armed_path = os.path.join(output_dir, f"armed-{stem}.txt")
    with open(armed_path, "w", encoding="utf-8") as handle:
        handle.write(
            f"FORENSICS_ARMED role={role} pid={pid} ppid={os.getppid()} "
            f"{_mpi_rank_hint()} at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"argv={' '.join(orig_argv)}\n"
        )

    try:
        interval = float(os.environ.get(_ENV_INTERVAL, _DEFAULT_INTERVAL))
    except ValueError:
        interval = _DEFAULT_INTERVAL
    interval = max(interval, 5.0)

    stack_path = os.path.join(output_dir, f"stack-{stem}.txt")

    def _dump_loop():
        sequence = 0
        while True:
            time.sleep(interval)
            sequence += 1
            try:
                # Rewrite in place: the final dump is the state at the hang. A
                # growing file would be unbounded across a multi-hour stage.
                with open(stack_path, "w", encoding="utf-8") as handle:
                    handle.write(
                        f"FORENSICS_STACK role={role} pid={pid} "
                        f"{_mpi_rank_hint()} dump_seq={sequence} "
                        f"at={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    handle.flush()
                    faulthandler.dump_traceback(file=handle, all_threads=True)
            except Exception:  # noqa: BLE001 - forensics must never fail a test
                continue

    thread = threading.Thread(target=_dump_loop, name="tllm-forensics-dump", daemon=True)
    thread.start()


try:
    _install()
except Exception as exc:  # noqa: BLE001 - never break interpreter startup
    print(f"[forensics] install failed: {exc!r}", file=sys.stderr)
