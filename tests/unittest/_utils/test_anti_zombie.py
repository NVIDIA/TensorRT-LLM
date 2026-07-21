# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Anti-zombie helpers: PR_SET_PDEATHSIG and kill_process_tree (no GPU)."""

import signal
import subprocess
import sys
import time

import psutil
import pytest

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux-only"),
]


def _alive(pid: int) -> bool:
    return psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return not _alive(pid)


# Parent spawns a child that arms PR_SET_PDEATHSIG, prints its pid, then sleeps.
_PARENT_SRC = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", '''
import signal, time
from tensorrt_llm._utils import set_parent_death_signal
set_parent_death_signal(signal.SIGKILL)
print("CHILD_READY", flush=True)
time.sleep(300)
'''], stdout=subprocess.PIPE, text=True)
# Relay the child's readiness + pid to our stdout.
child.stdout.readline()  # wait for CHILD_READY
print(f"CHILD_PID={child.pid}", flush=True)
time.sleep(300)
"""


def test_prctl_kills_child_when_parent_dies() -> None:
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT_SRC], stdout=subprocess.PIPE, text=True
    )
    child_pid = None
    try:
        line = parent.stdout.readline().strip()
        assert line.startswith("CHILD_PID="), f"unexpected: {line!r}"
        child_pid = int(line.split("=", 1)[1])
        assert _alive(child_pid)

        # Kill the parent; the kernel should SIGKILL the child via PDEATHSIG.
        parent.kill()
        parent.wait(timeout=10)

        assert _wait_gone(child_pid, timeout=10.0), f"child {child_pid} survived its parent's death"
    finally:
        if parent.poll() is None:
            parent.kill()
        # Best-effort: if an assertion failed after the parent was reaped, the
        # child may still be sleeping — don't leave it behind for 5 minutes.
        if child_pid is not None:
            try:
                psutil.Process(child_pid).kill()
            except psutil.Error:
                pass


# Builds a 3-level tree (top -> child -> grandchild), all sleeping, and prints
# each pid so the test can verify kill_process_tree reaps the whole tree.
_TREE_SRC = """
import subprocess, sys, time
gc_src = "import time; print('G', flush=True); time.sleep(300)"
ch_src = (
    "import subprocess, sys, time; "
    "g = subprocess.Popen([sys.executable, '-c', %r]); "
    "print('CHILD_PID=' + str(__import__('os').getpid()), flush=True); "
    "print('GRANDCHILD_PID=' + str(g.pid), flush=True); "
    "time.sleep(300)"
) % gc_src
child = subprocess.Popen([sys.executable, "-c", ch_src], stdout=subprocess.PIPE, text=True)
import os
print("TOP_PID=" + str(os.getpid()), flush=True)
for _ in range(2):
    print(child.stdout.readline().strip(), flush=True)
time.sleep(300)
"""


def test_kill_process_tree_reaps_grandchildren() -> None:
    from tensorrt_llm._utils import kill_process_tree

    top = subprocess.Popen([sys.executable, "-c", _TREE_SRC], stdout=subprocess.PIPE, text=True)
    pids = {}
    try:
        for _ in range(3):
            line = top.stdout.readline().strip()
            key, _, val = line.partition("=")
            pids[key] = int(val)
        assert {"TOP_PID", "CHILD_PID", "GRANDCHILD_PID"} <= set(pids)
        for pid in pids.values():
            assert _alive(pid), f"{pid} not alive at setup"

        kill_process_tree(pids["TOP_PID"], include_parent=True, wait_timeout=10.0)

        for name, pid in pids.items():
            assert _wait_gone(pid, timeout=10.0), f"{name} ({pid}) not reaped"
    finally:
        if top.poll() is None:
            top.kill()
        # Best-effort cleanup if the assertion failed mid-way.
        for pid in pids.values():
            try:
                psutil.Process(pid).kill()
            except psutil.Error:
                pass


def test_set_parent_death_signal_idempotent() -> None:
    """Calling it must not raise. Run in a subprocess so we don't arm
    PR_SET_PDEATHSIG on the pytest worker itself."""
    src = (
        "import signal\n"
        "from tensorrt_llm._utils import set_parent_death_signal\n"
        "set_parent_death_signal(signal.SIGTERM)\n"
        "set_parent_death_signal(signal.SIGTERM)\n"
        "print('OK')\n"
    )
    # Generous timeout: the subprocess pays a cold `import tensorrt_llm`, which
    # alone can take ~a minute on slower hosts, before the prctl calls run.
    proc = subprocess.run([sys.executable, "-c", src], timeout=300, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_prearming_parent_death_detected() -> None:
    """Regression for the arming race: PR_SET_PDEATHSIG only takes effect after
    the prctl syscall, so a parent that died first would never trigger it. When
    the spawner supplies expected_parent_pid, a reparented process must detect
    the mismatch right after arming and deliver the signal to itself."""
    src = (
        "import os, signal\n"
        "from tensorrt_llm._utils import set_parent_death_signal\n"
        "# Simulate 'parent already died before arming': expect a parent PID\n"
        "# that is guaranteed not to be our actual current parent.\n"
        "set_parent_death_signal(signal.SIGKILL, expected_parent_pid=os.getppid() + 1)\n"
        "print('UNREACHABLE')\n"
    )
    proc = subprocess.run([sys.executable, "-c", src], timeout=300, capture_output=True, text=True)
    assert proc.returncode == -signal.SIGKILL, (proc.returncode, proc.stderr)
    assert "UNREACHABLE" not in proc.stdout

    # And the happy path: the expected parent matches, no self-kill.
    src_ok = (
        "import os, signal\n"
        "from tensorrt_llm._utils import set_parent_death_signal\n"
        "set_parent_death_signal(signal.SIGKILL, expected_parent_pid=os.getppid())\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", src_ok], timeout=300, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
