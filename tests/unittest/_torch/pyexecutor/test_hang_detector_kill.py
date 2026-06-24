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
"""HangDetector timer behavior and the hard-kill propagation mechanism (no GPU)."""

import os
import signal
import subprocess
import sys
import time

from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector


def test_detector_fires_after_timeout():
    fired = []
    hd = HangDetector(timeout=2, on_detected=lambda: fired.append(time.monotonic()))
    with hd:
        hd.checkpoint()
        time.sleep(1.0)
        assert hd.detected() is False
        assert fired == []
        time.sleep(2.0)
        assert hd.detected() is True
        assert len(fired) == 1


def test_checkpoint_resets_timer():
    """Repeated checkpoints before the timeout keep the detector quiet."""
    fired = []
    hd = HangDetector(timeout=2, on_detected=lambda: fired.append(1))
    with hd:
        for _ in range(6):
            hd.checkpoint()
            time.sleep(0.4)  # < timeout, so the timer keeps resetting
        assert fired == []
        assert hd.detected() is False


def test_pause_suppresses_detection():
    fired = []
    hd = HangDetector(timeout=1, on_detected=lambda: fired.append(1))
    with hd:
        hd.checkpoint()
        with hd.pause():
            time.sleep(2.0)  # would have fired if not paused
        assert fired == []
        assert hd.detected() is False


def test_propagate_hard_kill_self_sigkills_without_mpi():
    """With MPI disabled, propagate_hard_kill self-SIGKILLs the process.

    A SIGKILL'd process reports returncode -SIGKILL (== -9) to the parent.
    """
    script = (
        "from tensorrt_llm._torch.pyexecutor.hang_detector import propagate_hard_kill; "
        "propagate_hard_kill()"
    )
    env = {**os.environ, "TLLM_DISABLE_MPI": "1"}
    # Generous timeout: the subprocess pays a cold `import tensorrt_llm` (full
    # _torch init), which alone can take a minute on slower hosts, before it
    # ever reaches propagate_hard_kill().
    proc = subprocess.run([sys.executable, "-c", script], env=env, timeout=300, capture_output=True)
    assert proc.returncode == -signal.SIGKILL, (
        f"expected self-SIGKILL (-9), got {proc.returncode}; "
        f"stderr={proc.stderr.decode(errors='replace')[-500:]}"
    )
