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
"""The benchmark-disagg fill gate's retry loop must have a deadline.

The gate retries until the fill completes. Its `continue` sits before
`iter_counter += 1`, so while it spins the iteration counter is frozen and
wall-clock advances -- which is why an archived wedge shows tens of thousands
of byte-identical iteration lines at ~110 ms apart (this gate's own
`time.sleep(0.1)`, observed from outside).

The bound is on *no progress*, not on elapsed time: a slow but advancing fill
resets the clock and is never killed.
"""

import types

import pytest

from tensorrt_llm._torch.pyexecutor.py_executor import (
    BENCHMARK_DISAGG_FILL_STALL_ENV_VAR_NAME,
    PyExecutor,
    _fill_stall_timeout_sec,
)

# Pure monkeypatch: no engine, no GPU. The marker is required, not
# decorative -- tests/unittest/conftest.py's pytest_ignore_collect drops any
# file whose source lacks this literal when pytest runs with -m cpu_only.
pytestmark = pytest.mark.cpu_only


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _executor(timeout_s, clock, rank=0):
    """A PyExecutor with only the fill-gate stall surface populated."""
    ex = object.__new__(PyExecutor)
    ex._benchmark_fill_stall_since = None
    ex._benchmark_fill_stall_timeout_sec = timeout_s
    ex.dist = types.SimpleNamespace(rank=rank)
    return ex


def _spin(ex, clock, monkeypatch, seconds, step=0.1, made_progress=False):
    """Drive the no-progress retry path for `seconds` of wall clock."""
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.time.monotonic", clock)
    for _ in range(int(seconds / step)):
        ex._fail_if_fill_gate_stalled(made_progress)
        clock.advance(step)


def test_raises_once_the_stall_window_elapses(monkeypatch):
    clock = _Clock()
    ex = _executor(5.0, clock, rank=3)
    with pytest.raises(RuntimeError, match="made no progress"):
        _spin(ex, clock, monkeypatch, seconds=8.0)


def test_message_names_the_rank_and_the_knob(monkeypatch):
    clock = _Clock()
    ex = _executor(5.0, clock, rank=7)
    with pytest.raises(RuntimeError) as exc:
        _spin(ex, clock, monkeypatch, seconds=8.0)
    msg = str(exc.value)
    assert "rank 7" in msg
    assert "TRTLLM_BENCHMARK_DISAGG_FILL_STALL_SEC" in msg


def test_a_slow_but_advancing_fill_is_never_killed(monkeypatch):
    """The bound is on no-progress, not on elapsed time.

    A fill that keeps making progress may legitimately take far longer than
    the window; killing it would be a regression, not a fix.
    """
    clock = _Clock()
    ex = _executor(5.0, clock)
    _spin(ex, clock, monkeypatch, seconds=60.0, made_progress=True)
    assert ex._benchmark_fill_stall_since is None


def test_progress_resets_the_clock(monkeypatch):
    """Stall, recover, stall again -- the second window starts from zero."""
    clock = _Clock()
    ex = _executor(5.0, clock)
    _spin(ex, clock, monkeypatch, seconds=4.0)  # just under
    ex._fail_if_fill_gate_stalled(True)  # progress
    assert ex._benchmark_fill_stall_since is None
    _spin(ex, clock, monkeypatch, seconds=4.0)  # under again
    assert ex._benchmark_fill_stall_since is not None  # armed, not fired


def test_zero_disables_the_bound(monkeypatch):
    clock = _Clock()
    ex = _executor(0.0, clock)
    _spin(ex, clock, monkeypatch, seconds=3600.0, step=10.0)
    assert ex._benchmark_fill_stall_since is None


def test_first_stalled_call_only_arms_the_clock(monkeypatch):
    """One no-progress pass is normal; it must not raise on its own."""
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.time.monotonic", _Clock())
    ex = _executor(5.0, _Clock())
    ex._fail_if_fill_gate_stalled(False)
    assert ex._benchmark_fill_stall_since is not None


@pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity"])
def test_non_finite_env_falls_back_to_the_default(monkeypatch, raw):
    """float() accepts nan/inf, and each breaks the bound a different way.

    Every nan comparison is False, so ``nan <= 0`` does not disable the
    bound and ``stalled_for < nan`` does not defer it: control falls through
    to the raise, firing on the second consecutive stalled call rather than
    after the window. ``inf`` is the mirror -- ``stalled_for < inf`` is
    always True, so the bound never fires and is silently equivalent to 0.
    """
    monkeypatch.setenv(BENCHMARK_DISAGG_FILL_STALL_ENV_VAR_NAME, raw)
    assert _fill_stall_timeout_sec() == 600.0


def test_a_valid_env_override_is_honoured(monkeypatch):
    monkeypatch.setenv(BENCHMARK_DISAGG_FILL_STALL_ENV_VAR_NAME, "42.5")
    assert _fill_stall_timeout_sec() == 42.5


def test_zero_env_is_preserved_as_the_disable_switch(monkeypatch):
    """0 must survive the finiteness check -- it is the documented opt-out."""
    monkeypatch.setenv(BENCHMARK_DISAGG_FILL_STALL_ENV_VAR_NAME, "0")
    assert _fill_stall_timeout_sec() == 0.0
