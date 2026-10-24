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
"""GPU smoke tests for the fill-gate GPU keepalive: the production path on a device."""

import subprocess
import sys
import time

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import gpu_keepalive as ka

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _run_production_path(k, timeout_s=300.0):
    """tick() until initialised, launch, then drain; returns (mode, evidence the chunks ran).

    Initialisation and the one-time hot recalibration both write the output buffer
    themselves, so the helper first gets past them, then clears the buffer and requires
    at least one further chunk. The buffer is read back only after drain(), without any
    other synchronisation, so the check proves both that a runtime chunk computed and
    that drain() waited for it.
    """
    deadline = time.monotonic() + timeout_s
    try:
        while not k._initialized and not k._disabled and time.monotonic() < deadline:
            k.tick()  # polls the self-test child; launches nothing until it reports
            time.sleep(0.2)
        assert k._initialized and not k._disabled, "keepalive did not initialise"
        mode = k.mode
        while (
            getattr(k, "_recalibrate_when_hot", False)
            and not k._disabled
            and time.monotonic() < deadline
        ):
            k.tick()  # spin: the second launch triggers the recalibration kernels
            time.sleep(0.05)
        assert not k._disabled, "keepalive disabled during recalibration"
        k._stream.synchronize()
        out = k._sink if mode == "spin" else k._c  # keep a reference: drain() drops the keepalive's
        with torch.cuda.stream(k._stream):
            out.zero_()
        k._stream.synchronize()
        launches_before = k.launches
        while k.launches <= launches_before and not k._disabled and time.monotonic() < deadline:
            k.tick()  # a free queue slot and (mm) the period cadence gate permitting
            time.sleep(0.05)
        assert not k._disabled, "keepalive disabled while launching"
        assert k.launches > launches_before, "no runtime chunk was launched"
    finally:
        k.drain()
    assert not k._inflight and k._stream is None and k.mode is None  # released when the gate opens
    # No synchronize() here on purpose: only drain() may have waited for the chunks.
    computed = bool(torch.all(out > 0)) if mode == "spin" else out[0, 0].item() == ka._MM_DIM
    return mode, computed


def test_mm_fallback_on_device(monkeypatch):
    """The torch.mm path taken when the Triton self-test does not pass."""
    monkeypatch.setattr(ka, "_SELFTEST_VERDICT", {torch.cuda.current_device(): False})
    k = ka.GpuKeepalive(torch.cuda.current_device())
    assert k.mode is None  # nothing on the device until the first tick
    assert _run_production_path(k) == ("mm", True)
    assert k._a is None  # the 96 MiB of operands are gone


def test_spin_kernel_on_device(monkeypatch):
    """The real spin kernel through the production path: async self-test, then launches."""
    if not ka._HAVE_TRITON:
        pytest.skip("triton not installed")
    device = torch.cuda.current_device()
    monkeypatch.setattr(ka, "_SELFTEST_VERDICT", {})
    k = ka.GpuKeepalive(device)
    mode, computed = _run_production_path(k)
    if mode != "spin":  # diagnose before failing: a compute mode may forbid the child's context
        child = subprocess.run(
            [sys.executable, ka.__file__, "--selftest", str(device)],
            capture_output=True,
            timeout=300,
        )
        if child.returncode == ka._SELFTEST_RC_NO_CONTEXT:
            pytest.skip("compute mode forbids a second CUDA context on this device")
        pytest.fail(
            f"self-test rc={child.returncode}: {child.stderr.decode(errors='replace')[-2000:]}"
        )
    assert computed
