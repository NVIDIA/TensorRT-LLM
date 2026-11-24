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
"""``TRTLLM_GRAPH_LAUNCH_RESET`` (``pyexecutor/graph_launch_reset.py``), on the host only.

The module is loaded from its file with ``torch`` stubbed, and the CUDA driver bindings are a
recording fake, so no device is touched. The tests pin:

* any value but ``1`` (unset, empty, ``0``, ``true``) makes no driver call and logs nothing;
* ``1`` creates one transient context, destroys it, restores the saved current context, and
  skips a second reset in the same process;
* a failed driver call, a current context that is not the primary one and a context, device
  or stream that differs afterwards raise, the saved context restored where it was changed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_MODULE = _ROOT / "tensorrt_llm/_torch/pyexecutor/graph_launch_reset.py"
ENV = "TRTLLM_GRAPH_LAUNCH_RESET"
SAVED, TEMP = 11, 22


def _load():
    saved = sys.modules.get("torch")
    sys.modules["torch"] = mock.MagicMock()
    try:
        spec = importlib.util.spec_from_file_location("_graph_launch_reset_under_test", _MODULE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved
    return module


GLR = _load()


class _Driver:
    """The driver calls the reset makes, recorded; ``fail`` names a call that returns 1."""

    def __init__(self, primary=SAVED, current_after=SAVED, fail=None):
        self.calls = []
        self.primary = primary
        self.current = SAVED
        self.current_after = current_after
        self.fail = fail

    def _call(self, name, *payload):
        self.calls.append(name)
        return (1,) if self.fail == name else (0, *payload)

    def cuCtxGetCurrent(self):
        if "cuCtxSetCurrent" in self.calls:
            self.current = self.current_after
        return self._call("cuCtxGetCurrent", self.current)

    def cuDeviceGet(self, device):
        return self._call("cuDeviceGet", 5)

    def cuDevicePrimaryCtxRetain(self, cudev):
        return self._call("cuDevicePrimaryCtxRetain", self.primary)

    def cuDevicePrimaryCtxRelease(self, cudev):
        return self._call("cuDevicePrimaryCtxRelease")

    def cuCtxDestroy(self, ctx):
        assert ctx == TEMP
        return self._call("cuCtxDestroy")

    def cuCtxSetCurrent(self, ctx):
        assert ctx == SAVED
        return self._call("cuCtxSetCurrent")


def _driver(monkeypatch, create="v4", **kwargs):
    """A fake driver whose context creation has the signature of cuda-python 13 (``v4``: a
    ``cuCtxCreate_v4``), of 13 without it (``params``: ``cuCtxCreate(params, flags, dev)``) or
    of 12.x (``flags``: ``cuCtxCreate(flags, dev)``)."""
    driver = _Driver(**kwargs)

    def create_v4(params, flags, cudev):
        return driver._call("cuCtxCreate_v4", TEMP)

    def create_params(params, flags, cudev):
        return driver._call("cuCtxCreate", TEMP)

    def create_flags(flags, cudev):
        return driver._call("cuCtxCreate", TEMP)

    if create == "v4":
        driver.cuCtxCreate_v4 = create_v4
    else:
        driver.cuCtxCreate = create_params if create == "params" else create_flags
    package = types.ModuleType("cuda")
    package.__path__ = []
    bindings = types.ModuleType("cuda.bindings")
    bindings.driver = driver
    package.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", package)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    return driver


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(GLR, "_done_pid", None)
    GLR.torch.reset_mock()
    GLR.torch.cuda.current_device.return_value = 0
    GLR.torch.cuda.current_stream.return_value = "stream"


@pytest.mark.parametrize("value", [None, "", "0", "true", "2", " 1"])
def test_any_value_but_one_does_nothing(monkeypatch, value):
    driver = _driver(monkeypatch)
    log = mock.MagicMock()
    if value is not None:
        monkeypatch.setenv(ENV, value)
    GLR.maybe_reset_graph_launch_mode(log)
    assert driver.calls == []
    GLR.torch.cuda.synchronize.assert_not_called()
    assert log.method_calls == []
    assert GLR._done_pid is None


@pytest.mark.parametrize("create", ["v4", "params", "flags"])
def test_one_creates_and_destroys_a_context_and_restores_the_saved_one(monkeypatch, create):
    driver = _driver(monkeypatch, create=create)
    log = mock.MagicMock()
    monkeypatch.setenv(ENV, "1")
    GLR.maybe_reset_graph_launch_mode(log)
    created = "cuCtxCreate_v4" if create == "v4" else "cuCtxCreate"
    assert driver.calls == [
        "cuCtxGetCurrent",
        "cuDeviceGet",
        "cuDevicePrimaryCtxRetain",
        "cuDevicePrimaryCtxRelease",
        created,
        "cuCtxDestroy",
        "cuCtxSetCurrent",
        "cuCtxGetCurrent",
    ]
    assert GLR.torch.cuda.synchronize.call_count == 2
    assert GLR._done_pid == os.getpid()
    log.info.assert_called_once()
    log.warning.assert_not_called()


def test_a_second_reset_in_one_process_is_skipped(monkeypatch):
    driver = _driver(monkeypatch)
    log = mock.MagicMock()
    monkeypatch.setenv(ENV, "1")
    GLR.maybe_reset_graph_launch_mode(log)
    calls = list(driver.calls)
    GLR.maybe_reset_graph_launch_mode(log)
    assert driver.calls == calls
    log.info.assert_called_once()
    log.warning.assert_called_once()


@pytest.mark.parametrize(
    "call",
    [
        "cuCtxGetCurrent",
        "cuDeviceGet",
        "cuDevicePrimaryCtxRetain",
        "cuDevicePrimaryCtxRelease",
        "cuCtxCreate_v4",
        "cuCtxDestroy",
        "cuCtxSetCurrent",
    ],
)
def test_a_failed_driver_call_raises(monkeypatch, call):
    driver = _driver(monkeypatch, fail=call)
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(RuntimeError, match=f"graph launch reset: {call} failed"):
        GLR.maybe_reset_graph_launch_mode(mock.MagicMock())
    assert GLR._done_pid is None
    if call in ("cuCtxDestroy",):
        assert "cuCtxSetCurrent" in driver.calls


def test_a_current_context_that_is_not_the_primary_one_raises(monkeypatch):
    driver = _driver(monkeypatch, primary=SAVED + 1)
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(RuntimeError, match="not the device's primary context"):
        GLR.maybe_reset_graph_launch_mode(mock.MagicMock())
    assert "cuCtxCreate_v4" not in driver.calls


def test_no_current_context_raises(monkeypatch):
    driver = _driver(monkeypatch)
    driver.current = 0
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(RuntimeError, match="no current context"):
        GLR.maybe_reset_graph_launch_mode(mock.MagicMock())


def test_a_context_device_or_stream_that_differs_afterwards_raises(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    _driver(monkeypatch, current_after=SAVED + 1)
    with pytest.raises(RuntimeError, match="changed across the reset"):
        GLR.maybe_reset_graph_launch_mode(mock.MagicMock())
    _driver(monkeypatch)
    GLR.torch.cuda.current_stream.side_effect = ["stream", "other"]
    with pytest.raises(RuntimeError, match="changed across the reset"):
        GLR.maybe_reset_graph_launch_mode(mock.MagicMock())
    assert GLR._done_pid is None
