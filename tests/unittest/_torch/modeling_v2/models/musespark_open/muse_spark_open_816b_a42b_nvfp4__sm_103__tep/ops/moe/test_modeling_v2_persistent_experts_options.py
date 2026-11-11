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
"""``MUSESPARK_OPEN_PERSISTENT_EXPERTS_OPTIONS`` of ``persistent_experts``, on the host only.

``persistent_experts.py`` is loaded from its file with ``torch``, the op catalog and the
logger stubbed for the load; no kernel is built or launched. The tests pin:

* unset, empty and blank values leave ``OPTIONS`` exactly as they are;
* the overrides (``key=value,...`` over ``TACTIC_RANGES``) replace only the named fields,
  are read and logged once per device, and are cached with the device's options;
* an unknown or repeated key, a missing or non-integer value and a value out of range raise.
"""

from __future__ import annotations

import enum
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_MODULE = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_103__tep/ops/moe/persistent_experts.py"
)
ENV = "MUSESPARK_OPEN_PERSISTENT_EXPERTS_OPTIONS"


class _Arch(enum.Enum):
    SM_103 = 103


class _OpWrapper:
    pass


def _load():
    catalog = "tensorrt_llm._torch._experimental.modeling_v2.catalog"
    stubs = {
        "torch": mock.MagicMock(),
        f"{catalog}._op": types.SimpleNamespace(Arch=_Arch, OpWrapper=_OpWrapper),
        f"{catalog}.moe.fp4_block_scale_moe_runner": types.SimpleNamespace(_E2M1_VALUES=[0.0]),
        "tensorrt_llm.logger": types.SimpleNamespace(logger=mock.MagicMock()),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_persistent_experts_under_test", _MODULE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


PE = _load()
DEVICE = types.SimpleNamespace(index=0)


@pytest.fixture(autouse=True)
def _fresh_device(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(PE, "_DEVICE_OPTIONS", {})
    PE.torch.cuda.get_device_capability.return_value = (10, 3)
    PE.logger.reset_mock()


@pytest.mark.parametrize("value", [None, "", " ", ",", " , ,"])
def test_no_overrides_leave_the_options_unchanged(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert PE._env_options() == {}
    assert PE._device_options(DEVICE) == PE.OPTIONS
    assert PE._DEVICE_OPTIONS == {0: PE.OPTIONS_BY_SM[(10, 3)]}
    PE.logger.info.assert_not_called()


def test_order_zero_changes_only_the_order(monkeypatch):
    monkeypatch.setenv(ENV, "order=0")
    assert PE.OPTIONS["order"] == 1
    assert PE._device_options(DEVICE) == {**PE.OPTIONS, "order": 0}
    assert PE.OPTIONS["order"] == 1


@pytest.mark.parametrize(
    "value, expected",
    [
        ("num_stages=8, balance=0", {"num_stages": 8, "balance": 0}),
        ("order=0", {"order": 0}),
        ("acc_stages=1,publish=1,handoff=0", {"acc_stages": 1, "publish": 1, "handoff": 0}),
    ],
)
def test_the_legal_overrides_parse_to_their_fields(monkeypatch, value, expected):
    monkeypatch.setenv(ENV, value)
    assert PE._env_options() == expected


def test_the_overrides_are_read_logged_and_cached_once_per_device(monkeypatch):
    monkeypatch.setenv(ENV, " order = 0 , balance=0,num_stages=4 ")
    first = PE._device_options(DEVICE)
    assert first == {**PE.OPTIONS, "order": 0, "balance": 0, "num_stages": 4}
    PE.logger.info.assert_called_once()
    assert ENV in PE.logger.info.call_args.args[0]
    monkeypatch.setenv(ENV, "order=1")
    assert PE._device_options(DEVICE) == first
    PE.logger.info.assert_called_once()
    other = types.SimpleNamespace(index=1)
    assert PE._device_options(other)["order"] == 1


@pytest.mark.parametrize(
    "value",
    [
        "no_such_key=1",
        "order=0,order=1",
        "order",
        "order=",
        "order=x",
        "order=2",
        "order=-1",
        "num_stages=0",
        "num_stages=9",
        "acc_stages=5",
        "balance=2",
        "publish=2",
        "=1",
        "foo=1",
        "fastroute=1",
        "trace=1",
        "delay=1",
    ],
)
def test_an_illegal_override_raises(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with pytest.raises(ValueError, match=ENV):
        PE._env_options()
    with pytest.raises(ValueError, match=ENV):
        PE._device_options(DEVICE)
    assert PE._DEVICE_OPTIONS == {}


def test_a_device_of_another_capability_still_raises(monkeypatch):
    monkeypatch.setenv(ENV, "order=0")
    PE.torch.cuda.get_device_capability.return_value = (9, 0)
    with pytest.raises(ValueError, match="no kernel options for SM"):
        PE._device_options(DEVICE)
