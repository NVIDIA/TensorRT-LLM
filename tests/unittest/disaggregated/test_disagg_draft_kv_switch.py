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
"""The disagg drafter context-window switch.

The strict environment contract, and the gate: the model's opt-in hook AND a separate draft pool
AND the kill switch unset.
"""

import inspect
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.disaggregation import draft_kv
from tensorrt_llm._torch.disaggregation.draft_kv import (
    DISABLE_DISAGG_DRAFT_KV_ENV,
    disagg_draft_kv_disabled,
    parse_strict_bool_env,
    select_disagg_draft_kv_cache_manager,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

pytestmark = pytest.mark.cpu_only

_TRUE = ["1", "true", "TRUE", "True", "on", "ON", "yes", "Yes"]
_FALSE = ["", "0", "false", "FALSE", "off", "Off", "no", "NO"]
_INVALID = ["2", "-1", "enable", "y", "n", "disable", "truee", "1.0"]


def test_name_is_the_documented_variable():
    assert DISABLE_DISAGG_DRAFT_KV_ENV == "TRTLLM_DISABLE_DISAGG_DRAFT_KV"


def test_unset_is_not_disabled():
    assert disagg_draft_kv_disabled({}) is False


@pytest.mark.parametrize("value", _TRUE)
def test_true_values_disable(value):
    assert disagg_draft_kv_disabled({DISABLE_DISAGG_DRAFT_KV_ENV: value}) is True


@pytest.mark.parametrize("value", _FALSE)
def test_false_values_do_not_disable(value):
    assert disagg_draft_kv_disabled({DISABLE_DISAGG_DRAFT_KV_ENV: value}) is False


@pytest.mark.parametrize("value", _INVALID)
def test_anything_else_raises_naming_the_variable(value):
    with pytest.raises(ValueError, match=DISABLE_DISAGG_DRAFT_KV_ENV):
        disagg_draft_kv_disabled({DISABLE_DISAGG_DRAFT_KV_ENV: value})


def test_process_environment_is_read(monkeypatch):
    monkeypatch.setenv(DISABLE_DISAGG_DRAFT_KV_ENV, "yes")
    assert disagg_draft_kv_disabled() is True
    monkeypatch.setenv(DISABLE_DISAGG_DRAFT_KV_ENV, "off")
    assert disagg_draft_kv_disabled() is False
    monkeypatch.setenv(DISABLE_DISAGG_DRAFT_KV_ENV, "maybe")
    with pytest.raises(ValueError, match=DISABLE_DISAGG_DRAFT_KV_ENV):
        disagg_draft_kv_disabled()


def test_parse_is_shared_code_not_a_model_module():
    # The shared parse must not depend on any model's switch module: every other model never
    # loads one.
    assert "modeling_v2" not in inspect.getsource(draft_kv)
    assert parse_strict_bool_env("X", {"X": " On "}) is True


class _Worker:
    def engage_disagg_draft_kv(self):
        raise AssertionError("selection must not engage the worker")


def _model(hook: bool, worker=None):
    model = SimpleNamespace(spec_worker=_Worker() if worker is None else worker)
    if hook:
        model.runtime_hooks = RuntimeHooks(disagg_draft_kv=True)
    return model


def _draft_manager(joint_reuse=False):
    return SimpleNamespace(enable_joint_kv_cache_reuse=joint_reuse)


def test_hook_defaults_off():
    assert RuntimeHooks().disagg_draft_kv is False


def test_selected_only_with_hook_and_separate_pool(monkeypatch):
    monkeypatch.delenv(DISABLE_DISAGG_DRAFT_KV_ENV, raising=False)
    manager = _draft_manager()
    assert select_disagg_draft_kv_cache_manager(_model(hook=True), manager) is manager
    # Every other model: no hook (all other DFlash / DSpark deployments).
    assert select_disagg_draft_kv_cache_manager(_model(hook=False), manager) is None
    assert select_disagg_draft_kv_cache_manager(SimpleNamespace(), manager) is None
    # No separate draft pool (one-model Eagle3 sharing the target's).
    assert select_disagg_draft_kv_cache_manager(_model(hook=True), None) is None
    # A worker with no paged drafter context to ship.
    no_ctx = SimpleNamespace(runtime_hooks=RuntimeHooks(disagg_draft_kv=True), spec_worker=object())
    assert select_disagg_draft_kv_cache_manager(no_ctx, manager) is None


@pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
def test_kill_switch_wins(monkeypatch, value):
    monkeypatch.setenv(DISABLE_DISAGG_DRAFT_KV_ENV, value)
    assert select_disagg_draft_kv_cache_manager(_model(hook=True), _draft_manager()) is None


def test_invalid_kill_switch_value_fails_selection(monkeypatch):
    monkeypatch.setenv(DISABLE_DISAGG_DRAFT_KV_ENV, "please")
    with pytest.raises(ValueError, match=DISABLE_DISAGG_DRAFT_KV_ENV):
        select_disagg_draft_kv_cache_manager(_model(hook=True), _draft_manager())


def test_joint_reuse_is_refused(monkeypatch):
    monkeypatch.delenv(DISABLE_DISAGG_DRAFT_KV_ENV, raising=False)
    with pytest.raises(ValueError, match="enable_joint_kv_cache_reuse"):
        select_disagg_draft_kv_cache_manager(_model(hook=True), _draft_manager(joint_reuse=True))
