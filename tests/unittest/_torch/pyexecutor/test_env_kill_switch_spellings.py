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
"""Runtime kill switches read through ``env_switch`` accept every documented spelling.

1 / true / on / yes set a switch and 0 / false / off / no leave it unset, in any case; any
other value raises an error naming the variable instead of silently keeping a feature on.
"""

import pytest

from tensorrt_llm._torch import flashinfer_autotune_cache
from tensorrt_llm._torch.pyexecutor import disagg_gen_prefetch, perf_metrics_manager
from tensorrt_llm._torch.pyexecutor.py_executor import _env_killed
from tensorrt_llm._torch.speculative.interface import onehot_index_fill_enabled

pytestmark = pytest.mark.cpu_only

_SET = ["1", "true", "TRUE", "on", "Yes", " yes "]
_UNSET = ["0", "false", "FALSE", "off", "No", ""]


_SWITCHES = {
    "TRTLLM_DISABLE_RESPONSE_GUARD": lambda: _env_killed("TRTLLM_DISABLE_RESPONSE_GUARD"),
    "TRTLLM_DISABLE_POSTLAUNCH_ADMISSION": lambda: _env_killed(
        "TRTLLM_DISABLE_POSTLAUNCH_ADMISSION"
    ),
    "TRTLLM_M31_DISABLE_ONEHOT_INDEX_FILL": lambda: not onehot_index_fill_enabled(True),
    "TRTLLM_M31_DISABLE_KVPREFETCH_N1": lambda: (
        disagg_gen_prefetch.get_disagg_gen_prefetch_requests() == 0
    ),
    "TRTLLM_DISABLE_GPU_TIMESTAMPS": lambda: not perf_metrics_manager.arm_gpu_timestamps(),
}


@pytest.fixture
def prefetch_on(monkeypatch):
    monkeypatch.delenv(disagg_gen_prefetch.DISAGG_GEN_PREFETCH_REQUESTS_ENV, raising=False)
    monkeypatch.setattr(disagg_gen_prefetch, "_model_default", 1)
    # arm_gpu_timestamps sets process state; give it back afterwards.
    for name in ("_GPU_TIMESTAMPS", "_STAMP_OP"):
        monkeypatch.setattr(perf_metrics_manager, name, getattr(perf_metrics_manager, name))


@pytest.mark.parametrize("name", sorted(_SWITCHES))
@pytest.mark.parametrize("raw", _SET)
def test_every_set_spelling_turns_the_feature_off(name, raw, monkeypatch, prefetch_on):
    monkeypatch.setenv(name, raw)
    assert _SWITCHES[name]() is True


@pytest.mark.parametrize(
    "name", sorted(set(_SWITCHES) - {"TRTLLM_DISABLE_GPU_TIMESTAMPS"})
)  # the stamps also need the native op, so their "unset" result depends on the build
@pytest.mark.parametrize("raw", _UNSET)
def test_every_unset_spelling_keeps_the_feature(name, raw, monkeypatch, prefetch_on):
    monkeypatch.setenv(name, raw)
    assert _SWITCHES[name]() is False


@pytest.mark.parametrize("name", sorted(_SWITCHES))
def test_an_unknown_value_names_the_variable(name, monkeypatch, prefetch_on):
    monkeypatch.setenv(name, "disable")
    with pytest.raises(ValueError, match=name):
        _SWITCHES[name]()


@pytest.mark.parametrize("raw, frozen", [("on", True), ("off", False), ("YES", True)])
def test_the_flashinfer_tactic_freeze_switch(raw, frozen, monkeypatch, tmp_path):
    monkeypatch.setenv(flashinfer_autotune_cache.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(flashinfer_autotune_cache.FROZEN_ENV, raw)
    assert flashinfer_autotune_cache.frozen() is frozen
