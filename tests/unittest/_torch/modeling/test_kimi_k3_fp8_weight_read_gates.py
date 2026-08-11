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
"""Pin the Kimi K3 FP8 weight-read switch defaults.

The FP8 weight read is lossy relative to BF16, so the master switch must stay
opt-in — a default run keeps BF16 and matches the published accuracy numbers.
These tests exist so that default cannot drift silently: it did once, and the
only symptom was accuracy measured against a configuration nobody selected.

CPU-only: ``is_sm_100f`` is patched so the gates can be exercised without
Blackwell hardware.
"""

import pytest

from tensorrt_llm._torch.models import modeling_kimi_linear
from tensorrt_llm._torch.models.modeling_kimi_linear import _resolve_fp8_weight_read_gates

ENVS = (
    "KIMI_K3_FP8_WEIGHT_READ",
    "KIMI_K3_FP8_WEIGHT_READ_KDA",
    "KIMI_K3_KDA_GLUE_FP8",
)


@pytest.fixture
def sm100f(monkeypatch):
    """Report Blackwell so the SM gate never masks the env behavior."""
    monkeypatch.setattr(modeling_kimi_linear, "is_sm_100f", lambda: True)
    for env in ENVS:
        monkeypatch.delenv(env, raising=False)


def test_master_switch_defaults_off(sm100f):
    """With nothing set, no FP8 weight read anywhere."""
    assert _resolve_fp8_weight_read_gates() == (False, False, False)


def test_master_switch_opt_in(sm100f, monkeypatch):
    """Setting the master switch enables it and the default-on sub-gates."""
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    assert _resolve_fp8_weight_read_gates() == (True, True, True)


@pytest.mark.parametrize(
    "env,expected",
    [
        ("KIMI_K3_FP8_WEIGHT_READ_KDA", (True, False, False)),
        ("KIMI_K3_KDA_GLUE_FP8", (True, True, False)),
    ],
)
def test_sub_gates_narrow_an_enabled_master(sm100f, monkeypatch, env, expected):
    """Sub-gates only ever narrow; they never enable on their own."""
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    monkeypatch.setenv(env, "0")
    assert _resolve_fp8_weight_read_gates() == expected


@pytest.mark.parametrize("env", ENVS[1:])
def test_sub_gates_are_inert_while_master_is_off(sm100f, monkeypatch, env):
    """A sub-gate set to 1 must not turn on FP8 reads by itself."""
    monkeypatch.setenv(env, "1")
    assert _resolve_fp8_weight_read_gates() == (False, False, False)


def test_non_blackwell_never_reads_fp8(monkeypatch):
    """The SM gate wins even with the master switch explicitly on."""
    monkeypatch.setattr(modeling_kimi_linear, "is_sm_100f", lambda: False)
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    assert _resolve_fp8_weight_read_gates() == (False, False, False)
