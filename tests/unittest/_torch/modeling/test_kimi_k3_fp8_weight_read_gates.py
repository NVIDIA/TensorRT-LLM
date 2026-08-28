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
"""Pin the Kimi K3 FP8 weight-read gate resolution and its defaults.

The FP8 weight read is lossy relative to BF16, so the master switch must stay
opt-in — a default run keeps BF16 and matches the published accuracy numbers.
These tests exist so that default cannot drift silently: it did once, and the
only symptom was accuracy measured against a configuration nobody selected.

The 5 gates now resolve from ``QuantConfig`` (an explicit config value wins),
else the deprecated ``KIMI_K3_*`` env var (still honored, warn-once), else the
historical default. Both surfaces are exercised here, along with the migration's
acceptance criteria at the config/unit level: each of the 5 former env vars has
a config equivalent, a run with neither env vars nor config values resolves a
complete and valid set of gates, and a set env var is honored exactly once with
a deprecation warning.

CPU-only: ``is_sm_100f`` is patched so the gates can be exercised without
Blackwell hardware.
"""

import pytest

from tensorrt_llm._torch.models import kimi_k3_knobs
from tensorrt_llm._torch.models.kimi_k3_knobs import resolve_fp8_weight_read_gates
from tensorrt_llm.models.modeling_utils import QuantConfig

ENVS = (
    "KIMI_K3_FP8_WEIGHT_READ",
    "KIMI_K3_FP8_WEIGHT_READ_KDA",
    "KIMI_K3_FP8_WEIGHT_READ_MLA",
    "KIMI_K3_FP8_WEIGHT_READ_GATE_UP",
    "KIMI_K3_KDA_GLUE_FP8",
)


@pytest.fixture
def sm100f(monkeypatch):
    """Report Blackwell so the SM gate never masks env/config behavior; clear env."""
    monkeypatch.setattr(kimi_k3_knobs, "is_sm_100f", lambda: True)
    for env in ENVS:
        monkeypatch.delenv(env, raising=False)


def _gates(quant_config=None, *, enable_attention_dp=False):
    return resolve_fp8_weight_read_gates(quant_config, enable_attention_dp=enable_attention_dp)


def _tuple(g):
    return (g.master, g.kda, g.kda_glue, g.mla, g.gate_up)


# --- Defaults (config surface) ---------------------------------------------


def test_master_switch_defaults_off(sm100f):
    """With nothing set, no FP8 weight read anywhere."""
    assert _tuple(_gates(QuantConfig())) == (False, False, False, False, False)


def test_master_switch_opt_in_via_config(sm100f):
    """Setting the master switch enables it and the default-on sub-gates."""
    g = _gates(QuantConfig(kimi_k3_fp8_weight_read=True), enable_attention_dp=True)
    assert _tuple(g) == (True, True, True, True, True)


def test_gate_up_default_follows_attention_dp(sm100f):
    """gate_up defaults on under attention-DP, off otherwise (master on)."""
    on = _gates(QuantConfig(kimi_k3_fp8_weight_read=True), enable_attention_dp=True)
    off = _gates(QuantConfig(kimi_k3_fp8_weight_read=True), enable_attention_dp=False)
    assert on.gate_up is True and off.gate_up is False


@pytest.mark.parametrize(
    "field,expected",
    [
        # kda off -> kda_glue collapses too; mla/gate_up independent.
        ("kimi_k3_fp8_weight_read_kda", (True, False, False, True, True)),
        ("kimi_k3_kda_glue_fp8", (True, True, False, True, True)),
        ("kimi_k3_fp8_weight_read_mla", (True, True, True, False, True)),
        ("kimi_k3_fp8_weight_read_gate_up", (True, True, True, True, False)),
    ],
)
def test_sub_gates_narrow_enabled_master_via_config(sm100f, field, expected):
    """Sub-gates only ever narrow an enabled master; they never enable alone."""
    g = _gates(
        QuantConfig(kimi_k3_fp8_weight_read=True, **{field: False}), enable_attention_dp=True
    )
    assert _tuple(g) == expected


@pytest.mark.parametrize(
    "field",
    [
        "kimi_k3_fp8_weight_read_kda",
        "kimi_k3_fp8_weight_read_mla",
        "kimi_k3_fp8_weight_read_gate_up",
        "kimi_k3_kda_glue_fp8",
    ],
)
def test_sub_gates_inert_while_master_off_via_config(sm100f, field):
    """A sub-gate set on must not turn on FP8 reads by itself."""
    g = _gates(QuantConfig(**{field: True}), enable_attention_dp=True)
    assert _tuple(g) == (False, False, False, False, False)


def test_non_blackwell_never_reads_fp8_config(monkeypatch):
    """The SM gate wins even with the master switch explicitly on (config)."""
    monkeypatch.setattr(kimi_k3_knobs, "is_sm_100f", lambda: False)
    for env in ENVS:
        monkeypatch.delenv(env, raising=False)
    g = _gates(QuantConfig(kimi_k3_fp8_weight_read=True), enable_attention_dp=True)
    assert _tuple(g) == (False, False, False, False, False)


# --- Deprecated env var back-compat (still honored) ------------------------


def test_master_switch_opt_in_via_deprecated_env(sm100f, monkeypatch):
    """The deprecated master env var is still honored when config is unset."""
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    g = _gates(QuantConfig(), enable_attention_dp=False)
    # gate_up follows attention_dp=False -> off; the rest default on.
    assert _tuple(g) == (True, True, True, True, False)


@pytest.mark.parametrize(
    "env,expected",
    [
        ("KIMI_K3_FP8_WEIGHT_READ_KDA", (True, False, False, True, True)),
        ("KIMI_K3_KDA_GLUE_FP8", (True, True, False, True, True)),
        ("KIMI_K3_FP8_WEIGHT_READ_MLA", (True, True, True, False, True)),
        ("KIMI_K3_FP8_WEIGHT_READ_GATE_UP", (True, True, True, True, False)),
    ],
)
def test_sub_gates_narrow_via_deprecated_env(sm100f, monkeypatch, env, expected):
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    monkeypatch.setenv(env, "0")
    assert _tuple(_gates(QuantConfig(), enable_attention_dp=True)) == expected


def test_non_blackwell_never_reads_fp8_env(monkeypatch):
    """The SM gate wins even with the master env var on."""
    monkeypatch.setattr(kimi_k3_knobs, "is_sm_100f", lambda: False)
    for env in ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    assert _tuple(_gates(QuantConfig(), enable_attention_dp=True)) == (
        False,
        False,
        False,
        False,
        False,
    )


# --- Precedence: config wins over the deprecated env var -------------------


def test_config_master_off_beats_env_master_on(sm100f, monkeypatch):
    """Explicit config value takes precedence over the deprecated env var."""
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    g = _gates(QuantConfig(kimi_k3_fp8_weight_read=False), enable_attention_dp=True)
    assert _tuple(g) == (False, False, False, False, False)


def test_config_subgate_off_beats_env_subgate_on(sm100f, monkeypatch):
    """A config sub-gate overrides a conflicting deprecated env sub-gate."""
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ_KDA", "1")
    g = _gates(
        QuantConfig(kimi_k3_fp8_weight_read=True, kimi_k3_fp8_weight_read_kda=False),
        enable_attention_dp=True,
    )
    assert g.master is True and g.kda is False and g.kda_glue is False


# --- Config surface: every former env var has a config equivalent -----------


def test_fp8_knobs_exist_on_quant_config():
    """Each of the 5 former KIMI_K3_* env vars has a QuantConfig field."""
    fields = QuantConfig.model_fields
    for name in kimi_k3_knobs.KIMI_K3_QUANT_KNOB_FIELDS:
        assert name in fields, f"QuantConfig missing {name}"
        assert fields[name].default is None, f"{name} must default to None (unset)"
    assert len(kimi_k3_knobs.KIMI_K3_QUANT_KNOB_FIELDS) == len(ENVS)


def test_knobs_settable_from_extra_llm_api_options_dict():
    """The knobs round-trip through a plain dict, as extra_llm_api_options does."""
    qc = QuantConfig.model_validate(
        {"kimi_k3_fp8_weight_read": True, "kimi_k3_fp8_weight_read_kda": False}
    )
    assert qc.kimi_k3_fp8_weight_read is True
    assert qc.kimi_k3_fp8_weight_read_kda is False


def test_zero_env_zero_config_resolves_complete_valid(sm100f):
    """No env vars + a default QuantConfig -> every gate resolves to a concrete
    value (the historical BF16 default), with no missing-knob error."""
    g = _gates(QuantConfig(), enable_attention_dp=False)
    assert _tuple(g) == (False, False, False, False, False)
    assert all(isinstance(v, bool) for v in _tuple(g))


# --- Deprecation warning: emitted once, and even when config overrides ------


class _RecordingLogger:
    """Mirrors ``logger.warning_once`` dedup so warn-once is deterministic."""

    def __init__(self):
        self._seen = set()
        self.emitted = []

    def warning_once(self, *msg, key):
        if key not in self._seen:
            self._seen.add(key)
            self.emitted.append((key, " ".join(str(m) for m in msg)))


def test_deprecated_env_warns_once_and_is_honored(sm100f, monkeypatch):
    rec = _RecordingLogger()
    monkeypatch.setattr(kimi_k3_knobs, "logger", rec)
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    # Resolve several times; the deprecation warning must appear exactly once.
    for _ in range(3):
        assert _gates(QuantConfig(), enable_attention_dp=True).master is True
    keys = [k for k, _ in rec.emitted]
    assert keys == ["kimi_k3_deprecated_env::KIMI_K3_FP8_WEIGHT_READ"]
    assert "extra_llm_api_options" in rec.emitted[0][1]


def test_warn_once_even_when_config_overrides(sm100f, monkeypatch):
    """The deprecation is about the env var existing; it warns even when a
    config value overrides the env value."""
    rec = _RecordingLogger()
    monkeypatch.setattr(kimi_k3_knobs, "logger", rec)
    monkeypatch.setenv("KIMI_K3_FP8_WEIGHT_READ", "1")
    assert _gates(QuantConfig(kimi_k3_fp8_weight_read=False), enable_attention_dp=True).master is (
        False
    )
    assert [k for k, _ in rec.emitted] == ["kimi_k3_deprecated_env::KIMI_K3_FP8_WEIGHT_READ"]
