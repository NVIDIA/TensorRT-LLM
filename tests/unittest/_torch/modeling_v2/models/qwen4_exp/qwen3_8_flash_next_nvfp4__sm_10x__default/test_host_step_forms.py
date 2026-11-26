# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-step host forms the shell declares as runtime hooks (CPU): `DP_STEPEDGE_ENV` and
`DP_RGAP_ENV` at a dep4 / dep8 rank on SM103, `TEP_STEPEDGE_ENV` at a tep4 rank on SM103, all on by
default where their row table reaches, off with `0`, and a warning where set to 1 outside it.

What the forms do is the engine's and the QSA metadata's (bitwise by construction): see
`tests/unittest/_torch/pyexecutor/test_steady_spec_prepare.py` (the padded attention-DP steps
against forced full passes) and `tests/unittest/_torch/attention/sparse/qsa/test_qsa_steady_decode.py`
(the one-launch block-offset copy and the QSA steady step)."""

import dataclasses
import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

_ENVS = (M.DP_STEPEDGE_ENV, M.DP_RGAP_ENV, M.TEP_STEPEDGE_ENV)


def _forms(monkeypatch, layout, sm103=True, **env):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(getattr(M, name), value)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: layout == "dep" and sm103)
    monkeypatch.setattr(M, "_tep4_layout", lambda core: layout == "tep4")
    monkeypatch.setattr(M, "_is_sm103", lambda device: sm103)
    warnings = []
    monkeypatch.setattr(M.logger, "warning", warnings.append)
    forms = M._host_step_forms(SimpleNamespace(), torch.device("cpu"))
    return forms, warnings


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "layout, sm103, env, expected, warned",
    [
        ("dep", True, {}, (True, True, True), 0),
        ("dep", True, dict(DP_RGAP_ENV="0"), (True, True, False), 0),
        ("dep", True, dict(DP_STEPEDGE_ENV="0"), (False, False, False), 0),
        # the replay gap rides on the steady step: on with the step off, it warns and stays off
        ("dep", True, dict(DP_STEPEDGE_ENV="0", DP_RGAP_ENV="1"), (False, False, False), 1),
        ("dep", False, {}, (False, False, False), 0),
        ("dep", False, dict(DP_STEPEDGE_ENV="1"), (False, False, False), 1),
        ("tep4", True, {}, (False, True, False), 0),
        ("tep4", True, dict(TEP_STEPEDGE_ENV="0"), (False, False, False), 0),
        ("tep4", False, dict(TEP_STEPEDGE_ENV="1"), (False, False, False), 1),
        ("tep4", True, dict(DP_STEPEDGE_ENV="1"), (False, True, False), 1),
        ("tp1", True, {}, (False, False, False), 0),
        ("tp1", True, dict(TEP_STEPEDGE_ENV="1"), (False, False, False), 1),
    ],
)
def test_row_table(monkeypatch, layout, sm103, env, expected, warned):
    forms, warnings = _forms(monkeypatch, layout, sm103, **env)
    assert (forms["dp_steady"], forms["qsa_steady"], forms["qsa_replay_gap"]) == expected
    assert len(warnings) == warned
    assert all("is set but ignored" in w for w in warnings)


@pytest.mark.cpu_only
def test_bad_value_raises(monkeypatch):
    with pytest.raises(ValueError):
        _forms(monkeypatch, "dep", DP_STEPEDGE_ENV="maybe")


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "forms",
    [
        dict(dp_steady=True, qsa_steady=True, qsa_replay_gap=True),
        dict(dp_steady=True, qsa_steady=True, qsa_replay_gap=False),
        dict(dp_steady=False, qsa_steady=True, qsa_replay_gap=False),
        dict(dp_steady=False, qsa_steady=False, qsa_replay_gap=False),
    ],
)
def test_runtime_hooks_carry_the_forms(monkeypatch, forms):
    """The row table reaches the engine: each form becomes its RuntimeHooks field, and the core keeps
    the forms for its decode-forms line. Without a core every form is off."""
    monkeypatch.setattr(M, "_host_step_forms", lambda core: dict(forms))
    fields = {field.name for field in dataclasses.fields(M.RuntimeHooks)}

    def runtime_hooks(**kw):
        # Every keyword is a field of the engine's RuntimeHooks.
        assert set(kw) <= fields, set(kw) - fields
        return SimpleNamespace(**kw)

    monkeypatch.setattr(M, "RuntimeHooks", runtime_hooks)
    monkeypatch.setattr(M, "QsaRuntimePolicy", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(M, "SpecStepMetadataPrologue", lambda: None)
    monkeypatch.delenv(M.CTX_COLDIDX_ENV, raising=False)
    core = SimpleNamespace()
    hooks = M._runtime_hooks(None, core)
    assert core.host_step_forms == forms
    assert hooks.steady_decode_prepare is forms["dp_steady"]
    assert hooks.qsa.steady_decode is forms["qsa_steady"]
    assert hooks.qsa.steady_replay_gap is forms["qsa_replay_gap"]
    hooks = M._runtime_hooks(None)
    assert hooks.steady_decode_prepare is False
    assert (hooks.qsa.steady_decode, hooks.qsa.steady_replay_gap) == (False, False)


@pytest.mark.cpu_only
def test_the_shell_passes_its_core():
    src = inspect.getsource(M.ModelingV2Qwen38FlashNextNvfp4Sm10xDefault.__init__)
    assert "_runtime_hooks(self.spec_config, self.model)" in src


@pytest.mark.cpu_only
def test_the_decode_forms_line_names_them():
    src = inspect.getsource(M._decode_forms)
    for key in ('"steady prepare"', '"qsa steady"', '"replay gap"'):
        assert key in src
    # the tep4 branch names the QSA steady step too
    assert src.count('"qsa steady"') == 2
