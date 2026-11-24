# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-target feature profiles of the Rubin K3 model (``k3_rubin/features.py``)."""

import dataclasses
import sys
import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin import features

pytestmark = pytest.mark.cpu_only

_TARGETS = {"tp1", "helix8", "helix16_moetp4ep4", "dep16"}


@pytest.fixture
def fresh(monkeypatch):
    """A process in which no profile is selected and no other k3_rubin module is imported."""
    monkeypatch.setattr(features, "_ACTIVE", features.K3Features())
    monkeypatch.setattr(features, "_SELECTED", None)
    monkeypatch.setattr(features, "_PACKAGE", "k3_features_test_no_such_package")
    warnings = []
    from tensorrt_llm.logger import logger

    monkeypatch.setattr(logger, "warning", lambda msg, *args, **kwargs: warnings.append(msg))
    return warnings


def test_one_profile_per_routing_target():
    assert set(features.PROFILES) == _TARGETS
    assert all(isinstance(p, features.K3Features) for p in features.PROFILES.values())


def test_every_field_replaces_one_retired_variable():
    fields = {f.name for f in dataclasses.fields(features.K3Features)}
    mapped = [f for f in features.RETIRED_ENV.values() if f is not None]
    assert sorted(mapped) == sorted(fields)


def test_context_profile_only_sets_context_features():
    assert set(features.PROFILES["tp1"].changed()) == {
        "kda_fused_conv_prefill",
        "kda_prepad_prefill",
        "mla_ctx_fp8_kv_fused",
        "mla_ctx_triton_quant",
    }


def test_active_is_the_plain_path_before_selection(fresh):
    assert features.active() == features.K3Features()
    assert features.active().changed() == {}
    assert features.selected() is None


def test_select_is_once_per_process(fresh):
    assert features.select("helix8") is features.PROFILES["helix8"]
    assert features.active() is features.PROFILES["helix8"]
    assert features.selected() == "helix8"
    assert features.select("helix8") is features.PROFILES["helix8"]
    with pytest.raises(RuntimeError, match="helix8"):
        features.select("dep16")


def test_select_rejects_unknown_targets(fresh):
    with pytest.raises(KeyError):
        features.select("tp8")


def test_select_after_a_k3_rubin_import_raises(fresh, monkeypatch):
    monkeypatch.setitem(
        sys.modules, f"{features._PACKAGE}.text_model", types.ModuleType("text_model")
    )
    with pytest.raises(RuntimeError, match="text_model"):
        features.select("tp1")
    assert features.selected() is None


def test_select_warns_about_retired_variables(fresh, monkeypatch):
    for name in features.RETIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    features.select("dep16")
    assert fresh == []
    monkeypatch.setattr(features, "_SELECTED", None)
    monkeypatch.setenv("TRTLLM_K3_ROUTE_TAIL", "1")
    features.select("dep16")
    assert len(fresh) == 1 and "TRTLLM_K3_ROUTE_TAIL" in fresh[0]
