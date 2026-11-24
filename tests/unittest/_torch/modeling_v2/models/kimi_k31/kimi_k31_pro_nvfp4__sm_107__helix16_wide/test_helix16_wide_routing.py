# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TLLM_K31_HELIX16_WIDE``: the ULL generation route (``routing._route_ull_generation``) returns
the wide-batch target instead of ``kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1`` (same recipe
checks, same ``ull16`` profile) and records its two switches in ``model_config.extra_attrs``:
``TLLM_K31_HELIX16_WIDE_TAIL`` (``tuned`` / ``rows``) and ``TLLM_K31_HELIX16_WIDE_FRONT``
(``tmem_merged`` / ``once``); another value of any of the three is refused before the target is
built. The recipe module's ``select`` refuses a module of either ULL package imported before it.
Skips on a host where the family's routing and recipe modules do not import.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from types import SimpleNamespace

import pytest
from _helix16_wide_cpu import FAMILY_PKG

pytestmark = pytest.mark.cpu_only

XIN = (
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1",
    "ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1",
)
WIDE = ("kimi_k31_pro_nvfp4__sm_107__helix16_wide", "ModelingV2KimiK31ProNvfp4Sm107Helix16Wide")
SWITCHES = ("TLLM_K31_HELIX16_WIDE", "TLLM_K31_HELIX16_WIDE_TAIL", "TLLM_K31_HELIX16_WIDE_FRONT")


def _import(name: str):
    try:
        return importlib.import_module(name)
    except Exception as error:  # noqa: BLE001 (any import failure of the engine's dependencies)
        pytest.skip(f"{name} does not import on this host: {type(error).__name__}: {error}")


@pytest.fixture
def ull_route(monkeypatch):
    """``_route_ull_generation`` on a ULL deployment that passes its recipe checks, with stand-ins
    for both targets' modeling modules and ``select`` recorded instead of run."""
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    routing = _import(f"{FAMILY_PKG}.routing")
    ull = _import(f"{FAMILY_PKG}.ops.common.rubin_ull_features")
    targets = {}
    for directory, name in (XIN, WIDE):
        module = types.ModuleType(f"{FAMILY_PKG}.{directory}.modeling")
        targets[directory] = type(name, (), {})
        setattr(module, name, targets[directory])
        monkeypatch.setitem(sys.modules, module.__name__, module)
    selected = []
    monkeypatch.setattr(ull, "select", selected.append)
    monkeypatch.setattr(ull, "validate_native_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(routing, "disagg_role", lambda llm_args=None: "generation")
    monkeypatch.setattr(routing, "_kv_cache_quant_algo_name", lambda *args: "FP8")
    for name in ("_dflash2_mla", "_history", "_history_only", "_draft_schedule"):
        monkeypatch.setattr(routing, f"{name}_mismatch", lambda *args: None)
    mapping = SimpleNamespace(
        world_size=16, tp_size=1, pp_size=1, cp_size=16, enable_attention_dp=False,
        moe_tp_size=16, moe_ep_size=1, moe_tp_ep_user_specified=True, rank=0,
        has_cp_helix=lambda: True,
    )  # fmt: skip
    spec = SimpleNamespace(spec_dec_mode=SimpleNamespace(name="DFLASH"), max_draft_len=7)
    config = SimpleNamespace(mapping=mapping, spec_config=spec, extra_attrs={})
    args = SimpleNamespace(
        moe_config=SimpleNamespace(load_balancer=None, use_low_precision_moe_combine=False,
                                   backend="CUTEDSL"),
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=True,
                                        mamba_ssm_cache_dtype="float32"),
        max_beam_width=1, max_batch_size=16,
    )  # fmt: skip

    def route():
        return routing._route_ull_generation(config, args, "fp8")

    return route, targets, selected, ull, config


@pytest.mark.parametrize(
    "env, record",
    [({}, None), ({"TLLM_K31_HELIX16_WIDE": "off", "TLLM_K31_HELIX16_WIDE_TAIL": "rows"}, None),
     ({"TLLM_K31_HELIX16_WIDE": " on "}, ("tuned", "tmem_merged")),
     ({"TLLM_K31_HELIX16_WIDE": "true", "TLLM_K31_HELIX16_WIDE_FRONT": ""},
      ("tuned", "tmem_merged")),
     ({"TLLM_K31_HELIX16_WIDE": "1", "TLLM_K31_HELIX16_WIDE_TAIL": " Rows ",
       "TLLM_K31_HELIX16_WIDE_FRONT": "ONCE"}, ("rows", "once"))],
)  # fmt: skip
def test_the_switches_pick_the_target_and_its_record(ull_route, monkeypatch, env, record):
    """Off: the ULL target, nothing recorded. On: the wide target and its two switches recorded
    (the fork reads the record, not the environment). The same recipe and profile either way."""
    route, targets, selected, ull, config = ull_route
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert route() is targets[(XIN if record is None else WIDE)[0]]
    assert selected == [ull.ULL16_GENERATION]
    if record is None:
        assert config.extra_attrs == {}
    else:
        want = dict(zip(("tail", "front"), record))
        assert config.extra_attrs.get("kimi_k31_helix16_wide") == want


@pytest.mark.parametrize(
    "name, value",
    [("TLLM_K31_HELIX16_WIDE", "2"), ("TLLM_K31_HELIX16_WIDE", "ture"),  # codespell:ignore ture
     ("TLLM_K31_HELIX16_WIDE_TAIL", "wide"), ("TLLM_K31_HELIX16_WIDE_TAIL", "0"),
     ("TLLM_K31_HELIX16_WIDE_FRONT", "tmem"), ("TLLM_K31_HELIX16_WIDE_FRONT", "off")],
)  # fmt: skip
def test_another_value_is_refused(ull_route, monkeypatch, name, value):
    """A misspelt selector does not route to the ULL target silently; a switch value is refused."""
    route, _, _, _, config = ull_route
    monkeypatch.setenv("TLLM_K31_HELIX16_WIDE", "1")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name + ("=" if name == SWITCHES[0] else "")):
        route()
    assert config.extra_attrs == {}


def test_select_checks_both_packages(monkeypatch):
    """A module of either ULL package imported before ``select`` is refused; each package's
    ``features`` is exempt, and ``_PACKAGE`` stays the ULL target's (its tests' conftest reads
    it)."""
    features = _import(f"{FAMILY_PKG}.ops.common.rubin_ull_features")
    packages = tuple(f"{FAMILY_PKG}.{directory}" for directory, _ in (XIN, WIDE))
    if any(m.startswith(p + ".") for p in packages for m in sys.modules):
        pytest.skip("a ULL target package is imported in this process")
    assert features._PACKAGE == packages[0] and features._PACKAGES == packages
    monkeypatch.setattr(features, "_SELECTED", None)
    monkeypatch.setattr(features, "_ACTIVE", features._ACTIVE)
    configured = []
    switchboard = SimpleNamespace(configure=lambda *args, **kwargs: configured.append(args))
    monkeypatch.setattr(features, "_switchboard", lambda: switchboard)
    for package in packages:
        early = f"{package}.modeling"
        monkeypatch.setitem(sys.modules, early, types.ModuleType(early))
        with pytest.raises(RuntimeError, match=re.escape(early)):
            features.select(features.ULL16_GENERATION)
        monkeypatch.delitem(sys.modules, early)
        monkeypatch.setitem(sys.modules, f"{package}.features", types.ModuleType("features"))
    profile = features.PROFILES[features.ULL16_GENERATION]
    assert features.select(features.ULL16_GENERATION) is profile
    assert len(configured) == 1
