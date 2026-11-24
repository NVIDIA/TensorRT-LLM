# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode switchboard (``tensorrt_llm/_torch/decode_switches.py``). Needs neither torch nor a GPU: each test
loads a fresh copy of the module by path, as a new process sees it."""

import dataclasses
import importlib.util
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

_PATH = (
    pathlib.Path(__file__).resolve().parents[3] / "tensorrt_llm" / "_torch" / "decode_switches.py"
)
_NAME = "_decode_switches_under_test"


class _Log:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, msg):
        self.infos.append(msg)

    def warning(self, msg):
        self.warnings.append(msg)


@pytest.fixture
def sw(monkeypatch):
    """A fresh copy of the switchboard with a recording logger (``sw.log``)."""
    spec = importlib.util.spec_from_file_location(_NAME, _PATH)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _NAME, mod)
    spec.loader.exec_module(mod)
    log = _Log()
    monkeypatch.setattr(mod, "_logger", lambda: log)
    monkeypatch.setattr(mod, "log", log, raising=False)
    for name in (mod.DISAGG_ROLE_ENV, *mod.RETIRED_ENV):
        monkeypatch.delenv(name, raising=False)
    return mod


def test_plain_path_before_configure(sw):
    """Before configure(), active() is the default instance, and each default has its field's type. The defaults'
    values are checked against v1 with the switches unset by the K3.1 Rubin switch-consistency test."""
    switches = sw.active()
    assert isinstance(switches, sw.DecodeSwitches)
    assert dataclasses.asdict(switches) == dataclasses.asdict(sw.DecodeSwitches())
    assert switches.changed() == {}
    assert sw.configured() is None
    for field in dataclasses.fields(switches):
        value = getattr(switches, field.name)
        assert field.type in (type(value).__name__, type(value)), field.name


def test_frozen(sw):
    with pytest.raises(dataclasses.FrozenInstanceError):
        sw.active().kv_v2_zero_gpu_pools = True


def test_configure_once(sw):
    switches = sw.DecodeSwitches(kv_v2_zero_gpu_pools=True, dsa_helix_key_shard=True)
    sw.configure(switches, "helix8/generation")
    assert sw.active() is switches
    assert sw.configured() == "helix8/generation"
    assert sw.log.infos == [
        "decode switches: helix8/generation (2 set): dsa_helix_key_shard=True, kv_v2_zero_gpu_pools=True"
    ]
    assert sw.log.warnings == []


def test_configure_same_settings_again_is_a_no_op(sw):
    sw.configure(sw.DecodeSwitches(dsa_helix_key_shard=True), "helix8/generation")
    sw.configure(sw.DecodeSwitches(dsa_helix_key_shard=True), "helix8/generation (drafter)")
    assert sw.active().dsa_helix_key_shard
    assert sw.configured() == "helix8/generation"
    assert len(sw.log.infos) == 1


def test_configure_different_settings_raises(sw):
    sw.configure(sw.DecodeSwitches(dsa_helix_key_shard=True), "helix8/generation")
    with pytest.raises(RuntimeError, match=r"dsa_helix_key_shard=True->False"):
        sw.configure(sw.DecodeSwitches(), "dep16/context")
    assert sw.active().dsa_helix_key_shard


def test_configure_plain_settings_counts_as_configured(sw):
    sw.configure(sw.DecodeSwitches(), "tep8/context")
    assert sw.log.infos == ["decode switches: tep8/context (0 set)"]
    with pytest.raises(RuntimeError):
        sw.configure(sw.DecodeSwitches(kv_v2_zero_gpu_pools=True), "dep16/context")


def test_configure_rejects_other_types(sw):
    with pytest.raises(TypeError):
        sw.configure({"kv_v2_zero_gpu_pools": True}, "dep16/context")
    assert sw.configured() is None


def test_override_restores(sw):
    sw.configure(sw.DecodeSwitches(kv_v2_zero_gpu_pools=True), "dep16/generation")
    with sw.override(dsa_helix_key_shard=True, dsa_helix_fused_meta=True) as inside:
        assert sw.active() is inside
        assert inside.dsa_helix_key_shard and inside.dsa_helix_fused_meta
        assert inside.kv_v2_zero_gpu_pools
        with sw.override(dsa_helix_key_shard=False):
            assert not sw.active().dsa_helix_key_shard
        assert sw.active().dsa_helix_key_shard
    assert sw.active() == sw.DecodeSwitches(kv_v2_zero_gpu_pools=True)


def test_override_restores_after_an_exception(sw):
    with pytest.raises(ValueError):
        with sw.override(dsa_helix_fused_meta=True):
            raise ValueError
    assert sw.active() == sw.DecodeSwitches()


def test_override_rejects_unknown_fields(sw):
    with pytest.raises(TypeError):
        with sw.override(no_such_field=True):
            pass


def test_read_before_configure_warns_when_configure_changes_a_setting(sw):
    assert not sw.active().kv_v2_zero_gpu_pools  # a read site that runs before route()
    sw.configure(sw.DecodeSwitches(kv_v2_zero_gpu_pools=True), "dep16/context")
    assert len(sw.log.warnings) == 1
    assert "read before they were configured" in sw.log.warnings[0]
    assert "test_decode_switches.py" in sw.log.warnings[0]


def test_read_before_configure_is_quiet_for_plain_settings(sw):
    sw.active()
    sw.configure(sw.DecodeSwitches(), "tep8/context")
    assert sw.log.warnings == []


def test_no_warning_without_an_early_read(sw):
    sw.configure(sw.DecodeSwitches(kv_v2_zero_gpu_pools=True), "dep16/context")
    sw.active()
    assert sw.log.warnings == []


def _v1_context_server(env):
    """The context-server predicate that ``disagg_role`` replaces (the role every v1 predicate used)."""
    return env.get("TRTLLM_DISAGG_ROLE") == "context"


@pytest.mark.parametrize("role", [None, "context", "generation", "", "CONTEXT"])
def test_disagg_role_matches_v1_context_server(sw, monkeypatch, role):
    env = {}
    if role is not None:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
        env["TRTLLM_DISAGG_ROLE"] = role
    expected = "context" if _v1_context_server(env) else "generation"
    assert sw.disagg_role() == expected
    assert sw.disagg_role(None) == expected


class _Args:
    def __init__(self, env_overrides):
        self.env_overrides = env_overrides


@pytest.mark.parametrize(
    "env_role, overrides, expected",
    [
        # An MPI rank spawned before trtllm-serve set the variable sees only the overrides.
        (None, {"TRTLLM_DISAGG_ROLE": "context"}, "context"),
        (None, {"TRTLLM_DISAGG_ROLE": "generation"}, "generation"),
        # The worker applies the overrides over its environment, so they win.
        ("context", {"TRTLLM_DISAGG_ROLE": "generation"}, "generation"),
        ("generation", {"TRTLLM_DISAGG_ROLE": "context"}, "context"),
        # No override of the role: the environment decides.
        ("context", {"OTHER": "1"}, "context"),
        ("context", None, "context"),
        (None, None, "generation"),
    ],
)
def test_disagg_role_reads_env_overrides_first(sw, monkeypatch, env_role, overrides, expected):
    if env_role is not None:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", env_role)
    assert sw.disagg_role(_Args(overrides)) == expected
    assert sw.disagg_role({"env_overrides": overrides}) == expected


def test_warn_retired_env_once(sw, monkeypatch):
    monkeypatch.setenv("K31_SWITCHES_TEST_RETIRED_A", "1")
    monkeypatch.delenv("K31_SWITCHES_TEST_RETIRED_B", raising=False)
    names = ["K31_SWITCHES_TEST_RETIRED_A", "K31_SWITCHES_TEST_RETIRED_B"]
    sw.warn_retired_env(names)
    assert len(sw.log.warnings) == 1
    assert "K31_SWITCHES_TEST_RETIRED_A" in sw.log.warnings[0]
    assert "K31_SWITCHES_TEST_RETIRED_B" not in sw.log.warnings[0]
    sw.warn_retired_env(names)
    assert len(sw.log.warnings) == 1


def test_configure_warns_about_retired_switch_env(sw, monkeypatch):
    monkeypatch.setattr(sw, "RETIRED_ENV", {"K31_SWITCHES_TEST_RETIRED_C": "kv_v2_zero_gpu_pools"})
    monkeypatch.setenv("K31_SWITCHES_TEST_RETIRED_C", "0")
    sw.configure(sw.DecodeSwitches(), "dep16/generation")
    assert len(sw.log.warnings) == 1
    assert "K31_SWITCHES_TEST_RETIRED_C" in sw.log.warnings[0]


def test_reset(sw):
    sw.active()
    sw.configure(sw.DecodeSwitches(dsa_helix_key_shard=True), "helix8/generation")
    sw._reset()
    assert sw.configured() is None
    assert sw.active() == sw.DecodeSwitches()
    sw.configure(sw.DecodeSwitches(), "tep8/context")


def test_package_import(sw):
    pytest.importorskip("torch")
    from tensorrt_llm._torch import decode_switches

    assert isinstance(decode_switches.active(), decode_switches.DecodeSwitches)
    package_defaults = dataclasses.asdict(decode_switches.DecodeSwitches())
    assert package_defaults == dataclasses.asdict(sw.DecodeSwitches())
