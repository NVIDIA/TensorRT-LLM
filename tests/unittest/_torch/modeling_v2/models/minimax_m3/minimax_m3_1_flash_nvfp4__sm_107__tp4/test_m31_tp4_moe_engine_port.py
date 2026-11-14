# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4_MOE_ENGINE without a GPU: the family engine (kernels/m3ll/moe_engine_kernel.py and its host) stands apart from
the hoisted block package, and ``tp4_moe_engine.engine_config`` keeps the m3_moe arithmetic, CTA split and plan keys,
adds only the reference engine's scheduling keys, never takes a key that changes a summation order, and refuses an
fc1_done ring that can alias a phase. The GPU bitwise checks are in
minimax_m3_1_flash_nvfp4__sm_107__tp4/moe/test_modeling_v2_m31_vr_tp4_moe_engine_m3ll.py."""

import re
from pathlib import Path

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.kernels.m3ll import (  # noqa: E501
    tp4_moe_engine as wc,
)

pytestmark = pytest.mark.cpu_only

_PKG = Path(wc.__file__).resolve().parent


def _ours(m: int) -> dict:
    """The sm_107 tp4 m3_moe state config at ``m`` rows, as kernels/m3_moe/op.py builds it with every switch on."""
    cfg = {
        "m_max": m,
        "num_ctas": 188,
        "nb2_cap": 2,
        "fused_router": 1,
        "timing": 0,
        "i_tp": 768,
        "sh_i": 768,
        "shared": 1,
        "sfa_cp": 1,
        "sfa_bulk": 1,
        "k128": 1,
        "b2img": 0,
        "ar_pre": 0 if m == 2 else 1,
        "out_skip": 1,
        "prod_trigger": 1,
    }
    if m <= 4:
        cfg.update({"b2img": 0, "lg_repl": 16, "ar_pre": 0, "late_trigger": 0})
    if m == 8:
        cfg.update({"stages": 5, "b2_poll_ns": 50, "early0": 0})
    return cfg


def test_engine_kernel_and_host_are_the_family_modules() -> None:
    assert (_PKG / "moe_engine_kernel.py").is_file()
    host = (_PKG / "moe_engine_host.py").read_text()
    assert "moe_engine_kernel" in host
    assert not re.search(r"^\s*(from|import)\s+\S*\btp4c1\b", host, re.M), (
        "the host must not import the TP4 C1 block package"
    )


@pytest.mark.parametrize("m", range(1, 9))
def test_config_keeps_the_m3_moe_arithmetic_and_split(m: int) -> None:
    ours = _ours(m)
    cfg = wc.engine_config(ours)
    for k in (
        "m_max",
        "num_ctas",
        "nb2_cap",
        "fused_router",
        "i_tp",
        "sh_i",
        "shared",
        "sfa_cp",
        "sfa_bulk",
        "k128",
        "ar_pre",
    ):
        assert cfg[k] == ours[k], k
    assert cfg["router_vllm"] == 0 and cfg["silu_vllm"] == 0
    for k in wc.OURS_ONLY_KEYS:
        assert k not in cfg
    for k in wc.NOT_TAKEN_KEYS:
        assert not cfg.get(k), k
    assert cfg["roll_fc2"] == 1
    if m >= 5:
        assert (
            cfg["b2img"],
            cfg["sf2_w6"],
            cfg["lag"],
            cfg["fc1d"],
            cfg["b2buf"],
            cfg["stages"],
        ) == (1, 1, 6, 8, 2, 6)
        assert (cfg["early_trig"], cfg["sh_trig"], cfg["late_trigger"]) == (1, 1, 1)
    else:
        assert "lag" not in cfg and "sf2_w6" not in cfg and "early_trig" not in cfg
        assert cfg["late_trigger"] == 0 and cfg["b2img"] == 0


def test_kill_knobs() -> None:
    cfg = wc.engine_config(_ours(8), b2lag=False)
    assert "lag" not in cfg and "fc1d" not in cfg and cfg["b2buf"] == 1 and cfg["stages"] == 5
    cfg = wc.engine_config(_ours(8), early_trig=False)
    assert "early_trig" not in cfg and "sh_trig" not in cfg
    assert wc.engine_config(_ours(8), stages=5)["stages"] == 5


@pytest.mark.parametrize(
    "m,kw,want",
    [
        (8, {}, None),  # B2LAG: 6 stages hold the 96 KB of rows
        (
            8,
            {"b2lag": False},
            0,
        ),  # m3_moe's 5-stage ring cannot (the s1u1b "x staging must fit the ring")
        (8, {"b2lag": False, "ours_xstage": False}, 0),
        (8, {"stages": 5}, 0),  # the B2LAG stage fallback
        (8, {"early_trig": False}, None),
        (6, {"b2lag": False, "ours_xstage": False}, 0),  # m3_moe's ring and its choice
        (6, {"b2lag": False, "ours_xstage": True}, None),
        (4, {"ours_xstage": True}, None),
    ],
)
def test_xstage_follows_the_ring(m: int, kw: dict, want) -> None:
    """The reference kernel asserts that the staged rows fit the ring where m3_moe silently reads them from global: the
    config states xstage 0 wherever the ring cannot hold M x 6144 x 2 bytes, or m3_moe runs without it."""
    cfg = wc.engine_config(_ours(m), **kw)
    assert cfg.get("xstage") == want
    if "stages" in cfg and cfg.get("xstage", 1):
        assert m * 6144 * 2 <= cfg["stages"] * wc.STAGE_BYTES_SFA_CP


@pytest.mark.parametrize(
    "key", ["dl", "sharded", "nsh", "ar_g3", "ext_route", "b1_x4", "fc1_per_token"]
)
def test_summation_order_keys_are_refused(key: str) -> None:
    with pytest.raises(ValueError, match="never takes"):
        wc.engine_config({**_ours(8), key: 1})


@pytest.mark.parametrize(
    "lag,fc1d,m,ok",
    [
        (2, 4, 8, True),
        (6, 8, 8, True),
        (6, 4, 8, False),
        (6, 16, 8, True),
        (3, 4, 8, False),
        (3, 8, 8, True),
        (6, 6, 8, False),
        (6, 8, 9, False),
    ],
)
def test_ring_depth(lag: int, fc1d: int, m: int, ok: bool) -> None:
    """A power of two > LAG + 1 (the kernel's assert), and past lag 2 one slot per group of the call (NG_CAP = m)."""
    cfg = {"m_max": m, "lag": lag, "fc1d": fc1d}
    if ok:
        wc.check_ring(cfg)
    else:
        with pytest.raises(ValueError):
            wc.check_ring(cfg)


def test_default_ring_depth_matches_the_kernel() -> None:
    assert [wc.default_fc1d(lag) for lag in range(0, 7)] == [4, 4, 4, 8, 8, 8, 8]
