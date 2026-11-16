# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Muse Spark Open GB300 deployment defaults (``deployment_defaults``)."""

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open import (
    deployment_defaults as dd,
)

pytestmark = pytest.mark.cpu_only


def _facts(max_batch_size, max_num_tokens, disaggregated):
    return types.SimpleNamespace(
        max_batch_size=max_batch_size,
        max_num_tokens=max_num_tokens,
        disaggregated=disaggregated,
    )


def _mapping(tp):
    return types.SimpleNamespace(tp_size=tp)


@pytest.mark.parametrize(
    "env, facts, role",
    [
        ({}, _facts(64, 8192, False), "agg"),
        ({}, _facts(64, 8192, True), "ctx"),
        ({}, _facts(4, 32, True), "gen"),
        ({}, _facts(31, 512, True), "gen"),
        ({}, _facts(1, 512, True), "gen"),
        ({dd.ROLE_ENV: "ctx"}, _facts(4, 32, True), "ctx"),
        ({}, None, None),
    ],
)
def test_infer_role(env, facts, role):
    assert dd.infer_role(env, facts)[0] == role


@pytest.mark.parametrize(
    "mbs, point", [(1, None), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8), (9, 16), (31, 31), (32, None)]
)
def test_f11_point(mbs, point):
    assert dd.f11_point(mbs) == point


def test_profiles_by_layout():
    switches, name = dd.profile(103, 4, "agg", 64)
    assert name == "tep4-agg"
    assert switches["MUSESPARK_OPEN_SWA_CL8"] == "all"
    assert "MUSESPARK_OPEN_SP_TAIL_PIPELINE" not in switches
    assert "early_expert_ids" in switches["MUSESPARK_OPEN_SM103_TEP4_EXTRA_PATHS"]
    switches, name = dd.profile(103, 4, "ctx", 64)
    assert switches["MUSESPARK_OPEN_SP_DROP_PARTS"] == "ce_gather"
    # A context server runs no decode step: none of the decode-path extra paths (early_expert_ids
    # refuses to start without the persistent expert window a static placement leaves unbuilt).
    assert not set(dd.TEP4_DECODE) & set(switches)
    assert "MUSESPARK_OPEN_SM103_TEP4_EXTRA_PATHS" in dd.profile(103, 4, "gen", 8)[0]
    switches, name = dd.profile(103, 8, "gen", 4)
    assert name == "tep8-gen-f11-b4" and switches[dd.F11_ENV] == "1"
    # TEP4-only switches never reach a TEP8 process (F11 reads MUSESPARK_OPEN_SWA_CL8 too).
    assert not set(dd.TEP4) & set(switches)
    switches, name = dd.profile(103, 8, "gen", 1)
    assert name == "tep8-gen-aria" and dd.F11_ENV not in switches
    assert "MUSESPARK_OPEN_LARGE_NORM_GATE" not in switches
    assert dd.profile(103, 8, "agg", 6)[1] == "tep8-agg-aria"
    assert dd.profile(107, 8, "gen", 4) == ({}, "none")
    assert dd.profile(103, 16, "gen", 4) == ({}, "none")


def test_apply_keeps_the_environment():
    env = {"MUSESPARK_OPEN_SWA_CL8": "0"}
    applied = dd.apply((10, 3), _mapping(4), _facts(64, 8192, False), env)
    assert env["MUSESPARK_OPEN_SWA_CL8"] == "0"
    assert "MUSESPARK_OPEN_SWA_CL8" not in applied
    assert env["MUSESPARK_OPEN_FMK_CONTEXT_FMHA"] == "1"
    assert dd.ROLE_ENV not in env  # agg: the lever table derives it


def test_apply_sets_the_inferred_role_and_f11():
    env = {}
    dd.apply((10, 3), _mapping(8), _facts(16, 128, True), env)
    assert env[dd.ROLE_ENV] == "gen"
    assert env[dd.F11_ENV] == "1"
    assert env["MUSESPARK_OPEN_TEP8_WIDE_WINDOW_GEOMETRY"] == "1"


def test_apply_f11_opt_out_keeps_aria():
    env = {dd.F11_ENV: "0"}
    applied = dd.apply((10, 3), _mapping(8), _facts(4, 32, True), env)
    assert env[dd.F11_ENV] == "0"
    assert not [
        k for k in applied if k.startswith(("MUSESPARK_OPEN_SM103_TEP8_", "MUSESPARK_OPEN_TEP8_"))
    ]


def test_apply_off():
    env = {dd.ENABLE_ENV: "0"}
    assert dd.apply((10, 3), _mapping(4), _facts(64, 8192, False), env) == {}
    assert env == {dd.ENABLE_ENV: "0"}


def test_apply_off_sm103():
    env = {}
    assert dd.apply((10, 7), _mapping(8), _facts(4, 32, True), env) == {}
    assert env == {}
