# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The high-cap row tables (`DP_HICAP_ENV`; unset, `route` and `fin` at dep8, nothing at dep4).

* The switch: unset takes the rank's default parts (dep8: `route`, `fin`; dep4 and no rank: none);
  0 off; 1 / all every part; a comma list of parts; another spelling (`release` too) raises.
* `route`: dep8's route front serves up to `_DP_HICAP_ROUTE_MAX_ROWS` gathered rows (dep4 keeps
  `MAX_ROWS`), the fast path's rows follow, the 64-expert window pins tile 64 past 1024 rows (held
  to the runner's listed tactic), and `DP_MOE_OWNK_ENV=all` covers the wider rows. The build
  compiles a row count past `MAX_ROWS` on the fast path only (never the two-token sort), skips it
  where the fast path is off, and drops it, alone, where its check fails.
* `fin`: the fused reduce-scatter's layouts end at one resident wave.
* `head`: the sharded verify head reaches its wider range; dep4's releasing gather keeps its own.
* Off: every table is the default dispatch's.

The ops are recorders; no kernel runs.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

# The build makes its table buffers on the device it is given (`device.type == "cuda"`).
_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="the build's buffers are on CUDA")


def _set(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(M.DP_HICAP_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_HICAP_ENV, value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, set()),
        ("", set()),
        ("0", set()),
        ("off", set()),
        ("1", {"route", "fin", "head"}),
        ("all", {"route", "fin", "head"}),
        ("route", {"route"}),
        ("Route, fin", {"route", "fin"}),
    ],
)
def test_parts(monkeypatch, value, expected):
    _set(monkeypatch, value)
    assert M._dp_hicap() == frozenset(expected)


@pytest.mark.parametrize(
    "value, dp_size, expected",
    [
        (None, 8, {"route", "fin"}),
        ("", 8, {"route", "fin"}),
        (None, 4, set()),
        (None, 1, set()),
        ("0", 8, set()),
        ("head", 8, {"head"}),  # a list replaces the default
        ("1", 4, {"route", "fin", "head"}),
    ],
)
def test_default_parts_by_layout(monkeypatch, value, dp_size, expected):
    _set(monkeypatch, value)
    assert M._dp_hicap(SimpleNamespace(dp_size=dp_size)) == frozenset(expected)


@pytest.mark.parametrize("value", ["2", "routes", "route,tile", ",", "release", "route,release"])
def test_parts_reject_another_spelling(monkeypatch, value):
    _set(monkeypatch, value)
    with pytest.raises(ValueError):
        M._dp_hicap()


@pytest.mark.parametrize(
    "dp_size, local, value, expected",
    [
        (8, 64, None, 1792),
        (8, 64, "0", 1024),
        (8, 64, "route", 1792),
        (8, 64, "fin,head", 1024),
        (4, 128, None, 1024),
        (4, 128, "1", 1024),  # dep4's caps 40-52 are 640-832 rows: inside MAX_ROWS
        (8, 128, "1", 1024),
    ],
)
def test_route_rows(monkeypatch, dp_size, local, value, expected):
    _set(monkeypatch, value)
    core = SimpleNamespace(dp_size=dp_size, local_experts=local)
    assert M._dp_hicap_route_rows(core) == expected


@pytest.mark.parametrize(
    "dp_size, local, rows, value, expected",
    [
        (8, 64, 1152, "0", 0),
        (8, 64, 1152, None, 1),
        (8, 64, 1152, "route", 1),
        (8, 64, 1792, "1", 1),
        (8, 64, 1824, "1", 0),
        (8, 64, 1024, None, 1),  # today's last row count, unchanged
        (4, 128, 1152, "1", 0),
    ],
)
def test_fast2_rows(monkeypatch, dp_size, local, rows, value, expected):
    monkeypatch.delenv(M.DP_ROUTE_FAST2_ENV, raising=False)
    _set(monkeypatch, value)
    core = SimpleNamespace(dp_size=dp_size, local_experts=local)
    assert M._route_front_fast2(core, rows) == expected


class _Runner:
    def __init__(self, valid):
        self.valid = valid

    def get_valid_configs(self, top_k, hidden, inter, num_experts, rows):
        return [list(t) for t in self.valid]


W64 = SimpleNamespace(dp_size=8, topk=10, hidden=2560, expert_inter=640, local_experts=64)


@pytest.mark.parametrize(
    "rows, valid, value, expected",
    [
        (1152, [(128, 4)], "0", (None, None)),  # off: no tactic past 1024 rows
        (1152, [(128, 4), (256, 2)], None, ((64, 5), (128, 4))),  # the dep8 default
        (1152, [(128, 4), (256, 2)], "route", ((64, 5), (128, 4))),
        (1792, [(64, 5), (128, 4)], "route", ((64, 5), None)),
        (1824, [(128, 4)], "route", (None, None)),
        (1024, [(128, 4)], "route", ((64, 5), (128, 4))),  # unchanged at 1024
    ],
)
def test_w64_tactic(monkeypatch, rows, valid, value, expected):
    _set(monkeypatch, value)
    assert M._route_front_w64_tactic(_Runner(valid), W64, rows) == expected


@pytest.mark.parametrize(
    "ownk, value, dp_size, expected",
    [
        ("all", None, None, (1, 1024)),
        ("all", None, 8, (1, 1792)),  # the dep8 default
        ("all", "0", 8, (1, 1024)),
        ("all", None, 4, (1, 1024)),
        ("all", "route", None, (1, 1792)),
        ("1", "route", 8, (513, 896)),  # the measured default range is not widened
    ],
)
def test_ownk_all_covers_the_wider_rows(monkeypatch, ownk, value, dp_size, expected):
    monkeypatch.setenv(M.DP_MOE_OWNK_ENV, ownk)
    _set(monkeypatch, value)
    core = None if dp_size is None else SimpleNamespace(dp_size=dp_size)
    assert M._dp_ownk_wanted(core) == expected


def _build_core(sizes):
    return SimpleNamespace(
        dp_size=8,
        tep_fast=False,
        one_rank=False,
        trunk_moe_backend="TRTLLM",
        num_experts=512,
        topk=10,
        hidden=2560,
        local_experts=64,
        expert_inter=640,
        expert_offset=0,
        num_layers=2,
        max_verify_rows=4,
        mapping=SimpleNamespace(tp_rank=0, tp_size=8),
        model_config=SimpleNamespace(cuda_graph_batch_sizes=sizes),
    )


def _fake_build(monkeypatch, bad_rows=()):
    """`_route_front_build` on CPU: a runner that lists tile 128 past 1024 rows and tile 64 below,
    a recording compile that takes every shape, a check that fails at `bad_rows`."""
    compiles = []
    monkeypatch.delenv(M.DP_ROUTE_FRONT_ENV, raising=False)
    monkeypatch.delenv(M.DP_ROUTE_FRONT_EARLY_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    monkeypatch.setattr(M, "get_env_enable_pdl", lambda: True)

    class Runner:
        def get_valid_configs(self, top_k, hidden, inter, num_experts, rows):
            return [[128, 4]] if rows > 1024 else [[64, 5], [32, 32]]

    monkeypatch.setattr(M, "_moe_runner", Runner)

    def compile_(
        rows, hidden, stride, local, tile, fast2=0, route_only=False, ids_in=False, early=False
    ):
        compiles.append((rows, int(fast2), early))
        return True

    monkeypatch.setattr(M, "_route_front_compile", compile_)
    monkeypatch.setattr(
        M,
        "_route_front_check",
        lambda core, st, L, i, rows, gen: "routing" if rows in bad_rows else "",
    )
    # the real device context and synchronize stay (the buffers are on CUDA)
    layers = [SimpleNamespace(fc1_w=torch.zeros(1)), SimpleNamespace(fc1_w=torch.zeros(2))]
    return compiles, layers


@_CUDA
@pytest.mark.parametrize("value", ["route", None])
def test_build_compiles_wide_rows_on_the_fast_path_only(monkeypatch, value):
    compiles, layers = _fake_build(monkeypatch)
    monkeypatch.delenv(M.DP_ROUTE_FAST2_ENV, raising=False)
    _set(monkeypatch, value)
    st = M._route_front_build(_build_core([28, 32, 36, 40, 56, 60]), layers, torch.device("cuda"))
    assert sorted(st.tactics) == [896, 1024, 1152, 1280, 1792]
    assert st.tactics[1152] == (64, 5) and st.refs[1152] == (128, 4)
    assert {r: st.fast2[r] for r in (1152, 1280, 1792)} == {1152: 1, 1280: 1, 1792: 1}
    assert {1152, 1280, 1792} <= st.early
    # never the two-token sort past MAX_ROWS
    assert not [c for c in compiles if c[0] > 1024 and c[1] == 0]


@_CUDA
def test_build_skips_wide_rows_without_the_fast_path(monkeypatch):
    compiles, layers = _fake_build(monkeypatch)
    monkeypatch.setenv(M.DP_ROUTE_FAST2_ENV, "0")
    _set(monkeypatch, "route")
    st = M._route_front_build(_build_core([32, 36, 40]), layers, torch.device("cuda"))
    assert sorted(st.tactics) == [1024]
    assert not [c for c in compiles if c[0] > 1024]


@_CUDA
def test_build_drops_a_failing_wide_row_count_alone(monkeypatch):
    compiles, layers = _fake_build(monkeypatch, bad_rows=(1280,))
    monkeypatch.delenv(M.DP_ROUTE_FAST2_ENV, raising=False)
    _set(monkeypatch, "route")
    st = M._route_front_build(_build_core([32, 36, 40]), layers, torch.device("cuda"))
    assert sorted(st.tactics) == [1024, 1152]
    assert 1280 not in st.fast2 and 1280 not in st.early and 1280 not in st.flats


@_CUDA
def test_build_off_is_the_default_dispatch_rows(monkeypatch):
    compiles, layers = _fake_build(monkeypatch)
    monkeypatch.delenv(M.DP_ROUTE_FAST2_ENV, raising=False)
    _set(monkeypatch, "0")
    st = M._route_front_build(_build_core([32, 36, 40]), layers, torch.device("cuda"))
    assert sorted(st.tactics) == [1024]
    assert max(c[0] for c in compiles) == 1024


@pytest.mark.parametrize(
    "value, dp_size, expected",
    [
        ("0", 8, ((1, 304), (2, 608))),
        (None, 8, ((1, 304), (2, 304))),  # the dep8 default
        (None, 4, ((1, 192), (2, 608))),
        ("fin", 8, ((1, 304), (2, 304))),
        ("fin", 4, ((1, 192), (2, 304))),
        ("route", 4, ((1, 192), (2, 608))),
    ],
)
def test_fin_layouts_end_at_one_resident_wave(monkeypatch, value, dp_size, expected):
    """The layouts `_dp_rs_inject_build` hands the fused launch: with `fin` past 152 rows a rank
    every stream of a row shares one CTA, so the fused finalize stays one resident wave."""
    from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe import (
        dep_lamport_rs_inject as rsi,
    )

    monkeypatch.delenv(M.DP_RS_INJECT_ENV, raising=False)
    monkeypatch.delenv(M.DP_RS_INJECT_LAYOUTS_ENV, raising=False)
    _set(monkeypatch, value)
    seen = {}

    class _Stop(Exception):
        pass

    def build(trip, hc_count, layouts, fin, topk, pdl):
        seen["layouts"] = layouts
        raise _Stop

    monkeypatch.setattr(M._rs_inject_op, "build", build)
    monkeypatch.setattr(M._rs_inject_op, "supported", lambda hidden, hc: True)
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: True)
    monkeypatch.setattr(M, "_graph_rows", lambda core, lo, hi: [144, 160])
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())

    class _Dist:
        def tp_allgather(self, x):
            return [x]

    import tensorrt_llm._torch.distributed.communicator as comm

    monkeypatch.setattr(comm.Distributed, "get", staticmethod(lambda mapping: _Dist()))
    trip = SimpleNamespace(max_rows=512, world=dp_size)
    core = SimpleNamespace(
        _lamport_trip=trip,
        dp_size=dp_size,
        hidden=2560,
        hc_count=4,
        _route_front=None,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[]),
        mtp_enabled=False,
        mapping=None,
        dp_rank=0,
        topk=10,
    )
    # the recorded build raises: every rank reports it and the switch stays off
    assert M._dp_rs_inject_build(core, torch.device("cpu")) is None
    assert seen["layouts"] == expected
    if value == "fin" or (value is None and dp_size == 8):
        # 160 rows a rank: one CTA a row (4 streams), 160 CTAs, resident on 152 SMs
        assert rsi.streams_per_cta(160, 4, expected) == 4
        assert rsi.fin_resident(160, 4, expected, 152)
        assert rsi.streams_per_cta(144, 4, expected) == 2  # unchanged up to 152 rows
    else:
        assert not rsi.fin_resident(160, 4, expected, 152)


def test_head_range_and_the_releasing_gather_rows(monkeypatch):
    assert M._DP_HICAP_HEAD_VERIFY == (1, 56)
    assert not hasattr(M, "_DP_HICAP_RELEASE_ROWS")
    assert M._DP_ROUTER_FRONT_RELEASE_ROWS == {4: (16, 384), 8: (32, 768)}
    assert M._DP_HEAD_SHARD_REQUESTS["verify"] == (1, 32)


class _Nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize(
    "value, dp_size, expected",
    [
        # dep4, four rows a request: graph sizes 8 / 40 / 64 -> 128 / 640 / 1024 gathered rows
        (None, 4, {128}),
        ("1", 4, {128}),
        # dep8 keeps its own range (32..768: 256 only of 256 / 1280 / 2048)
        (None, 8, {256}),
        ("1", 8, {256}),
    ],
)
def test_releasing_gather_rows_are_not_widened(monkeypatch, value, dp_size, expected):
    """The releasing gather's rows `_dp_router_front_build` engages are its own range whatever
    `DP_HICAP_ENV` says (the widening it once offered was not taken)."""
    from tensorrt_llm._torch.distributed import communicator

    _set(monkeypatch, value)
    monkeypatch.setenv(M.DP_ROUTER_FRONT_ENV, "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))
    monkeypatch.setattr(torch.cuda, "device", lambda device: _Nothing())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(M, "MidWeight", lambda w: ("mw", w.data_ptr()))
    monkeypatch.setattr(M, "mid_rows_precompile", lambda mw, lo, hi: None)
    monkeypatch.setattr(
        communicator.Distributed,
        "get",
        staticmethod(lambda m: SimpleNamespace(tp_allgather=lambda v: [v])),
    )
    core = SimpleNamespace(
        dp_size=dp_size,
        dp_rank=0,
        max_verify_rows=4,
        mapping=None,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[8, 40, 64]),
        _lamport_trip=SimpleNamespace(releasing_gather_fn=lambda: None),
    )
    layers = [SimpleNamespace(router=torch.zeros(4, 2))]
    front = M._dp_router_front_build(core, layers, torch.device("cuda"))
    assert front is not None and front.release_rows == expected and not front.rows


def _head_shard_fakes():
    """`test_dp_head_shard`'s build fakes (the shell, the group, the collectives)."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_dp_head_shard.py")
    spec = importlib.util.spec_from_file_location("_hicap_head_shard_fakes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "value, verify_requests, max_rows",
    [
        (None, (1, 2, 4, 8, 16, 32), 128),
        ("head", (1, 2, 4, 8, 16, 32, 40, 56), 224),
        ("route,fin", (1, 2, 4, 8, 16, 32), 128),
    ],
)
def test_head_verify_rows_reach_the_build(monkeypatch, value, verify_requests, max_rows):
    """`head` widens the sharded verify head's rows `_dp_head_shard_build` serves to 56 requests a
    rank (four rows a request) and sizes the group's buffers for them; the draft head is unchanged."""
    fakes = _head_shard_fakes()
    _set(monkeypatch, value)
    monkeypatch.setenv(M.DP_HEAD_SHARD_ENV, "1")
    monkeypatch.setenv(M.DP_HEAD_SHARD_GROUP_ENV, "node")
    created = fakes._patch_build(monkeypatch)
    shell = fakes._build_shell(sizes=(1, 2, 4, 8, 16, 32, 40, 56, 64, 80))
    shard = M._dp_head_shard_build(shell, torch.device("cpu"))
    assert shard.verify_rows == {4 * b for b in verify_requests}
    assert shard.draft_rows == {1, 2, 4, 8, 16, 32, 40, 56, 64}
    assert created == [(4, max_rows, "ipc")]
