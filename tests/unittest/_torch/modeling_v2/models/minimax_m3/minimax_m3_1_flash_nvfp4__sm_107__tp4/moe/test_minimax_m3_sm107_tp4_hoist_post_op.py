# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4WC_HOIST: trtllm::minimax_m3_sm107tp4_hoist_post on one sm_107 GPU (no peers): the op and its fake are registered,
the rows-0 idle launch (the reference build-time module load) touches nothing, and every argument check the peer
extension makes still refuses."""

import struct

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
    native_ops,
)

H = 6144
EMPTY = -(2**31)
FMT = "<8qq5i4x"


def _sm107():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


# The skipif below reads torch.ops.trtllm: load the TRT-LLM op library first (without this import every case skipped,
# 15/15, in build_cut26_v1).
import tensorrt_llm  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not _sm107() or not hasattr(native_ops, "minimax_m3_sm107tp4_hoist_post"),
    reason="sm_107 with trtllm::minimax_m3_sm107tp4_hoist_post",
)


def _pp(world=4, maxt=8, cluster=4, ndst=None, rank=0):
    ex = [
        torch.full((4 * maxt * world * (H // 2),), EMPTY, dtype=torch.int32, device="cuda")
        for _ in range(world)
    ]
    flags = torch.zeros(maxt * 8, dtype=torch.int32, device="cuda")
    peers = [t.data_ptr() for t in ex] + [0] * (8 - world)
    b = struct.pack(
        FMT, *peers, flags.data_ptr(), rank, world, maxt, cluster, world if ndst is None else ndst
    )
    return torch.tensor(list(b), dtype=torch.uint8), (ex, flags)


def _bufs():
    bf = dict(dtype=torch.bfloat16, device="cuda")
    return dict(
        residual=torch.randn(8, H, **bf),
        gamma=torch.randn(H, **bf),
        out_res=torch.full((8, H), 7.0, **bf),
        out_norm=torch.full((8, H), 7.0, **bf),
    )


def _call(pp, b, rows=0, **kw):
    a = dict(
        out_q=None,
        out_sf=None,
        cl=8,
        rx_bases=[],
        route=None,
        rctr=None,
        bias=None,
        tab=None,
        delay_ns=0,
        lg_dbg=None,
        tdbg=None,
        repl=16,
        npart=1,
        rxc=None,
        deadline_ns=2_000_000_000,
        hang_dbg=0,
        roll=0,
        x4_q=None,
        x4_sf=None,
        x4_sfsc=None,
        prr=1,
        rx2=1,
        pt=0,
        x4_pts=None,
        pwait=0,
    )
    a.update(kw)
    native_ops.minimax_m3_sm107tp4_hoist_post(
        pp,
        rows,
        b["residual"],
        b["gamma"],
        1e-6,
        1,
        b["out_norm"],
        b["out_res"],
        a["out_q"],
        a["out_sf"],
        a["cl"],
        a["rx_bases"],
        a["route"],
        a["rctr"],
        a["bias"],
        a["tab"],
        a["delay_ns"],
        a["lg_dbg"],
        a["tdbg"],
        a["repl"],
        a["npart"],
        a["rxc"],
        a["deadline_ns"],
        a["hang_dbg"],
        a["roll"],
        a["x4_q"],
        a["x4_sf"],
        a["x4_sfsc"],
        a["prr"],
        a["rx2"],
        a["pt"],
        a["x4_pts"],
        a["pwait"],
    )


def test_fake_registered():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        fake_ops,
    )

    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        f"{fake_ops._NS}::minimax_m3_sm107tp4_hoist_post", "Meta"
    )


@pytest.mark.parametrize("cl", [4, 8, 16])
@pytest.mark.parametrize("x4", [0, 2])
def test_rows0_idle_launch_touches_nothing(cl, x4):
    pp, keep = _pp()
    b = _bufs()
    kw = dict(cl=cl)
    if x4:
        kw.update(
            x4_q=torch.zeros(8, H // 2, dtype=torch.uint8, device="cuda"),
            x4_sf=torch.zeros(8, H // 16, dtype=torch.uint8, device="cuda"),
            x4_sfsc=torch.ones(1, device="cuda"),
            pt=1,
            x4_pts=torch.zeros(8, device="cuda"),
        )
    _call(pp, b, rows=0, **kw)
    torch.cuda.synchronize()
    assert bool((b["out_res"] == 7.0).all()) and bool((b["out_norm"] == 7.0).all())
    assert all(bool((e == EMPTY).all()) for e in keep[0]) and int(keep[1].abs().sum()) == 0


@pytest.mark.parametrize(
    "bad",
    [
        "cluster8",
        "world2",
        "prr0",
        "pwait_cl4",
        "x4_without_sf",
        "route_without_bias",
        "rows_over_maxt",
        "pp_on_gpu",
    ],
)
def test_peer_checks_refuse(bad):
    pp, keep = _pp(cluster=8 if bad == "cluster8" else 4, world=2 if bad == "world2" else 4)
    b = _bufs()
    kw, rows = {}, 0
    if bad == "prr0":
        kw["prr"] = 0
    elif bad == "pwait_cl4":
        kw.update(pwait=1, cl=4)
    elif bad == "x4_without_sf":
        kw["x4_q"] = torch.zeros(8, H // 2, dtype=torch.uint8, device="cuda")
    elif bad == "route_without_bias":
        kw.update(
            route=torch.zeros(16 * 8 * 16, dtype=torch.int64, device="cuda"),
            rctr=torch.zeros(8 * 32, dtype=torch.int32, device="cuda"),
        )
    elif bad == "rows_over_maxt":
        rows = 9
    elif bad == "pp_on_gpu":
        pp = pp.cuda()
    with pytest.raises(RuntimeError):
        _call(pp, b, rows=rows, **kw)
