# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L1 / Q1 / S2 of ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` (CPU only): ``fuse_routing`` gives
the wide rows ``RouteSituFc12`` (FC12's sort tables and NVFP4 operand written by the routing kernel)
and refuses it without S2, the FP32 routing weights or the glue, or on another expert window."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _helix16_wide_cpu import modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

LOCAL = 112


def _fuse_namespace():
    host = SimpleNamespace(prepare_shared_carveout=lambda code, i, v: ("carved", code),
                           to_cute=lambda t: ("cute", tuple(t.shape)))  # fmt: skip
    ns = {
        "module": {"cutedsl.common.host": host}.__getitem__, "torch": torch,
        "replace": __import__("dataclasses").replace,
        "Int32": lambda v: ("i32", v), "EXPERTS_PER_RANK": LOCAL,
        "compile_route_fc12": lambda *a: ("code", a),
    }  # fmt: skip
    source_objects(modeling_path(), ("fuse_routing",), ns)
    return ns


def _fuse_moe(**over):
    from dataclasses import make_dataclass

    launch_cls = make_dataclass("Launch", ["code", "extra", "mailbox"])
    plans = {b: ("gemm", "logits", launch_cls("route", ("route32",), torch.zeros((1, 128, 4))), "views")
             for b in (72, 96, 128)}  # fmt: skip
    once = SimpleNamespace(plans=plans, scales="route32", device=torch.device("cpu"))
    tables = {
        (p, b): tuple(torch.zeros(1) for _ in range(6)) for p in (0, 1) for b in (72, 96, 128)
    }
    sorter = SimpleNamespace(rows=(72, 96, 128), local=LOCAL, offset=224, tables=tables)
    quant = tuple((torch.zeros((128, 1792)), torch.zeros((128, 224))) for _ in range(2))
    glue = SimpleNamespace(tickets=torch.zeros((2, 4), dtype=torch.int32), quant=quant)
    experts = SimpleNamespace(offset=224, bank={"fc31_in_scale": torch.ones(4)})
    moe = SimpleNamespace(sorter=sorter, glue=glue, fronted="event", experts=experts, layer=3,
                          route_tables={}, route_quant={})  # fmt: skip
    for k, v in over.items():
        setattr(moe if hasattr(moe, k) else once, k, v)
    weights = SimpleNamespace(shared=SimpleNamespace(gate_up=SimpleNamespace(weight=torch.zeros((4, 1)))),
                              renormalize=True)  # fmt: skip
    return moe, once, weights


def test_fuse_routing_takes_the_wide_rows_and_needs_s2_route32_and_the_glue():
    ns = _fuse_namespace()
    moe, once, weights = _fuse_moe()
    ns["fuse_routing"](moe, once, weights, 8)
    assert sorted(moe.route_tables) == sorted(moe.route_quant) == [72, 96, 128]
    assert moe.route_tables[96] is moe.sorter.tables[1, 96]  # the layer's parity
    code, extra = once.plans[96][2].code, once.plans[96][2].extra
    assert code[0] == "carved" and code[1][1][0] == 96  # the verify steps' carve-out, as FrontOnce
    assert extra[0] == "route32" and extra[1] == ("i32", 224) and extra[-1] == ("cute", (1,))
    assert len(extra) == 1 + 2 + 6 + 3
    for bad in (dict(fronted=None), dict(glue=None), dict(scales=None)):
        moe, once, weights = _fuse_moe(**bad)
        with pytest.raises(ValueError, match="S2"):
            ns["fuse_routing"](moe, once, weights, 8)
    moe, once, weights = _fuse_moe()
    moe.sorter.local = 111
    with pytest.raises(ValueError, match="window"):
        ns["fuse_routing"](moe, once, weights, 8)
