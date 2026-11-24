# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The front above 64 rows (``FrontOnce``; CPU only, on the sources): ``FRONT_ONCE`` covers 65-128
rows; ``FrontOnce`` compiles per step one front GEMM and MR8's routing kernel with the 64-row call's
options (carve-out on verify steps only) and launches both on the current stream; rows that are not
whole requests above 64, or a short mailbox, are refused.
"""

from __future__ import annotations

import functools
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from _helix16_wide_cpu import family_dir, literal, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

VENDOR = "kernels/vendor/vllm_mr8/specialized_decode/cutedsl"
# The packed front rows at TP16 (56 router rows padded to 64, then the 224 latent-down rows).
N, VALID = 288, 56
WIDER = tuple(range(72, 129, 8))


@pytest.fixture(scope="module")
def policy():
    path = family_dir() / VENDOR / "common/policy.py"
    spec = importlib.util.spec_from_file_location("_k31_helix16_wide_policy_once", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def front(policy):
    """MR8's front tables and ``front_geometry`` (pure Python), as extracted from front.py."""
    names = ("FRONT", "FRONT_WIDE", "front_table", "front_ks", "front_geometry")
    ns = {"functools": functools, "batch_table": policy.batch_table}
    return SimpleNamespace(**{
        k: v for k, v in source_objects(family_dir() / VENDOR / "moe/front.py", names, ns).items()
        if k in names
    })  # fmt: skip


@pytest.fixture(scope="module")
def once_table(policy):
    return policy.batch_table(literal(modeling_path(), "FRONT_ONCE"))


def test_the_table_covers_the_wider_steps(once_table):
    assert sorted(set(once_table) & set(range(1, 200))) == list(range(65, 129))
    assert {once_table[b][:3] + once_table[b][4:] for b in WIDER} == {(16, 7, 16, 2)}


# --- FrontOnce on stand-ins -------------------------------------------------------------------


class _Recorder:
    def __init__(self, name, log, result=None):
        self.name, self.log, self.result = name, log, result

    def __call__(self, *args, **kwargs):
        self.log.append((self.name, args, kwargs))
        return self.result if self.result is not None else (self.name, len(self.log))


def _front_once(front, policy, log, *, rows=WIDER, q=8, cap=128):
    compiled_route = _Recorder("route", log)
    front_module = SimpleNamespace(
        WIDTH=3584 + 2 * 896,
        front_geometry=front.front_geometry,
        compile_route=lambda *a, **k: (log.append(("compile_route", a, k)), compiled_route)[1],
    )
    host = SimpleNamespace(
        to_cute=lambda t: ("cute", t.data_ptr(), tuple(t.shape)),
        stream=lambda device: "stream",
        prepare_shared_carveout=lambda code, dev, value: ("carved", code, value),
    )
    modules = {
        "cutedsl.moe.front": front_module,
        "cutedsl.common.host": host,
        "cutedsl.common.policy": policy,
    }
    ns = {
        "module": modules.__getitem__,
        "torch": torch,
        "Int32": lambda v: ("i32", v),
        "Int64": lambda v: ("i64", v),
        "Float32": lambda v: ("f32", v),
        "MoeRouteLaunch": lambda **kw: SimpleNamespace(**kw),
        "moe_route": lambda launch, s: log.append(("moe_route", launch, s)),
        "compile_front_once": lambda *a: (log.append(("compile_front_once", a)), _Gemm(log))[1],
        "Sequence": tuple,
    }
    names = ("FRONT_ROWS", "FRONT_ONCE", "_MOE_LATENT", "_MOE_EXPERTS", "_MOE_SITU", "FrontOnce")
    names += ("WIDE_FRONTS", "FRONT_TMEM_N")
    source_objects(modeling_path(), names, ns)
    weights = SimpleNamespace(
        front=torch.zeros((N, 8), dtype=torch.bfloat16),
        bias=torch.zeros(896),
        scale=2.827,
        renormalize=True,
        shared=SimpleNamespace(gate_up=SimpleNamespace(weight=torch.zeros((768, 1)))),
    )
    box = SimpleNamespace(buffer=torch.zeros((1, cap, 3584 + 2 * 896), dtype=torch.bfloat16))
    box.mc, box.rank = 0x7000, 5
    shapes = ((cap, 896), (cap, 768), (cap, 16), (cap, 16), (cap, 384), (cap, 7168), (cap, 3584))
    storage = tuple(torch.zeros(s) for s in shapes)
    once = ns["FrontOnce"](weights, box=box, storage=storage, rows=rows, q=q)
    return once, weights, box, storage, ns


class _Gemm:
    def __init__(self, log):
        self.log = log

    def __call__(self, *args):
        self.log.append(("gemm", args))


def test_front_once_builds_and_launches_the_steps(front, policy, once_table):
    log = []
    once, weights, box, storage, _ = _front_once(front, policy, log)
    assert sorted(once.plans) == list(WIDER)
    gemms = [entry for entry in log if entry[0] == "compile_front_once"]
    routes = [entry for entry in log if entry[0] == "compile_route"]
    assert len(gemms) == len(routes) == len(WIDER)
    for (_, args), (_, rargs, rkwargs), rows in zip(gemms, routes, WIDER):
        ks, kb, tg, rt, ch = once_table[rows]
        geometry = front.front_geometry(rows, N, ks, kb=kb, tg=tg, rt=rt, ch=ch, wide=True)
        assert args == (N, N - 224, VALID, rows, None, ks, geometry, 128)
        # The 64-row call's routing (``SharedFront``'s keys at 64 rows: polled, no plan, lean by
        # default), with its copy barrier and the wide mailbox's rows.
        assert rargs == (rows, 384, True, (True, True, False), None, True, False, None)
        assert rkwargs == {"sync_copy": True, "box_rows": 128}
        assert once.label(rows) == "once:" + ".".join(map(str, once_table[rows]))
    # The mailbox and this rank's slots: router column 5 * 56, latent column 5 * 224.
    assert once.box[1:] == (("i64", 0x7000), ("i32", 5 * 56), ("i32", 5 * 224))
    for rows in WIDER:
        gemm, logits, launch, views = once.plans[rows]
        assert launch.code[0] == "carved" and launch.code[2] == 100
        assert launch.rows == rows and launch.mailbox is box.buffer and launch.plan is None
        assert launch.extra == () and launch.routes is None and launch.image is False
        for view, whole in zip(views, storage):
            assert view.shape[0] == rows and view.data_ptr() == whole.data_ptr()
        assert launch.args[4] == once.box[0] and launch.args[8:] == (
            ("f32", 2.827), ("f32", 4.0), ("f32", 25.0),
        )  # fmt: skip
        log.clear()
        x = torch.zeros((rows, 7168), dtype=torch.bfloat16)
        out = once.route(x)
        assert [entry[0] for entry in log] == ["gemm", "moe_route"]
        assert log[0][1] == (("cute", x.data_ptr(), (rows, 7168)), once.packed, *once.box, logits,
                             "stream")  # fmt: skip
        assert log[1][1] is launch and log[1][2] == "stream"
        assert [t.data_ptr() for t in out] == [storage[i].data_ptr() for i in (2, 3, 6)]


def test_front_once_without_verify_keeps_mr8s_carveout_off(front, policy):
    log = []
    once, *_ = _front_once(front, policy, log, rows=(72,), q=1)
    assert isinstance(once.plans[72][2].code, _Recorder)


@pytest.mark.parametrize(
    "rows, cap, match",
    [((64,), 136, None), ((68,), 136, None), ((136,), 136, None), ((72, 129), 136, None),
     ((128,), 96, "mailbox")],
)  # fmt: skip
def test_front_once_refuses_other_rows(front, policy, rows, cap, match):
    with pytest.raises(ValueError, match=match):
        _front_once(front, policy, [], rows=rows, cap=cap)
