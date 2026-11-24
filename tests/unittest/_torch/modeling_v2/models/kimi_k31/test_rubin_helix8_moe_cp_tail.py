# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's column-parallel MoE tail (``moe_cp_tail``, a generation constant) on the CPU.

With the switch on, a decode step whose rows fall in ``moe_cp_tail_bands`` runs the tail: FC12,
the shared experts' partial pushed into the tail's Lamport call (``rubin_lamport_push``, one-shot,
PDL; on a CUDA graph's steps on the aux stream between the shared fork and join), the latent
all-reduce row on its own workspace, then this rank's 896 rows of latent_up pushed into the call's
all-gather rows (``rubin_ull_cp_tail``). It returns ``_MoeTail``; the next mix finishes the call:

* ``moe_norm`` (and ``moe_fold_norm``, which hands a tail to it): the PDL mix with the finish
  (one-shot, all-gather rows, the round-to-nearest add, ``moe_tail_early_poll``) where it serves the
  rows, else the finish kernel and the add, then the mix;
* ``moe_norm_mx`` (a KDA layer under ``attn_res_mx_pdl``): the PDL mix with the finish that also
  stores the MXFP8 form of one 128-row tile in the QKVG GEMM's own rule (``_kda_mx_exact``);
* ``moe_add`` (a tapped layer, the output mix without the fold): the finish kernel, then the add;
* ``_mix_in`` / ``_mix_out`` hand a tail to those slots as they do a ``_MoeOut``.

Rows outside the bands, and every step with the switch off, run today's replicated tail. The
bands parse as ``lo-hi`` pairs, ascending and disjoint, up to 72 rows. latent_up's row counts have
one owner each: the import refuses a tail band that shares a row count with the resident
latent_up's bands (``moe_latent_up_resident``) or with gemm_pins_moe's latent_up pins, and any two
of those with each other. The product's functions run from modeling.py over the recording
stand-ins of ``test_rubin_helix8_moe_front.py``.
"""

from __future__ import annotations
import __future__

import ast
import re
from types import MappingProxyType, SimpleNamespace
from typing import Any, NamedTuple, Optional

import pytest
import test_rubin_helix8_moe_front as F  # noqa: E402 -- the block's recording stand-ins
import torch

pytestmark = pytest.mark.cpu_only

_MODELING, _README = F._MODELING, F._README
_TAIL_BLOCK = (
    *F._BLOCK,
    "_MoeTail",
    "_moe_cp_tail",
    "_moe_tp_gen_tail",
    "_moe_tp_gen_streams_tail",
)
_SLOT_NAMES = (
    "_TARGET",
    "_ATTN_RES_OP_MAX_TOKENS",
    "_FUSED_MAX_ROWS",
    "_PERSISTENT_MAX_CANDIDATES",
    "_X_LEVERS",
    "_X_MAX_ROWS",
    "_MX_TILE_ROWS",
    "_KDA_MX_PDL_LINE",
    "_MoeOut",
    "_MoeTail",
    "_DensePartial",
    "_MixW",
    "_x_levers",
    "_mix_slots",
    "_capture",
    "_mix_in",
    "_mix_out",
)
_WORLD, _RANK, _C0, _N = 8, 3, 3 * 896, 896


def _tree(**switches) -> ast.Module:
    return F._tree(**switches)


def _constants(tree: ast.Module) -> dict:
    """``F._constants`` with the band parser defined first, so the bands evaluate with the switch
    on as well."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    (parser,) = F._definitions(tree, ("_moe_row_bands",))
    exec(compile(ast.fix_missing_locations(ast.Module([parser], [])), str(_MODELING), "exec"), ns)
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        names = [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
        if not names or not isinstance(target, (ast.Name, ast.Tuple)):
            continue
        try:
            value = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
        if isinstance(target, ast.Name):
            ns[target.id] = value
        else:
            ns.update(zip(names, value))
    return ns


def _block(tree: ast.Module, namespace: dict, names=_TAIL_BLOCK) -> dict:
    module = ast.fix_missing_locations(ast.Module(F._definitions(tree, names), []))
    ns = {**_constants(tree), "NamedTuple": NamedTuple, "Any": Any, "Optional": Optional}
    ns.update(namespace)
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


def _tail_entries(plan: F._Plan, seen: dict) -> dict:
    def rubin_lamport_push(ws, data, one_shot, use_pdl):
        seen["push"] = dict(ws=ws, data=data, flags=(one_shot, use_pdl))
        plan.kernel("lamport_push", (data,))

    def rubin_ull_cp_tail(latent, w_rows, ws, tactic, *, c0):
        seen["cp_tail"] = dict(latent=latent, w=w_rows, ws=ws, tactic=tactic, c0=c0)
        plan.kernel("cp_tail", (latent, w_rows))

    return dict(rubin_lamport_push=rubin_lamport_push, rubin_ull_cp_tail=rubin_ull_cp_tail)


def _rt(plan: F._Plan) -> SimpleNamespace:
    rt = F._rt(plan)
    rt.ull_tail = SimpleNamespace(
        ws="tail_ws",
        c0=_C0,
        n=_N,
        tactics=tuple(f"tactic_{m}" for m in range(73)),
        cpp_one_shot=True,
        chunk=8,
    )
    return rt


def _run(rows: int, cuda_graph: bool = True, on: bool = True, tree=None):
    plan, seen = F._Plan(), {}
    log = F._Logger(plan)
    tree = tree or (_tree(moe_cp_tail=True) if on else _tree())
    block = _block(
        tree,
        {
            "torch": F._Torch(plan),
            "logger": log,
            **F._entries(plan),
            **_tail_entries(plan, seen),
        },
    )
    layer, rt = F._layer(False), _rt(plan)
    x = torch.empty(rows, F._HIDDEN, dtype=torch.bfloat16)
    with F._Record(plan):
        out = block["latent_moe"](layer, F._LAYER, x, SimpleNamespace(cuda_graph=cuda_graph), rt)
    return plan, log, block, seen, layer, rt, out


# --- the constants ---


def test_the_tail_is_a_generation_constant_on_with_its_bands_and_poll():
    """On in the source (``_tree`` turns it off unless asked: ``test_rubin_helix8_moe_front.py``'s
    switch default); the README states the default."""
    gen = _constants(ast.parse(_MODELING.read_text()))["_CONSTANTS"]["generation"]
    assert gen["moe_cp_tail"] is True
    assert "`moe_cp_tail` (a helix8 generation constant, on)" in _README.read_text()
    assert gen["moe_cp_tail_bands"] == "1-60" and gen["moe_tail_early_poll"] is True
    details = _constants(_tree())["_DETAILS"]
    assert details["moe_cp_tail_bands"] == details["moe_tail_early_poll"] == "moe_cp_tail"
    off, on = _constants(_tree()), _constants(_tree(moe_cp_tail=True))
    assert (off["_MOE_CP_TAIL_BANDS"], off["_MOE_CP_TAIL_MAX_M"]) == ((), 0)
    # The tail's workspace and push GEMM tactics are sized to the bands' largest row count.
    assert (on["_MOE_CP_TAIL_BANDS"], on["_MOE_CP_TAIL_MAX_M"]) == (((1, 60),), 60)
    wide = _constants(_tail_tree("1-72"))
    assert (wide["_MOE_CP_TAIL_BANDS"], wide["_MOE_CP_TAIL_MAX_M"]) == (((1, 72),), 72)
    line = on["_MOE_CP_TAIL_LINE"]
    assert line == (
        "Kimi K3.1 Rubin MoE tail: column-parallel latent_up pushed into the shared experts' "
        "Lamport call (moe_cp_tail)"
    )
    assert f"\n{line}\n" in _README.read_text()


def test_row_bands_parse_ascending_disjoint_and_in_range():
    parse = _constants(_tree())["_moe_row_bands"]
    assert parse("1-48,61-72", 72) == ((1, 48), (61, 72))
    assert parse(" 6-6 , 12-24", 72) == ((6, 6), (12, 24))
    assert parse("", 72) == ()
    for bad in ("48-1", "1-48,40-50", "0-3", "1-73", "a-b", "12", "1-48,48-60"):
        with pytest.raises(ValueError):
            parse(bad, 72)


def _source(**values) -> str:
    """modeling.py's source with the generation constants ``values`` over the source's (string
    values as quoted constants)."""
    text = _MODELING.read_text()
    for name, value in values.items():
        rendered = f'"{value}"' if isinstance(value, str) else repr(value)
        # The generation table's line (16 spaces in), not a _DETAILS entry of the same key.
        text, hits = re.subn(rf'(?m)^( {{16}}"{name}": )[^,\n]+,$', rf"\g<1>{rendered},", text)
        assert hits == 1, (name, hits)
    return text


def _tail_tree(bands: str) -> ast.Module:
    """modeling.py's tree with the tail on over ``bands`` in place of the source's bands."""
    return ast.parse(_source(moe_cp_tail=True, moe_cp_tail_bands=bands), filename=str(_MODELING))


def _owner_check(**values) -> None:
    """Run modeling.py's import-time owner check of latent_up's row counts with the generation
    constants ``values`` over the source's (``_source``)."""
    tree = ast.parse(_source(**values))
    ns = _constants(tree)
    (overlaps,) = F._definitions(tree, ("_moe_band_overlaps",))
    (check,) = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and "_MOE_LATENT_UP_OWNERS" in ast.unparse(n.test)
    ]
    run = ast.fix_missing_locations(ast.Module([overlaps, check], []))
    exec(compile(run, str(_MODELING), "exec"), ns)


_PINS_ON = dict(gemm_pins_moe=True, gemm_pins_mla=True, gemm_pins_dense=True, gemm_pins_draft=True)


@pytest.mark.parametrize(
    "values",
    [
        dict(moe_cp_tail=True, moe_cp_tail_bands="1-60", gemm_pins_a_class=False, **_PINS_ON),
        dict(moe_cp_tail=True, moe_cp_tail_bands="1-72", gemm_pins_a_class=False, **_PINS_ON),
        dict(moe_cp_tail=True, moe_cp_tail_bands="1-30", gemm_pins_a_class=False, **_PINS_ON),
        dict(
            moe_cp_tail=True,
            moe_cp_tail_bands="1-30",
            moe_latent_up_resident=True,
            moe_latent_up_resident_bands="31-48",
            **_PINS_ON,
        ),
        dict(moe_cp_tail=True, moe_cp_tail_bands="1-30,49-60", gemm_pins_moe=False),
        dict(
            moe_cp_tail=False,
            moe_latent_up_resident=True,
            moe_latent_up_resident_bands="1-48",
            gemm_pins_moe=True,
        ),
    ],
    ids=[
        "shipped",
        "bands_1_72",
        "batch1",
        "resident_31_48",
        "tail_without_pins",
        "resident_without_tail",
    ],
)
def test_latent_up_rows_with_one_owner_each_import(values):
    _owner_check(**values)


def test_the_shipped_owners_leave_rows_past_the_bands_to_cublas():
    """With the source's constants latent_up's owners are the tail at 1-60 rows and gemm_pins_moe's
    144-row pin: the 54 / 60-row pins stay out of the table (the tail owns those rows), and 61-72
    rows (b11 / b12 at k5) have no owner, so they run cuBLAS's latent_up after the latent
    all-reduce."""
    ns = _constants(ast.parse(_MODELING.read_text()))
    owners = dict(ns["_MOE_LATENT_UP_OWNERS"])
    assert owners == {
        "moe_cp_tail": ((1, 60),),
        "moe_latent_up_resident": (),
        "gemm_pins_moe": ((144, 144),),
    }
    for rows in range(61, 73):
        assert not any(lo <= rows <= hi for bands in owners.values() for lo, hi in bands), rows


@pytest.mark.parametrize(
    "values, rows",
    [
        (
            dict(
                moe_cp_tail=True,
                moe_cp_tail_bands="1-30",
                moe_latent_up_resident=True,
                moe_latent_up_resident_bands="1-60",
                gemm_pins_moe=False,
            ),
            "((1, 30),) of moe_cp_tail and moe_latent_up_resident",
        ),
        (
            dict(
                moe_cp_tail=False,
                moe_latent_up_resident=True,
                moe_latent_up_resident_bands="1-30",
                gemm_pins_moe=True,
                gemm_pins_a_class=True,
            ),
            "((12, 12),) of moe_latent_up_resident and gemm_pins_moe",
        ),
        (
            dict(
                moe_cp_tail=True,
                moe_cp_tail_bands="1-72",
                gemm_pins_moe=True,
                gemm_pins_a_class=True,
            ),
            "((12, 12),) of moe_cp_tail and gemm_pins_moe",
        ),
    ],
    ids=["tail_and_resident", "resident_and_a_class_pin", "tail_and_a_class_pin"],
)
def test_latent_up_rows_with_two_owners_refuse_to_import(values, rows):
    with pytest.raises(NotImplementedError, match="latent_up rows with two owners") as err:
        _owner_check(**values)
    assert rows in str(err.value), str(err.value)


@pytest.mark.parametrize(
    "rows, served, wide",
    [
        (1, True, True),
        (6, True, True),
        (24, True, True),
        (30, True, True),
        (31, True, True),
        (36, True, True),
        (54, True, True),
        (60, True, True),
        (61, False, True),
        (66, False, True),
        (72, False, True),
        (78, False, False),
        (144, False, False),
    ],
)
def test_the_tail_serves_exactly_its_bands(rows, served, wide):
    """``served``: the source's bands; ``wide``: bands 1-72, the push GEMM's widest."""
    names = ("_moe_in_bands", "_moe_cp_tail_serves")
    on = _block(_tree(moe_cp_tail=True), {}, names)
    on_wide = _block(_tail_tree("1-72"), {}, names)
    off = _block(_tree(), {}, names)
    assert on["_moe_cp_tail_serves"](rows) is served
    assert on_wide["_moe_cp_tail_serves"](rows) is wide
    assert off["_moe_cp_tail_serves"](rows) is False


# --- the decode step's MoE ---

_GRAPH_TAIL_BACK = [
    "wait:shared_fork@aux",
    "shared@aux",
    "lamport_push@aux",
    "record:shared_join@aux",
    "latent_ar_norm@main",
    "cp_tail@main",
    "log",
    "wait:shared_join@main",
]
_EAGER_TAIL = [
    "router_gemm@main",
    "latent_down@main",
    "route_tail@main",
    "fp4_quantize@main",
    "fc12@main",
    "shared@main",
    "lamport_push@main",
    "latent_ar_norm@main",
    "cp_tail@main",
    "log",
]


@pytest.mark.parametrize(
    "rows, bands",
    [
        (6, None),
        (12, None),
        (24, None),
        (30, None),
        (36, None),
        (54, None),
        (60, None),
        (66, "1-72"),
        (72, "1-72"),
    ],
)
def test_a_graph_step_in_a_band_runs_the_tail_on_the_stream_plan(rows, bands):
    """``bands``: None for the source's, else the tail's bands."""
    plan, log, block, seen, layer, rt, out = _run(rows, tree=bands and _tail_tree(bands))
    assert isinstance(out, block["_MoeTail"]), type(out)
    assert tuple(out) == ("tail_ws", True, 8)
    front = F._front(plan.ops)
    # The front of today's graph step, unchanged.
    want_front = F._front(F._run(rows)[0].ops)
    assert front == want_front
    assert plan.ops[len(front) :] == ["log", *_GRAPH_TAIL_BACK], plan.ops
    assert not plan.problems, "\n".join(plan.problems)
    # Every side stream is joined before the step returns.
    for side in ("latent", "prezero", "aux"):
        last = max(at for at, stream, *_ in plan.kernels if stream == side)
        assert plan.clock["main"].get(side, 0) > last, f"the {side} stream is not joined"
    # The shared experts' partial goes into the tail's call, one-shot with PDL.
    push = seen["push"]
    assert push["ws"] == "tail_ws" and push["flags"] == (True, True)
    assert push["data"] is plan.kernel_at("shared")[3][1]
    # The tail: this rank's rows of latent_up, the push GEMM's tactic for the row count, over the
    # latent all-reduce row's output.
    tail = seen["cp_tail"]
    assert tail["ws"] == "tail_ws" and tail["c0"] == _C0 and tail["tactic"] == f"tactic_{rows}"
    assert tail["w"].data_ptr() == layer.latent_up[_C0].data_ptr()
    assert tuple(tail["w"].shape) == (_N, F._LATENT)
    assert tail["latent"] is plan.kernel_at("latent_ar_norm")[3][2]
    line = block["_MOE_CP_TAIL_LINE"]
    assert (line, line) in log.once


@pytest.mark.parametrize(
    "rows, bands", [(6, None), (24, None), (48, None), (60, None), (72, "1-72")]
)
def test_an_eager_step_in_a_band_runs_the_tail_on_the_current_stream(rows, bands):
    plan, log, block, seen, _, _, out = _run(
        rows, cuda_graph=False, tree=bands and _tail_tree(bands)
    )
    assert isinstance(out, block["_MoeTail"])
    assert plan.ops == _EAGER_TAIL and not plan.problems
    assert seen["push"]["flags"] == (True, True)


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize(
    "rows, on", [(66, True), (72, True), (78, True), (96, True), (144, True), (24, False)]
)
def test_rows_outside_the_bands_and_the_switch_off_keep_todays_tail(rows, on, graph):
    """Rows past the source's bands (66 / 72 rows, b11 / b12 at k5, among them) run the step of
    the switch off: today's replicated tail."""
    plan, _, block, seen, _, _, out = _run(rows, cuda_graph=graph, on=on)
    assert isinstance(out, block["_MoeOut"]), type(out)
    assert plan.ops == F._run(rows, cuda_graph=graph)[0].ops
    assert seen == {}


# --- the next mix finishes the call ---


class _LamportMix:
    def __init__(self, calls: list, name: str, mx: bool):
        self.calls, self.name, self.mx = calls, name, mx

    def __call__(
        self, prefix, uc, flags_ptr, bank, proj, res, out, *args, early_trigger, updated=None
    ):
        self.calls.append(
            (self.name, prefix, uc, flags_ptr, bank, (proj, res, out), args, early_trigger)
        )
        if updated is not None:  # mix_bank_in_mix: the bank row the updated prefix goes to
            self.calls.append(("updated", updated))
            if self.mx:
                return updated, "normed", "codes", "scales"
            return updated, "normed"
        if self.mx:
            return "updated", "normed", "codes", "scales"
        return "updated", "normed"


def _slots_ns(calls: list, lines: list, exact_rows: set) -> dict:
    def entry(name, result):
        def call(*args, **kwargs):
            calls.append((name, args))
            return result(*args)

        return call

    def rubin_lamport_finish(ws, out, *flags):
        calls.append(("finish", (ws, tuple(out.shape), out.dtype, flags)))
        return out.fill_(0.5)

    ns = dict(
        torch=torch,
        Any=Any,
        Optional=Optional,
        NamedTuple=NamedTuple,
        MappingProxyType=MappingProxyType,
        logger=SimpleNamespace(
            warning=lambda m: lines.append(("warn", m)),
            info_once=lambda m, key=None: lines.append(("info", m)),
        ),
        flashinfer_rmsnorm=entry("rmsnorm", lambda x, *a: "normed"),
        attn_res_fwd=entry("attn_res_fwd", lambda p, *a: (p,)),
        rubin_attn_res_rmsnorm=entry("pdl", lambda p, *a: "normed"),
        rubin_attn_res_add_rmsnorm=entry("pdl_add", lambda p, *a: (p, p)),
        rubin_attn_res_add_rmsnorm_persistent=entry("persistent", lambda p, *a: (p, p)),
        attn_res_rmsnorm_fwd=entry("per_token", lambda p, *a: p),
        attn_res_add_rmsnorm_fwd=entry("per_token_add", lambda p, *a: (p, p)),
        rubin_moe_residual_add=entry("moe_residual_add", lambda r, s, p: p),
        rubin_attn_res_moe_add_rmsnorm=entry("pdl_moe", lambda p, *a: (p, p)),
        rubin_attn_res_moe_add_rmsnorm_mx=entry("pdl_moe_mx", lambda p, *a: (p, p, "c", "s")),
        rubin_attn_res_add_rmsnorm_persistent_x=entry("x", lambda p, *a: None),
        persistent_x_available=lambda: True,
        rubin_lamport_finish=rubin_lamport_finish,
        rubin_attn_res_lamport_add_rmsnorm=_LamportMix(calls, "lamport_mix", False),
        rubin_attn_res_lamport_add_rmsnorm_mx=_LamportMix(calls, "lamport_mix_mx", True),
        _kda_mx_exact=lambda rows: rows in exact_rows,
        # The low-M QKVG GEMM quantizes up to 32 rows inside its kernel (``_kda_mx_pdl_rows``).
        _kda_mx_pdl_rows=lambda rows: rows > 32,
    )
    tree = _tree(moe_cp_tail=True)
    module = ast.fix_missing_locations(ast.Module(F._definitions(tree, _SLOT_NAMES), []))
    ns = {**_constants(tree), **ns}
    exec(compile(module, str(_MODELING), "exec"), ns)
    return ns


def _tail(ns):
    ws = SimpleNamespace(
        uc_local=1234, flags=torch.zeros(9, dtype=torch.int32), mc_base=5678, rank=_RANK, world=8
    )
    # Stand-in call fields: four ranks per partial sum, the two-shot C++ sums.
    return ns["_MoeTail"](ws, False, 4)


def _inputs(rows: int, snapshots: int):
    prefix = torch.ones(rows, F._HIDDEN, dtype=torch.bfloat16)
    bank = torch.zeros(snapshots, rows, F._HIDDEN, dtype=torch.bfloat16)
    mw = SimpleNamespace(proj="proj", res_norm="res_norm", out_norm="out_norm")
    return prefix, bank, mw


@pytest.mark.parametrize("poll", [True, False], ids=["early_poll", "no_poll"])
@pytest.mark.parametrize("slot", ["moe_norm", "moe_fold_norm"])
def test_the_pdl_mix_finishes_a_tails_call(slot, poll):
    calls, lines = [], []
    ns = _slots_ns(calls, lines, set())
    gen = dict(ns["_CONSTANTS"]["generation"], moe_cp_tail=True, moe_tail_early_poll=poll)
    slots = ns["_mix_slots"](gen, 1e-5)
    tail = _tail(ns)
    prefix, bank, mw = _inputs(54, 3)
    assert slots[slot](prefix, tail, bank, mw) == ("updated", "normed")
    ((name, got_prefix, uc, flags_ptr, got_bank, weights, args, early),) = calls
    assert name == "lamport_mix" and got_prefix is prefix and got_bank is bank
    assert (uc, flags_ptr, weights) == (
        1234,
        tail.ws.flags.data_ptr(),
        ("proj", "res_norm", "out_norm"),
    )
    # eps, eps, world, ranks per partial sum, two-shot finish, multicast base, rank,
    # round-to-nearest add, early poll, entry trigger, the all-gather rows.
    assert args == (1e-5, 1e-5, 8, 4, False, 5678, _RANK, True, poll, False, True)
    assert early is gen["attn_res_early_trigger"] is True


@pytest.mark.parametrize("rows, exact", [(36, False), (54, False), (72, True)])
def test_a_kda_layers_mix_finishes_the_call_and_stores_the_mxfp8_form(rows, exact):
    calls, lines = [], []
    ns = _slots_ns(calls, lines, {72})
    gen = dict(ns["_CONSTANTS"]["generation"], moe_cp_tail=True)
    assert gen["attn_res_mx_pdl"] and gen["fuse_moe_residual"]
    slots = ns["_mix_slots"](gen, 1e-5)
    prefix, bank, mw = _inputs(rows, 3)
    updated, normed = slots["moe_norm_mx"](prefix, _tail(ns), bank, mw)
    assert (updated, normed) == ("updated", ("normed", "codes", "scales"))
    ((name, _, _, _, _, _, args, early),) = calls
    # ... the all-gather rows, then one 128-row tile's atoms (one block) in the QKVG GEMM's rule.
    assert name == "lamport_mix_mx" and early is True
    assert args[-4:] == (True, True, 1, exact)
    assert args[:-3] == (1e-5, 1e-5, 8, 4, False, 5678, _RANK, True, True, False, True)
    assert lines == [("info", ns["_KDA_MX_PDL_LINE"])]


@pytest.mark.parametrize("rows", [6, 24, 30, 32])
def test_a_kda_layers_mix_finishes_the_call_in_bf16_where_the_gemm_quantizes_the_rows(rows):
    """Up to the low-M GEMM's in-kernel rows the tail's mix stores no MXFP8 form, as for a
    ``_MoeOut`` (``_kda_mx_pdl_rows``)."""
    calls, lines = [], []
    ns = _slots_ns(calls, lines, set())
    gen = dict(ns["_CONSTANTS"]["generation"], moe_cp_tail=True)
    assert gen["attn_res_mx_pdl"] and gen["fuse_moe_residual"]
    slots = ns["_mix_slots"](gen, 1e-5)
    prefix, bank, mw = _inputs(rows, 3)
    assert slots["moe_norm_mx"](prefix, _tail(ns), bank, mw) == ("updated", "normed")
    assert [c[0] for c in calls] == ["lamport_mix"] and lines == []


def test_without_mx_pdl_a_kda_layers_mix_finishes_the_call_in_bf16():
    calls, lines = [], []
    ns = _slots_ns(calls, lines, set())
    gen = dict(ns["_CONSTANTS"]["generation"], moe_cp_tail=True, attn_res_mx_pdl=False)
    slots = ns["_mix_slots"](gen, 1e-5)
    prefix, bank, mw = _inputs(24, 3)
    assert slots["moe_norm_mx"](prefix, _tail(ns), bank, mw) == ("updated", "normed")
    assert [c[0] for c in calls] == ["lamport_mix"] and lines == []


def test_moe_add_and_a_mix_without_snapshots_finish_with_the_finish_kernel():
    calls, lines = [], []
    ns = _slots_ns(calls, lines, set())
    gen = dict(ns["_CONSTANTS"]["generation"], moe_cp_tail=True)
    slots = ns["_mix_slots"](gen, 1e-5)
    tail = _tail(ns)
    prefix, bank, mw = _inputs(8, 2)
    want = (prefix.float() + 0.5).to(torch.bfloat16)
    # one-shot, PDL, no wait first, the call's C++ sums, the all-gather rows.
    finish = ("finish", (tail.ws, (8, F._HIDDEN), torch.bfloat16, (True, True, False, False, True)))
    assert torch.equal(slots["moe_add"](prefix, tail), want) and calls == [finish]
    for slot in ("moe_norm", "moe_fold_norm", "moe_norm_mx"):
        calls.clear()
        updated, normed = slots[slot](prefix, tail, bank[:0], mw)
        assert torch.equal(updated, want) and normed == "normed"
        assert [c[0] for c in calls] == ["finish", "rmsnorm"] and calls[0] == finish


def _slot_rt(calls: list) -> SimpleNamespace:
    def slot(name, result):
        def call(*args):
            calls.append((name, args))
            return result

        return call

    return SimpleNamespace(
        attn_res_norm=slot("attn_res_norm", "normed"),
        add_norm=slot("add_norm", ("added", "normed")),
        moe_norm=slot("moe_norm", ("moe_added", "normed")),
        moe_fold_norm=slot("moe_fold_norm", ("moe_added", "normed")),
        moe_norm_mx=slot("moe_norm_mx", ("moe_added", "normed")),
        moe_add=slot("moe_add", torch.full((2, F._HIDDEN), 5.0)),
        attn_res_mix=slot("attn_res_mix", "mix"),
        rms_norm=slot("rms_norm", "normed"),
        taps=None,
    )


@pytest.mark.parametrize("tapped", [False, True], ids=["untapped", "tapped"])
@pytest.mark.parametrize("kda", [False, True], ids=["mla_next", "kda_next"])
def test_the_next_mix_takes_a_tail_as_a_moe_output(kda, tapped):
    ns = _slots_ns([], [], set())
    tail, prefix, bank = ns["_MoeTail"]("tail_ws", True, 8), torch.ones(2, F._HIDDEN), "bank"
    mw = SimpleNamespace(out_norm="out_norm")
    taps = []
    tap = (SimpleNamespace(maybe_capture_hidden_states=lambda *a: taps.append(a)), 5)
    calls = []
    ns["_mix_in"](_slot_rt(calls), prefix, tail, bank, mw, tap if tapped else None, kda)
    names = [name for name, _ in calls]
    if tapped:
        assert names == ["moe_add", "attn_res_mix", "rms_norm"] and taps == [(5, "mix", None)]
    else:
        assert names == (["moe_norm_mx"] if kda else ["moe_fold_norm"])
    assert calls[0][1][1] is tail
    calls = []
    ns["_mix_out"](
        _slot_rt(calls), prefix, tail, bank, SimpleNamespace(out=mw), tap if tapped else None
    )
    names = [name for name, _ in calls]
    assert names == (["moe_add", "attn_res_mix", "attn_res_norm"] if tapped else ["moe_fold_norm"])
    assert calls[0][1][1] is tail


def test_the_runtime_builds_the_tail_on_the_moe_tp_group_only_when_on():
    """``_build_runtime`` builds the tail's workspace and share on ``core.tp_mapping`` (the helix
    ranks as TP ranks), sized to the bands' largest row count, under ``moe_cp_tail`` only."""
    (fn,) = F._definitions(_tree(), ("_build_runtime",))
    (branch,) = [
        n for n in ast.walk(fn) if isinstance(n, ast.If) and "moe_cp_tail" in ast.unparse(n.test)
    ]
    assert ast.unparse(branch.test) == "c.get('moe_cp_tail')"
    text = ast.unparse(branch)
    assert "rubin_ull_tail_workspace(core.tp_mapping, _WORLD_SIZE, _MOE_CP_TAIL_MAX_M)" in text
    assert (
        "rubin_ull_cp_tail_share(_WORLD_SIZE, core.tp_mapping.tp_rank, _MOE_CP_TAIL_MAX_M)" in text
    )
    assert "ull_tail=ull_tail" in ast.unparse(fn)
