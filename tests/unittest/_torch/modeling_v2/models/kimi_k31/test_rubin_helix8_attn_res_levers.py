# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's attention-residual epilogue levers and the indexer logits' PDL lever (CPU, from source).

``attn_res_mx_pdl`` (the PDL kernel's fused MoE mix stores the MXFP8 form of a KDA layer's normed
rows, up to one 128-row scale tile, where the QKVG GEMM would quantize them in a kernel of its own)
is checked here too: it sets ``moe_norm_mx`` with or without the persistent kernel, hands over codes
and R128c4 scales (one atom block) on exactly the rows it serves, and says so once. It serves the
PDL rows past the ones the low-M GEMM quantizes inside its kernel (``mxfp8_quantizes_in_gemm`` for
the fused projection's ``[6288, 7168]`` weight: up to 32 rows on sm_107, asked once per row count);
those rows make the calls they make with the lever off. Their rule is the QKVG GEMM's own
quantization of the rows (``_kda_mx_exact``): the shared low-M / mid-M GEMM's floored amax where
``mxfp8_act_layout`` names its layout for that weight, the catalog quantizer's (``mx_exact``) where
it names the dense GEMM's R128c4; asked once per row count.

Four launch constants of the generation role, all on: ``attn_res_mx_moe`` (a KDA layer's mix
folds the MoE residual update and stores the MXFP8 form its QKVG GEMM multiplies),
``attn_res_moe_fold`` (every other MoE-fed mix and the output mix fold the update),
``attn_res_grid_cap`` (the persistent mixes launch min(SMs, rows) CTAs) and
``dsa_idx_logits_pdl`` (the key-shard indexer logits kernel launches with PDL). The first three
take ``rubin_attn_res_add_rmsnorm_persistent_x`` (a CuTe DSL kernel); on a device it does not run on
they stay off with one WARNING line.

The product's functions run from their source over recording stand-ins of the mix entries:

* the constants: all four on in the table, read by ``_x_levers`` and the logits constant; no
  environment variable;
* with the table's constants the persistent rows take the epilogue op with every flag;
* with every lever off (switched off here) the slots call the parent's entries and the two new
  slots are None;
* each lever moves exactly the persistent kernel's rows (past the PDL kernel's 128, at most 9
  candidates, up to ``_X_MAX_ROWS``) to the epilogue op with its flags, and leaves the PDL rows
  (96 at b16 k5) on today's entries;
* ``_mix_in`` / ``_mix_out`` take the new slots only when they are set; the decode loop flags the
  KDA layers; a KDA block multiplies the mix's MXFP8 form when it comes with the rows;
* ``_idx_decode`` passes ``pdl`` to the key-shard logits only with the constant on, and the
  logits op keys it.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, NamedTuple, Optional

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_HELIX8 = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_HELIX8_WEIGHTS = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "weights.py"
_ROUTING = _FAMILY / "routing.py"
_OP = _FAMILY / "ops/norm/rubin_attn_res_add_rmsnorm_persistent_x.py"
_LOGITS_OP = _FAMILY / "ops/attention/rubin_fp8_paged_mqa_logits.py"

_NAMES = (
    "_TARGET",
    "_CONSTANTS",
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
    "_K31",
    "_K31_KDA",
    "_KDA_TP",
    "_KDA_H",
    "_KDA_D",
    "_KDA_P",
    "_KDA_FA0",
    "_KDA_B0",
    "_KDA_B1",
    "_KDA_FAB_N",
    "_kda_mx_exact",
    "_kda_mx_pdl_rows",
    "_kda_project",
    "_kda_fb",
    "_kda_split",
)
_LEVERS = ("attn_res_mx_moe", "attn_res_moe_fold", "attn_res_grid_cap")
_OFF = dict.fromkeys((*_LEVERS, "attn_res_mx_pdl"), False)
_HIDDEN = 64


def _definitions(path: Path, names) -> list:
    """The module-level definitions and assignments of ``names`` in ``path``, in source order;
    each name must be defined exactly once."""
    wanted, nodes, seen = set(names), [], {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined = {node.name}
        elif isinstance(node, ast.Assign):
            defined = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if defined & wanted:
            nodes.append(node)
            for name in defined & wanted:
                seen[name] = seen.get(name, 0) + 1
    assert seen == dict.fromkeys(wanted, 1), (path.name, seen)
    return nodes


def _function(path: Path, name: str) -> ast.FunctionDef:
    (node,) = _definitions(path, (name,))
    return node


class _X(NamedTuple):
    updated: Any
    output: Any
    output_f32: Any
    codes: Any
    scales: Any


class _XStandIn:
    """``rubin_attn_res_add_rmsnorm_persistent_x``: records its arguments, returns marked tensors
    (without an addend, ``prefix`` itself, as the entry returns it)."""

    def __init__(self, calls: list):
        self.calls = calls

    def __call__(self, prefix, addend, addend2, bank, *w, f32=False, mx=False, grid_cap=False):
        self.calls.append(
            (
                "x",
                dict(
                    addend=addend is not None,
                    addend2=addend2 is not None,
                    mx=mx,
                    grid_cap=grid_cap,
                    rows=prefix.shape[0],
                    k=bank.shape[0],
                ),
            )
        )
        updated = prefix.view(prefix.shape) if addend is None else prefix + 1
        return _X(updated, prefix + 2, None, "codes" if mx else None, "scales" if mx else None)


class _PdlMxStandIn:
    """``rubin_attn_res_moe_add_rmsnorm_mx``: records its flags, returns marked outputs."""

    def __init__(self, calls: list):
        self.calls = calls

    def __call__(
        self,
        prefix,
        routed,
        shared,
        bank,
        proj,
        res_norm,
        out_norm,
        eps,
        out_eps,
        entry_trigger,
        mx_atom,
        mx_atom_blocks,
        *,
        early_trigger,
        mx_exact=False,
        updated=None,
    ):
        assert updated is None, "mix_bank_in_mix is off: no bank row reaches the mix"
        self.calls.append(
            (
                "pdl_moe_mx",
                dict(
                    entry_trigger=entry_trigger,
                    mx_atom=mx_atom,
                    mx_atom_blocks=mx_atom_blocks,
                    early_trigger=early_trigger,
                    mx_exact=mx_exact,
                    rows=prefix.shape[0],
                    k=bank.shape[0],
                ),
            )
        )
        return prefix + 1, prefix + 2, "codes", "scales"


def _ns(
    calls: list,
    available: bool = True,
    lines: Optional[list] = None,
    layout: str = "mid",
    asked: Optional[list] = None,
    in_gemm_rows: int = 32,
    gate: Optional[list] = None,
) -> dict:
    """The definitions over stand-ins. ``layout``: what ``mxfp8_act_layout`` names (``"mid"``, the
    shared low-M / mid-M GEMM, as on sm_107); ``asked`` records its ``(m, n, k)``.
    ``in_gemm_rows``: the rows up to which ``mxfp8_quantizes_in_gemm`` holds (the low-M kernel's
    32 for the fused projection on sm_107; 0 where the dense GEMM runs at every row count);
    ``gate`` records its ``(m, n, k)``."""
    lines = [] if lines is None else lines
    asked = [] if asked is None else asked
    gate = [] if gate is None else gate

    def entry(name: str, result: Callable) -> Callable:
        def call(*args, **kwargs):
            calls.append((name, None))
            return result(*args)

        return call

    def mxfp8_act_layout(m: int, n: int, k: int) -> tuple[int, str]:
        asked.append((m, n, k))
        return 128, layout

    def mxfp8_quantizes_in_gemm(m: int, n: int, k: int) -> bool:
        gate.append((m, n, k))
        return m <= in_gemm_rows

    ns = dict(
        torch=torch,
        functools=functools,
        mxfp8_act_layout=mxfp8_act_layout,
        mxfp8_quantizes_in_gemm=mxfp8_quantizes_in_gemm,
        Any=Any,
        Callable=Callable,
        NamedTuple=NamedTuple,
        Optional=Optional,
        MappingProxyType=MappingProxyType,
        logger=SimpleNamespace(
            warning=lambda m: lines.append(("warn", m)),
            info_once=lambda m, key=None: lines.append(("info", m)),
        ),
        flashinfer_rmsnorm=entry("rmsnorm", lambda x, *a: x),
        attn_res_fwd=entry("attn_res_fwd", lambda p, *a: (p,)),
        rubin_attn_res_rmsnorm=entry("pdl", lambda p, *a: p),
        rubin_attn_res_add_rmsnorm=entry("pdl_add", lambda p, *a: (p, p)),
        rubin_attn_res_add_rmsnorm_persistent=entry("persistent", lambda p, *a: (p, p)),
        attn_res_rmsnorm_fwd=entry("per_token", lambda p, *a: p),
        attn_res_add_rmsnorm_fwd=entry("per_token_add", lambda p, *a: (p, p)),
        rubin_moe_residual_add=entry("moe_residual_add", lambda r, s, p: p),
        rubin_attn_res_moe_add_rmsnorm=entry("pdl_moe", lambda p, *a: (p, p)),
        rubin_attn_res_moe_add_rmsnorm_mx=_PdlMxStandIn(calls),
        rubin_attn_res_add_rmsnorm_persistent_x=_XStandIn(calls),
        persistent_x_available=lambda: available,
        rubin_mxfp8_linear=entry("mxfp8_linear", lambda x, w, s: x),
        rubin_mxfp8_linear_prequantized=entry(
            "mxfp8_linear_prequantized", lambda c, s, w, ws: torch.zeros(4, 8192)
        ),
        rubin_linear=entry("linear", lambda x, w, t: x),
    )
    module = ast.fix_missing_locations(ast.Module(_definitions(_HELIX8, _NAMES), []))
    exec(compile(module, _HELIX8.name, "exec"), ns)
    return ns


def _gen(ns, **on) -> dict:
    """The generation constants with ``on`` set."""
    return dict(ns["_CONSTANTS"]["generation"], **on)


def _inputs(ns, rows: int, snapshots: int):
    prefix = torch.zeros(rows, _HIDDEN, dtype=torch.bfloat16)
    moe = ns["_MoeOut"](torch.zeros_like(prefix), torch.zeros_like(prefix))
    bank = torch.zeros(snapshots, rows, _HIDDEN, dtype=torch.bfloat16)
    mw = ns["_MixW"](torch.ones(_HIDDEN), torch.ones(_HIDDEN), torch.ones(_HIDDEN))
    return prefix, moe, bank, mw


def _call(ns, calls, c, slot, rows, snapshots):
    calls.clear()
    slots = ns["_mix_slots"](c, 1e-5)
    prefix, moe, bank, mw = _inputs(ns, rows, snapshots)
    if slot in ("moe_norm", "moe_norm_mx", "moe_fold_norm"):
        out = slots[slot](prefix, moe, bank, mw)
    elif slot == "add_norm":
        out = slots[slot](prefix, prefix, bank, mw)
    else:
        out = slots[slot](prefix, bank, mw)
    return out, list(calls)


def _pdl_mx(c, rows: int, k: int = 3, exact: bool = False) -> tuple:
    """The PDL ``_mx`` call; ``exact``: the catalog rule (the stand-in layout ``"r128c4"``)."""
    want = dict(entry_trigger=False, mx_atom=True, mx_atom_blocks=1, mx_exact=exact)
    return ("pdl_moe_mx", dict(want, early_trigger=c["attn_res_early_trigger"], rows=rows, k=k))


def _x(**flags) -> tuple:
    want = dict(addend=False, addend2=False, mx=False, grid_cap=False, rows=144, k=3)
    want.update(flags)
    return ("x", want)


# --- the constants ---


def test_the_levers_are_launch_constants_on():
    ns = _ns([])
    gen = ns["_CONSTANTS"]["generation"]
    for key in (*_LEVERS, "dsa_idx_logits_pdl"):
        assert gen[key] is True, key
    assert gen["attn_res_mx_pdl"] is True
    assert ns["_MX_TILE_ROWS"] == 128 and gen["attn_res_pdl_max_m"] == 128
    assert ns["_X_LEVERS"] == ("attn_res_mx_moe", "attn_res_moe_fold", "attn_res_grid_cap")
    (pdl,) = _definitions(_HELIX8, ("_IDX_LOGITS_PDL",))
    assert ast.unparse(pdl.value) == "_CONSTANTS['generation']['dsa_idx_logits_pdl']"
    # The target reads no environment variable; routing records none of these.
    (node,) = _definitions(_ROUTING, ("LEVER_ENV",))
    assert not [v for v in ast.literal_eval(node.value) if "ATTN_RES" in v or "LOGITS_PDL" in v]


@pytest.mark.parametrize("lever", _LEVERS)
def test_x_levers_reads_each_constant(lever):
    lines = []
    ns = _ns([], lines=lines)
    assert ns["_x_levers"](_gen(ns)) == (True, True, True)
    assert ns["_x_levers"](_gen(ns, **_OFF)) == (False, False, False)
    got = ns["_x_levers"](_gen(ns, **{**_OFF, lever: True}))
    assert got == tuple(k == lever for k in _LEVERS)
    assert not lines


def test_a_device_without_the_kernel_keeps_the_levers_off_and_says_so():
    lines = []
    ns = _ns([], available=False, lines=lines)
    got = ns["_x_levers"](_gen(ns, **dict.fromkeys(_LEVERS, True)))
    assert got == (False, False, False)
    (warn,) = [m for kind, m in lines if kind == "warn"]
    assert (
        "rubin_attn_res_add_rmsnorm_persistent_x" in warn and "does not run on this device" in warn
    )
    assert all(k in warn for k in _LEVERS)


# --- the slots ---


def test_a_generation_server_runs_every_lever_on_the_persistent_rows():
    calls = []
    ns = _ns(calls)
    c = _gen(ns)  # the generation constants as they are: every lever on
    assert ns["_x_levers"](c) == (True, True, True)
    _, got = _call(ns, calls, c, "attn_res_norm", 144, 3)
    assert got == [_x(grid_cap=True)]
    (_, (_, codes, scales)), got = _call(ns, calls, c, "moe_norm_mx", 144, 3)
    assert got == [_x(addend=True, addend2=True, mx=True, grid_cap=True)]
    assert (codes, scales) == ("codes", "scales")
    _, got = _call(ns, calls, c, "moe_fold_norm", 144, 3)
    assert got == [_x(addend=True, addend2=True, grid_cap=True)]
    # The PDL kernel's rows (96 at b16 k5) keep their calls, but a KDA layer's mix stores the
    # MXFP8 form there too (attn_res_mx_pdl).
    _, got = _call(ns, calls, c, "moe_fold_norm", 96, 3)
    assert [n for n, _ in got] == ["pdl_moe"]
    (_, (_, codes, scales)), got = _call(ns, calls, c, "moe_norm_mx", 96, 3)
    assert got == [_pdl_mx(c, 96)] and (codes, scales) == ("codes", "scales")
    # Up to the rows the low-M GEMM quantizes in its kernel (24 at b4 k5) a KDA layer's mix is
    # every other MoE-fed mix's.
    (_, normed), got = _call(ns, calls, c, "moe_norm_mx", 24, 3)
    assert [n for n, _ in got] == ["pdl_moe"] and isinstance(normed, torch.Tensor)


@pytest.mark.parametrize("rows", [96, 144])
def test_with_every_lever_off_the_slots_are_todays(rows):
    calls = []
    ns = _ns(calls)
    c = _gen(ns, **_OFF)
    slots = ns["_mix_slots"](c, 1e-5)
    assert slots["moe_fold_norm"] is None and slots["moe_norm_mx"] is None
    pdl = rows <= c["attn_res_pdl_max_m"]
    for slot, want in (
        ("attn_res_norm", ["pdl"] if pdl else ["persistent"]),
        ("add_norm", ["pdl_add"] if pdl else ["persistent"]),
        ("moe_norm", ["pdl_moe"] if pdl else ["moe_residual_add", "persistent"]),
    ):
        _, got = _call(ns, calls, c, slot, rows, 3)
        assert [n for n, _ in got] == want, (slot, got)


@pytest.mark.parametrize(
    "rows, moved",
    [(144, True), (129, True), (4096, True), (128, False), (96, False), (4097, False)],
)
def test_grid_cap_moves_exactly_the_persistent_rows(rows, moved):
    calls = []
    ns = _ns(calls)
    c = _gen(ns, **{**_OFF, "attn_res_grid_cap": True})
    for slot, addend in (("attn_res_norm", False), ("add_norm", True)):
        _, got = _call(ns, calls, c, slot, rows, 3)
        if moved:
            assert got == [_x(addend=addend, grid_cap=True, rows=rows)], got
        else:
            assert [n for n, _ in got] in (["pdl"], ["pdl_add"], ["persistent"]), got
    # 10 candidates: not the persistent kernel's
    _, got = _call(ns, calls, c, "attn_res_norm", 144, 9)
    assert [n for n, _ in got] != ["x"]


@pytest.mark.parametrize("cap", [False, True])
def test_mx_moe_folds_the_update_and_hands_the_mxfp8_form_over(cap):
    calls = []
    ns = _ns(calls)
    c = _gen(ns, **{**_OFF, "attn_res_mx_moe": True, "attn_res_grid_cap": cap})
    (prefix, (normed, codes, scales)), got = _call(ns, calls, c, "moe_norm_mx", 144, 3)
    assert got == [_x(addend=True, addend2=True, mx=True, grid_cap=cap)]
    assert (codes, scales) == ("codes", "scales")
    # The PDL kernel's rows keep today's fused MoE mix (no MXFP8 form); an empty bank keeps moe_norm.
    (_, normed), got = _call(ns, calls, c, "moe_norm_mx", 96, 3)
    assert [n for n, _ in got] == ["pdl_moe"] and isinstance(normed, torch.Tensor)
    _, got = _call(ns, calls, c, "moe_norm_mx", 144, 0)
    assert [n for n, _ in got] == ["moe_residual_add", "rmsnorm"]
    assert ns["_mix_slots"](c, 1e-5)["moe_fold_norm"] is None


@pytest.mark.parametrize("rows", [36, 54, 60, 66, 96, 126, 128])
def test_mx_pdl_hands_the_pdl_rows_mxfp8_form_over(rows):
    calls, lines = [], []
    ns = _ns(calls, lines=lines)
    c = _gen(ns, **{**_OFF, "attn_res_mx_pdl": True})
    (prefix, (normed, codes, scales)), got = _call(ns, calls, c, "moe_norm_mx", rows, 3)
    assert got == [_pdl_mx(c, rows)]
    assert (codes, scales) == ("codes", "scales")
    assert isinstance(prefix, torch.Tensor) and isinstance(normed, torch.Tensor)
    assert lines == [("info", ns["_KDA_MX_PDL_LINE"])]
    # The other MoE-fed slots keep today's PDL call.
    _, got = _call(ns, calls, c, "moe_norm", rows, 3)
    assert [n for n, _ in got] == ["pdl_moe"]


@pytest.mark.parametrize(
    "rows, served",
    [
        (1, False),
        (6, False),
        (12, False),
        (18, False),
        (24, False),
        (30, False),
        (32, False),
        (33, True),
        (36, True),
        (128, True),
        (129, False),
        (144, False),
    ],
)
def test_mx_pdl_serves_the_rows_the_qkvg_gemm_quantizes_apart(rows, served):
    """Up to the rows the low-M GEMM quantizes inside its kernel (32 for the fused projection on
    sm_107) a KDA layer's mix makes the calls it makes with the lever off (the slot ``_mix_in``
    takes then) and hands over no MXFP8 form; past them, up to one tile, the PDL mix stores it;
    past one tile the gate is not asked. Alone and with every other lever on."""
    calls, lines, asked, gate = [], [], [], []
    ns = _ns(calls, lines=lines, asked=asked, gate=gate)
    for on in (_gen(ns, **{**_OFF, "attn_res_mx_pdl": True}), _gen(ns, attn_res_mx_pdl=True)):
        (_, normed), got = _call(ns, calls, on, "moe_norm_mx", rows, 3)
        if served:
            assert got == [_pdl_mx(on, rows)]
            continue
        off = dict(on, attn_res_mx_pdl=False)
        slots = ns["_mix_slots"](off, 1e-5)
        slot = next(s for s in ("moe_norm_mx", "moe_fold_norm", "moe_norm") if slots[s])
        _, want = _call(ns, calls, off, slot, rows, 3)
        assert got == want, (slot, got, want)
        if rows <= 128:  # the PDL rows: no MXFP8 form (past them attn_res_mx_moe's may come)
            assert isinstance(normed, torch.Tensor)
    assert gate == ([(rows, 6288, 7168)] if rows <= 128 else [])
    assert asked == ([(rows, 6288, 7168)] if served else [])
    assert lines == ([("info", ns["_KDA_MX_PDL_LINE"])] * 2 if served else [])


def test_mx_pdl_leaves_every_other_row_to_its_slot():
    calls = []
    ns = _ns(calls)
    c = _gen(ns, **{**_OFF, "attn_res_mx_pdl": True})
    # Past one scale tile: today's unfused update and the persistent mix, no MXFP8 form.
    (_, normed), got = _call(ns, calls, c, "moe_norm_mx", 144, 3)
    assert [n for n, _ in got] == ["moe_residual_add", "persistent"]
    assert isinstance(normed, torch.Tensor)
    # An empty bank: moe_norm.
    _, got = _call(ns, calls, c, "moe_norm_mx", 54, 0)
    assert "pdl_moe_mx" not in [n for n, _ in got]
    # With every lever on, the persistent rows keep attn_res_mx_moe's epilogue op.
    c = _gen(ns, attn_res_mx_pdl=True)
    _, got = _call(ns, calls, c, "moe_norm_mx", 144, 3)
    assert got == [_x(addend=True, addend2=True, mx=True, grid_cap=True)]
    _, got = _call(ns, calls, c, "moe_norm_mx", 54, 3)
    assert got == [_pdl_mx(c, 54)]


def test_mx_pdl_needs_no_persistent_kernel_but_the_pdl_moe_mix():
    calls, lines = [], []
    ns = _ns(calls, available=False, lines=lines)
    c = _gen(ns, attn_res_mx_pdl=True)  # every lever asked for, the persistent_x kernel absent
    assert ns["_x_levers"](c) == (False, False, False)
    _, got = _call(ns, calls, c, "moe_norm_mx", 54, 3)
    assert got == [_pdl_mx(c, 54)]
    _, got = _call(ns, calls, c, "moe_norm_mx", 144, 3)
    assert [n for n, _ in got] == ["moe_residual_add", "persistent"]
    for off in (dict(fuse_moe_residual=False), dict(attn_res_pdl=1)):
        slots = ns["_mix_slots"](_gen(ns, attn_res_mx_pdl=True, **off), 1e-5)
        assert slots["moe_norm_mx"] is None, off


@pytest.mark.parametrize("layout, exact, in_gemm_rows", [("mid", False, 32), ("r128c4", True, 0)])
@pytest.mark.parametrize("rows", [6, 24, 32, 36, 54, 126, 128])
def test_mx_pdl_quantizes_the_rows_as_the_qkvg_gemm_does(rows, layout, exact, in_gemm_rows):
    """The shared low-M / mid-M GEMM (its layout named) quantizes the rows itself, amax floored:
    ``mx_exact`` off; it quantizes up to 32 rows inside its kernel, where the form is not stored.
    The dense GEMM (R128c4 named, at every row count where the shared GEMM does not run) takes
    the catalog quantizer's form after a quantization kernel at every row count: on. The layout
    is asked for the fused projection's weight, once per row count it serves."""
    calls, asked = [], []
    ns = _ns(calls, layout=layout, asked=asked, in_gemm_rows=in_gemm_rows)
    c = _gen(ns, **{**_OFF, "attn_res_mx_pdl": True})
    served = rows > in_gemm_rows
    for _ in range(2):
        _, got = _call(ns, calls, c, "moe_norm_mx", rows, 3)
        assert got == ([_pdl_mx(c, rows, exact=exact)] if served else [("pdl_moe", None)])
    assert asked == ([(rows, 6288, 7168)] if served else [])


def test_the_rule_is_asked_for_the_weight_weights_py_loads():
    ns = _ns([])
    # q | k | v | g (4 x 12 heads x 128), f_a (128), b (12 heads): 6284 rows, padded to 16.
    assert ns["_KDA_B1"] == 6284 and ns["_KDA_FAB_N"] == 6288
    (align,) = _definitions(_HELIX8_WEIGHTS, ("_W1_KDA_FAB_ROW_ALIGN",))
    assert ast.literal_eval(align.value) == 16
    # Past one tile the layout no longer names the GEMM (a mid-M shape reads R128c4 there).
    with pytest.raises(AssertionError, match="past one tile"):
        ns["_kda_mx_exact"](129)


def test_moe_fold_is_a_slot_of_its_own():
    calls = []
    ns = _ns(calls)
    c = _gen(ns, **{**_OFF, "attn_res_moe_fold": True})
    _, got = _call(ns, calls, c, "moe_fold_norm", 144, 3)
    assert got == [_x(addend=True, addend2=True)]
    _, got = _call(ns, calls, c, "moe_fold_norm", 96, 3)
    assert [n for n, _ in got] == ["pdl_moe"]
    assert ns["_mix_slots"](c, 1e-5)["moe_norm_mx"] is None
    _, got = _call(ns, calls, c, "attn_res_norm", 144, 3)  # no grid cap: the plain slots stay
    assert [n for n, _ in got] == ["persistent"]


# --- _mix_in, _mix_out, the decode loop, the KDA blocks ---


def _rt(calls, fold: bool, mx: bool):
    def slot(name, result):
        def call(*args):
            calls.append(name)
            return result

        return call

    return SimpleNamespace(
        moe_norm=slot("moe_norm", ("u", "n")),
        moe_norm_mx=slot("moe_norm_mx", ("u", ("n", "c", "s"))) if mx else None,
        moe_fold_norm=slot("moe_fold_norm", ("u", "folded")) if fold else None,
        moe_add=slot("moe_add", "u"),
        attn_res_norm=slot("attn_res_norm", "n"),
        attn_res_mix=slot("attn_res_mix", "mix"),
        add_norm=slot("add_norm", ("u", "n")),
        rms_norm=slot("rms_norm", "n"),
        taps=None,
    )


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("mx", [False, True])
def test_mix_in_and_mix_out_take_the_new_slots_only_when_set(fold, mx):
    calls = []
    ns = _ns([])
    moe = ns["_MoeOut"]("r", "s")
    rt = _rt(calls, fold, mx)
    ns["_mix_in"](rt, "p", moe, "bank", "mw", None)  # an MLA layer
    ns["_mix_in"](rt, "p", moe, "bank", "mw", None, True)  # a KDA layer
    head = SimpleNamespace(out="mw")
    out = ns["_mix_out"](rt, "p", moe, "bank", head, None)
    plain = "moe_fold_norm" if fold else "moe_norm"
    want = [plain, "moe_norm_mx" if mx else plain]
    want += ["moe_fold_norm"] if fold else ["moe_add", "attn_res_norm"]
    assert calls == want
    assert out == ("folded" if fold else "n")
    calls.clear()  # a tapped layer runs apart whatever is on
    tap = (SimpleNamespace(maybe_capture_hidden_states=lambda *a: calls.append("tap")), 3)
    ns["_mix_in"](rt, "p", moe, "bank", SimpleNamespace(out_norm="o"), tap, True)
    ns["_mix_out"](rt, "p", moe, "bank", head, tap)
    assert calls == ["moe_add", "attn_res_mix", "tap", "rms_norm"] + [
        "moe_add",
        "attn_res_mix",
        "tap",
        "attn_res_norm",
    ]


def test_the_decode_loop_flags_the_kda_layers():
    loop = ast.unparse(_definitions(_HELIX8, ("DecodeTarget",))[0])
    assert (
        "prefix, x = _mix_in(rt, prefix, pending, bank[:L.n_bank], L.pre, "
        "_tap(st.spec_md, i - 1), L.kda is not None, snap)"
    ) in loop
    assert "o = kda(L.kda, x, st, rt)" in loop


def test_a_kda_block_multiplies_the_mixs_mxfp8_form():
    calls = []
    ns = _ns(calls)
    L = SimpleNamespace(qkvgfab_w="W", qkvgfab_s="S", fb="fb")
    x = torch.zeros(4, 8192)
    assert ns["_KDA_B0"] + 16 <= x.shape[1]
    rows, mx = ns["_kda_split"]((x, "codes", "scales"))
    assert rows is x and mx == ("codes", "scales")
    assert ns["_kda_split"](x) == (x, None)
    calls.clear()
    ns["_kda_project"](L, x, None)
    assert [n for n, _ in calls] == ["mxfp8_linear", "linear"]
    calls.clear()
    ns["_kda_project"](L, x, ("codes", "scales"))
    assert [n for n, _ in calls] == ["mxfp8_linear_prequantized", "linear"]
    calls.clear()
    # kda_verify_fused_fb: the verify kernel computes the forget gate, so no f_b GEMM.
    assert ns["_kda_project"](L, x, ("codes", "scales"), False)[1] is None
    assert [n for n, _ in calls] == ["mxfp8_linear_prequantized"]
    for name in ("kda_decode", "kda_verify"):
        body = [ast.unparse(s) for s in _function(_HELIX8, name).body]
        assert "x, mx = _kda_split(x)" in body, name
    assert "y, g = _kda_project(L, x, mx)" in ast.unparse(_function(_HELIX8, "_kda_decode_core"))
    assert "y, g = _kda_project(L, x, mx, gate=fused is False)" in ast.unparse(
        _function(_HELIX8, "_kda_verify_core")
    )


# --- the logits lever, the op entry ---


def test_idx_decode_passes_the_logits_constant_and_the_op_keys_it():
    calls = [n for n in ast.walk(_function(_HELIX8, "_idx_decode")) if isinstance(n, ast.Call)]
    logits = [c for c in calls if ast.unparse(c.func) == "rubin_fp8_paged_mqa_logits"]
    # The key-shard call takes pdl only when the constant is on; the eager fallback takes none.
    (shard,) = [c for c in logits if c.keywords]
    assert [(k.arg, ast.unparse(k.value)) for k in shard.keywords] == [
        (None, "{'pdl': True} if _IDX_LOGITS_PDL else {}")
    ]
    text = _LOGITS_OP.read_text()
    assert "if pdl:  # its own cache entry" in text and "key = key + (True,)" in text


def test_the_op_entry_launches_the_dsl_kernel_and_says_where_it_does_not_run():
    avail, cls = _definitions(
        _OP, ("persistent_x_available", "RubinAttnResAddRmsnormPersistentX")
    )  # source order
    assert ast.unparse(avail.body[-1]) == "return IS_CUTLASS_DSL_AVAILABLE and _kernel_supported()"
    assert "persistent_x_available()" in ast.unparse(_function(_HELIX8, "_x_levers"))
    imports = {
        a.asname: (n.module, n.level, a.name)
        for n in ast.walk(ast.parse(_OP.read_text()))
        if isinstance(n, ast.ImportFrom)
        for a in n.names
        if a.asname
    }
    kernel = "kernels.norm.rubin_attn_res_persistent_x"
    assert imports["_launch"] == (kernel, 3, "attn_res_add_rmsnorm_persistent_x")
    assert imports["_kernel_supported"] == (kernel, 3, "supported")
    (call,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"]
    (launch,) = [
        n for n in ast.walk(call) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_launch"
    ]
    assert len(launch.args) == 12  # the launcher's twelve arguments, in order
    assert "torch.ops.trtllm" not in _OP.read_text()
