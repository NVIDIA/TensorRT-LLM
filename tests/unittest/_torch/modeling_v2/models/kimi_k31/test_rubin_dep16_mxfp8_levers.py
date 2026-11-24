# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's MXFP8 levers (CPU, from source): some BF16 GEMMs of a generation server in MXFP8.

Three switches of ``_Levers`` (``rt.levers``), off on both server roles (a generation server's
constants set them False: they would hold BF16 checkpoint weights below the recipe's precision):
``mla_mxfp8`` (the DSA layers' g_proj and o_proj), ``moe_latent_mxfp8`` (latent_down) and
``draft_mxfp8`` (each drafter layer's o_proj, gate_up and down). Off, a generation server's table
holds those rows in BF16 and the block projects them with the BF16 GEMMs. The tests below turn a
lever on explicitly to keep its path checked. Under one:

* the weight table loads each of the lever's rows (``weights.MXFP8_*``) as an MXFP8 pair,
  ``{name}_w`` (E4M3) and ``{name}_s`` (UE8M0 1x32 scales, R128c4, flat), quantized once at load by
  the checkpoint MXFP8 weights' recipe; the BF16 parameter is not declared;
* the views hand those weights out as ``_Mx`` and the block projects them with
  ``rubin_mxfp8_linear`` (``_dense``, ``_moe_latent_down``), pinned except the o_proj GEMMs, on
  the same streams, after the same waits; every other kernel is unchanged;
* ``rubin_mxfp8_gemm.PINNED_TACTICS`` holds a split-K-free tactic for each of those GEMM shapes.

The block functions run from modeling.py over the stand-ins of ``test_rubin_dep16_gemm_levers.py``
and ``test_rubin_dep16_moe_front.py``; an MXFP8 stand-in multiplies the same exact operands, so a
lever's run keeps the lever-off bits and differs only in which entries ran.
"""

from __future__ import annotations

import ast
import math
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["."]
import test_rubin_dep16_gemm_levers as G  # noqa: E402
import test_rubin_dep16_moe_front as F  # noqa: E402

pytestmark = pytest.mark.cpu_only

_GEMM_OP = G._FAMILY / "ops/gemm/rubin_mxfp8_gemm.py"
_MX = ("mla_mxfp8", "moe_latent_mxfp8", "draft_mxfp8")


def _family_quantizer() -> dict:
    """``quantize_mxfp8_reference`` and ``mxfp8_scales_to_r128c4`` of the family's GEMM entry,
    executed from its source (no package import)."""
    tree = G._tree(_GEMM_OP)
    names = ("_pad_up", "mxfp8_scales_to_r128c4", "_pow2", "quantize_mxfp8_reference")
    ns = G._constants(tree, {"torch": torch})
    return G._exec(tree, names, ns, _GEMM_OP)


def _weights_ns() -> dict:
    """weights.py's body (``G._weights_ns``) with the family quantizer it imports."""
    ns = G._weights_ns()
    fam = _family_quantizer()
    ns.update(
        quantize_mxfp8_reference=fam["quantize_mxfp8_reference"],
        mxfp8_scales_to_r128c4=fam["mxfp8_scales_to_r128c4"],
    )
    return ns


def _independent_mxfp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 written here from the rule: per 32 values the smallest power of two at or above
    amax / 448 (exponent from frexp), E4M3 by round-to-nearest-even; the scale bytes row-major."""
    rows, k = w.shape
    b = w.double().reshape(rows, k // 32, 32)
    amax = b.abs().amax(-1)
    m, e = torch.frexp(amax / 448.0)  # amax / 448 = m * 2^e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)  # an exact power of two keeps its exponent
    e = torch.where(amax == 0, torch.full_like(e, -127), e)
    vals = (b / torch.exp2(e.double()).unsqueeze(-1)).reshape(rows, k).to(torch.float8_e4m3fn)
    return vals, (e + 127).to(torch.uint8)


def _r128c4(byte: torch.Tensor) -> torch.Tensor:
    """R128c4, written here from the layout: 128-row x 4-column tiles, 32 x 4 x 4 atoms."""
    rows, cols = byte.shape
    pr, pc = -(-rows // 128) * 128, -(-cols // 4) * 4
    out = torch.zeros(pr * pc, dtype=torch.uint8)
    for r in range(rows):
        for c in range(cols):
            tile = (r // 128) * (pc // 4) + c // 4
            out[tile * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4] = byte[r, c]
    return out


# --- the levers ---


def test_the_three_mxfp8_levers_are_off_on_both_roles():
    """A generation server's constants name the three levers, off; a context server's do not name
    them. ``weights.dims`` still reads each from the core's levers."""
    ns = G._levers_ns()
    gen, ctx, consts = ns["_levers"]("generation"), ns["_levers"]("context"), ns["_CONSTANTS"]
    dims = ast.unparse(G._function("dims", G._tree(G._WEIGHTS)))
    for key in _MX:
        assert key in ns["_Levers"]._fields, key
        assert consts["generation"][key] is False and key not in consts["context"], key
        assert not getattr(gen, key) and not getattr(ctx, key), key
        assert f"{key}=bool(getattr(levers, '{key}', False))" in dims, key


# --- the weight table ---

_LEVER_ROWS = {
    "mla_mxfp8": ("w1_rows", ("mla_g", "mla_o")),
    "moe_latent_mxfp8": ("w1_rows", ("moe_latent_down",)),
    "draft_mxfp8": (
        "w3_rows",
        tuple(f"d_l{j}_{n}" for j in range(5) for n in ("o", "gate_up", "down")),
    ),
}


@pytest.mark.parametrize("lever", sorted(_LEVER_ROWS))
def test_a_generation_server_table_holds_the_lever_rows_in_bf16(lever):
    """The table a generation server's own levers (``_levers("generation")``) lay out: each MXFP8
    lever's rows are the BF16 rows a context server's table holds, with no MXFP8 pair."""
    ns = _weights_ns()
    gen = G._levers_ns()["_levers"]("generation")
    table, names = _LEVER_ROWS[lever]
    fields = ns["Dims"]._fields
    d = G._dims(ns, **{k: v for k, v in gen._asdict().items() if k in fields})
    assert d.role == "generation" and not getattr(d, lever)
    rows = {r.name: r for r in ns[table](d)}
    for name in names:
        assert name in rows and f"{name}_w" not in rows and f"{name}_s" not in rows, name
        assert rows[name].dtype in (None, torch.bfloat16), (name, rows[name].dtype)
    assert set(names) <= {r.name for r in ns[table](G._dims(ns, role="context"))}


def _bytes(ns: dict, d, rows) -> int:
    with torch.device("meta"):
        return sum(p.numel() * p.element_size() for p in ns["declare"](d, rows).values())


@pytest.mark.parametrize("lever", sorted(_LEVER_ROWS))
def test_the_table_holds_the_lever_rows_once_in_mxfp8(lever):
    ns = _weights_ns()
    table, names = _LEVER_ROWS[lever]
    d, bf16 = G._dims(ns, **{lever: True}), G._dims(ns)
    rows, before = ns[table](d), ns[table](bf16)
    want = []
    for r in before:
        want += [f"{r.name}_w", f"{r.name}_s"] if r.name in names else [r.name]
    assert [r.name for r in rows] == want
    by = {r.name: r for r in rows}
    saved = 0
    for name in names:
        (old,) = [r for r in before if r.name == name]
        w, s = by[f"{name}_w"], by[f"{name}_s"]
        n, k = old.shape(d)
        assert w.shape(d) == (n, k) and w.dtype == torch.float8_e4m3fn
        assert (
            s.shape(d) == (-(-n // 128) * 128 * (-(-(k // 32) // 4) * 4),)
            and s.dtype == torch.uint8
        )
        assert w.src is s.src and w.cut is s.cut and w.layers is s.layers
        i = (old.layer_indices(d) or [None])[0]
        assert w.sources(d, i) == s.sources(d, i) == old.sources(d, i)
        assert w.layer_indices(d) == old.layer_indices(d)
        assert (w.cut is None) == (old.cut is None) and (w.cut is None or w.cut(d) == old.cut(d))
        layers = len(old.layer_indices(d) or [None])
        saved += layers * (2 * n * k - n * k - s.shape(d)[0])

    # Every checkpoint key is still read, once per source; the lever's bytes drop by about half.
    def keys(rs):
        return {k for r in rs for i in (r.layer_indices(d) or [None]) for k, _ in r.sources(d, i)}

    assert keys(rows) == keys(before)
    assert _bytes(ns, bf16, before) - _bytes(ns, d, rows) == saved > 0


@pytest.mark.parametrize("name", ["moe_latent_down", "mla_o"])
def test_a_lever_row_loads_the_checkpoint_mxfp8_recipe(name):
    """Read through the table's own path: the E4M3 values and R128c4 scales of the rule, written
    independently here; the transforms also run on meta tensors (the coverage test's shapes)."""
    ns = _weights_ns()
    lever = "moe_latent_mxfp8" if name.startswith("moe") else "mla_mxfp8"
    d = G._dims(ns, **{lever: True})
    by = {r.name: r for r in ns["w1_rows"](d)}
    w_row, s_row = by[f"{name}_w"], by[f"{name}_s"]
    n, k = w_row.shape(d)
    gen = torch.Generator().manual_seed(11)
    rows = 256  # the rule is per row: a 256-row slice keeps the test short
    src = (torch.randn(rows, k, generator=gen) * 0.02).to(torch.bfloat16)
    src[3, :32] = 0  # an all-zero block
    src[5, 64:96] = 2.0**-3 * 448  # amax / 448 an exact power of two
    ((key, _),) = w_row.sources(d, 3)
    vals = ns["read"]({key: src}, key, None, w_row.transform, None)
    sf = ns["read"]({key: src}, key, None, s_row.transform, None)
    want_v, want_b = _independent_mxfp8(src)
    assert torch.equal(vals.view(torch.uint8), want_v.view(torch.uint8))
    assert torch.equal(sf, _r128c4(want_b))
    with torch.device("meta"):
        meta = torch.empty(n, k, dtype=torch.bfloat16)
        assert tuple(w_row.transform(meta, None).shape) == w_row.shape(d)
        assert tuple(s_row.transform(meta, None).shape) == s_row.shape(d)
        assert w_row.transform(meta, None).dtype == torch.float8_e4m3fn


# --- the views and the projections ---


def test_the_views_hand_out_mxfp8_weights_and_dense_projects_them_pinned():
    calls = []
    ns = G._exec(
        G._tree(),
        ("_Mx", "_mx_view", "_dense", "_moe_latent_down"),
        dict(
            NamedTuple=__import__("typing").NamedTuple,
            Any=object,
            rubin_linear=lambda x, w, tf32: calls.append(("bf16", w, tf32)) or "bf16",
            rubin_mxfp8_linear=lambda x, w, s, pinned=False: calls.append(("mx", w, s, pinned))
            or "mx",
            _moe_latent_gemm=lambda x, w: calls.append(("latent_bf16", w)) or "latent_bf16",
        ),
    )
    a, b, c = torch.zeros(2, 2), torch.zeros(4, dtype=torch.uint8), torch.zeros(2, 2)
    w = {"l3_mla_g_w": a, "l3_mla_g_s": b, "l3_mla_o": c}
    g, o = ns["_mx_view"](w, "l3_mla_g"), ns["_mx_view"](w, "l3_mla_o")
    assert isinstance(g, ns["_Mx"]) and g.w is a and g.s is b and o is c
    x = torch.zeros(1, 2)
    assert ns["_dense"](x, o) == "bf16" and calls[-1] == ("bf16", c, False)
    assert ns["_dense"](x, g) == "mx" and calls[-1] == ("mx", a, b, True)
    assert ns["_dense"](x, g, pinned=False) == "mx" and calls[-1] == ("mx", a, b, False)
    assert ns["_dense"](x, o, pinned=False) == "bf16" and calls[-1] == ("bf16", c, False)
    assert ns["_moe_latent_down"](x, c) == "latent_bf16" and calls[-1] == ("latent_bf16", c)
    assert ns["_moe_latent_down"](x, g) == "mx" and calls[-1] == ("mx", a, b, True)


def test_the_views_and_calls_name_the_lever_rows():
    """``_layer_views`` reads g_proj, o_proj and the MoE weights through ``_mx_view``; every
    generation-server latent_down goes through ``_moe_latent_down``; the drafter's o_proj, gate_up
    and down through ``_dense``."""
    tree = G._tree()
    views = ast.unparse(G._function("_layer_views", tree)).replace("'", '"')
    for text in (
        'g=_mx_view(w, f"l{i}_mla_g")',
        'o=_mx_view(w, f"l{i}_mla_o")',
        '_mx_view(w, f"l{i}_moe_{f}")',
    ):
        assert text in views, text
    for fn in ("_moe_dep_gen_routed", "_moe_dep_gen_streams"):
        body = ast.unparse(G._function(fn, tree))
        assert "_moe_latent_down(x, L.latent_down)" in body
        assert "_moe_latent_gemm(x, L.latent_down)" not in body
    step = ast.unparse(G._function("draft_step", tree))
    assert (
        "_dense(attn, L.o, pinned=False)" in step
        and "_dense(flashinfer_silu_and_mul(_dense(x, L.gate_up)), L.down)" in step
    )


# --- the blocks under the levers: same plan, MXFP8 entries ---


@pytest.mark.parametrize("setting", sorted(G._SETTINGS))
def test_mla_generate_runs_g_and_o_in_mxfp8_on_the_same_streams(setting):
    bf16 = G._mla(G._Plan(), G._SETTINGS[setting])
    mx = G._mla(G._Plan(), G._SETTINGS[setting], mx=True)
    swap = {
        "g_proj@fork": "mx:g_proj@fork",
        "g_proj@main": "mx:g_proj@main",
        "o_proj@main": "mx:o_proj@main",
    }
    assert mx.plan.ops == [swap.get(o, o) for o in bf16.plan.ops]
    assert sum(o.startswith("mx:") for o in mx.plan.ops) == 2
    assert not mx.plan.problems, mx.plan.problems
    assert torch.equal(mx.out, bf16.out)


def _moe_mx_run(case: str, cuda_graph: bool = True) -> SimpleNamespace:
    """``F._run`` with latent_down an ``_Mx`` whose stand-in GEMM is the BF16 product."""
    counts, rank, chunk_tokens = F._CASES[case]
    plan, seen = F._Plan(), {}
    log = F._Logger(plan)
    entries = F._entries(plan, seen)
    weight = {}

    def rubin_mxfp8_linear(x, w, s, pinned=False):
        assert pinned is True and s.dtype == torch.uint8
        plan.quiet += 1
        try:
            out = (x.float() @ weight["bf16"].t().float()).to(torch.bfloat16)
            plan.kernel("mx_latent_down", (x, w, s), out)
            return out
        finally:
            plan.quiet -= 1

    block = F._block(
        F._tree(),
        {
            "torch": F._Torch(plan),
            "logger": log,
            **entries,
            "rubin_mxfp8_linear": rubin_mxfp8_linear,
        },
    )
    gen = torch.Generator().manual_seed(2026)
    layer = F._layer(gen)
    weight["bf16"] = layer.latent_down
    layer.latent_down = block["_Mx"](
        layer.latent_down.to(torch.float8_e4m3fn), torch.zeros(4, dtype=torch.uint8)
    )
    x = torch.randn(counts[rank], F._HIDDEN, generator=gen).to(torch.bfloat16)
    ws = F._workspace(rank, chunk_tokens)
    rt = F._rt(plan, ws)
    st = SimpleNamespace(cuda_graph=cuda_graph, all_rank_num_tokens=counts)
    with F._Record(plan):
        out = block["latent_moe"](layer, F._LAYER, x, st, rt)
    return SimpleNamespace(plan=plan, out=out)


@pytest.mark.parametrize("cuda_graph", [True, False], ids=["graph", "eager"])
@pytest.mark.parametrize("case", sorted(F._CASES))
def test_the_moe_front_runs_latent_down_in_mxfp8_where_it_ran(case, cuda_graph):
    bf16, mx = F._run(case, cuda_graph), _moe_mx_run(case, cuda_graph)
    ops = [o.replace("latent_down@", "mx_latent_down@") for o in bf16.plan.ops]
    assert mx.plan.ops == ops
    assert sum("mx_latent_down@" in o for o in mx.plan.ops) == sum(
        "latent_down@" in o for o in bf16.plan.ops
    )
    for a, b in ((mx.out.routed, bf16.out.routed), (mx.out.shared, bf16.out.shared)):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("fused", [False, True], ids=["q_a_kv_a_off", "q_a_kv_a_fused"])
def test_the_drafter_runs_o_gate_up_and_down_in_mxfp8(fused):
    bf16, mx = G._draft_run(fused), G._draft_run(fused, mx=True)
    lever = {"o", "gate_up", "down"}
    assert mx.gemms == [("mx:" + g) if g in lever else g for g in bf16.gemms]
    assert torch.equal(mx.out, bf16.out)


# --- the pinned tactics ---


def _pinned_tactics() -> dict:
    tree = G._tree(_GEMM_OP)
    (node,) = [
        n
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "PINNED_TACTICS"
    ]
    return ast.literal_eval(node.value)


def test_every_lever_gemm_has_a_split_k_free_pinned_tactic():
    ns = _weights_ns()
    pins = _pinned_tactics()
    shapes = set()
    for lever, (table, names) in _LEVER_ROWS.items():
        d = G._dims(ns, **{lever: True})
        shapes |= {r.shape(d) for r in ns[table](d) if r.name in {f"{n}_w" for n in names}}
    assert shapes == set(pins), (sorted(shapes), sorted(pins))
    for t in pins.values():
        assert t[0] == "base" and len(t) == 9 and t[8] == 1, t
        assert math.prod(t[3]) <= 16 and t[4] is True, t  # a cluster the kernel takes; swap_ab
