# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4WC_HOIST CPU checks: no GPU. The hoisted block package (kernels/hoist/tp4c1) holds every module the seams bend,
with the names the seams patch; the post's C++ headers are this target's csrc; the seams, the block adapter, the op and
the TP4 wiring agree with each other (read off the sources)."""

import ast
import os
import re
import struct

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 7))
FAM = os.path.join(ROOT, "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3")
FAMPKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
TP4 = os.path.join(FAM, "minimax_m3_1_flash_nvfp4__sm_107__tp4")
TP4PKG = f"{FAMPKG}.minimax_m3_1_flash_nvfp4__sm_107__tp4"
HOIST = os.path.join(TP4, "kernels/hoist")
TP4C1 = os.path.join(HOIST, "tp4c1")
PKG = f"{TP4PKG}.kernels.hoist.tp4c1"
KDIR = os.path.join(TP4, "csrc")
OP = os.path.join(KDIR, "hoistPostTp4Op.cpp")
# passB preflight P3_HYB_engine_push (every rank)
PASSB_HYB_ENGINE = {
    "AR_G3": 0,
    "AR_V3": 0,
    "AR_CL": 8,
    "AR_ND": 4,
    "AR_PRE": 1,
    "DL": 1,
    "SHARED": 1,
    "NSH": 24,
    "LAG": 6,
    "FC1D": 8,
    "XSTAGE": 0,
    "EXT_ROUTE": 1,
    "FUSED_ROUTER": 1,
}
# Imports in the block package that resolve outside it, all on branches the block never takes (recorded, not
# carried): flashinfer_ugpu (expert_shards' fallback cut; the block registers its shards first), the expert_local
# engine family (router_hoist_host's C2 builds), m3block / h12r_slot (M3-mini harness helpers in router_hoist_host).
UNREACHED = {f"{PKG}.kernels.moe.engine.expert_local_host"}


def module_consts(path, names):
    out = {}
    for node in ast.parse(open(path).read()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            if node.targets[0].id in names:
                try:
                    out[node.targets[0].id] = ast.literal_eval(node.value)
                except (
                    ValueError
                ):  # integer arithmetic (shifts, products, negation) of literals only
                    expr = ast.Expression(node.value)
                    ok = all(
                        isinstance(
                            x,
                            (
                                ast.Expression,
                                ast.BinOp,
                                ast.UnaryOp,
                                ast.Constant,
                                ast.operator,
                                ast.unaryop,
                            ),
                        )
                        for x in ast.walk(expr)
                    )
                    out[node.targets[0].id] = (
                        eval(compile(expr, path, "eval"), {"__builtins__": {}}) if ok else None
                    )
    return out


# The block modules the seams bend, with the names they patch (seams.install / seams.load).
SEAMED = {
    "toolchain/paths.py": {"csrc_root"},
    "kernels/comm/allreduce_norm_quant/route_warp_tp4_fc1_per_token_host.py": {"src_hash", "dist"},
    "kernels/comm/allreduce_norm_quant/route_warp_tp4_host.py": {"dist"},
    "kernels/comm/allreduce_norm_quant/router_exchange.py": {"dist"},
    "kernels/comm/allreduce_norm_quant/lamport_allreduce_host.py": {"dist"},
    "kernels/moe/router/fused_o_proj_tp4_host.py": {"dist", "_rank"},
    "kernels/moe/engine/router_hoist_host.py": {"_rank_world"},
    "kernels/moe/engine/fused_die_local_host.py": {"_dev_b2lag", "_op_dl"},
}


def _top_names(path):
    names = set()
    for n in ast.parse(open(path).read()).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names


def test_block_package_holds_every_seamed_module_and_name():
    for rel, names in SEAMED.items():
        path = os.path.join(TP4C1, rel)
        assert os.path.isfile(path), rel
        assert names <= _top_names(path), (rel, names - _top_names(path))
    seams = open(os.path.join(HOIST, "seams.py")).read()
    for rel in SEAMED:
        assert ".tp4c1." + rel[:-3].replace("/", ".") in seams, rel


def test_post_headers_are_this_targets_csrc():
    """The post's translation unit includes the routing-warp header, which includes the boundary consumer, which
    includes the Lamport exchange; all three are this target's csrc files."""
    chain = (
        ("hoistPostTp4.cu", "route_warp_tp4_fc1_per_token.cuh"),
        ("route_warp_tp4_fc1_per_token.cuh", "allreduce_norm_quant.cuh"),
        ("allreduce_norm_quant.cuh", "lamport_exchange.cuh"),
    )
    for src, inc in chain:
        text = open(os.path.join(KDIR, src)).read()
        assert re.search(r'#include "(?:[^"]*/)?' + re.escape(inc) + '"', text), (src, inc)
        assert os.path.isfile(os.path.join(KDIR, inc)), inc


def _imports(path, mod):
    tree = ast.parse(open(path).read())
    is_pkg = path.endswith("__init__.py")
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name, None
        elif isinstance(n, ast.ImportFrom):
            parts = mod.split(".")
            pkg = parts if is_pkg else parts[:-1]
            base = (
                ".".join(pkg[: len(pkg) - (n.level - 1)] + ([n.module] if n.module else []))
                if n.level
                else n.module
            )
            yield base, [a.name for a in n.names]


def _resolves(mod):
    base = os.path.join(ROOT, *mod.split("."))
    return os.path.isfile(base + ".py") or os.path.isfile(os.path.join(base, "__init__.py"))


def test_block_package_is_closed():
    missing = set()
    for d, _, fs in os.walk(TP4C1):
        for f in fs:
            if not f.endswith(".py"):
                continue
            p = os.path.join(d, f)
            rel = os.path.relpath(p, ROOT)[:-3].replace(os.sep, ".")
            mod = rel[: -len(".__init__")] if rel.endswith(".__init__") else rel
            for base, names in _imports(p, mod):
                if not base or not (base == PKG or base.startswith(PKG + ".")):
                    continue
                if not _resolves(base):
                    missing.add(base)
                for n in names or ():
                    if not _resolves(base) and not _resolves(f"{base}.{n}"):
                        missing.add(f"{base}.{n}")
            for s in re.findall(r"\"(" + re.escape(PKG) + r"\.[A-Za-z0-9_.]+)", open(p).read()):
                if not _resolves(s):
                    missing.add(s)
    assert missing == UNREACHED


def test_seams_constants():
    c = module_consts(
        os.path.join(HOIST, "seams.py"),
        {
            "HDR_HASH",
            "B2LAG",
            "PUSH_PARAMS_FMT",
            "EMPTY_WORD",
            "CLUSTER_SZ",
            "HANG_CAP_NS",
        },
    )
    assert c["HDR_HASH"] == "40c1042cbf7d"
    lr4q = open(os.path.join(TP4C1, "families/router/lr4_prr4_q.py")).read()
    assert f'HDR_HASH = "{c["HDR_HASH"]}"' in lr4q
    seams = ast.parse(open(os.path.join(HOIST, "seams.py")).read())
    (engine,) = [
        n
        for n in seams.body
        if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["ENGINE_KERNEL"]
    ]
    assert "'kernels', 'm3ll', 'moe_engine_kernel.py'" in ast.unparse(engine.value)
    assert os.path.isfile(os.path.join(FAM, "kernels/m3ll/moe_engine_kernel.py"))
    assert c["B2LAG"] == {"b2buf": 2, "stages": 6, "lag": 6}
    assert c["EMPTY_WORD"] == -(2**31) and c["CLUSTER_SZ"] == 8
    assert c["HANG_CAP_NS"] == (1 << 31) * 200  # the reference families/moe/_common.HANG_CAP_NS
    fmt = "<" + c["PUSH_PARAMS_FMT"]
    assert struct.calcsize(fmt) == 96  # sizeof(PushParamsV3): the .cu static_asserts the same
    b = struct.pack(fmt, *range(1, 9), 99, 3, 4, 8, 4, 4)
    assert struct.unpack_from("8q", b, 0) == tuple(
        range(1, 9)
    )  # router_exchange.prx_spec / push_spec offsets
    assert struct.unpack_from("q", b, 64)[0] == 99
    assert struct.unpack_from("5i", b, 72) == (3, 4, 8, 4, 4)


def test_block_engine_is_the_measured_hyb_engine():
    c = module_consts(
        os.path.join(HOIST, "tp4_hoist_block.py"), {"HYB_ENGINE", "HYB_ENGINE_OVERLAY", "ROWS"}
    )
    assert c["HYB_ENGINE"] == PASSB_HYB_ENGINE
    assert c["HYB_ENGINE_OVERLAY"] == {"ar_v3": 0, "ar_cl": 8}
    assert c["ROWS"] == 8


def _schema_args():
    src = open(OP).read()
    m = re.search(
        r'"minimax_m3_sm107tp4_hoist_post\((.*?)\) -> \(\)"', src.replace('"\n        "', ""), re.S
    )
    assert m, "schema"
    return [a.strip().split()[-1].split("=")[0] for a in m.group(1).split(",")]


def test_op_schema_matches_the_reference_extension_signature():
    tree = ast.parse(open(os.path.join(HOIST, "seams.py")).read())
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "allreduce_norm_quant_route"
    )
    peer = [a.arg for a in fn.args.args][1:]  # minus self
    assert peer == [
        "pp",
        "rows",
        "residual",
        "gamma",
        "eps",
        "gemma",
        "out_norm",
        "out_res",
        "out_q",
        "out_sf",
        "cl",
        "rx_bases",
        "route",
        "rctr",
        "bias",
        "tab",
        "delay_ns",
        "lg_dbg",
        "tdbg",
        "repl",
        "npart",
        "rxc",
        "deadline_ns",
        "hang_dbg",
        "roll",
        "x4_q",
        "x4_sf",
        "x4_sfsc",
        "prr",
        "rx2",
        "pt",
        "x4_pts",
        "pwait",
    ]
    assert _schema_args() == peer


def test_op_build_wiring():
    cm = open(os.path.join(FAM, "csrc/CMakeLists.txt")).read()
    assert re.search(
        r"add_modeling_v2_target\(\s*minimax_m3 minimax_m3_1_flash_nvfp4__sm_107__tp4\s+ARCHS 100f 107a\s+"
        r"SOURCES \*\.cu \*\.cpp",
        cm,
    )
    assert "TRTLLM_MV2_LIBRARY_FRAGMENT(m)" in open(OP).read()
    fakes = open(os.path.join(TP4, "fake_ops.py")).read()
    assert '_in_place("minimax_m3_sm107tp4_hoist_post")' in fakes
    seams = open(os.path.join(HOIST, "seams.py")).read()
    assert "native_ops.minimax_m3_sm107tp4_hoist_post(" in seams


def test_tp4_wiring_and_kill_switches():
    src = open(os.path.join(TP4, "modeling.py")).read()
    assert (
        'TP4WC_HOIST = not (m31_disabled("TP4_WHOLECHAIN") or m31_disabled("TP4WC_HOIST"))' in src
    )
    assert 'TP4WC_HOIST_PATH = "m31_tp4wc_hoist"' in src
    assert "TP4WC_HOIST_ROWS = 8" in src
    assert "TP4WC_HOIST_HANG_CAP_NS = (1 << 30) * 200" in src
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"_prebuild_hoist", "hoist_block", "_hoist_arm", "_hoist_take", "_hoist_layer"} <= names
    assert "self._prebuild_hoist()" in src
    assert "hoist=hoist is not None," in src and "out_residual = hoist.residual_mid" in src
    assert src.count("return self._run_front(i, step, push, hoist)") == 1


def _bf16(x):
    """round-to-nearest-even fp32 -> bf16 -> fp32 (pure Python)."""
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


@pytest.mark.parametrize("lo,hi", [(0.5, 1.0), (1.0, 2.0)])
def test_unfold_gemma_is_exact_on_the_folded_range(lo, hi):
    """Every bf16 folded weight in [0.5, 2) (Gemma 1 + w for |w| < 1) gives bf16(f - 1) with (f - 1) + 1 == f in fp32:
    the reference post's fp32 1 + w reproduces this target's folded weight bit for bit
    (tp4_hoist_block.unfold_gemma)."""
    f, n = lo, 0
    step = 2.0**-8 if lo < 1.0 else 2.0**-7
    while f < hi:
        raw = _bf16(f - 1.0)
        assert struct.unpack("<f", struct.pack("<f", raw + 1.0))[0] == f, f
        f += step
        n += 1
    assert n == 128


# ------------------------------------------------------------------------------------- relative-import regressions
def _relative_imports(path, pkg):
    for n in ast.walk(ast.parse(open(path).read())):
        if isinstance(n, ast.ImportFrom) and n.level:
            yield n.lineno, "." * n.level + (n.module or ""), pkg


def test_every_relative_import_resolves_in_the_tree():
    """A relative import one level short resolves to a package that does not exist, the ImportError is taken as a
    failed build, and DL never engages. Every relative import of kernels/m3ll/**, of the hoisted block package and of
    the TP4 modeling must name a module or package of this tree."""
    import importlib.util

    files = [os.path.join(TP4, "modeling.py")]
    for top in (os.path.join(FAM, "kernels/m3ll"), HOIST):
        for d, _, fs in os.walk(top):
            files += [os.path.join(d, f) for f in fs if f.endswith(".py")]
    broken, n = [], 0
    for p in files:
        root = ROOT
        mod = os.path.relpath(p, root)[:-3].replace(os.sep, ".")
        pkg = mod[: -len(".__init__")] if p.endswith("__init__.py") else mod.rsplit(".", 1)[0]
        for line, spec, pkg_ in _relative_imports(p, pkg):
            n += 1
            tgt = importlib.util.resolve_name(spec, pkg_)
            base = os.path.join(root, *tgt.split("."))
            if not (
                os.path.isfile(base + ".py") or os.path.isfile(os.path.join(base, "__init__.py"))
            ):
                broken.append(f"{os.path.relpath(p, ROOT)}:{line} {spec} -> {tgt}")
    assert n > 50 and not broken, broken


def test_domain_tensor_crosses_the_locality_import(monkeypatch):
    """The DL = 1 build path reaches the family locality module: tp4_hoist_block.domain_tensor with a non-None
    LocalityMap imports kernels/m3ll/locality (a stub here, keyed by the module's real dotted name, which must be a file
    of this tree) and calls its _Mapping."""
    import importlib.util
    import sys
    import types

    hoist_pkg = f"{TP4PKG}.kernels.hoist"
    loc_name = f"{FAMPKG}.kernels.m3ll.locality"
    assert os.path.isfile(os.path.join(ROOT, *loc_name.split(".")) + ".py")

    class Reached(Exception):
        pass

    class _Mapping:
        def __init__(self, *args, **kwargs):
            raise Reached(args)

    loc = types.ModuleType(loc_name)
    loc._Mapping = _Mapping

    def ensure_pkg(name):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [os.path.join(ROOT, *name.split("."))]
            monkeypatch.setitem(sys.modules, name, m)
        return sys.modules[name]

    for full in (hoist_pkg, loc_name.rsplit(".", 1)[0]):
        parts = full.split(".")
        for k in range(1, len(parts) + 1):
            ensure_pkg(".".join(parts[:k]))
    monkeypatch.setitem(sys.modules, loc_name, loc)
    monkeypatch.setattr(sys.modules[loc_name.rsplit(".", 1)[0]], "locality", loc, raising=False)
    # seams imports the target's native ops and switch registry: stand-ins (no library, no environment read).
    monkeypatch.setattr(sys.modules[TP4PKG], "native_ops", types.SimpleNamespace(), raising=False)
    switches = types.ModuleType(f"{TP4PKG}.m31_switches")
    switches.m31_disabled = lambda item_id: False
    monkeypatch.setitem(sys.modules, switches.__name__, switches)
    try:
        import torch  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    for sub in ("seams", "tp4_hoist_block"):
        name = f"{hoist_pkg}.{sub}"
        spec = importlib.util.spec_from_file_location(name, os.path.join(HOIST, sub + ".py"))
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
    blk = sys.modules[f"{hoist_pkg}.tp4_hoist_block"]
    src = types.SimpleNamespace(numel=lambda: 1000, element_size=lambda: 1)
    with pytest.raises(Reached) as got:
        blk.domain_tensor(types.SimpleNamespace(granularity=2 << 20, device=0), src, 1, [])
    assert got.value.args[0] == (0, [(2 << 20, 1)])


def test_engagement_line_carries_dl_and_placement():
    src = open(os.path.join(HOIST, "tp4_hoist_block.py")).read()
    assert "dl {c['DL']} placed {int(block.placed)}" in src
    assert "placed=locmap is not None" in src


def test_oproj_view_is_the_reference_hosts_shape():
    """The o_proj view the block hands the reference host is the served TP4 per-rank shape, which is the peer
    host's (H, KO): weights.py o_w = (hidden, q_width), q_width = heads x head_dim with heads = 64 / tp; the
    hand-off buffer is [rows, KO]; and every fixture of the block's GPU UTs uses that KO (a [6144, 6144] fixture
    reached fused_o_proj_tp4_host.py:166 on GPU)."""
    peer = module_consts(
        os.path.join(TP4C1, "kernels/moe/router/fused_o_proj_tp4_host.py"),
        {"H", "KO"},
    )
    ours = module_consts(os.path.join(HOIST, "tp4_hoist_block.py"), {"H", "KO", "ROWS"})
    served_heads, head_dim, tp, hidden = (
        64,
        128,
        4,
        6144,
    )  # M3.1 config.json: num_attention_heads, head_dim
    assert (
        (peer["H"], peer["KO"])
        == (hidden, served_heads // tp * head_dim)
        == (ours["H"], ours["KO"])
        == (6144, 2048)
    )
    wsrc = open(os.path.join(TP4, "weights.py")).read()
    assert '"o_w",\n            shape=lambda d: (d.hidden, d.q_width),' in wsrc
    assert "q_width=heads * head_dim," in wsrc
    assert '"o_s",\n            shape=lambda d: (d.hidden * (d.q_width // MX_BLOCK),),' in wsrc
    blk = open(os.path.join(HOIST, "tp4_hoist_block.py")).read()
    assert "tuple(o_w.shape) != (H, KO)" in blk and "o_s.numel() != H * (KO // 32)" in blk
    assert "oq.shape[1] != KO" in blk and "(ra4.H, ra4.KO) != (H, KO)" in blk
    gpu = os.path.join(os.path.dirname(__file__), "moe/test_modeling_v2_m31_vr_tp4_hoist.py")
    tree = ast.parse(open(gpu).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [t.id for t in node.targets[0].elts]
            if "KO" in names:
                assert ast.literal_eval(node.value)[names.index("KO")] == 2048
                break
    else:
        raise AssertionError("the 4-rank UT names no KO")


def _defs(path, names, env):
    """exec the named top-level function defs of a source file into env (stdlib-only bodies)."""
    tree = ast.parse(open(path).read())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in body) == sorted(names), path
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), env)
    return env


def test_b2lag_switch_reaches_seam_7():
    """B2LAG is default on (seam 7 returns the HYB schedule); TRTLLM_M31_DISABLE_TP4WC_B2LAG=1 (the target's switch
    registry) restores the reference's shipped schedule."""
    tree = ast.parse(open(os.path.join(HOIST, "seams.py")).read())
    stmts = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Assign)}
    assert "B2LAG_OFF = m31_disabled('TP4WC_B2LAG')" in stmts
    assert "dl._dev_b2lag = (lambda: None) if B2LAG_OFF else lambda: dict(B2LAG)" in stmts
    (assign,) = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["B2LAG_OFF"]
    ]
    for disabled in (False, True):
        scope = {"m31_disabled": lambda item_id, _d=disabled: _d and item_id == "TP4WC_B2LAG"}
        exec(compile(ast.Module(body=[assign], type_ignores=[]), "seams.py", "exec"), scope)
        assert scope["B2LAG_OFF"] is disabled
    registry = open(os.path.join(TP4, "m31_switches.py")).read()
    assert '"TP4WC_B2LAG":' in registry


def test_engine_check_follows_the_b2lag_switch():
    """With B2LAG on the engine must be exactly HYB_ENGINE; with it off exactly HYB_ENGINE_B2LAG_OFF, the
    shipped
    engine measured on all 4 ranks of the pair (LAG 2, FC1D 4; 0040 required FC1D 8 and rejected it on every layer). The
    engaged line reports the setting."""
    path = os.path.join(HOIST, "tp4_hoist_block.py")
    c = module_consts(path, {"HYB_ENGINE", "HYB_ENGINE_B2LAG_LAG", "HYB_ENGINE_B2LAG_OFF"})
    assert c["HYB_ENGINE_B2LAG_LAG"] == 6 == c["HYB_ENGINE"]["LAG"]
    assert c["HYB_ENGINE_B2LAG_OFF"] == dict(PASSB_HYB_ENGINE, LAG=2, FC1D=4)
    for off in (False, True):
        env = {
            "Dict": dict,
            "HYB_ENGINE": dict(c["HYB_ENGINE"]),
            "HYB_ENGINE_B2LAG_OFF": dict(c["HYB_ENGINE_B2LAG_OFF"]),
            "seams": type("S", (), {"B2LAG_OFF": off}),
        }
        m = _defs(path, ["engine_matches"], env)["engine_matches"]
        hyb, shipped = dict(c["HYB_ENGINE"]), dict(c["HYB_ENGINE_B2LAG_OFF"])
        assert m(hyb) is (not off)
        assert m(shipped) is off
        assert m(dict(hyb, LAG=2)) is False  # FC1D 8 at lag 2: neither engine
        assert m(dict(shipped, NSH=48)) is False
        assert m(dict(hyb, FC1D=4)) is False
    src = open(path).read()
    assert "if not engine_matches(cfg):" in src
    assert "b2lag {'off' if seams.B2LAG_OFF else 'on'}" in src
