# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2-wide commgemv track CPU controls (no CUDA, no torch import): the row bounds and kill switches of TP2WIDE_PUSH,
TP2WIDE_GEMVCFG (cfg only), DRAFT_AR_NORM_CLUSTER_WIDE, TP2WIDE_INPUT9 and FRONT_CAPFORK_WIDE, the multicast
all-reduce's certification cells for the new rows, the drafter-form bucket gate and the S9B-MC build probe. Each
switch statement is executed from the actual source with stub switch readers."""

import ast
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[7]
TARGET = ROOT / (
    "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2"
)
MODELING = TARGET / "modeling.py"
DSPARK = TARGET / "dspark.py"
AR_OP = TARGET / "ops/comm/m3_mc_allreduce_norm.py"
AR_KERNEL = TARGET / "kernels/m3_mc_allreduce.py"
CU = ROOT / (
    "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/csrc/m3/sm107Tp4McAllreduceNorm.cu"
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _assign(tree: ast.Module, name: str) -> ast.stmt:
    for node in tree.body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node
    raise AssertionError(f"no module-level assignment of {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name}")


def _run(stmts, env: dict) -> dict:
    module = ast.Module(body=list(stmts), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "commgemv", "exec"), env)
    return env


def _switches(*killed: str) -> dict:
    return {
        "m31_disabled": lambda i: i in killed,
    }


# ------------------------------------------------------------------ W2 TP2WIDE_PUSH


@pytest.mark.parametrize(
    "killed,mc,want",
    [
        ((), True, 32),
        (("TP2WIDE_PUSH",), True, 8),
        (("TP2WIDE_COMMGEMV",), True, 8),
        ((), False, 0),
        (("TP2WIDE_PUSH",), False, 0),
    ],
)
def test_attn_push_bound_and_switches(killed, mc, want) -> None:
    t = _tree(MODELING)
    env = {"MC_AR_TP4": mc, **_switches(*killed)}
    names = (
        "TP2WIDE_PUSH",
        "ATTN_PUSH_BASE_TOKENS",
        "ATTN_PUSH_WIDE_TOKENS",
        "ATTN_PUSH_MAX_TOKENS",
    )
    _run([_assign(t, n) for n in names], env)
    assert env["ATTN_PUSH_MAX_TOKENS"] == want
    assert env["ATTN_PUSH_BASE_TOKENS"] == (8 if mc else 0)


def test_dense_push_stays_at_eight_rows() -> None:
    """The dense down projection's bf16 SwiGLU input declines the bf16 GEMV past 8 rows (24 k-tiles): no pushing
    producer exists at 9..32, so DENSE_PUSH is not widened."""
    t = _tree(MODELING)
    env = {"MC_AR_TP4": True, "DENSE_PUSH_TP4": True}
    _run([_assign(t, "DENSE_PUSH_MAX_TOKENS")], env)
    assert env["DENSE_PUSH_MAX_TOKENS"] == 8


def test_wide_push_miss_is_soft_past_the_base_rows() -> None:
    """In DecodeTarget._layer an armed push that the o_proj did not take is a miss (raises under require) only up to
    ATTN_PUSH_BASE_TOKENS rows; past them it is ``uncovered`` and the unpushed boundary runs (same result)."""
    t = _tree(MODELING)
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "DecodeTarget")
    layer = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_layer")
    src = ast.unparse(layer)
    soft = src.index("if armed is not None and num_tokens > ATTN_PUSH_BASE_TOKENS:")
    assert src.index("uncovered(ATTN_PUSH.path", soft) < src.index("elif armed is not None:", soft)
    assert src.index("missed(ATTN_PUSH.path", soft) > src.index("elif armed is not None:", soft)
    assert "_note_wide_push(core, num_tokens, armed[4])" in src


def _cells() -> list:
    t = _tree(AR_OP)
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "M3McAllreduceNorm")
    stmt = next(
        n
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "CELLS"
    )
    env = {"Cell": lambda why, spec: dict(spec)}
    return list(eval(compile(ast.Expression(stmt.value), "cells", "eval"), env))


def _bucket(m: int) -> int:
    return 8 if m <= 8 else 16 if m <= 16 else 48


def test_prepushed_cells_certify_every_wide_push_row_in_its_bucket() -> None:
    cells = _cells()
    pushed = {c["m"]: c for c in cells if c.get("prepushed") == 2 and not c.get("drafter")}
    assert set(range(1, 33)) <= set(pushed), sorted(set(range(1, 33)) - set(pushed))
    for m in range(1, 33):
        assert pushed[m]["maxt"] == _bucket(m), (m, pushed[m])
        assert pushed[m]["opts"] in ((7, 15) if m <= 8 else (15,)), (m, pushed[m])


def test_drafter_cells_cover_the_wide_buckets_with_option_31() -> None:
    cells = [c for c in _cells() if c.get("drafter") and c["opts"] in (23, 31)]
    wide = {c["m"]: c for c in cells if c["maxt"] in (16, 48)}
    assert {14, 21, 28, 32} <= set(wide)
    for m, c in wide.items():
        assert c["maxt"] == _bucket(m) and c["opts"] == 31, c


def test_the_kernel_build_instantiates_option_31_at_buckets_16_and_48() -> None:
    src = CU.read_text()
    assert "(maxTokens == 16 || maxTokens == 48) && (opts == 15 || opts == 31 || opts == 47)" in src
    for bucket in (16, 48):
        assert f"launch<kCluster, kThreads, {bucket}, 31>(params, pdl, stream);" in src


# ------------------------------------------------------------------ W9 S9B_MC_WIDE


class _T:
    """A 2-D tensor stand-in: shape, dtype, contiguity."""

    def __init__(self, rows: int, cols: int = 6144, dtype: str = "bf16"):
        self.shape = (rows, cols)
        self.dtype = dtype

    def dim(self):
        return 2

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self


def _ar_wrapper(capturing: bool = False):
    """kernels/m3_mc_allreduce.m3_mc_allreduce_norm with its op replaced by a recorder."""
    t = _tree(AR_KERNEL)
    calls = []
    torch = types.SimpleNamespace(
        bfloat16="bf16", cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: capturing)
    )
    env = {
        "torch": torch,
        "_allreduce_norm_op": lambda *args: calls.append(args) or ["normed", "updated"],
        "_capture_args": lambda capture, drafter: (None, None, 0.0),
    }
    names = (
        "MAX_TOKENS",
        "AR_OPTS_BY_M",
        "AR_OPTS_BY_BUCKET",
        "AR_FIN_OPTS_BY_BUCKET",
        "DRAFTER_WIDE_BUCKETS",
        "_OPT_ONESHOT_CLUSTER",
        "_OPT_ROW_CARVEOUT",
        "_OPT_DRAFTER_CLUSTER",
        "_OPT_CLUSTER_BARRIER",
    )
    _run(
        [_assign(t, n) for n in names]
        + [_function(t, n) for n in ("_ar_opts", "_bucket", "m3_mc_allreduce_norm")],
        env,
    )
    return env, calls


class _Bucket:
    def __init__(self, maxt: int, world: int):
        self.max_tokens, self.world_size, self.hidden, self.rank = maxt, world, 6144, 0
        self.uc = object()
        self.mc = types.SimpleNamespace(data_ptr=lambda: 0)

    def flags_for(self, cluster):
        return object()


class _Buckets:
    def __init__(self, world: int):
        self.ws = [_Bucket(b, world) for b in (8, 16, 48)]

    def for_tokens(self, m):
        return next((w for w in self.ws if m <= w.max_tokens), None)


@pytest.mark.parametrize("m", [9, 14, 16, 17, 21, 28, 32])
def test_drafter_form_takes_the_wide_buckets_only_when_asked_at_world_2(m) -> None:
    env, calls = _ar_wrapper()
    env["McBuckets"] = _Buckets
    fn = env["m3_mc_allreduce_norm"]
    # _bucket dispatches on McBuckets: point it at the double.
    env["_bucket"] = lambda ws, n: ws.for_tokens(n)
    x, res = _T(m), _T(m)
    assert fn(x, res, None, 1e-6, _Buckets(2), drafter_norm=True) is None
    assert not calls
    out = fn(x, res, None, 1e-6, _Buckets(2), drafter_norm=True, drafter_wide=True)
    assert out == ("normed", "updated")
    args = calls[-1]
    maxt, opts = args[12], args[15]
    assert (maxt, opts) == (_bucket(m), 31), (maxt, opts)
    # World 4 (the row kernel's drafter form) and the cluster drafter option stay 8-token only.
    assert fn(x, res, None, 1e-6, _Buckets(4), drafter_norm=True, drafter_wide=True) is None
    assert (
        fn(
            x,
            res,
            None,
            1e-6,
            _Buckets(2),
            drafter_norm=True,
            drafter_wide=True,
            drafter_cluster=True,
        )
        is None
    )
    assert len(calls) == 1


def test_drafter_form_at_eight_rows_is_unchanged() -> None:
    env, calls = _ar_wrapper()
    env["_bucket"] = lambda ws, n: ws.for_tokens(n)
    fn = env["m3_mc_allreduce_norm"]
    for m in range(1, 9):
        assert fn(_T(m), _T(m), None, 1e-6, _Buckets(2), drafter_norm=True) is not None
        assert calls[-1][12] == 8 and calls[-1][15] == (23 if m in (1, 2, 7, 8) else 31)


@pytest.mark.parametrize(
    "killed,want",
    [
        ((), 32),
        (("DRAFT_AR_NORM_CLUSTER_WIDE",), 8),
        (("TP2WIDE_COMMGEMV",), 8),
        (("DRAFT_AR_NORM_CLUSTER",), 8),
    ],
)
def test_s9b_mc_bound_and_switches(killed, want) -> None:
    t = _tree(DSPARK)
    env = _switches(*killed)
    names = ("S9B_MC", "S9B_MC_BASE_ROWS", "S9B_MC_WIDE", "S9B_MC_MAX_ROWS")
    _run([_assign(t, n) for n in names], env)
    assert env["S9B_MC_MAX_ROWS"] == want


def _s9b_op(behaviour, capturing=None):
    """dspark._s9b_mc_op (+ _s9b_mc_wide_ready) over a recording multicast wrapper; ``behaviour(rows, wide)`` returns
    the wrapper's result or raises."""
    t = _tree(DSPARK)
    capturing = [False] if capturing is None else capturing
    log = []
    calls = []
    work = []

    def ar(partial, residual, w, eps, ws, use_pdl, drafter_norm, drafter_wide):
        calls.append((partial.shape[0], drafter_wide))
        return behaviour(partial.shape[0], drafter_wide)

    env = {
        "S9B_MC_BASE_ROWS": 8,
        "S9B_MC_MAX_ROWS": 32,
        "_S9B_MC_WIDE_STATE": {"ok": None},
        "m3_mc_allreduce": types.SimpleNamespace(
            get_workspace=lambda mapping, h: object(), m3_mc_allreduce_norm=ar
        ),
        "get_env_enable_pdl": lambda: True,
        "logger": types.SimpleNamespace(info=log.append, warning=log.append),
        "m3_report": types.SimpleNamespace(work=lambda item, detail="": work.append(item)),
        "torch": types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_current_stream_capturing=lambda: capturing[0])
        ),
    }
    _run([_function(t, "_s9b_mc_op"), _function(t, "_s9b_mc_wide_ready")], env)
    op = env["_s9b_mc_op"](types.SimpleNamespace(tp_rank=0, tp_size=2))
    return op, env, calls, log, work


def test_s9b_mc_wide_probe_on_a_build_without_the_instances() -> None:
    capturing = [False]

    def behaviour(rows, wide):
        if wide:
            raise RuntimeError(
                "minimax_m3_sm107_tp4_mc_allreduce_norm: unsupported (bucket 16, opts 31)"
            )
        return ("normed", "updated")

    op, env, calls, log, work = _s9b_op(behaviour, capturing)
    assert op(_T(7), _T(7), None, 1e-6) == ("normed", "updated")
    assert op(_T(14), _T(14), None, 1e-6) is None  # declined: S9b's fused path runs
    assert env["_S9B_MC_WIDE_STATE"]["ok"] is False
    assert op(_T(21), _T(21), None, 1e-6) is None and calls[-1][0] == 14  # no further wide attempt
    assert any("S9B_MC_WIDE off" in m for m in log)
    assert "S9B_MC_WIDE" not in work
    with pytest.raises(RuntimeError):  # any other failure is not swallowed
        _s9b_op(lambda r, w: (_ for _ in ()).throw(RuntimeError("boom")))[0](
            _T(14), _T(14), None, 1e-6
        )


def test_s9b_mc_wide_engages_and_waits_for_an_eager_decision() -> None:
    capturing = [True]
    op, env, calls, log, work = _s9b_op(lambda rows, wide: ("normed", "updated"), capturing)
    assert op(_T(28), _T(28), None, 1e-6) is None and not calls  # unknown while capturing: decline
    capturing[0] = False
    assert op(_T(28), _T(28), None, 1e-6) == ("normed", "updated")
    assert env["_S9B_MC_WIDE_STATE"]["ok"] is True and calls[-1] == (28, True)
    capturing[0] = True
    assert op(_T(14), _T(14), None, 1e-6) == ("normed", "updated")  # known good: captured too
    assert work.count("S9B_MC_WIDE") == 2
    assert sum("S9B_MC_WIDE engaged" in m for m in log) == 1
    assert op(_T(33), _T(33), None, 1e-6) is None  # past the bound


# ------------------------------------------------------------------ W5 TP2WIDE_GEMVCFG


def _gemvcfg(table: dict, killed=()):
    t = _tree(MODELING)
    shared = {(4992, 6144): (4, 12), (6144, 4096): (2, 12), (3072, 6144): (4, 12)}
    mx = types.SimpleNamespace(
        get_config=lambda n, k, m=None: shared.get((n, k)), _CONTIG=True, _NACC=1
    )
    notes = []
    env = {
        **_switches(*killed),
        "mx_gemv": mx,
        "Dict": dict,
        "Tuple": tuple,
        "Optional": __import__("typing").Optional,
        "GEMV_QKV_CFG_MAX_ROWS": 8,
        "GEMV_QKV_SPLIT8_TP4": True,
        "GEMV_OPROJ_STRIDED_TP4": True,
        "MX_GEMV_CFG": {(4992, 6144): (8, 8)},
        "MX_GEMV_CFG_STRIDED": {(6144, 4096): (2, 8, 0)},
        "_note_wide_gemvcfg": lambda *a: notes.append(a),
    }
    stmts = [
        _assign(t, "TP2WIDE_GEMVCFG"),
        _assign(t, "GEMV_WIDE_CFG_MAX_ROWS"),
        _function(t, "_wide_mma"),
        _function(t, "_wide_cfg_is_order_preserving"),
        _function(t, "mx_gemv_cfg"),
    ]
    loop = next(
        n for n in t.body if isinstance(n, ast.For) and "MX_GEMV_CFG_WIDE" in ast.unparse(n.iter)
    )
    _run(stmts, env)
    env["MX_GEMV_CFG_WIDE"] = table
    _run([loop], env)
    return env, notes


def test_wide_gemv_cfg_table_is_empty_until_the_mb_fills_it() -> None:
    t = _tree(MODELING)
    assert ast.literal_eval(_assign(t, "MX_GEMV_CFG_WIDE").value) == {}
    env, notes = _gemvcfg({})
    for rows in range(9, 49):
        assert env["mx_gemv_cfg"](4992, 6144, rows) is None
    assert env["mx_gemv_cfg"](4992, 6144, 4) == (8, 8)
    assert not notes


def test_wide_gemv_cfg_admits_order_preserving_entries_only() -> None:
    ok = {(6144, 4096, 16): (2, 4, 1, 1, 1, 1), (4992, 6144, 32): (4, 6, 1, 1, 0, 1)}
    env, notes = _gemvcfg(ok)
    assert env["mx_gemv_cfg"](6144, 4096, 12) == (2, 4, 1, 1, 1, 1)
    assert env["mx_gemv_cfg"](6144, 4096, 24) is None
    assert env["mx_gemv_cfg"](4992, 6144, 24) == (4, 6, 1, 1, 0, 1)
    assert env["mx_gemv_cfg"](4992, 6144, 33) is None
    assert env["mx_gemv_cfg"](4992, 6144, 8) == (8, 8)  # the batch-1 override is untouched
    assert len(notes) == 2
    for bad in ((8, 8), (4, 12, 0), (4, 12, 1, 2), (2, 12)):
        with pytest.raises(ValueError, match="not order-preserving"):
            _gemvcfg({(4992, 6144, 16): bad})
    with pytest.raises(ValueError):
        _gemvcfg({(4992, 6144, 24): (4, 6)})  # not an MMA width
    env, _ = _gemvcfg(ok, killed=("TP2WIDE_GEMVCFG",))
    assert env["mx_gemv_cfg"](6144, 4096, 12) is None


# ------------------------------------------------------------------ W11 small guards


@pytest.mark.parametrize(
    "killed,want", [((), True), (("TP2WIDE_INPUT9",), False), (("TP2WIDE_COMMGEMV",), False)]
)
def test_wide_input9_switch(killed, want) -> None:
    t = _tree(MODELING)
    env = _switches(*killed)
    _run([_assign(t, "TP2WIDE_INPUT9"), _assign(t, "TP2WIDE_INPUT_MID_MAX_TOKENS")], env)
    assert env["TP2WIDE_INPUT9"] is want and env["TP2WIDE_INPUT_MID_MAX_TOKENS"] == 64


def test_embed_routes_wide_rows_to_the_fast_first_layer() -> None:
    """_embed at 9..64 rows (TP2WIDE_INPUT9) returns (normed, raw): a residual, so layer 0 runs the fast decode layer
    instead of ``_generic_layer`` (the decoder loop sends ``residual is None`` to the generic layer)."""
    t = _tree(MODELING)
    cls = next(n for n in t.body if isinstance(n, ast.ClassDef) and n.name == "DecodeTarget")
    src = ast.unparse(
        next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_embed")
    )
    assert (
        "EMBED_FIRSTNORM_MAX_TOKENS < num_tokens <= min(TP2WIDE_INPUT_MID_MAX_TOKENS, DECODE_STEP_MAX)"
        in src
    )
    assert src.index(
        "if mid or (TP2WIDE_DECODE and 64 < num_tokens <= DECODE_STEP_MAX):"
    ) < src.index("if not EMBED_FIRSTNORM.covers(num_tokens):")
    loop = ast.unparse(
        next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    )
    assert "elif not kernels or residual is None or is_capture:" in loop


@pytest.mark.parametrize(
    "killed,want",
    [((), 32), (("FRONT_CAPFORK_WIDE",), 8), (("TP2WIDE_COMMGEMV",), 8)],
)
def test_front_capfork_bound(killed, want) -> None:
    t = _tree(DSPARK)
    env = _switches(*killed)
    names = ("FRONT_CAPFORK_BASE_ROWS", "FRONT_CAPFORK_WIDE", "FRONT_CAPFORK_MAX_ROWS")
    _run([_assign(t, n) for n in names], env)
    assert env["FRONT_CAPFORK_MAX_ROWS"] == want


def test_capfork_wide_note_matches_the_fork_conditions() -> None:
    t = _tree(DSPARK)
    log, work = [], []
    env = {
        "FRONT_CAPFORK": True,
        "FRONT_CAPFORK_BASE_ROWS": 8,
        "FRONT_CAPFORK_MAX_ROWS": 32,
        "_CAPFORK_WIDE_LOGGED": [],
        "logger": types.SimpleNamespace(info=log.append),
        "m3_report": types.SimpleNamespace(work=lambda item, detail="": work.append(item)),
    }
    _run([_function(t, "_note_capfork_wide")], env)
    note = env["_note_capfork_wide"]

    class H:
        def __init__(self, rows, cuda=True):
            self.shape, self.is_cuda = (rows, 6144), cuda

        def dim(self):
            return 2

    note(3, H(8), None)  # batch-1 rows: the base fork, not counted here
    note(3, H(16), object())  # residual addend: in line
    note(-1, H(16), None)  # final hidden: in line
    note(3, H(40), None)  # past the bound: in line
    assert not work
    note(3, H(16), None)
    note(5, H(32), None)
    assert work == ["FRONT_CAPFORK_WIDE"] * 2 and len(log) == 1
