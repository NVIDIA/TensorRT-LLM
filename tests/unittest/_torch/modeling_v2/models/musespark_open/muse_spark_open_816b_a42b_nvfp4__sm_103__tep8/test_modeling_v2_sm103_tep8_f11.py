# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The F11 TEP8 composition on sm_103 (``MUSESPARK_OPEN_SM103_TEP8_F11=1``), on the host only (no torch, no device).

Pins:

* off by default: ``(103, 8)`` is the pre-F11 composition, and every sm_103 lever table is unchanged;
* on: ``(103, 8)`` is sm107's TEP8 composition less ``wide_dense_gemms`` (sm107-only table), without the sm107
  fused-kernel precondition, and its lever tables resolve as sm107's TEP8 ones do (same rules) except where the
  drafter arena's structure differs by architecture (``windowed_commit_parallel``, ``dispatch.WINDOWED_COMMIT_SMS``);
* the TEP4 composition and tables are identical with the switch on;
* ``SM103_TEP8_DEFAULTS``: a subset of sm107's TEP8 defaults (same values), no TEP4 / context / oversized-SMEM switch,
  and ``env`` reads them only on sm_103 with the switch on;
* the sm103 wide drafter tactics keep sm103's 8-row ``split_k`` and fit GB300's opt-in shared memory.
"""

from __future__ import annotations

import ast
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_modeling_v2_levers as _base  # noqa: E402

LEVERS, DISPATCH = _base.LEVERS, _base.DISPATCH
_TARGET = _base._TARGET
ENV = "MUSESPARK_OPEN_SM103_TEP8_F11"


def _defaults():
    package = sys.modules[_base._PACKAGE]
    assert package.__path__ == [str(_TARGET)]
    return importlib.import_module(f"{_base._PACKAGE}.sm107_defaults")


DEFAULTS = _defaults()


def _compositions(switch_on: bool):
    """``modeling.py``'s composition statements over ``dispatch`` (as the levers test runs them), then the module's
    ``if _sm103_tep8_f11(): TARGET_PATHS[(103, 8)] = ...`` statement with the switch ``switch_on``."""
    path = _TARGET / "modeling.py"
    tree = ast.parse(path.read_text())
    wanted = re.compile(r"^(_(TEP|SM10|PREFETCH)\w*|_DROP_PATHS|TARGET_PATHS)$")
    body = [n for n in tree.body if _base._assigned(n) and all(wanted.match(x) for x in _base._assigned(n))]
    switch = [
        n
        for n in tree.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Call) and getattr(n.test.func, "id", "") == "_sm103_tep8_f11"
    ]
    assert len(switch) == 1
    namespace = {
        "_dispatch": DISPATCH,
        "_paths_from_env": lambda name: frozenset(),
        "_sm103_tep8_f11": lambda: switch_on,
    }
    exec(compile(ast.Module(body=body + switch, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["TARGET_PATHS"], namespace


OFF, NS = _compositions(False)
ON, _ = _compositions(True)


def test_off_by_default_the_sm103_tep8_composition_is_the_pre_f11_one():
    assert OFF[(103, 8)].served == NS["_TEP8_PATHS"] | NS["_PREFETCH_LEVERS"]
    assert OFF == _base.TARGET_PATHS
    assert DEFAULTS.SM103_TEP8_F11_ENV == ENV


def test_on_the_sm103_tep8_composition_is_f11_less_the_sm107_only_paths():
    f11 = ON[(107, 8)]
    sm103 = ON[(103, 8)]
    assert sm103.served == f11.served - {"wide_dense_gemms"}
    assert f11.served - sm103.served == {"wide_dense_gemms"}
    assert sm103.preconditions == ()
    assert sm103.topology == f11.topology == DISPATCH.TWO_TRAYS_OF_4
    for key in ((103, 4), (107, 4), (107, 8)):
        assert ON[key] == OFF[key], key


def test_switch_reads_the_environment_only(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not DEFAULTS.sm103_tep8_f11()
    for value in ("0", "", "true", "yes"):
        monkeypatch.setenv(ENV, value)
        assert not DEFAULTS.sm103_tep8_f11()
    monkeypatch.setenv(ENV, "1")
    assert DEFAULTS.sm103_tep8_f11()


DEFAULTS_ALLOWED_EXTRA = set()
TEP4_OR_CONTEXT = re.compile(
    r"^MUSESPARK_OPEN_(TEP4_|SWA_CL8$|FA_STAGES$|MR_STAGES$|POOL_SPLITK8$|PREROUTED_META$|"
    r"ROUTER_ROWS_TILING$|FC_SHARD_PREWARM$|SP_SM107_PARTS$|SPLITK_OVERSIZED_SMEM$|SPLITK_RS)"
)


def test_sm103_tep8_defaults_are_sm107s_tep8_values_without_tep4_context_or_oversized_switches():
    sm103, sm107 = DEFAULTS.SM103_TEP8_DEFAULTS, DEFAULTS.SM107_DEFAULTS
    assert set(sm103) <= set(sm107)
    for name, value in sm103.items():
        assert value == sm107[name], name
        assert not TEP4_OR_CONTEXT.match(name), name
    # Every TEP8 switch of sm107 is carried, except the QKV override that needs oversized shared memory.
    tep8 = {name for name in sm107 if name.startswith("MUSESPARK_OPEN_TEP8_")}
    assert tep8 - set(sm103) == {"MUSESPARK_OPEN_TEP8_WIDE64_KERNELS"}
    assert "MUSESPARK_OPEN_SPLITK_OVERSIZED_SMEM" not in sm103
    # The shared attention switches F11's TEP8 shapes need are carried (CL8-TEP8 is gated on ONE2).
    assert {"MUSESPARK_OPEN_SWA_ONE2", "MUSESPARK_OPEN_SWA_PREP_SPLIT", "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI"} <= set(sm103)


@pytest.mark.parametrize(
    "capability, switch, expect",
    [
        ((10, 3), "1", "sm103"),
        ((10, 3), None, None),
        ((10, 3), "0", None),
        ((10, 7), "1", "sm107"),
        ((10, 7), None, "sm107"),
        ((10, 0), "1", None),
        (None, "1", None),
    ],
)
def test_env_reads_sm103_defaults_only_on_sm103_with_the_switch(monkeypatch, capability, switch, expect):
    monkeypatch.setattr(DEFAULTS, "_capability", lambda: capability)
    if switch is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, switch)
    name = "MUSESPARK_OPEN_TEP8_HOP_DIRECT"
    monkeypatch.delenv(name, raising=False)
    want = {"sm103": DEFAULTS.SM103_TEP8_DEFAULTS[name], "sm107": DEFAULTS.SM107_DEFAULTS[name], None: "dflt"}[expect]
    assert DEFAULTS.env(name, "dflt") == want
    # A TEP4 switch never takes an sm107 value on sm103.
    monkeypatch.delenv("MUSESPARK_OPEN_SWA_CL8", raising=False)
    assert DEFAULTS.env("MUSESPARK_OPEN_SWA_CL8", "x") == ("all" if expect == "sm107" else "x")
    # The environment still wins, including "0".
    monkeypatch.setenv(name, "0")
    assert DEFAULTS.env(name, "dflt") == "0"


def _table(compositions, key, mbs, role, arena):
    sm, tp = key
    return DISPATCH.table_for(compositions, sm, tp, mbs, role, arena)


def _on_levers(table):
    return {name for name in LEVERS.RULES if table.on(name)}


GEN_CASES = [
    pytest.param(mbs, role, arena, id=f"{role}-mbs{mbs}-{arena}")
    for mbs in (1, 4, 8, 16)
    for role in ("gen", "agg")
    for arena in ("windowed", "dense", "paged")
]
#: Lever names whose resolution depends on the architecture's drafter-arena structure, not on the F11 rules.
ARENA_SM = {"windowed_commit_parallel"}


@pytest.mark.parametrize("mbs, role, arena", GEN_CASES)
def test_on_sm103_tep8_resolves_like_sm107_tep8(monkeypatch, mbs, role, arena):
    monkeypatch.setenv(ENV, "1")
    t103 = _table(ON, (103, 8), mbs, role, arena)
    t107 = _table(ON, (107, 8), mbs, role, arena)
    assert _on_levers(t103) - ARENA_SM == _on_levers(t107) - ARENA_SM
    assert not t103.on("windowed_commit_parallel")
    assert not t103.on("lm_head_inbox")
    assert not t103.on("graph_start_qkv_prefetch")
    assert t103.on("vocab_parallel_head") == (mbs <= 8)


@pytest.mark.parametrize("mbs, role, arena", GEN_CASES)
def test_the_switch_leaves_every_tep4_table_and_the_off_tep8_table_unchanged(monkeypatch, mbs, role, arena):
    monkeypatch.delenv(ENV, raising=False)
    t4_off = _table(OFF, (103, 4), mbs, role, arena)
    t8_off = _table(OFF, (103, 8), mbs, role, arena)
    monkeypatch.setenv(ENV, "1")
    t4_on = _table(ON, (103, 4), mbs, role, arena)
    assert {n: d.on for n, d in t4_on.decisions.items()} == {n: d.on for n, d in t4_off.decisions.items()}
    # Off: the sm103 tep8 rules are the pre-F11 ones (inbox on, prefetch on, the head at any capacity).
    assert t8_off.on("lm_head_inbox") and t8_off.on("graph_start_qkv_prefetch")
    # The rules alone (switch on, old composition) do not change a TEP4 deployment.
    monkeypatch.setenv(ENV, "1")
    assert {n: d.on for n, d in _table(OFF, (103, 4), mbs, role, arena).decisions.items()} == {
        n: d.on for n, d in t4_off.decisions.items()
    }


def _drafter_tables():
    path = _TARGET / "ops/gemm/fixed_low_m_gemm.py"
    src = path.read_text()
    tree = ast.parse(src)
    keep = {
        "SPLITK_TACTICS",
        "_TP8_DRAFTER_WIDE_SHAPES",
        "_TP8_DRAFTER_WIDE_TILES",
        "SM107_TP8_DRAFTER_WIDE_TACTICS",
        "SM107_TP8_FC_WIDE_TACTICS",
        "SM103_TP8_DRAFTER_WIDE_TACTICS",
        "SM103_TP8_FC_WIDE_TACTICS",
    }
    body = []
    for node in tree.body:
        names = _base._assigned(node)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if (names and set(names) <= keep) or (isinstance(node, ast.FunctionDef) and node.name == "_sm103_tp8_wide"):
            body.append(node)
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _smem(tactic):
    """splitk_gemm._smem_bytes (no RS mailbox), restated."""
    mma_m, mma_n, split_k, st = tactic
    up = lambda v, a: -(-v // a) * a  # noqa: E731
    cur = up(mma_m * 128 * 2 * st, 1024) + mma_n * 128 * 2 * st
    cur = up(cur, 8) + 2 * st * 8 + 3 * 8
    cur = up(cur, 4) + 4
    if split_k == 1:
        return cur
    return up(up(cur, 128) + (split_k - 1) * mma_m * mma_n * 4, 8) + 8


def test_sm103_wide_drafter_tactics_keep_the_8_row_split_and_fit_gb300():
    ns = _drafter_tables()
    eight = ns["SPLITK_TACTICS"]
    for name, src in (("SM103_TP8_DRAFTER_WIDE_TACTICS", "SM107_TP8_DRAFTER_WIDE_TACTICS"),
                      ("SM103_TP8_FC_WIDE_TACTICS", "SM107_TP8_FC_WIDE_TACTICS")):
        sm103, sm107 = ns[name], ns[src]
        assert set(sm103) == set(sm107)
        for key, tactic in sm103.items():
            rows, k, n = key
            assert tactic[2] == eight[(8, k, n)][2], key  # the one-request bits
            assert tactic[:2] == sm107[key][:2], key
            assert 1 <= tactic[3] <= sm107[key][3], key
            assert _smem(tactic) <= 232448, (key, tactic, _smem(tactic))
            assert _smem((*tactic[:3], tactic[3] + 1)) > 232448 or tactic[3] == sm107[key][3], key
    # gate_up is the one site whose split differs (sm103 2, sm107 1).
    assert ns["SM103_TP8_DRAFTER_WIDE_TACTICS"][(64, 8192, 7168)] == (64, 64, 2, 6)


def test_sm103_wide64_module_tactics_fit_gb300_without_oversized_smem():
    """dense_gemms' module wide64 tactics (served on sm103: no override) fit the 232448 B opt-in; F11's QKV override
    (8 stages, oversized on sm107) does not, which is why SM103_TEP8_DEFAULTS leaves it out."""
    src = (_TARGET / "kernels/dense_gemms/dispatcher.py").read_text()
    m = re.search(r"SM107_TP8_WIDE64_KERNELS: Mapping\[str, str\] = (\{[^}]*\})", src)
    kernels = ast.literal_eval(m.group(1))
    for site, kernel in kernels.items():
        tactic = tuple(int(v) for v in kernel.split(":")[1].split(","))
        assert _smem(tactic) <= 232448, (site, kernel)
    assert _smem((64, 64, 4, 8)) > 232448
    # QKV keeps sm103's 8-row split_k (4).
    assert kernels["qkv"].split(",")[2] == "4"



def test_sm103_wide64_sites_keep_the_bits_of_sm103s_8_row_cells():
    """``dense_gemms.install`` serves a wide64 site only when it keeps each row's bits of the site's 8-row cell on this
    device: a split-K cell with the same ``split_k``, or a cuBLAS cell measured bitwise equal to a ``split_k``-1
    tactic (``TEP8_WIDE64_CUBLAS_CELLS``). On sm103 at TP8 QKV is the former, o_proj the latter (GB300), so the
    module's default sites (qkv, o_proj; the direct hop needs o_proj) stand under the switch."""
    src = (_TARGET / "kernels/dense_gemms/dispatcher.py").read_text()
    wide = ast.literal_eval(re.search(r"SM107_TP8_WIDE64_KERNELS: Mapping\[str, str\] = (\{[^}]*\})", src).group(1))
    m = re.search(r"^DEFAULT_KERNELS: Mapping\[.*?\] = (\{.*?^\})", src, re.S | re.M)
    cells = eval(m.group(1), {"MAX_ROWS": 8, "CUBLAS": "cublas"})  # noqa: S307 - the module's own literal
    m = re.search(r"TEP8_WIDE64_CUBLAS_CELLS: .*? = frozenset\((\{.*?\})\)", src)
    cublas_ok = ast.literal_eval(m.group(1))
    assert cublas_ok == {("o_proj", (10, 3))}
    assert "MUSESPARK_OPEN_TEP8_WIDE64_SITES" not in DEFAULTS.SM103_TEP8_DEFAULTS  # module default: qkv,o_proj
    for site in ("qkv", "o_proj"):
        cell = cells[site][-1][1]
        if cell.startswith("splitk:"):
            assert cell.split(",")[2] == wide[site].split(",")[2], (site, cell, wide[site])
        else:
            assert cell == "cublas" and wide[site].split(",")[2] == "1" and (site, (10, 3)) in cublas_ok, site
