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


def _compositions(switch_on: bool, wide_sites=frozenset()):
    """``modeling.py``'s composition statements over ``dispatch`` (as the levers test runs them), then the module's
    ``if _sm103_tep8_f11(): TARGET_PATHS[(103, 8)] = ...`` statement with the switch ``switch_on``."""
    path = _TARGET / "modeling.py"
    tree = ast.parse(path.read_text())
    wanted = re.compile(r"^(_(TEP|SM10|PREFETCH)\w*|_DROP_PATHS|TARGET_PATHS)$")
    body = [
        n
        for n in tree.body
        if _base._assigned(n) and all(wanted.match(x) for x in _base._assigned(n))
    ]
    switch = [
        n
        for n in tree.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and getattr(n.test.func, "id", "") == "_sm103_tep8_f11"
    ]
    assert len(switch) == 1
    namespace = {
        "_dispatch": DISPATCH,
        "_paths_from_env": lambda name: frozenset(),
        "_sm103_tep8_f11": lambda: switch_on,
        "_sm103_tep8_wide_gemm_sites": lambda: frozenset(wide_sites) if switch_on else frozenset(),
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
    assert {
        "MUSESPARK_OPEN_SWA_ONE2",
        "MUSESPARK_OPEN_SWA_PREP_SPLIT",
        "MUSESPARK_OPEN_FA_PREP_FOLD_MULTI",
    } <= set(sm103)


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
def test_env_reads_sm103_defaults_only_on_sm103_with_the_switch(
    monkeypatch, capability, switch, expect
):
    monkeypatch.setattr(DEFAULTS, "_capability", lambda: capability)
    if switch is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, switch)
    name = "MUSESPARK_OPEN_TEP8_HOP_DIRECT"
    monkeypatch.delenv(name, raising=False)
    want = {
        "sm103": DEFAULTS.SM103_TEP8_DEFAULTS[name],
        "sm107": DEFAULTS.SM107_DEFAULTS[name],
        None: "dflt",
    }[expect]
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
def test_the_switch_leaves_every_tep4_table_and_the_off_tep8_table_unchanged(
    monkeypatch, mbs, role, arena
):
    monkeypatch.delenv(ENV, raising=False)
    t4_off = _table(OFF, (103, 4), mbs, role, arena)
    t8_off = _table(OFF, (103, 8), mbs, role, arena)
    monkeypatch.setenv(ENV, "1")
    t4_on = _table(ON, (103, 4), mbs, role, arena)
    assert {n: d.on for n, d in t4_on.decisions.items()} == {
        n: d.on for n, d in t4_off.decisions.items()
    }
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
        if (names and set(names) <= keep) or (
            isinstance(node, ast.FunctionDef) and node.name == "_sm103_tp8_wide"
        ):
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
    for name, src in (
        ("SM103_TP8_DRAFTER_WIDE_TACTICS", "SM107_TP8_DRAFTER_WIDE_TACTICS"),
        ("SM103_TP8_FC_WIDE_TACTICS", "SM107_TP8_FC_WIDE_TACTICS"),
    ):
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
    wide = ast.literal_eval(
        re.search(r"SM107_TP8_WIDE64_KERNELS: Mapping\[str, str\] = (\{[^}]*\})", src).group(1)
    )
    m = re.search(r"^DEFAULT_KERNELS: Mapping\[.*?\] = (\{.*?^\})", src, re.S | re.M)
    cells = eval(m.group(1), {"MAX_ROWS": 8, "CUBLAS": "cublas"})  # noqa: S307 - the module's own literal
    m = re.search(r"TEP8_WIDE64_CUBLAS_CELLS: .*? = frozenset\((\{.*?\})\)", src)
    cublas_ok = ast.literal_eval(m.group(1))
    assert cublas_ok == {("o_proj", (10, 3))}
    assert (
        "MUSESPARK_OPEN_TEP8_WIDE64_SITES" not in DEFAULTS.SM103_TEP8_DEFAULTS
    )  # module default: qkv,o_proj
    for site in ("qkv", "o_proj"):
        cell = cells[site][-1][1]
        if cell.startswith("splitk:"):
            assert cell.split(",")[2] == wide[site].split(",")[2], (site, cell, wide[site])
        else:
            assert (
                cell == "cublas"
                and wide[site].split(",")[2] == "1"
                and (site, (10, 3)) in cublas_ok
            ), site


def test_sm103_tep8_finalize_transport_takes_128_tokens():
    """The TP8 finalize transport's token cap admits 128 (the built-in TP8 kernel's bound; one CTA per token at
    17..128, the flat build's order), and SM103_TEP8_DEFAULTS keeps 64 (128 is set per deployment)."""
    src = (_TARGET / "ops/moe/finalize_allreduce_transport.py").read_text()
    assert "assert TP8_MAX_TOKENS in (16, 32, 64, 128)" in src
    legacy = (_TARGET.parents[4] / "models/musespark_open_finalize_ar.py").read_text()
    assert "_MAX_TOKENS = 128" in legacy and "8: ((16, 8), (_MAX_TOKENS, 1))" in legacy
    assert DEFAULTS.SM103_TEP8_DEFAULTS["MUSESPARK_OPEN_TEP8_FINALIZE_TRANSPORT_TOKENS"] == "64"


def test_sm103_tep8_wide_gemms_switch_serves_wide_dense_gemms_for_the_named_sites(monkeypatch):
    """``MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS`` (default off, in no defaults table) adds ``wide_dense_gemms`` to the
    sm103 F11 TEP8 composition, which is then sm107's TEP8 composition exactly; ``1`` names the sites whose sm103 8-row
    cell is split-K; an unknown site is refused; the F11 switch off leaves it inert."""
    wide_on, _ = _compositions(True, {"unpool"})
    assert wide_on[(103, 8)].served == ON[(107, 8)].served
    wide_off_switch, _ = _compositions(False, {"unpool"})
    assert wide_off_switch == OFF
    name = DEFAULTS.SM103_TEP8_WIDE_GEMMS_ENV
    assert name == "MUSESPARK_OPEN_SM103_TEP8_WIDE_GEMMS"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    monkeypatch.setenv(ENV, "1")
    for value, sites in (
        ("", set()),
        ("0", set()),
        ("1", {"qkv", "pool", "unpool"}),
        ("unpool", {"unpool"}),
        ("qkv, unpool", {"qkv", "unpool"}),
        ("o_proj,pool", {"o_proj", "pool"}),
    ):
        monkeypatch.setenv(name, value)
        assert DEFAULTS.sm103_tep8_wide_gemm_sites() == sites, value
    monkeypatch.setenv(name, "lm_head")
    with pytest.raises(ValueError):
        DEFAULTS.sm103_tep8_wide_gemm_sites()
    monkeypatch.setenv(name, "1")
    monkeypatch.setenv(ENV, "0")
    assert DEFAULTS.sm103_tep8_wide_gemm_sites() == frozenset()


def test_sm103_tep8_wide_gemm_tiles_keep_the_8_row_split_and_fit_gb300():
    """sm107's TP8 wide tiles keep sm103's 8-row split_k (4 for qkv / pool / unpool; o_proj split_k 1 over
    sm103's cuBLAS cell, TEP8_WIDE64_CUBLAS_CELLS), and fit GB300's opt-in once ``wide_steps`` lowers the stages."""
    src = (_TARGET / "kernels/dense_gemms/dispatcher.py").read_text()
    wide = eval(  # noqa: S307 - the module's own literal
        re.search(r"^SM107_TP8_WIDE_KERNELS: Mapping\[.*?\] = (\{.*?^\})", src, re.S | re.M).group(
            1
        ),
        {"WIDE_MAX_ROWS": 32},
    )
    m = re.search(r"^DEFAULT_KERNELS: Mapping\[.*?\] = (\{.*?^\})", src, re.S | re.M)
    cells = eval(m.group(1), {"MAX_ROWS": 8, "CUBLAS": "cublas"})  # noqa: S307 - the module's own literal
    for site, steps in wide.items():
        cell = cells[site][-1][1]
        for _, kernel in steps:
            tactic = tuple(int(v) for v in kernel.split(":")[1].split(","))
            if cell.startswith("splitk:"):
                assert tactic[2] == int(cell.split(",")[2]) == 4, (site, kernel, cell)
            else:
                assert site == "o_proj" and cell == "cublas" and tactic[2] == 1, (
                    site,
                    kernel,
                    cell,
                )
            stages = tactic[3]
            while stages > 2 and _smem((*tactic[:3], stages)) > 232448:
                stages -= 1
            assert _smem((*tactic[:3], stages)) <= 232448 and stages >= 6, (site, kernel, stages)
    assert (
        "_sm103_tep8_wide_gemm_sites()\n"
        "            if tuple(capability) == (10, 3) and tp_size == 8 and sharded\n" in src
    )


def test_sm103_tep8_unpool_direct_wide_switch(monkeypatch):
    """Unpool-fuse: ``MUSESPARK_OPEN_SM103_TEP8_UNPOOL_DIRECT_WIDE`` (default off, in no defaults table) is read
    only under the F11 switch, takes 0 / 1 only; the direct transport accepts the 32-row layout at TP8 and
    ``unpool_into_boundary.install`` builds the wide workspace at TP8 only through it (TP4 keeps ``_wide_direct``)."""
    name = DEFAULTS.SM103_TEP8_UNPOOL_DIRECT_WIDE_ENV
    assert name == "MUSESPARK_OPEN_SM103_TEP8_UNPOOL_DIRECT_WIDE"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    assert DEFAULTS.SM103_TEP8_DEFAULTS["MUSESPARK_OPEN_TEP8_WIDE_UNPOOL_GATHER"] == "1"
    monkeypatch.setenv(ENV, "1")
    for value, on in (("", False), ("0", False), ("1", True)):
        monkeypatch.setenv(name, value)
        assert DEFAULTS.sm103_tep8_unpool_direct_wide() is on, value
    monkeypatch.setenv(name, "yes")
    with pytest.raises(ValueError):
        DEFAULTS.sm103_tep8_unpool_direct_wide()
    monkeypatch.setenv(name, "1")
    monkeypatch.setenv(ENV, "0")
    assert DEFAULTS.sm103_tep8_unpool_direct_wide() is False
    direct = (_TARGET / "ops/comm/unpool_direct_boundary.py").read_text()
    assert "TP4 only" not in direct.split("def check_layout", 1)[1].split("\ndef ", 1)[0]
    install = (_TARGET / "kernels/unpool_into_boundary/boundary.py").read_text()
    assert (
        "tep8_wide = int(layers[0].mapping.tp_size) == 8 and _sm103_tep8_unpool_direct_wide()"
        in install
    )
    assert '(tep8_wide or getattr(layer.feed_forward, "_wide_direct", False))' in install


def test_sm103_tep8_publish_splitk_only_keeps_the_8_row_cells_bits_and_is_opt_in():
    """``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_SPLITK_ONLY=1`` arms GB300's TP8 pool / unpool publication for the
    split-K producer's rows only (the TP8 cuBLAS census covers Rubin's 212 SMs). The producer's tactics keep the
    ``split_k`` of sm103's TP8 sharded 8-row pool / unpool cells (one-request bits), fit GB300's opt-in shared memory,
    the split-K census holds a 152-SM (GB300) key, and the switch is opt-in (not among the F11 defaults)."""
    src = (_TARGET / "ops/comm/projection_publish.py").read_text()
    tactics = re.search(r"TP8_PUBLISH_SPLITK_TACTICS = \{(.*?)\n\}", src, re.S).group(1)
    published = [
        tuple(int(v) for v in t) for t in re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", tactics)
    ]
    assert len(published) == 2
    dispatcher = (_TARGET / "kernels/dense_gemms/dispatcher.py").read_text()
    m = re.search(r"^DEFAULT_KERNELS: Mapping\[.*?\] = (\{.*?^\})", dispatcher, re.S | re.M)
    cells = eval(m.group(1), {"MAX_ROWS": 8, "CUBLAS": "cublas"})  # noqa: S307 - the module's own literal
    for site in ("pool", "unpool"):
        cell = cells[site][-1][1]
        assert cell.startswith("splitk:"), (site, cell)
        assert {t[2] for t in published} == {int(cell.split(",")[2])}, (site, cell, published)
    for tactic in published:
        assert _smem(tactic) <= 232448, tactic
    certified = re.search(
        r"CERTIFIED_SPLITK_PRODUCERS_TP8 = frozenset\((.*?)\n\)", src, re.S
    ).group(1)
    assert re.search(r"^\s*152,$", certified, re.M), (
        "no GB300 (152 SM) split-K producer census entry"
    )
    publish = (_TARGET / "kernels/projection_publish/publish.py").read_text()
    assert 'SPLITK_ONLY_ENV = "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_SPLITK_ONLY"' in publish
    assert 'SPLITK_ONLY = _sm107_env(SPLITK_ONLY_ENV, "0") == "1"' in publish
    assert "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_SPLITK_ONLY" not in DEFAULTS.SM103_TEP8_DEFAULTS
    assert DEFAULTS.SM103_TEP8_DEFAULTS["MUSESPARK_OPEN_TEP8_PUBLISH_SPLITK"] == "1"


@pytest.mark.parametrize(
    "value, expect",
    [(None, 5), ("5", 5), ("2", 2), ("4", 4), ("1", 2), ("0", 2), ("9", 5), ("", 5)],
)
def test_tep8_cl8_min_requests_defaults_to_5_and_clamps_to_2_5(value, expect):
    """MUSESPARK_OPEN_TEP8_SWA_CL8_MIN_REQUESTS (unset = 5, the pre-knob band) is clamped to 2..5."""
    path = _TARGET / "ops" / "attention" / "swa_block.py"
    tree = ast.parse(path.read_text())
    node = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and [getattr(t, "id", "") for t in n.targets] == ["_CL8_TEP8_MIN_REQUESTS"]
    ]
    assert len(node) == 1
    env = {} if value is None else {"MUSESPARK_OPEN_TEP8_SWA_CL8_MIN_REQUESTS": value}
    expr = compile(ast.Expression(node[0].value), str(path), "eval")
    got = eval(
        expr,
        {
            "_sm107_env": lambda name, default=None: env.get(name, default),
            "min": min,
            "max": max,
            "int": int,
        },
    )
    assert got == expect


def test_tep8_cl8_2cta_build_is_opt_in_and_fits_its_smem():
    """MUSESPARK_OPEN_TEP8_SWA_CL8_2CTA is off unless set to "1"; the two-CTA build only rewrites the one
    ``__launch_bounds__(NT, 1)`` of the source; its dynamic SMEM covers the kernel's layout and two CTAs fit GB300."""
    path = _TARGET / "ops" / "attention" / "swa_block.py"
    text = path.read_text()
    tree = ast.parse(text)
    assigns = {
        t.id: n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name)
    }
    for value, expect in ((None, False), ("0", False), ("1", True)):
        env = {} if value is None else {"MUSESPARK_OPEN_TEP8_SWA_CL8_2CTA": value}
        expr = compile(ast.Expression(assigns["_CL8_2CTA"]), str(path), "eval")
        assert eval(expr, {"_sm107_env": lambda n, d=None: env.get(n, d)}) is expect
    src = ast.literal_eval(assigns["_CL8_SRC"])
    assert src.count("__launch_bounds__(NT, 1)") == 1
    defines = dict(re.findall(r"^#define (\w+) (.+?)(?:\s*//.*)?$", src, re.M))

    def val(name):
        expr = defines[name]
        for _ in range(20):
            for k in defines:
                expr = re.sub(rf"\b{k}\b", f"({defines[k]})", expr)
        return eval(expr)

    end = val("BAR_OFF") + 8
    smem2 = eval(compile(ast.Expression(assigns["_CL8_SMEM_2CTA"]), str(path), "eval"))
    assert end <= smem2 and 2 * (smem2 + 1024) <= 233472


def test_sm103_tep8_publish_wide_is_opt_in_and_chunks_into_certified_rows():
    """``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE=splitk|splitk1|cublas`` arms GB300's TP8 pool / unpool
    publication for 65..WIDE_MAX_ROWS-row steps (bs16 = 128 rows); opt-in (not among the F11 defaults), 33..64 rows
    only with SPLITK_ONLY, every 64-row chunk inside the split-K producer's certified 33..64 band, split_k-1 tactics on
    the certified tiles, and the GB300 cuBLAS key consulted only by the WIDE build."""
    publish = (_TARGET / "kernels/projection_publish/publish.py").read_text()
    assert 'WIDE_ENV = "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE"' in publish
    assert 'WIDE_MAX_ROWS = int(_sm107_env(WIDE_MAX_ROWS_ENV, "") or 128)' in publish
    assert "self._min_rows = MIN_ROWS if SPLITK_ONLY else _publish.SPLITK_MAX_ROWS + 1" in publish
    for env in (
        "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE",
        "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_MAX_ROWS",
    ):
        assert env not in DEFAULTS.SM103_TEP8_DEFAULTS
    src = (_TARGET / "ops/comm/projection_publish.py").read_text()
    k4 = re.search(r"TP8_PUBLISH_SPLITK_TACTICS = \{(.*?)\n\}", src, re.S).group(1)
    k1 = re.search(r"TP8_PUBLISH_SPLITK1_TACTICS = \{(.*?)\n\}", src, re.S).group(1)
    t4 = [tuple(int(v) for v in t) for t in re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", k4)]
    t1 = [tuple(int(v) for v in t) for t in re.findall(r"\((\d+), (\d+), (\d+), (\d+)\)", k1)]
    assert [t[:2] for t in t1] == [t[:2] for t in t4] and {t[2] for t in t1} == {1}
    for tactic in t1:
        assert _smem(tactic) <= 232448, tactic
    assert (
        'CERTIFIED_PRODUCERS_TP8 = frozenset({("2.14.0a0+b2c75dd062.nv26.09", 130800, 212)})' in src
    )
    assert re.search(r"CERTIFIED_PRODUCERS_TP8_SM103_WIDE = frozenset\(\{\(.*, 152\)\}\)", src)
    # The chunking of _publish_chunked: 64-row launches, the remainder last.
    assert "starts = list(range(0, rows, SPLITK_MAX_ROWS))" in src
    assert "head = _splitk_tactic(hidden[:SPLITK_MAX_ROWS], weight, slot[:SPLITK_MAX_ROWS])" in src


def test_sm103_tep8_hop_direct128_switch_and_scatter_addresses(monkeypatch):
    """hop128: ``MUSESPARK_OPEN_SM103_TEP8_HOP_DIRECT128`` (default off, in no defaults table) is read only under
    the F11 switch and takes 0 / 1; off, the direct hop keeps its 64-row workspace, its pad rule and the scatter
    kernel's 8-slot stride (the image's code paths); on, 65..128-row o_proj calls publish as 64-row chunks whose
    stores land exactly where the consumer polls (a host model of both address formulas, every token and feature)."""
    name = DEFAULTS.SM103_TEP8_HOP_DIRECT128_ENV
    assert name == "MUSESPARK_OPEN_SM103_TEP8_HOP_DIRECT128"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    monkeypatch.setenv(ENV, "1")
    for value, on in (("", False), ("0", False), ("1", True)):
        monkeypatch.setenv(name, value)
        assert DEFAULTS.sm103_tep8_hop_direct128() is on, value
    monkeypatch.setenv(name, "2")
    with pytest.raises(ValueError):
        DEFAULTS.sm103_tep8_hop_direct128()
    monkeypatch.setenv(name, "1")
    monkeypatch.setenv(ENV, "0")
    assert DEFAULTS.sm103_tep8_hop_direct128() is False

    layers = (_TARGET / "layers.py").read_text()
    assert "max_rows=_wide.DIRECT128_MAX_ROWS if _wide.DIRECT128 else 64," in layers
    assert "pad_rows=_wide.DIRECT128_PAD_ROWS if _wide.DIRECT128 else (0, 1024)," in layers
    wide = (_TARGET / "ops/allreduce/post_attn_hop_wide.py").read_text()
    assert "+ cutlass.Int64(bidx) * ((self.slots - 1) * self.cta_m)" in wide
    assert "* (7 * self.cta_m)" not in wide
    assert "DIRECT128_PAD_ROWS = (0, 64)" in wide
    assert "DIRECT128_MAX_ROWS = _sm103_tep8_hop_direct_max_rows() if DIRECT128 else 128" in wide
    assert (
        "if DIRECT128:\n    SERVED_DIRECT_ROWS = "
        "(SERVED_DIRECT_ROWS[0], max(SERVED_DIRECT_ROWS[1], DIRECT128_MAX_ROWS))" in wide
    )
    dispatcher = (_TARGET / "kernels/dense_gemms/dispatcher.py").read_text()
    # hop256: the band reaches 256 rows; the publication arms only what its workspace takes.
    assert (
        'if TEP8_WIDE64_MAX_ROWS < rows <= 4 * TEP8_WIDE64_MAX_ROWS and site.role == "o_proj":'
        in dispatcher
    )
    assert 'if publication is None or not getattr(_wide, "DIRECT128", False):' in dispatcher

    # Host model of the fb-layout addresses (elements, BF16), ranks 8, cta_m 64, one rank's window peer stride PS.
    ranks, cta_m, hidden, PS = 8, 64, 8192, 1 << 24
    for max_rows in (64, 128, 256):
        slots = max_rows // ranks
        polled = {}
        for row in range(
            max_rows
        ):  # the consumer: owner row % 8 polls source slab p at its receive base
            for offset in range(0, hidden, 8):
                polled[
                    (row % ranks, (offset // 64) * (slots * 64) + (row // ranks) * 64 + offset % 64)
                ] = (row, offset)
        stored = {}
        for row0 in range(
            0, max_rows, 64
        ):  # the producer: chunk at token row0 through the view advanced row0 // 8 slots
            base = (row0 // 8) * 64
            for bidx in range(hidden // cta_m):
                origin = base + bidx * cta_m  # gD_tile's origin: feature m0 = bidx * cta_m
                for tok in range(64):
                    p, s = tok % 8, tok // 8
                    for fchunk in range(cta_m // 8):
                        dst = (
                            origin + bidx * ((slots - 1) * cta_m) + p * PS + s * cta_m + fchunk * 8
                        )
                        stored[(dst // PS, dst % PS)] = (row0 + tok, bidx * cta_m + fchunk * 8)
        assert stored == polled, max_rows


def test_sm103_tep8_hop_direct_max_rows_switch(monkeypatch):
    """hop256: ``MUSESPARK_OPEN_SM103_TEP8_HOP_DIRECT_MAX_ROWS`` (128 when unset, in no defaults table) takes
    128 or 256 only; the one-launch producer's row guard admits 256 rows."""
    name = DEFAULTS.SM103_TEP8_HOP_DIRECT_MAX_ROWS_ENV
    assert name == "MUSESPARK_OPEN_SM103_TEP8_HOP_DIRECT_MAX_ROWS"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    for value, want in (("", 128), ("128", 128), ("256", 256)):
        monkeypatch.setenv(name, value)
        assert DEFAULTS.sm103_tep8_hop_direct_max_rows() == want, value
    for bad in ("64", "192", "512", "x"):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError):
            DEFAULTS.sm103_tep8_hop_direct_max_rows()
    wide = (_TARGET / "ops/allreduce/post_attn_hop_wide.py").read_text()
    assert "if not 1 <= m <= 256 or n != 8192" in wide


def test_sm103_tep8_hop128_producer_switch_and_wide_tile_addresses(monkeypatch):
    """``MUSESPARK_OPEN_SM103_TEP8_HOP128_PRODUCER`` (default ``chunks``, in no defaults table, read only with
    the hop128 switch) picks the 65..128-row o_proj producer; ``grid`` / ``tile128`` publish the whole call in one
    launch whose stores land exactly where the direct consumer polls (host model of the wide-tile epilogue: token tile
    y of cta_n tokens, warp / lane / j map, every token and feature at 128 rows / 16 slots)."""
    name = DEFAULTS.SM103_TEP8_HOP128_PRODUCER_ENV
    assert name == "MUSESPARK_OPEN_SM103_TEP8_HOP128_PRODUCER"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    for value, want in (
        ("", ("chunks", 7)),
        ("chunks", ("chunks", 7)),
        ("grid", ("grid", 3)),
        ("grid:4", ("grid", 4)),
        ("tile128", ("tile128", 4)),
        ("tile128:3", ("tile128", 3)),
    ):
        monkeypatch.setenv(name, value)
        assert DEFAULTS.sm103_tep8_hop128_producer() == want, value
    for bad in ("chunks:3", "grid:0", "grid:x", "tile64", "tile128:13"):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError):
            DEFAULTS.sm103_tep8_hop128_producer()
    monkeypatch.delenv(name)

    wide = (_TARGET / "ops/allreduce/post_attn_hop_wide.py").read_text()
    assert (
        'DIRECT128_PRODUCER = _sm103_tep8_hop128_producer() if DIRECT128 else ("chunks", 7)' in wide
    )
    # The image's scatter build is unchanged: whole-capacity request, the 64-token fb loop, the view at (bidx, n_idx).
    assert 'self.scatter_smem = utils.get_smem_capacity_in_bytes("sm_100")' in wide
    assert "cute.local_tile(mC, c_tiler_mn, (bidx, 0 if self.wide_tiles else n_idx, l_idx))" in wide
    assert "p = warp + 4 * (j // 2)\n" in wide and "s = 4 * (j % 2) + lane // chunks\n" in wide
    dispatcher = (_TARGET / "kernels/dense_gemms/dispatcher.py").read_text()
    assert 'mode, stages = _wide.DIRECT128_PRODUCER\n        if mode != "chunks":' in dispatcher

    ranks, cta_m, hidden, PS, max_rows = 8, 64, 8192, 1 << 24, 128
    slots = max_rows // ranks
    for rows in (65, 96, 127, 128):
        polled = {}
        for row in range(rows):
            for offset in range(0, hidden, 8):
                polled[
                    (row % ranks, (offset // 64) * (slots * 64) + (row // ranks) * 64 + offset % 64)
                ] = (row, offset)
        for cta_n in (64, 128):
            stored = {}
            for y in range((rows + cta_n - 1) // cta_n):
                t0 = y * cta_n
                for bidx in range(hidden // cta_m):
                    origin = bidx * cta_m  # local_tile(mC, ..., (bidx, 0, l)): the view of token 0
                    for epi_tid in range(128):
                        warp, lane = epi_tid // 32, epi_tid % 32
                        fchunk = lane % (cta_m // 8)
                        for j in range(cta_n // 16):
                            p = warp + 4 * ((j // 2) % 2)
                            s = 4 * (j % 2) + lane // (cta_m // 8) + 8 * (j // 4)
                            tok = p + 8 * s
                            if t0 + tok < rows:
                                dst = (
                                    origin
                                    + bidx * ((slots - 1) * cta_m)
                                    + p * PS
                                    + (t0 // 8 + s) * cta_m
                                    + fchunk * 8
                                )
                                key = (dst // PS, dst % PS)
                                assert key not in stored, (rows, cta_n, key)
                                stored[key] = (t0 + tok, bidx * cta_m + fchunk * 8)
            assert stored == polled, (rows, cta_n)


def test_sm103_tep8_publish_wide_one_switch():
    """``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_ONE`` (default off, environment only, in no defaults table) takes
    the split_k-1 publication of a 65..128-row slice in one launch compiled as the 64-row launch (same producer key);
    off, ``_publish_chunked`` keeps the forked 64-row launches."""
    src = (_TARGET / "ops/comm/projection_publish.py").read_text()
    name = "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_ONE"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    assert (
        f'WIDE_ONE_ENV = "{name}"' in src
        and '_WIDE_ONE = os.environ.get(WIDE_ONE_ENV, "").strip()' in src
    )
    assert (
        "    if WIDE_ONE and table is TP8_PUBLISH_SPLITK1_TACTICS and rows <= 2 * SPLITK_MAX_ROWS:\n"
        "        _run_splitk_one(hidden, weight, slot, tactic)\n"
    ) in src
    assert (
        '"publish in one launch",\n'
        '            key="musespark_open.t8r4.publish_wide_one",\n'
        "        )\n        return True\n"
        "    main = torch.cuda.current_stream(hidden.device)\n" in src
    )
    assert "**sk._launch_options(SPLITK_MAX_ROWS, tactic.split_k)," in src
    assert "sk.validate_tactic(tactic, SPLITK_MAX_ROWS, n, k)" in src


def test_sm103_tep8_publish_wide_one_k4_switch():
    """With ``PUBLISH_WIDE=splitk`` + ``PUBLISH_WIDE_ONE=1`` a 65..448-row slice publishes from ONE split_k-4
    launch, compiled as the 64-row launch; ``MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_TACTICS`` (in no defaults table)
    picks the tile per row range and accepts split_k 4 only; the one-launch helper refuses other K partitions."""
    src = (_TARGET / "ops/comm/projection_publish.py").read_text()
    name = "MUSESPARK_OPEN_SM103_TEP8_PUBLISH_WIDE_TACTICS"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    assert f'WIDE_TACTICS_ENV = "{name}"' in src and "WIDE_ONE_K4_MAX_ROWS = 448" in src
    k4 = src.index("    if table is None and WIDE_ONE:\n")
    assert k4 < src.index(
        "    if table is None:\n        for (a, b), tactic in zip(bounds, tactics):"
    )
    assert "wt = wide_splitk_tactic(hidden.shape[1], weight.shape[0], rows)" in src
    assert "or tactic[2] != 4\n            or " in src
    assert "top = {1: 2 * SPLITK_MAX_ROWS, 4: WIDE_ONE_K4_MAX_ROWS}.get(tactic.split_k, 0)" in src


def test_sm103_tep8_prerouted_128_switch():
    """``MUSESPARK_OPEN_TEP8_PREROUTED_MAX_ROWS`` (64 default, or 128; in no defaults table) extends the
    prerouted two-window calls to 65..128 rows at tile 16 (the stock call's fallback tile there) and keeps tile 8 up to
    64 rows; ring_buddy_assign covers 2048 slots (1024 before the wide program)."""
    src = (_TARGET / "kernels/replicated_experts/ring_buddy.py").read_text()
    asg = (_TARGET / "ops/moe/ring_buddy_assign.py").read_text()
    name = "MUSESPARK_OPEN_TEP8_PREROUTED_MAX_ROWS"
    assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    assert DEFAULTS.SM103_TEP8_DEFAULTS["MUSESPARK_OPEN_TEP8_RING_BUDDY_MAX_ROWS"] == "64"
    assert f'_TEP8_PREROUTED_MAX_ROWS = int(_sm107_env("{name}", "") or 64)' in src
    assert "if _TEP8_PREROUTED_MAX_ROWS not in (64, 128):" in src
    assert "_TEP8_PREROUTED_ROWS = range(9, _TEP8_PREROUTED_MAX_ROWS + 1)" in src
    assert "    return _PREROUTED_TILE if rows <= 64 else 2 * _PREROUTED_TILE\n" in src
    assert "meta_tile=_prerouted_tile(router_logits.shape[0])," in src
    assert (
        'or meta.get("tile") != _prerouted_tile(rows)' in src
        and "if tile != _prerouted_tile(rows):" in src
    )
    assert "\nMAX_SLOTS = 2048\n" in asg  # widened for the 129..256-row wide program


def test_sm103_tep8_f11_two_shot_and_wide_switches():
    """``MUSESPARK_OPEN_SM103_TEP8_F11_EXTRA_PATHS`` (e.g. ``finalize_two_shot``) adds paths to the F11 TEP8
    composition only, through ``_paths_from_env`` (so the composition tables stay exec-able here);
    ``MUSESPARK_OPEN_TEP8_WIDE_BALANCE_ROWS`` takes 129..256 and only widens NVFP4 balanced layers;
    the staged router GEMM switch keeps 129..256 rows. None of them is in a defaults table."""
    model = (_TARGET / "modeling.py").read_text()
    rb = (_TARGET / "kernels/replicated_experts/ring_buddy.py").read_text()
    rg = (_TARGET / "ops/moe/router_gemm.py").read_text()
    for name in (
        "MUSESPARK_OPEN_SM103_TEP8_F11_EXTRA_PATHS",
        "MUSESPARK_OPEN_TEP8_WIDE_BALANCE_ROWS",
        "TRTLLM_MUSESPARK_ROUTER_STAGED_GEMM",
    ):
        assert name not in DEFAULTS.SM103_TEP8_DEFAULTS and name not in DEFAULTS.SM107_DEFAULTS
    assert (
        '_SM103_TEP8_F11_EXTRA_PATHS = _paths_from_env("MUSESPARK_OPEN_SM103_TEP8_F11_EXTRA_PATHS")'
        in model
    )
    assert "    | _SM103_TEP8_F11_EXTRA_PATHS\n" in model
    assert "if _TEP8_WIDE_BALANCE_ROWS and not 129 <= _TEP8_WIDE_BALANCE_ROWS <= 256:" in rb
    assert 'if _TEP8_WIDE_BALANCE_ROWS and kind == "nvfp4" and windows >= 2 and balance:' in rb
    assert '**({"wide": True} if router_logits.shape[0] > 128 else {}),' in rb
    assert "STAGED_MIN_ROWS = ROW_TILE + 1\nSTAGED_MAX_ROWS = 2 * ROW_TILE\n" in rg
