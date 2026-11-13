# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of INDEX_SCORE_T2SC_WIDE: the sm_107 TP2 target scores 3 and 4 requests on the T2SC index scorer.

Source-level (no torch / tensorrt_llm import): the request-count table, its kill switch, the engagement line, the
schedule the dispatch derives (no early issue above INDEX_SCORE_EARLY_MAX_REQUESTS) and the catalog cells that
certify every covered request count at the TP2 K7 geometry (2 index heads, 8 query rows per request).
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

TARGET = (
    Path(__file__).resolve().parents[7]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__tp2"
)
MODELING = TARGET / "modeling.py"
ENTRY = TARGET / "ops/attention/m3_index_score_t2sc.py"


def _assign(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return node.value
    raise AssertionError(f"{name} is not assigned at module level")


def _proxy_dict(node: ast.AST, tree: ast.Module) -> dict:
    """The dict inside ``MappingProxyType({...})``: literals, or module-level names assigned a literal (e.g. the
    2-request split ``INDEX_SCORE_T2SC_SPLIT2_DEFAULT``)."""
    assert isinstance(node, ast.Call) and node.func.id == "MappingProxyType"

    def value(v: ast.AST):
        if isinstance(v, ast.Name):
            return ast.literal_eval(_assign(tree, v.id))
        return ast.literal_eval(v)

    d = node.args[0]
    assert isinstance(d, ast.Dict)
    return {ast.literal_eval(k): value(v) for k, v in zip(d.keys, d.values)}


def _modeling():
    return ast.parse(MODELING.read_text())


def _cells() -> list:
    tree = ast.parse(ENTRY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "M3IndexScoreT2sc":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.target.id == "CELLS":
                    specs = []
                    for cell in stmt.value.elts:
                        spec = next(k.value for k in cell.keywords if k.arg == "spec")
                        specs.append({k.arg: ast.literal_eval(k.value) for k in spec.keywords})
                    return specs
    raise AssertionError("M3IndexScoreT2sc.CELLS not found")


def test_the_batch_one_two_splits_are_unchanged():
    tree = _modeling()
    assert _proxy_dict(_assign(tree, "INDEX_SCORE_T2SC_SPLIT_K_B12"), tree) == {1: 1272, 2: 768}


def test_three_and_four_requests_take_t2sc():
    tree = _modeling()
    wide = _proxy_dict(_assign(tree, "INDEX_SCORE_T2SC_SPLIT_K_WIDE"), tree)
    assert set(wide) == {3, 4}
    # Every split keeps the 4-request grid inside one dimension of the kernel's (batch, split_k) launch.
    assert all(1 <= split <= 1536 for split in wide.values())


def test_the_kill_switch_drops_only_the_wide_counts():
    tree = _modeling()
    gate = ast.unparse(_assign(tree, "INDEX_SCORE_T2SC_WIDE"))
    assert gate == "not m31_disabled('INDEX_SCORE_T2SC_WIDE')"
    merged = ast.unparse(_assign(tree, "INDEX_SCORE_T2SC_SPLIT_K"))
    assert "INDEX_SCORE_T2SC_SPLIT_K_B12" in merged
    assert "INDEX_SCORE_T2SC_SPLIT_K_WIDE if INDEX_SCORE_T2SC_WIDE else {}" in merged


def test_the_wide_counts_keep_the_prefetch_schedule():
    """index_score_t2sc_schedule: early issue (and the prestage it gates) only up to INDEX_SCORE_EARLY_MAX_REQUESTS."""
    tree = _modeling()
    early_max = ast.literal_eval(_assign(tree, "INDEX_SCORE_EARLY_MAX_REQUESTS"))
    wide = _proxy_dict(_assign(tree, "INDEX_SCORE_T2SC_SPLIT_K_WIDE"), tree)
    assert all(n > early_max for n in wide)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "index_score_t2sc_schedule"
    )
    assert "requests <= INDEX_SCORE_EARLY_MAX_REQUESTS" in ast.unparse(fn)


def test_every_wide_count_is_certified_at_the_tp2_k7_geometry():
    """The dispatch's certification rule (_index_score_t2sc_requests): 2 index heads, the count's (split, early)."""
    tree = _modeling()
    early_max = ast.literal_eval(_assign(tree, "INDEX_SCORE_EARLY_MAX_REQUESTS"))
    table = {
        **_proxy_dict(_assign(tree, "INDEX_SCORE_T2SC_SPLIT_K_B12"), tree),
        **_proxy_dict(_assign(tree, "INDEX_SCORE_T2SC_SPLIT_K_WIDE"), tree),
    }
    certified = set()
    for spec in _cells():
        n = len(spec["seq_lens"])
        if spec["heads"] != 2 or n not in table:
            continue
        if (spec["split_k"], spec["early"]) == (table[n], n <= early_max):
            certified.add(n)
            assert spec["qlen"] == 8 or n <= 2
    assert certified == set(table)
    # The wide cells include a short request (early tiles / blocks past the length) and a looping CTA grid.
    for n in (3, 4):
        lens = [s["seq_lens"] for s in _cells() if len(s["seq_lens"]) == n]
        assert any(min(x) <= 300 for x in lens)
        assert any(max(x) > 128 * table[n] for x in lens)


def test_the_engagement_line_is_logged_once_per_count():
    src = MODELING.read_text()
    assert "_INDEX_SCORE_T2SC_WIDE_LOGGED.add(rows)" in src
    assert "[TP2WIDE] T2SC_WIDE taken:" in src
