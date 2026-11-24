# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FC12's launch levers per row range above 64 rows (CPU only, on the sources): the table, the
per-row lookup of ``Fc12Experts`` (launch, prezeroed launch, tuner lookup) and the chain setting
them on every MoE layer before FC12's preparation."""

from __future__ import annotations

import ast
from typing import Any, Dict, Iterable

import pytest
from _helix16_wide_cpu import function_node, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

NAMES = (
    "_FC12_MMA_N", "_FC12_DIE_SCHED", "_FC12_DIE_CLAIM", "_FC12_DIE_MIN_TOKENS",
    "_FC12_LOOKAHEAD", "_FC12_SM_RESERVE", "_FC12_L2_PREFETCH", "_FC12_LEVERS",
    "FC12_ROW_LEVERS", "fc12_row_levers",
)  # fmt: skip


@pytest.fixture(scope="module")
def ns():
    return source_objects(modeling_path(), NAMES, {"Dict": Dict, "Any": Any, "Iterable": Iterable})


def test_the_row_ranges(ns):
    table, base = ns["fc12_row_levers"](range(1, 140)), ns["_FC12_LEVERS"]
    assert sorted(table) == list(range(65, 129))
    want = {**dict.fromkeys(range(65, 81), (0, 1, 2)), **dict.fromkeys(range(81, 113), (16, 1, 4)),
            **dict.fromkeys(range(113, 129), (0, 1, 2))}  # fmt: skip
    for rows, levers in table.items():
        assert (levers["sm_reserve"], levers["lookahead"], levers["l2_prefetch"]) == want[rows]
        # Only the three launch levers move; the rest are FC12's defaults.
        assert {
            k: v for k, v in levers.items() if k not in ("sm_reserve", "lookahead", "l2_prefetch")
        } == {k: v for k, v in base.items() if k not in ("sm_reserve", "lookahead", "l2_prefetch")}
    # One dict per lever set, shared by its rows (one tuner runner each).
    assert len({id(v) for v in table.values()}) == 2
    assert table[72] is table[120] and table[96] is not table[72]


@pytest.mark.parametrize("q", [8, 6, 5, 4])
def test_every_wide_row_of_each_q_takes_its_range(ns, q):
    rows = [q * b for b in range(1, 128 // q + 1)]
    table = ns["fc12_row_levers"](rows)
    assert sorted(table) == [r for r in rows if r > 64]
    assert ns["fc12_row_levers"](range(1, 65)) == {}


def test_fc12_takes_each_row_counts_levers():
    run = ast.unparse(function_node(modeling_path(), "Fc12Experts.run"))
    launch = ast.unparse(function_node(modeling_path(), "Fc12Experts.launch"))
    levers = ast.unparse(function_node(modeling_path(), "Fc12Experts.levers"))
    assert "levers = self.levers(out.shape[0])" in run and "**levers)" in run
    assert "levers = self.levers(operands[13].shape[0])" in launch and "**levers)" in launch
    assert "return self.row_levers.get(rows, _FC12_LEVERS)" in levers


def test_the_chain_sets_the_levers_before_fc12_is_prepared():
    init = ast.unparse(function_node(modeling_path(), "FlatChainRuntime.__init__"))
    assert init.index("self._prepare_fc12_levers()") < init.index("experts.prepare(self.rows")
    prepare = ast.unparse(function_node(modeling_path(), "FlatChainRuntime._prepare_fc12_levers"))
    assert "levers = fc12_row_levers(self.rows)" in prepare
    assert "provider.experts.row_levers = dict(levers)" in prepare
