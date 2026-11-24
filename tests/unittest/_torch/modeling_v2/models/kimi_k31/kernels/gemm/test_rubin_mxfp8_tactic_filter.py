# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tactics the family's MXFP8 runner profiles on SM107 (``_is_kimi_k3_mxfp8_tactic_profiled``).

Read from the kernel module's source, so the check needs no CuTe DSL: the filter keeps the
prefetching tactics of a (4, 2) cluster, the fastest on some K3.1 MXFP8 weights over the SM107
tactic space, besides the prefetching unswapped 2-CTA b-reuse tile.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_MODULE = (
    Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0])
    / "_torch/_experimental/modeling_v2/models/kimi_k31/kernels/gemm/rubin_dense_blockscaled_gemm.py"
)
_NAMES = ("_KIMI_K3_MXFP8_MAX_SPLIT_K_M", "_is_kimi_k3_mxfp8_tactic_profiled")


def _filter():
    tree = ast.parse(_MODULE.read_text())
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _NAMES)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in _NAMES for t in node.targets)
        )
    ]
    assert len(body) == len(_NAMES), "the filter or its split-K bound moved"
    ns: dict = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(_MODULE), "exec"), ns)
    return ns["_is_kimi_k3_mxfp8_tactic_profiled"]


def _base(tiler_m, inst_m, cluster, swap_ab, prefetch, split_k=1):
    return (
        "base",
        (tiler_m, 128, 128),
        (inst_m, 128, 32),
        cluster,
        swap_ab,
        prefetch,
        "static",
        "m",
        split_k,
    )


def _mixed(tiler_m, inst_m, cluster, swap_ab, prefetch):
    return ("mixed_clusters", (tiler_m, 128, 128), (inst_m, 128, 32), cluster, (2, 1), swap_ab,
            prefetch, "static", "m")  # fmt: skip


@pytest.mark.parametrize(
    "tactic, kept",
    [
        (_base(256, 256, (4, 2), False, True), True),  # prefetch on a (4, 2) cluster
        (_base(256, 256, (4, 2), True, True), True),  # swapped too
        (_mixed(256, 256, (4, 2), False, True), True),
        (_base(512, 256, (2, 1), False, True), True),  # the unswapped 2-CTA b-reuse tile
        (_base(256, 256, (2, 1), False, True), False),  # prefetch elsewhere
        (_base(512, 256, (2, 1), True, True), False),
        (_mixed(256, 256, (2, 2), False, True), False),
        (_base(256, 256, (2, 1), False, False), True),  # no prefetch
        (_base(256, 128, (4, 2), False, True), False),  # a 1-CTA b-reuse tile
    ],
)
def test_the_filter_keeps_the_prefetching_4x2_cluster_tactics(tactic, kept):
    assert _filter()(tactic, 64) is kept


def test_split_k_stays_up_to_m_256():
    profiled = _filter()
    tactic = _base(256, 256, (2, 1), False, False, split_k=2)
    assert profiled(tactic, 256) and not profiled(tactic, 512)
