# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's key-shard merge re-selection (``dsa_key_shard_failing_fallback``): off, a merge row
whose certificate fails is re-selected from every rank's local logits; on, from the failing ranks'
logits and the other ranks' received candidates (``rubin_dsa_key_shard_merge``'s
``failing_only``: the same rows, counts and mask, which its own tests compare byte for byte).

CPU (``cpu_only``), from helix8's modeling.py over its constants:

* the generation constant is on, and ``_KS_FAILING_ONLY`` follows it;
* ``_idx_decode`` on a key-sharded step calls the logits, the CuTe local Top-K, the pack and the
  merge on the same buffers, bounds and mask either way; only the switch passes
  ``failing_only=True`` and logs the README's INFO line through ``logger.info_once``.
"""

from __future__ import annotations
import __future__

import ast
import copy
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_FAMILY = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31"
_MODELING = _FAMILY / "kimi_k31_pro_nvfp4__sm_107__helix8" / "modeling.py"
_README = _FAMILY / "README.md"
_LINE_PREFIX = "Kimi K3.1 Rubin DSA key shard: "


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _constants(tree: ast.Module) -> dict:
    """``tree``'s module-level constants that evaluate on their own, in source order."""
    ns: dict = {"MappingProxyType": MappingProxyType, "torch": torch}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            ns[target.id] = eval(compile(ast.Expression(node.value), str(_MODELING), "eval"), ns)
        except Exception:
            continue
    return ns


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    return fn


class _Log:
    def __init__(self):
        self.once = []

    def info_once(self, msg, key=None):
        self.once.append((msg, key))


def _exec(failing_only: bool):
    """helix8's ``_idx_decode`` from its modeling.py, with ``_KS_FAILING_ONLY`` = ``failing_only``
    and recording stand-ins for the kernels it calls."""
    tree = _tree(_MODELING)
    ns = _constants(tree)
    calls, log = [], _Log()
    ns["_KS_FAILING_ONLY"] = failing_only
    ns["_IDX_HEADS"], ns["_IDX_DIM"] = 2, 4

    def rec(name, result=None):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result(*args) if result else None

        return run

    ns.update(
        rubin_fp8_paged_mqa_logits=rec("logits", lambda *a: a[-1]),
        rubin_topk_decode_cute=rec("topk"),
        rubin_dsa_key_shard_pack=rec("pack"),
        rubin_dsa_key_shard_merge=rec("merge"),
        logger=log,
    )
    body = [copy.deepcopy(_function(tree, "_idx_decode"))]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns, calls, log


def _step(n: int, bounds: bool):
    ws = SimpleNamespace(
        local_logits=torch.zeros(16, 512),
        local_topk=torch.zeros(16, 512, dtype=torch.int32),
        local_k=512,
    )
    dsa = SimpleNamespace(
        skip_gen=False,
        topk_out=torch.zeros(16, 2048, dtype=torch.int32),
        ks_next_n=1,
        ks_ctx_lens=torch.zeros(n, dtype=torch.int32),
        ks_block_table=torch.zeros(n, 4, dtype=torch.int32),
        ks_sched=torch.zeros(4, 2, dtype=torch.int32),
        ks_local_len=256,
        ks_row_lens=torch.zeros(n, dtype=torch.int32),
        row_bounds=torch.zeros(16, dtype=torch.int32) if bounds else None,
        counts_out=torch.zeros(16, dtype=torch.int32) if bounds else None,
    )
    st = SimpleNamespace(dsa=dsa, num_tokens=n, num_generations=n)
    # ks_sel: no dsa_ks_select_pack runner, so today's chain (the merge under test) runs.
    rt = SimpleNamespace(ks_ws=ws, ks_sel=None, dsa_pools=SimpleNamespace(idx_k_cache=[None, None]))
    return st, rt, torch.zeros(16, 8), torch.zeros(16, 2)


def test_the_switch_is_on():
    ns = _constants(_tree(_MODELING))
    assert ns["_CONSTANTS"]["generation"]["dsa_key_shard_failing_fallback"] is True
    assert ns["_KS_FAILING_ONLY"] is True


@pytest.mark.parametrize("failing_only", [False, True], ids=["off", "on"])
@pytest.mark.parametrize("bounds", [True, False], ids=["bounds", "no_bounds"])
def test_idx_decode_passes_the_switch_to_the_merge(failing_only, bounds):
    ns, calls, log = _exec(failing_only)
    line = ns["_KS_FAILING_LINE"]
    for n in (6, 12):
        del calls[:]
        st, rt, q8, w = _step(n, bounds)
        topk, mask, kv_rows = ns["_idx_decode"](1, q8, w, st, rt)
        ws, d = rt.ks_ws, st.dsa
        assert kv_rows is False, "local token indices: the remap follows"
        assert [c[0] for c in calls] == ["logits", "topk", "pack", "merge"]
        assert topk.data_ptr() == d.topk_out.data_ptr() and topk.shape[0] == n
        assert (mask is not None) is bounds and (mask is None or mask.shape == (n,))
        pack, (_, merge, kwargs) = calls[2][1], calls[3]
        assert pack[0].data_ptr() == ws.local_topk.data_ptr() and pack[2] is ws
        assert merge[0] is topk and merge[1] is d.counts_out and merge[4] is ws and len(merge) == 5
        assert merge[2] is (d.row_bounds if bounds else None) and merge[3] is mask
        assert kwargs == ({"failing_only": True} if failing_only else {})
    # The stand-in logger records every call; the engine's info_once logs the first per process.
    assert log.once == ([(line, line)] * 2 if failing_only else [])


def test_the_info_line():
    line = _constants(_tree(_MODELING))["_KS_FAILING_LINE"]
    assert line.startswith(_LINE_PREFIX) and "\n" not in line
    assert f"\n{line}\n" in _README.read_text()
