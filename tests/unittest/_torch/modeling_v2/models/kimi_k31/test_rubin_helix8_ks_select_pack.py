# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""helix8's ``dsa_ks_select_pack``: the key-sharded Top-K in one launch of helix16's select / pack /
merge (``rubin_dsa_key_shard_select_pack``), at the row counts whose first-use device check passed.

CPU (``cpu_only``), from helix8's modeling.py over recording stand-ins:

* the generation constant is on as shipped and ``_KS_SELECT_PACK`` follows it; off, ``_idx_decode``
  runs today's chain and returns local token indices (the remap follows), the workspace keeps
  today's size and no runner is built;
* on: a row count's first eager call runs the check (every cluster size compiled first, the check
  driven with today's chain and the select / pack as its two sides) and today's chain; a passed
  check makes the later calls at that row count one select / pack launch that writes KV pool rows
  (no remap), with the step's row lengths, counts, bounds and mask and the layer's KV-row arguments,
  and logs the README's INFO line once; a failed check warns and keeps today's chain; a capture
  before the check keeps today's chain and records nothing; other page sizes never take it;
* both ``_mla_generation`` sites skip the remap only for KV rows; the first decode step builds the
  runner only with the constant on, over a workspace sized for its LL region.
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
_FUNCTIONS = ("_idx_decode", "_ks_select", "_ks_select_check", "_ks_kv_rows", "_idx_remap")


def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


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
        self.once, self.warnings, self.warnings_once = [], [], []

    def info_once(self, msg, key=None):
        self.once.append((msg, key))

    def warning(self, msg):
        self.warnings.append(msg)

    def warning_once(self, msg, key=None):
        self.warnings_once.append((msg, key))


class _Torch:
    """``torch`` with a switchable ``torch.cuda.is_current_stream_capturing``."""

    def __init__(self):
        self.capturing = False
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: self.capturing)

    def __getattr__(self, name):
        return getattr(torch, name)


def _exec(on: bool, why: str = "", names=_FUNCTIONS):
    """helix8's functions ``names`` from its modeling.py with ``_KS_SELECT_PACK`` = ``on``, recording
    stand-ins for the kernels and the check (which drives both of its sides once and returns
    ``why``)."""
    tree = _tree()
    ns = _constants(tree)
    calls, log, fake_torch = [], _Log(), _Torch()
    ns["_KS_SELECT_PACK"] = on
    ns["_KS_SELECT_CHECKED"] = {}
    ns["_IDX_HEADS"], ns["_IDX_DIM"] = 2, 4
    ns["_KS_SELECT_PAGE"] = 64  # the entry's page size, imported by modeling.py

    def rec(name, result=None):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result(*args, **kwargs) if result else None

        return run

    def check(ws, n, width, row_lens, today, select, counters):
        calls.append(("check", (ws, n, width, row_lens, counters), {}))
        lens = torch.full((n,), 7, dtype=torch.int32)
        sides = (today(lens), select(lens))
        calls.append(("check sides", sides, {}))
        return why

    ns.update(
        torch=fake_torch,
        rubin_fp8_paged_mqa_logits=rec("logits", lambda *a, **k: a[-1]),
        rubin_topk_decode_cute=rec("topk"),
        rubin_dsa_key_shard_pack=rec("pack"),
        rubin_dsa_key_shard_merge=rec("merge"),
        rubin_dsa_key_shard_select_pack=rec("select"),
        key_shard_kv_rows=lambda *a: ("kv",) + a,
        key_shard_select_check=check,
        convert_req_index_to_global=rec("remap", lambda *a: a[2] + 1000),
        rubin_dsa_topk_remap=rec("remap_triton"),
        logger=log,
    )
    body = [copy.deepcopy(_function(tree, name)) for name in names]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns, calls, log, fake_torch


def _step(n: int, bounds: bool = True, tpb: int = 64, counts: bool = True):
    ws = SimpleNamespace(
        local_logits=torch.zeros(16, 512),
        local_topk=torch.zeros(16, 512, dtype=torch.int32),
        local_k=512,
        rows_max=16,
        counts=torch.zeros(16, dtype=torch.int32),
        state=torch.zeros(4, dtype=torch.int32),
    )
    sel = SimpleNamespace(
        merge=SimpleNamespace(state=torch.zeros(4, dtype=torch.int32)),
        compiled=[],
    )
    sel.precompile = lambda *a: sel.compiled.append(a)
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
        counts_out=torch.zeros(16, dtype=torch.int32) if counts else None,
        tokens_per_block=tpb,
        req_idx_gen=torch.zeros(n, dtype=torch.int32),
        block_table_gen=torch.zeros(2, 8, dtype=torch.int32),
    )
    st = SimpleNamespace(dsa=dsa, num_tokens=n, num_generations=n)
    pools = SimpleNamespace(idx_k_cache=[None, None], pool_params={1: (24, 5)})
    rt = SimpleNamespace(ks_ws=ws, ks_sel=sel, dsa_pools=pools)
    return st, rt, torch.zeros(16, 8), torch.zeros(16, 2)


def _names(calls):
    return [c[0] for c in calls]


# --- defaults --------------------------------------------------------------------------------


def test_the_switch_is_on_as_shipped():
    ns = _constants(_tree())
    gen = ns["_CONSTANTS"]["generation"]
    # On in the shipped stack (a row count whose check fails keeps today's chain).
    assert gen["dsa_ks_select_pack"] is True and ns["_KS_SELECT_PACK"] is True
    # The per-step cluster's cap: 1, the one-kernel exchange (measured fastest at 16.5k local keys);
    # logged inside dsa_ks_select_pack.
    assert gen["dsa_ks_select_cluster_max"] == ns["_KS_SELECT_CLUSTER_MAX"] == 1
    assert ns["_DETAILS"]["dsa_ks_select_cluster_max"] == "dsa_ks_select_pack"


def test_the_info_line():
    ns = _constants(_tree())
    line = ns["_KS_SELECT_LINE"]
    assert line.startswith(_LINE_PREFIX) and "\n" not in line
    assert f"\n{line}\n" in _README.read_text()
    assert ns["_KS_SELECT_OFF"].startswith(_LINE_PREFIX)
    assert ns["_KS_SELECT_UNSUPPORTED"].startswith(_LINE_PREFIX)


@pytest.mark.parametrize("bounds", [True, False], ids=["bounds", "no_bounds"])
def test_off_runs_todays_chain(bounds):
    ns, calls, log, _ = _exec(False)
    st, rt, q8, w = _step(6, bounds)
    del rt.ks_sel  # off: the runner is never read
    topk, mask, kv = ns["_idx_decode"](1, q8, w, st, rt)
    assert kv is False and topk.data_ptr() == st.dsa.topk_out.data_ptr()
    assert _names(calls) == ["logits", "topk", "pack", "merge"]
    assert log.once[-1][0] == ns["_KS_FAILING_LINE"] and len(log.once) == 1


# --- on: the check, then the select / pack -----------------------------------------------------


def test_first_use_checks_then_launches_the_select_pack():
    ns, calls, log, _ = _exec(True)
    st, rt, q8, w = _step(6)
    d, ws, sel = st.dsa, rt.ks_ws, rt.ks_sel
    # The first call checks and runs today's chain.
    topk, mask, kv = ns["_idx_decode"](1, q8, w, st, rt)
    assert kv is False
    assert _names(calls) == ["logits", "check", "topk", "pack", "merge", "remap", "select",
                             "check sides", "topk", "pack", "merge"]  # fmt: skip
    check = calls[1][1]
    assert check[0] is ws and check[1] == 6 and check[2] == d.ks_local_len
    assert check[3] is d.ks_row_lens and check[4] == (ws.state, sel.merge.state)
    assert sel.compiled == [(16, True, False)], "every cluster size compiled before the check"
    # The check's two sides: today's chain into scratch outputs, the remap; the select / pack with
    # the layer's KV-row arguments into other scratch outputs.
    (_, (tl, lens, local, k, nn), _), (_, (packed, plens, pws), _) = calls[2], calls[3]
    assert local.data_ptr() == ws.local_topk.data_ptr() and packed is local
    assert plens is lens and pws is ws
    assert tl.shape == (6, 256) and k == 512 and nn == 1
    merge = calls[4]
    assert merge[2] == {"failing_only": ns["_KS_FAILING_ONLY"]}
    assert merge[1][2] is d.row_bounds and merge[1][4] is ws
    assert merge[1][0].data_ptr() != d.topk_out.data_ptr(), "scratch rows"
    select = calls[6][1]
    assert select[0] is sel and select[1] is lens and select[2] == 6
    assert select[3].shape == (6, 2048) and select[3].data_ptr() != d.topk_out.data_ptr()
    assert select[5] is d.row_bounds and select[6].dtype == torch.bool
    assert select[7] == ("kv", d.block_table_gen, d.req_idx_gen, 24 * 64, 5 * 64)
    assert ns["_KS_SELECT_CHECKED"] == {(6, True): True} and not log.warnings
    # The checked row count: one select / pack launch into the step's buffers, no remap.
    del calls[:]
    topk, mask, kv = ns["_idx_decode"](1, q8, w, st, rt)
    assert kv is True and _names(calls) == ["logits", "select"]
    args = calls[1][1]
    assert args[0] is sel and args[1] is d.ks_row_lens and args[2] == 6
    assert args[3] is topk and topk.data_ptr() == d.topk_out.data_ptr()
    assert args[4] is d.counts_out and args[5] is d.row_bounds and args[6] is mask
    assert args[7] == ("kv", d.block_table_gen, d.req_idx_gen, 24 * 64, 5 * 64)
    line = ns["_KS_SELECT_LINE"]
    assert log.once.count((line, line)) == 1
    # Another row count is checked on its own first call.
    del calls[:]
    st12, rt12, q8, w = _step(12)
    rt12.ks_sel, rt12.ks_ws = sel, ws
    assert ns["_idx_decode"](1, q8, w, st12, rt12)[2] is False and "check" in _names(calls)
    assert ns["_KS_SELECT_CHECKED"] == {(6, True): True, (12, True): True}


def test_a_failed_check_warns_and_keeps_todays_chain():
    ns, calls, log, _ = _exec(True, why="KV rows differ on the ties logits")
    st, rt, q8, w = _step(6)
    for _ in range(3):
        assert ns["_idx_decode"](1, q8, w, st, rt)[2] is False
    assert _names(calls).count("check") == 1 and "select" in _names(calls)  # the check's side only
    assert _names(calls)[-4:] == ["logits", "topk", "pack", "merge"]
    assert ns["_KS_SELECT_CHECKED"] == {(6, True): False}
    want = ns["_KS_SELECT_OFF"].format(rows=6, why="KV rows differ on the ties logits")
    assert log.warnings == [want]
    assert not [m for m, _ in log.once if m == ns["_KS_SELECT_LINE"]]


def test_a_capture_before_the_check_keeps_todays_chain():
    ns, calls, log, fake_torch = _exec(True)
    st, rt, q8, w = _step(6)
    fake_torch.capturing = True
    assert ns["_idx_decode"](1, q8, w, st, rt)[2] is False
    assert _names(calls) == ["logits", "topk", "pack", "merge"]
    assert ns["_KS_SELECT_CHECKED"] == {} and rt.ks_sel.compiled == []
    # A capture after the check launches the select / pack.
    fake_torch.capturing = False
    ns["_idx_decode"](1, q8, w, st, rt)
    fake_torch.capturing = True
    del calls[:]
    assert ns["_idx_decode"](1, q8, w, st, rt)[2] is True and _names(calls) == ["logits", "select"]


def test_no_bounds_and_no_counts():
    ns, calls, _, _ = _exec(True)
    st, rt, q8, w = _step(6, bounds=False, counts=False)
    ns["_idx_decode"](1, q8, w, st, rt)
    assert ns["_KS_SELECT_CHECKED"] == {(6, False): True}
    assert rt.ks_sel.compiled == [(16, False, False)]
    del calls[:]
    topk, mask, kv = ns["_idx_decode"](1, q8, w, st, rt)
    assert kv is True and mask is None
    args = calls[1][1]
    assert args[4].data_ptr() == rt.ks_ws.counts.data_ptr() and args[5] is None and args[6] is None


def test_other_page_sizes_and_no_runner_keep_todays_chain():
    ns, calls, _, _ = _exec(True)
    st, rt, q8, w = _step(6, tpb=32)
    assert ns["_idx_decode"](1, q8, w, st, rt)[2] is False
    assert "check" not in _names(calls) and ns["_KS_SELECT_CHECKED"] == {}
    del calls[:]
    st, rt, q8, w = _step(6)
    rt.ks_sel = None
    assert ns["_idx_decode"](1, q8, w, st, rt)[2] is False
    assert _names(calls) == ["logits", "topk", "pack", "merge"]


# --- the call sites and the workspace --------------------------------------------------------


def test_mla_generation_skips_the_remap_only_for_kv_rows():
    body = ast.unparse(_function(_tree(), "_mla_generation"))
    decode = "topk, mask, kv = _idx_decode(i, q8, w, st, rt)"
    remap = (
        "indices = topk if kv else _idx_remap(i, topk, d.req_idx_gen, d.block_table_gen, st, rt)"
    )
    assert body.count(decode) == 2 and body.count(remap) == 2
    assert "_idx_remap(" not in body.replace(remap, "")


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_the_workspace_and_the_runner(on):
    calls = []
    ns, _, log, fake_torch = _exec(on, names=("_key_shard_ws", "_key_shard_select"))
    ns.update(
        rubin_dsa_key_shard_ws=lambda *a, **k: calls.append(("ws", a, k)) or "ws",
        rubin_dsa_key_shard_select_ws=lambda *a: calls.append(("sel", a, {})),
    )
    mapping = SimpleNamespace(tp_size=8)
    md = SimpleNamespace(key_shard_rows_max=78, key_shard_max_local_len=16512)
    assert ns["_key_shard_ws"](mapping, md) == "ws"
    # On, the buffer also holds the select / pack's LL region; off, the call is today's.
    want = {"select_pack": True} if on else {}
    assert calls == [("ws", (mapping, 78, 2048, 16512), want)]
    # A workspace the select / pack does not take: one warning, today's chain.
    del calls[:]
    assert ns["_key_shard_select"]("ws") is None
    assert calls == [("sel", ("ws", ns["_KS_SELECT_CLUSTER_MAX"]), {})], "the cluster cap"
    assert log.warnings_once == [(ns["_KS_SELECT_UNSUPPORTED"], ns["_KS_SELECT_UNSUPPORTED"])]
    forward = [
        ast.unparse(n)
        for n in ast.walk(_tree())
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("._replace")
        and any(k.arg == "ks_sel" for k in n.keywords)
    ]  # fmt: skip
    assert forward == [
        "rt._replace(ks_ws=ks_ws, ks_sel=_key_shard_select(ks_ws) if _KS_SELECT_PACK else None)"
    ]
