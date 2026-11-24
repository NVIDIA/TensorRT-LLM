# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cep target's DKV read (``_mla_read``): the fused indices against the plain steps.

Under DKV a DSA layer's sparse MLA reads a local buffer: page 0 (never written), the step's new
tokens in pages, then the history rows the rank's Top-K selects. ``_DKV_READ_FUSED`` makes the
indices in one CuTe DSL launch (``rubin_dkv_topk_read_fused``: the Top-K remap over the new
tokens' pages, the copied history rows, the -1 padding at row 0, page 0 zeroed); off, the target
runs page 0's ``zero_``, ``_idx_remap``, ``rubin_dkv_topk_read`` and ``clamp(min=0)``.

CPU (``cpu_only``), on modeling.py's own ``_mla_read`` and ``_idx_remap`` (read from the source,
run over stand-ins of the entries that follow their documented rules): both settings give the
same indices and the same buffer bytes over the new tokens' pages, page 0 and the copied rows,
with the new rows appended (no owner writes), copied from their owners (owner writes) or none
copied (owner writes, no new row of this rank's requests); the fused setting calls only the fused
entry, the plain one the remap of its row count (Triton from 256 rows, C++ below) and the read.

On one sm_107 GPU, the target's ``_mla_read`` with the real entries (the append stood in): the
same equalities bit for bit, page 0 zero in both over memory first filled with E4M3 NaN bytes.
"""

from __future__ import annotations
import __future__

import ast
import copy
import functools
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_TARGET = "kimi_k31_pro_nvfp4__sm_107__cep"
_MODELING = _PKG_ROOT / "_torch/_experimental/modeling_v2/models/kimi_k31" / _TARGET / "modeling.py"
_TPB = 64
_ROW_BYTES = 576
_TOPK = 2048
#: The layer the reads stand for, and its offset in the latent pool's pages.
_LAYER = 3
_LAYER_OFFSET = 2
#: An E4M3 NaN byte.
_POISON = 0x7F

on_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the cep target runs on sm_107",
)

# (cached, new, this rank's rows per request): the Triton remap side with a history, the C++
# side with a cached length off the page grid and a fresh request (-1 padding), and a step
# without history tokens.
_CASES = {
    "history_300_rows": ([5000], [300], [(0, 300)]),
    "two_requests_70_rows": ([700, 0], [40, 30], [(0, 40), (1, 30)]),
    "no_history": ([0], [100], [(0, 100)]),
}
_OWNERS = ("append", "owner_write", "owner_none_new")


# --- the target's code, run from modeling.py ---


@functools.lru_cache(maxsize=None)
def _tree() -> ast.Module:
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _assigned(name: str) -> ast.Assign:
    for node in _tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node
    raise AssertionError(f"modeling.py assigns no {name}")


def _value(name: str, ns: dict):
    return eval(compile(ast.Expression(_assigned(name).value), "<constant>", "eval"), ns)


def _from_source(fused: bool, entries: dict) -> dict:
    """``_mla_read`` and ``_idx_remap`` of modeling.py over ``entries``, ``_DKV_READ_FUSED`` set
    to ``fused``."""
    ns = {
        "torch": torch,
        "_TARGET": _TARGET,
        "_IDX_TOPK": _TOPK,
        "_IDX_REMAP_TRITON_ROWS": _value("_IDX_REMAP_TRITON_ROWS", {}),
        "_DKV_READ_FUSED": fused,
        **entries,
    }
    body = [
        copy.deepcopy(n)
        for n in _tree().body
        if isinstance(n, ast.FunctionDef) and n.name in ("_mla_read", "_idx_remap")
    ]
    assert len(body) == 2
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    flags = __future__.annotations.compiler_flag
    exec(compile(module, str(_MODELING), "exec", flags=flags, dont_inherit=True), ns)
    return ns


# --- a step's DKV state ---


def _new_token_table(cached, new, tpb):
    """The read buffer's new-token pages (as ``rubin_cp_dkv.new_token_table``)."""
    blocks = [-(-(c + n) // tpb) for c, n in zip(cached, new)]
    table = torch.zeros(len(blocks), max(blocks), dtype=torch.int32)
    first = 1
    for r, (c, b) in enumerate(zip(cached, blocks)):
        table[r, c // tpb : b] = torch.arange(first, first + b - c // tpb, dtype=torch.int32)
        first += b - c // tpb
    return table, first


def _step(case: str, owner: str, device):
    """``(st, rt, topk, new_bytes)``: one step's DKV plan over a peer pool of random rows, the
    rank's Top-K rows (all positions up to the Top-K width, else random ones, ascending, -1 after)
    and the bytes the append leaves in the new tokens' pages."""
    cached, new, held = _CASES[case]
    gen = torch.Generator().manual_seed(7)
    lens = [c for c in cached]
    starts = [sum(lens[:r]) for r in range(len(lens))]
    tokens = sum(lens)
    table, pages = _new_token_table(cached, new, _TPB)
    new_rows = pages * _TPB
    peer_rows = 2 * tokens + 4 * new_rows + (_LAYER_OFFSET + 1) * _TPB
    src = torch.randint(0, 256, (peer_rows, _ROW_BYTES), dtype=torch.uint8, generator=gen)
    hist_src = torch.randperm(2 * tokens, generator=gen)[:tokens].to(torch.int64)
    rows, req = [], []
    cols = torch.arange(_TOPK)
    for r, count in held:
        for q in range(cached[r] + new[r] - count, cached[r] + new[r]):
            if q + 1 <= _TOPK:
                rows.append(torch.where(cols <= q, cols, -1))
            else:
                rows.append(torch.randperm(q + 1, generator=gen)[:_TOPK].sort().values)
            req.append(r)
    topk = torch.stack(rows).to(torch.int32)
    req = torch.tensor(req, dtype=torch.int32)
    new_bytes = torch.randint(
        0, 256, (new_rows - _TPB, _ROW_BYTES), dtype=torch.uint8, generator=gen
    )
    own = None
    if owner != "append":
        dst = []
        if owner == "owner_write":
            for r, _ in held:
                for pos in range(cached[r], cached[r] + new[r]):
                    dst.append(int(table[r, pos // _TPB]) * _TPB + pos % _TPB)
        n = len(dst)
        own = SimpleNamespace(
            new_src=torch.randperm(2 * tokens + 4 * new_rows, generator=gen)[:n],
            new_pos=torch.arange(n, dtype=torch.int64),
            new_count=torch.tensor([n], dtype=torch.int32),
            new_dst=torch.tensor(dst, dtype=torch.int64),
        )
        own = SimpleNamespace(**{k: v.to(device) for k, v in vars(own).items()})
    blocks, width = table.shape
    led = SimpleNamespace(
        pages=pages,
        table=table.to(device),
        offsets=table.view(1, blocks, 1, width).expand(1, blocks, 2, width).contiguous().to(device),
        hist_start=torch.tensor(starts, dtype=torch.int32, device=device),
        hist_len=torch.tensor(lens, dtype=torch.int32, device=device),
        hist_src=hist_src.to(device),
    )
    c = SimpleNamespace(
        dkv=led,
        owner=own,
        req_idx=req.to(device),
        sparse={"stand_in": True},
        req=torch.arange(len(cached), dtype=torch.int64, device=device),
    )
    pools = SimpleNamespace(
        tokens_per_block=_TPB,
        pool_params={_LAYER: (1, _LAYER_OFFSET)},
        row_bytes=_ROW_BYTES,
        num_local_layers=_LAYER + 1,
        peer_tokens=src.to(device),
    )
    st = SimpleNamespace(
        cp_mla=c, dsa=SimpleNamespace(tokens_per_block=_TPB), cp=SimpleNamespace(rank=0)
    )
    rt = SimpleNamespace(dsa_pools=pools)
    return st, rt, topk.to(device), new_bytes.to(device)


def _append_stand_in(new_bytes):
    """The append, as it leaves the new tokens' pages (every row of them, so all are defined)."""

    def append(i, latent, kv, pool, st, rt):
        assert latent is None and i == _LAYER
        kv[_TPB : _TPB + new_bytes.shape[0]] = new_bytes

    return append


def _defined_rows(st, kv, indices):
    """The buffer rows both settings define: page 0, the new tokens' rows the step writes and
    the copied history rows. With new rows (not ``owner_none_new``, whose Top-K still names new
    tokens no copy wrote) every row the indices name is one of them."""
    c = st.cp_mla
    new_rows = c.dkv.pages * _TPB
    rows = [torch.arange(_TPB, device=kv.device)]
    if c.owner is None:
        rows.append(torch.arange(_TPB, new_rows, device=kv.device))
    else:
        rows.append(c.owner.new_dst)
    copied = int((indices >= new_rows).sum()) and int(indices.max()) - new_rows + 1
    rows.append(torch.arange(new_rows, new_rows + copied, device=kv.device))
    rows = torch.cat(rows)
    if c.owner is None or c.owner.new_dst.numel():
        assert torch.isin(indices.view(-1).long(), rows).all(), "an index names an undefined row"
    return rows


# --- CPU: the stand-ins follow the entries' documented rules ---


def _remap_rule(req, table, topk, bs, k, stride, layer):
    """``convert_req_index_to_global`` / ``rubin_dsa_topk_remap``: -1 for a negative entry, a
    page past the table or a negative table entry."""
    tok = topk[:, :k].long()
    page = tok.clamp_min(0) // bs
    base = table[req.long()[:, None], page.clamp(max=table.shape[1] - 1)].long()
    rows = base * stride + tok.clamp_min(0) % bs + layer * bs
    bad = (tok < 0) | (page >= table.shape[1]) | (base < 0)
    return torch.where(bad, -1, rows).to(torch.int32)


def _read_rule(kv, new_rows, topk, req, new_idx, starts, lens, hist_src, src, off):
    """``rubin_dkv_topk_read``: the selected history rows copied into ``kv`` once each in
    ascending order behind the new rows; their entries replaced by those rows."""
    if hist_src.numel() == 0:
        return new_idx
    q = req.long()[:, None]
    hist = (topk >= 0) & (topk < lens.long()[q])
    flat = topk.long() + starts.long()[q]
    chosen = torch.unique(flat[hist])
    kv[new_rows : new_rows + chosen.numel()] = src[hist_src[chosen] + off]
    slot = torch.full((hist_src.numel(),), -1, dtype=torch.long)
    slot[chosen] = torch.arange(chosen.numel())
    out = new_idx.long().clone()
    out[hist] = new_rows + slot[flat[hist]]
    return out.to(torch.int32)


def _cpu_entries(calls: list, new_bytes):
    def remap(name):
        def run(*args):
            calls.append(name)
            return _remap_rule(*args)

        return run

    def read(*args):
        calls.append("read")
        return _read_rule(*args)

    def read_fused(kv, new_rows, topk, req, table, bs, stride, layer, *rest):
        calls.append("fused")
        kv[:bs] = 0
        new_idx = _remap_rule(req, table, topk, bs, topk.shape[1], stride, layer)
        return _read_rule(kv, new_rows, topk, req, new_idx, *rest).clamp(min=0)

    def rows_read(dst, src, src_rows, pos, count, off):
        calls.append("rows_read")
        n = int(count[0])
        dst[:n] = src[src_rows[pos[:n]] + off]
        return dst

    return {
        "rubin_dsa_topk_remap": remap("triton"),
        "convert_req_index_to_global": remap("cpp"),
        "rubin_dkv_topk_read": read,
        "rubin_dkv_topk_read_fused": read_fused,
        "rubin_dkv_rows_read": rows_read,
        "read_pointers": lambda kv, n: (kv.data_ptr(), n),
        "_mla_read_append": _append_stand_in(new_bytes),
    }


@pytest.mark.cpu_only
@pytest.mark.parametrize("owner", _OWNERS)
@pytest.mark.parametrize("case", sorted(_CASES))
def test_fused_read_equals_the_plain_steps(case, owner):
    results = {}
    for fused in (False, True):
        st, rt, topk, new_bytes = _step(case, owner, "cpu")
        calls = []
        ns = _from_source(fused, _cpu_entries(calls, new_bytes))
        kv, sparse, indices = ns["_mla_read"](_LAYER, None, topk, st, rt)
        results[fused] = (kv, sparse, indices, calls)
    (kv_p, sp_p, idx_p, calls_p), (kv_f, sp_f, idx_f, calls_f) = results[False], results[True]
    rows = topk.shape[0]
    remap = "triton" if rows >= 256 else "cpp"
    rows_read = [] if owner != "owner_write" else ["rows_read"]
    assert calls_p == rows_read + [remap, "read"], calls_p
    assert calls_f == rows_read + ["fused"], calls_f
    assert torch.equal(idx_f, idx_p)
    assert (idx_f >= 0).all()
    defined = _defined_rows(st, kv_f, idx_f)
    assert torch.equal(kv_f[defined], kv_p[defined])
    assert (kv_f[:_TPB] == 0).all()
    assert sp_f.keys() == sp_p.keys() and torch.equal(
        sp_f["kv_cache_block_offsets"], sp_p["kv_cache_block_offsets"]
    )


# --- GPU: the target's _mla_read with the real entries ---


@on_sm107
@pytest.mark.parametrize("owner", _OWNERS)
@pytest.mark.parametrize("case", sorted(_CASES))
def test_fused_read_equals_the_plain_steps_on_the_gpu(case, owner, monkeypatch):
    cep = importlib.import_module(f"{_PKG}.{_TARGET}.modeling")
    results = {}
    for fused in (False, True):
        st, rt, topk, new_bytes = _step(case, owner, "cuda")
        monkeypatch.setattr(cep, "_mla_read_append", _append_stand_in(new_bytes))
        monkeypatch.setattr(cep, "_DKV_READ_FUSED", fused)
        new_rows = st.cp_mla.dkv.pages * _TPB
        size = new_rows + min(st.cp_mla.dkv.hist_src.numel(), topk.numel())
        # Hand the buffer's allocation E4M3 NaN bytes: free a block of its size filled with them.
        poison = torch.full((size, _ROW_BYTES), _POISON, dtype=torch.uint8, device="cuda")
        del poison
        kv, sparse, indices = cep._mla_read(_LAYER, None, topk, st, rt)
        torch.cuda.synchronize()
        results[fused] = (kv.clone(), indices.clone(), st)
    (kv_p, idx_p, _), (kv_f, idx_f, st) = results[False], results[True]
    assert torch.equal(idx_f, idx_p)
    defined = _defined_rows(st, kv_f, idx_f)
    assert torch.equal(kv_f[defined], kv_p[defined])
    assert (kv_f[:_TPB] == 0).all() and (kv_p[:_TPB] == 0).all()
