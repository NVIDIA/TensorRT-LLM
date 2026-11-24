# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 window plan entry (``rubin_dflash2_window_plan``).

CPU (any host): on every cell and every CP rank the reference plan gives each block query
exactly its window: over the ranks its body, head and block keys cover ``[lo_j, L + j]`` once,
each key on the rank whose pages hold it, at the slot ``HelixDraftShard.local_slots`` gives it
there; the chunks partition each body, whose pages are valid up to its length and -1 after. The
history length
the plan reads is the block's RoPE base: for every live row the engine's fused prepare
advertises ``num_ctx`` equal to the first query position (the prepare kernel runs in Triton's
interpreter in a fresh process, which loads only that kernel file and this file's two check
functions).

GPU (SM107): the plan kernel equals the reference exactly on every cell; a plan captured in a
CUDA graph and replayed on new lengths and a new table equals an eager plan; no compile after
the warmup pass; the RoPE-base check on the device.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
    rubin_dflash2_window as kw,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    plan_buffers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_dflash2_window_plan import (  # noqa: E501
    rubin_dflash2_window_plan as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

_ON_SM107 = torch.cuda.is_available() and get_sm_version() in (103, 107)
gpu = pytest.mark.skipif(
    not _ON_SM107, reason="the window entries run on SM103 (GB300) and SM107 (Rubin)"
)


def _build(spec: dict, seed: int, device):
    """Lengths, a global page table with a distinct pool page per (request, page), and the
    request -> table row map when the cell reads rows of a shared table."""
    g = torch.Generator().manual_seed(seed)
    lengths = list(spec["lengths"])
    batch, blk, page, cp = len(lengths), spec["block_size"], spec["page_size"], spec["cp_size"]
    span = page * cp
    gcols = -(-(max(lengths) + blk + span) // span) * cp
    table = torch.randperm(batch * gcols + 3, generator=g)[: batch * gcols].view(batch, gcols)
    rows = None
    if spec["table_rows"]:
        rows = torch.randperm(batch + 5, generator=g)[:batch]
    num_ctx = torch.tensor(lengths, dtype=torch.int64)
    return num_ctx.to(device), table.to(device), None if rows is None else rows.to(device)


def _local(table: torch.Tensor, rows, cp: int, rank: int, num_rows: int) -> torch.Tensor:
    """Rank ``rank``'s table (local page ``c`` = global page ``c * cp + rank``), placed at the
    requests' rows of a larger shared table when ``rows`` is given."""
    local = table[:, rank::cp].to(torch.int32)
    if rows is None:
        return local.contiguous()
    shared = torch.full((num_rows, local.shape[1]), 2**30, dtype=torch.int32, device=table.device)
    shared[rows] = local
    return shared


def _args(spec, num_ctx, table, rows, rank):
    cp = spec["cp_size"]
    num_rows = 0 if rows is None else int(rows.max()) + 3
    return (
        num_ctx,
        _local(table, rows, cp, rank, num_rows),
        rows,
        spec["block_size"],
        spec["window"],
        spec["page_size"],
        cp,
        rank,
        spec["num_splits"],
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def _ids(cells):
    return [c.why[:56] for c in cells]


def _helix_draft_shard_class():
    """``HelixDraftShard`` as the engine defines it.

    Imported when the engine's import chain loads; on a CPU host without flashinfer the class
    is compiled from its own source in ``speculative/dflash_attention.py`` instead (it needs only
    torch and ``dataclass``), so the check still runs against the engine's code, not a copy.
    """
    try:
        from tensorrt_llm._torch.speculative.dflash_attention import HelixDraftShard

        return HelixDraftShard
    except Exception:  # noqa: BLE001 - any import failure of the heavy chain
        pass
    import tensorrt_llm

    path = Path(tensorrt_llm.__path__[0]) / "_torch" / "speculative" / "dflash_attention.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HelixDraftShard")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, "dataclass": dataclasses.dataclass}
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - the engine's own source
    return namespace["HelixDraftShard"]


# CPU


@pytest.mark.cpu_only
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
def test_reference_plan_is_each_querys_window(cell):
    spec = cell.spec
    num_ctx, table, rows = _build(spec, _seed(cell), "cpu")
    blk, page, window, cp = spec["block_size"], spec["page_size"], spec["window"], spec["cp_size"]
    # pool page -> (request, global page)
    owner_req = torch.full((int(table.max()) + 1,), -1, dtype=torch.long)
    owner_page = torch.full_like(owner_req, -1)
    owner_req[table.reshape(-1)] = torch.arange(table.shape[0]).repeat_interleave(table.shape[1])
    owner_page[table.reshape(-1)] = torch.arange(table.shape[1]).repeat(table.shape[0])
    shard_cls = _helix_draft_shard_class()
    seen = {}
    for rank in range(cp):
        plan = op.reference(*_args(spec, num_ctx, table, rows, rank), None)
        shard = shard_cls(cp, rank, page)
        local = table[:, rank::cp].long()
        for b, length in enumerate(num_ctx.tolist()):
            body_len = int(plan.body_len[b])
            t = torch.arange(body_len)
            body_rows = plan.body_pages[b].long()[t // page] * page + t % page
            bounds = plan.chunk_bounds[b].tolist()
            assert bounds[0] == 0 and bounds[-1] == body_len, "the chunks partition the body"
            assert all(lo <= hi and lo % page == 0 for lo, hi in zip(bounds, bounds[1:]))
            pages = int((plan.body_pages[b] >= 0).sum())
            assert pages == -(-body_len // page) and (plan.body_pages[b, pages:] == -1).all()
            head = plan.head_rows[b].long()
            for j in range(blk):
                key_rows = torch.cat(
                    (
                        body_rows,
                        head[(head >= 0) & (torch.arange(len(head)) >= plan.head_first[b, j])],
                    )
                )
                assert (owner_req[key_rows // page] == b).all(), "a key of another request"
                gpage = owner_page[key_rows // page]
                assert (gpage % cp == rank).all(), "a key on another rank's page"
                positions = (gpage * page + key_rows % page).tolist()
                bits = int(plan.block_bits[b])
                positions += [length + k for k in range(j + 1) if (bits >> k) & 1]
                assert all(
                    ((length + k) // page) % cp == rank for k in range(blk) if (bits >> k) & 1
                )
                # The store's rule: the rank owns each key at the slot local_slots names.
                pos = torch.tensor(positions, dtype=torch.long)
                slots, owned = shard.local_slots(pos)
                assert bool(owned.all()), f"rank {rank} reads a position it does not own"
                hist = slots[pos < length]
                stored = local[b][hist // page] * page + hist % page
                assert sorted(stored.tolist()) == sorted(key_rows.tolist())
                seen.setdefault((b, j), []).extend(positions)
    for (b, j), positions in seen.items():
        p = int(num_ctx[b]) + j
        assert sorted(positions) == list(range(max(0, p - window + 1), p + 1)), (b, j)


def _prepare_case(prepare, device):
    """One fused-prepare call over live rows (and a dummy row) with the draft reserve in place."""
    g = torch.Generator().manual_seed(29)
    blk = width = 8
    page = 64
    old = [100, 1023, 0, 5000, 64, 1, 9, 0]  # per slot; slot 7 is the dummy slot
    slots = torch.tensor([3, 0, 5, 1, 7, 2, 6], dtype=torch.int64)
    gens = slots.shape[0]
    dummy = 7
    nacc = torch.randint(1, width + 1, (gens,), generator=g)
    capacity = [-(-(old[s] + width + blk) // page) for s in slots.tolist()]
    max_blocks = max(capacity) + 2
    tables = torch.randperm(gens * max_blocks, generator=g).view(gens, max_blocks).to(torch.int32)
    prep = prepare(
        nacc.to(device),
        torch.randint(0, 1000, (gens, width), generator=g).to(device),
        slots.to(device),
        torch.tensor(old, dtype=torch.int64, device=device),
        torch.tensor(capacity, dtype=torch.int32, device=device),
        tables.to(device),
        torch.zeros(gens, dtype=torch.int32, device=device),
        0,
        dummy,
        1 << 30,
        page,
        blk,
        width,
    )
    return prep, slots != dummy, blk


def _check_rope_base(prepare, device):
    prep, live, blk = _prepare_case(prepare, device)
    live = live.to(prep.num_ctx.device)
    first = prep.query_positions[:, 0]
    assert torch.equal(prep.num_ctx[live], first[live]), "num_ctx is not the block's RoPE base"
    # The plan places block query j at num_ctx + j: the positions the block was roped at.
    j = torch.arange(blk, device=first.device)
    assert torch.equal((prep.num_ctx[:, None] + j)[live], prep.query_positions[live])
    assert (prep.num_ctx[~live] == 0).all(), "dummy rows attend an empty history"


_INTERPRETED_MAIN = """
import importlib.util
import sys

import torch

{functions}

spec = importlib.util.spec_from_file_location("_dflash2_spec_tail_interpreted", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
_check_rope_base(module.dflash_paged_latent_prepare, "cpu")
print("ok rope base")
"""


@pytest.mark.cpu_only
def test_num_ctx_is_the_block_rope_base_interpreted():
    """The fused prepare in Triton's interpreter, in a fresh process (see the module doc):
    TRITON_INTERPRET must be set before triton is first imported."""
    source = Path(kw.__file__).resolve().parents[6] / "speculative" / "spec_tail_kernels.py"
    functions = inspect.getsource(_prepare_case) + "\n\n" + inspect.getsource(_check_rope_base)
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-c", _INTERPRETED_MAIN.format(functions=functions), str(source)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0 and "ok rope base" in proc.stdout, proc.stdout + proc.stderr


# GPU


def _filled_buffers(spec, batch, device):
    buffers = plan_buffers(
        batch + 2,
        spec["block_size"],
        spec["window"],
        spec["page_size"],
        spec["cp_size"],
        spec["num_splits"],
        device,
    )
    for t in buffers:
        t.fill_(-7)  # what a larger earlier step could have left
    return buffers


@gpu
@pytest.mark.parametrize("cell", op.CELLS, ids=_ids(op.CELLS))
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    num_ctx, table, rows = _build(spec, _seed(cell), "cuda")
    for rank in range(spec["cp_size"]):
        args = _args(spec, num_ctx, table, rows, rank)
        got = op(*args, _filled_buffers(spec, num_ctx.shape[0], "cuda"))
        want = op.reference(*args, None)
        for name, a, b in zip(want._fields, got, want):
            assert torch.equal(a, b), f"rank {rank}: {name} differs"


@gpu
def test_capture_on_one_input_set_replay_on_another():
    spec = dict(op.CELLS[3].spec)  # CP4, 15 requests
    batch = len(spec["lengths"])
    sets = [
        _build(spec, 1, "cuda"),
        _build(dict(spec, lengths=tuple(reversed(spec["lengths"]))), 2, "cuda"),
    ]
    rank = 2
    tables = [_local(t, None, spec["cp_size"], rank, 0) for _, t, _ in sets]
    num_ctx = torch.zeros(batch, dtype=torch.int64, device="cuda")
    table = torch.zeros(batch, max(t.shape[1] for t in tables), dtype=torch.int32, device="cuda")
    buffers = _filled_buffers(spec, batch, "cuda")

    def load(i):
        num_ctx.copy_(sets[i][0])
        table.zero_()
        table[:, : tables[i].shape[1]] = tables[i]

    def step():
        return op(
            num_ctx,
            table,
            None,
            spec["block_size"],
            spec["window"],
            spec["page_size"],
            spec["cp_size"],
            rank,
            spec["num_splits"],
            buffers,
        )

    load(0)
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    load(1)
    graph.replay()
    replayed = [t.clone() for t in captured]
    want = op.reference(
        num_ctx,
        table,
        None,
        spec["block_size"],
        spec["window"],
        spec["page_size"],
        spec["cp_size"],
        rank,
        spec["num_splits"],
        None,
    )
    for name, a, b in zip(want._fields, replayed, want):
        assert torch.equal(a, b), f"replay: {name} differs from the plan of the new inputs"


@gpu
def test_no_compile_after_the_warmup_pass():
    spec = dict(op.CELLS[0].spec)

    def variants():
        caches = getattr(kw._window_plan_kernel, "device_caches", None)
        if caches is None:
            pytest.skip("this Triton keeps no per-device kernel cache to count")
        return sum(len(entry[0]) for entry in caches.values())

    counts = []
    for seed in (1, 2):
        for batch in (1, 2, 4, 8, 12):
            sub = dict(spec, lengths=spec["lengths"][seed : seed + batch] or (5,))
            num_ctx, table, rows = _build(sub, seed * 100 + batch, "cuda")
            op(*_args(sub, num_ctx, table, rows, 0), _filled_buffers(sub, num_ctx.shape[0], "cuda"))
        torch.cuda.synchronize()
        counts.append(variants())
    assert counts[1] == counts[0], f"compiled {counts[1] - counts[0]} variants after warmup"


@gpu
def test_num_ctx_is_the_block_rope_base():
    from tensorrt_llm._torch.speculative.spec_tail_kernels import dflash_paged_latent_prepare

    _check_rope_base(dflash_paged_latent_prepare, "cuda")
