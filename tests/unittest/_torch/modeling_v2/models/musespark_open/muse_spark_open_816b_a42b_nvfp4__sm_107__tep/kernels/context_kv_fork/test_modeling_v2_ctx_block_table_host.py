# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``drafter_block_table_decode`` on the host only: the one-launch decode's arithmetic and its
routing, without a device or Triton.

* The kernel's program (``block_table._ctx_block_table_kernel``), stepped in numpy column block
  by column block exactly as it is written, gives the words of the built-in
  ``DFlashWorker._refresh_ctx_block_tables`` (its body compiled from ``speculative/dflash.py``
  onto a stand-in worker) at the generation server's shapes, every column included.
* ``block_table.refresh`` leaves to the built-in what the built-in answers (no table, no rows,
  no source) without reporting a decline, and reports a call it cannot take.

The device check (the Triton kernel itself, in a CUDA graph, through the fork) is
``test_modeling_v2_ctx_block_table.py``.
"""

from __future__ import annotations
import __future__

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_BLOCK_TABLE = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/kernels/context_kv_fork/block_table.py"
)
_DFLASH = _ROOT / "tensorrt_llm/_torch/speculative/dflash.py"

BAD_PAGE_INDEX = -1
SLOTS, WIDTH = 97, 31180


def _load_block_table():
    spec = importlib.util.spec_from_file_location("_block_table_under_test", _BLOCK_TABLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


block_table = _load_block_table()


def _builtin_refresh():
    """``DFlashWorker._refresh_ctx_block_tables``, compiled from its source as a function."""
    tree = ast.parse(_DFLASH.read_text(), filename=str(_DFLASH))
    worker_cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DFlashWorker"
    )
    method = next(
        n
        for n in worker_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_refresh_ctx_block_tables"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": torch}
    exec(
        compile(
            module,
            str(_DFLASH),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace["_refresh_ctx_block_tables"]


builtin_refresh = _builtin_refresh()


def _kernel_program(src_row, num_cols, offset, divisor, block):
    """One program of ``_ctx_block_table_kernel`` (one row), as written, in int64/int32 numpy."""
    table_row = np.empty(num_cols, dtype=np.int32)
    extent = np.zeros(block, dtype=np.int32)
    for start in range(0, num_cols, block):
        cols = start + np.arange(block, dtype=np.int32)
        live = cols < num_cols
        loaded = np.where(live, src_row[np.minimum(cols, num_cols - 1)], 0).astype(np.int32)
        encoded = loaded.astype(np.int64) - offset
        valid = live & (encoded >= 0)
        extent = np.maximum(extent, np.where(valid, cols + 1, 0))
        # Triton's ``//`` on int64 truncates; the operand is non-negative, so it is the floor.
        decoded = np.maximum(encoded, 0) // divisor
        table_row[cols[live]] = decoded[live].astype(np.int32)
    return table_row, np.int64(extent.max())


def _decode_like_the_kernel(worker, src, num_seqs):
    rows = src[worker._ctx_pool_idx, :, 0].numpy()
    num_cols = rows.shape[1]
    block = min(block_table.BLOCK, 1 << (num_cols - 1).bit_length())
    tables = worker._ctx_block_tables.numpy()
    counts = worker._ctx_block_counts.numpy()
    for r in range(num_seqs):
        tables[r], counts[r] = _kernel_program(
            rows[r], num_cols, worker._ctx_block_offset, worker._ctx_block_divisor, block
        )


def _page_map(gen, pools, rows, width, offset, window):
    """Encoded offsets ``[pools, rows, 2, width]``: per row a live tail behind evicted
    placeholders, some entries below ``offset``, an empty row and a row live to its end."""
    src = torch.full((pools, rows, 2, width), BAD_PAGE_INDEX, dtype=torch.int32)
    for row in range(rows):
        extent = int(torch.randint(0, width + 1, (1,), generator=gen))
        if row == 1:
            extent = 0
        elif row == 2:
            extent = width
        lo = max(0, extent - int(torch.randint(1, window + 1, (1,), generator=gen)))
        live = extent - lo
        if live:
            pages = torch.randint(0, 1 << 22, (pools, 2, live), generator=gen, dtype=torch.int32)
            holes = torch.randint(0, live, (max(1, live // 16),), generator=gen)
            pages[..., holes] = BAD_PAGE_INDEX
            if offset:
                low = torch.randint(0, live, (max(1, live // 32),), generator=gen)
                pages[..., low] = torch.randint(0, offset, (len(low),), generator=gen).int()
            src[:, row, :, lo:extent] = pages
    return src


def _worker(src, *, pool_idx, offset, divisor, unified=True, slots=SLOTS):
    width = src.shape[3]
    gen = torch.Generator().manual_seed(7)
    return SimpleNamespace(
        _ctx_unified_cache=unified,
        _ctx_pool_idx=pool_idx,
        _ctx_block_offset=offset,
        _ctx_block_divisor=divisor,
        _ctx_block_tables=torch.randint(
            -(1 << 30), 1 << 30, (slots, width), generator=gen, dtype=torch.int32
        ),
        _ctx_block_counts=torch.randint(-5, 5, (slots,), generator=gen),
        _ctx_block_cols=torch.arange(1, width + 1, dtype=torch.long),
    )


def _twin(worker):
    other = SimpleNamespace(**vars(worker))
    other._ctx_block_tables = worker._ctx_block_tables.clone()
    other._ctx_block_counts = worker._ctx_block_counts.clone()
    return other


def _metadata(src, unified=True):
    name = "kv_cache_block_offsets" if unified else "draft_kv_cache_block_offsets"
    return SimpleNamespace(**{name: src})


@pytest.mark.parametrize(
    "num_seqs,width,offset,divisor,unified",
    [
        (1, 7, 0, 2, True),
        (5, 1000, 3, 10, True),
        (40, 4097, 1, 1, False),
        (96, WIDTH, 0, 2, True),
        (97, WIDTH, 5, 10, True),
    ],
)
def test_the_kernel_program_is_the_builtin_decode(num_seqs, width, offset, divisor, unified):
    gen = torch.Generator().manual_seed(num_seqs)
    src = _page_map(gen, 3, SLOTS, width, offset, window=130)
    want = _worker(src, pool_idx=2, offset=offset, divisor=divisor, unified=unified)
    got = _twin(want)
    assert builtin_refresh(want, _metadata(src, unified), num_seqs)
    _decode_like_the_kernel(got, src, num_seqs)
    # Every column of the batch's rows, the extents, and nothing outside the rows.
    assert torch.equal(got._ctx_block_tables, want._ctx_block_tables)
    assert torch.equal(got._ctx_block_counts, want._ctx_block_counts)


def test_what_the_builtin_answers_is_left_to_it(monkeypatch):
    src = _page_map(torch.Generator().manual_seed(0), 3, 4, 16, 0, window=8)
    worker = _worker(src, pool_idx=2, offset=0, divisor=2, slots=4)
    declined = []
    # No rows, no source, no table: the built-in returns False or raises; nothing reported.
    assert block_table.refresh(worker, _metadata(src), 0, declined.append) is None
    assert block_table.refresh(worker, SimpleNamespace(), 2, declined.append) is None
    unbound = _twin(worker)
    unbound._ctx_block_tables = None
    assert block_table.refresh(unbound, _metadata(src), 2, declined.append) is None
    assert not declined
    # A call it cannot take is reported with its reason and left to the built-in.
    monkeypatch.setattr(block_table, "_KERNEL", False)
    assert block_table.refresh(worker, _metadata(src), 2, declined.append) is None
    assert declined == ["no Triton"]
    monkeypatch.setattr(block_table, "_KERNEL", True)
    declined.clear()
    # A host source: not the CUDA table the kernel addresses.
    assert block_table.refresh(worker, _metadata(src), 2, declined.append) is None
    assert declined == ["the attention-op table is not a CUDA int32 [pools, rows, 2, blocks] table"]
