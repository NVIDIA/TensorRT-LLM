# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The staged pool-group block-offset copy with ``pool_group_tail_marks``, on the host only.

``runtime._PrepareStager.block_offsets`` copies a KV cache manager's attention-op pool-group
tables: per row the readable columns ``[0, cols)`` from the host page indices and the bad-page
word in every column past them. With ``_BlockTailMarks`` it fills only the columns a row may
not already hold as that word. The device table must end every step with the words the copy
without marks leaves.

The real ``_BlockTailMarks`` class and the real ``block_offsets`` body (with the helpers it
calls) are compiled from ``runtime.py`` onto a stand-in stager, with a stand-in manager and a
staging block that records the descriptors and executes them on host memory as the launch
kernel does (``BLOCKS``: ``K = scale * page``, ``V = K + kv_offset``, both the bad-page word
for ``BAD_PAGE_INDEX``; ``FILL``: the value). No torch device, no Triton.
"""

from __future__ import annotations
import __future__

import ast
import math
import sys
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_RUNTIME = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_107__tep/runtime.py"
)

BAD_PAGE_INDEX = -1
SENTINEL = -7
#: The stager methods ``block_offsets`` runs.
_STAGER_METHODS = (
    "block_offsets",
    "_forget_blocks",
    "_decline_blocks",
    "_decline",
    "_after_decline",
    "_dst",
)


def _compile_runtime_parts():
    """``_BlockTailMarks`` and a stager class holding ``_PrepareStager``'s ``_STAGER_METHODS``."""
    tree = ast.parse(_RUNTIME.read_text(), filename=str(_RUNTIME))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_addr":
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BLOCK_DECLINES" for t in node.targets
        ):
            keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_BlockTailMarks":
            keep.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "_PrepareStager":
            methods = [
                n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in _STAGER_METHODS
            ]
            assert {m.name for m in methods} == set(_STAGER_METHODS)
            keep.append(
                ast.ClassDef(
                    name="StagerMethods",
                    bases=[],
                    keywords=[],
                    body=methods,
                    decorator_list=[],
                    type_params=[],
                )
            )
    module = ast.fix_missing_locations(ast.Module(body=keep, type_ignores=[]))
    namespace = {"np": np, "math": math, "torch": torch, "Any": Any, "Optional": Optional}
    code = compile(
        module, str(_RUNTIME), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True
    )
    exec(code, namespace)
    return namespace


PARTS = _compile_runtime_parts()
BlockTailMarks = PARTS["_BlockTailMarks"]


class Stager(PARTS["StagerMethods"]):
    """The attributes ``_PrepareStager.__init__`` sets that ``block_offsets`` reads."""

    def __init__(self, staging, tail_marks):
        self.staging = staging
        self.tail_marks = tail_marks
        self.counts = Counter()
        self.blocks_memo = None
        self.plan = None
        self.attn_metadata = SimpleNamespace()
        self.declined = []
        self.block_cols = None
        self.block_seqs = 0
        self._watch = {}
        self.input_writes = {}


@pytest.fixture(autouse=True)
def _leaf_modules(monkeypatch):
    """The two names ``block_offsets`` imports inside its body."""
    resource_manager = types.ModuleType("tensorrt_llm._torch.pyexecutor.resource_manager")
    resource_manager.CacheTypeCpp = SimpleNamespace(CROSS="cross", SELF="self")
    kv_v2 = types.ModuleType("tensorrt_llm.runtime.kv_cache_manager_v2")
    kv_v2.BAD_PAGE_INDEX = BAD_PAGE_INDEX
    monkeypatch.setitem(sys.modules, resource_manager.__name__, resource_manager)
    monkeypatch.setitem(sys.modules, kv_v2.__name__, kv_v2)


class DeviceTable:
    """A host tensor presented as the device table ``[tables, rows, 2, columns]`` int32."""

    is_cuda = True

    def __init__(self, tensor):
        self.t = tensor
        self.dtype = tensor.dtype
        self.shape = tensor.shape

    def dim(self):
        return self.t.dim()

    def is_contiguous(self):
        return self.t.is_contiguous()

    def data_ptr(self):
        return self.t.data_ptr()

    def element_size(self):
        return self.t.element_size()


class Staging:
    """Records ``blocks`` / ``fill`` descriptors; ``launch`` executes them on host memory."""

    def __init__(self, memory):
        #: Every int32 buffer a descriptor may address: (first byte, flat int32 view).
        self.memory = memory
        self.pending = []
        self.fill_words = 0

    def room(self, descriptors, data_bytes=0, payloads=0):
        return True

    def blocks(self, dst_k, dst_v, src, count, scale, kv_offset, bad_page=0):
        self.pending.append(("blocks", dst_k, dst_v, src, count, scale, kv_offset, bad_page))

    def fill(self, dst, stride, count, value):
        assert stride == 1
        self.pending.append(("fill", dst, count, value))
        self.fill_words += count

    def _view(self, addr, count):
        for base, flat in self.memory:
            end = base + flat.numel() * 4
            if base <= addr and addr + count * 4 <= end:
                assert (addr - base) % 4 == 0
                start = (addr - base) // 4
                return flat[start : start + count]
        raise AssertionError(f"address {addr:#x} outside every buffer")

    def launch(self):
        for d in self.pending:
            if d[0] == "fill":
                _, dst, count, value = d
                self._view(dst, count).fill_(value)
            else:
                _, k, v, src, count, scale, kv_offset, bad_page = d
                pages = self._view(src, count).clone()
                bad = pages == BAD_PAGE_INDEX
                kw = torch.where(bad, torch.tensor(bad_page, dtype=torch.int32), scale * pages)
                vw = torch.where(
                    bad, torch.tensor(bad_page, dtype=torch.int32), scale * pages + kv_offset
                )
                self._view(k, count).copy_(kw)
                self._view(v, count).copy_(vw)
        self.pending.clear()
        return True


TOKENS_PER_BLOCK = 4
WIDTH = 48
HOST_ROWS = 8
#: (pool, scale, K/V offset) of the three attention-op tables: two life cycles of one page size
#: and one of another, as a target and a drafter in one manager.
GROUPS = ((0, 2, 1), (0, 2, 1), (1, 4, 2))


class Manager:
    """The ``KVCacheManagerV2`` attributes the staged copy reads."""

    def __init__(self, seed):
        gen = torch.Generator().manual_seed(seed)
        self.tokens_per_block = TOKENS_PER_BLOCK
        self.max_copy_beam_width = 1
        self.sparse_metadata_batch = None
        self._use_per_layer_page_tables = False
        self.kv_cache_type = "self"
        self._attention_op_pool_groups = [
            SimpleNamespace(pool_id=p, scale=s, kv_offset=o) for p, s, o in GROUPS
        ]
        self.host_kv_cache_block_offsets = torch.full(
            (2, HOST_ROWS, 2, WIDTH), BAD_PAGE_INDEX, dtype=torch.int32
        )
        self.kv_cache_map = {}
        self._gen = gen
        self._rows = {}

    def set_request(self, rid, row, capacity):
        """Request ``rid`` on host row ``row`` with ``capacity`` tokens: its pages, BAD past them."""
        self._rows[rid] = row
        self.kv_cache_map[rid] = SimpleNamespace(capacity=capacity)
        blocks = math.ceil(capacity / TOKENS_PER_BLOCK)
        host = self.host_kv_cache_block_offsets
        host[:, row] = BAD_PAGE_INDEX
        pages = torch.randint(0, 1 << 16, (2, 2, blocks), generator=self._gen, dtype=torch.int32)
        # Evicted sliding-window pages inside the allocation.
        pages[..., : blocks // 3] = BAD_PAGE_INDEX
        host[:, row, :, :blocks] = pages

    @property
    def index_mapper(self):
        rows = self._rows

        def get_copy_index(request_ids, num_contexts, beam_width, flag):
            return torch.tensor([rows[r] for r in request_ids], dtype=torch.int32)

        return SimpleNamespace(get_copy_index=get_copy_index)


def _run(manager, dst, request_ids, marks):
    staging = Staging(
        [
            (dst.data_ptr(), dst.t.view(-1)),
            (
                manager.host_kv_cache_block_offsets.data_ptr(),
                manager.host_kv_cache_block_offsets.view(-1),
            ),
        ]
    )
    stager = Stager(staging, marks)
    assert stager.block_offsets(manager, dst, list(request_ids), 1, 0, len(request_ids))
    assert not stager.declined
    staging.launch()
    return stager, staging


def _tables(rows=6, buffer=None):
    shape = (len(GROUPS), rows, 2, WIDTH)
    if buffer is None:
        t = torch.full(shape, SENTINEL, dtype=torch.int32)
    else:
        t = buffer[: math.prod(shape)].view(shape)
    return DeviceTable(t)


#: Steps: (request id, host row, capacity) per batch row; capacities that grow, shrink, a batch
#: that narrows and widens, rows that change request.
STEPS = (
    ((1, 0, 40), (2, 1, 9), (3, 2, 17)),
    ((1, 0, 44), (2, 1, 9), (3, 2, 17)),
    ((4, 3, 12), (2, 1, 13), (3, 2, 21)),
    ((4, 3, 12), (2, 1, 13)),
    ((5, 4, 150), (4, 3, 16), (2, 1, 13), (6, 5, 4)),
    ((7, 6, 8), (4, 3, 16)),
    ((7, 6, 8), (4, 3, 20), (8, 7, 60)),
)


def _apply(manager, step):
    for rid, row, capacity in step:
        manager.set_request(rid, row, capacity)
    return [rid for rid, _, _ in step]


def test_the_marked_copy_leaves_the_words_of_the_full_tail_copy():
    plain, marked = Manager(1), Manager(1)
    dst_plain, dst_marked = _tables(), _tables()
    marks = BlockTailMarks()
    filled_plain, filled_marked = [], []
    for step in STEPS:
        ids = _apply(plain, step)
        assert _apply(marked, step) == ids
        assert torch.equal(plain.host_kv_cache_block_offsets, marked.host_kv_cache_block_offsets)
        s_plain, st_plain = _run(plain, dst_plain, ids, None)
        s_marked, st_marked = _run(marked, dst_marked, ids, marks)
        assert s_plain.block_cols == s_marked.block_cols
        assert torch.equal(dst_plain.t, dst_marked.t), step
        filled_plain.append(st_plain.fill_words)
        filled_marked.append(st_marked.fill_words)
        cols = s_marked.block_cols
        # Every staged row holds the bad-page word from its readable width on.
        assert (dst_marked.t[:, : len(ids), :, cols:] == BAD_PAGE_INDEX).all()
        assert s_marked.counts["block tail columns filled"] + s_marked.counts[
            "block tail columns kept"
        ] == (WIDTH - cols) * len(GROUPS) * len(ids)
    # Without marks every step fills every row's tail; with them the first step does, a step
    # whose readable width does not drop over the same rows fills nothing (step 1), and every
    # other step fills only the drops.
    assert filled_marked[0] == filled_plain[0]
    assert filled_marked[1] == 0 < filled_plain[1]
    assert all(m <= p for m, p in zip(filled_marked, filled_plain))
    assert sum(filled_marked) < sum(filled_plain) / 2


def test_a_forgotten_mark_fills_the_whole_tail_again():
    manager = Manager(2)
    dst, want = _tables(), _tables()
    marks = BlockTailMarks()
    ids = _apply(manager, STEPS[0])
    _run(manager, dst, ids, marks)
    _run(manager, want, ids, None)
    # A native copy (a step the stager did not see) leaves other words in the tail; every
    # such step forgets the marks (``StepInputStaging.begin_if_eligible``, ``_forget_blocks``).
    dst.t[:, :, :, -5:] = 123
    want.t[:, :, :, -5:] = 123
    marks.forget()
    ids = _apply(manager, STEPS[1])
    _, staging = _run(manager, dst, ids, marks)
    _run(manager, want, ids, None)
    assert torch.equal(dst.t, want.t)
    assert staging.fill_words > 0


def test_a_table_sharing_words_with_a_marked_one_drops_its_marks():
    # Attention metadata of different batch sizes may take their tables from one buffer.
    buffer = torch.full((len(GROUPS) * 6 * 2 * WIDTH,), SENTINEL, dtype=torch.int32)
    small, large = _tables(4, buffer), _tables(6, buffer)
    ref_buffer = buffer.clone()
    ref_small, ref_large = _tables(4, ref_buffer), _tables(6, ref_buffer)
    manager, ref = Manager(3), Manager(3)
    marks = BlockTailMarks()
    sequence = (
        (small, ref_small, STEPS[0]),
        (large, ref_large, STEPS[4]),
        (small, ref_small, STEPS[1]),
    )
    for dst, want, step in sequence:
        ids = _apply(manager, step)
        _apply(ref, step)
        _run(manager, dst, ids, marks)
        _run(ref, want, ids, None)
        assert torch.equal(buffer, ref_buffer)


def test_a_per_pool_copy_or_a_decline_forgets_the_marks():
    marks = BlockTailMarks()
    dst = _tables()
    marks.marks(dst)[:] = 3
    manager = Manager(4)
    ids = _apply(manager, STEPS[0])
    manager._attention_op_pool_groups = None
    manager._index_scale_ints = [2, 4]
    manager.kv_offset = torch.tensor([1, 2])
    per_pool = DeviceTable(torch.full((2, 6, 2, WIDTH), SENTINEL, dtype=torch.int32))
    _run(manager, per_pool, ids, marks)
    assert (marks.marks(dst) == WIDTH).all()
    marks.marks(dst)[:] = 3
    stager = Stager(Staging([]), marks)
    stager.declined.append("an earlier write")
    assert not stager.block_offsets(manager, dst, ids, 1, 0, len(ids))
    assert (marks.marks(dst) == WIDTH).all()


def test_marks_keep_at_most_their_table_count():
    marks = BlockTailMarks(max_tables=2)
    tables = [_tables() for _ in range(3)]
    for t in tables:
        marks.marks(t)[:] = 1
    # The least recently marked table is dropped: unknown again, the row width.
    assert (marks.marks(tables[0]) == WIDTH).all()
    assert (marks.marks(tables[2]) == 1).all()
