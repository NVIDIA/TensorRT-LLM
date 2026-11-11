# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""``context_fmha_fmk`` on 32- and 64-token cache blocks, on the host only.

The op's host side (``ops/attention/context_fmha_fmk.py``) is loaded from its file with ``torch``
and ``tensorrt_llm._utils`` stubbed for the load; no kernel is built or launched. The tests pin:

* the planner covers every (request, 128-row block, head group, key tile) exactly once, whole or
  as pieces, for any lengths, and its cut pieces tile an item's key range without a gap;
* the kernel's block index math (a 64-token block read as its two 32-token halves) addresses, for
  every key tile and every TMA box of it, the bytes of the same 32 keys as the 32-token path over a
  pool that holds the same keys, and the box past a request's last block is out of bounds.

The kernel itself needs a GPU.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[9]
_OP = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_103__tep/ops/attention/context_fmha_fmk.py"
)
D, HKV, TILE_N, IDS_PER_BATCH = 64, 4, 64, 32


@pytest.fixture(scope="module")
def fmk():
    stubs = {
        "torch": types.ModuleType("torch"),
        "tensorrt_llm": types.ModuleType("tensorrt_llm"),
        "tensorrt_llm._utils": types.ModuleType("tensorrt_llm._utils"),
    }
    stubs["torch"].Tensor = object
    stubs["torch"].device = object
    stubs["tensorrt_llm._utils"].maybe_pin_memory = lambda t: t
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("fmk_under_test", _OP)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # the module's dataclass resolves its annotations here
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("fmk_under_test", None)
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


def _lengths(rng, n):
    q = [int(rng.integers(1, 700)) for _ in range(n)]
    return q, [x + int(rng.integers(0, 9000)) for x in q]


@pytest.mark.parametrize("cache_page", [32, 64])
def test_block_index_math_addresses_the_same_keys(cache_page):
    """Over a pool holding the same keys, key tile ``t`` / box ``p`` of the kernel's index math
    addresses the bytes of keys ``t * 64 + 32 p .. +31`` of the request."""
    rng = np.random.default_rng(5)
    boxes_per_block = cache_page // 32
    blocks_per_tile = TILE_N // cache_page
    tiles_per_batch = IDS_PER_BATCH // blocks_per_tile
    pages_per_tile = TILE_N // 32
    for _ in range(50):
        kv_len = int(rng.integers(1, 5000))
        num_blocks = -(-kv_len // cache_page)
        pool_blocks = num_blocks + int(rng.integers(0, 50))
        ids = rng.permutation(pool_blocks)[:num_blocks].astype(np.int64)
        kvh = int(rng.integers(0, HKV))
        oob_box = HKV * pool_blocks * boxes_per_block
        last_tile = (kv_len - 1) // TILE_N
        for t in range(last_tile + 1):
            # load_page_ids: lane l holds block 32 * bj + l (-1 past the last block)
            bj = t // tiles_per_batch
            lane_ids = np.full(IDS_PER_BATCH, -1, dtype=np.int64)
            for lane in range(IDS_PER_BATCH):
                page = bj * IDS_PER_BATCH + lane
                if page < num_blocks:
                    lane_ids[lane] = ids[page]
            for p in range(pages_per_tile):
                blk, half = p // boxes_per_block, p % boxes_per_block
                pid = lane_ids[(t % tiles_per_batch) * blocks_per_tile + blk]
                box = (pid * HKV + kvh) * boxes_per_block + half if pid >= 0 else oob_box
                first_key = t * TILE_N + 32 * p
                if first_key >= num_blocks * cache_page:
                    assert box == oob_box  # a box past the last block is zero-filled by the TMA unit
                    continue
                # The pool is [blocks, kv_heads, cache_page, D]; a box is 32 consecutive tokens of
                # D bytes: its byte offset is box * 32 * D.
                block = ids[first_key // cache_page]
                want = ((block * HKV + kvh) * cache_page + first_key % cache_page) * D
                assert box * 32 * D == want


def test_planner_covers_every_item_once(fmk):
    rng = np.random.default_rng(11)
    for n, ctas in ((1, 152), (5, 152), (23, 152), (40, 64)):
        q_lens, kv_lens = _lengths(rng, n)
        items = fmk._items(q_lens, kv_lens, 32, 4, 4, 64)
        sorted_items = sorted(items, key=lambda it: -it[3])
        per_cta, loads, pieces, num_slots = fmk._stream_k(
            sorted_items, ctas, sum(it[3] + 8.0 for it in items) / ctas + 30.0, 8.0, 13.0, 16
        )
        covered = {}
        for units in per_cta:
            for b, m, hg, t0, t1, mode, _slot in units:
                covered.setdefault((b, m, hg), []).append((t0, t1))
        assert len(covered) == len(items)
        for b, m, hg, t_end, _first_masked in items:
            spans = sorted(covered[(b, m, hg)])
            assert spans[0][0] == 0 and spans[-1][1] == t_end
            assert all(a[1] == c[0] for a, c in zip(spans, spans[1:]))
