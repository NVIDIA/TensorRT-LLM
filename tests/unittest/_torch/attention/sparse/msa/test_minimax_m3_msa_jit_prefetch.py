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
"""The MiniMax-M3 MSA decode JIT-prefetch planner covers other batches.

One recorded warmup-like call of the block-sparse decode fills the records;
the planner's replays must then cover the real launches of other request
counts, also when the query, output and top-k views start behind context
rows, and must not allocate from the persistent buffer arena.
"""

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import jit_prefetch_msa as jpm
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import (
    triton_sparse_decode as tsd,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

KV_HEADS, GROUP, HEAD_DIM, PAGE, TOPK = 4, 16, 128, 128, 16
MAX_REQUESTS, SPAN, MAX_BLOCKS = 12, 4, 8


def _keys(calls):
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key

    keys = set()
    for call in calls:
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        for kw in variants:
            kw = dict(kw)
            kw["debug"] = kw.get("debug", jit_fn.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            _, kkc, _, _, binder = jit_fn.device_caches[driver.active.get_current_device()]
            _, spec, opts = binder(*call.args, **kw)
            keys.add((jit_fn.fn.__name__, compute_cache_key(kkc, spec, opts)))
    return keys


class _Engine:
    """Engine-sized buffers; a generation step is a row slice of them.
    ``blocks`` is the block table's width, blocks per sequence."""

    def __init__(self, blocks=MAX_BLOCKS):
        dev = torch.device("cuda")
        rows = (MAX_REQUESTS + 3) * SPAN
        heads = KV_HEADS * GROUP
        self.q = torch.zeros(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.out = torch.zeros(rows, heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        # The indexer builds the top-k table token-major; the decode reads it
        # head-major through a permuted view.
        self.topk = torch.zeros(rows, KV_HEADS, TOPK, dtype=torch.int32, device=dev)
        pages = (MAX_REQUESTS + 3) * MAX_BLOCKS
        self.k = torch.zeros(pages, KV_HEADS, PAGE, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.v = torch.zeros_like(self.k)
        self.block_table = torch.zeros(MAX_REQUESTS + 3, blocks, dtype=torch.int32, device=dev)
        self.seq_lens = torch.full((MAX_REQUESTS + 3,), 300, dtype=torch.int32, device=dev)

    def decode(self, nctx, ngen):
        tokens = slice(nctx * SPAN, (nctx + ngen) * SPAN)
        tsd.minimax_m3_sparse_attn_decode(
            self.q[tokens],
            self.k,
            self.v,
            self.topk[tokens].permute(1, 0, 2),
            self.block_table[nctx : nctx + ngen],
            self.seq_lens[nctx : nctx + ngen],
            sm_scale=HEAD_DIM**-0.5,
            output=self.out[tokens],
            decode_query_len=SPAN,
        )


def _record(eng):
    jpm._RECORDS.clear()
    jpm.start_recording()
    try:
        eng.decode(0, 2)
        torch.cuda.synchronize()
    finally:
        jpm.stop_recording()


def test_planned_keys_cover_other_batches_without_allocating():
    eng = _Engine()
    _record(eng)
    assert jpm.recorded_launch_classes() == 1
    provider = jpm.MiniMaxM3MsaDecodeProvider(max_num_requests=MAX_REQUESTS)

    allocated = []
    orig = tsd.get_memory_buffers().get_buffer

    def counting_get_buffer(*args, **kwargs):
        allocated.append(args)
        return orig(*args, **kwargs)

    tsd.get_memory_buffers().get_buffer = counting_get_buffer
    try:
        planned = _keys(provider.plan_all())
    finally:
        del tsd.get_memory_buffers().get_buffer
    assert not allocated, "planning allocated from the persistent buffer arena"
    for nctx, ngen in [(0, 1), (0, 5), (1, 1), (3, 7), (2, 12)]:
        with jp.shadow_launches() as rec:
            eng.decode(nctx, ngen)
        real = _keys(rec)
        assert real, "no launches"
        missing = real - planned
        assert not missing, f"nctx={nctx} ngen={ngen}: unplanned {sorted(missing)}"


@pytest.mark.parametrize("blocks", [(16, 17), (17, 16)], ids=["to_unaligned", "to_aligned"])
def test_a_block_table_of_another_width_reuses_the_kernels(blocks):
    """The final executor's block table is wider than the KV-cache estimation
    executor's (a larger max_seq_len), so its row stride differs, divisible by
    16 or not. The decode launches of either width reach the same kernels: the
    final executor's warmup compiles none of them again."""
    estimation, final = blocks
    _record(_Engine(blocks=estimation))
    planned = _keys(jpm.MiniMaxM3MsaDecodeProvider(max_num_requests=MAX_REQUESTS).plan_all())
    eng = _Engine(blocks=final)
    # Generation-only steps, as warmup's: every row starts aligned, so only
    # the table's width differs from the recorded launch.
    for ngen in (2, 5):
        with jp.shadow_launches() as rec:
            eng.decode(0, ngen)
        real = _keys(rec)
        assert real, "no launches"
        missing = real - planned
        assert not missing, f"width {final} after {estimation}: unplanned {sorted(missing)}"


def test_each_new_launch_class_calls_the_new_record_hook_once():
    """Warmup plans a decode launch class as soon as it records it: the hook
    runs once per new class, not again for a launch of a class recorded
    before, and no longer once recording stops."""
    jpm._RECORDS.clear()
    calls = []
    jpm.start_recording()
    try:
        jpm.set_on_new_record(lambda: calls.append(jpm.recorded_launch_classes()))
        eng = _Engine()
        eng.decode(0, 2)
        eng.decode(0, 5)
        assert calls == [1]
        _Engine(blocks=MAX_BLOCKS + 1).decode(0, 2)
        assert calls == [1, 2]
        torch.cuda.synchronize()
    finally:
        jpm.stop_recording()
    jpm.start_recording()
    try:
        _Engine(blocks=MAX_BLOCKS + 2).decode(0, 2)
    finally:
        jpm.stop_recording()
    assert calls == [1, 2]
