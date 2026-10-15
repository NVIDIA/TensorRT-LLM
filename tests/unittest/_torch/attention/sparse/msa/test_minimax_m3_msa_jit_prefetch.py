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
    """Engine-sized buffers; a generation step is a row slice of them."""

    def __init__(self):
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
        self.block_table = torch.zeros(MAX_REQUESTS + 3, MAX_BLOCKS, dtype=torch.int32, device=dev)
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


def test_planned_keys_cover_other_batches_without_allocating():
    eng = _Engine()
    jpm._RECORDS.clear()
    jpm.start_recording()
    try:
        eng.decode(0, 2)
        torch.cuda.synchronize()
    finally:
        jpm.stop_recording()
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
