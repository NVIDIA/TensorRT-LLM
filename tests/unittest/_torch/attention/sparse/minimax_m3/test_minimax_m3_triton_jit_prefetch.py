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
"""The MiniMax-M3 Triton sparse attention JIT-prefetch planner covers the
backend's two kernels for any batch: the real launches of other row counts
and other longest-sequence lengths must each hit a planned Triton cache key.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import triton_kernels as tk
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.jit_prefetch import (
    MiniMaxM3TritonProvider,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

INDEX_HEADS, KV_HEADS, Q_HEADS, BLOCK = 4, 4, 64, 128
MAX_SEQ_LEN = 8192


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


def _provider():
    p = MiniMaxM3TritonProvider.__new__(MiniMaxM3TritonProvider)
    p.configs = [
        SimpleNamespace(
            num_index_heads=INDEX_HEADS,
            num_kv_heads=KV_HEADS,
            num_q_heads=Q_HEADS,
            block_size=BLOCK,
        )
    ]
    p.max_seq_len = MAX_SEQ_LEN
    return p


def _real(rows, max_k):
    dev = torch.device("cuda")
    g = Q_HEADS // KV_HEADS
    with jp.shadow_launches() as rec:
        tk.triton_block_max_score(torch.zeros(rows, INDEX_HEADS, max_k, device=dev), BLOCK)
        tk.triton_sparse_softmax(
            torch.zeros(rows, KV_HEADS, g, max_k, device=dev),
            torch.zeros(rows, KV_HEADS, max_k, dtype=torch.bool, device=dev),
        )
    return _keys(rec)


@pytest.mark.parametrize("rows", [1, 2, 7, 16, 32, 33, 512])
@pytest.mark.parametrize("max_k", [1, 2, 17, 64, 100, 128, 129, 256, 300, 2048, 2689, 4100, 8192])
def test_planned_keys_cover_any_batch(rows, max_k):
    planned = _keys(_provider().plan_all())
    real = _real(rows, max_k)
    assert real, "no launches"
    missing = real - planned
    assert not missing, f"rows={rows} max_k={max_k}: unplanned {sorted(missing)}"
