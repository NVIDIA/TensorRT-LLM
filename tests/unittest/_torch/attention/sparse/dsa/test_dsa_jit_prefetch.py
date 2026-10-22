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
"""The DSA decode-metadata planner covers every batch size's Triton variant.

One real call fills the launch template (as warmup does); the planner then
plans every BLOCK_S class, and each real launch at a range of decode batch
sizes must hit a planned cache key.
"""

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp
from tensorrt_llm._torch.attention.backends.sparse.dsa import kernels
from tensorrt_llm._torch.attention.backends.sparse.dsa.jit_prefetch import DsaDecodeMetadataProvider

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

_MAX_BLOCKS, _MAX_SEQS, _NEXT_N = 512, 256, 4


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


def _call(n, record=True):
    """A real decode-step call, sliced from engine-sized buffers as the
    metadata object does."""
    dev = torch.device("cuda")
    tokens = n * _NEXT_N
    seq_lens = torch.full((_MAX_SEQS,), _NEXT_N, dtype=torch.int32, device=dev)
    kv_lens = torch.full((_MAX_SEQS,), 1000, dtype=torch.int32, device=dev)
    block_offsets = torch.zeros(_MAX_SEQS, _MAX_BLOCKS, dtype=torch.int32, device=dev)
    big = _MAX_SEQS * _NEXT_N
    req = torch.empty(big, dtype=torch.int32, device=dev)
    fp8 = torch.empty(big, dtype=torch.int64, device=dev)
    scale = torch.empty(big, dtype=torch.int64, device=dev)
    kv_ind = torch.empty(_MAX_SEQS + 1, dtype=torch.int64, device=dev)
    c_ind = torch.empty(_MAX_SEQS + 1, dtype=torch.int64, device=dev)
    args = (seq_lens[:n], kv_lens[:n], block_offsets[:n], req[:tokens], fp8[:tokens],
            scale[:tokens], kv_ind[: n + 1], c_ind[: n + 1])  # fmt: skip
    kw = dict(num_tokens=tokens, max_query_len=_NEXT_N, tokens_per_block=64,
              index_head_dim=128, quant_block_size=128, data_bytes_per_token=128)  # fmt: skip
    if not record:
        kernels.fused_dsa_decode_metadata(*args, **kw)
        return []
    with jp.shadow_launches() as rec:
        kernels.fused_dsa_decode_metadata(*args, **kw)
    return list(rec)


def test_planned_keys_cover_every_batch_size():
    kernels.FUSED_DSA_DECODE_TEMPLATE.clear()
    _call(8, record=False)  # warmup's call fills the template
    p = DsaDecodeMetadataProvider(max_batch_size=_MAX_SEQS)
    assert p
    planned = _keys(p.plan_all())
    assert len(planned) == len(p.num_seqs_classes())
    for n in (1, 2, 3, 5, 31, 64, 100, 129, 200, 256):
        real = _keys(_call(n))
        assert real <= planned, f"num_seqs={n}: unplanned {sorted(real - planned)}"
