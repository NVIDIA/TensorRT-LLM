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
"""nvfp4_ds_mla context prefill through an E4M3 scratch (Kimi K3.1 context servers): the paged record
conversion equals the PrimTS nvfp4.fp8rope loader's, and the native E4M3 kernel over the scratch returns the in-place
NVFP4 kernel's output bit for bit."""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse import nvfp4_ds_mla_kv as rec
from tensorrt_llm._torch.attention.backends.sparse.dsa import prims_ts_prefill as ptp
from tensorrt_llm._utils import get_sm_version

TPB, NUM_LAYERS, LAYER = 64, 2, 1
HEADS, TOPK = 96, 2048


def _skip_unless_sm100():
    if not torch.cuda.is_available() or get_sm_version() not in (100, 103):
        pytest.skip("PrimTS nvfp4.fp8rope requires SM100 / SM103")


def _paged_pool(kv_lens, seed):
    """Records of ``len(kv_lens)`` requests in a paged pool (``NUM_LAYERS`` layers interleaved per block, layer
    ``LAYER`` written), blocks in random order. Returns pool, block table, kv_indptr and the latent rows."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks = [math.ceil(n / TPB) for n in kv_lens]
    total_blocks = sum(blocks) + 7
    phys = torch.randperm(total_blocks, generator=g, device="cuda").to(torch.int32)
    table = torch.full((len(kv_lens), max(blocks)), -1, dtype=torch.int32, device="cuda")
    o = 0
    for r, b in enumerate(blocks):
        table[r, :b] = phys[o : o + b]
        o += b
    pool = torch.zeros(
        total_blocks * NUM_LAYERS * TPB, rec.TOKEN_BYTES, dtype=torch.uint8, device="cuda"
    )
    rows = []
    for r, n in enumerate(kv_lens):
        x = torch.randn(n, 576, generator=g, device="cuda") * 0.5
        x[torch.randint(0, n, (max(1, n // 97),), generator=g, device="cuda"), :512] *= (
            900.0  # saturating blocks
        )
        x = x.to(torch.bfloat16)
        pos = torch.arange(n, device="cuda")
        loc = table[r, pos // TPB].long() * (NUM_LAYERS * TPB) + LAYER * TPB + pos % TPB
        rec.quant_scatter(pool, loc, x)
        rows.append(x)
    kv_indptr = torch.zeros(len(kv_lens) + 1, dtype=torch.int64, device="cuda")
    kv_indptr[1:] = torch.tensor(kv_lens, device="cuda").cumsum(0)
    return pool, table, kv_indptr, rows


@pytest.mark.parametrize(
    "kv_lens", [[1000, 4097, 130], [64], [1100000]]
)  # > 65535 x 16-token blocks
def test_paged_records_to_e4m3_matches_the_loader(kv_lens):
    """Each NoPE value = E2M1 x block scale rounded once to E4M3 with saturation; RoPE bytes copied."""
    _skip_unless_sm100()
    pool, table, kv_indptr, _ = _paged_pool(kv_lens, seed=len(kv_lens))
    out = torch.empty(sum(kv_lens), 576, dtype=torch.uint8, device="cuda")
    rec.paged_records_to_e4m3(
        pool, table, kv_indptr, max(kv_lens), TPB, NUM_LAYERS * TPB, LAYER, out
    )
    for r, n in enumerate(kv_lens):
        pos = torch.arange(n, device="cuda")
        loc = table[r, pos // TPB].long() * (NUM_LAYERS * TPB) + LAYER * TPB + pos % TPB
        records = pool[loc]
        ref = rec.dequant_rows(records)
        ref_nope = ref[:, :512].clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
        got = out[kv_indptr[r] : kv_indptr[r + 1]]
        assert torch.equal(got[:, :512], ref_nope)
        assert torch.equal(got[:, 512:], records[:, 256:320])


def _local_topk(rows_per_req, kv_lens, seed):
    """Row j of request r draws TOPK causal positions of [0, kv_len[r] - rows[r] + j] (-1 past the window)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    his = [
        torch.arange(n - m + 1, n + 1, device="cuda").clamp(min=1)
        for m, n in zip(rows_per_req, kv_lens)
    ]
    hi = torch.cat(his)
    t = (
        (torch.rand(hi.numel(), TOPK, generator=g, device="cuda") * hi[:, None])
        .long()
        .to(torch.int32)
    )
    t[torch.arange(TOPK, device="cuda")[None, :] >= hi[:, None]] = -1
    return t


@pytest.mark.parametrize(
    "rows_per_req,kv_lens",
    [
        ([8192], [131072]),  # an 8K-row chunk of a 123K-token prefix
        ([8192], [608192]),  # the 600K-prefix chunk of the mixed AgentX point
        ([1237], [24576]),  # a short tail chunk
        ([3000, 2000, 1001], [9000, 40000, 1001]),  # three requests packed into one chunk
    ],
)
def test_dequant_route_is_bitwise_with_in_place(rows_per_req, kv_lens):
    _skip_unless_sm100()
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts_sparse_mla import (
        get_prims_ts_toolchain_unsupported_reason,
    )

    if get_prims_ts_toolchain_unsupported_reason() is not None:
        pytest.skip("PrimTS toolchain unavailable")
    device = torch.device("cuda")
    pool, table, kv_indptr, _ = _paged_pool(kv_lens, seed=sum(kv_lens) % 997)
    req_of_row = torch.repeat_interleave(
        torch.arange(len(kv_lens), dtype=torch.int32, device=device),
        torch.tensor(rows_per_req, device=device),
    )
    n = req_of_row.numel()
    topk = _local_topk(rows_per_req, kv_lens, seed=n)
    g = torch.Generator(device="cuda").manual_seed(7)
    query = (torch.randn(n, HEADS, 576, generator=g, device=device) * 0.5).to(torch.float8_e4m3fn)
    lengths = torch.full((n,), TOPK, dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(192)

    global_rows = torch.ops.trtllm.convert_req_index_to_global(
        req_of_row, table, topk, TPB, TOPK, NUM_LAYERS * TPB, LAYER
    ).view(n, TOPK)
    w4, qo4 = ptp._plan(device, n, HEADS, TOPK, "nvfp4.fp8rope")
    out4 = torch.empty(n, HEADS, 512, dtype=torch.bfloat16, device=device)
    w4.run(
        query,
        pool,
        global_rows,
        lengths,
        qo_indptr=qo4.clamp(max=n),
        softmax_scale=scale,
        output_scale=1.0,
        out=out4,
        validate=False,
    )

    scratch = torch.empty(sum(kv_lens), 576, dtype=torch.uint8, device=device)
    rec.paged_records_to_e4m3(
        pool, table, kv_indptr, max(kv_lens), TPB, NUM_LAYERS * TPB, LAYER, scratch
    )
    local = ptp._rebase_topk(topk, req_of_row, kv_indptr)
    w8, qo8 = ptp._plan(device, n, HEADS, TOPK, "native")
    out8 = torch.empty(n, HEADS, 512, dtype=torch.bfloat16, device=device)
    w8.run(
        query,
        scratch.view(torch.float8_e4m3fn),
        local,
        lengths,
        qo_indptr=qo8.clamp(max=n),
        softmax_scale=scale,
        output_scale=1.0,
        out=out8,
        validate=False,
    )
    assert torch.equal(out8.view(torch.int16), out4.view(torch.int16))
