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
"""``paged_context_fmha_fmk`` against the trtllm-gen context FMHA it replaces on mixed steps.

Both run on the same E4M3 Q and paged E4M3 K / V pool (32- and 64-token cache blocks), the same
block table and the same scales; the library is flashinfer's ``trtllm_batch_context_with_kv_cache``,
the call the library's ``run_context`` makes. The batch has context lengths that are not a
multiple of the cache block (the last block holds stale keys past the length, which both must
mask), queries that start mid-context, and one that is the whole context.

The kernels share the precision (E4M3 inputs and P, fp32 accumulation, BF16 O) but not the
summation order or the P rounding point, so they are compared with a tolerance, not bitwise: the
relative RMS error stays within 5 % (E4M3 rounds P to 3 mantissa bits, a 6 % relative step that
averages out over the keys), and no element is off by more than 10 % of the largest library
output (a bound on the output scale, not per element: rows with 1-3 keys and cancelling V terms
can have a tiny reference but an error of 6 % of |V|). A wrong key address or a missed mask
moves the rows by their own size (about 100 %).
"""

import flashinfer
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    context_fmha_fmk as fmk,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the fmk context FMHA is built for sm_103",
)

HEADS, KV_HEADS, HEAD_DIM = 16, 4, 64
# (new rows, context length): 100, 333 and 1000 are not multiples of 32 or 64.
CASES = [(37, 100), (130, 333), (5, 5), (200, 1000)]


@pytest.mark.parametrize("cache_page", [32, 64])
def test_fmk_matches_library_context_fmha(cache_page):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    q_lens = [q for q, _ in CASES]
    kv_lens = [k for _, k in CASES]
    batch, total_q = len(CASES), sum(q_lens)

    def e4m3(*shape):
        return torch.randn(*shape, device=dev).to(torch.float8_e4m3fn)

    q = e4m3(total_q, HEADS, HEAD_DIM)
    blocks = [-(-k // cache_page) for k in kv_lens]
    # Shuffled pages: K of page p is pool block 2p, V is 2p + 1. Every block, the stale tail of a
    # last block included, holds random keys.
    pages = torch.randperm(sum(blocks), device=dev).to(torch.int32)
    kv_pool = e4m3(2 * sum(blocks), KV_HEADS, cache_page, HEAD_DIM)
    table = torch.zeros(batch, 2, max(blocks), dtype=torch.int32, device=dev)
    start = 0
    for b, n in enumerate(blocks):
        table[b, 0, :n] = 2 * pages[start : start + n]
        table[b, 1, :n] = 2 * pages[start : start + n] + 1
        start += n

    i32 = dict(dtype=torch.int32, device=dev)
    cu_q = torch.tensor([0, *q_lens], **i32).cumsum(0, dtype=torch.int32)
    cu_kv = torch.tensor([0, *kv_lens], **i32).cumsum(0, dtype=torch.int32)
    kv_len_dev = torch.tensor(kv_lens, **i32)
    scale = HEAD_DIM**-0.5

    ref = torch.empty(total_q, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=q,
        kv_cache=(kv_pool, kv_pool),
        workspace_buffer=torch.zeros(256 * 2**20, dtype=torch.uint8, device=dev),
        block_tables=table,
        seq_lens=kv_len_dev,
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
        bmm1_scale=scale,
        bmm2_scale=1.0,
        batch_size=batch,
        cum_seq_lens_q=cu_q,
        cum_seq_lens_kv=cu_kv,
        window_left=-1,
        out=ref,
        kv_layout="HND",
        uses_shared_paged_kv_idx=False,
        causal=True,
        multi_ctas_kv_counter_buffer=torch.zeros(2**20, dtype=torch.uint8, device=dev),
    )

    plan = fmk.plan_context_fmha(
        q_lens,
        kv_lens,
        num_ctas=torch.cuda.get_device_properties(dev).multi_processor_count,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        num_inst=4,
        tile_n=64,
        device=dev,
    )
    out = torch.full_like(ref, float("nan"))
    scales = torch.tensor([scale * 1.4426950408889634, 1.0], dtype=torch.float32, device=dev)
    fmk.paged_context_fmha_fmk(
        q, kv_pool, table, cu_q, kv_len_dev, plan, scales[0:1], scales[1:2], out
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out).all(), "fmk left rows unwritten"
    assert ref.float().abs().max() > 0.1, "the reference is trivially small"
    diff = (out.float() - ref.float()).abs()
    rel_rms = (diff.pow(2).mean() / ref.float().pow(2).mean()).sqrt()
    assert rel_rms < 0.05, f"relative RMS error {rel_rms:.4f}"
    assert diff.max() < 0.1 * ref.float().abs().max(), f"max abs error {diff.max():.4f}"
