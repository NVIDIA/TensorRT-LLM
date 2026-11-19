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
"""SM100/SM103 split-K FP8 MLA context FMHA (``TRTLLM_K3_FP8KV_SPLIT_K_CONTEXT_SM100``).

Runs chunked prefill the way ``MLA.forward_context_with_chunked_prefill_splitk`` does: the cached prefix
(non-causal, overwrite) and then the new tokens (causal, merge). The merged output must match attention over
the whole sequence.
"""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the split-K MLA context FMHA requires SM100/SM103",
)

_FP8 = torch.float8_e4m3fn


def _fp8(*shape):
    # Unit-scale inputs give peaked softmax rows and outputs of order 0.1-1, so a wrong mask or merge
    # cannot hide inside the tolerance.
    return torch.randn(*shape, device="cuda").to(_FP8)


def _kv(tokens, heads):
    """k_nope and v as strided views of one kv_b_proj-shaped buffer, as MLA passes them."""
    kv = _fp8(tokens, 2 * heads * 128)
    return (
        kv[:, : heads * 128].view(tokens, heads, 128),
        _fp8(tokens, 64),
        kv[:, heads * 128 :].view(tokens, heads, 128),
    )


def _reference(q, k_nope, k_pe, v, prefix_len, sm_scale):
    """Attention of new tokens over prefix + new tokens; new token i sees every prefix token and new tokens <= i."""
    heads = q.shape[1]
    k = torch.cat((k_nope.float(), k_pe.float()[:, None, :].expand(-1, heads, -1)), dim=-1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), k) * sm_scale
    new_len, total_len = q.shape[0], k.shape[0]
    rows = torch.arange(new_len, device="cuda")[:, None]
    cols = torch.arange(total_len, device="cuda")[None, :]
    scores.masked_fill_(cols > prefix_len + rows, float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v.float())


@pytest.mark.parametrize("heads", [8, 96])
def test_split_k_chunked_prefill_matches_reference(heads):
    from tensorrt_llm._torch.attention_backend.fmha import mla_context_splitk as splitk

    assert splitk._IMPORT_ERROR is None, splitk._IMPORT_ERROR
    torch.manual_seed(heads)
    # Two sequences: lengths that are not multiples of the 128-row tile, a long and a short prefix.
    new_lens, prefix_lens = [100, 257], [300, 50]
    sm_scale = 1.0 / math.sqrt(192)

    q = _fp8(sum(new_lens), heads, 192)
    prefix_nope, prefix_pe, prefix_v = _kv(sum(prefix_lens), heads)
    new_nope, new_pe, new_v = _kv(sum(new_lens), heads)
    cu_q = torch.tensor([0, new_lens[0], sum(new_lens)], dtype=torch.int32, device="cuda")
    cu_prefix = torch.tensor(
        [0, prefix_lens[0], sum(prefix_lens)], dtype=torch.int32, device="cuda"
    )
    output = torch.full((sum(new_lens), heads, 128), torch.nan, dtype=torch.bfloat16, device="cuda")
    stats = torch.full((sum(new_lens), heads, 2), torch.nan, dtype=torch.float32, device="cuda")

    def merge_op(op):
        return torch.full((len(new_lens),), op, dtype=torch.int64, device="cuda")

    splitk.run_partition(
        q,
        prefix_nope,
        prefix_pe,
        prefix_v,
        output,
        stats,
        cu_q,
        cu_prefix,
        merge_op(2),
        max_q=max(new_lens),
        max_kv=max(prefix_lens),
        causal=False,
        sm_scale=sm_scale,
    )
    splitk.run_partition(
        q,
        new_nope,
        new_pe,
        new_v,
        output,
        stats,
        cu_q,
        cu_q,
        merge_op(1),
        max_q=max(new_lens),
        max_kv=max(new_lens),
        causal=True,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()

    for b in range(len(new_lens)):
        rows = slice(cu_q[b].item(), cu_q[b + 1].item())
        kv_rows = slice(cu_prefix[b].item(), cu_prefix[b + 1].item())
        expected = _reference(
            q[rows],
            torch.cat((prefix_nope[kv_rows], new_nope[rows])),
            torch.cat((prefix_pe[kv_rows], new_pe[rows])),
            torch.cat((prefix_v[kv_rows], new_v[rows])),
            prefix_lens[b],
            sm_scale,
        )
        actual = output[rows].float()
        assert torch.isfinite(actual).all()
        rel = ((actual - expected).norm() / expected.norm()).item()
        print(f"heads={heads} seq={b}: relative error {rel:.4f}")
        # FP8 attention: a few percent of relative error; a wrong mask or merge is far above this.
        assert rel < 0.05, rel
