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
"""Correctness tests for the halfspec (TMA-load + sync-MMA warp-specialized)
sm_120 / sm_121 context FMHA.

The halfspec kernel is the sm_120 / sm_121 context attention path that backs
skip-softmax, so it is exercised here by attaching a ``SkipSoftmaxAttentionConfig``
to ``TrtllmAttention`` (which flips ``use_halfspec_fmha`` on). A tiny
``threshold_scale_factor`` (1e-30) keeps skip-softmax effectively disabled --
``log(thr / seqlen)`` is hugely negative so no tile is ever skipped -- so the
kernel computes full softmax and must match a standard attention reference.

For the supported config (BF16 in/out, head_dim == head_dim_v in {128, 256},
causal, packed QKV) the C++ dispatch in ``FusedMultiHeadAttentionXMMAKernelV2::run``
routes unconditionally to the halfspec kernel, so a correct result here is a
direct check of the halfspec kernel itself.

Regression coverage: the multi-request (batch > 1) cases guard against a
per-request TMA offset bug where every batch element re-read request 0's tokens.
"""

import math
from typing import List

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import PredefinedAttentionMask
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import SkipSoftmaxAttentionConfig

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (120, 121),
    reason=f"halfspec FMHA only dispatches on SM 120 / 121, got SM {get_sm_version()}",
)


def _causal_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: List[int],
) -> torch.Tensor:
    """Per-request causal attention in fp32. q/k/v are packed ``[total, H*D]``."""
    outs = []
    off = 0
    for sl in seq_lens:
        qi = q[off : off + sl].view(sl, num_heads, head_dim).transpose(0, 1)
        ki = k[off : off + sl].view(sl, num_kv_heads, head_dim).transpose(0, 1)
        vi = v[off : off + sl].view(sl, num_kv_heads, head_dim).transpose(0, 1)
        if num_heads > num_kv_heads:
            rep = num_heads // num_kv_heads
            ki = ki.repeat_interleave(rep, dim=0)
            vi = vi.repeat_interleave(rep, dim=0)
        scores = torch.matmul(qi.float(), ki.float().transpose(1, 2)) / math.sqrt(head_dim)
        causal = torch.triu(torch.ones(sl, sl, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        oi = torch.matmul(probs, vi.float())  # [H, sl, D]
        outs.append(oi.transpose(0, 1).contiguous().view(sl, num_heads * head_dim).to(q.dtype))
        off += sl
    return torch.cat(outs)


def _run_context(num_heads, num_kv_heads, head_dim, seq_lens, q, k, v, sparse_attention_config):
    """Build a TRTLLM attention layer + no-cache context metadata and run a
    packed-QKV causal prefill. Mirrors ``test_attention_no_cache``."""
    AttentionCls = get_attention_backend("TRTLLM")
    layer = AttentionCls(
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        sparse_attention_config=sparse_attention_config,
    )

    metadata = AttentionCls.Metadata(
        max_num_requests=len(seq_lens),
        max_num_tokens=8192,
        kv_cache_manager=None,
        mapping=None,
        runtime_features=None,
    )
    metadata.seq_lens = torch.tensor(seq_lens, dtype=torch.int)
    metadata.num_contexts = len(seq_lens)
    metadata.request_ids = torch.tensor(range(len(seq_lens)), dtype=torch.int)
    metadata.max_seq_len = max(seq_lens)
    metadata.prepare()

    qkv = torch.cat((q, k, v), dim=-1)
    return (
        layer,
        metadata,
        layer.forward(qkv, None, None, metadata, attention_mask=PredefinedAttentionMask.CAUSAL),
    )


# (num_heads, num_kv_heads) -- MHA configs exercised by the halfspec kernel.
HEADS = [(8, 8)]
# Single request and several multi-request batches (the regression target:
# equal- and variable-length batches, and a larger fan-out).
SEQ_LENS = [
    [64],
    [128],
    [137],
    [64, 64],
    [64, 128],
    [16, 64, 137, 256],
    [64, 64, 64, 64],
]


@pytest.mark.parametrize("head_dim", [128], ids=lambda d: f"head_dim_{d}")
@pytest.mark.parametrize("num_heads,num_kv_heads", HEADS, ids=lambda x: f"h_{x}")
@pytest.mark.parametrize("seq_lens", SEQ_LENS, ids=lambda s: f"seqs_{'_'.join(map(str, s))}")
def test_halfspec_context_matches_reference(num_heads, num_kv_heads, head_dim, seq_lens):
    """The halfspec context FMHA must match (a) a standard fp32 causal-attention
    reference and (b) the default (non-halfspec) TRTLLM context kernel on the
    same inputs, for both single- and multi-request batches."""
    torch.manual_seed(720)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    total = sum(seq_lens)

    gen = torch.Generator(device=device).manual_seed(720)
    q = torch.randn(total, num_heads * head_dim, dtype=dtype, device=device, generator=gen)
    k = torch.randn(total, num_kv_heads * head_dim, dtype=dtype, device=device, generator=gen)
    v = torch.randn(total, num_kv_heads * head_dim, dtype=dtype, device=device, generator=gen)

    ref = _causal_reference(q, k, v, num_heads, num_kv_heads, head_dim, seq_lens)

    # Trusted baseline: default TRTLLM context FMHA (no skip-softmax cfg).
    _, _, out_default = _run_context(
        num_heads, num_kv_heads, head_dim, seq_lens, q, k, v, sparse_attention_config=None
    )
    # Under test: halfspec, forced on via a skip-softmax cfg with a tiny
    # threshold so no tile is actually skipped (full softmax).
    halfspec_layer, _, out_halfspec = _run_context(
        num_heads,
        num_kv_heads,
        head_dim,
        seq_lens,
        q,
        k,
        v,
        sparse_attention_config=SkipSoftmaxAttentionConfig(threshold_scale_factor=1e-30),
    )
    torch.cuda.synchronize()

    assert halfspec_layer.use_halfspec_fmha, "skip-softmax cfg did not enable halfspec"

    # Sanity: the trusted baseline matches the fp32 reference.
    torch.testing.assert_close(out_default.float(), ref.float(), atol=2e-2, rtol=2e-2)
    # The actual regression assertions: halfspec matches the reference and the
    # trusted kernel. Pre-fix, multi-request batches returned request 0's result
    # for every request (mean abs error ~0.2), far outside these tolerances.
    torch.testing.assert_close(out_halfspec.float(), ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out_halfspec.float(), out_default.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
