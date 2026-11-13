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
"""TRTLLM generation attention over a long INT8 KV cache with XQA multi-block mode.

The generation step runs on a paged INT8 cache and, as a control, on a BF16 cache that holds the
same dequantized values (the KV scale is a power of two, so ``int8 * scale`` is exact in BF16).
Each case checks that the XQA kernel splits every sequence into more than one sub-sequence, that
the INT8 output matches the BF16-cache output and a float64 reference, that the appended tokens
are stored exactly, and that repeated steps are bitwise identical.
"""

import json
import math

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (80, 86, 89, 90),
    reason="INT8 KV decode uses the generic XQA mha.cu kernel on SM80-SM90",
)

KV_SCALE = 2.0**-5
PAGE = 64
# The XQA kernel of the generation step (JIT-compiled from cpp/kernels/xqa/mha.cu).
XQA_KERNEL = "kernel_mha"


def _inputs(num_heads, num_kv_heads, head_dim, cached, q_len, seed):
    """Query, cached K/V and new K/V; the K/V values are integers in [-127, 127]."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    num_tokens = len(cached) * q_len
    q = torch.randn(num_tokens, num_heads, head_dim, generator=gen, device="cuda")
    group = num_heads // num_kv_heads

    def kv_ints(*shape):
        return (torch.randn(*shape, generator=gen, device="cuda") * 32).round().clamp(-127, 127)

    cached_k = [kv_ints(c, num_kv_heads, head_dim) for c in cached]
    cached_v = [kv_ints(c, num_kv_heads, head_dim) for c in cached]
    # Keys aligned with the first query of the sequence at a few positions spread over the
    # history, so the running softmax maximum differs between the sub-sequences that the
    # multi-block kernel reduces.
    for i, c in enumerate(cached):
        positions = torch.randint(0, c, (4,), generator=gen, device="cuda").tolist()
        for j, p in enumerate(positions):
            for h in range(num_kv_heads):
                cached_k[i][p, h] = (q[i * q_len, h * group] * 8 * (j + 1)).round().clamp(-127, 127)
    new_k = kv_ints(num_tokens, num_kv_heads, head_dim)
    new_v = kv_ints(num_tokens, num_kv_heads, head_dim)
    return q.to(torch.bfloat16), cached_k, cached_v, new_k, new_v


def _pool(manager, int8):
    """The layer's pool as ``[pages, 2, kv_heads, page, head_dim]``.

    The KV cache managers do not take ``DataType.INT8``; an INT8 cache uses a pool of another
    1-byte type, which has the same block layout, and is viewed as int8 here.
    """
    buf = manager.get_buffers(0, kv_layout="HND")
    return buf.view(torch.int8) if int8 else buf


def _fill(manager, cached_k, cached_v, int8):
    """Write the cached prefix of every sequence; the rest of the pool is zero."""
    buf = _pool(manager, int8)
    buf.zero_()
    blocks = manager.get_batch_cache_indices(list(range(len(cached_k))), 0)
    for i, (k, v) in enumerate(zip(cached_k, cached_v, strict=True)):
        if int8:
            k, v = k.to(torch.int8), v.to(torch.int8)
        else:
            k, v = (k * KV_SCALE).to(buf.dtype), (v * KV_SCALE).to(buf.dtype)
        for n, block in enumerate(blocks[i][: math.ceil(k.shape[0] / PAGE)]):
            tokens = slice(n * PAGE, (n + 1) * PAGE)
            num = k[tokens].shape[0]
            buf[block, 0, :, :num].copy_(k[tokens].transpose(0, 1))
            buf[block, 1, :, :num].copy_(v[tokens].transpose(0, 1))


def _cached_kv(manager, total, int8):
    """Dequantized cache contents per sequence, ``[2, total_len, num_kv_heads, head_dim]``."""
    buf = _pool(manager, int8)
    kv_scale = KV_SCALE if int8 else 1.0
    blocks = manager.get_batch_cache_indices(list(range(len(total))), 0)
    result = []
    for i, length in enumerate(total):
        ids = torch.tensor(blocks[i][: math.ceil(length / PAGE)], device="cuda")
        pages = buf[ids].permute(1, 0, 3, 2, 4)  # [2, n, page, kv_heads, head_dim]
        kv = pages.reshape(2, -1, *pages.shape[3:])[:, :length]
        result.append(kv.double() * kv_scale)
    return result


def _reference(q, cached_kv, cached, q_len):
    """Float64 causal attention of each new query over its sequence's cache."""
    num_heads, head_dim = q.shape[1], q.shape[2]
    outputs = []
    for i, c in enumerate(cached):
        k, v = cached_kv[i]
        group = num_heads // k.shape[1]
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
        qi = q[i * q_len : (i + 1) * q_len].double()
        scores = torch.einsum("qhd,khd->hqk", qi, k) / math.sqrt(head_dim)
        key_pos = torch.arange(k.shape[0], device="cuda")
        last_key = c + torch.arange(q_len, device="cuda")
        scores = scores.masked_fill(key_pos[None, None, :] > last_key[None, :, None], -math.inf)
        out = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v)
        outputs.append(out.reshape(q_len, num_heads * head_dim))
    return torch.cat(outputs)


def _generation_step(kv_dtype, num_heads, num_kv_heads, head_dim, cached, q_len, inputs, tmp_path):
    """One generation step; returns (output, cache after the step, XQA grids, repeat outputs)."""
    q, cached_k, cached_v, new_k, new_v = inputs
    int8 = kv_dtype == DataType.INT8
    pool_dtype = DataType.FP8 if int8 else kv_dtype
    num_seqs = len(cached)
    request_ids = list(range(num_seqs))
    total = [c + q_len for c in cached]
    pages = math.ceil(max(total) / PAGE)

    attn = create_attention(
        "TRTLLM",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.INT8) if int8 else None,
    )
    if int8:
        attn.kv_cache_scaling_factor.fill_(KV_SCALE)
        attn.kv_scale_quant_orig = attn.kv_cache_scaling_factor
        attn.kv_scale_orig_quant = 1.0 / attn.kv_cache_scaling_factor

    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=num_seqs * pages * PAGE),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=PAGE,
        max_seq_len=pages * PAGE,
        max_batch_size=num_seqs,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=pool_dtype,
    )
    try:
        manager.add_dummy_requests(request_ids, total)
        _fill(manager, cached_k, cached_v, int8)

        metadata = TrtllmAttentionMetadata(
            num_contexts=0,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
            seq_lens=torch.tensor([q_len] * num_seqs, dtype=torch.int),
            max_num_requests=num_seqs,
            max_num_tokens=num_seqs * q_len,
            kv_cache_manager=manager,
            request_ids=request_ids,
            prompt_lens=total,
        )
        if q_len > 1:
            # Linear-tree speculative decoding: the draft tokens attend causally to each other.
            metadata.update_spec_dec_param(
                batch_size=num_seqs,
                is_spec_decoding_enabled=True,
                is_spec_dec_tree=False,
                is_spec_dec_dynamic_tree=False,
                max_draft_len=q_len - 1,
                max_total_draft_tokens=q_len - 1,
            )
        metadata.prepare()

        num_tokens = num_seqs * q_len
        qkv = torch.cat(
            [
                q.reshape(num_tokens, -1),
                (new_k * KV_SCALE).to(q.dtype).reshape(num_tokens, -1),
                (new_v * KV_SCALE).to(q.dtype).reshape(num_tokens, -1),
            ],
            dim=-1,
        )
        forward_args = AttentionForwardArgs(attention_mask=PredefinedAttentionMask.CAUSAL)

        def step():
            out = attn.forward(qkv, None, None, metadata, forward_args=forward_args)
            return out[0] if isinstance(out, tuple) else out

        output = step().clone()
        torch.cuda.synchronize()
        cache = _cached_kv(manager, total, int8)

        # The step rewrites the same new tokens into the same slots, so it can be repeated.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            repeats = [step().clone() for _ in range(4)]
            torch.cuda.synchronize()
        trace = tmp_path / f"{kv_dtype.name}.json"
        prof.export_chrome_trace(str(trace))
        grids = [
            event["args"]["grid"]
            for event in json.loads(trace.read_text())["traceEvents"]
            if event.get("cat") == "kernel" and XQA_KERNEL in event["name"]
        ]
        return output, cache, grids, repeats
    finally:
        manager.shutdown()


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,cached,q_len",
    [
        (32, 8, 128, [8192], 1),
        (24, 4, 256, [8192], 1),
        (32, 8, 128, [1000, 4097, 16384], 1),
        (24, 4, 256, [32768], 1),
        (24, 4, 256, [8192], 4),
        (32, 8, 128, [3000, 20000], 4),
    ],
    ids=[
        "h32kv8d128-8k",
        "h24kv4d256-8k",
        "h32kv8d128-mixed",
        "h24kv4d256-32k",
        "h24kv4d256-8k-specdec4",
        "h32kv8d128-mixed-specdec4",
    ],
)
def test_int8_kv_generation_uses_multi_block(
    tmp_path, num_heads, num_kv_heads, head_dim, cached, q_len
):
    inputs = _inputs(num_heads, num_kv_heads, head_dim, cached, q_len, seed=0)
    args = (num_heads, num_kv_heads, head_dim, cached, q_len, inputs, tmp_path)
    out_int8, cache_int8, grids_int8, repeats_int8 = _generation_step(DataType.INT8, *args)
    out_bf16, cache_bf16, grids_bf16, _ = _generation_step(DataType.BF16, *args)

    # Both caches hold exactly the input values, including the INT8-quantized new tokens.
    for i, (kv_int8, kv_bf16) in enumerate(zip(cache_int8, cache_bf16, strict=True)):
        tokens = slice(i * q_len, (i + 1) * q_len)
        expected = (
            torch.stack(
                [
                    torch.cat([inputs[1][i], inputs[3][tokens]]),
                    torch.cat([inputs[2][i], inputs[4][tokens]]),
                ]
            ).double()
            * KV_SCALE
        )
        torch.testing.assert_close(kv_int8, expected, atol=0, rtol=0)
        torch.testing.assert_close(kv_bf16, expected, atol=0, rtol=0)

    # The XQA kernel serves both steps, and splits each sequence into sub-sequences.
    assert grids_int8 and grids_bf16
    assert all(grid[0] > 1 for grid in grids_int8), grids_int8
    assert all(grid[0] > 1 for grid in grids_bf16), grids_bf16

    assert all(torch.equal(r, out_int8) for r in repeats_int8)
    assert not out_int8.isnan().any()

    reference = _reference(inputs[0], cache_int8, cached, q_len)
    rel_err = ((out_int8.double() - reference).norm() / reference.norm()).item()
    rel_err_bf16 = ((out_bf16.double() - reference).norm() / reference.norm()).item()
    assert rel_err < 1e-2, rel_err
    assert rel_err_bf16 < 1e-2, rel_err_bf16
    torch.testing.assert_close(out_int8.double(), reference, atol=2e-2, rtol=2e-2)
    # Both kernels read the same values; they differ only in rounding of the partial results.
    torch.testing.assert_close(out_int8, out_bf16, atol=1e-3, rtol=1e-2)
