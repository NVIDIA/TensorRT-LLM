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
"""Decode attention with locality domains on matches it with them off (Rubin only).

The same requests, cached K/V and query run through ``TrtllmAttention`` twice: once on an ordinary
KV cache with one kernel over the batch, once on a localized KV cache where each locality domain's
rows run on that domain's SMs. Per-domain kernels choose their configuration for the domain's SMs,
so the two runs may use different kernel variants and KV splits; the outputs must still agree.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from kv_cache_utils import fill_kv_cache_logical

import tensorrt_llm
from tensorrt_llm._torch.attention.backends import TrtllmAttention
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_locality_domain_enabled(),
    reason="needs a GPU with locality domains (Rubin)",
)

_KV_LORA_RANK = 512
_ROPE_DIM = 64
_MLA_HEAD_DIM = _KV_LORA_RANK + _ROPE_DIM


def _domains(batch: int, localized: bool):
    # Domain 0 first, so the positional split runs every row on the domain holding its KV.
    if not localized:
        return None
    half = (batch + 1) // 2
    return [0] * half + [1] * (batch - half)


def _manager(
    *, mla: bool, batch: int, cached: int, num_kv_heads: int, head_dim: int, localized: bool
):
    tokens_per_block = 32 if mla else 64
    max_seq_len = cached + 1 + tokens_per_block
    blocks_per_seq = math.ceil(max_seq_len / tokens_per_block)
    cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType
    manager = KVCacheManagerV2(
        # Each domain holds half the pool, so leave room for a full batch in either one.
        KvCacheConfig(
            max_tokens=blocks_per_seq * tokens_per_block * 2 * (batch + 2), enable_block_reuse=False
        ),
        cache_type.SELFKONLY if mla else cache_type.SELF,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        dtype=DataType.FP8 if mla else DataType.BF16,
        enable_locality_domains=localized,
    )
    requests = manager.add_dummy_requests(
        list(range(batch)), [cached + 1] * batch, locality_domain_ids=_domains(batch, localized)
    )
    assert requests is not None
    if localized:
        assert manager.num_locality_domains == 2
    return manager


def _metadata(manager, *, mla: bool, batch: int, cached: int, localized: bool):
    kwargs = dict(
        num_contexts=0,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[cached] * batch),
        seq_lens=torch.ones(batch, dtype=torch.int),
        max_num_requests=batch,
        max_num_tokens=8192,
        kv_cache_manager=manager,
        request_ids=list(range(batch)),
        prompt_lens=[cached + 1] * batch,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        locality_domain_enabled=localized,
    )
    if mla:
        kwargs["enable_flash_mla"] = False
    metadata = TrtllmAttention.Metadata(**kwargs)
    metadata.prepare()
    return metadata


def _fill_mla_latent(manager, latent_per_seq):
    """Write each request's cached latent rows through its own page list (layout-independent)."""
    buf = manager.get_buffers(0, kv_layout="NHD")  # [pages, 1, tokens_per_block, 1, 576]
    tokens_per_block = buf.shape[2]
    for blocks, latent in zip(
        manager.get_batch_cache_indices(list(range(len(latent_per_seq))), 0), latent_per_seq
    ):
        blocks = [b for b in blocks if b != -1]
        # Zero only owned pages: a localized pool leaves unmapped VA between the domains' pages.
        buf[blocks] = 0
        written = 0
        for blk in blocks:
            if written >= latent.shape[0]:
                break
            n = min(tokens_per_block, latent.shape[0] - written)
            buf[blk, 0, :n, 0].copy_(latent[written : written + n].to(buf.dtype))
            written += n


def _gqa_output(*, batch: int, cached: int, localized: bool) -> torch.Tensor:
    num_heads, num_kv_heads, head_dim = 48, 4, 128
    gen = torch.Generator(device="cuda").manual_seed(0)
    k = [
        torch.randn(cached, num_kv_heads, head_dim, device="cuda", generator=gen) * 0.5
        for _ in range(batch)
    ]
    v = [
        torch.randn(cached, num_kv_heads, head_dim, device="cuda", generator=gen)
        for _ in range(batch)
    ]
    qkv = torch.randn(
        batch, (num_heads + 2 * num_kv_heads) * head_dim, device="cuda", generator=gen
    ).to(torch.bfloat16)
    manager = _manager(
        mla=False,
        batch=batch,
        cached=cached,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        localized=localized,
    )
    try:
        fill_kv_cache_logical(manager, 0, list(range(batch)), k, v, kv_layout="HND")
        metadata = _metadata(manager, mla=False, batch=batch, cached=cached, localized=localized)
        backend = TrtllmAttention(
            layer_idx=0, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
        )
        mask = PredefinedAttentionMask.CAUSAL
        output = backend.create_output(
            qkv, is_quantize_output=False, metadata=metadata, attention_mask=mask, is_gen_only=True
        )[0]
        backend.forward(
            qkv,
            None,
            None,
            metadata,
            attention_mask=mask,
            attention_input_type=AttentionInputType.generation_only,
            output=output,
        )
        torch.cuda.synchronize()
        return output.float().cpu()
    finally:
        manager.shutdown()


def _mla_output(*, batch: int, cached: int, localized: bool) -> torch.Tensor:
    num_heads = 64
    gen = torch.Generator(device="cuda").manual_seed(0)
    latent = [
        torch.randn(cached, _MLA_HEAD_DIM, device="cuda", generator=gen) * 0.5 for _ in range(batch)
    ]
    fused_q = torch.randn(batch, num_heads * _MLA_HEAD_DIM, device="cuda", generator=gen).to(
        torch.bfloat16
    )
    q_pe = torch.randn(batch, num_heads, _ROPE_DIM, device="cuda", generator=gen).to(torch.bfloat16)
    new_latent = torch.randn(batch, _MLA_HEAD_DIM, device="cuda", generator=gen).to(torch.bfloat16)
    manager = _manager(
        mla=True,
        batch=batch,
        cached=cached,
        num_kv_heads=1,
        head_dim=_MLA_HEAD_DIM,
        localized=localized,
    )
    try:
        _fill_mla_latent(manager, latent)
        metadata = _metadata(manager, mla=True, batch=batch, cached=cached, localized=localized)
        # DeepSeek-V3 yarn RoPE, as in test_attention_mla.py.
        rope = RopeParams.from_config(
            SimpleNamespace(
                hidden_size=7168,
                num_attention_heads=128,
                rope_scaling={
                    "beta_fast": 32,
                    "beta_slow": 1,
                    "factor": 40.0,
                    "mscale": 1.0,
                    "mscale_all_dim": 1.0,
                    "original_max_position_embeddings": 4096,
                    "type": "yarn",
                },
                max_position_embeddings=163840,
                rope_theta=10000.0,
                qk_rope_head_dim=_ROPE_DIM,
                model_type="deepseek_v3",
            )
        )
        backend = TrtllmAttention(
            layer_idx=0,
            num_heads=num_heads,
            head_dim=_MLA_HEAD_DIM,
            num_kv_heads=1,
            quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value),
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.yarn, rope=rope, is_neox=False
            ),
            mla_params=MLAParams(
                q_lora_rank=1536,
                kv_lora_rank=_KV_LORA_RANK,
                qk_rope_head_dim=_ROPE_DIM,
                qk_nope_head_dim=128,
                v_head_dim=128,
                predicted_tokens_per_seq=1,
            ),
        )
        cu_q = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
        cu_kv = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
        counter = torch.empty(1, dtype=torch.uint32, device="cuda")
        bmm1_scale = torch.empty(2, dtype=torch.float32, device="cuda")
        bmm2_scale = torch.empty(1, dtype=torch.float32, device="cuda")
        quant_q = torch.empty(batch, num_heads * _MLA_HEAD_DIM, dtype=torch.uint8, device="cuda")
        backend.mla_rope_generation(
            fused_q,
            q_pe,
            new_latent,
            metadata,
            cu_q,
            cu_kv,
            counter,
            bmm1_scale,
            bmm2_scale,
            quant_q,
        )
        output = backend.create_output(
            fused_q,
            is_quantize_output=False,
            metadata=metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            is_gen_only=True,
        )[0]
        backend.forward(
            fused_q,
            None,
            None,
            metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=new_latent,
            q_pe=q_pe,
            cu_q_seqlens=cu_q,
            cu_kv_seqlens=cu_kv,
            fmha_scheduler_counter=counter,
            mla_bmm1_scale=bmm1_scale,
            mla_bmm2_scale=bmm2_scale,
            quant_q_buffer=quant_q,
            output=output,
        )
        torch.cuda.synchronize()
        return output.float().cpu()
    finally:
        manager.shutdown()


@pytest.mark.parametrize("batch,cached", [(8, 1000), (64, 4096)])
def test_gqa_decode_matches_with_locality_domains(batch, cached):
    off = _gqa_output(batch=batch, cached=cached, localized=False)
    on = _gqa_output(batch=batch, cached=cached, localized=True)
    assert torch.isfinite(on).all()
    torch.testing.assert_close(on, off, atol=2e-2, rtol=2e-2)


# At 64 x 16k each domain's kernel sizes its KV split for half the SMs, unlike the single full-device kernel.
@pytest.mark.parametrize("batch,cached", [(8, 1000), (64, 16384)])
def test_mla_decode_matches_with_locality_domains(batch, cached):
    off = _mla_output(batch=batch, cached=cached, localized=False)
    on = _mla_output(batch=batch, cached=cached, localized=True)
    assert torch.isfinite(on).all()
    torch.testing.assert_close(on, off, atol=2e-2, rtol=2e-2)
