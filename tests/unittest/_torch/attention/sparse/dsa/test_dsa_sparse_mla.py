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

"""
Tests for sparse MLA attention using explicit sparse indices.
"""

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._utils import str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype = torch.bfloat16
    num_layers: int = 1
    num_heads: int = 128
    num_kv_heads: int = 1
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 512
    rope_append: bool = True
    hidden_size: int = 7168
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1
    rope_factor: float = 40.0
    rope_mscale: float = 1.0
    rope_mscale_all_dim: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_type: str = "yarn"
    model_type: str = "deepseek_v3"
    kv_cache_tokens_per_block: int = 64


@dataclass(kw_only=True, frozen=True)
class RopeConfig:
    hidden_size: int = 7168
    num_attention_heads: int = 128
    rope_scaling: dict = field(
        default_factory=lambda: {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        }
    )
    max_position_embeddings: int = 163840
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64
    model_type: str = "deepseek_v3"


def _build_sparse_topk_indices_context(
    seq_lens: List[int],
    topk: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    total_tokens = sum(seq_lens)
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    token_offset = 0
    for seq_len in seq_lens:
        for token_idx in range(seq_len):
            max_index = token_idx
            valid_len = min(max_index + 1, topk)
            indices = torch.randperm(max_index + 1, device=device, generator=generator)[:valid_len]
            indices, _ = torch.sort(indices)
            topk_indices[token_offset + token_idx, :valid_len] = indices.to(torch.int32)
        token_offset += seq_len
    return topk_indices


def _build_sparse_topk_indices_generation(
    cached_lens: List[int],
    seq_len_q: int,
    topk: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    total_tokens = len(cached_lens) * seq_len_q
    topk_indices = torch.full((total_tokens, topk), -1, dtype=torch.int32, device=device)
    row = 0
    for cached_len in cached_lens:
        for q_idx in range(seq_len_q):
            max_index = cached_len + q_idx
            valid_len = min(max_index + 1, topk)
            indices = torch.randperm(max_index + 1, device=device, generator=generator)[:valid_len]
            indices, _ = torch.sort(indices)
            topk_indices[row, :valid_len] = indices.to(torch.int32)
            row += 1
    return topk_indices


def _allocate_kv_cache_for_generation(kv_cache_manager, request_ids, num_tokens: int):
    for request_id in request_ids:
        for _ in range(num_tokens):
            kv_cache_manager.impl.add_token(request_id)
            if hasattr(kv_cache_manager, "indexer_k_cache_manager"):
                kv_cache_manager.indexer_k_cache_manager.add_tokens(request_id, 1)


# Define test data
context_sequence_lengths = [[10], [3000, 3100], [508, 4399, 9981]]
# Use MTP by default if seqlen_q > 1.
generation_seq_len_q = [1, 4]
num_generation_steps = [2]

tokens_per_block = 64

kv_cache_dtype_list = [torch.bfloat16, torch.float8_e4m3fn]
# DSA only supports rope_append=True
rope_append_values = [True]
scenarios = [
    Scenario(
        kv_cache_dtype=kv_cache_dtype,
        num_layers=num_layers,
        kv_cache_tokens_per_block=tokens_per_block,
        rope_append=rope_append,
    )
    for kv_cache_dtype in kv_cache_dtype_list
    for num_layers in [1]
    for rope_append in rope_append_values
]

accuracy_dict = {
    torch.bfloat16: (0.1, 0.01),
    torch.float8_e4m3fn: (0.12, 0.01),
}

SPARSE_TOPK = 2048


def _assert_matches_vanilla(
    actual: dict[str, torch.Tensor],
    golden: dict[str, torch.Tensor],
    kv_cache_dtype: torch.dtype,
    backend_name: str,
) -> None:
    assert actual.keys() == golden.keys(), (
        f"[{backend_name} vs VANILLA golden] phase mismatch: {list(actual)} != {list(golden)}"
    )
    atol, rtol = accuracy_dict[kv_cache_dtype]
    for phase, golden_output in golden.items():
        torch.testing.assert_close(
            actual[phase],
            golden_output,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"[{backend_name} vs VANILLA golden, {phase}]\n{message}",
        )


# Convert parameterized tests to pytest parametrize
@skip_pre_blackwell
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.parametrize("scenario", scenarios, ids=lambda x: f"scenario: {x}")
@pytest.mark.parametrize(
    "context_sequence_lengths",
    context_sequence_lengths,
    ids=lambda x: f"context_sequence_lengths: {x}",
)
@pytest.mark.parametrize(
    "generation_seq_len_q", generation_seq_len_q, ids=lambda x: f"generation_seq_len_q: {x}"
)
@pytest.mark.parametrize(
    "num_generation_steps", num_generation_steps, ids=lambda x: f"num_generation_steps: {x}"
)
def test_sparse_attention_mla(
    scenario: Scenario,
    context_sequence_lengths: List[int],
    generation_seq_len_q: int,
    num_generation_steps: int,
):
    """Compare TRTLLM sparse MLA against the VanillaAttention golden."""
    golden = _test_sparse_attention_mla(
        "VANILLA",
        scenario,
        context_sequence_lengths,
        generation_seq_len_q,
        num_generation_steps,
    )
    actual = _test_sparse_attention_mla(
        "TRTLLM",
        scenario,
        context_sequence_lengths,
        generation_seq_len_q,
        num_generation_steps,
    )
    _assert_matches_vanilla(actual, golden, scenario.kv_cache_dtype, "TRTLLM")


def _test_sparse_attention_mla(
    backend_name: str,
    scenario: Scenario,
    context_sequence_lengths: List[int],
    generation_seq_len_q: int,
    num_generation_steps: int,
    sparse_topk: int = SPARSE_TOPK,
    seed: int = 123,
    topk_seed: int = 456,
) -> dict[str, torch.Tensor]:
    num_heads = scenario.num_heads
    num_kv_heads = scenario.num_kv_heads
    q_lora_rank = scenario.q_lora_rank
    qk_nope_head_dim = scenario.qk_nope_head_dim
    qk_rope_head_dim = scenario.qk_rope_head_dim
    v_head_dim = scenario.v_head_dim
    rope_append = scenario.rope_append
    if rope_append is False:
        print("rope_append is False, setting num_heads to 64")
        num_heads = 64
    kv_lora_rank = scenario.kv_lora_rank
    rope_config = RopeConfig(
        hidden_size=scenario.hidden_size,
        num_attention_heads=scenario.num_heads,
        rope_scaling={
            "beta_fast": scenario.rope_beta_fast,
            "beta_slow": scenario.rope_beta_slow,
            "factor": scenario.rope_factor,
            "mscale": scenario.rope_mscale,
            "mscale_all_dim": scenario.rope_mscale_all_dim,
            "original_max_position_embeddings": scenario.rope_original_max_position_embeddings,
            "type": scenario.rope_type,
        },
        max_position_embeddings=scenario.max_position_embeddings,
        rope_theta=scenario.rope_theta,
        qk_rope_head_dim=scenario.qk_rope_head_dim,
        model_type=scenario.model_type,
    )
    kv_cache_tokens_per_block = scenario.kv_cache_tokens_per_block
    num_layers = scenario.num_layers
    device = torch.device("cuda")
    dtype = scenario.dtype
    kv_cache_dtype = scenario.kv_cache_dtype

    assert sparse_topk % 128 == 0

    print(
        f"--------------------------------Test for scenario: {scenario} start--------------------------------"
    )

    return _run_test_for_backend(
        backend_name,
        num_heads,
        num_kv_heads,
        num_layers,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        rope_append,
        rope_config,
        kv_cache_tokens_per_block,
        device,
        dtype,
        kv_cache_dtype,
        context_sequence_lengths,
        generation_seq_len_q,
        num_generation_steps,
        sparse_topk,
        seed,
        topk_seed,
    )


def _run_test_for_backend(
    backend_name,
    num_heads,
    num_kv_heads,
    num_layers,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    rope_append,
    rope_config,
    kv_cache_tokens_per_block,
    device,
    dtype,
    kv_cache_dtype,
    context_sequence_lengths,
    generation_seq_len_q,
    num_generation_steps,
    sparse_topk,
    seed,
    topk_seed,
) -> dict[str, torch.Tensor]:
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=sparse_topk,
        skip_indexer_for_short_seqs=False,
    )
    is_vanilla = backend_name == "VANILLA"
    AttentionCls = (
        VanillaAttention if is_vanilla else get_attention_backend(backend_name, sparse_config)
    )
    # When rope_append is False, [448: 512) are used for qk_rope_head_dim
    kv_lora_rank = kv_lora_rank - qk_rope_head_dim if not rope_append else kv_lora_rank
    head_dim = kv_lora_rank + qk_rope_head_dim

    # Set seed for reproducibility.
    torch.manual_seed(seed)
    topk_generator = torch.Generator(device=device).manual_seed(topk_seed)

    # Create inputs
    inputs_per_layer = []
    for _ in range(num_layers):
        ctx_compressed_kv = torch.cat(
            [
                torch.empty(
                    [ctx_len, kv_lora_rank],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_k_pe = torch.cat(
            [
                torch.empty(
                    [ctx_len, qk_rope_head_dim],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_q = torch.cat(
            [
                torch.empty(
                    [ctx_len, num_heads, kv_lora_rank],  # sparse MLA uses absorption mode
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_q_pe = torch.cat(
            [
                torch.empty(
                    [ctx_len, num_heads, qk_rope_head_dim],
                    dtype=dtype,
                    device=device,
                ).uniform_(-1, 1)
                for ctx_len in context_sequence_lengths
            ]
        )
        ctx_fused_q = torch.cat([ctx_q, ctx_q_pe], dim=-1).view(-1, num_heads * head_dim)

        gen_compressed_kv_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, kv_lora_rank],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_k_pe_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, qk_rope_head_dim],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_q_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, num_heads, kv_lora_rank],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_q_pe_list = [
            torch.cat(
                [
                    torch.empty(
                        [generation_seq_len_q, num_heads, qk_rope_head_dim],
                        dtype=dtype,
                        device=device,
                    ).uniform_(-1, 1)
                    for _ in context_sequence_lengths
                ]
            )
            for _ in range(num_generation_steps)
        ]
        gen_fused_q_list = [
            torch.cat([gen_q_list[i], gen_q_pe_list[i]], dim=-1).view(-1, num_heads * head_dim)
            for i in range(num_generation_steps)
        ]

        inputs = {
            "ctx_compressed_kv": ctx_compressed_kv,
            "ctx_k_pe": ctx_k_pe,
            "ctx_q_pe": ctx_q_pe,
            "ctx_fused_q": ctx_fused_q,
            "gen_compressed_kv_list": gen_compressed_kv_list,
            "gen_k_pe_list": gen_k_pe_list,
            "gen_fused_q_list": gen_fused_q_list,
            "gen_q_pe_list": gen_q_pe_list,
        }
        inputs_per_layer.append(inputs)
        print(f"context sequence lengths: {context_sequence_lengths}")
        for key, val in inputs.items():
            if key.endswith("_list"):
                print(f"{key}: [{val[0].shape}] * {len(val)}")
            else:
                print(f"{key}: {val.shape}")

    # Setup attention module and metadata
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )
    mla_params = MLAParams(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
        rope_append=rope_append,
        predicted_tokens_per_seq=1,
    )

    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = pos_embd_params.rope.mscale_all_dim
    scaling_factor = pos_embd_params.rope.scale
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)

    quant_config = None
    if kv_cache_dtype == torch.float8_e4m3fn:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8.value)

    def create_layer(layer_idx: int, num_kv_heads: int):
        sparse_kwargs = (
            {"sparse_params": sparse_config.to_sparse_params(layer_idx=layer_idx)}
            if is_vanilla
            else {"sparse_attention_config": sparse_config}
        )
        return AttentionCls(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            **sparse_kwargs,
        )

    ctx_layers = [create_layer(layer_idx, num_kv_heads) for layer_idx in range(num_layers)]
    gen_layers = [create_layer(layer_idx, 1) for layer_idx in range(num_layers)]
    if is_vanilla:
        assert all(type(layer) is VanillaAttention for layer in ctx_layers + gen_layers)

    # NOTE: set up metadata, refer to tensorrt_llm/_torch/pyexecutor/model_engine.py
    # all layers share the same metadata
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    max_context_sequence_length = max(context_sequence_lengths)
    max_num_contexts = len(context_sequence_lengths)
    max_tokens = (
        (
            max_context_sequence_length
            + (num_generation_steps + 1) * generation_seq_len_q
            + kv_cache_tokens_per_block
            - 1
        )
        // kv_cache_tokens_per_block
        * kv_cache_tokens_per_block
        * max_num_contexts
    )

    pretrained_config = SimpleNamespace(
        rms_norm_eps=1e-6,
    )
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=kv_cache_tokens_per_block,
        max_seq_len=max_context_sequence_length + (num_generation_steps + 1) * generation_seq_len_q,
        max_batch_size=max_num_contexts,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(kv_cache_dtype)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    outputs = {}
    try:
        request_ids = list(range(max_num_contexts))
        kv_cache_manager.add_dummy_requests(request_ids, context_sequence_lengths)
        metadata_sparse_kwargs = {} if is_vanilla else {"sparse_attention_config": sparse_config}

        ctx_seq_lens = torch.tensor(context_sequence_lengths, dtype=torch.int)
        total_ctx_tokens = sum(context_sequence_lengths)
        attn_metadata = AttentionCls.Metadata(
            seq_lens=ctx_seq_lens,
            request_ids=request_ids,
            max_num_requests=max_num_contexts,
            num_contexts=max_num_contexts,
            prompt_lens=context_sequence_lengths,
            max_num_tokens=total_ctx_tokens,
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0 for _ in context_sequence_lengths],
            ),
            mapping=mapping,
            **metadata_sparse_kwargs,
        )
        attn_metadata.prepare()

        # run forward for each step and each layer
        for step in range(num_generation_steps + 1):
            if step > 0:
                _allocate_kv_cache_for_generation(
                    kv_cache_manager, request_ids, generation_seq_len_q
                )
                gen_seq_lens = torch.tensor(
                    [generation_seq_len_q] * max_num_contexts, dtype=torch.int
                )
                total_gen_tokens = max_num_contexts * generation_seq_len_q
                attn_metadata = AttentionCls.Metadata(
                    seq_lens=gen_seq_lens,
                    request_ids=request_ids,
                    max_num_requests=max_num_contexts,
                    num_contexts=0,
                    prompt_lens=context_sequence_lengths,
                    max_num_tokens=total_gen_tokens,
                    kv_cache_manager=kv_cache_manager,
                    kv_cache_params=KVCacheParams(
                        use_cache=True,
                        num_cached_tokens_per_seq=[
                            ctx_len + (step - 1) * generation_seq_len_q
                            for ctx_len in context_sequence_lengths
                        ],
                    ),
                    mapping=mapping,
                    enable_flash_mla=torch.cuda.get_device_capability() == (9, 0),
                    **metadata_sparse_kwargs,
                )
                attn_metadata.prepare()
            for layer_idx in range(num_layers):
                print(f"---- step {step} layer {layer_idx} start ----")
                if step == 0:
                    fused_q = inputs_per_layer[layer_idx]["ctx_fused_q"]
                    compressed_kv = inputs_per_layer[layer_idx]["ctx_compressed_kv"]
                    k_pe = inputs_per_layer[layer_idx]["ctx_k_pe"]
                    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
                    q_pe = inputs_per_layer[layer_idx]["ctx_q_pe"]
                    topk_indices = _build_sparse_topk_indices_context(
                        context_sequence_lengths,
                        sparse_topk,
                        device,
                        generator=topk_generator,
                    )
                    result = ctx_layers[layer_idx].forward(
                        fused_q.clone(),
                        None,
                        None,
                        attn_metadata,
                        attention_input_type=AttentionInputType.context_only,
                        latent_cache=latent_cache.clone(),
                        q_pe=q_pe,
                        topk_indices=topk_indices,
                    )
                else:
                    fused_q = inputs_per_layer[layer_idx]["gen_fused_q_list"][step - 1]
                    q_pe = inputs_per_layer[layer_idx]["gen_q_pe_list"][step - 1]
                    compressed_kv = inputs_per_layer[layer_idx]["gen_compressed_kv_list"][step - 1]
                    k_pe = inputs_per_layer[layer_idx]["gen_k_pe_list"][step - 1]
                    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
                    cached_lens = [
                        ctx_len + (step - 1) * generation_seq_len_q
                        for ctx_len in context_sequence_lengths
                    ]
                    topk_indices = _build_sparse_topk_indices_generation(
                        cached_lens,
                        generation_seq_len_q,
                        sparse_topk,
                        device,
                        generator=topk_generator,
                    )
                    if is_vanilla:
                        backend_fused_q = fused_q.clone()
                        backend_latent_cache = latent_cache.clone()
                        generation_kwargs = {}
                    else:
                        num_tokens = fused_q.size(0)
                        num_seqs = attn_metadata.kv_lens_cuda_runtime.size(0)
                        cu_q_seqlens = torch.empty(
                            num_seqs + 1, dtype=torch.int32, device=fused_q.device
                        )
                        cu_kv_seqlens = torch.empty(
                            num_seqs + 1, dtype=torch.int32, device=fused_q.device
                        )
                        fmha_scheduler_counter = torch.empty(
                            1, dtype=torch.uint32, device=fused_q.device
                        )
                        has_fp8_kv_cache = (
                            gen_layers[layer_idx].has_fp8_kv_cache
                            if hasattr(gen_layers[layer_idx], "has_fp8_kv_cache")
                            else False
                        )

                        if has_fp8_kv_cache:
                            mla_bmm1_scale = torch.empty(
                                2, dtype=torch.float32, device=fused_q.device
                            )
                            mla_bmm2_scale = torch.empty(
                                1, dtype=torch.float32, device=fused_q.device
                            )
                            quant_q_buffer = torch.empty(
                                num_tokens,
                                num_heads * head_dim,
                                dtype=torch.uint8,
                                device=fused_q.device,
                            )
                        else:
                            mla_bmm1_scale = None
                            mla_bmm2_scale = None
                            quant_q_buffer = None

                        gen_layers[layer_idx].mla_rope_generation(
                            fused_q,
                            q_pe,
                            latent_cache,
                            attn_metadata,
                            cu_q_seqlens,
                            cu_kv_seqlens,
                            fmha_scheduler_counter,
                            mla_bmm1_scale,
                            mla_bmm2_scale,
                            quant_q_buffer,
                        )
                        backend_fused_q = fused_q
                        backend_latent_cache = latent_cache
                        generation_kwargs = {
                            "cu_q_seqlens": cu_q_seqlens,
                            "cu_kv_seqlens": cu_kv_seqlens,
                            "fmha_scheduler_counter": fmha_scheduler_counter,
                            "mla_bmm1_scale": mla_bmm1_scale,
                            "mla_bmm2_scale": mla_bmm2_scale,
                            "quant_q_buffer": quant_q_buffer,
                        }
                    result = gen_layers[layer_idx].forward(
                        backend_fused_q,
                        None,
                        None,
                        attn_metadata,
                        attention_input_type=AttentionInputType.generation_only,
                        latent_cache=backend_latent_cache,
                        q_pe=q_pe,
                        topk_indices=topk_indices,
                        **generation_kwargs,
                    )
                # Record results for the Vanilla-golden comparison.
                print(
                    f"{backend_name} output mean: {result.abs().mean().item()}, max: {result.abs().max().item()}"
                )
                print(
                    f"Test for sparse MLA in {backend_name} backend passed at layer {layer_idx} in step {step}"
                )
                print(f"---- step {step} layer {layer_idx} end ----")
                phase = "context" if step == 0 else f"generation_{step - 1}"
                outputs[f"{phase}_layer_{layer_idx}"] = result.detach().clone()

        print(f"Test for sparse MLA in {backend_name} backend passed")
        return outputs
    finally:
        kv_cache_manager.shutdown()


if __name__ == "__main__":
    test_sparse_attention_mla(
        scenario=scenarios[0],
        context_sequence_lengths=context_sequence_lengths[0],
        generation_seq_len_q=generation_seq_len_q[0],
        num_generation_steps=num_generation_steps[0],
    )
