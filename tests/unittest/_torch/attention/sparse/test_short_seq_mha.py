# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Test suite for the short-sequence MHA optimization path in MLA.

When TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD is set and the total number of
packed context tokens is within the threshold, MLA.forward_context_dsa
dispatches to forward_context which routes to the appropriate MHA path:
forward_context_default (no cached tokens), forward_context_with_cached_kv
(cached tokens), or forward_context_with_chunked_prefill (chunked prefill).
"""

import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm._utils import get_sm_version, str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

# DSACacheManager creates background ThreadPoolExecutor threads that outlive
# the test. Disable pytest-threadleak for this entire module.
pytestmark = pytest.mark.threadleak(enabled=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embedding(x, cos_sin):
    """Apply rotary position embedding to x."""
    original_dtype = x.dtype
    cos, sin = cos_sin.chunk(2, dim=-2)
    cos = cos.squeeze(1)
    sin = sin.squeeze(1)
    x_interleaved = x.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
    cos_expanded = cos.view(cos.shape[0], *([1] * (x.ndim - 2)), cos.shape[-1])
    sin_expanded = sin.view(sin.shape[0], *([1] * (x.ndim - 2)), sin.shape[-1])
    x_rotated = (x_interleaved * cos_expanded) + (rotate_half(x_interleaved) * sin_expanded)
    return x_rotated.to(original_dtype)


def calculate_reference_output(
    q,
    compressed_kv,
    k_pe,
    kv_b_proj_weight,
    rope_cos_sin,
    sequence_lengths,
    num_heads,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    softmax_scale,
    device,
):
    """Reference implementation using kv_b_proj expansion + causal SDPA.

    This mirrors the short-seq MHA path (forward_context_default):
    1. Apply RoPE to q (rope portion) and k_pe.
    2. Expand compressed_kv via kv_b_proj to get k_nope and v.
    3. Construct K = [k_nope, RoPE(k_pe)] per head.
    4. Run causal attention per sequence.
    """
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    results = []
    offset = 0

    for seq_len in sequence_lengths:
        q_seq = q[offset : offset + seq_len].view(-1, num_heads, qk_head_dim)
        compressed_kv_seq = compressed_kv[offset : offset + seq_len]
        k_pe_seq = k_pe[offset : offset + seq_len]
        cos_sin = rope_cos_sin[:seq_len]

        # Apply RoPE to q rope portion
        q_nope = q_seq[..., :qk_nope_head_dim]
        q_pe = q_seq[..., qk_nope_head_dim:]
        q_pe_roped = apply_rotary_embedding(q_pe, cos_sin)
        q_full = torch.cat([q_nope, q_pe_roped], dim=-1)

        # Apply RoPE to k_pe
        k_pe_roped = apply_rotary_embedding(k_pe_seq, cos_sin)

        # Expand via kv_b_proj: weight is [out_features, kv_lora_rank]
        kv = torch.nn.functional.linear(compressed_kv_seq, kv_b_proj_weight)
        k_nope, v = kv.split([num_heads * qk_nope_head_dim, num_heads * v_head_dim], -1)

        k_nope_r = k_nope.view(-1, num_heads, qk_nope_head_dim)
        k_pe_expanded = k_pe_roped.view(-1, 1, qk_rope_head_dim).expand(-1, num_heads, -1)
        k_full = torch.cat([k_nope_r, k_pe_expanded], dim=-1)

        v_r = v.view(-1, num_heads, v_head_dim)

        # SDPA: [1, H, S, D]
        q_b = q_full.unsqueeze(0).transpose(1, 2)
        k_b = k_full.unsqueeze(0).transpose(1, 2)
        v_b = v_r.unsqueeze(0).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_b, k_b, v_b, is_causal=True, scale=softmax_scale
        )
        attn_out = attn_out.transpose(1, 2).squeeze(0)
        results.append(attn_out.reshape(seq_len, num_heads * v_head_dim))
        offset += seq_len

    return torch.cat(results, dim=0)


@dataclass
class RopeConfig:
    hidden_size: int
    num_attention_heads: int
    rope_scaling: dict
    max_position_embeddings: int
    rope_theta: float
    qk_rope_head_dim: int
    model_type: str


# Model configuration matching DeepSeek V3
NUM_HEADS = 128
Q_LORA_RANK = 512
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
HIDDEN_SIZE = 2048
MAX_POSITION_EMBEDDINGS = 4096
TOKENS_PER_BLOCK = 64
NUM_LAYERS = 1
LAYER_IDX = 0
TOPK_TOKENS = 2048
NN_INIT_STD = 0.02

# Test batch specifications: (name, sequence_lengths)
BATCH_SPECS = [
    ("single_short", [16]),
    ("single_medium", [64]),
    ("multi_short", [8, 12, 16]),
    ("multi_varied", [4, 32, 10]),
    ("single_one_token", [1]),
]

# Chunked context specifications: (name, [(cached_tokens, new_tokens), ...])
# Each tuple represents one sequence with its cached/new token split.
CHUNKED_CONTEXT_SPECS = [
    ("single_small", [(16, 8)]),
    ("single_medium", [(64, 32)]),
    ("large_cache_small_new", [(128, 4)]),
    ("single_one_new_token", [(64, 1)]),
    ("multi_seq", [(64, 32), (32, 16)]),
    ("multi_varied", [(128, 8), (16, 64)]),
    ("three_seqs", [(32, 16), (16, 8), (48, 4)]),
]


def _make_rope_config():
    return RopeConfig(
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rope_theta=10000.0,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        model_type="deepseek_v2",
    )


def _make_rope_cos_sin(rope_config, device):
    return (
        torch.tensor(
            RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                rope_config.max_position_embeddings,
                rope_config.qk_rope_head_dim,
                rope_config.rope_theta,
                rope_config.rope_scaling["factor"],
                rope_config.rope_scaling["original_max_position_embeddings"],
                rope_config.rope_scaling["beta_fast"],
                rope_config.rope_scaling["beta_slow"],
                rope_config.rope_scaling["mscale"],
                rope_config.rope_scaling["mscale_all_dim"],
            )[1],
            dtype=torch.float32,
            device=device,
        )
        .reshape(rope_config.max_position_embeddings, -1, 2)
        .transpose(-2, -1)
    )


def _compute_softmax_scale(rope_config):
    def yarn_get_mscale(scale=1, mscale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    mscale_all_dim = rope_config.rope_scaling["mscale_all_dim"]
    scaling_factor = rope_config.rope_scaling["factor"]
    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
    q_scaling = 1.0 / (mscale * mscale)
    return 1.0 / (math.sqrt(QK_HEAD_DIM) * q_scaling)


def _build_mla(rope_config, device, threshold):
    """Build an MLA module with DSA config and the given threshold."""
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=TOPK_TOKENS,
    )
    pretrained_config = SimpleNamespace(rms_norm_eps=1e-6)
    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(rope_config),
        is_neox=False,
    )

    # Set env var BEFORE constructing MLA so __init__ picks it up.
    old_val = os.environ.get("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD")
    os.environ["TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD"] = str(threshold)
    try:
        mla = MLA(
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            num_key_value_heads=1,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            q_lora_rank=Q_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            predicted_tokens_per_seq=1,
            max_position_embeddings=MAX_POSITION_EMBEDDINGS,
            bias=False,
            pos_embd_params=pos_embd_params,
            layer_idx=LAYER_IDX,
            dtype=torch.bfloat16,
            config=model_config,
        ).to(device)
    finally:
        # Restore env var.
        if old_val is None:
            os.environ.pop("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", None)
        else:
            os.environ["TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD"] = old_val

    # mla.mqa (DSATrtllmAttention) does not inherit from nn.Module, so
    # mla.to(device) does NOT recursively move its children. Explicitly
    # move the indexer (which IS an nn.Module) so its weights are on CUDA.
    if hasattr(mla, "mqa") and hasattr(mla.mqa, "indexer"):
        mla.mqa.indexer.to(device)

    return mla, mapping, sparse_config, model_config


def _init_mla_weights(mla):
    """Initialize MLA weights deterministically.

    The kv_b_proj weight must use the LOADED layout (as produced by
    modeling_deepseekv3.py's load_kv_b_proj_and_k_b_proj_trans), NOT the raw
    HuggingFace layout. The loaded layout is:
        [all_heads_k_nope, all_heads_v] along the output dimension,
    i.e. rows 0..num_heads*qk_nope-1 are k_nope weights for all heads, and
    rows num_heads*qk_nope..end are v weights for all heads.

    This layout is what forward_context_default expects when it does:
        kv.split([num_heads * qk_nope, num_heads * v], dim=-1)
    """
    with torch.no_grad():
        # Generate k_nope and v weights separately, then concatenate in the
        # loaded layout: [all_k_nope || all_v].
        k_nope_weight = torch.empty(
            NUM_HEADS,
            QK_NOPE_HEAD_DIM,
            KV_LORA_RANK,
            dtype=mla.kv_b_proj.weight.dtype,
            device=mla.kv_b_proj.weight.device,
        )
        k_nope_weight.normal_(mean=0.0, std=NN_INIT_STD)

        v_weight = torch.empty(
            NUM_HEADS,
            V_HEAD_DIM,
            KV_LORA_RANK,
            dtype=mla.kv_b_proj.weight.dtype,
            device=mla.kv_b_proj.weight.device,
        )
        v_weight.normal_(mean=0.0, std=NN_INIT_STD)

        # kv_b_proj.weight: [num_heads*(qk_nope+v_head), kv_lora_rank]
        # Loaded layout: first num_heads*qk_nope rows are k_nope, rest are v.
        mla.kv_b_proj.weight.data = torch.cat(
            [
                k_nope_weight.reshape(NUM_HEADS * QK_NOPE_HEAD_DIM, KV_LORA_RANK),
                v_weight.reshape(NUM_HEADS * V_HEAD_DIM, KV_LORA_RANK),
            ],
            dim=0,
        )

        # k_b_proj_trans: [num_heads, kv_lora_rank, qk_nope_head_dim]
        mla.k_b_proj_trans.data = k_nope_weight.transpose(1, 2).contiguous()

        # v_b_proj: [num_heads, v_head_dim, kv_lora_rank]
        mla.v_b_proj.data = v_weight.contiguous()

        mla.mqa.indexer.wq_b.weight.normal_(mean=0.0, std=NN_INIT_STD)
        mla.mqa.indexer.wk.weight.normal_(mean=0.0, std=NN_INIT_STD)
        mla.mqa.indexer.weights_proj.weight.normal_(mean=0.0, std=NN_INIT_STD)


def _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device):
    """Build a DSACacheManager for the given batch."""
    max_seqlen = max(seq_lens)
    max_tokens = 16384

    cache_dtype = torch.bfloat16
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max_seqlen,
        max_batch_size=len(seq_lens),
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(cache_dtype)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )

    # Allocate cache for all requests as pure prefill.
    for req_idx, seq_len in enumerate(seq_lens):
        kv_cache_manager.add_dummy_requests(
            request_ids=[req_idx],
            token_nums=[seq_len],
            is_gen=False,
            prepare_resource=True,
        )

    return kv_cache_manager


def _build_kv_cache_manager_with_cached(
    mapping, sparse_config, model_config, cached_per_seq, new_per_seq, device
):
    """Build a DSACacheManager with blocks allocated for cached + new tokens.

    The cache manager allocates enough blocks for the full sequence
    (cached + new tokens per request), which is needed so that
    DSAMetadata.prepare() can compute correct kv_lens and
    num_ctx_cached_tokens.
    """
    total_per_seq = [c + n for c, n in zip(cached_per_seq, new_per_seq)]
    max_seqlen = max(total_per_seq)
    max_tokens = 16384

    cache_dtype = torch.bfloat16
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(max_tokens=max_tokens, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max_seqlen,
        max_batch_size=len(cached_per_seq),
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(cache_dtype)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )

    for req_idx, total in enumerate(total_per_seq):
        kv_cache_manager.add_dummy_requests(
            request_ids=[req_idx],
            token_nums=[total],
            is_gen=False,
            prepare_resource=True,
        )

    return kv_cache_manager


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
@pytest.mark.parametrize("batch_name,seq_lens", BATCH_SPECS, ids=[b[0] for b in BATCH_SPECS])
def test_forward_context_short_mha(batch_name: str, seq_lens: List[int]):
    """Test that the short-seq MHA path produces correct attention output.

    Compares the output of the short-seq MHA path (forward_context_default)
    against a standalone reference implementation that performs the same math:
    kv_b_proj expansion, RoPE application, and causal attention.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    rope_cos_sin = _make_rope_cos_sin(rope_config, device)
    softmax_scale = _compute_softmax_scale(rope_config)

    # Set threshold high enough that total packed tokens take the short MHA path.
    threshold = sum(seq_lens) + 100
    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)

    # Verify short-seq MHA is configured.
    assert mla.short_seq_mha_threshold == threshold
    assert mla.mha is not None
    assert not mla.apply_rotary_emb, "Expected rope_fusion=True for DSA"

    kv_cache_manager = _build_kv_cache_manager(
        mapping, sparse_config, model_config, seq_lens, device
    )
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    # Generate inputs.
    total_tokens = sum(seq_lens)
    num_contexts = len(seq_lens)
    q = torch.randn(total_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv = torch.randn(total_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(total_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat(
        [torch.arange(slen, device=device, dtype=torch.int32) for slen in seq_lens]
    )
    output = torch.empty(total_tokens, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)

    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(range(num_contexts)),
        max_num_requests=num_contexts,
        num_contexts=num_contexts,
        prompt_lens=seq_lens,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_contexts,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    # Verify the threshold guard will trigger (total packed tokens <= threshold).
    assert total_tokens <= threshold

    # Clone inputs since forward_context_dsa may modify them in-place.
    q_for_ref = q.clone()
    compressed_kv_for_ref = compressed_kv.clone()
    k_pe_for_ref = k_pe.clone()

    # Run the actual forward (should dispatch to forward_context_default).
    # topk_indices=None: the short MHA path does not use sparse routing,
    # so the indexer computation can be skipped entirely for short sequences.
    mla.forward_context_dsa(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        attn_metadata=attn_metadata,
        output=output,
        latent_cache=latent_cache,
        topk_indices=None,
        position_ids=position_ids,
    )

    # Compute reference.
    kv_b_proj_weight = mla.kv_b_proj.weight.data
    ref_output = calculate_reference_output(
        q_for_ref,
        compressed_kv_for_ref,
        k_pe_for_ref,
        kv_b_proj_weight,
        rope_cos_sin,
        seq_lens,
        NUM_HEADS,
        KV_LORA_RANK,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        softmax_scale,
        device,
    )

    # Compare.
    assert output.shape == ref_output.shape
    assert torch.isfinite(output).all(), "Output contains non-finite values"
    assert torch.isfinite(ref_output).all(), "Reference contains non-finite values"

    abs_error = (output - ref_output).abs()
    torch.testing.assert_close(output, ref_output, rtol=0.05, atol=0.05)
    print(
        f"[{batch_name}] PASSED: max_error={abs_error.max():.4f}, mean_error={abs_error.mean():.6f}"
    )

    kv_cache_manager.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
def test_short_mha_not_triggered_when_threshold_zero():
    """Verify that with threshold=0, the short MHA path is NOT active."""
    device = torch.device("cuda")
    rope_config = _make_rope_config()
    mla, _, _, _ = _build_mla(rope_config, device, threshold=0)

    assert mla.short_seq_mha_threshold == 0
    assert mla.mha is None


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
def test_short_mha_boundary_threshold_equals_max_seq():
    """Test the boundary condition where threshold == total packed tokens.

    The dispatch guard uses `<=`, so threshold == num_ctx_tokens should still
    trigger the short MHA path. This verifies correctness at the exact boundary.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    rope_cos_sin = _make_rope_cos_sin(rope_config, device)
    softmax_scale = _compute_softmax_scale(rope_config)

    seq_lens = [24, 16]
    # Threshold exactly equals total packed tokens — short MHA should trigger.
    threshold = sum(seq_lens)
    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)

    kv_cache_manager = _build_kv_cache_manager(
        mapping, sparse_config, model_config, seq_lens, device
    )
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    total_tokens = sum(seq_lens)
    num_contexts = len(seq_lens)
    q = torch.randn(total_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv = torch.randn(total_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(total_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat(
        [torch.arange(slen, device=device, dtype=torch.int32) for slen in seq_lens]
    )
    output = torch.empty(total_tokens, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)

    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(range(num_contexts)),
        max_num_requests=num_contexts,
        num_contexts=num_contexts,
        prompt_lens=seq_lens,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_contexts,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    # Verify boundary: total packed tokens == threshold.
    assert total_tokens == threshold

    q_for_ref = q.clone()
    compressed_kv_for_ref = compressed_kv.clone()
    k_pe_for_ref = k_pe.clone()

    # topk_indices=None: the short MHA path does not use sparse routing.
    mla.forward_context_dsa(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        attn_metadata=attn_metadata,
        output=output,
        latent_cache=latent_cache,
        topk_indices=None,
        position_ids=position_ids,
    )

    kv_b_proj_weight = mla.kv_b_proj.weight.data
    ref_output = calculate_reference_output(
        q_for_ref,
        compressed_kv_for_ref,
        k_pe_for_ref,
        kv_b_proj_weight,
        rope_cos_sin,
        seq_lens,
        NUM_HEADS,
        KV_LORA_RANK,
        QK_NOPE_HEAD_DIM,
        QK_ROPE_HEAD_DIM,
        V_HEAD_DIM,
        softmax_scale,
        device,
    )

    abs_error = (output - ref_output).abs()
    torch.testing.assert_close(output, ref_output, rtol=0.05, atol=0.05)
    print(
        f"[boundary_threshold] PASSED: max_error={abs_error.max():.4f}, "
        f"mean_error={abs_error.mean():.6f}"
    )

    kv_cache_manager.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
def test_short_mha_not_triggered_when_seq_exceeds_threshold():
    """Verify that when total packed tokens > threshold, the standard path is used.

    We set threshold=16 but use seq_len=32 (total_tokens=32 > 16). The output
    should match the standard absorption path (not the short MHA path). We
    verify this indirectly: since both paths must produce the same result for
    correctness, we just confirm the module runs without error and produces
    finite output.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    seq_lens = [32]

    # Threshold is below total packed tokens — short MHA should NOT trigger.
    threshold = 16
    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)

    kv_cache_manager = _build_kv_cache_manager(
        mapping, sparse_config, model_config, seq_lens, device
    )
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    total_tokens = sum(seq_lens)
    num_contexts = len(seq_lens)
    q = torch.randn(total_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv = torch.randn(total_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(total_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE, dtype=dtype, device=device)
    qr = torch.randn(total_tokens, Q_LORA_RANK, dtype=dtype, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat(
        [torch.arange(slen, device=device, dtype=torch.int32) for slen in seq_lens]
    )
    output = torch.empty(total_tokens, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)

    attn_metadata = AttentionCls.Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(range(num_contexts)),
        max_num_requests=num_contexts,
        num_contexts=num_contexts,
        prompt_lens=seq_lens,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_contexts,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata.prepare()

    # Threshold is below total packed tokens; standard path should be used.
    assert total_tokens > threshold

    topk_indices = mla.mqa.indexer(
        qr,
        hidden_states,
        attn_metadata,
        position_ids,
        indexer_k=mla.mqa.indexer.wk(hidden_states),
    )

    mla.forward_context_dsa(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        attn_metadata=attn_metadata,
        output=output,
        latent_cache=latent_cache,
        topk_indices=topk_indices,
        position_ids=position_ids,
    )

    assert torch.isfinite(output).all(), "Output contains non-finite values"
    print(
        f"[threshold_exceeded] PASSED: standard path ran without error, "
        f"output mean={output.abs().mean():.4f}"
    )

    kv_cache_manager.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
def test_short_mha_agrees_with_absorption_path():
    """Compare short MHA path output against the absorption path output.

    Both paths should produce numerically close results for the same inputs.
    This is an A/B test: run the same inputs through both paths and verify
    they agree within tolerance.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    seq_lens = [16]
    total_tokens = sum(seq_lens)

    # Build two MLA instances: one with short MHA, one without.
    mla_short, mapping, sparse_config, model_config = _build_mla(
        rope_config, device, threshold=total_tokens + 100
    )
    mla_absorption, _, _, _ = _build_mla(rope_config, device, threshold=0)

    # Copy weights from short to absorption so they are identical.
    with torch.no_grad():
        _init_mla_weights(mla_short)
        # Copy all nn.Module parameters (registered on MLA directly).
        for (name_s, param_s), (name_a, param_a) in zip(
            mla_short.named_parameters(), mla_absorption.named_parameters()
        ):
            assert name_s == name_a, f"Parameter name mismatch: {name_s} vs {name_a}"
            param_a.data.copy_(param_s.data)
        # mla.mqa is NOT an nn.Module, so named_parameters() above misses the
        # indexer weights. Copy them explicitly so both paths use the same
        # sparse routing (topk_indices).
        for (name_s, param_s), (name_a, param_a) in zip(
            mla_short.mqa.indexer.named_parameters(), mla_absorption.mqa.indexer.named_parameters()
        ):
            assert name_s == name_a, f"Indexer param mismatch: {name_s} vs {name_a}"
            param_a.data.copy_(param_s.data)

    # Shared KV cache managers (need separate ones since they have state).
    kv_cache_manager_short = _build_kv_cache_manager(
        mapping, sparse_config, model_config, seq_lens, device
    )
    kv_cache_manager_absorption = _build_kv_cache_manager(
        mapping, sparse_config, model_config, seq_lens, device
    )
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    num_contexts = len(seq_lens)
    q = torch.randn(total_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv = torch.randn(total_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(total_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE, dtype=dtype, device=device)
    qr = torch.randn(total_tokens, Q_LORA_RANK, dtype=dtype, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat(
        [torch.arange(slen, device=device, dtype=torch.int32) for slen in seq_lens]
    )

    def _run_forward(mla_module, kv_cache_mgr):
        output = torch.empty(total_tokens, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
        attn_metadata = AttentionCls.Metadata(
            seq_lens=torch.tensor(seq_lens, dtype=torch.int),
            request_ids=list(range(num_contexts)),
            max_num_requests=num_contexts,
            num_contexts=num_contexts,
            prompt_lens=seq_lens,
            max_num_tokens=total_tokens,
            kv_cache_manager=kv_cache_mgr,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0] * num_contexts,
            ),
            mapping=mapping,
            sparse_attention_config=sparse_config,
        )
        attn_metadata.prepare()

        # Only compute topk_indices when needed: the short MHA path
        # does not use sparse routing, so the indexer can be skipped.
        use_short_mha = (
            mla_module.short_seq_mha_threshold > 0
            and total_tokens <= mla_module.short_seq_mha_threshold
        )
        if use_short_mha:
            topk_indices = None
        else:
            topk_indices = mla_module.mqa.indexer(
                qr.clone(),
                hidden_states.clone(),
                attn_metadata,
                position_ids.clone(),
                indexer_k=mla_module.mqa.indexer.wk(hidden_states.clone()),
            )

        mla_module.forward_context_dsa(
            q=q.clone(),
            compressed_kv=compressed_kv.clone(),
            k_pe=k_pe.clone(),
            attn_metadata=attn_metadata,
            output=output,
            latent_cache=latent_cache.clone(),
            topk_indices=topk_indices,
            position_ids=position_ids.clone(),
        )
        return output

    output_short = _run_forward(mla_short, kv_cache_manager_short)
    output_absorption = _run_forward(mla_absorption, kv_cache_manager_absorption)

    assert torch.isfinite(output_short).all()
    assert torch.isfinite(output_absorption).all()

    abs_error = (output_short - output_absorption).abs()
    # The two paths use different intermediate representations (kv_b_proj
    # expansion + dense attention vs absorption BMMs + sparse MLA kernel),
    # so bf16 accumulation differences are expected. The tolerance is set
    # based on observed max errors (~0.06) with headroom for variance.
    torch.testing.assert_close(output_short, output_absorption, rtol=0.08, atol=0.08)
    print(f"[a_b_test] PASSED: max_error={abs_error.max():.4f}, mean_error={abs_error.mean():.6f}")

    kv_cache_manager_short.shutdown()
    kv_cache_manager_absorption.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
def test_short_mha_cached_kv_correctness():
    """Verify forward_context_with_cached_kv produces correct output.

    Runs chunk 1 (pure prefill, 200 tokens) to populate the KV cache,
    then chunk 2 (100 new tokens with 200 cached) through the
    forward_context_with_cached_kv path.  Compares against a single-pass
    reference over all 300 tokens.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()

    cached_per_seq = [200]
    new_per_seq = [100]
    total_per_seq = [c + n for c, n in zip(cached_per_seq, new_per_seq)]
    num_seqs = len(cached_per_seq)
    total_tokens = sum(total_per_seq)
    total_chunk1 = sum(cached_per_seq)
    total_chunk2 = sum(new_per_seq)
    threshold = total_tokens + 100

    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    # ---- Generate full-sequence input data ----
    q_full = torch.randn(total_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv_full = torch.randn(total_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe_full = torch.randn(total_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    latent_cache_full = torch.cat([compressed_kv_full, k_pe_full], dim=-1)
    position_ids_full = torch.cat(
        [torch.arange(t, device=device, dtype=torch.int32) for t in total_per_seq]
    )

    chunk1_idx = torch.arange(total_chunk1, dtype=torch.long, device=device)
    chunk2_idx = torch.arange(total_chunk1, total_tokens, dtype=torch.long, device=device)

    # ---- Reference: single-pass full prefill ----
    kv_cache_ref = _build_kv_cache_manager(
        mapping, sparse_config, model_config, total_per_seq, device
    )
    output_ref = torch.empty(total_tokens, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_ref = AttentionCls.Metadata(
        seq_lens=torch.tensor(total_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=total_per_seq,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_ref,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata_ref.prepare()
    mla.forward_context_dsa(
        q=q_full.clone(),
        compressed_kv=compressed_kv_full.clone(),
        k_pe=k_pe_full.clone(),
        attn_metadata=attn_metadata_ref,
        output=output_ref,
        latent_cache=latent_cache_full.clone(),
        topk_indices=None,
        position_ids=position_ids_full.clone(),
    )
    assert torch.isfinite(output_ref).all(), "Reference output has non-finite values"
    ref_chunk2_output = output_ref[chunk2_idx]

    # ---- Chunk 1: pure prefill to populate KV cache ----
    kv_cache_chunked = _build_kv_cache_manager(
        mapping, sparse_config, model_config, total_per_seq, device
    )
    output_c1 = torch.empty(total_chunk1, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_c1 = AttentionCls.Metadata(
        seq_lens=torch.tensor(cached_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=cached_per_seq,
        max_num_tokens=total_chunk1,
        kv_cache_manager=kv_cache_chunked,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata_c1.prepare()
    position_ids_c1 = torch.cat(
        [torch.arange(c, device=device, dtype=torch.int32) for c in cached_per_seq]
    )
    mla.forward_context_dsa(
        q=q_full[chunk1_idx].clone(),
        compressed_kv=compressed_kv_full[chunk1_idx].clone(),
        k_pe=k_pe_full[chunk1_idx].clone(),
        attn_metadata=attn_metadata_c1,
        output=output_c1,
        latent_cache=latent_cache_full[chunk1_idx].clone(),
        topk_indices=None,
        position_ids=position_ids_c1,
    )

    # ---- Chunk 2: cached KV → forward_context_with_cached_kv ----
    output_c2 = torch.empty(total_chunk2, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_c2 = AttentionCls.Metadata(
        seq_lens=torch.tensor(new_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=new_per_seq,
        max_num_tokens=total_chunk2,
        kv_cache_manager=kv_cache_chunked,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_per_seq,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
        enable_context_mla_with_cached_kv=True,
    )
    attn_metadata_c2.prepare()
    assert attn_metadata_c2.num_ctx_cached_tokens == sum(cached_per_seq)

    position_ids_c2 = torch.cat(
        [
            torch.arange(c, c + n, device=device, dtype=torch.int32)
            for c, n in zip(cached_per_seq, new_per_seq)
        ]
    )
    mla.forward_context_dsa(
        q=q_full[chunk2_idx].clone(),
        compressed_kv=compressed_kv_full[chunk2_idx].clone(),
        k_pe=k_pe_full[chunk2_idx].clone(),
        attn_metadata=attn_metadata_c2,
        output=output_c2,
        latent_cache=latent_cache_full[chunk2_idx].clone(),
        topk_indices=None,
        position_ids=position_ids_c2,
    )

    assert torch.isfinite(output_c2).all(), "Chunk 2 output has non-finite values"
    abs_error = (output_c2 - ref_chunk2_output).abs()
    torch.testing.assert_close(output_c2, ref_chunk2_output, rtol=0.08, atol=0.08)
    print(
        f"[cached_kv_single] PASSED: max_error={abs_error.max():.4f}, "
        f"mean_error={abs_error.mean():.6f}"
    )

    kv_cache_ref.shutdown()
    kv_cache_chunked.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
def test_short_mha_cached_kv_multi_seq_correctness():
    """Multi-sequence variant of forward_context_with_cached_kv correctness.

    Two sequences with different cached/new counts processed through
    chunk 1 + chunk 2, compared against a single-pass reference.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()

    cached_per_seq = [128, 64]
    new_per_seq = [32, 16]
    total_per_seq = [c + n for c, n in zip(cached_per_seq, new_per_seq)]
    num_seqs = len(cached_per_seq)
    total_tokens = sum(total_per_seq)
    total_chunk1 = sum(cached_per_seq)
    total_chunk2 = sum(new_per_seq)
    threshold = total_tokens + 100

    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    # ---- Generate full-sequence input data ----
    q_full = torch.randn(total_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv_full = torch.randn(total_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe_full = torch.randn(total_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    latent_cache_full = torch.cat([compressed_kv_full, k_pe_full], dim=-1)
    position_ids_full = torch.cat(
        [torch.arange(t, device=device, dtype=torch.int32) for t in total_per_seq]
    )

    # Build per-sequence index arrays.
    chunk1_indices, chunk2_indices = [], []
    offset = 0
    for c, n in zip(cached_per_seq, new_per_seq):
        chunk1_indices.extend(range(offset, offset + c))
        chunk2_indices.extend(range(offset + c, offset + c + n))
        offset += c + n
    chunk1_idx = torch.tensor(chunk1_indices, dtype=torch.long, device=device)
    chunk2_idx = torch.tensor(chunk2_indices, dtype=torch.long, device=device)

    # ---- Reference: single-pass full prefill ----
    kv_cache_ref = _build_kv_cache_manager(
        mapping, sparse_config, model_config, total_per_seq, device
    )
    output_ref = torch.empty(total_tokens, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_ref = AttentionCls.Metadata(
        seq_lens=torch.tensor(total_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=total_per_seq,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_ref,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata_ref.prepare()
    mla.forward_context_dsa(
        q=q_full.clone(),
        compressed_kv=compressed_kv_full.clone(),
        k_pe=k_pe_full.clone(),
        attn_metadata=attn_metadata_ref,
        output=output_ref,
        latent_cache=latent_cache_full.clone(),
        topk_indices=None,
        position_ids=position_ids_full.clone(),
    )
    assert torch.isfinite(output_ref).all(), "Reference output has non-finite values"
    ref_chunk2_output = output_ref[chunk2_idx]

    # ---- Chunk 1: pure prefill ----
    kv_cache_chunked = _build_kv_cache_manager(
        mapping, sparse_config, model_config, total_per_seq, device
    )
    output_c1 = torch.empty(total_chunk1, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_c1 = AttentionCls.Metadata(
        seq_lens=torch.tensor(cached_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=cached_per_seq,
        max_num_tokens=total_chunk1,
        kv_cache_manager=kv_cache_chunked,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata_c1.prepare()
    position_ids_c1 = torch.cat(
        [torch.arange(c, device=device, dtype=torch.int32) for c in cached_per_seq]
    )
    mla.forward_context_dsa(
        q=q_full[chunk1_idx].clone(),
        compressed_kv=compressed_kv_full[chunk1_idx].clone(),
        k_pe=k_pe_full[chunk1_idx].clone(),
        attn_metadata=attn_metadata_c1,
        output=output_c1,
        latent_cache=latent_cache_full[chunk1_idx].clone(),
        topk_indices=None,
        position_ids=position_ids_c1,
    )

    # ---- Chunk 2: cached KV → forward_context_with_cached_kv ----
    output_c2 = torch.empty(total_chunk2, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_c2 = AttentionCls.Metadata(
        seq_lens=torch.tensor(new_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=new_per_seq,
        max_num_tokens=total_chunk2,
        kv_cache_manager=kv_cache_chunked,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_per_seq,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
        enable_context_mla_with_cached_kv=True,
    )
    attn_metadata_c2.prepare()
    assert attn_metadata_c2.num_ctx_cached_tokens == sum(cached_per_seq)

    position_ids_c2 = torch.cat(
        [
            torch.arange(c, c + n, device=device, dtype=torch.int32)
            for c, n in zip(cached_per_seq, new_per_seq)
        ]
    )
    mla.forward_context_dsa(
        q=q_full[chunk2_idx].clone(),
        compressed_kv=compressed_kv_full[chunk2_idx].clone(),
        k_pe=k_pe_full[chunk2_idx].clone(),
        attn_metadata=attn_metadata_c2,
        output=output_c2,
        latent_cache=latent_cache_full[chunk2_idx].clone(),
        topk_indices=None,
        position_ids=position_ids_c2,
    )

    assert torch.isfinite(output_c2).all(), "Chunk 2 output has non-finite values"
    abs_error = (output_c2 - ref_chunk2_output).abs()
    torch.testing.assert_close(output_c2, ref_chunk2_output, rtol=0.08, atol=0.08)
    print(
        f"[cached_kv_multi] PASSED: max_error={abs_error.max():.4f}, "
        f"mean_error={abs_error.mean():.6f}"
    )

    kv_cache_ref.shutdown()
    kv_cache_chunked.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90 (Hopper) or later")
@pytest.mark.parametrize(
    "spec_name,chunk_specs",
    CHUNKED_CONTEXT_SPECS,
    ids=[s[0] for s in CHUNKED_CONTEXT_SPECS],
)
def test_chunked_context_correctness(spec_name: str, chunk_specs: List[Tuple[int, int]]):
    """Verify that chunked context produces the same output as full prefill.

    For each sequence, (C, N) means C tokens are processed in chunk 1
    (populating the KV cache via short-seq MHA) and N tokens in chunk 2
    (with C cached tokens, routed to forward_context_with_cached_kv via
    forward_context).  The reference is a single pass of all C+N tokens
    via the short-seq MHA path.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()

    cached_per_seq = [c for c, _n in chunk_specs]
    new_per_seq = [n for _c, n in chunk_specs]
    total_per_seq = [c + n for c, n in chunk_specs]
    num_seqs = len(chunk_specs)
    total_tokens = sum(total_per_seq)
    total_chunk1 = sum(cached_per_seq)
    total_chunk2 = sum(new_per_seq)

    # Threshold high enough for both full-prefill and chunk-1 passes to use
    # the short-seq MHA path.
    threshold = total_tokens + 100

    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)
    AttentionCls = get_attention_backend("TRTLLM", sparse_config)

    # ---- Generate full-sequence input data ----
    q_full = torch.randn(total_tokens, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv_full = torch.randn(total_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe_full = torch.randn(total_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    latent_cache_full = torch.cat([compressed_kv_full, k_pe_full], dim=-1)
    position_ids_full = torch.cat(
        [torch.arange(t, device=device, dtype=torch.int32) for t in total_per_seq]
    )

    # ---- Build index arrays for splitting packed data into chunks ----
    chunk1_indices = []
    chunk2_indices = []
    offset = 0
    for c, n in chunk_specs:
        chunk1_indices.extend(range(offset, offset + c))
        chunk2_indices.extend(range(offset + c, offset + c + n))
        offset += c + n
    chunk1_idx = torch.tensor(chunk1_indices, dtype=torch.long, device=device)
    chunk2_idx = torch.tensor(chunk2_indices, dtype=torch.long, device=device)

    # ---- Reference pass: full prefill via short-seq MHA ----
    kv_cache_ref = _build_kv_cache_manager(
        mapping, sparse_config, model_config, total_per_seq, device
    )
    output_ref = torch.empty(total_tokens, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_ref = AttentionCls.Metadata(
        seq_lens=torch.tensor(total_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=total_per_seq,
        max_num_tokens=total_tokens,
        kv_cache_manager=kv_cache_ref,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata_ref.prepare()
    assert total_tokens <= threshold, "Reference must use short-seq MHA"

    mla.forward_context_dsa(
        q=q_full.clone(),
        compressed_kv=compressed_kv_full.clone(),
        k_pe=k_pe_full.clone(),
        attn_metadata=attn_metadata_ref,
        output=output_ref,
        latent_cache=latent_cache_full.clone(),
        topk_indices=None,
        position_ids=position_ids_full.clone(),
    )
    assert torch.isfinite(output_ref).all(), "Reference output has non-finite values"
    ref_chunk2_output = output_ref[chunk2_idx]

    # ---- Chunked pass: chunk 1 (pure prefill, populates KV cache) ----
    kv_cache_chunked = _build_kv_cache_manager(
        mapping, sparse_config, model_config, total_per_seq, device
    )

    output_chunk1 = torch.empty(total_chunk1, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_c1 = AttentionCls.Metadata(
        seq_lens=torch.tensor(cached_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=cached_per_seq,
        max_num_tokens=total_chunk1,
        kv_cache_manager=kv_cache_chunked,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[0] * num_seqs,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    attn_metadata_c1.prepare()
    assert total_chunk1 <= threshold, "Chunk 1 must use short-seq MHA"

    position_ids_c1 = torch.cat(
        [torch.arange(c, device=device, dtype=torch.int32) for c in cached_per_seq]
    )
    mla.forward_context_dsa(
        q=q_full[chunk1_idx].clone(),
        compressed_kv=compressed_kv_full[chunk1_idx].clone(),
        k_pe=k_pe_full[chunk1_idx].clone(),
        attn_metadata=attn_metadata_c1,
        output=output_chunk1,
        latent_cache=latent_cache_full[chunk1_idx].clone(),
        topk_indices=None,
        position_ids=position_ids_c1,
    )

    # ---- Chunked pass: chunk 2 (cached tokens → MHA via forward_context) ----
    output_chunk2 = torch.empty(total_chunk2, NUM_HEADS * V_HEAD_DIM, dtype=dtype, device=device)
    attn_metadata_c2 = AttentionCls.Metadata(
        seq_lens=torch.tensor(new_per_seq, dtype=torch.int),
        request_ids=list(range(num_seqs)),
        max_num_requests=num_seqs,
        num_contexts=num_seqs,
        prompt_lens=new_per_seq,
        max_num_tokens=total_chunk2,
        kv_cache_manager=kv_cache_chunked,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_per_seq,
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
        enable_context_mla_with_cached_kv=True,
    )
    attn_metadata_c2.prepare()
    assert attn_metadata_c2.num_ctx_cached_tokens == sum(cached_per_seq)

    position_ids_c2 = torch.cat(
        [torch.arange(c, c + n, device=device, dtype=torch.int32) for c, n in chunk_specs]
    )

    # MHA path via forward_context does not need topk_indices.
    mla.forward_context_dsa(
        q=q_full[chunk2_idx].clone(),
        compressed_kv=compressed_kv_full[chunk2_idx].clone(),
        k_pe=k_pe_full[chunk2_idx].clone(),
        attn_metadata=attn_metadata_c2,
        output=output_chunk2,
        latent_cache=latent_cache_full[chunk2_idx].clone(),
        topk_indices=None,
        position_ids=position_ids_c2,
    )

    # ---- Compare chunk 2 output vs reference ----
    assert torch.isfinite(output_chunk2).all(), "Chunk 2 output has non-finite values"
    abs_error = (output_chunk2 - ref_chunk2_output).abs()
    torch.testing.assert_close(output_chunk2, ref_chunk2_output, rtol=0.08, atol=0.08)
    print(
        f"[chunked_{spec_name}] PASSED: max_error={abs_error.max():.4f}, "
        f"mean_error={abs_error.mean():.6f}"
    )

    kv_cache_ref.shutdown()
    kv_cache_chunked.shutdown()


if __name__ == "__main__":
    for batch_name, seq_lens in BATCH_SPECS:
        test_forward_context_short_mha(batch_name, seq_lens)
    test_short_mha_not_triggered_when_threshold_zero()
    test_short_mha_boundary_threshold_equals_max_seq()
    test_short_mha_not_triggered_when_seq_exceeds_threshold()
    test_short_mha_agrees_with_absorption_path()
    test_short_mha_cached_kv_correctness()
    test_short_mha_cached_kv_multi_seq_correctness()
    for spec_name, chunk_specs in CHUNKED_CONTEXT_SPECS:
        test_chunked_context_correctness(spec_name, chunk_specs)
    print("All tests passed!")
