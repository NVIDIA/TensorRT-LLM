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
"""Test short-sequence MHA optimization path in MLA.

Covers: pure prefill correctness (vs reference), threshold boundary condition,
threshold=0 disables MHA, above-threshold fallback to standard path,
A/B comparison vs absorption path, chunked context with cached KV, and the
indexer K cache a short-MHA prefill leaves for later decode steps.
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
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManager
from tensorrt_llm._torch.attention.backends.sparse.dsa.module import (
    _forward_dsa_attn,
    forward_sparse_attn,
    should_use_short_mha,
)
from tensorrt_llm._torch.attention.backends.utils import get_attention_backend
from tensorrt_llm._torch.attention.mla import MLA
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._utils import get_sm_version, str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType, RopeEmbeddingUtils
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

# DSACacheManager creates background ThreadPoolExecutor threads.
pytestmark = pytest.mark.threadleak(enabled=False)


# ---------------------------------------------------------------------------
# Model constants (DeepSeek V3-like)
# ---------------------------------------------------------------------------
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
WEIGHT_INIT_STD = 0.02

# ---------------------------------------------------------------------------
# Parametrized test specs
# ---------------------------------------------------------------------------
# (name, seq_lens, threshold_offset): threshold = sum(seq_lens) + offset.
# offset=0 tests the boundary condition (threshold == total tokens).
PREFILL_SPECS = [
    ("single", [16], 100),
    ("multi_boundary", [8, 12, 16], 0),
]

# (name, [(cached_tokens, new_tokens), ...], chunk_size or None)
# chunk_size=None -> cached-KV path; chunk_size=int -> chunked-prefill path.
CHUNKED_SPECS = [
    ("cached_kv", [(64, 32), (32, 16)], None),
    ("chunked_prefill", [(64, 16), (64, 8)], 32),
]

# Short-MHA prefills longer than TOPK_TOKENS (and not block-aligned): the
# following decode step keeps TOPK_TOKENS of more keys, so its TopK depends on
# every prefill key in the indexer K cache.
DECODE_PREFILL_LENS = [TOPK_TOKENS + 4 * TOKENS_PER_BLOCK + 35, TOPK_TOKENS + 69]


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_embedding(x, cos_sin):
    original_dtype = x.dtype
    cos, sin = cos_sin.chunk(2, dim=-2)
    cos = cos.squeeze(1)
    sin = sin.squeeze(1)
    x_interleaved = x.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
    cos_expanded = cos.view(cos.shape[0], *([1] * (x.ndim - 2)), cos.shape[-1])
    sin_expanded = sin.view(sin.shape[0], *([1] * (x.ndim - 2)), sin.shape[-1])
    x_rotated = (x_interleaved * cos_expanded) + (_rotate_half(x_interleaved) * sin_expanded)
    return x_rotated.to(original_dtype)


def _reference_attention(
    q, compressed_kv, k_pe, kv_b_proj_weight, rope_cos_sin, seq_lens, softmax_scale
):
    """kv_b_proj expansion + RoPE + causal SDPA reference."""
    results = []
    offset = 0
    for slen in seq_lens:
        qs = q[offset : offset + slen].view(-1, NUM_HEADS, QK_HEAD_DIM)
        cs = rope_cos_sin[:slen]

        q_nope = qs[..., :QK_NOPE_HEAD_DIM]
        q_pe = _apply_rotary_embedding(qs[..., QK_NOPE_HEAD_DIM:], cs)
        q_full = torch.cat([q_nope, q_pe], dim=-1)

        k_pe_roped = _apply_rotary_embedding(k_pe[offset : offset + slen], cs)
        kv = torch.nn.functional.linear(compressed_kv[offset : offset + slen], kv_b_proj_weight)
        k_nope, v = kv.split([NUM_HEADS * QK_NOPE_HEAD_DIM, NUM_HEADS * V_HEAD_DIM], -1)
        k_full = torch.cat(
            [
                k_nope.view(-1, NUM_HEADS, QK_NOPE_HEAD_DIM),
                k_pe_roped.view(-1, 1, QK_ROPE_HEAD_DIM).expand(-1, NUM_HEADS, -1),
            ],
            dim=-1,
        )
        v_r = v.view(-1, NUM_HEADS, V_HEAD_DIM)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q_full.unsqueeze(0).transpose(1, 2),
            k_full.unsqueeze(0).transpose(1, 2),
            v_r.unsqueeze(0).transpose(1, 2),
            is_causal=True,
            scale=softmax_scale,
        )
        results.append(attn_out.transpose(1, 2).squeeze(0).reshape(slen, NUM_HEADS * V_HEAD_DIM))
        offset += slen
    return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
@dataclass
class RopeConfig:
    hidden_size: int
    num_attention_heads: int
    rope_scaling: dict
    max_position_embeddings: int
    rope_theta: float
    qk_rope_head_dim: int
    model_type: str


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


def _build_mla(rope_config, device, threshold, indexer_k_dtype="fp8", index_rope_dim=None):
    """Build an MLA module with DSA config and the given threshold."""
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=TOPK_TOKENS,
        indexer_k_dtype=indexer_k_dtype,
    )
    pretrained_config = SimpleNamespace(rms_norm_eps=1e-6)
    if index_rope_dim is not None:
        pretrained_config.index_rope_dim = index_rope_dim
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
        if old_val is None:
            os.environ.pop("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", None)
        else:
            os.environ["TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD"] = old_val

    # mla.mqa (DSATrtllmAttention) is not an nn.Module, so mla.to(device)
    # does not move its children. Explicitly move the indexer.
    if hasattr(mla, "mqa") and hasattr(mla.mqa, "indexer"):
        mla.mqa.indexer.to(device)

    return mla, mapping, sparse_config, model_config


def _attention_cls_for_sparse_config(sparse_config, model_config):
    sparse_params = sparse_config.to_sparse_params(pretrained_config=model_config.pretrained_config)
    return get_attention_backend("TRTLLM", sparse_params=sparse_params)


def _init_mla_weights(mla):
    """Initialize MLA weights deterministically in the loaded layout.

    The loaded layout (as produced by modeling_deepseekv3.py) is:
    [all_heads_k_nope, all_heads_v] along the output dimension.
    """
    with torch.no_grad():
        dev = mla.kv_b_proj.weight.device
        dt = mla.kv_b_proj.weight.dtype

        k_nope_weight = torch.empty(NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK, dtype=dt, device=dev)
        k_nope_weight.normal_(mean=0.0, std=WEIGHT_INIT_STD)

        v_weight = torch.empty(NUM_HEADS, V_HEAD_DIM, KV_LORA_RANK, dtype=dt, device=dev)
        v_weight.normal_(mean=0.0, std=WEIGHT_INIT_STD)

        mla.kv_b_proj.weight.data = torch.cat(
            [
                k_nope_weight.reshape(NUM_HEADS * QK_NOPE_HEAD_DIM, KV_LORA_RANK),
                v_weight.reshape(NUM_HEADS * V_HEAD_DIM, KV_LORA_RANK),
            ],
            dim=0,
        )
        mla.k_b_proj_trans.data = k_nope_weight.transpose(1, 2).contiguous()
        mla.v_b_proj.data = v_weight.contiguous()

        mla.mqa.indexer.wq_b.weight.normal_(mean=0.0, std=WEIGHT_INIT_STD)
        mla.mqa.indexer.wk.weight.normal_(mean=0.0, std=WEIGHT_INIT_STD)
        mla.mqa.indexer.weights_proj.weight.normal_(mean=0.0, std=WEIGHT_INIT_STD)
        # Build fused wk+weights_proj weight after random init
        mla.mqa.indexer.post_load_weights()


def _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device):
    """Build a DSACacheManager for the given batch."""
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(max_tokens=16384, enable_block_reuse=False),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=NUM_LAYERS,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=max(seq_lens),
        max_batch_size=len(seq_lens),
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.bfloat16)),
        sparse_attention_config=sparse_config,
        model_config=model_config,
    )
    for req_idx, seq_len in enumerate(seq_lens):
        kv_cache_manager.add_dummy_requests(
            request_ids=[req_idx],
            token_nums=[seq_len],
            is_gen=False,
        )
    return kv_cache_manager


def _make_inputs(seq_lens, device, dtype=torch.bfloat16):
    """Generate random MLA inputs for the given sequence lengths."""
    total = sum(seq_lens)
    q = torch.randn(total, NUM_HEADS * QK_HEAD_DIM, dtype=dtype, device=device)
    compressed_kv = torch.randn(total, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(total, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    position_ids = torch.cat([torch.arange(s, device=device, dtype=torch.int32) for s in seq_lens])
    return q, compressed_kv, k_pe, latent_cache, position_ids


def _make_metadata(
    attn_cls,
    seq_lens,
    kv_cache_manager,
    mapping,
    sparse_config,
    cached_per_seq=None,
    runtime_features=None,
):
    """Build and prepare attention metadata.

    When cached_per_seq is provided, enables the cached-KV context path.
    """
    sparse_metadata_params = sparse_config.to_sparse_metadata_params()
    num_ctx = len(seq_lens)
    kwargs = {}
    if cached_per_seq is not None:
        kwargs["enable_context_mla_with_cached_kv"] = True
    if runtime_features is not None:
        kwargs["runtime_features"] = runtime_features
    metadata = attn_cls.Metadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        request_ids=list(range(num_ctx)),
        max_num_requests=num_ctx,
        num_contexts=num_ctx,
        prompt_lens=seq_lens,
        max_num_tokens=sum(seq_lens),
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_per_seq or [0] * num_ctx,
        ),
        mapping=mapping,
        sparse_metadata_params=sparse_metadata_params,
        **kwargs,
    )
    metadata.prepare()
    return metadata


@torch.no_grad()
def _short_mha_indexer_inputs(mla, position_ids):
    """The [k_fp8, k_scale] Op 1 hands a short-MHA batch, from random hidden states.

    No grad: some callers run outside inference mode, and the indexer rotates
    views of its keys in place.
    """
    num_tokens = position_ids.shape[-1]
    device = position_ids.device
    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    qr = torch.randn(num_tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    return list(mla.mqa.indexer.pre_indexer_proj(qr, hidden_states, position_ids)[1:3])


def _run_forward(
    mla, q, compressed_kv, k_pe, latent_cache, position_ids, metadata, indexer_inputs=None
):
    """Run the production DSA attention dispatcher on cloned inputs.

    Without indexer_inputs the batch is a short-MHA one, which still writes
    its tokens' indexer keys.
    """
    if indexer_inputs is None:
        indexer_inputs = _short_mha_indexer_inputs(mla, position_ids)
    output = torch.empty(q.shape[0], NUM_HEADS * V_HEAD_DIM, dtype=q.dtype, device=q.device)
    _forward_dsa_attn(
        mla,
        q=q.clone(),
        compressed_kv=compressed_kv.clone(),
        k_pe=k_pe.clone(),
        latent_cache=latent_cache.clone(),
        indexer_intermediates=indexer_inputs,
        position_ids=position_ids.clone(),
        attn_metadata=metadata,
        output=output,
    )
    return output


def _init_proj_weights(mla):
    """Initialize the projections forward_dsa_proj runs, then the attention weights."""
    with torch.no_grad():
        mla.kv_a_proj_with_mqa.weight.normal_(mean=0.0, std=WEIGHT_INIT_STD)
        mla.q_b_proj.weight.normal_(mean=0.0, std=WEIGHT_INIT_STD)
    _init_mla_weights(mla)


def _copy_mla_weights(src, dst):
    """Give dst (built like src by _build_mla) the weights of src."""
    with torch.no_grad():
        for (_, ps), (_, pd) in zip(src.named_parameters(), dst.named_parameters()):
            pd.data.copy_(ps.data)
        for (_, ps), (_, pd) in zip(
            src.mqa.indexer.named_parameters(), dst.mqa.indexer.named_parameters()
        ):
            pd.data.copy_(ps.data)
    # Build fused wk+weights_proj weight after copying params
    dst.mqa.indexer.post_load_weights()


def _build_short_and_standard_mla(device, threshold):
    """A short-MHA module (threshold) and a standard one (threshold 0) with equal weights."""
    rope_config = _make_rope_config()
    mla_short, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    mla_std, _, _, _ = _build_mla(rope_config, device, threshold=0)
    _init_proj_weights(mla_short)
    _copy_mla_weights(mla_short, mla_std)
    return mla_short, mla_std, mapping, sparse_config, model_config


def _build_zeroed_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device):
    """_build_kv_cache_manager with a zeroed indexer K cache: unwritten keys read as zero."""
    kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device)
    kv_mgr.get_indexer_k_cache_buffers(LAYER_IDX).zero_()
    return kv_mgr


def _make_gen_metadata(attn_cls, cached_per_seq, kv_cache_manager, mapping, sparse_config):
    """Build and prepare metadata for one decode token per sequence."""
    num_gen = len(cached_per_seq)
    metadata = attn_cls.Metadata(
        seq_lens=torch.ones(num_gen, dtype=torch.int),
        request_ids=list(range(num_gen)),
        max_num_requests=num_gen,
        num_contexts=0,
        prompt_lens=cached_per_seq,
        max_num_tokens=num_gen,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=cached_per_seq,
        ),
        mapping=mapping,
        sparse_metadata_params=sparse_config.to_sparse_metadata_params(),
    )
    metadata.prepare()
    return metadata


def _run_dsa_forward(mla, hidden_states, position_ids, metadata):
    """Run the whole DSA forward: Op 1 projections from hidden states, then Op 2 attention."""
    output = torch.empty(
        hidden_states.shape[0],
        NUM_HEADS * V_HEAD_DIM,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    forward_sparse_attn(mla, position_ids, hidden_states, metadata, [output])
    return output


def _indexer_k_cache_rows(kv_mgr, metadata, num_tokens, data_bytes):
    """Indexer K cache bytes (data, scale) of the batch's first num_tokens new tokens."""
    flat = kv_mgr.get_indexer_k_cache_buffers(LAYER_IDX).reshape(-1).view(torch.uint8)
    data_offsets = torch.arange(data_bytes, device=flat.device)
    scale_offsets = torch.arange(4, device=flat.device)
    data = flat[metadata.slot_mapping_fp8[:num_tokens, None] + data_offsets]
    scale = flat[metadata.slot_mapping_scale[:num_tokens, None] + scale_offsets]
    return data, scale


def _record_decode_topk(indexer):
    """Keep a copy of every generation-phase TopK the indexer predicts."""
    records = []
    predict = indexer.forward_from_projected

    def forward_from_projected(*args, is_generation=None, **kwargs):
        topk = predict(*args, is_generation=is_generation, **kwargs)
        if is_generation:
            records.append(topk.clone())
        return topk

    indexer.forward_from_projected = forward_from_projected
    return records


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@pytest.mark.parametrize(
    "name,seq_lens,threshold_offset",
    PREFILL_SPECS,
    ids=[s[0] for s in PREFILL_SPECS],
)
def test_forward_context_short_mha(name: str, seq_lens: List[int], threshold_offset: int):
    """Short-seq MHA output vs standalone reference (kv_b_proj + RoPE + SDPA)."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    threshold = sum(seq_lens) + threshold_offset
    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)

    kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device)
    attn_cls = _attention_cls_for_sparse_config(sparse_config, model_config)
    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(seq_lens, device)
    metadata = _make_metadata(attn_cls, seq_lens, kv_mgr, mapping, sparse_config)

    output = _run_forward(mla, q, compressed_kv, k_pe, latent_cache, position_ids, metadata)

    rope_cos_sin = _make_rope_cos_sin(rope_config, device)
    softmax_scale = _compute_softmax_scale(rope_config)
    ref = _reference_attention(
        q, compressed_kv, k_pe, mla.kv_b_proj.weight.data, rope_cos_sin, seq_lens, softmax_scale
    )

    torch.testing.assert_close(output, ref, rtol=0.05, atol=0.05)
    kv_mgr.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
def test_threshold_zero_disables_mha():
    """With threshold=0, the short MHA path is NOT active."""
    device = torch.device("cuda")
    rope_config = _make_rope_config()
    mla, _, _, _ = _build_mla(rope_config, device, threshold=0)

    assert mla.short_seq_mha_threshold == 0
    assert mla.mha is None


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@torch.inference_mode()
def test_standard_path_when_exceeds_threshold():
    """When total tokens > threshold, the standard absorption path is used."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    seq_lens = [32]
    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold=16)
    _init_mla_weights(mla)

    kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device)
    attn_cls = _attention_cls_for_sparse_config(sparse_config, model_config)
    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(seq_lens, device)

    total_tokens = sum(seq_lens)
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    qr = torch.randn(total_tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)

    metadata = _make_metadata(attn_cls, seq_lens, kv_mgr, mapping, sparse_config)
    inputs = list(mla.mqa.indexer.pre_indexer_proj(qr, hidden_states, position_ids))

    output = _run_forward(mla, q, compressed_kv, k_pe, latent_cache, position_ids, metadata, inputs)
    assert torch.isfinite(output).all()
    kv_mgr.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@torch.inference_mode()
def test_agrees_with_absorption_path():
    """Short MHA and absorption paths produce numerically close results."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    seq_lens = [16]
    total_tokens = sum(seq_lens)

    mla_short, mapping, sparse_config, model_config = _build_mla(
        rope_config, device, threshold=total_tokens + 100
    )
    mla_absorb, _, _, _ = _build_mla(rope_config, device, threshold=0)

    # Share identical weights between the two modules.
    _init_mla_weights(mla_short)
    _copy_mla_weights(mla_short, mla_absorb)

    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(seq_lens, device)
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    qr = torch.randn(total_tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    attn_cls = _attention_cls_for_sparse_config(sparse_config, model_config)

    def _run(mla_module):
        kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device)
        meta = _make_metadata(attn_cls, seq_lens, kv_mgr, mapping, sparse_config)
        use_short = (
            mla_module.short_seq_mha_threshold > 0
            and total_tokens <= mla_module.short_seq_mha_threshold
        )
        inputs = None
        if not use_short:
            inputs = list(
                mla_module.mqa.indexer.pre_indexer_proj(
                    qr.clone(), hidden_states.clone(), position_ids.clone()
                )
            )
        out = _run_forward(
            mla_module, q, compressed_kv, k_pe, latent_cache, position_ids, meta, inputs
        )
        kv_mgr.shutdown()
        return out

    out_short = _run(mla_short)
    out_absorb = _run(mla_absorb)
    torch.testing.assert_close(out_short, out_absorb, rtol=0.08, atol=0.08)


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@pytest.mark.parametrize(
    "name,chunk_specs,chunk_size",
    CHUNKED_SPECS,
    ids=[s[0] for s in CHUNKED_SPECS],
)
def test_chunked_correctness(name: str, chunk_specs: List[Tuple[int, int]], chunk_size):
    """Chunked context (cached KV or chunked prefill) matches single-pass prefill."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    cached_per_seq = [c for c, _ in chunk_specs]
    new_per_seq = [n for _, n in chunk_specs]
    total_per_seq = [c + n for c, n in chunk_specs]
    threshold = sum(total_per_seq) + 100

    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)
    attn_cls = _attention_cls_for_sparse_config(sparse_config, model_config)

    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(total_per_seq, device)

    # Build chunk index tensors.
    c1_idx, c2_idx = [], []
    offset = 0
    for c, n in chunk_specs:
        c1_idx.extend(range(offset, offset + c))
        c2_idx.extend(range(offset + c, offset + c + n))
        offset += c + n
    c1_idx = torch.tensor(c1_idx, dtype=torch.long, device=device)
    c2_idx = torch.tensor(c2_idx, dtype=torch.long, device=device)

    # Reference: single-pass full prefill.
    kv_ref = _build_kv_cache_manager(mapping, sparse_config, model_config, total_per_seq, device)
    meta_ref = _make_metadata(attn_cls, total_per_seq, kv_ref, mapping, sparse_config)
    out_ref = _run_forward(mla, q, compressed_kv, k_pe, latent_cache, position_ids, meta_ref)

    # Chunk 1: pure prefill (populates KV cache).
    kv_chunked = _build_kv_cache_manager(
        mapping, sparse_config, model_config, total_per_seq, device
    )
    meta_c1 = _make_metadata(attn_cls, cached_per_seq, kv_chunked, mapping, sparse_config)
    pos_c1 = torch.cat([torch.arange(c, device=device, dtype=torch.int32) for c in cached_per_seq])
    _run_forward(
        mla, q[c1_idx], compressed_kv[c1_idx], k_pe[c1_idx], latent_cache[c1_idx], pos_c1, meta_c1
    )

    # Chunk 2: cached KV or chunked prefill path.
    runtime_features = None
    if chunk_size is not None:
        runtime_features = AttentionRuntimeFeatures(
            chunked_prefill=True,
            chunk_size=chunk_size,
            chunked_prefill_buffer_batch_size=1,
        )
    meta_c2 = _make_metadata(
        attn_cls,
        new_per_seq,
        kv_chunked,
        mapping,
        sparse_config,
        cached_per_seq=cached_per_seq,
        runtime_features=runtime_features,
    )
    pos_c2 = torch.cat(
        [torch.arange(c, c + n, device=device, dtype=torch.int32) for c, n in chunk_specs]
    )
    out_c2 = _run_forward(
        mla, q[c2_idx], compressed_kv[c2_idx], k_pe[c2_idx], latent_cache[c2_idx], pos_c2, meta_c2
    )

    torch.testing.assert_close(out_c2, out_ref[c2_idx], rtol=0.08, atol=0.08)

    kv_ref.shutdown()
    kv_chunked.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
def test_chunked_context_rejects_when_kv_exceeds_threshold():
    """When max KV length (cached + new) exceeds threshold, short MHA is not used.

    Regression test: previously, only new token count was checked against
    the threshold.  With cached tokens, the effective attention span can far
    exceed the threshold even when new_tokens is small.
    """
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    # 1 sequence: 64 cached + 32 new = 96 total KV.
    # threshold=80: chunk 1 (64 tokens) passes, chunk 2's max KV (96) exceeds.
    cached_per_seq = [64]
    new_per_seq = [32]
    total_per_seq = [96]
    threshold = 80

    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)
    attn_cls = _attention_cls_for_sparse_config(sparse_config, model_config)

    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(total_per_seq, device)

    # Chunk 1: populate KV cache with cached tokens.
    kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, total_per_seq, device)
    meta_c1 = _make_metadata(attn_cls, cached_per_seq, kv_mgr, mapping, sparse_config)
    c1_idx = torch.arange(cached_per_seq[0], dtype=torch.long, device=device)
    pos_c1 = torch.arange(cached_per_seq[0], device=device, dtype=torch.int32)
    _run_forward(
        mla, q[c1_idx], compressed_kv[c1_idx], k_pe[c1_idx], latent_cache[c1_idx], pos_c1, meta_c1
    )

    # Chunk 2: cached KV path — threshold check matters here.
    meta_c2 = _make_metadata(
        attn_cls, new_per_seq, kv_mgr, mapping, sparse_config, cached_per_seq=cached_per_seq
    )
    pos_c2 = torch.arange(cached_per_seq[0], total_per_seq[0], device=device, dtype=torch.int32)

    # max_ctx_kv_len (96) > threshold (80) -> short MHA should NOT be used.
    assert not should_use_short_mha(mla, meta_c2, pos_c2)

    # With threshold large enough for the full KV -> short MHA IS used.
    mla.short_seq_mha_threshold = total_per_seq[0] + 100
    assert should_use_short_mha(mla, meta_c2, pos_c2)

    kv_mgr.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@pytest.mark.parametrize(
    "name,seq_lens,threshold_offset",
    PREFILL_SPECS,
    ids=[s[0] for s in PREFILL_SPECS],
)
@torch.inference_mode()
def test_short_mha_prefill_writes_indexer_k_cache(
    name: str, seq_lens: List[int], threshold_offset: int
):
    """A short-MHA prefill leaves the indexer K cache the standard prefill leaves.

    Short MHA needs no indexer TopK, but later decode steps read the prefill
    tokens' keys from the indexer K cache.
    """
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_tokens = sum(seq_lens)
    mla_short, mla_std, mapping, sparse_config, model_config = _build_short_and_standard_mla(
        device, threshold=num_tokens + threshold_offset
    )
    attn_cls = _attention_cls_for_sparse_config(sparse_config, model_config)
    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    position_ids = torch.cat([torch.arange(s, device=device, dtype=torch.int32) for s in seq_lens])
    data_bytes = mla_std.mqa.indexer.k_data_bytes_per_token

    kv_mgrs = []
    try:
        cache_rows = []
        for mla in (mla_short, mla_std):
            kv_mgr = _build_zeroed_kv_cache_manager(
                mapping, sparse_config, model_config, seq_lens, device
            )
            kv_mgrs.append(kv_mgr)
            metadata = _make_metadata(attn_cls, seq_lens, kv_mgr, mapping, sparse_config)
            assert should_use_short_mha(mla, metadata, position_ids) == (mla is mla_short)
            _run_dsa_forward(mla, hidden_states, position_ids, metadata)
            cache_rows.append(_indexer_k_cache_rows(kv_mgr, metadata, num_tokens, data_bytes))

        (k_short, scale_short), (k_std, scale_std) = cache_rows
        assert k_std.any(), "the standard prefill wrote no indexer keys"
        torch.testing.assert_close(k_short, k_std, rtol=0, atol=0)
        torch.testing.assert_close(scale_short, scale_std, rtol=0, atol=0)
    finally:
        for kv_mgr in kv_mgrs:
            kv_mgr.shutdown()


@pytest.mark.skipif(get_sm_version() < 100, reason="DSA sparse MLA decode requires SM100+")
@torch.inference_mode()
def test_decode_after_short_mha_prefill():
    """A decode step after a short-MHA prefill matches one after the standard prefill.

    The prefills are longer than index_topk, so the decode indexer drops keys
    and its TopK depends on every prefill key in the indexer K cache.
    """
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    prefill_lens = DECODE_PREFILL_LENS
    num_ctx_tokens = sum(prefill_lens)
    mla_short, mla_std, mapping, sparse_config, model_config = _build_short_and_standard_mla(
        device, threshold=max(prefill_lens)
    )
    attn_cls = _attention_cls_for_sparse_config(sparse_config, model_config)
    ctx_hidden = torch.randn(num_ctx_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    ctx_pos = torch.cat([torch.arange(n, device=device, dtype=torch.int32) for n in prefill_lens])
    gen_hidden = torch.randn(len(prefill_lens), HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    gen_pos = torch.tensor(prefill_lens, dtype=torch.int32, device=device)
    data_bytes = mla_std.mqa.indexer.k_data_bytes_per_token

    kv_mgrs = []
    try:
        results = []
        for mla in (mla_short, mla_std):
            # Blocks for the prefill and the decode token.
            kv_mgr = _build_zeroed_kv_cache_manager(
                mapping, sparse_config, model_config, [n + 1 for n in prefill_lens], device
            )
            kv_mgrs.append(kv_mgr)
            ctx_meta = _make_metadata(attn_cls, prefill_lens, kv_mgr, mapping, sparse_config)
            assert should_use_short_mha(mla, ctx_meta, ctx_pos) == (mla is mla_short)
            _run_dsa_forward(mla, ctx_hidden, ctx_pos, ctx_meta)
            prefill_k = _indexer_k_cache_rows(kv_mgr, ctx_meta, num_ctx_tokens, data_bytes)

            decode_topk = _record_decode_topk(mla.mqa.indexer)
            gen_meta = _make_gen_metadata(attn_cls, prefill_lens, kv_mgr, mapping, sparse_config)
            gen_out = _run_dsa_forward(mla, gen_hidden, gen_pos, gen_meta)
            assert len(decode_topk) == 1
            results.append((prefill_k, decode_topk[0], gen_out))

        (k_short, topk_short, out_short), (k_std, topk_std, out_std) = results
        # Every row keeps TOPK_TOKENS valid keys out of more: the TopK selects.
        assert topk_std.shape == (len(prefill_lens), TOPK_TOKENS)
        assert (topk_std >= 0).all()
        torch.testing.assert_close(
            topk_short.sort(dim=-1).values, topk_std.sort(dim=-1).values, rtol=0, atol=0
        )
        torch.testing.assert_close(out_short, out_std, rtol=1e-2, atol=1e-3)
        for rows_short, rows_std in zip(k_short, k_std):
            torch.testing.assert_close(rows_short, rows_std, rtol=0, atol=0)
    finally:
        for kv_mgr in kv_mgrs:
            kv_mgr.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@pytest.mark.parametrize("packed", [False, True], ids=["fused_gemm", "packed"])
@pytest.mark.parametrize("index_rope_dim", [None, 0], ids=["rope", "position_free"])
@pytest.mark.parametrize("indexer_k_dtype", ["fp8", "bf16", "fp4"])
@torch.inference_mode()
def test_indexer_k_projection_matches_full_projection(indexer_k_dtype, index_rope_dim, packed):
    """pre_indexer_k_proj returns bitwise the K that pre_indexer_proj returns."""
    if indexer_k_dtype == "fp4" and get_sm_version() < 100:
        pytest.skip("The FP4 indexer K requires SM100+")
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    mla, _, _, _ = _build_mla(
        _make_rope_config(),
        device,
        threshold=0,
        indexer_k_dtype=indexer_k_dtype,
        index_rope_dim=index_rope_dim,
    )
    _init_mla_weights(mla)
    indexer = mla.mqa.indexer
    assert (indexer.rotary_emb is None) == (index_rope_dim == 0)

    num_tokens = 37
    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    qr = torch.randn(num_tokens, Q_LORA_RANK, dtype=torch.bfloat16, device=device)
    position_ids = torch.arange(100, 100 + num_tokens, dtype=torch.int32, device=device)
    indexer_k = projected = None
    if packed:
        # Column slices of a packed projection output, as forward_dsa_proj passes them.
        packed_out = torch.randn(
            num_tokens, 32 + indexer.head_dim + indexer.n_heads, dtype=torch.bfloat16, device=device
        )
        _, indexer_k, weights = packed_out.split([32, indexer.head_dim, indexer.n_heads], -1)
        projected = (indexer_k, weights, None)

    _, k_ref, k_scale_ref, _, _ = indexer.pre_indexer_proj(
        qr, hidden_states, position_ids, projected=projected
    )
    k, k_scale = indexer.pre_indexer_k_proj(hidden_states, position_ids, indexer_k=indexer_k)

    assert k.dtype == k_ref.dtype and k.shape == k_ref.shape
    assert k_scale.dtype == k_scale_ref.dtype and k_scale.shape == k_scale_ref.shape
    torch.testing.assert_close(k.view(torch.uint8), k_ref.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(
        k_scale.view(torch.uint8), k_scale_ref.view(torch.uint8), rtol=0, atol=0
    )
