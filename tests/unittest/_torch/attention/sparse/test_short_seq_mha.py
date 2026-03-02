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
A/B comparison vs absorption path, and chunked context with cached KV.
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
NN_INIT_STD = 0.02

# "boundary_exact" tests the dispatch guard boundary (threshold == total_tokens).
BATCH_SPECS = [
    ("single_short", [16]),
    ("single_medium", [64]),
    ("multi_short", [8, 12, 16]),
    ("multi_varied", [4, 32, 10]),
    ("single_one_token", [1]),
    ("boundary_exact", [24, 16]),
]

# (name, [(cached_tokens, new_tokens), ...])
CHUNKED_CONTEXT_SPECS = [
    ("single_small", [(16, 8)]),
    ("single_medium", [(64, 32)]),
    ("large_cache_small_new", [(128, 4)]),
    ("single_one_new_token", [(64, 1)]),
    ("multi_seq", [(64, 32), (32, 16)]),
    ("multi_varied", [(128, 8), (16, 64)]),
    ("three_seqs", [(32, 16), (16, 8), (48, 4)]),
]


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


def _init_mla_weights(mla):
    """Initialize MLA weights deterministically in the loaded layout.

    The loaded layout (as produced by modeling_deepseekv3.py) is:
    [all_heads_k_nope, all_heads_v] along the output dimension.
    """
    with torch.no_grad():
        dev = mla.kv_b_proj.weight.device
        dt = mla.kv_b_proj.weight.dtype

        k_nope_weight = torch.empty(NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK, dtype=dt, device=dev)
        k_nope_weight.normal_(mean=0.0, std=NN_INIT_STD)

        v_weight = torch.empty(NUM_HEADS, V_HEAD_DIM, KV_LORA_RANK, dtype=dt, device=dev)
        v_weight.normal_(mean=0.0, std=NN_INIT_STD)

        mla.kv_b_proj.weight.data = torch.cat(
            [
                k_nope_weight.reshape(NUM_HEADS * QK_NOPE_HEAD_DIM, KV_LORA_RANK),
                v_weight.reshape(NUM_HEADS * V_HEAD_DIM, KV_LORA_RANK),
            ],
            dim=0,
        )
        mla.k_b_proj_trans.data = k_nope_weight.transpose(1, 2).contiguous()
        mla.v_b_proj.data = v_weight.contiguous()

        mla.mqa.indexer.wq_b.weight.normal_(mean=0.0, std=NN_INIT_STD)
        mla.mqa.indexer.wk.weight.normal_(mean=0.0, std=NN_INIT_STD)
        mla.mqa.indexer.weights_proj.weight.normal_(mean=0.0, std=NN_INIT_STD)


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
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    for req_idx, seq_len in enumerate(seq_lens):
        kv_cache_manager.add_dummy_requests(
            request_ids=[req_idx],
            token_nums=[seq_len],
            is_gen=False,
            prepare_resource=True,
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
    enable_cached_kv=False,
):
    """Build and prepare attention metadata."""
    num_ctx = len(seq_lens)
    kwargs = {"enable_context_mla_with_cached_kv": True} if enable_cached_kv else {}
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
        sparse_attention_config=sparse_config,
        **kwargs,
    )
    metadata.prepare()
    return metadata


def _run_forward(
    mla, q, compressed_kv, k_pe, latent_cache, position_ids, metadata, topk_indices=None
):
    """Run forward_context_dsa on cloned inputs and return the output tensor."""
    output = torch.empty(q.shape[0], NUM_HEADS * V_HEAD_DIM, dtype=q.dtype, device=q.device)
    mla.forward_context_dsa(
        q=q.clone(),
        compressed_kv=compressed_kv.clone(),
        k_pe=k_pe.clone(),
        attn_metadata=metadata,
        output=output,
        latent_cache=latent_cache.clone(),
        topk_indices=topk_indices,
        position_ids=position_ids.clone(),
    )
    return output


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@pytest.mark.parametrize("batch_name,seq_lens", BATCH_SPECS, ids=[b[0] for b in BATCH_SPECS])
def test_forward_context_short_mha(batch_name: str, seq_lens: List[int]):
    """Short-seq MHA output vs standalone reference (kv_b_proj + RoPE + SDPA)."""
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    # "boundary_exact" tests threshold == total; others use threshold > total.
    threshold = sum(seq_lens) if batch_name == "boundary_exact" else sum(seq_lens) + 100
    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)

    kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device)
    attn_cls = get_attention_backend("TRTLLM", sparse_config)
    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(seq_lens, device)
    metadata = _make_metadata(attn_cls, seq_lens, kv_mgr, mapping, sparse_config)

    output = _run_forward(mla, q, compressed_kv, k_pe, latent_cache, position_ids, metadata)

    rope_cos_sin = _make_rope_cos_sin(rope_config, device)
    softmax_scale = _compute_softmax_scale(rope_config)
    ref = _reference_attention(
        q,
        compressed_kv,
        k_pe,
        mla.kv_b_proj.weight.data,
        rope_cos_sin,
        seq_lens,
        softmax_scale,
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
def test_standard_path_when_exceeds_threshold():
    """When total tokens > threshold, the standard absorption path is used."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    seq_lens = [32]
    threshold = 16
    mla, mapping, sparse_config, model_config = _build_mla(rope_config, device, threshold)
    _init_mla_weights(mla)

    kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device)
    attn_cls = get_attention_backend("TRTLLM", sparse_config)
    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(seq_lens, device)

    total_tokens = sum(seq_lens)
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE, dtype=dtype, device=device)
    qr = torch.randn(total_tokens, Q_LORA_RANK, dtype=dtype, device=device)

    metadata = _make_metadata(attn_cls, seq_lens, kv_mgr, mapping, sparse_config)
    topk_indices = mla.mqa.indexer(
        qr,
        hidden_states,
        metadata,
        position_ids,
        indexer_k=mla.mqa.indexer.wk(hidden_states),
    )

    output = _run_forward(
        mla, q, compressed_kv, k_pe, latent_cache, position_ids, metadata, topk_indices
    )
    assert torch.isfinite(output).all()
    kv_mgr.shutdown()


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
def test_agrees_with_absorption_path():
    """Short MHA and absorption paths produce numerically close results."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    rope_config = _make_rope_config()
    seq_lens = [16]
    total_tokens = sum(seq_lens)

    mla_short, mapping, sparse_config, model_config = _build_mla(
        rope_config, device, threshold=total_tokens + 100
    )
    mla_absorb, _, _, _ = _build_mla(rope_config, device, threshold=0)

    _init_mla_weights(mla_short)
    with torch.no_grad():
        for (_, ps), (_, pa) in zip(mla_short.named_parameters(), mla_absorb.named_parameters()):
            pa.data.copy_(ps.data)
        for (_, ps), (_, pa) in zip(
            mla_short.mqa.indexer.named_parameters(),
            mla_absorb.mqa.indexer.named_parameters(),
        ):
            pa.data.copy_(ps.data)

    q, compressed_kv, k_pe, latent_cache, position_ids = _make_inputs(seq_lens, device)
    hidden_states = torch.randn(total_tokens, HIDDEN_SIZE, dtype=dtype, device=device)
    qr = torch.randn(total_tokens, Q_LORA_RANK, dtype=dtype, device=device)
    attn_cls = get_attention_backend("TRTLLM", sparse_config)

    def _run(mla_module):
        kv_mgr = _build_kv_cache_manager(mapping, sparse_config, model_config, seq_lens, device)
        meta = _make_metadata(attn_cls, seq_lens, kv_mgr, mapping, sparse_config)
        use_short = (
            mla_module.short_seq_mha_threshold > 0
            and total_tokens <= mla_module.short_seq_mha_threshold
        )
        topk = None
        if not use_short:
            topk = mla_module.mqa.indexer(
                qr.clone(),
                hidden_states.clone(),
                meta,
                position_ids.clone(),
                indexer_k=mla_module.mqa.indexer.wk(hidden_states.clone()),
            )
        out = _run_forward(
            mla_module, q, compressed_kv, k_pe, latent_cache, position_ids, meta, topk
        )
        kv_mgr.shutdown()
        return out

    out_short = _run(mla_short)
    out_absorb = _run(mla_absorb)
    torch.testing.assert_close(out_short, out_absorb, rtol=0.08, atol=0.08)


@pytest.mark.skipif(get_sm_version() < 90, reason="MLA requires SM90+")
@pytest.mark.parametrize(
    "spec_name,chunk_specs",
    CHUNKED_CONTEXT_SPECS,
    ids=[s[0] for s in CHUNKED_CONTEXT_SPECS],
)
def test_chunked_context_correctness(spec_name: str, chunk_specs: List[Tuple[int, int]]):
    """Chunked context (chunk1 prefill + chunk2 cached KV) matches single-pass prefill."""
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
    attn_cls = get_attention_backend("TRTLLM", sparse_config)

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

    # Chunk 2: cached KV path.
    meta_c2 = _make_metadata(
        attn_cls,
        new_per_seq,
        kv_chunked,
        mapping,
        sparse_config,
        cached_per_seq=cached_per_seq,
        enable_cached_kv=True,
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


if __name__ == "__main__":
    for batch_name, seq_lens in BATCH_SPECS:
        test_forward_context_short_mha(batch_name, seq_lens)
    test_threshold_zero_disables_mha()
    test_standard_path_when_exceeds_threshold()
    test_agrees_with_absorption_path()
    for spec_name, chunk_specs in CHUNKED_CONTEXT_SPECS:
        test_chunked_context_correctness(spec_name, chunk_specs)
    print("All tests passed!")
