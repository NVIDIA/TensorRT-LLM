# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real MiniMax-M3 attention runtime state at the TP4 per-rank geometry, for the backend-step entries' GPU tests.

One hybrid NVFP4 KV cache manager (``MiniMaxM3KVCacheManagerV2``) laid out as the target's: a dense layer in the FP8
half, a sparse MSA layer in the NVFP4 half with its FP8 index-K cache, and (``draft=True``) the one-model Eagle3
drafter's shared layer, read through the manager's FP8 draft view as the runtime's draft swap reads it. Metadata is
the runtime's own type, filled by its own ``prepare()``; the backends are built the way the target builds them.

The geometry is one rank's: 16 query heads over one KV head and one index head of dim 128. The mapping is a single
rank holding exactly those heads, so nothing is sharded again.

History rows are written by the in-tree scatters (``fused_write_layer_caches`` for the FP8 layers,
``fused_write_layer_caches_nvfp4`` for the NVFP4 layer and its index-K), at the slots the manager assigned, which are
not in request order.
"""

from __future__ import annotations

from typing import Sequence

import torch

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import (
    MiniMaxM3KVCacheManagerV2,
    MiniMaxM3MsaSparseAttention,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
    fused_write_layer_caches,
    fused_write_layer_caches_nvfp4,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend import (
    MiniMaxM3MsaSparseAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import (
    Eagle3DecodingConfig,
    KvCacheConfig,
    MiniMaxM3SparseAttentionConfig,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

HEADS = 16
KV_HEADS = 1
INDEX_HEADS = 1
HEAD_DIM = 128
PAGE = 128
TOPK = 16
DENSE, SPARSE, DRAFT = 0, 1, 2
MAX_BATCH = 8
MAX_SEQ_LEN = 8192
MAX_NUM_TOKENS = 8192
#: The Eagle3 drafter checkpoint's RoPE: NeoX over the whole head, theta 5e6.
DRAFT_ROPE_THETA = 5.0e6

#: E4M3-exact planted index levels, 1/8 apart in relative terms (0.25 .. 15): 48 distinct block scores.
LEVELS = tuple(2.0**e * (1 + m / 8) for e in range(-2, 4) for m in range(8))
#: The index value of a block's non-peak rows; its score (8) is below every level's (>= 16).
LOW = 0.125


def mapping() -> Mapping:
    return Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)


def sparse_attention_config() -> MiniMaxM3SparseAttentionConfig:
    """The target's runtime: MSA, the FP8 indexer, the fused five-way projection (M3's checkpoint selection rule:
    no forced initial blocks, the query's own block forced)."""
    return MiniMaxM3SparseAttentionConfig(
        implementation="msa",
        indexer_kv_dtype="fp8",
        fuse_qkv_index_projection=True,
        sparse_num_index_heads=INDEX_HEADS,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
    )


def make_manager(*, draft: bool) -> MiniMaxM3KVCacheManagerV2:
    torch.cuda.init()
    return MiniMaxM3KVCacheManagerV2(
        KvCacheConfig(
            max_tokens=4 * MAX_SEQ_LEN,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
            host_cache_size=0,
            event_buffer_max_size=0,
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=[KV_HEADS] * (3 if draft else 2),
        head_dim=HEAD_DIM,
        tokens_per_block=PAGE,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH,
        max_num_tokens=MAX_NUM_TOKENS,
        mapping=mapping(),
        dtype=DataType.NVFP4,
        vocab_size=16,
        spec_config=(
            Eagle3DecodingConfig(max_draft_len=3, speculative_model="unused") if draft else None
        ),
        sparse_layer_ids=[SPARSE],
        sparse_attention_config=sparse_attention_config(),
    )


def make_msa_backend(layer: int) -> MiniMaxM3MsaSparseAttention:
    """One layer's MSA backend as ``MiniMaxM3Tp4ModelingV2`` builds it (NVFP4 KV quant config on every layer)."""
    quant = QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)
    backend = create_attention(
        "TRTLLM",
        layer,
        HEADS,
        HEAD_DIM,
        KV_HEADS,
        pos_embd_params=None,
        quant_config=quant,
        q_scaling=1.0,
        sparse_params=sparse_attention_config().to_sparse_params(layer_idx=layer),
    )
    backend.update_quant_config(quant)
    assert isinstance(backend, MiniMaxM3MsaSparseAttention), type(backend).__name__
    return backend


def make_draft_backend() -> TrtllmAttention:
    """The drafter layer's attention as ``M3Eagle3Draft`` builds it: fused NeoX RoPE, an FP8 KV cache (the target
    overrides the drafter's KV dtype to FP8), no other quantization."""
    quant = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    backend = create_attention(
        "TRTLLM",
        DRAFT,
        HEADS,
        HEAD_DIM,
        KV_HEADS,
        pos_embd_params=PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams(dim=HEAD_DIM, theta=DRAFT_ROPE_THETA, max_positions=MAX_SEQ_LEN),
        ),
        quant_config=quant,
        q_scaling=1.0,
        attention_chunk_size=None,
        sparse_params=None,
    )
    backend.update_quant_config(quant)
    assert type(backend) is TrtllmAttention, type(backend).__name__
    return backend


def make_msa_metadata(manager) -> MiniMaxM3MsaSparseAttentionMetadata:
    metadata = MiniMaxM3MsaSparseAttentionMetadata(
        max_num_requests=MAX_BATCH,
        max_num_tokens=MAX_NUM_TOKENS,
        max_num_sequences=MAX_BATCH,
        kv_cache_manager=manager,
        mapping=mapping(),
        sparse_metadata_params=sparse_attention_config().to_sparse_metadata_params(),
    )
    # The target's MSA preparation choice that changes what prepare() stages (the identity tables of short steps).
    metadata.msa_policy = MsaRuntimePolicy(short_sequence_identity=True)
    return metadata


def make_draft_metadata(manager) -> TrtllmAttentionMetadata:
    """Drafter metadata over the shared manager's FP8 draft view, the manager the runtime's draft swap installs."""
    view = manager.get_draft_kv_cache_view()
    assert view is not None and view.dtype == DataType.FP8
    return TrtllmAttentionMetadata(
        max_num_requests=MAX_BATCH,
        max_num_tokens=MAX_NUM_TOKENS,
        kv_cache_manager=view,
        mapping=mapping(),
    )


def allocate(manager, request_ids: Sequence[int], qlens: Sequence[int], cached: Sequence[int]):
    manager.add_dummy_requests(list(request_ids), token_nums=[c + n for c, n in zip(cached, qlens)])


def prepare(metadata, request_ids, qlens, cached, num_contexts: int) -> None:
    metadata.seq_lens = torch.tensor(list(qlens), dtype=torch.int)
    metadata.num_contexts = num_contexts
    metadata.request_ids = list(request_ids)
    metadata.prompt_lens = [
        c + n if row < num_contexts else c for row, (c, n) in enumerate(zip(cached, qlens))
    ]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached))
    metadata.prepare()


def flat_slots(manager, layer: int, request_id: int, positions: torch.Tensor) -> torch.Tensor:
    """The cache slots (page slot id * PAGE + offset) of a request's positions in ``layer``."""
    pages = torch.tensor(
        manager.get_batch_cache_indices([request_id], layer_idx=layer)[0],
        dtype=torch.int64,
        device="cuda",
    )
    positions = positions.to(device="cuda", dtype=torch.int64)
    return (pages[positions // PAGE] * PAGE + positions % PAGE).to(torch.int32)


def write_fp8_rows(manager, layer, request_id, positions, k, v) -> None:
    """K / V rows [n, 128] (E4M3) into an FP8 layer's cache at ``positions``."""
    view = manager.get_buffers(layer, "HND")
    assert view.dtype == torch.float8_e4m3fn
    slots = flat_slots(manager, layer, request_id, positions)
    assert fused_write_layer_caches(view[:, 0], view[:, 1], None, slots, k, v, None)


def write_nvfp4_rows(manager, metadata, layer, request_id, positions, k, v, idx_k, quant) -> None:
    """BF16 K / V rows quantized into the NVFP4 layer's cache, and index-K rows into its FP8 index cache."""
    data = manager.get_buffers(layer, "HND")
    scales = manager.get_block_scale_buffers(layer, "HND")
    slots = flat_slots(manager, layer, request_id, positions)
    assert fused_write_layer_caches_nvfp4(
        data[:, 0],
        data[:, 1],
        scales[:, 0],
        scales[:, 1],
        metadata.msa_idx_k_cache(layer),
        slots,
        k,
        v,
        idx_k,
        quant,
    )


def index_directions(gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Two orthogonal ±1 index directions (the halves of the 128 dims), E4M3-exact."""
    signs = torch.randint(0, 2, (HEAD_DIM,), generator=gen, device="cuda").float() * 2 - 1
    u0 = torch.zeros(HEAD_DIM, device="cuda")
    u1 = torch.zeros(HEAD_DIM, device="cuda")
    u0[: HEAD_DIM // 2] = signs[: HEAD_DIM // 2]
    u1[HEAD_DIM // 2 :] = signs[HEAD_DIM // 2 :]
    return u0, u1


def planted_index_keys(length: int, u0, u1, gen: torch.Generator) -> torch.Tensor:
    """Index-K rows [length, 128] (bf16, E4M3-exact) whose per-block maximum score is unambiguous.

    Every block has one peak row at a random position, ``a_b u0 + c_b u1`` with ``a`` and ``c`` two independent
    permutations of ``LEVELS``; its other rows are ``LOW (u0 + u1)``. A query along ``u0`` then ranks the blocks by
    ``a``, one along ``u1`` by ``c``, every score 1/8 or more apart from the next."""
    blocks = (length + PAGE - 1) // PAGE
    assert blocks <= len(LEVELS)
    levels = torch.tensor(LEVELS, device="cuda")
    a = levels[torch.randperm(len(LEVELS), generator=gen, device="cuda")[:blocks]]
    c = levels[torch.randperm(len(LEVELS), generator=gen, device="cuda")[:blocks]]
    rows = (LOW * (u0 + u1)).expand(length, HEAD_DIM).clone()
    for b in range(blocks):
        span = min(PAGE, length - b * PAGE)
        peak = b * PAGE + int(torch.randint(0, span, (1,), generator=gen, device="cuda"))
        rows[peak] = a[b] * u0 + c[b] * u1
    return rows.to(torch.bfloat16)


def nvfp4_quant(*values: torch.Tensor) -> torch.Tensor:
    """The (Q, K, V) quantization scales ``fused_write_layer_caches_nvfp4`` takes: each role's amax at the top of the
    E2M1 x E4M3 range."""
    quant = torch.ones(3, dtype=torch.float32, device="cuda")
    for role, x in enumerate(values, start=1):
        quant[role] = (448.0 * 6.0) / x.abs().max().float()
    return quant
