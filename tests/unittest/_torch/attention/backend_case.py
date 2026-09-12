# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified harness for exercising PyTorch attention backends.

A :class:`BackendCase` fully describes one attention forward (shapes, dtypes,
mask, RoPE, cache state) in a JSON-serializable form. :func:`run_case` runs the
VanillaAttention *golden* plus every supported backend through a real
``KVCacheManager`` and asserts the backends match the golden.

The same machinery is reused by the synthetic sweep (random cases) and the
replay suite (captured cases) — a captured case is just a serialized
``BackendCase``.
"""

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
from unittest.mock import patch

import torch
from backend_capability import BACKEND_CAPS, unsupported_reason
from kv_cache_utils import apply_rope, fill_kv_cache_logical, make_position_ids

import tensorrt_llm
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse import get_sparse_attn_kv_cache_manager
from tensorrt_llm._torch.attention.backends.sparse.dsa import (
    DSABackendForwardArgs,
    _effective_compress_ratio_divisor,
    _select_indexer_compress_ratio,
)
from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_torch, torch_dtype_to_binding
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, SparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

# Default tolerances (match the existing test_attention.py).
ATOL = 1e-2
RTOL = 1e-3
BF16_ATOL = 3e-2
BF16_RTOL = 3e-3
# Quantized-backend-vs-fp16-golden tolerances. Looser than backend-vs-backend at
# equal precision: e4m3 carries ~0.125 relative error, so attention outputs
# differ from the fp16 golden by ~0.1-0.4. Matches test_attention_mla.py (fp8=4e-1).
FP8_ATOL = 4e-1
FP4_ATOL = 6e-1
# Golden-vs-backend tolerance for selected sparse MLA (bf16 latent-gather
# accumulation).
SPARSE_ATOL = 1e-1
SPARSE_RTOL = 1e-2

# Backends compared against the VanillaAttention golden. FlashInfer is only
# included when available, so callers can iterate this list unconditionally.
BACKENDS_UNDER_TEST = ("TRTLLM",) + (("FLASHINFER",) if IS_FLASHINFER_AVAILABLE else ())
DEFAULT_MAX_NUM_TOKENS = 8192


def _dtype_to_torch(dtype: str):
    """Convert harness dtype names to torch dtypes, preserving the NVFP4 sentinel."""
    if dtype == "nvfp4":
        return dtype
    return str_dtype_to_torch(dtype)


@dataclass(kw_only=True)
class BackendCase:
    """A single attention forward, fully described and JSON-serializable."""

    num_heads: int
    num_kv_heads: int
    head_dim: int
    # Query tokens per request (self-attention: also the number of new KV tokens).
    seq_lens: List[int]
    # Already-cached KV tokens per request (0 for pure prefill).
    num_cached_tokens: List[int]
    # Number of context-phase requests (the rest are generation-phase).
    num_contexts: int

    # Cross-attention: new KV (encoder) tokens per request. None => self-attention
    # (KV tokens == seq_lens). When set, the case is cross-attention (must be
    # non-causal).
    seq_lens_kv: Optional[List[int]] = None

    dtype: str = "float16"
    # KV cache dtype. None mirrors the compute dtype (the realistic default);
    # set explicitly only to quantize the cache ("fp8" / "nvfp4").
    kv_dtype: Optional[str] = None
    causal: bool = True
    sliding_window: Optional[int] = None
    q_scaling: float = 1.0
    page_size: int = 64
    cache: str = "paged"  # "paged" | "none"
    # User-facing sparse config, lowered into backend params, metadata params, and
    # the sparse KV-cache manager exactly as in production. The selection unit and
    # top-k are derived from it (properties below); the attention family is
    # ``is_mla``.
    sparse_attention_config: Optional[SparseAttentionConfig] = None
    # RoPE config: RopeParams kwargs (+ optional "is_neox"), or None to disable.
    rope: Optional[dict] = None
    # When True (and rope set), exercise TRTLLM's in-kernel fused RoPE: TRTLLM
    # receives pre-RoPE q/k + pos_embd_params and rotates internally, while the
    # Vanilla golden / FlashInfer get harness-applied RoPE. Only affects TRTLLM.
    fused_rope: bool = False
    # Paged KV-cache block layout for the backends under test: "NHD" or "HND".
    # None lets each backend use its native layout (Vanilla/TRTLLM are fixed;
    # FlashInfer defaults to HND). A backend that cannot store the requested
    # layout is skipped via the capability matrix.
    kv_layout: Optional[str] = None
    is_mla: bool = False
    # MLA latent dims (only meaningful when is_mla). The unified harness runs the
    # *absorbed generation* MLA path: fused_q does MQA over a single-latent cache
    # ([compressed_kv | k_pe]); value is the kv_lora_rank slice. Validated as the
    # Vanilla golden vs FlashInfer and TRTLLM. The harness skips
    # mla_rope_generation (feeding a pre-formed fused_q and explicit q_pe), so
    # all three backends run RoPE-free and stay aligned; TRTLLM additionally
    # pre-writes the new latent into the cache and Python-initializes the
    # trtllm-gen scheduler buffers. MLA context uses coherent up-projected K/V
    # and latent-cache inputs: TRTLLM fuses RoPE while Vanilla/FlashInfer receive
    # the equivalent pre-rotated tensors.
    v_head_dim: Optional[int] = None
    hidden_size: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    use_kv_cache_manager_v2: bool = False

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def nnz_q(self) -> int:
        return sum(self.seq_lens)

    @property
    def is_cross(self) -> bool:
        return self.seq_lens_kv is not None

    @property
    def is_sparse(self) -> bool:
        return self.sparse_attention_config is not None

    @property
    def sparse_topk(self) -> Optional[int]:
        """Per-token selection budget (``index_topk``) from the sparse config."""
        cfg = self.sparse_attention_config
        if cfg is None:
            return None
        return cfg.index_topk

    @property
    def prompt_lens(self) -> List[int]:
        """Original prompt lengths expected by fused generation kernels."""
        return [
            seq_len if i < self.num_contexts else cached_len
            for i, (seq_len, cached_len) in enumerate(
                zip(self.seq_lens, self.num_cached_tokens, strict=True)
            )
        ]

    @property
    def is_gen_only(self) -> bool:
        """A uniform pure-decode batch eligible for a captured CUDA graph.

        CUDA graphs require a fixed batch and shape at capture; production only
        captures the generation phase. So a case qualifies only when it is paged,
        has no context requests.
        """
        return self.cache != "none" and self.num_contexts == 0 and not self.is_cross

    @property
    def is_context_only(self) -> bool:
        return self.num_contexts == self.num_seqs

    @property
    def kv_new_lens(self) -> List[int]:
        """New KV tokens per request (encoder side for cross; == seq_lens for self)."""
        return self.seq_lens_kv if self.seq_lens_kv is not None else self.seq_lens

    @property
    def nnz_kv(self) -> int:
        return sum(self.kv_new_lens)

    @property
    def token_nums(self) -> List[int]:
        # Total KV tokens per request after the backend appends the new KV.
        return [c + s for c, s in zip(self.num_cached_tokens, self.kv_new_lens, strict=True)]

    @property
    def compute_dtype(self) -> torch.dtype:
        return _dtype_to_torch(self.dtype)

    @property
    def kv_torch_dtype(self):
        if self.kv_dtype is None:
            return self.compute_dtype
        return _dtype_to_torch(self.kv_dtype)

    @property
    def max_num_tokens(self) -> int:
        """Metadata token capacity needed by the largest packed tensor in the case."""
        return max(DEFAULT_MAX_NUM_TOKENS, self.nnz_q, self.nnz_kv, *self.token_nums)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "BackendCase":
        fields = BackendCase.__dataclass_fields__
        return BackendCase(**{k: v for k, v in d.items() if k in fields})


def _rope_params_from_dict(d: dict) -> RopeParams:
    fields = RopeParams.__dataclass_fields__
    kwargs = {k: v for k, v in d.items() if k in fields}
    st = kwargs.get("scale_type")
    if isinstance(st, str):
        kwargs["scale_type"] = RotaryScalingType.from_string(st)
    elif isinstance(st, int):
        kwargs["scale_type"] = RotaryScalingType(st)
    # short_factor / long_factor are tuples in RopeParams but JSON gives lists.
    for key in ("short_factor", "long_factor"):
        if isinstance(kwargs.get(key), list):
            kwargs[key] = tuple(kwargs[key])
    return RopeParams(**kwargs)


def _validate_sparse_case(case: BackendCase) -> None:
    """Reject unsupported sparse contracts (called only for sparse cases)."""
    if case.sparse_topk is None or case.sparse_topk <= 0:
        raise ValueError("Sparse backend cases require a positive top-k")


def _randn(gen: torch.Generator, dtype: torch.dtype, *shape) -> torch.Tensor:
    """Seeded random tensor on cuda in ``dtype`` (shared by all input builders)."""
    return torch.randn(*shape, generator=gen, device="cuda").to(dtype)


def generate_inputs(case: BackendCase, seed: int) -> Dict[str, object]:
    """Generate seeded, reproducible pre-RoPE inputs in compute dtype.

    Returns packed ``q`` and new ``k``/``v`` plus per-sequence cached K/V.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cdt = case.compute_dtype
    H, Hkv, D = case.num_heads, case.num_kv_heads, case.head_dim

    q = _randn(gen, cdt, case.nnz_q, H * D)
    # New KV tokens: == q tokens for self-attention, encoder length for cross.
    new_k = _randn(gen, cdt, case.nnz_kv, Hkv * D)
    new_v = _randn(gen, cdt, case.nnz_kv, Hkv * D)
    cached_k = [_randn(gen, cdt, c, Hkv, D) for c in case.num_cached_tokens]
    cached_v = [_randn(gen, cdt, c, Hkv, D) for c in case.num_cached_tokens]
    return dict(q=q, new_k=new_k, new_v=new_v, cached_k=cached_k, cached_v=cached_v)


def _build_kv_cache_manager(case: BackendCase, backend: str, kv_dtype: torch.dtype):
    paged = BACKEND_CAPS[backend]["paged"]
    max_total = max(case.token_nums)
    if paged:
        tokens_per_block = case.page_size
        pages_per_seq = math.ceil(max_total / tokens_per_block)
    else:
        tokens_per_block = max(max_total, 1)
        pages_per_seq = 1
    num_blocks = case.num_seqs * pages_per_seq
    max_seq_len = pages_per_seq * tokens_per_block

    if kv_dtype == "nvfp4":
        bindings_dtype = tensorrt_llm.bindings.DataType.NVFP4
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block, dtype="nvfp4")
    else:
        bindings_dtype = torch_dtype_to_binding(kv_dtype)
        kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    cache_types = tensorrt_llm.bindings.internal.batch_manager.CacheType
    cache_type = cache_types.CROSS if case.is_cross else cache_types.SELF
    cls = KVCacheManagerV2 if case.use_kv_cache_manager_v2 else KVCacheManager

    mgr = cls(
        kv_cache_config,
        cache_type,
        num_layers=1,
        num_kv_heads=case.num_kv_heads,
        head_dim=case.head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=case.num_seqs,
        mapping=mapping,
        dtype=bindings_dtype,
    )
    return mgr


# ---------------------------------------------------------------------------
# MLA (DeepSeek-style absorbed latent attention) generation.
#
# The MLA module's absorbed-generation step is module-orchestrated: a
# q_nope @ W_UK absorption BMM produces fused_q's lora part, RoPE is applied to
# q_pe / k_pe, and the backend then does MQA of fused_q over a single-latent KV
# cache ([compressed_kv | k_pe], head_dim kv_lora_rank + qk_rope_head_dim) with
# value = the kv_lora_rank slice. We do not test the BMM/projection here -- the
# harness feeds the *absorbed* fused_q + latent directly (random), exercising
# only the MQA. Because the identical inputs go to both the Vanilla golden and
# FlashInfer, the comparison validates the absorbed-MQA math regardless of the
# RoPE values (RoPE correctness is covered by test_attention_mla.py).
# ---------------------------------------------------------------------------
def _build_mla_kv_cache_manager(
    case: BackendCase,
    backend: str,
    sparse_config=None,
):
    """A SELFKONLY KV cache for MLA: one latent head, head_dim kv_lora+qk_rope."""
    d_latent = case.kv_lora_rank + case.qk_rope_head_dim
    # Sparse selected-attention tests deliberately exercise multiple pages in
    # Vanilla too. Dense Vanilla keeps its historical single-block setup.
    paged = case.is_sparse or BACKEND_CAPS[backend]["paged"]
    max_total = max(case.token_nums)
    if paged:
        tokens_per_block = case.page_size
        pages_per_seq = math.ceil(max_total / tokens_per_block)
    else:
        tokens_per_block = max(max_total, 1)
        pages_per_seq = 1
    num_blocks = case.num_seqs * pages_per_seq
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    cache_types = tensorrt_llm.bindings.internal.batch_manager.CacheType
    kwargs = dict(
        kv_cache_config=KvCacheConfig(
            max_tokens=num_blocks * tokens_per_block,
            enable_block_reuse=False,
        ),
        kv_cache_type=cache_types.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=d_latent,
        tokens_per_block=tokens_per_block,
        max_seq_len=pages_per_seq * tokens_per_block,
        max_batch_size=case.num_seqs,
        mapping=mapping,
        dtype=torch_dtype_to_binding(case.compute_dtype),
    )

    if sparse_config is not None:
        cls = get_sparse_attn_kv_cache_manager(sparse_config)
        kwargs.update(sparse_attention_config=sparse_config)
    else:
        cls = KVCacheManagerV2 if case.use_kv_cache_manager_v2 else KVCacheManager

    return cls(**kwargs)


def generate_mla_gen_inputs(case: BackendCase, seed: int = 0) -> Dict:
    """Random absorbed-MLA generation inputs (shared by all backends).

    ``fused_q`` already contains the rope slot (also returned as ``q_pe``) so
    the harness never calls ``mla_rope_generation`` -- that op ropes inside the
    kernel for fusing backends but not for non-fusing ones, which would desync
    the comparison. Feeding identical pre-formed ``fused_q`` to every backend
    keeps them aligned.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cdt = case.compute_dtype
    H = case.num_heads
    d_latent = case.kv_lora_rank + case.qk_rope_head_dim
    fused_q = _randn(gen, cdt, case.nnz_q, H * d_latent)
    return dict(
        # [num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim)].
        fused_q=fused_q,
        # The q_pe view is passed explicitly when the backend's fused MLA RoPE
        # step is skipped.
        q_pe=fused_q.view(case.nnz_q, H, d_latent)[..., case.kv_lora_rank :],
        # New latent token per query token: [compressed_kv | k_pe].
        latent_cache=_randn(gen, cdt, case.nnz_q, d_latent),
        # Cached latent prefix per request.
        cached_latent=[_randn(gen, cdt, c, d_latent) for c in case.num_cached_tokens],
    )


def generate_sparse_mla_inputs(case: BackendCase, seed: int = 0) -> Dict:
    """Generate raw absorbed-MLA inputs plus the indexer inputs and weights.

    The selection itself is not generated here: every backend runs the
    production indexer over these inputs, so the top-k under test is the
    model's, not the harness's.
    """
    if not case.is_mla:
        raise ValueError("This generator supports selected sparse MLA only")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cdt = case.compute_dtype
    num_heads = case.num_heads
    kv_lora_rank = case.kv_lora_rank
    qk_rope_head_dim = case.qk_rope_head_dim
    d_latent = kv_lora_rank + qk_rope_head_dim

    q_nope = _randn(gen, cdt, case.nnz_q, num_heads, kv_lora_rank)
    q_pe = _randn(gen, cdt, case.nnz_q, num_heads, qk_rope_head_dim)
    compressed_kv = _randn(gen, cdt, case.nnz_q, kv_lora_rank)
    k_pe = _randn(gen, cdt, case.nnz_q, qk_rope_head_dim)

    pos_embd_params = _mla_context_pos_embd_params(case)
    rope_params = pos_embd_params.rope
    assert rope_params is not None
    new_positions = make_position_ids(case.seq_lens, case.num_cached_tokens)
    # Two input flavors, because RoPE happens in different places per phase:
    #   * generation: the standalone harness skips mla_rope_generation, so feed
    #     the RoPE'd inputs to every backend.
    #   * context: both DSA backends own RoPE and receive raw inputs.
    # q_pe rotates per head; k_pe is shared across heads.
    rotated_q_pe = apply_rope(
        q_pe.reshape(case.nnz_q, num_heads * qk_rope_head_dim),
        new_positions,
        rope_params,
        qk_rope_head_dim,
        is_neox=pos_embd_params.is_neox,
    ).reshape(case.nnz_q, num_heads, qk_rope_head_dim)
    fused_q = torch.cat((q_nope, rotated_q_pe), dim=-1).reshape(case.nnz_q, num_heads * d_latent)
    fused_q_raw = torch.cat((q_nope, q_pe), dim=-1).reshape(case.nnz_q, num_heads * d_latent)
    rotated_new_k_pe = apply_rope(
        k_pe,
        new_positions,
        rope_params,
        qk_rope_head_dim,
        is_neox=pos_embd_params.is_neox,
    )
    expected_new_latent = torch.cat((compressed_kv, rotated_new_k_pe), dim=-1)
    # RoPE'd new-token latent: with skip_mla_rope_generation the backend appends it
    # verbatim. The raw variant is roped in-kernel by the TRTLLM context path.
    latent_cache = expected_new_latent
    latent_cache_raw = torch.cat((compressed_kv, k_pe), dim=-1)

    cached_latent = []
    for cached_len in case.num_cached_tokens:
        cached_compressed = _randn(gen, cdt, cached_len, kv_lora_rank)
        cached_k_pe = _randn(gen, cdt, cached_len, qk_rope_head_dim)
        if cached_len:
            cached_positions = torch.arange(cached_len, dtype=torch.int32, device="cuda")
            cached_k_pe = apply_rope(
                cached_k_pe,
                cached_positions,
                rope_params,
                qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )
        cached_latent.append(torch.cat((cached_compressed, cached_k_pe), dim=-1))

    sparse_params = case.sparse_attention_config.to_sparse_params(
        layer_idx=None, pretrained_config=None
    )
    # Indexer inputs. ``qr`` is the q_a_layernorm output the model feeds to
    # wq_b; ``hidden_states`` drives the key and per-head weight projections.
    # The cached variants replay the prefix that produced the already-cached
    # KV, so the harness can prime the indexer K cache the way generation
    # steps do in production.
    index_n_heads = sparse_params.index_n_heads
    index_head_dim = sparse_params.index_head_dim
    hidden_size = case.hidden_size
    q_lora_rank = case.q_lora_rank
    if hidden_size is None or q_lora_rank is None:
        raise ValueError("Sparse MLA cases require hidden_size and q_lora_rank for the indexer")
    # Small weight scale keeps the indexer logits inside the FP8 range the
    # production quantization assumes.
    weight_scale = 0.02
    indexer_weights = dict(
        wq_b=_randn(gen, cdt, index_n_heads * index_head_dim, q_lora_rank) * weight_scale,
        wk=_randn(gen, torch.float32, index_head_dim, hidden_size) * weight_scale,
        weights_proj=_randn(gen, torch.float32, index_n_heads, hidden_size) * weight_scale,
    )

    return dict(
        fused_q=fused_q,
        # The RoPE'd q_pe view is passed explicitly since the MLA RoPE step is
        # skipped (skip_mla_rope_generation); it must match the fused_q pe slot.
        q_pe=fused_q.view(case.nnz_q, num_heads, d_latent)[..., kv_lora_rank:],
        latent_cache=latent_cache,
        # Raw (un-RoPE'd) variants for both DSA context paths.
        fused_q_raw=fused_q_raw,
        q_pe_raw=q_pe,
        latent_cache_raw=latent_cache_raw,
        cached_latent=cached_latent,
        expected_new_latent=expected_new_latent,
        indexer_weights=indexer_weights,
        hidden_states=_randn(gen, cdt, case.nnz_q, hidden_size),
        qr=_randn(gen, cdt, case.nnz_q, q_lora_rank),
        cached_hidden_states=[_randn(gen, cdt, c, hidden_size) for c in case.num_cached_tokens],
        cached_qr=[_randn(gen, cdt, c, q_lora_rank) for c in case.num_cached_tokens],
    )


def _fill_mla_cache(mgr, layer_idx, request_ids, cached_latent, *, kv_layout="NHD"):
    """Write the per-request cached latent prefix into the MLA cache pool."""
    if all(c.shape[0] == 0 for c in cached_latent):
        return
    buf = mgr.get_buffers(layer_idx, kv_layout=kv_layout)
    if kv_layout == "NHD":
        tokens_per_block = buf.shape[2]
    elif kv_layout == "HND":
        tokens_per_block = buf.shape[3]
    else:
        raise ValueError(f"Unsupported kv_layout: {kv_layout}")
    blocks_per_req = mgr.get_batch_cache_indices(list(request_ids), layer_idx)
    for i, blocks in enumerate(blocks_per_req):
        blocks = [b for b in blocks if b != -1]
        lat = cached_latent[i]
        written = 0
        for blk in blocks:
            if written >= lat.shape[0]:
                break
            n = min(tokens_per_block, lat.shape[0] - written)
            if kv_layout == "NHD":
                buf[blk, 0, :n, 0, :].copy_(lat[written : written + n].to(buf.dtype))
            else:
                buf[blk, 0, 0, :n, :].copy_(lat[written : written + n].to(buf.dtype))
            written += n


def _kv_cache_tokens_per_block(buf: torch.Tensor, kv_layout: str) -> int:
    if kv_layout == "NHD":
        return buf.shape[2]
    if kv_layout == "HND":
        return buf.shape[3]
    raise ValueError(f"Unsupported kv_layout: {kv_layout}")


def _slice_cache_tokens(
    buf: torch.Tensor,
    block: int,
    block_offset: int,
    n: int,
    kv_layout: str,
    *,
    cache_kind: str,
) -> torch.Tensor:
    if cache_kind == "mla":
        if kv_layout == "NHD":
            return buf[block, 0, block_offset : block_offset + n, 0, :]
        return buf[block, 0, 0, block_offset : block_offset + n, :]

    if cache_kind == "kv":
        if kv_layout == "NHD":
            return buf[block, :, block_offset : block_offset + n, :, :]
        return buf[block, :, :, block_offset : block_offset + n, :].transpose(1, 2)

    raise ValueError(f"Unsupported cache kind: {cache_kind}")


def _split_packed_tokens(packed: torch.Tensor, lengths, *tail_shape) -> list[torch.Tensor]:
    chunks = []
    offset = 0
    for length in lengths:
        chunk = packed[offset : offset + length]
        if tail_shape:
            chunk = chunk.view(length, *tail_shape)
        chunks.append(chunk)
        offset += length
    return chunks


def _expected_standard_cache_tokens(
    case: BackendCase, new_k: torch.Tensor, new_v: torch.Tensor
) -> list[torch.Tensor]:
    k_per_seq = _split_packed_tokens(new_k, case.kv_new_lens, case.num_kv_heads, case.head_dim)
    v_per_seq = _split_packed_tokens(new_v, case.kv_new_lens, case.num_kv_heads, case.head_dim)
    return [torch.stack((k, v), dim=0) for k, v in zip(k_per_seq, v_per_seq, strict=True)]


def _assert_cache_contains_new_tokens(
    mgr,
    layer_idx,
    request_ids,
    new_lens,
    num_cached_tokens,
    expected_per_seq: list[torch.Tensor],
    *,
    kv_layout: str,
    cache_kind: str,
    atol: float = 0.0,
    rtol: float = 0.0,
):
    """Assert the backend appended each request's expected cache tokens."""
    buf = mgr.get_buffers(layer_idx, kv_layout=kv_layout)
    tokens_per_block = _kv_cache_tokens_per_block(buf, kv_layout)
    blocks_per_req = mgr.get_batch_cache_indices(list(request_ids), layer_idx)
    concat_dim = 1 if cache_kind == "kv" else 0
    for i, new_len in enumerate(new_lens):
        if new_len == 0:
            continue
        start = num_cached_tokens[i]
        blocks = [b for b in blocks_per_req[i] if b != -1]
        pieces = []
        read = 0
        while read < new_len:
            pos = start + read
            block = blocks[pos // tokens_per_block]
            block_offset = pos % tokens_per_block
            n = min(tokens_per_block - block_offset, new_len - read)
            pieces.append(
                _slice_cache_tokens(
                    buf,
                    block,
                    block_offset,
                    n,
                    kv_layout,
                    cache_kind=cache_kind,
                )
            )
            read += n

        actual = torch.cat(pieces, dim=concat_dim).to(torch.float32)
        expected = expected_per_seq[i].to(buf.dtype).to(torch.float32)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _create_indexer(attn, inputs) -> None:
    """Create and load the indexer weights every backend shares.

    The backends are built with ``skip_create_weights_in_init``, so the
    indexer's Linear layers still need their storage; loading the same random
    weights into each backend is what makes their selections comparable.
    """
    indexer = attn.indexer
    for linear in (indexer.wq_b, indexer.wk, indexer.weights_proj):
        linear.create_weights()
    indexer.to("cuda")
    # Production runs the indexer under inference_mode; without that, autograd
    # tracks the weights and rejects the in-place RoPE on projection views.
    indexer.requires_grad_(False)
    weights = inputs["indexer_weights"]
    with torch.no_grad():
        indexer.wq_b.weight.copy_(weights["wq_b"])
        indexer.wk.weight.copy_(weights["wk"])
        indexer.weights_proj.weight.copy_(weights["weights_proj"])
    indexer.cache_derived_state()


def _make_sparse_metadata(
    AttentionCls,
    case: BackendCase,
    mgr,
    request_ids: List[int],
    seq_lens: List[int],
    num_cached_tokens: List[int],
    num_contexts: int,
    prompt_lens: List[int],
    *,
    kv_layout: str,
    mapping,
    sparse_metadata_params,
):
    metadata = AttentionCls.Metadata(
        num_contexts=num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=num_cached_tokens,
        ),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int),
        max_num_requests=case.num_seqs,
        max_num_tokens=case.max_num_tokens,
        kv_cache_manager=mgr,
        request_ids=request_ids,
        prompt_lens=prompt_lens,
        kv_layout=kv_layout,
        mapping=mapping,
        sparse_metadata_params=sparse_metadata_params,
    )
    metadata.prepare()
    return metadata


def _run_indexer_projections(attn, metadata, qr, hidden_states, position_ids) -> List[torch.Tensor]:
    """Project the indexer inputs and append the new keys to the indexer cache.

    Mirrors what the MLA module does around the backend forward
    (``forward_dsa_proj`` + the ``_update_k_cache`` in ``_forward_dsa_attn``).
    """
    with torch.no_grad():
        q_fp8, k_fp8, k_scale, weights, q_scale = attn.indexer.pre_indexer_proj(
            qr, hidden_states, position_ids
        )
        attn.indexer._update_k_cache(k_fp8, k_scale, metadata)
    return [q_fp8, k_fp8, k_scale, weights, q_scale]


def _seed_indexer_k_cache(
    attn,
    case: BackendCase,
    inputs: Dict,
    AttentionCls,
    mgr,
    request_ids: List[int],
    *,
    kv_layout: str,
    mapping,
    sparse_metadata_params,
) -> None:
    """Prime the indexer K cache with each request's cached prefix.

    Production fills the cache incrementally as tokens are produced; the
    harness replays that as a single prefill over the cached tokens, so the
    indexer scores the same keys it would score mid-generation.
    """
    seeded = [(i, c) for i, c in enumerate(case.num_cached_tokens) if c > 0]
    if not seeded:
        return
    indices = [i for i, _ in seeded]
    cached_lens = [c for _, c in seeded]
    metadata = _make_sparse_metadata(
        AttentionCls,
        case,
        mgr,
        [request_ids[i] for i in indices],
        cached_lens,
        [0] * len(indices),
        len(indices),
        cached_lens,
        kv_layout=kv_layout,
        mapping=mapping,
        sparse_metadata_params=sparse_metadata_params,
    )
    _run_indexer_projections(
        attn,
        metadata,
        torch.cat([inputs["cached_qr"][i] for i in indices]),
        torch.cat([inputs["cached_hidden_states"][i] for i in indices]),
        make_position_ids(cached_lens, [0] * len(indices)),
    )


def _run_sparse_mla_backend(
    case: BackendCase,
    backend: str,
    inputs: Dict,
    *,
    kv_layout: str,
    indexer_outputs: Optional[Dict[str, torch.Tensor]] = None,
    indexer_topk_override: Optional[Dict[str, torch.Tensor]] = None,
    native_outputs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Run selected sparse MLA through production backend/config lowering.

    ``indexer_outputs`` records request-local selections independently of the
    attention result. ``indexer_topk_override`` lets an implementation consume
    the golden selection after its own indexer has run, so the attention and
    indexer contracts are compared separately. When both an override and
    ``native_outputs`` are supplied, the backend is also run with its own
    selection and that composed-path result is returned through the dictionary.
    """
    sparse_config = case.sparse_attention_config
    assert sparse_config is not None
    # The DSA config carries every field the lowering needs, so no pretrained
    # config is required.
    sparse_params = sparse_config.to_sparse_params(layer_idx=None, pretrained_config=None)
    sparse_metadata_params = sparse_config.to_sparse_metadata_params(pretrained_config=None)
    AttentionCls = get_attention_backend(backend, sparse_params=sparse_params)
    request_ids = list(range(case.num_seqs))
    d_latent = case.kv_lora_rank + case.qk_rope_head_dim
    pos_embd_params = _mla_context_pos_embd_params(case)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    attn = create_attention(
        backend,
        layer_idx=0,
        num_heads=case.num_heads,
        head_dim=d_latent,
        num_kv_heads=1,
        q_scaling=case.q_scaling,
        pos_embd_params=pos_embd_params,
        is_mla_enable=True,
        q_lora_rank=case.q_lora_rank,
        kv_lora_rank=case.kv_lora_rank,
        qk_nope_head_dim=case.qk_nope_head_dim,
        qk_rope_head_dim=case.qk_rope_head_dim,
        # Selected sparse MLA returns latent values; the model's V projection
        # lives outside the standalone backend, so v_head_dim is the latent width.
        v_head_dim=case.kv_lora_rank,
        hidden_size=case.hidden_size,
        predicted_tokens_per_seq=1,
        sparse_params=sparse_params,
        dtype=case.compute_dtype,
        skip_create_weights_in_init=True,
    )
    # update_quant_config initializes the quant/FMHA state needed before forward;
    # _create_indexer then materializes the indexer weights it skipped.
    attn.update_quant_config(None)
    _create_indexer(attn, inputs)
    mgr = _build_mla_kv_cache_manager(case, backend, sparse_config)

    try:
        mgr.add_dummy_requests(request_ids, case.token_nums)
        _fill_mla_cache(
            mgr,
            0,
            request_ids,
            inputs["cached_latent"],
            kv_layout=kv_layout,
        )
        _seed_indexer_k_cache(
            attn,
            case,
            inputs,
            AttentionCls,
            mgr,
            request_ids,
            kv_layout=kv_layout,
            mapping=mapping,
            sparse_metadata_params=sparse_metadata_params,
        )
        metadata = _make_sparse_metadata(
            AttentionCls,
            case,
            mgr,
            request_ids,
            case.seq_lens,
            case.num_cached_tokens,
            case.num_contexts,
            case.prompt_lens,
            kv_layout=kv_layout,
            mapping=mapping,
            sparse_metadata_params=sparse_metadata_params,
        )
        # Both phases score against one set of projections, exactly as the MLA
        # module does: the indexer runs once per batch, before the phase split.
        indexer_intermediates = _run_indexer_projections(
            attn,
            metadata,
            inputs["qr"],
            inputs["hidden_states"],
            make_position_ids(case.seq_lens, case.num_cached_tokens),
        )

        num_context_tokens = sum(case.seq_lens[: case.num_contexts])
        phases = []
        if case.num_contexts:
            phases.append((AttentionInputType.context_only, slice(0, num_context_tokens)))
        if case.num_contexts < case.num_seqs:
            phases.append(
                (AttentionInputType.generation_only, slice(num_context_tokens, case.nnz_q))
            )

        outputs = []
        native_phase_outputs = []
        for attention_input_type, token_slice in phases:
            # Both DSA backends own context RoPE, so context receives raw
            # inputs. The standalone generation harness does not invoke the
            # module's mla_rope_generation hook and therefore feeds generation
            # inputs with RoPE already applied.
            kernel_ropes = attention_input_type == AttentionInputType.context_only
            input_suffix = "_raw" if kernel_ropes else ""

            def run_attention(topk: torch.Tensor) -> torch.Tensor:
                # Context RoPE mutates Q/K in place, so each comparison needs
                # fresh inputs. Cache appends target the same logical slots and
                # are therefore idempotent across these two calls.
                phase_fused_q = inputs[f"fused_q{input_suffix}"][token_slice].clone()
                phase_q_pe = inputs[f"q_pe{input_suffix}"][token_slice].clone()
                phase_latent_cache = inputs[f"latent_cache{input_suffix}"][token_slice].clone()
                phase_forward_args = AttentionForwardArgs(
                    latent_cache=phase_latent_cache,
                    q_pe=phase_q_pe,
                    sparse_backend_args=DSABackendForwardArgs(
                        indexer_intermediates=indexer_intermediates
                    ),
                    attention_input_type=attention_input_type,
                    skip_mla_rope_generation=not kernel_ropes,
                )
                with patch.object(
                    attn.indexer,
                    "forward_from_projected",
                    return_value=topk,
                ):
                    phase_output = attn.forward(
                        phase_fused_q,
                        None,
                        None,
                        metadata,
                        forward_args=phase_forward_args,
                    )
                assert phase_forward_args.sparse_runtime_params.sparse_attn_indices is not None
                return phase_output[0] if isinstance(phase_output, tuple) else phase_output

            is_generation = attention_input_type == AttentionInputType.generation_only
            phase_name = "generation" if is_generation else "context"
            indexer_hidden_states = inputs[f"fused_q{input_suffix}"][token_slice]
            computed_topk = (
                attn.indexer.forward_from_projected(
                    metadata,
                    indexer_hidden_states,
                    indexer_intermediates,
                    is_generation=is_generation,
                )
                .detach()
                .clone()
            )
            if indexer_outputs is not None:
                indexer_outputs[phase_name] = computed_topk
            if native_outputs is not None and indexer_topk_override is not None:
                # Backend outputs may alias a reusable workspace. Preserve the
                # native result before the isolated-attention call reuses it.
                native_phase_outputs.append(run_attention(computed_topk).clone())

            attention_topk = computed_topk
            if indexer_topk_override is not None:
                attention_topk = indexer_topk_override[phase_name]
            # Preserve each phase before the next phase can reuse its workspace.
            phase_output = run_attention(attention_topk).clone()
            outputs.append(phase_output)
            if native_outputs is not None and indexer_topk_override is None:
                native_phase_outputs.append(phase_output)

        expected_latents = _split_packed_tokens(inputs["expected_new_latent"], case.seq_lens)
        cache_atol, cache_rtol = _tolerances(case, case.compute_dtype)
        _assert_cache_contains_new_tokens(
            mgr,
            0,
            request_ids,
            case.seq_lens,
            case.num_cached_tokens,
            expected_latents,
            kv_layout=metadata.kv_layout,
            cache_kind="mla",
            atol=cache_atol,
            rtol=cache_rtol,
        )
        output = torch.cat(outputs, dim=0)[: case.nnz_q].contiguous()
        if native_outputs is not None:
            native_outputs["output"] = torch.cat(native_phase_outputs, dim=0)[
                : case.nnz_q
            ].contiguous()
        return output
    finally:
        mgr.shutdown()


def _run_mla_gen_backend(
    case, backend, inputs, *, kv_layout: str, cuda_graph=False
) -> torch.Tensor:
    """Run one backend's absorbed-MLA generation; return [nnz_q, heads*kv_lora].

    No ``mla_rope_generation`` call (see ``generate_mla_gen_inputs``): ``fused_q``
    is passed as-is and the backend's ``forward`` appends the new latent + MQA.
    """
    AttentionCls = get_attention_backend(backend)
    H = case.num_heads
    d_latent = case.kv_lora_rank + case.qk_rope_head_dim
    request_ids = list(range(case.num_seqs))
    attn = create_attention(
        backend,
        layer_idx=0,
        num_heads=H,
        head_dim=d_latent,
        num_kv_heads=1,
        q_scaling=case.q_scaling,
        is_mla_enable=True,
        q_lora_rank=case.q_lora_rank,
        kv_lora_rank=case.kv_lora_rank,
        qk_nope_head_dim=case.qk_nope_head_dim,
        qk_rope_head_dim=case.qk_rope_head_dim,
        v_head_dim=case.v_head_dim,
    )
    mgr = _build_mla_kv_cache_manager(case, backend)
    mgr.add_dummy_requests(request_ids, case.token_nums)
    _fill_mla_cache(mgr, 0, request_ids, inputs["cached_latent"], kv_layout=kv_layout)
    fused_q = inputs["fused_q"]
    q_pe = inputs["q_pe"]
    latent_cache = inputs["latent_cache"]
    expected_latents = _split_packed_tokens(latent_cache, case.seq_lens)

    def _forward(metadata, q, q_pe):
        out = attn.forward(
            q,
            None,
            None,
            metadata,
            forward_args=AttentionForwardArgs(
                latent_cache=latent_cache,
                q_pe=q_pe,
                attention_input_type=AttentionInputType.generation_only,
                # The harness feeds a pre-RoPE'd fused_q, so skip the RoPE step;
                # the TRTLLM backend still appends the new latent and inits its
                # scheduler buffers. Vanilla/FlashInfer ignore this flag.
                skip_mla_rope_generation=True,
            ),
        )
        return out[0] if isinstance(out, tuple) else out

    def _create_metadata(AttentionCls, case, mgr):
        return AttentionCls.Metadata(
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=case.num_cached_tokens
            ),
            seq_lens=torch.tensor(case.seq_lens, dtype=torch.int),
            max_num_requests=case.num_seqs,
            max_num_tokens=case.max_num_tokens,
            kv_cache_manager=mgr,
            request_ids=request_ids,
            prompt_lens=case.token_nums,
            kv_layout=kv_layout,
        )

    try:
        if cuda_graph:
            out = _capture_replay(
                AttentionCls,
                case,
                mgr,
                _create_metadata,
                {"q": fused_q, "q_pe": q_pe},
                lambda md, bufs: _forward(md, bufs["q"], bufs["q_pe"]),
            )
            _assert_cache_contains_new_tokens(
                mgr,
                0,
                request_ids,
                case.seq_lens,
                case.num_cached_tokens,
                expected_latents,
                kv_layout=kv_layout,
                cache_kind="mla",
            )
            return out
        metadata = _create_metadata(AttentionCls, case, mgr)
        metadata.prepare()
        out = _forward(metadata, fused_q, q_pe)[: case.nnz_q].contiguous()
        _assert_cache_contains_new_tokens(
            mgr,
            0,
            request_ids,
            case.seq_lens,
            case.num_cached_tokens,
            expected_latents,
            kv_layout=metadata.kv_layout,
            cache_kind="mla",
        )
        return out
    finally:
        mgr.shutdown()


def _mla_context_pos_embd_params(case: BackendCase) -> PositionalEmbeddingParams:
    """Build the GPT-J-style RoPE configuration required by TRTLLM MLA context."""
    if case.rope is None:
        raise ValueError("TRTLLM MLA context requires RoPE parameters.")

    rope_config = dict(case.rope)
    rope_config.update(dim=case.qk_rope_head_dim, duplicate_data=True)
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gptj,
        rope=_rope_params_from_dict(rope_config),
        is_neox=False,
    )


def generate_mla_context_inputs(case: BackendCase, seed: int = 0) -> Dict:
    """Random production-layout MLA context inputs before RoPE."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cdt = case.compute_dtype
    H, Hkv = case.num_heads, case.num_kv_heads
    qk_head = case.qk_nope_head_dim + case.qk_rope_head_dim
    compressed_kv = _randn(gen, cdt, case.nnz_q, case.kv_lora_rank)
    k_pe = _randn(gen, cdt, case.nnz_q, case.qk_rope_head_dim)
    packed_kv = _randn(
        gen,
        cdt,
        case.nnz_q,
        Hkv * (case.qk_nope_head_dim + case.v_head_dim),
    )
    k_nope, v = packed_kv.split([Hkv * case.qk_nope_head_dim, Hkv * case.v_head_dim], dim=-1)
    k = torch.cat(
        [
            k_nope.view(-1, Hkv, case.qk_nope_head_dim),
            k_pe.view(-1, 1, case.qk_rope_head_dim).expand(-1, Hkv, -1),
        ],
        dim=-1,
    ).view(-1, Hkv * qk_head)
    return dict(
        q=_randn(gen, cdt, case.nnz_q, H * qk_head),
        k=k,
        # Keep the split view: TRTLLM MLA context expects token stride to include
        # the packed k_nope portion, and Vanilla/FlashInfer support that layout.
        v=v,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        latent_cache=torch.cat([compressed_kv, k_pe], dim=-1),
    )


def _prepare_mla_context_inputs(
    case: BackendCase,
    inputs: Dict,
    pos_embd_params: PositionalEmbeddingParams,
    *,
    fuse_rope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare backend inputs and the expected post-RoPE latent cache."""
    position_ids = make_position_ids(case.seq_lens, case.num_cached_tokens)
    rope_params = pos_embd_params.rope
    assert rope_params is not None

    rotated_k_pe = apply_rope(
        inputs["k_pe"],
        position_ids,
        rope_params,
        case.qk_rope_head_dim,
        is_neox=pos_embd_params.is_neox,
    )
    expected_latent_cache = torch.cat([inputs["compressed_kv"], rotated_k_pe], dim=-1)

    if fuse_rope:
        return (
            inputs["q"].clone(),
            inputs["k"].clone(),
            inputs["v"],
            inputs["latent_cache"],
            expected_latent_cache,
        )

    q = inputs["q"].clone().view(-1, case.num_heads, case.head_dim)
    q_pe = q[..., case.qk_nope_head_dim :].reshape(
        case.nnz_q, case.num_heads * case.qk_rope_head_dim
    )
    q[..., case.qk_nope_head_dim :] = apply_rope(
        q_pe,
        position_ids,
        rope_params,
        case.qk_rope_head_dim,
        is_neox=pos_embd_params.is_neox,
    ).view(case.nnz_q, case.num_heads, case.qk_rope_head_dim)

    k = inputs["k"].clone().view(-1, case.num_kv_heads, case.head_dim)
    k[..., case.qk_nope_head_dim :] = rotated_k_pe.view(case.nnz_q, 1, case.qk_rope_head_dim)
    return (
        q.view(case.nnz_q, -1),
        k.view(case.nnz_q, -1),
        inputs["v"],
        expected_latent_cache,
        expected_latent_cache,
    )


def _run_mla_context_backend(case, backend, inputs, *, kv_layout: str) -> torch.Tensor:
    """Run one backend's up-projected MLA context; return [nnz_q, heads*v_head]."""
    AttentionCls = get_attention_backend(backend)
    qk_head = case.qk_nope_head_dim + case.qk_rope_head_dim
    request_ids = list(range(case.num_seqs))
    pos_embd_params = _mla_context_pos_embd_params(case)
    attn = create_attention(
        backend,
        layer_idx=0,
        num_heads=case.num_heads,
        head_dim=qk_head,
        num_kv_heads=case.num_kv_heads,
        q_scaling=case.q_scaling,
        pos_embd_params=pos_embd_params,
        is_mla_enable=True,
        q_lora_rank=case.q_lora_rank,
        kv_lora_rank=case.kv_lora_rank,
        qk_nope_head_dim=case.qk_nope_head_dim,
        qk_rope_head_dim=case.qk_rope_head_dim,
        v_head_dim=case.v_head_dim,
    )
    mgr = _build_mla_kv_cache_manager(case, backend)
    mgr.add_dummy_requests(request_ids, case.token_nums)
    fuse_rope = AttentionCls.support_fused_rope()
    q, k, v, latent_cache, expected_latent_cache = _prepare_mla_context_inputs(
        case,
        inputs,
        pos_embd_params,
        fuse_rope=fuse_rope,
    )
    expected_latents = _split_packed_tokens(expected_latent_cache, case.seq_lens)
    cache_atol, cache_rtol = _tolerances(case, case.compute_dtype) if fuse_rope else (0.0, 0.0)
    metadata = AttentionCls.Metadata(
        num_contexts=case.num_contexts,
        kv_cache_params=KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=case.num_cached_tokens
        ),
        seq_lens=torch.tensor(case.seq_lens, dtype=torch.int),
        max_num_requests=case.num_seqs,
        max_num_tokens=case.max_num_tokens,
        kv_cache_manager=mgr,
        request_ids=request_ids,
        prompt_lens=case.token_nums,
        kv_layout=kv_layout,
    )
    metadata.prepare()
    try:
        out = attn.forward(
            q,
            k,
            v,
            metadata,
            forward_args=AttentionForwardArgs(
                latent_cache=latent_cache,
                attention_input_type=AttentionInputType.context_only,
            ),
        )
        if isinstance(out, tuple):
            out = out[0]
        _assert_cache_contains_new_tokens(
            mgr,
            0,
            request_ids,
            case.seq_lens,
            case.num_cached_tokens,
            expected_latents,
            kv_layout=metadata.kv_layout,
            cache_kind="mla",
            atol=cache_atol,
            rtol=cache_rtol,
        )
        return out[: case.nnz_q].contiguous()
    finally:
        mgr.shutdown()


def _quant_config(kv_dtype) -> Optional[QuantConfig]:
    # Quantize only the KV cache (not activations) — there are no projection
    # layers in the standalone backend, and activation FP8 QDQ here yields NaNs.
    if kv_dtype == torch.float8_e4m3fn:
        return QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    if kv_dtype == "nvfp4":
        return QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)
    return None


def _tolerances(case: "BackendCase", kv_dtype) -> tuple:
    """Dtype-appropriate (atol, rtol).

    fp8/fp4 quantization error dominates, so it sets the atol. When the compute
    dtype is bf16 its coarser mantissa compounds with the quant error, so the
    quantized atol gets extra headroom and the rtol relaxes to the bf16 rtol.
    """
    if case.is_sparse:
        return SPARSE_ATOL, SPARSE_RTOL

    bf16 = case.compute_dtype == torch.bfloat16
    if kv_dtype == torch.float8_e4m3fn:
        return (FP8_ATOL + BF16_ATOL, BF16_RTOL) if bf16 else (FP8_ATOL, RTOL)
    if kv_dtype == "nvfp4":
        return (FP4_ATOL + BF16_ATOL, BF16_RTOL) if bf16 else (FP4_ATOL, RTOL)
    if bf16:
        return BF16_ATOL, BF16_RTOL
    return ATOL, RTOL


def _assert_sparse_indexer_matches_golden(
    actual: Dict[str, torch.Tensor],
    golden: Dict[str, torch.Tensor],
    *,
    case: BackendCase,
    compress_ratio: int,
    min_row_overlap: float = 0.95,
) -> None:
    """Compare request-local TopK sets against the Vanilla indexer golden.

    The fused FP8 quantizer intentionally uses an approximate reciprocal while
    the torch reference uses exact division. Entries at the TopK boundary may
    therefore differ even when the scoring implementation is correct. Compare
    sets rather than score order and require every row to retain at least 95%
    of the golden selection. Causal limits are expressed in the indexer's
    compressed KV coordinate system.
    """
    if actual.keys() != golden.keys():
        raise AssertionError(
            f"Indexer phase mismatch: actual={list(actual)}, golden={list(golden)}"
        )

    def causal_limits(phase: str, device: torch.device) -> torch.Tensor:
        if phase == "context":
            seq_start, seq_end = 0, case.num_contexts
        elif phase == "generation":
            seq_start, seq_end = case.num_contexts, case.num_seqs
        else:
            raise AssertionError(f"Unknown Indexer phase: {phase}")
        visible_counts = [
            (
                torch.arange(
                    case.num_cached_tokens[i],
                    case.num_cached_tokens[i] + case.seq_lens[i],
                    device=device,
                    dtype=torch.int64,
                )
                + 1
            )
            // compress_ratio
            for i in range(seq_start, seq_end)
        ]
        if not visible_counts:
            return torch.empty(0, device=device, dtype=torch.int64)
        return torch.cat(visible_counts) - 1

    def validate_topk(
        topk: torch.Tensor,
        phase: str,
        source: str,
        limits: torch.Tensor,
    ) -> torch.Tensor:
        if topk.ndim != 2 or topk.shape[0] != limits.numel():
            raise AssertionError(
                f"Indexer row mismatch for {source} {phase}: "
                f"actual={topk.shape[0] if topk.ndim else 0}, expected={limits.numel()}"
            )
        if topk.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise AssertionError(f"Indexer {source} {phase} must contain integer indices")
        invalid_padding = topk < -1
        if invalid_padding.any():
            row, column = torch.nonzero(invalid_padding, as_tuple=False)[0].tolist()
            raise AssertionError(
                f"Indexer {source} {phase} has invalid padding at row {row}, "
                f"column {column}: {int(topk[row, column].item())}"
            )

        valid = topk >= 0
        out_of_range = valid & (topk.to(torch.int64) > limits.unsqueeze(1))
        if out_of_range.any():
            row, column = torch.nonzero(out_of_range, as_tuple=False)[0].tolist()
            raise AssertionError(
                f"Indexer {source} {phase} selected future index "
                f"{int(topk[row, column].item())} in row {row}, whose maximum is "
                f"{int(limits[row].item())}"
            )

        valid_counts = valid.sum(dim=1)
        expected_counts = (limits + 1).clamp_max(topk.shape[1])
        if not torch.equal(valid_counts, expected_counts):
            row = int(torch.nonzero(valid_counts != expected_counts, as_tuple=False)[0].item())
            raise AssertionError(
                f"Indexer {source} {phase} has {int(valid_counts[row].item())} "
                f"valid entries in row {row}, expected {int(expected_counts[row].item())}"
            )

        sentinel = torch.iinfo(topk.dtype).max
        sorted_topk = torch.where(valid, topk, sentinel).sort(dim=1).values
        duplicates = (sorted_topk[:, 1:] == sorted_topk[:, :-1]) & (sorted_topk[:, 1:] != sentinel)
        if duplicates.any():
            row, column = torch.nonzero(duplicates, as_tuple=False)[0].tolist()
            raise AssertionError(
                f"Indexer {source} {phase} repeats index "
                f"{int(sorted_topk[row, column].item())} in row {row}"
            )
        return sorted_topk

    for phase, golden_topk in golden.items():
        actual_topk = actual[phase]
        if actual_topk.shape != golden_topk.shape:
            raise AssertionError(
                f"Indexer shape mismatch for {phase}: "
                f"actual={tuple(actual_topk.shape)}, golden={tuple(golden_topk.shape)}"
            )
        if actual_topk.dtype != golden_topk.dtype:
            raise AssertionError(
                f"Indexer dtype mismatch for {phase}: "
                f"actual={actual_topk.dtype}, golden={golden_topk.dtype}"
            )

        limits = causal_limits(phase, golden_topk.device)
        actual_sorted = validate_topk(actual_topk, phase, "actual", limits)
        golden_sorted = validate_topk(golden_topk, phase, "golden", limits)
        actual_valid = actual_sorted >= 0
        golden_valid = golden_sorted >= 0
        sentinel = torch.iinfo(actual_sorted.dtype).max
        actual_valid &= actual_sorted != sentinel
        golden_valid &= golden_sorted != sentinel
        actual_counts = actual_valid.sum(dim=1)
        golden_counts = golden_valid.sum(dim=1)
        if not torch.equal(actual_counts, golden_counts):
            bad_row = int(torch.nonzero(actual_counts != golden_counts)[0].item())
            raise AssertionError(
                f"Indexer valid-entry count mismatch for {phase} row {bad_row}: "
                f"actual={int(actual_counts[bad_row].item())}, "
                f"golden={int(golden_counts[bad_row].item())}"
            )

        if actual_sorted.shape[1] == 0:
            continue
        positions = torch.searchsorted(golden_sorted, actual_sorted)
        safe_positions = positions.clamp_max(golden_sorted.shape[1] - 1)
        matches = (
            actual_valid
            & (positions < golden_sorted.shape[1])
            & (golden_sorted.gather(1, safe_positions) == actual_sorted)
        )
        intersections = matches.sum(dim=1)
        overlap = intersections.float() / golden_counts.clamp_min(1).float()
        overlap = torch.where(golden_counts == 0, torch.ones_like(overlap), overlap)
        worst_overlap, worst_row = overlap.min(dim=0)
        if float(worst_overlap.item()) < min_row_overlap:
            raise AssertionError(
                f"Indexer TopK overlap is too low for {phase} row {int(worst_row.item())}: "
                f"{float(worst_overlap.item()):.2%} < {min_row_overlap:.2%}"
            )


def _assert_sparse_end_to_end_matches_golden(
    actual: torch.Tensor,
    golden: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    min_close_fraction: float = 0.995,
    min_row_close_fraction: float = 0.8,
    max_mean_abs_error: float = 1e-2,
    max_row_mean_abs_error: float = 1e-1,
) -> None:
    """Bound the composed backend path while allowing TopK boundary swaps.

    Attention with an identical selection is checked strictly elsewhere. This
    check keeps each backend's own Indexer connected to its attention, but
    tolerates the small output tail caused by the fused Indexer's approximate
    reciprocal changing entries exactly at the TopK boundary.
    """
    if actual.shape != golden.shape:
        raise AssertionError(
            f"Composed sparse output shape mismatch: {tuple(actual.shape)} != {tuple(golden.shape)}"
        )
    if actual.dtype != golden.dtype:
        raise AssertionError(
            f"Composed sparse output dtype mismatch: {actual.dtype} != {golden.dtype}"
        )
    if not torch.isfinite(actual).all() or not torch.isfinite(golden).all():
        raise AssertionError("Composed sparse output contains non-finite values")

    close = torch.isclose(actual, golden, atol=atol, rtol=rtol)
    close_fraction = close.float().mean()
    row_close_fraction = close.reshape(close.shape[0], -1).float().mean(dim=1)
    abs_error = (actual.float() - golden.float()).abs()
    mean_abs_error = abs_error.mean()
    row_mean_abs_error = abs_error.reshape(abs_error.shape[0], -1).mean(dim=1)
    worst_close, worst_close_row = row_close_fraction.min(dim=0)
    worst_mean_error, worst_error_row = row_mean_abs_error.max(dim=0)
    if (
        float(close_fraction.item()) < min_close_fraction
        or float(worst_close.item()) < min_row_close_fraction
        or float(mean_abs_error.item()) > max_mean_abs_error
        or float(worst_mean_error.item()) > max_row_mean_abs_error
    ):
        raise AssertionError(
            "Composed sparse output drift is too large: "
            f"close={float(close_fraction.item()):.3%} "
            f"(required {min_close_fraction:.3%}), "
            f"worst_row_close={float(worst_close.item()):.3%} at row "
            f"{int(worst_close_row.item())} "
            f"(required {min_row_close_fraction:.3%}), "
            f"mean_abs_error={float(mean_abs_error.item()):.6f} "
            f"(allowed {max_mean_abs_error:.6f}), "
            f"worst_row_mean_abs_error={float(worst_mean_error.item()):.6f} at row "
            f"{int(worst_error_row.item())} "
            f"(allowed {max_row_mean_abs_error:.6f}), "
            f"max_abs_error={float(abs_error.max().item()):.6f}"
        )


def _maybe_rope(case: BackendCase, inputs, *, fuse_rope: bool):
    """Apply RoPE per the routing rules; returns (q, new_k, cached_k_per_seq).

    cached K is always returned POST-RoPE (the cache holds roped keys). For
    non-fused backends q and new-k are returned POST-RoPE; for the fused path
    they stay PRE-RoPE (the kernel rotates them).
    """
    q, new_k, cached_k = inputs["q"], inputs["new_k"], inputs["cached_k"]
    if case.rope is None:
        return q, new_k, cached_k

    rope_params = _rope_params_from_dict(case.rope)
    is_neox = case.rope.get("is_neox", True)
    Hkv, D = case.num_kv_heads, case.head_dim

    cached_eff = []
    for i, c in enumerate(case.num_cached_tokens):
        if c == 0:
            cached_eff.append(cached_k[i])
            continue
        pos = torch.arange(0, c, dtype=torch.int32, device="cuda")
        roped = apply_rope(cached_k[i].reshape(c, Hkv * D), pos, rope_params, D, is_neox=is_neox)
        cached_eff.append(roped.reshape(c, Hkv, D))

    if not fuse_rope:
        q_pos = make_position_ids(case.seq_lens, case.num_cached_tokens)
        q = apply_rope(q, q_pos, rope_params, D, is_neox=is_neox)
        # Self-attention K shares the query positions; cross-attention K is the
        # encoder side (its own positions, length seq_lens_kv), so it must be
        # roped at encoder positions -- otherwise q_len != kv_len mismatches.
        if case.is_cross:
            k_pos = make_position_ids(case.seq_lens_kv, [0] * case.num_seqs)
        else:
            k_pos = q_pos
        new_k = apply_rope(new_k, k_pos, rope_params, D, is_neox=is_neox)
    return q, new_k, cached_eff


def _expected_new_k_for_cache(case: BackendCase, inputs, new_k: torch.Tensor, *, fuse_rope: bool):
    if not fuse_rope or case.rope is None:
        return new_k

    rope_params = _rope_params_from_dict(case.rope)
    is_neox = case.rope.get("is_neox", True)
    if case.is_cross:
        k_pos = make_position_ids(case.seq_lens_kv, [0] * case.num_seqs)
    else:
        k_pos = make_position_ids(case.seq_lens, case.num_cached_tokens)
    return apply_rope(inputs["new_k"], k_pos, rope_params, case.head_dim, is_neox=is_neox)


def _capture_replay(
    AttentionCls, case, mgr, make_metadata, static_inputs, forward_fn
) -> torch.Tensor:
    """Capture a gen-phase graph once and replay it with the case's inputs.

    Mirrors production graph reuse: the cuda-graph metadata holds pre-allocated,
    fixed-address buffers (seq_lens refreshed via ``copy_``, never reallocated);
    inputs are copied into static buffers. The decode kernel appends the new K/V
    to the same fixed cache slot on every warmup/capture/replay pass, so the
    repeated append is idempotent. Shared by the standard decode and the absorbed
    MLA generation paths -- they differ only in ``static_inputs`` / ``forward_fn``.
    """
    cg_md = make_metadata(AttentionCls, case, mgr).create_cuda_graph_metadata(case.num_seqs)
    cg_md.seq_lens = torch.tensor(case.seq_lens, dtype=torch.int)
    cg_md.num_contexts = 0
    cg_md.prepare()

    bufs = {k: torch.zeros_like(v) for k, v in static_inputs.items()}
    for k, v in static_inputs.items():
        bufs[k].copy_(v)

    def _fwd():
        return forward_fn(cg_md, bufs)

    # Warm up on a side stream so metadata-dependent host work that host-syncs
    # (e.g. FlashInfer ``plan()``) runs OUTSIDE the capture region.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _fwd()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = _fwd()
    graph.replay()
    torch.cuda.synchronize()
    return graph_out[: case.nnz_q].contiguous().clone()


def run_backend(
    case: BackendCase,
    backend: str,
    inputs,
    *,
    kv_dtype: torch.dtype,
    fuse_rope: bool = False,
    cuda_graph: bool = False,
    kv_layout: str = "NHD",
    sparse_indexer_outputs: Optional[Dict[str, torch.Tensor]] = None,
    sparse_topk_override: Optional[Dict[str, torch.Tensor]] = None,
    sparse_native_outputs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Run one backend on ``case`` and return ``[nnz_q, num_heads*head_dim]``.

    With ``cuda_graph=True`` (only valid for a gen-only batch) the forward is
    captured into a CUDA graph and replayed, exercising the graph-reuse path.
    ``kv_layout`` is the paged-cache block layout to fill + read ("NHD"/"HND");
    the caller passes a layout the backend supports (gated by the capability
    matrix). MLA cases are dispatched to the absorbed-generation path.
    """
    if case.is_sparse:
        if case.is_mla:
            return _run_sparse_mla_backend(
                case,
                backend,
                inputs,
                kv_layout=kv_layout,
                indexer_outputs=sparse_indexer_outputs,
                indexer_topk_override=sparse_topk_override,
                native_outputs=sparse_native_outputs,
            )
        raise ValueError(f"Unsupported sparse contract: is_mla={case.is_mla}")

    if case.is_mla:
        if case.is_context_only:
            return _run_mla_context_backend(case, backend, inputs, kv_layout=kv_layout)
        return _run_mla_gen_backend(
            case, backend, inputs, kv_layout=kv_layout, cuda_graph=cuda_graph
        )

    AttentionCls = get_attention_backend(backend)
    H, Hkv, D = case.num_heads, case.num_kv_heads, case.head_dim
    request_ids = list(range(case.num_seqs))
    mask = PredefinedAttentionMask.CAUSAL if case.causal else PredefinedAttentionMask.FULL

    q, new_k, cached_k = _maybe_rope(case, inputs, fuse_rope=fuse_rope)
    new_v = inputs["new_v"]
    cached_v = inputs["cached_v"]
    cache_new_k = _expected_new_k_for_cache(case, inputs, new_k, fuse_rope=fuse_rope)
    cache_atol, cache_rtol = _tolerances(case, kv_dtype) if fuse_rope else (0.0, 0.0)
    expected_cache_tokens = None
    if case.kv_dtype != "nvfp4":
        expected_cache_tokens = _expected_standard_cache_tokens(case, cache_new_k, new_v)

    pos_embd_params = None
    if fuse_rope and case.rope is not None:
        rope_params = _rope_params_from_dict(case.rope)
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
            is_neox=case.rope.get("is_neox", True),
        )

    # Build via the real factory (the same path the Attention/MLA modules use),
    # so the env-gated capture hook in create_attention is exercised too.
    attn = create_attention(
        backend,
        layer_idx=0,
        num_heads=H,
        head_dim=D,
        num_kv_heads=Hkv,
        quant_config=_quant_config(kv_dtype),
        q_scaling=case.q_scaling,
        pos_embd_params=pos_embd_params,
    )

    forward_args = AttentionForwardArgs(
        attention_mask=mask,
        attention_window_size=case.sliding_window if case.sliding_window else None,
    )

    # TRTLLM fuses QKV for self-attention, but cross-attention needs separate
    # q/k/v (it sets is_fused_qkv = not is_cross and k is None).
    use_fused_qkv = AttentionCls.support_fused_qkv() and not case.is_cross

    mgr = None
    if case.cache == "none":
        metadata = AttentionCls.Metadata(
            max_num_requests=case.num_seqs,
            max_num_tokens=case.max_num_tokens,
            kv_cache_manager=None,
            mapping=None,
            runtime_features=None,
        )
        metadata.seq_lens = torch.tensor(case.seq_lens, dtype=torch.int)
        metadata.num_contexts = case.num_seqs
        metadata.request_ids = torch.tensor(request_ids, dtype=torch.int)
        metadata.max_seq_len = max(case.seq_lens)
        metadata.prepare()
    else:
        mgr = _build_kv_cache_manager(case, backend, kv_dtype)
        mgr.add_dummy_requests(request_ids, case.token_nums)
        # The cached prefix must be filled in the layout the backend reads:
        # Vanilla uses NHD, TRTLLM HND, FlashInfer whatever metadata.kv_layout
        # says. The caller passes a supported ``kv_layout`` (capability-gated).
        fill_kv_cache_logical(mgr, 0, request_ids, cached_k, cached_v, kv_layout=kv_layout)

        def create_metadata(AttentionCls, case, mgr, *, num_contexts: int = 0):
            seq_lens_kv = torch.tensor(case.seq_lens_kv, dtype=torch.int) if case.is_cross else None
            return AttentionCls.Metadata(
                num_contexts=num_contexts,
                kv_cache_params=KVCacheParams(
                    use_cache=True, num_cached_tokens_per_seq=case.num_cached_tokens
                ),
                seq_lens=torch.tensor(case.seq_lens, dtype=torch.int),
                seq_lens_kv=seq_lens_kv,
                max_num_requests=case.num_seqs,
                max_num_tokens=case.max_num_tokens,
                kv_cache_manager=mgr,
                request_ids=request_ids,
                prompt_lens=case.seq_lens if case.is_cross else case.token_nums,
                kv_layout=kv_layout,
            )

        if cuda_graph:
            static = (
                {"q": torch.cat([q, new_k, new_v], dim=-1)}
                if use_fused_qkv
                else {"q": q, "k": new_k, "v": new_v}
            )

            def _cg_fwd(md, b):
                out = attn.forward(
                    b["q"],
                    b.get("k"),
                    b.get("v"),
                    md,
                    forward_args=forward_args,
                )
                return out[0] if isinstance(out, tuple) else out

            try:
                out = _capture_replay(
                    AttentionCls,
                    case,
                    mgr,
                    create_metadata,
                    static,
                    _cg_fwd,
                )
                if expected_cache_tokens is not None:
                    _assert_cache_contains_new_tokens(
                        mgr,
                        0,
                        request_ids,
                        case.kv_new_lens,
                        case.num_cached_tokens,
                        expected_cache_tokens,
                        kv_layout=kv_layout,
                        cache_kind="kv",
                        atol=cache_atol,
                        rtol=cache_rtol,
                    )
                return out
            finally:
                mgr.shutdown()
        metadata = create_metadata(AttentionCls, case, mgr, num_contexts=case.num_contexts)
        metadata.prepare()
        if case.is_cross:
            # TRTLLM cross reads metadata.cu_q_seqlens / cu_kv_seqlens (indptr,
            # num_seqs+1). The model engine sets these; the standalone harness
            # must too, else kv indexing falls back to q lengths and breaks when
            # q_len != kv_len. cu_kv uses total KV per seq (cached + new encoder).
            def _cu(lengths):
                return torch.tensor(
                    [0, *torch.tensor(lengths).cumsum(0).tolist()],
                    dtype=torch.int32,
                    device="cuda",
                )

            metadata.cu_q_seqlens = _cu(case.seq_lens)
            metadata.cu_kv_seqlens = _cu(case.token_nums)

    try:
        if use_fused_qkv:
            qkv = torch.cat([q, new_k, new_v], dim=-1)
            out = attn.forward(qkv, None, None, metadata, forward_args=forward_args)
        else:
            out = attn.forward(q, new_k, new_v, metadata, forward_args=forward_args)
        if isinstance(out, tuple):
            out = out[0]
        if mgr is not None and expected_cache_tokens is not None:
            _assert_cache_contains_new_tokens(
                mgr,
                0,
                request_ids,
                case.kv_new_lens,
                case.num_cached_tokens,
                expected_cache_tokens,
                kv_layout=metadata.kv_layout,
                cache_kind="kv",
                atol=cache_atol,
                rtol=cache_rtol,
            )
        return out[: case.nnz_q].contiguous()
    finally:
        if mgr is not None:
            mgr.shutdown()


def run_case(case: BackendCase, *, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Run the VanillaAttention golden and every supported backend; assert match.

    Handles both standard attention and absorbed-MLA generation (dispatched
    inside ``run_backend``). The Vanilla golden always runs in its native NHD
    layout; each backend under test runs in the case's requested layout (or HND).
    Sparse cases compare each backend's Indexer selection against Vanilla by
    set overlap, run attention with Vanilla's exact selection to isolate the
    attention implementation, and retain a bounded end-to-end check with the
    backend's own selection.
    A gen-only batch is additionally replayed through a captured CUDA graph.

    Returns the per-backend outputs (including ``"VANILLA"`` golden) for callers
    that want the raw tensors (e.g. the minimizer).
    """
    is_mla = case.is_mla
    if case.is_sparse:
        _validate_sparse_case(case)
        if case.is_mla:
            inputs = generate_sparse_mla_inputs(case, seed)
        else:
            raise ValueError(f"Unsupported sparse contract: is_mla={case.is_mla}")
    elif is_mla:
        if case.is_context_only:
            inputs = generate_mla_context_inputs(case, seed)
        else:
            inputs = generate_mla_gen_inputs(case, seed)
    else:
        inputs = generate_inputs(case, seed)
    golden_indexer_outputs = {} if case.is_sparse else None
    indexer_compress_ratio = 1
    if case.is_sparse:
        compress_ratios = getattr(case.sparse_attention_config, "compress_ratios", None)
        if compress_ratios:
            indexer_compress_ratio = _effective_compress_ratio_divisor(
                _select_indexer_compress_ratio(compress_ratios)
            )
    golden = run_backend(
        case,
        "VANILLA",
        inputs,
        kv_dtype=case.compute_dtype,
        kv_layout="NHD",
        sparse_indexer_outputs=golden_indexer_outputs,
    )
    results = {"VANILLA": golden}

    # Evaluate every supported backend before asserting, so one backend's
    # mismatch does not mask another's.
    failures = []
    for backend in BACKENDS_UNDER_TEST:
        if unsupported_reason(backend, case) is not None:
            continue
        kv_dtype = case.compute_dtype if (is_mla or case.cache == "none") else case.kv_torch_dtype
        # Fused RoPE only applies to TRTLLM (the sole support_fused_rope backend).
        fuse_rope = case.fused_rope and backend == "TRTLLM"
        layout = case.kv_layout or "HND"  # native for TRTLLM/FlashInfer
        backend_indexer_outputs = {} if case.is_sparse else None
        backend_native_outputs = {} if case.is_sparse else None
        out = run_backend(
            case,
            backend,
            inputs,
            kv_dtype=kv_dtype,
            fuse_rope=fuse_rope,
            kv_layout=layout,
            sparse_indexer_outputs=backend_indexer_outputs,
            sparse_topk_override=golden_indexer_outputs,
            sparse_native_outputs=backend_native_outputs,
        )
        results[backend] = out

        atol, rtol = _tolerances(case, kv_dtype)
        if case.is_sparse:
            try:
                _assert_sparse_indexer_matches_golden(
                    backend_indexer_outputs,
                    golden_indexer_outputs,
                    case=case,
                    compress_ratio=indexer_compress_ratio,
                )
            except AssertionError as exc:
                failures.append(f"[{backend} indexer vs VANILLA indexer golden]\n{exc}")
            native_output = backend_native_outputs["output"]
            results[f"{backend}+native_topk"] = native_output
            try:
                _assert_sparse_end_to_end_matches_golden(
                    native_output,
                    golden,
                    atol=atol,
                    rtol=rtol,
                )
            except AssertionError as exc:
                failures.append(f"[{backend} end-to-end vs VANILLA golden]\n{exc}")
        try:
            torch.testing.assert_close(out, golden, atol=atol, rtol=rtol)
        except AssertionError as exc:
            failures.append(f"[{backend} vs VANILLA golden]\n{exc}")

        # A gen-only batch also exercises the captured-CUDA-graph path
        # (production replays a captured decode graph); it must still match the
        # eager golden. The sparse runner rebuilds each request's logical cache on
        # the host, which is not graph-capturable, so sparse cases are skipped.
        if case.is_gen_only and not case.is_sparse:
            cg_out = run_backend(
                case,
                backend,
                inputs,
                kv_dtype=kv_dtype,
                fuse_rope=fuse_rope,
                kv_layout=layout,
                cuda_graph=True,
            )
            results[f"{backend}+cudagraph"] = cg_out
            try:
                torch.testing.assert_close(cg_out, golden, atol=atol, rtol=rtol)
            except AssertionError as exc:
                failures.append(f"[{backend}+cudagraph vs VANILLA golden]\n{exc}")

    if failures:
        raise AssertionError("\n\n".join(failures))
    return results
