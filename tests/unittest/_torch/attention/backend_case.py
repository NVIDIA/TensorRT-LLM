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

import torch
from backend_capability import BACKEND_CAPS, unsupported_reason
from kv_cache_utils import apply_rope, fill_kv_cache_logical, make_position_ids

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention, get_attention_backend
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import str_dtype_to_torch, torch_dtype_to_binding
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
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
    sparse: str = "off"  # "off" | "degenerate"
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
def _build_mla_kv_cache_manager(case: BackendCase, backend: str):
    """A SELFKONLY KV cache for MLA: one latent head, head_dim kv_lora+qk_rope."""
    d_latent = case.kv_lora_rank + case.qk_rope_head_dim
    paged = BACKEND_CAPS[backend]["paged"]
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
    cls = KVCacheManagerV2 if case.use_kv_cache_manager_v2 else KVCacheManager
    return cls(
        KvCacheConfig(max_tokens=num_blocks * tokens_per_block, enable_block_reuse=False),
        cache_types.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=d_latent,
        tokens_per_block=tokens_per_block,
        max_seq_len=pages_per_seq * tokens_per_block,
        max_batch_size=case.num_seqs,
        mapping=mapping,
        dtype=torch_dtype_to_binding(case.compute_dtype),
    )


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
    bf16 = case.compute_dtype == torch.bfloat16
    if kv_dtype == torch.float8_e4m3fn:
        return (FP8_ATOL + BF16_ATOL, BF16_RTOL) if bf16 else (FP8_ATOL, RTOL)
    if kv_dtype == "nvfp4":
        return (FP4_ATOL + BF16_ATOL, BF16_RTOL) if bf16 else (FP4_ATOL, RTOL)
    if bf16:
        return BF16_ATOL, BF16_RTOL
    return ATOL, RTOL


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
) -> torch.Tensor:
    """Run one backend on ``case`` and return ``[nnz_q, num_heads*head_dim]``.

    With ``cuda_graph=True`` (only valid for a gen-only batch) the forward is
    captured into a CUDA graph and replayed, exercising the graph-reuse path.
    ``kv_layout`` is the paged-cache block layout to fill + read ("NHD"/"HND");
    the caller passes a layout the backend supports (gated by the capability
    matrix). MLA cases are dispatched to the absorbed-generation path.
    """
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
    A gen-only batch is additionally replayed through a captured CUDA graph.

    Returns the per-backend outputs (including ``"VANILLA"`` golden) for callers
    that want the raw tensors (e.g. the minimizer).
    """
    is_mla = case.is_mla
    if is_mla:
        if case.is_context_only:
            inputs = generate_mla_context_inputs(case, seed)
        else:
            inputs = generate_mla_gen_inputs(case, seed)
    else:
        inputs = generate_inputs(case, seed)
    golden = run_backend(case, "VANILLA", inputs, kv_dtype=case.compute_dtype, kv_layout="NHD")
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
        out = run_backend(
            case, backend, inputs, kv_dtype=kv_dtype, fuse_rope=fuse_rope, kv_layout=layout
        )
        results[backend] = out

        atol, rtol = _tolerances(case, kv_dtype)
        try:
            torch.testing.assert_close(out, golden, atol=atol, rtol=rtol)
        except AssertionError as exc:
            failures.append(f"[{backend} vs VANILLA golden]\n{exc}")

        # A gen-only batch also exercises the captured-CUDA-graph path
        # (production replays a captured decode graph); it must still match the
        # eager golden.
        if case.is_gen_only:
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
