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
from capability_matrix import BACKEND_CAPS, unsupported_reason
from kv_cache_utils import apply_rope, fill_kv_cache_logical, make_position_ids

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention, get_attention_backend
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager, KVCacheManagerV2
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

_STR_TO_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "nvfp4": "nvfp4",  # sentinel; not a torch dtype
}

_BINDINGS_DTYPE = {
    torch.float16: tensorrt_llm.bindings.DataType.HALF,
    torch.bfloat16: tensorrt_llm.bindings.DataType.BF16,
    torch.float8_e4m3fn: tensorrt_llm.bindings.DataType.FP8,
}

# Backends compared against the VanillaAttention golden.
BACKENDS_UNDER_TEST = ("TRTLLM", "FLASHINFER")


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
    # non-causal); TRTLLM is skipped (it asserts no cross), FlashInfer/Vanilla run.
    seq_lens_kv: Optional[List[int]] = None

    dtype: str = "float16"
    # KV cache dtype. None mirrors the compute dtype (the realistic default);
    # set explicitly only to quantize the cache ("float8_e4m3fn" / "nvfp4").
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
    is_mla: bool = False
    # MLA latent dims (only meaningful when is_mla). Carried for capture/replay
    # schema completeness so captured MLA cases round-trip. MLA *numerical*
    # validation lives in test_attention_mla.py: the MLA backend's latent
    # interface (compressed_kv / k_pe / fused_q / latent_cache, with separate
    # ctx/gen references) does not fit run_backend's standard q/k/v path, so the
    # unified harness does not execute MLA cases (run_case is never given one;
    # the replay suite skips is_mla).
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
    def kv_new_lens(self) -> List[int]:
        """New KV tokens per request (encoder side for cross; == seq_lens for self)."""
        return self.seq_lens_kv if self.seq_lens_kv is not None else self.seq_lens

    @property
    def nnz_kv(self) -> int:
        return sum(self.kv_new_lens)

    @property
    def token_nums(self) -> List[int]:
        # Total KV tokens per request after the backend appends the new KV.
        return [c + s for c, s in zip(self.num_cached_tokens, self.kv_new_lens)]

    @property
    def compute_dtype(self) -> torch.dtype:
        return _STR_TO_DTYPE[self.dtype]

    @property
    def kv_torch_dtype(self):
        if self.kv_dtype is None:
            return self.compute_dtype
        return _STR_TO_DTYPE[self.kv_dtype]

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


def generate_inputs(case: BackendCase, seed: int) -> Dict[str, object]:
    """Generate seeded, reproducible pre-RoPE inputs in compute dtype.

    Returns packed ``q`` and new ``k``/``v`` plus per-sequence cached K/V.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    cdt = case.compute_dtype
    H, Hkv, D = case.num_heads, case.num_kv_heads, case.head_dim

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda").to(cdt)

    q = randn(case.nnz_q, H * D)
    # New KV tokens: == q tokens for self-attention, encoder length for cross.
    new_k = randn(case.nnz_kv, Hkv * D)
    new_v = randn(case.nnz_kv, Hkv * D)
    cached_k = [randn(c, Hkv, D) for c in case.num_cached_tokens]
    cached_v = [randn(c, Hkv, D) for c in case.num_cached_tokens]
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
        bindings_dtype = _BINDINGS_DTYPE[kv_dtype]
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


def _quant_config(kv_dtype) -> Optional[QuantConfig]:
    # Quantize only the KV cache (not activations) — there are no projection
    # layers in the standalone backend, and activation FP8 QDQ here yields NaNs.
    if kv_dtype == torch.float8_e4m3fn:
        return QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    if kv_dtype == "nvfp4":
        return QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4)
    return None


def _tolerances(case: "BackendCase", kv_dtype) -> tuple:
    """Dtype-appropriate (atol, rtol). fp8/fp4 dominate; else bf16 is looser."""
    if kv_dtype == torch.float8_e4m3fn:
        return FP8_ATOL, RTOL
    if kv_dtype == "nvfp4":
        return FP4_ATOL, RTOL
    if case.compute_dtype == torch.bfloat16:
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
        new_k = apply_rope(new_k, q_pos, rope_params, D, is_neox=is_neox)
    return q, new_k, cached_eff


def run_backend(
    case: BackendCase, backend: str, inputs, *, kv_dtype: torch.dtype, fuse_rope: bool = False
) -> torch.Tensor:
    """Run one backend on ``case`` and return ``[nnz_q, num_heads*head_dim]``."""
    AttentionCls = get_attention_backend(backend)
    H, Hkv, D = case.num_heads, case.num_kv_heads, case.head_dim
    request_ids = list(range(case.num_seqs))
    mask = PredefinedAttentionMask.CAUSAL if case.causal else PredefinedAttentionMask.FULL

    q, new_k, cached_k = _maybe_rope(case, inputs, fuse_rope=fuse_rope)
    new_v = inputs["new_v"]
    cached_v = inputs["cached_v"]

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

    fwd_kwargs = dict(attention_mask=mask)
    if case.sliding_window:
        fwd_kwargs["attention_window_size"] = case.sliding_window

    use_fused_qkv = AttentionCls.support_fused_qkv()

    mgr = None
    if case.cache == "none":
        metadata = AttentionCls.Metadata(
            max_num_requests=case.num_seqs,
            max_num_tokens=8192,
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
        # Vanilla reads the cache via the NHD ``get_buffers`` view; TRTLLM (C++
        # paged FMHA/XQA) and FlashInfer use the HND head-major block layout.
        # The cached prefix must be filled in the layout the backend reads.
        layout = "NHD" if backend == "VANILLA" else "HND"
        fill_kv_cache_logical(mgr, 0, request_ids, cached_k, cached_v, kv_layout=layout)
        seq_lens_kv = torch.tensor(case.seq_lens_kv, dtype=torch.int) if case.is_cross else None
        metadata = AttentionCls.Metadata(
            num_contexts=case.num_contexts,
            kv_cache_params=KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=case.num_cached_tokens
            ),
            seq_lens=torch.tensor(case.seq_lens, dtype=torch.int),
            seq_lens_kv=seq_lens_kv,
            max_num_requests=case.num_seqs,
            max_num_tokens=8192,
            kv_cache_manager=mgr,
            request_ids=request_ids,
            prompt_lens=case.token_nums,
        )
        metadata.prepare()

    try:
        if use_fused_qkv:
            qkv = torch.cat([q, new_k, new_v], dim=-1)
            out = attn.forward(qkv, None, None, metadata, **fwd_kwargs)
        else:
            out = attn.forward(q, new_k, new_v, metadata, **fwd_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out[: case.nnz_q].contiguous()
    finally:
        if mgr is not None:
            mgr.shutdown()


def run_case(case: BackendCase, *, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Run the VanillaAttention golden and every supported backend; assert match.

    Returns the per-backend outputs (including ``"VANILLA"`` golden) for callers
    that want the raw tensors (e.g. the minimizer).
    """
    inputs = generate_inputs(case, seed)
    golden = run_backend(case, "VANILLA", inputs, kv_dtype=case.compute_dtype)
    results = {"VANILLA": golden}

    # Evaluate every supported backend before asserting, so one backend's
    # mismatch does not mask another's.
    failures = []
    for backend in BACKENDS_UNDER_TEST:
        if unsupported_reason(backend, case) is not None:
            continue
        if backend == "FLASHINFER" and not IS_FLASHINFER_AVAILABLE:
            continue
        kv_dtype = case.compute_dtype if case.cache == "none" else case.kv_torch_dtype
        # Fused RoPE only applies to TRTLLM (the sole support_fused_rope backend).
        fuse_rope = case.fused_rope and backend == "TRTLLM"
        out = run_backend(case, backend, inputs, kv_dtype=kv_dtype, fuse_rope=fuse_rope)
        results[backend] = out

        atol, rtol = _tolerances(case, kv_dtype)
        try:
            torch.testing.assert_close(out, golden, atol=atol, rtol=rtol)
        except AssertionError as exc:
            failures.append(f"[{backend} vs VANILLA golden]\n{exc}")

    if failures:
        raise AssertionError("\n\n".join(failures))
    return results
