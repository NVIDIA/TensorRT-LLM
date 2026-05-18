# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom ops for the auto path.

`torch.ops.visgen_auto.sdpa` is the lowering target the SDPA pattern matcher
in `rewrite.py` points at. The op is intentionally narrower than
`F.scaled_dot_product_attention`:

- Non-causal only (no `is_causal` parameter).
- No KV cache (no cache id, no `n_rep`).
- No attention mask.
- No dropout.

This matches the diffusers DiT invariant. The body dispatches to whichever
visual-gen attention backend the rewriter selected (the `backend` kwarg on
the FX node, threaded through from `RewritePolicy.attention_backend`).

**Backend-generic dispatcher.** Backend construction goes through visual_gen's
own factory (`attention_backend/utils.py:get_visual_gen_attention_backend`),
so any class that implements the `AttentionBackend` ABC and registers in
that factory is reachable from the auto path without code changes here.
Layout selection reads `backend.preferred_layout` (HND vs NHD) instead of
hard-coding per backend name. Future parallel backends (Ulysses, Ring, Star,
Attention2D) plug in via the same path — they just need to declare a
`preferred_layout` and implement `forward(q, k, v, **kwargs)`.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.library
import torch.nn.functional as F

_BACKEND_CACHE: dict[tuple, object] = {}

# Process-wide current `VisualGenMapping`. Set by `AutoDiffusersPipeline.__init__`
# when multi-GPU is requested. Read inside `_get_backend` to decide whether to
# wrap an inner backend with `UlyssesAttention`. Mapping objects can't be passed
# through FX node kwargs (process groups aren't serializable), so we route via a
# module-level slot — analogous to how the handwritten path reads
# `model_config.visual_gen_mapping` at construction time.
_CURRENT_MAPPING = None


def set_current_mapping(mapping) -> None:
    """Register the active VisualGenMapping for backend construction."""
    global _CURRENT_MAPPING
    _CURRENT_MAPPING = mapping
    # Mapping changed → invalidate cached backends so the next `_get_backend`
    # constructs with the new ulysses_size / process_group.
    _BACKEND_CACHE.clear()


def get_current_mapping():
    return _CURRENT_MAPPING


def _get_backend(name: str, num_heads: int, head_dim: int, dtype: torch.dtype):
    """Construct (and cache) a visual_gen `AttentionBackend` for `name`.

    Uses `attention_backend/utils.py:get_visual_gen_attention_backend` so any
    backend registered in that factory (current: VANILLA, TRTLLM, FA4; future:
    Ulysses, Ring, Star, Attention2D, ...) is reachable without per-backend
    branches here.

    When a multi-GPU `VisualGenMapping` is active and `ulysses_size > 1`, the
    inner backend is constructed with `num_heads // ulysses_size` (Ulysses
    shards heads across the inner backend) and wrapped with
    `UlyssesAttention(inner, vgm.ulysses_group)`. The wrapper handles the
    sequence ↔ head all-to-all on each forward call.
    """
    vgm = _CURRENT_MAPPING
    ulysses_size = getattr(vgm, "ulysses_size", 1) if vgm is not None else 1

    key = (name, num_heads, head_dim, dtype, ulysses_size)
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    from ..attention_backend.utils import get_visual_gen_attention_backend

    # Inner backend's head count is sharded across the Ulysses dim. When
    # ulysses_size == 1 the math collapses to the plain single-GPU path.
    inner_num_heads = num_heads // ulysses_size

    cls = get_visual_gen_attention_backend(name)
    init_kwargs = dict(
        layer_idx=0,
        num_heads=inner_num_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    # TRTLLM mandates `attention_metadata_state` at construction (it raises
    # otherwise). Provide a fresh per-site state — populated lazily on first
    # forward. Other backends ignore the extra kwarg via their `**kwargs`.
    init_kwargs["attention_metadata_state"] = {"metadata": None, "capacity": (0, 0)}

    backend = cls(**init_kwargs)

    # Wrap with Ulysses sequence parallelism if requested. Composes around the
    # inner backend — UlyssesAttention's `preferred_layout` is NHD; the
    # dispatcher's input/output normalization in `_dispatch_attention` reads
    # `.preferred_layout` after the wrap, so layout handling stays uniform.
    if ulysses_size > 1:
        from ..attention_backend.parallel import UlyssesAttention

        backend = UlyssesAttention(
            inner_backend=backend,
            process_group=vgm.ulysses_group,
        )

    _BACKEND_CACHE[key] = backend
    return backend


def _dispatch_attention(
    backend_obj, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Call `backend_obj.forward(q, k, v)` and normalize input/output to HND.

    The captured graph hands us HND `(B, H, S, D)` — what
    `F.scaled_dot_product_attention` receives from a Diffusers DiT block.
    Different backends prefer different layouts; we drive the conversion from
    `backend.preferred_layout` so no branches per backend name are needed.

    Output normalization: most backends return their preferred layout (4D),
    but TRTLLM returns flattened `(B, S, H*D)` (3D). Detect and reshape.
    """
    from ..attention_backend.interface import AttentionTensorLayout

    layout = backend_obj.preferred_layout
    B, H, S_q, D = q.shape
    S_kv = k.shape[2]

    if layout == AttentionTensorLayout.NHD:
        q_in = q.transpose(1, 2).contiguous()  # (B, S_q, H, D)
        k_in = k.transpose(1, 2).contiguous()
        v_in = v.transpose(1, 2).contiguous()
        # TRTLLM consumes batch_size / seq_len kwargs; other backends accept
        # them via the ABC's `**kwargs` and ignore. See `AttentionBackend`
        # docstring in `attention_backend/interface.py`.
        out = backend_obj.forward(
            q_in,
            k_in,
            v_in,
            batch_size=B,
            seq_len=S_q,
            seq_len_kv=S_kv,
        )
        if out.dim() == 3:  # TRTLLM returns (B, S, H*D)
            out = out.reshape(B, S_q, H, D)
        return out.transpose(1, 2).contiguous()  # → HND
    # HND-preferring backend (VANILLA today) — no conversion needed.
    return backend_obj.forward(q, k, v)


@torch.library.custom_op("visgen_auto::sdpa", mutates_args=())
def visgen_auto_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    backend: str = "VANILLA",
) -> torch.Tensor:
    """Non-causal, cacheless joint MM-DiT attention.

    Tensors are in `[B, H, S, D]` (HND) layout — what `F.scaled_dot_product_attention`
    receives from a Diffusers DiT attention block.

    Args:
        q: `(B, H, S_q, D)` query.
        k: `(B, H_kv, S_kv, D)` key.
        v: `(B, H_kv, S_kv, D)` value.
        scale: explicit softmax scale; `None` falls back to `1/sqrt(D)`.
        backend: one of `"VANILLA"`, `"TRTLLM"`, `"FA4"`. Currently only
            VANILLA is wired; the others log a warning and fall back.

    Returns:
        `(B, H, S_q, D)` attention output.
    """
    # VanillaAttention bakes scale = 1/sqrt(head_dim) into the instance.
    # If the caller passes a non-default scale, route through F.SDPA directly
    # — preserves correctness while keeping the common path fast.
    if scale is not None and not math.isclose(scale, 1.0 / math.sqrt(q.shape[-1])):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    be = _get_backend(backend, num_heads=q.shape[1], head_dim=q.shape[-1], dtype=q.dtype)
    return _dispatch_attention(be, q, k, v)


@visgen_auto_sdpa.register_fake
def _visgen_auto_sdpa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    backend: str = "VANILLA",
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("visgen_auto::dit_qk_norm_rope", mutates_args=())
def visgen_auto_dit_qk_norm_rope(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    interleave: bool,
    norm_q_add_weight: Optional[torch.Tensor] = None,
    norm_k_add_weight: Optional[torch.Tensor] = None,
    num_txt_tokens: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-stream per-head RMSNorm on Q/K + RoPE on Q/K, packed QKV in.

    Wraps `torch.ops.trtllm.fused_dit_qk_norm_rope`, which does the math
    in-register with FP32 cos/sin — equivalent to handwritten Flux's
    `fusedDiTQKNormRopeKernel` path. The pattern matcher in
    The QK-rope fusion pattern matchers in `auto/qk_rope_fusion.py` target this op as the
    lowering site.

    Args:
        qkv: packed QKV ``(B, S, (H_q + H_k + H_v) * D)`` in BF16 — comes
            directly from the QKV-fused linear output. Will be cloned
            internally; caller's tensor is not mutated.
        cos: cosine table ``(S, D)`` or any shape reshapeable to that,
            FP32 inside the kernel (cast if needed).
        sin: sine table, same shape as ``cos``.
        norm_q_weight: ``(D,)`` per-head RMSNorm weight for Q.
        norm_k_weight: ``(D,)`` per-head RMSNorm weight for K.
        num_heads_q, num_heads_k, num_heads_v: head counts.
        head_dim: per-head dimension D.
        eps: RMSNorm epsilon.
        interleave: True for Flux-style adjacent-pair rotation
            (`x.reshape(..., D//2, 2).unbind(-1)`), False for rotate-half.

    Returns:
        ``(Q, K, V)`` each shaped ``(B, H, S, D)``, ready for SDPA.
    """
    B, S, total_dim = qkv.shape
    expected_total = (num_heads_q + num_heads_k + num_heads_v) * head_dim
    assert total_dim == expected_total, (
        f"QKV total dim {total_dim} != expected "
        f"({num_heads_q} + {num_heads_k} + {num_heads_v}) * {head_dim} = {expected_total}"
    )

    # Clone for in-place safety: callers may still hold references to the
    # original QKV (e.g., the FX node feeding this op has other uses).
    qkv_2d = qkv.reshape(B * S, total_dim).contiguous().clone()

    cos_2d = cos.reshape(-1, head_dim).float().contiguous()
    sin_2d = sin.reshape(-1, head_dim).float().contiguous()
    assert cos_2d.shape[0] == S, f"cos seq dim {cos_2d.shape[0]} != S {S}"
    assert sin_2d.shape == cos_2d.shape, "sin/cos shape mismatch"

    cos_tiled = cos_2d.repeat(B, 1) if B > 1 else cos_2d
    sin_tiled = sin_2d.repeat(B, 1) if B > 1 else sin_2d

    # Dual-stream: when num_txt_tokens > 0, the kernel uses different norm
    # weights for the first `num_txt_tokens` tokens (encoder side, from
    # `*_add_weight`) versus the rest (hidden side, from `*_weight`).
    # tokens_per_batch tells the kernel the per-batch seq length so it can
    # find the encoder/hidden boundary within each batch via modulo.
    tokens_per_batch = S if num_txt_tokens > 0 else 0
    q_add = norm_q_add_weight.contiguous() if norm_q_add_weight is not None else None
    k_add = norm_k_add_weight.contiguous() if norm_k_add_weight is not None else None

    torch.ops.trtllm.fused_dit_qk_norm_rope(
        qkv_2d,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        norm_q_weight.contiguous(),
        norm_k_weight.contiguous(),
        q_add,
        k_add,
        cos_tiled,
        sin_tiled,
        num_txt_tokens,
        interleave,
        tokens_per_batch,
    )

    q_dim = num_heads_q * head_dim
    k_dim = num_heads_k * head_dim

    qkv_3d = qkv_2d.reshape(B, S, total_dim)
    q = qkv_3d[..., :q_dim].reshape(B, S, num_heads_q, head_dim).transpose(1, 2).contiguous()
    k = (
        qkv_3d[..., q_dim : q_dim + k_dim]
        .reshape(B, S, num_heads_k, head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    v = (
        qkv_3d[..., q_dim + k_dim :]
        .reshape(B, S, num_heads_v, head_dim)
        .transpose(1, 2)
        .contiguous()
    )
    return q, k, v


@visgen_auto_dit_qk_norm_rope.register_fake
def _visgen_auto_dit_qk_norm_rope_fake(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    interleave: bool,
    norm_q_add_weight: Optional[torch.Tensor] = None,
    norm_k_add_weight: Optional[torch.Tensor] = None,
    num_txt_tokens: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, _ = qkv.shape
    q = qkv.new_empty(B, num_heads_q, S, head_dim)
    k = qkv.new_empty(B, num_heads_k, S, head_dim)
    v = qkv.new_empty(B, num_heads_v, S, head_dim)
    return q, k, v


@torch.library.custom_op("visgen_auto::all_gather_seq", mutates_args=())
def visgen_auto_all_gather_seq(
    x: torch.Tensor, dim: int = 1, ulysses_size: int = 1
) -> torch.Tensor:
    """All-gather a sequence-sharded tensor along `dim` across the current
    Ulysses process group.

    `ulysses_size` is an explicit kwarg (not derived from a global) so the
    `register_fake` meta below is deterministic over its args — Inductor's
    stride propagation relies on this when the captured graph is later
    re-traced under `torch.compile`. Earlier versions read `ulysses_size`
    from `_CURRENT_MAPPING`; if the global changed between capture and
    Inductor re-trace, codegen produced buggy strides. The runtime group
    is still resolved from `_CURRENT_MAPPING.ulysses_group` (a process
    handle is not a constexpr).

    `pre_capture_patch` in each family closes over `ulysses_size` as a
    Python int and passes it here, so the value gets baked into the
    captured FX graph as a literal node arg.

    Single-rank fast path (`ulysses_size == 1`) is a no-op clone so the
    op is safe to use without distributed init.

    `dim` defaults to 1 (WAN/SD3 callers); LTX-2's split-mode RoPE tensors
    have seq at dim 2.
    """
    if ulysses_size == 1:
        return x.clone()

    vgm = _CURRENT_MAPPING
    if vgm is None or getattr(vgm, "ulysses_group", None) is None:
        raise RuntimeError(
            "visgen_auto.all_gather_seq called with ulysses_size > 1 but no "
            "VisualGenMapping is active (or it has no ulysses_group). Call "
            "`auto.ops.set_current_mapping(vgm)` before running the captured "
            "graph."
        )

    import torch.distributed as dist

    x = x.contiguous()
    gathered = [torch.empty_like(x) for _ in range(ulysses_size)]
    dist.all_gather(gathered, x, group=vgm.ulysses_group)
    return torch.cat(gathered, dim=dim)


@visgen_auto_all_gather_seq.register_fake
def _visgen_auto_all_gather_seq_fake(
    x: torch.Tensor, dim: int = 1, ulysses_size: int = 1
) -> torch.Tensor:
    # Deterministic over args — no global lookup. Inductor uses this fake
    # meta for stride propagation across torch.compile re-traces.
    out_shape = list(x.shape)
    out_shape[dim] = out_shape[dim] * ulysses_size
    return x.new_empty(out_shape)
