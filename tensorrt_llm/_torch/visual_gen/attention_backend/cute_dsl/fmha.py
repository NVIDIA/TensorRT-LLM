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
CuTe DSL (NVIDIA kernels) Dense FMHA Backend for Visual Generation Models

JIT-compiles the dense FMHA kernel and caches the compiled artifact for each kernel configuration.
Expects NHD layout ([B, S, H, D]) and supports float16/bfloat16 inputs. The VSA sparse path uses
VSAAttention from vsa.py instead.
"""

import math
from typing import Any, NamedTuple, Tuple

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from ....attention_backend.interface import PredefinedAttentionMask
from ..interface import AttentionBackend, AttentionTensorLayout

_cute_dsl_import_error: BaseException | None = None
try:
    import cutlass
    from cuda.bindings import driver as cuda_driver
    from cutlass import cute
    from cutlass.cute import typing as cute_typing
    from cutlass.cute.runtime import from_dlpack

    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell.attention import (
        BlackwellFusedMultiHeadAttentionForward,
        BlackwellFusedMultiHeadBlockScaledAttentionForward,
    )
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.helpers import fmha_helpers as fmha_utils
except (ImportError, OSError) as e:
    cutlass = None
    cuda_driver = None
    cute = None
    cute_typing = None
    from_dlpack = None
    BlackwellFusedMultiHeadAttentionForward = None
    BlackwellFusedMultiHeadBlockScaledAttentionForward = None
    fmha_utils = None
    _cute_dsl_import_error = e


SUPPORTED_GPU_ARCHS: Tuple[str, ...] = ("sm_100a", "sm_103a")


# ============================================================================
# Runtime helpers
# ============================================================================


def _check_cute_runtime_available() -> None:
    if _cute_dsl_import_error is None:
        return
    raise ImportError(
        f"CuTe DSL runtime is not available. Import error: {_cute_dsl_import_error}"
    ) from _cute_dsl_import_error


def _get_gpu_arch(device: torch.device | None = None) -> str:
    capability = torch.cuda.get_device_capability(device)
    gpu_arch = f"sm_{capability[0]}{capability[1]}a"
    if gpu_arch not in SUPPORTED_GPU_ARCHS:
        supported = ", ".join(SUPPORTED_GPU_ARCHS)
        raise ValueError(
            f"Unsupported GPU architecture {gpu_arch}. Supported architectures: {supported}."
        )
    return gpu_arch


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
) -> None:
    """Validate the (B, S, H, D) layout the backend feeds the kernel."""
    if not (q.dim() == k.dim() == v.dim() == o.dim() == 4):
        raise ValueError("FMHA expects 4D (B, S, H, D) Q/K/V/O tensors.")
    if q.dtype != k.dtype:
        raise ValueError(f"Q/K dtype mismatch: {q.dtype} vs {k.dtype}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"Batch size mismatch: q={q.shape[0]} vs k={k.shape[0]}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"Q/K head dim mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(f"K/V shape mismatch: {k.shape[:-1]} vs {v.shape[:-1]}")
    expected_o_shape = (*q.shape[:-1], v.shape[-1])
    if tuple(o.shape) != expected_o_shape:
        raise ValueError(f"Output shape mismatch: {tuple(o.shape)} vs {expected_o_shape}")


def _to_cute_tensor(tensor: torch.Tensor, leading_dim: int, cutlass_element_type=None):
    """Wrap a torch tensor as a CuTe tensor.

    For sub-byte / non-torch-dtype elements (FP4 packed as uint8, MXFP8 SF exponents stored as
    uint8), pass `cutlass_element_type` to override the interpretation; the tensor storage must be
    byte-addressable (uint8 / int8). Otherwise the FP8-e4m3 path and the default
    direct-from-dlpack path apply.
    """
    # Match cutlass.torch.cute_tensor_like: set element_type BEFORE mark_layout_dynamic so the
    # layout transformation sees the override-typed tensor and carries it forward.
    if cutlass_element_type is not None:
        cute_tensor = from_dlpack(tensor.view(torch.int8), assumed_align=16)
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        cute_tensor.element_type = cutlass_element_type
        return cute_tensor
    if tensor.dtype == torch.float8_e4m3fn:
        cute_tensor = from_dlpack(tensor.view(torch.int8), assumed_align=16)
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        cute_tensor.element_type = cutlass.Float8E4M3FN
        return cute_tensor
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


# ============================================================================
# JIT compile + cache
# ============================================================================


def _torch_to_cutlass_dtype(t: torch.dtype):
    table = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    }
    try:
        return table[t]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype for CuTe DSL FMHA: {t}") from exc


class _CacheKey(NamedTuple):
    qk_cutlass_dtype: Any
    pv_cutlass_dtype: Any
    out_cutlass_dtype: Any
    qk_acc_dtype: Any
    pv_acc_dtype: Any
    head_dim: int
    head_dim_v: int
    mma_tiler_mn: Tuple[int, int]
    qk_sf_vec: int  # 0 = dense Q/K; 32 = MXFP8; 16 = NVFP4
    is_persistent: bool
    mask_type: Any  # fmha_utils.MaskEnum
    with_lse: bool
    with_sink: bool
    with_scale_v_channels: bool
    has_window: bool
    has_skip_softmax: bool
    use_tma_store: bool
    enable_ex2_emulation: bool
    enable_skip_correction: bool
    gpu_arch_str: str


_COMPILE_CACHE: dict = {}


def clear_cute_dsl_fmha_cache() -> None:
    """Drop all compiled CuTe DSL FMHA kernels (for tests / teardown)."""
    _COMPILE_CACHE.clear()


def _get_or_compile(key: _CacheKey, compile_args: tuple):
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    hd, hd_v = key.head_dim, key.head_dim_v
    head_dim_arg = hd if hd == hd_v else (hd, hd_v)
    if key.qk_sf_vec != 0:
        fmha = BlackwellFusedMultiHeadBlockScaledAttentionForward(
            key.qk_acc_dtype,
            key.pv_acc_dtype,
            key.mma_tiler_mn,
            head_dim_arg,
            key.is_persistent,
            key.mask_type,
            key.enable_ex2_emulation,
            key.enable_skip_correction,
            key.qk_sf_vec,
            use_tma_store=key.use_tma_store,
        )
    else:
        fmha = BlackwellFusedMultiHeadAttentionForward(
            key.qk_acc_dtype,
            key.pv_acc_dtype,
            key.mma_tiler_mn,
            head_dim_arg,
            key.is_persistent,
            key.mask_type,
            key.enable_ex2_emulation,
            key.enable_skip_correction,
            use_tma_store=key.use_tma_store,
        )
    logger.info(
        f"Compiling CuTe DSL FMHA kernel for {key.qk_cutlass_dtype.__name__}/"
        f"{key.pv_cutlass_dtype.__name__} head_dim={key.head_dim} "
        f"mask={key.mask_type.name} persistent={key.is_persistent} "
        f"lse={key.with_lse} on {key.gpu_arch_str} ..."
        f"qk_sf_vec={key.qk_sf_vec} "
    )
    compiled = cute.compile(fmha, *compile_args)
    _COMPILE_CACHE[key] = compiled
    return compiled


@torch.compiler.disable
def cute_dsl_fmha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    *,
    is_causal: bool = False,
    sm_scale: float | None = None,
    window_left: int = -1,
    window_right: int = -1,
    lse: torch.Tensor | None = None,
    scale_q: float | torch.Tensor = 1.0,
    scale_k: float | torch.Tensor = 1.0,
    scale_v: float | torch.Tensor = 1.0,
    scale_v_channels: torch.Tensor | None = None,
    scale_o: float | torch.Tensor = 1.0,
    is_persistent: bool = True,
    skip_softmax_threshold_scale_factor: float | None = None,
    qk_sf_vec: int = 0,
    q_sf: torch.Tensor | None = None,
    k_sf: torch.Tensor | None = None,
    qk_cutlass_dtype: Any = None,
) -> None:
    """JIT-compile (or fetch from cache) and launch the CuTe DSL FMHA kernel.

    Expects contiguous 4D (B, S, H, D) Q/K/V/O tensors with uniform per-batch sequence lengths.
    Varlen via indptr is intentionally not exposed — callers pack uniform-length sequences.

    When `qk_sf_vec` is non-zero, dispatches to the block-scaled kernel class:
    32 selects MXFP8 (Q/K stored as FP8 e4m3, SFs as Float8E8M0FNU uint8 storage);
    16 selects NVFP4 (Q/K stored as packed FP4 in torch.uint8, SFs as Float8E4M3FN in uint8 storage).
    """
    _check_cute_runtime_available()
    _validate_inputs(q, k, v, o)
    if qk_sf_vec != 0:
        if q_sf is None or k_sf is None:
            raise ValueError("Block-scaled path (qk_sf_vec != 0) requires q_sf and k_sf tensors.")
        if not q_sf.is_contiguous() or not k_sf.is_contiguous():
            raise ValueError("q_sf and k_sf must be contiguous.")
    elif scale_v_channels is not None:
        raise ValueError("scale_v_channels is only supported by MXFP8 and NVFP4 kernels.")

    # The kernel hard-codes dense strides in its CuTe layout (fmha.py:447-461) and ignores the
    # input tensor's actual strides, so non-dense inputs (e.g. `qkv.split(...)` views) would
    # be read at wrong offsets. .contiguous() is a no-op when the tensor is already dense.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if not o.is_contiguous():
        raise ValueError("Output tensor `o` must be contiguous (writes happen in place).")
    if lse is not None and not lse.is_contiguous():
        raise ValueError("LSE tensor must be contiguous (writes happen in place).")

    # Delay scalar extraction to inside @torch.compiler.disable decoration
    def _scalar_float(t):
        if isinstance(t, torch.Tensor):
            return t.item()
        else:
            return t

    scale_q = _scalar_float(scale_q)
    scale_k = _scalar_float(scale_k)
    scale_v = _scalar_float(scale_v)
    scale_o = _scalar_float(scale_o)

    # Reshape (B, S, H, D) → (B, S, h_kv, h_r, D) for Q/O and (B, S, h_kv, 1, D) for K/V; LSE
    # goes (B, S, H) → (B, S, h_kv, h_r). Layout matches fmha.py:run() (no head_dim split).
    # For NVFP4 (qk_cutlass_dtype == Float4E2M1FN), Q/K's storage last-dim is head_dim//2.
    # We keep the packed shape through the view; the kernel's FP4 element_type override resolves
    # element-unit strides to the right byte addresses.
    batch_size, seq_len_q, num_heads_q, qk_storage_dim = q.shape
    _, seq_len_kv, num_heads_kv, _ = k.shape
    value_head_dim = v.shape[-1]
    num_head_groups = num_heads_q // num_heads_kv
    is_fp4 = qk_cutlass_dtype is cutlass.Float4E2M1FN
    head_dim = qk_storage_dim * 2 if is_fp4 else qk_storage_dim
    if qk_sf_vec != 0 and head_dim != 128:
        raise ValueError(
            f"MXFP8 / NVFP4 (qk_sf_vec={qk_sf_vec}) currently requires head_dim=128, "
            f"got head_dim={head_dim}."
        )
    if scale_v_channels is not None:
        expected_scale_shape = (num_heads_kv, value_head_dim)
        if tuple(scale_v_channels.shape) != expected_scale_shape:
            raise ValueError(
                f"scale_v_channels must have shape {expected_scale_shape}; "
                f"got {tuple(scale_v_channels.shape)}."
            )
        if scale_v_channels.dtype != torch.float32:
            raise ValueError("scale_v_channels must use torch.float32.")
        if scale_v_channels.device != v.device:
            raise ValueError("scale_v_channels must be on the same device as V.")
        if not scale_v_channels.is_contiguous():
            raise ValueError("scale_v_channels must be contiguous.")

    q_5d = q.view(batch_size, seq_len_q, num_heads_kv, num_head_groups, qk_storage_dim)
    o_5d = o.view(batch_size, seq_len_q, num_heads_kv, num_head_groups, value_head_dim)
    k_5d = k.view(batch_size, seq_len_kv, num_heads_kv, 1, qk_storage_dim)
    v_5d = v.view(batch_size, seq_len_kv, num_heads_kv, 1, value_head_dim)
    lse_4d = (
        lse.view(batch_size, seq_len_q, num_heads_kv, num_head_groups) if lse is not None else None
    )

    # Map is_causal / window args onto the published MaskEnum surface.
    has_window = is_causal or window_left != -1 or window_right != -1
    if is_causal:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK
        ws_l_int = None if window_left == -1 else window_left
        ws_r_int = 0 if window_right == -1 else window_right
    elif has_window:
        mask_type = fmha_utils.MaskEnum.WINDOW_MASK
        ws_l_int = None if window_left == -1 else window_left
        ws_r_int = None if window_right == -1 else window_right
    else:
        mask_type = fmha_utils.MaskEnum.RESIDUAL_MASK
        ws_l_int = None
        ws_r_int = None

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    scale_softmax = scale_q * scale_k * sm_scale
    scale_softmax_log2 = scale_softmax * math.log2(math.exp(1.0))
    scale_output = scale_v / scale_o

    use_skip_softmax = (
        skip_softmax_threshold_scale_factor is not None and skip_softmax_threshold_scale_factor > 0
    )
    skip_threshold_log2 = (
        cute_typing.Float32(math.log2(skip_softmax_threshold_scale_factor / seq_len_kv))
        if use_skip_softmax
        else None
    )

    # For the block-scaled paths Q/K may be FP4 (stored as torch.uint8) or FP8 e4m3; pass
    # qk_cutlass_dtype to override the inferred element type for FP4 (FP8 e4m3 is auto-detected).
    q_cute = _to_cute_tensor(q_5d, leading_dim=4, cutlass_element_type=qk_cutlass_dtype)
    k_cute = _to_cute_tensor(k_5d, leading_dim=4, cutlass_element_type=qk_cutlass_dtype)
    v_cute = _to_cute_tensor(v_5d, leading_dim=4)
    o_cute = _to_cute_tensor(o_5d, leading_dim=4)
    if qk_sf_vec != 0:
        # MXFP8 SFs are Float8E8M0FNU (uint8 storage); NVFP4 SFs are Float8E4M3FN.
        sf_dtype = cutlass.Float8E8M0FNU if qk_sf_vec == 32 else cutlass.Float8E4M3FN
        q_sf_cute = _to_cute_tensor(q_sf, leading_dim=0, cutlass_element_type=sf_dtype)
        k_sf_cute = _to_cute_tensor(k_sf, leading_dim=0, cutlass_element_type=sf_dtype)
        scale_v_channels_cute = (
            _to_cute_tensor(scale_v_channels.view(-1), leading_dim=0)
            if scale_v_channels is not None
            else None
        )
    else:
        q_sf_cute = None
        k_sf_cute = None
        scale_v_channels_cute = None
    # lse_4d is (B, S_q, h_kv, h_r) contiguous → h_r is the stride-1 inner dim (index 3).
    lse_cute = (
        from_dlpack(lse_4d, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        if lse_4d is not None
        else None
    )

    ws_left = None if ws_l_int is None else cute_typing.Int32(ws_l_int)
    ws_right = None if ws_r_int is None else cute_typing.Int32(ws_r_int)

    # problem_size = (b, s_q_max, s_lse_max, s_k_max, h_q, h_k, d, dv); with cum_seqlen_* = None,
    # s_lse_max collapses to s_q (per fmha.py:run()).
    problem_size = (
        batch_size,
        seq_len_q,
        seq_len_q,
        seq_len_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        value_head_dim,
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    gpu_arch_str = _get_gpu_arch(q.device)
    qk_dtype_cache = (
        qk_cutlass_dtype if qk_cutlass_dtype is not None else _torch_to_cutlass_dtype(q.dtype)
    )
    # SM100 needs ex2 emulation; SM103 (and any other SM10X SKU) does not.
    enable_ex2_emulation = gpu_arch_str == "sm_100a"

    key = _CacheKey(
        qk_cutlass_dtype=qk_dtype_cache,
        pv_cutlass_dtype=_torch_to_cutlass_dtype(v.dtype),
        out_cutlass_dtype=_torch_to_cutlass_dtype(o.dtype),
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        head_dim=head_dim,
        head_dim_v=value_head_dim,
        mma_tiler_mn=(128, 128),
        qk_sf_vec=qk_sf_vec,
        is_persistent=is_persistent,
        mask_type=mask_type,
        with_lse=lse is not None,
        with_sink=False,
        with_scale_v_channels=scale_v_channels is not None,
        has_window=has_window,
        has_skip_softmax=use_skip_softmax,
        use_tma_store=True,
        enable_ex2_emulation=enable_ex2_emulation,
        enable_skip_correction=True,
        gpu_arch_str=gpu_arch_str,
    )

    if qk_sf_vec != 0:
        launch_args = (
            q_cute,
            k_cute,
            q_sf_cute,
            k_sf_cute,
            v_cute,
            o_cute,
            problem_size,
            None,  # cum_seqlen_q
            None,  # cum_seqlen_k
            lse_cute,
            None,  # sink
            cute_typing.Float32(scale_softmax_log2),
            cute_typing.Float32(scale_softmax),
            cute_typing.Float32(scale_output),
            scale_v_channels_cute,
            skip_threshold_log2,
            ws_left,
            ws_right,
            None,  # skip_softmax_count
            None,  # total_softmax_count
            stream,
            False,  # use_pdl
        )
    else:
        launch_args = (
            q_cute,
            k_cute,
            v_cute,
            o_cute,
            problem_size,
            None,  # cum_seqlen_q
            None,  # cum_seqlen_k
            lse_cute,
            None,  # sink
            cute_typing.Float32(scale_softmax_log2),
            cute_typing.Float32(scale_softmax),
            cute_typing.Float32(scale_output),
            skip_threshold_log2,
            ws_left,
            ws_right,
            None,  # skip_softmax_count
            None,  # total_softmax_count
            stream,
            False,  # use_pdl
        )

    compiled = _get_or_compile(key, launch_args)
    compiled(*launch_args)


# ============================================================================
# Block-scaled (MXFP8 / NVFP4) Q/K quantization
# ============================================================================


_FP8_E4M3_MAX = 448.0  # FP8 e4m3 max magnitude
_FP4_E2M1_MAX = 6.0  # FP4 e2m1 max magnitude


def _quantize_fp8_v(
    v_bshd: torch.Tensor, per_head_channel: bool
) -> Tuple[torch.Tensor, float | torch.Tensor, torch.Tensor | None]:
    """Quantize V to FP8 with either one tensor scale or an (H, D) scale tensor."""
    if per_head_channel:
        v_qscale = _FP8_E4M3_MAX / v_bshd.float().abs().amax(dim=(0, 1)).clamp(min=1e-3)
        v_quantized = (v_bshd * v_qscale).to(torch.float8_e4m3fn)
        return v_quantized, 1.0, v_qscale.reciprocal().contiguous()

    v_qscale = _FP8_E4M3_MAX / v_bshd.abs().amax().clamp(min=1e-3)
    v_quantized = (v_bshd * v_qscale).to(torch.float8_e4m3fn)
    return v_quantized, v_qscale.reciprocal(), None


def _quantize_blockscaled_one(
    x_bshd: torch.Tensor, qk_sf_vec: int
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Quantize a single (B, S, H, D) bf16/fp16 tensor for the block-scaled kernel.

    Steps:
      1. Transpose to (B, H, S, D) — GEMM-compatible (num_heads is batch-like).
      2. Pad S -> S_pad (multiple of 128) along the row axis.
      3. Invoke the TRT-LLM op with proper options.
      4. Reshape the quantized data, unpad to S, and transpose back to BSHD.
         The SF tensor stays at S_pad per kernel constraints.

    The returned data tensor's last dim is the *storage* dim: D for MXFP8 ; D/2 for NVFP4.

    Returns:
        x_q:    quantized data in (B, S, H, D_storage);
        x_sf:   swizzled SF tensor produced by the op (kept padded to S_pad).
        scale:  scalar dequant factor to fold into scale_softmax.
                1.0 for MXFP8 (per-block SFs carry the full range);
                amax/(448*6) for NVFP4 (the per-tensor global scale).
    """
    if x_bshd.dim() != 4:
        raise ValueError(f"_quantize_blockscaled_one expects (B, S, H, D); got {x_bshd.shape}")
    batch_size, seq_len, num_heads, head_dim = x_bshd.shape

    x_bhsd = x_bshd.transpose(1, 2).contiguous()  # (B, H, S, D)
    s_pad = ((seq_len + 127) // 128) * 128
    if s_pad != seq_len:
        pad = torch.zeros(
            batch_size,
            num_heads,
            s_pad - seq_len,
            head_dim,
            dtype=x_bhsd.dtype,
            device=x_bhsd.device,
        )
        x_bhsd = torch.cat([x_bhsd, pad], dim=2)
    x_2d = x_bhsd.reshape(batch_size * num_heads * s_pad, head_dim)

    if qk_sf_vec == 32:
        # MXFP8: per-32-element block, UE8M0 SFs in swizzled layout.
        x_q_2d, x_sf = torch.ops.trtllm.mxfp8_quantize(x_2d, True, alignment=32)
        # x_q_2d shape: (M, D) fp8_e4m3fn; storage last-dim == logical head_dim.
        x_q = x_q_2d.view(batch_size, num_heads, s_pad, head_dim)
        x_q = x_q[:, :, :seq_len, :].transpose(1, 2).contiguous()  # (B, S, H, D)
        return x_q, x_sf, 1.0

    if qk_sf_vec == 16:
        # NVFP4: per-16-element block, E4M3 SFs in swizzled layout; per-tensor global scale folded
        # into the returned `scale` (caller multiplies it into scale_softmax via scale_q / scale_k).
        amax = x_2d.float().abs().amax().clamp(min=1e-6)
        global_sf = (_FP8_E4M3_MAX * _FP4_E2M1_MAX) / amax
        global_sf_t = global_sf.to(torch.float32).reshape(1)
        x_q_2d, x_sf = torch.ops.trtllm.fp4_quantize(x_2d, global_sf_t, 16, False)
        # x_q_2d shape: (M, D/2) uint8 — natural packed layout (2 FP4 / byte).
        head_dim_packed = head_dim // 2
        x_q = x_q_2d.view(batch_size, num_heads, s_pad, head_dim_packed)
        x_q = x_q[:, :, :seq_len, :].transpose(1, 2).contiguous()  # (B, S, H, D/2)
        return x_q, x_sf, amax / (_FP8_E4M3_MAX * _FP4_E2M1_MAX)

    raise ValueError(f"Unsupported qk_sf_vec={qk_sf_vec}; expected 0, 16, or 32.")


# ============================================================================
# VisualGen AttentionBackend class
# ============================================================================


class CuTeDSLAttention(AttentionBackend):
    """
    CuTe DSL (NVIDIA kernels) backend for diffusion models.

    JIT-compiles BlackwellFusedMultiHeadAttentionForward on first use and caches the
    compiled artifact per (dtype, mask, head_dim, ...) configuration.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        skip_softmax_threshold_scale: float | None = None,
        **kwargs,
    ):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or num_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        self.skip_softmax_threshold_scale = skip_softmax_threshold_scale
        self.scale = 1.0 / math.sqrt(head_dim)

        # CuTe DSL expects [B, S, H, D] format
        self._preferred_layout = AttentionTensorLayout.NHD

    def _smooth_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        alpha: float = 0.7,
    ):
        """Smooth-Smooth technique:
        Performs SmoothQuant-like handling for Qk, leveraging the additional fact that
        K - K_mean doesn't affect attention result.
        """
        k = k.unflatten(2, (-1, 1))  # (B, S, H_K, 1, D)
        q = q.unflatten(2, (k.shape[2], -1))  # (B, S, H_K, H_R, D)
        q_max = (
            q.abs().amax(dim=(0, 1, 3), keepdim=True).float().clamp_min(1e-4)
        )  # (1, 1, H_K, 1, D)
        k_max = (
            k.abs().amax(dim=(0, 1, 3), keepdim=True).float().clamp_min(1e-4)
        )  # (1, 1, H_K, 1, D)
        s = (q_max.pow(alpha) / k_max.pow(1 - alpha)).clamp(1e-4, 1e4)
        q = q * s.reciprocal().bfloat16()
        # Per-channel shift commutes with the per-channel smooth scale `s`.
        k = k - k.mean(dim=(0, 1, 3), keepdim=True)
        k = k * s.bfloat16()
        return q.flatten(2, 3), k.flatten(2, 3)

    def _prepare_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.dtype]:
        """Cast inputs to CuTeDSL-compatible dtype and resolve causal flag."""
        if _cute_dsl_import_error is not None:
            raise ImportError(
                f"CuTe DSL kernels are not available. Import error: {_cute_dsl_import_error}"
            ) from _cute_dsl_import_error

        is_causal = attention_mask == PredefinedAttentionMask.CAUSAL

        # Perform QK-smoothing if Bmm1 is to be quantized.
        qac = self.quant_attention_config
        smooth_qk = qac is not None and qac.qk_dtype not in ["bf16", "fp16"]

        # Published kernel supports float16 and bfloat16 only.
        origin_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        if smooth_qk:
            q, k = self._smooth_qk(q, k)
        return q, k, v, is_causal, origin_dtype

    # cute_dsl_fmha_fwd is already @torch.compiler.disable'd, so torch.compile may still fuse
    # preceding linear/norm with the V quantization below.
    def _fwd(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len_q, num_heads, _ = q.shape
        value_head_dim = v.shape[-1]
        out = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            value_head_dim,
            dtype=q.dtype,
            device=q.device,
        )
        lse = torch.empty(
            batch_size,
            seq_len_q,
            num_heads,
            dtype=torch.float32,
            device=q.device,
        )

        # V is tensor-scaled by default. MXFP8/NVFP4 with v_block_size=1 use an (H, D) scale.
        scale_v = kwargs.get("scale_v", 1.0)
        scale_q = kwargs.get("scale_q", 1.0)
        scale_k = kwargs.get("scale_k", 1.0)
        qac = self.quant_attention_config
        q_sf = k_sf = qk_cutlass_dtype = None
        qk_sf_vec = 0
        scale_v_channels = None
        if qac is not None:
            if qac.qk_dtype in ("mxfp8", "nvfp4"):
                qk_sf_vec = 32 if qac.qk_dtype == "mxfp8" else 16
                q, q_sf, gs_q = _quantize_blockscaled_one(q, qk_sf_vec)
                k, k_sf, gs_k = _quantize_blockscaled_one(k, qk_sf_vec)
                scale_q = scale_q * gs_q
                scale_k = scale_k * gs_k
                qk_cutlass_dtype = cutlass.Float4E2M1FN if qk_sf_vec == 16 else cutlass.Float8E4M3FN
            v, v_dequant_scale, scale_v_channels = _quantize_fp8_v(
                v, per_head_channel=qk_sf_vec != 0 and qac.v_block_size == 1
            )
            scale_v = scale_v * v_dequant_scale

        # Skip softmax.
        skip_softmax_threshold_scale = self.skip_softmax_threshold_scale
        if skip_softmax_threshold_scale is not None and skip_softmax_threshold_scale <= 0.0:
            skip_softmax_threshold_scale = None

        cute_dsl_fmha_fwd(
            q,
            k,
            v,
            out,
            is_causal=is_causal,
            sm_scale=self.scale,
            lse=lse,
            scale_q=scale_q,
            scale_k=scale_k,
            scale_v=scale_v,
            scale_v_channels=scale_v_channels,
            scale_o=kwargs.get("scale_o", 1.0),
            skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale,
            qk_sf_vec=qk_sf_vec,
            q_sf=q_sf,
            k_sf=k_sf,
            qk_cutlass_dtype=qk_cutlass_dtype,
        )
        return out, lse

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using CuTe DSL (NVIDIA kernels).

        Dimensions are derived from tensor shapes (NHD layout: ``[B, S, H, D]``).

        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            v: Value tensor [batch_size, seq_len_kv, num_kv_heads, head_dim]
            attention_mask: Attention mask type (CAUSAL or FULL)

        Returns:
            Output tensor [batch_size, seq_len, num_heads, head_dim]
        """
        output, _ = self.forward_with_lse(q, k, v, attention_mask=attention_mask, **kwargs)
        return output

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both output and log-sum-exp (LSE).

        Returns:
            output: [batch_size, seq_len, num_heads, head_dim]
            lse:    [batch_size, num_heads, seq_len] - log-sum-exp per query position,
                    always in float32. Used for numerically stable combination of
                    partial attention results in Attention2D parallelism.
        """
        q, k, v, is_causal, origin_dtype = self._prepare_inputs(q, k, v, attention_mask)
        output, lse = self._fwd(q, k, v, is_causal, **kwargs)
        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output, lse.transpose(1, 2)

    @classmethod
    def support_lse(cls) -> bool:
        return True

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Return the preferred tensor layout for this backend."""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
