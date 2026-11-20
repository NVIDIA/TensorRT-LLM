# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any, NamedTuple

import torch

from tensorrt_llm.logger import logger

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


SUPPORTED_GPU_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")


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


def is_cute_fmha_supported(device: torch.device | None = None) -> bool:
    if _cute_dsl_import_error is not None or not torch.cuda.is_available():
        return False
    try:
        _get_gpu_arch(device)
    except (AssertionError, RuntimeError, ValueError):
        return False
    return True


def _to_cute_tensor(tensor: torch.Tensor, leading_dim: int, cutlass_element_type=None):
    if cutlass_element_type is None and tensor.dtype == torch.float8_e4m3fn:
        cutlass_element_type = cutlass.Float8E4M3FN
    if cutlass_element_type is not None:
        # DLPack carries FP4, E4M3 and the block scales through byte storage.
        tensor = tensor.view(torch.int8)
    wrapped = from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)
    if cutlass_element_type is not None:
        wrapped.element_type = cutlass_element_type
    return wrapped


class _CacheKey(NamedTuple):
    q_dtype: torch.dtype
    v_dtype: torch.dtype
    out_dtype: torch.dtype
    head_dim: int
    qk_sf_vec: int
    tensor_softmax_scale: bool
    tensor_output_scale: bool
    per_channel_v: bool
    skip_softmax: bool
    gpu_arch: str


_COMPILE_CACHE: dict = {}


def _get_or_compile(key: _CacheKey, launch_args: tuple):
    compiled = _COMPILE_CACHE.get(key)
    if compiled is not None:
        return compiled
    config = dict(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=(128, 128),
        head_dim=key.head_dim,
        is_persistent=True,
        mask_type=fmha_utils.MaskEnum.RESIDUAL_MASK,
        enable_ex2_emulation=key.gpu_arch == "sm_100a",
        enable_skip_correction=True,
        use_tma_store=True,
    )
    if key.qk_sf_vec:
        kernel = BlackwellFusedMultiHeadBlockScaledAttentionForward(
            **config, qk_sf_vec_size=key.qk_sf_vec
        )
    else:
        kernel = BlackwellFusedMultiHeadAttentionForward(**config)
    logger.info(f"Compiling CuTe FMHA {key}")
    compiled = cute.compile(kernel, *launch_args)
    _COMPILE_CACHE[key] = compiled
    return compiled


def _scale_args(scale: float | torch.Tensor):
    if isinstance(scale, torch.Tensor):
        return cute_typing.Float32(1.0), _to_cute_tensor(scale.view(-1), leading_dim=0)
    return cute_typing.Float32(scale), None


@torch.compiler.disable
def cute_dsl_fmha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    *,
    sm_scale: float | torch.Tensor,
    lse: torch.Tensor,
    scale_output: float | torch.Tensor,
    scale_v_channels: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    qk_sf_vec: int,
    q_sf: torch.Tensor | None,
    k_sf: torch.Tensor | None,
    qk_cutlass_dtype: Any,
) -> None:
    """Launch full self-attention on equal BSHD Q/K/V with contiguous BSH logsumexp."""
    batch, seq, heads, head_dim = v.shape
    # Two NVFP4 elements occupy one byte of the Q/K storage dimension.
    qk_storage_dim = head_dim // 2 if qk_sf_vec == 16 else head_dim
    if q.shape != (batch, seq, heads, qk_storage_dim) or k.shape != q.shape or o.shape != v.shape:
        raise ValueError("CuTe FMHA requires Q/K/V/O with matching logical BSHD shapes.")
    if lse.shape != (batch, seq, heads):
        raise ValueError("CuTe FMHA requires a BSH logsumexp buffer.")
    if q.dtype != k.dtype or not all(t.is_contiguous() for t in (q, k, v, o, lse)):
        raise ValueError("CuTe FMHA requires matching Q/K dtypes and contiguous buffers.")

    # CuTe represents heads as (KV heads, query heads per KV head).
    q_cute = _to_cute_tensor(q.unsqueeze(3), leading_dim=4, cutlass_element_type=qk_cutlass_dtype)
    k_cute = _to_cute_tensor(k.unsqueeze(3), leading_dim=4, cutlass_element_type=qk_cutlass_dtype)
    v_cute = _to_cute_tensor(v.unsqueeze(3), leading_dim=4)
    o_cute = _to_cute_tensor(o.unsqueeze(3), leading_dim=4)
    lse_cute = from_dlpack(lse.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=3)
    qk_scale_args = ()
    v_channel_args = ()
    if qk_sf_vec:
        if head_dim != 128:
            raise ValueError("Block-scaled CuTe FMHA requires head_dim 128.")
        if q_sf is None or k_sf is None or not (q_sf.is_contiguous() and k_sf.is_contiguous()):
            raise ValueError("Block-scaled CuTe FMHA requires contiguous Q/K block scales.")
        sf_dtype = cutlass.Float8E8M0FNU if qk_sf_vec == 32 else cutlass.Float8E4M3FN
        qk_scale_args = (
            _to_cute_tensor(q_sf, leading_dim=0, cutlass_element_type=sf_dtype),
            _to_cute_tensor(k_sf, leading_dim=0, cutlass_element_type=sf_dtype),
        )
        v_channel_args = (
            _to_cute_tensor(scale_v_channels.view(-1), leading_dim=0)
            if scale_v_channels is not None
            else None,
        )

    skip = (
        skip_softmax_threshold_scale_factor is not None and skip_softmax_threshold_scale_factor > 0
    )
    skip_threshold_log2 = (
        cute_typing.Float32(math.log2(skip_softmax_threshold_scale_factor / seq)) if skip else None
    )
    # CuTe expects (B, Sq, S_lse, Sk, Hq, Hkv, Dqk, Dv).
    problem_size = (batch, seq, seq, seq, heads, heads, head_dim, head_dim)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    launch_args = (
        q_cute,
        k_cute,
        *qk_scale_args,
        v_cute,
        o_cute,
        problem_size,
        None,  # cum_seqlen_q
        None,  # cum_seqlen_k
        lse_cute,
        None,  # sink
        *_scale_args(sm_scale * math.log2(math.exp(1.0))),
        *_scale_args(sm_scale),
        *_scale_args(scale_output),
        *v_channel_args,
        skip_threshold_log2,
        None,  # window_left
        None,  # window_right
        None,  # skip_softmax_count
        None,  # total_softmax_count
        stream,
        False,  # use_pdl
    )
    key = _CacheKey(
        q.dtype,
        v.dtype,
        o.dtype,
        head_dim,
        qk_sf_vec,
        isinstance(sm_scale, torch.Tensor),
        isinstance(scale_output, torch.Tensor),
        scale_v_channels is not None,
        skip,
        _get_gpu_arch(q.device),
    )
    compiled = _get_or_compile(key, launch_args)
    compiled(*launch_args)


_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0


def _quantize_fp8(
    x_bshd: torch.Tensor,
    *,
    per_head_channel: bool = False,
) -> tuple[torch.Tensor, float | torch.Tensor, torch.Tensor | None]:
    if per_head_channel:
        qscale = _FP8_E4M3_MAX / x_bshd.float().abs().amax(dim=(0, 1)).clamp(min=1e-3)
        quantized = (x_bshd * qscale).to(torch.float8_e4m3fn)
        return quantized, 1.0, qscale.reciprocal().contiguous()
    quantized, dequant_scale = torch.ops.trtllm.quantize_e4m3_per_tensor(x_bshd.contiguous())
    return quantized, dequant_scale.float(), None


def _quantize_blockscaled_one(
    x_bshd: torch.Tensor, qk_sf_vec: int
) -> tuple[torch.Tensor, torch.Tensor, float | torch.Tensor]:
    """Return packed BSHD data, padded swizzled block scales, and the global dequant scale."""
    batch_size, seq_len, num_heads, head_dim = x_bshd.shape
    x_bhsd = x_bshd.transpose(1, 2).contiguous()
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
        x_q_2d, x_sf = torch.ops.trtllm.mxfp8_quantize(x_2d, True, alignment=32)
        global_dequant_scale = 1.0
    elif qk_sf_vec == 16:
        # The global-scale reduction supports rows up to 8192 elements.
        x_reduction_shape = x_2d.reshape(-1, min(8 * num_heads * head_dim, 8192))
        reduced_rows = torch.ops.trtllm.calculate_nvfp4_global_scale(x_reduction_shape, None)
        global_sf = reduced_rows.amin().clamp(max=(_FP8_E4M3_MAX * _FP4_E2M1_MAX) / 1e-6)
        global_sf_t = global_sf.to(torch.float32).reshape(1)
        x_q_2d, x_sf = torch.ops.trtllm.fp4_quantize(x_2d, global_sf_t, 16, False)
        global_dequant_scale = global_sf.reciprocal()
    else:
        raise ValueError(f"Expected a Q/K block-scale vector of 16 or 32, got {qk_sf_vec}.")
    x_q = x_q_2d.view(batch_size, num_heads, s_pad, -1)
    x_q = x_q[:, :, :seq_len, :].transpose(1, 2).contiguous()
    return x_q, x_sf, global_dequant_scale
