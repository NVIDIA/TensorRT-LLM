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

from __future__ import annotations

import functools
import math
import platform
from pathlib import Path
from typing import Optional, Union

import torch

_cute_runtime_import_error = None
try:
    import cutlass
    from cuda.bindings import driver as cuda_driver
    from cutlass import cute
    from cutlass.cute import typing as cute_typing
    from cutlass.cute.runtime import from_dlpack
except (ImportError, OSError) as e:
    cutlass = None
    cute = None
    from_dlpack = None
    cuda_driver = None
    _cute_runtime_import_error = e


CUBINS_ROOT = Path(__file__).resolve().parent / "cubins"
SUPPORTED_GPU_ARCHS = ("sm_100a", "sm_103a")


def _dtype_to_str(dtype: torch.dtype) -> str:
    float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    dtype_map = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }
    if float8_e4m3fn is not None:
        dtype_map[float8_e4m3fn] = "e4m3"
    try:
        return dtype_map[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported CuTe DSL FMHA dtype: {dtype}") from exc


def _gpu_cpu_arch() -> str:
    arch = platform.machine().lower()
    if arch in ("amd64", "x64"):
        return "x86_64"
    if arch in ("arm64",):
        return "aarch64"
    return arch


def _get_gpu_arch(device: torch.device | str | None = None) -> str:
    capability = torch.cuda.get_device_capability(device)
    gpu_arch = f"sm_{capability[0]}{capability[1]}a"
    if gpu_arch not in SUPPORTED_GPU_ARCHS:
        supported = ", ".join(SUPPORTED_GPU_ARCHS)
        raise ValueError(
            f"Unsupported GPU architecture {gpu_arch}. Supported architectures: {supported}."
        )
    return gpu_arch


def _get_cubins_dir(gpu_arch: str) -> Path:
    return CUBINS_ROOT / _gpu_cpu_arch() / gpu_arch


def _get_variant_name(
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    is_persistent: bool = True,
    varlen: bool = False,
    with_lse: bool = False,
    enable_skip_softmax: bool = False,
    enable_tvm_ffi: bool = False,
) -> str:
    qk_str = _dtype_to_str(qk_dtype)
    pv_str = _dtype_to_str(pv_dtype)
    out_str = _dtype_to_str(out_dtype)
    if qk_dtype != pv_dtype:
        dtype_str = f"{qk_str}_{pv_str}_{out_str}"
    elif qk_dtype != out_dtype:
        dtype_str = f"{qk_str}_{out_str}"
    else:
        dtype_str = qk_str

    causal_str = "causal" if is_causal else "nocausal"
    persist_str = "persistent" if is_persistent else "nonpersistent"
    varlen_str = "_varlen" if varlen else ""
    lse_str = "_lse" if with_lse else ""
    skip_str = "_skipsm" if enable_skip_softmax else ""
    ffi_str = "_tvmffi" if enable_tvm_ffi else ""
    return (
        f"cute_dsl_fmha_{dtype_str}_h{head_dim}_{causal_str}_{persist_str}"
        f"{varlen_str}{lse_str}{skip_str}{ffi_str}"
    )


def _get_candidate_paths(variant_name: str, gpu_arch: str | None) -> list[Path]:
    names = [f"{variant_name}.so", f"{variant_name}.o"]
    if gpu_arch is not None:
        return [_get_cubins_dir(gpu_arch) / name for name in names]

    host_dir = CUBINS_ROOT / _gpu_cpu_arch()
    return [
        host_dir / supported_gpu_arch / name
        for supported_gpu_arch in SUPPORTED_GPU_ARCHS
        for name in names
    ]


def _resolve_cubin_path(
    variant_name: str,
    gpu_arch: str | None = None,
) -> Path:
    tried = []
    for candidate in _get_candidate_paths(variant_name, gpu_arch):
        tried.append(candidate)
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"  - {path}" for path in tried)
    default_dir = (
        _get_cubins_dir(gpu_arch)
        if gpu_arch is not None
        else CUBINS_ROOT / _gpu_cpu_arch() / "<gpu_arch>"
    )
    raise FileNotFoundError(
        f"Could not find packaged CuTe DSL FMHA cubins for '{variant_name}'.\n"
        f"Expected a .so or .o under {default_dir}.\nSearched:\n{searched}"
    )


def _check_cute_runtime_available() -> None:
    if cute is not None:
        return
    raise ImportError(
        f"CuTe DSL runtime is not available. Import error: {_cute_runtime_import_error}"
    ) from _cute_runtime_import_error


def _load_cubin_from_path(
    path: str | Path,
    variant_name: str | None = None,
    enable_tvm_ffi: bool = True,
):
    _check_cute_runtime_available()

    cubin_path = Path(path).expanduser().resolve()
    if variant_name is None:
        variant_name = cubin_path.stem

    module = cute.runtime.load_module(str(cubin_path), enable_tvm_ffi=enable_tvm_ffi)
    try:
        return getattr(module, variant_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Loaded {cubin_path}, but symbol '{variant_name}' was not found. "
            "The symbol name must match the .so filename stem / function prefix."
        ) from exc


@functools.lru_cache(maxsize=None)
def _load_cute_dsl_fmha_cubin_cached(
    variant_name: str,
    gpu_arch: str | None,
    enable_tvm_ffi: bool,
):
    path = _resolve_cubin_path(variant_name, gpu_arch)
    return _load_cubin_from_path(path, variant_name, enable_tvm_ffi)


def get_cute_dsl_fmha_cubin(
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    head_dim: int,
    is_causal: bool,
    is_persistent: bool = True,
    enable_tvm_ffi: bool = True,
    varlen: bool = False,
    with_lse: bool = False,
    enable_skip_softmax: bool = False,
    gpu_arch: str | None = None,
):
    variant_name = _get_variant_name(
        qk_dtype,
        pv_dtype,
        out_dtype,
        head_dim,
        is_causal,
        is_persistent,
        varlen,
        with_lse,
        enable_skip_softmax,
        enable_tvm_ffi,
    )
    return _load_cute_dsl_fmha_cubin_cached(
        variant_name,
        gpu_arch,
        enable_tvm_ffi,
    )


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
) -> bool:
    if q.dim() not in (3, 4) or k.dim() != q.dim() or v.dim() != q.dim():
        raise ValueError("Expected 3D or 4D Q/K/V tensors with matching ranks.")
    if q.dtype != k.dtype:
        raise ValueError(f"Q/K dtype mismatch: {q.dtype} vs {k.dtype}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"Q/K head dim mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(f"K/V shape mismatch: {k.shape[:-1]} vs {v.shape[:-1]}")
    expected_o_shape = (*q.shape[:-1], v.shape[-1])
    if tuple(o.shape) != expected_o_shape:
        raise ValueError(f"Output shape mismatch: {tuple(o.shape)} vs {expected_o_shape}")
    return q.dim() == 3


def _to_cint_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.int32).contiguous()


def _to_cute_tensor(tensor: torch.Tensor, leading_dim: int):
    float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    if tensor.dtype == float8_e4m3fn:
        cute_tensor = from_dlpack(tensor.view(torch.int8), assumed_align=16)
        cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        cute_tensor.element_type = cutlass.Float8E4M3FN
        return cute_tensor
    return from_dlpack(tensor, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


def _get_runtime_problem(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: Optional[torch.Tensor],
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
    max_qo_len: Optional[int],
    max_kv_len: Optional[int],
    varlen: bool,
) -> tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if varlen:
        if qo_indptr is None or kv_indptr is None or max_qo_len is None or max_kv_len is None:
            raise ValueError("Varlen FMHA requires indptr tensors and max sequence lengths.")
        if qo_indptr.dim() != 1 or kv_indptr.dim() != 1:
            raise ValueError("Varlen FMHA indptr tensors must be 1D.")
        if qo_indptr.numel() != kv_indptr.numel() or qo_indptr.numel() < 2:
            raise ValueError("Varlen FMHA indptr tensors must have matching non-empty sizes.")
        total_q, num_heads_q, head_dim = q.shape
        _, num_heads_kv, _ = k.shape
        batch_size = qo_indptr.numel() - 1
        max_s_q = max_qo_len
        max_s_k = max_kv_len
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        o_4d = o.unsqueeze(0)
        lse_3d = lse.unsqueeze(0) if lse is not None else None
    else:
        batch_size, max_s_q, num_heads_q, head_dim = q.shape
        _, max_s_k, num_heads_kv, _ = k.shape
        total_q = batch_size * max_s_q
        q_4d = q
        k_4d = k
        v_4d = v
        o_4d = o
        lse_3d = lse
    value_head_dim = v.shape[-1]
    return (
        batch_size,
        max_s_q,
        total_q,
        max_s_k,
        num_heads_q,
        num_heads_kv,
        head_dim,
        value_head_dim,
        q_4d,
        k_4d,
        v_4d,
        o_4d,
        lse_3d,
    )


@torch.compiler.disable
def cute_dsl_fmha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    window_right: int = -1,
    lse: Optional[torch.Tensor] = None,
    scale_q: Union[float, torch.Tensor] = 1.0,
    scale_k: Union[float, torch.Tensor] = 1.0,
    scale_v: Union[float, torch.Tensor] = 1.0,
    scale_o: Union[float, torch.Tensor] = 1.0,
    enable_tvm_ffi: bool = True,
    is_persistent: bool = False,
    max_qo_len: Optional[int] = None,
    max_kv_len: Optional[int] = None,
    kernel_fn=None,
    skip_softmax_threshold_scale_factor: Optional[float] = None,
) -> None:
    varlen = _check_inputs(q, k, v, o)

    scale_q = float(scale_q.item()) if isinstance(scale_q, torch.Tensor) else scale_q
    scale_k = float(scale_k.item()) if isinstance(scale_k, torch.Tensor) else scale_k
    scale_v = float(scale_v.item()) if isinstance(scale_v, torch.Tensor) else scale_v
    scale_o = float(scale_o.item()) if isinstance(scale_o, torch.Tensor) else scale_o

    (
        batch_size,
        max_s_q,
        total_q,
        max_s_k,
        num_heads_q,
        num_heads_kv,
        head_dim,
        value_head_dim,
        q_4d,
        k_4d,
        v_4d,
        o_4d,
        lse_3d,
    ) = _get_runtime_problem(q, k, v, o, lse, qo_indptr, kv_indptr, max_qo_len, max_kv_len, varlen)
    use_skip_softmax = (
        skip_softmax_threshold_scale_factor is not None and skip_softmax_threshold_scale_factor > 0
    )
    problem_size = (
        batch_size,
        max_s_q,
        total_q,
        max_s_k,
        num_heads_q,
        num_heads_kv,
        head_dim,
        value_head_dim,
    )

    if kernel_fn is None:
        kernel_fn = get_cute_dsl_fmha_cubin(
            q.dtype,
            v.dtype,
            o.dtype,
            head_dim,
            is_causal,
            is_persistent=is_persistent,
            varlen=varlen,
            enable_tvm_ffi=enable_tvm_ffi,
            with_lse=lse is not None,
            enable_skip_softmax=use_skip_softmax,
            gpu_arch=_get_gpu_arch(q.device),
        )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    scale_softmax = scale_q * scale_k * sm_scale
    scale_softmax_log2 = scale_softmax * math.log2(math.exp(1.0))
    scale_output = scale_v / scale_o

    skip_threshold_log2 = None
    if use_skip_softmax:
        threshold = skip_softmax_threshold_scale_factor / max_s_k
        skip_threshold_log2 = cute_typing.Float32(math.log2(threshold))

    ws_left = None if window_left == -1 else cute_typing.Int32(window_left)
    ws_right = None if window_right == -1 else cute_typing.Int32(window_right)
    if is_causal and ws_right is None:
        ws_right = cute_typing.Int32(0)

    # CUBIN path
    num_head_groups = num_heads_q // num_heads_kv
    q_5d = q_4d.unflatten(2, (num_heads_kv, num_head_groups))
    k_5d = k_4d.unsqueeze(3)
    v_5d = v_4d.unsqueeze(3)
    o_5d = o_4d.unflatten(2, (num_heads_kv, num_head_groups))
    lse_4d = lse_3d.unflatten(2, (num_heads_kv, num_head_groups)) if lse is not None else None

    if enable_tvm_ffi:
        qo_indptr_i32 = _to_cint_contiguous(qo_indptr) if varlen else None
        kv_indptr_i32 = _to_cint_contiguous(kv_indptr) if varlen else None
        kernel_fn(
            q_5d,
            k_5d,
            v_5d,
            o_5d,
            problem_size,
            qo_indptr_i32,
            kv_indptr_i32,
            lse_4d,
            None,  # sink
            cute_typing.Float32(scale_softmax_log2),
            cute_typing.Float32(scale_softmax),
            cute_typing.Float32(scale_output),
            skip_threshold_log2,
            ws_left,
            ws_right,
            None,
            None,
            False,  # reserved
        )
        return

    q_cute = _to_cute_tensor(q_5d, leading_dim=4)
    k_cute = _to_cute_tensor(k_5d, leading_dim=4)
    v_cute = _to_cute_tensor(v_5d, leading_dim=4)
    o_cute = _to_cute_tensor(o_5d, leading_dim=4)

    cum_seqlen_q_cute = None
    cum_seqlen_k_cute = None
    if varlen:
        qo_indptr_i32 = _to_cint_contiguous(qo_indptr)
        kv_indptr_i32 = _to_cint_contiguous(kv_indptr)
        cum_seqlen_q_cute = from_dlpack(qo_indptr_i32, assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        )
        cum_seqlen_k_cute = from_dlpack(kv_indptr_i32, assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        )

    lse_iter = None
    if lse is not None:
        lse_cute = from_dlpack(lse_4d, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        lse_iter = lse_cute.iterator

    stream = cuda_driver.CUstream(torch.cuda.current_stream(q).cuda_stream)
    kernel_fn(
        q_cute.iterator,
        k_cute.iterator,
        v_cute.iterator,
        o_cute.iterator,
        problem_size,
        cum_seqlen_q_cute,
        cum_seqlen_k_cute,
        lse_iter,
        None,  # sink_iter
        cute_typing.Float32(scale_softmax_log2),
        cute_typing.Float32(scale_softmax),
        cute_typing.Float32(scale_output),
        skip_threshold_log2,
        ws_left,
        ws_right,
        None,
        None,
        False,  # reserved
        stream,
    )
