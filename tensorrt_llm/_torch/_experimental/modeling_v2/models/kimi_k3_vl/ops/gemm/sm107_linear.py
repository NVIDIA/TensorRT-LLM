# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Projections on K3-owned weights: tensor-parallel shards, BF16 GEMM and FP8 block-scale GEMM.

Weight containers register the parameters under the names the checkpoint loader looks up and keep
this rank's shard. Their forward is one explicit kernel chain. A row-parallel BF16 projection given
an all-reduce (``layers.TpAllReduce``, as ``all_reduce``) reduces its partial with it, with
the caller's ``all_reduce_params`` (``enable_allreduce=False``: the partial; an RMSNorm fusion:
the fused outputs), as main's ``Linear(ROW, reduce_output=True)``.

**Sharding** (:func:`shard_range` / :func:`load_shard`): rank ``r`` of ``tp`` takes a contiguous
range of the split dim (output features for ``COLUMN``, input features for ``ROW``), the first
``width % tp`` ranks one unit more; for FP8 block-scale weights the unit is a 128-wide block and
the block-scale grid follows the weight's range (``scale_span=128``). :func:`load_weight_shard`
is main's ceil-divided split (``ceil(width / tp)`` per rank), which the NVFP4 experts load with.

**BF16** (:class:`Bf16Linear`): ``F.linear`` (cuBLASLt).

**FP8 block scales** (:class:`Fp8BlockScaleLinear`): E4M3 ``weight [N, K]`` with one FP32 scale per
128 x 128 block (``weight_scale [ceil(N/128), ceil(K/128)]``). :meth:`Fp8BlockScaleLinear.prepare`
rounds the scales to UE8M0 (``resmooth_to_fp8_e8m0`` rescales the codes to match) and lays them
out for the GEMM of this GPU; :func:`fp8_block_scale_gemm` then runs, on BF16 rows:

* SM107 (CuTe DSL Rubin): ``fp8_quantize_1x128_cutedsl_ue8m0`` (activation to E4M3 with K32
  R128c4 UE8M0 scales) and ``ops.gemm.k3_mxfp8_gemm`` (``ops/gemm/_mxfp8_gemm.py``), the weight
  scale in the K32 R128c4 layout;
* SM100 / SM103: ``fp8_swap_ab_gemm(disable_ue8m0_cast=True)`` (DeepGEMM; activation quantized in
  the kernel), the weight scale packed for DeepGEMM;
* either, where ``low_m_max`` covers the rows: ``trtllm::fp8_block_scale_low_m_gemm`` on the
  DeepGEMM-packed scale (on SM107 a copy built from the K32 R128c4 one).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_device_integrated
from tensorrt_llm.functional import AllReduceParams
from tensorrt_llm.logger import logger

from ...common.parallel import TensorParallelMode

FP8_BLOCK = 128


def _rubin_mxfp8() -> bool:
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE

    return get_sm_version() == 107 and IS_CUTLASS_DSL_RUBIN_AVAILABLE


def _split(width: int, parts: int, index: int) -> int:
    return (width // parts) * index + min(width % parts, index)


def shard_range(width: int, tp_size: int, tp_rank: int, alignment: int = 1) -> Tuple[int, int]:
    """``[start, end)`` of rank ``tp_rank`` over a split dim of ``width`` elements."""
    if alignment <= 1:
        return _split(width, tp_size, tp_rank), _split(width, tp_size, tp_rank + 1)
    if width % alignment:
        raise ValueError(f"split dim {width} is not a multiple of {alignment}")
    blocks = width // alignment
    return (
        _split(blocks, tp_size, tp_rank) * alignment,
        _split(blocks, tp_size, tp_rank + 1) * alignment,
    )


def load_shard(
    tensor: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    mode: Optional[TensorParallelMode],
    *,
    features: Optional[int] = None,
    alignment: int = 1,
    scale_span: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """This rank's slice of a checkpoint tensor (a weight, or with ``scale_span`` its block-scale
    grid) on ``device``. ``features``: the full split-dim width of the weight (default: the
    tensor's). A 1-D tensor of a ``ROW`` projection, or a split dim of width 1, is not split."""
    device = device if device is not None else tensor.device
    if mode is None or tp_size <= 1:
        return tensor.to(device)
    dim = TensorParallelMode.split_dim(mode)
    if (tensor.dim() == 1 and dim == 1) or tensor.shape[dim] == 1:
        return tensor.to(device)
    start, end = shard_range(features or tensor.shape[dim], tp_size, tp_rank, alignment)
    if scale_span:
        if start % scale_span or end % scale_span:
            raise ValueError(f"shard [{start}, {end}) does not align with {scale_span}-wide scales")
        start, end = start // scale_span, end // scale_span
    return tensor.narrow(dim, start, end - start).to(device)


def load_weight_shard(
    weight,
    tensor_parallel_size: int = 1,
    tensor_parallel_rank: int = 0,
    tensor_parallel_mode: Optional[TensorParallelMode] = None,
    device: torch.device = torch.device("cpu"),
    return_slice_indices: bool = False,
) -> torch.Tensor:
    """This rank's ceil-divided slice of ``weight`` (a tensor or a safetensors slice) on ``device``:
    ``ceil(width / tp)`` elements of the split dim per rank, the last rank the remainder. A 1-D
    tensor of a ``ROW`` projection, or a split dim of width 1, is not split. On an integrated GPU
    the tensor stays on its own device. ``return_slice_indices`` (tensors only): also the slices."""
    # Skip device transfers on integrated GPUs to conserve shared memory
    if weight.device.type != device.type and is_device_integrated():
        # For integrated GPU systems (e.g., DGX Spark), CPU and GPU share limited physical memory.
        # Avoiding device transfers reduces memory consumption and unnecessary data copies,
        # enabling support for larger models on memory-constrained systems.
        logger.warning_once(
            f"[load_weight_shard] Skipping device transfer from {weight.device} to {device} "
            "on integrated GPU to conserve shared memory.",
            key="load_weight_shard_skip_device_transfer_with_integrated_gpu",
        )
        device = weight.device
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor, indices: list[slice] | None = None):
            if indices is None:
                # Avoid unnecessary copy
                result = (tensor.to(device), [slice(d) for d in tensor.shape])
            else:
                result = (tensor[indices].to(device), indices)
            return result if return_slice_indices else result[0]

    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to
    # the module; safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice, tuple[slice]] = slice(None)
        ):
            return tensor[indices].to(device)
    else:
        raise ValueError(f"unsupported weight type: {type(weight)}")
    if tensor_parallel_mode is None or tensor_parallel_size <= 1:
        return maybe_convert_to_torch_tensor(weight)

    split_dim = TensorParallelMode.split_dim(tensor_parallel_mode)

    if len(tensor_shape) == 1 and split_dim == 1:
        return maybe_convert_to_torch_tensor(weight)

    width = tensor_shape[split_dim]
    if width == 1:
        return maybe_convert_to_torch_tensor(weight)

    slice_width = math.ceil(width / tensor_parallel_size)
    slice_start = tensor_parallel_rank * slice_width
    slice_end = min((tensor_parallel_rank + 1) * slice_width, width)
    slice_obj = [slice(d) for d in tensor_shape]
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))


class _ShardedLinear(nn.Module):
    alignment = 1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int,
        tp_rank: int,
        tp_mode: Optional[TensorParallelMode],
    ) -> None:
        super().__init__()
        self.tp_size, self.tp_rank, self.tp_mode = tp_size, tp_rank, tp_mode
        self.full_in_features, self.full_out_features = in_features, out_features
        self.in_features, self.out_features = in_features, out_features
        if tp_size > 1 and tp_mode is not None:
            dim_width = in_features if tp_mode == TensorParallelMode.ROW else out_features
            start, end = shard_range(dim_width, tp_size, tp_rank, self.alignment)
            if tp_mode == TensorParallelMode.ROW:
                self.in_features = end - start
            else:
                self.out_features = end - start

    def split_features(self) -> int:
        return (
            self.full_in_features
            if self.tp_mode == TensorParallelMode.ROW
            else self.full_out_features
        )

    def shard(self, tensor: torch.Tensor, scale_span: Optional[int] = None) -> torch.Tensor:
        """This rank's slice of a full checkpoint tensor of the projection, on its device."""
        return load_shard(
            tensor,
            self.tp_size,
            self.tp_rank,
            self.tp_mode,
            features=self.split_features(),
            alignment=self.alignment,
            scale_span=scale_span,
            device=self.weight.device,
        )


def bf16_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``x @ weight.T (+ bias)``: cuBLASLt via ``F.linear``."""
    return F.linear(x, weight, bias)


class Bf16Linear(_ShardedLinear):
    """BF16 (or FP16 / FP32) ``weight [out, in]`` (and ``bias [out]``) of one rank."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_mode: Optional[TensorParallelMode] = None,
        all_reduce: Optional[Callable[..., torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, tp_size, tp_rank, tp_mode)
        if all_reduce is not None and tp_mode != TensorParallelMode.ROW:
            raise ValueError("only a row-parallel projection reduces its output")
        self.all_reduce = all_reduce
        self.reduce_output = all_reduce is not None
        self.weight = nn.Parameter(
            torch.empty(self.out_features, self.in_features, dtype=dtype, device=device),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, dtype=dtype, device=device), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

    def load(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """Copy this rank's shard of the full checkpoint ``weight`` (and ``bias``)."""
        self.weight.data.copy_(self.shard(weight).to(self.weight.dtype))
        if bias is not None:
            self.bias.data.copy_(self.shard(bias).to(self.bias.dtype))

    def forward(
        self, x: torch.Tensor, *, all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        # Bias on rank 0 only for a row-parallel projection (its partial sums are all-reduced).
        bias = self.bias if self.tp_mode != TensorParallelMode.ROW or self.tp_rank == 0 else None
        out = bf16_linear(x, self.weight, bias)
        if self.all_reduce is not None:
            out = self.all_reduce(out, all_reduce_params=all_reduce_params)
        return out


def prepare_fp8_block_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(codes, scale)`` for :func:`fp8_block_scale_gemm` from checkpoint E4M3 codes and their FP32
    128 x 128 grid: scales rounded to UE8M0 with the codes rescaled to match (in place), then the
    scale laid out for this GPU's GEMM (K32 R128c4 UE8M0 bytes on SM107, DeepGEMM-packed UE8M0
    otherwise)."""
    from tensorrt_llm.quantization.utils.fp8_utils import (
        resmooth_to_fp8_e8m0,
        transform_k128_scales_to_cutedsl_mxfp8_layout,
        transform_sf_into_required_layout,
    )

    weight, weight_scale = resmooth_to_fp8_e8m0(
        weight.contiguous(), weight_scale.contiguous().float()
    )
    n, k = weight.shape
    if _rubin_mxfp8():
        return weight, transform_k128_scales_to_cutedsl_mxfp8_layout(weight_scale, mn=n, k=k)
    return weight, transform_sf_into_required_layout(
        weight_scale, mn=n, k=k, recipe=(1, 128, 128), is_sfa=False
    )


def low_m_gemm_rows(n: int, k: int) -> int:
    """Largest M ``trtllm::fp8_block_scale_low_m_gemm`` serves for an ``[n, k]`` weight here (0:
    none)."""
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not IS_CUTLASS_DSL_AVAILABLE:
        return 0
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.max_m_for(n, k)


def low_m_gemm_scale(weight_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """The DeepGEMM-packed scale the low-M GEMM reads, from a :func:`prepare_fp8_block_weight`
    scale (itself on SM100 / SM103, repacked from K32 R128c4 on SM107)."""
    if not _rubin_mxfp8():
        return weight_scale
    from tensorrt_llm._torch.kernels.blackwell import fp8_block_scale_low_m_gemm

    return fp8_block_scale_low_m_gemm.pack_block_scale_from_mxfp8_layout(weight_scale, n, k)


def fp8_block_scale_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    low_m_max: int = 0,
    low_m_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """BF16 ``x [M, K] @ weight.T`` on :func:`prepare_fp8_block_weight` operands (see the module
    docstring for the chain per GPU)."""
    if (
        low_m_max
        and 1 <= x.shape[0] <= low_m_max
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    ):
        return torch.ops.trtllm.fp8_block_scale_low_m_gemm(x, weight, low_m_scale)
    if _rubin_mxfp8():
        from .k3_mxfp8_gemm import k3_mxfp8_gemm

        act, act_scale = torch.ops.trtllm.fp8_quantize_1x128_cutedsl_ue8m0(x)
        return k3_mxfp8_gemm(act, weight, act_scale, weight_scale)
    return torch.ops.trtllm.fp8_swap_ab_gemm(
        x, weight, weight_scale, output_dtype=x.dtype, disable_ue8m0_cast=True
    )


class Fp8BlockScaleLinear(_ShardedLinear):
    """FP8 block-scale ``weight`` / ``weight_scale`` of one rank (plus the unit ``input_scale`` /
    ``inv_input_scale`` checkpoint FP8 projections also register), bias-free."""

    alignment = FP8_BLOCK

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_mode: Optional[TensorParallelMode] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(in_features, out_features, tp_size, tp_rank, tp_mode)
        n, k = self.out_features, self.in_features
        self.weight = nn.Parameter(
            torch.empty(n, k, dtype=torch.float8_e4m3fn, device=device), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(-(-n // FP8_BLOCK), -(-k // FP8_BLOCK), dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.input_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.inv_input_scale = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=False
        )
        self.register_parameter("bias", None)
        self.prepared = False
        self.low_m_max = 0
        self.low_m_scale: Optional[torch.Tensor] = None

    def load(self, weight: torch.Tensor, weight_scale: torch.Tensor) -> None:
        """Copy this rank's shard of the full checkpoint codes and their FP32 block-scale grid
        (a modelopt ``[N/128, 1, K/128, 1]`` grid is squeezed first)."""
        if weight_scale.dim() == 4:
            weight_scale = weight_scale.squeeze(1).squeeze(-1)
        self.weight.data.copy_(self.shard(weight))
        self.weight_scale.data.copy_(self.shard(weight_scale, scale_span=FP8_BLOCK).float())
        self.prepared = False

    def prepare(self, low_m: bool = True) -> None:
        """Once after loading: :func:`prepare_fp8_block_weight` in place of the parameters, and
        with ``low_m`` the rows the low-M GEMM serves and its scale."""
        if self.prepared:
            return
        weight, scale = prepare_fp8_block_weight(self.weight.data, self.weight_scale.data)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale = nn.Parameter(scale, requires_grad=False)
        n, k = self.out_features, self.in_features
        self.low_m_max = low_m_gemm_rows(n, k) if low_m else 0
        self.low_m_scale = low_m_gemm_scale(scale, n, k) if self.low_m_max else None
        self.prepared = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.prepared:
            raise RuntimeError("Fp8BlockScaleLinear.prepare() must run after loading")
        x_flat = x.reshape(-1, x.shape[-1])
        out = fp8_block_scale_gemm(
            x_flat,
            self.weight,
            self.weight_scale,
            low_m_max=self.low_m_max,
            low_m_scale=self.low_m_scale,
        )
        return out.reshape(*x.shape[:-1], self.out_features)
