# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)
from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm
from tensorrt_llm._torch.modules.low_m_gemm import low_m_gemm_dispatcher, low_m_gemm_takes_rows
from tensorrt_llm._torch.utils import get_model_extra_attrs
from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization.fp8_block_weight_layout import (
    fp8_block_weight_layout,
)
from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization.fp8_quantize import (
    native_fp8_per_tensor_quantize,
    native_static_fp8_per_tensor_quantize,
)
from tensorrt_llm._torch.visual_gen.modeling_v2.catalog.quantization.fp8_static_quantize import (
    fp8_static_quantize,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f, mpi_disabled
from tensorrt_llm.bindings import ipc_nvls_supported
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.mapping import Mapping, MpiTopology
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..ops.nvfp4_gemm_allreduce import nvfp4_gemm_allreduce
from ..ops.quantized_gemm import fp8_block_gemm, fp8_scaled_gemm, low_m_limit, tunable_fp4_quantize
from .projection_weights import load_projection


class TensorParallelSubgroup(MpiTopology):
    """A tensor-parallel mapping for the first `size` ranks and their explicit process group."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, size: int, rank: int, group: dist.ProcessGroup | None):
        super().__init__(world_size=size, rank=rank, tp_size=size, gpus_per_node=size)
        self.process_group = group

    @property
    def tp_group_pg(self) -> dist.ProcessGroup:
        if self.process_group is None:
            raise RuntimeError("Tensor-parallel subgroup is not initialized.")
        return self.process_group

    @property
    def tp_group_name(self) -> str:
        return self.tp_group_pg.group_name


class TensorParallelSum(nn.Module):
    """Sum across a TP mapping and supply its GEMM output allocation policy."""

    def __init__(self, mapping: Mapping, strategy: AllReduceStrategy, dtype: torch.dtype | None):
        super().__init__()
        self.mapping = mapping
        self._process_group = None
        self._engine = None
        if mapping.tp_size > 1:
            if isinstance(mapping, TensorParallelSubgroup) and mpi_disabled():
                self._process_group = mapping.tp_group_pg
            else:
                self._engine = AllReduce(mapping=mapping, strategy=strategy, dtype=dtype)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._process_group is not None:
            result = tensor.contiguous().clone()
            dist.all_reduce(result, group=self._process_group)
            return result
        if self._engine is not None:
            return self._engine(tensor)
        return tensor

    def output_allocation(self) -> tuple[int, list[int] | None]:
        if self._engine is not None and self._engine.uses_nccl_symmetric_memory_window():
            return int(BufferKind.NCCL_WINDOW), self.mapping.tp_group
        return int(BufferKind.DEFAULT), None


def shard_bounds(size: int, tp_size: int, rank: int) -> tuple[int, int]:
    chunk, extra = divmod(size, tp_size)
    start = rank * chunk + min(rank, extra)
    return start, start + chunk + (rank < extra)


def projection_shard_bounds(
    width: int, tp_size: int, rank: int, quant_algo: QuantAlgo | None, deferred: bool
) -> tuple[int, int]:
    if tp_size == 1:
        return 0, width
    if deferred:
        if width % tp_size:
            raise ValueError(
                f"Deferred H3 projection width {width} must divide across {tp_size} ranks"
            )
        return shard_bounds(width, tp_size, rank)
    alignment = {QuantAlgo.FP8_BLOCK_SCALES: 128, QuantAlgo.NVFP4: 32}.get(quant_algo, 1)
    if width % alignment:
        raise ValueError(
            f"TP projection width {width} must be divisible by quantization alignment {alignment}"
        )
    start, end = shard_bounds(width // alignment, tp_size, rank)
    return start * alignment, end * alignment


class Linear(nn.Module):
    """H3 projection with rank-local parameters in the layout consumed by GEMM and LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        mapping: Mapping | None = None,
        tensor_parallel_mode: str | None = None,
        quant_config: QuantConfig | None = None,
        reduce_output: bool = True,
        skip_create_weights_in_init: bool = False,
        force_dynamic_quantization: bool = False,
        allreduce_strategy=AllReduceStrategy.AUTO,
        override_tp_sharding: tuple[int, int] | dict[str, tuple[int, int]] | None = None,
        fused_weight_shard_indices_mapping: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.mapping = mapping or Mapping()
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.tp_mode = tensor_parallel_mode
        self.quant_config = quant_config
        self.has_bias = bias
        self.force_dynamic_quantization = force_dynamic_quantization
        self.fused_weight_shard_indices_mapping = fused_weight_shard_indices_mapping
        self.tp_sharding = override_tp_sharding
        if self.tp_mode not in (None, "column", "row"):
            raise ValueError(f"Invalid H3 projection TP mode {self.tp_mode}")
        if self.tp_sharding is None and self.tp_mode and self.tp_size > 1:
            width = in_features if self.tp_mode == "row" else out_features
            self.tp_sharding = projection_shard_bounds(
                width, self.tp_size, self.tp_rank, self.quant_algo, skip_create_weights_in_init
            )
        self.in_features = in_features
        self.out_features = out_features
        if self.tp_sharding is not None:
            ranges = (
                self.tp_sharding.values()
                if isinstance(self.tp_sharding, dict)
                else [self.tp_sharding]
            )
            local_width = sum(stop - start for start, stop in ranges)
            if self.tp_mode == "column":
                self.out_features = local_width
            if self.tp_mode == "row":
                self.in_features = local_width
        self.all_reduce = (
            TensorParallelSum(mapping=self.mapping, strategy=allreduce_strategy, dtype=dtype)
            if self.tp_mode == "row" and self.tp_size > 1 and reduce_output
            else None
        )
        self.use_fused_gemm_allreduce = bool(
            self.all_reduce is not None
            and self.has_nvfp4
            and not mpi_disabled()
            and dtype in (torch.bfloat16, torch.float16)
            and self.in_features % 128 == 0
            and self.out_features % 64 == 0
            and os.getenv("TRTLLM_GEMM_ALLREDUCE_FUSION_ENABLED", "0") == "1"
            and get_sm_version() >= 100
            and ipc_nvls_supported()
        )
        sm = get_sm_version() if torch.cuda.is_available() else 0
        self.enable_cuda_core = sm in (89, 120, 121)
        self.use_rubin_fp8 = (
            sm == 107
            and IS_CUTLASS_DSL_RUBIN_AVAILABLE
            and os.getenv("USE_CUTE_DSL_FP8_PER_TENSOR_MM", "0") == "1"
        )
        self.nvfp4_backends = ",".join(
            (get_model_extra_attrs() or {}).get("nvfp4_gemm_allowed_backends")
            or ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
        )
        self.packed_sm100 = torch.cuda.is_available() and is_sm_100f()
        self.packed_sm120 = sm in (120, 121)
        self.fp8_low_m_limit = 0
        self._weights_created = False
        self._weights_transformed = False
        if not skip_create_weights_in_init:
            self.create_weights()

    @property
    def quant_algo(self):
        return self.quant_config.quant_algo if self.quant_config is not None else None

    @property
    def has_any_quant(self):
        return self.quant_algo is not None

    @property
    def has_fp8_qdq(self):
        return self.quant_algo == QuantAlgo.FP8

    @property
    def has_fp8_block_scales(self):
        return self.quant_algo == QuantAlgo.FP8_BLOCK_SCALES

    @property
    def has_nvfp4(self):
        return self.quant_algo == QuantAlgo.NVFP4

    def disable_quantization(self):
        if self.quant_config is None:
            return
        self.quant_config = QuantConfig(kv_cache_quant_algo=self.quant_config.kv_cache_quant_algo)
        self.use_fused_gemm_allreduce = False
        if self._weights_created:
            self._parameters.clear()
            self._weights_created = False
            self.create_weights()

    def create_weights(self):
        if self._weights_created:
            return
        n, k = self.out_features, self.in_features

        def param(tensor):
            return nn.Parameter(tensor, requires_grad=False)

        if self.has_nvfp4:
            group_size = (
                self.quant_config.group_size
                if "group_size" in self.quant_config.model_fields_set
                else None
            )
            if group_size not in (None, 16) or k % 16:
                raise ValueError("H3 NVFP4 projections require 16-wide weight scale blocks")
            self.weight = param(torch.empty(n, k // 2, dtype=torch.uint8))
            self.weight_scale = param(
                torch.empty(
                    ((n + 127) // 128 * 128) * ((k // 16 + 3) // 4 * 4),
                    dtype=torch.uint8,
                )
            )
            self.input_scale = param(torch.empty(1, dtype=torch.float32))
            self.inv_input_scale = param(torch.empty(1, dtype=torch.float32))
            self.weight_scale_2 = param(torch.empty(1, dtype=torch.float32))
            self.alpha = param(torch.empty(1, dtype=torch.float32))
        elif self.has_fp8_qdq or self.has_fp8_block_scales:
            self.weight = param(torch.empty(n, k, dtype=torch.float8_e4m3fn))
            self.weight_scale = param(
                torch.ones((), dtype=torch.float32)
                if self.has_fp8_qdq
                else torch.empty((n + 127) // 128, (k + 127) // 128, dtype=torch.float32)
            )
            self.input_scale = param(torch.ones((), dtype=torch.float32))
            self.inv_input_scale = param(torch.ones((), dtype=torch.float32))
        elif not self.has_any_quant:
            self.weight = param(torch.empty(n, k, dtype=self.dtype))
        else:
            raise ValueError(f"Unsupported H3 projection quantization {self.quant_algo}")
        if self.has_nvfp4 or self.has_fp8_qdq:
            self.kv_scales = param(torch.ones(3, dtype=torch.float32))
            self.inv_kv_scales = param(torch.ones(3, dtype=torch.float32))
        if self.has_bias:
            self.bias = param(torch.empty(n, dtype=self.dtype))
        else:
            self.register_parameter("bias", None)
        self._weights_created = True
        self._weights_transformed = False

    def load_shard(self, tensor, *, name=None, device=None, span=1):
        if self.tp_size > 1 and self.tp_mode is not None:
            axis = int(self.tp_mode == "row")
            if tensor.ndim > axis and tensor.shape[axis] != 1:
                bounds = (
                    self.tp_sharding[name]
                    if isinstance(self.tp_sharding, dict)
                    else self.tp_sharding
                )
                start, end = bounds
                if start % span or end % span:
                    raise ValueError(
                        f"H3 {self.tp_mode} shard {bounds} splits a quantization block of {span}"
                    )
                tensor = tensor.narrow(axis, start // span, (end - start) // span)
        return tensor.to(device) if device is not None else tensor

    def load_checkpoint(self, name, weights, model_config=None):
        return load_projection(self, name, weights, model_config)

    @torch.no_grad()
    def post_load_weights(self):
        if self._weights_transformed:
            return
        if self.has_fp8_block_scales and (self.packed_sm100 or self.packed_sm120):
            weight, scales = self.weight, self.weight_scale
            if weight.device.type == "cpu":
                weight, scales = weight.cuda(), scales.cuda()
            weight, scales = fp8_block_weight_layout(weight, scales)
            self.weight.data = weight
            self.weight_scale.data = scales
            if self.packed_sm100 and get_sm_version() != 107 and IS_CUTLASS_DSL_AVAILABLE:
                self.fp8_low_m_limit = low_m_limit(self.out_features, self.in_features)
        if self.has_nvfp4:
            pad_n, pad_k = (-self.weight.size(0)) % 32, (-self.weight.size(1)) % 16
            if pad_n or pad_k:
                self.weight.data = F.pad(self.weight, (0, pad_k, 0, pad_n))
        self._weights_transformed = True

    def _output_allocation(self):
        if self.all_reduce is not None:
            return self.all_reduce.output_allocation()
        return int(BufferKind.DEFAULT), None

    def _low_m_runner_for(self, x: torch.Tensor):
        if torch.compiler.is_compiling() or x.ndim == 0 or x.shape[-1] == 0:
            return None
        if low_m_gemm_takes_rows(self, x.numel() // x.shape[-1]):
            return low_m_gemm_dispatcher(self)
        return None

    def _fp8(self, x, bias):
        orig = x.shape
        if (
            self.input_scale is not None
            and not self.force_dynamic_quantization
            and bias is None
            and (runner := self._low_m_runner_for(x)) is not None
        ):
            result = runner.apply_fp8(
                x, self.weight, self.weight_scale, self.input_scale, self.dtype
            )
            if result is not None:
                return result
        x = x.reshape(-1, orig[-1])
        scale = self.input_scale
        if x.dtype != torch.float8_e4m3fn:
            if scale is None or self.force_dynamic_quantization:
                x, scale = native_fp8_per_tensor_quantize(x)
                scale = scale.float()
            elif (
                not torch.compiler.is_compiling()
                and x.dtype == torch.bfloat16
                and x.is_contiguous()
                and x.numel() >= 1 << 23
            ):
                x = fp8_static_quantize(x, scale)
            else:
                x = native_static_fp8_per_tensor_quantize(x, scale)
        kind, group = self._output_allocation()
        output = fp8_scaled_gemm(
            x,
            self.weight,
            scale,
            self.weight_scale,
            self.dtype,
            self.enable_cuda_core and x.shape[0] <= 8,
            self.use_rubin_fp8,
            kind,
            group,
        )
        output = output.reshape(*orig[:-1], self.out_features)
        return output + bias if bias is not None else output

    def _nvfp4(self, x, bias):
        orig = x.shape
        x = x.reshape(-1, orig[-1])
        amax = x.abs().amax().float()
        scale = (448.0 * 6.0) / amax
        alpha = (amax / (448.0 * 6.0)) * self.weight_scale_2
        q, sf = tunable_fp4_quantize(x, scale, 16)
        if self.use_fused_gemm_allreduce:
            output = nvfp4_gemm_allreduce(
                q,
                self.weight,
                sf,
                self.weight_scale,
                alpha,
                self.dtype,
                self.tp_rank,
                self.mapping.tp_group,
            )
        else:
            kind, group = self._output_allocation()
            fuse_bias = (
                bias is not None
                and kind == int(BufferKind.DEFAULT)
                and self.weight.shape[0] == self.out_features
            )
            output = nvfp4_gemm(
                q,
                self.weight,
                sf,
                self.weight_scale,
                alpha,
                self.dtype,
                kind,
                self.nvfp4_backends,
                group,
                bias if fuse_bias else None,
            )
            if fuse_bias:
                bias = None
        output = (
            output[..., : self.out_features].contiguous().reshape(*orig[:-1], self.out_features)
        )
        return output + bias if bias is not None else output

    def forward(self, x):
        bias = self.bias if self.tp_mode != "row" or self.tp_rank == 0 else None
        if self.has_nvfp4:
            y = self._nvfp4(x, bias)
        elif self.has_fp8_qdq:
            y = self._fp8(x, bias)
        elif self.has_fp8_block_scales:
            orig = x[0].shape if isinstance(x, tuple) else x.shape
            if not isinstance(x, tuple):
                x = x.reshape(-1, orig[-1])
                if x.dtype == torch.float8_e4m3fn:
                    x = x.to(torch.bfloat16) * self.input_scale
            y = fp8_block_gemm(
                x,
                self.weight,
                self.weight_scale,
                self.packed_sm100,
                self.packed_sm120,
                self.fp8_low_m_limit,
            ).reshape(*orig[:-1], self.out_features)
            if bias is not None:
                y = y + bias
        else:
            runner = self._low_m_runner_for(x)
            y = runner.apply(self, x, self.weight, bias) if runner is not None else None
            if y is None:
                y = F.linear(x, self.weight, bias)
        if self.all_reduce is not None and not self.use_fused_gemm_allreduce:
            y = self.all_reduce(y)
        return y
