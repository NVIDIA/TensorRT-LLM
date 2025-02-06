import enum
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams

from ...models.modeling_utils import QuantConfig
from ..distributed import ParallelConfig, TensorParallelMode

E2M1_MAX = 6.0


class WeightMode(str, enum.Enum):
    # weight of a vanilla layer
    VANILLA = 'vanilla'
    # weight of a fused QKV linear layer
    FUSED_QKV_LINEAR = 'fused_qkv_linear'
    # weight of a fused gate and up linear layer
    FUSED_GATE_UP_LINEAR = 'fused_gate_up_linear'


@dataclass(kw_only=True)
class WeightsLoadingConfig:
    weight_mode: WeightMode = WeightMode.VANILLA
    ignore_tensor_parallel: bool = False


def load_weight_shard(
        weight,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    if isinstance(weight, torch.Tensor):
        tensor_shape = weight.shape

        def maybe_convert_to_torch_tensor(tensor: torch.Tensor,
                                          indices: slice = None):
            if indices == None:
                # Avoid unnecessary copy
                return tensor.to(device)
            else:
                return tensor[indices].to(device)
    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to the module
    # safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice | tuple[slice]] = slice(None)):
            return tensor[indices].to(device)
    else:
        raise ValueError(f'unsupported weight type: {type(weight)}')
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
    slice_obj = [slice(None)] * len(tensor_shape)
    slice_obj[split_dim] = slice(slice_start, slice_end)
    return maybe_convert_to_torch_tensor(weight, tuple(slice_obj))


def load_weight_scales_fp8_qdq(weights: List[Dict]):
    input_scale, weight_scale = [], []
    for w in weights:
        if "input_scale" in w:
            input_scale.append(w["input_scale"][...].reshape([]))
        if "weight_scale" in w:
            weight_scale.append(w["weight_scale"][...].reshape([]))
    return input_scale, weight_scale


def load_weight_scales_nvfp4(weights: List[Dict],
                             tp_size: int = 1,
                             tp_rank: int = 0,
                             tp_mode: Optional[TensorParallelMode] = None):
    # For concatenated weights (qkv_proj / up_gate_proj), the global scaling factors and input scaling factors should be shared.
    input_scale = None
    weight_scale_2 = None
    weight_scale = []

    for w in weights:
        if "input_scale" in w:
            if input_scale is None:
                input_scale = w["input_scale"][...]
            else:
                assert input_scale == w["input_scale"][
                    ...], "The input_scale should be same for all the weights"
        if "weight_scale" in w:
            ws = load_weight_shard(w["weight_scale"], tp_size, tp_rank,
                                   tp_mode).contiguous()
            assert ws.dtype == torch.float8_e4m3fn  # TODO: or e8m0 for mxfp4 recipe?
            weight_scale.append(ws.view(fp4_utils.float4_sf_dtype))
        if "weight_scale_2" in w:
            if weight_scale_2 is None:
                weight_scale_2 = w["weight_scale_2"][...]
            else:
                assert weight_scale_2 == w["weight_scale_2"][
                    ...], "The weight_scale_2 should be same for all the weights"

    # Compute scaling factor and alpha required by GEMM kernels
    # TODO: ModelOpt's o_proj.weight_scale_2 is bfloat16, which should be float32
    alpha = input_scale.float() * weight_scale_2.float()
    # modelopt ckpt stores amax/(448*6), convert to (448*6)/amax
    input_scale = 1.0 / input_scale

    return input_scale, weight_scale, alpha


class Linear(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dtype: torch.dtype = None,
                 parallel_config: Optional[ParallelConfig] = None,
                 quant_config: Optional[QuantConfig] = None,
                 weights_loading_config: Optional[WeightsLoadingConfig] = None):
        from tensorrt_llm._torch.distributed import AllReduce

        super().__init__()
        self.dtype = dtype
        self.parallel_config = parallel_config or ParallelConfig()
        self.quant_config = quant_config
        self.weights_loading_config = weights_loading_config or WeightsLoadingConfig(
        )
        self.tp_size = self.parallel_config.tensor_parallel_size
        self.tp_rank = self.parallel_config.tensor_parallel_rank
        self.tp_mode = self.parallel_config.tensor_parallel_mode

        local_in_features = in_features
        local_out_features = out_features

        if self.parallel_config.tensor_parallel_mode == TensorParallelMode.ROW:
            assert in_features % self.tp_size == 0, (
                f'in_features {in_features} must be divisible by tp_size {self.tp_size}'
            )
            local_in_features = in_features // self.tp_size
        elif self.parallel_config.tensor_parallel_mode == TensorParallelMode.COLUMN:
            assert out_features % self.tp_size == 0, (
                f'out_features {out_features} must be divisible by tp_size {self.tp_size}'
            )
            local_out_features = out_features // self.tp_size
        else:
            assert self.parallel_config.tensor_parallel_mode is None, (
                'unsupported tensor parallel mode: {self.parallel_config.tensor_parallel_mode}'
            )

        self.in_features = local_in_features
        self.out_features = local_out_features

        self.all_reduce = AllReduce(self.parallel_config)

        weight_shape = (self.out_features, self.in_features)
        self.has_any_quant = False
        self.has_fp8_qdq = False
        self.has_nv_fp4 = False
        # only create_weights, and load quantized weight directly.
        if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
        ):
            self.has_any_quant = True
            qc = self.quant_config
            if qc.layer_quant_mode.has_fp8_qdq():
                self.has_fp8_qdq = True
                self.weight = Parameter(torch.empty(weight_shape,
                                                    dtype=torch.float8_e4m3fn),
                                        requires_grad=False)
                self.weight_scale = Parameter(torch.tensor(1.,
                                                           dtype=torch.float32),
                                              requires_grad=False)
                self.input_scale = Parameter(torch.tensor(1.,
                                                          dtype=torch.float32),
                                             requires_grad=False)
                self.inv_input_scale = Parameter(torch.tensor(
                    1., dtype=torch.float32),
                                                 requires_grad=False)
            elif qc.layer_quant_mode.has_nvfp4():
                self.has_nv_fp4 = True
                self.scaling_vector_size = 16
                assert self.in_features % self.scaling_vector_size == 0, f"in_features {self.in_features} must be divisible by scaling_vector_size {self.scaling_vector_size}"

                # Quantized weights
                self.weight = Parameter(torch.empty(
                    [self.out_features, self.in_features // 2],
                    dtype=fp4_utils.float4_e2m1x2),
                                        requires_grad=False)

                # FP8 per-block scaling factors. dtype must be aligned with SF_DTYPE
                # Padding is required. See computeSFSize in quantization.h
                nrows = fp4_utils.pad_up(self.out_features, 128)
                ncols = fp4_utils.pad_up(
                    self.in_features // self.scaling_vector_size, 4)
                self.weight_scale = Parameter(torch.empty(
                    [nrows * ncols], dtype=fp4_utils.float4_sf_dtype),
                                              requires_grad=False)

                # FP32 per-tensor global scaling factor = 448*6/amax_input
                self.input_scale = Parameter(torch.empty([1],
                                                         dtype=torch.float32),
                                             requires_grad=False)
                self.inv_input_scale = Parameter(torch.empty(
                    [1], dtype=torch.float32),
                                                 requires_grad=False)

                # (amax_input*amax_weight) / (448*6*448*6)
                self.alpha = Parameter(torch.empty([1], dtype=torch.float32),
                                       requires_grad=False)

                self.profiler = torch.classes.trtllm.FP4GemmRunner.get_instance(
                    dtype)
                self.needs_profiling = True

            else:
                # TODO(zhenhuanc): support other quant mode
                raise ValueError(f'unsupported quant mode: {qc.quant_mode}')
        else:
            self.weight = Parameter(torch.empty(weight_shape, dtype=dtype),
                                    requires_grad=False)

        if bias:
            self.bias = Parameter(torch.empty((self.out_features, ),
                                              dtype=dtype),
                                  requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def apply_linear(self, input, weight, bias):
        if self.has_any_quant:
            qc = self.quant_config
            if self.has_fp8_qdq:
                if input.dtype != torch.float8_e4m3fn:
                    qinput, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                        input, self.input_scale)
                else:
                    qinput = input
                output = torch.ops.trtllm.cublas_scaled_mm(
                    qinput,
                    weight.t(),
                    scale_a=self.input_scale,
                    scale_b=self.weight_scale,
                    bias=bias,
                    out_dtype=self.dtype or input.dtype,
                )
            elif self.has_nv_fp4:
                m = math.prod(input.shape[:-1])
                n = self.weight.shape[0]
                k = self.weight.shape[1] * 2

                if self.needs_profiling:
                    self.needs_profiling = False
                    self.profiler.run_profile(n, k, fp4_utils.fp4_buckets)

                act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                    input, self.input_scale, self.scaling_vector_size, False)

                best_config_id = self.profiler.get_best_config_id(m, n, k)
                output = self.profiler.run_gemm(act_fp4, self.weight, act_sf,
                                                self.weight_scale, self.alpha,
                                                False, best_config_id)
            else:
                # TODO(zhenhuanc): support other quant mode
                raise ValueError(f'unsupported quant mode: {qc.quant_mode}')
        else:
            output = F.linear(input, self.weight, bias)
        return output

    def forward(
            self,
            input: torch.Tensor,
            *,
            all_reduce_params: Optional[AllReduceParams] = None
    ) -> torch.Tensor:
        from tensorrt_llm._torch.distributed import allgather

        if self.tp_mode == TensorParallelMode.ROW:
            bias = None if (self.tp_rank > 0) else self.bias
            if self.tp_size > 1:
                fuse_bias_into_all_reduce = (
                    bias is not None and all_reduce_params is not None
                    and (all_reduce_params.fusion_op
                         == AllReduceFusionOp.RESIDUAL_RMS_NORM))
                if fuse_bias_into_all_reduce:
                    all_reduce_params.bias = bias
                    bias = None
            else:
                assert all_reduce_params is None, "Cannot fuse norm/residual/bias ops into allreduce op since we do not call allreduce op when tp_size is 1."
            output = self.apply_linear(input, self.weight, bias)
            output = self.all_reduce(
                output,
                all_reduce_params=all_reduce_params,
            )
        elif self.tp_mode == TensorParallelMode.COLUMN:
            output = self.apply_linear(input, self.weight, self.bias)
            if self.parallel_config.gather_output:
                output = allgather(output, self.parallel_config)
        else:
            output = self.apply_linear(input, self.weight, self.bias)

        return output

    def load_weights(self, weights: List[Dict]):

        def copy(dst: Parameter, src: torch.Tensor):
            assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
            dst.data.copy_(src)

        weight_mode = self.weights_loading_config.weight_mode
        quant_mode = self.quant_config.quant_mode if self.quant_config else None
        # load weight shard onto GPU to speed up operations on the shards
        device = torch.device('cuda')

        if weight_mode == WeightMode.VANILLA:
            assert len(weights) == 1

            weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                       self.tp_rank, self.tp_mode, device)
            copy(self.weight, weight)

            if self.bias is not None:
                bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
                copy(self.bias, bias)

            if quant_mode:
                if quant_mode.has_fp8_qdq():
                    input_scale, weight_scale = load_weight_scales_fp8_qdq(
                        weights)
                    copy(self.input_scale, input_scale[0])
                    copy(self.weight_scale, weight_scale[0])
                    self.inv_input_scale.data = 1.0 / self.input_scale
                elif quant_mode.has_nvfp4():
                    input_scale, weight_scale, alpha = load_weight_scales_nvfp4(
                        weights,
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        tp_mode=self.tp_mode)
                    assert len(weights) == 1
                    weight_scale = weight_scale[0]
                    # Swizzle weight scale
                    weight_scale = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        weight_scale)
                    copy(self.input_scale, input_scale)
                    copy(self.weight_scale, weight_scale)
                    self.inv_input_scale.data = self.input_scale / E2M1_MAX
                    copy(self.alpha, alpha)

        elif weight_mode == WeightMode.FUSED_QKV_LINEAR:
            assert len(weights) == 3

            q_weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
            k_weight = load_weight_shard(weights[1]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
            v_weight = load_weight_shard(weights[2]['weight'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)

            if quant_mode:
                if quant_mode.has_fp8_qdq():
                    input_scale, weight_scale = load_weight_scales_fp8_qdq(
                        weights)
                    copy(self.input_scale, max(input_scale))
                    copy(self.weight_scale, max(weight_scale))
                    q_weight = q_weight.to(self.dtype) * weight_scale[0]
                    k_weight = k_weight.to(self.dtype) * weight_scale[1]
                    v_weight = v_weight.to(self.dtype) * weight_scale[2]
                elif quant_mode.has_nvfp4():
                    input_scale, weight_scale, alpha = load_weight_scales_nvfp4(
                        weights,
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        tp_mode=self.tp_mode)
                    # Swizzle weight scales after concatenation
                    weight_scale = torch.cat(weight_scale, 0)
                    weight_scale = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        weight_scale)
                    copy(self.input_scale, input_scale)
                    copy(self.weight_scale, weight_scale)
                    copy(self.alpha, alpha)

            fused_weight = torch.cat((q_weight, k_weight, v_weight))

            if quant_mode and quant_mode.has_fp8_qdq():
                fused_weight = (fused_weight / self.weight_scale).to(
                    torch.float8_e4m3fn)

            copy(self.weight, fused_weight)

            if self.bias is not None:
                q_bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                k_bias = load_weight_shard(weights[1]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                v_bias = load_weight_shard(weights[2]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                copy(self.bias, torch.cat((q_bias, k_bias, v_bias)))
        elif weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
            assert len(weights) == 2

            gate_weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                            self.tp_rank, self.tp_mode, device)
            up_weight = load_weight_shard(weights[1]['weight'], self.tp_size,
                                          self.tp_rank, self.tp_mode, device)
            if quant_mode:
                if quant_mode.has_fp8_qdq():
                    input_scale, weight_scale = load_weight_scales_fp8_qdq(
                        weights)
                    copy(self.input_scale, max(input_scale))
                    copy(self.weight_scale, max(weight_scale))
                    gate_weight = gate_weight.to(self.dtype) * weight_scale[0]
                    up_weight = up_weight.to(self.dtype) * weight_scale[1]
                elif quant_mode.has_nvfp4():
                    input_scale, weight_scale, alpha = load_weight_scales_nvfp4(
                        weights,
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        tp_mode=self.tp_mode)
                    # Swizzle weight scales after concatenation
                    weight_scale = torch.cat(weight_scale, 0)
                    weight_scale = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        weight_scale)
                    copy(self.input_scale, input_scale)
                    copy(self.weight_scale, weight_scale)
                    copy(self.alpha, alpha)

            fused_weight = torch.cat((gate_weight, up_weight))

            if quant_mode and quant_mode.has_fp8_qdq():
                fused_weight = (fused_weight / self.weight_scale).to(
                    torch.float8_e4m3fn)

            copy(self.weight, fused_weight)

            if self.bias is not None:
                gate_bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                              self.tp_rank, self.tp_mode,
                                              device)
                up_bias = load_weight_shard(weights[1]['bias'], self.tp_size,
                                            self.tp_rank, self.tp_mode, device)
                copy(self.bias, torch.cat((up_bias, gate_bias)))
        else:
            raise ValueError(f'unsupported weight mode: {weight_mode}')
