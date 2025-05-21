import enum
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.peft.lora.layer import LoraLayer
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams
from tensorrt_llm.mapping import Mapping

from ...models.modeling_utils import QuantConfig
from ..quantization.quant import LinearQuantCreator
from ..utils import Fp4QuantizedTensor

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


class TensorParallelMode(str, enum.Enum):
    COLUMN = 'column'
    ROW = 'row'

    @classmethod
    def split_dim(cls, mode):
        return 1 if mode == cls.ROW else 0


INPUT_SCALE_NAME = "input_scale"
WEIGHT_SCALE_NAME = "weight_scale"
WEIGHT_SCALE_2_NAME = "weight_scale_2"


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
            if indices is None:
                # Avoid unnecessary copy
                return tensor.to(device)
            else:
                return tensor[indices].to(device)
    # WAR to check whether it is a safetensor slice since safetensor didn't register the type to the module
    # safetensors slice, supports lazy loading, type(weight) is `builtin.PySafeSlice`
    elif hasattr(weight, "get_shape"):
        tensor_shape = weight.get_shape()

        def maybe_convert_to_torch_tensor(
            tensor, indices: Union[slice, tuple[slice]] = slice(None)):
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
    weight_scale = []
    for w in weights:
        if WEIGHT_SCALE_NAME in w:
            weight_scale.append(w[WEIGHT_SCALE_NAME][...].reshape([]))
    return weight_scale


def load_weight_scales_nvfp4(weights: List[Dict],
                             tp_size: int = 1,
                             tp_rank: int = 0,
                             tp_mode: Optional[TensorParallelMode] = None):
    # For concatenated weights (qkv_proj / up_gate_proj), the global scaling factors and input scaling factors should be shared.
    input_scale = None
    weight_scale_2 = None
    weight_scale = []

    device = torch.device("cuda")

    for w in weights:
        if INPUT_SCALE_NAME in w:
            if input_scale is None:
                input_scale = w[INPUT_SCALE_NAME][...]
            else:
                assert input_scale == w[INPUT_SCALE_NAME][
                    ...], "The input_scale should be same for all the weights"
        if WEIGHT_SCALE_NAME in w:
            ws = load_weight_shard(w[WEIGHT_SCALE_NAME],
                                   tp_size,
                                   tp_rank,
                                   tp_mode,
                                   device=device).contiguous()
            assert ws.dtype == torch.float8_e4m3fn  # TODO: or e8m0 for mxfp4 recipe?
            weight_scale.append(ws.view(fp4_utils.float4_sf_dtype))
        if WEIGHT_SCALE_2_NAME in w:
            if weight_scale_2 is None:
                weight_scale_2 = w[WEIGHT_SCALE_2_NAME][...]
            else:
                assert weight_scale_2 == w[WEIGHT_SCALE_2_NAME][
                    ...], "The weight_scale_2 should be same for all the weights"

    # Compute scaling factor and alpha required by GEMM kernels
    # TODO: ModelOpt's o_proj.weight_scale_2 is bfloat16, which should be float32
    alpha = input_scale.float() * weight_scale_2.float()

    return weight_scale, alpha


class Linear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,  # COLUMN parallel only
        quant_config: Optional[QuantConfig] = None,
        weights_loading_config: Optional[WeightsLoadingConfig] = None,
        reduce_output: bool = True,  # ROW parallel only
        skip_create_weights_in_init: bool = False,
        use_custom_cublas_mm: bool = False,
        lora: Optional[LoraLayer] = None,
    ):
        from ..distributed import AllReduce

        super().__init__()
        self.has_bias = bias
        self.dtype = dtype
        self.mapping = mapping or Mapping()
        # could be modified later
        self.quant_config = quant_config
        self.weights_loading_config = weights_loading_config or WeightsLoadingConfig(
        )
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.tp_mode = tensor_parallel_mode
        self.gather_output = gather_output

        local_in_features = in_features
        local_out_features = out_features

        if self.tp_mode == TensorParallelMode.ROW:
            assert in_features % self.tp_size == 0, (
                f'in_features {in_features} must be divisible by tp_size {self.tp_size}'
            )
            local_in_features = in_features // self.tp_size
        elif self.tp_mode == TensorParallelMode.COLUMN:
            assert out_features % self.tp_size == 0, (
                f'out_features {out_features} must be divisible by tp_size {self.tp_size}'
            )
            local_out_features = out_features // self.tp_size
        else:
            assert self.tp_mode is None, (
                'unsupported tensor parallel mode: {self.tp_mode}')

        self.in_features = local_in_features
        self.out_features = local_out_features

        self.all_reduce = AllReduce(self.mapping) if reduce_output else None
        self._weights_created = False
        self.reduce_output = reduce_output
        self.use_custom_cublas_mm = use_custom_cublas_mm
        self.lora = lora
        self.input_quantizer = None
        self._do_linear = None
        self.inv_input_scale = None

        if not skip_create_weights_in_init:
            self.create_weights()

    def _has_any_quant(self):
        return self.has_fp8_qdq or self.has_fp8_block_scales or self.has_nvfp4

    def create_weights(self):
        if self._weights_created:
            return
        weight_shape = (self.out_features, self.in_features)
        self.has_fp8_qdq = False
        self.has_fp8_block_scales = False
        self.has_nvfp4 = False

        self.input_quantizer = LinearQuantCreator.create_quantizer(self.quant_config)

        if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            qc = self.quant_config
            if qc.layer_quant_mode.has_fp8_qdq():
                self.has_fp8_qdq = True
                self.weight = Parameter(torch.empty(weight_shape,
                                                    dtype=torch.float8_e4m3fn),
                                        requires_grad=False)
                self.weight_scale = Parameter(torch.tensor(1.,
                                                           dtype=torch.float32),
                                              requires_grad=False)
                self._do_linear = self._apply_linear_fp8_qdq
            elif qc.layer_quant_mode.has_fp8_block_scales():
                self.has_fp8_block_scales = True
                self.weight = Parameter(torch.empty(weight_shape,
                                                    dtype=torch.float8_e4m3fn),
                                        requires_grad=False)
                scale_shape = (math.ceil(self.out_features / 128),
                               math.ceil(self.in_features / 128))
                self.weight_scale = Parameter(torch.empty(scale_shape,
                                                          dtype=torch.float32),
                                              requires_grad=False)
                self._do_linear = self._apply_linear_fp8_block_scale
            elif qc.layer_quant_mode.has_nvfp4():
                self.has_nvfp4 = True
                self.scaling_vector_size = self.input_quantizer.scaling_vector_size
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
                self.alpha = Parameter(torch.empty([1], dtype=torch.float32),
                                       requires_grad=False)
                self._do_linear = self._apply_linear_nvfp4
            else:
                # TODO(zhenhuanc): support other quant mode
                raise ValueError(f'unsupported quant mode: {qc.quant_mode}')
        else:
            self.weight = Parameter(torch.empty(weight_shape, dtype=self.dtype),
                                    requires_grad=False)
            self._do_linear = self._apply_linear

        if self.has_bias:
            self.bias = Parameter(torch.empty((self.out_features, ),
                                              dtype=self.dtype),
                                  requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._weights_created = True

    def _apply_linear_fp8_qdq(self, input, input_scale, bias):
        # This op does not support bias now.
        output = torch.ops.trtllm.cublas_scaled_mm(
            input,
            self.weight.t(),
            scale_a=input_scale,
            scale_b=self.weight_scale,
            bias=None,
            out_dtype=self.dtype or input.dtype,
        )
        if bias is not None:
            output = output + bias
        return output

    def _apply_linear_fp8_block_scale(self, act_input, input_scale, bias):
        output = torch.ops.trtllm.fp8_block_scaling_gemm(
            act_input, self.weight, input_scale, self.weight_scale)
        if bias is not None:
            output = output + bias
        return output

    def _apply_linear_nvfp4(self, act_input, input_scale, bias):
        output = torch.ops.trtllm.nvfp4_gemm(act_input, self.weight,
                                             input_scale, self.weight_scale,
                                             self.alpha, False, self.dtype)
        if bias is not None:
            output = output + bias

        return output

    def _apply_linear_not_supported_quant(self, act_input, input_scale, bias):
        # TODO(zhenhuanc): support other quant mode
        raise ValueError(
            f'unsupported quant mode: {self.quant_config.quant_mode}')

    def _apply_linear(self, act_input, input_scale, bias):
        if self.use_custom_cublas_mm:
            output = torch.ops.trtllm.cublas_mm(act_input,
                                                self.weight.t(),
                                                bias,
                                                out_dtype=None)
        else:
            output = F.linear(act_input, self.weight, bias)
        return output

    def apply_linear(self,
                     input,
                     bias,
                     lora_params: Optional[dict] | None = None,
                     layer_idx: Optional[int] | None = None):
        act_input = input
        input_scale = None
        if self._has_any_quant():
            if self.input_quantizer is not None:
                act_input, input_scale = self.input_quantizer(input)

        output = self._do_linear(act_input, input_scale, bias)

        if self.lora is not None and bool(lora_params):
            lora_result = self.lora(input, lora_params, layer_idx)
            if lora_result is not None:
                output = output + lora_result
        return output

    def _maybe_fuse_bias_into_allreduce(
        self,
        bias: Optional[torch.Tensor],
        all_reduce_params: Optional[AllReduceParams] = None,
    ) -> bool:
        if self.tp_size > 1:
            fuse_bias_into_all_reduce = (
                bias is not None and all_reduce_params is not None
                and (all_reduce_params.fusion_op
                     == AllReduceFusionOp.RESIDUAL_RMS_NORM))
            if fuse_bias_into_all_reduce:
                all_reduce_params.bias = bias
                return True
        else:
            assert all_reduce_params is None or all_reduce_params.enable_allreduce is False, "Cannot fuse norm/residual/bias ops into allreduce op since we do not call allreduce op when tp_size is 1."
            return False

    def forward(
        self,
        input: Union[torch.Tensor, Fp4QuantizedTensor],
        *,
        all_reduce_params: Optional[AllReduceParams] = None,
        lora_params: Optional[dict] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        from ..distributed import allgather

        if self.tp_mode == TensorParallelMode.ROW:
            bias = None if (self.tp_rank > 0) else self.bias
            if self.reduce_output:
                fuse_bias = self._maybe_fuse_bias_into_allreduce(
                    bias, all_reduce_params)
                bias = None if fuse_bias else bias
                output = self.apply_linear(input, bias, lora_params, layer_idx)
                output = self.all_reduce(
                    output,
                    all_reduce_params=all_reduce_params,
                )
            else:
                output = self.apply_linear(input, bias, lora_params, layer_idx)
        elif self.tp_mode == TensorParallelMode.COLUMN:
            output = self.apply_linear(input, self.bias, lora_params, layer_idx)
            if self.gather_output:
                output = allgather(output, self.mapping)
        else:
            output = self.apply_linear(input, self.bias, lora_params, layer_idx)
        return output

    def load_weights(self, weights: List[Dict]):
        assert self._weights_created

        def _copy(dst: Parameter, src: torch.Tensor):
            # TODO check that is it a reasonable change or not
            if dst.dtype != src.dtype:
                src = src.to(dst.dtype)
            assert dst.dtype == src.dtype, f"Incompatible dtype. dst: {dst.dtype}, src: {src.dtype}"
            dst.data.copy_(src)

        # load weight shard onto GPU to speed up operations on the shards
        device = torch.device('cuda')

        weight_mode = self.weights_loading_config.weight_mode
        quant_mode = self.quant_config.quant_mode if self.quant_config else None
        if self.input_quantizer:
            self.input_quantizer.load_weights(weights, INPUT_SCALE_NAME, device)
            self.input_scale = self.input_quantizer.scale
            self.inv_input_scale = self.input_quantizer.inv_scale

        if weight_mode == WeightMode.VANILLA:
            assert len(weights) == 1

            weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                       self.tp_rank, self.tp_mode, device)
            _copy(self.weight, weight)

            if self.bias is not None:
                bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                         self.tp_rank, self.tp_mode, device)
                _copy(self.bias, bias)

            if quant_mode:
                if quant_mode.has_fp8_qdq():
                    weight_scale = load_weight_scales_fp8_qdq(weights)
                    _copy(self.weight_scale, weight_scale[0])
                elif quant_mode.has_nvfp4():
                    weight_scale, alpha = load_weight_scales_nvfp4(
                        weights,
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        tp_mode=self.tp_mode)
                    assert len(weights) == 1
                    weight_scale = weight_scale[0]
                    # Swizzle weight scale
                    weight_scale = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        weight_scale)
                    _copy(self.weight_scale, weight_scale)
                    _copy(self.alpha, alpha)

                elif quant_mode.has_fp8_block_scales():
                    # `weight_scale_inv` for DS recipe and  `weight_scale` for ModelOpt recipe.
                    # Actually they hold identical values of data_amax / 448.
                    scale_name = "weight_scale_inv"
                    if scale_name not in weights[0]:
                        scale_name = WEIGHT_SCALE_NAME
                    weight_scale = load_weight_shard(weights[0][scale_name],
                                                     self.tp_size, self.tp_rank,
                                                     self.tp_mode, device)
                    _copy(self.weight_scale, weight_scale)

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
                    weight_scale = load_weight_scales_fp8_qdq(weights)
                    _copy(self.weight_scale, max(weight_scale))
                    q_weight = q_weight.to(self.dtype) * weight_scale[0]
                    k_weight = k_weight.to(self.dtype) * weight_scale[1]
                    v_weight = v_weight.to(self.dtype) * weight_scale[2]
                elif quant_mode.has_nvfp4():
                    weight_scale, alpha = load_weight_scales_nvfp4(
                        weights,
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        tp_mode=self.tp_mode)
                    # Swizzle weight scales after concatenation
                    weight_scale = torch.cat(weight_scale, 0)
                    weight_scale = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        weight_scale)
                    _copy(self.weight_scale, weight_scale)
                    _copy(self.alpha, alpha)
                elif quant_mode.has_fp8_block_scales():
                    scale_name = "weight_scale_inv"
                    if scale_name not in weights[0]:
                        scale_name = WEIGHT_SCALE_NAME
                    q_scale = load_weight_shard(weights[0][scale_name],
                                                self.tp_size, self.tp_rank,
                                                self.tp_mode).contiguous()
                    k_scale = load_weight_shard(weights[1][scale_name],
                                                self.tp_size, self.tp_rank,
                                                self.tp_mode).contiguous()
                    v_scale = load_weight_shard(weights[2][scale_name],
                                                self.tp_size, self.tp_rank,
                                                self.tp_mode).contiguous()
                    fused_fp8_block_scale = torch.cat(
                        (q_scale, k_scale, v_scale))
                    _copy(self.weight_scale, fused_fp8_block_scale)

            fused_weight = torch.cat((q_weight, k_weight, v_weight))

            if quant_mode and quant_mode.has_fp8_qdq():
                fused_weight = (fused_weight / self.weight_scale).to(
                    torch.float8_e4m3fn)

            _copy(self.weight, fused_weight)

            if self.bias is not None:
                q_bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                k_bias = load_weight_shard(weights[1]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                v_bias = load_weight_shard(weights[2]['bias'], self.tp_size,
                                           self.tp_rank, self.tp_mode, device)
                _copy(self.bias, torch.cat((q_bias, k_bias, v_bias)))
        elif weight_mode == WeightMode.FUSED_GATE_UP_LINEAR:
            assert len(weights) == 2

            gate_weight = load_weight_shard(weights[0]['weight'], self.tp_size,
                                            self.tp_rank, self.tp_mode, device)
            up_weight = load_weight_shard(weights[1]['weight'], self.tp_size,
                                          self.tp_rank, self.tp_mode, device)
            if quant_mode:
                if quant_mode.has_fp8_qdq():
                    weight_scale = load_weight_scales_fp8_qdq(weights)
                    _copy(self.weight_scale, max(weight_scale))
                    gate_weight = gate_weight.to(self.dtype) * weight_scale[0]
                    up_weight = up_weight.to(self.dtype) * weight_scale[1]
                elif quant_mode.has_nvfp4():
                    weight_scale, alpha = load_weight_scales_nvfp4(
                        weights,
                        tp_size=self.tp_size,
                        tp_rank=self.tp_rank,
                        tp_mode=self.tp_mode)
                    # Swizzle weight scales after concatenation
                    weight_scale = torch.cat(weight_scale, 0)
                    weight_scale = torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                        weight_scale)
                    _copy(self.weight_scale, weight_scale)
                    _copy(self.alpha, alpha)
                elif quant_mode.has_fp8_block_scales():
                    scale_name = "weight_scale_inv"
                    if scale_name not in weights[0]:
                        scale_name = WEIGHT_SCALE_NAME
                    left_scale = load_weight_shard(weights[0][scale_name],
                                                   self.tp_size, self.tp_rank,
                                                   self.tp_mode, device)
                    right_scale = load_weight_shard(weights[1][scale_name],
                                                    self.tp_size, self.tp_rank,
                                                    self.tp_mode, device)
                    fused_scale = torch.cat([left_scale, right_scale], dim=0)
                    _copy(self.weight_scale, fused_scale)

            fused_weight = torch.cat((gate_weight, up_weight))

            if quant_mode and quant_mode.has_fp8_qdq():
                fused_weight = (fused_weight / self.weight_scale).to(
                    torch.float8_e4m3fn)

            _copy(self.weight, fused_weight)

            if self.bias is not None:
                gate_bias = load_weight_shard(weights[0]['bias'], self.tp_size,
                                              self.tp_rank, self.tp_mode,
                                              device)
                up_bias = load_weight_shard(weights[1]['bias'], self.tp_size,
                                            self.tp_rank, self.tp_mode, device)
                _copy(self.bias, torch.cat((up_bias, gate_bias)))
        else:
            raise ValueError(f'unsupported weight mode: {weight_mode}')
