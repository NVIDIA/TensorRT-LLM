from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from tensorrt_llm._torch.peft.lora.layer import LoraLayer
from tensorrt_llm.mapping import Mapping

from ...models.modeling_utils import QuantConfig
# Reuse the common Triton import setup
from .fused_moe.fused_moe_triton import (IS_TRITON_KERNELS_AVAILABLE,
                                         maybe_update_stride,
                                         swizzle_weight_and_scale)

if IS_TRITON_KERNELS_AVAILABLE:
    from triton_kernels.matmul_ogs import (FlexCtx, PrecisionConfig, matmul_ogs)
    from triton_kernels.numerics import InFlexData

from .linear import (Linear, LinearMethodBase, TensorParallelMode,
                     WeightsLoadingConfig, copy_weight, load_weight_shard,
                     load_weights_fused_gate_up_helper,
                     load_weights_fused_qkv_helper, load_weights_vanilla_helper)


class TritonUnquantizedLinearMethod(LinearMethodBase):

    def __init__(self):
        super().__init__()
        self.param_transform = {
            "weight_transform": lambda x: x.T.unsqueeze(0),
            "bias_transform": lambda x: x.unsqueeze(0)
        }

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        weight_shape = (1, in_features, out_features)
        module.weight = Parameter(torch.empty(weight_shape, dtype=dtype),
                                  requires_grad=False)

        if bias:
            module.bias = Parameter(
                torch.empty((1, out_features), dtype=torch.float32
                            ),  # Triton kernels expect bias in float32
                requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        output = matmul_ogs(
            input,
            module.weight,
            module.bias,
            None,  # Routing data is not used here
            gather_indx=None,
            precision_config=None)
        return output

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]):
        load_weights_vanilla_helper(module, weights, **self.param_transform)
        module.weight.data = maybe_update_stride(module.weight.data)

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]):
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights, **self.param_transform)
        fused_weight = torch.cat(
            (q_weight, k_weight, v_weight), axis=-1
        )  #Each of them has shape (1, in_features, out_features_part)
        copy_weight(module.weight, fused_weight)
        module.weight.data = maybe_update_stride(module.weight.data)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]):
        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights, **self.param_transform)
        fused_weight = torch.cat(
            (gate_weight, up_weight), axis=-1
        )  #Each of them has shape (1, in_features, out_features_part)
        copy_weight(module.weight, fused_weight)
        module.weight.data = maybe_update_stride(module.weight.data)


class TritonFP8QDQLinearMethod(LinearMethodBase):

    def __init__(self):
        super().__init__()
        self.param_transform = {
            "weight_transform": lambda x: x.T.unsqueeze(0),
            "bias_transform": lambda x: x.unsqueeze(0)
        }

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        weight_shape = (1, in_features, out_features)
        module.weight = Parameter(torch.empty(weight_shape,
                                              dtype=torch.float8_e4m3fn),
                                  requires_grad=False)
        module.weight_scale = Parameter(torch.empty((1, ), dtype=torch.float32),
                                        requires_grad=False)
        module.input_scale = Parameter(torch.empty((1, ), dtype=torch.float32),
                                       requires_grad=False)

        if bias:
            module.bias = Parameter(
                torch.empty((1, out_features), dtype=torch.float32
                            ),  # Triton kernels expect bias in float32
                requires_grad=False)
        else:
            module.register_parameter("bias", None)

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        cur_input_scale = module.input_scale
        if input.dtype != torch.float8_e4m3fn:
            if module.input_scale is not None:
                # Static quantization
                qinput, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    input, module.input_scale)
            else:
                # Dynamic quantization
                qinput, cur_input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                    input)
                cur_input_scale = cur_input_scale.to(torch.float32)

        else:
            qinput = input

        flex_ctx = FlexCtx(
            lhs_data=InFlexData(scale=cur_input_scale),
            rhs_data=InFlexData(scale=module.weight_scale),
        )
        pc = PrecisionConfig(flex_ctx=flex_ctx,
                             allow_tf32=False,
                             out_dtype=module.dtype)
        output = matmul_ogs(
            qinput,
            module.weight,
            module.bias,
            None,  # Routing data is not used here
            gather_indx=None,
            precision_config=pc)
        return output

    def load_weight_scales(self, weights: List[Dict]):
        input_scale, weight_scale = [], []
        for w in weights:
            if "input_scale" in w:
                input_scale.append(w["input_scale"][...].reshape((1, )))
            if "weight_scale" in w:
                weight_scale.append(w["weight_scale"][...].reshape((1, )))
        return input_scale, weight_scale

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]):
        load_weights_vanilla_helper(module, weights, **self.param_transform)
        input_scale, weight_scale = self.load_weight_scales(weights)
        if len(input_scale) != 0:
            # Static quantization
            copy_weight(module.input_scale, input_scale[0])
        else:
            # Dynamic quantization
            module.input_scale = None
        copy_weight(module.weight_scale, weight_scale[0])
        module.weight.data = maybe_update_stride(module.weight.data)

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]):
        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(
            module, weights, **self.param_transform)

        input_scale, weight_scale = self.load_weight_scales(weights)
        if len(input_scale) != 0:
            # Static quantization
            copy_weight(module.input_scale, max(input_scale))
        else:
            # Dynamic quantization
            module.input_scale = None
        copy_weight(module.weight_scale, max(weight_scale))

        q_weight = q_weight.to(module.dtype) * weight_scale[0]
        k_weight = k_weight.to(module.dtype) * weight_scale[1]
        v_weight = v_weight.to(module.dtype) * weight_scale[2]

        fused_weight = torch.cat((q_weight, k_weight, v_weight))
        fused_weight = (fused_weight / module.weight_scale).to(
            torch.float8_e4m3fn)
        copy_weight(module.weight,
                    self.param_transform["weight_transform"](fused_weight))
        module.weight.data = maybe_update_stride(module.weight.data)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]):
        input_scale, weight_scale = self.load_weight_scales(weights)
        if len(input_scale) != 0:
            # Static quantization
            copy_weight(module.input_scale, max(input_scale))
        else:
            # Dynamic quantization
            module.input_scale = None
        copy_weight(module.weight_scale, max(weight_scale))

        gate_weight, up_weight = load_weights_fused_gate_up_helper(
            module, weights, **self.param_transform)

        gate_weight = gate_weight.to(module.dtype) * weight_scale[0]
        up_weight = up_weight.to(module.dtype) * weight_scale[1]
        fused_weight = torch.cat((gate_weight, up_weight))
        fused_weight = (fused_weight / module.weight_scale).to(
            torch.float8_e4m3fn)
        copy_weight(module.weight,
                    self.param_transform["weight_transform"](fused_weight))
        module.weight.data = maybe_update_stride(module.weight.data)


class TritonMXFP4LinearMethod(LinearMethodBase):

    def __init__(self, activation_dtype):
        super().__init__()
        assert activation_dtype in [torch.float8_e4m3fn, torch.bfloat16], \
            f"TritonMXFP4LinearMethod only supports float8_e4m3fn or bfloat16 activation, got {activation_dtype}"
        self.activation_dtype = activation_dtype

    def create_weights(self, module: Linear, in_features: int,
                       out_features: int, bias: bool, dtype: torch.dtype):
        # Create weight
        assert in_features % 2 == 0, "in_features must be even for MXFP4"
        weight_shape = (1, in_features // 2, out_features)
        module.weight = Parameter(torch.empty(weight_shape, dtype=torch.uint8),
                                  requires_grad=False)

        # Create weight scale
        scale_shape = (1, in_features // 32, out_features
                       )  # Block size is 32 for MXFP4
        module.weight_scale = Parameter(torch.empty(scale_shape,
                                                    dtype=torch.uint8),
                                        requires_grad=False)

        # Create bias
        if bias:
            module.bias = Parameter(
                torch.empty((1, out_features), dtype=torch.float32
                            ),  # Triton kernels expect bias in float32
                requires_grad=False)
        else:
            module.bias = None

        # Create input scale
        if self.activation_dtype == torch.float8_e4m3fn:
            module.input_scale = Parameter(torch.empty((1, ),
                                                       dtype=torch.float32),
                                           requires_grad=False)
        else:
            module.input_scale = None

    def apply(self, module: Linear, input: torch.Tensor,
              bias: Optional[torch.Tensor]):
        if self.activation_dtype == torch.float8_e4m3fn:
            if input.dtype != torch.float8_e4m3fn:
                if module.input_scale is not None:
                    # Static quantization
                    input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                        input, module.input_scale)
                    input_scale = module.input_scale
                else:
                    # Dynamic quantization
                    input, input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(
                        input)
            else:
                assert module.input_scale is not None
                input_scale = module.input_scale

        if self.activation_dtype == torch.float8_e4m3fn:
            flex_ctx = FlexCtx(lhs_data=InFlexData(scale=input_scale), )
        else:
            flex_ctx = FlexCtx()
        pc = PrecisionConfig(weight_scale=module.weight_scale,
                             flex_ctx=flex_ctx,
                             allow_tf32=False,
                             out_dtype=module.dtype)
        output = matmul_ogs(
            input,
            module.weight,
            module.bias,
            None,  # Routing data is not used here
            gather_indx=None,
            precision_config=pc)
        return output

    def load_weights_common(self, module: Linear, weights_list: List[Dict]):
        device = torch.device('cuda')
        processed_weights = []
        weight_scales = []
        biases = []
        input_scales = []
        for w in weights_list:
            current_weight = load_weight_shard(w['weight'], module.tp_size,
                                               module.tp_rank, module.tp_mode,
                                               device)
            current_scale = load_weight_shard(w['weight_scale'], module.tp_size,
                                              module.tp_rank, module.tp_mode,
                                              device)
            current_bias = load_weight_shard(
                w['bias'], module.tp_size, module.tp_rank, module.tp_mode,
                device) if module.bias is not None else None

            processed_weights.append(current_weight)
            weight_scales.append(current_scale)
            if current_bias is not None:
                biases.append(current_bias)
            if "input_scale" in w:
                input_scales.append(w["input_scale"][...].reshape([]))
        # handle weights
        fused_weight = torch.cat(
            processed_weights)  # (out_features, in_features//2)
        fused_weight = fused_weight.T.unsqueeze(
            0)  # (1, in_features//2, out_features)

        # handle scales
        fused_scale = torch.cat(
            weight_scales)  # (out_features, in_features//32)
        fused_scale = fused_scale.T.unsqueeze(
            0)  # (1, in_features//32, out_features)
        fused_weight, fused_scale = swizzle_weight_and_scale(
            fused_weight, fused_scale)
        assert module.weight_scale.dtype == fused_scale.dtype
        # We need to use Triton tensor wrapper instead of Torch tensor to maintain the correct swizzling layout
        module._parameters.pop('weight', None)
        module._parameters.pop('weight_scale', None)
        torch.cuda.empty_cache()
        module.weight = fused_weight
        module.weight_scale = fused_scale

        # handle biases
        if module.bias is not None:
            fused_bias = torch.cat(biases)  # (out_features, )
            fused_bias = fused_bias.unsqueeze(0)  # (1, out_features)
            copy_weight(module.bias, fused_bias)

        # handle input scales
        if len(input_scales) != 0:
            # Static quantization
            max_input_scale = torch.tensor(max(input_scales)).reshape((1, ))
            copy_weight(module.input_scale, max_input_scale)
        else:
            # Dynamic quantization
            module.input_scale = None

    def load_weights_vanilla(self, module: Linear, weights: List[Dict]):
        assert len(weights) == 1
        self.load_weights_common(module, weights)

    def load_weights_fused_qkv_linear(self, module: Linear,
                                      weights: List[Dict]):
        assert len(weights) == 3
        self.load_weights_common(module, weights)

    def load_weights_fused_gate_up_linear(self, module: Linear,
                                          weights: List[Dict]):
        assert len(weights) == 2
        self.load_weights_common(module, weights)


class TritonLinear(Linear):
    """
    A Linear module that uses Triton for the forward pass.
    """

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
        if not IS_TRITON_KERNELS_AVAILABLE:
            raise ImportError("Triton kernels are not available. "
                              "Please install the required dependencies.")
        assert not use_custom_cublas_mm, "TritonLinear does not support custom cublas mm."

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
            quant_config=quant_config,
            weights_loading_config=weights_loading_config,
            reduce_output=reduce_output,
            skip_create_weights_in_init=skip_create_weights_in_init,
            use_custom_cublas_mm=use_custom_cublas_mm,
            lora=lora)

    # Most of the code can be reused, only change the quant method offloading here.
    def get_quant_method(self, quant_config: Optional[QuantConfig] = None):
        if quant_config is None or not quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            return TritonUnquantizedLinearMethod()
        if quant_config.layer_quant_mode.has_fp8_qdq():
            return TritonFP8QDQLinearMethod()
        if quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
            return TritonMXFP4LinearMethod(activation_dtype=torch.float8_e4m3fn)
        if quant_config.layer_quant_mode.has_w4a16_mxfp4():
            assert self.dtype == torch.bfloat16, "Only bfloat16 is supported for W4A16 MXFP4"
            return TritonMXFP4LinearMethod(activation_dtype=self.dtype)
        raise ValueError(f'unsupported quant mode: {quant_config.quant_mode}')
