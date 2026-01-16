# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import gc
import os
import types

import torch
from torch import nn

from visual_gen.configs.op_manager import LinearOpManager
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.utils.auto_tuner import get_auto_tuner
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import tensorrt_llm  # noqa
except ImportError:
    tensorrt_llm = None

try:
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quant_api  # noqa
except ImportError:
    Float8DynamicActivationFloat8WeightConfig = None
    PerRow = None
    quant_api = None

try:
    import transformer_engine.pytorch.cpp_extensions as ext  # noqa: F401
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import MXFP8Quantizer
except ImportError:
    MXFP8Quantizer = None
    logger.warning("Transformer_engine is not installed")

try:
    from flashinfer import SfLayout, nvfp4_quantize
except ImportError:
    logger.warning("Flashinfer is not installed")


class ditLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = None  # module name in the model
        self.linear_impl = None
        self.input_scale = None
        self.weight_scale = None
        self.offloading = False
        self.weight_scaling_factor_2 = None
        self.hasgelu = False

        if os.environ.get("LOADING_QUANT_CHECKPOINT", "False") == "True":
            self.select_linear_impl()

    def get_offloading_weights(self):
        weight, _, _, _ = self.select_linear_impl()
        return weight

    def set_offloading(self):
        offloading_weight = self.get_offloading_weights()
        self.host_weight = offloading_weight.pin_memory()
        self.offloading = True
        self.main_event = torch.cuda.Event()

        self.pingpong_in = None
        self.pingpong_out = None
        self.offloading_stream = None
        self.offloading_event = None
        self.device_weight = None
        self.next_offloading_layer_weight = None

    def ceil_to_ue8m0(self, x: torch.Tensor):
        return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

    def create_blockwise_quantized_weight(
        self,
        param_value: torch.Tensor,
        block_size: int = 128,
        use_ue8m0: bool = False,
    ):
        # refer to transfromers fp8 128*128 block quantization
        # (https://github.com/huggingface/transformers/blob/main/src/transformers/quantizers/quantizer_finegrained_fp8.py)

        param_value = param_value.to(torch.float32)

        # Get FP8 min/max values
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        block_size_m, block_size_n = block_size, block_size
        rows, cols = param_value.shape[-2:]
        if rows % block_size_m != 0 or cols % block_size_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
            )
        param_value_orig_shape = param_value.shape
        param_value = param_value.reshape(
            -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
        ).permute(0, 1, 3, 2, 4)

        # Calculate scaling factor for each block
        max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
        if use_ue8m0:
            scale = self.ceil_to_ue8m0(max_abs / fp8_max)
        else:
            scale = fp8_max / max_abs
        scale_orig_shape = scale.shape
        scale = scale.unsqueeze(-1).unsqueeze(-1)

        @torch.compiler.disable
        def _quantize(param_value, scale, fp8_min, fp8_max):
            # Quantize the weights
            quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

            quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
            # Reshape back to matrix shape
            quantized_param = quantized_param.reshape(param_value_orig_shape)

            # Reshape scale to match the number of blocks
            scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()

            return quantized_param, scale

        if use_ue8m0:
            quantized_param, scale = _quantize(param_value, 1.0 / scale, fp8_min, fp8_max)
        else:
            quantized_param, scale = _quantize(param_value, scale, fp8_min, fp8_max)
        return quantized_param, scale

    def create_nvfp4_quantized_weight(
        self,
        param_value: torch.Tensor,
        block_size: int = 16,
    ):
        weights_scaling_factor_2 = (
            torch.finfo(torch.float8_e4m3fn).max * 6.0 / torch.max(torch.abs(param_value).to(torch.float))
        )
        param_value = param_value.to(torch.bfloat16)
        packed_weight, weights_scaling_factor = torch.ops.trtllm.fp4_quantize(
            param_value, weights_scaling_factor_2, block_size, False
        )
        return packed_weight, weights_scaling_factor, weights_scaling_factor_2

    def create_per_tensor_quantized_weight(self, param_value: torch.Tensor):
        param_value = param_value.to(torch.float32)

        # Get FP8 min/max values
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        max_abs = torch.amax(torch.abs(param_value))
        scale = fp8_max / max_abs

        @torch.compiler.disable
        def _quantize(param_value, scale, fp8_min, fp8_max):
            quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
            quantized_param = quantized_param.t()
            scale = scale.reshape(1, 1).reciprocal()
            return quantized_param, scale

        quantized_param, scale = _quantize(param_value, scale, fp8_min, fp8_max)
        return quantized_param, scale

    @torch.compiler.disable
    def create_torch_ao_quantized_weight(self, param_value: torch.Tensor):
        if Float8DynamicActivationFloat8WeightConfig is None:
            logger.error("Torchao is not installed")

        param_value = param_value.to(torch.bfloat16)
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        weight = quant_api._float8_dynamic_activation_float8_weight_quantize_tensor(param_value, config)
        weight = torch.nn.Parameter(weight, requires_grad=False)
        weight.extra_repr = types.MethodType(quant_api._linear_extra_repr, self)
        return weight

    def _register_fp8_weight(self, linear_type: str):
        weight_name = linear_type + "_weight"
        weight_scale_name = linear_type + "_weight_scale"
        # compute quantized weight and weight scale if needed
        if linear_type == "trtllm-fp8-blockwise":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_blockwise_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight, requires_grad=False))
                self.register_buffer(weight_scale_name, weight_scale)
        elif linear_type == "trtllm-fp8-per-tensor":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_per_tensor_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight, requires_grad=False))
                self.register_buffer(weight_scale_name, weight_scale)
        elif linear_type == "deepgemm-MXFP8":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_blockwise_quantized_weight(self.weight, use_ue8m0=True)
                self.register_parameter(weight_name, torch.nn.Parameter(weight, requires_grad=False))
                self.register_buffer(weight_scale_name, weight_scale)
        elif linear_type == "torch-ao-fp8":
            if not hasattr(self, weight_name):
                weight = self.create_torch_ao_quantized_weight(self.weight)
                self.register_parameter(weight_name, weight)
        elif linear_type == "te-fp8-blockwise":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_blockwise_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight, requires_grad=False))
                self.register_buffer(weight_scale_name, weight_scale)
        elif linear_type == "te-fp8-per-tensor":
            if not hasattr(self, weight_name) or not hasattr(self, weight_scale_name):
                weight, weight_scale = self.create_per_tensor_quantized_weight(self.weight)
                self.register_parameter(weight_name, torch.nn.Parameter(weight, requires_grad=False))
                self.register_buffer(weight_scale_name, weight_scale)
        else:
            raise NotImplementedError(f"Linear type {linear_type} not implemented")

    def _register_nvfp4_weight(self, linear_type: str):
        weight_name = linear_type + "_weight"
        weight_scale_name = linear_type + "_weight_scale"
        weight_scale_2_name = linear_type + "_weight_scale_2"
        if (
            not hasattr(self, weight_name)
            or not hasattr(self, weight_scale_name)
            or not hasattr(self, weight_scale_2_name)
        ):
            weight, weight_scale, weights_scaling_factor_2 = self.create_nvfp4_quantized_weight(self.weight)
            self.register_parameter(weight_name, torch.nn.Parameter(weight, requires_grad=False))
            self.register_buffer(weight_scale_name, weight_scale)
            self.register_buffer(weight_scale_2_name, weights_scaling_factor_2)

    def _register_fi_nvfp4_weight(self, linear_type: str, sf_layout: SfLayout, do_shuffle: bool):
        weight_name = linear_type + "_weight"
        weight_scale_name = linear_type + "_weight_scale"
        weight_scale_2_name = linear_type + "_weight_scale_2"
        if (
            not hasattr(self, weight_name)
            or not hasattr(self, weight_scale_name)
            or not hasattr(self, weight_scale_2_name)
        ):
            b = self.weight.to(torch.bfloat16)
            b_global_sf = (448 * 6) / b.float().abs().nan_to_num().max()
            b_fp4, b_sf = nvfp4_quantize(b, b_global_sf, sfLayout=sf_layout, do_shuffle=do_shuffle)
            b_fp4 = b_fp4.T
            b_sf = b_sf.T
            self.register_parameter(weight_name, torch.nn.Parameter(b_fp4, requires_grad=False))
            self.register_buffer(weight_scale_name, b_global_sf)
            self.register_buffer(weight_scale_2_name, b_sf)

    def _register_mxfp8_weight(self, linear_type: str):
        if MXFP8Quantizer is None:
            logger.error("TransformerEngine is not installed")

        weight_name = linear_type + "_weight"
        if not hasattr(self, weight_name):
            input_quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
            self.weight = self.weight.to(torch.bfloat16)
            if self.bias is not None:
                self.bias = self.bias.to(torch.bfloat16)
            weight = input_quantizer(self.weight)
            setattr(self, weight_name, weight)

    def _delete_unused_weight(self, linear_type: str):
        weight_name = linear_type + "_weight"
        weight_scale_name = linear_type + "_weight_scale"
        # Free default weight to save memory
        keys_to_delete = []
        for key, _ in self.named_parameters():
            if key == "weight" and linear_type != "default":
                keys_to_delete.append(key)
            elif key.endswith("_weight") and key != weight_name:
                keys_to_delete.append(key)
        for key, _ in self.named_buffers():
            if key.endswith("_weight_scale") and key != weight_scale_name:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            delattr(self, key)
        if keys_to_delete:  # Only clear cache if we actually deleted something
            gc.collect()
            torch.cuda.empty_cache()

    def load_fp4_weight(self, weights, svd_weight_name_table):
        name_array = self.name.split(".")
        prefill = ".".join(name_array[:2])
        name = ".".join(name_array[2:])
        if name in svd_weight_name_table:
            self.svd_qweight = weights[prefill + "." + svd_weight_name_table[name] + ".qweight"].to("cuda").contiguous()
            self.svd_lora_up = weights[prefill + "." + svd_weight_name_table[name] + ".lora_up"].to("cuda").contiguous()
            self.svd_lora_down = (
                weights[prefill + "." + svd_weight_name_table[name] + ".lora_down"].to("cuda").contiguous()
            )
            self.svd_smooth = weights[prefill + "." + svd_weight_name_table[name] + ".smooth"].to("cuda").contiguous()
            self.svd_wscales = weights[prefill + "." + svd_weight_name_table[name] + ".wscales"].to("cuda").contiguous()
            if prefill + "." + svd_weight_name_table[name] + ".wtscale" in weights:
                self.svd_wtscale = weights[prefill + "." + svd_weight_name_table[name] + ".wtscale"].item()
                self.svd_wcscales = None
            else:
                self.svd_wtscale = torch.ones(1, dtype=torch.float32).item()
                self.svd_wcscales = (
                    weights[prefill + "." + svd_weight_name_table[name] + ".wcscales"].to("cuda").contiguous()
                )
            self.svd_bias = weights[prefill + "." + svd_weight_name_table[name] + ".bias"].to("cuda").contiguous()

            if LinearOpManager.linear_type == "svd-nvfp4":
                self.weight = None
                self.bias = None

    @torch.compiler.disable()
    def get_linear_type(self):
        linear_type = LinearOpManager.linear_type
        # in svdquant not all layers are quantized, so we need to check if the layer is quantized
        if linear_type == "svd-nvfp4" and not hasattr(self, "svd_qweight"):
            linear_type = "default"

        if linear_type in ["trtllm-fp8-blockwise", "te-fp8-blockwise"] and hasattr(self, "weight"):
            rows, cols = self.weight.shape[-2:]
            if rows % 128 != 0 or cols % 128 != 0:
                logger.debug(
                    f"layer {self.name} weight shape {rows}x{cols} is not divisible by 128, cannot use trtllm-fp8-blockwise"
                )
                linear_type = "default"
        if linear_type == "te-fp8-per-tensor" and hasattr(self, "weight"):
            rows, cols = self.weight.shape[-2:]
            if cols % 16 != 0:
                logger.debug(
                    f"layer {self.name} weight shape {rows}x{cols} is not divisible by 16, cannot use te-fp8-per-tensor"
                )
                linear_type = "default"
        return linear_type

    @torch.compiler.disable()
    def select_linear_impl(self, input: torch.Tensor = None):
        linear_type = self.get_linear_type()

        if linear_type == "auto":
            auto_tuner = get_auto_tuner()
            assert auto_tuner is not None, "AutoTuner is not initialized"
            if auto_tuner.mode == "inference":
                layer_name = self.name
                assert layer_name is not None, f"layer_name is not set for {type(self)}"
                mse_threshold = LinearOpManager.mse_threshold
                cosine_similarity_threshold = LinearOpManager.cosine_similarity_threshold
                # need inputs info to select the best impl
                assert input is not None, "input is not set for autotuning"
                linear_inputs = {
                    "input": input,
                    "weight": self.weight,
                    "bias": self.bias,
                    "input_scale": self.input_scale,
                    "weight_scale": None,
                }
                linear_type = auto_tuner.select_best_impl(
                    step=PipelineConfig.current_denoising_step,
                    layer_name=layer_name,
                    op_type=LinearOpManager.op_type(),
                    inputs=linear_inputs,
                    mse_threshold=mse_threshold,
                    cosine_similarity_threshold=cosine_similarity_threshold,
                )
            else:
                assert (
                    auto_tuner.mode == "tuning"
                ), f"Got unexpected auto tuner mode: {auto_tuner.mode}, choices: inference, tuning"
                for linear_type in LinearOpManager.get_registered_types():
                    # prepare fp8 weights and scale for autotuning
                    if linear_type == "default":
                        continue
                    if linear_type == "trtllm-fp8-blockwise":
                        rows, cols = self.weight.shape[-2:]
                        if rows % 128 != 0 or cols % 128 != 0:
                            logger.debug(
                                f"layer {self.name} weight shape {rows}x{cols} is not divisible by 128, cannot use trtllm-fp8-blockwise"
                            )
                            self.register_parameter(linear_type + "_weight", None)
                            self.register_buffer(linear_type + "_weight_scale", None)
                            continue
                    if linear_type == "trtllm-nvfp4" or linear_type == "comfy-kitchen-nvfp4":
                        self._register_nvfp4_weight(linear_type)
                    elif linear_type == "flashinfer-nvfp4-trtllm":
                        self._register_fi_nvfp4_weight(linear_type, sf_layout=SfLayout.layout_128x4, do_shuffle=True)
                    elif linear_type in ["flashinfer-nvfp4-cudnn", "flashinfer-nvfp4-cutlass"]:
                        self._register_fi_nvfp4_weight(linear_type, sf_layout=SfLayout.layout_128x4, do_shuffle=False)
                    else:
                        self._register_fp8_weight(linear_type)
                # No impl is selected for now, we need to tune firstly
                return self.weight, None, None, "auto"

        # select linear implementation
        if self.linear_impl is None:
            self.linear_impl = LinearOpManager.get_impl(linear_type)
        weight_scale = None
        weight_scale_2 = None
        if linear_type == "default":
            weight = self.weight
        elif linear_type == "svd-nvfp4":
            weight = self.svd_qweight
        elif linear_type == "trtllm-fp8-blockwise" or linear_type == "trtllm-fp8-per-tensor" or linear_type == "deepgemm-MXFP8":
            self._register_fp8_weight(linear_type)
            weight_name = linear_type + "_weight"
            weight_scale_name = linear_type + "_weight_scale"
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            assert weight is not None, f"weight of {linear_type} is not set"
            assert weight_scale is not None, f"weight_scale of {linear_type} is not set"
        elif linear_type == "te-fp8-per-tensor" or linear_type == "te-fp8-blockwise":
            self._register_fp8_weight(linear_type)
            weight_name = linear_type + "_weight"
            weight_scale_name = linear_type + "_weight_scale"
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            assert weight is not None, f"weight of {linear_type} is not set"
            assert weight_scale is not None, f"weight_scale of {linear_type} is not set"
        elif linear_type == "trtllm-nvfp4" or linear_type == "comfy-kitchen-nvfp4":
            self._register_nvfp4_weight(linear_type)
            weight_name = linear_type + "_weight"
            weight_scale_name = linear_type + "_weight_scale"
            weight_scale_name_2 = linear_type + "_weight_scale_2"
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            weight_scale_2 = getattr(self, weight_scale_name_2)
        elif linear_type == "torch-ao-fp8":
            self._register_fp8_weight(linear_type)
            weight = getattr(self, linear_type + "_weight")
        elif linear_type == "te-MXFP8-blockwise-32":
            self._register_mxfp8_weight(linear_type)
            weight = getattr(self, linear_type + "_weight")
        elif linear_type == "flashinfer-nvfp4-trtllm":
            self._register_fi_nvfp4_weight(linear_type, sf_layout=SfLayout.layout_128x4, do_shuffle=True)
            weight_name = linear_type + "_weight"
            weight_scale_name = linear_type + "_weight_scale"
            weight_scale_2_name = linear_type + "_weight_scale_2"
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            weight_scale_2 = getattr(self, weight_scale_2_name)
        elif linear_type in ["flashinfer-nvfp4-cudnn", "flashinfer-nvfp4-cutlass"]:
            self._register_fi_nvfp4_weight(linear_type, sf_layout=SfLayout.layout_128x4, do_shuffle=False)
            weight_name = linear_type + "_weight"
            weight_scale_name = linear_type + "_weight_scale"
            weight_scale_2_name = linear_type + "_weight_scale_2"
            weight = getattr(self, weight_name)
            weight_scale = getattr(self, weight_scale_name)
            weight_scale_2 = getattr(self, weight_scale_2_name)

        # In auto mode, we don't delete unused weights, because same layer may have different impls in different steps
        if LinearOpManager.linear_type != "auto":
            self._delete_unused_weight(linear_type)

        logger.debug(f"Selected linear implementation: {type(self.linear_impl).__name__}")
        return weight, weight_scale, weight_scale_2, linear_type

    # @torch.cuda.nvtx.range("ditLinear.forward")
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight, weight_scale, weight_scale_2, linear_type = self.select_linear_impl(input)
        svd_kwargs = {}
        if hasattr(self, "svd_qweight"):
            svd_kwargs["svd_qweight"] = self.svd_qweight
            svd_kwargs["svd_lora_up"] = self.svd_lora_up
            svd_kwargs["svd_lora_down"] = self.svd_lora_down
            svd_kwargs["svd_smooth"] = self.svd_smooth
            svd_kwargs["svd_wscales"] = self.svd_wscales
            svd_kwargs["svd_wtscale"] = self.svd_wtscale
            svd_kwargs["svd_wcscales"] = self.svd_wcscales
            svd_kwargs["svd_bias"] = self.svd_bias

        auto_tuner = get_auto_tuner()
        if linear_type == "auto" and auto_tuner.mode == "tuning":
            assert auto_tuner is not None, "AutoTuner is not initialized"
            # record input and output tensors for autotuning
            layer_name = self.name
            assert layer_name is not None, f"layer_name is not set for {type(self)}"
            step = PipelineConfig.current_denoising_step
            assert step is not None, "current_denoising_step of the pipeline is not set"
            linear_inputs = {
                "input": input,
                "weight": weight,
                "bias": self.bias,
                "input_scale": self.input_scale,
                "weight_scale": weight_scale,
                "special_inputs": {
                    "trtllm-fp8-blockwise": {
                        "weight": getattr(self, "trtllm-fp8-blockwise_weight"),
                        "weight_scale": getattr(self, "trtllm-fp8-blockwise_weight_scale"),
                    },
                    "trtllm-fp8-per-tensor": {
                        "weight": getattr(self, "trtllm-fp8-per-tensor_weight"),
                        "weight_scale": getattr(self, "trtllm-fp8-per-tensor_weight_scale"),
                    },
                    "deepgemm-MXFP8": {
                        "weight": getattr(self, "deepgemm-MXFP8_weight"),
                        "weight_scale": getattr(self, "deepgemm-MXFP8_weight_scale"),
                    },
                    "te-fp8-blockwise": {
                        "weight": getattr(self, "te-fp8-blockwise_weight"),
                        "weight_scale": getattr(self, "te-fp8-blockwise_weight_scale"),
                    },
                    "te-MXFP8-blockwise-32": {
                        "weight": getattr(self, "te-MXFP8-blockwise-32_weight"),
                    },
                    "te-fp8-per-tensor": {
                        "weight": getattr(self, "te-fp8-per-tensor_weight"),
                        "weight_scale": getattr(self, "te-fp8-per-tensor_weight_scale"),
                    },
                    "svd-nvfp4": svd_kwargs,
                    "trtllm-nvfp4": {
                        "weight": getattr(self, "trtllm-nvfp4_weight"),
                        "weight_scale": getattr(self, "trtllm-nvfp4_weight_scale"),
                        "weight_scale_2": getattr(self, "trtllm-nvfp4_weight_scale_2"),
                    },
                    "comfy-kitchen-nvfp4": {
                        "weight": getattr(self, "comfy-kitchen-nvfp4_weight", None),
                        "weight_scale": getattr(self, "comfy-kitchen-nvfp4_weight_scale", None),
                        "weight_scale_2": getattr(self, "comfy-kitchen-nvfp4_weight_scale_2", None),
                    },
                    "torch-ao-fp8": {
                        "weight": getattr(self, "torch-ao-fp8_weight"),
                    },
                    "flashinfer-nvfp4-trtllm": {
                        "weight": getattr(self, "flashinfer-nvfp4-trtllm_weight"),
                        "weight_scale": getattr(self, "flashinfer-nvfp4-trtllm_weight_scale"),
                        "weight_scale_2": getattr(self, "flashinfer-nvfp4-trtllm_weight_scale_2"),
                    },
                    "flashinfer-nvfp4-cudnn": {
                        "weight": getattr(self, "flashinfer-nvfp4-cudnn_weight"),
                        "weight_scale": getattr(self, "flashinfer-nvfp4-cudnn_weight_scale"),
                        "weight_scale_2": getattr(self, "flashinfer-nvfp4-cudnn_weight_scale_2"),
                    },
                    "flashinfer-nvfp4-cutlass": {
                        "weight": getattr(self, "flashinfer-nvfp4-cutlass_weight"),
                        "weight_scale": getattr(self, "flashinfer-nvfp4-cutlass_weight_scale"),
                        "weight_scale_2": getattr(self, "flashinfer-nvfp4-cutlass_weight_scale_2"),
                    },
                },
            }
            outputs = auto_tuner.tune(
                layer_name=layer_name,
                op_manager=LinearOpManager,
                baseline_impl="default",
                inputs=linear_inputs,
                step=step,
            )
            return outputs

        # kwargs is not empty for both svd-nvfp4 and auto,
        # in auto mode, if the selected impl is not svd-nvfp4, we should not pass the kwargs
        kwargs = {}
        if linear_type == "svd-nvfp4":
            kwargs = svd_kwargs
        elif linear_type == "trtllm-nvfp4" or linear_type == "comfy-kitchen-nvfp4":
            kwargs = {
                "weight_scale_2": weight_scale_2,
                "scaling_vector_size": 16,
            }
        elif linear_type == "te-fp8-per-tensor":
            kwargs = {
                "hasgelu": self.hasgelu,
            }
        elif linear_type in ["flashinfer-nvfp4-trtllm", "flashinfer-nvfp4-cudnn", "flashinfer-nvfp4-cutlass"]:
            kwargs = {
                "weight_scale_2": weight_scale_2,
            }

        if self.offloading:
            disable_offloading = os.environ.get("DISABLE_WEIGHT_MANAGEMENT", "False") == "True"
            if disable_offloading:
                weight_clc = self.host_weight.to(torch.cuda.current_device())
            else:
                self.pingpong_out = 0 if self.pingpong_in == 1 else 1
                # make sure the weight of this layer is copied to device
                self.offloading_event.wait(torch.cuda.current_stream())
                weight_clc = self.device_weight[self.pingpong_in]
                self.main_event.record(torch.cuda.current_stream())
                with torch.cuda.stream(stream=self.offloading_stream):
                    # make sure the buffer to copy next layer's weight is ready
                    self.main_event.wait(self.offloading_stream)
                    if self.next_offloading_layer_weight is None:
                        raise RuntimeError(f"next_offloading_layer_weight of {self.name} is None")
                    self.device_weight[self.pingpong_out].copy_(self.next_offloading_layer_weight, non_blocking=True)
                    self.offloading_event.record(self.offloading_stream)
            return self.linear_impl(
                input, weight_clc, self.bias, input_scale=self.input_scale, weight_scale=weight_scale, **kwargs
            )
        else:
            return self.linear_impl(
                input, weight, self.bias, input_scale=self.input_scale, weight_scale=weight_scale, **kwargs
            )

    @classmethod
    def from_linear(cls, linear: nn.Linear, load_parameters: bool = False) -> "ditLinear":
        device = linear.weight.device
        dtype = linear.weight.dtype
        bias = linear.bias is not None
        visual_gen_linear = cls(
            in_features=linear.in_features, out_features=linear.out_features, bias=bias, device=device, dtype=dtype
        )
        if load_parameters:
            visual_gen_linear.weight.data = linear.weight.data
            if bias:
                visual_gen_linear.bias.data = linear.bias.data
        return visual_gen_linear 