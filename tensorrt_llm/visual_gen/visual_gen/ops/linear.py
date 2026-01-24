# Adapted from https://github.com/nunchaku-tech/deepcompressor
# @article{
#   li2024svdquant,
#   title={SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models},
#   author={Li*, Muyang and Lin*, Yujun and Zhang*, Zhekai and Cai, Tianle and Li, Xiuyu and Guo, Junxian and Xie, Enze and Meng, Chenlin and Zhu, Jun-Yan and Han, Song},
#   journal={arXiv preprint arXiv:2411.05007},
#   year={2024}
# }

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from visual_gen.configs.op_manager import LinearOpManager
from visual_gen.utils.auto_tuner import TunableParam
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import visual_gen.csrc._C
except ImportError:
    logger.warning("svdquant kernels are not compiles, svdquant only supports SM 120.")

try:
    import tensorrt_llm  # noqa: F401
except ImportError:
    tensorrt_llm = None
    logger.warning("TensorRT-LLM is not installed.")

try:
    import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
    from tensorrt_llm._torch.autotuner import autotune
except ImportError:
    tensorrt_llm = None
    logger.warning("TensorRT-LLM nvfp4 kernel dependency import failed.")

try:
    import transformer_engine  # noqa: F401
    import transformer_engine.pytorch.cpp_extensions as ext
    import transformer_engine_torch as tex
    from transformer_engine.pytorch import MXFP8Quantizer
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
        Float8BlockQuantizer,
        Float8BlockwiseQTensor,
    )
    from transformer_engine.pytorch.tensor.float8_tensor import (
        Float8CurrentScalingQuantizer,
        Float8Tensor,
    )
except ImportError as e:
    logger.warning(f"Transformer_engine is not installed: {e}")


try:
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize  # noqa: F401
except ImportError:
    logger.warning("Flashinfer is not installed")


try:
    from deep_gemm import fp8_gemm_nt

    from visual_gen.ops.deep_gemm_quant import quant_and_transform_ue8m0
except ImportError:
    logger.warning("Deep_gemm is not installed")


class BaseLinear:
    def __init__(self):
        pass

    def register_tunable_params(
        self,
        value: Any,
        param_range: Optional[List[Any]],
        name: Optional[str],
        description: Optional[str],
    ):
        return TunableParam(value, param_range, name, description)


@LinearOpManager.register_linear("default")
class DefaultLinear(BaseLinear):
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return F.linear(input, weight, bias)


@LinearOpManager.register_linear("trtllm-fp8-blockwise")
class TrtllmFp8BlockLinear(BaseLinear):
    # @torch.cuda.nvtx.range("TrtllmFp8BlockLinear.forward")
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        if tensorrt_llm is None:
            logger.error("TensorRT-LLM is not installed")

        # input
        origin_shape = input.shape
        origin_dtype = input.dtype
        input = input.to(torch.bfloat16)

        if input.dim() > 2:
            input = input.reshape(-1, input.shape[-1])

        act_input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128(input)
        output = torch.ops.trtllm.fp8_block_scaling_gemm(
            act_input_fp8, weight, input_scale, weight_scale
        )

        if bias is not None:
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        output = output.to(origin_dtype)

        return output


@LinearOpManager.register_linear("trtllm-fp8-per-tensor")
class TrtllmFp8PerTensorLinear(BaseLinear):
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        if tensorrt_llm is None:
            logger.error("TensorRT-LLM is not installed")

        origin_shape = input.shape
        origin_dtype = input.dtype
        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        # Dynamic quantization
        qinput, cur_input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(input)
        cur_input_scale = cur_input_scale.to(torch.float32)
        # This op does not support bias now.
        if qinput.dim() == 3:
            qinput = qinput.reshape(-1, qinput.shape[-1])

        output = torch.ops.trtllm.cublas_scaled_mm(
            qinput,
            weight,
            scale_a=cur_input_scale,
            scale_b=weight_scale,
            bias=None,
            out_dtype=input.dtype,
        )

        if bias is not None:
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        output = output.to(origin_dtype)

        return output


@LinearOpManager.register_linear("te-fp8-blockwise")
class TeFp8BlockLinear(BaseLinear):
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        # input
        origin_shape = input.shape
        origin_dtype = input.dtype
        input = input.to(torch.bfloat16)

        if bias is not None:
            if bias.dtype != torch.bfloat16:
                bias = bias.to(torch.bfloat16)

        if input.dim() > 2:
            input = input.reshape(-1, input.shape[-1])
        if input.shape[0] % 8 != 0:
            act_input_fp8, input_scale = torch.ops.trtllm.fp8_quantize_1x128(input)
            output = torch.ops.trtllm.fp8_block_scaling_gemm(
                act_input_fp8, weight, input_scale, weight_scale
            )
            if bias is not None:
                output = output + bias
        else:
            # Dynamic quantization
            x_quantizer_cur = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                amax_epsilon=0.0,
                force_pow_2_scales=False,
                block_scaling_dim=1,
            )
            w_quantizer_cur = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                amax_epsilon=0.0,
                force_pow_2_scales=False,
                block_scaling_dim=2,
            )
            x_fp8_te = x_quantizer_cur.quantize(input)
            w_fp8_te = Float8BlockwiseQTensor(
                shape=weight.shape,
                dtype=torch.float32,
                rowwise_data=weight.contiguous().view(torch.uint8),
                rowwise_scale_inv=weight_scale.to(dtype=torch.float32),
                columnwise_data=None,
                columnwise_scale_inv=None,
                fp8_dtype=tex.DType.kFloat8E4M3,
                is_2D_scaled=True,
                quantizer=w_quantizer_cur,
            )
            output, *_ = ext.general_gemm(
                A=w_fp8_te, B=x_fp8_te, out_dtype=torch.bfloat16, bias=bias
            )

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        output = output.to(origin_dtype)

        return output


@LinearOpManager.register_linear("te-MXFP8-blockwise-32")
class TeMXFP8Blockwise32Linear(BaseLinear):
    @torch.compiler.disable()
    def run_te_gemm(self, input, weight, bias):
        input_quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)

        inp_fp8 = input_quantizer(input)
        # weight_fp8 = input_quantizer(weight)
        outp_type = torch.bfloat16

        output, *_ = ext.general_gemm(
            weight,
            inp_fp8,
            outp_type,
            quantization_params=None,
            bias=bias,
            use_split_accumulator=False,
        )
        return output

    def __call__(
        self,
        input: torch.Tensor,
        weight,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        # input
        origin_shape = input.shape
        origin_dtype = input.dtype
        input = input.to(torch.bfloat16)

        if bias is not None:
            if bias.dtype != torch.bfloat16:
                bias = bias.to(torch.bfloat16)

        if input.dim() > 2:
            input = input.reshape(-1, input.shape[-1])

        # Pad input if it is not divisible by 32
        original_batch_size = input.shape[0]
        if input.shape[0] % 32 != 0:
            pad_size = 32 - (input.shape[0] % 32)
            input = F.pad(input, (0, 0, 0, pad_size), mode="constant", value=0)

        output = self.run_te_gemm(input, weight, bias)

        # Trim padding if it was added
        if output.shape[0] != original_batch_size:
            output = output[:original_batch_size]

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        output = output.to(origin_dtype)

        return output


@LinearOpManager.register_linear("te-fp8-per-tensor")
class TeFp8PerTensorLinear(BaseLinear):
    def __init__(self):
        self.quantizer_cur = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, device="cuda"
        )

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        hasgelu: torch.Tensor,
    ) -> torch.Tensor:
        origin_shape = input.shape
        origin_dtype = input.dtype
        if origin_dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        if bias is not None:
            if bias.dtype != torch.bfloat16:
                bias = bias.to(torch.bfloat16)

        input_fp8_te = self.quantizer_cur.quantize(input)
        w_fp8_te = Float8Tensor(
            shape=weight.t().shape,
            dtype=torch.bfloat16,
            data=weight.t().contiguous().view(torch.uint8),
            fp8_dtype=tex.DType.kFloat8E4M3,
            fp8_scale_inv=weight_scale.flatten().to(dtype=torch.float32),
        )
        if hasgelu:
            gelu_in = torch.randn(
                (input.shape[0], weight.shape[1]), dtype=torch.bfloat16, device="cuda"
            )
            output, *_ = ext.general_gemm(
                A=w_fp8_te,
                B=input_fp8_te,
                out_dtype=torch.bfloat16,
                bias=bias,
                gelu=True,
                gelu_in=gelu_in,
            )
        else:
            output, *_ = ext.general_gemm(
                A=w_fp8_te, B=input_fp8_te, out_dtype=torch.bfloat16, bias=bias
            )

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)

        return output


@torch.compiler.disable()
@LinearOpManager.register_linear("svd-nvfp4")
class SvdNvfp4Linear(BaseLinear):
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        svd_qweight=None,
        svd_lora_up=None,
        svd_lora_down=None,
        svd_smooth=None,
        svd_wscales=None,
        svd_wtscale=None,
        svd_wcscales=None,
        svd_bias=None,
    ) -> torch.Tensor:
        B, M, K = input.shape
        N = svd_qweight.shape[0]
        R = svd_lora_up.shape[1]

        assert B * M % 256 == 0, "B * M must be divisible by 256"
        assert K % 128 == 0, "K must be divisible by 128"
        assert N % 128 == 0, "N must be divisible by 128"

        act = torch.empty(B * M, int(K / 2), dtype=torch.int8, device=input.device).contiguous()
        ascales = torch.empty(
            int(K / 16), B * M, dtype=torch.float8_e4m3fn, device=input.device
        ).contiguous()
        lora_act = torch.empty(B * M, R, dtype=torch.float32, device=input.device).contiguous()
        out = torch.empty(B, M, N, dtype=torch.bfloat16, device=input.device).contiguous()
        lora_scales = [1.0] * (R // 16)
        input = input.to(torch.bfloat16).contiguous()

        visual_gen.csrc._C.ops.quantize_w4a4_act_fuse_lora(
            input, act, ascales, svd_lora_down, lora_act, svd_smooth, False, True
        )

        visual_gen.csrc._C.ops.gemm_w4a4(
            act,  # packed act [M, K / 2]
            svd_qweight,  # packed act [N, K / 2]
            out,  # linear     [M, N]
            None,  # packed act [M, N / 2]
            ascales,  # packed as  [K / 16, M]
            svd_wscales,  # packed ws  [K / 16, N]
            None,  # packed as  [N / 16, M]
            None,  # linear     [M / PoolSize, N]
            lora_act,  # packed lora_act [M, R]
            svd_lora_up,  # packed lora_wgt [N, R]
            None,  # packed lora_wgt [N, R]
            None,  # packed lora_act [M, R]
            None,  # norm_q     [HEAD_DIM]
            None,  # norm_k     [HEAD_DIM]
            None,  # rotary_emb [M, HEAD_DIM / 2, 2, 2]
            svd_bias,  # packed ws  [N]
            None,  # smooth_factor  packed ws  [N], for quantization of the next layer
            None,  # out_vk   [B, num_heads, head_dim + 1, head_dim]
            None,  # out_linearattn   [B, (M), N / 3]
            False,  # act_unsigned
            lora_scales,
            False,  # fuse_silu
            True,  # fp4
            svd_wtscale,
            svd_wcscales,  # wcscales
            None,  # out_q packed attention [B, H, M, D]
            None,  # out_k packed attention [B, H, M, D]
            None,  # out_v packed attention [B, H, M, D]
            0,
        )

        return out


@LinearOpManager.register_linear("trtllm-nvfp4")
class TrtllmNVFP4Linear(BaseLinear):
    def __init__(self):
        self.input_scale_2 = torch.tensor([8.0], dtype=torch.float32).cuda()

        self.trtllm_tuned = False

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        scaling_vector_size: int = 16,
        input_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if tensorrt_llm is None:
            logger.error("TensorRT-LLM is not installed")

        origin_dtype = input.dtype
        if origin_dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        assert scaling_vector_size % 16 == 0

        if LinearOpManager.linear_recipe == "dynamic":
            input_scale_2 = 448.0 * 6.0 / torch.amax(torch.abs(input)).to(torch.float)
        else:
            input_scale_2 = self.input_scale_2

        alpha = 1 / (input_scale_2 * weight_scale_2)
        if input.dim() == 3:
            act_fp4, act_sf = torch.ops.trtllm.fp4_batched_quantize(
                input, input_scale_2, scaling_vector_size, False
            )
            if not self.trtllm_tuned:
                with torch.inference_mode(), autotune():
                    output = torch.ops.trtllm.fp4_bmm(
                        act_fp4,
                        weight.unsqueeze(0),
                        act_sf,
                        weight_scale.unsqueeze(0),
                        alpha,
                        fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                        out_dtype=origin_dtype,
                    )
                self.trtllm_tuned = True
            else:
                output = torch.ops.trtllm.fp4_bmm(
                    act_fp4,
                    weight.unsqueeze(0),
                    act_sf,
                    weight_scale.unsqueeze(0),
                    alpha,
                    fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4,
                    out_dtype=origin_dtype,
                )
        else:
            act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                input, input_scale_2, scaling_vector_size, False
            )
            output = torch.ops.trtllm.nvfp4_gemm(
                act_fp4, weight, act_sf, weight_scale, alpha, output_dtype=origin_dtype
            )

        if bias is not None:
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        return output


@LinearOpManager.register_linear("comfy-kitchen-nvfp4")
class ComfyKitchenNVFP4Linear(BaseLinear):
    def __init__(self):
        self.input_scale_2 = torch.tensor([8.0], dtype=torch.float32).cuda()

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        scaling_vector_size: int = 16,
        input_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        origin_dtype = input.dtype
        if origin_dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        assert scaling_vector_size % 16 == 0

        if LinearOpManager.linear_recipe == "dynamic":
            input_scale_2 = 448.0 * 6.0 / torch.amax(torch.abs(input)).to(torch.float)
        elif input_scale is not None:
            # using user provided input_scale
            input_scale_2 = input_scale
        else:
            input_scale_2 = self.input_scale_2

        from comfy_kitchen import scaled_mm_nvfp4

        if input.dim() == 3:
            # The batched gemm is not supported in comfy-kitchen yet.
            batch_size = input.shape[0]
            input = input.reshape(-1, input.shape[-1])
            # currently still reuse trtllm fp4 quantize kernel for better performance
            act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                input, input_scale_2, scaling_vector_size, False
            )

            output = scaled_mm_nvfp4(
                act_fp4,
                weight,
                1 / input_scale_2,
                1 / weight_scale_2,
                act_sf.reshape(-1, input.shape[1] // scaling_vector_size)
                .view(torch.float8_e4m3fn)
                .contiguous(),
                weight_scale.reshape(-1, weight.shape[1] * 2 // scaling_vector_size)
                .view(torch.float8_e4m3fn)
                .contiguous(),
                out_dtype=origin_dtype,
                bias=bias.to(origin_dtype) if bias is not None else None,
            )
            output = output.reshape(batch_size, -1, output.shape[-1])
        else:
            act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(
                input, input_scale_2, scaling_vector_size, False
            )
            output = scaled_mm_nvfp4(
                act_fp4,
                weight,
                1 / input_scale_2,
                1 / weight_scale_2,
                act_sf.reshape(-1, input.shape[1] // scaling_vector_size).view(torch.float8_e4m3fn),
                weight_scale.reshape(-1, weight.shape[1] * 2 // scaling_vector_size).view(
                    torch.float8_e4m3fn
                ),
                out_dtype=origin_dtype,
                bias=bias.to(origin_dtype) if bias is not None else None,
            )

        return output


@LinearOpManager.register_linear("torch-ao-fp8")
class TorchAOLinear(BaseLinear):
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        origin_dtype = input.dtype
        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)
            bias = bias.to(torch.bfloat16)
        output = F.linear(input, weight, bias)
        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)
        return output


class FlashInferNVFP4Linear:
    def __init__(self):
        self.input_scale_2 = torch.tensor(
            [8.0], dtype=torch.float32, device=torch.cuda.current_device()
        )

    def _nvfp4_gemm(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        scaling_vector_size: int = 16,
        input_scale: torch.Tensor = None,
        sfLayout: SfLayout = SfLayout.layout_128x4,
        do_shuffle: bool = False,
        backend: str = "trtllm",
    ) -> torch.Tensor:
        origin_dtype = input.dtype
        origin_shape = input.shape
        if input.dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)
            bias = bias.to(torch.bfloat16)
        if input.dim() != 2:
            input = input.reshape(-1, input.shape[-1]).contiguous()

        if LinearOpManager.linear_recipe == "dynamic":
            input_global_sf = (448 * 6) / input.float().abs().nan_to_num().max()
        elif input_scale is not None:
            # using user provided input_scale
            input_global_sf = input_scale
        else:
            # TODO: magic number for static quantization scale, it should be passed by a quantized ckpt
            input_global_sf = self.input_scale_2
        input_fp4, input_sf = nvfp4_quantize(
            input, input_global_sf, sfLayout=sfLayout, do_shuffle=do_shuffle
        )

        output = mm_fp4(
            input_fp4,
            weight,
            input_sf,
            weight_scale_2,
            1.0 / (input_global_sf * weight_scale),
            torch.bfloat16,
            None,
            backend=backend,
        )

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        if output.dtype != origin_dtype:
            output = output.to(origin_dtype)

        if bias is not None:
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        return output


@LinearOpManager.register_linear("flashinfer-nvfp4-trtllm")
class FlashInferNVFP4TRTLLMLinear(BaseLinear, FlashInferNVFP4Linear):
    def __init__(self):
        BaseLinear.__init__(self)
        FlashInferNVFP4Linear.__init__(self)

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        scaling_vector_size: int = 16,
        input_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        return self._nvfp4_gemm(
            input,
            weight,
            bias,
            weight_scale,
            weight_scale_2,
            scaling_vector_size,
            input_scale,
            SfLayout.layout_128x4,
            False,
            "trtllm",
        )


@LinearOpManager.register_linear("flashinfer-nvfp4-cudnn")
class FlashInferNVFP4CudnnLinear(BaseLinear, FlashInferNVFP4Linear):
    def __init__(self):
        BaseLinear.__init__(self)
        FlashInferNVFP4Linear.__init__(self)

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        scaling_vector_size: int = 16,
        input_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        return self._nvfp4_gemm(
            input,
            weight,
            bias,
            weight_scale,
            weight_scale_2,
            scaling_vector_size,
            input_scale,
            SfLayout.layout_128x4,
            False,
            "cudnn",
        )


@LinearOpManager.register_linear("flashinfer-nvfp4-cutlass")
class FlashInferNVFP4CutlassLinear(BaseLinear, FlashInferNVFP4Linear):
    def __init__(self):
        BaseLinear.__init__(self)
        FlashInferNVFP4Linear.__init__(self)

    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: torch.Tensor,
        scaling_vector_size: int = 16,
        input_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        return self._nvfp4_gemm(
            input,
            weight,
            bias,
            weight_scale,
            weight_scale_2,
            scaling_vector_size,
            input_scale,
            SfLayout.layout_128x4,
            False,
            "cutlass",
        )


@LinearOpManager.register_linear("deepgemm-MXFP8")
class DeepgemmFp8Linear(BaseLinear):
    def __call__(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        origin_dtype = input.dtype
        if origin_dtype != torch.bfloat16:
            input = input.to(torch.bfloat16)

        assert input.dim() == 3
        b, m, k = input.shape
        n = weight.shape[0]
        output = torch.empty((b * m, n), device=input.device, dtype=input.dtype).contiguous()

        input = input.reshape(b * m, k)
        input_fp8, input_scale = quant_and_transform_ue8m0(input)
        fp8_gemm_nt(
            (input_fp8, input_scale),
            (weight, weight_scale),
            output,
            None,  # bias
            disable_ue8m0_cast=False,
        )

        output = output.reshape(b, m, n)

        if bias is not None:
            if bias.dtype != output.dtype:
                bias = bias.to(output.dtype)
            output = output + bias

        return output
