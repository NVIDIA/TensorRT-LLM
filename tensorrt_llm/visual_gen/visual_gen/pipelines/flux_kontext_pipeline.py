# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py
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

import torch
from diffusers import FluxKontextPipeline
from safetensors.torch import load_file

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.op_manager import LinearOpManager
from visual_gen.configs.parallel import VAEParallelConfig
from visual_gen.layers.linear import ditLinear
from visual_gen.models.transformers.flux_transformer import ditFluxTransformer2DModel
from visual_gen.pipelines.base_pipeline import ditBasePipeline
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


class ditFluxKontextPipeline(FluxKontextPipeline, ditBasePipeline):
    """visual_gen Pipeline for Flux Kontext (reference image-based generation and editing)."""

    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        """Post-processing hook after load model checkpoints in 'from_pretrained' method."""
        if not isinstance(self.transformer, ditFluxTransformer2DModel):
            logger.debug("Loading ditFluxTransformer2DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditFluxTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.debug("Using ditFluxTransformer2DModel from kwargs for transformer")

        self._fuse_qkv(self.transformer)

        linear_type = LinearOpManager.linear_type
        if linear_type == "te-fp8-per-tensor":
            self._fuse_gemm_gelu(self.transformer)

        self.transformer.teacache_coefficients = [
            2.57151496e05,
            -3.54229917e04,
            1.40286849e03,
            -1.35890334e01,
            1.32517977e-01,
        ]

        if not VAEParallelConfig.disable_parallel_vae:
            logger.warning("VAE parallel is not supported for FluxKontextPipeline")

    def load_fp4_weights(self, path, svd_weight_name_table):
        """Load FP4 quantized weights."""
        weights_table = load_file(path)
        for name, module in self.transformer.named_modules():
            if isinstance(module, ditLinear):
                module.load_fp4_weight(weights_table, svd_weight_name_table)

    def run_vae_decode(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """VAE decode with latent unpacking and scaling.

        This method can be wrapped with CUDA Graph for better performance.
        """
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, "shift_factor"):
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        return image

    def __call__(self, *args, num_inference_steps: int = 28, **kwargs):
        """Override __call__ to initialize TeaCache before inference."""
        if TeaCacheConfig.enable_teacache():
            step_size = 1
            if TeaCacheConfig.use_ret_steps():
                TeaCacheConfig.set_config(
                    cutoff_steps=num_inference_steps * step_size,
                    num_steps=num_inference_steps * step_size,
                    cnt=0,
                )
            else:
                TeaCacheConfig.set_config(
                    cutoff_steps=num_inference_steps * step_size - 2,
                    num_steps=num_inference_steps * step_size,
                    cnt=0,
                )
            logger.debug(f"TeaCache initialized: num_steps={num_inference_steps * step_size}")

        return super().__call__(*args, num_inference_steps=num_inference_steps, **kwargs)
