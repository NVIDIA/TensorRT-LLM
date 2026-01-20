# Adapted from: https://github.com/huggingface/diffusers/blob/8af8e86bc7a2a2a038b6d5954793cdcc7b20f1e3/src/diffusers/pipelines/flux2/pipeline_flux2.py
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Adapted from: https://github.com/ali-vilab/TeaCache
# @article{
#   title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
#   author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
#   journal={arXiv preprint arXiv:2411.19108},
#   year={2024}
# }
#
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
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import PIL
from diffusers import Flux2Pipeline
from diffusers.pipelines.flux2.pipeline_flux2 import Flux2PipelineOutput, retrieve_timesteps, compute_empirical_mu
from diffusers.utils import is_torch_xla_available
from safetensors.torch import load_file

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.op_manager import LinearOpManager
from visual_gen.configs.parallel import VAEParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.linear import ditLinear
from visual_gen.models.transformers.flux2_transformer import ditFlux2Transformer2DModel
from visual_gen.pipelines.base_pipeline import ditBasePipeline
from visual_gen.utils.logger import get_logger
from visual_gen.utils import cudagraph_wrapper
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

logger = get_logger(__name__)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class ditFlux2Pipeline(Flux2Pipeline, ditBasePipeline):

    def _set_teacache_coefficients(self, **kwargs) -> None:
            teacache_configs = kwargs.pop("teacache", None)
            
            if teacache_configs is not None:
                logger.debug("Setting up TeaCache configuration")
                self.transformer.teacache_coefficients = [
                    1.04582360e+02,
                    -6.87605554e+00,
                    -8.61659379e-02,
                    5.37600252e-02
                ]
                TeaCacheConfig.set_config(
                    enable_teacache=teacache_configs.pop("enable_teacache", False),
                    teacache_thresh=teacache_configs.pop("teacache_thresh", 0.),
                    use_ret_steps=teacache_configs.pop("use_ret_steps", False),
                    ret_steps=teacache_configs.pop("ret_steps", 0),
                    cutoff_steps=teacache_configs.pop("cutoff_steps", 50),
                    cnt=0
                )

    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        """Post-processing hook after load model checkpoints in 'from_pretrained' method."""
        if not isinstance(self.transformer, ditFlux2Transformer2DModel):
            logger.debug("Loading ditFlux2Transformer2DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditFlux2Transformer2DModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.debug("Using ditWanTransformer3DModel from kwargs for transformer")

        self._fuse_qkv(self.transformer)

        linear_type = LinearOpManager.linear_type
        if linear_type == "te-fp8-per-tensor":
            self._fuse_gemm_gelu(self.transformer)

        if not VAEParallelConfig.disable_parallel_vae:
            logger.warning("VAE parallel is not supported for FluxPipeline")

        self._set_teacache_coefficients(**kwargs)
            

    def enable_cuda_graph(self):
        if TeaCacheConfig.enable_teacache():
            logger.info("capturing cuda graph for teacache..")
            self.transformer.run_pre_processing = cudagraph_wrapper(
                self.transformer.run_pre_processing
                )
            self.transformer.run_teacache_check = cudagraph_wrapper(
                self.transformer.run_teacache_check
            )
            self.transformer.run_transformer_blocks = cudagraph_wrapper(
                self.transformer.run_transformer_blocks
            )
            self.transformer.run_post_processing = cudagraph_wrapper(
                self.transformer.run_post_processing
            )
        else:
            logger.info("capturing cuda graph..")
            self.transformer.forward = cudagraph_wrapper(self.transformer.forward)


    def load_fp4_weights(self, path, svd_weight_name_table):
        weights_table = load_file(path)
        for name, module in self.transformer.named_modules():
            if isinstance(module, ditLinear):
                module.load_fp4_weight(weights_table, svd_weight_name_table)

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
        caption_upsample_temperature: float = None,
    ):
        #setup teacache num_steps
        if TeaCacheConfig.enable_teacache():
            TeaCacheConfig.set_config(
                    num_steps=num_inference_steps
            )
            if TeaCacheConfig.cutoff_steps() > TeaCacheConfig.num_steps():
                logger.warning("Number of cutoff_steps > num_steps.")

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]


        batch_size, prompt, _, prompt_embeds, _ = self.dit_dp_split(
            batch_size, prompt, None, prompt_embeds, None
        )

        device = self._execution_device

        # 3. prepare text embeddings
        if caption_upsample_temperature:
            prompt = self.upsample_prompt(
                prompt, images=image, temperature=caption_upsample_temperature, device=device
            )

        if "text_encoder" in PipelineConfig.model_wise_offloading:
            # only copy part of params to bypass the _execution_device issue
            if hasattr(self, "text_encoder"):
                self.text_encoder.to(torch.cuda.current_device())
            if hasattr(self, "text_encoder_2"):
                self.text_encoder_2.to(torch.cuda.current_device())
            
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if "text_encoder" in PipelineConfig.model_wise_offloading:
            logger.warning(f"text_encoder.device: {self.text_encoder.device}")
            if hasattr(self, "text_encoder"):
                self.text_encoder.to("cpu")
            if hasattr(self, "text_encoder_2"):
                self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = latents.to(self.transformer.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                inputs_dict = {
                    "hidden_states": latent_model_input,  # (B, image_seq_len, C)
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "encoder_hidden_states": prompt_embeds,
                    "txt_ids": text_ids,  # B, text_seq_len, 4
                    "img_ids": latent_image_ids,  # B, image_seq_len, 4
                    "joint_attention_kwargs": self._attention_kwargs,
                    "return_dict": False,
                }

                noise_pred, _ = self.visual_gen_transformer(
                    self.transformer,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=False,
                    cfg_positive_inputs=inputs_dict
                )

                noise_pred = noise_pred[:, : latents.size(1) :]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latents = self.dit_dp_gather(latents)

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents_with_ids(latents, latent_ids)

            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
                latents.device, latents.dtype
            )
            latents = latents * latents_bn_std + latents_bn_mean
            latents = self._unpatchify_latents(latents)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
