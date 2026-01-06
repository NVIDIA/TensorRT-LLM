# Adapted from: https://github.com/huggingface/diffusers/blob/9d313fc718c8ace9a35f07dad9d5ce8018f8d216/src/diffusers/pipelines/flux/pipeline_flux.py
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
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput, calculate_shift, retrieve_timesteps
from diffusers.utils import is_torch_xla_available
from safetensors.torch import load_file

from visual_gen.configs.op_manager import LinearOpManager
from visual_gen.configs.parallel import VAEParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.linear import ditLinear
from visual_gen.models.transformers.flux_transformer import ditFluxTransformer2DModel
from visual_gen.pipelines.base_pipeline import ditBasePipeline
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class ditFluxPipeline(FluxPipeline, ditBasePipeline):

    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        """Post-processing hook after load model checkpoints in 'from_pretrained' method."""
        if not isinstance(self.transformer, ditFluxTransformer2DModel):
            logger.debug("Loading ditFluxTransformer2DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditFluxTransformer2DModel.from_pretrained(
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

        self.transformer.teacache_coefficients = [
            2.57151496e05,
            -3.54229917e04,
            1.40286849e03,
            -1.35890334e01,
            1.32517977e-01,
        ]

        if not VAEParallelConfig.disable_parallel_vae:
            logger.warning("VAE parallel is not supported for FluxPipeline")

    def load_fp4_weights(self, path, svd_weight_name_table):
        weights_table = load_file(path)
        for name, module in self.transformer.named_modules():
            if isinstance(module, ditLinear):
                module.load_fp4_weight(weights_table, svd_weight_name_table)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Leverage ditBasePipeline's dit_dp_split method, which splits the data for data parallel
        batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds = self.dit_dp_split(
            batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        device = self._execution_device

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        if "text_encoder" in PipelineConfig.model_wise_offloading:
            # only copy part of params to bypass the _execution_device issue
            if hasattr(self, "text_encoder"):
                self.text_encoder.to(torch.cuda.current_device())
            if hasattr(self, "text_encoder_2"):
                self.text_encoder_2.to(torch.cuda.current_device())

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        if "text_encoder" in PipelineConfig.model_wise_offloading:
            logger.warning(f"text_encoder.device: {self.text_encoder.device}")
            if hasattr(self, "text_encoder"):
                self.text_encoder.to("cpu")
            if hasattr(self, "text_encoder_2"):
                self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
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
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                cfg_positive_inputs = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "pooled_projections": pooled_prompt_embeds,
                    "encoder_hidden_states": prompt_embeds,
                    "txt_ids": text_ids,
                    "img_ids": latent_image_ids,
                    "joint_attention_kwargs": self.joint_attention_kwargs,
                    "return_dict": False,
                }

                cfg_negative_inputs = None
                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

                    cfg_negative_inputs = {
                        "hidden_states": latents,
                        "timestep": timestep / 1000,
                        "guidance": guidance,
                        "pooled_projections": negative_pooled_prompt_embeds,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "txt_ids": negative_text_ids,
                        "img_ids": latent_image_ids,
                        "joint_attention_kwargs": self.joint_attention_kwargs,
                        "return_dict": False,
                    }

                noise_cond, noise_uncond = self.visual_gen_transformer(
                    self.transformer,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=do_true_cfg,
                    cfg_positive_inputs=cfg_positive_inputs,
                    cfg_negative_inputs=cfg_negative_inputs,
                )
                if do_true_cfg:
                    assert noise_uncond is not None, "noise_uncond is required"
                    noise_pred = noise_uncond + true_cfg_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond

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

        # Leverage ditBasePipeline's dp_gather method, which gathers the results from all data parallel processes
        latents = self.dit_dp_gather(latents)

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
