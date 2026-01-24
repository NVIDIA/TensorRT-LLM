# Adapted from: https://github.com/huggingface/diffusers/blob/9d313fc718c8ace9a35f07dad9d5ce8018f8d216/src/diffusers/pipelines/wan/pipeline_wan.py
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import torch
import torch.cuda.nvtx as nvtx
from diffusers import WanImageToVideoPipeline, WanPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.wan.pipeline_wan import WanPipelineOutput
from diffusers.utils import is_torch_xla_available

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.op_manager import (
    AttentionOpManager,
    SparseVideogenConfig,
    SparseVideogenConfig2,
)
from visual_gen.configs.parallel import DiTParallelConfig, VAEParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.models.transformers.wan_transformer import ditWanTransformer3DModel
from visual_gen.models.vaes.wan_vae import ditWanAutoencoderKL
from visual_gen.utils.logger import get_logger, log_execution_time

from .base_pipeline import ditBasePipeline

logger = get_logger(__name__)
XLA_AVAILABLE = is_torch_xla_available()
if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


class ditWanPipeline(WanPipeline, ditBasePipeline):
    """Implementation of WanPipeline with additional functionality.

    This class inherits from both WanPipeline and ditBasePipeline, using
    WanPipeline's methods for core functionality while using ditBasePipeline's
    from_pretrained, dit_dp_split and dit_dp_gather methods.
    """

    def _set_teacache_coefficients(self, pretrained_model_name_or_path, **kwargs) -> None:
        logger.debug("Setting up TeaCache configuration")
        if TeaCacheConfig.use_ret_steps():
            logger.debug("Using ret_steps mode for TeaCache")
            if "1.3B" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    -5.21862437e04,
                    9.23041404e03,
                    -5.28275948e02,
                    1.36987616e01,
                    -4.99875664e-02,
                ]
                logger.debug("Set TeaCache coefficients for 1.3B model")
            elif "14B" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    -3.03318725e05,
                    4.90537029e04,
                    -2.65530556e03,
                    5.87365115e01,
                    -3.15583525e-01,
                ]
                logger.debug("Set TeaCache coefficients for 14B model")
            else:
                error_msg = f"Model {pretrained_model_name_or_path} not supported for TeaCache"
                logger.error(error_msg)
                raise ValueError(error_msg)
            TeaCacheConfig.set_config(ret_steps=5 * 2)
        else:
            logger.debug("Using standard mode for TeaCache")
            if "1.3B" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    2.39676752e03,
                    -1.31110545e03,
                    2.01331979e02,
                    -8.29855975e00,
                    1.37887774e-01,
                ]
                logger.debug("Set TeaCache coefficients for 1.3B model")
            elif "14B" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    -5784.54975374,
                    5449.50911966,
                    -1811.16591783,
                    256.27178429,
                    -13.02252404,
                ]
                logger.debug("Set TeaCache coefficients for 14B model")
            else:
                error_msg = f"Model {pretrained_model_name_or_path} not supported for TeaCache"
                logger.error(error_msg)
                raise ValueError(error_msg)
            TeaCacheConfig.set_config(ret_steps=1 * 2)

    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        logger.debug("Executing ditWanPipeline post-load hook")
        self._set_teacache_coefficients(pretrained_model_name_or_path, **kwargs)
        logger.debug(f"TeaCache config: {TeaCacheConfig.get_instance().to_dict()}")

        # setup pipeline config
        pipe_config = kwargs.get("pipeline", {})
        pipe_config["transformer_type"] = "ditWanTransformer3DModel"
        PipelineConfig.set_config(**pipe_config)

        if not isinstance(self.transformer, ditWanTransformer3DModel):
            logger.debug("Loading ditWanTransformer3DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditWanTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.debug("Using ditWanTransformer3DModel from kwargs for transformer")

        # wan 2.2 has transformer_2, but wan 2.1 does not
        if self.transformer_2 is not None and not isinstance(
            self.transformer_2, ditWanTransformer3DModel
        ):
            logger.debug("Loading ditWanTransformer3DModel from diffusers transformer_2")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer_2 = ditWanTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer_2",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.debug("Using ditWanTransformer3DModel from kwargs for transformer_2")

        self._fuse_qkv(self.transformer)
        if self.transformer_2 is not None:
            self._fuse_qkv(self.transformer_2)

        if not VAEParallelConfig.disable_parallel_vae:
            if not isinstance(self.vae, ditWanAutoencoderKL):
                logger.debug("Loading ditWanAutoencoderKL from diffusers vae")
                self.vae = ditWanAutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
                )
                gc.collect()
                torch.cuda.empty_cache()
            else:
                logger.debug("Using ditWanAutoencoderKL from kwargs")
            self.vae.parallel_vae(split_dim=VAEParallelConfig.parallel_vae_split_dim)

        logger.debug("Post-load hook completed")

    def _set_sparse_videogen_attrs(
        self,
        context_length,
        height,
        width,
        num_frames,
        cfg_size,
        num_head,
        head_dim,
        dtype=torch.bfloat16,
    ):
        masks = ["spatial", "temporal"]
        if PipelineConfig.transformer_type == "ditWanTransformer3DModel":
            from svg.models.wan.attention import prepare_flexattention
            from svg.models.wan.utils import get_attention_mask, sparsity_to_width
        else:
            raise NotImplementedError(
                f"Sparse VideoGen is not implemented for {PipelineConfig.transformer_type}"
            )

        sample_mse_max_row = SparseVideogenConfig.sample_mse_max_row()
        sparsity = SparseVideogenConfig.sparsity()

        num_frame = 1 + num_frames // (
            self.vae_scale_factor_temporal * self.transformer.config.patch_size[0]
        )
        mod_value = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        frame_size = int(height // mod_value) * int(width // mod_value)
        attention_masks = [
            get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size)
            for mask_name in masks
        ]
        multiplier = diag_width = sparsity_to_width(sparsity, context_length, num_frame, frame_size)

        block_mask = prepare_flexattention(
            cfg_size,
            num_head // DiTParallelConfig.ulysses_size(),
            head_dim,
            dtype,
            "cuda",
            context_length,
            context_length,
            num_frame,
            frame_size,
            diag_width,
            multiplier,
        )
        SparseVideogenConfig.update(
            attention_masks=attention_masks,
            context_length=context_length,
            num_frame=num_frame,
            frame_size=frame_size,
            block_mask=block_mask,
        )

    def _set_sparse_videogen2_attrs(
        self,
        context_length,
        height,
        width,
        num_frames,
        cfg_size,
        num_head,
        head_dim,
        dtype=torch.bfloat16,
    ):
        num_frame = 1 + num_frames // (
            self.vae_scale_factor_temporal * self.transformer.config.patch_size[0]
        )
        mod_value = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        frame_size = int(height // mod_value) * int(width // mod_value)
        SparseVideogenConfig2.update(
            context_length=context_length,
            num_frame=num_frame,
            frame_size=frame_size,
        )

    @torch.no_grad()
    @log_execution_time("visual_gen.wan_pipeline")
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        num_frames = max(num_frames, 1)

        num_attention_heads = self.transformer.config.num_attention_heads
        if num_attention_heads % DiTParallelConfig.ulysses_size() != 0:
            logger.error(
                f"`num_attention_heads` ({num_attention_heads}) has to be divisible by ulysses size ({DiTParallelConfig.ulysses_size()})."
            )
            raise ValueError(
                f"`num_attention_heads` ({num_attention_heads}) has to be divisible by ulysses size ({DiTParallelConfig.ulysses_size()})."
            )

        if AttentionOpManager.attn_type == "sparse-videogen":
            self._set_sparse_videogen_attrs(
                context_length=0,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_size=1,
                num_head=num_attention_heads,
                head_dim=self.transformer.config.attention_head_dim,
            )

        if AttentionOpManager.attn_type == "sparse-videogen2":
            self._set_sparse_videogen2_attrs(
                context_length=0,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_size=1,
                num_head=num_attention_heads,
                head_dim=self.transformer.config.attention_head_dim,
            )

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Leverage ditBasePipeline's dit_dp_split method, which splits the data for data parallel
        batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds = (
            self.dit_dp_split(
                batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds
            )
        )

        # 3. Encode input prompt
        logger.debug("Encoding prompts")
        if "text_encoder" in PipelineConfig.model_wise_offloading:
            # only copy part of params to bypass the _execution_device issue
            self.text_encoder.to(torch.cuda.current_device())
        nvtx.range_push("encode_prompt")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        nvtx.range_pop()
        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        nvtx.range_push("prepare_latents")
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )
        nvtx.range_pop()

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = (
                self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        else:
            boundary_timestep = None

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to(torch.cuda.current_device())
            self.transformer_2.to(torch.cuda.current_device())
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                logger.debug(f"Denoising step {i}:{t} of {num_inference_steps}")
                if self.interrupt:
                    continue

                self._current_timestep = t
                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                # Leverage ditBasePipeline's visual_gen_transformer method, which handles cfg parallel, etc.
                nvtx.range_push("visual_gen_transformer")
                cfg_positive_inputs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_image": None,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                cfg_negative_inputs = None
                if self.do_classifier_free_guidance:
                    cfg_negative_inputs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_hidden_states_image": None,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                    }
                noise_cond, noise_uncond = self.visual_gen_transformer(
                    current_model,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    cfg_positive_inputs=cfg_positive_inputs,
                    cfg_negative_inputs=cfg_negative_inputs,
                )
                if self.do_classifier_free_guidance:
                    assert noise_uncond is not None, "noise_uncond is required"
                    noise_pred = noise_uncond + current_guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond
                nvtx.range_pop()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to("cpu")
            self.transformer_2.to("cpu")
            torch.cuda.empty_cache()

        self._current_timestep = None

        # Leverage ditBasePipeline's dp_gather method, which gathers the results from all data parallel processes
        latents = self.dit_dp_gather(latents)

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            nvtx.range_push("vae.decode")
            video = self.vae.decode(latents, return_dict=False)[0]
            nvtx.range_pop()
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


class ditWanImageToVideoPipeline(WanImageToVideoPipeline, ditBasePipeline):
    """Implementation of WanImageToVideoPipeline with additional functionality.

    This class inherits from both WanImageToVideoPipeline and ditBasePipeline, using
    WanImageToVideoPipeline's methods for core functionality while using ditBasePipeline's
    from_pretrained, dit_dp_split and dit_dp_gather methods.
    """

    def _set_teacache_coefficients(self, pretrained_model_name_or_path, **kwargs) -> None:
        if not TeaCacheConfig.enable_teacache():
            return

        logger.debug("Setting up TeaCache configuration")

        if TeaCacheConfig.use_ret_steps():
            logger.debug("Using ret_steps mode for TeaCache")
            if "480P" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    2.57151496e05,
                    -3.54229917e04,
                    1.40286849e03,
                    -1.35890334e01,
                    1.32517977e-01,
                ]
                logger.debug("Set TeaCache coefficients for Wan2.1 I2V 480P model")
            elif "720P" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    8.10705460e03,
                    2.13393892e03,
                    -3.72934672e02,
                    1.66203073e01,
                    -4.17769401e-02,
                ]
                logger.debug("Set TeaCache coefficients for Wan2.1 I2V 720P model")
            else:
                error_msg = f"Model {pretrained_model_name_or_path} not supported for TeaCache"
                logger.error(error_msg)
                raise ValueError(error_msg)
            TeaCacheConfig.set_config(ret_steps=5 * 2)
        else:
            logger.debug("Using standard mode for TeaCache")
            if "480P" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    -3.02331670e02,
                    2.23948934e02,
                    -5.25463970e01,
                    5.87348440e00,
                    -2.01973289e-01,
                ]
                logger.debug("Set TeaCache coefficients for Wan2.1 I2V 480P model")
            elif "720P" in pretrained_model_name_or_path:
                self.transformer.teacache_coefficients = [
                    -114.36346466,
                    65.26524496,
                    -18.82220707,
                    4.91518089,
                    -0.23412683,
                ]
                logger.debug("Set TeaCache coefficients for Wan2.1 I2V 720P model")
            else:
                error_msg = f"Model {pretrained_model_name_or_path} not supported for TeaCache"
                logger.error(error_msg)
                raise ValueError(error_msg)
            TeaCacheConfig.set_config(ret_steps=1 * 2)

    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        logger.debug("Executing ditWanPipeline post-load hook")

        self._set_teacache_coefficients(pretrained_model_name_or_path, **kwargs)
        logger.debug(f"TeaCache config: {TeaCacheConfig.get_instance().to_dict()}")

        # setup pipeline config
        pipe_config = kwargs.get("pipeline", {})
        pipe_config["transformer_type"] = "ditWanTransformer3DModel"
        PipelineConfig.set_config(**pipe_config)

        if not isinstance(self.transformer, ditWanTransformer3DModel):
            logger.debug("Loading ditWanTransformer3DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditWanTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.debug("Using ditWanTransformer3DModel from kwargs")

        if self.transformer_2 is not None and not isinstance(
            self.transformer_2, ditWanTransformer3DModel
        ):
            logger.debug("Loading ditWanTransformer3DModel from diffusers transformer_2")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer_2 = ditWanTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer_2",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.debug("Using ditWanTransformer3DModel from kwargs")

        if not VAEParallelConfig.disable_parallel_vae:
            if not isinstance(self.vae, ditWanAutoencoderKL):
                logger.debug("Loading ditWanAutoencoderKL from diffusers vae")
                self.vae = ditWanAutoencoderKL.from_pretrained(
                    pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
                )
                gc.collect()
                torch.cuda.empty_cache()
            else:
                logger.debug("Using ditWanAutoencoderKL from kwargs")
            self.vae.parallel_vae(split_dim=VAEParallelConfig.parallel_vae_split_dim)

    def _set_sparse_videogen_attrs(
        self,
        context_length,
        height,
        width,
        num_frames,
        cfg_size,
        num_head,
        head_dim,
        dtype=torch.bfloat16,
    ):
        masks = ["spatial", "temporal"]
        if PipelineConfig.transformer_type == "ditWanTransformer3DModel":
            from svg.models.wan.attention import prepare_flexattention
            from svg.models.wan.utils import get_attention_mask, sparsity_to_width
        else:
            raise NotImplementedError(
                f"Sparse VideoGen is not implemented for {PipelineConfig.transformer_type}"
            )

        sample_mse_max_row = SparseVideogenConfig.sample_mse_max_row()
        sparsity = SparseVideogenConfig.sparsity()

        num_frame = 1 + num_frames // (
            self.vae_scale_factor_temporal * self.transformer.config.patch_size[0]
        )
        mod_value = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        frame_size = int(height // mod_value) * int(width // mod_value)
        attention_masks = [
            get_attention_mask(mask_name, sample_mse_max_row, context_length, num_frame, frame_size)
            for mask_name in masks
        ]
        multiplier = diag_width = sparsity_to_width(sparsity, context_length, num_frame, frame_size)

        block_mask = prepare_flexattention(
            cfg_size,
            num_head // DiTParallelConfig.ulysses_size(),
            head_dim,
            dtype,
            "cuda",
            context_length,
            context_length,
            num_frame,
            frame_size,
            diag_width,
            multiplier,
        )
        SparseVideogenConfig.update(
            attention_masks=attention_masks,
            context_length=context_length,
            num_frame=num_frame,
            frame_size=frame_size,
            block_mask=block_mask,
        )

    def _set_sparse_videogen2_attrs(
        self,
        context_length,
        height,
        width,
        num_frames,
        cfg_size,
        num_head,
        head_dim,
        dtype=torch.bfloat16,
    ):
        num_frame = 1 + num_frames // (
            self.vae_scale_factor_temporal * self.transformer.config.patch_size[0]
        )
        mod_value = self.vae_scale_factor_spatial * self.transformer.config.patch_size[1]
        frame_size = int(height // mod_value) * int(width // mod_value)

        SparseVideogenConfig2.update(
            context_length=context_length,
            num_frame=num_frame,
            frame_size=frame_size,
        )

    @torch.no_grad()
    @log_execution_time("visual_gen_wan_i2v_pipeline")
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        last_image: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""The call function to the pipeline for generation.

        Args:
            image (`PipelineImageInput`):
                The input image to condition the generation on. Must be an image, a list of images or a `torch.Tensor`.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, defaults to `480`):
                The height of the generated video.
            width (`int`, defaults to `832`):
                The width of the generated video.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `negative_prompt` input argument.
            image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings. Can be used to easily tweak image inputs (weighting). If not provided,
                image embeddings are generated from the `image` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        num_frames = max(num_frames, 1)

        num_attention_heads = self.transformer.config.num_attention_heads
        if num_attention_heads % DiTParallelConfig.ulysses_size() != 0:
            logger.error(
                f"`num_attention_heads` ({num_attention_heads}) has to be divisible by ulysses size ({DiTParallelConfig.ulysses_size()})."
            )
            raise ValueError(
                f"`num_attention_heads` ({num_attention_heads}) has to be divisible by ulysses size ({DiTParallelConfig.ulysses_size()})."
            )

        if AttentionOpManager.attn_type == "sparse-videogen":
            self._set_sparse_videogen_attrs(
                context_length=0,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_size=1,
                num_head=num_attention_heads,
                head_dim=self.transformer.config.attention_head_dim,
            )

        if AttentionOpManager.attn_type == "sparse-videogen2":
            self._set_sparse_videogen2_attrs(
                context_length=0,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_size=1,
                num_head=num_attention_heads,
                head_dim=self.transformer.config.attention_head_dim,
            )

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Leverage ditBasePipeline's dit_dp_split method, which splits the data for data parallel
        batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds = (
            self.dit_dp_split(
                batch_size, prompt, negative_prompt, prompt_embeds, negative_prompt_embeds
            )
        )

        # 3. Encode input prompt
        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to(torch.cuda.current_device())
        nvtx.range_push("encode_prompt")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        nvtx.range_pop()
        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if self.config.boundary_ratio is None and not self.config.expand_timesteps:
            if image_embeds is None:
                if "image_encoder" in PipelineConfig.model_wise_offloading:
                    self.image_encoder.to(torch.cuda.current_device())
                nvtx.range_push("encode_image")
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
                nvtx.range_pop()
                if "image_encoder" in PipelineConfig.model_wise_offloading:
                    self.image_encoder.to("cpu")
                    torch.cuda.empty_cache()

            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        nvtx.range_push("video_processor.preprocess")
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=torch.float32
        )
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )
        nvtx.range_pop()

        nvtx.range_push("prepare_latents")
        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )
        if self.config.expand_timesteps:
            latents, condition, first_frame_mask = latents_outputs
        else:
            latents, condition = latents_outputs
        nvtx.range_pop()

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = (
                self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        else:
            boundary_timestep = None

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to(torch.cuda.current_device())
            self.transformer_2.to(torch.cuda.current_device())

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                if self.config.expand_timesteps:
                    latent_model_input = (
                        1 - first_frame_mask
                    ) * condition + first_frame_mask * latents
                    latent_model_input = latent_model_input.to(transformer_dtype)

                    # seq_len: num_latent_frames * (latent_height // patch_size) * (latent_width // patch_size)
                    temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    latent_model_input = torch.cat([latents, condition], dim=1).to(
                        transformer_dtype
                    )
                    timestep = t.expand(latents.shape[0])

                # Leverage ditBasePipeline's visual_gen_transformer method, which handles cfg parallel, etc.
                nvtx.range_push("visual_gen_transformer")
                cfg_positive_inputs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "encoder_hidden_states_image": image_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                cfg_negative_inputs = None
                if self.do_classifier_free_guidance:
                    cfg_negative_inputs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "encoder_hidden_states_image": image_embeds,
                        "attention_kwargs": attention_kwargs,
                        "return_dict": False,
                    }
                noise_cond, noise_uncond = self.visual_gen_transformer(
                    current_model,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    cfg_positive_inputs=cfg_positive_inputs,
                    cfg_negative_inputs=cfg_negative_inputs,
                )
                if self.do_classifier_free_guidance:
                    assert noise_uncond is not None, "noise_uncond is required"
                    noise_pred = noise_uncond + current_guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = noise_cond
                nvtx.range_pop()

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to("cpu")
            self.transformer_2.to("cpu")
            torch.cuda.empty_cache()

        self._current_timestep = None

        # Leverage ditBasePipeline's dit_dp_gather method, which gathers the results from all data parallel processes
        latents = self.dit_dp_gather(latents)

        if self.config.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            nvtx.range_push("vae.decode")
            video = self.vae.decode(latents, return_dict=False)[0]
            nvtx.range_pop()
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
