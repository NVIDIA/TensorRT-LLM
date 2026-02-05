# Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py
# Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.0/src/diffusers/pipelines/cosmos/pipeline_cosmos2_text2image.py
# Adapted from: https://github.com/huggingface/diffusers/blob/v0.35.0/src/diffusers/pipelines/cosmos/pipeline_cosmos_text2world.py
#
# Adapted from: https://github.com/ali-vilab/TeaCache
# @article{
#   title={Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model},
#   author={Liu, Feng and Zhang, Shiwei and Wang, Xiaofeng and Wei, Yujie and Qiu, Haonan and Zhao, Yuzhong and Zhang, Yingya and Ye, Qixiang and Wan, Fang},
#   journal={arXiv preprint arXiv:2411.19108},
#   year={2024}
# }
#
# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import torchvision.transforms.functional as F2
from diffusers import (
    Cosmos2TextToImagePipeline,
    Cosmos2VideoToWorldPipeline,
    Cosmos2_5_PredictBasePipeline,
    CosmosTextToWorldPipeline,
)
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.cosmos.pipeline_cosmos_video2world import CosmosPipelineOutput, retrieve_timesteps
from diffusers.pipelines.cosmos.pipeline_output import CosmosImagePipelineOutput

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.op_manager import AttentionOpManager
from visual_gen.configs.parallel import DiTParallelConfig, VAEParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.models.transformers.cosmos_transformer import ditCosmosTransformer3DModel
from visual_gen.utils.logger import get_logger

from .base_pipeline import ditBasePipeline

logger = get_logger(__name__)


class ditCosmos2_5_PredictBasePipeline(Cosmos2_5_PredictBasePipeline, ditBasePipeline):
    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        logger.debug(f"TeaCache config: {TeaCacheConfig.get_instance().to_dict()}")
        pipe_config = kwargs.get("pipeline", {})
        pipe_config["transformer_type"] = "ditCosmosTransformer3DModel"
        PipelineConfig.set_config(**pipe_config)
        if not isinstance(self.transformer, ditCosmosTransformer3DModel):
            logger.debug("Loading ditCosmosTransformer3DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditCosmosTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                revision=kwargs.get('revision', None),
                subfolder="transformer",
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        logger.debug("Post-load hook completed")

        # unlike other models, Cosmos2.5' final denoising t-range: [0.75, 1.0] has a very different
        # input-output correlation pattern. Generally speaking, output would change drastically even
        # when input delta is relatively small. This model-specific switch is to block use of
        # TeaCache for the specific t-range.
        self.teacache_allowed_t_thres = 0.75
        self.transformer.teacache_coefficients = [
            1.00000000e+02, # Mathematical soundness: Keep highest order coeffs positive. Not part of the fit
            -1.1689676e+03,
            4.07797856e+02,
            -4.3215575e+01,
            1.55959631e+00,
            1.02673255e-01,
        ]

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        conditional_frame_timestep: float = 0.1,
    ):
        if False:  # self.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = "cuda"  # self._execution_device

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        num_frames_in = None
        if image is not None:
            if batch_size != 1:
                raise ValueError(f"batch_size must be 1 for image input (given {batch_size})")

            image = F2.to_tensor(image).unsqueeze(0)
            video = torch.cat([image, torch.zeros_like(image).repeat(num_frames - 1, 1, 1, 1)], dim=0)
            video = video.unsqueeze(0)
            num_frames_in = 1
        elif video is None:
            video = torch.zeros(batch_size, num_frames, 3, height, width, dtype=torch.uint8)
            num_frames_in = 0
        else:
            num_frames_in = len(video)

            if batch_size != 1:
                raise ValueError(f"batch_size must be 1 for video input (given {batch_size})")

        assert video is not None
        video = self.video_processor.preprocess_video(video, height, width)

        # pad with last frame (for video2world)
        num_frames_out = num_frames
        if video.shape[2] < num_frames_out:
            n_pad_frames = num_frames_out - num_frames_in
            last_frame = video[0, :, -1:, :, :]  # [C, T==1, H, W]
            pad_frames = last_frame.repeat(1, 1, n_pad_frames, 1, 1)  # [B, C, T, H, W]
            video = torch.cat((video, pad_frames), dim=2)

        assert num_frames_in <= num_frames_out, f"expected ({num_frames_in=}) <= ({num_frames_out=})"

        video = video.to(device=device, dtype=vae_dtype)

        num_channels_latents = self.transformer.config.in_channels - 1
        latents, cond_latent, cond_mask, cond_indicator = self.prepare_latents(
            video=video,
            batch_size=batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames_in=num_frames_in,
            num_frames_out=num_frames,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        cond_timestep = torch.ones_like(cond_indicator) * conditional_frame_timestep
        cond_mask = cond_mask.to(transformer_dtype)

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        # Denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        gt_velocity = (latents - cond_latent) * cond_mask
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t.cpu().item()

                # NOTE: assumes sigma(t) \in [0, 1]
                sigma_t = (
                    torch.tensor(self.scheduler.sigmas[i].item())
                    .unsqueeze(0)
                    .to(device=device, dtype=transformer_dtype)
                )

                in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
                in_latents = in_latents.to(transformer_dtype)
                in_timestep = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t

                cfg_positive_inputs = {
                    "hidden_states": in_latents,
                    "condition_mask": cond_mask,
                    "timestep": in_timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "padding_mask": padding_mask,
                    "return_dict": False,
                    "disallow_teacache": i > num_inference_steps * self.teacache_allowed_t_thres,
                }
                cfg_negative_inputs = None
                if self.do_classifier_free_guidance:
                    cfg_negative_inputs = {
                        "hidden_states": in_latents,
                        "condition_mask": cond_mask,
                        "timestep": in_timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "padding_mask": padding_mask,
                        "return_dict": False,
                        "disallow_teacache": i > num_inference_steps * self.teacache_allowed_t_thres,
                    }

                noise_pred, noise_pred_uncond = self.visual_gen_transformer(
                    self.transformer,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    cfg_positive_inputs=cfg_positive_inputs,
                    cfg_negative_inputs=cfg_negative_inputs,
                )

                # NOTE: replace velocity (noise_pred) with gt_velocity for conditioning inputs only
                noise_pred = gt_velocity + noise_pred * (1 - cond_mask)

                if self.do_classifier_free_guidance:
                    assert noise_pred_uncond is not None, "noise_pred_uncond is required"
                    noise_pred_uncond = gt_velocity + noise_pred_uncond * (1 - cond_mask)
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents_mean = self.latents_mean.to(latents.device, latents.dtype)
            latents_std = self.latents_std.to(latents.device, latents.dtype)
            latents = latents * latents_std + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            video = self._match_num_frames(video, num_frames)

            # assert self.safety_checker is not None
            # self.safety_checker.to(device)
            video = self.video_processor.postprocess_video(video, output_type="np")
            video = (video * 255).astype(np.uint8)
            video_batch = []
            for vid in video:
                # vid = self.safety_checker.check_video_safety(vid)
                video_batch.append(vid)
            video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
            video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)


class ditCosmos2VideoToWorldPipeline(Cosmos2VideoToWorldPipeline, ditBasePipeline):
    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        logger.debug(f"TeaCache config: {TeaCacheConfig.get_instance().to_dict()}")
        pipe_config = kwargs.get("pipeline", {})
        pipe_config["transformer_type"] = "ditCosmosTransformer3DModel"
        PipelineConfig.set_config(**pipe_config)
        if not isinstance(self.transformer, ditCosmosTransformer3DModel):
            logger.debug("Loading ditCosmosTransformer3DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditCosmosTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )
            gc.collect()
            torch.cuda.empty_cache()
        logger.debug("Post-load hook completed")

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        fps: int = 16,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        sigma_conditioning: float = 0.0001,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*):
                The image to be used as a conditioning input for the video generation.
            video (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                The video to be used as a conditioning input for the video generation.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `704`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `93`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `35`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`.
            fps (`int`, defaults to `16`):
                The frames per second of the generated video.
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
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`CosmosPipelineOutput`] instead of a plain tuple.
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
                The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
                the prompt is shorter than this length, it will be padded.
            sigma_conditioning (`float`, defaults to `0.0001`):
                The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
                set to a small value close to zero.

        Examples:

        Returns:
            [`~CosmosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CosmosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if self.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        n_heads = self.transformer.config.num_attention_heads
        if n_heads % DiTParallelConfig.ulysses_size() != 0:
            raise ValueError(
                f"`num_attention_heads` ({n_heads}) must be divisible by ulysses_size ({DiTParallelConfig.ulysses_size()})"
            )

        if AttentionOpManager.attn_type == "sparse-videogen":
            self._set_sparse_videogen_attrs(
                context_length=0,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_size=1,
                num_head=n_heads,
                head_dim=self.transformer.config.attention_head_dim,
            )

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )
            self.safety_checker.to("cpu")

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

        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to(torch.cuda.current_device())
        nvtx.range_push("encode_prompt")
        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        nvtx.range_pop()
        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 4. Prepare timesteps
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        if self.scheduler.config.final_sigmas_type == "sigma_min":
            # Replace the last sigma (which is zero) with the minimum sigma value
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        # 5. Prepare latent variables
        vae_dtype = self.vae.dtype
        transformer_dtype = vae_dtype  # self.transformer.dtype could drift to float8 if linear type is set to float8

        if image is not None:
            video = self.video_processor.preprocess(image, height, width).unsqueeze(2)
        else:
            video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=vae_dtype)

        nvtx.range_push("prepare_latents")
        num_channels_latents = self.transformer.config.in_channels - 1
        latents, conditioning_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask = self.prepare_latents(
            video,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            self.do_classifier_free_guidance,
            torch.float32,
            device,
            generator,
            latents,
        )
        nvtx.range_pop()
        unconditioning_latents = None

        cond_mask = cond_mask.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            uncond_mask = uncond_mask.to(transformer_dtype)
            unconditioning_latents = conditioning_latents

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)
        sigma_conditioning = torch.tensor(sigma_conditioning, dtype=torch.float32, device=device)
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to(torch.cuda.current_device())
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]

                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    latents.size(0), -1, latents.size(2), -1, -1
                )  # [B, 1, T, 1, 1]

                cond_latent = latents * c_in
                cond_latent = cond_indicator * conditioning_latents + (1 - cond_indicator) * cond_latent
                cond_latent = cond_latent.to(transformer_dtype)
                cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                cond_timestep = cond_timestep.to(transformer_dtype)

                cfg_positive_inputs = {
                    "hidden_states": cond_latent,
                    "timestep": cond_timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "fps": fps,
                    "condition_mask": cond_mask,
                    "padding_mask": padding_mask,
                    "return_dict": False,
                }
                cfg_negative_inputs = None
                if self.do_classifier_free_guidance:
                    uncond_latent = latents * c_in
                    uncond_latent = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * uncond_latent
                    uncond_latent = uncond_latent.to(transformer_dtype)
                    uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                    uncond_timestep = uncond_timestep.to(transformer_dtype)
                    cfg_negative_inputs = {
                        "hidden_states": uncond_latent,
                        "timestep": uncond_timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "fps": fps,
                        "condition_mask": uncond_mask,
                        "padding_mask": padding_mask,
                        "return_dict": False,
                    }

                noise_pred, noise_pred_uncond = self.visual_gen_transformer(
                    self.transformer,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    cfg_positive_inputs=cfg_positive_inputs,
                    cfg_negative_inputs=cfg_negative_inputs,
                )

                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)
                noise_pred = cond_indicator * conditioning_latents + (1 - cond_indicator) * noise_pred

                if self.do_classifier_free_guidance:
                    assert noise_pred_uncond is not None, "noise_uncond is required"
                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
                    noise_pred_uncond = (
                        uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                    )
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                noise_pred = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
        self._current_timestep = None
        latents = self.dit_dp_gather(latents)

        if not output_type == "latent":
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            if self.safety_checker is not None:
                self.safety_checker.to(device)
                video = self.video_processor.postprocess_video(video, output_type="np")
                video = (video * 255).astype(np.uint8)
                video_batch = []
                for vid in video:
                    vid = self.safety_checker.check_video_safety(vid)
                    video_batch.append(vid)
                video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
                video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
                video = self.video_processor.postprocess_video(video, output_type=output_type)
                self.safety_checker.to("cpu")
            else:
                video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)


class ditCosmos2TextToImagePipeline(Cosmos2TextToImagePipeline, ditBasePipeline):
    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        logger.debug(f"TeaCache config: {TeaCacheConfig.get_instance().to_dict()}")
        pipe_config = kwargs.get("pipeline", {})
        pipe_config["transformer_type"] = "ditCosmosTransformer3DModel"
        PipelineConfig.set_config(**pipe_config)
        if not isinstance(self.transformer, ditCosmosTransformer3DModel):
            logger.debug("Loading ditCosmosTransformer3DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditCosmosTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )
            gc.collect()
            torch.cuda.empty_cache()
        logger.debug("Post-load hook completed")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 768,
        width: int = 1360,
        num_inference_steps: int = 35,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if self.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_frames = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )
            self.safety_checker.to("cpu")

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

        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to(torch.cuda.current_device())
        nvtx.range_push("encode_prompt")
        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        nvtx.range_pop()
        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 4. Prepare timesteps
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
        if self.scheduler.config.get("final_sigmas_type", "zero") == "sigma_min":
            # Replace the last sigma (which is zero) with the minimum sigma value
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

        # 5. Prepare latent variables
        transformer_dtype = self.transformer.dtype
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to(torch.cuda.current_device())
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]

                current_t = current_sigma / (current_sigma + 1)
                c_in = 1 - current_t
                c_skip = 1 - current_t
                c_out = -current_t
                timestep = current_t.expand(latents.shape[0]).to(transformer_dtype)  # [B, 1, T, 1, 1]

                latent_model_input = latents * c_in
                latent_model_input = latent_model_input.to(transformer_dtype)

                cfg_positive_inputs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "padding_mask": padding_mask,
                    "return_dict": False,
                }
                cfg_negative_inputs = None
                if self.do_classifier_free_guidance:
                    cfg_negative_inputs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "padding_mask": padding_mask,
                        "return_dict": False,
                    }

                noise_pred, noise_pred_uncond = self.visual_gen_transformer(
                    self.transformer,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    cfg_positive_inputs=cfg_positive_inputs,
                    cfg_negative_inputs=cfg_negative_inputs,
                )
                noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)
                if self.do_classifier_free_guidance:
                    noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
                    noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                noise_pred = (latents - noise_pred) / current_sigma
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()

        self._current_timestep = None
        latents = self.dit_dp_gather(latents)

        if not output_type == "latent":
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std / self.scheduler.config.sigma_data + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            if self.safety_checker is not None:
                self.safety_checker.to(device)
                video = self.video_processor.postprocess_video(video, output_type="np")
                video = (video * 255).astype(np.uint8)
                video_batch = []
                for vid in video:
                    vid = self.safety_checker.check_video_safety(vid)
                    video_batch.append(vid)
                video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
                video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
                video = self.video_processor.postprocess_video(video, output_type=output_type)
                self.safety_checker.to("cpu")
            else:
                video = self.video_processor.postprocess_video(video, output_type=output_type)
            image = [batch[0] for batch in video]
            if isinstance(video, torch.Tensor):
                image = torch.stack(image)
            elif isinstance(video, np.ndarray):
                image = np.stack(image)
        else:
            image = latents[:, :, 0]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return CosmosImagePipelineOutput(images=image)


class ditCosmosTextToWorldPipeline(CosmosTextToWorldPipeline, ditBasePipeline):

    def _after_load(self, pretrained_model_name_or_path, *args, **kwargs) -> None:
        logger.debug(f"TeaCache config: {TeaCacheConfig.get_instance().to_dict()}")
        pipe_config = kwargs.get("pipeline", {})
        pipe_config["transformer_type"] = "ditCosmosTransformer3DModel"
        PipelineConfig.set_config(**pipe_config)
        if not isinstance(self.transformer, ditCosmosTransformer3DModel):
            logger.debug("Loading ditCosmosTransformer3DModel from diffusers transformer")
            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            self.transformer = ditCosmosTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path, subfolder="transformer", torch_dtype=torch_dtype, low_cpu_mem_usage=True
            )
            gc.collect()
            torch.cuda.empty_cache()
        logger.debug("Post-load hook completed")

        self.transformer.teacache_coefficients = [
            2.71156237e02,
            -9.19775607e01,
            2.24437250e00,
            2.08355751e00,
            1.41776330e-01,
        ]

        if not VAEParallelConfig.disable_parallel_vae:
            logger.warning("VAE parallel is not supported for CosmosPipeline")

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 121,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        fps: int = 30,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if self.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        n_heads = self.transformer.config.num_attention_heads
        if n_heads % DiTParallelConfig.ulysses_size() != 0:
            raise ValueError(
                f"`num_attention_heads` ({n_heads}) must be divisible by ulysses_size ({DiTParallelConfig.ulysses_size()})"
            )

        if AttentionOpManager.attn_type == "sparse-videogen":
            self._set_sparse_videogen_attrs(
                context_length=0,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_size=1,
                num_head=n_heads,
                head_dim=self.transformer.config.attention_head_dim,
            )

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if self.safety_checker is not None:
            self.safety_checker.to(device)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )
            self.safety_checker.to("cpu")

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

        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to(torch.cuda.current_device())
        nvtx.range_push("encode_prompt")
        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )
        nvtx.range_pop()
        if "text_encoder" in PipelineConfig.model_wise_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device)

        nvtx.range_push("prepare_latents")
        # 5. Prepare latent variables
        transformer_dtype = self.transformer.dtype
        num_channels_latents = self.transformer.config.in_channels
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

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to(torch.cuda.current_device())
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(transformer_dtype)

                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = latent_model_input.to(transformer_dtype)

                cfg_positive_inputs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "fps": fps,
                    "padding_mask": padding_mask,
                    "return_dict": False,
                }
                cfg_negative_inputs = None
                if self.do_classifier_free_guidance:
                    cfg_negative_inputs = {
                        "hidden_states": latent_model_input,
                        "timestep": timestep,
                        "encoder_hidden_states": negative_prompt_embeds,
                        "fps": fps,
                        "padding_mask": padding_mask,
                        "return_dict": False,
                    }

                noise_pred, noise_pred_uncond = self.visual_gen_transformer(
                    self.transformer,
                    current_denoising_step=i,
                    num_inference_steps=num_inference_steps,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    cfg_positive_inputs=cfg_positive_inputs,
                    cfg_negative_inputs=cfg_negative_inputs,
                )

                sample = latents
                if self.do_classifier_free_guidance:
                    noise_pred = torch.cat([noise_pred_uncond, noise_pred])
                    sample = torch.cat([sample, sample])

                # pred_original_sample (x0)
                noise_pred = self.scheduler.step(noise_pred, t, sample, return_dict=False)[1]
                self.scheduler._step_index -= 1

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_cond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # pred_sample (eps)
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False, pred_original_sample=noise_pred
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        if "transformer" in PipelineConfig.model_wise_offloading:
            self.transformer.to("cpu")
            torch.cuda.empty_cache()

        self._current_timestep = None
        latents = self.dit_dp_gather(latents)

        if not output_type == "latent":
            if self.vae.config.latents_mean is not None:
                latents_mean, latents_std = self.vae.config.latents_mean, self.vae.config.latents_std
                latents_mean = (
                    torch.tensor(latents_mean)
                    .view(1, self.vae.config.latent_channels, -1, 1, 1)[:, :, : latents.size(2)]
                    .to(latents)
                )
                latents_std = (
                    torch.tensor(latents_std)
                    .view(1, self.vae.config.latent_channels, -1, 1, 1)[:, :, : latents.size(2)]
                    .to(latents)
                )
                latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean
            else:
                latents = latents / self.scheduler.config.sigma_data
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            if self.safety_checker is not None:
                self.safety_checker.to(device)
                video = self.video_processor.postprocess_video(video, output_type="np")
                video = (video * 255).astype(np.uint8)
                video_batch = []
                for vid in video:
                    vid = self.safety_checker.check_video_safety(vid)
                    video_batch.append(vid)
                video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
                video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
                video = self.video_processor.postprocess_video(video, output_type=output_type)
                self.safety_checker.to("cpu")
            else:
                video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CosmosPipelineOutput(frames=video)
