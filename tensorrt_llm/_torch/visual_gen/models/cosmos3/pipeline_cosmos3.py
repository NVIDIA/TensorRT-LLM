# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import os
import time
from typing import List, Optional, Union

import PIL.Image
import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import Qwen2Tokenizer

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

from .defaults import COSMOS3_720P_PARAMS, COSMOS3_EXTRA_SPECS
from .guardrails import check_video_safety, download_guardrail_checkpoint
from .transformer_cosmos3 import Cosmos3VFMTransformer

COSMOS3_DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, "
    "low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, "
    "unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of "
    "poor quality."
)
COSMOS3_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate videos from a given prompt."
)
COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_DEFAULT_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
TRTLLM_DISABLE_COSMOS3_GUARDRAILS = os.environ.get("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"

# Public offload component names for the two transformer towers. The "reasoner"
# (understanding) pathway is the causal language model that processes text; the
# "generator" (generation) pathway is the stack of cross-attention layers that
# produces video tokens. Only the heavy decoder-layer ModuleLists are offloaded;
# the small shared embeddings/projections/norms stay resident on GPU.
COSMOS3_REASONER_OFFLOAD_COMPONENT = "reasoner"
COSMOS3_GENERATOR_OFFLOAD_COMPONENT = "generator"
# Opt-in guardrail offload components (rank 0 only): the Qwen3Guard text checker and
# the RetinaFace video face-blur model. Kept out of the CPU defaults below.
COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT = "text_guardrail"
COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT = "video_guardrail"
_COSMOS3_DEFAULT_OFFLOAD_STAGES = (
    (COSMOS3_REASONER_OFFLOAD_COMPONENT,),
    (COSMOS3_GENERATOR_OFFLOAD_COMPONENT,),
)


@register_pipeline(
    "Cosmos3OmniMoTPipeline",
    hf_ids=[
        "nvidia/Cosmos3-Nano",
        "nvidia/Cosmos3-Super",
        "nvidia/Cosmos3-Super-Image2Video",
        "nvidia/Cosmos3-Super-Text2Image",
    ],
    doc="Cosmos3 Omnimodal world models.",
)
class Cosmos3OmniMoTPipeline(BasePipeline):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)

    def _init_transformer(self) -> None:
        logger.info("Initializing Cosmos3VFMTransformer")
        self.transformer = Cosmos3VFMTransformer(self.pipeline_config.model_configs["transformer"])

    # =========================================================================
    # Offloading
    # =========================================================================

    def default_offload_stages(self) -> tuple[tuple[str, ...], ...]:
        """Offload the reasoner and generator towers as separate stages.

        Only invoked when ``cpu_offload_config.enable`` is true (the base class
        short-circuits before calling this). Defaults are CPU-staged today;
        non-CPU offload devices skip the defaults and require explicit
        ``cpu_offload_config.stages`` from the user.
        """
        cpu_offload_config = self.pipeline_config.cpu_offload_config
        if cpu_offload_config.device != "cpu":
            logger.warning(
                "Cosmos3 default offload stages are CPU-only; "
                f"cpu_offload_config.device='{cpu_offload_config.device}' has no "
                "default stages. Set cpu_offload_config.stages explicitly to "
                "stage components on a non-CPU device."
            )
            return ()
        return _COSMOS3_DEFAULT_OFFLOAD_STAGES

    def offload_pipeline_components(self) -> dict[str, nn.Module]:
        """Expose the two transformer towers, VAE, and guardrails as offload components.

        Cosmos3 packs both pathways into a single ``transformer`` module, so the
        default ``BasePipeline.offload_pipeline_components`` (which looks for a
        ``transformer.blocks`` ModuleList) does not apply. We expose the heavy
        decoder-layer ModuleLists of each tower individually so they can be
        brought on/off the GPU independently. The opt-in guardrail components wrap
        the underlying safety nn.Modules (loaded on rank 0 only).
        """
        components: dict[str, nn.Module] = {}

        transformer = getattr(self, "transformer", None)
        if transformer is not None:
            language_model = getattr(transformer, "language_model", None)
            reasoner_layers = (
                getattr(language_model, "layers", None) if language_model is not None else None
            )
            if reasoner_layers is not None:
                components[COSMOS3_REASONER_OFFLOAD_COMPONENT] = reasoner_layers

            generator_layers = getattr(transformer, "gen_layers", None)
            if generator_layers is not None:
                components[COSMOS3_GENERATOR_OFFLOAD_COMPONENT] = generator_layers

        vae = getattr(self, PipelineComponent.VAE.value, None)
        if vae is not None:
            components[PipelineComponent.VAE.value] = vae

        # Guardrails (rank 0 only). CosmosSafetyChecker is an nn.Module but its
        # GuardrailRunner children are plain objects, so expose the real safety
        # nn.Modules (Qwen3Guard, RetinaFaceFilter) wrapped in a ModuleList.
        safety_checker = getattr(self, "safety_checker", None)
        if safety_checker is not None:
            for component_name, runner in (
                (COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT, safety_checker.text_guardrail),
                (COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT, safety_checker.video_guardrail),
            ):
                modules = [m for m in runner.models if isinstance(m, nn.Module)]
                if modules:
                    components[component_name] = nn.ModuleList(modules)

        return components

    def extra_offload_component_names(self) -> set[str]:
        # Guardrails load on rank 0 only; treat their names as valid on all ranks
        # so explicit multi-GPU stages don't fail during validation. Other ranks
        # drop them later via the offloader's stage filtering.
        return {COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT, COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT}

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.eval()

    def load_standard_components(
        self, checkpoint_dir: str, device: torch.device, skip_components: Optional[list] = []
    ) -> None:
        skip_components = skip_components or []

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading tokenizer...")
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                checkpoint_dir,
                subfolder="text_tokenizer",
            )

        # Cosmos3 canonical defaults — overwritten if VAE is loaded
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 16

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading VAE...")
            vae_device = (
                torch.device("cpu")
                if PipelineComponent.VAE.value in self.offloader.requested_components()
                else device
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.bfloat16,  # load VAE in BF16 for memory saving
            ).to(vae_device)

            self.vae_scale_factor_temporal = getattr(
                self.vae.config, "scale_factor_temporal", self.vae_scale_factor_temporal
            )
            self.vae_scale_factor_spatial = getattr(
                self.vae.config, "scale_factor_spatial", self.vae_scale_factor_spatial
            )
            self.transformer.temporal_compression_factor = self.vae_scale_factor_temporal

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading scheduler...")
            self.scheduler = UniPCMultistepScheduler.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.SCHEDULER,
            )

        # Re-check the env var in case it was changed after initialization like in unit tests.
        guardrails_disabled = os.environ.get("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"
        global TRTLLM_DISABLE_COSMOS3_GUARDRAILS
        TRTLLM_DISABLE_COSMOS3_GUARDRAILS = guardrails_disabled
        if not TRTLLM_DISABLE_COSMOS3_GUARDRAILS:
            # lazy import
            try:
                from cosmos_guardrail import CosmosSafetyChecker
            except (ImportError, ModuleNotFoundError):
                raise ValueError(
                    "Cosmos Guardrail is not installed. This is in violation of the "
                    "[NVIDIA Open Model License Agreement]"
                    "(https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                    "Please run the following installation commands or "
                    "explicitly disable guardrails by setting TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1 "
                    "(user is responsible for deploying the model without guardrails). "
                    "- `pip install cosmos_guardrail==0.3.0 && pip uninstall opencv-python`"
                )
            # Guardrails are only evaluated on rank 0; load them only there to avoid
            # dead model weights occupying GPU memory on every other rank.
            if self.rank == 0:
                # the download guardrail checkpoint will bypass CosmosSafetyChecker's checkpoint download.
                # Both will use HF_HOME as the cache directory.
                download_guardrail_checkpoint()
                self.safety_checker = CosmosSafetyChecker()
                self.safety_checker.to(device)

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @property
    def default_warmup_resolutions(self):
        return [(720, 1280)]

    @property
    def default_warmup_num_frames(self):
        return [189]

    @property
    def default_generation_params(self):
        return dict(COSMOS3_720P_PARAMS)

    @property
    def extra_param_specs(self):
        return dict(COSMOS3_EXTRA_SPECS)

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        with torch.no_grad():
            self.forward(
                prompt="warmup",
                negative_prompt="",
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=steps,
                guidance_scale=COSMOS3_720P_PARAMS["guidance_scale"],
                seed=42,
                max_sequence_length=COSMOS3_720P_PARAMS["max_sequence_length"],
                use_guardrails=False,
                image=None,
            )

    def infer(self, req):
        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.params.negative_prompt,
            image=req.params.image,
            height=req.params.height,
            width=req.params.width,
            num_frames=req.params.num_frames,
            num_inference_steps=req.params.num_inference_steps,
            guidance_scale=req.params.guidance_scale,
            seed=req.params.seed,
            max_sequence_length=req.params.max_sequence_length,
            frame_rate=req.params.frame_rate,
            use_duration_template=req.params.extra_params.get("use_duration_template", True),
            use_resolution_template=req.params.extra_params.get("use_resolution_template", True),
            use_system_prompt=req.params.extra_params.get("use_system_prompt", False),
            use_guardrails=req.params.extra_params.get("use_guardrails", True),
        )

    def _format_prompt_with_template(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        use_duration_template: bool = True,
        use_resolution_template: bool = True,
    ) -> str:
        prompt = prompt.strip()

        if use_duration_template and num_frames > 1:
            duration = num_frames / frame_rate
            dur_text = COSMOS3_DURATION_TEMPLATE.format(duration=duration, fps=frame_rate)
            prompt = prompt.rstrip(".") + ". " + dur_text

        prompt = prompt.strip()
        if use_resolution_template:
            res_text = COSMOS3_DEFAULT_RESOLUTION_TEMPLATE.format(height=height, width=width)
            prompt = prompt.rstrip(".") + ". " + res_text

        return prompt

    def _resize_and_center_crop_image(
        self, image: PIL.Image.Image, height: int, width: int
    ) -> PIL.Image.Image:
        """Match Cosmos3 reference preprocessing for conditioning images."""
        orig_w, orig_h = image.size
        scaling_ratio = max(width / orig_w, height / orig_h)
        resize_w = int(math.ceil(scaling_ratio * orig_w))
        resize_h = int(math.ceil(scaling_ratio * orig_h))

        image = image.resize((resize_w, resize_h), PIL.Image.Resampling.LANCZOS)

        left = max((resize_w - width) // 2, 0)
        top = max((resize_h - height) // 2, 0)
        return image.crop((left, top, left + width, top + height))

    @nvtx_range("_tokenize_prompt", color="blue")
    def _tokenize_prompt(
        self, text: str, max_sequence_length: int, use_system_prompt: bool = False
    ):
        """Tokenize a prompt using the Qwen2 chat template.

        Returns (input_ids, attention_mask) as [1, S] tensors on device.
        """
        conversations = (
            [{"role": "system", "content": COSMOS3_DEFAULT_SYSTEM_PROMPT}]
            if use_system_prompt
            else []
        )
        conversations.append(
            {"role": "user", "content": text},
        )
        token_ids = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        reserved_tokens = 2
        if max_sequence_length < reserved_tokens:
            raise ValueError(
                f"max_sequence_length must be at least {reserved_tokens}, got {max_sequence_length}"
            )
        token_ids = token_ids[: max_sequence_length - reserved_tokens]
        token_ids.append(self.tokenizer.eos_token_id)  # 151645
        token_ids.append(self.tokenizer.convert_tokens_to_ids("<|vision_start|>"))  # 151652
        seq_len = len(token_ids)

        # Pad to max_sequence_length
        pad_len = max_sequence_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        token_ids = token_ids + [self.tokenizer.pad_token_id or 0] * pad_len

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=self.device)
        return input_ids, attention_mask

    # =========================================================================
    # Latent preparation
    # =========================================================================

    @nvtx_range("_prepare_latents", color="blue")
    def _prepare_latents(self, height, width, num_frames, generator):
        num_channels_latents = self.transformer.latent_channel_size
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

    # -- I2V latent preparation -----------------------------------------------

    def _encode_conditioning_video(
        self,
        image_tensor: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """VAE-encode a conditioning image as a full-length video.

        The WAN VAE has temporal compression (factor 4), so encoding a single
        frame produces degenerate temporal features.  Following imaginaire4's
        ``build_conditioned_video_batch``, we fill the entire pixel-space video
        with the conditioning image (repeating it across all frames) so the
        temporal encoder sees plausible content everywhere.  The caller then
        keeps only the conditioned latent frame(s) and replaces the rest with
        noise.

        Args:
            image_tensor: [1, 3, H, W] in [-1, 1]
            num_frames: total pixel frames for the video
            height: pixel height
            width: pixel width

        Returns:
            [1, C, T_latent, H_latent, W_latent] normalized latent of the
            full conditioning video.
        """
        # Build pixel-space video: repeat the conditioning image across all frames
        # image_tensor: [1, 3, H, W] -> [1, 3, 1, H, W] -> [1, 3, num_frames, H, W]
        video = image_tensor.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        video = video.to(device=self.device, dtype=self.vae.dtype)

        with self.offloader.context_if_requested(PipelineComponent.VAE.value):
            latent = self.vae.encode(video).latent_dist.mode()

        # Normalize (inverse of _decode_latents denormalization)
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, -1, 1, 1, 1)
                .to(latent.device, latent.dtype)
            )
            latent = (latent - latents_mean) / latents_std
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latent = latent * scaling_factor

        return latent.to(self.dtype)

    def _prepare_latents_i2v(
        self,
        image_tensor: torch.Tensor,
        height: int,
        width: int,
        num_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare initial latents with frame 0 conditioned on the input image.

        The conditioning image is repeated across all pixel frames before VAE
        encoding so the temporal encoder sees plausible content everywhere
        (avoids degenerate single-frame encoding with the WAN VAE's temporal
        compression).  Only frame 0 of the resulting latent is kept clean;
        the rest is replaced with noise.

        Returns:
            latents: [1, C, T_lat, H_lat, W_lat] with frame 0 = image, rest = noise
            velocity_mask: [1, 1, T_lat, 1, 1] with frame 0 = 0, rest = 1
            image_latent: [1, C, 1, H_lat, W_lat] clean frame 0 for re-injection
        """
        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # Pure noise
        noise = randn_tensor(
            (
                1,
                C,
                T_lat,
                height // self.vae_scale_factor_spatial,
                width // self.vae_scale_factor_spatial,
            ),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Encode full conditioning video (image repeated across all frames)
        cond_latent = self._encode_conditioning_video(
            image_tensor,
            num_frames,
            height,
            width,
        )  # [1, C, T_lat, H_lat, W_lat]

        # Keep only frame 0 for conditioning; replace rest with noise
        image_latent = cond_latent[:, :, 0:1, :, :]  # [1, C, 1, H_lat, W_lat]

        condition_mask = torch.zeros(1, 1, T_lat, 1, 1, device=self.device, dtype=self.dtype)
        condition_mask[:, :, 0, :, :] = 1.0

        latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise

        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, image_latent

    # =========================================================================
    # VAE decode
    # =========================================================================

    @nvtx_range("_decode_latents", color="blue")
    def _decode_latents(self, latents):
        latents = latents.to(self.vae.dtype)

        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            if not hasattr(self, "_latents_mean"):
                self._latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
                self._latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, -1, 1, 1, 1)
                    .to(self.device, self.vae.dtype)
                )
            latents = (latents * self._latents_std) + self._latents_mean
        else:
            scaling_factor = self.vae.config.get("scaling_factor", 1.0)
            latents = latents / scaling_factor

        with self.offloader.context_if_requested(PipelineComponent.VAE.value):
            video = self.vae.decode(latents, return_dict=False)[0]
        video = postprocess_video_tensor(video)
        return video

    # =========================================================================
    # Forward (main generation entry point)
    # =========================================================================

    @nvtx_range("Cosmos3OmniMoTPipeline.forward")
    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        seed: int,
        negative_prompt: Optional[str] = None,
        image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
        height: int = COSMOS3_720P_PARAMS["height"],
        width: int = COSMOS3_720P_PARAMS["width"],
        num_frames: int = COSMOS3_720P_PARAMS["num_frames"],
        num_inference_steps: int = COSMOS3_720P_PARAMS["num_inference_steps"],
        guidance_scale: float = COSMOS3_720P_PARAMS["guidance_scale"],
        max_sequence_length: int = COSMOS3_720P_PARAMS["max_sequence_length"],
        frame_rate: float = COSMOS3_720P_PARAMS["frame_rate"],
        use_duration_template: bool = COSMOS3_EXTRA_SPECS["use_duration_template"].default,
        use_resolution_template: bool = COSMOS3_EXTRA_SPECS["use_resolution_template"].default,
        use_system_prompt: bool = COSMOS3_EXTRA_SPECS["use_system_prompt"].default,
        use_guardrails: bool = COSMOS3_EXTRA_SPECS["use_guardrails"].default,
    ):
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        use_guardrails = use_guardrails and not TRTLLM_DISABLE_COSMOS3_GUARDRAILS

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        if batch_size > 1:
            # TODO: support batch generation
            raise ValueError("Batch generation is not supported for Cosmos3")

        # Validate image input — only single image is supported for batch generation
        if image is not None and not isinstance(image, (PIL.Image.Image, torch.Tensor, str)):
            raise ValueError(
                f"`image` must be a PIL.Image, torch.Tensor, or file path string, "
                f"got {type(image)}. Batch of different images is not supported; "
                f"use a single image with multiple prompts instead."
            )

        # Text guardrail — check both positive and user-supplied negative prompts.
        # None negative_prompt means the hardcoded default will be used (safe); skip it.
        text_blocked = torch.zeros((), device=self.device, dtype=torch.int32)
        if self.rank == 0 and use_guardrails and self.safety_checker is not None:
            prompts_to_check = list(prompt)
            if negative_prompt is not None:
                prompts_to_check.append(negative_prompt)
            with self.offloader.context_if_requested(COSMOS3_TEXT_GUARDRAIL_OFFLOAD_COMPONENT):
                for p in prompts_to_check:
                    is_safe = self.safety_checker.check_text_safety(p)
                    if not is_safe:
                        logger.warning("Text guardrail blocked prompt")
                        text_blocked.fill_(1)
                        break

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(text_blocked, src=0)

        if text_blocked.item():
            timer.mark_end()
            return timer.fill(PipelineOutput())

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if negative_prompt is None:
            negative_prompt = COSMOS3_DEFAULT_NEGATIVE_PROMPT

        negative_prompt = self._format_prompt_with_template(
            negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            use_duration_template=use_duration_template,
            use_resolution_template=use_resolution_template,
        )

        prompt = [
            self._format_prompt_with_template(
                p,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                use_duration_template=use_duration_template,
                use_resolution_template=use_resolution_template,
            )
            for p in prompt
        ]
        logger.info(f"Prompt with metadata: '{prompt}'")

        prompt = prompt[0]

        # 1. Tokenize prompts (no separate text encoder — transformer embeds internally)
        logger.info("Tokenizing prompts...")
        cond_ids, cond_mask = self._tokenize_prompt(prompt, max_sequence_length, use_system_prompt)
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt
        )

        # 2. Prepare latents
        if image is not None:
            if isinstance(image, str):
                image = PIL.Image.open(image).convert("RGB")

            if isinstance(image, PIL.Image.Image):
                image = image.convert("RGB")
                image = self._resize_and_center_crop_image(image, height=height, width=width)
                image = self.video_processor.preprocess(
                    image,
                    height=height,
                    width=width,
                )

            latents, velocity_mask, image_latent = self._prepare_latents_i2v(
                image, height=height, width=width, num_frames=num_frames, generator=generator
            )
        else:
            latents = self._prepare_latents(height, width, num_frames, generator)
            velocity_mask = None
            image_latent = None

        # Compute video shape in latent space
        T_latent = latents.shape[2]
        H_latent = latents.shape[3]
        W_latent = latents.shape[4]
        video_shape = (T_latent, H_latent, W_latent)

        # 3. Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 4. Build forward_fn for the denoise loop
        def forward_fn(
            latent_input, extra_stream_latents, timestep, encoder_hidden_states, extra_tensors
        ):
            """Cosmos3 forward function for BasePipeline.denoise().

            Since Cosmos3 embeds text internally, we pass token IDs via extra_tensors
            rather than through encoder_hidden_states.
            """
            noise_pred = self.transformer(
                hidden_states=latent_input,
                timestep=timestep,
                text_ids=extra_tensors["text_ids"],
                text_mask=extra_tensors["text_mask"],
                video_shape=video_shape,
                fps=frame_rate,
                noisy_frame_mask=velocity_mask,
                offload_context=self.offloader.context_if_requested,
            )
            if velocity_mask is not None:
                noise_pred = noise_pred * velocity_mask
            return noise_pred

        # 5. Build CFG tensors — text_ids and text_mask need to be split for CFG
        #    BasePipeline.denoise batches [uncond, cond] when guidance_scale > 1
        #    We pass text IDs/masks through extra_cfg_tensors so they get split correctly
        extra_cfg_tensors = {
            "text_ids": (cond_ids, uncond_ids),
            "text_mask": (cond_mask, uncond_mask),
        }

        self.transformer.reset_cache()

        # 6. Denoise
        timer.mark_denoise_start()
        latents = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=cond_ids,  # placeholder — actual conditioning via extra_cfg_tensors
            neg_prompt_embeds=uncond_ids,
            guidance_scale=guidance_scale,
            forward_fn=forward_fn,
            extra_cfg_tensors=extra_cfg_tensors,
        )
        timer.mark_post_start()

        # 7. Decode
        logger.info("Decoding video...")
        decode_start = time.time()

        if image_latent is not None:
            latents = latents.clone()
            latents[:, :, 0:1, :, :] = image_latent.to(device=latents.device, dtype=latents.dtype)

        video = self.decode_latents(latents, self._decode_latents)

        # Video guardrails
        if self.rank == 0:
            logger.info(f"Video decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

            if use_guardrails and self.safety_checker is not None:
                with self.offloader.context_if_requested(COSMOS3_VIDEO_GUARDRAIL_OFFLOAD_COMPONENT):
                    video = check_video_safety(video, self.safety_checker)

        timer.mark_end()
        return timer.fill(PipelineOutput(video=video, frame_rate=frame_rate))
