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

import json
import math
import os
import time
from typing import List, Optional, Union

import PIL.Image
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import Qwen2Tokenizer

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.logger import logger

from .defaults import COSMOS3_720P_PARAMS, COSMOS3_EXTRA_SPECS, COSMOS3_T2I_PARAMS
from .guardrails import check_video_safety, download_guardrail_checkpoint
from .sound_tokenizer import LatentAutoEncoderV2
from .transformer_cosmos3 import Cosmos3VFMTransformer

COSMOS3_DEFAULT_NEGATIVE_PROMPT = ""
COSMOS3_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate videos from a given prompt."
)
COSMOS3_T2I_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate images from a given prompt."
)
COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_DEFAULT_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
COSMOS3_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."

TRTLLM_DISABLE_COSMOS3_GUARDRAILS = os.environ.get("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"


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
        primary_pretrained_config = pipeline_config.primary_pretrained_config
        self.audio_gen = False
        self.action_gen = False
        if getattr(
            primary_pretrained_config,
            "audio_gen",
            getattr(primary_pretrained_config, "sound_gen", False),
        ):
            logger.info("Initializing Cosmos3OmniMoTPipeline with audio generation.")
            self.audio_gen = True

        if getattr(primary_pretrained_config, "action_gen", False):
            logger.info("Initializing Cosmos3OmniMoTPipeline with action generation.")
            self.action_gen = True

        super().__init__(pipeline_config)

    def _init_transformer(self) -> None:
        logger.info("Initializing Cosmos3VFMTransformer")
        model_config = self.pipeline_config.model_configs["transformer"]
        self.transformer = Cosmos3VFMTransformer(model_config)

    def load_weights(self, weights: dict) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            self.transformer.eval()

    def load_standard_components(
        self, checkpoint_dir: str, device: torch.device, skip_components: Optional[list] = []
    ) -> None:
        skip_components = skip_components or []

        if self.audio_gen and PipelineComponent.SOUND_TOKENIZER not in skip_components:
            logger.info("Loading audio tokenizer...")
            self.audio_tokenizer = (
                LatentAutoEncoderV2.from_pretrained(
                    checkpoint_dir,
                    subfolder=PipelineComponent.SOUND_TOKENIZER,
                )
                .to(device)
                .to(self.dtype)
                .eval()
            )

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
            self.vae = AutoencoderKLWan.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.bfloat16,  # load VAE in BF16 for memory saving
            ).to(device)

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
            # Snapshot the checkpoint scheduler config so the scheduler can be
            # rebuilt at request time when a mode-specific ``flow_shift`` is
            # needed (T2I uses shift=3.0; T2V/I2V keep the checkpoint default).
            self._base_scheduler_config = self.scheduler.config
            self._engine_init_flow_shift = float(
                getattr(self.scheduler.config, "flow_shift", 1.0) or 1.0
            )
            self._current_flow_shift = self._engine_init_flow_shift
            if self.audio_gen:
                # Separate instance so video and audio scheduler states don't collide
                # (UniPC mutates internal correction buffers on every .step() call).
                self.audio_scheduler = UniPCMultistepScheduler.from_config(self.scheduler.config)

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

    def _set_flow_shift(self, target_shift: float) -> None:
        """Rebuild the UniPC scheduler with ``flow_shift=target_shift`` if needed.

        T2I uses ``flow_shift=3.0`` while T2V/I2V use the checkpoint default.
        ``self._current_flow_shift`` is tracked explicitly so a prior T2I rebuild
        does not leak into a subsequent video request.
        """
        if not hasattr(self, "_base_scheduler_config"):
            return
        target = float(target_shift)
        if target == float(self._current_flow_shift):
            return
        self.scheduler = UniPCMultistepScheduler.from_config(
            self._base_scheduler_config, flow_shift=target
        )
        self._current_flow_shift = target

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
                enable_audio=False,
            )

    @staticmethod
    def _resolve_t2i_default(merged_value, video_default, t2i_default):
        """Pick the T2I default when the field still carries the merged video default.

        The executor merges a single ``default_generation_params`` dict (the
        video params) into the request before ``infer()``, so an unspecified
        field arrives equal to its video default.  For T2I we substitute the
        T2I default in that case while honoring any explicit user override.
        """
        return t2i_default if merged_value == video_default else merged_value

    def infer(self, req):
        extra_params = req.params.extra_params or {}
        output_type = extra_params.get("output_type", "video")
        is_t2i = str(output_type).lower() == "image"

        height = req.params.height
        width = req.params.width
        num_inference_steps = req.params.num_inference_steps
        guidance_scale = req.params.guidance_scale
        if is_t2i:
            height = self._resolve_t2i_default(
                height, COSMOS3_720P_PARAMS["height"], COSMOS3_T2I_PARAMS["height"]
            )
            width = self._resolve_t2i_default(
                width, COSMOS3_720P_PARAMS["width"], COSMOS3_T2I_PARAMS["width"]
            )
            num_inference_steps = self._resolve_t2i_default(
                num_inference_steps,
                COSMOS3_720P_PARAMS["num_inference_steps"],
                COSMOS3_T2I_PARAMS["num_inference_steps"],
            )
            guidance_scale = self._resolve_t2i_default(
                guidance_scale,
                COSMOS3_720P_PARAMS["guidance_scale"],
                COSMOS3_T2I_PARAMS["guidance_scale"],
            )

        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.params.negative_prompt,
            image=req.params.image,
            height=height,
            width=width,
            num_frames=req.params.num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=req.params.seed,
            max_sequence_length=req.params.max_sequence_length,
            frame_rate=req.params.frame_rate,
            use_duration_template=extra_params.get(
                "use_duration_template",
                COSMOS3_EXTRA_SPECS["use_duration_template"].default,
            ),
            use_resolution_template=extra_params.get(
                "use_resolution_template",
                COSMOS3_EXTRA_SPECS["use_resolution_template"].default,
            ),
            use_system_prompt=extra_params.get("use_system_prompt", False),
            use_guardrails=extra_params.get("use_guardrails", True),
            enable_audio=extra_params.get("enable_audio", False),
            output_type=output_type,
        )

    def _apply_metadata_templates(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        duration_template: Optional[str] = COSMOS3_DURATION_TEMPLATE,
        resolution_template: Optional[str] = COSMOS3_DEFAULT_RESOLUTION_TEMPLATE,
        force_duration_template: bool = False,
    ) -> str:
        """Append duration and resolution metadata to a plain-text prompt.

        ``duration_template`` / ``resolution_template`` of ``None`` disables that
        template.  JSON prompts are handled by ``_format_prompt_with_metadata``.
        """
        parts: List[str] = []
        head = prompt.rstrip(".").strip()
        if head:
            parts.append(head)
        if duration_template is not None and (num_frames > 1 or force_duration_template):
            duration = num_frames / frame_rate
            parts.append(duration_template.format(duration=duration, fps=frame_rate).rstrip("."))
        if resolution_template is not None:
            parts.append(resolution_template.format(height=height, width=width).rstrip("."))
        if not parts:
            return ""
        return ". ".join(parts) + "."

    def _format_prompt_with_metadata(
        self,
        prompt: str,
        *,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        duration_template: Optional[str],
        resolution_template: Optional[str],
        force_duration_template: bool = False,
    ) -> str:
        """Apply cosmos-framework-style metadata to plain text or JSON prompts."""
        stripped = prompt.strip()
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                data = None
            else:
                if isinstance(data, dict):
                    if duration_template is not None and (
                        num_frames > 1 or force_duration_template
                    ):
                        duration = num_frames / frame_rate
                        data["duration"] = f"{duration:.1f}s"
                        data["fps"] = (
                            int(frame_rate) if frame_rate == int(frame_rate) else frame_rate
                        )
                    if resolution_template is not None:
                        data["resolution"] = {"W": width, "H": height}
                        divisor = math.gcd(height, width)
                        data["aspect_ratio"] = f"{height // divisor},{width // divisor}"
                    return json.dumps(data, ensure_ascii=False)

        return self._apply_metadata_templates(
            prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            duration_template=duration_template,
            resolution_template=resolution_template,
            force_duration_template=force_duration_template,
        )

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
        self,
        text: str,
        max_sequence_length: int,
        use_system_prompt: bool = False,
        system_prompt: Optional[str] = None,
    ):
        """Tokenize a prompt using the Qwen2 chat template.

        Returns (input_ids, attention_mask) as [1, S] tensors on device.
        """
        conversations = (
            [{"role": "system", "content": system_prompt or COSMOS3_DEFAULT_SYSTEM_PROMPT}]
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

        video = self.vae.decode(latents, return_dict=False)[0]
        video = postprocess_video_tensor(video)
        return video

    # =========================================================================
    # Audio generation
    # =========================================================================

    def decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode audio latent tokens back to waveform.

        Args:
            latent: Audio latent tensor of shape (B, C, T).

        Returns:
            Waveform tensor of shape (B, audio_channels, N_samples).
        """
        return self.audio_tokenizer.decode(latent).float()  # [B, audio_channels, N_samples]

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
        enable_audio: bool = COSMOS3_EXTRA_SPECS["enable_audio"].default,
        output_type: str = COSMOS3_EXTRA_SPECS["output_type"].default,
    ):
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        use_guardrails = use_guardrails and not TRTLLM_DISABLE_COSMOS3_GUARDRAILS

        # Text-to-image mode: same checkpoint/forward path as T2V, but a single
        # latent frame, image-flavored prompt templates, flow_shift=3.0, a CFG
        # guidance interval, and an image (rather than video) output.
        is_t2i = str(output_type).lower() == "image"
        guidance_interval = None
        if is_t2i:
            if image is not None:
                raise ValueError(
                    "Cosmos3 text-to-image (output_type='image') does not accept an image input."
                )
            num_frames = 1
            enable_audio = False
            guidance_interval = COSMOS3_T2I_PARAMS["guidance_interval"]
            self._set_flow_shift(COSMOS3_T2I_PARAMS["flow_shift"])
        else:
            # Restore the checkpoint flow_shift in case a prior T2I request
            # rebuilt the scheduler with shift=3.0.
            self._set_flow_shift(getattr(self, "_engine_init_flow_shift", 1.0))

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
        # None negative_prompt means the empty default will be used (safe); skip it.
        text_blocked = torch.zeros((), device=self.device, dtype=torch.int32)
        if self.rank == 0 and use_guardrails and self.safety_checker is not None:
            prompts_to_check = list(prompt)
            if negative_prompt is not None:
                prompts_to_check.append(negative_prompt)
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

        # Positive prompt: forward duration/resolution templates.  T2I has no
        # duration concept (single image) and uses the image-flavored
        # resolution template.
        use_duration_template = use_duration_template and not is_t2i
        dur_tmpl = COSMOS3_DURATION_TEMPLATE if use_duration_template else None
        if use_resolution_template:
            res_tmpl = (
                COSMOS3_IMAGE_RESOLUTION_TEMPLATE if is_t2i else COSMOS3_DEFAULT_RESOLUTION_TEMPLATE
            )
        else:
            res_tmpl = None

        # Negative prompt: mirror positive metadata (cosmos-framework CLI default
        # when ``negative_prompt_keep_metadata`` promotes mode to ``same``).
        negative_prompt = self._format_prompt_with_metadata(
            negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            duration_template=dur_tmpl,
            resolution_template=res_tmpl,
            force_duration_template=False,
        )

        prompt = [
            self._format_prompt_with_metadata(
                p,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                duration_template=dur_tmpl,
                resolution_template=res_tmpl,
            )
            for p in prompt
        ]
        logger.info(f"Prompt with metadata: '{prompt}'")

        prompt = prompt[0]

        # 1. Tokenize prompts (no separate text encoder — transformer embeds internally)
        logger.info("Tokenizing prompts...")
        system_prompt = COSMOS3_T2I_SYSTEM_PROMPT if is_t2i else COSMOS3_DEFAULT_SYSTEM_PROMPT
        cond_ids, cond_mask = self._tokenize_prompt(
            prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )

        # 2. Prepare latents
        if image is not None:
            if isinstance(image, str):
                image = load_image(image, format="pil")

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

        # 3b. Audio noise init — latent length matches diffusers Cosmos3OmniPipeline.prepare_latents.
        do_audio = enable_audio and self.audio_gen and hasattr(self, "audio_tokenizer")
        audio_latents = None
        if do_audio:
            audio_cfg = self.audio_tokenizer.model_config
            n_audio_samples = int(num_frames / frame_rate * audio_cfg["sampling_rate"])
            hop_size = math.prod(audio_cfg["dec_strides"])
            T_audio = (n_audio_samples + hop_size - 1) // hop_size
            audio_latents = randn_tensor(
                (1, self.transformer.audio_dim, T_audio),
                generator=generator,
                device=self.device,
                dtype=latents.dtype,
            )
            # Audio uses the same scheduler type/config as video.
            self.audio_scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 4. Build forward_fn for the denoise loop
        def forward_fn(
            latent_input,
            extra_stream_latents,
            step_index,
            timestep,
            encoder_hidden_states,
            extra_tensors,
        ):
            """Cosmos3 forward function for BasePipeline.denoise().

            Since Cosmos3 embeds text internally, we pass token IDs via extra_tensors
            rather than through encoder_hidden_states.
            """
            current_audio = extra_stream_latents.get("audio") if extra_stream_latents else None

            result = self.transformer(
                hidden_states=latent_input,
                timestep=timestep / self.scheduler.config.num_train_timesteps,
                raw_timestep=timestep,
                text_ids=extra_tensors["text_ids"],
                text_mask=extra_tensors["text_mask"],
                video_shape=video_shape,
                fps=frame_rate,
                noisy_frame_mask=velocity_mask,
                audio_latents=current_audio,
            )

            video_noise_pred = result.video
            audio_noise_pred = result.audio

            if velocity_mask is not None:
                video_noise_pred = video_noise_pred * velocity_mask

            if audio_noise_pred is not None:
                return video_noise_pred, {"audio": audio_noise_pred}
            return video_noise_pred

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
        extra_streams = {"audio": (audio_latents, self.audio_scheduler)} if do_audio else None
        denoise_result = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=cond_ids,  # placeholder — actual conditioning via extra_cfg_tensors
            neg_prompt_embeds=uncond_ids,
            guidance_scale=guidance_scale,
            forward_fn=forward_fn,
            extra_cfg_tensors=extra_cfg_tensors,
            extra_streams=extra_streams,
            guidance_interval=guidance_interval,
        )

        if extra_streams is not None:
            latents, extra_latents = denoise_result
            audio_latents = extra_latents.get("audio")
        else:
            latents = denoise_result
            audio_latents = None

        timer.mark_post_start()

        # 7. Decode video
        logger.info("Decoding video...")
        decode_start = time.time()

        if image_latent is not None:
            latents = latents.clone()
            latents[:, :, 0:1, :, :] = image_latent.to(device=latents.device, dtype=latents.dtype)

        video = self.decode_latents(latents, self._decode_latents)

        # 7b. Decode audio
        waveform = None
        if do_audio and audio_latents is not None:
            logger.info("Decoding audio...")
            waveform = self.decode_audio(audio_latents)  # [B, audio_channels, N_samples]

        # Video guardrail
        if self.rank == 0:
            logger.info(f"Video decoded in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

            if use_guardrails and self.safety_checker is not None:
                video = check_video_safety(video, self.safety_checker)

        timer.mark_end()

        if is_t2i:
            # Collapse the single decoded frame [B, T=1, H, W, C] -> [B, H, W, C].
            image = video[:, 0] if video is not None else None
            return timer.fill(PipelineOutput(image=image))

        return timer.fill(
            PipelineOutput(
                video=video,
                frame_rate=frame_rate,
                audio=waveform,
                audio_sample_rate=self.audio_tokenizer.model_config["sampling_rate"]
                if waveform is not None
                else None,
            )
        )
