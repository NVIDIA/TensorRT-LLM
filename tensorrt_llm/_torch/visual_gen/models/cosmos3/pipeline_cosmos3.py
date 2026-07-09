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
from collections.abc import Mapping
from typing import Any, Iterable, List, Optional, Union

import PIL.Image
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import Qwen2Tokenizer

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    def tqdm(iterable, **kwargs):
        return iterable

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.logger import logger

from .transfer import (
    Cosmos3TransferConfig,
    load_or_compute_control_frames,
    media_hw,
    media_to_uint8_cthw,
    pad_temporal_frames,
    resize_center_crop_uint8_cthw,
    resolve_transfer_config,
    uint8_cthw_to_normalized_5d,
)
from .defaults import (
    COSMOS3_720P_PARAMS,
    COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP,
    COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES,
    COSMOS3_EXTRA_SPECS,
    COSMOS3_PIPELINE_DEFAULTS,
    COSMOS3_T2I_PARAMS,
    COSMOS3_V2V_DEFAULT_FLOW_SHIFT,
)
from .guardrails import check_video_safety, download_guardrail_checkpoint
from .sound_tokenizer import LatentAutoEncoderV2
from .transformer_cosmos3 import Cosmos3VFMTransformer
from .utils import normalize_video_input, pil_to_rgb

COSMOS3_DEFAULT_NEGATIVE_PROMPT = ""
# NOTE: Intentional typo in "give" instead of "given" to match training setup.
COSMOS3_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate videos from a give prompt."
)
COSMOS3_T2I_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate images from a give prompt."
)
COSMOS3_DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
COSMOS3_DEFAULT_RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
COSMOS3_IMAGE_RESOLUTION_TEMPLATE = "This image is of {height}x{width} resolution."

TRTLLM_DISABLE_COSMOS3_GUARDRAILS = os.environ.get("TRTLLM_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"


def _normalize_condition_video_latent_indexes(
    indexes: Iterable[int] | int | str | None,
) -> tuple[int, ...]:
    if indexes is None:
        return COSMOS3_DEFAULT_CONDITION_VIDEO_LATENT_INDEXES
    if isinstance(indexes, int):
        normalized = (indexes,)
    elif isinstance(indexes, str):
        parts = [part.strip() for part in indexes.split(",") if part.strip()]
        normalized = tuple(int(part) for part in parts)
    else:
        normalized = tuple(int(index) for index in indexes)

    if not normalized:
        raise ValueError("Cosmos3 condition_video_latent_indexes must not be empty.")
    if any(index < 0 for index in normalized):
        raise ValueError(
            f"Cosmos3 condition_video_latent_indexes must be non-negative, got {normalized}."
        )
    return normalized


def _condition_pixel_frame_count(
    condition_video_latent_indexes: Iterable[int],
    temporal_compression: int,
) -> int:
    return max(condition_video_latent_indexes) * int(temporal_compression) + 1


def _normalize_condition_video_keep(keep: str | None) -> str:
    normalized = str(keep or COSMOS3_DEFAULT_CONDITION_VIDEO_KEEP).strip().lower()
    if normalized not in {"first", "last"}:
        raise ValueError("Cosmos3 condition_video_keep must be either first or last.")
    return normalized


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
        if getattr(
            primary_pretrained_config,
            "audio_gen",
            getattr(primary_pretrained_config, "sound_gen", False),
        ):
            logger.info("Initializing Cosmos3OmniMoTPipeline with audio generation.")
            self.audio_gen = True

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
            # needed.
            self._base_scheduler_config = self.scheduler.config
            self._engine_init_flow_shift = float(
                getattr(self.scheduler.config, "flow_shift", 1.0) or 1.0
            )
            self._current_flow_shift = self._engine_init_flow_shift
            self._base_scheduler_use_karras_sigmas = self._scheduler_use_karras_sigmas(
                self.scheduler.config
            )
            self._current_scheduler_use_karras_sigmas = self._base_scheduler_use_karras_sigmas
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

    @staticmethod
    def _scheduler_use_karras_sigmas(config: Any) -> Optional[bool]:
        value = getattr(config, "use_karras_sigmas", None)
        return None if value is None else bool(value)

    def _set_flow_shift(
        self, target_shift: float, *, use_karras_sigmas: Optional[bool] = None
    ) -> None:
        """Rebuild the UniPC scheduler when request scheduler defaults change.

        The effective flow-shift changes when switching between mode defaults
        (T2I=3.0, V2V=10.0, T2V/I2V=checkpoint default) or when a
        request provides ``flow_shift``. V2V also forces Karras sigmas off.
        """
        if not hasattr(self, "_base_scheduler_config"):
            return
        target = float(target_shift)
        target_use_karras_sigmas = (
            self._base_scheduler_use_karras_sigmas
            if use_karras_sigmas is None
            else bool(use_karras_sigmas)
        )
        if (
            target == float(self._current_flow_shift)
            and target_use_karras_sigmas == self._current_scheduler_use_karras_sigmas
        ):
            return

        scheduler_kwargs = {"flow_shift": target}
        if use_karras_sigmas is not None:
            scheduler_kwargs["use_karras_sigmas"] = bool(use_karras_sigmas)
        self.scheduler = UniPCMultistepScheduler.from_config(
            self._base_scheduler_config, **scheduler_kwargs
        )
        if self.audio_gen:
            self.audio_scheduler = UniPCMultistepScheduler.from_config(
                self._base_scheduler_config, **scheduler_kwargs
            )
        self._current_flow_shift = target
        self._current_scheduler_use_karras_sigmas = self._scheduler_use_karras_sigmas(
            self.scheduler.config
        )

    @property
    def default_warmup_resolutions(self):
        return [(720, 1280)]

    @property
    def default_warmup_num_frames(self):
        return [189]

    @property
    def default_generation_params(self):
        return dict(COSMOS3_PIPELINE_DEFAULTS)

    @property
    def extra_param_specs(self):
        return dict(COSMOS3_EXTRA_SPECS)

    def _transfer_bucket_size(
        self,
        transfer_config: Cosmos3TransferConfig,
        source_hw: tuple[int, int] | None,
    ) -> tuple[int, int]:
        resolution = transfer_config.resolution if transfer_config.resolution is not None else 720
        source_h, source_w = source_hw or (
            COSMOS3_720P_PARAMS["height"],
            COSMOS3_720P_PARAMS["width"],
        )
        target_w, target_h = find_closest_target_size(int(source_h), int(source_w), resolution)
        return int(target_h), int(target_w)

    @staticmethod
    def _video_payload_value(video: Any, key: str) -> Any:
        if isinstance(video, Mapping):
            return video.get(key)
        return getattr(video, key, None)

    @classmethod
    def _video_payload_fps(cls, video: Any) -> Optional[float]:
        for key in ("fps", "frame_rate", "source_fps", "input_fps", "avg_fps", "average_fps"):
            fps = cls.positive_float(cls._video_payload_value(video, key))
            if fps is not None:
                return fps
        for key in ("metadata", "info"):
            metadata = cls._video_payload_value(video, key)
            if metadata is None or metadata is video:
                continue
            fps = cls._video_payload_fps(metadata)
            if fps is not None:
                return fps
        if isinstance(video, Mapping):
            for key in ("frames", "data", "video"):
                nested = video.get(key)
                if nested is None or nested is video:
                    continue
                fps = cls._video_payload_fps(nested)
                if fps is not None:
                    return fps
        if isinstance(video, (str, os.PathLike)):
            try:
                import imageio.v3 as iio

                metadata = iio.immeta(os.fspath(video))
            except Exception:
                return None
            if metadata is not video:
                return cls._video_payload_fps(metadata)
        return None

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

    def infer(self, req):
        extra_params = req.params.extra_params or {}
        output_type = extra_params.get("output_type", "video")
        transfer_config = resolve_transfer_config(extra_params, req.params, req.prompt)

        # The V2V reference rides in ``multi_modal_data["video"]`` as a
        # ``VideoData`` (framework convention); the worker crops + VAE-encodes its
        # frames. Both producers (offline and serve) build it, so there is no
        # legacy-path fallback.
        mm_data = req.params.multi_modal_data or {}
        video_data = mm_data.get("video")
        video = video_data.frames if video_data is not None else None

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
            use_duration_template=extra_params.get(
                "use_duration_template",
                COSMOS3_EXTRA_SPECS["use_duration_template"].default,
            ),
            use_resolution_template=extra_params.get(
                "use_resolution_template",
                COSMOS3_EXTRA_SPECS["use_resolution_template"].default,
            ),
            use_system_prompt=extra_params.get(
                "use_system_prompt", COSMOS3_EXTRA_SPECS["use_system_prompt"].default
            ),
            use_guardrails=extra_params.get("use_guardrails", True),
            enable_audio=extra_params.get("enable_audio", False),
            output_type=output_type,
            video=video,
            condition_video_latent_indexes=extra_params.get("condition_video_latent_indexes"),
            condition_video_keep=extra_params.get("condition_video_keep"),
            flow_shift=extra_params.get("flow_shift"),
            transfer_config=transfer_config
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

        # Pad to max_sequence_length. TRT's shared denoiser concatenates CFG
        # prompt tensors, so cond/uncond sequence lengths must match.
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

    # =========================================================================
    # I2V latent preparation
    # =========================================================================

    def _encode_conditioning_video(
        self,
        image_tensor: torch.Tensor,
        num_frames: int,
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

        Returns:
            [1, C, T_latent, H_latent, W_latent] normalized latent of the
            full conditioning video.
        """
        # Build pixel-space video: repeat the conditioning image across all frames
        # image_tensor: [1, 3, H, W] -> [1, 3, 1, H, W] -> [1, 3, num_frames, H, W]
        video = image_tensor.unsqueeze(2).expand(-1, -1, num_frames, -1, -1).contiguous()
        return self._encode_video_tensor(video)

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

    def _decode_latents_raw(self, latents):
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

        return self.vae.decode(latents, return_dict=False)[0]

    @nvtx_range("_decode_latents", color="blue")
    def _decode_latents(self, latents):
        return postprocess_video_tensor(self._decode_latents_raw(latents))

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

    def _preprocess_condition_video(
        self, frames: List[Any], target_h: int, target_w: int
    ) -> torch.Tensor:
        if not frames:
            raise ValueError("Cosmos3 condition video input must contain at least one frame.")
        processed = [
            self.video_processor.preprocess(
                self._resize_and_center_crop_image(pil_to_rgb(frame), target_h, target_w),
                height=target_h,
                width=target_w,
            ).squeeze(0)
            for frame in frames
        ]
        return torch.stack(processed, dim=1).unsqueeze(0).contiguous()

    def _encode_video_tensor(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """VAE-encode a preprocessed pixel video [1, 3, T, H, W]."""
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)
        if video_tensor.ndim != 5 or video_tensor.shape[0] != 1 or video_tensor.shape[1] != 3:
            raise ValueError(
                f"Cosmos3 video tensor must have shape [1, 3, T, H, W], got {tuple(video_tensor.shape)}."
            )

        video = video_tensor.to(device=self.device, dtype=self.vae.dtype)
        latent = self.vae.encode(video).latent_dist.mode()

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

    # =========================================================================
    # Video to video
    # =========================================================================

    def _prepare_latents_v2v(
        self,
        video_tensor: torch.Tensor,
        num_frames: int,
        generator: torch.Generator,
        condition_video_latent_indexes: Iterable[int] | int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare V2V latents with explicit clean conditioned latent frames."""
        if video_tensor.ndim == 4:
            video_tensor = video_tensor.unsqueeze(0)
        if video_tensor.ndim != 5 or video_tensor.shape[0] != 1 or video_tensor.shape[1] != 3:
            raise ValueError(
                "Cosmos3 video tensor must have shape [1, 3, T, H, W], "
                f"got {tuple(video_tensor.shape)}."
            )
        if video_tensor.shape[2] < 1:
            raise ValueError("Cosmos3 V2V video tensor must contain at least one frame.")

        C = self.transformer.latent_channel_size
        T_lat = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        H_lat = video_tensor.shape[-2] // self.vae_scale_factor_spatial
        W_lat = video_tensor.shape[-1] // self.vae_scale_factor_spatial
        indexes = _normalize_condition_video_latent_indexes(condition_video_latent_indexes)
        out_of_range = [index for index in indexes if index >= T_lat]
        if out_of_range:
            raise ValueError(
                "Cosmos3 condition_video_latent_indexes contains indexes outside the latent video: "
                f"indexes={indexes}, latent_frames={T_lat}."
            )

        noise = randn_tensor(
            (1, C, T_lat, H_lat, W_lat),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        condition_pixel_frames = _condition_pixel_frame_count(
            indexes, self.vae_scale_factor_temporal
        )
        condition_video = video_tensor[:, :, :condition_pixel_frames]
        if condition_video.shape[2] < condition_pixel_frames:
            pad = condition_video[:, :, -1:].repeat(
                1, 1, condition_pixel_frames - condition_video.shape[2], 1, 1
            )
            condition_video = torch.cat([condition_video, pad], dim=2)

        cond_latent = self._encode_video_tensor(condition_video)
        expected_prefix = (1, C, max(indexes) + 1, H_lat, W_lat)
        if (
            cond_latent.shape[0] != expected_prefix[0]
            or cond_latent.shape[1] != expected_prefix[1]
            or cond_latent.shape[2] < expected_prefix[2]
            or cond_latent.shape[3:] != expected_prefix[3:]
        ):
            raise ValueError(
                "Cosmos3 V2V condition latent shape mismatch: "
                f"encoded={tuple(cond_latent.shape)}, expected at least {expected_prefix}."
            )

        condition_mask = torch.zeros(1, 1, T_lat, 1, 1, device=self.device, dtype=self.dtype)
        condition_latents = torch.zeros_like(noise)
        for index in indexes:
            condition_mask[:, :, index, :, :] = 1.0
            condition_latents[:, :, index : index + 1] = cond_latent[:, :, index : index + 1]
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, condition_latents

    # =========================================================================
    # Transfer
    # =========================================================================

    def _prepare_transfer_latents(
        self,
        target_video: torch.Tensor,
        current_conditional_frames: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_latents = self._encode_video_tensor(target_video)
        noise = randn_tensor(
            condition_latents.shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros(
            1,
            1,
            condition_latents.shape[2],
            1,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        if current_conditional_frames > 0:
            latent_frames = (current_conditional_frames - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[:, :, :latent_frames] = 1.0
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        velocity_mask = 1.0 - condition_mask
        return latents, velocity_mask, condition_mask * condition_latents

    # =========================================================================
    # Forward (main generation entry point)
    # =========================================================================

    @nvtx_range("Cosmos3OmniMoTPipeline.forward")
    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        image: Optional[Union[PIL.Image.Image, torch.Tensor, str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 42,
        max_sequence_length: Optional[int] = None,
        frame_rate: Optional[float] = None,
        use_duration_template: bool = COSMOS3_EXTRA_SPECS["use_duration_template"].default,
        use_resolution_template: bool = COSMOS3_EXTRA_SPECS["use_resolution_template"].default,
        use_system_prompt: Optional[bool] = COSMOS3_EXTRA_SPECS["use_system_prompt"].default,
        use_guardrails: bool = COSMOS3_EXTRA_SPECS["use_guardrails"].default,
        enable_audio: bool = COSMOS3_EXTRA_SPECS["enable_audio"].default,
        output_type: str = COSMOS3_EXTRA_SPECS["output_type"].default,
        video: Any = None,
        condition_video_latent_indexes: Any = None,
        condition_video_keep: Any = None,
        flow_shift: Optional[float] = None,
        transfer_config: Optional[Cosmos3TransferConfig] = None
    ):
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        use_guardrails = use_guardrails and not TRTLLM_DISABLE_COSMOS3_GUARDRAILS

        # Text-to-image mode: same checkpoint/forward path as T2V, but a single
        # latent frame, image-flavored prompt templates, flow_shift=3.0, a CFG
        # guidance interval, and an image (rather than video) output.
        is_t2i = str(output_type).lower() == "image"
        if image is not None and video is not None:
            raise ValueError(
                "Cosmos3 generation supports text-only, text + image, "
                "or text + video input, but not both image and video."
            )
        if is_t2i and video is not None:
            raise ValueError(
                "Cosmos3 video-to-video generation is supported only for video outputs."
            )
        is_v2v = video is not None and not is_t2i
        if use_system_prompt is None:
            use_system_prompt = is_v2v
        else:
            use_system_prompt = bool(use_system_prompt)
        if transfer_config is not None:
            if is_t2i:
                raise ValueError("Cosmos3 transfer inference is supported only for video outputs.")
            if do_action:
                raise ValueError("Cosmos3 transfer inference cannot be combined with action generation.")
            if enable_audio:
                raise ValueError("Cosmos3 transfer inference cannot be combined with sound generation.")
        guidance_interval = None
        if is_t2i:
            if image is not None:
                raise ValueError(
                    "Cosmos3 text-to-image (output_type='image') does not accept an image input."
                )
            if enable_audio:
                raise ValueError("Cosmos3 audio generation does not support output_type='image'.")
            num_frames = 1
            height = height or COSMOS3_T2I_PARAMS["height"]
            width = width or COSMOS3_T2I_PARAMS["width"]
            num_inference_steps = num_inference_steps or COSMOS3_T2I_PARAMS["num_inference_steps"]
            if guidance_scale is None:
                guidance_scale = COSMOS3_T2I_PARAMS["guidance_scale"]
            guidance_interval = COSMOS3_T2I_PARAMS["guidance_interval"]
            self._set_flow_shift(
                flow_shift if flow_shift is not None else COSMOS3_T2I_PARAMS["flow_shift"]
            )
        else:
            height = height or COSMOS3_720P_PARAMS["height"]
            width = width or COSMOS3_720P_PARAMS["width"]
            num_frames = num_frames or COSMOS3_720P_PARAMS["num_frames"]
            num_inference_steps = num_inference_steps or COSMOS3_720P_PARAMS["num_inference_steps"]
            if guidance_scale is None:
                guidance_scale = COSMOS3_720P_PARAMS["guidance_scale"]
            if is_v2v:
                self._set_flow_shift(
                    flow_shift if flow_shift is not None else COSMOS3_V2V_DEFAULT_FLOW_SHIFT,
                    use_karras_sigmas=False,
                )
            else:
                # Restore the checkpoint flow_shift in case a prior T2I/V2V
                # request rebuilt the scheduler with a mode-specific shift.
                self._set_flow_shift(
                    flow_shift
                    if flow_shift is not None
                    else getattr(self, "_engine_init_flow_shift", 1.0)
                )

        max_sequence_length = max_sequence_length or COSMOS3_720P_PARAMS["max_sequence_length"]
        if frame_rate is None:
            frame_rate = COSMOS3_720P_PARAMS["frame_rate"]

        if self.rank == 0:
            logger.info(
                f"Cosmos3 generation dims: {width}x{height} (WxH), num_frames={num_frames}, "
                f"num_inference_steps={num_inference_steps}, guidance_scale={guidance_scale:.2f}, "
                f"frame_rate={frame_rate:.1f}"
            )

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

        if transfer_config is not None:
            return self._forward_transfer(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                max_frames=transfer_config.max_frames,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                use_system_prompt=use_system_prompt,
                use_duration_template=False,
                use_resolution_template=False,
                seed=seed,
                transfer_config=transfer_config,
                video=video,
            )

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
        condition_latents = None
        image_latent = None
        velocity_mask = None

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
        elif video is not None:
            condition_video_latent_indexes = _normalize_condition_video_latent_indexes(
                condition_video_latent_indexes
            )
            condition_video_keep = _normalize_condition_video_keep(condition_video_keep)
            condition_pixel_frames = min(
                _condition_pixel_frame_count(
                    condition_video_latent_indexes, self.vae_scale_factor_temporal
                ),
                num_frames,
            )
            # ``video`` is the reference frames (a list, from the producer's
            # ``VideoData``). The worker crops the first/last conditioning window
            # uniformly — producers never crop, so behavior never depends on them.
            video = normalize_video_input(
                video,
                max_frames=None if condition_video_keep == "last" else condition_pixel_frames,
            )
            video = (
                video[-condition_pixel_frames:]
                if condition_video_keep == "last"
                else video[:condition_pixel_frames]
            )
            video = self._preprocess_condition_video(video, height, width)

            if self.rank == 0:
                logger.info(
                    f"Cosmos3 V2V conditioning: frames={video.shape[2]}, "
                    f"latent_indexes={condition_video_latent_indexes}"
                )
            latents, velocity_mask, condition_latents = self._prepare_latents_v2v(
                video,
                num_frames=num_frames,
                generator=generator,
                condition_video_latent_indexes=condition_video_latent_indexes,
            )
        else:
            latents = self._prepare_latents(height, width, num_frames, generator)

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

        def post_step_fn(step_latents):
            if velocity_mask is not None and condition_latents is not None:
                step_latents = (
                    velocity_mask * step_latents + (1.0 - velocity_mask) * condition_latents
                )
            elif velocity_mask is not None and image_latent is not None:
                step_latents = step_latents.clone()
                step_latents[:, :, 0:1, :, :] = image_latent.to(
                    device=step_latents.device, dtype=step_latents.dtype
                )
            return step_latents

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
        extra_streams = None
        if do_audio:
            extra_streams = {"audio": (audio_latents, self.audio_scheduler)}
        should_pin_condition_latents = condition_latents is not None or image_latent is not None
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
            post_step_fn=post_step_fn if should_pin_condition_latents else None,
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

    @staticmethod
    def _first_transfer_control_hw(transfer_config: Cosmos3TransferConfig) -> tuple[int, int] | None:
        for hint in transfer_config.ordered_hints:
            if hint.control is not None:
                detected = media_hw(hint.control)
                if detected is not None:
                    return detected
            if hint.control_path is not None:
                detected = media_hw(hint.control_path)
                if detected is not None:
                    return detected
        return None
    
    @staticmethod
    def _get_transfer_num_chunks(
        total_frames: int,
        frames_per_chunk: int,
        conditional_frames: int,
    ) -> tuple[int, int]:
        if frames_per_chunk <= 0:
            raise ValueError("Cosmos3 transfer frames_per_chunk must be positive.")
        if total_frames <= frames_per_chunk:
            return 1, frames_per_chunk
        stride = frames_per_chunk - conditional_frames
        if stride <= 0:
            raise ValueError("Cosmos3 transfer num_conditional_frames must be smaller than num_video_frames_per_chunk.")
        remaining = total_frames - frames_per_chunk
        extra_chunks = remaining // stride + (1 if remaining % stride else 0)
        return 1 + extra_chunks, stride

    @staticmethod
    def positive_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if value <= 0:
            return None
        return value

    @staticmethod
    def _transfer_active_at(
        timestep: torch.Tensor,
        interval: tuple[float, float] | None,
    ) -> bool:
        if interval is None:
            return True
        t_scalar = float(timestep.item()) if torch.is_tensor(timestep) else float(timestep)
        lo, hi = interval
        return float(lo) <= t_scalar <= float(hi)

    @staticmethod
    def _combine_transfer_predictions(
        *,
        cond_full: torch.Tensor,
        cond_no_control: torch.Tensor | None,
        uncond_full: torch.Tensor | None,
        guidance_scale: float,
        control_guidance: float,
    ) -> torch.Tensor:
        needs_control_cfg = cond_no_control is not None and control_guidance != 1.0
        needs_text_cfg = uncond_full is not None and guidance_scale > 1.0

        if needs_control_cfg and needs_text_cfg:
            control_cond = cond_no_control + control_guidance * (cond_full - cond_no_control)
            return uncond_full + guidance_scale * (control_cond - uncond_full)
        if needs_control_cfg:
            return cond_no_control + control_guidance * (cond_full - cond_no_control)
        if needs_text_cfg:
            return uncond_full + guidance_scale * (cond_full - uncond_full)
        return cond_full

    def diffuse_transfer(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        cond_ids: torch.Tensor,
        cond_mask: torch.Tensor,
        uncond_ids: torch.Tensor,
        uncond_mask: torch.Tensor,
        guidance_scale: float,
        control_guidance: float,
        control_guidance_interval: tuple[float, float] | None,
        control_latents: list[torch.Tensor],
        shared_kwargs: dict[str, Any],
        velocity_mask: torch.Tensor,
        condition_latents: torch.Tensor,
        guidance_interval: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        """Run Cosmos3 transfer denoising with sequential control/text CFG branches."""

        branch_caches: dict[str, tuple[Any, Any]] = {}

        def run_branch(
            cache_key: str,
            *,
            text_ids: torch.Tensor,
            text_mask: torch.Tensor,
            branch_control_latents: list[torch.Tensor] | None,
            timestep: torch.Tensor,
        ) -> torch.Tensor:
            self.transformer.cached_kv, self.transformer.cached_freqs_gen = branch_caches.get(
                cache_key,
                (None, None),
            )
            result = self.transformer(
                hidden_states=latents,
                timestep=timestep / self.scheduler.config.num_train_timesteps,
                raw_timestep=timestep,
                text_ids=text_ids,
                text_mask=text_mask,
                control_latents=branch_control_latents,
                **shared_kwargs,
            )
            branch_caches[cache_key] = (
                self.transformer.cached_kv,
                self.transformer.cached_freqs_gen,
            )
            if result.video is None:
                raise ValueError("Cosmos3 transfer diffusion expects video predictions.")
            return result.video

        self.transformer.reset_cache()
        try:
            transfer_steps = tqdm(
                timesteps,
                total=len(timesteps),
                desc="Transfer denoising",
                disable=self.rank != 0,
                dynamic_ncols=True,
            )
            for t in transfer_steps:
                timestep = t.expand(latents.shape[0])
                step_guidance = (
                    float(guidance_scale)
                    if self._transfer_active_at(t, guidance_interval)
                    else 1.0
                )
                step_control = (
                    float(control_guidance)
                    if self._transfer_active_at(t, control_guidance_interval)
                    else 1.0
                )

                cond_full = run_branch(
                    "transfer_cond_full",
                    text_ids=cond_ids,
                    text_mask=cond_mask,
                    branch_control_latents=control_latents,
                    timestep=timestep,
                )
                cond_no_control = None
                if step_control != 1.0:
                    cond_no_control = run_branch(
                        "transfer_cond_no_control",
                        text_ids=cond_ids,
                        text_mask=cond_mask,
                        branch_control_latents=None,
                        timestep=timestep,
                    )

                uncond_full = None
                if step_guidance > 1.0:
                    uncond_full = run_branch(
                        "transfer_uncond_full",
                        text_ids=uncond_ids,
                        text_mask=uncond_mask,
                        branch_control_latents=control_latents,
                        timestep=timestep,
                    )

                noise_pred = self._combine_transfer_predictions(
                    cond_full=cond_full,
                    cond_no_control=cond_no_control,
                    uncond_full=uncond_full,
                    guidance_scale=step_guidance,
                    control_guidance=step_control,
                )
                noise_pred = noise_pred * velocity_mask
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents = velocity_mask * latents + (1.0 - velocity_mask) * condition_latents
        finally:
            self.transformer.reset_cache()

        return latents

    def _forward_transfer(self,
                          *,
                          prompt: str,
                          negative_prompt: str,
                          height: int,
                          width: int,
                          max_frames: int,
                          num_inference_steps: int,
                          max_sequence_length,
                          use_system_prompt,
                          use_duration_template: str,
                          use_resolution_template: str,
                          seed,
                          transfer_config: Cosmos3TransferConfig,
                          video: Any
    ) -> PipelineOutput:
        input_frames = None
        transfer_input_fps = self._video_payload_fps(video) if video is not None else None
        source_hw = media_hw(video) if video is not None else self._first_transfer_control_hw(transfer_config)
        height, width = self._transfer_bucket_size(transfer_config, source_hw)
        if self.rank == 0:
            logger.info(
                "Cosmos3 transfer bucket: "
                f"source_hw={source_hw}, resolution={transfer_config.resolution or 720}, "
                f"target={width}x{height} (WxH), transfer_input_fps={transfer_input_fps}"
            )
        if video is not None:
            input_frames = media_to_uint8_cthw(
                video,
                height=height,
                width=width,
                max_frames=max_frames,
            )
            input_frames = input_frames[:, : transfer_config.max_frames]

        per_hint_frames: dict[str, torch.Tensor] = {}
        for hint in transfer_config.ordered_hints:
            frames = load_or_compute_control_frames(
                hint,
                height=height,
                width=width,
                max_frames=transfer_config.max_frames,
                input_frames=input_frames,
            )
            if frames.shape[1] < 1:
                raise ValueError(f"Cosmos3 transfer hint '{hint.key}' produced no frames.")
            per_hint_frames[hint.key] = frames
        if not per_hint_frames:
            raise ValueError("Cosmos3 transfer requires at least one control hint.")
        
        total_frames = next(iter(per_hint_frames.values())).shape[1]
        if transfer_config.num_frames is not None:
            total_frames = min(total_frames, int(transfer_config.num_frames))
        total_frames = max(1, total_frames)
        per_hint_frames = {key: pad_temporal_frames(frames, total_frames) for key, frames in per_hint_frames.items()}
        if input_frames is not None:
            input_frames = pad_temporal_frames(input_frames, total_frames)

        temporal_compression = self.vae_scale_factor_temporal
        chunk_frames = 1 if total_frames == 1 else transfer_config.num_video_frames_per_chunk
        chunk_frames = math.ceil((chunk_frames - 1) / temporal_compression) * temporal_compression + 1
        num_chunks, stride = self._get_transfer_num_chunks(
            total_frames,
            chunk_frames,
            transfer_config.num_conditional_frames,
        )
        padded_frames = max(total_frames, chunk_frames)
        per_hint_frames = {key: pad_temporal_frames(frames, padded_frames) for key, frames in per_hint_frames.items()}
        if input_frames is not None:
            input_frames = pad_temporal_frames(input_frames, padded_frames)

        configured_frame_rate = self.positive_float(transfer_config.fps)
        input_frame_rate = self.positive_float(transfer_input_fps)
        sampling_frame_rate = self.positive_float(transfer_config.fps)
        is_wsm_only = len(transfer_config.hints) == 1 and "wsm" in transfer_config.hints
        if is_wsm_only:
            frame_rate = configured_frame_rate or input_frame_rate or sampling_frame_rate or 24.0
        else:
            frame_rate = input_frame_rate or configured_frame_rate or sampling_frame_rate or 24.0
        num_inference_steps = num_inference_steps or COSMOS3_720P_PARAMS["num_inference_steps"]
        guidance_scale = (
            float(transfer_config.guidance_scale)
            if transfer_config.guidance_scale is not None
            else COSMOS3_720P_PARAMS["guidance_scale"]
        )
        flow_shift_target = float(
            transfer_config.flow_shift
            if transfer_config.flow_shift is not None
            else COSMOS3_V2V_DEFAULT_FLOW_SHIFT
        )
        max_sequence_length = (
            max_sequence_length
            or COSMOS3_720P_PARAMS["max_sequence_length"]
        )
        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps
        self._set_flow_shift(flow_shift_target, use_karras_sigmas=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if negative_prompt is None:
            negative_prompt = COSMOS3_DEFAULT_NEGATIVE_PROMPT

        # Transfer prompts are already upsampled by the benchmark/config path.
        # Keep them verbatim; duration/resolution templates would change parity.
        prompt = [prompt] if isinstance(prompt, str) else list(prompt)
        prompt = prompt[0]
        logger.info(f"Transfer prompt: '{prompt}'")

        # 1. Tokenize prompts (no separate text encoder — transformer embeds internally)
        logger.info("Tokenizing prompts...")
        system_prompt = COSMOS3_DEFAULT_SYSTEM_PROMPT
        cond_ids, cond_mask = self._tokenize_prompt(
            prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )
        uncond_ids, uncond_mask = self._tokenize_prompt(
            negative_prompt, max_sequence_length, use_system_prompt, system_prompt=system_prompt
        )

        output_chunks: list[torch.Tensor] = []
        control_chunks_per_hint: dict[str, list[torch.Tensor]] = {key: [] for key in per_hint_frames}
        previous_output: torch.Tensor | None = None


        for chunk_id in range(num_chunks):
            start_frame = chunk_id * stride
            end_frame = min(start_frame + chunk_frames, total_frames)
            control_norms = {
                key: uint8_cthw_to_normalized_5d(
                    pad_temporal_frames(frames[:, start_frame:end_frame], chunk_frames),
                    dtype=self.dtype,
                )
                for key, frames in per_hint_frames.items()
            }
            target_norm = torch.zeros_like(next(iter(control_norms.values())))
            current_conditional_frames = 0

            if chunk_id == 0 and transfer_config.num_first_chunk_conditional_frames > 0:
                if input_frames is None:
                    raise ValueError("Cosmos3 transfer num_first_chunk_conditional_frames > 0 requires a video input.")
                current_conditional_frames = min(
                    transfer_config.num_first_chunk_conditional_frames,
                    input_frames.shape[1],
                    chunk_frames,
                )
                if current_conditional_frames > 0:
                    input_cond = uint8_cthw_to_normalized_5d(
                        input_frames[:, :current_conditional_frames],
                        dtype=self.dtype,
                    )
                    target_norm[:, :, :current_conditional_frames] = input_cond
                    if current_conditional_frames < chunk_frames:
                        fill = target_norm[:, :, current_conditional_frames - 1 : current_conditional_frames]
                        target_norm[:, :, current_conditional_frames:] = fill.expand(
                            -1,
                            -1,
                            chunk_frames - current_conditional_frames,
                            -1,
                            -1,
                        )
            elif chunk_id > 0 and previous_output is not None:
                current_conditional_frames = min(
                    transfer_config.num_conditional_frames,
                    previous_output.shape[2],
                    chunk_frames,
                )
                if current_conditional_frames > 0:
                    target_norm[:, :, :current_conditional_frames] = previous_output[
                        :, :, -current_conditional_frames:
                    ].to(target_norm)
                    if current_conditional_frames < chunk_frames:
                        fill = target_norm[:, :, current_conditional_frames - 1 : current_conditional_frames]
                        target_norm[:, :, current_conditional_frames:] = fill.expand(
                            -1,
                            -1,
                            chunk_frames - current_conditional_frames,
                            -1,
                            -1,
                        )

            control_latents = [self._encode_video_tensor(video) for video in control_norms.values()]
            latents, velocity_mask, condition_latents = self._prepare_transfer_latents(
                target_norm,
                current_conditional_frames,
                generator,
            )
            video_shape = (latents.shape[2], latents.shape[3], latents.shape[4])
            shared_kwargs = dict(
                video_shape=video_shape,
                fps=frame_rate,
                noisy_frame_mask=velocity_mask,
                transfer_share_vision_temporal_positions=transfer_config.share_vision_temporal_positions,
            )

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            latents = self.diffuse_transfer(
                latents=latents,
                timesteps=self.scheduler.timesteps,
                cond_ids=cond_ids,
                cond_mask=cond_mask,
                uncond_ids=uncond_ids,
                uncond_mask=uncond_mask,
                guidance_scale=guidance_scale,
                control_guidance=transfer_config.control_guidance,
                control_guidance_interval=transfer_config.control_guidance_interval,
                control_latents=control_latents,
                shared_kwargs=shared_kwargs,
                velocity_mask=velocity_mask,
                condition_latents=condition_latents,
            )
            output_video = self._decode_latents_raw(latents).clamp(-1, 1)
            previous_output = output_video

            if chunk_id == 0:
                output_chunks.append(output_video)
                for key, control in control_norms.items():
                    control_chunks_per_hint[key].append(control)
            else:
                output_chunks.append(output_video[:, :, current_conditional_frames:])
                for key, control in control_norms.items():
                    control_chunks_per_hint[key].append(control[:, :, current_conditional_frames:])

        full_output = torch.cat(output_chunks, dim=2)[:, :, :total_frames]
        full_controls = {
            key: torch.cat(chunks, dim=2)[:, :, :total_frames] for key, chunks in control_chunks_per_hint.items()
        }

        if transfer_config.show_control_condition:
            all_controls = torch.cat([full_controls[key] for key in per_hint_frames], dim=-1)
            all_controls = all_controls.to(full_output)
            full_output = torch.cat([all_controls, full_output], dim=-1)
        if transfer_config.show_input and input_frames is not None:
            normalized_input = uint8_cthw_to_normalized_5d(input_frames[:, :total_frames], dtype=torch.float32)
            full_output = torch.cat([normalized_input.to(full_output), full_output], dim=-1)
        video = postprocess_video_tensor(full_output)
        return PipelineOutput(
            video=video,
            frame_rate=frame_rate,
        )
