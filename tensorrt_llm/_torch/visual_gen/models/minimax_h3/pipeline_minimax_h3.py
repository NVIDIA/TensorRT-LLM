# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.

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

"""TRT-LLM VisualGen pipeline for MiniMax-H3 FL2VA checkpoints."""

import time
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.logger import logger

from .autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from .autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from .modeling_utils import MiniMaxH3DiagonalGaussianDistribution, minimax_h3_randn_tensor
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    prepare_keyframe_image,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .scheduler import MiniMaxH3Scheduler
from .transformer_minimax_h3 import MiniMaxH3Transformer3DModel


def _component_skipped(
    skip_components: list[str | PipelineComponent],
    component: PipelineComponent,
) -> bool:
    return component in skip_components or component.value in skip_components


@register_pipeline(
    "MiniMaxH3ModularPipeline",
    hf_ids=["MiniMaxAI/MiniMax-H3"],
    download_patterns=[
        "modular_model_index.json",
        "LICENSE",
        "README.md",
        "transformer/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "vae/*",
        "audio_vae/*",
        "scheduler/*",
        "audio_scheduler/*",
    ],
    doc=(
        "MiniMax-H3 initial BF16 support for text-to-video-with-audio and "
        "first/last-frame FL2VA using the converted top-level checkpoint."
    ),
)
class MiniMaxH3Pipeline(BasePipeline):
    """Joint packed-sequence video and stereo-audio generation with MiniMax-H3.

    Args:
        pipeline_config: Internal VisualGen pipeline configuration resolved from
            the checkpoint and ``VisualGenArgs``.
    """

    VIDEO_SCHEDULER_SHIFT = 12.0
    AUDIO_SCHEDULER_SHIFT = 3.0
    derive_output_size_from_reference = True

    @staticmethod
    def _request_generator(seed: int) -> torch.Generator:
        """Create the CPU generator required by the released H3 RNG contract."""
        return torch.Generator().manual_seed(seed)

    def __init__(self, pipeline_config: DiffusionPipelineConfig) -> None:
        if pipeline_config.mapping.world_size != 1:
            raise NotImplementedError("MiniMax-H3 initial support is single-GPU only.")
        if pipeline_config.attention.backend != "VANILLA":
            raise NotImplementedError(
                "MiniMax-H3 initial support is quality-validated only with VANILLA attention."
            )
        if pipeline_config.cache is not None:
            raise NotImplementedError(
                "TeaCache and Cache-DiT are not quality-validated for MiniMax-H3."
            )
        if pipeline_config.cuda_graph.enable:
            raise NotImplementedError(
                "CUDA graphs are not yet supported for MiniMax-H3's packed layout inputs."
            )
        self.audio_vae = None
        self.audio_scheduler = None
        self.processor = None
        super().__init__(pipeline_config)

    @property
    def default_generation_params(self) -> dict:
        return {
            # MiniMax-H3 derives the default canvas from the first present
            # keyframe, or from 16:9 for T2VA.
            "height": None,
            "width": None,
            "num_frames": 124,
            "frame_rate": float(MINIMAX_H3_FPS),
            "num_inference_steps": 50,
        }

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        return {
            "last_image": ExtraParamSchema(
                type="str",
                default=None,
                description=(
                    "Optional last-frame path. It may be used alone or together "
                    "with image for MiniMax-H3 FL2VA generation."
                ),
            )
        }

    @property
    def default_warmup_resolutions(self) -> list[tuple[int, int]]:
        return [(768, 1344)]

    @property
    def default_warmup_num_frames(self) -> list[int]:
        return [124]

    @property
    def resolution_multiple_of(self) -> tuple[int, int]:
        return (MINIMAX_H3_CANVAS_MULTIPLE, MINIMAX_H3_CANVAS_MULTIPLE)

    def _run_warmup(
        self,
        height: int,
        width: int,
        num_frames: int,
        steps: int,
    ) -> None:
        self.forward(
            prompt="warmup",
            seed=42,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(MINIMAX_H3_FPS),
            num_inference_steps=steps,
        )

    def _init_transformer(self) -> None:
        self.transformer = MiniMaxH3Transformer3DModel(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        )

    def load_weights(self, weights: dict) -> None:
        transformer_weights = weights.get(PipelineComponent.TRANSFORMER, weights)
        self.transformer.load_weights(transformer_weights)
        self.transformer.eval()

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list[str | PipelineComponent]] = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        skip_components = skip_components or []

        if not _component_skipped(skip_components, PipelineComponent.TOKENIZER):
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TOKENIZER,
            )
        if not _component_skipped(skip_components, PipelineComponent.PROCESSOR):
            self.processor = Qwen3VLProcessor.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.PROCESSOR,
            )
        if not _component_skipped(skip_components, PipelineComponent.TEXT_ENCODER):
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=torch.bfloat16,
            ).to(device)
            self.text_encoder.eval()
        if not _component_skipped(skip_components, PipelineComponent.VAE):
            self.vae = AutoencoderKLMiniMaxH3.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.float32,
            ).to(device)
            self.vae.eval()
        if not _component_skipped(skip_components, PipelineComponent.AUDIO_VAE):
            self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.AUDIO_VAE,
                torch_dtype=torch.float32,
            ).to(device)
            self.audio_vae.eval()
        if not _component_skipped(skip_components, PipelineComponent.SCHEDULER):
            self.scheduler = MiniMaxH3Scheduler.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.SCHEDULER,
            )
        if not _component_skipped(skip_components, PipelineComponent.AUDIO_SCHEDULER):
            self.audio_scheduler = MiniMaxH3Scheduler.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.AUDIO_SCHEDULER,
            )

    def post_load_weights(self) -> None:
        super().post_load_weights()
        if self.scheduler is not None:
            self.scheduler.register_to_config(shift=self.VIDEO_SCHEDULER_SHIFT)
        if self.audio_scheduler is not None:
            self.audio_scheduler.register_to_config(shift=self.AUDIO_SCHEDULER_SHIFT)

    def _load_request_keyframes(
        self,
        req: Any,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Load first/last keyframes and preserve their temporal anchors."""

        images = req.params.image
        if images is None:
            images = []
        elif not isinstance(images, list):
            images = [images]
        if len(images) > 2:
            raise ValueError(
                "MiniMax-H3 FL2VA accepts at most two images: first frame, then last frame."
            )

        extra = req.params.extra_params or {}
        last_image = extra.get("last_image")
        if len(images) == 2 and last_image is not None:
            raise ValueError(
                "Pass the last frame either as image[1] or extra_params.last_image, not both."
            )

        keyframes = []
        keyframe_anchors = []
        if images:
            keyframes.append(load_image(images[0], format="pil"))
            keyframe_anchors.append("first")
        if len(images) == 2:
            keyframes.append(load_image(images[1], format="pil"))
            keyframe_anchors.append("last")
        elif last_image is not None:
            keyframes.append(load_image(last_image, format="pil"))
            keyframe_anchors.append("last")
        return keyframes, tuple(keyframe_anchors)

    def prepare_request(self, req: Any) -> None:
        """Resolve mode-dependent canvas geometry before warmup lookup."""

        keyframes, keyframe_anchors = self._load_request_keyframes(req)
        req.prepared_inputs["keyframes"] = keyframes
        req.prepared_inputs["keyframe_anchors"] = keyframe_anchors

        height = req.params.height
        width = req.params.width
        if (height is None) != (width is None):
            raise ValueError("MiniMax-H3 height and width must be set together.")
        if height is None:
            source_width, source_height = keyframes[0].size if keyframes else (16, 9)
            req.params.height, req.params.width = resolve_canvas_size(
                source_width,
                source_height,
            )

    def infer(self, req: Any) -> PipelineOutput:
        if req.params.negative_prompt:
            raise ValueError(
                "MiniMax-H3 is guidance-distilled and does not accept negative_prompt."
            )
        if req.params.num_images_per_prompt != 1:
            raise ValueError("MiniMax-H3 initial support generates one video per request.")

        prompt = req.prompt
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("MiniMax-H3 initial support accepts one prompt per request.")
            prompt = prompt[0]

        prepared_inputs = getattr(req, "prepared_inputs", {})
        keyframes = prepared_inputs.get("keyframes")
        keyframe_anchors = prepared_inputs.get("keyframe_anchors")
        if keyframes is None or keyframe_anchors is None:
            keyframes, keyframe_anchors = self._load_request_keyframes(req)

        return self.forward(
            prompt=prompt,
            seed=req.params.seed,
            height=req.params.height,
            width=req.params.width,
            num_frames=req.params.num_frames,
            frame_rate=req.params.frame_rate,
            num_inference_steps=req.params.num_inference_steps,
            keyframes=keyframes,
            keyframe_anchors=keyframe_anchors,
        )

    def _validate_request(
        self,
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
    ) -> int:
        if not isinstance(prompt, str):
            raise ValueError("MiniMax-H3 accepts one prompt string per request.")
        if frame_rate != MINIMAX_H3_FPS:
            raise ValueError(f"MiniMax-H3 uses a fixed {MINIMAX_H3_FPS} fps, got {frame_rate}.")
        self.validate_resolution(height, width, num_frames)
        aligned_num_frames = align_num_frames(num_frames)
        duration = aligned_num_frames / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMax-H3 duration must be between {MINIMAX_H3_MIN_DURATION} "
                f"and {MINIMAX_H3_MAX_DURATION} seconds after frame alignment; "
                f"got {aligned_num_frames} frames."
            )
        if aligned_num_frames != num_frames:
            logger.warning(
                f"MiniMax-H3 requires 17*n+5 frames; rounding {num_frames} "
                f"up to {aligned_num_frames}."
            )
        return aligned_num_frames

    def _prepare_keyframes(
        self,
        keyframes: list[Image.Image],
        keyframe_anchors: tuple[str, ...],
        height: int,
        width: int,
    ) -> list[Image.Image]:
        if len(keyframes) != len(keyframe_anchors):
            raise ValueError("Every MiniMax-H3 keyframe must have a matching anchor.")
        if any(anchor not in ("first", "last") for anchor in keyframe_anchors):
            raise ValueError("MiniMax-H3 keyframe anchors must be 'first' or 'last'.")
        prepared = []
        for index, image in enumerate(keyframes):
            image = ImageOps.exif_transpose(image).convert("RGB")
            prepared.append(
                prepare_keyframe_image(
                    image,
                    height,
                    width,
                    stretch=index == 0,
                )
            )
        return prepared

    def _encode_prompt(
        self,
        prompt: str,
        keyframes: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        if num_layers <= MINIMAX_H3_TEXT_ENCODER_LAYER:
            raise ValueError(
                "MiniMax-H3 requires the unnormalized hidden state after Qwen3-VL "
                f"layer {MINIMAX_H3_TEXT_ENCODER_LAYER}, but the encoder has "
                f"{num_layers} layers."
            )

        pixel_values = None
        image_grid_thw = None
        token_ids = []
        token_tags = []
        if keyframes:
            vision = self.processor.image_processor(images=keyframes, return_tensors="pt")
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
            merge_size = self.processor.image_processor.merge_size**2
            for index in range(len(keyframes)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = self.tokenizer(
                    f"<Picture {index + 1}>: ",
                    add_special_tokens=False,
                )["input_ids"]
                vision_ids = [
                    self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                    *[
                        self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                        for _ in range(num_image_tokens)
                    ],
                    self.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
                ]
                token_ids.extend(label_ids + vision_ids)
                token_tags.extend(
                    [MINIMAX_H3_TEXT_TAG] * len(label_ids)
                    + [MINIMAX_H3_VIDEO_TAG] * len(vision_ids)
                )

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids.extend(prompt_ids)
        token_tags.extend([MINIMAX_H3_TEXT_TAG] * len(prompt_ids))
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        mm_token_type_ids = torch.tensor(
            self.processor.create_mm_token_type_ids([token_ids]),
            dtype=torch.long,
            device=self.device,
        )
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=None
            if pixel_values is None
            else pixel_values.to(self.device, self.text_encoder.dtype),
            image_grid_thw=None if image_grid_thw is None else image_grid_thw.to(self.device),
            use_cache=False,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[MINIMAX_H3_TEXT_ENCODER_LAYER].to(
            device=self.device,
            dtype=self.dtype,
        )
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long)

    def _encode_keyframes(
        self,
        keyframes: list[Image.Image],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
    ) -> Optional[torch.Tensor]:
        if not keyframes:
            return None

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        rows = []
        for image in keyframes:
            pixels = torch.from_numpy(np.array(image)).to(self.device)
            pixels = pixels.permute(2, 0, 1)[None, :, None]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            moments = self.vae._encode_clip(pixels)
            posterior = MiniMaxH3DiagonalGaussianDistribution(moments)
            encode_generator = torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
            latents = posterior.sample(generator=encode_generator)
            latents = latents.to(torch.float16).float().cpu()
            rows.append(
                patchify_video_latents(
                    (latents - latents_mean) / latents_std,
                    self.transformer.config.patch_size,
                )
            )
        condition_latents = torch.cat(rows).to(self.device)
        noise = keyframe_condition_noise(
            ((1, latent_height, latent_width),) * len(keyframes),
            self.transformer.config.patch_size,
            self.vae.config.latent_channels,
            generator=generator,
            device=self.device,
        )
        return self.scheduler.scale_noise(
            condition_latents,
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            noise,
        )

    def _prepare_latents(
        self,
        *,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        generator: torch.Generator,
        condition_latents: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = minimax_h3_randn_tensor(
            (
                1,
                self.vae.config.latent_channels,
                num_latent_frames,
                latent_height,
                latent_width,
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        video_rows = patchify_video_latents(
            latents,
            self.transformer.config.patch_size,
        )
        if condition_latents is not None:
            video_rows = torch.cat((condition_latents, video_rows))

        audio_rows = minimax_h3_randn_tensor(
            (
                num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS,
                self.audio_vae.config.latent_channels,
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        return video_rows, audio_rows

    def _decode_video(
        self,
        rows: torch.Tensor,
        num_condition_rows: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        latents = unpatchify_video_tokens(
            rows[num_condition_rows:],
            num_latent_frames,
            latent_height,
            latent_width,
            self.vae.config.latent_channels,
            self.transformer.config.patch_size,
        )
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=self.device,
        ).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=self.device,
        ).view(1, -1, 1, 1, 1)
        latents = latents * latents_std + latents_mean
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            video = self.vae.decode(latents, return_dict=False)[0]
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)
        return (video.permute(0, 2, 3, 4, 1) * 255).round().to(torch.uint8)

    def _decode_audio(
        self,
        rows: torch.Tensor,
        num_audio_latents: int,
    ) -> torch.Tensor:
        latents = unpack_audio_tokens(rows, num_audio_latents)
        latents_mean = torch.tensor(
            self.audio_vae.config.latents_mean,
            device=self.device,
        ).view(1, -1, 1)
        latents_std = torch.tensor(
            self.audio_vae.config.latents_std,
            device=self.device,
        ).view(1, -1, 1)
        audio = self.audio_vae.decode(
            latents * latents_std + latents_mean,
            return_dict=False,
        )[0]
        return audio.float().permute(1, 0, 2)

    @torch.inference_mode()
    def forward(
        self,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        keyframes: Optional[list[Image.Image]] = None,
        keyframe_anchors: Optional[tuple[str, ...]] = None,
    ) -> PipelineOutput:
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        keyframes = keyframes or []
        num_frames = self._validate_request(
            prompt,
            height,
            width,
            num_frames,
            frame_rate,
        )
        if keyframe_anchors is None:
            keyframe_anchors = ("first", "last")[: len(keyframes)]
        keyframes = self._prepare_keyframes(
            keyframes,
            keyframe_anchors,
            height,
            width,
        )
        generator = self._request_generator(seed)
        num_latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // self.vae.spatial_compression_ratio
        latent_width = width // self.vae.spatial_compression_ratio
        num_audio_latents = audio_latent_num_frames(num_frames)

        prompt_embeds, text_token_tags = self._encode_prompt(prompt, keyframes)
        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            self.transformer.config.patch_size,
            keyframe_anchors,
        )
        position_ids = layout.position_ids.to(self.device)
        token_tags = layout.token_tags.to(self.device)
        video_indices = layout.video_indices.to(self.device)
        audio_indices = layout.audio_indices.to(self.device)
        text_indices = layout.text_indices.to(self.device)

        condition_latents = self._encode_keyframes(
            keyframes,
            latent_height,
            latent_width,
            generator,
        )
        latents, audio_latents = self._prepare_latents(
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            generator=generator,
            condition_latents=condition_latents,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.audio_scheduler.set_timesteps(num_inference_steps, device=self.device)
        row_timestep_plan = [
            tuple(
                tensor.to(self.device)
                for tensor in build_row_timesteps(
                    layout,
                    float(video_timestep),
                    float(audio_timestep),
                    max(float(video_timestep), MINIMAX_H3_KEYFRAME_NOISE_AUG),
                    1.0,
                )
            )
            for video_timestep, audio_timestep in zip(
                self.scheduler.timesteps,
                self.audio_scheduler.timesteps,
            )
        ]
        static_context = self.transformer.prepare_static_context(
            prompt_embeds,
            position_ids,
        )
        timer.mark_denoise_start()

        for index, video_timestep in self._profile_denoise_steps(self.scheduler.timesteps):
            unique_timesteps, timestep_indices = row_timestep_plan[index]
            video_velocity, audio_velocity = self.transformer(
                hidden_states=latents[None],
                audio_hidden_states=audio_latents[None],
                encoder_hidden_states=None,
                timestep=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=None,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                return_dict=False,
                static_context=static_context,
            )
            condition_rows = layout.num_condition_video_rows
            latents[condition_rows:] = self.scheduler.step(
                video_velocity[0, condition_rows:].float(),
                video_timestep,
                latents[condition_rows:],
                return_dict=False,
            )[0]
            audio_latents[:] = self.audio_scheduler.step(
                audio_velocity[0].float(),
                self.audio_scheduler.timesteps[index],
                audio_latents,
                return_dict=False,
            )[0]

        timer.mark_post_start()
        video = self._decode_video(
            latents,
            layout.num_condition_video_rows,
            num_latent_frames,
            latent_height,
            latent_width,
        )
        audio = self._decode_audio(audio_latents, num_audio_latents)
        timer.mark_end()
        logger.info(f"MiniMax-H3 inference completed in {time.time() - pipeline_start:.2f}s")
        return timer.fill(
            PipelineOutput(
                video=video,
                audio=audio,
                frame_rate=float(MINIMAX_H3_FPS),
                audio_sample_rate=int(self.audio_vae.config.sampling_rate),
            )
        )
