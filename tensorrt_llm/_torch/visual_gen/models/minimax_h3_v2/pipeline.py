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

"""TRT-LLM VisualGen pipeline for MiniMax-H3 FL2VA and Ref2VA checkpoints."""

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from io import BytesIO
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3Audio, MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageOps
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, RefSlotSpec, RoleSpec
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.logger import logger

from .components.text_encoder import encode_text_tokens
from .components.vae import _prepare_vae_decoder, decode_audio, decode_video
from .ops.vsa import build_h3_vsa_metadata
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
    video_latent_num_frames,
)
from .parallel.vae import TiledAutoencoderKLMiniMaxH3

if TYPE_CHECKING:
    from ...sleep import PipelineSleepManager, SleepRestoreMode


def _component_skipped(
    skip_components: list[str | PipelineComponent],
    component: PipelineComponent,
) -> bool:
    return component in skip_components or component.value in skip_components


def _load_keyframe_image(content: Any) -> Image.Image:
    """Decode one keyframe reference into RGB.

    ``prepare_reference_slots`` resolves every reference to raw bytes before the
    request reaches a worker, so bytes are the canonical form here; ``load_image``
    would try to ``urlparse`` them.  Strings and PIL images are still accepted so
    the pipeline stays usable when driven directly.
    """
    if isinstance(content, Image.Image):
        return content.convert("RGB")
    if isinstance(content, (bytes, bytearray, memoryview)):
        return Image.open(BytesIO(bytes(content))).convert("RGB")
    return load_image(content, format="pil")


def _check_denoise_step(velocity: torch.Tensor, name: str, step: int) -> None:
    """Reject a denoising step that cannot produce a usable frame.

    An all-zero or non-finite velocity yields an all-zero latent, which decodes
    to a black frame (and silent audio) without raising anywhere.
    """
    # Reduce both conditions on device and read them back together, so the
    # check costs one sync per call rather than one per condition.
    finite, nonzero = torch.stack([torch.isfinite(velocity).all(), velocity.any()]).tolist()
    if not finite:
        raise RuntimeError(
            f"MiniMax-H3 {name} velocity is not finite at denoising step {step}; "
            "refusing to emit a corrupt result."
        )
    if not nonzero:
        raise RuntimeError(
            f"MiniMax-H3 {name} velocity is all zeros at denoising step {step}; "
            "this decodes to a blank output, so the result is rejected."
        )


class H3PipelineBase(BasePipeline):
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
        if (
            pipeline_config.attention.backend == "TRTLLM"
            and torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        ):
            raise NotImplementedError(
                "MiniMax-H3 currently restricts TRTLLM attention to SM100 or SM103. "
                "Other architectures need model-specific numerical validation; use VANILLA."
            )
        if pipeline_config.cache is not None:
            raise NotImplementedError(
                "TeaCache and Cache-DiT are not quality-validated for MiniMax-H3."
            )
        if pipeline_config.cuda_graph.enable:
            raise NotImplementedError(
                "CUDA graphs are not yet supported for MiniMax-H3's packed layout inputs."
            )
        vgm = getattr(pipeline_config, "visual_gen_mapping", None)
        world_size = vgm.world_size if vgm is not None else pipeline_config.mapping.world_size
        vae_size = pipeline_config.parallel.parallel_vae_size
        if not 1 <= vae_size <= world_size:
            raise ValueError("MiniMax-H3 parallel_vae_size must be between 1 and world_size.")
        self.audio_vae = None
        self.audio_scheduler = None
        self.processor = None
        self._sleep_manager: Optional["PipelineSleepManager"] = None
        super().__init__(pipeline_config)

    def default_offload_stages(self) -> tuple[tuple[str, ...], ...]:
        return (("text_encoder",), ("transformer",), ("vae",), ("audio_vae",))

    def extra_offload_component_names(self) -> set[str]:
        # Non-participating ranks receive embeddings instead of loading Qwen.
        return {PipelineComponent.TEXT_ENCODER.value}

    def offload_pipeline_components(self) -> dict[str, torch.nn.Module]:
        return {
            name: component
            for name in ("text_encoder", "transformer", "vae", "audio_vae")
            if (component := getattr(self, name, None)) is not None
        }

    def _check_sleep_world_size(self) -> None:
        config = self.pipeline_config
        vgm = config.visual_gen_mapping
        if (
            self.mapping.world_size != 1
            or config.parallel.n_workers != 1
            or (vgm is not None and vgm.world_size != 1)
        ):
            raise ValueError("MiniMax-H3 sleep/wake currently requires world_size=1")

    @contextmanager
    def _sleep_loading_context(
        self,
        restore_mode: "SleepRestoreMode",
        *,
        device: torch.device,
        release_host_backup: bool = False,
    ) -> Iterator[None]:
        """Back persistent H3 allocations while keeping warmup outside the pool."""
        config = self.pipeline_config
        self._check_sleep_world_size()
        if config.cpu_offload_config.enable or config.cuda_graph.enable:
            raise ValueError(
                "MiniMax-H3 sleep cannot be combined with CPU offloading or CUDA graphs"
            )
        if config.cache is not None or config.runtime_lora is not None:
            raise ValueError(
                "MiniMax-H3 sleep does not yet support cache acceleration or runtime LoRA"
            )
        if self._sleep_manager is not None:
            raise RuntimeError("Pipeline sleep was already enabled during loading")

        from ...sleep import PipelineSleepManager

        manager = PipelineSleepManager(
            restore_mode, device, release_host_backup=release_host_backup
        )
        with manager.loading():
            yield
        self._sleep_manager = manager

    def sleep(self) -> None:
        """Reject new generation and wait for the active call before releasing GPU memory."""
        if self._sleep_manager is None:
            raise RuntimeError("Pipeline sleep was not enabled during loading")
        self._check_sleep_world_size()
        self._sleep_manager.sleep()

    def wake_up(self) -> None:
        """Restore managed GPU memory before admitting more inference."""
        if self._sleep_manager is None:
            raise RuntimeError("Pipeline sleep was not enabled during loading")
        self._check_sleep_world_size()
        self._sleep_manager.wake_up()

    @property
    def is_sleeping(self) -> bool:
        return self._sleep_manager is not None and self._sleep_manager.is_sleeping

    def setup_parallel_vae(self) -> None:
        """Use the shared VAE group for independent spatial tile decoding."""
        size = self.pipeline_config.parallel.parallel_vae_size
        self._parallel_vae_enabled = size > 1 and self.vae is not None
        if not self._parallel_vae_enabled:
            return
        vgm = self.pipeline_config.visual_gen_mapping
        if vgm is None:
            raise RuntimeError("MiniMax-H3 parallel VAE requires a VisualGen mapping.")
        if self.rank not in vgm.vae_ranks:
            return
        if vgm.vae_group is None:
            raise RuntimeError("MiniMax-H3 parallel VAE requires a VAE group.")
        if torch.distributed.get_world_size(vgm.vae_group) != size:
            raise RuntimeError("MiniMax-H3 VAE group size does not match parallel_vae_size.")
        self.vae.configure_parallel(group=vgm.vae_group)
        logger.info(f"MiniMax-H3 spatial VAE tile decode ranks={size}")

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
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        return {
            "image_reference": RefSlotSpec(
                modality="image",
                roles=[
                    RoleSpec(role="first_frame", min=0, max=1),
                    RoleSpec(role="last_frame", min=0, max=1),
                ],
            ),
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

    def load_weights(self, weights: dict) -> None:
        transformer_weights = weights.get(PipelineComponent.TRANSFORMER, weights)
        self.transformer.load_weights(transformer_weights)
        self.transformer.eval()

    def _load_text_encoder(self, checkpoint_dir, device, offloaded):
        return (
            Qwen3VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=torch.bfloat16,
            )
            .to("cpu" if "text_encoder" in offloaded else device)
            .eval()
        )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list[str | PipelineComponent]] = None,
        **kwargs: object,
    ) -> None:
        del kwargs
        skip_components = skip_components or []
        offloaded = self.offloader.requested_components()

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
            self.text_encoder = self._load_text_encoder(checkpoint_dir, device, offloaded)
        if not _component_skipped(skip_components, PipelineComponent.VAE):
            self.vae = TiledAutoencoderKLMiniMaxH3.from_pretrained(
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
            ).to("cpu" if "audio_vae" in offloaded else device)
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
        self._configure_scheduler_shifts()
        if self.vae is not None and self.device.type == "cuda":
            _prepare_vae_decoder(self.vae)

    def _configure_scheduler_shifts(self):
        pass

    def _set_timesteps(self, num_inference_steps: int) -> None:
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.audio_scheduler.set_timesteps(num_inference_steps, device=self.device)

    def torch_compile(self) -> None:
        super().torch_compile()
        # The ViT VAE decoder is 36 identical blocks run once per clip; compile them like the transformer blocks.
        if self.vae is not None:
            tc_config = self.pipeline_config.torch_compile
            blocks = self.vae.decoder.transformer_blocks
            logger.info(
                f"torch.compile: vae.decoder.transformer_blocks ({len(blocks)} blocks, mode=default)"
            )
            self.vae.decoder.transformer_blocks = torch.nn.ModuleList(
                torch.compile(
                    block, mode="default", dynamic=None, fullgraph=tc_config.enable_fullgraph
                )
                for block in blocks
            )

    def _load_request_keyframes(
        self,
        req: Any,
    ) -> tuple[list[Image.Image], tuple[str, ...]]:
        """Load first/last keyframes and preserve their temporal anchors.

        Both keyframe roles are optional, so ``validate_visual_gen_params``
        cannot infer one and requires callers to name it; there is deliberately
        no positional fallback.
        """

        refs = req.params.image_reference or []
        if not isinstance(refs, list):
            refs = [refs]
        if len(refs) > 2:
            raise ValueError(
                "MiniMax-H3 FL2VA accepts at most two images: first frame, then last frame."
            )

        by_role: dict[str, Any] = {}
        for ref in refs:
            role = getattr(ref, "role", None)
            if role not in ("first_frame", "last_frame"):
                raise ValueError(
                    "MiniMax-H3 image_reference requires role 'first_frame' or "
                    f"'last_frame', got {role!r}."
                )
            if role in by_role:
                raise ValueError(f"MiniMax-H3 accepts a single {role} image.")
            by_role[role] = ref.content

        keyframes = []
        keyframe_anchors = []
        for role, anchor in (("first_frame", "first"), ("last_frame", "last")):
            content = by_role.get(role)
            if content is not None:
                keyframes.append(_load_keyframe_image(content))
                keyframe_anchors.append(anchor)
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

        return self.forward(
            prompt=prompt,
            seed=req.params.seed,
            height=req.params.height,
            width=req.params.width,
            num_frames=req.params.num_frames,
            frame_rate=req.params.frame_rate,
            num_inference_steps=req.params.num_inference_steps,
            **self._request_conditions(req),
        )

    def _request_conditions(self, req: Any) -> dict:
        prepared_inputs = getattr(req, "prepared_inputs", {})
        keyframes = prepared_inputs.get("keyframes")
        keyframe_anchors = prepared_inputs.get("keyframe_anchors")
        if keyframes is None or keyframe_anchors is None:
            keyframes, keyframe_anchors = self._load_request_keyframes(req)

        return dict(keyframes=keyframes, keyframe_anchors=keyframe_anchors, references=None)

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

    def _reference_conditioning(
        self, references, prompt, height, width, num_frames, keyframes, generator
    ):
        if references:
            raise ValueError("This MiniMax-H3 target accepts text and first/last keyframes.")
        return None

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
            # Deliberate deviation: the Diffusers reference does no EXIF
            # handling, so an EXIF-rotated input will not match a golden.
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
        self, prompt: str, keyframes: list[Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_prompt_local(prompt, keyframes)

    def _encode_prompt_local(
        self,
        prompt: str,
        keyframes: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None:
            raise RuntimeError("MiniMax-H3 text encoder is not loaded on this rank.")

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
        vision_inputs = {}
        if pixel_values is not None:
            vision_inputs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
        return self._encode_text_tokens(token_ids, vision_inputs), torch.tensor(
            token_tags, dtype=torch.long
        )

    def _encode_text_tokens(
        self, token_ids: list[int], vision_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode one H3 presentation, shared by keyframes and mixed references."""
        return encode_text_tokens(
            self.text_encoder,
            self.processor,
            token_ids,
            vision_inputs,
            device=self.device,
            dtype=self.dtype,
            offloader=self.offloader,
        )

    def _encode_keyframes(
        self,
        keyframes: list[Image.Image],
        latent_height: int,
        latent_width: int,
        generator: torch.Generator,
    ) -> Optional[torch.Tensor]:
        if not keyframes:
            return None

        del latent_height, latent_width
        latents = []
        for image in keyframes:
            pixels = torch.from_numpy(np.array(image)).to(self.device)
            pixels = pixels.permute(2, 0, 1)[None, :, None]
            latents.append(self._encode_visual_condition(pixels))
        return self._prepare_condition_rows(latents, generator)

    def _encode_visual_condition(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode uint8 [1, 3, T, H, W] pixels as normalized FP32 CPU latents."""
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
        with self.offloader.context_if_requested("vae"):
            posterior = self.vae.encode(pixels).latent_dist
        encode_generator = torch.Generator().manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        latents = posterior.sample(generator=encode_generator).to(torch.float16).float().cpu()
        return (latents - latents_mean) / latents_std

    def _prepare_condition_rows(
        self, latents: list[torch.Tensor], generator: torch.Generator
    ) -> torch.Tensor:
        """Pack and noise heterogeneous visual references in request order."""
        patch_size = self.transformer.config.patch_size
        condition_rows = torch.cat(
            [patchify_video_latents(value, patch_size) for value in latents]
        ).to(self.device)
        noise = keyframe_condition_noise(
            tuple(tuple(value.shape[2:]) for value in latents),
            patch_size,
            self.vae.config.latent_channels,
            generator=generator,
            device=self.device,
        )
        return self.scheduler.scale_noise(condition_rows, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)

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
        latents = randn_tensor(
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

        audio_rows = randn_tensor(
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
        return decode_video(
            self.vae,
            self.offloader,
            self.device,
            self.transformer.config.patch_size,
            rows,
            num_condition_rows,
            num_latent_frames,
            latent_height,
            latent_width,
        )

    def _decode_audio(
        self,
        rows: torch.Tensor,
        num_audio_latents: int,
    ) -> torch.Tensor:
        return decode_audio(self.audio_vae, self.offloader, self.device, rows, num_audio_latents)

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
        references: Optional[list] = None,
    ) -> PipelineOutput:
        manager = getattr(self, "_sleep_manager", None)
        with manager.generation() if manager is not None else nullcontext():
            return self._forward(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                keyframes=keyframes,
                keyframe_anchors=keyframe_anchors,
                references=references,
            )

    def _forward(
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
        references: Optional[list] = None,
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
        attention = self.pipeline_config.primary_model_config.attention
        sparse = getattr(attention, "sparse_attention_config", None)
        vsa = (
            getattr(attention, "backend", "VANILLA") in ("CUTEDSL", "TRTLLM")
            and getattr(sparse, "algorithm", None) == "vsa"
        )
        if vsa and (keyframes or references):
            raise ValueError("MiniMax-H3 VSA currently supports text-to-video only.")
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

        conditioning = self._reference_conditioning(
            references, prompt, height, width, num_frames, keyframes, generator
        )
        if conditioning is None:
            audio_condition_latents = None
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
            condition_latents = self._encode_keyframes(
                keyframes,
                latent_height,
                latent_width,
                generator,
            )
        else:
            prompt_embeds, layout, condition_latents, audio_condition_latents = conditioning
        position_ids = layout.position_ids.to(self.device)
        token_tags = layout.token_tags.to(self.device)
        video_indices = layout.video_indices.to(self.device)
        audio_indices = layout.audio_indices.to(self.device)
        text_indices = layout.text_indices.to(self.device)

        latents, audio_latents = self._prepare_latents(
            num_latent_frames=num_latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            generator=generator,
            condition_latents=condition_latents,
        )

        self._set_timesteps(num_inference_steps)
        # The denoise loop indexes both schedules with one counter, and
        # set_timesteps collapses float32 duplicates, so unequal lengths would
        # silently truncate a stream instead of failing.
        if len(self.scheduler.timesteps) != len(self.audio_scheduler.timesteps):
            raise RuntimeError(
                "MiniMax-H3 video and audio schedules disagree: "
                f"{len(self.scheduler.timesteps)} vs "
                f"{len(self.audio_scheduler.timesteps)} steps."
            )
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
        with self.offloader.context_if_requested("transformer"):
            static_context = self.transformer.prepare_static_context(
                prompt_embeds,
                position_ids,
            )
        condition_rows = layout.num_condition_video_rows
        condition_prefix = latents[:condition_rows][None]
        audio_condition_rows = layout.num_condition_audio_rows
        audio_condition_prefix = (
            audio_condition_latents[None]
            if audio_condition_latents is not None
            else audio_latents[None, :0]
        )
        # H3's native time increases toward clean. Attention and graph-phase
        # scheduling use descending normalized noise, protecting both streams.
        attention_timesteps = 1.0 - torch.minimum(
            self.scheduler.timesteps, self.audio_scheduler.timesteps
        )

        vsa_metadata = None
        if vsa:
            patch = self.transformer.config.patch_size
            grid = tuple(
                size // p
                for size, p in zip(
                    (num_latent_frames, latent_height, latent_width), patch, strict=True
                )
            )
            vsa_metadata = build_h3_vsa_metadata(
                (text_indices.numel(), audio_indices.numel(), grid),
                sparse.vsa_sparsity,
                self.device,
            )

        def forward_fn(
            video_latents: torch.Tensor,
            extra_stream_latents: dict[str, torch.Tensor],
            index: int,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            extra_tensors: dict[str, torch.Tensor],
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            del timestep, encoder_hidden_states, extra_tensors
            unique_timesteps, timestep_indices = row_timestep_plan[index]
            transformer_kwargs = dict(
                hidden_states=torch.cat((condition_prefix, video_latents), dim=1),
                audio_hidden_states=torch.cat(
                    (audio_condition_prefix, extra_stream_latents["audio"]), dim=1
                ),
                encoder_hidden_states=None,
                timestep=attention_timesteps[index : index + 1],
                conditioning_timesteps=unique_timesteps,
                timestep_indices=timestep_indices,
                token_tags=token_tags,
                position_ids=None,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                return_dict=False,
                static_context=static_context,
            )
            if vsa_metadata is not None:
                transformer_kwargs["vsa_metadata"] = vsa_metadata
            video_velocity, audio_velocity = self.transformer(**transformer_kwargs)
            _check_denoise_step(video_velocity[:, condition_rows:], "video", index)
            audio_velocity = audio_velocity[:, audio_condition_rows:]
            _check_denoise_step(audio_velocity, "audio", index)
            return video_velocity[:, condition_rows:].float(), {"audio": audio_velocity.float()}

        timer.mark_denoise_start()
        with self.offloader.context_if_requested("transformer"):
            generated_latents, extra_latents = self.denoise(
                latents=latents[condition_rows:][None],
                scheduler=self.scheduler,
                prompt_embeds=prompt_embeds,
                guidance_scale=1.0,
                forward_fn=forward_fn,
                extra_streams={"audio": (audio_latents[None], self.audio_scheduler)},
                extra_stream_timesteps={"audio": self.audio_scheduler.timesteps},
            )
        latents = torch.cat((condition_prefix, generated_latents), dim=1)[0]
        audio_latents = extra_latents["audio"][0]
        timer.mark_post_start()
        video, audio = self.decode_latents(
            latents,
            decode_fn=lambda rows: self._decode_video(
                rows,
                layout.num_condition_video_rows,
                num_latent_frames,
                latent_height,
                latent_width,
            ),
            extra_latents={
                "audio": (audio_latents, lambda rows: self._decode_audio(rows, num_audio_latents))
            },
        )
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
