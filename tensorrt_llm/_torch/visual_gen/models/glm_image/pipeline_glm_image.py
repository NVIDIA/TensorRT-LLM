# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-Image VisualGen pipeline.

This is the first TRTLLM VisualGen enablement layer for GLM-Image. It
registers the family in the VisualGen registry and runs the upstream
Diffusers GLM-Image pipeline through the TRTLLM VisualGen executor. The
adapter deliberately avoids copying GLM-Image's large AR+DiT transformer
stack into TRTLLM until a native port can reuse the shared VisualGen
attention, VAE, quantization, cache, and mapping infrastructure.
"""

from __future__ import annotations

import time
from io import BytesIO
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm.inputs.utils import load_image
from tensorrt_llm.logger import logger

_DEFAULT_GENERATION_PARAMS = {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 1.5,
    "max_sequence_length": 2048,
}


@register_pipeline(
    "GlmImagePipeline",
    hf_ids=["zai-org/GLM-Image"],
    defaults={},
    doc="ZhipuAI/Z.ai GLM-Image text-to-image and image-conditioned generation.",
)
class GlmImagePipeline(BasePipeline):
    """GLM-Image adapter backed by the upstream Diffusers implementation."""

    DEFAULT_GENERATION_PARAMS = _DEFAULT_GENERATION_PARAMS
    derive_output_size_from_reference = True

    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)
        self._diffusers_pipeline = None
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        pipe = getattr(self, "_diffusers_pipeline", None)
        if pipe is not None:
            return pipe._execution_device
        default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return getattr(self, "_device", default_device)

    @property
    def default_generation_params(self) -> dict:
        return dict(_DEFAULT_GENERATION_PARAMS)

    @property
    def resolution_multiple_of(self) -> tuple[int, int]:
        # Diffusers GLM-Image enforces multiples of 32 for image dimensions.
        return (32, 32)

    def warmup_cache_key(self, height: int | None, width: int | None, **kwargs) -> tuple:
        return (height, width)

    @property
    def default_warmup_resolutions(self) -> List[tuple[int, int]]:
        # The Diffusers-backed adapter does not do TRTLLM CUDA graph or
        # torch.compile warmup. The first request is the first real run.
        return []

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [1]

    def _init_transformer(self) -> None:
        # Native TRTLLM transformer support is intentionally not duplicated
        # here. load_standard_components() materializes the Diffusers pipeline.
        self.transformer = None

    def load_transformer_weights(self, checkpoint_dir: str) -> dict[str, torch.Tensor]:
        return {}

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        return None

    def post_load_weights(self) -> None:
        return None

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
        **kwargs,
    ) -> None:
        try:
            from diffusers import GlmImagePipeline as DiffusersGlmImagePipeline
        except ImportError:
            try:
                from diffusers.pipelines.glm_image import (
                    GlmImagePipeline as DiffusersGlmImagePipeline,
                )
            except ImportError as exc:  # pragma: no cover - depends on installed diffusers
                raise ImportError(
                    "GLM-Image requires a Diffusers release with GlmImagePipeline "
                    "and a Transformers release with the GLM-Image processor/model."
                ) from exc

        if skip_components:
            logger.warning(
                "GLM-Image Diffusers adapter ignores skip_components=%s", skip_components
            )

        logger.info("Loading GLM-Image Diffusers pipeline from %s", checkpoint_dir)
        pipe = DiffusersGlmImagePipeline.from_pretrained(
            checkpoint_dir,
            torch_dtype=self.pipeline_config.torch_dtype,
        )
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self._diffusers_pipeline = pipe
        self._device = torch.device(device)

    @staticmethod
    def _load_condition_images(image: Any) -> Any:
        if image is None:
            return None
        inputs = image if isinstance(image, list) else [image]
        loaded = []
        for item in inputs:
            if isinstance(item, Image.Image):
                loaded.append(item.convert("RGB"))
            elif isinstance(item, bytes):
                loaded.append(Image.open(BytesIO(item)).convert("RGB"))
            elif isinstance(item, np.ndarray):
                loaded.append(Image.fromarray(item).convert("RGB"))
            elif torch.is_tensor(item):
                array = item.detach().cpu()
                if array.ndim == 4:
                    array = array[0]
                if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                    array = array.permute(1, 2, 0)
                if array.dtype != torch.uint8:
                    if torch.is_floating_point(array) and float(array.max()) <= 1.0:
                        array = array * 255.0
                    array = array.clamp(0, 255).to(torch.uint8)
                loaded.append(Image.fromarray(array.numpy()).convert("RGB"))
            else:
                loaded.append(load_image(item, format="pil"))
        return loaded

    @staticmethod
    def _pil_images_to_tensor(images: List[Image.Image]) -> torch.Tensor:
        arrays = [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images]
        if not arrays:
            raise ValueError("GLM-Image pipeline returned no images.")
        return torch.from_numpy(np.stack(arrays, axis=0)).contiguous()

    def infer(self, req):
        params = req.params
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        if len(prompts) > 1 and (params.num_images_per_prompt or 1) != 1:
            raise NotImplementedError(
                "GLM-Image batched prompts currently require num_images_per_prompt=1."
            )
        if params.negative_prompt not in (None, ""):
            raise ValueError(
                "GLM-Image Diffusers pipeline does not expose a string negative_prompt; "
                "use guidance_scale or provide upstream negative_prompt_embeds in a native port."
            )
        return self.forward(
            prompt=prompts,
            image=params.image,
            height=params.height,
            width=params.width,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            num_images_per_prompt=params.num_images_per_prompt,
            seed=params.seed,
            max_sequence_length=params.max_sequence_length,
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        image: Any = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        num_images_per_prompt: int = 1,
        seed: int = 42,
        max_sequence_length: int = 2048,
    ) -> PipelineOutput:
        if self._diffusers_pipeline is None:
            raise RuntimeError("GLM-Image Diffusers pipeline has not been loaded.")

        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()

        condition_images = self._load_condition_images(image)
        generator = torch.Generator(device=self.device).manual_seed(seed)

        timer.mark_denoise_start()
        result = self._diffusers_pipeline(
            prompt=prompt,
            image=condition_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            max_sequence_length=max_sequence_length,
            output_type="pil",
            return_dict=True,
        )
        timer.mark_post_start()

        image_tensor = self._pil_images_to_tensor(result.images)
        if getattr(self, "rank", 0) == 0:
            logger.info("GLM-Image pipeline total: %.2fs", time.time() - pipeline_start)

        timer.mark_end()
        return timer.fill(PipelineOutput(image=image_tensor))
