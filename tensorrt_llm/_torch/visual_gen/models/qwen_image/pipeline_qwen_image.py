# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image pipeline (Phase 0 stub).

Registers ``QwenImagePipeline`` with :class:`AutoPipeline` so that Qwen-Image
checkpoints stop erroring with ``Unknown pipeline: ''``. All non-transformer
components (VAE, text encoder, tokenizer, scheduler) are loaded from the HF
checkpoint using diffusers/transformers, mirroring the FLUX pattern. The
denoising path raises a clear :class:`NotImplementedError` pointing at the
Phase 1 follow-up work.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm.logger import logger

from .transformer_qwen_image import QwenImageTransformer2DModel

_PHASE1_MSG = (
    "Qwen-Image native inference is not yet implemented in TensorRT-LLM "
    "(Phase 0 scaffolding only). The MMDiT transformer, MSRoPE, joint "
    "text-image attention, VAE decode packing, and denoise loop will land "
    "in the Phase 1 upstream PR."
)


@register_pipeline("QwenImagePipeline")
class QwenImagePipeline(BasePipeline):
    """Qwen-Image text-to-image pipeline (Phase 0 stub).

    The transformer backbone is a stub that raises ``NotImplementedError``
    on any forward pass. This class exists so that:

    1. ``AutoPipeline`` can route Qwen-Image checkpoints through VisualGen
       with a sensible error message instead of ``Unknown pipeline: ''``.
    2. Phase 1 only needs to implement the transformer module and the
       image-specific parts of ``forward`` / ``infer``; the wiring to
       ``trtllm-serve``, executor, and multi-GPU parallelism is already in
       place via :class:`BasePipeline`.

    References:
        - Qwen-Image model card: https://huggingface.co/Qwen/Qwen-Image
        - Tech report: https://arxiv.org/abs/2508.02324
        - diffusers implementation: ``diffusers.pipelines.qwenimage``
    """

    def __init__(self, model_config):
        super().__init__(model_config)

    @property
    def dtype(self):
        return self.model_config.torch_dtype

    @property
    def device(self):
        if self.transformer is not None:
            return next(self.transformer.parameters()).device
        return torch.device("cuda:0")

    # ------------------------------------------------------------------
    # Warmup / resolution constraints.
    # Mirrors FLUX defaults; Phase 1 should confirm against the
    # Qwen-Image aspect-ratio presets on the HF model card.
    # ------------------------------------------------------------------
    @property
    def default_warmup_resolutions(self) -> List[Tuple[int, int]]:
        return [(1328, 1328)]

    @property
    def default_warmup_num_frames(self) -> List[int]:
        return [1]

    def warmup_cache_key(self, height: int, width: int, **kwargs) -> tuple:
        return (height, width)

    @property
    def resolution_multiple_of(self) -> Tuple[int, int]:
        # Qwen-Image's VAE uses a patchified 2x2 latent packing similar
        # to FLUX. Phase 1 should verify the actual constraint from the
        # diffusers pipeline; 16 is a safe lower bound for any VAE with
        # 8x downsampling + 2x packing.
        return (16, 16)

    # ------------------------------------------------------------------
    # Transformer / component loading.
    # ------------------------------------------------------------------
    def _init_transformer(self) -> None:
        logger.info("Creating Qwen-Image transformer stub (Phase 0)")
        self.transformer = QwenImageTransformer2DModel(
            model_config=self.model_config
        )

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
    ) -> None:
        """Load VAE, text encoder, tokenizer, and scheduler.

        The Qwen-Image checkpoint on HuggingFace bundles these inside the
        standard diffusers directory layout (``<ckpt>/vae``, ``<ckpt>/
        text_encoder``, etc.). We load them via diffusers / transformers
        directly -- no TRT-LLM-side optimization is applied to these
        components in Phase 0. Phase 1 may replace the text encoder with
        an accelerated Qwen2.5-VL path.
        """
        skip_components = skip_components or []

        # Imports are local so that installs without the Qwen-Image
        # components (older diffusers / transformers) can still import
        # this module; failures here surface with a clear message.
        try:
            from diffusers import (  # type: ignore[attr-defined]
                AutoencoderKLQwenImage,
                FlowMatchEulerDiscreteScheduler,
            )
        except ImportError as e:  # pragma: no cover - exercised on old diffusers
            raise ImportError(
                "Qwen-Image requires a diffusers build that includes "
                "AutoencoderKLQwenImage and the Qwen-Image pipeline. "
                "Install a compatible version from HuggingFace "
                "(e.g. `pip install -U diffusers`)."
            ) from e

        try:
            from transformers import (  # type: ignore[attr-defined]
                Qwen2_5_VLForConditionalGeneration,
                Qwen2Tokenizer,
            )
        except ImportError as e:  # pragma: no cover - exercised on old transformers
            raise ImportError(
                "Qwen-Image requires a transformers build that includes "
                "Qwen2_5_VLForConditionalGeneration. Install "
                "transformers >= 4.49 (`pip install -U transformers`)."
            ) from e

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info("Loading Qwen2 tokenizer...")
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.TOKENIZER
            )

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info("Loading Qwen2.5-VL text encoder...")
            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.TEXT_ENCODER,
                torch_dtype=self.model_config.torch_dtype,
            ).to(device)

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading Qwen-Image VAE...")
            self.vae = AutoencoderKLQwenImage.from_pretrained(
                checkpoint_dir,
                subfolder=PipelineComponent.VAE,
                torch_dtype=torch.bfloat16,
            ).to(device)
            # 8x downsample is standard for 16-channel diffusion VAEs.
            # Phase 1 must verify this matches the actual Qwen-Image VAE.
            self.vae_scale_factor = 2 ** (
                len(getattr(self.vae.config, "block_out_channels", [1] * 4)) - 1
            )

        if PipelineComponent.SCHEDULER not in skip_components:
            logger.info("Loading Qwen-Image scheduler...")
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                checkpoint_dir, subfolder=PipelineComponent.SCHEDULER
            )

        self.default_height = 1328
        self.default_width = 1328
        self.max_sequence_length = 1024

    def load_weights(self, weights: dict) -> None:
        """Delegate to the transformer's stub ``load_weights``.

        This will raise :class:`NotImplementedError` with a Phase-1
        pointer until the native transformer lands.
        """
        if self.transformer is not None and hasattr(
            self.transformer, "load_weights"
        ):
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
        self._target_dtype = self.model_config.torch_dtype

    # ------------------------------------------------------------------
    # Inference entry points (Phase 1 will implement).
    # ------------------------------------------------------------------
    @property
    def default_generation_params(self) -> dict:
        # Matches Qwen-Image's documented defaults on the HF model card.
        return {
            "height": 1328,
            "width": 1328,
            "num_inference_steps": 50,
            "guidance_scale": 4.0,
            "max_sequence_length": 1024,
        }

    def infer(self, req):
        raise NotImplementedError(_PHASE1_MSG)

    def forward(
        self,
        prompt: Union[str, List[str]],  # noqa: ARG002
        height: int = 1328,  # noqa: ARG002
        width: int = 1328,  # noqa: ARG002
        num_inference_steps: int = 50,  # noqa: ARG002
        guidance_scale: float = 4.0,  # noqa: ARG002
        seed: int = 42,  # noqa: ARG002
        max_sequence_length: int = 1024,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ):
        raise NotImplementedError(_PHASE1_MSG)
