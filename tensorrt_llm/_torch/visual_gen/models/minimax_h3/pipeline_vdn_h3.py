# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDN-H3 reuses H3's packed video/audio generation pipeline."""

from pathlib import Path

import torch

from tensorrt_llm._torch.visual_gen.checkpoints import WeightLoader
from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline

from .pipeline_minimax_h3 import MiniMaxH3Pipeline
from .transformer_vdn_h3 import VDNMiniMaxH3Transformer3DModel
from .vdn_checkpoint import VDN_STAGE, load_vdn_overlay


@register_pipeline(
    "VDNMiniMaxH3Transformer3DModel",
    hf_ids=["OpenVDN/vdn-minimax-h3"],
    defaults={"workflow": "fl2va", "stage": VDN_STAGE, "base_checkpoint": ""},
    download_patterns=[
        "modular_model_index.json",
        "h3-base/*",
        "stage-dmd-step-250/*",
        "stage-b-step-2000/*",
        "LICENSE",
        "README.md",
    ],
    doc="VDN MiniMax-H3 BF16 single-GPU video with audio, using H3 packing and VAEs.",
)
class VDNMiniMaxH3Pipeline(MiniMaxH3Pipeline):
    def __init__(self, pipeline_config: DiffusionPipelineConfig) -> None:
        if pipeline_config.extra_attrs.get("workflow", "fl2va") != "fl2va":
            raise NotImplementedError(
                "VDN-H3 uses the FL2VA backbone; native Ref2VA weights are not supported."
            )
        self.stage = pipeline_config.extra_attrs.get("stage", VDN_STAGE)
        self.base_checkpoint = pipeline_config.extra_attrs["vdn_base_checkpoint"]
        super().__init__(pipeline_config)

    def _request_generator(self, seed: int) -> torch.Generator:
        """VDN's released renderer draws its initial noise on the compute device."""
        return torch.Generator(self.device).manual_seed(seed)

    def _init_transformer(self) -> None:
        self.transformer = VDNMiniMaxH3Transformer3DModel(
            self.pipeline_config.model_configs[PipelineComponent.TRANSFORMER]
        )

    @property
    def default_generation_params(self) -> dict:
        params = super().default_generation_params
        # H3 counts sigma grid points: N evaluations require N + 1 points.
        params["num_inference_steps"] = 9 if self.stage == VDN_STAGE else 51
        return params

    def load_transformer_weights(self, checkpoint_dir: str) -> dict:
        weights = WeightLoader().load_weights(self.base_checkpoint, self.mapping)
        return load_vdn_overlay(Path(checkpoint_dir), self.stage, weights)

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: list[str | PipelineComponent] | None = None,
        **kwargs: object,
    ) -> None:
        # The base path may point at an existing full H3 snapshot. The VDN
        # repository itself omits the text encoder, tokenizer and processor.
        skipped = list(skip_components or [])
        text_components = [
            PipelineComponent.TEXT_ENCODER,
            PipelineComponent.TOKENIZER,
            PipelineComponent.PROCESSOR,
        ]
        missing_text = not (Path(self.base_checkpoint) / "text_encoder").is_dir()
        super().load_standard_components(
            self.base_checkpoint,
            device,
            skip_components=skipped + (text_components if missing_text else []),
            **kwargs,
        )
        if missing_text:
            media_components = [
                PipelineComponent.VAE,
                PipelineComponent.AUDIO_VAE,
                PipelineComponent.SCHEDULER,
                PipelineComponent.AUDIO_SCHEDULER,
            ]
            super().load_standard_components(
                "MiniMaxAI/MiniMax-H3", device, skip_components=skipped + media_components, **kwargs
            )
