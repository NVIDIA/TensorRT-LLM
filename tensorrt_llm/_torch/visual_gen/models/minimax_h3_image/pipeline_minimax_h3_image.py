# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3-Image text-to-image pipeline for the converted image checkpoint."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2TokenizerFast

from ...checkpoints import WeightLoader
from ...output import CudaPhaseTimer, PipelineOutput
from ...pipeline import BasePipeline, ExtraParamSchema
from ...pipeline_registry import register_pipeline
from ..minimax_h3.packing import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    unpatchify_video_tokens,
)
from .image_decoder import MiniMaxH3ImageDecoder
from .packing import DEFAULT_MIN_AREA, MIN_AREA_RANGE, generation_canvas, text_image_positions
from .schedule import NUM_PDD_STEPS, ImagePDDSchedule
from .text_encoder import MiniMaxH3ImageTextEncoder
from .transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel


@register_pipeline("MiniMaxH3ImageModularPipeline")
@register_pipeline(
    "MiniMaxH3ImagePipeline",
    hf_ids=["MiniMaxAI/MiniMax-H3-Image"],
    download_patterns=[
        "model_index.json",
        "modular_model_index.json",
        "transformer/config.json",
        "transformer_diffusers/*",
        "vae_diffusers/*",
        "text_encoder/*",
        "tokenizer/*",
        "scheduler/scheduler_config.json",
        "LICENSE",
    ],
    doc="MiniMax-H3-Image text-to-image generation with six distilled PDD steps.",
)
class MiniMaxH3ImagePipeline(BasePipeline):
    @property
    def default_generation_params(self):
        return dict(
            height=2048,
            width=2048,
            num_frames=1,
            num_inference_steps=NUM_PDD_STEPS,
            guidance_scale=1.0,
            max_sequence_length=4096,
        )

    @property
    def extra_param_specs(self):
        return {
            "min_generation_area": ExtraParamSchema(
                type="int",
                range=MIN_AREA_RANGE,
                default=DEFAULT_MIN_AREA,
                description="Minimum internal canvas area, in pixels, before resizing to the requested output.",
            )
        }

    def _init_transformer(self):
        config = self.pipeline_config
        if config.cuda_graph.enable or config.cache is not None:
            raise ValueError("H3 image PDD requires eager steps without activation caching")
        model = config.primary_model_config
        # The generic loader discovers transformer/config.json. This repository
        # also publishes converted weights under a *different* subfolder with
        # their own config schema. Do not load raw tensors against that schema.
        cfg = model.pretrained_config
        if getattr(cfg, "_class_name", None) != "MiniMaxH3ImageTransformer2DModel":
            path = Path(cfg._name_or_path) / "transformer_diffusers" / "config.json"
            with path.open() as f:
                cfg = SimpleNamespace(**json.load(f))
            if cfg._class_name != "MiniMaxH3ImageTransformer2DModel":
                raise ValueError(f"Unsupported converted transformer config: {path}")
            model = model.model_copy(update={"pretrained_config": cfg})
        self.transformer = MiniMaxH3ImageTransformer2DModel(model)

    def load_transformer_weights(self, checkpoint_dir):
        return WeightLoader(components="transformer_diffusers").load_weights(
            checkpoint_dir, self.mapping
        )

    def load_standard_components(self, checkpoint_dir, device, skip_components=None, **kwargs):
        skip = set(skip_components or ())
        if "tokenizer" not in skip:
            self.tokenizer = Qwen2TokenizerFast.from_pretrained(
                Path(checkpoint_dir) / "tokenizer", local_files_only=True
            )
        if "text_encoder" not in skip:
            encoder = MiniMaxH3ImageTextEncoder.from_pretrained_config(
                checkpoint_dir, self.mapping, device
            )
            encoder.load_weights(
                WeightLoader(components="text_encoder").load_weights(checkpoint_dir, self.mapping)
            )
            self.text_encoder = encoder.eval()
        if "vae" not in skip:
            with (Path(checkpoint_dir) / "vae_diffusers/config.json").open() as f:
                decoder = MiniMaxH3ImageDecoder(SimpleNamespace(**json.load(f)))
            weights = WeightLoader(components="vae_diffusers").load_weights(
                checkpoint_dir, self.mapping
            )
            decoder.load_state_dict(
                {k: v for k, v in weights.items() if k.startswith(("decoder.", "post_quant_conv."))}
            )
            self.vae = decoder.to(device).eval()
        self.scheduler = ImagePDDSchedule(device)

    def prepare_request(self, req):
        p = req.params
        if p.num_frames not in (None, 1) or p.num_images_per_prompt != 1:
            raise ValueError("MiniMax-H3-Image generates a single image per request")
        if p.num_inference_steps not in (None, NUM_PDD_STEPS):
            raise ValueError("MiniMax-H3-Image uses exactly six learned PDD heads")
        if p.negative_prompt or p.guidance_scale not in (None, 1.0):
            raise ValueError("MiniMax-H3-Image is guidance-distilled and does not support CFG")
        if p.image_reference or p.video_reference or p.audio_reference:
            raise ValueError("This MiniMax-H3-Image pipeline supports text-to-image only")

    @torch.inference_mode()
    def infer(self, req):
        self.prepare_request(req)
        p = req.params
        prompt = req.prompt
        # The executor represents even one prompt as a list; forward is scalar.
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("MiniMax-H3-Image requires exactly one prompt per request")
            prompt = prompt[0]
        return self.forward(
            prompt,
            p.height,
            p.width,
            p.seed,
            (p.extra_params or {}).get("min_generation_area", DEFAULT_MIN_AREA),
            p.max_sequence_length,
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: str,
        height: int,
        width: int,
        seed: int,
        min_generation_area: int = DEFAULT_MIN_AREA,
        max_sequence_length: int = 4096,
    ):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected a single nonempty prompt string")
        if not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096")
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        h, w = generation_canvas(height, width, min_generation_area)
        ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"][:max_sequence_length]
        text = self.text_encoder.encode(torch.tensor(ids, dtype=torch.long, device=self.device))
        ph, pw = h // 32, w // 32
        positions = text_image_positions(len(ids), ph, pw, self.device)
        context = self.transformer.prepare_static_context(text, positions)
        channels = self.transformer.config.in_channels
        noise = torch.randn(
            1,
            channels,
            ph * 2,
            pw * 2,
            generator=torch.Generator().manual_seed(seed),
            dtype=torch.float32,
            device="cpu",
        )
        rows = (
            noise.view(1, channels, ph, 2, pw, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(1, ph * pw, -1)
            .to(self.device)
        )
        timer.mark_denoise_start()
        for i, sigma in self._profile_denoise_steps(self.scheduler.sigmas[:-1]):
            v = self.transformer(rows, timestep=sigma, static_context=context, pdd_step_index=i)
            rows = self.scheduler.step(i, rows, v)
        timer.mark_post_start()
        latents = unpatchify_video_tokens(rows, 1, ph * 2, pw * 2, channels, (1, 2, 2)).squeeze(2)
        pixels = self.vae(latents).float()
        pixels = (
            pixels * pixels.new_tensor(MINIMAX_H3_PIXEL_STD)[None, :, None, None]
            + pixels.new_tensor(MINIMAX_H3_PIXEL_MEAN)[None, :, None, None]
        )
        if not pixels.isfinite().all():
            raise RuntimeError("Non-finite H3 image decoder output")
        rgb = (pixels.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        if (h, w) != (height, width):
            resized = Image.fromarray(rgb[0].cpu().numpy()).resize(
                (width, height), Image.Resampling.LANCZOS
            )
            rgb = torch.from_numpy(np.array(resized))[None].to(self.device)
        timer.mark_end()
        return timer.fill(PipelineOutput(image=rgb))
