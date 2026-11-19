# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import copy
import hashlib
import math
import os
import weakref
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from ..ltx2.ltx2_core.audio_vae import AudioDecoderConfigurator, decode_audio
from ..ltx2.ltx2_core.patchifier import VideoLatentPatchifier, get_pixel_coords
from ..ltx2.ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from ..ltx2.ltx2_core.video_vae import TilingConfig
from ..ltx2.parallel_vae import tile_parallel_decode
from ..ltx2.pipeline_ltx2 import LTX2Pipeline, _load_component_weights, _LTX2CUDAGraphRunner
from ..ltx2.text_cache import TextCache
from .ltx23_core.audio_vae import LTX23VocoderConfigurator
from .ltx23_core.connector import (
    LTX23AudioConnectorConfigurator,
    LTX23GemmaFeaturesExtractor,
    LTX23VideoConnectorConfigurator,
)
from .ltx23_core.modality import LTX23Modality
from .ltx23_core.video_vae import LTX23VideoDecoderConfigurator
from .transformer_ltx23 import LTX23Model

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig


class _LTX23CUDAGraphRunner(_LTX2CUDAGraphRunner):
    """Shared generation/retake graph runner for LTX23Modality and TextCache.

    Tensor metadata keys dynamic latents, sigma, timesteps and prompt tensors.
    Padding-mask contents also key the immutable FlashInfer plans. Equal masks
    and differently valued prompts share a graph; replay copies all tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mask_fingerprints = {}

    def _key_parts_for(self, prefix, value):
        if isinstance(value, (LTX23Modality, TextCache)):
            for field in fields(value):
                tensor = getattr(value, field.name)
                label = f"{prefix}.{field.name}"
                yield from self._key_parts_for(label, tensor)
                if field.name.endswith("mask") and isinstance(tensor, torch.Tensor):
                    cached = self._mask_fingerprints.get(label)
                    # Inference tensors have no version counter, so their small
                    # masks are re-read outside capture to avoid stale plans.
                    version = None if torch.is_inference(tensor) else tensor._version
                    if (
                        version is None
                        or cached is None
                        or cached[0]() is not tensor
                        or cached[1] != version
                    ):
                        data = (
                            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        )
                        cached = (weakref.ref(tensor), version, hashlib.sha256(data).hexdigest())
                        self._mask_fingerprints[label] = cached
                    yield (f"{label}.contents", cached[2])
            return
        if isinstance(value, torch.Tensor):
            yield (prefix, (tuple(value.shape), str(value.dtype), str(value.device)))
            return
        if isinstance(value, (tuple, list)):
            yield (f"{prefix}.length", len(value))
            for index, item in enumerate(value):
                yield from self._key_parts_for(f"{prefix}.{index}", item)
            return
        if isinstance(value, bool):
            yield (prefix, value)
            return
        yield from super()._key_parts_for(prefix, value)

    def capture(self, key, fn, args, kwargs):
        outputs = []

        def remember_outputs(*args, **kwargs):
            output = fn(*args, **kwargs)
            outputs[:] = [output]
            return output

        super().capture(key, remember_outputs, args, kwargs)
        # Outputs of distinct mask/shape graphs must not alias freed storage.
        self._graph_output_refs[key] = outputs[0]

    @staticmethod
    def _clone_value(value):
        cloned = _LTX2CUDAGraphRunner._clone_value(value)
        if isinstance(value, LTX23Modality):
            return replace(
                cloned,
                sigma=value.sigma.clone(),
                cross_modality_sigma=(
                    value.cross_modality_sigma.clone()
                    if value.cross_modality_sigma is not None
                    else None
                ),
            )
        return cloned

    @staticmethod
    def _copy_value(dst, src):
        copied = _LTX2CUDAGraphRunner._copy_value(dst, src)
        if isinstance(src, LTX23Modality) and isinstance(dst, LTX23Modality):
            dst.sigma.copy_(src.sigma)
            if dst.cross_modality_sigma is not None and src.cross_modality_sigma is not None:
                dst.cross_modality_sigma.copy_(src.cross_modality_sigma)
        return copied


def detect_native_ltx_pipeline(config: dict) -> str:
    """Return LTX2Pipeline or LTX23Pipeline from native safetensors config."""
    ltx23_text_config = {
        "caption_proj_before_connector": True,
        "caption_projection_first_linear": False,
        "caption_projection_second_linear": False,
        "caption_proj_input_norm": False,
    }
    transformer = config.get("transformer", {})
    expected = set(ltx23_text_config)
    present = expected.intersection(transformer)

    if not present:
        if transformer.get("cross_attention_adaln", False):
            raise ValueError(
                "Ambiguous LTX checkpoint: cross_attention_adaln=True without "
                "the LTX-2.3 text-projection configuration."
            )
        return "LTX2Pipeline"

    missing = expected - set(transformer)
    mismatched = {
        k: transformer[k]
        for k, v in ltx23_text_config.items()
        if k in transformer and transformer[k] != v
    }
    if missing or mismatched:
        raise ValueError(
            "Unsupported/partial LTX text-projection config: "
            f"missing={sorted(missing)}, mismatched={mismatched}"
        )
    if transformer.get("cross_attention_adaln") is not True:
        raise ValueError("LTX-2.3 text projection found without cross_attention_adaln=True.")
    return "LTX23Pipeline"


@register_pipeline(
    "LTX23Pipeline",
    hf_ids=["Lightricks/LTX-2.3"],
    defaults={
        "text_encoder_path": "google/gemma-3-12b-it-qat-q4_0-unquantized",
        "workflow": "generation",
    },
    doc="Lightricks LTX-2.3 text-to-video generation and retake with audio.",
)
class LTX23Pipeline(BasePipeline):
    """Text-to-video pipeline for the LTX-2.3 checkpoint (native single-file safetensors)."""

    _supports_quantized_attention = True

    # Connectors share the transformer prefix but load as separate components.
    _TRANSFORMER_PREFIX = LTX2Pipeline._TRANSFORMER_PREFIX
    _TRANSFORMER_EXCLUDE_PREFIXES = LTX2Pipeline._TRANSFORMER_EXCLUDE_PREFIXES

    def __init__(self, pipeline_config: "DiffusionPipelineConfig") -> None:
        if pipeline_config.parallel.cfg_size != 1:
            raise NotImplementedError("LTX-2.3 generation requires parallel.cfg_size=1.")
        super().__init__(pipeline_config)

    @classmethod
    def resolve_variant(cls, config):
        workflow = config.extra_attrs.get("workflow", "generation")
        if workflow == "retake":
            from .pipeline_ltx23_retake import LTX23RetakePipeline

            return LTX23RetakePipeline
        if workflow != "generation":
            raise ValueError(f"Unsupported LTX-2.3 workflow: {workflow!r}")
        return cls

    default_warmup_resolutions = LTX2Pipeline.default_warmup_resolutions
    default_warmup_num_frames = LTX2Pipeline.default_warmup_num_frames
    default_generation_params = LTX2Pipeline.default_generation_params
    load_transformer_weights = LTX2Pipeline.load_transformer_weights
    load_weights = LTX2Pipeline.load_weights
    load_standard_components = LTX2Pipeline.load_standard_components
    setup_parallel_vae = LTX2Pipeline.setup_parallel_vae
    _encode_prompt = LTX2Pipeline._encode_prompt
    _extra_transformer_kwargs = LTX2Pipeline._extra_transformer_kwargs
    _setup_cuda_graphs = LTX2Pipeline._setup_cuda_graphs
    _transformer_class = LTX23Model
    _cuda_graph_runner_class = _LTX23CUDAGraphRunner

    def _run_warmup(self, height: int, width: int, num_frames: int, steps: int) -> None:
        self.forward(
            prompt="warmup",
            negative_prompt="",
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=4.0,
            seed=42,
        )

    def warmup(self) -> None:
        """Tune with capture off, then capture from the warm tactic cache.

        A single pass would record autotuner torch.rand kernels into the graph.
        """
        if not (
            self.pipeline_config.torch_compile.enable_autotune
            and self.pipeline_config.cuda_graph.enable
            and self.world_size == 1
        ):
            super().warmup()
            return

        shapes, steps = self.resolve_warmup_plan()
        if not shapes:
            super().warmup()
            return

        shape_list = ", ".join(f"{h}x{w}x{f}" for h, w, f in shapes)
        logger.info(
            f"Running warmup for {self.__class__.__name__}: {len(shapes)} shape(s) "
            f"[{shape_list}], {steps} steps, tuning pass then capture pass..."
        )
        self._is_warmup = True
        try:
            with (
                self.disallow_cuda_graph_capture(),
                autotune(
                    cache_path=os.environ.get("TLLM_AUTOTUNER_CACHE_PATH"),
                    skip_dynamic_tuning_buckets=True,
                ),
            ):
                self._run_warmup_pass(shapes, steps)
            self._run_warmup_pass(shapes, steps)
        finally:
            self._is_warmup = False

        self._warmed_up_shapes = set(
            self.warmup_cache_key(h, w, num_frames=f) for h, w, f in shapes
        )

    def _init_transformer(self) -> None:
        model_config = self.pipeline_config.model_configs["transformer"]
        attention = model_config.attention
        quant_attention = attention.quant_attention_config
        if (
            attention.backend == "FLASHINFER"
            and quant_attention is not None
            and quant_attention.qk_dtype in ("mxfp8", "nvfp4")
            and quant_attention.v_dtype == "fp8"
        ):
            if model_config.attention_metadata_state is None:
                model_config.attention_metadata_state = {}
            model_config.attention_metadata_state["_ltx23_static_global_scale"] = 1.0
        LTX2Pipeline._init_transformer(self)
        self.transformer_in_channels = self.transformer._transformer_config.get("in_channels", 128)

    def _load_native_components(
        self,
        config: Dict[str, Any],
        sft_paths: List[str],
        device: torch.device,
        dtype: torch.dtype,
        skip_components: Optional[list] = None,
    ) -> None:
        skip_components = skip_components or []

        def _load(module, prefixes, load_dtype):
            _load_component_weights(sft_paths, module, prefixes)
            return module.to(device=device, dtype=load_dtype)

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading native video decoder...")
            self.video_decoder = _load(
                LTX23VideoDecoderConfigurator.from_config(config),
                ["vae.decoder.", "vae."],
                dtype,
            )

        if "connectors" not in skip_components:
            logger.info("Loading native text feature extractor + connectors...")
            self.feature_extractor = _load(
                LTX23GemmaFeaturesExtractor.from_config(config),
                "text_embedding_projection.",
                dtype,
            )
            self.video_connector = _load(
                LTX23VideoConnectorConfigurator.from_config(config),
                "model.diffusion_model.video_embeddings_connector.",
                dtype,
            )
            self.audio_connector = _load(
                LTX23AudioConnectorConfigurator.from_config(config),
                "model.diffusion_model.audio_embeddings_connector.",
                dtype,
            )

        if PipelineComponent.VAE not in skip_components and "audio_decoder" not in skip_components:
            logger.info("Loading native audio decoder...")
            self.audio_decoder = _load(
                AudioDecoderConfigurator.from_config(config),
                ["audio_vae.decoder.", "audio_vae."],
                dtype,
            )
        if PipelineComponent.VAE not in skip_components and "vocoder" not in skip_components:
            logger.info("Loading native vocoder (BigVGAN-v2 + BWE)...")
            self.vocoder = _load(
                LTX23VocoderConfigurator.from_config(config), "vocoder.", torch.float32
            )

        patch_size = self.transformer._transformer_config.get("patch_size", 1)
        self.video_patchifier = VideoLatentPatchifier(patch_size=patch_size)

        if PipelineComponent.VAE not in skip_components and hasattr(self, "audio_decoder"):
            self.audio_patchifier = self.audio_decoder.patchifier
            self.audio_sampling_rate = self.audio_decoder.sample_rate
            self.audio_hop_length = self.audio_decoder.mel_hop_length
            self.audio_mel_bins = self.audio_decoder.mel_bins

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        device: Union[str, torch.device],
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """LTX-2.3 packs Gemma states with per-token RMS normalization."""
        variance = text_hidden_states.pow(2).mean(dim=2, keepdim=True)
        normalized = (text_hidden_states * torch.rsqrt(variance + eps)).flatten(2, 3)
        positions = torch.arange(text_hidden_states.shape[1], device=text_hidden_states.device)
        if padding_side == "left":
            mask = positions[None, :] >= text_hidden_states.shape[1] - sequence_lengths[:, None]
        elif padding_side == "right":
            mask = positions[None, :] < sequence_lengths[:, None]
        else:
            raise ValueError(f"Unsupported text padding side: {padding_side!r}")
        return torch.where(mask[..., None], normalized, torch.zeros_like(normalized))

    def _process_connectors(self, prompt_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Feature extractor, then video and audio connectors. Mask is shared."""
        additive_mask = (1 - attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        additive_mask = additive_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        video_proj, audio_proj = self.feature_extractor(prompt_embeds)
        video_embeds, video_mask = self.video_connector(video_proj, additive_mask)
        audio_embeds, _ = self.audio_connector(audio_proj, additive_mask)
        return video_embeds, audio_embeds, video_mask

    @property
    def extra_param_specs(self):
        return {
            "output_type": ExtraParamSchema(
                type="str",
                default="pt",
                description="Output type: 'pt' for tensors, 'latent' for raw latents.",
            ),
            "guidance_rescale": ExtraParamSchema(
                type="float",
                default=0.0,
                description="Guidance rescale factor to prevent overexposure.",
            ),
        }

    def infer(self, req):
        extra = req.params.extra_params or {}
        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.params.negative_prompt,
            height=req.params.height,
            width=req.params.width,
            num_frames=req.params.num_frames,
            frame_rate=req.params.frame_rate,
            num_inference_steps=req.params.num_inference_steps,
            guidance_scale=req.params.guidance_scale,
            seed=req.params.seed,
            output_type=extra.get("output_type", "pt"),
            guidance_rescale=extra.get("guidance_rescale", 0.0),
            max_sequence_length=req.params.max_sequence_length,
        )

    def _decode_video(
        self,
        latents: torch.Tensor,
        video_shape: VideoLatentShape,
        tiling_config: TilingConfig,
        generator: torch.Generator,
        output_type: str = "pt",
    ) -> torch.Tensor:
        """Unpatchify and decode video latents with the supplied tiling geometry."""
        latents = self.video_patchifier.unpatchify(latents, video_shape)
        if output_type == "latent":
            return latents
        if self._parallel_vae_enabled:
            video = tile_parallel_decode(
                self.video_decoder,
                latents.to(self.dtype),
                tiling_config,
                pg=self.pipeline_config.visual_gen_mapping.vae_group,
                generator=generator,
            )
        else:
            chunks = self.video_decoder.tiled_decode(
                latents.to(self.dtype), tiling_config, generator=generator
            )
            video = torch.cat(list(chunks), dim=2)
        return postprocess_video_tensor(video)

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        seed: int,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        output_type: str = "pt",
        max_sequence_length: int = 1024,
    ):
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        do_cfg = guidance_scale > 1.0

        logger.info("Encoding prompts...")
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length
        )
        neg_prompt_embeds = neg_prompt_attention_mask = None
        if do_cfg:
            negative_prompt = negative_prompt or ""
            neg_prompt_embeds, neg_prompt_attention_mask = self._encode_prompt(
                negative_prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length
            )

        if do_cfg:
            combined_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat([neg_prompt_attention_mask, prompt_attention_mask], dim=0)
            video_embeds_combined, audio_embeds_combined, connector_mask_combined = (
                self._process_connectors(combined_embeds, combined_mask)
            )
            neg_video_embeds, video_embeds = video_embeds_combined.chunk(2, dim=0)
            neg_audio_embeds, audio_embeds = audio_embeds_combined.chunk(2, dim=0)
            neg_connector_mask, connector_mask = connector_mask_combined.chunk(2, dim=0)
        else:
            video_embeds, audio_embeds, connector_mask = self._process_connectors(
                prompt_embeds, prompt_attention_mask
            )
            neg_video_embeds = neg_audio_embeds = neg_connector_mask = None

        logger.info("Preparing latents...")
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=frame_rate
        )
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=self.transformer_in_channels
        )
        # Mel bins are pre-patch, so divide by 4 to align with the video timeline.
        audio_shape = AudioLatentShape.from_video_pixel_shape(
            pixel_shape,
            channels=getattr(self.audio_decoder, "z_channels", 8),
            mel_bins=self.audio_mel_bins // 4,
            sample_rate=self.audio_sampling_rate,
            hop_length=self.audio_hop_length,
        )
        self.transformer.configure_audio_ulysses(audio_shape.frames)

        latents = torch.randn(
            (
                video_shape.batch,
                self.video_patchifier.get_token_count(video_shape),
                video_shape.channels * math.prod(self.video_patchifier.patch_size),
            ),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        audio_latents = torch.randn(
            (audio_shape.batch, audio_shape.frames, audio_shape.channels * audio_shape.mel_bins),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        video_positions = self.video_patchifier.get_patch_grid_bounds(
            video_shape, device=self.device
        )
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate
        audio_positions = self.audio_patchifier.get_patch_grid_bounds(
            audio_shape, device=self.device
        )

        # Shared sigma schedule; deepcopy so each stream has its own step index.
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.sigmas = self.scheduler.sigmas.to(self.device)
        audio_scheduler = copy.deepcopy(self.scheduler)
        timesteps = self.scheduler.timesteps

        if do_cfg:
            v_ctx = torch.cat([neg_video_embeds, video_embeds])
            a_ctx = torch.cat([neg_audio_embeds, audio_embeds])
            v_mask = (
                torch.cat([neg_connector_mask, connector_mask])
                if connector_mask is not None
                else None
            )
        else:
            v_ctx, a_ctx, v_mask = video_embeds, audio_embeds, connector_mask

        text_cache = self.transformer.prepare_text_cache(
            video_context=v_ctx,
            video_context_mask=v_mask,
            video_positions=video_positions,
            audio_context=a_ctx,
            audio_context_mask=v_mask,
            audio_positions=audio_positions,
            dtype=self.dtype,
        )

        def _run_transformer(v_latents, a_latents, timestep_val, v_context, a_context, mask):
            v_latents_f32 = v_latents.float()
            v_latents_bf = v_latents.to(self.dtype)
            a_latents_f32 = a_latents.float()
            a_latents_bf = a_latents.to(self.dtype)

            video_mod = LTX23Modality(
                latent=v_latents_bf,
                timesteps=timestep_val,
                sigma=timestep_val,
                positions=video_positions,
                context=v_context,
                context_mask=mask,
            )
            audio_mod = LTX23Modality(
                latent=a_latents_bf,
                timesteps=timestep_val,
                sigma=timestep_val,
                positions=audio_positions,
                context=a_context,
                context_mask=mask,
            )
            vel_v, vel_a = self.transformer(
                video=video_mod,
                audio=audio_mod,
                text_cache=text_cache,
                timestep=timestep_val.new_tensor(0.0),
            )

            # x0 prediction (rectified flow): x0 = sample - v * sigma.
            def to_x0(latent, velocity):
                sigma = timestep_val.float()
                while sigma.dim() < velocity.dim():
                    sigma = sigma.unsqueeze(-1)
                return (latent - velocity.float() * sigma).to(self.dtype)

            return to_x0(v_latents_f32, vel_v), to_x0(a_latents_f32, vel_a)

        def forward_fn(
            video_latents,
            extra_stream_latents,
            step_index,
            timestep,
            encoder_hidden_states,
            extra_tensors,
        ):
            dn_v, dn_a = _run_transformer(
                video_latents,
                extra_stream_latents["audio"],
                timestep,
                encoder_hidden_states,
                extra_tensors.get("audio_embeds", audio_embeds),
                extra_tensors.get("attention_mask", connector_mask),
            )
            return dn_v, {"audio": dn_a}

        timer.mark_denoise_start()
        result = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=video_embeds,
            neg_prompt_embeds=neg_video_embeds,
            guidance_scale=guidance_scale,
            forward_fn=forward_fn,
            timesteps=timesteps,
            guidance_rescale=guidance_rescale,
            extra_cfg_tensors=(
                {
                    "audio_embeds": (audio_embeds, neg_audio_embeds),
                    "attention_mask": (connector_mask, neg_connector_mask),
                }
                if do_cfg
                else None
            ),
            extra_streams={"audio": (audio_latents, audio_scheduler)},
        )
        latents, extra_stream_latents = result
        audio_latents = extra_stream_latents["audio"]

        timer.mark_post_start()

        logger.info("Decoding video and audio...")

        def decode_video_fn(vid_latents):
            return self._decode_video(
                vid_latents, video_shape, TilingConfig.default(), generator, output_type
            )

        def decode_audio_fn(aud_latents):
            aud_latents = self.audio_patchifier.unpatchify(aud_latents, audio_shape)
            if output_type == "latent":
                return aud_latents
            aud_latents = aud_latents.to(self.dtype)
            return decode_audio(aud_latents, self.audio_decoder, self.vocoder)

        decoded = self.decode_latents(
            latents=latents,
            decode_fn=decode_video_fn,
            extra_latents={"audio": (audio_latents, decode_audio_fn)},
        )
        if isinstance(decoded, (tuple, list)):
            video, audio = decoded[0], decoded[1]
        else:
            video, audio = decoded, None

        timer.mark_end()
        return timer.fill(
            PipelineOutput(
                video=video,
                audio=audio,
                frame_rate=float(frame_rate),
                audio_sample_rate=(
                    int(self.vocoder.output_sampling_rate) if audio is not None else None
                ),
            )
        )
