# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2
"""LTX-2.3 ("V2") text-to-video pipeline (Phase-0: BF16, 1 GPU, eager, video-only).

This is a *separate* pipeline from ``LTX2Pipeline`` (it does not subclass it),
matching the reviewer guidance: the two share the module-level helpers and the
reusable native components, but LTX-2.3 diverges in ways that live inside the
text path and denoise loop:

* Text features: all 49 Gemma hidden states are stacked, **per-token RMS
  normalized** (``text_encoder_norm_type=per_token_rms``, not LTX-2's masked
  min-max), then projected by a **split** feature extractor
  (``video_aggregate_embed`` / ``audio_aggregate_embed``) before the connectors.
* The transformer needs a global ``sigma`` per step for the sigma-driven text
  cross-attention K/V modulation (``prompt_adaln_single`` /
  ``prompt_scale_shift_table``), so the text K/V cannot be pre-projected once
  like LTX-2 — it is (re)projected inside each denoise step.

Phase-0 scope (agreed): text-to-video only, plain CFG, BF16, single GPU, eager.
Audio VAE + BigVGAN-v2 BWE vocoder (48 kHz) and the optimization stack
(FP8/NVFP4, CUDA graph, torch.compile, Ulysses) are added in later phases.

Reused verbatim from LTX-2 (same numerical contract):
* ``self.denoise`` — CFG is applied in x0-space in LTX-2.3, which is
  algebraically identical to velocity-space CFG for rectified flow
  (x0 = sample - v*sigma is affine in v), so the base CFG combine is exact.
* ``self.decode_latents`` + tiled video VAE decode + ``postprocess_video_tensor``.
* The RoPE / patchifier / scheduler / shape utilities.
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger


def _ltx_bench_emit(msg: str) -> None:
    """Perf-only: surface a bench line regardless of logger routing.

    The pipeline runs inside the DiffusionClient worker, whose ``logger.info``
    output does not always reach the driver stdout. Print (flushed) is reliably
    captured, and we also append to ``LTX_BENCH_FILE`` when set so the A/B runner
    can read the numbers straight from a file.
    """
    print(msg, flush=True)
    path = os.environ.get("LTX_BENCH_FILE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(msg + "\n")
        except OSError:
            pass

# Reuse the LTX-2 native components + module-level loaders (shared utilities).
from ..ltx2.ltx2_core.audio_vae import AudioDecoderConfigurator, decode_audio
from ..ltx2.ltx2_core.patchifier import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from ..ltx2.ltx2_core.rope import LTXRopeType
from ..ltx2.ltx2_core.scheduler_adapter import NativeSchedulerAdapter
from ..ltx2.ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from ..ltx2.ltx2_core.video_vae import TilingConfig
from ..ltx2.pipeline_ltx2 import (
    _find_safetensors_files,
    _load_component_weights,
    _load_ltx2_transformer_weights,
    _prefetch_ltx2_safetensors_files,
    _read_safetensors_config,
)
from ..ltx2.transformer_ltx2 import LTXModelType
from .ltx23_core.connector import (
    LTX23AudioConnectorConfigurator,
    LTX23GemmaFeaturesExtractor,
    LTX23VideoConnectorConfigurator,
)
from .ltx23_core.video_vae_ltx23 import LTX23VideoDecoderConfigurator
from .ltx23_core.vocoder_ltx23 import LTX23VocoderConfigurator
from .ltx23_core.modality import LTX23Modality
from .transformer_ltx23 import LTX23Model

# Gemma-3-12b-it exposes 49 hidden states (48 layers + embeddings). The split
# feature extractor's [out, 3840*49] weights expect all of them stacked.
_NUM_GEMMA_HIDDEN_STATES = 49


@register_pipeline(
    "LTX23Pipeline",
    hf_ids=["Lightricks/LTX-2.3"],
    defaults={"text_encoder_path": "google/gemma-3-12b-it"},
    doc="Lightricks LTX-2.3 audio-video generation (Phase-0: video-only, BF16).",
)
class LTX23Pipeline(BasePipeline):
    """Text-to-video pipeline for the LTX-2.3 checkpoint (native single-file safetensors)."""

    # LTX-2.3 stores the transformer under this prefix; the connectors live
    # under the same prefix and are loaded as separate pipeline components, so
    # they are excluded from the transformer state dict.
    _TRANSFORMER_PREFIX = "model.diffusion_model."
    _TRANSFORMER_EXCLUDE_PREFIXES = [
        "audio_embeddings_connector.",
        "video_embeddings_connector.",
    ]

    def __init__(self, model_config):
        super().__init__(model_config)

    @property
    def dtype(self):
        return self.pipeline_config.torch_dtype

    @classmethod
    def resolve_variant(cls, config):
        # Phase-0: single-stage only (no spatial upsampler / distilled LoRA path).
        return cls

    @property
    def default_warmup_resolutions(self):
        return [(512, 768)]

    @property
    def default_warmup_num_frames(self):
        return [121]

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

    # ------------------------------------------------------------------
    # Transformer init + weight loading
    # ------------------------------------------------------------------

    def _init_transformer(self) -> None:
        attn_cfg = getattr(self.pipeline_config, "attention", None)
        if attn_cfg is not None and getattr(attn_cfg, "quant_attention_config", None) is not None:
            raise NotImplementedError("Quantized attention is not yet supported for LTX-2.3.")

        model_config = self.pipeline_config.model_configs["transformer"]
        cfg = model_config.pretrained_config

        rope_type = LTXRopeType(getattr(cfg, "rope_type", "interleaved"))
        double_precision_rope = getattr(cfg, "frequencies_precision", False) == "float64"
        apply_gated_attention = getattr(cfg, "apply_gated_attention", False)

        self.transformer_in_channels = getattr(cfg, "in_channels", 128)
        self.audio_in_channels = getattr(cfg, "audio_in_channels", 128)
        self.audio_out_channels = getattr(cfg, "audio_out_channels", 128)

        logger.info(
            f"LTX-2.3 transformer config: rope_type={rope_type.value}, "
            f"double_precision_rope={double_precision_rope}, "
            f"apply_gated_attention={apply_gated_attention} (AudioVideo)"
        )

        # Phase-1: AudioVideo. The dual-stream block + bidirectional A<->V
        # cross-attention are already implemented; enabling the audio stream
        # here makes the transformer emit (video_velocity, audio_velocity).
        self.transformer = LTX23Model(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=getattr(cfg, "num_attention_heads", 32),
            attention_head_dim=getattr(cfg, "attention_head_dim", 128),
            in_channels=self.transformer_in_channels,
            out_channels=getattr(cfg, "out_channels", 128),
            num_layers=getattr(cfg, "num_layers", 48),
            cross_attention_dim=getattr(cfg, "cross_attention_dim", 4096),
            audio_num_attention_heads=getattr(cfg, "audio_num_attention_heads", 32),
            audio_attention_head_dim=getattr(cfg, "audio_attention_head_dim", 64),
            audio_in_channels=self.audio_in_channels,
            audio_out_channels=self.audio_out_channels,
            audio_cross_attention_dim=getattr(cfg, "audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=getattr(
                cfg, "audio_positional_embedding_max_pos", [20]
            ),
            norm_eps=float(getattr(cfg, "norm_eps", 1e-6)),
            caption_channels=getattr(cfg, "caption_channels", 3840),
            positional_embedding_theta=float(getattr(cfg, "positional_embedding_theta", 10000.0)),
            positional_embedding_max_pos=getattr(
                cfg, "positional_embedding_max_pos", [20, 2048, 2048]
            ),
            timestep_scale_multiplier=getattr(cfg, "timestep_scale_multiplier", 1000),
            use_middle_indices_grid=getattr(cfg, "use_middle_indices_grid", True),
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=apply_gated_attention,
            model_config=model_config,
        )
        self.transformer._transformer_config = vars(cfg)

    def load_transformer_weights(self, checkpoint_dir: str) -> Dict[str, torch.Tensor]:
        logger.info("Loading LTX-2.3 transformer weights (native checkpoint)")
        return _load_ltx2_transformer_weights(
            checkpoint_dir,
            self._TRANSFORMER_PREFIX,
            exclude_prefixes=self._TRANSFORMER_EXCLUDE_PREFIXES,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            logger.info("Loading transformer weights...")
            self.transformer.load_weights(weights.get("transformer", weights))
            logger.info("Transformer weights loaded successfully.")

    # ------------------------------------------------------------------
    # Component loading
    # ------------------------------------------------------------------

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
        *,
        text_encoder_path: str = "",
        **kwargs,
    ) -> None:
        skip_components = skip_components or []
        dtype = self.pipeline_config.torch_dtype

        needs_text = (
            PipelineComponent.TOKENIZER not in skip_components
            or PipelineComponent.TEXT_ENCODER not in skip_components
        )
        if needs_text and not text_encoder_path:
            raise ValueError(
                "text_encoder_path is required for the Gemma3 tokenizer/encoder. "
                "Set the LTX-2.3 pipeline_config 'text_encoder_path' entry."
            )

        if PipelineComponent.TOKENIZER not in skip_components:
            logger.info(f"Loading tokenizer (Gemma3) from {text_encoder_path}...")
            self.tokenizer = GemmaTokenizerFast.from_pretrained(text_encoder_path)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        if PipelineComponent.TEXT_ENCODER not in skip_components:
            logger.info(f"Loading text encoder (Gemma3) from {text_encoder_path}...")
            self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                text_encoder_path,
                torch_dtype=dtype,
            ).to(device)

        native_config = self.pipeline_config.extra_attrs.get("monolithic_safetensors_config")
        sft_paths = _find_safetensors_files(checkpoint_dir)
        _prefetch_ltx2_safetensors_files(sft_paths)
        if native_config is None and sft_paths:
            native_config = _read_safetensors_config(sft_paths[0])
        if native_config is None:
            raise ValueError(
                "LTX-2.3 native checkpoint requires embedded 'config' metadata in the "
                f"safetensors file(s) at {checkpoint_dir}."
            )
        self._native_config = native_config

        self._load_native_components(native_config, sft_paths, device, dtype, skip_components)

        if PipelineComponent.SCHEDULER not in skip_components:
            self.scheduler = NativeSchedulerAdapter()

    def _load_native_components(
        self,
        config: Dict[str, Any],
        sft_paths: List[str],
        device: torch.device,
        dtype: torch.dtype,
        skip_components: Optional[list] = None,
    ) -> None:
        skip_components = skip_components or []

        # Video decoder (config-driven; the LTX-2 VAE configurator reads the
        # LTX-2.3 decoder_blocks recipe from the same native config).
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading native video decoder...")
            self.video_decoder = LTX23VideoDecoderConfigurator.from_config(config)
            _load_component_weights(sft_paths, self.video_decoder, ["vae.decoder.", "vae."])
            self.video_decoder = self.video_decoder.to(device=device, dtype=dtype)

        # Split feature extractor (video + audio projections) + video connector.
        # Phase-0 is video-only, so the audio connector is skipped; the audio
        # projection weights are still loaded (both keys live under
        # text_embedding_projection.*) and simply unused this phase.
        logger.info("Loading native text feature extractor + video connector...")
        self.feature_extractor = LTX23GemmaFeaturesExtractor.from_config(config)
        _load_component_weights(sft_paths, self.feature_extractor, "text_embedding_projection.")
        self.feature_extractor = self.feature_extractor.to(device=device, dtype=dtype)

        self.video_connector = LTX23VideoConnectorConfigurator.from_config(config)
        _load_component_weights(
            sft_paths,
            self.video_connector,
            "model.diffusion_model.video_embeddings_connector.",
        )
        self.video_connector = self.video_connector.to(device=device, dtype=dtype)

        # Audio connector (32 x 64 = 2048, 8 layers, gated) consuming its own
        # audio projection from the split feature extractor.
        self.audio_connector = LTX23AudioConnectorConfigurator.from_config(config)
        _load_component_weights(
            sft_paths,
            self.audio_connector,
            "model.diffusion_model.audio_embeddings_connector.",
        )
        self.audio_connector = self.audio_connector.to(device=device, dtype=dtype)

        # Audio VAE decoder (reused verbatim from LTX-2; config-driven and
        # matches the LTX-2.3 audio_vae recipe). Latent -> mel spectrogram.
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading native audio decoder...")
            self.audio_decoder = AudioDecoderConfigurator.from_config(config)
            _load_component_weights(
                sft_paths, self.audio_decoder, ["audio_vae.decoder.", "audio_vae."]
            )
            self.audio_decoder = self.audio_decoder.to(device=device, dtype=dtype)

            # Vocoder (BigVGAN-v2 AMP1 + BWE -> 48 kHz). Kept in fp32: the BWE
            # forward runs fp32 regardless, and bf16 accumulation degrades the
            # spectral output across the ~108 sequential convs.
            logger.info("Loading native vocoder (BigVGAN-v2 + BWE)...")
            self.vocoder = LTX23VocoderConfigurator.from_config(config)
            _load_component_weights(sft_paths, self.vocoder, "vocoder.")
            self.vocoder = self.vocoder.to(device=device, dtype=torch.float32)

        # Patchifier (structural, no weights).
        t_cfg = self.transformer._transformer_config
        patch_size = t_cfg.get("patch_size", 1)
        self.video_patchifier = VideoLatentPatchifier(patch_size=patch_size)

        # Audio decode-side properties (mirror LTX-2). The audio VAE decoder
        # exposes the patchifier and the mel/sample-rate metadata; the *output*
        # sample rate is the vocoder's (48 kHz via BWE), not the VAE's 16 kHz.
        # These are only meaningful when the audio decoder was loaded (skipped
        # alongside the VAE component in transformer-only loads / unit tests).
        if PipelineComponent.VAE not in skip_components:
            self.audio_patchifier = self.audio_decoder.patchifier
            self.audio_sampling_rate = self.audio_decoder.sample_rate
            self.audio_hop_length = self.audio_decoder.mel_hop_length
            self.audio_mel_bins = self.audio_decoder.mel_bins

    # ------------------------------------------------------------------
    # Text encoding (LTX-2.3: per-token RMS over stacked Gemma states)
    # ------------------------------------------------------------------

    @staticmethod
    def _per_token_rms_pack(hidden_states: List[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
        """Stack all Gemma hidden states, per-token RMS normalize, flatten.

        ``hidden_states``: tuple of ``num_layers+1`` tensors ``[B, S, 3840]``.
        Returns ``[B, S, 3840 * num_states]`` to feed the split feature
        extractor's ``[out, 3840*49]`` projections.

        LTX-2.3 uses ``text_encoder_norm_type='per_token_rms'`` with
        ``caption_proj_input_norm=False`` (the norm is applied here, before the
        projection, not inside caption projection). We normalize each token's
        hidden vector per layer by its RMS over the hidden dim — the placement
        the vLLM-Omni reference documents (unflatten -> per_token_rms -> project).

        NOTE (Phase-0 validation target): the exact reduction axis for
        per_token_rms (per-layer hidden dim vs. flattened across all layers) is
        the first thing to check against a reference render if colors/structure
        look off. Isolated here for that reason.
        """
        stacked = torch.stack(hidden_states, dim=-1)  # [B, S, 3840, num_states]
        # RMS over the hidden dim (3840), per token and per layer.
        rms = torch.sqrt(stacked.float().pow(2).mean(dim=2, keepdim=True) + eps)
        normed = (stacked.float() / rms).to(dtype=stacked.dtype)
        return normed.flatten(2, 3)  # [B, S, 3840 * num_states]

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        prompt = [p.strip() for p in prompt]

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_attention_mask = text_inputs.attention_mask.to(self.device)

        outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = self._per_token_rms_pack(outputs.hidden_states)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)
        return prompt_embeds, prompt_attention_mask

    def _process_connectors(self, prompt_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """Split feature extractor -> video + audio connectors.

        Returns ``(video_embeds, audio_embeds, video_mask)``. The connector
        (register-augmented) mask is shared across modalities in LTX-2.3.
        """
        additive_mask = (1 - attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        additive_mask = additive_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        video_proj, audio_proj = self.feature_extractor(prompt_embeds)
        video_embeds, video_mask = self.video_connector(video_proj, additive_mask)
        audio_embeds, _ = self.audio_connector(audio_proj, additive_mask)
        return video_embeds, audio_embeds, video_mask

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def default_generation_params(self):
        return {
            "height": 512,
            "width": 768,
            "num_inference_steps": 40,
            "guidance_scale": 4.0,
            "max_sequence_length": 1024,
            "num_frames": 121,
            "frame_rate": 24.0,
        }

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
        pipeline_start = time.time()
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        do_cfg = guidance_scale > 1.0

        # ---- 1. Encode prompts -----------------------------------------
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

        # ---- 2. Connectors (feature extractor + video/audio connectors) -
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

        # ---- 3. Latent shape + noise -----------------------------------
        logger.info("Preparing latents...")
        pixel_shape = VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=frame_rate
        )
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape, latent_channels=self.transformer_in_channels
        )
        # Audio latent grid is derived from the same pixel timeline so the two
        # streams stay temporally aligned (mel bins are pre-patch, hence // 4).
        audio_shape = AudioLatentShape.from_video_pixel_shape(
            pixel_shape,
            channels=getattr(self.audio_decoder, "z_channels", 8),
            mel_bins=self.audio_mel_bins // 4,
            sample_rate=self.audio_sampling_rate,
            hop_length=self.audio_hop_length,
        )
        self.transformer.configure_audio_ulysses(audio_shape.frames)

        latents = torch.randn(
            video_shape.to_torch_shape(),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        latents = self.video_patchifier.patchify(latents)

        audio_latents = torch.randn(
            audio_shape.to_torch_shape(),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        audio_latents = self.audio_patchifier.patchify(audio_latents)

        # ---- 4. Position embeddings (RoPE) -----------------------------
        video_positions = self.video_patchifier.get_patch_grid_bounds(
            video_shape, device=self.device
        )
        video_positions = get_pixel_coords(
            video_positions.float(), VIDEO_SCALE_FACTORS, causal_fix=True
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate
        video_positions = video_positions.to(self.dtype)
        audio_positions = self.audio_patchifier.get_patch_grid_bounds(
            audio_shape, device=self.device
        )

        # ---- 5. Scheduler (video + lockstep audio) ---------------------
        # Both streams share one sigma schedule (derived from the video latent)
        # and step together; the deep copy keeps an independent step index.
        latents_5d = torch.randn(video_shape.to_torch_shape(), device=self.device)
        self.scheduler.set_timesteps(num_inference_steps, latent=latents_5d)
        audio_scheduler = copy.deepcopy(self.scheduler)
        audio_scheduler.set_timesteps(num_inference_steps, latent=latents_5d)
        timesteps = self.scheduler.timesteps

        # ---- 6. Text cache (sigma-independent; no static K/V) ----------
        # Batched CFG: cache is [neg, cond] to align with the base denoise
        # loop's [uncond, cond] latent stacking.
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

        # ---- 7. Joint (video + audio) denoising loop -------------------
        def _run_transformer(v_latents, a_latents, timestep_val, v_context, a_context, mask):
            v_latents_f32 = v_latents.float()
            v_latents_bf = v_latents.to(self.dtype)
            a_latents_f32 = a_latents.float()
            a_latents_bf = a_latents.to(self.dtype)

            video_mod = LTX23Modality(
                latent=v_latents_bf,
                timesteps=timestep_val,
                sigma=timestep_val,  # LTX-2.3: global current sigma drives prompt_adaln
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
            sigma = timestep_val.float()
            while sigma.dim() < vel_v.dim():
                sigma = sigma.unsqueeze(-1)
            dn_v = v_latents_f32 - vel_v.float() * sigma

            sigma_a = timestep_val.float()
            while sigma_a.dim() < vel_a.dim():
                sigma_a = sigma_a.unsqueeze(-1)
            dn_a = a_latents_f32 - vel_a.float() * sigma_a
            return dn_v, dn_a

        # Perf-only: accumulate transformer (DiT) GPU time per forward call so we
        # can isolate it from scheduler/CFG-combine cost. Guarded by LTX_BENCH=1.
        _bench = os.environ.get("LTX_BENCH", "0") == "1"
        _bench_events: List[tuple] = []
        if _bench:
            _ltx_bench_emit(
                f"[LTX_BENCH] enabled (steps={len(timesteps)}, "
                f"FREEZE_TEXT_KV={os.environ.get('LTX23_FREEZE_TEXT_KV', '0')})"
            )

        def forward_fn(
            video_latents,
            extra_stream_latents,
            step_index,
            timestep,
            encoder_hidden_states,
            extra_tensors,
        ):
            if _bench and torch.cuda.is_available():
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                dn_v, dn_a = _run_transformer(
                    video_latents,
                    extra_stream_latents["audio"],
                    timestep,
                    encoder_hidden_states,
                    extra_tensors.get("audio_embeds", audio_embeds),
                    extra_tensors.get("attention_mask", connector_mask),
                )
                ev1.record()
                _bench_events.append((ev0, ev1))
                return dn_v, {"audio": dn_a}
            dn_v, dn_a = _run_transformer(
                video_latents,
                extra_stream_latents["audio"],
                timestep,
                encoder_hidden_states,
                extra_tensors.get("audio_embeds", audio_embeds),
                extra_tensors.get("attention_mask", connector_mask),
            )
            return dn_v, {"audio": dn_a}

        # Perf-only A/B: re-prime the frozen text-K/V cache each generation so it
        # rebuilds on step 0 (see LTX23_FREEZE_TEXT_KV in transformer_ltx23.py).
        if os.environ.get("LTX23_FREEZE_TEXT_KV", "0") == "1" and hasattr(
            self.transformer, "reset_text_kv_cache"
        ):
            self.transformer.reset_text_kv_cache()

        if _bench:
            _ltx_bench_emit(
                f"LTX_BENCH workload: video_latents={tuple(latents.shape)} "
                f"audio_latents={tuple(audio_latents.shape)} "
                f"do_cfg={do_cfg} (LTX23)"
            )

        timer.mark_denoise_start()
        if _bench and torch.cuda.is_available():
            torch.cuda.synchronize()
            _denoise_t0 = time.time()
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

        if _bench and torch.cuda.is_available():
            torch.cuda.synchronize()
            _denoise_dt = time.time() - _denoise_t0
            _n_steps = len(timesteps)
            _freeze = os.environ.get("LTX23_FREEZE_TEXT_KV", "0") == "1"
            # Authoritative denoise-phase wall time (includes scheduler + CFG).
            _ltx_bench_emit(f"Denoising done: {_denoise_dt:.4f}s")
            _ltx_bench_emit(
                f"LTX_BENCH denoise: {_denoise_dt:.4f}s over {_n_steps} steps "
                f"= {_denoise_dt / max(_n_steps, 1):.4f}s/step "
                f"(FREEZE_TEXT_KV={int(_freeze)})"
            )
            if _bench_events:
                _trans_ms = sum(e0.elapsed_time(e1) for e0, e1 in _bench_events)
                _calls = len(_bench_events)
                _ltx_bench_emit(
                    f"LTX_BENCH transformer: {_trans_ms / 1000:.4f}s total over "
                    f"{_calls} forward calls = {_trans_ms / max(_calls, 1):.2f}ms/call "
                    f"(FREEZE_TEXT_KV={int(_freeze)})"
                )

        timer.mark_post_start()

        # ---- 8. Decode video + audio -----------------------------------
        logger.info("Decoding video and audio...")
        decode_start = time.time()

        def decode_video_fn(vid_latents):
            vid_latents = self.video_patchifier.unpatchify(vid_latents, video_shape)
            if output_type == "latent":
                return vid_latents
            vid_latents = vid_latents.to(self.dtype)
            tiling_config = TilingConfig.default()
            chunks = list(
                self.video_decoder.tiled_decode(vid_latents, tiling_config, generator=generator)
            )
            video = torch.cat(chunks, dim=2)
            return postprocess_video_tensor(video)

        def decode_audio_fn(aud_latents):
            aud_latents = self.audio_patchifier.unpatchify(aud_latents, audio_shape)
            if output_type == "latent":
                return aud_latents
            # Audio VAE runs in the model dtype; the vocoder/BWE stay fp32.
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

        if self.rank == 0:
            logger.info(f"Decoding completed in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

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
