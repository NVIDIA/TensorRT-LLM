# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import copy
import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline, ExtraParamSchema
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent, register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

# Reuse the LTX-2 native components and module-level loaders (shared utilities).
from ..ltx2.ltx2_core.audio_vae import AudioDecoderConfigurator, decode_audio
from ..ltx2.ltx2_core.patchifier import VideoLatentPatchifier, get_pixel_coords
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
    _LTX2CUDAGraphRunner,
    _prefetch_ltx2_safetensors_files,
    _read_safetensors_config,
)
from ..ltx2.transformer_ltx2 import LTXModelType
from .ltx23_core.connector import (
    LTX23AudioConnectorConfigurator,
    LTX23GemmaFeaturesExtractor,
    LTX23VideoConnectorConfigurator,
)
from .ltx23_core.modality import LTX23Modality
from .ltx23_core.video_vae_ltx23 import LTX23VideoDecoderConfigurator
from .ltx23_core.vocoder_ltx23 import LTX23VocoderConfigurator
from .text_conditioning_ltx23 import LTX23TextConditioning
from .transformer_ltx23 import LTX23Model


class _LTX23CUDAGraphRunner(_LTX2CUDAGraphRunner):
    """CUDAGraphRunner extended for LTX-2.3's dataclass-based transformer inputs.

    LTX-2.3 passes tensors inside LTX23Modality and LTX23TextConditioning, so
    those tensors are cloned into static buffers at capture time and copied
    in-place at replay time. sigma is included because it changes on every
    denoise step and drives the prompt K/V modulation.
    """

    @staticmethod
    def _tensor_pair_shapes(pair):
        if pair is None:
            return None
        return tuple(tuple(t.shape) for t in pair)

    def _key_parts_for(self, prefix, value):
        if isinstance(value, LTX23Modality):
            yield (f"{prefix}.latent", tuple(value.latent.shape))
            yield (f"{prefix}.timesteps", tuple(value.timesteps.shape))
            yield (f"{prefix}.sigma", tuple(value.sigma.shape))
            yield (f"{prefix}.positions", tuple(value.positions.shape))
            yield (f"{prefix}.context", tuple(value.context.shape))
            yield (f"{prefix}.enabled", value.enabled)
            yield (
                f"{prefix}.context_mask",
                tuple(value.context_mask.shape) if value.context_mask is not None else None,
            )
            return
        if isinstance(value, LTX23TextConditioning):
            for name in ("video_context", "video_mask", "audio_context", "audio_mask"):
                tensor = getattr(value, name)
                yield (
                    f"{prefix}.{name}",
                    tuple(tensor.shape) if tensor is not None else None,
                )
            for name in ("video_pe", "video_cross_pe", "audio_pe", "audio_cross_pe"):
                yield (
                    f"{prefix}.{name}",
                    self._tensor_pair_shapes(getattr(value, name)),
                )
            return
        yield from super()._key_parts_for(prefix, value)

    @staticmethod
    def _clone_value(value):
        if isinstance(value, LTX23Modality):
            return LTX23Modality(
                latent=value.latent.clone(),
                timesteps=value.timesteps.clone(),
                sigma=value.sigma.clone(),
                positions=value.positions.clone(),
                context=value.context.clone(),
                enabled=value.enabled,
                context_mask=(
                    value.context_mask.clone() if value.context_mask is not None else None
                ),
            )
        if isinstance(value, LTX23TextConditioning):
            clone_pair = _LTX23CUDAGraphRunner._clone_tensor_pair
            return LTX23TextConditioning(
                video_context=(
                    value.video_context.clone() if value.video_context is not None else None
                ),
                video_mask=value.video_mask.clone() if value.video_mask is not None else None,
                video_pe=clone_pair(value.video_pe),
                video_cross_pe=clone_pair(value.video_cross_pe),
                audio_context=(
                    value.audio_context.clone() if value.audio_context is not None else None
                ),
                audio_mask=value.audio_mask.clone() if value.audio_mask is not None else None,
                audio_pe=clone_pair(value.audio_pe),
                audio_cross_pe=clone_pair(value.audio_cross_pe),
            )
        return _LTX2CUDAGraphRunner._clone_value(value)

    @staticmethod
    def _copy_optional_tensor(dst, src):
        if dst is not None and src is not None:
            dst.copy_(src)

    @staticmethod
    def _copy_value(dst, src):
        if isinstance(src, LTX23Modality) and isinstance(dst, LTX23Modality):
            dst.latent.copy_(src.latent)
            dst.timesteps.copy_(src.timesteps)
            dst.sigma.copy_(src.sigma)
            dst.positions.copy_(src.positions)
            dst.context.copy_(src.context)
            _LTX23CUDAGraphRunner._copy_optional_tensor(dst.context_mask, src.context_mask)
            return dst
        if isinstance(src, LTX23TextConditioning) and isinstance(dst, LTX23TextConditioning):
            copy_tensor = _LTX23CUDAGraphRunner._copy_optional_tensor
            copy_pair = _LTX23CUDAGraphRunner._copy_tensor_pair
            copy_tensor(dst.video_context, src.video_context)
            copy_tensor(dst.video_mask, src.video_mask)
            copy_pair(dst.video_pe, src.video_pe)
            copy_pair(dst.video_cross_pe, src.video_cross_pe)
            copy_tensor(dst.audio_context, src.audio_context)
            copy_tensor(dst.audio_mask, src.audio_mask)
            copy_pair(dst.audio_pe, src.audio_pe)
            copy_pair(dst.audio_cross_pe, src.audio_cross_pe)
            return dst
        return _LTX2CUDAGraphRunner._copy_value(dst, src)


@register_pipeline(
    "LTX23Pipeline",
    hf_ids=["Lightricks/LTX-2.3"],
    defaults={"text_encoder_path": "google/gemma-3-12b-it"},
    doc="Lightricks LTX-2.3 text-to-video generation with audio.",
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

    @property
    def dtype(self):
        return self.pipeline_config.torch_dtype

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

    def warmup(self) -> None:
        """Tune with capture suppressed, then capture from the warm tactic cache.

        Tuning and capturing in one pass records the autotuner's torch.rand
        candidate-input kernels into the graph, where they replay on every
        denoise step. Other configurations defer to the base implementation.
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
        warmup_start = time.time()

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
            # Hits the warm tactic cache, so the graph holds only model kernels.
            self._run_warmup_pass(shapes, steps)
        finally:
            self._is_warmup = False

        self._warmed_up_shapes = set(
            self.warmup_cache_key(h, w, num_frames=f) for h, w, f in shapes
        )
        logger.info(f"Warmup completed in {time.time() - warmup_start:.2f}s")

    def _setup_cuda_graphs(self):
        """Wrap the transformer with LTX-2.3-aware CUDA graph capture/replay."""
        if not self.pipeline_config.cuda_graph.enable:
            return

        runner = _LTX23CUDAGraphRunner(
            CUDAGraphRunnerConfig(use_cuda_graph=True),
        )
        self.transformer.register_cuda_graph_extra_key_fns(runner)
        compile_note = " (with torch.compile)" if self.pipeline_config.torch_compile.enable else ""
        logger.info(
            "CUDA graph runner: wrapping transformer.forward "
            f"(LTX-2.3 Modality-aware){compile_note}"
        )
        self.transformer.forward = runner.wrap(self.transformer.forward)
        self._cuda_graph_runners["transformer"] = runner

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

        # AudioVideo mode: the transformer emits (video_velocity, audio_velocity).
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

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading native video decoder...")
            self.video_decoder = LTX23VideoDecoderConfigurator.from_config(config)
            _load_component_weights(sft_paths, self.video_decoder, ["vae.decoder.", "vae."])
            self.video_decoder = self.video_decoder.to(device=device, dtype=dtype)

        # The video and audio projection weights both live under
        # text_embedding_projection.*, and each feeds its own connector.
        logger.info("Loading native text feature extractor + connectors...")
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

        self.audio_connector = LTX23AudioConnectorConfigurator.from_config(config)
        _load_component_weights(
            sft_paths,
            self.audio_connector,
            "model.diffusion_model.audio_embeddings_connector.",
        )
        self.audio_connector = self.audio_connector.to(device=device, dtype=dtype)

        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading native audio decoder...")
            self.audio_decoder = AudioDecoderConfigurator.from_config(config)
            _load_component_weights(
                sft_paths, self.audio_decoder, ["audio_vae.decoder.", "audio_vae."]
            )
            self.audio_decoder = self.audio_decoder.to(device=device, dtype=dtype)

            # Kept in fp32: the BWE forward runs fp32 regardless, and bf16
            # accumulation degrades the spectral output across ~108 sequential convs.
            logger.info("Loading native vocoder (BigVGAN-v2 + BWE)...")
            self.vocoder = LTX23VocoderConfigurator.from_config(config)
            _load_component_weights(sft_paths, self.vocoder, "vocoder.")
            self.vocoder = self.vocoder.to(device=device, dtype=torch.float32)

        patch_size = self.transformer._transformer_config.get("patch_size", 1)
        self.video_patchifier = VideoLatentPatchifier(patch_size=patch_size)

        # Audio decode-side metadata comes off the audio VAE decoder, so it is
        # only available when that component was loaded.
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
        """Stack all Gemma hidden states, per-token RMS normalize, then flatten.

        hidden_states is a tuple of num_layers+1 tensors [B, S, 3840]; returns
        [B, S, 3840 * num_states] for the split feature extractor. LTX-2.3 sets
        text_encoder_norm_type=per_token_rms with caption_proj_input_norm=False,
        so the norm belongs here rather than in the caption projection.
        """
        stacked = torch.stack(hidden_states, dim=-1)  # [B, S, 3840, num_states]
        # RMS over the hidden dim, per token and per layer.
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

        Returns (video_embeds, audio_embeds, video_mask). The connector
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
            def to_x0(latent, velocity):
                sigma = timestep_val.float()
                while sigma.dim() < velocity.dim():
                    sigma = sigma.unsqueeze(-1)
                return latent - velocity.float() * sigma

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
