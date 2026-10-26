# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2x
"""BF16 LTX-2.5 IC-LoRA two-stage video editing."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline import (
    BasePipeline,
    ExtraParamSchema,
    RefSlotSpec,
    RoleSpec,
)
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.inputs.media_io import VideoMediaIO

from ..ltx2.ltx2_core.audio_vae import AudioDecoderConfigurator, decode_audio
from ..ltx2.ltx2_core.patchifier import VideoLatentPatchifier
from ..ltx2.ltx2_core.types import AudioLatentShape, VideoLatentShape, VideoPixelShape
from ..ltx2.ltx2_core.upsampler import LatentUpsamplerConfigurator, upsample_video
from ..ltx2.ltx2_core.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ..ltx2.pipeline_ltx2 import _find_safetensors_files, _load_ltx2_transformer_weights
from .audio.vocoder import LTX25VocoderConfigurator
from .conditioning import (
    STAGE_1_SIGMAS,
    STAGE_2_SIGMAS,
    ICLoraState,
    add_noise,
    append_reference,
    euler_step,
    video_state,
)
from .config import LTX25Config
from .connector import (
    LTX25AudioConnectorConfigurator,
    LTX25GemmaFeaturesExtractor,
    LTX25VideoConnectorConfigurator,
)
from .loading import (
    _build_ltx25_transformer,
    _checkpoint_gemma_version,
    _has_component,
    _load_component_strict,
    validate_transformer_weights,
)
from .lora import StageLoRAWeights
from .modality import LTX25Modality
from .text_encoder import (
    load_gemma4_config,
    load_gemma4_model,
    load_gemma4_tokenizer,
    pack_gemma4_hidden_states,
    tokenize_gemma4,
)
from .video_vae import LTX25VideoDecoderConfigurator, LTX25VideoEncoderConfigurator

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.executor import DiffusionRequest


def _pad_audio_for_video(
    audio: torch.Tensor, frames: int, fps: float, sample_rate: int
) -> torch.Tensor:
    # The VAE rounds audio duration down to its latent grid. VisualGen's muxer
    # uses -shortest, so preserve all video frames by padding only trailing silence.
    missing = math.ceil(frames * sample_rate / fps) - audio.shape[-1]
    return F.pad(audio, (0, missing)) if missing > 0 else audio


def _read_config(path: str) -> dict:
    with safe_open(path, framework="pt") as checkpoint:
        metadata = checkpoint.metadata() or {}
    if "config" not in metadata:
        raise ValueError(f"Missing embedded config in {path}")
    return json.loads(metadata["config"])


def _load_split_component(path: str, module: torch.nn.Module, prefix: str) -> None:
    """Load only the component's declared state, including shared VAE statistics."""
    weights = {}
    with safe_open(path, framework="pt") as checkpoint:
        keys = set(checkpoint.keys())
        for name in module.state_dict():
            candidates = [prefix + name]
            if name.startswith("per_channel_statistics."):
                candidates.append(name)
                if prefix.startswith("audio_vae."):
                    candidates.append("audio_vae." + name)
            candidates += ["vae." + key for key in candidates]
            matches = [key for key in candidates if key in keys]
            if len(matches) != 1:
                raise ValueError(f"Expected one checkpoint tensor for {prefix + name}: {matches}")
            weights[name] = checkpoint.get_tensor(matches[0])
    module.load_state_dict(weights, strict=True)


@register_pipeline(
    "LTX25ICLoraPipeline",
    hf_ids=["Lightricks/LTX-2.5"],
    defaults=LTX25Config().model_dump(),
    doc="Single-GPU BF16 LTX-2.5 IC-LoRA two-stage video editing with audio.",
)
class LTX25ICLoraPipeline(BasePipeline):
    """Run the distilled 8+3-step recipe with one SDR control video.

    Requests must execute serially: stage selection mutates transformer weights.
    """

    def __init__(self, pipeline_config) -> None:
        self.options = LTX25Config.model_validate(
            {
                key: value
                for key, value in pipeline_config.extra_attrs.items()
                if key != "monolithic_safetensors_config"
            }
        )
        for name in (
            "text_encoder_path",
            "video_vae_path",
            "audio_vae_path",
            "spatial_upsampler_path",
            "iclora_path",
        ):
            if not getattr(self.options, name):
                raise ValueError(f"LTX-2.5 requires pipeline_config.{name}")
        if pipeline_config.torch_dtype != torch.bfloat16:
            raise ValueError("LTX-2.5 IC-LoRA requires BF16")
        attention = pipeline_config.attention
        if (
            attention.backend != "VANILLA"
            or attention.quant_attention_config is not None
            or attention.sparse_attention_config is not None
        ):
            raise ValueError("LTX-2.5 BF16 correctness requires dense VANILLA attention")
        if pipeline_config.runtime_lora is not None:
            raise ValueError("Use pipeline_config.iclora_path for stage-specific IC-LoRA")
        parallel = pipeline_config.parallel
        if any(
            getattr(parallel, name) != 1
            for name in ("cfg_size", "tp_size", "seq_parallel_size", "parallel_vae_size")
        ):
            raise ValueError("LTX-2.5 IC-LoRA supports one GPU")
        if (
            pipeline_config.cache_backend is not None
            or pipeline_config.cuda_graph.enable
            or pipeline_config.torch_compile.enable
        ):
            raise ValueError(
                "Disable cache acceleration, CUDA graphs and torch.compile for BF16 reference inference"
            )
        if pipeline_config.get_quant_config().quant_algo is not None:
            raise ValueError("LTX-2.5 IC-LoRA does not support quantized weights")
        self._stage_weights = None
        super().__init__(pipeline_config)

    def load_transformer_weights(self, checkpoint_dir: str) -> dict[str, torch.Tensor]:
        with safe_open(checkpoint_dir, framework="pt") as checkpoint:
            if "_quantization_metadata" in (checkpoint.metadata() or {}):
                raise ValueError("LTX-2.5 IC-LoRA requires an unquantized BF16 checkpoint")
        return _load_ltx2_transformer_weights(
            checkpoint_dir,
            "model.diffusion_model.",
            exclude_prefixes=["audio_embeddings_connector.", "video_embeddings_connector."],
        )

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        transformer_weights = weights.get("transformer", weights)
        validate_transformer_weights(self.transformer, transformer_weights)
        self.transformer.load_weights(transformer_weights)

    def post_load_weights(self) -> None:
        super().post_load_weights()
        self._stage_weights = StageLoRAWeights(
            self.transformer, self.options.iclora_path, self.options.iclora_scale
        )

    def _set_stage(self, stage_one: bool) -> None:
        if self._stage_weights is None:
            raise RuntimeError("IC-LoRA weights have not been initialized")
        self._stage_weights.select(stage_one)

    @property
    def ref_slot_specs(self) -> dict[str, RefSlotSpec]:
        return {
            "video_reference": RefSlotSpec(
                modality="video", roles=[RoleSpec(role="reference", min=1, max=1)]
            )
        }

    @property
    def extra_param_specs(self) -> dict[str, ExtraParamSchema]:
        return {
            "reference_strength": ExtraParamSchema(
                type="float",
                default=1.0,
                range=(0.0, 1.0),
                description="Reference latent conditioning strength.",
            )
        }

    def warmup(self) -> None:
        # An IC-LoRA request supplies the control video required to run inference.
        return

    def load_standard_components(
        self, checkpoint_dir, device, skip_components=None, *, text_encoder_path="", **kwargs
    ) -> None:
        if skip_components:
            raise ValueError("LTX-2.5 IC-LoRA requires all conditioning and media components")
        text_encoder_path = text_encoder_path or self.options.text_encoder_path
        paths = _find_safetensors_files(checkpoint_dir)
        self._gemma_config = load_gemma4_config(text_encoder_path, _checkpoint_gemma_version(paths))
        self._gemma_weight_paths = _find_safetensors_files(text_encoder_path)
        self.tokenizer = load_gemma4_tokenizer(text_encoder_path)
        self.text_encoder = load_gemma4_model(
            text_encoder_path, self._gemma_config, self.dtype, device
        )
        config = _read_config(paths[0])
        self._load_native_components(config, paths, device, self.dtype)

    def _process_connectors(self, prompt_embeds, attention_mask):
        mask = ((1 - attention_mask.to(prompt_embeds.dtype)) * -1000000.0)[:, None, None, :]
        video_proj, audio_proj = self.feature_extractor(prompt_embeds)
        with sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        ):
            video, video_mask = self.video_connector(video_proj, mask)
            audio, _ = self.audio_connector(audio_proj, mask)
        return video, audio, video_mask

    def _init_transformer(self) -> None:
        cfg = self.pipeline_config.model_configs["transformer"].pretrained_config
        self.transformer_in_channels = getattr(cfg, "in_channels", 128)
        self.audio_in_channels = getattr(cfg, "audio_in_channels", 128)
        self.audio_out_channels = getattr(cfg, "audio_out_channels", 128)
        self.transformer = _build_ltx25_transformer(self.pipeline_config)

    def _encode_prompt(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompts = [prompt] if isinstance(prompt, str) else prompt
        input_ids, attention_mask = tokenize_gemma4(
            self.tokenizer, prompts, max_sequence_length, self.device
        )
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        embeddings = pack_gemma4_hidden_states(outputs.hidden_states, attention_mask).to(
            dtype=self.dtype
        )
        return (
            embeddings.repeat_interleave(num_videos_per_prompt, dim=0),
            attention_mask.repeat_interleave(num_videos_per_prompt, dim=0),
        )

    def _load_native_components(self, config, sft_paths, device, dtype) -> None:
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        cfg = config["transformer"]
        text_config = self._gemma_config.text_config
        if text_config.hidden_size != cfg.get("caption_channels", 3840):
            raise ValueError("Gemma4 hidden size does not match LTX-2.5 caption_channels.")
        self.feature_extractor = LTX25GemmaFeaturesExtractor(
            caption_channels=text_config.hidden_size,
            video_dim=cfg.get("cross_attention_dim", 4096),
            audio_dim=cfg.get("audio_cross_attention_dim", 2048),
            num_hidden_states=text_config.num_hidden_layers + 1,
        )
        prefix = "text_embedding_projection."
        projection_paths = (
            self._gemma_weight_paths
            if _has_component(self._gemma_weight_paths, prefix)
            else sft_paths
        )
        _load_component_strict(projection_paths, self.feature_extractor, prefix)
        self.feature_extractor.to(device=device, dtype=dtype).eval()
        self.video_connector = LTX25VideoConnectorConfigurator.from_config(config)
        self.audio_connector = LTX25AudioConnectorConfigurator.from_config(config)
        for name, connector in (("video", self.video_connector), ("audio", self.audio_connector)):
            _load_component_strict(
                sft_paths, connector, f"model.diffusion_model.{name}_embeddings_connector."
            )
            connector.to(device=device, dtype=dtype).eval()
        attrs = self.pipeline_config.extra_attrs
        video_path, audio_path = attrs["video_vae_path"], attrs["audio_vae_path"]
        video_config, audio_config = _read_config(video_path), _read_config(audio_path)
        if video_config["vae"].get("_class_name") != "CausalVideoAutoencoder":
            raise ValueError("LTX25 V2V currently requires the convolutional video VAE.")
        self.video_encoder = LTX25VideoEncoderConfigurator.from_config(video_config)
        _load_split_component(video_path, self.video_encoder, "encoder.")
        self.video_decoder = LTX25VideoDecoderConfigurator.from_config(video_config)
        # Only these two statistics are present in the LTX-2.5 conv-VAE export.
        self.video_decoder.per_channel_statistics = self.video_encoder.per_channel_statistics
        _load_split_component(video_path, self.video_decoder, "decoder.")
        self.audio_decoder = AudioDecoderConfigurator.from_config(audio_config)
        _load_split_component(audio_path, self.audio_decoder, "audio_vae.decoder.")
        self.vocoder = LTX25VocoderConfigurator.from_config(audio_config)
        _load_split_component(audio_path, self.vocoder, "vocoder.")
        upsampler_path = attrs["spatial_upsampler_path"]
        self.spatial_upsampler = LatentUpsamplerConfigurator.from_config(
            _read_config(upsampler_path)
        )
        _load_split_component(upsampler_path, self.spatial_upsampler, "")
        for module in (
            self.video_encoder,
            self.video_decoder,
            self.audio_decoder,
            self.spatial_upsampler,
        ):
            module.to(device=device, dtype=dtype).eval()
        self.vocoder.to(device=device, dtype=torch.float32).eval()
        self.audio_patchifier = self.audio_decoder.patchifier
        self.audio_sampling_rate = self.audio_decoder.sample_rate
        self.audio_hop_length = self.audio_decoder.mel_hop_length
        self.audio_mel_bins = self.audio_decoder.mel_bins
        with safe_open(attrs["iclora_path"], framework="pt") as checkpoint:
            metadata = checkpoint.metadata() or {}
        self.reference_downscale_factor = int(metadata.get("reference_downscale_factor", 1))
        self.reference_temporal_scale_factor = int(
            metadata.get("reference_temporal_scale_factor", 1)
        )
        if min(self.reference_downscale_factor, self.reference_temporal_scale_factor) < 1:
            raise ValueError("IC-LoRA reference scales must be positive integers.")

    @property
    def default_generation_params(self) -> dict:
        return {
            "height": 768,
            "width": 1280,
            "num_frames": 65,
            "frame_rate": 24.0,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": 42,
            "max_sequence_length": 1024,
        }

    def _audio_state(self, latent: torch.Tensor) -> ICLoraState:
        tokens = self.audio_patchifier.patchify(latent)
        positions = self.audio_patchifier.get_patch_grid_bounds(
            AudioLatentShape.from_torch_shape(latent.shape), device=latent.device
        )
        mask = torch.ones((*tokens.shape[:2], 1), device=latent.device, dtype=torch.float32)
        return ICLoraState(tokens, tokens.clone(), mask, positions)

    def _denoise(
        self,
        video: ICLoraState,
        audio: ICLoraState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[ICLoraState, ICLoraState]:
        # Reference ICLoraPipeline consumes post-connector contexts without a mask.
        cache = self.transformer.prepare_text_cache(
            video_context=video_context,
            video_context_mask=None,
            video_positions=video.positions,
            audio_context=audio_context,
            audio_context_mask=None,
            audio_positions=audio.positions,
            dtype=self.dtype,
        )
        # Resolve scalar divisors once, avoiding a device sync at every Euler step.
        sigma_values = sigmas.detach().cpu().tolist()
        for index, sigma in self._profile_denoise_steps(sigmas[:-1]):

            def modality(state: ICLoraState, context: torch.Tensor) -> LTX25Modality:
                return LTX25Modality(
                    latent=state.latent,
                    timesteps=state.mask * sigma,
                    sigma=sigma.reshape(1),
                    cross_modality_sigma=sigma.reshape(1),
                    positions=state.positions,
                    context=context,
                    context_mask=None,
                    keyframes_mask=state.keyframes_mask,
                )

            # Lightricks AUTOMATIC SDPA prioritizes cuDNN on this BF16 path.
            # Preserve that order rather than inheriting Torch's global default.
            with sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            ):
                video_velocity, audio_velocity = self.transformer(
                    video=modality(video, video_context),
                    audio=modality(audio, audio_context),
                    text_cache=cache,
                    step_index=index,
                )
            video = euler_step(
                video, video_velocity, sigma, sigmas[index + 1], sigma_value=sigma_values[index]
            )
            audio = euler_step(
                audio, audio_velocity, sigma, sigmas[index + 1], sigma_value=sigma_values[index]
            )
        return video, audio

    @torch.inference_mode()
    def infer(self, req: DiffusionRequest) -> PipelineOutput:
        params = req.params
        if params.image_reference or params.audio_reference:
            raise ValueError(
                "LTX-2.5 IC-LoRA requires video_reference and does not accept image or audio references."
            )
        if getattr(params, "negative_prompt", None):
            raise ValueError("LTX25 V2V does not support negative_prompt without CFG.")
        if getattr(params, "num_images_per_prompt", 1) != 1:
            raise ValueError("LTX25 V2V requires num_images_per_prompt=1.")
        extra = params.extra_params or {}
        references = params.video_reference or []
        if len(references) != 1 or references[0].format != "bytes":
            raise ValueError("Expected one resolved video reference")
        path = references[0].content
        prompt = req.prompt
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise ValueError("LTX25 V2V requires a single text prompt.")
        height, width, frames, fps = (
            params.height,
            params.width,
            params.num_frames,
            params.frame_rate,
        )
        if height <= 0 or width <= 0 or height % 64 or width % 64:
            raise ValueError("Two-stage V2V dimensions must be positive multiples of 64.")
        if frames <= 0 or (frames - 1) % 8 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("V2V requires 8n+1 frames and a positive finite frame rate.")
        if params.num_inference_steps != 8 or params.guidance_scale != 1.0:
            raise ValueError("Distilled IC-LoRA uses a fixed 8+3 step schedule without CFG.")
        strength = extra.get("reference_strength", 1.0)
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not 0 <= strength <= 1
        ):
            raise ValueError("reference_strength must be in [0, 1].")
        scale = self.reference_downscale_factor
        if height % (64 * scale) or width % (64 * scale):
            raise ValueError("V2V dimensions must align with the scaled reference VAE grid.")
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        generator = torch.Generator(device=self.device).manual_seed(params.seed)
        embeddings, mask = self._encode_prompt(
            prompt, max_sequence_length=params.max_sequence_length
        )
        video_context, audio_context, _ = self._process_connectors(embeddings, mask)
        reference = self._read_reference(path, height // 2, width // 2, frames)
        pixel_shape = VideoPixelShape(1, frames, height // 2, width // 2, fps)
        vshape = VideoLatentShape.from_pixel_shape(pixel_shape)
        ashape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        self.transformer.configure_audio_ulysses(ashape.frames)
        video = video_state(
            torch.zeros(vshape.to_torch_shape(), device=self.device, dtype=self.dtype), fps
        )
        target_tokens = video.latent.shape[1]
        video = append_reference(
            video,
            reference,
            fps,
            self.reference_downscale_factor,
            self.reference_temporal_scale_factor,
            strength,
        )
        video = add_noise(video, generator, 1.0)
        audio = add_noise(
            self._audio_state(
                torch.zeros(ashape.to_torch_shape(), device=self.device, dtype=self.dtype)
            ),
            generator,
            1.0,
        )
        sigmas = torch.tensor(STAGE_1_SIGMAS, device=self.device, dtype=torch.float32)
        try:
            self._set_stage(True)
            timer.mark_denoise_start()
            video, audio = self._denoise(video, audio, video_context, audio_context, sigmas)
        finally:
            self._set_stage(False)
        latent = self.video_patchifier.unpatchify(video.latent[:, :target_tokens], vshape)
        latent = upsample_video(
            latent[:1], self.video_encoder.per_channel_statistics, self.spatial_upsampler
        )
        video = add_noise(video_state(latent, fps), generator, STAGE_2_SIGMAS[0])
        audio_latent = self.audio_patchifier.unpatchify(audio.latent, ashape)
        audio = add_noise(self._audio_state(audio_latent), generator, STAGE_2_SIGMAS[0])
        sigmas = torch.tensor(STAGE_2_SIGMAS, device=self.device, dtype=torch.float32)
        video, audio = self._denoise(video, audio, video_context, audio_context, sigmas)
        timer.mark_post_start()
        output_shape = VideoLatentShape.from_pixel_shape(
            VideoPixelShape(1, frames, height, width, fps)
        )
        latent = self.video_patchifier.unpatchify(video.latent, output_shape)
        # Match ICLoraPipeline's conv-VAE AUTO_TILING, without changing LTX25 defaults.
        tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(768, 64),
            temporal_config=TemporalTilingConfig(80, 24),
        )
        chunks = list(self.video_decoder.tiled_decode(latent, tiling, generator=generator))
        output_video = postprocess_video_tensor(torch.cat(chunks, dim=2))
        output_audio = decode_audio(
            self.audio_patchifier.unpatchify(audio.latent, ashape), self.audio_decoder, self.vocoder
        )
        sample_rate = int(self.vocoder.output_sampling_rate)
        output_audio = _pad_audio_for_video(output_audio, frames, fps, sample_rate)
        timer.mark_end()
        return timer.fill(
            PipelineOutput(
                video=output_video,
                audio=output_audio,
                frame_rate=float(fps),
                audio_sample_rate=sample_rate,
            )
        )

    def _read_reference(self, content: bytes, height: int, width: int, frames: int) -> torch.Tensor:
        scale = self.reference_downscale_factor
        height, width = height // scale, width // scale
        # Disable uniform sampling: IC-LoRA consumes the first N source frames.
        source = VideoMediaIO(num_frames=-1, fps=-1, format="np").load_bytes(content).frames
        decoded = []
        for frame in source[:frames]:
            image = torch.from_numpy(frame).to(device=self.device, dtype=torch.float32)
            image = image.permute(2, 0, 1).unsqueeze(0)
            ratio = max(height / image.shape[-2], width / image.shape[-1])
            image = F.interpolate(
                image,
                size=(math.ceil(image.shape[-2] * ratio), math.ceil(image.shape[-1] * ratio)),
                mode="bilinear",
                align_corners=False,
            )
            top, left = (image.shape[-2] - height) // 2, (image.shape[-1] - width) // 2
            decoded.append(
                (image[:, :, top : top + height, left : left + width] / 127.5 - 1).to(self.dtype)
            )
        if len(decoded) != frames:
            raise ValueError(f"Reference video requires at least {frames} frames")
        video = torch.stack(decoded, dim=2)
        if self.reference_temporal_scale_factor > 1:
            video = video[:, :, [0, *range(1, frames, self.reference_temporal_scale_factor)]]
        return self.video_encoder(video)
