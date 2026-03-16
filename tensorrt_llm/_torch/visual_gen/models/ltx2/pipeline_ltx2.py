# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import safetensors.torch
import torch
import torch.distributed as dist
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from tensorrt_llm._torch.visual_gen.config import PipelineComponent
from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.teacache import CacheContext
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from .ltx2_core.audio_vae import AudioDecoderConfigurator, VocoderConfigurator, decode_audio
from .ltx2_core.connector import Embeddings1DConnectorConfigurator, GemmaFeaturesExtractorProjLinear
from .ltx2_core.guiders import MultiModalGuider, MultiModalGuiderParams
from .ltx2_core.modality import Modality
from .ltx2_core.patchifier import AudioPatchifier, VideoLatentPatchifier, get_pixel_coords
from .ltx2_core.perturbations import (
    BatchedPerturbationConfig,
    PerturbationConfig,
    build_stg_perturbation_config,
)
from .ltx2_core.rope import LTXRopeType
from .ltx2_core.scheduler_adapter import NativeSchedulerAdapter
from .ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from .ltx2_core.video_vae import TilingConfig, VideoDecoderConfigurator, VideoEncoderConfigurator
from .transformer_ltx2 import LTXModel, LTXModelType


def _assert_resolution(height: int, width: int) -> None:
    """Validate that height/width are divisible by the VAE spatial scale factor (32)."""
    divisor = 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"Height and width must be multiples of {divisor}."
        )


def _load_ltx2_transformer_weights(
    checkpoint_dir: str,
    prefix: str,
    exclude_prefixes: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Read transformer weights from an LTX-2 single-safetensor checkpoint.

    Scans all ``.safetensors`` files in *checkpoint_dir* for keys starting
    with *prefix*, strips the prefix, and returns the resulting state dict.

    Args:
        checkpoint_dir: Path to the checkpoint directory (or single file).
        prefix: Key prefix for transformer weights
            (e.g. ``model.diffusion_model.``).
        exclude_prefixes: Sub-prefixes (after stripping *prefix*) to skip.
            Used to filter out non-transformer components that share the
            same checkpoint prefix (e.g. ``audio_embeddings_connector.``).
    """
    d = Path(checkpoint_dir)
    if d.is_file() and d.suffix == ".safetensors":
        sft_paths = [str(d)]
    else:
        sft_paths = sorted(str(f) for f in d.glob("*.safetensors"))

    if not sft_paths:
        raise ValueError(f"No safetensors files found in {checkpoint_dir}")

    exclude_prefixes = tuple(exclude_prefixes) if exclude_prefixes else ()

    weights: Dict[str, torch.Tensor] = {}
    for path in sft_paths:
        with safetensors.torch.safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    stripped = key[len(prefix) :]
                    if stripped.startswith(exclude_prefixes):
                        continue
                    weights[stripped] = f.get_tensor(key)

    if not weights:
        raise ValueError(f"No transformer weights found with prefix '{prefix}' in {sft_paths}")

    logger.info(
        f"Loaded {len(weights)} transformer weight tensors from LTX-2 checkpoint ({prefix}*)"
    )
    return weights


# TeaCache polynomial coefficients for LTX-0.9.1-HFIE. Calibrated from the LTX-Video model family.
# Maps raw embedding L1 distances to rescaled distances for cache decisions.
# Coefficients are from:
# https://huggingface.co/jbilcke-hf/LTX-Video-0.9.1-HFIE/blob/main/teacache.py#L42
# TODO: Need to verify coefficients for correctness.

# LTX2_TEACACHE_COEFFICIENTS = {
#     "ltx": {
#         "ret_steps": [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03],
#         "standard": [2.14700694e+01, -1.28016453e+01, 2.31279151e+00, 7.92487521e-01, 9.69274326e-03],
#     },
# }


class LTX2TeaCacheExtractor:
    """Custom TeaCache extractor for LTX-2's Modality-based interface.

    LTX-2's transformer takes ``(video: Modality, audio: Modality)`` rather
    than flat ``(hidden_states, timestep, ...)`` parameters, so the generic
    extractor cannot locate the timestep or hidden states.

    Video and audio velocity outputs are concatenated along the token
    dimension for the hook's single-tensor residual logic (both share
    ``out_channels=128``), then split back in ``postprocess``.
    """

    _FORWARD_PARAMS = ["video", "audio", "perturbations"]

    def __init__(self, timestep_embed_fn):
        self.timestep_embed_fn = timestep_embed_fn
        self._video_tokens = 0
        self._audio_tokens = 0

    def __call__(self, module, *args, **kwargs):
        params = {
            self._FORWARD_PARAMS[i]: arg
            for i, arg in enumerate(args)
            if i < len(self._FORWARD_PARAMS)
        }
        params.update(kwargs)

        video = params.get("video")
        audio = params.get("audio")

        # --- timestep embedding (for cache distance) ---
        ts = video.timesteps if video is not None else audio.timesteps
        if ts.ndim >= 2:
            ts = ts.amax(dim=-1)
        t_emb = self.timestep_embed_fn(module, ts)

        # --- combined hidden_states for residual computation ---
        v_lat = video.latent if video is not None else None
        a_lat = audio.latent if audio is not None else None
        self._video_tokens = v_lat.shape[1] if v_lat is not None else 0
        self._audio_tokens = a_lat.shape[1] if a_lat is not None else 0

        if v_lat is not None and a_lat is not None:
            hidden_states = torch.cat([v_lat, a_lat], dim=1)
        else:
            hidden_states = v_lat if v_lat is not None else a_lat

        def run_blocks():
            vel_v, vel_a = module._original_forward(**params)
            if vel_v is not None and vel_a is not None:
                return (torch.cat([vel_v, vel_a], dim=1),)
            return (vel_v if vel_v is not None else vel_a,)

        def postprocess(output):
            n_v, n_a = self._video_tokens, self._audio_tokens
            if n_v > 0 and n_a > 0 and output.shape[1] == n_v + n_a:
                return output[:, :n_v], output[:, n_v:]
            if n_v > 0:
                return output, None
            return None, output

        return CacheContext(
            modulated_input=t_emb,
            hidden_states=hidden_states,
            run_transformer_blocks=run_blocks,
            postprocess=postprocess,
        )


# ---------------------------------------------------------------------------
# Weight-loading helpers
# ---------------------------------------------------------------------------


def _find_safetensors_files(directory: str) -> List[str]:
    """Return sorted list of .safetensors files in *directory*."""
    d = Path(directory)
    if d.is_file() and d.suffix == ".safetensors":
        return [str(d)]
    files = sorted(d.glob("*.safetensors"))
    return [str(f) for f in files]


def _read_safetensors_config(path: str) -> Optional[Dict[str, Any]]:
    """Read the ``config`` key from safetensors metadata header."""
    try:
        with safetensors.torch.safe_open(path, framework="pt") as f:
            meta = f.metadata()
            if meta and "config" in meta:
                return json.loads(meta["config"])
    except Exception:
        pass
    return None


def _load_component_weights(
    safetensors_paths: List[str],
    module: torch.nn.Module,
    prefix: Union[str, List[str]],
) -> None:
    """Load weights from safetensors file(s) into *module* by filtering on *prefix*.

    *prefix* can be a single string or a list of strings.  When multiple
    prefixes are given, keys matching any prefix are collected (each prefix
    is stripped independently).
    """
    prefixes = [prefix] if isinstance(prefix, str) else prefix

    state_dict: Dict[str, torch.Tensor] = {}
    for path in safetensors_paths:
        with safetensors.torch.safe_open(path, framework="pt") as f:
            for key in f.keys():
                for pfx in prefixes:
                    if key.startswith(pfx):
                        stripped = key[len(pfx) :]
                        state_dict[stripped] = f.get_tensor(key)
                        break

    if not state_dict:
        logger.warning(f"No weights found with prefix '{prefixes}'")
        return

    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(
            f"Keys missing for prefix '{prefixes}' ({len(missing)}): "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if unexpected:
        logger.warning(
            f"Unexpected keys for prefix '{prefixes}' ({len(unexpected)}): "
            f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@register_pipeline("LTX2Pipeline")
class LTX2Pipeline(BasePipeline):
    """Pipeline for text-to-video generation with audio using LTX2 model.

    All components use native LTX-2 implementations ported from
    https://github.com/Lightricks/LTX-2.
    Only the text encoder (Gemma3) and tokenizer come from the
    ``transformers`` library.
    """

    @property
    def dtype(self):
        return self.model_config.torch_dtype

    @property
    def common_warmup_shapes(self) -> list:
        """Return list of common warmup shapes (height, width, num_frames)."""
        return [(512, 768, 121)]

    def _run_warmup(self, warmup_steps: int) -> None:
        """Run warmup inference to trigger torch.compile and CUDA init."""
        for height, width, num_frames in self.common_warmup_shapes:
            logger.info(f"Warmup: LTX2 {height}x{width}, {num_frames} frames, {warmup_steps} steps")
            self.forward(
                prompt="warmup",
                negative_prompt="",
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=warmup_steps,
                guidance_scale=4.0,
                seed=0,
            )

    # ------------------------------------------------------------------
    # Transformer weight loading
    # ------------------------------------------------------------------

    _TRANSFORMER_PREFIX = "model.diffusion_model."
    _TRANSFORMER_EXCLUDE_PREFIXES = [
        "audio_embeddings_connector.",
        "video_embeddings_connector.",
    ]

    def load_transformer_weights(self, checkpoint_dir: str) -> Dict[str, torch.Tensor]:
        """Load transformer weights from LTX-2 native checkpoint format."""
        logger.info("Loading transformer weights via LTX-2 native checkpoint")
        return _load_ltx2_transformer_weights(
            checkpoint_dir,
            self._TRANSFORMER_PREFIX,
            exclude_prefixes=self._TRANSFORMER_EXCLUDE_PREFIXES,
        )

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load transformer weights.

        Args:
            weights: State dict with parameter names already stripped of
                any checkpoint prefix (e.g. ``model.diffusion_model.``).
                The transformer's own :meth:`load_weights` handles any
                remaining key remapping (``net.0.proj → up_proj``, etc.).
        """
        if self.transformer is not None and hasattr(self.transformer, "load_weights"):
            logger.info("Loading transformer weights...")
            transformer_weights = weights.get("transformer", weights)
            self.transformer.load_weights(transformer_weights)
            logger.info("Transformer weights loaded successfully.")

    @staticmethod
    def _compute_ltx2_timestep_embedding(module, timestep, guidance=None):
        """Compute timestep embedding for TeaCache hook."""
        scaled_ts = timestep * module.timestep_scale_multiplier
        _, embedded_ts = module.adaln_single(scaled_ts, hidden_dtype=torch.bfloat16)
        return embedded_ts

    # ------------------------------------------------------------------
    # Transformer init (called from BasePipeline.__init__)
    # ------------------------------------------------------------------

    def _init_transformer(self) -> None:
        """Create LTXModel from pretrained_config.

        Reads all architecture parameters from the checkpoint config to match
        the reference ``LTXModelConfigurator.from_config()``.  Missing keys
        fall back to the same defaults the reference uses.
        """
        cfg = self.model_config.pretrained_config

        rope_type = LTXRopeType(getattr(cfg, "rope_type", "interleaved"))
        freq_prec = getattr(cfg, "frequencies_precision", False)
        double_precision_rope = freq_prec == "float64"
        apply_gated_attention = getattr(cfg, "apply_gated_attention", False)

        logger.info(
            f"LTX2 transformer config: rope_type={rope_type.value}, "
            f"double_precision_rope={double_precision_rope}, "
            f"apply_gated_attention={apply_gated_attention}"
        )

        self.transformer = LTXModel(
            model_type=LTXModelType.AudioVideo,
            num_attention_heads=getattr(cfg, "num_attention_heads", 32),
            attention_head_dim=getattr(cfg, "attention_head_dim", 128),
            in_channels=getattr(cfg, "in_channels", 128),
            out_channels=getattr(cfg, "out_channels", 128),
            num_layers=getattr(cfg, "num_layers", 48),
            cross_attention_dim=getattr(cfg, "cross_attention_dim", 4096),
            norm_eps=float(getattr(cfg, "norm_eps", 1e-6)),
            caption_channels=getattr(cfg, "caption_channels", 3840),
            positional_embedding_theta=float(getattr(cfg, "positional_embedding_theta", 10000.0)),
            positional_embedding_max_pos=getattr(
                cfg, "positional_embedding_max_pos", [20, 2048, 2048]
            ),
            timestep_scale_multiplier=getattr(cfg, "timestep_scale_multiplier", 1000),
            use_middle_indices_grid=getattr(cfg, "use_middle_indices_grid", True),
            audio_num_attention_heads=getattr(cfg, "audio_num_attention_heads", 32),
            audio_attention_head_dim=getattr(cfg, "audio_attention_head_dim", 64),
            audio_in_channels=getattr(cfg, "audio_in_channels", 128),
            audio_out_channels=getattr(cfg, "audio_out_channels", 128),
            audio_cross_attention_dim=getattr(cfg, "audio_cross_attention_dim", 2048),
            audio_positional_embedding_max_pos=getattr(
                cfg, "audio_positional_embedding_max_pos", [20]
            ),
            av_ca_timestep_scale_multiplier=getattr(cfg, "av_ca_timestep_scale_multiplier", 1),
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=apply_gated_attention,
            model_config=self.model_config,
        )
        self.transformer._transformer_config = vars(cfg)

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
        """Load all non-transformer components.

        The text encoder (Gemma3) and tokenizer are loaded from a
        **separate** directory (``text_encoder_path``), matching the
        reference LTX-2 implementation which keeps the Gemma model
        independent of the diffusion checkpoint.  All other native
        components are loaded from the single safetensors checkpoint.

        Args:
            checkpoint_dir: Path to the native LTX-2 checkpoint
                (directory containing ``*.safetensors`` files).
            device: Target device.
            skip_components: Components to skip.
            text_encoder_path: Path to the Gemma3 model directory.
                Must contain model weights (``model*.safetensors``),
                tokenizer files, and ``preprocessor_config.json``.
        """
        skip_components = skip_components or []
        dtype = self.model_config.torch_dtype

        needs_text = (
            PipelineComponent.TOKENIZER not in skip_components
            or PipelineComponent.TEXT_ENCODER not in skip_components
        )
        if needs_text and not text_encoder_path:
            raise ValueError(
                "text_encoder_path is required for loading the tokenizer "
                "and text encoder. Set VisualGenArgs.text_encoder_path to "
                "the Gemma3 model directory."
            )

        # --- Tokenizer & text encoder (from separate Gemma directory) -----
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

        # --- Resolve native config ----------------------------------------
        native_config = self.model_config.extra_attrs.get("monolithic_safetensors_config")
        sft_paths = _find_safetensors_files(checkpoint_dir)

        if native_config is None and sft_paths:
            native_config = _read_safetensors_config(sft_paths[0])

        if native_config is None:
            raise ValueError(
                "LTX-2 native checkpoint format required but could not find "
                "config metadata in safetensors file(s) at "
                f"{checkpoint_dir}. Ensure the checkpoint contains embedded "
                "metadata under the 'config' key."
            )

        self._native_config = native_config

        # --- Native components --------------------------------------------
        self._load_native_components(native_config, sft_paths, device, dtype, skip_components)

        # --- Scheduler (algorithmic, no weights) --------------------------
        if PipelineComponent.SCHEDULER not in skip_components:
            self.scheduler = NativeSchedulerAdapter()

    def _load_native_components(
        self,
        config: Dict[str, Any],
        sft_paths: List[str],
        device: torch.device,
        dtype: torch.dtype,
        skip_components: List,
    ) -> None:
        """Instantiate and load weights for native LTX-2 components."""

        # Video decoder — native checkpoint stores decoder weights under
        # "vae.decoder." and statistics under "vae.per_channel_statistics.".
        # Prefixes are tried in order: "vae.decoder." strips to bare param
        # names (e.g. conv_in.*), while "vae." catches per_channel_statistics
        # and keeps the submodule prefix intact.
        if PipelineComponent.VAE not in skip_components:
            logger.info("Loading native video decoder...")
            self.video_decoder = VideoDecoderConfigurator.from_config(config)
            _load_component_weights(
                sft_paths,
                self.video_decoder,
                ["vae.decoder.", "vae."],
            )
            self.video_decoder = self.video_decoder.to(device=device, dtype=dtype)

        # Audio decoder — same prefix layout as video VAE
        if "audio_vae" not in skip_components:
            logger.info("Loading native audio decoder...")
            self.audio_decoder = AudioDecoderConfigurator.from_config(config)
            _load_component_weights(
                sft_paths,
                self.audio_decoder,
                ["audio_vae.decoder.", "audio_vae."],
            )
            self.audio_decoder = self.audio_decoder.to(device=device, dtype=dtype)

        # Vocoder (kept in float32 for output precision)
        if "vocoder" not in skip_components:
            logger.info("Loading native vocoder...")
            self.vocoder = VocoderConfigurator.from_config(config)
            _load_component_weights(sft_paths, self.vocoder, "vocoder.")
            self.vocoder = self.vocoder.to(device=device, dtype=dtype)

        # Feature extractor + connectors — native checkpoint uses
        # "text_embedding_projection." for the feature extractor and stores
        # connectors inside the diffusion model prefix.
        if "connectors" not in skip_components:
            logger.info("Loading native text connectors...")
            self.feature_extractor = GemmaFeaturesExtractorProjLinear.from_config(config)
            _load_component_weights(
                sft_paths,
                self.feature_extractor,
                "text_embedding_projection.",
            )
            self.feature_extractor = self.feature_extractor.to(device=device, dtype=dtype)

            self.video_connector = Embeddings1DConnectorConfigurator.from_config(config)
            _load_component_weights(
                sft_paths,
                self.video_connector,
                "model.diffusion_model.video_embeddings_connector.",
            )
            self.video_connector = self.video_connector.to(device=device, dtype=dtype)

            self.audio_connector = Embeddings1DConnectorConfigurator.from_config(config)
            _load_component_weights(
                sft_paths,
                self.audio_connector,
                "model.diffusion_model.audio_embeddings_connector.",
            )
            self.audio_connector = self.audio_connector.to(device=device, dtype=dtype)

        # Video encoder (for image-to-video conditioning)
        if "video_encoder" not in skip_components:
            encoder_blocks = config.get("vae", {}).get("encoder_blocks", [])
            if encoder_blocks:
                logger.info("Loading native video encoder (for i2v)...")
                self.video_encoder = VideoEncoderConfigurator.from_config(config)
                _load_component_weights(
                    sft_paths,
                    self.video_encoder,
                    ["vae.encoder.", "vae."],
                )
                self.video_encoder = self.video_encoder.to(device=device, dtype=dtype)
            else:
                logger.info("No encoder_blocks in config; video encoder not loaded.")
                self.video_encoder = None
        else:
            self.video_encoder = None

        # Patchifiers (no weights, purely structural)
        t_cfg = self.transformer._transformer_config
        patch_size = t_cfg.get("patch_size", 1)
        self.video_patchifier = VideoLatentPatchifier(patch_size=patch_size)

        if hasattr(self, "audio_decoder") and self.audio_decoder is not None:
            self.audio_patchifier = self.audio_decoder.patchifier
        else:
            self.audio_patchifier = AudioPatchifier(patch_size=1)

    # ------------------------------------------------------------------
    # Post-load
    # ------------------------------------------------------------------

    def post_load_weights(self) -> None:
        """Finalize after weight loading: TeaCache, derived attributes."""
        super().post_load_weights()

        # TODO: TeaCache disabled: LTX2_TEACACHE_COEFFICIENTS are unverified.
        # To re-enable, uncomment the following lines and verify coefficients.
        # register_extractor(
        #     "LTXModel",
        #     LTX2TeaCacheExtractor(self._compute_ltx2_timestep_embedding),
        # )
        # self._setup_teacache(self.transformer, coefficients=LTX2_TEACACHE_COEFFICIENTS)

        # Compression ratios from native scale factors
        self.vae_spatial_compression_ratio = VIDEO_SCALE_FACTORS.width
        self.vae_temporal_compression_ratio = VIDEO_SCALE_FACTORS.time

        # Audio properties
        if hasattr(self, "audio_decoder") and self.audio_decoder is not None:
            self.audio_sampling_rate = self.audio_decoder.sample_rate
            self.audio_hop_length = self.audio_decoder.mel_hop_length
            self.audio_mel_bins = self.audio_decoder.mel_bins

        # Transformer patch config
        t_cfg = self.transformer._transformer_config
        self.transformer_in_channels = t_cfg.get("in_channels", 128)

        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.tokenizer_max_length = self.tokenizer.model_max_length

        logger.info("LTX2 pipeline post-load complete")

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        device: Union[str, torch.device],
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Pack and normalize text encoder hidden states."""
        batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
        original_dtype = text_hidden_states.dtype

        token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        if padding_side == "right":
            mask = token_indices < sequence_lengths[:, None]
        elif padding_side == "left":
            start_indices = seq_len - sequence_lengths[:, None]
            mask = token_indices >= start_indices
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
        mask = mask[:, :, None, None]

        masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
        num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (
            num_valid_positions + eps
        )

        x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

        normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
        normalized_hidden_states = normalized_hidden_states * scale_factor

        normalized_hidden_states = normalized_hidden_states.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
        normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)
        return normalized_hidden_states

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
    ):
        """Encode prompt into text embeddings via Gemma3."""
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

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=self.device,
            padding_side=self.tokenizer.padding_side,
            scale_factor=scale_factor,
        )
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    # ------------------------------------------------------------------
    # Connector processing
    # ------------------------------------------------------------------

    def _process_connectors(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """Run feature extraction and video/audio connectors.

        Returns (video_embeds, audio_embeds, connector_mask).
        """
        additive_mask = (1 - attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        additive_mask = additive_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        projected = self.feature_extractor(prompt_embeds)
        video_embeds, video_mask = self.video_connector(projected, additive_mask)
        audio_embeds, _ = self.audio_connector(projected, additive_mask)

        return video_embeds, audio_embeds, video_mask

    # ------------------------------------------------------------------
    # Image conditioning helpers (for image-to-video)
    # ------------------------------------------------------------------

    def _load_and_preprocess_image(
        self,
        image: Union[str, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Load and preprocess an image for VAE encoding.

        Args:
            image: File path (str) or tensor. Tensor should be ``(3, H, W)``
                or ``(B, 3, H, W)`` in ``[0, 1]`` range.
            height: Target height in pixels.
            width: Target width in pixels.

        Returns:
            Tensor of shape ``(1, 3, 1, H, W)`` in ``[-1, 1]``.
        """
        if isinstance(image, str):
            from PIL import Image

            pil_img = Image.open(image).convert("RGB")
            pil_img = pil_img.resize((width, height), Image.LANCZOS)
            import numpy as np

            img_np = np.array(pil_img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        else:
            img_tensor = image
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]
            if img_tensor.shape[1] != height or img_tensor.shape[2] != width:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        img_tensor = img_tensor * 2.0 - 1.0
        return (
            img_tensor.unsqueeze(0)
            .unsqueeze(2)
            .to(
                device=self.device,
                dtype=self.dtype,
            )
        )

    @torch.inference_mode()
    def _encode_image(self, image_5d: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed image tensor through the VAE encoder.

        Args:
            image_5d: ``(B, 3, 1, H, W)`` tensor in ``[-1, 1]``.

        Returns:
            Latent tensor ``(B, C, 1, H_lat, W_lat)``.
        """
        if self.video_encoder is None:
            raise RuntimeError(
                "Image-to-video requires a VAE encoder but video_encoder was "
                "not loaded. Ensure the checkpoint contains encoder weights "
                "(vae.encoder.*) and encoder_blocks config."
            )
        return self.video_encoder(image_5d)

    def _build_denoise_mask(
        self,
        video_shape: "VideoLatentShape",
        num_cond_latent_frames: int = 1,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Create a per-token denoise mask for image conditioning.

        Convention follows LTX-2: ``0.0`` = conditioned (don't denoise),
        ``1.0`` = unconditioned (fully denoise).

        Args:
            video_shape: Latent shape for the video.
            num_cond_latent_frames: Number of latent frames to condition on.
            strength: Conditioning strength (1.0 = fully conditioned).

        Returns:
            ``(1, T)`` mask in patchified token space.
        """
        patch_t, patch_h, patch_w = self.video_patchifier.patch_size
        grid_f = video_shape.frames // patch_t
        grid_h = video_shape.height // patch_h
        grid_w = video_shape.width // patch_w
        tokens_per_frame = grid_h * grid_w
        total_tokens = grid_f * tokens_per_frame
        cond_tokens = num_cond_latent_frames * tokens_per_frame

        mask = torch.ones(1, total_tokens, device=self.device, dtype=torch.float32)
        mask[:, :cond_tokens] = 1.0 - strength
        return mask

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, req):
        """Run inference with request parameters."""
        return self.forward(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            frame_rate=req.frame_rate,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            output_type=req.output_type,
            guidance_rescale=req.guidance_rescale,
            max_sequence_length=req.max_sequence_length,
            image=getattr(req, "image", None),
            image_cond_strength=getattr(req, "image_cond_strength", 1.0),
            stg_scale=getattr(req, "stg_scale", 0.0),
            stg_blocks=getattr(req, "stg_blocks", None),
            modality_scale=getattr(req, "modality_scale", 1.0),
            rescale_scale=getattr(req, "rescale_scale", 0.0),
            guidance_skip_step=getattr(req, "guidance_skip_step", 0),
            enhance_prompt=getattr(req, "enhance_prompt", False),
        )

    # ------------------------------------------------------------------
    # Prompt enhancement
    # ------------------------------------------------------------------

    def _enhance_prompt(self, prompt: str, seed: int = 42) -> str:
        """Use Gemma3 as an LLM to enhance the prompt for video generation."""
        system_prompt = (
            "You are a helpful assistant that enhances text prompts for video generation. "
            "Given a user prompt, rewrite it to be more descriptive, vivid, and detailed "
            "while preserving the original intent. Focus on visual details, motion, "
            "lighting, camera angles, and atmosphere. Keep it concise (1-3 sentences)."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": f"User prompt: {prompt}"}]},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.inference_mode(), torch.random.fork_rng(devices=[self.device]):
            torch.manual_seed(seed)
            outputs = self.text_encoder.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"Enhanced prompt: {enhanced}")
        return enhanced.strip()

    @torch.inference_mode()
    def forward(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        seed: int = 42,
        output_type: str = "pt",
        max_sequence_length: int = 1024,
        image: Optional[Union[str, torch.Tensor]] = None,
        image_cond_strength: float = 1.0,
        stg_scale: float = 0.0,
        stg_blocks: Optional[List[int]] = None,
        modality_scale: float = 1.0,
        rescale_scale: float = 0.0,
        guidance_skip_step: int = 0,
        enhance_prompt: bool = False,
    ):
        """Generate video (and audio) from text, optionally conditioned on an image.

        When *image* is provided, the first frame of the generated video is
        seeded with the VAE-encoded image latent.  Per-token timesteps ensure
        conditioned tokens are treated as clean while the remaining tokens are
        denoised normally (LTX-2 image-to-video conditioning).

        Args:
            image: Optional conditioning image. Either a file path (str) or a
                tensor ``(3, H, W)`` / ``(1, 3, H, W)`` in ``[0, 1]``.
            image_cond_strength: Conditioning strength for the image
                (``1.0`` = fully conditioned first frame).
        """
        if image is not None:
            _assert_resolution(height, width)
        pipeline_start = time.time()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Build guider params
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=guidance_scale,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks or [],
            rescale_scale=rescale_scale,
            modality_scale=modality_scale,
            skip_step=guidance_skip_step,
        )
        # Audio CFG scale is floored at 7.0 when classifier-free guidance is
        # active (guidance_scale > 1).  This matches the reference LTX-2
        # implementation where the audio stream requires a stronger guidance
        # signal than video to maintain audio-visual coherence.
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=max(guidance_scale, 7.0) if guidance_scale > 1.0 else 1.0,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks or [],
            rescale_scale=rescale_scale,
            modality_scale=modality_scale,
            skip_step=guidance_skip_step,
        )
        video_guider = MultiModalGuider(video_guider_params)
        audio_guider = MultiModalGuider(audio_guider_params)

        do_cfg = video_guider.do_unconditional_generation()
        do_stg = video_guider.do_perturbed_generation()
        do_modality = video_guider.do_isolated_modality_generation()
        # Only activate multi-modal guidance when STG or modality guidance is
        # requested.  Plain CFG stays on the original BasePipeline path so that
        # guidance_rescale and other existing behaviour is preserved.
        use_multi_modal_guidance = do_stg or do_modality

        # CFG parallel for multi-modal guidance: each GPU handles one
        # CFG pass (cond or uncond), results are all-gathered, then
        # STG/modality passes run on every GPU before the guidance formula.
        cfg_size = self.model_config.parallel.dit_cfg_size
        ulysses_size = self.model_config.parallel.dit_ulysses_size
        do_cfg_parallel_mm = use_multi_modal_guidance and cfg_size >= 2 and do_cfg
        cfg_group = self.rank // ulysses_size
        if do_cfg_parallel_mm and self.rank == 0:
            logger.info(
                f"CFG parallel (multi-modal guidance): cfg_size={cfg_size}, "
                f"ulysses_size={ulysses_size}"
            )

        # ---- 0. Optional prompt enhancement -----------------------------
        if enhance_prompt:
            logger.info("Enhancing prompt with Gemma3...")
            prompt_text = prompt if isinstance(prompt, str) else prompt[0]
            prompt = self._enhance_prompt(prompt_text, seed=seed)

        # ---- 1. Encode prompts ------------------------------------------
        logger.info("Encoding prompts...")
        encode_start = time.time()
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt, num_videos_per_prompt=1, max_sequence_length=max_sequence_length
        )

        neg_prompt_embeds, neg_prompt_attention_mask = None, None
        if do_cfg:
            negative_prompt = negative_prompt or ""
            neg_prompt_embeds, neg_prompt_attention_mask = self._encode_prompt(
                negative_prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

        logger.info(f"Prompt encoding completed in {time.time() - encode_start:.2f}s")

        # ---- 2. Process through connectors ------------------------------
        if do_cfg:
            combined_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat([neg_prompt_attention_mask, prompt_attention_mask], dim=0)
            (
                video_embeds_combined,
                audio_embeds_combined,
                connector_mask_combined,
            ) = self._process_connectors(combined_embeds, combined_mask)

            neg_video_embeds, video_embeds = video_embeds_combined.chunk(2, dim=0)
            neg_audio_embeds, audio_embeds = audio_embeds_combined.chunk(2, dim=0)
            neg_connector_mask, connector_mask = connector_mask_combined.chunk(2, dim=0)
        else:
            video_embeds, audio_embeds, connector_mask = self._process_connectors(
                prompt_embeds,
                prompt_attention_mask,
            )
            neg_video_embeds = None
            neg_audio_embeds = None
            neg_connector_mask = None

        # ---- 3. Prepare latent shapes -----------------------------------
        logger.info("Preparing latents...")
        pixel_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            height=height,
            width=width,
            fps=frame_rate,
        )
        video_shape = VideoLatentShape.from_pixel_shape(
            pixel_shape,
            latent_channels=self.transformer_in_channels,
        )
        audio_shape = AudioLatentShape.from_video_pixel_shape(
            pixel_shape,
            channels=getattr(self.audio_decoder, "z_channels", 8)
            if hasattr(self, "audio_decoder")
            else 8,
            mel_bins=getattr(self, "audio_mel_bins", 64) // 4,
            sample_rate=getattr(self, "audio_sampling_rate", 16000),
            hop_length=getattr(self, "audio_hop_length", 160),
        )

        self.transformer.configure_audio_ulysses(audio_shape.frames)

        # ---- 4. Generate initial noise / image conditioning ---------------
        latents = torch.randn(
            video_shape.to_torch_shape(),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

        denoise_mask: Optional[torch.Tensor] = None
        clean_latent: Optional[torch.Tensor] = None

        if image is not None:
            logger.info("Encoding conditioning image for i2v...")
            image_5d = self._load_and_preprocess_image(image, height, width)
            encoded_image = self._encode_image(image_5d).float()  # (1, C, 1, H_lat, W_lat)

            latents[:, :, :1, :, :] = encoded_image

            # 5D mask for mixing noise with clean latents (before patchification)
            cond_strength = image_cond_strength
            mask_5d = torch.ones(
                1,
                1,
                video_shape.frames,
                video_shape.height,
                video_shape.width,
                device=self.device,
                dtype=torch.float32,
            )
            mask_5d[:, :, :1, :, :] = 1.0 - cond_strength

            noise = torch.randn_like(latents)
            latents = noise * mask_5d + latents * (1.0 - mask_5d)

            # Token-space mask for per-token timesteps (after patchification)
            denoise_mask = self._build_denoise_mask(
                video_shape,
                num_cond_latent_frames=1,
                strength=cond_strength,
            )
            # Full-size clean latent in patchified form for post-step blending.
            # Non-conditioned positions are zero (masked out by denoise_mask).
            clean_5d = torch.zeros_like(latents)
            clean_5d[:, :, :1, :, :] = encoded_image
            clean_latent = self.video_patchifier.patchify(clean_5d)

            num_cond_tokens = int((denoise_mask < 0.5).sum().item())
            logger.info(
                f"i2v conditioning: {num_cond_tokens} conditioned tokens "
                f"of {clean_latent.shape[1]} total, strength={cond_strength}"
            )

        latents = self.video_patchifier.patchify(latents)

        audio_latents = torch.randn(
            audio_shape.to_torch_shape(),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        audio_latents = self.audio_patchifier.patchify(audio_latents)

        # ---- 5. Position embeddings (RoPE) ------------------------------
        video_positions = self.video_patchifier.get_patch_grid_bounds(
            video_shape,
            device=self.device,
        )
        video_positions = get_pixel_coords(
            video_positions.float(),
            VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate
        video_positions = video_positions.to(self.dtype)
        audio_positions = self.audio_patchifier.get_patch_grid_bounds(
            audio_shape,
            device=self.device,
        )

        # ---- 6. Prepare scheduler / timesteps ---------------------------
        latents_5d = torch.randn(
            video_shape.to_torch_shape(),
            device=self.device,
        )
        self.scheduler.set_timesteps(num_inference_steps, latent=latents_5d)
        audio_scheduler = copy.deepcopy(self.scheduler)
        audio_scheduler.set_timesteps(num_inference_steps, latent=latents_5d)
        timesteps = self.scheduler.timesteps

        # ---- 7. Build perturbation config for STG -----------------------
        stg_perturbation: PerturbationConfig | None = None
        if do_stg and stg_blocks:
            stg_perturbation = build_stg_perturbation_config(stg_blocks)

        # ---- 8. Denoising loop ------------------------------------------
        def _run_transformer(
            v_latents,
            a_latents,
            timestep_val,
            v_context,
            a_context,
            mask,
            perturbations=None,
        ):
            """Single transformer pass → (denoised_video, denoised_audio).

            Either *v_latents* or *a_latents* (but not both) may be ``None``
            for modality-isolated passes.

            When *denoise_mask* is active (i2v mode), video timesteps are
            converted to per-token values (conditioned tokens → 0) and the
            denoised prediction is blended with the clean conditioning latent.
            """
            v_latents_f32 = v_latents.float() if v_latents is not None else None
            v_latents_bf = v_latents.to(self.dtype) if v_latents is not None else None
            a_latents_f32 = a_latents.float() if a_latents is not None else None
            a_latents_bf = a_latents.to(self.dtype) if a_latents is not None else None

            # Per-token timesteps for image conditioning
            if denoise_mask is not None and v_latents_bf is not None:
                v_timestep = denoise_mask * timestep_val.unsqueeze(-1)  # (B, T)
            else:
                v_timestep = timestep_val

            video_mod = (
                Modality(
                    latent=v_latents_bf,
                    timesteps=v_timestep,
                    positions=video_positions,
                    context=v_context,
                    context_mask=mask,
                )
                if v_latents_bf is not None
                else None
            )

            audio_mod = (
                Modality(
                    latent=a_latents_bf,
                    timesteps=timestep_val,
                    positions=audio_positions,
                    context=a_context,
                    context_mask=mask,
                )
                if a_latents_bf is not None
                else None
            )

            vel_v, vel_a = self.transformer(
                video=video_mod,
                audio=audio_mod,
                perturbations=perturbations,
            )

            dn_v = None
            if vel_v is not None and v_latents_f32 is not None:
                sigma = timestep_val.float()
                while sigma.dim() < vel_v.dim():
                    sigma = sigma.unsqueeze(-1)
                dn_v = v_latents_f32 - vel_v.float() * sigma

                if denoise_mask is not None and clean_latent is not None:
                    dm = denoise_mask.unsqueeze(-1)  # (B, T, 1)
                    dn_v = dn_v * dm + clean_latent.float() * (1.0 - dm)

            dn_a = None
            if vel_a is not None and a_latents_f32 is not None:
                sigma = timestep_val.float()
                while sigma.dim() < vel_a.dim():
                    sigma = sigma.unsqueeze(-1)
                dn_a = a_latents_f32 - vel_a.float() * sigma

            return dn_v, dn_a

        step_counter = [0]

        def forward_fn(
            video_latents,
            extra_stream_latents,
            timestep,
            encoder_hidden_states,
            extra_tensors,
        ):
            audio_latents_in = extra_stream_latents.get("audio")
            cur_step = step_counter[0]
            step_counter[0] += 1

            if not use_multi_modal_guidance or video_guider.should_skip_step(cur_step):
                dn_v, dn_a = _run_transformer(
                    video_latents,
                    audio_latents_in,
                    timestep,
                    encoder_hidden_states,
                    extra_tensors.get("audio_embeds", audio_embeds),
                    extra_tensors.get("attention_mask", connector_mask),
                )
                return dn_v, {"audio": dn_a}

            # --- CFG: conditional + unconditional passes --------------------
            if do_cfg_parallel_mm:
                # CFG parallel: split cond/uncond across GPUs, all-gather
                if cfg_group == 0:
                    local_v, local_a = _run_transformer(
                        video_latents,
                        audio_latents_in,
                        timestep,
                        video_embeds,
                        audio_embeds,
                        connector_mask,
                    )
                else:
                    local_v, local_a = _run_transformer(
                        video_latents,
                        audio_latents_in,
                        timestep,
                        neg_video_embeds,
                        neg_audio_embeds,
                        neg_connector_mask,
                    )

                local_v = local_v.contiguous()
                gather_v = [torch.empty_like(local_v) for _ in range(self.world_size)]
                dist.all_gather(gather_v, local_v)
                cond_v = gather_v[0]
                uncond_v = gather_v[ulysses_size]

                if local_a is not None:
                    local_a = local_a.contiguous()
                    gather_a = [torch.empty_like(local_a) for _ in range(self.world_size)]
                    dist.all_gather(gather_a, local_a)
                    cond_a = gather_a[0]
                    uncond_a = gather_a[ulysses_size]
                else:
                    cond_a = None
                    uncond_a = 0.0
            else:
                cond_v, cond_a = _run_transformer(
                    video_latents,
                    audio_latents_in,
                    timestep,
                    video_embeds,
                    audio_embeds,
                    connector_mask,
                )
                uncond_v = 0.0
                uncond_a = 0.0
                if do_cfg and neg_video_embeds is not None:
                    uncond_v, uncond_a = _run_transformer(
                        video_latents,
                        audio_latents_in,
                        timestep,
                        neg_video_embeds,
                        neg_audio_embeds,
                        neg_connector_mask,
                    )

            # STG: perturbed attention pass
            perturbed_v: torch.Tensor | float = 0.0
            perturbed_a: torch.Tensor | float = 0.0
            if do_stg and stg_perturbation is not None:
                batched = BatchedPerturbationConfig(
                    perturbations=[stg_perturbation] * video_latents.shape[0]
                )
                perturbed_v, perturbed_a = _run_transformer(
                    video_latents,
                    audio_latents_in,
                    timestep,
                    video_embeds,
                    audio_embeds,
                    connector_mask,
                    perturbations=batched,
                )

            # Modality guidance: disable cross-modal attention
            iso_v: torch.Tensor | float = 0.0
            iso_a: torch.Tensor | float = 0.0
            if do_modality:
                iso_v, _ = _run_transformer(
                    video_latents,
                    None,
                    timestep,
                    video_embeds,
                    None,
                    connector_mask,
                )
                if audio_latents_in is not None:
                    _, iso_a = _run_transformer(
                        None,
                        audio_latents_in,
                        timestep,
                        None,
                        audio_embeds,
                        connector_mask,
                    )

            guided_v = video_guider.calculate(cond_v, uncond_v, perturbed_v, iso_v)
            guided_a = cond_a
            if cond_a is not None:
                ua = uncond_a if isinstance(uncond_a, torch.Tensor) else 0.0
                pa = perturbed_a if isinstance(perturbed_a, torch.Tensor) else 0.0
                ia = iso_a if isinstance(iso_a, torch.Tensor) else 0.0
                guided_a = audio_guider.calculate(cond_a, ua, pa, ia)

            return guided_v, {"audio": guided_a}

        # When using multi-modal guidance, we handle everything inside
        # forward_fn, so tell BasePipeline not to apply its own CFG.
        effective_guidance = 1.0 if use_multi_modal_guidance else guidance_scale

        result = self.denoise(
            latents=latents,
            scheduler=self.scheduler,
            prompt_embeds=video_embeds,
            neg_prompt_embeds=neg_video_embeds if not use_multi_modal_guidance else None,
            guidance_scale=effective_guidance,
            forward_fn=forward_fn,
            timesteps=timesteps,
            guidance_rescale=guidance_rescale,
            extra_cfg_tensors=(
                {
                    "audio_embeds": (audio_embeds, neg_audio_embeds),
                    "attention_mask": (connector_mask, neg_connector_mask),
                }
                if not use_multi_modal_guidance and do_cfg
                else None
            ),
            extra_streams={
                "audio": (audio_latents, audio_scheduler),
            },
        )

        latents, extra_stream_latents = result
        audio_latents = extra_stream_latents["audio"]

        # ---- 8. Decode --------------------------------------------------
        logger.info("Decoding video and audio...")
        decode_start = time.time()

        def decode_video_fn(vid_latents):
            vid_latents = self.video_patchifier.unpatchify(vid_latents, video_shape)

            if output_type == "latent":
                return vid_latents

            vid_latents = vid_latents.to(self.dtype)
            tiling_config = TilingConfig.default()
            chunks = list(
                self.video_decoder.tiled_decode(
                    vid_latents,
                    tiling_config,
                    generator=generator,
                )
            )
            video = torch.cat(chunks, dim=2)
            video = postprocess_video_tensor(video, remove_batch_dim=True)
            return video

        def decode_audio_fn(aud_latents):
            aud_latents = self.audio_patchifier.unpatchify(aud_latents, audio_shape)

            if output_type == "latent":
                return aud_latents

            aud_latents = aud_latents.to(self.dtype)
            return decode_audio(aud_latents, self.audio_decoder, self.vocoder)

        video, audio = self.decode_latents(
            latents=latents,
            decode_fn=decode_video_fn,
            extra_latents={"audio": (audio_latents, decode_audio_fn)},
        )

        if self.rank == 0:
            logger.info(f"Decoding completed in {time.time() - decode_start:.2f}s")
            logger.info(f"Total pipeline time: {time.time() - pipeline_start:.2f}s")

        return MediaOutput(video=video, audio=audio)
