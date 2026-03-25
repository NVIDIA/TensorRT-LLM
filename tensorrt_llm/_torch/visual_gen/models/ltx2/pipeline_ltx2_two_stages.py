# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import time
from typing import Any, Dict, List, Optional, Union

import safetensors.torch
import torch

from tensorrt_llm._torch.visual_gen.output import MediaOutput
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm.logger import logger

from .ltx2_core.audio_vae import decode_audio
from .ltx2_core.modality import Modality
from .ltx2_core.patchifier import get_pixel_coords
from .ltx2_core.types import (
    VIDEO_SCALE_FACTORS,
    AudioLatentShape,
    VideoLatentShape,
    VideoPixelShape,
)
from .ltx2_core.upsampler import LatentUpsamplerConfigurator, upsample_video
from .ltx2_core.video_vae import TilingConfig
from .pipeline_ltx2 import LTX2Pipeline, _assert_resolution, _find_safetensors_files

STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------


def _load_lora_deltas(
    lora_path: str,
    transformer: torch.nn.Module,
    transformer_prefix: str = "model.diffusion_model.",
    strength: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Load LoRA weights and pre-compute parameter deltas.

    Supports two LoRA key layouts:
      * ``<prefix><param_name>.lora_A.weight`` / ``.lora_B.weight``
      * ``<prefix><param_name>.lora_down.weight`` / ``.lora_up.weight``

    The returned dict maps base parameter names (e.g. ``transformer_blocks.0.
    attn1.to_q.weight``) to pre-computed delta tensors ``(B @ A) * strength``.
    """
    sft_paths = _find_safetensors_files(lora_path)
    if not sft_paths:
        raise ValueError(f"No safetensors files found at {lora_path}")

    raw: Dict[str, torch.Tensor] = {}
    alpha_dict: Dict[str, float] = {}
    for path in sft_paths:
        with safetensors.torch.safe_open(path, framework="pt") as f:
            for key in f.keys():
                raw[key] = f.get_tensor(key)

    down_suffix = (".lora_A.weight", ".lora_down.weight")
    up_suffix = (".lora_B.weight", ".lora_up.weight")

    def _strip(key, suffixes):
        for s in suffixes:
            if key.endswith(s):
                return key[: -len(s)]
        return None

    down_keys: Dict[str, torch.Tensor] = {}
    up_keys: Dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        base = _strip(key, down_suffix)
        if base is not None:
            down_keys[base] = tensor
            continue
        base = _strip(key, up_suffix)
        if base is not None:
            up_keys[base] = tensor
            continue
        if key.endswith(".alpha"):
            base_name = key[: -len(".alpha")]
            alpha_dict[base_name] = tensor.item()

    # Build list of prefixes to try stripping, ordered from longest to
    # shortest so we prefer the most specific match.  LoRA checkpoints
    # may use the full ``model.diffusion_model.`` prefix, or just the
    # inner ``diffusion_model.`` part, depending on the exporter.
    strip_prefixes = []
    if transformer_prefix:
        strip_prefixes.append(transformer_prefix)
        parts = transformer_prefix.split(".")
        for i in range(1, len(parts)):
            suffix = ".".join(parts[i:])
            if suffix:
                strip_prefixes.append(suffix)

    deltas: Dict[str, torch.Tensor] = {}
    for base_name in down_keys:
        if base_name not in up_keys:
            continue
        A = down_keys[base_name]  # (rank, in_features)
        B = up_keys[base_name]  # (out_features, rank)
        rank = A.shape[0]
        alpha = alpha_dict.get(base_name, float(rank))
        scale = strength * alpha / rank

        param_name = base_name
        for prefix in strip_prefixes:
            if param_name.startswith(prefix):
                param_name = param_name[len(prefix) :]
                break

        # Apply the same key remapping as LTXModel.load_weights() so
        # that LoRA delta keys match TRT-LLM parameter names.
        for ff_prefix in (".ff.", ".audio_ff."):
            if ff_prefix + "net.0.proj" in param_name:
                param_name = param_name.replace(ff_prefix + "net.0.proj", ff_prefix + "up_proj")
            elif ff_prefix + "net.2" in param_name:
                param_name = param_name.replace(ff_prefix + "net.2", ff_prefix + "down_proj")
        param_name = param_name.replace(".q_norm.", ".norm_q.")
        param_name = param_name.replace(".k_norm.", ".norm_k.")

        delta = (B.float() @ A.float()) * scale
        deltas[param_name] = delta

    logger.info(f"Loaded {len(deltas)} LoRA deltas from {lora_path} (strength={strength})")
    return deltas


_E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _dequantize_fp4_weight(
    packed_weight: torch.Tensor,
    interleaved_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 packed weight (uint8) back to BF16.

    The stored weight_scale is in interleaved layout (after
    ``block_scale_interleave``).  We reverse that, unpack the FP4
    nibbles via E2M1 lookup, and apply per-block + global scales.
    """
    scale_cols = in_features // block_size
    padded_rows = _pad_up(out_features, 128)
    padded_cols = _pad_up(scale_cols, 4)

    linear_scale = torch.ops.trtllm.block_scale_interleave_reverse(
        interleaved_scale.view(padded_rows, padded_cols)
    )

    scale_fp32 = (
        linear_scale[:out_features, :scale_cols].view(torch.float8_e4m3fn).to(torch.float32)
    )

    device = packed_weight.device
    high = (packed_weight >> 4) & 0x0F
    low = packed_weight & 0x0F
    idx = torch.empty(out_features, in_features, dtype=torch.long, device=device)
    idx[..., 0::2] = low.long()
    idx[..., 1::2] = high.long()

    e2m1_table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
    vals = e2m1_table[idx]

    scale_real = (scale_fp32 * weight_scale_2.float()).view(out_features, scale_cols, 1)
    vals = vals.view(out_features, scale_cols, block_size) * scale_real
    return vals.view(out_features, in_features).to(torch.bfloat16)


def _requantize_fp4_weight(
    bf16_weight: torch.Tensor,
    block_size: int = 16,
) -> tuple:
    """Quantize BF16 weight to NVFP4 with interleaved scales.

    Returns ``(qweight, interleaved_scale, weight_scale_2)``.
    """
    from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_nvfp4

    qweight, linear_scale, weight_scale_2 = quantize_nvfp4(bf16_weight, block_size)
    # block_scale_interleave expects uint8; quantize_nvfp4 returns float8_e4m3fn
    interleaved_scale = torch.ops.trtllm.block_scale_interleave(linear_scale.view(torch.uint8))
    return qweight, interleaved_scale, weight_scale_2


# -- FP8 block-scale helpers ------------------------------------------------


def _is_fp8_scale_packed(
    weight_scale: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 128,
) -> bool:
    """Return True when the FP8 block scale is in the packed int32 layout.

    After ``post_load_weights`` on SM100f / SM120, block scales are
    resmoothed and packed into a TMA-aligned int32 col-major tensor by
    ``transform_sf_into_required_layout``.  When that has **not**
    happened, the scale is a plain ``(nb_out, nb_in)`` float32 grid
    (the *standard* format).
    """
    if weight_scale.dtype != torch.int32 or weight_scale.ndim != 2:
        return False

    import math

    from tensorrt_llm.quantization.utils.fp8_utils import align

    nb_in = math.ceil(in_features / block_size)
    aligned_k = align(nb_in, 4)
    expected_cols = aligned_k // 4
    return weight_scale.shape[0] == out_features and weight_scale.shape[1] == expected_cols


def _dequantize_fp8_weight(
    fp8_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    packed: bool = False,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize FP8 E4M3 weight with per-block scales back to BF16.

    Handles both the standard float32 grid and the packed int32 layout.
    """
    out_features, in_features = fp8_weight.shape

    if packed:
        from tensorrt_llm.quantization.utils.fp8_utils import inverse_transform_sf

        block_scale = inverse_transform_sf(
            weight_scale,
            mn=out_features,
            k=in_features,
            block_size=block_size,
        )
    else:
        block_scale = weight_scale

    bf16 = fp8_weight.to(torch.bfloat16)
    scale = block_scale.repeat_interleave(block_size, dim=0)[:out_features]
    scale = scale.repeat_interleave(block_size, dim=1)[:, :in_features]
    return bf16 * scale.to(bf16.device)


def _requantize_fp8_weight(
    bf16_weight: torch.Tensor,
    repack: bool = False,
    block_size: int = 128,
) -> tuple:
    """Quantize BF16 weight to FP8 E4M3 with 128x128 block scales.

    When *repack* is True the returned weight/scale pair is post-processed
    through ``resmooth_to_fp8_e8m0`` + ``transform_sf_into_required_layout``
    so they match the packed layout expected by SM100f / SM120 GEMM kernels.

    Returns ``(qweight, block_scales)``.
    """
    from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_blockwise

    qw, scale = quantize_fp8_blockwise(bf16_weight, block_size)

    if repack:
        from tensorrt_llm.quantization.utils.fp8_utils import (
            resmooth_to_fp8_e8m0,
            transform_sf_into_required_layout,
        )

        qw, scale = resmooth_to_fp8_e8m0(qw, scale)
        scale = transform_sf_into_required_layout(
            scale,
            mn=qw.shape[0],
            k=qw.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False,
        )

    return qw, scale


def _apply_lora_deltas(
    module: torch.nn.Module,
    deltas: Dict[str, torch.Tensor],
    sign: float = 1.0,
) -> tuple:
    """Add (sign=+1) or remove (sign=-1) pre-computed LoRA deltas.

    For standard (BF16/FP16) weights the delta is added directly.
    For FP8-quantized weights (same shape, float8 dtype) and
    NVFP4-quantized weights (packed, half the last dim) we
    dequantize → apply → requantize.

    Returns ``(applied_count, saved_quant_state)`` where
    *saved_quant_state* maps parameter names to their original
    quantized tensors so that un-merging can restore them exactly
    (avoiding double round-trip loss).
    """
    applied = 0
    saved_state: Dict[str, torch.Tensor] = {}
    # Build a lookup that maps *clean* parameter names to the actual
    # Parameter objects.  torch.compile wraps each block in an
    # OptimizedModule, inserting ``._orig_mod.`` into the parameter
    # path (e.g. ``transformer_blocks.0._orig_mod.attn1.to_q.weight``).
    # We strip those segments so LoRA delta keys (which don't contain
    # ``_orig_mod``) can match.
    state: Dict[str, torch.nn.Parameter] = {}
    for raw_name, param in module.named_parameters():
        clean = raw_name.replace("._orig_mod.", ".")
        state[clean] = param
    _FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

    for name, delta in deltas.items():
        param_name = name if name in state else f"{name}.weight"
        if param_name not in state:
            continue

        param = state[param_name]
        base = param_name.rsplit(".weight", 1)[0]

        # --- same shape ---------------------------------------------------
        if param.shape == delta.shape:
            if param.dtype in _FP8_DTYPES:
                # FP8 block-scale: dequant → apply → requant
                scale_key = f"{base}.weight_scale"
                if scale_key not in state:
                    raise RuntimeError(
                        f"Cannot apply LoRA delta to FP8 param '{param_name}': missing {scale_key}."
                    )
                ws_param = state[scale_key]
                out_f, in_f = delta.shape
                is_packed = _is_fp8_scale_packed(
                    ws_param.data,
                    out_f,
                    in_f,
                )

                saved_state[param_name] = param.data.clone()
                saved_state[scale_key] = ws_param.data.clone()

                bf16 = _dequantize_fp8_weight(
                    param.data,
                    ws_param.data,
                    packed=is_packed,
                )
                bf16.add_(delta.to(bf16.device, bf16.dtype), alpha=sign)

                qw, new_scale = _requantize_fp8_weight(
                    bf16,
                    repack=is_packed,
                )
                param.data.copy_(qw)
                ws_param.data.copy_(new_scale)
            else:
                # BF16/FP16/FP32 — direct in-place addition
                param.data.add_(
                    delta.to(param.device, param.dtype),
                    alpha=sign,
                )
            applied += 1

        # --- FP4 packed (half last dim) -----------------------------------
        elif (
            param.ndim == 2
            and delta.ndim == 2
            and param.shape[0] == delta.shape[0]
            and param.shape[1] * 2 == delta.shape[1]
        ):
            scale_key = f"{base}.weight_scale"
            scale2_key = f"{base}.weight_scale_2"
            if scale_key not in state or scale2_key not in state:
                raise RuntimeError(
                    f"Cannot apply LoRA delta to quantized param "
                    f"'{param_name}': missing {scale_key} or {scale2_key}."
                )

            ws_param = state[scale_key]
            ws2_param = state[scale2_key]
            out_features, in_features = delta.shape

            saved_state[param_name] = param.data.clone()
            saved_state[scale_key] = ws_param.data.clone()
            saved_state[scale2_key] = ws2_param.data.clone()

            bf16 = _dequantize_fp4_weight(
                param.data,
                ws_param.data,
                ws2_param.data,
                out_features,
                in_features,
            )
            bf16.add_(delta.to(bf16.device, bf16.dtype), alpha=sign)

            qw, new_scale, new_s2 = _requantize_fp4_weight(bf16)
            param.data.copy_(qw)
            ws_param.data.copy_(new_scale)
            ws2_param.data.fill_(new_s2.item())
            applied += 1
        else:
            logger.warning(
                f"Shape mismatch for LoRA param '{param_name}': "
                f"param={list(param.shape)}, delta={list(delta.shape)}. "
                f"Skipping."
            )
    return applied, saved_state


def _restore_lora_state(
    module: torch.nn.Module,
    saved_state: Dict[str, torch.Tensor],
) -> None:
    """Restore quantized parameters saved by ``_apply_lora_deltas``."""
    state: Dict[str, torch.nn.Parameter] = {}
    for raw_name, param in module.named_parameters():
        clean = raw_name.replace("._orig_mod.", ".")
        state[clean] = param
    for name, data in saved_state.items():
        if name in state:
            state[name].data.copy_(data)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@register_pipeline("LTX2TwoStagesPipeline")
class LTX2TwoStagesPipeline(LTX2Pipeline):
    """Two-stage text-to-video with audio.

    Stage 1: denoise at half spatial resolution with full guidance.
    Stage 2: learned 2x spatial upsample, refinement denoising
             (distilled sigma schedule, no guidance, distilled LoRA),
             then decode.
    """

    @property
    def common_warmup_shapes(self) -> list:
        return [(512, 768, 121)]

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
        spatial_upsampler_path: str = "",
        distilled_lora_path: str = "",
        **kwargs,
    ) -> None:
        super().load_standard_components(
            checkpoint_dir,
            device,
            skip_components,
            text_encoder_path=text_encoder_path,
            **kwargs,
        )

        dtype = self.model_config.torch_dtype

        # --- Spatial upsampler ---
        if spatial_upsampler_path:
            logger.info(f"Loading spatial upsampler from {spatial_upsampler_path}...")
            sft_paths = _find_safetensors_files(spatial_upsampler_path)
            if not sft_paths:
                raise ValueError(f"No safetensors files found at {spatial_upsampler_path}")

            config: Dict[str, Any] = {}
            try:
                with safetensors.torch.safe_open(sft_paths[0], framework="pt") as f:
                    meta = f.metadata()
                    if meta and "config" in meta:
                        import json

                        config = json.loads(meta["config"])
            except Exception:
                pass

            self.spatial_upsampler = LatentUpsamplerConfigurator.from_config(config)

            state_dict: Dict[str, torch.Tensor] = {}
            for path in sft_paths:
                with safetensors.torch.safe_open(path, framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)

            missing, unexpected = self.spatial_upsampler.load_state_dict(
                state_dict,
                strict=False,
            )
            if missing:
                logger.warning(f"Upsampler missing keys ({len(missing)}): {missing[:5]}")
            self.spatial_upsampler = self.spatial_upsampler.to(device=device, dtype=dtype)
            logger.info("Spatial upsampler loaded")
        else:
            self.spatial_upsampler = None

        # --- Distilled LoRA (pre-compute deltas) ---
        self._distilled_lora_deltas: Dict[str, torch.Tensor] = {}
        if distilled_lora_path:
            logger.info(f"Loading distilled LoRA from {distilled_lora_path}...")
            self._distilled_lora_deltas = _load_lora_deltas(
                distilled_lora_path,
                self.transformer,
                transformer_prefix=self._TRANSFORMER_PREFIX,
            )
            logger.info(
                f"Distilled LoRA ready: {len(self._distilled_lora_deltas)} parameter deltas"
            )

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

    def infer(self, req):
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
    # Forward
    # ------------------------------------------------------------------

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
        guidance_scale: float = 3.0,
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
        """Generate video and audio via two stages.

        Stage 1: Denoise at half spatial resolution (height//2, width//2)
                 with full guidance.
        Stage 2: Learned 2x spatial upsample → refinement denoising
                 (distilled sigma schedule, no guidance, distilled LoRA)
                 → decode.
        """
        # Optional prompt enhancement (applied once and reused for both stages).
        if enhance_prompt:
            logger.info("Enhancing prompt with Gemma3 (two-stage)...")
            prompt_text = prompt if isinstance(prompt, str) else prompt[0]
            prompt = self._enhance_prompt(prompt_text, seed=seed)
            # Downstream calls should not re-enhance the prompt.
            enhance_prompt = False

        _assert_resolution(height, width, is_two_stage=True)
        pipeline_start = time.time()
        height_s1 = height // 2
        width_s1 = width // 2
        logger.info(f"LTX2 two-stage: stage1 at {height_s1}x{width_s1}, final {height}x{width}")

        # ================================================================
        # Stage 1: denoise at half resolution
        # ================================================================
        out = super().forward(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height_s1,
            width=width_s1,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            seed=seed,
            output_type="latent",
            max_sequence_length=max_sequence_length,
            image=image,
            image_cond_strength=image_cond_strength,
            stg_scale=stg_scale,
            stg_blocks=stg_blocks,
            modality_scale=modality_scale,
            rescale_scale=rescale_scale,
            guidance_skip_step=guidance_skip_step,
            enhance_prompt=enhance_prompt,
        )

        video_latents = out.video  # (B, C, F_lat, H_lat_s1, W_lat_s1)
        audio_latents = out.audio  # (B, C, F_aud, M) or None

        # Non-primary workers (rank != 0) receive None from
        # decode_latents and exit here.  Rank 0 continues with Stage 2.
        if video_latents is None:
            return MediaOutput(video=None, audio=None)

        # ================================================================
        # Spatial upsample: 2x via learned upsampler
        # ================================================================
        per_ch_stats = self._get_per_channel_statistics()
        video_latents = upsample_video(
            video_latents[:1],
            per_ch_stats,
            self.spatial_upsampler,
        )
        logger.info("Upsampled video latents via learned upsampler")

        # ================================================================
        # Stage 2: refinement denoising with distilled LoRA
        # ================================================================
        n, saved_quant_state = _apply_lora_deltas(
            self.transformer,
            self._distilled_lora_deltas,
            sign=1.0,
        )
        logger.info(f"Merged distilled LoRA ({n} params) for stage 2")

        # Disable Ulysses for Stage 2: only rank 0 is active, so
        # cross-rank collectives in the attention backend would hang.
        self.transformer.set_ulysses_enabled(False)
        try:
            video_latents, audio_latents = self._refinement_denoise(
                video_latents=video_latents,
                audio_latents=audio_latents,
                prompt=prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                seed=seed,
                max_sequence_length=max_sequence_length,
                image=image,
                image_cond_strength=image_cond_strength,
            )
        finally:
            self.transformer.set_ulysses_enabled(True)
            if saved_quant_state:
                _restore_lora_state(self.transformer, saved_quant_state)
            else:
                _apply_lora_deltas(
                    self.transformer,
                    self._distilled_lora_deltas,
                    sign=-1.0,
                )
            logger.info("Un-merged distilled LoRA after stage 2")

        # ================================================================
        # Decode
        # ================================================================
        if output_type == "latent":
            if self.rank == 0:
                logger.info(f"Two-stage total time: {time.time() - pipeline_start:.2f}s")
            return MediaOutput(video=video_latents, audio=audio_latents)

        logger.info("Decoding upsampled video (tiled)...")
        video_latents = video_latents.to(self.dtype)
        tiling_config = TilingConfig.default()
        chunks = list(
            self.video_decoder.tiled_decode(
                video_latents,
                tiling_config,
                generator=None,
            )
        )
        video = torch.cat(chunks, dim=2)
        video = postprocess_video_tensor(video)

        audio_out = None
        if audio_latents is not None:
            audio_latents = audio_latents.to(self.dtype)
            audio_out = decode_audio(audio_latents, self.audio_decoder, self.vocoder)

        if self.rank == 0:
            logger.info(f"Two-stage total time: {time.time() - pipeline_start:.2f}s")
        return MediaOutput(video=video, audio=audio_out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_per_channel_statistics(self) -> torch.nn.Module:
        """Return per-channel statistics for un-normalize/normalize.

        Prefers the encoder (matches reference), falls back to decoder.
        """
        if getattr(self, "video_encoder", None) is not None:
            return self.video_encoder.per_channel_statistics
        return self.video_decoder.per_channel_statistics

    # ------------------------------------------------------------------
    # Refinement denoising
    # ------------------------------------------------------------------

    def _refinement_denoise(
        self,
        video_latents: torch.Tensor,
        audio_latents: Optional[torch.Tensor],
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        seed: int,
        max_sequence_length: int,
        image: Optional[Union[str, torch.Tensor]] = None,
        image_cond_strength: float = 1.0,
    ) -> tuple:
        """Run stage 2 refinement denoising on upsampled latents.

        Uses the distilled sigma schedule (3 steps) with no guidance,
        matching the reference two-stage pipeline.  Both video and audio
        are jointly refined.

        Returns:
            (video_latents, audio_latents) in 5-D un-patchified form.
        """
        logger.info("Stage 2: refinement denoising...")
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # --- Text conditioning (positive only, no CFG) ---
        prompt_embeds, prompt_attention_mask = self._encode_prompt(
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        video_embeds, audio_embeds, connector_mask = self._process_connectors(
            prompt_embeds,
            prompt_attention_mask,
        )

        # --- Shapes at full resolution ---
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

        # --- Video: patchify, positions ---
        v_latents = self.video_patchifier.patchify(video_latents)
        video_positions = self.video_patchifier.get_patch_grid_bounds(
            video_shape,
            device=self.device,
        )
        video_positions = get_pixel_coords(
            video_positions.float(),
            VIDEO_SCALE_FACTORS,
            causal_fix=True,
        )
        video_positions[:, 0, ...] /= frame_rate
        video_positions = video_positions.to(self.dtype)

        # --- Image conditioning at full resolution ---
        denoise_mask: Optional[torch.Tensor] = None
        clean_latent: Optional[torch.Tensor] = None

        if image is not None and getattr(self, "video_encoder", None) is not None:
            logger.info("Stage 2: encoding image at full resolution for i2v...")
            image_5d = self._load_and_preprocess_image(image, height, width)
            encoded_image = self._encode_image(image_5d).float()

            full_clean = torch.zeros_like(video_latents)
            full_clean[:, :, :1, :, :] = encoded_image

            denoise_mask = self._build_denoise_mask(
                video_shape,
                num_cond_latent_frames=1,
                strength=image_cond_strength,
            )
            clean_latent = self.video_patchifier.patchify(full_clean)

            noise_5d = torch.randn_like(video_latents)
            mask_5d = torch.ones(
                1,
                1,
                video_shape.frames,
                video_shape.height,
                video_shape.width,
                device=self.device,
                dtype=torch.float32,
            )
            mask_5d[:, :, :1, :, :] = 1.0 - image_cond_strength
            blended = noise_5d * mask_5d + video_latents * (1.0 - mask_5d)
            v_latents = self.video_patchifier.patchify(blended)

        # --- Audio: patchify, positions ---
        if audio_latents is not None:
            a_latents = self.audio_patchifier.patchify(audio_latents)
            audio_positions = self.audio_patchifier.get_patch_grid_bounds(
                audio_shape,
                device=self.device,
            )
        else:
            a_latents = None
            audio_positions = None

        # --- Distilled sigma schedule ---
        sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            device=self.device,
            dtype=torch.float32,
        )

        # Add noise at the first sigma level using flow-matching interpolation:
        #   z_t = noise * sigma + clean * (1 - sigma)
        # This matches the reference GaussianNoiser, NOT additive noise.
        sigma_0 = sigmas[0]
        v_noise = torch.randn_like(v_latents, generator=generator)
        v_working = v_noise * sigma_0 + v_latents * (1.0 - sigma_0)

        a_working = a_latents
        if a_working is not None:
            a_noise = torch.randn_like(a_working, generator=generator)
            a_working = a_noise * sigma_0 + a_working * (1.0 - sigma_0)

        # --- Euler denoising loop (no guidance) ---
        for i in range(len(sigmas) - 1):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]
            dt = sigma_next - sigma
            timestep = sigma.unsqueeze(0).expand(v_working.shape[0])

            if denoise_mask is not None:
                v_timestep = denoise_mask * sigma
            else:
                v_timestep = timestep

            video_mod = Modality(
                latent=v_working.to(self.dtype),
                timesteps=v_timestep,
                positions=video_positions,
                context=video_embeds,
                context_mask=connector_mask,
            )

            audio_mod = None
            if a_working is not None:
                audio_mod = Modality(
                    latent=a_working.to(self.dtype),
                    timesteps=timestep,
                    positions=audio_positions,
                    context=audio_embeds,
                    context_mask=connector_mask,
                )

            vel_v, vel_a = self.transformer(
                video=video_mod,
                audio=audio_mod,
            )

            # Video: velocity → x0 → post-process → Euler step
            sigma_v = sigma.float()
            while sigma_v.dim() < vel_v.dim():
                sigma_v = sigma_v.unsqueeze(-1)

            denoised_v = v_working.float() - vel_v.float() * sigma_v

            if denoise_mask is not None and clean_latent is not None:
                dm = denoise_mask.unsqueeze(-1)
                denoised_v = denoised_v * dm + clean_latent.float() * (1.0 - dm)

            velocity_v = (v_working.float() - denoised_v) / sigma_v
            v_working = (v_working.float() + velocity_v * dt).to(v_working.dtype)

            # Audio: velocity → x0 → Euler step
            if vel_a is not None and a_working is not None:
                sigma_a = sigma.float()
                while sigma_a.dim() < vel_a.dim():
                    sigma_a = sigma_a.unsqueeze(-1)
                denoised_a = a_working.float() - vel_a.float() * sigma_a
                velocity_a = (a_working.float() - denoised_a) / sigma_a
                a_working = (a_working.float() + velocity_a * dt).to(a_working.dtype)

        # --- Unpatchify ---
        video_out = self.video_patchifier.unpatchify(v_working, video_shape)
        audio_out = None
        if a_working is not None:
            audio_out = self.audio_patchifier.unpatchify(a_working, audio_shape)

        logger.info("Stage 2 refinement complete")
        return video_out, audio_out
