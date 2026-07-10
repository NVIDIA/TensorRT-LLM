# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import safetensors.torch
import torch
import torch.distributed as dist

from tensorrt_llm._torch.modules.linear import Linear, UnquantizedLinearMethod
from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunnerConfig
from tensorrt_llm._torch.visual_gen.output import CudaPhaseTimer, PipelineOutput
from tensorrt_llm._torch.visual_gen.pipeline_registry import register_pipeline
from tensorrt_llm._torch.visual_gen.quantization.ops import quantize_fp8_blockwise, quantize_nvfp4
from tensorrt_llm._torch.visual_gen.utils import postprocess_video_tensor
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.utils.fp8_utils import (
    align,
    inverse_transform_sf,
    resmooth_to_fp8_e8m0,
    transform_sf_into_required_layout,
)

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
from .parallel_vae import tile_parallel_decode
from .pipeline_ltx2 import (
    LTX2Pipeline,
    _assert_resolution,
    _find_safetensors_files,
    _LTX2CUDAGraphRunner,
    _prefetch_ltx2_safetensors_files,
)

STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# Baseline BF16 peak memory ~75 GiB, saving BF16 weights snapshot total ~108 GiB.
_BF16_WEIGHTS_SNAPSHOT_FREE_MEMORY_THRESHOLD_GIB = 115.0
_QKV_SUFFIXES = (".to_q", ".to_k", ".to_v")


# ---------------------------------------------------------------------------
# LoRA helpers
# ---------------------------------------------------------------------------


def _get_free_gpu_memory_gib(
    device: Optional[Union[torch.device, str, int]] = None,
) -> Optional[float]:
    """Return free memory on the requested CUDA device, or ``None`` if unavailable."""
    if not torch.cuda.is_available():
        return None

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device=device)
    except (RuntimeError, OSError) as exc:
        logger.warning(
            f"Unable to query CUDA free memory for BF16 weight snapshots on device {device}: {exc}"
        )
        return None

    return free_bytes / (1024**3)


def _should_save_bf16_weights(
    device: Optional[Union[torch.device, str, int]] = None,
    preload_free_gib: Optional[float] = None,
    threshold_gib: float = _BF16_WEIGHTS_SNAPSHOT_FREE_MEMORY_THRESHOLD_GIB,
) -> bool:
    free_gib = preload_free_gib
    source = "pre-load"
    if free_gib is None:
        free_gib = _get_free_gpu_memory_gib(device=device)
        source = "current"

    if free_gib is None:
        logger.debug("BF16 weight snapshots disabled: CUDA free memory is unavailable")
        return False

    save_state = free_gib > threshold_gib
    relation = ">" if save_state else "<="
    logger.debug(
        f"BF16 weight snapshots {'enabled' if save_state else 'disabled'} "
        f"on device {device}: {source} free GPU memory {free_gib:.2f} GiB "
        f"{relation} {threshold_gib:.2f} GiB threshold"
    )
    return save_state


def _map_lora_param_name(base_name: str, strip_prefixes: List[str]) -> str:
    param_name = base_name
    for prefix in strip_prefixes:
        if param_name.startswith(prefix):
            param_name = param_name[len(prefix) :]
            break

    # Apply the same key remapping as LTXModel.load_weights() so LoRA delta
    # keys match TRT-LLM parameter names.
    for ff_prefix in (".ff.", ".audio_ff."):
        if ff_prefix + "net.0.proj" in param_name:
            param_name = param_name.replace(ff_prefix + "net.0.proj", ff_prefix + "up_proj")
        elif ff_prefix + "net.2" in param_name:
            param_name = param_name.replace(ff_prefix + "net.2", ff_prefix + "down_proj")
    param_name = param_name.replace(".q_norm.", ".norm_q.")
    param_name = param_name.replace(".k_norm.", ".norm_k.")
    return param_name


def _has_lora_target(param_name: str, model_params: set[str]) -> bool:
    if param_name in model_params or f"{param_name}.weight" in model_params:
        return True

    for suffix in _QKV_SUFFIXES:
        if param_name.endswith(suffix):
            attn_prefix = param_name[: -len(suffix)]
            return f"{attn_prefix}.qkv_proj.weight" in model_params
    return False


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
    # This helper can run in a background thread while base components load.
    # Avoid distributed prefetch collectives here; every rank must enter those
    # from the same foreground load sequence to avoid hangs.

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

    model_params = {n for n, _ in transformer.named_parameters()}
    deltas: Dict[str, torch.Tensor] = {}
    skipped_non_targets = 0
    for base_name in down_keys:
        if base_name not in up_keys:
            continue

        param_name = _map_lora_param_name(base_name, strip_prefixes)
        if not _has_lora_target(param_name, model_params):
            skipped_non_targets += 1
            continue

        A = down_keys[base_name]  # (rank, in_features)
        B = up_keys[base_name]  # (out_features, rank)
        rank = A.shape[0]
        alpha = alpha_dict.get(base_name, float(rank))
        scale = strength * alpha / rank
        delta = (B.float() @ A.float()) * scale
        deltas[param_name] = delta

    # Fuse separate to_q / to_k / to_v deltas into a single qkv_proj
    # delta when the transformer uses QKV fusion (FUSE_QKV mode).
    fused_keys: set = set()
    qkv_groups: Dict[str, List[str]] = {}
    for key in list(deltas.keys()):
        for suffix in _QKV_SUFFIXES:
            if key.endswith(suffix):
                attn_prefix = key[: -len(suffix)]
                qkv_groups.setdefault(attn_prefix, []).append(key)
                break

    for attn_prefix, keys in qkv_groups.items():
        q_key = f"{attn_prefix}.to_q"
        k_key = f"{attn_prefix}.to_k"
        v_key = f"{attn_prefix}.to_v"
        fused_key = f"{attn_prefix}.qkv_proj"

        if not (q_key in deltas and k_key in deltas and v_key in deltas):
            continue

        if f"{fused_key}.weight" not in model_params:
            continue

        deltas[fused_key] = torch.cat([deltas[q_key], deltas[k_key], deltas[v_key]], dim=0)
        fused_keys.update((q_key, k_key, v_key))

    for key in fused_keys:
        del deltas[key]

    logger.info(
        f"Loaded {len(deltas)} LoRA deltas from {lora_path} "
        f"(strength={strength}, skipped_non_targets={skipped_non_targets})"
    )
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
    """Dequantize FP8 E4M3 weight back to BF16.

    Handles per-tensor scales (scalar), standard float32 block-scale grids,
    and the packed int32 layout.
    """
    bf16 = fp8_weight.to(torch.bfloat16)

    if weight_scale.numel() == 1:
        return bf16 * weight_scale.float().item()

    out_features, in_features = fp8_weight.shape

    if packed:
        block_scale = inverse_transform_sf(
            weight_scale,
            mn=out_features,
            k=in_features,
            block_size=block_size,
        )
    else:
        block_scale = weight_scale

    scale = block_scale.repeat_interleave(block_size, dim=0)[:out_features]
    scale = scale.repeat_interleave(block_size, dim=1)[:, :in_features]
    return bf16 * scale.to(bf16.device)


def _requantize_fp8_weight(
    bf16_weight: torch.Tensor,
    repack: bool = False,
    block_size: int = 128,
    per_tensor: bool = False,
) -> tuple:
    """Quantize BF16 weight to FP8 E4M3.

    When *per_tensor* is True, a single scalar scale is computed from the
    tensor-wide absmax.  Otherwise 128x128 block scales are used.

    When *repack* is True (block-scale only) the returned weight/scale pair
    is post-processed through ``resmooth_to_fp8_e8m0`` +
    ``transform_sf_into_required_layout`` so they match the packed layout
    expected by SM100f / SM120 GEMM kernels.

    Returns ``(qweight, scale)``.
    """
    if per_tensor:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        amax = bf16_weight.float().abs().max().clamp(min=1e-12)
        scale = amax / fp8_max
        qw = (bf16_weight.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
        return qw, scale.to(bf16_weight.device)

    qw, scale = quantize_fp8_blockwise(bf16_weight, block_size)

    if repack:
        qw, scale = resmooth_to_fp8_e8m0(qw, scale)
        scale = transform_sf_into_required_layout(
            scale,
            mn=qw.shape[0],
            k=qw.shape[1],
            recipe=(1, 128, 128),
            is_sfa=False,
        )

    return qw, scale


def _scale_like(scale: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return scale.to(device=reference.device, dtype=reference.dtype).reshape(reference.shape)


def _apply_lora_deltas(
    module: torch.nn.Module,
    deltas: Dict[str, torch.Tensor],
    sign: float = 1.0,
    save_bf16_weights: bool = False,
) -> tuple:
    """Add (sign=+1) or remove (sign=-1) pre-computed LoRA deltas.

    For BF16 weights the delta is added directly and later removed either
    by restoring an optional saved snapshot or by subtracting the same delta.
    For FP8-quantized weights (same shape, float8 dtype), we
    dequantize → apply → requantize.  FP4 weights are handled through
    the packed-FP4 branch because the current static and dynamic NVFP4
    load paths both store packed FP4 weights by the time LoRA deltas are
    applied.  For packed FP4, merged weights are kept in BF16 for stage 2
    inference.  The parent ``Linear`` module's ``quant_method`` is swapped
    to ``UnquantizedLinearMethod`` so inference runs with plain
    ``F.linear``.  The quantized original tensors and ``quant_method`` are
    saved so that ``_restore_lora_state`` can fully restore the original
    state afterwards.

    Returns ``(applied_count, saved_lora_state, snapshot_required_count)`` where
    *saved_lora_state* maps each touched snapshotted parameter name to its
    original tensor.  BF16 weights are snapshotted only when
    *save_bf16_weights* is true.  This does not change FP8 or FP4 LoRA
    handling: those paths always snapshot quantized state.  For packed FP4 it
    also stores the parent
    ``quant_method`` so that stage 2 can run with BF16 weights and then
    restore the exact original FP4 state without another quantization round
    trip.  *snapshot_required_count* is the number of weights that must be
    restored from saved snapshots.
    """
    applied = 0
    snapshot_required = 0
    saved_state: Dict[str, Any] = {}
    applied_deltas: Dict[str, torch.Tensor] = {}
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

    # Always build module path → module mapping for quant_method swapping.
    module_dict: Dict[str, torch.nn.Module] = {}
    for raw_name, mod in module.named_modules():
        clean = raw_name.replace("._orig_mod.", ".")
        module_dict[clean] = mod

    try:
        for name, delta in deltas.items():
            param_name = name if name in state else f"{name}.weight"
            if param_name not in state:
                continue

            param = state[param_name]
            base = param_name.rsplit(".weight", 1)[0]

            # --- same shape ---------------------------------------------------
            if param.shape == delta.shape:
                if param.dtype in _FP8_DTYPES:
                    # FP8: dequant -> apply -> requant
                    scale_key = f"{base}.weight_scale"
                    if scale_key not in state:
                        raise RuntimeError(
                            f"Cannot apply LoRA delta to FP8 param '{param_name}': missing {scale_key}."
                        )
                    ws_param = state[scale_key]
                    out_f, in_f = delta.shape
                    is_per_tensor = ws_param.data.numel() == 1
                    is_packed = not is_per_tensor and _is_fp8_scale_packed(
                        ws_param.data, out_f, in_f
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
                        per_tensor=is_per_tensor,
                    )
                    new_scale = _scale_like(new_scale, ws_param.data)
                    param.data.copy_(qw)
                    ws_param.data.copy_(new_scale)
                    snapshot_required += 1
                else:
                    if save_bf16_weights and param.dtype == torch.bfloat16:
                        saved_state[param_name] = param.data.clone()
                        snapshot_required += 1
                    # BF16: direct in-place addition, then restore by snapshot
                    # copy when memory allows or by subtracting the delta.
                    param.data.add_(
                        delta.to(param.device, param.dtype),
                        alpha=sign,
                    )
                applied_deltas[name] = delta
                applied += 1

            # --- packed FP4 (half last dim) -----------------------------------
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

                linear_mod = module_dict.get(base)
                if linear_mod is None or not isinstance(linear_mod, Linear):
                    raise RuntimeError(
                        f"Packed FP4 LoRA merge: could not find Linear module at '{base}'."
                    )

                # Packed FP4: keep the LoRA-merged weight in BF16 for stage 2.
                # This replaces packed FP4 storage (out, in//2) with BF16 storage
                # (out, in) and swaps the parent Linear to plain F.linear.
                param.data = bf16
                saved_state[f"__quant_method__{base}"] = linear_mod.quant_method
                linear_mod.quant_method = UnquantizedLinearMethod()
                snapshot_required += 1
                applied_deltas[name] = delta
                applied += 1
            else:
                logger.warning(
                    f"Shape mismatch for LoRA param '{param_name}': "
                    f"param={list(param.shape)}, delta={list(delta.shape)}. "
                    f"Skipping."
                )
    except Exception:
        if saved_state:
            _restore_lora_state(module, saved_state)
        if applied_deltas:
            _subtract_dense_lora_deltas(module, applied_deltas, saved_state)
        raise
    return applied, saved_state, snapshot_required


def _subtract_dense_lora_deltas(
    module: torch.nn.Module,
    deltas: Dict[str, torch.Tensor],
    saved_state: Dict[str, Any],
) -> int:
    """Remove LoRA deltas from dense floating-point weights.

    Quantized weights are skipped here because FP8 and FP4 are restored from
    exact snapshots in ``saved_state``.
    """
    restored = 0
    state: Dict[str, torch.nn.Parameter] = {}
    for raw_name, param in module.named_parameters():
        clean = raw_name.replace("._orig_mod.", ".")
        state[clean] = param

    for name, delta in deltas.items():
        param_name = name if name in state else f"{name}.weight"
        if param_name not in state or param_name in saved_state:
            continue

        param = state[param_name]
        if param.shape != delta.shape or param.dtype in _FP8_DTYPES:
            continue

        if not param.data.is_floating_point():
            continue

        param.data.add_(
            delta.to(param.device, param.dtype),
            alpha=-1.0,
        )
        restored += 1

    return restored


def _count_saved_lora_weight_tensors(saved_state: Dict[str, Any]) -> int:
    """Count LoRA-touched base weights restored from snapshots."""
    return sum(
        1
        for name in saved_state
        if not name.startswith("__quant_method__")
        and (name == "weight" or name.endswith(".weight"))
    )


def _restore_lora_state(
    module: torch.nn.Module,
    saved_state: Dict[str, Any],
) -> None:
    """Restore parameters and quantization state saved by ``_apply_lora_deltas``.

    Handles three cases:
    - Regular tensors (same shape/dtype): restored via ``.data.copy_()``.
    - BF16-swapped FP4 weights (shape/dtype changed for stage 2):
      restored via ``.data =`` assignment to replace the storage.
    - Saved ``quant_method`` objects (keys starting with
      ``__quant_method__``): re-assigned to the corresponding Linear module.
    """
    state: Dict[str, torch.nn.Parameter] = {}
    for raw_name, param in module.named_parameters():
        clean = raw_name.replace("._orig_mod.", ".")
        state[clean] = param

    module_dict: Dict[str, torch.nn.Module] = {}
    for raw_name, mod in module.named_modules():
        clean = raw_name.replace("._orig_mod.", ".")
        module_dict[clean] = mod

    for name, data in saved_state.items():
        if name.startswith("__quant_method__"):
            mod_path = name[len("__quant_method__") :]
            mod = module_dict.get(mod_path)
            if mod is None or not isinstance(mod, Linear):
                raise RuntimeError(
                    f"Could not restore quant_method for Linear module '{mod_path}'."
                )
            mod.quant_method = data
        elif name in state:
            param = state[name]
            if param.data.shape == data.shape and param.data.dtype == data.dtype:
                param.data.copy_(data)
            else:
                # Shape or dtype changed (e.g. BF16 swap): replace storage.
                param.data = data


@dataclass
class _PersistentLoRAParamState:
    param_name: str
    precision: str
    weight_param: torch.nn.Parameter
    original_weight: torch.Tensor
    merged_weight: torch.Tensor
    scale_params: Dict[str, torch.nn.Parameter] = field(default_factory=dict)
    original_scales: Dict[str, torch.Tensor] = field(default_factory=dict)
    merged_scales: Dict[str, torch.Tensor] = field(default_factory=dict)
    linear_module: Optional[Linear] = None
    original_quant_method: Optional[Any] = None
    merged_quant_method: Optional[Any] = None


class _PersistentLoRAWeightCache:
    """Keep unmerged and merged LoRA-touched weights resident.

    The cache is built when distilled LoRA is loaded and used only for LTX-2
    Stage 2 distilled LoRA.
    It removes per-request merge/unmerge math by rebinding parameter storage to
    precomputed resident tensors.  FP8 and FP4 keep exact original quantized
    state.  FP4's merged state is BF16 and swaps the parent Linear to
    UnquantizedLinearMethod, matching the existing per-request Stage 2 path.
    """

    def __init__(
        self,
        entries: List[_PersistentLoRAParamState],
    ) -> None:
        self._entries = entries
        self._bound_state = "original"
        self.applied_count = len(entries)

    @staticmethod
    def _module_state(
        module: torch.nn.Module,
    ) -> tuple[Dict[str, torch.nn.Parameter], Dict[str, torch.nn.Module]]:
        state: Dict[str, torch.nn.Parameter] = {}
        for raw_name, param in module.named_parameters():
            clean = raw_name.replace("._orig_mod.", ".")
            state[clean] = param

        module_dict: Dict[str, torch.nn.Module] = {}
        for raw_name, mod in module.named_modules():
            clean = raw_name.replace("._orig_mod.", ".")
            module_dict[clean] = mod

        return state, module_dict

    @classmethod
    def build(
        cls,
        module: torch.nn.Module,
        deltas: Dict[str, torch.Tensor],
    ) -> "_PersistentLoRAWeightCache":
        state, module_dict = cls._module_state(module)
        entries: List[_PersistentLoRAParamState] = []

        for name, delta in deltas.items():
            param_name = name if name in state else f"{name}.weight"
            if param_name not in state:
                continue

            param = state[param_name]
            base = param_name.rsplit(".weight", 1)[0]

            if param.shape == delta.shape:
                if param.dtype in _FP8_DTYPES:
                    scale_key = f"{base}.weight_scale"
                    if scale_key not in state:
                        raise RuntimeError(
                            f"Cannot build persistent LoRA state for FP8 param "
                            f"'{param_name}': missing {scale_key}."
                        )

                    ws_param = state[scale_key]
                    out_f, in_f = delta.shape
                    is_per_tensor = ws_param.data.numel() == 1
                    is_packed = not is_per_tensor and _is_fp8_scale_packed(
                        ws_param.data, out_f, in_f
                    )

                    bf16 = _dequantize_fp8_weight(
                        param.data,
                        ws_param.data,
                        packed=is_packed,
                    )
                    bf16.add_(delta.to(bf16.device, bf16.dtype))
                    qw, new_scale = _requantize_fp8_weight(
                        bf16,
                        repack=is_packed,
                        per_tensor=is_per_tensor,
                    )
                    new_scale = _scale_like(new_scale, ws_param.data)

                    entries.append(
                        _PersistentLoRAParamState(
                            param_name=param_name,
                            precision="fp8",
                            weight_param=param,
                            original_weight=param.data,
                            merged_weight=qw,
                            scale_params={scale_key: ws_param},
                            original_scales={scale_key: ws_param.data},
                            merged_scales={scale_key: new_scale},
                        )
                    )
                else:
                    merged = param.data.clone()
                    merged.add_(delta.to(merged.device, merged.dtype))
                    precision = "bf16" if param.dtype == torch.bfloat16 else str(param.dtype)
                    entries.append(
                        _PersistentLoRAParamState(
                            param_name=param_name,
                            precision=precision,
                            weight_param=param,
                            original_weight=param.data,
                            merged_weight=merged,
                        )
                    )
                continue

            if (
                param.ndim == 2
                and delta.ndim == 2
                and param.shape[0] == delta.shape[0]
                and param.shape[1] * 2 == delta.shape[1]
            ):
                scale_key = f"{base}.weight_scale"
                scale2_key = f"{base}.weight_scale_2"
                if scale_key not in state or scale2_key not in state:
                    raise RuntimeError(
                        f"Cannot build persistent LoRA state for packed FP4 param "
                        f"'{param_name}': missing {scale_key} or {scale2_key}."
                    )

                ws_param = state[scale_key]
                ws2_param = state[scale2_key]
                out_features, in_features = delta.shape

                bf16 = _dequantize_fp4_weight(
                    param.data,
                    ws_param.data,
                    ws2_param.data,
                    out_features,
                    in_features,
                )
                bf16.add_(delta.to(bf16.device, bf16.dtype))

                linear_mod = module_dict.get(base)
                if linear_mod is None or not isinstance(linear_mod, Linear):
                    raise RuntimeError(
                        f"Packed FP4 persistent LoRA state: could not find "
                        f"Linear module at '{base}'."
                    )

                entries.append(
                    _PersistentLoRAParamState(
                        param_name=param_name,
                        precision="fp4",
                        weight_param=param,
                        original_weight=param.data,
                        merged_weight=bf16,
                        scale_params={
                            scale_key: ws_param,
                            scale2_key: ws2_param,
                        },
                        original_scales={
                            scale_key: ws_param.data,
                            scale2_key: ws2_param.data,
                        },
                        merged_scales={
                            scale_key: ws_param.data,
                            scale2_key: ws2_param.data,
                        },
                        linear_module=linear_mod,
                        original_quant_method=linear_mod.quant_method,
                        merged_quant_method=UnquantizedLinearMethod(),
                    )
                )
                continue

            logger.warning(
                f"Shape mismatch for persistent LoRA param '{param_name}': "
                f"param={list(param.shape)}, delta={list(delta.shape)}. "
                f"Skipping."
            )

        return cls(entries)

    def bind_original(self) -> None:
        for entry in self._entries:
            entry.weight_param.data = entry.original_weight
            for scale_name, scale_param in entry.scale_params.items():
                scale_param.data = entry.original_scales[scale_name]
            if entry.linear_module is not None:
                entry.linear_module.quant_method = entry.original_quant_method
        self._bound_state = "original"

    def bind_merged(self) -> None:
        for entry in self._entries:
            entry.weight_param.data = entry.merged_weight
            for scale_name, scale_param in entry.scale_params.items():
                scale_param.data = entry.merged_scales[scale_name]
            if entry.linear_module is not None:
                entry.linear_module.quant_method = entry.merged_quant_method
        self._bound_state = "merged"

    def precision_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self._entries:
            counts[entry.precision] = counts.get(entry.precision, 0) + 1
        return counts


class _LTX2TwoStageCUDAGraphRunner(_LTX2CUDAGraphRunner):
    """CUDA graph runner keyed by LTX-2 two-stage LoRA weight state."""

    def __init__(
        self,
        config: CUDAGraphRunnerConfig,
        lora_state_getter: Callable[[], str],
    ) -> None:
        super().__init__(config)
        self._lora_state_getter = lora_state_getter

    def get_graph_key(self, *args, **kwargs):
        return (
            *super().get_graph_key(*args, **kwargs),
            ("ltx2_two_stage_lora_state", self._lora_state_getter()),
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


# Registered without ``hf_ids`` / ``defaults`` on purpose: this class is
# only reached via ``LTX2Pipeline.resolve_variant()`` swap, never through
# class-name dispatch from the registry, so its own discovery surface
# entry would duplicate the canonical ``LTX2Pipeline`` entry above and
# make ``supported_models()`` / ``pipeline_config()`` ambiguous.
@register_pipeline("LTX2TwoStagesPipeline")
class LTX2TwoStagesPipeline(LTX2Pipeline):
    """Lightricks LTX-Video two-stage text-to-video with audio.

    Stage 1: denoise at half spatial resolution with full guidance.
    Stage 2: learned 2x spatial upsample, refinement denoising with
             the distilled sigma schedule (no guidance, distilled LoRA),
             then decode.
    """

    @property
    def common_warmup_shapes(self) -> list:
        return [(512, 768, 121)]

    def _current_lora_cuda_graph_state(self) -> str:
        return getattr(self, "_lora_cuda_graph_state", "original")

    def _is_cuda_graph_enabled(self) -> bool:
        for config_name in ("pipeline_config", "model_config"):
            config = getattr(self, config_name, None)
            cuda_graph = getattr(config, "cuda_graph", None)
            if getattr(cuda_graph, "enable", False):
                return True
        return False

    def _assert_cuda_graph_safe_lora_bindings(self) -> None:
        if not self._is_cuda_graph_enabled():
            return
        if not getattr(self, "_distilled_lora_deltas", {}):
            return
        if getattr(self, "_distilled_lora_weight_cache", None) is not None:
            return

        raise RuntimeError(
            "LTX-2 two-stage CUDA graph requires persistent LoRA weights. "
            "The non-persistent distilled LoRA path mutates parameter storage "
            "and quantization state during Stage 2, which is not CUDA-graph safe. "
            "Disable CUDA graph or ensure the persistent LoRA cache can be built."
        )

    def _setup_cuda_graphs(self):
        """Wrap transformer.forward with a LoRA-state-aware CUDA graph key."""
        if not self.pipeline_config.cuda_graph.enable:
            return

        runner = _LTX2TwoStageCUDAGraphRunner(
            CUDAGraphRunnerConfig(use_cuda_graph=True),
            self._current_lora_cuda_graph_state,
        )
        compile_note = " (with torch.compile)" if self.pipeline_config.torch_compile.enable else ""
        logger.info(
            "CUDA graph runner: wrapping LTX-2 two-stage transformer.forward "
            f"with LoRA state key{compile_note}"
        )
        self.transformer.forward = runner.wrap(self.transformer.forward)
        self._cuda_graph_runners["transformer"] = runner

    # ------------------------------------------------------------------
    # Component loading
    # ------------------------------------------------------------------

    def load_standard_components(
        self,
        checkpoint_dir: str,
        device: torch.device,
        skip_components: Optional[list] = None,
        **kwargs,
    ) -> None:
        # The BF16 snapshot threshold is a whole-pipeline capacity gate, so
        # record it before loading model/runtime components that consume HBM.
        self._bf16_snapshot_preload_free_gib = _get_free_gpu_memory_gib(device=device)
        dtype = self.pipeline_config.torch_dtype
        spatial_upsampler_path = self.pipeline_config.extra_attrs.get("spatial_upsampler_path", "")
        distilled_lora_path = self.pipeline_config.extra_attrs.get("distilled_lora_path", "")

        lora_executor = None
        lora_future = None
        if distilled_lora_path:
            logger.info(f"Starting distilled LoRA pre-compute from {distilled_lora_path}...")
            lora_executor = ThreadPoolExecutor(max_workers=1)
            lora_future = lora_executor.submit(
                _load_lora_deltas,
                distilled_lora_path,
                self.transformer,
                self._TRANSFORMER_PREFIX,
            )

        try:
            super().load_standard_components(
                checkpoint_dir,
                device,
                skip_components=skip_components,
                **kwargs,
            )

            self._load_two_stage_components(
                device,
                dtype,
                spatial_upsampler_path,
                distilled_lora_path,
                lora_future,
            )
        finally:
            if lora_executor is not None:
                lora_executor.shutdown(wait=True)

    def _load_two_stage_components(
        self,
        device: torch.device,
        dtype: torch.dtype,
        spatial_upsampler_path: str,
        distilled_lora_path: str,
        lora_future,
    ) -> None:
        # --- Spatial upsampler ---
        if spatial_upsampler_path:
            logger.info(f"Loading spatial upsampler from {spatial_upsampler_path}...")
            sft_paths = _find_safetensors_files(spatial_upsampler_path)
            if not sft_paths:
                raise ValueError(f"No safetensors files found at {spatial_upsampler_path}")
            _prefetch_ltx2_safetensors_files(sft_paths)

            config: Dict[str, Any] = {}
            try:
                with safetensors.torch.safe_open(sft_paths[0], framework="pt") as f:
                    meta = f.metadata()
                    if meta and "config" in meta:
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
        self._distilled_lora_weight_cache: Optional[_PersistentLoRAWeightCache] = None
        if distilled_lora_path:
            logger.info("Waiting for distilled LoRA pre-compute...")
            if lora_future is None:
                raise RuntimeError("Distilled LoRA pre-compute was not started.")
            self._distilled_lora_deltas = lora_future.result()
            logger.info(
                f"Distilled LoRA ready: {len(self._distilled_lora_deltas)} parameter deltas"
            )
            try:
                self._distilled_lora_weight_cache = _PersistentLoRAWeightCache.build(
                    self.transformer,
                    self._distilled_lora_deltas,
                )
            except torch.cuda.OutOfMemoryError as exc:
                logger.warning(
                    "Persistent LTX-2 LoRA weights disabled after CUDA OOM "
                    f"during cache build: {exc}"
                )
                torch.cuda.empty_cache()
                self._distilled_lora_weight_cache = None
            else:
                self._distilled_lora_weight_cache.bind_original()
                logger.info(
                    "Persistent LTX-2 LoRA weights ready: "
                    f"{self._distilled_lora_weight_cache.applied_count} params, "
                    f"precision_counts={self._distilled_lora_weight_cache.precision_counts()}"
                )
        self._assert_cuda_graph_safe_lora_bindings()

    # ------------------------------------------------------------------
    # Inference entry point
    # ------------------------------------------------------------------

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
            output_type=extra["output_type"],
            guidance_rescale=extra["guidance_rescale"],
            max_sequence_length=req.params.max_sequence_length,
            image=req.params.image,
            image_cond_strength=extra["image_cond_strength"],
            stg_scale=extra["stg_scale"],
            stg_blocks=extra["stg_blocks"],
            modality_scale=extra["modality_scale"],
            rescale_scale=extra["rescale_scale"],
            guidance_skip_step=extra["guidance_skip_step"],
            enhance_prompt=extra["enhance_prompt"],
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
        guidance_scale: float = 3.0,
        guidance_rescale: float = 0.0,
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
        self._lora_cuda_graph_state = "original"
        if enhance_prompt:
            logger.info("Enhancing prompt with Gemma3 (two-stage)...")
            prompt_text = prompt if isinstance(prompt, str) else prompt[0]
            prompt = self._enhance_prompt(prompt_text, seed=seed)
            # Downstream calls should not re-enhance the prompt.
            enhance_prompt = False

        _assert_resolution(height, width, is_two_stage=True)
        pipeline_start = time.time()
        # Two-stage timing: stage 1 is reported as ``denoise``; stage 2
        # (spatial upsample + refinement denoise + decode) folds into
        # ``post_denoise``. Only the outer timer's numbers reach
        # ``PipelineOutput``.
        timer = CudaPhaseTimer()
        timer.mark_pre_start()
        height_s1 = height // 2
        width_s1 = width // 2
        logger.info(f"LTX2 two-stage: stage1 at {height_s1}x{width_s1}, final {height}x{width}")

        # ================================================================
        # Stage 1: denoise at half resolution
        # ================================================================
        timer.mark_denoise_start()
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

        timer.mark_post_start()

        # Non-primary workers (rank != 0) receive None from
        # decode_latents and exit here.  Rank 0 continues with Stage 2.
        if video_latents is None:
            timer.mark_end()
            return timer.fill(PipelineOutput(video=None, audio=None, frame_rate=float(frame_rate)))

        # ================================================================
        # Stage 2: spatial upsample + refinement denoise (rank-0 only)
        # ================================================================
        # Only rank 0 refines; other vae_ranks skip Stage 2 and rejoin at the
        # collective decode below, receiving the refined latents via broadcast.
        if self.rank == 0:
            video_latents, audio_latents = self._upsample_and_refine(
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
        else:
            video_latents, audio_latents = None, None

        # ================================================================
        # Decode
        # ================================================================
        if output_type == "latent":
            # No decode: only rank 0 holds the refined latents.
            video_out, audio_out = (
                (video_latents, audio_latents) if self.rank == 0 else (None, None)
            )
            if self.rank == 0:
                logger.info(f"Two-stage total time: {time.time() - pipeline_start:.2f}s")
            timer.mark_end()
            return timer.fill(
                PipelineOutput(
                    video=video_out,
                    audio=audio_out,
                    frame_rate=float(frame_rate),
                    audio_sample_rate=(
                        int(self.audio_sampling_rate)
                        if getattr(self, "audio_sampling_rate", None) is not None
                        and audio_out is not None
                        else None
                    ),
                )
            )

        if self._parallel_vae_enabled:
            # Broadcast rank-0's refined Stage-2 latents to every vae_rank, then
            # decode collectively (tile-parallel over vgm.vae_group).
            vgm = self.pipeline_config.visual_gen_mapping
            video_latents = self._broadcast_video_latents(video_latents, vgm.vae_group)
            logger.info("Decoding upsampled video (tile-parallel)...")
            video = tile_parallel_decode(
                self.video_decoder,
                video_latents,
                TilingConfig.default(),
                pg=vgm.vae_group,
            )
            video = postprocess_video_tensor(video)
        else:
            logger.info("Decoding upsampled video (tiled)...")
            video_latents = video_latents.to(self.dtype)
            chunks = list(
                self.video_decoder.tiled_decode(
                    video_latents,
                    TilingConfig.default(),
                    generator=None,
                )
            )
            video = torch.cat(chunks, dim=2)
            video = postprocess_video_tensor(video)

        # Audio decode is rank-0 only (not tile-parallel).
        audio_out = None
        if self.rank == 0 and audio_latents is not None:
            audio_latents = audio_latents.to(self.dtype)
            audio_out = decode_audio(audio_latents, self.audio_decoder, self.vocoder)

        if self.rank == 0:
            logger.info(f"Two-stage total time: {time.time() - pipeline_start:.2f}s")
        timer.mark_end()
        return timer.fill(
            PipelineOutput(
                video=video,
                audio=audio_out,
                frame_rate=float(frame_rate),
                audio_sample_rate=(
                    int(self.audio_sampling_rate)
                    if getattr(self, "audio_sampling_rate", None) is not None
                    and audio_out is not None
                    else None
                ),
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _upsample_and_refine(
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
        """Stage 2 (rank-0 only): learned 2x spatial upsample + refinement denoise.

        Returns the refined ``(video_latents, audio_latents)`` in 5-D form.
        """
        per_ch_stats = self._get_per_channel_statistics()
        video_latents = upsample_video(
            video_latents[:1],
            per_ch_stats,
            self.spatial_upsampler,
        )
        logger.info("Upsampled video latents via learned upsampler")

        # The persistent cache owns original and merged tensors when it can be
        # built at load time. Stage 2 only rebinds pointers and FP4 quant_method
        # state, so no per-request clone, merge, or unmerge math is needed.
        self._assert_cuda_graph_safe_lora_bindings()
        lora_cache = self._distilled_lora_weight_cache
        using_persistent_lora = lora_cache is not None
        saved_lora_state: Dict[str, Any] = {}
        snapshot_required = 0
        n = 0
        dense_lora_merge_completed = False
        stage2_start = time.time()
        try:
            if using_persistent_lora:
                lora_cache.bind_merged()
                self._lora_cuda_graph_state = "merged"
                n = lora_cache.applied_count
                logger.info(f"Bound persistent distilled LoRA ({n} params) for stage 2")
            else:
                transformer_device = next(self.transformer.parameters()).device
                preload_free_gib = getattr(self, "_bf16_snapshot_preload_free_gib", None)
                save_bf16_weights = _should_save_bf16_weights(
                    device=transformer_device,
                    preload_free_gib=preload_free_gib,
                )
                n, saved_lora_state, snapshot_required = _apply_lora_deltas(
                    self.transformer,
                    self._distilled_lora_deltas,
                    sign=1.0,
                    save_bf16_weights=save_bf16_weights,
                )
                dense_lora_merge_completed = True
                self._lora_cuda_graph_state = "merged"
                logger.info(f"Merged distilled LoRA ({n} params) for stage 2 (BF16 weights)")

            # Disable Ulysses for Stage 2: only rank 0 is active, so
            # cross-rank collectives in the attention backend would hang.
            self.transformer.set_ulysses_enabled(False)
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
            stage2_denoise_time = time.time() - stage2_start
            logger.info(f"Stage 2 denoising time: {stage2_denoise_time:.2f}s (BF16 weights)")
            self.transformer.set_ulysses_enabled(True)
            if using_persistent_lora:
                lora_cache.bind_original()
                self._lora_cuda_graph_state = "original"
                logger.info("Re-bound persistent distilled LoRA original weights after stage 2")
            elif dense_lora_merge_completed:
                if snapshot_required and not saved_lora_state:
                    raise RuntimeError(
                        "LoRA state was not saved; cannot safely restore stage 2 weights."
                    )

                snapshot_restored = 0
                if snapshot_required:
                    # Restore every LoRA-touched parameter from its snapshot. Packed
                    # FP4 also restores the original quant_method.
                    _restore_lora_state(self.transformer, saved_lora_state)
                    snapshot_restored = _count_saved_lora_weight_tensors(saved_lora_state)

                # BF16 weights that were not snapshotted are restored by
                # subtracting LoRA deltas. FP8 and FP4 are exact snapshot restores.
                dense_restored = _subtract_dense_lora_deltas(
                    self.transformer,
                    self._distilled_lora_deltas,
                    saved_lora_state,
                )
                restored = snapshot_restored + dense_restored
                if restored != n:
                    raise RuntimeError(
                        f"Restored {restored} LoRA-touched weights after stage 2, but {n} were applied."
                    )
                self._lora_cuda_graph_state = "original"
                logger.info("Un-merged distilled LoRA after stage 2")
            else:
                self._lora_cuda_graph_state = "original"

        return video_latents, audio_latents

    def _broadcast_video_latents(
        self, video_latents: Optional[torch.Tensor], vae_group
    ) -> torch.Tensor:
        """Broadcast rank-0's refined Stage-2 latents to every ``vae_rank``.

        ``tile_parallel_decode`` needs the full latent replicated on each rank of
        ``vae_group``; only rank 0 ran Stage 2, so it is the broadcast source.
        Video latents are 5-D ``(B, C, F, H, W)``.
        """
        if vae_group is None:
            raise ValueError(
                "parallel VAE decode requires a valid vae_group, got None "
                "(a None group would fall back to the world group and hang on non-VAE ranks)."
            )
        if self.rank == 0:
            video_latents = video_latents.to(self.dtype).contiguous()
            shape = torch.tensor(video_latents.shape, dtype=torch.long, device=self.device)
        else:
            shape = torch.empty(5, dtype=torch.long, device=self.device)
        dist.broadcast(shape, src=0, group=vae_group)
        if self.rank != 0:
            video_latents = torch.empty(
                torch.Size(shape.tolist()), dtype=self.dtype, device=self.device
            )
        dist.broadcast(video_latents, src=0, group=vae_group)
        return video_latents

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

        # --- Text conditioning: reuse Stage 1 encoder output ---
        # Gemma3 + Connector outputs depend only on prompt text, not on
        # resolution or LoRA weights.
        video_embeds, audio_embeds, connector_mask = self._cached_encoder_output

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

        # --- Pre-compute static preproc (context, PE, KV) for Stage 2 ---
        _s2_static = self.transformer.prepare_text_cache(
            video_context=video_embeds,
            video_context_mask=connector_mask,
            video_positions=video_positions,
            audio_context=audio_embeds if a_working is not None else None,
            audio_context_mask=connector_mask if a_working is not None else None,
            audio_positions=audio_positions if a_working is not None else None,
            dtype=self.dtype,
        )

        # --- Euler denoising loop (no guidance) ---
        for i in range(len(sigmas) - 1):
            with nvtx_range(f"refinement_step {i}"):
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
                    text_cache=_s2_static,
                    step_index=i,
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
