# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in K3.1 checkpoint conversion at load time.

``TLLM_K31_LOAD_QUANTIZE=1`` selects NVFP4 experts and block-scaled FP8 attention.
The adapter owns source-format handling; runtime modules use the native loaders.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F

from ...logger import logger
from ...models.modeling_utils import QuantConfig
from ...quantization import QuantAlgo
from ..model_config import ModelConfig
from .checkpoints.base_weight_loader import ConsumableWeightsDict

if TYPE_CHECKING:
    from .modeling_kimi_linear import KimiLinearForCausalLM

_SWITCH = "TLLM_K31_LOAD_QUANTIZE"
_DEQUANT_MAX_CHUNK_ELEMENTS = 1 << 22
_KDA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "g_proj",
    "o_proj",
    "f_a_proj",
    "f_b_proj",
    "b_proj",
)
_MLA_PROJECTIONS = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "g_proj", "o_proj")


class _TensorSlice(Protocol):
    def get_shape(self) -> list[int]: ...

    def __getitem__(self, index: object) -> torch.Tensor: ...


def dequantize_mxfp4_weight(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode uint8 E2M1 [N,K/2] and E8M0 [N,K/32] into BF16 [N,K]."""
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("MXFP4 conversion requires uint8 packed weights and E8M0 scales")
    if packed.ndim != 2 or scales.ndim != 2 or packed.numel() == 0 or packed.shape[1] % 16:
        raise ValueError("MXFP4 conversion requires a matrix with K divisible by 32")
    if tuple(scales.shape) != (packed.shape[0], packed.shape[1] // 16):
        raise ValueError("MXFP4 scale shape does not cover the packed weight")
    if packed.device != scales.device:
        raise ValueError("MXFP4 weights and scales must be on the same device")
    if (scales == 255).any():
        raise ValueError("MXFP4 E8M0 scales contain NaN (255)")
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=packed.device,
    )
    rows, cols = packed.shape[0], packed.shape[1] * 2
    weight = torch.empty((rows, cols), dtype=torch.bfloat16, device=packed.device)
    chunk_rows = max(1, _DEQUANT_MAX_CHUNK_ELEMENTS // cols)
    # Bound unpacked indices and FP32 intermediates independently of the matrix size.
    for start in range(0, rows, chunk_rows):
        block = packed[start : start + chunk_rows]
        block_scales = scales[start : start + chunk_rows]
        indices = torch.stack((block & 15, block >> 4), dim=-1).int()
        values = table[indices].reshape(*block_scales.shape, 32)
        decoded = torch.ldexp(values, (block_scales.int() - 127).unsqueeze(-1)).bfloat16()
        weight[start : start + chunk_rows].copy_(decoded.reshape(block.shape[0], cols))
        del indices, values, decoded
    if not torch.isfinite(weight).all():
        raise ValueError("MXFP4 weights are not representable in BF16")
    return weight


def quantize_expert_weight(
    weight: torch.Tensor, *, amax: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize CUDA BF16 [N,K] to packed NVFP4, linear E4M3 scales and an FP32 multiplier.

    ``amax`` may cover both gate/up projections so a fused GEMM can share their multiplier.
    """
    if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        raise TypeError("K3.1 expert quantization requires a CUDA BF16 matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("K3.1 expert projection contains non-finite values")
    if amax is None:
        amax = weight.abs().amax().float()
    global_scale = torch.where(amax == 0, torch.ones_like(amax), amax / (448.0 * 6.0))
    inverse_scale = global_scale.reciprocal()
    if not torch.isfinite(inverse_scale) or inverse_scale <= 0:
        raise ValueError("K3.1 expert global scale is not representable in FP32")
    packed, scales = torch.ops.trtllm.fp4_quantize(
        weight.contiguous(), inverse_scale, 16, False, False
    )
    return packed, scales.view(weight.shape[0], -1), global_scale


def quantize_attention_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 [N,K] projection into E4M3 and FP32 128x128 block scales."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise TypeError("K3.1 attention conversion requires a BF16 matrix")
    if not torch.isfinite(weight).all():
        raise ValueError("K3.1 attention projection contains non-finite values")
    rows, cols = weight.shape
    block = 128
    nr, nc = math.ceil(rows / block), math.ceil(cols / block)
    padded = F.pad(weight.float(), (0, nc * block - cols, 0, nr * block - rows))
    tiles = padded.reshape(nr, block, nc, block)
    amax = tiles.abs().amax(dim=(1, 3))
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 448.0)
    codes = (tiles / scale[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return codes.reshape(nr * block, nc * block)[:rows, :cols].contiguous(), scale


class _AttentionPair:
    """Hold at most one converted projection until its scale is consumed."""

    def __init__(self, source: torch.Tensor | _TensorSlice, device: torch.device) -> None:
        self.source = source
        self.device = device
        self.shape = (
            tuple(source.shape) if isinstance(source, torch.Tensor) else tuple(source.get_shape())
        )
        self.codes: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None

    def materialize(self) -> None:
        if self.scales is None:
            source = self.source if isinstance(self.source, torch.Tensor) else self.source[:]
            codes, scales = quantize_attention_weight(source.to(self.device))
            self.codes, self.scales = codes.cpu(), scales.cpu()


class _AttentionValue:
    """Safetensors-slice interface retained through the VLM prefix filter."""

    def __init__(self, pair: _AttentionPair, *, scale: bool = False) -> None:
        self.pair = pair
        self.scale = scale

    def get_shape(self) -> list[int]:
        if self.scale:
            return [math.ceil(dim / 128) for dim in self.pair.shape]
        return list(self.pair.shape)

    def __getitem__(self, index: object) -> torch.Tensor:
        self.pair.materialize()
        if self.scale:
            result = self.pair.scales[index]
            self.pair.codes = self.pair.scales = None
            return result
        return self.pair.codes[index]


def _load_converted_expert(
    backend: torch.nn.Module,
    base: str,
    ckpt_expert_idx: int,
    slot_expert_idx: int,
    local_slot_id: int,
    get_tensor: Callable[[str], torch.Tensor],
    get_optional_tensor: Callable[[str], torch.Tensor | None],
) -> None:
    del get_optional_tensor
    device = backend.w2_weight.device

    def dequantize_projection(projection: str) -> torch.Tensor:
        key = f"{base}.{ckpt_expert_idx}.{projection}"
        try:
            return dequantize_mxfp4_weight(
                get_tensor(key + ".weight_packed").to(device),
                get_tensor(key + ".weight_scale").to(device),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key}: {error}") from error

    weights = {projection: dequantize_projection(projection) for projection in ("w1", "w3")}
    # Single-alpha backends fuse gate/up into one GEMM, so both halves use
    # the same global scale when choosing their per-16 block scales.
    gate_up_amax = torch.maximum(weights["w1"].abs().amax(), weights["w3"].abs().amax()).float()
    tensors = {}
    for projection in ("w1", "w3", "w2"):
        key = f"{base}.{ckpt_expert_idx}.{projection}"
        try:
            packed, scales, global_scale = quantize_expert_weight(
                dequantize_projection(projection)
                if projection == "w2"
                else weights.pop(projection),
                amax=gate_up_amax if projection != "w2" else None,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key}: {error}") from error
        tensors.update(
            {
                f"{projection}_weight": packed.cpu(),
                f"{projection}_weight_scale": scales.cpu(),
                f"{projection}_weight_scale_2": global_scale.cpu(),
                f"{projection}_input_scale": torch.ones((), dtype=torch.float32),
            }
        )
        del packed, scales, global_scale
    backend.quant_method.load_streaming_nvfp4_expert(
        backend, global_expert_id=slot_expert_idx, local_slot_id=local_slot_id, **tensors
    )


class K31LoadQuantization:
    """Configure native quantized modules and adapt the source checkpoint lazily."""

    def __init__(self, attention_keys: set[str]) -> None:
        self.attention_keys = attention_keys

    @classmethod
    def configure(cls, config: ModelConfig) -> K31LoadQuantization | None:
        """Apply the opt-in runtime config before constructing a K3.1 VLM."""
        enabled = os.environ.get(_SWITCH, "0")
        if enabled not in ("0", "1"):
            raise ValueError(f"{_SWITCH} must be 0 or 1, got {enabled!r}")
        if enabled == "0":
            logger.info(f"{_SWITCH}=0: using native K3.1 checkpoint precision")
            return None
        cfg = config.pretrained_config
        if cfg.model_type != "kimi_k31":
            raise ValueError(f"{_SWITCH} requires a K3.1 composite checkpoint")
        text = cfg.text_config
        source_quant = getattr(text, "quantization_config", None) or getattr(
            cfg, "quantization_config", {}
        )
        from .modeling_kimi_linear import KimiK3MoERuntime

        expert_layers = [
            i
            for i in range(text.num_hidden_layers)
            if i >= text.first_k_dense_replace and i % text.moe_layer_freq == 0
        ]
        routed = [KimiK3MoERuntime._resolve_routed_quant_config(config, i) for i in expert_layers]
        if routed and all(q.quant_algo == QuantAlgo.NVFP4 for q in routed):
            logger.info(f"{_SWITCH}=1: native NVFP4 checkpoint; bypassing conversion")
            return None
        if source_quant.get("format") != "mxfp4-pack-quantized" or any(
            q.quant_algo != QuantAlgo.W4A8_MXFP4_MXFP8 for q in routed
        ):
            raise ValueError(
                f"{_SWITCH} requires MXFP4 routed experts with no conflicting overrides"
            )
        if config.moe_backend not in ("TRTLLM", "CUTLASS"):
            raise ValueError(f"{_SWITCH} supports the TRTLLM and CUTLASS MoE backends")
        if config.quant_config_dict:
            raise ValueError(f"{_SWITCH} does not accept source per-layer quantization overrides")

        # Source exclusions include all attention. The explicit per-layer map
        # now owns precision, leaving unspecified modules unquantized.
        quant = config.quant_config.model_copy(
            update={"quant_algo": QuantAlgo.MIXED_PRECISION, "exclude_modules": None}
        )
        # Preserve executor-normalized cache dtypes and recompute cached modes
        # from the new weight precision.
        quant.__dict__.pop("quant_mode", None)
        quant.__dict__.pop("layer_quant_mode", None)
        declarations = {}
        attention_keys = set()
        kda_layers = set(text.linear_attn_config["kda_layers"])
        for layer in range(text.num_hidden_layers):
            stem = f"model.layers.{layer}"
            if layer in expert_layers:
                declarations[f"{stem}.block_sparse_moe.experts"] = QuantConfig(
                    quant_algo=QuantAlgo.NVFP4, group_size=16
                )
            projections = _KDA_PROJECTIONS if layer + 1 in kda_layers else _MLA_PROJECTIONS
            for projection in projections:
                name = f"{stem}.self_attn.{projection}"
                declarations[name] = QuantConfig(
                    quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128
                )
                attention_keys.add(name + ".weight")
        frozen = config._frozen
        config._frozen = False
        config.quant_config = quant
        config.quant_config_dict = declarations
        config._frozen = frozen
        logger.info(
            f"{_SWITCH}=1: converting {len(expert_layers)} expert banks to NVFP4 and "
            f"{len(attention_keys)} attention projections to FP8; DSA indexers stay unquantized"
        )
        return cls(attention_keys)

    def prepare_weights(self, llm: KimiLinearForCausalLM, weights) -> ConsumableWeightsDict:
        """Preserve file-grouped expert streaming and add lazy FP8 projection values."""
        from .modeling_kimi_linear import _K3_NVFP4_EXPERT_CKPT_SPEC

        spec = _K3_NVFP4_EXPERT_CKPT_SPEC._replace(
            kinds=("weight_packed", "weight_scale"),
            loader=_load_converted_expert,
            optional_suffixes=(),
        )
        for layer in llm.model.layers:
            if getattr(layer, "is_moe", False):
                layer.block_sparse_moe.expert_ckpt_spec = spec
        adapted = dict(weights.items())
        prefix = "language_model." if any(k.startswith("language_model.") for k in adapted) else ""
        device = next(llm.parameters()).device
        for name in self.attention_keys:
            key = prefix + name
            if key not in adapted:
                raise KeyError(f"K3.1 load conversion is missing attention projection {key}")
            if key + "_scale" in adapted or key + "_scale_inv" in adapted:
                raise ValueError(f"K3.1 load conversion expects unquantized attention: {key}")
            pair = _AttentionPair(adapted[key], device)
            adapted[key] = _AttentionValue(pair)
            adapted[key + "_scale"] = _AttentionValue(pair, scale=True)
        result = ConsumableWeightsDict(adapted)
        checkpoint_dir = getattr(weights, "checkpoint_dir", None)
        if checkpoint_dir is not None:
            result.checkpoint_dir = checkpoint_dir
        result.checkpoint_prefix = getattr(weights, "checkpoint_prefix", "")
        return result
