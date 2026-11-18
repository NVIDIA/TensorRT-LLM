# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 routed experts of a draft (MTP) layer quantized to NVFP4 at load time for the TRTLLM-gen MoE backend.

Selected by ``MTPDecodingConfig.draft_moe_dtype="nvfp4"`` (lossy): an NVFP4 checkpoint that keeps the MTP layer in
BF16 then runs it on the target layers' FP4 MoE path. Weight scales come from the weights' amax (ModelOpt recipe,
no calibration data); activation global scales from DRAFT_MOE_NVFP4_FC1_AMAX / _FC2_AMAX. Only draft proposals
(acceptance length) can change.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

NVFP4_BLOCK = 16
# NVFP4 global scale convention (ModelOpt / TRT-LLM): amax maps to E2M1 max (6) x E4M3 max (448).
_E2M1_MAX = 6.0
_NVFP4_RANGE = _E2M1_MAX * 448.0
# Activation amax of the FC1 input / SwiGLU output the NVFP4 input scales assume: values above it clip, values
# far below only lose block-scale range.
DRAFT_MOE_NVFP4_FC1_AMAX = 256.0
DRAFT_MOE_NVFP4_FC2_AMAX = 2048.0
# E2M1 magnitudes in code order 0..7.
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def experts_unquantized(model_config, expert_quant_config) -> bool:
    """Whether a layer's routed experts load unquantized: no per-layer quant entry and a global quant config
    that does not quantize them (MIXED_PRECISION exports that keep the MTP layer in BF16)."""
    from tensorrt_llm.quantization.mode import QuantAlgo

    if expert_quant_config is not None:
        return not expert_quant_config.layer_quant_mode.has_any_quant(exclude_kv_cache=True)
    qc = getattr(model_config, "quant_config", None)
    if qc is None or qc.quant_algo in (None, QuantAlgo.MIXED_PRECISION):
        return True
    return not qc.layer_quant_mode.has_any_quant(exclude_kv_cache=True)


def quant_config_for(kv_cache_quant_algo=None):
    """The quantized draft experts' ``QuantConfig`` (NVFP4)."""
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    return QuantConfig(
        quant_algo=QuantAlgo.NVFP4,
        kv_cache_quant_algo=kv_cache_quant_algo,
        group_size=NVFP4_BLOCK,
    )


# ----------------------------------------------------------------------------------------------
# Quantizer ([N, K] BF16 -> quantized values + scales)
# ----------------------------------------------------------------------------------------------
def _e2m1_codes(mag: torch.Tensor) -> torch.Tensor:
    """Magnitudes (>= 0, <= 6) -> E2M1 magnitude codes 0..7 (uint8), round to nearest, ties to the even code (what
    cvt.rn.satfinite.e2m1x2 does): boundaries 0.25 / 0.75 / 1.25 / 1.75 / 2.5 / 3.5 / 5, ties at odd-code midpoints
    go down (0.25, 1.25, 2.5, 5) and at even-code midpoints up (0.75, 1.75, 3.5)."""
    code = (mag > 0.25).to(torch.uint8)
    for thr, inclusive in (
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        code += (mag >= thr if inclusive else mag > thr).to(torch.uint8)
    return code


def _nvfp4_codes(blocks: torch.Tensor, eff: torch.Tensor) -> torch.Tensor:
    """Signed E2M1 codes (uint8, sign in bit 3) of ``blocks`` [..., 16] for per-block multipliers ``eff`` [...]. A
    negative value that rounds to zero keeps its sign (-0, code 8), as cvt.rn.satfinite.e2m1x2 and the TRT-LLM
    quantization kernel produce it (same value as +0)."""
    scaled = torch.where(
        eff[..., None] > 0, blocks / eff[..., None].clamp_min(1e-30), torch.zeros_like(blocks)
    )
    code = _e2m1_codes(scaled.abs().clamp(max=_E2M1_MAX))
    return code | ((scaled < 0).to(torch.uint8) << 3)


def quantize_nvfp4(
    w: torch.Tensor, amax: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``w`` [N, K] -> (packed E2M1 uint8 [N, K/2], E4M3 block scales [N, K/16], FP32 ``weight_scale_2`` 0-d) in
    the ModelOpt NVFP4 checkpoint layout: ``weight_scale_2 = amax / (6 * 448)`` (``amax`` defaults to the tensor's),
    block scale ``e4m3(block_amax / 6 / weight_scale_2)``, ``w ~ e2m1 * block_scale * weight_scale_2``; element ``2j``
    in the low nibble of byte ``j``."""
    n, k = w.shape
    assert k % NVFP4_BLOCK == 0, k
    wf = w.float()
    a = wf.abs().amax() if amax is None else torch.tensor(float(amax), device=w.device)
    ws2 = (a / _NVFP4_RANGE).clamp_min(1e-30).to(torch.float32)
    blocks = wf.view(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    sf = (blocks.abs().amax(dim=-1) / 6.0 / ws2).clamp(max=448.0).to(torch.float8_e4m3fn)
    code = _nvfp4_codes(blocks, sf.float() * ws2).view(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, sf.contiguous(), ws2.reshape(())


def dequantize_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, weight_scale_2: torch.Tensor
) -> torch.Tensor:
    n, half = packed.shape
    k = 2 * half
    lo = (packed & 0x0F).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    code = torch.stack([lo, hi], dim=-1).view(n, k)
    grid = torch.tensor(_E2M1_VALUES, device=packed.device)
    val = grid[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    scale = scales.view(torch.float8_e4m3fn).float() * weight_scale_2.float()
    return (val.view(n, k // NVFP4_BLOCK, NVFP4_BLOCK) * scale[..., None]).view(n, k)


def convert_bf16_experts(
    weights: Dict, expert_ids: List[int], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Per-expert (VANILLA) NVFP4 weights of ``expert_ids`` from the checkpoint's BF16 expert tensors, either
    fused (``gate_up_proj`` [E, H, 2I] gate first, ``down_proj`` [E, I, H], as a fused-expert weight mapper hands
    them over) or per expert (``{e}.w1.weight`` / ``w3`` [I, H], ``{e}.w2.weight`` [H, I])."""
    out: Dict[str, torch.Tensor] = {}
    fused = "gate_up_proj" in weights
    in1 = torch.tensor(DRAFT_MOE_NVFP4_FC1_AMAX / _NVFP4_RANGE, dtype=torch.float32)
    in2 = torch.tensor(DRAFT_MOE_NVFP4_FC2_AMAX / _NVFP4_RANGE, dtype=torch.float32)
    for e in expert_ids:
        if fused:
            gu = weights["gate_up_proj"][e].to(device=device, dtype=torch.bfloat16).transpose(0, 1)
            w1, w3 = gu.chunk(2, dim=0)  # gate, up: [I, H]
            w2 = weights["down_proj"][e].to(device=device, dtype=torch.bfloat16).transpose(0, 1)
        else:
            w1, w3, w2 = (
                weights[f"{e}.{n}.weight"].to(device=device, dtype=torch.bfloat16)
                for n in ("w1", "w3", "w2")
            )
        # w1 / w3 share one per-tensor scale (the fused FC1 GEMM has one alpha per expert); the loader shards
        # the full-expert tensors and keeps the per-tensor scales.
        a13 = torch.maximum(w1.abs().amax(), w3.abs().amax()).float()
        for name, w, a, in_scale in (
            ("w1", w1, a13, in1),
            ("w3", w3, a13, in1),
            ("w2", w2, None, in2),
        ):
            q, sf, ws2 = quantize_nvfp4(w.contiguous(), amax=None if a is None else float(a))
            out[f"{e}.{name}.weight"] = q
            out[f"{e}.{name}.weight_scale"] = sf
            out[f"{e}.{name}.weight_scale_2"] = ws2.cpu()
            out[f"{e}.{name}.input_scale"] = in_scale.clone()
    return out


def _is_bf16_expert_dict(weights: Dict) -> bool:
    if "gate_up_proj" in weights:
        return weights["gate_up_proj"].dtype == torch.bfloat16
    w1 = next((v for k, v in weights.items() if k.endswith(".w1.weight")), None)
    return w1 is not None and w1.dtype == torch.bfloat16


def wrap_experts_loader(experts) -> None:
    """Make the experts' ``load_weights`` take the checkpoint's BF16 expert tensors and load this rank's experts
    as NVFP4 in the per-expert (VANILLA) layout. ``experts`` must have been built with ``quant_config_for()``.

    Both the module and a ``ConfigurableMoE``'s ``backend`` are wrapped (the model loader may hand the weights to
    either); already converted weights pass through, so a delegating wrapper converts once."""
    backend = getattr(experts, "backend", experts)
    for target in (experts,) if backend is experts else (experts, backend):
        _wrap_load_weights(target, experts, backend)


def _wrap_load_weights(target, experts, backend) -> None:
    from ..moe.fused_moe.interface import MoEWeightLoadingMode

    load = target.load_weights

    def load_weights(weights: List[Dict], allow_partial_loading: bool = False):
        assert len(weights) == 1, len(weights)
        w = weights[0]
        if not _is_bf16_expert_dict(w):  # converted already, or a reload of quantized weights
            return load(weights, allow_partial_loading=allow_partial_loading)
        if getattr(backend, "layer_load_balancer", None):
            raise NotImplementedError("NVFP4 draft experts do not support the MoE load balancer")
        device = torch.device("cuda", torch.cuda.current_device())
        converted = convert_bf16_experts(w, list(backend.initial_local_expert_ids), device)
        # The weight mapper sets the checkpoint's (fused) layout on the module; the converted
        # tensors use the per-expert one.
        experts.weight_loading_mode = MoEWeightLoadingMode.VANILLA
        backend.weight_loading_mode = MoEWeightLoadingMode.VANILLA
        return load([converted], allow_partial_loading=allow_partial_loading)

    target.load_weights = load_weights


__all__ = [
    "DRAFT_MOE_NVFP4_FC1_AMAX",
    "DRAFT_MOE_NVFP4_FC2_AMAX",
    "convert_bf16_experts",
    "dequantize_nvfp4",
    "experts_unquantized",
    "quant_config_for",
    "quantize_nvfp4",
    "wrap_experts_loader",
]
