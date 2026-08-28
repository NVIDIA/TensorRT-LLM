# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only native SiTU dispatch for the Kimi K3 sparse MoE reference.

The K3 MoE block has two mutually exclusive kernel paths:

1. **Python fallback** — MXFP4 group-32 routed expert weights are
   dequantized on the fly, then fed through per-expert
   ``gate_up_proj + activation + down_proj`` linears. Byte-exact HF
   parity under random weights.

2. **Native fused SiTU path** — routed compute goes through the in-tree
   ``torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner`` custom op
   with ``act_type=ActType_TrtllmGen.SiTu``. Activations are dynamically
   quantized to MXFP8 (``mxfp8_quantize``); weights are the checkpoint's
   MXFP4 group-32 tensors padded and shuffled with the same
   pad/shuffle/interleave contract as
   ``MXFP4WeightTRTLLMGenFusedMoEMethod`` (see
   ``fused_moe/quantization.py``). No FlashInfer private-cubin
   environment variable is involved.

Routing always goes through the ``topk_weights``/``topk_ids`` bypass:
K3's gate semantics (sigmoid scoring, ``e_score_correction_bias``
affecting selection only, renormalize, no groups) match none of the
built-in trtllm-gen routing methods, so the K3 gate computes top-k on
the host module and the op consumes the precomputed result verbatim.

Weight/activation packing conventions (must stay in sync with the
generated ``GemmGatedActOptions.h`` SiTuGlu definition):

* FC1 packs ``w3`` (up/linear) in the first half and ``w1`` (gate) in
  the second half — the cubin evaluates ``x0`` (linear) from the first
  half and ``x1`` (gate) from the second. This is the **opposite** of
  the HF/Python reference layout where gate comes first.
* ``gemm1_alpha``  (cubin ``alpha``, gate side ``x1``)  <- ``activation_situ_beta``
* ``gemm1_beta``   (cubin ``beta``, linear side ``x0``) <- ``activation_situ_linear_beta``
* ``gemm1_clamp_limit=None`` means +inf (no clamping) per the generated
  ``KernelParamsDecl.h`` contract.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# TRTLLM-Gen backend alignment contract — keep in sync with
# MXFP4WeightTRTLLMGenFusedMoEMethod in fused_moe/quantization.py and the
# roundUp calls in blockScaleMoe/runner.cu.
INPUT_HIDDEN_ALIGNMENT = 512
WEIGHT_ALIGNMENT = 128
SCALING_VECTOR_SIZE = 32
EPILOGUE_TILE_M = 128

# Local memo for the (expensive) shuffle permute-index computation, shared
# across experts/instances with identical shapes.
_CACHE_PERMUTE_INDICES: Dict[tuple, torch.Tensor] = {}


def _round_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


def get_moe_sm_version() -> int:
    """Return the runtime SM version used for kernel-support checks."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return -1
    try:
        from tensorrt_llm._utils import get_sm_version as _tllm_get_sm_version
    except ImportError:  # pragma: no cover — source-loader stub path
        prop = torch.cuda.get_device_properties(0)
        return prop.major * 10 + prop.minor
    return int(_tllm_get_sm_version())


def is_native_situ_supported() -> bool:
    """Native SiTU cubins are Blackwell sm_100f: the whole SM100 family."""
    return 100 <= get_moe_sm_version() < 110


def assert_native_situ_supported(
    *,
    hidden_size: int,
    intermediate_size: int,
    group_size: int = SCALING_VECTOR_SIZE,
) -> None:
    """Fail loudly (before any launch) when the fused SiTU path cannot run."""
    if not torch.cuda.is_available():
        raise RuntimeError("native SiTU MoE requires CUDA; no CUDA device is available")
    sm = get_moe_sm_version()
    if not 100 <= sm < 110:
        raise RuntimeError(
            f"native SiTU MoE requires the SM100 (Blackwell) family; running on SM{sm}"
        )
    if group_size != SCALING_VECTOR_SIZE:
        raise RuntimeError(
            f"native SiTU MoE requires MXFP4 group_size {SCALING_VECTOR_SIZE}, got {group_size}"
        )
    if hidden_size % group_size != 0 or intermediate_size % group_size != 0:
        raise RuntimeError(
            f"hidden_size {hidden_size} and intermediate_size {intermediate_size} must be "
            f"multiples of the MXFP4 group size {group_size}"
        )


def padded_fused_shapes(hidden_size: int, intermediate_size: int) -> Tuple[int, int, int]:
    """Return (hidden_padded_fc1, hidden_padded_fc2, intermediate_padded).

    FC1 consumes activations along hidden (K dim, 512-aligned); FC2 produces
    hidden (N dim, 128-aligned); intermediate is 128-aligned on both sides.
    """
    return (
        _round_up(hidden_size, INPUT_HIDDEN_ALIGNMENT),
        _round_up(hidden_size, WEIGHT_ALIGNMENT),
        _round_up(intermediate_size, WEIGHT_ALIGNMENT),
    )


def pack_routed_expert_weights(
    *,
    w1_packed: torch.Tensor,
    w1_scales: torch.Tensor,
    w3_packed: torch.Tensor,
    w3_scales: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Pad + shuffle checkpoint MXFP4 expert weights into the TRTLLM-Gen layout.

    Inputs are the per-expert MXFP4 tensors as stored by
    the test reference's routed expert bank (HF layout, group_size=32):

    * ``w1_packed``/``w3_packed``: ``uint8 [E, I, H // 2]`` (w1 = gate, w3 = up)
    * ``w1_scales``/``w3_scales``: ``uint8 [E, I, H // 32]`` (E8M0 biased exponents)
    * ``w2_packed``: ``uint8 [E, H, I // 2]``, ``w2_scales``: ``uint8 [E, H, I // 32]``

    Returns CUDA buffers matching ``MXFP4WeightTRTLLMGenFusedMoEMethod``'s
    device layout:

    * ``gemm1_weights``: ``uint8 [E, 2 * I_pad, H_pad512 // 2]`` — w3 first,
      w1 second, then row-shuffled for the gated-act GEMM.
    * ``gemm1_weights_scale``: ``uint8 [E, 2 * I_pad, H_pad512 // 32]`` —
      shuffled + block-scale interleaved.
    * ``gemm2_weights``: ``uint8 [E, H_pad128, I_pad // 2]`` — row-shuffled.
    * ``gemm2_weights_scale``: ``uint8 [E, H_pad128, I_pad // 32]`` —
      shuffled + block-scale interleaved.
    """
    from tensorrt_llm._torch.modules.fused_moe.quantization import (
        maybe_pad_for_mxfp4,
        trtllmgen_maybe_get_cached_w2_permute_indices,
        trtllmgen_maybe_get_cached_w3_w1_permute_indices,
    )
    from tensorrt_llm.quantization.utils.fp4_utils import float4_sf_dtype

    for name, t in (
        ("w1_packed", w1_packed),
        ("w1_scales", w1_scales),
        ("w3_packed", w3_packed),
        ("w3_scales", w3_scales),
        ("w2_packed", w2_packed),
        ("w2_scales", w2_scales),
    ):
        if t.dtype != torch.uint8:
            raise RuntimeError(f"{name} must be uint8 MXFP4 data, got {t.dtype}")

    num_experts, intermediate_size, hidden_half = w1_packed.shape
    hidden_size = hidden_half * 2
    h_pad_fc1, h_pad_fc2, i_pad = padded_fused_shapes(hidden_size, intermediate_size)

    gemm1_weights = torch.zeros(
        (num_experts, 2 * i_pad, h_pad_fc1 // 2), dtype=torch.uint8, device=device
    )
    gemm1_weights_scale = torch.zeros(
        (num_experts, 2 * i_pad, h_pad_fc1 // SCALING_VECTOR_SIZE),
        dtype=torch.uint8,
        device=device,
    )
    gemm2_weights = torch.zeros(
        (num_experts, h_pad_fc2, i_pad // 2), dtype=torch.uint8, device=device
    )
    gemm2_weights_scale = torch.zeros(
        (num_experts, h_pad_fc2, i_pad // SCALING_VECTOR_SIZE),
        dtype=torch.uint8,
        device=device,
    )

    for e in range(num_experts):
        # ---- FC1 weights: pad, place w3 (linear) first / w1 (gate) second, shuffle.
        dst = gemm1_weights[e]
        dst_w3, dst_w1 = dst.chunk(2, dim=0)
        dst_w3.copy_(maybe_pad_for_mxfp4(w3_packed[e].to(device), h_pad_fc1 // 2, i_pad))
        dst_w1.copy_(maybe_pad_for_mxfp4(w1_packed[e].to(device), h_pad_fc1 // 2, i_pad))
        permute = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst, _CACHE_PERMUTE_INDICES, EPILOGUE_TILE_M
        )
        dst.copy_(torch.ops.trtllm.shuffle_matrix(dst, permute.to(device)))

        # ---- FC1 scales: pad, place, shuffle with sf indices, interleave.
        dst_sf = gemm1_weights_scale[e]
        dst_sf_w3, dst_sf_w1 = dst_sf.chunk(2, dim=0)
        dst_sf_w3.copy_(
            maybe_pad_for_mxfp4(w3_scales[e].to(device), h_pad_fc1 // SCALING_VECTOR_SIZE, i_pad)
        )
        dst_sf_w1.copy_(
            maybe_pad_for_mxfp4(w1_scales[e].to(device), h_pad_fc1 // SCALING_VECTOR_SIZE, i_pad)
        )
        permute_sf = trtllmgen_maybe_get_cached_w3_w1_permute_indices(
            dst_sf.view(float4_sf_dtype),
            _CACHE_PERMUTE_INDICES,
            EPILOGUE_TILE_M,
            num_elts_per_sf=SCALING_VECTOR_SIZE,
        )
        shuffled_sf = torch.ops.trtllm.shuffle_matrix(
            dst_sf.view(float4_sf_dtype), permute_sf.to(device)
        )
        dst_sf.copy_(
            torch.ops.trtllm.block_scale_interleave(
                shuffled_sf.view(float4_sf_dtype).reshape(dst_sf.shape)
            )
            .view(torch.uint8)
            .reshape(dst_sf.shape)
        )

        # ---- FC2 weights: pad + shuffle.
        dst2 = gemm2_weights[e]
        dst2.copy_(maybe_pad_for_mxfp4(w2_packed[e].to(device), i_pad // 2, h_pad_fc2))
        permute2 = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst2, _CACHE_PERMUTE_INDICES, EPILOGUE_TILE_M
        )
        dst2.copy_(torch.ops.trtllm.shuffle_matrix(dst2, permute2.to(device)))

        # ---- FC2 scales: pad + shuffle with sf indices + interleave.
        dst2_sf = gemm2_weights_scale[e]
        dst2_sf.copy_(
            maybe_pad_for_mxfp4(w2_scales[e].to(device), i_pad // SCALING_VECTOR_SIZE, h_pad_fc2)
        )
        permute2_sf = trtllmgen_maybe_get_cached_w2_permute_indices(
            dst2_sf.view(float4_sf_dtype),
            _CACHE_PERMUTE_INDICES,
            EPILOGUE_TILE_M,
            num_elts_per_sf=SCALING_VECTOR_SIZE,
        )
        shuffled2_sf = torch.ops.trtllm.shuffle_matrix(
            dst2_sf.view(float4_sf_dtype), permute2_sf.to(device)
        )
        dst2_sf.copy_(
            torch.ops.trtllm.block_scale_interleave(shuffled2_sf.view(float4_sf_dtype))
            .view(torch.uint8)
            .reshape(dst2_sf.shape)
        )

    return {
        "gemm1_weights": gemm1_weights,
        "gemm1_weights_scale": gemm1_weights_scale,
        "gemm2_weights": gemm2_weights,
        "gemm2_weights_scale": gemm2_weights_scale,
    }


def make_situ_alpha_beta(
    *,
    local_num_experts: int,
    situ_beta: float,
    situ_linear_beta: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the per-expert CUDA float32 alpha/beta buffers for the op boundary.

    The kernel reads one alpha/beta pair per local expert; scalar broadcast is
    not part of the contract. ``alpha`` is the gate-side (x1) parameter and maps
    to Kimi's ``activation_situ_beta``; ``beta`` is the linear-side (x0)
    parameter and maps to ``activation_situ_linear_beta``.
    """
    if situ_beta <= 0.0 or situ_linear_beta <= 0.0:
        raise RuntimeError(
            f"SiTu alpha/beta must be > 0 (got alpha={situ_beta}, beta={situ_linear_beta})"
        )
    gemm1_alpha = torch.full(
        (local_num_experts,), float(situ_beta), dtype=torch.float32, device=device
    ).contiguous()
    gemm1_beta = torch.full(
        (local_num_experts,), float(situ_linear_beta), dtype=torch.float32, device=device
    ).contiguous()
    return gemm1_alpha, gemm1_beta


def invoke_native_situ_moe(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm1_alpha: torch.Tensor,
    gemm1_beta: torch.Tensor,
    num_experts: int,
    top_k: int,
    valid_hidden_size: int,
    valid_intermediate_size: int,
    local_expert_offset: int = 0,
    local_num_experts: Optional[int] = None,
    act_type: Optional[int] = None,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    """Run routed-expert compute through the in-tree SiTU fused MoE op.

    Parameters
    ----------
    hidden_states
        bf16 ``[num_tokens, valid_hidden_size]`` routed input (unpadded).
    topk_ids / topk_weights
        Precomputed K3 routing (any integer/float dtype; converted to the
        op's int32/bfloat16 contract here). ``topk_weights`` must already
        include renormalization and ``routed_scaling_factor`` — the bypass
        feeds them into finalize verbatim. ``topk_ids`` are GLOBAL expert
        ids; under EP the kernel skips ids outside
        ``[local_expert_offset, local_expert_offset + local_num_experts)``
        and those tokens contribute zeros (caller allreduces partials).
    gemm*_weights / gemm*_weights_scale
        Output of :func:`pack_routed_expert_weights` for this rank's
        local expert slice (leading dim = ``local_num_experts``).
    gemm1_alpha / gemm1_beta
        Output of :func:`make_situ_alpha_beta` (``[local_num_experts]``).

    Returns bf16 ``[num_tokens, valid_hidden_size]``.
    """
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    if act_type is None:
        act_type = int(ActType_TrtllmGen.SiTu)
    if local_num_experts is None:
        local_num_experts = num_experts
    if gemm1_weights.shape[0] != local_num_experts:
        raise RuntimeError(
            f"gemm1_weights holds {gemm1_weights.shape[0]} experts but "
            f"local_num_experts={local_num_experts}"
        )

    if hidden_states.dtype != torch.bfloat16:
        raise RuntimeError(f"native SiTU MoE expects bf16 hidden_states, got {hidden_states.dtype}")
    if not hidden_states.is_cuda:
        raise RuntimeError("native SiTU MoE requires CUDA hidden_states")
    num_tokens, hidden_size = hidden_states.shape
    if hidden_size != valid_hidden_size:
        raise RuntimeError(
            f"hidden_states last dim {hidden_size} != valid_hidden_size {valid_hidden_size}"
        )

    intermediate_size_padded = gemm1_weights.shape[-2] // 2
    hidden_padded_fc1 = gemm1_weights.shape[-1] * 2
    expected_h_pad, _, expected_i_pad = padded_fused_shapes(
        valid_hidden_size, valid_intermediate_size
    )
    if hidden_padded_fc1 != expected_h_pad or intermediate_size_padded != expected_i_pad:
        raise RuntimeError(
            f"fused weight shapes do not match the padding contract: "
            f"padded hidden {hidden_padded_fc1} (expected {expected_h_pad}, "
            f"valid {valid_hidden_size}), padded intermediate "
            f"{intermediate_size_padded} (expected {expected_i_pad}, "
            f"valid {valid_intermediate_size})"
        )

    # topk contract: int32 / bfloat16, contiguous, [num_tokens, top_k].
    topk_ids = topk_ids.to(device=hidden_states.device, dtype=torch.int32).contiguous()
    topk_weights = topk_weights.to(device=hidden_states.device, dtype=torch.bfloat16).contiguous()
    if topk_ids.shape != (num_tokens, top_k) or topk_weights.shape != (num_tokens, top_k):
        raise RuntimeError(
            f"topk tensors must be [num_tokens={num_tokens}, top_k={top_k}]; got "
            f"topk_ids {tuple(topk_ids.shape)}, topk_weights {tuple(topk_weights.shape)}"
        )

    # Dynamic MXFP8 activation quantization; pads hidden to the FC1 alignment.
    x_fp8, x_sf = torch.ops.trtllm.mxfp8_quantize(
        hidden_states.contiguous(), False, alignment=INPUT_HIDDEN_ALIGNMENT
    )

    output = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
        None,  # routing_logits — unused with the topk bypass
        None,  # routing_bias
        x_fp8,
        x_sf.flatten(),
        gemm1_weights,
        gemm1_weights_scale,
        None,  # gemm1_bias
        gemm1_alpha,
        gemm1_beta,
        None,  # gemm1_clamp_limit — nullptr means +inf for clmp kernels
        gemm2_weights,
        gemm2_weights_scale,
        None,  # gemm2_bias
        num_experts,
        top_k,
        None,  # n_group — K3 has no expert groups
        None,  # topk_group
        intermediate_size_padded,
        valid_hidden_size,
        valid_intermediate_size,
        local_expert_offset,
        local_num_experts,
        None,  # routed_scaling_factor — already folded into topk_weights
        1,  # routing_method_type (Renormalize) — inert under the topk bypass
        act_type,
        topk_weights,
        topk_ids,
        tune_max_num_tokens=tune_max_num_tokens,
    )

    if output.shape[-1] > valid_hidden_size:
        output = output[:, :valid_hidden_size].contiguous()
    return output
