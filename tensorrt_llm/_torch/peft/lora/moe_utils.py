# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert (MoE) LoRA validation and synthetic-adapter tooling."""

from collections.abc import Iterable
from typing import Literal

import torch

from tensorrt_llm.lora_helper import MOE_MODULE_SHARED_FLAG, LoraConfig

SharedSide = Literal["A", "B", None]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _normalize_targets(lora_target_modules: Iterable[str]) -> set[str]:
    return {name.lower() for name in lora_target_modules or []}


def has_moe_lora_targets(lora_config: LoraConfig | None) -> bool:
    """Return True iff `lora_config` requests LoRA on any routed-expert MoE module."""
    if lora_config is None:
        return False
    targets = _normalize_targets(lora_config.lora_target_modules)
    return bool(MOE_MODULE_SHARED_FLAG.keys() & targets)


def check_moe_lora_supported(
    *,
    moe_backend_name: str,
    lora_config: LoraConfig | None,
    quant_config,
    layer_idx: int | None = None,
) -> None:
    """Raise `ValueError` if a routed-expert MoE LoRA cannot run on the chosen
    backend / quant combination.

    Args:
        moe_backend_name: The resolved `moe_backend` string (e.g. "CUTLASS",
            "WIDEEP", "TRTLLM"). Comparison is case-insensitive.
        lora_config: The model's `LoraConfig`, or None.
        quant_config: The model's `QuantConfig`, or None. We only reject when
            the layer is actually quantized. The per-layer `layer_quant_mode`
            is preferred over the global `quant_mode` so this matches the view
            used for backend selection under mixed/per-layer quantization.
        layer_idx: Optional layer index for diagnostic messages.

    Constraints (MVP):
        - MoE backend MUST be CUTLASS.
        - Base weight quantization MUST be off (no FP8 / FP4 / INT8 / INT4 / W4A8 ...).

    Other constraints (alltoall, min-latency, FP4, CUDA-graph) are enforced at
    runtime; we do not pre-check them here because they depend on per-call
    state that isn't available at factory time.
    """
    if not has_moe_lora_targets(lora_config):
        return

    prefix = f"[layer_idx={layer_idx}] " if layer_idx is not None else ""

    if (moe_backend_name or "").upper() != "CUTLASS":
        raise ValueError(
            f"{prefix}Routed-expert MoE LoRA requires moe_backend='CUTLASS'; got "
            f"moe_backend={moe_backend_name!r}. Disable LoRA on MoE modules "
            f"(remove {sorted(MOE_MODULE_SHARED_FLAG)} from "
            "lora_config.lora_target_modules) or switch to the Cutlass MoE backend."
        )

    if quant_config is not None:
        # Prefer the per-layer quant mode so this check agrees with the
        # backend-selection logic in create_moe.py, which also uses
        # `layer_quant_mode`. Fall back to the global `quant_mode`.
        quant_mode = getattr(quant_config, "layer_quant_mode", None)
        if quant_mode is None:
            quant_mode = getattr(quant_config, "quant_mode", None)
        is_quantized = False
        if quant_mode is not None and hasattr(quant_mode, "has_any_quant"):
            try:
                is_quantized = bool(quant_mode.has_any_quant(exclude_kv_cache=True))
            except TypeError:
                # Older signatures may not accept the kwarg; fall back.
                is_quantized = bool(quant_mode.has_any_quant())
        if is_quantized:
            raise ValueError(
                f"{prefix}Routed-expert MoE LoRA MVP only supports unquantized "
                f"fp16/bf16 base weights; got quant_mode={quant_mode}. "
                "FP8/FP4/INT4/INT8 base weights combined with MoE LoRA are not "
                "supported yet."
            )


# ---------------------------------------------------------------------------
# Synthetic adapter tooling (unit tests)
# ---------------------------------------------------------------------------


def make_per_expert_lora(
    num_experts: int,
    rank: int,
    in_dim: int,
    out_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    shared_side: SharedSide = None,
    seed: int | None = None,
) -> dict[str, torch.Tensor]:
    """Generate a (A, B) LoRA tensor pair for an MoE module.

    Always returns shapes `A: [E, rank, in_dim]` and `B: [E, out_dim, rank]`,
    suitable for `torch.stack(...)` in `tensorrt_llm/lora_manager.py`.

    When `shared_side == "A"`, the same `[rank, in_dim]` matrix is replicated
    across all `E` experts (load-time replication for shared-outer). Likewise
    for `shared_side == "B"`. When `shared_side is None`, each expert gets
    independent A and B matrices.

    Args:
        num_experts: number of experts in this MoE layer.
        rank: LoRA rank for this module.
        in_dim: input hidden size (e.g. `hidden_size` for moe_h_to_4h).
        out_dim: output hidden size (e.g. `intermediate_size` for moe_h_to_4h).
        dtype: tensor dtype.
        device: tensor device.
        shared_side: which side to replicate ("A", "B", or None).
        seed: optional torch RNG seed for deterministic generation.

    Returns:
        Dict with keys "A" (shape [E, rank, in_dim]) and "B" (shape [E, out_dim, rank]).
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    def _randn(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=gen)

    if shared_side == "A":
        shared_a = _randn(rank, in_dim)
        a = shared_a.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
        b = _randn(num_experts, out_dim, rank)
    elif shared_side == "B":
        a = _randn(num_experts, rank, in_dim)
        shared_b = _randn(out_dim, rank)
        b = shared_b.unsqueeze(0).expand(num_experts, -1, -1).contiguous()
    elif shared_side is None:
        a = _randn(num_experts, rank, in_dim)
        b = _randn(num_experts, out_dim, rank)
    else:
        raise ValueError(f"shared_side must be 'A', 'B', or None; got {shared_side!r}")

    return {"A": a, "B": b}


def reference_moe_lora_delta(
    a_stacked: torch.Tensor,
    b_stacked: torch.Tensor,
    x: torch.Tensor,
    token_to_expert: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute the reference LoRA delta `delta[t] = scale * (B[e_t] @ A[e_t] @ x[t])`
    for a routed MoE layer, in eager PyTorch.

    This is used by unit tests to check the fused MoE LoRA op against a
    straightforward per-expert reference. Works for both per-expert and
    shared-outer adapters as long as `a_stacked`/`b_stacked` are already
    expanded to `[E, ...]`.

    Args:
        a_stacked: `[E, rank, in_dim]`
        b_stacked: `[E, out_dim, rank]`
        x:         `[num_tokens, in_dim]`
        token_to_expert: `[num_tokens]` int tensor of expert indices.
        scale: LoRA scale factor (`alpha / rank` for vanilla LoRA, `alpha / sqrt(rank)`
            for rs-LoRA).

    Returns:
        Tensor of shape `[num_tokens, out_dim]`.
    """
    num_tokens = x.shape[0]
    out_dim = b_stacked.shape[1]
    output = torch.zeros(num_tokens, out_dim, dtype=x.dtype, device=x.device)
    for t in range(num_tokens):
        e = int(token_to_expert[t].item())
        # (rank, in) @ (in,) -> (rank,)
        mid = a_stacked[e] @ x[t]
        # (out, rank) @ (rank,) -> (out,)
        output[t] = scale * (b_stacked[e] @ mid)
    return output
