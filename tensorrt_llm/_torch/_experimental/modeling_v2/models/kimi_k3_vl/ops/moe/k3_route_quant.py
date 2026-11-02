# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's top-16 routing and MXFP8 latent quantization at decode size, ``trtllm::k3_route_quant``: the CuTe DSL
form of ``trtllm::kimi_k3_noaux_tc_mxfp8_quant``, the same four outputs bit for bit, for up to 64 tokens."""

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Importing the op module registers trtllm::k3_route_quant; CuTe DSL is imported on its first call.
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_route_quant import (
    op as _k3_route_quant_op,  # noqa: F401
)

__all__ = ["k3_route_quant"]


class K3RouteQuant(OpWrapper):
    """``(topk_ids, topk_weights, quantized, scales)`` for ``router_logits`` (fp32 ``[M, 896]``) and ``latent`` (bf16
    ``[M, 3584]``), 1 <= M <= 64: the 16 experts with the largest sigmoid + ``e_score_correction_bias`` (int32
    ``[M, 16]``), their unbiased sigmoids renormalized times ``routed_scaling_factor`` (bf16 ``[M, 16]``), and the
    latent as MXFP8 (float8_e4m3fn ``[M, 3584]``, one UE8M0 scale per 32 columns, uint8 ``[M, 112]``).

    ``early_trigger``: let the next kernel launch (as a programmatic dependent) as soon as every CTA has passed its own
    grid-dependency wait, before the outputs are written; for a dependent that waits for this whole grid before
    reading them, as ``trtllm::k3_moe`` does."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"M={m}, early_trigger={early}: decode batch and M=64 row invariance",
                spec=dict(m=m, case="random", early_trigger=early, routed_scaling_factor=2.827),
            )
            for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 64)
            for early in (False, True)
        ),
        *(
            Cell(
                why=f"M={m}, early_trigger={early}: {reason}",
                spec=dict(m=m, case=case, early_trigger=early, routed_scaling_factor=2.827),
            )
            for case, reason in (
                ("40_tied_keys", "40 equal selection keys choose the lowest 16 expert ids"),
                ("all_equal_logits", "all experts tied with zero correction bias"),
                ("huge_logits", "saturated tanh-form sigmoids"),
                ("zero_large_denormal_rows", "zero blocks, large and denormal latent values"),
            )
            for m in (1, 3, 8)
            for early in (False, True)
        ),
    )

    note = """
    Inputs are contiguous CUDA tensors on one device: fp32 [M,896] logits,
    fp32 [896] selection bias and bf16 [M,3584] latent, 1<=M<=64.
    Outputs are new contiguous tensors on the logits' device; inputs
    are neither mutated nor retained. The schema calls these inputs
    scores, bias and hidden_states, with mutates_args=().

    Selection uses 0.5*tanh(0.5*logit)+0.5 plus bias, descending with
    lower expert ids winning ties. Weights use unbiased sigmoids.
    The kernel sums the selected scores in fp32 with a 16-lane
    xor-butterfly reduction, then normalizes with a 1e-20 denominator
    offset and applies routed_scaling_factor in fp64, rounding once
    to bf16. The reference uses fp32 accumulation and one output cast.
    MXFP8 has linear UE8M0 scales rounded up from each 32-column
    block's amax/448; byte t*112+b scales latent[t,32*b:32*b+32].
    The kernel's approximate reciprocal flushes denormal scales to zero;
    all-denormal nonzero blocks and non-finite inputs are not modeled by
    the reference. The cells include denormal elements in normal blocks.

    The four outputs are also checked bit for bit against
    kimi_k3_noaux_tc_mxfp8_quant, with and without early trigger,
    including tied/saturated routing and zero/large/denormal latent
    rows. Reruns and each M's rows agree bitwise with the M=64 call.
    The kernel test also compares noaux_tc_op plus mxfp8_quantize.
    Router and latent-down GEMMs, expert execution and combine remain
    outside the fusion. k3_moe consumes the outputs; k3_moe_front fuses
    the GEMMs, head all-gather and this op's device code.

    No metadata or persistent workspace. The result-neutral process
    compile cache is keyed by (early_trigger,PDL); each build must run
    eagerly before capture or raises RuntimeError. TRTLLM_ENABLE_PDL,
    default on, is read every call and changes scheduling only.
    Before its dependency wait the kernel reads the correction bias:
    preceding work must not be writing that weight. With early_trigger,
    dependents may start before outputs are written and must wait for
    this whole grid before reading them, as k3_moe with use_pdl does.
    Without early trigger, release follows output stores and a GPU
    fence. Captured M=8/3/64 calls replay rewritten inputs four times
    with interleaved eager calls and retain the isolated calls' bits.
    Invalid shapes, dtypes, layouts and token counts raise ValueError;
    the tests retain those refusals and both cold-capture refusals.
    The stock comparator requires SM10.x.
    """

    def __call__(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        latent: torch.Tensor,
        routed_scaling_factor: float,
        early_trigger: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.k3_route_quant(
            router_logits,
            e_score_correction_bias,
            latent,
            routed_scaling_factor,
            early_trigger=early_trigger,
        )

    def reference(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        latent: torch.Tensor,
        routed_scaling_factor: float,
        early_trigger: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = 0.5 * torch.tanh(0.5 * router_logits.float()) + 0.5
        ids = torch.sort(
            scores + e_score_correction_bias.float(), dim=-1, descending=True, stable=True
        ).indices[:, :16]
        weights = scores.gather(1, ids)
        total = weights.sum(-1, keepdim=True).double()
        weights = weights.double() / (total + 1e-20) * routed_scaling_factor
        blocks = latent.float().reshape(latent.shape[0], 112, 32)
        amax = blocks.abs().amax(dim=-1)
        bits = (amax / 448.0).view(torch.int32)
        byte = (((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
        scale = torch.exp2(byte.float() - 127.0).unsqueeze(-1)
        data = (blocks / scale).clamp(-448.0, 448.0).reshape(latent.shape).to(torch.float8_e4m3fn)
        return ids.to(torch.int32), weights.to(torch.bfloat16), data, byte.to(torch.uint8)

    def is_valid(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
        latent: torch.Tensor,
        routed_scaling_factor: float,
        early_trigger: bool = False,
    ) -> None:
        assert e_score_correction_bias.device == latent.device == router_logits.device, (
            "logits, correction bias and latent must be on one device"
        )


k3_route_quant = K3RouteQuant()
