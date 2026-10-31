# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp hyper-connection gating: low-rank stream mixer plus block-injection
weights.

A macro entry composed from native torch (three ``F.linear`` GEMMs and four
elementwise passes); it launches no TensorRT-LLM kernel. The ``mhc_*`` op family
(``mhc_gemm_sqrsum_fma``, ``mhc_big_fuse``, ``mhc_post_mapping``) computes a
structurally different hyper-connection -- a per-stream *scalar* gate from one
linear rather than a rank-320 SiLU bottleneck, a weighted *sum* rather than a
mean, the *raw* stream gated rather than the normalized one -- and its fused
members do not instantiate hidden 2560. The built-in Qwen4-Exp implementation
has fused ops for this (``trtllm::qwen4_exp_hc_*``), not wrapped here; the
semantics follow the HuggingFace ``Qwen4ExpTextGatedResidual`` minus its norm.

Shared by the qwen4_exp tp1 target
(`qwen3_8_flash_next_nvfp4__sm_103__tp1`) and the unified tp4/tep4/dep4 target
(`hf_fc694b5_orig__sm_103__unified`); promoted from their copies.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .._op import Arch, Cell, OpWrapper

#: This checkpoint's geometry: 4 streams of 2560, rank-320 mixer bottleneck.
CKPT_HC_COUNT = 4
CKPT_HIDDEN = 2560
CKPT_LOWRANK = 320


def gate_ratio(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst element's distance from `ref` as a fraction of the test's band:
    `rtol = eps`, `atol = eps * max|ref|`, both of `ref`'s dtype. Below 1 is
    inside."""
    eps = torch.finfo(ref.dtype).eps
    ref64 = ref.double()
    atol = eps * ref64.abs().max().item()
    return ((out.double() - ref64).abs() / (atol + eps * ref64.abs())).max().item()


class Qwen4ExpGatedResidual(OpWrapper):
    """Collapse the `hc_count`-stream residual to one stream, and weight the
    block's injection. With `C = hc_count`, `H = width // C` and `normed` the
    `[..., C*H]` stream **after** its grouped norm, viewed stream-major as
    `[..., C, H]`::

        low    = silu(normed @ down^T / C)                 # [..., lowrank]
        gate   = sigmoid(low @ up^T)                       # [..., C*H]
        mixed  = mean over streams of (gate * normed)      # [..., H]
        inject = 2 * sigmoid(normed @ block_inject^T / C)  # [..., C]

    Returns fresh contiguous `(mixed, inject)` in the activation dtype;
    `inject` is `None` exactly when `block_inject_weight` is `None` (the
    stack-final mixer), and `mixed` is then bitwise what it is with one.

    Both divisions by `C` sit on the pre-activation logit; `mixed` gates the
    **normed** stream; `inject` carries a factor of 2 (a zero logit is an
    identity injection). GEMMs run in the activation dtype (fp32 accumulate),
    the pointwise chain in fp32, each result rounded once.

    Fusion boundary: both projections, both activations, the gating, the
    stream mean and the injection projection are inside. The grouped gemma
    norm producing `normed`, and the accumulate `qwen4_exp_post_inject` (into
    the **pre-norm** stream the caller still holds), are not. Weights are used
    as stored (`nn.Linear` layout), nothing is cached.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: leading (token dims), inject (True). Every cell is this
    # checkpoint's 4x2560 bf16 geometry with synthetic weights of its measured
    # element statistics, the normed stream reaching ~+-30 as the real one does.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="checkpoint 4x2560 bf16, decode batch 8 / MTP draft rows",
            spec=dict(leading=(8,)),
        ),
        Cell(
            why="checkpoint 4x2560 bf16, prefill 8192 tokens, largest measured",
            spec=dict(leading=(8192,)),
        ),
        Cell(
            why="stack-final mixer: block_inject_weight None",
            spec=dict(leading=(256,), inject=False),
        ),
        # The MTP layer's shared_head mixer (block_inject_weight None) at the
        # rows DecodeTarget.mtp_forward runs: one request, the largest decode
        # graph batch, and mtp7's first draft over 128 x (1 + 7) verify rows.
        Cell(
            why="1 row, no injection: one request's MTP shared_head",
            spec=dict(leading=(1,), inject=False),
        ),
        Cell(
            why="128 rows, no injection: MTP shared_head at the largest decode graph batch",
            spec=dict(leading=(128,), inject=False),
        ),
        Cell(
            why="1024 rows, no injection: mtp7's first draft over 128 x 8 verify rows",
            spec=dict(leading=(1024,), inject=False),
        ),
        Cell(
            why="8192 rows, no injection: the envelope's largest, a max_num_tokens prefill chunk",
            spec=dict(leading=(8192,), inject=False),
        ),
    )

    note = """
    Metadata consumed: none; every row independent. Call sites: twice per
    decoder layer (attention and MoE hyper-connections, 96 calls) plus the
    stack-final mixer with `block_inject_weight=None` -- whose `mixed` *is*
    the checkpoint's final norm -- and three more in the MTP layer (two
    blocks and `shared_head`'s own mixer).

    Preconditions, all loud (each measured to raise `RuntimeError` before
    producing anything; the width one is tested): the width divisible by `hc_count` (unflatten), the
    weights' input width equal to it and their dtype equal to the
    activation's (`F.linear`).

    float32 activations are accepted and not certified: `F.linear` on fp32
    follows torch's TF32 default here, ~10 mantissa bits. A literal bf16
    transcription of the pointwise chain is ~2x this entry's error (0.0449 vs
    0.0227 on `mixed` with the real layer-0 weights).

    Operand range: with the real layer-0 weights the normed stream reaches
    |30.1|, the logits 47.8 / 67.5, `inject` spans 0.430 .. 1.578. At up to
    4096x that stream every output stays finite, `|mixed| <= max|normed|`,
    `inject` in `[0, 2]`; agreement with float64 degrades there because the
    sigmoid saturates (a property of the function, outside the checkpoint's
    range).

    Certified: bf16 2-D `[rows, 10240]` at 8 and 8192 rows with the
    injection; without it (the stack-final mixer, the MTP layer's
    shared_head) at 1, 128, 256, 1024 and 8192 rows; CUDA-graph replay at 8
    rows, bitwise equal to eager. Measured on sm_103 at the 1 / 128 / 1024 /
    8192-row cells without injection: `mixed` at 0.271 / 0.399 / 0.276 /
    0.371 of the test's band, cosine >= 0.9999982 and relative L2 <= 1.88e-3
    against float64; five wrong formulas (logit not divided by C, stream sum,
    ungated mean, mean gate times mean stream, hidden-major gate) at
    31.9x..192x the band. Cost, not gated, shared device: ~150-235 us at
    1..1024 rows (overhead-bound), 475 us at 4096, ~850-900 us at 8192.

    Not certified: other row counts; `inject` at 128 and 1024 rows (the MTP
    layer's unfused attention and MoE sites; `mixed` there is bitwise the
    no-injection cells'); more than one leading dim, hc_count other than 4,
    hidden other than 2560, lowrank other than 320, float16 and float32,
    weight/activation dtype mixing; the operand-range bounds above are
    measured. Hard limits: not probed (largest operand 8192 x 10240 = 2^26.3
    elements; a 2^31-element operand would take 209716 rows, beyond the
    envelope's 8192).
    """

    def __call__(
        self,
        hyper_input_normed: torch.Tensor,
        input_mix_weight_down: torch.Tensor,
        input_mix_weight_up: torch.Tensor,
        block_inject_weight: Optional[torch.Tensor],
        hc_count: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        activation_dtype = hyper_input_normed.dtype
        hidden_size = hyper_input_normed.shape[-1] // hc_count

        # GEMMs in the activation dtype: a bf16 product accumulates in fp32 with
        # no ambient state, an fp32 one would follow torch's TF32 default.
        low_rank = F.linear(hyper_input_normed, input_mix_weight_down)
        low_rank = F.silu(low_rank.float() / hc_count).to(activation_dtype)

        mix_weight = F.linear(low_rank, input_mix_weight_up)
        mix_weight = torch.sigmoid(mix_weight.float()).unflatten(-1, (hc_count, hidden_size))
        streams = hyper_input_normed.float().unflatten(-1, (hc_count, hidden_size))
        mixed_input = (mix_weight * streams).mean(dim=-2).to(activation_dtype)

        injection_weights = None
        if block_inject_weight is not None:
            inject_logits = F.linear(hyper_input_normed, block_inject_weight)
            injection_weights = 2.0 * torch.sigmoid(inject_logits.float() / hc_count)
            injection_weights = injection_weights.to(activation_dtype)

        return mixed_input, injection_weights

    def reference(
        self,
        hyper_input_normed: torch.Tensor,
        input_mix_weight_down: torch.Tensor,
        input_mix_weight_up: torch.Tensor,
        block_inject_weight: Optional[torch.Tensor],
        hc_count: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """float64 throughout -- an fp32 GEMM here would be TF32 and the less
        accurate side -- rounded once to the activation dtype."""
        dtype = hyper_input_normed.dtype
        hidden = hyper_input_normed.shape[-1] // hc_count
        x = hyper_input_normed.double()
        low = F.silu((x @ input_mix_weight_down.double().t()) / hc_count)
        gate = torch.sigmoid(low @ input_mix_weight_up.double().t()).unflatten(
            -1, (hc_count, hidden)
        )
        mixed = (gate * x.unflatten(-1, (hc_count, hidden))).mean(dim=-2).to(dtype)
        inject = None
        if block_inject_weight is not None:
            inject = 2.0 * torch.sigmoid((x @ block_inject_weight.double().t()) / hc_count)
            inject = inject.to(dtype)
        return mixed, inject


qwen4_exp_gated_residual = Qwen4ExpGatedResidual()
