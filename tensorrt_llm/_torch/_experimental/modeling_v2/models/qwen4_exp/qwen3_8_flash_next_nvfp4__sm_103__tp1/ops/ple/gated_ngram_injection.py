# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE gating: one shared n-gram value injected into every hyper-connection
stream, each scaled by its own key-query gate.

Tier-3 composed from native torch (two upcasts, one reduction, the signed
square root, one broadcast multiply); it launches no TensorRT-LLM kernel.
Nothing registered under ``torch.ops.trtllm`` computes a dot-product gate or a
signed square root (``fused_sigmoid_mul_inplace`` takes a per-element gate and
could at most stand in for the last multiply). ``Engram.forward`` in
``tensorrt_llm/_torch/modules/engram/engram.py`` computes this gate as four
statements inside a forward that also runs the projections, both norms (plain
``x * w``, not this checkpoint's grouped gemma ``(1 + w)``) and a stateless
short conv -- not an atomic callable, and a different composite. The built-in
Qwen4-Exp implementation has its own, not wrapped here; the semantics follow
the HuggingFace ``Qwen4ExpTextPLELayer``.
"""

from __future__ import annotations

import math

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's PLE geometry and its gate floor (`_PLE_CLAMP_MIN`).
CKPT_HC_COUNT = 4
CKPT_HIDDEN = 2560
CKPT_CLAMP_MIN = 1e-6


def gate_ratio(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst element's distance from `ref` as a fraction of the test's 16-bit
    band: `rtol = eps`, `atol = tiny * eps` (one subnormal spacing). Below 1 is
    inside."""
    info = torch.finfo(ref.dtype)
    ref64 = ref.double()
    return (
        ((out.double() - ref64).abs() / (info.tiny * info.eps + info.eps * ref64.abs()))
        .max()
        .item()
    )


class GatedNgramInjection(OpWrapper):
    """For token `t`, stream `s`, `D = value.shape[-1]`, with `key_normed` and
    `query_normed` packed stream-major as `hc_count` blocks of `D`::

        dot[t, s]  = sum_i key_normed[t, s*D + i] * query_normed[t, s*D + i]
        g          = dot / sqrt(D)
        signed     = sign(g) * sqrt(max(|g|, clamp_min))
        out[t, s*D + i] = sigmoid(signed[t, s]) * value[t, i]

    Returns a fresh contiguous `[..., hc_count * D]` in **`value`'s dtype**
    (a bf16 key/query with an fp32 value returns fp32). The value is shared:
    every stream carries a differently scaled copy of the same vector.
    `sign(0) = 0`, so an exactly zero dot gives `sigmoid(0) = 0.5` in every
    dtype; `clamp_min` only replaces a small nonzero magnitude. The whole
    chain runs in fp32 and rounds once.

    Fusion boundary: the per-stream dot, scale, floor, signed root, sigmoid
    and broadcast multiply. The `key_proj` / `value_proj` GEMMs, both grouped
    gemma norms producing the operands, the short conv over the result and
    the add into the stream are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: kind "ckpt" (default) -- operands with this checkpoint's norm-gamma
    # and value statistics: leading, value_dtype (bf16), clamp_min. kind
    # "single_channel" -- one nonzero channel per stream so each dot is an exact
    # product under the test's control: dots, value_dtype, clamp_min.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="checkpoint 4x2560 bf16, 33 tokens / MTP verify 8 x 4 + 1",
            spec=dict(leading=(33,)),
        ),
        Cell(
            why="checkpoint 4x2560 bf16, prefill 8192 tokens, largest measured",
            spec=dict(leading=(8192,)),
        ),
        Cell(
            why="dots 0, +-5e-8 (inside the floor), 0.05, fp32 value",
            spec=dict(
                kind="single_channel",
                dots=(0.0, 5e-8, -5e-8, 0.05),
                value_dtype=torch.float32,
                clamp_min=CKPT_CLAMP_MIN,
            ),
        ),
    )

    note = """
    Metadata consumed: none; every token row and stream independent. Call
    site: once per forward, on the PLE layer, positional `clamp_min` =
    `_PLE_CLAMP_MIN` (1e-6), `key_normed` / `query_normed` `[T, 10240]` and
    `value` `[T, 2560]`, all bf16.

    Preconditions: `key_normed.shape[-1] == hc_count * value.shape[-1]` (the
    unflatten raises otherwise, which also catches a wrong `hc_count`). Leading
    dims must match exactly (`is_valid`); measured, a single query or value
    row broadcasts silently over every key row.

    fp32 rather than the HF reference's bf16 chain: against the unrounded
    float64 the fp32 chain sat at 0.498 of a one-epsilon bf16 band, the
    literal bf16 chain at 4.693 (4.81 against `reference`), and 31.5% of the
    output's bf16 elements differ between the two -- a bitwise comparison
    against a bf16 implementation will not match.

    The floor is invisible in bf16: a sub-floor dot moves the sigmoid by ~5e-4
    of its value, measurable in fp32 and below half a bf16 ulp
    (`clamp_min` 1e-6 and 0.0 bitwise identical with a bf16 `value`).

    Every near-miss tried (no root, unsigned root, no / misplaced / `1/D`
    scale, sigmoid before the root, tanh, stream-minor packing) stays bounded
    by `|value|`; they land 105x..1115x outside the band.

    CUDA-graph capture certified, bitwise equal to eager; deterministic.
    Cost, not gated: 0.114 ms at 8 tokens, 0.128 at 512, 0.495 at 4096,
    0.956 at 8192 (launch-bound below ~512); transient allocation 3.56x the
    operand bytes (the fp32 upcasts).

    Hard limits: not probed (largest operand 8192 x 10240 = 2^26.3 elements).
    """

    def __call__(
        self,
        key_normed: torch.Tensor,
        query_normed: torch.Tensor,
        value: torch.Tensor,
        hc_count: int,
        clamp_min: float = 1e-6,
    ) -> torch.Tensor:
        activation_dtype = value.dtype
        hidden_size = value.shape[-1]

        keys = key_normed.unflatten(-1, (hc_count, hidden_size)).float()
        queries = query_normed.unflatten(-1, (hc_count, hidden_size)).float()

        gate = (keys * queries).sum(dim=-1, keepdim=True) / math.sqrt(hidden_size)
        gate = gate.abs().clamp_min(clamp_min).sqrt() * gate.sign()

        gated = torch.sigmoid(gate) * value.float().unsqueeze(-2)
        return gated.flatten(-2).to(activation_dtype)

    def reference(
        self,
        key_normed: torch.Tensor,
        query_normed: torch.Tensor,
        value: torch.Tensor,
        hc_count: int,
        clamp_min: float = 1e-6,
    ) -> torch.Tensor:
        """float64 -- the entry already computes in fp32, so an fp32 reference
        would be the same arithmetic -- rounded once to `value`'s dtype."""
        hidden = value.shape[-1]
        keys = key_normed.double().unflatten(-1, (hc_count, hidden))
        queries = query_normed.double().unflatten(-1, (hc_count, hidden))
        gate = (keys * queries).sum(dim=-1, keepdim=True) / math.sqrt(hidden)
        gate = gate.abs().clamp_min(clamp_min).sqrt() * gate.sign()
        out = torch.sigmoid(gate) * value.double().unsqueeze(-2)
        return out.flatten(-2).to(value.dtype)

    def is_valid(
        self,
        key_normed: torch.Tensor,
        query_normed: torch.Tensor,
        value: torch.Tensor,
        hc_count: int,
        clamp_min: float = 1e-6,
    ) -> None:
        # Both measured: a [1, width] query beside [64, width] keys, or a
        # [1, D] value, returns 64 rows computed from the wrong row.
        assert key_normed.shape == query_normed.shape, (
            f"key_normed {tuple(key_normed.shape)} and query_normed "
            f"{tuple(query_normed.shape)} must have the same shape; a mismatch "
            f"broadcasts silently instead of raising"
        )
        assert value.shape[:-1] == key_normed.shape[:-1], (
            f"value {tuple(value.shape)} must carry the same leading dims as "
            f"key_normed {tuple(key_normed.shape)}; a mismatch broadcasts silently "
            f"instead of raising"
        )


gated_ngram_injection = GatedNgramInjection()
