# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram branch: the gated memory residual Hy4-Flash adds to its hyper-connection streams.

HF ``HYV4FlashEngramLayer.forward``, per stream ``hc``, all in the hidden dtype::

    key    = RMSNorm(key_projs[hc](embeddings))              # gamma: norm_key[hc]
    query  = RMSNorm(hidden_states[..., hc, :])              # gamma: norm_query[hc]
    g      = (key * query).sum(-1) * H**-0.5
    gate   = sigmoid(sign(g) * sqrt(|g|.clamp_min(1e-6)))
    signal = pre_gamma * RMSNorm(value_proj(embeddings))     # no affine
    out    = gate[..., None] * signal + hc_scales[hc] * signal
    out    = out.masked_fill(~engram_mask[..., None, None], 0.0)

The caller adds ``out`` to the streams. The key projections stay per-stream ``F.linear`` calls
so the result is bit-identical to HF's module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: HF's ``nn.RMSNorm(size)`` has no ``eps``, which resolves to the fp32 eps (not bf16's).
_NORM_EPS = torch.finfo(torch.float32).eps

#: HF's floor on the gate magnitude, applied before the signed square root.
_GATE_FLOOR = 1e-6


class EngramBranch(OpWrapper):
    """``hidden_states`` ``[..., HC, H]`` and ``embeddings`` ``[..., E]`` -> ``[..., HC, H]``.

    Returns the residual, ``gate * signal + hc_scales * signal`` per stream, zeroed where
    ``engram_mask`` is false; the caller adds it to ``hidden_states``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `layer` selects the checkpoint's weight statistics: layer 31 is the widest branch, layer 1
    # the dimmest.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode: one token at the widest branch", spec=dict(rows=1, layer=31)),
        Cell(why="a six-token DFly2 verify step", spec=dict(rows=6, layer=31)),
        Cell(why="the largest padded decode graph batch", spec=dict(rows=64, layer=31)),
        Cell(why="a 512-token chunked-prefill chunk", spec=dict(rows=512, layer=31)),
        Cell(why="the widest prefill chunk: 5120 tokens", spec=dict(rows=5120, layer=31)),
        Cell(why="the 8192-token max_num_tokens envelope", spec=dict(rows=8192, layer=31)),
        Cell(why="the dimmest branch: layer 1 at 64 tokens", spec=dict(rows=64, layer=1)),
    )

    note = """
    The in-tree `_torch/modules/engram` gate is the DeepSeek-V4.1 variant and computes a different
    function. An fp32 `pre_gamma`/`hc_scales` would silently promote the branch to fp32.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        key_weights: torch.Tensor,
        key_biases: torch.Tensor,
        key_norm_weights: torch.Tensor,
        query_norm_weights: torch.Tensor,
        value_weight: torch.Tensor,
        value_bias: torch.Tensor,
        pre_gamma: torch.Tensor,
        hc_scales: torch.Tensor,
        engram_mask: torch.Tensor,
    ) -> torch.Tensor:
        streams = hidden_states.shape[-2]
        hidden = hidden_states.shape[-1]
        if streams != key_weights.shape[0]:
            raise ValueError(
                f"hidden_states carries {streams} streams and key_weights "
                f"{key_weights.shape[0]}: the Engram layer has one key projection per stream"
            )
        if engram_mask.shape != hidden_states.shape[:-2]:
            raise ValueError(
                f"engram_mask {tuple(engram_mask.shape)} does not cover the "
                f"{tuple(hidden_states.shape[:-2])} positions of hidden_states"
            )
        gates = []
        for hc in range(streams):
            key = F.linear(embeddings, key_weights[hc], key_biases[hc])
            key = F.rms_norm(key, (hidden,), key_norm_weights[hc], _NORM_EPS)
            query = F.rms_norm(
                hidden_states[..., hc, :], (hidden,), query_norm_weights[hc], _NORM_EPS
            )
            gate = (key * query).sum(dim=-1) * hidden**-0.5
            gate = gate.abs().clamp_min(_GATE_FLOOR).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim=-1).unsqueeze(-1)
        signal = pre_gamma * F.rms_norm(
            F.linear(embeddings, value_weight, value_bias), (hidden,), None, _NORM_EPS
        )
        out = gates * signal.unsqueeze(-2) + hc_scales * signal.unsqueeze(-2)
        return out.masked_fill(~engram_mask[..., None, None], 0.0)

    def reference(
        self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
        key_weights: torch.Tensor,
        key_biases: torch.Tensor,
        key_norm_weights: torch.Tensor,
        query_norm_weights: torch.Tensor,
        value_weight: torch.Tensor,
        value_bias: torch.Tensor,
        pre_gamma: torch.Tensor,
        hc_scales: torch.Tensor,
        engram_mask: torch.Tensor,
    ) -> torch.Tensor:
        """HF's formula in float64, rounded once to ``hidden_states.dtype`` at the end."""
        streams = hidden_states.shape[-2]
        hidden = hidden_states.shape[-1]
        h = hidden_states.double()
        e = embeddings.double()
        gates = []
        for hc in range(streams):
            key = F.linear(e, key_weights[hc].double(), key_biases[hc].double())
            key = (
                key
                * torch.rsqrt(key.pow(2).mean(dim=-1, keepdim=True) + _NORM_EPS)
                * key_norm_weights[hc].double()
            )
            query = h[..., hc, :]
            query = (
                query
                * torch.rsqrt(query.pow(2).mean(dim=-1, keepdim=True) + _NORM_EPS)
                * query_norm_weights[hc].double()
            )
            gate = (key * query).sum(dim=-1) * hidden**-0.5
            gate = gate.abs().clamp_min(_GATE_FLOOR).sqrt() * gate.sign()
            gates.append(torch.sigmoid(gate))
        gates = torch.stack(gates, dim=-1).unsqueeze(-1)
        value = F.linear(e, value_weight.double(), value_bias.double())
        signal = (
            pre_gamma.double()
            * value
            * torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + _NORM_EPS)
        )
        out = gates * signal.unsqueeze(-2) + hc_scales.double() * signal.unsqueeze(-2)
        return out.masked_fill(~engram_mask[..., None, None], 0.0).to(hidden_states.dtype)

    # No `is_valid`: a wrong stream count or mask shape raises in `__call__`; dtype or width
    # mismatches raise in the projections and norms.


engram_branch = EngramBranch()
