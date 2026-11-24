# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""iHC hyper-connection mapping: gate the checkpoint's residual streams into a layer's input.

A torch macro entry (HF ``HYV4FlashHyperConnection`` / ``HYV4FlashHyperHead``)::

    flat = streams.flatten(-2).float()  # [..., N*H]
    logits = F.linear(flat, fn) * rsqrt(flat.pow(2).mean(-1, keepdim=True) + norm_eps)
    pre = sigmoid(logits[..., :N] * scale[0] + base[:N]) + eps
    post = post_magnitude * sigmoid(logits[..., N:] * scale[1] + base[N:]) + eps
    out = (pre.unsqueeze(-1) * streams).sum(-2)

The head (``fn`` ``[N, N*H]``) has one gate set, ``sigmoid(logits * scale + base) + eps``. The
post-mapping back into the streams is the caller's.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class IhcMapping(OpWrapper):
    """``(out [..., H] in streams.dtype, gates [..., N or 2N] fp32)``; the caller applies ``post``."""

    ARCHS = frozenset({Arch.SM_103})

    # Four streams of 3072; `fn_rms` spans the checkpoint's hc_fn tensors (0.0130..0.0493).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="connection, decode: 1 row, 4 streams, 8 gate rows",
            spec=dict(
                rows=1,
                streams=4,
                hidden=3072,
                groups=2,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0133,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="connection, 64 rows: the largest padded decode graph batch",
            spec=dict(
                rows=64,
                streams=4,
                hidden=3072,
                groups=2,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0133,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="connection, prefill: 5120 rows at the model's widest hc_fn",
            spec=dict(
                rows=5120,
                streams=4,
                hidden=3072,
                groups=2,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0493,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="head, decode: 1 row, 4 streams, 4 gate rows",
            spec=dict(
                rows=1,
                streams=4,
                hidden=3072,
                groups=1,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0130,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="head, 64 rows: the largest padded decode graph batch",
            spec=dict(
                rows=64,
                streams=4,
                hidden=3072,
                groups=1,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0130,
                stream_sigma=8.0,
            ),
        ),
        Cell(
            why="head, prefill: 5120 rows at the model's widest hc_fn",
            spec=dict(
                rows=5120,
                streams=4,
                hidden=3072,
                groups=1,
                dtype=torch.bfloat16,
                post_magnitude=2.0,
                eps=1e-6,
                norm_eps=1e-5,
                fn_rms=0.0493,
                stream_sigma=8.0,
            ),
        ),
    )

    note = """
    The projection runs under `true_fp32_matmul()`: TF32 at 5120 rows moves it far outside the
    band. A non-fp32 `base`/`scale` promotes silently and loses precision.
    """

    def __call__(
        self,
        streams: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        post_magnitude: float,
        eps: float,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        streams_per_row = streams.shape[-2]
        with true_fp32_matmul():
            flat = streams.flatten(-2).float()
            inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + norm_eps)
            logits = F.linear(flat, fn) * inv_rms
        gate_rows = logits.shape[-1]
        if gate_rows == streams_per_row:
            gates = torch.sigmoid(logits * scale + base) + eps
        elif gate_rows == 2 * streams_per_row:
            pre = (
                torch.sigmoid(logits[..., :streams_per_row] * scale[0] + base[:streams_per_row])
                + eps
            )
            post = (
                post_magnitude
                * torch.sigmoid(logits[..., streams_per_row:] * scale[1] + base[streams_per_row:])
                + eps
            )
            gates = torch.cat((pre, post), dim=-1)
        else:
            raise ValueError(
                f"fn has {gate_rows} rows for {streams_per_row} streams: expected "
                f"{streams_per_row} (head) or {2 * streams_per_row} (connection)"
            )
        out = (gates[..., :streams_per_row].unsqueeze(-1) * streams).sum(dim=-2)
        return out.to(streams.dtype), gates

    def reference(
        self,
        streams: torch.Tensor,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        post_magnitude: float,
        eps: float,
        norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """float64 throughout, rounded once at the end."""
        streams_per_row = streams.shape[-2]
        flat = streams.double().flatten(-2)
        logits = F.linear(flat, fn.double()) * torch.rsqrt(
            flat.pow(2).mean(dim=-1, keepdim=True) + norm_eps
        )
        gate_rows = logits.shape[-1]
        if gate_rows == streams_per_row:
            gates = torch.sigmoid(logits * scale.double() + base.double()) + eps
        elif gate_rows == 2 * streams_per_row:
            pre = (
                torch.sigmoid(
                    logits[..., :streams_per_row] * scale[0].double()
                    + base[:streams_per_row].double()
                )
                + eps
            )
            post = (
                post_magnitude
                * torch.sigmoid(
                    logits[..., streams_per_row:] * scale[1].double()
                    + base[streams_per_row:].double()
                )
                + eps
            )
            gates = torch.cat((pre, post), dim=-1)
        else:
            raise ValueError(
                f"fn has {gate_rows} rows for {streams_per_row} streams: expected "
                f"{streams_per_row} (head) or {2 * streams_per_row} (connection)"
            )
        out = (gates[..., :streams_per_row].unsqueeze(-1) * streams.double()).sum(dim=-2)
        return out.to(streams.dtype), gates.float()

    # No `is_valid`: a wrong stream layout raises in the projection or the gate-row check.


ihc_mapping = IhcMapping()
