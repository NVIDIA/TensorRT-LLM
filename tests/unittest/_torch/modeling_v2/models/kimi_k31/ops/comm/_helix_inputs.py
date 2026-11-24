# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and gates shared by the Helix entries' tests (one rank here, four in ``_helix_matrix``).

Only torch at module scope: the 4-rank launcher imports this before any tensorrt_llm state.
"""

from __future__ import annotations

import torch

HEADS = 96
LATENT = 512
VALUE = 128
#: The target's receive workspace per rank (``comm._TAIL_BYTES``) and its halves.
TAIL_BYTES = 16 << 20
HALF_BYTES = TAIL_BYTES // 2 // 16 * 16


def _gen(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def w_uv(seed: int) -> torch.Tensor:
    """BF16 [96, 128, 512] V up-projection, at the checkpoint's scale (unit-variance outputs)."""
    w = torch.randn(HEADS, VALUE, LATENT, generator=_gen(seed), device="cuda") / LATENT**0.5
    return w.to(torch.bfloat16)


def rank_inputs(spec, seed: int, rank: int):
    """One rank's latent rows [T * 96, 512], their stats [T * 96, 2] and its zero-KV mask [T]
    (``spec["masked"]``: random tokens on ranks above 0, so every token keeps a source)."""
    g = _gen(seed * 131 + rank)
    tokens = spec["tokens"]
    rows = torch.randn(tokens * HEADS, LATENT, generator=g, device="cuda").to(torch.bfloat16)
    peak = torch.randn(tokens * HEADS, generator=g, device="cuda") * 4
    total = torch.rand(tokens * HEADS, generator=g, device="cuda") * 50 + 1
    stats = torch.stack((peak, total), dim=-1).contiguous()
    mask = None
    if spec["masked"]:
        mask = torch.rand(tokens, generator=g, device="cuda") < 0.4
        if rank == 0:
            mask.zero_()
    return rows, stats, mask


def gate(spec, seed: int) -> torch.Tensor:
    width = HEADS // spec["cp_size"] * VALUE
    g = _gen(seed * 7 + 3)
    return (torch.randn(spec["tokens"], width, generator=g, device="cuda") * 2).to(torch.bfloat16)


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    mag = x.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


def assert_within_bf16_ulp(out: torch.Tensor, ref: torch.Tensor, ulps: int, where: str) -> None:
    """``out`` within ``ulps`` BF16 ulp of ``ref`` per element, against the row's scale: the V
    projection's FP32 accumulation (one ulp of the FP64 value) and the FP32 combine. A missing or
    doubled source moves the rows by tens of ulp."""
    assert out.shape == ref.shape and out.dtype == ref.dtype, (where, out.shape, ref.shape)
    o, r = out.float(), ref.float()
    assert not bool(o.isnan().any()), f"{where}: NaN in the output"
    scale = r.abs().amax(dim=-1, keepdim=True)
    err = (o - r).abs()
    bound = ulps * bf16_ulp(scale)
    if not bool((err <= bound).all()):
        worst = (err / bound).argmax().item()
        raise AssertionError(
            f"{where}: {int((err > bound).sum())} elements past {ulps} ulp of the row scale, "
            f"worst {err.flatten()[worst].item():.3e} at {worst}"
        )


class LocalWorkspace:
    """A one-rank receive workspace: the target's 16 MiB, armed with 0xFF bytes, and its launch
    counter."""

    def __init__(self, device: str = "cuda"):
        self.buffer = torch.full((TAIL_BYTES,), 0xFF, dtype=torch.uint8, device=device)
        self.state = torch.zeros(8, dtype=torch.int64, device=device)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count

    def args(self):
        """``(ws_addr, rank_stride, half_bytes, state, max_ctas, use_pdl)``."""
        return (self.buffer.data_ptr(), TAIL_BYTES, HALF_BYTES, self.state, self.sms, True)
