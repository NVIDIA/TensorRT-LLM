# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Masked store of a step's context latents of every drafter layer into the paged pools
(Triton, one program per token and layer)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.spec_tail_kernels import store_context_latents, store_mla_context_latents


def to_e4m3_saturated(x: torch.Tensor) -> torch.Tensor:
    """``to_draft_fp8``: E4M3 at unit scale, saturated to +-448, round to nearest even."""
    return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


class K3StoreContextLatents(OpWrapper):
    """``pools[l][pages[t], 0, 0, offsets[t]] = latent[t, l] * mask[t]`` for every token ``t``
    and layer ``l``. Returns None.

    ``latent`` is ``[N, L, D]`` with a unit inner stride; ``pools`` the ``L`` per-layer
    ``[pages, 1, 1, page_size, D]`` views (identical strides), ``pool_ptrs`` their ``[L]`` int64
    base addresses, ``pages`` / ``offsets`` ``[N]`` and ``mask`` ``[N]`` int8. ``mask`` 0 stores a
    zero (the signed zero of ``x * 0``). With ``mla`` (the MLA drafter's store), ``mask`` -1 leaves
    the slot untouched and an FP8 E4M3 pool stores ``to_draft_fp8`` of the masked latent;
    without it the pools have ``latent``'s dtype.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS = (
        Cell(
            why="dep16 (c383): 16 requests x 6 latents of 5 layers into FP8 pools, 0 / 1 masks",
            spec=dict(num_gens=16, width=6, cp_size=1, pool="fp8", mla=True),
        ),
        Cell(
            why="helix8 c75/c174: 8 requests x 8 latents, -1 on the other 7 ranks' positions",
            spec=dict(num_gens=8, width=8, cp_size=8, pool="fp8", mla=True),
        ),
        Cell(
            why="helix16_moetp4ep4 (ULL c4): 4 requests x 8 latents over 16 ranks",
            spec=dict(num_gens=4, width=8, cp_size=16, pool="fp8", mla=True),
        ),
        Cell(
            why="the unfused preparation's store into BF16 pools (no shipped profile takes it)",
            spec=dict(num_gens=16, width=6, cp_size=1, pool="bf16", mla=False),
        ),
    )

    note = """
    The multiply by the mask runs in FP32 and rounds back to the latent dtype, which is exact for
    0 and 1. The MLA store is the one the tail-fused generation step runs (possibly on the
    drafter's side stream); the plain one belongs to the unfused preparation, which no shipped
    K3 generation profile takes.
    """

    def __call__(
        self,
        latent: torch.Tensor,
        pools: list,
        pool_ptrs: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        mask: torch.Tensor,
        mla: bool = True,
    ) -> None:
        if mla:
            store_mla_context_latents(latent, pools, pool_ptrs, pages, offsets, mask)
        else:
            store_context_latents(latent, pools, pool_ptrs, pages, offsets, mask)

    def reference(
        self,
        latent: torch.Tensor,
        pools: list,
        pool_ptrs: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        mask: torch.Tensor,
        mla: bool = True,
    ) -> list[torch.Tensor]:
        """The ``L`` pools after the call (clones); ``pool_ptrs`` is not read."""
        out = []
        pages_i, offsets_i = pages.to(torch.int64), offsets.to(torch.int64)
        write = mask >= 0 if mla else torch.ones_like(mask, dtype=torch.bool)
        keep = (mask > 0).to(torch.float32)[:, None]
        for layer, pool in enumerate(pools):
            new = pool.clone()
            vals = (latent[:, layer].float() * keep).to(latent.dtype)
            target = new[:, 0, 0]
            if new.dtype == torch.float8_e4m3fn:
                # Assigned as bytes: the bits are the same.
                vals = to_e4m3_saturated(vals).view(torch.uint8)
                target = target.view(torch.uint8)
            target[pages_i[write], offsets_i[write]] = vals[write].to(target.dtype)
            out.append(new)
        return out

    def is_valid(
        self,
        latent: torch.Tensor,
        pools: list,
        pool_ptrs: torch.Tensor,
        pages: torch.Tensor,
        offsets: torch.Tensor,
        mask: torch.Tensor,
        mla: bool = True,
    ) -> None:
        # The kernel stores through pool_ptrs, not through the views it is handed.
        expected = [p.data_ptr() for p in pools]
        assert pool_ptrs.tolist() == expected, "pool_ptrs must hold the pools' base addresses"
        assert mask.dtype == torch.int8, "mask must be int8"
        if not mla:
            # The plain kernel multiplies by the mask: -1 would store a negated latent.
            assert bool((mask >= 0).all()), "the plain store takes a 0 / 1 mask"


k3_store_context_latents = K3StoreContextLatents()
