# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MR8 decode kernel chain's per-layer entry points (``kernels/k3r_*/ull/``, through ``chain.K3UllChain``).

The chain is one persistent run of fixed-shape decode kernels per step over the target's layers:
every KDA + MoE layer whole (in-projection, KDA verify, o-projection, MoE), the MoE of every MLA
layer, and layer 0's KDA core. Each kernel hands its output to the next through the chain's
symmetric-memory mailboxes, so its inputs are the chain's prepared weights and buffers
(:class:`~..chain.K3UllChain`, built once by the target) plus the step's activations passed here.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_NOTE = """
    The chain's kernels run on 8, 16, 24 or 32 rows (1 + 7 verified tokens of 1 to 4 requests) on
    all 16 ranks of the Helix CP16 / MoE TP16 world at once, on the weights, KDA checkpoint state
    and mailboxes `chain` holds; a reference would be the whole decoder layer in torch on the 16
    ranks' shards. Until one is written the entry has no cells: the chain is checked bitwise
    against the reference implementation it was flattened from on 16 GB300 ranks with the
    checkpoint's weights (notes `1110_k3_ull_v2_chain`), and the model's step against eager in
    `test_k3_ull_small_model_graph.py`.
"""


class Mr8ChainedLayer(OpWrapper):
    """``chain.run_layer``: a KDA + MoE layer whole on its normed attention input ``x``; updates
    ``prefix`` / ``blocks`` (AttnRes) in place and returns the next layer's normed attention
    input (``aux``: also stores it, the next layer's DSpark tap; ``mx``: a run's last layer also
    stores the next MLA layer's MXFP8 input rows)."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = ()
    note = _NOTE

    def __call__(
        self,
        chain: Any,
        layer_idx: int,
        x: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        aux: Optional[torch.Tensor],
        mx: bool,
    ) -> torch.Tensor:
        return chain.run_layer(layer_idx, x, prefix, blocks, aux, mx)

    def reference(self, *args: Any) -> torch.Tensor:
        """Not modelled yet (see the module note)."""
        raise NotImplementedError("Mr8ChainedLayer.reference: no plain-torch model yet")


class Mr8ChainMoe(OpWrapper):
    """``chain.run_moe``: an MLA + MoE layer's MoE on its normed MoE input ``x`` (``prefix``
    already holds the attention output); updates ``prefix`` / ``blocks`` in place and returns the
    next layer's normed attention input, or after the last layer the output AttnRes + final norm
    rows (``aux``, ``mx``: as :class:`Mr8ChainedLayer`)."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = ()
    note = _NOTE

    def __call__(
        self,
        chain: Any,
        layer_idx: int,
        x: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        aux: Optional[torch.Tensor],
        mx: bool,
    ) -> torch.Tensor:
        return chain.run_moe(layer_idx, x, prefix, blocks, aux, mx)

    def reference(self, *args: Any) -> torch.Tensor:
        """Not modelled yet (see the module note)."""
        raise NotImplementedError("Mr8ChainMoe.reference: no plain-torch model yet")


class Mr8SoloKda(OpWrapper):
    """``chain.run_solo``: layer 0's KDA in-projection + verify on the normed attention input
    ``x``, on the step's checkpoint slots (``chain.bind_slots`` first); returns the KDA core rows
    for the layer's o-projection."""

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = ()
    note = _NOTE

    def __call__(self, chain: Any, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        return chain.run_solo(layer_idx, x)

    def reference(self, *args: Any) -> torch.Tensor:
        """Not modelled yet (see the module note)."""
        raise NotImplementedError("Mr8SoloKda.reference: no plain-torch model yet")


mr8_chained_layer = Mr8ChainedLayer()
mr8_chain_moe = Mr8ChainMoe()
mr8_solo_kda = Mr8SoloKda()
