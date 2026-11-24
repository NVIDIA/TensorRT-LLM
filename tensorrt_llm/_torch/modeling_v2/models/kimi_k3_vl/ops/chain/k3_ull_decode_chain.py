# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer entry points for the ULL decode kernels in ``kernels/k3r_*/ull/``.

The chain is one persistent run of fixed-shape decode kernels per step over the target's layers:
every KDA + MoE layer whole (in-projection, KDA verify, o-projection, MoE), the MoE of every MLA
layer, and layer 0's KDA core. Each kernel hands its output to the next through the chain's
symmetric-memory mailboxes, so its inputs are the chain's prepared weights and buffers
(:class:`~.k3_ull_chain.K3UllChain`, built once by the target) plus the step's activations passed here.
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
    checkpoint's weights, and the model's step against eager in
    `test_k3_ull_small_model_graph.py`.
"""


class K3UllChainedLayer(OpWrapper):
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
        raise NotImplementedError("K3UllChainedLayer.reference: no plain-torch model yet")


class K3UllChainMoe(OpWrapper):
    """``chain.run_moe``: an MLA + MoE layer's MoE on its normed MoE input ``x`` (``prefix``
    already holds the attention output); updates ``prefix`` / ``blocks`` in place and returns the
    next layer's normed attention input, or after the last layer the output AttnRes + final norm
    rows (``aux``, ``mx``: as :class:`K3UllChainedLayer`)."""

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
        raise NotImplementedError("K3UllChainMoe.reference: no plain-torch model yet")


class K3UllSoloKda(OpWrapper):
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
        raise NotImplementedError("K3UllSoloKda.reference: no plain-torch model yet")


ull_chained_layer = K3UllChainedLayer()
ull_chain_moe = K3UllChainMoe()
ull_solo_kda = K3UllSoloKda()
