# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One step of the multi-launch vocab-parallel DSpark Markov chain: the Markov bias added to a
block position's shard logits and each split's argmax published for the exchange (Triton)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_spec.vocab_parallel_kernels import _PairLayout, markov_add_local_argmax
from ._reference import publish_pairs


class K3MarkovAddLocalArgmax(OpWrapper):
    """``k3_vocab_local_select`` (argmax) of ``step_logits + bias`` (``[requests, shard]``, each
    upcast to FP32 and added in FP32): every split's ``(value, col_base + column)`` pair into
    ``layout.buffer``, other ranks' columns zeroed when ``zero_width > 0``. Returns None."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    # The multi-launch chain of helix8 c240 (20 requests, a 6-position block, 20480-column
    # shards): the step's logits are a strided view of the block logits.
    CELLS = (
        Cell(
            why="helix8 c240, rank 0, step 0: 20 requests, 10 splits of 2048 columns",
            spec=dict(tp=8, rank=0, reqs=20, block=6, step=0),
        ),
        Cell(
            why="helix8 c240, last rank, last step: the zero fill around the top slots",
            spec=dict(tp=8, rank=7, reqs=20, block=6, step=5),
        ),
    )

    note = """
    ``step_logits`` is one block position of the shard block logits (a strided view with a unit
    inner stride); ``bias`` is ``F.linear(markov_w1[prev], w2_shard)`` in the weights' dtype.
    Grid (requests, splits), 8 warps.
    """

    def __call__(
        self, step_logits: torch.Tensor, bias: torch.Tensor, col_base: int, layout: _PairLayout
    ) -> None:
        markov_add_local_argmax(step_logits, bias, col_base, layout)

    def reference(
        self, step_logits: torch.Tensor, bias: torch.Tensor, col_base: int, layout: _PairLayout
    ) -> torch.Tensor:
        """``layout.buffer`` after the call (a clone)."""
        return publish_pairs(layout, step_logits.float() + bias.float(), col_base)

    def is_valid(
        self, step_logits: torch.Tensor, bias: torch.Tensor, col_base: int, layout: _PairLayout
    ) -> None:
        assert step_logits.stride(-1) == 1 and bias.stride(-1) == 1, "unit inner strides"
        assert bias.shape == step_logits.shape, "one bias row per request, shard wide"
        assert layout.splits * layout.chunk >= step_logits.shape[1], "the splits miss columns"


k3_markov_add_local_argmax = K3MarkovAddLocalArgmax()
