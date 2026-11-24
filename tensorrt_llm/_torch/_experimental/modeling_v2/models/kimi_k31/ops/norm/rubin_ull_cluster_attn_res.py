# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MR8's seven-CTA cluster AttnRes on a tensor delta
(``kernels/vendor/vllm_mr8/specialized_decode/cutedsl/attn_res``'s ``ClusterAttnRes``, launched by
``PreparedAttnRes.run``): residual add, block-attention residual mixture and the layer's RMSNorm in
one launch."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import _rubin_ull_attn_res_reference as _ref


def _cell(why: str, rows: int, nb: int, has_delta: bool, write: int, aux: bool) -> Cell:
    key = (nb, has_delta, write, False)
    return Cell(why=why, spec=dict(rows=rows, key=key, aux=aux, slots=0))


class ClusterAttnRes(OpWrapper):
    """One token per seven-CTA cluster over ``prefix`` [T, 7168]: the new prefix row
    ``bf16(prefix + delta)`` (``bf16(delta)`` where ``prepared.key``'s ``has_delta`` is False) is
    stored into ``prefix`` (and ``blocks[:, write]`` where ``write >= 0``); its logit and the
    ``nb`` saved blocks' (``blocks`` [T, >= nb, 7168], any row/block strides) are softmax-mixed
    and the RMSNorm of the mixture times the output norm is stored into ``out`` [T, 7168]; ``aux``
    [T, 7168] or None gets the BF16 mixture (``bind_aux(mix=True)``). ``prepared`` is the layer's
    ``PreparedAttnRes``: its compiled variant, weights, epsilons and ``key = (nb, has_delta, write,
    final)``. Returns ``out``.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The served sites (93 layers, a block every 12): the AttnRes before a layer whose input the
    # MoE tail ring does not finish (L0, L1, the unfused paths), and the post-attention AttnRes
    # after an unfused o-projection (L0 KDA, the DSA fallback). nb = ceil(layer / 12).
    CELLS: tuple[Cell, ...] = (
        _cell("L0 before: no blocks, the embedding is the new block 0", 8, 0, False, 0, True),
        _cell("L1 before: one saved block, b1 q8", 8, 1, False, -1, True),
        _cell("L1 before: one saved block, b8 q8 (64 rows)", 64, 1, False, -1, True),
        _cell("L12 before: a block boundary (write 1)", 32, 1, False, 1, True),
        _cell("L5 post: delta added, one saved block", 8, 1, True, -1, False),
        _cell(
            "L0 post: the boundary layer's post site (its row starts block 1)",
            8,
            1,
            False,
            -1,
            False,
        ),
        _cell("L50 post: five blocks, b4 q8", 32, 5, True, -1, False),
        _cell("L12 post: a boundary post site (nb + 1 blocks, no delta)", 64, 2, False, -1, False),
    )

    note = """
    The K3.1 ULL target's ``FlatChainRuntime`` sites ``before_plain`` and ``post``. The launch
    triggers its PDL dependents at entry and reads ``prefix`` and ``blocks`` before its own
    ``griddepcontrol.wait``: they must be complete when it launches (its predecessors trigger only
    after their waits); ``delta`` is read after the wait. ``out`` must not alias ``prefix``.
    GB300 (152 SMs) compiles the LSU-poll, stores-after-exchange variant; Rubin the others (same
    values). The first launch per variant compiles at construction, outside graph capture.
    """

    def __call__(
        self,
        prepared,
        delta: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        out: torch.Tensor,
        aux: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return prepared.run(delta, prefix, blocks, out, aux=aux)

    def reference(
        self,
        prepared,
        delta: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        out: torch.Tensor,
        aux: Optional[torch.Tensor],
        *,
        weights: Sequence[torch.Tensor],
        eps: Sequence[float],
    ) -> dict:
        """The values the launch writes (``out``, ``prefix``, ``aux``; ``blocks[:, write]`` is the
        new ``prefix``), FP32, from the unmodified inputs. ``prepared`` holds its weights only as
        kernel views: ``weights`` (selection norm, projection, output norm) and ``eps`` are those
        it was built from. ``out`` and ``aux`` are not read."""
        rows = prefix.shape[0]
        return _ref.attn_res(
            prepared.key, weights, eps, prepared.aux_mix, _ref.as_delta(delta, rows), prefix, blocks
        )

    def is_valid(
        self,
        prepared,
        delta: torch.Tensor,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        out: torch.Tensor,
        aux: Optional[torch.Tensor],
    ) -> None:
        """Every operand is read through 16-byte row fragments of a 7168-wide BF16 row."""
        nb = prepared.key[0]
        rows = prefix.shape[0]
        for name, t in (("delta", delta), ("prefix", prefix), ("out", out), ("aux", aux)):
            if t is None:
                continue
            if (
                t.dtype != torch.bfloat16
                or t.shape[-1] != _ref.HIDDEN
                or t.stride(-1) != 1
                or t.shape[0] < rows
            ):
                raise ValueError(f"cluster_attn_res: {name} must be BF16 [>= T, 7168] rows")
        if (
            blocks.dtype != torch.bfloat16
            or blocks.shape[0] < rows
            or blocks.shape[1] < max(nb, prepared.key[2] + 1)
        ):
            raise ValueError("cluster_attn_res: blocks must be BF16 [T, > max(nb, write), 7168]")
        if aux is not None and prepared.aux is None:
            raise ValueError("cluster_attn_res: aux needs bind_aux")
        if out.data_ptr() == prefix.data_ptr():
            raise ValueError("cluster_attn_res: out aliases prefix, which the launch rewrites")


cluster_attn_res = ClusterAttnRes()
