# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine's low-M Hyper-Connections pieces a target's trunk uses around the split boundaries
(`tensorrt_llm/_torch/modules/mhc/mhc_lowm.py`), as entries: whether a boundary of `n` rows takes the split path
(`hc_split_enabled`), the Lamport all-reduce workspace of the split boundaries (`hc_lamport_workspace`, a setup entry
called once at load time), the un-finalized MoE output a boundary can combine itself (`moe_unfinalized`, a container
builder) and its plain top-k combine (`moe_unfinalized_finalize`). Each calls the engine's code unchanged."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

if TYPE_CHECKING:
    from tensorrt_llm._torch.modules.mhc.mhc_lowm import (  # noqa: F401
        LamportAllReduce,
        MnnvlLamportAllReduce,
    )


class HcSplitEnabled(OpWrapper):
    """`(num_tokens, hidden, mult, mega_mhc=None) -> bool`: whether a Hyper-Connections boundary of `num_tokens` rows
    can take the split (critical half + side-stream coefficients) path: an sm_100 / sm_103 / sm_107 device, 4 copies,
    a width that is a multiple of 256 and at most 8192, and 1 to `top` rows; never under ``FORCE_DETERMINISTIC=1``.
    With `mega_mhc` given (a caller whose coefficients past 64 rows are the one-launch kernels,
    `hc_lagged_coefficients`) `top` is the engine's `split_max_tokens(mega_mhc)` (`mhc_split_enabled`, #22360): on
    sm_100 / sm_103 1023 for an attention entry with a post update (`mega_mhc=True`: from 1024 rows it keeps the
    one-pass boundary) and 4096 for the others, 64 elsewhere; such a caller still checks `hc_lagged_coefficients.ready`
    past 64 rows. Without it (two-launch coefficients only) `top` is 64."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="tep8 decode: six rows at the model's width", spec=dict(n=6, hidden=5120, mult=4)),
        Cell(why="the two-launch split's last row count", spec=dict(n=64, hidden=5120, mult=4)),
        Cell(
            why="dep8 c7000 decode, 160 rows: past the two-launch split",
            spec=dict(n=160, hidden=5120, mult=4),
        ),
        Cell(
            why="dep8 c7000 decode, 160 rows, an attention entry",
            spec=dict(n=160, hidden=5120, mult=4, mega_mhc=True),
        ),
        Cell(
            why="an attention entry's last split row count",
            spec=dict(n=1023, hidden=5120, mult=4, mega_mhc=True),
        ),
        Cell(
            why="1024 rows: past an attention entry's split",
            spec=dict(n=1024, hidden=5120, mult=4, mega_mhc=True),
        ),
        Cell(
            why="dep8 batch 320, FFN boundary of 1280 rows",
            spec=dict(n=1280, hidden=5120, mult=4, mega_mhc=False),
        ),
        Cell(
            why="4097 rows: past every split",
            spec=dict(n=4097, hidden=5120, mult=4, mega_mhc=False),
        ),
        Cell(why="an empty step", spec=dict(n=0, hidden=5120, mult=4)),
        Cell(why="a width the split kernel does not take", spec=dict(n=6, hidden=5000, mult=4)),
    )
    note = "A host predicate (no device work): the test gates it against the restated rule."

    def __call__(
        self, num_tokens: int, hidden: int, mult: int, mega_mhc: Optional[bool] = None
    ) -> bool:
        from tensorrt_llm._torch.modules.mhc.mhc_lowm import LOWM_MAX_TOKENS, mhc_split_enabled

        if mega_mhc is None:
            return num_tokens <= LOWM_MAX_TOKENS and mhc_split_enabled(num_tokens, hidden, mult)
        return mhc_split_enabled(num_tokens, hidden, mult, mega_mhc)

    def reference(
        self, num_tokens: int, hidden: int, mult: int, mega_mhc: Optional[bool] = None
    ) -> bool:
        cap = torch.cuda.is_available() and torch.cuda.get_device_capability()
        if cap not in ((10, 0), (10, 3), (10, 7)):
            return False
        top = 64
        if mega_mhc is not None and cap in ((10, 0), (10, 3)):
            top = 1023 if mega_mhc else 4096
        # FORCE_DETERMINISTIC=1 keeps every boundary on the fused path (stock's pinned tactic)
        deterministic = os.getenv("FORCE_DETERMINISTIC", "0") == "1"
        return (
            mult == 4
            and hidden % 256 == 0
            and hidden // 8 <= 1024
            and 0 < num_tokens <= top
            and not deterministic
        )


class HcLamportWorkspace(OpWrapper):
    """`(mapping, strategy) -> workspace or None`: the one-shot Lamport all-reduce workspace the split boundaries
    reduce their sub-layer output with (`hc_allreduce_post_collapse_norm`). The IPC workspace of the one-shot
    all-reduce for a tensor-parallel group of 2, 4 or 8 ranks inside one node with peer access (AUTO / ONESHOT), the
    MNNVL workspace for a group spanning trays (AUTO / MNNVL), None otherwise -- always None under attention DP (no
    tensor-parallel sum). A setup entry: collective over the group, called once by every rank at load time, never
    under graph capture."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep: attention DP, no tensor-parallel sum",
            spec=dict(world=8, tp=8, adp=True, strategy="AUTO"),
        ),
        Cell(
            why="a single rank: nothing to reduce",
            spec=dict(world=1, tp=1, adp=False, strategy="AUTO"),
        ),
        Cell(
            why="a group size the kernel does not take",
            spec=dict(world=1, tp=3, adp=False, strategy="AUTO"),
        ),
        Cell(
            why="tep8 inside one node, NCCL strategy: the boundary leaves the reduce to the engine",
            spec=dict(world=1, tp=4, adp=False, strategy="NCCL"),
        ),
    )
    note = """
    `reference` states the decision as a kind ("ipc", "mnnvl" or None); the single-process test drives the cells that
    need no peer memory. The IPC workspace itself (tep8's TP4 path) is exercised by the 4-rank
    `hc_allreduce_post_collapse_norm` test, which builds the same `LamportAllReduce`.
    """

    def __call__(self, mapping, strategy) -> "LamportAllReduce | MnnvlLamportAllReduce | None":
        from tensorrt_llm._torch.modules.mhc.mhc_lowm import mhc_lamport_allreduce

        return mhc_lamport_allreduce(mapping, strategy)

    def reference(self, mapping, strategy) -> Optional[str]:
        from tensorrt_llm._torch.distributed.ops import AllReduceStrategy

        dev = torch.cuda.is_available() and torch.cuda.get_device_capability() in (
            (10, 0),
            (10, 3),
            (10, 7),
        )
        if not dev or mapping.enable_attention_dp or mapping.tp_size not in (2, 4, 8):
            return None
        if mapping.is_multi_node():
            return (
                "mnnvl" if strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL) else None
            )
        return "ipc" if strategy in (AllReduceStrategy.AUTO, AllReduceStrategy.ONESHOT) else None


def moe_unfinalized(
    shared: torch.Tensor, fc2: torch.Tensor, scales: torch.Tensor, perm: torch.Tensor
):
    """The engine's `MoeUnfinalized` container (no device work): a rank's MoE output before the top-k combine --
    `fc2` [P, H] the permuted expert rows, `perm` [M, top_k] int32 the row of each (token, k) (-1: not on this rank),
    `scales` [M, top_k] the routing weights, `shared` [M, H] the shared-expert output. The split boundary
    (`hc_allreduce_post_collapse_norm`) combines it in its all-reduce; `moe_unfinalized_finalize` combines it alone."""
    from tensorrt_llm._torch.modules.mhc.mhc_lowm import MoeUnfinalized

    return MoeUnfinalized(shared=shared, fc2=fc2, scales=scales, perm=perm)


class MoeUnfinalizedFinalize(OpWrapper):
    """`parts` (`moe_unfinalized`) -> `[M, H]` the rank's partial MoE output: `shared + sum_k scales[:, k] *
    fc2[perm[:, k]]` over the valid `perm` entries, in fp32, one rounding to `shared`'s dtype (the engine's
    `MoeUnfinalized.finalize`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: six rows, top-6, about half the (token, k) pairs on this rank",
            spec=dict(m=6, k=6, h=5120),
        ),
        Cell(why="tep8 verify of four requests: 24 rows", spec=dict(m=24, k=6, h=5120)),
        Cell(
            why="every pair off this rank: the shared output alone",
            spec=dict(m=4, k=6, h=512, none=True),
        ),
    )
    note = (
        "fp32 sum of at most top_k + 1 terms against a float64 reference: one bf16 rounding step."
    )

    def __call__(self, parts) -> torch.Tensor:
        return parts.finalize()

    def reference(self, parts) -> torch.Tensor:
        perm = parts.perm.long()
        valid = (perm >= 0).unsqueeze(-1)
        rows = parts.fc2[perm.clamp(min=0)].double() * valid
        return ((rows * parts.scales.double().unsqueeze(-1)).sum(1) + parts.shared.double()).to(
            parts.shared.dtype
        )

    def is_valid(self, parts) -> None:
        m, k = parts.perm.shape
        assert (
            parts.scales.shape == (m, k)
            and parts.shared.shape[0] == m
            and parts.fc2.shape[1] == parts.shared.shape[1]
        )
        assert parts.perm.dtype == torch.int32 and int(parts.perm.max()) < parts.fc2.shape[0]


hc_split_enabled = HcSplitEnabled()
hc_lamport_workspace = HcLamportWorkspace()
moe_unfinalized_finalize = MoeUnfinalizedFinalize()
