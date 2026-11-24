# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE layer's routing at TP16: ``kernels/vendor/vllm_mr8/.../cutedsl/moe/front.py``'s
``RouteSitu`` with the route and copy roles (vLLM MR8), which polls the 896 router logits and the
latent rows the 16 ranks' front GEMMs publish into the front mailbox, ranks the top 16 and writes
the NVFP4 latent plan of the routed experts (the TensorRT-LLM NVFP4 writers)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

EXPERTS = 896
TOP_K = 16
LATENT = 3584
#: The front mailbox row: BF16 latent [3584], then the FP32 logits [896] as BF16 pairs.
WIDTH = LATENT + 2 * EXPERTS
SENTINEL = -0x80000000  # an unwritten mailbox word (``common.mailbox``)


@dataclass(frozen=True, eq=False)
class MoeRouteLaunch:
    """One row count's prepared routing launch (``moe.route_launches``, at load).

    ``code``: the compiled ``RouteSitu`` (route + copy roles; the carve-out launcher where the
    target sets one); ``args`` / ``extra``: its CuTe arguments (logits, bias, ids, weights,
    mailbox, latent, gate / up, hidden, the routed scaling factor and the SiTU constants; then
    the plan, the experts' route mailbox when it signals them, the activation global scale).
    The torch tensors those views were made from follow, for ``reference`` and ``is_valid``
    only: ``mailbox`` the front mailbox [1, 64, WIDTH], ``bias`` [896], ``ids`` / ``weights`` /
    ``latent`` the rows' outputs, ``plan`` the experts' plan buffer, ``routes`` the route
    mailbox (None unless signalled), and ``scale`` / ``renorm`` / ``gsf``.
    """

    code: Any
    args: tuple
    extra: tuple
    rows: int
    mailbox: torch.Tensor
    bias: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    latent: torch.Tensor
    plan: torch.Tensor
    image: bool
    routes: torch.Tensor | None
    scale: float
    renorm: bool
    gsf: float


def _cell(why: str, rows: int, logits: str = "random") -> Cell:
    """``logits``: "random" (N(0, 2) logits, small bias) or "ties" (equal scores everywhere)."""
    return Cell(why=why, spec=dict(rows=rows, logits=logits))


class MoeRoute(OpWrapper):
    """For each row ``t``: ``s = sigmoid(logits[t])`` (the FP32 words of front mailbox row ``t``,
    columns [3584, 5376)), ``ids[t]`` = the 16 experts of largest ``s + bias`` (a tie: the lower
    expert first), ``weights[t] = BF16(scale * s[ids] / (sum s[ids] + 1e-20))`` (renormalized;
    else ``scale * s[ids]``); ``latent[t]`` = the mailbox row's BF16 latent, bitwise; and the
    latent as NVFP4 (16-element blocks, UE4M3 scales under ``gsf``) into the experts' plan (the
    image at 1-8 rows, the grouped plan at 9-64). The mailbox cells it read are re-armed; with
    ``routes`` the ids and the image count also go to the experts' route mailbox. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("q1 decode: 1 row (polled logits, not lean)", 1),
        _cell("b1 verify: 8 rows, lean, two copy CTAs a row, signals the experts", 8),
        _cell("b2 verify: 16 rows into the grouped plan", 16),
        _cell("b4 verify: 32 rows into the grouped plan", 32),
        _cell("every score equal: the lowest 16 experts, in order", 8, "ties"),
    )

    note = """
    The routing of the K3.1 ULL generation target (MoE TP16 EP1; 896 experts, top 16, sigmoid
    + bias, renormalized and scaled). Lamport: the kernel polls the front mailbox cells until the
    16 ranks' front GEMMs have written them, so it has no data dependency on its predecessor and
    returns only when every row it routes has been published. The cells write the mailbox rows
    from the test (the other ranks' front GEMMs stand in) and run the launch the GB300 target
    prepares; Rubin's ``release_early`` variant (the experts released after the front GEMM) is
    the same kernel with another release point and is not driven here.
    """

    def __call__(self, launch: MoeRouteLaunch, stream: Any) -> None:
        launch.code(*launch.args, *launch.extra, stream)

    def reference(self, launch: MoeRouteLaunch, stream: Any) -> dict[str, torch.Tensor]:
        """FP64 / exact: ``ids`` [rows, 16] int64 (by rank), ``keys`` [rows, 896] (s + bias, to
        judge near ties), ``weights`` [rows, 16], ``latent`` [rows, 3584] BF16 (the copy) and
        ``plan`` [rows, 3584] (the NVFP4 values, code x scale) from the mailbox rows."""
        from .rubin_ull_nv_routed_experts import e2m1_round, hidden_scales

        rows = launch.rows
        cells = launch.mailbox[0, :rows]
        logits = cells[:, LATENT:].contiguous().view(torch.float32).double()
        scores = torch.sigmoid(logits)
        keys = scores + launch.bias.double()
        # Descending key, then ascending expert: a stable sort of the experts in order.
        order = torch.sort(keys, dim=1, descending=True, stable=True).indices[:, :TOP_K]
        picked = scores.gather(1, order)
        weights = picked * launch.scale
        if launch.renorm:
            weights = weights / (picked.sum(1, keepdim=True) + 1e-20)
        latent = cells[:, :LATENT].clone()  # the kernel re-arms the cells it read
        blocks = latent.double().view(rows, -1, 16)
        scale = hidden_scales(blocks.abs().amax(-1, keepdim=True), launch.gsf)
        multiplier = torch.where(scale > 0, launch.gsf / scale, torch.finfo(torch.float32).max)
        plan = (e2m1_round(blocks * multiplier) * scale).view(rows, LATENT)
        return dict(ids=order, keys=keys, weights=weights, latent=latent, plan=plan)

    def is_valid(self, launch: MoeRouteLaunch, stream: Any) -> None:
        """A signalling launch whose experts' route mailbox is not armed (all ones): the experts
        would take the previous layer's ids before this kernel writes them."""
        if launch.routes is not None and not bool((launch.routes == -1).all()):
            raise ValueError("moe_route: the experts' route mailbox is not armed")


moe_route = MoeRoute()
