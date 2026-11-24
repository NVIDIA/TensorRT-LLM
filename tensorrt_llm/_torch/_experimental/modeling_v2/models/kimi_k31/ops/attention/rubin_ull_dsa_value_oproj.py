# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's Helix value merge, V projection and output gate fused into MR8's o-projection
+ TP16 all-reduce + post-attention AttnRes (``kernels/attention/rubin_ull_dsa_oproj_fused.py``,
``features.dsa_value_oproj``): ``dsa_dcp_value`` followed by ``OprojFused`` in one kernel."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .rubin_ull_dsa_dcp_value import dsa_dcp_value

LOCAL, V_DIM, ROWS = 6, 128, 8


def _cell(why: str, rows: int, kept: tuple[int, ...], nb: int, delta: bool, late=False) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, kept=kept, nb=nb, delta=delta, late=late))


class DsaValueOproj(OpWrapper):
    """For this rank's 6 heads, the gated value rows ``dsa_dcp_value`` computes (the pushed
    partials of ``core.mail``'s parity of ``layer_idx`` merged with weights ``2^(lse_s - max) /
    sum`` in FP32 and rounded to BF16, times the head's ``W_UV`` (BF16 MMA, FP32 accumulation,
    rounded to BF16), times ``sigmoid(gate)``, rounded to BF16) written into ``x`` (BF16 [rows,
    768], this layer's parity of the x buffers, every word armed to ``PAIR_SENT`` before the
    call), then MR8's fused o-projection of ``x`` (``OprojFused`` with ``x_shared``): the
    [7168, 768] product's TP16 all-reduce through ``prepared``'s Lamport mailbox and the
    post-attention AttnRes, updating ``prefix`` and writing the MoE input ``out``. Re-arms the
    partial / LSE slots it read, the o-projection slots it read and every word of ``x_other``
    (the other parity, BF16 [8, 768]).

    ``code``: ``compile_value_oproj`` for these rows and this layer's AttnRes shape; ``core``: the
    prepared ``FlatDsaCore`` (the ULL target's ``modeling.py``: mailboxes, ``W_UV`` views, totals,
    poll counters, parity);
    ``prepared``: the layer's MR8 ``PreparedOproj`` (weight view, ``head`` = (mailbox, multicast
    address, rank), the AttnRes weight views and eps); ``gate``: BF16 [rows, 768] at the packed
    projection's pitch; ``prefix`` [rows, 7168], ``blocks`` [rows, nb, 7168] (a view),
    ``out`` [rows, 7168] BF16. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1: 8 rows, AttnRes 2 blocks + prefix", 8, (2048, 0, 7, 517, 130, 1, 2048, 64), 2, True),
        _cell("8 rows, a block-write layer (no prefix delta)", 8, (300, 2048, 65, 1, 7, 9, 2048, 0), 3, False),
        _cell("4 rows: merge CTAs 4-7 hold no row", 4, (300, 0, 2048, 65), 1, True),
        _cell("q1 row", 1, (2048,), 1, True),
        _cell("8 rows, partials pushed while the kernel polls", 8, (2048, 0, 7, 517, 130, 1, 2048, 64), 2, True, True),
    )  # fmt: skip

    note = """
    The K3.1 ULL generation target's DSA tail at the fused o-projection's rows (up to 8) under
    ``features.dsa_value_oproj``, after ``dsa_sparse_dcp_decode`` or the speculative decode,
    replacing ``dsa_dcp_value`` + ``OprojFused``. The partial / LSE polls are bounded like
    DcpValue's (``common/spin.py``, about 67 s): a timed-out source is counted in ``spins[1]``
    (LSE) or ``spins[2]`` (rows), then writes the ``kernels/common/rubin_ull_spin_trap.py`` record
    (sites 195 / 196) and traps; the x poll of every CTA on the merge CTAs traps at site 197.
    Every read of a buffer this layer's kernels wrote follows the kernel's griddepcontrol.wait
    (the decode before it may release it at its start); the kernel releases its dependents at its
    start, as ``OprojFused``. Numerics: the operations and rounding points of ``dsa_dcp_value``
    then ``OprojFused(x_shared)``; x differs from ``dsa_dcp_value``'s rows at most in the sign of
    a zero (the o-projection treats both alike).
    """

    def __call__(
        self, code, core, prepared, layer_idx, x, x_other, gate, prefix, blocks, out, stream
    ):
        from ...kernels.vendor.vllm_mr8 import module

        host = module("cutedsl.common.host")
        to_cute, dynamic = core.to_cute, host.dynamic_view
        parity, rows = core.mla_query.parity[layer_idx], x.shape[0]
        attn_res = (dynamic(prefix), dynamic(blocks), *prepared.views, dynamic(out), None, None)
        merge = (
            *core.box_views[parity],
            core.uv_views[layer_idx],
            to_cute(gate),
            to_cute(core.totals[:rows], 4),
            core.mla_query.spins_view,
            to_cute(x_other),
        )
        code(to_cute(x), prepared.wview, *prepared.head, *attn_res, *prepared.eps, *merge, stream)

    def reference(
        self, code, core, prepared, layer_idx, x, x_other, gate, prefix, blocks, out, stream
    ) -> torch.Tensor:
        """The BF16 [rows, 768] the kernel writes into ``x`` (``dsa_dcp_value.reference`` of this
        rank's mailbox as it holds the pushes: call it before the kernel re-arms them). The
        o-projection stage is MR8's ``OprojFused`` on these rows (the test runs it)."""
        rows = x.shape[0]
        parity = core.mla_query.parity[layer_idx]
        latent = core.latent[:rows]  # only its row count is read
        return dsa_dcp_value.reference(
            core, None, parity, layer_idx, None, gate, None, latent, None, None
        )

    def is_valid(
        self, code, core, prepared, layer_idx, x, x_other, gate, prefix, blocks, out, stream
    ):
        """A call the kernel would answer wrongly: ``x`` / ``x_other`` not the two parities of
        one armed [2, 8, 768] buffer (a stale word of ``x`` is taken as written), or more rows
        than the merge clusters hold."""
        rows = x.shape[0]
        if not 1 <= rows <= ROWS or x.shape[1] != LOCAL * V_DIM or x_other.shape != (ROWS, 768):
            return f"x [{rows}, {x.shape[1]}] / x_other {tuple(x_other.shape)}"
        if x.dtype != torch.bfloat16 or x_other.dtype != torch.bfloat16:
            return "x buffers must be BF16"
        if not bool((x.view(torch.int32) == -(1 << 31)).all()):
            return "x is not armed (PAIR_SENT words)"
        return None


dsa_value_oproj = DsaValueOproj()
