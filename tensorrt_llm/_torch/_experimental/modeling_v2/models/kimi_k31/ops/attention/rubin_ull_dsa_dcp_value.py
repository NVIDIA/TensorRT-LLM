# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA layer's Helix combine + V projection + output gate for this rank's 6 heads, from the
partials every rank's sparse decode pushed (vLLM e3ddbdbd60's ``DcpValue``, ``pushed``,
``kernels/vendor/vllm_main/.../mla/dcp.py`` with TensorRT-LLM's bounded polls)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

HEADS, LATENT, V_DIM = 96, 512, 128


def _cell(why: str, rows: int, kept: tuple[int, ...]) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, kept=kept))


class DsaDcpValue(OpWrapper):
    """For each row ``b < rows`` and each of this rank's ``core.local`` heads ``j`` (global head
    ``core.rank * local + j``): the sources' pushed partials in this rank's mailbox
    (``core.mail.partial[parity, s, b, j]`` BF16 [512], base-2 LSE ``core.mail.lse[parity, s, b,
    j]``, ``s`` over the ``core.world`` ranks; empty sources pushed only ``EMPTY_LSE``) merged
    with weights ``2^(lse_s - max) / sum`` in FP32 and rounded to BF16, times ``W_UV`` of the head
    (``core.w_uv[layer_idx][j]`` [512, 128], BF16 MMA with FP32 accumulation, rounded to BF16),
    times ``sigmoid(gate[b, 128 j : 128 j + 128])``, rounded to BF16, into ``out[b, 128 j : ...]``.
    Polls each source's LSE and rows until pushed, then re-arms what it read.

    ``core``: the prepared ``FlatDsaCore`` (the ULL target's ``modeling.py``); ``value``: its
    compiled kernel for ``rows`` (``core.kernels[rows][1]``); ``valid``: the kept counts the decode
    used (int32 [rows]);
    ``gate``: BF16 [rows, 768] at the packed projection's pitch; ``out``: contiguous BF16
    [rows, 768]; ``latent`` / ``lse``: the decode's unused operands (read here only as the
    [rows, 1, 96] row / LSE views the kernel takes). Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1: 8 rows (2 rows per CTA)", 8, (2048, 0, 7, 517, 130, 1, 2048, 64)),
        _cell("q1 row", 1, (2048,)),
        _cell("4 rows, one empty on every rank (zero output)", 4, (300, 0, 2048, 65)),
    )

    note = """
    The K3.1 ULL generation target's DcpValue after ``dsa_sparse_dcp_decode`` (rows 1-8, 16
    ranks x 6 heads). The mailbox polls are bounded (``common/spin.py``, a local change to the
    vendored vLLM file): a source not pushed within ``common.spin.SPIN_NS`` (2^26 us, about 67 s)
    is counted in ``spins[1]`` (LSE) or ``spins[2]`` (rows), then, compiled with the target's
    ``record`` (``compile_value(record=spin_trap.record_address())``), writes the
    ``kernels/common/rubin_ull_spin_trap.py`` record (sites 191-194) and traps instead of merging
    it as it is. Waits (griddepcontrol.wait) before any read and releases its dependents after it: the
    DSA o-projection that follows reads its prefix / blocks before its own wait on that basis.
    """

    def __call__(self, core, value, parity, layer_idx, valid, gate, out, latent, lse, stream):
        cutlass, cute_view = core.cutlass, core.to_cute
        rows = latent.shape[0]
        # DcpValue (pushed): its row / LSE operands are the unused [rows, 1, 96] views.
        gathered = HEADS
        rows_view = latent.as_strided((rows, 1, gathered, LATENT), (HEADS * LATENT, 0, LATENT, 1))
        lse_view = lse.as_strided((rows, 1, gathered), (HEADS, 0, 1))
        value(
            cute_view(rows_view),
            cute_view(lse_view, 4),
            cute_view(valid, 4),
            *core.box_views[parity],
            *core.peer_views,
            core.q_views[parity],
            core.uv_views[layer_idx],
            cute_view(gate),
            cute_view(core.totals[:rows], 4),
            cute_view(out),
            cutlass.Int32(parity),
            cutlass.Int32(core.rank),
            core.mla_query.spins_view,
            stream,
        )

    def reference(
        self, core, value, parity, layer_idx, valid, gate, out, latent, lse, stream
    ) -> torch.Tensor:
        """BF16 [rows, local * 128] from this rank's mailbox as it holds the pushes (read before
        the call re-arms it, so call this first): the FP32 merge rounded to BF16, an FP32
        projection rounded to BF16, the FP32 gate product rounded to BF16. ``valid``, ``out``,
        ``latent``, ``lse`` and ``stream`` are not read."""
        rows, local = latent.shape[0], core.local
        partial = core.mail.partial[parity, :, :rows].float()  # [world, rows, local, 512]
        lse2 = core.mail.lse[parity, :, :rows].float()  # [world, rows, local]
        sent = lse2 > -1e38  # EMPTY_LSE (-3e38) and the -1 sentinel are not mergeable sources
        top = torch.where(sent, lse2, torch.full_like(lse2, -torch.inf)).amax(0)
        weight = torch.where(sent, torch.exp2(lse2 - top), torch.zeros_like(lse2))
        total = weight.sum(0)
        partial = torch.where(sent.unsqueeze(-1), partial, torch.zeros_like(partial))
        merged = (weight.unsqueeze(-1) * partial).sum(0) / total.clamp_min(1e-30).unsqueeze(-1)
        merged = torch.where((total > 0).unsqueeze(-1), merged, torch.zeros_like(merged))
        with true_fp32_matmul():
            values = torch.einsum(
                "rjl,jlv->rjv", merged.bfloat16().float(), core.w_uv[layer_idx].float()
            ).bfloat16()
        g = torch.sigmoid(gate.float().reshape(rows, local, V_DIM))
        return (values.float() * g).bfloat16().reshape(rows, local * V_DIM)

    def is_valid(self, core, value, parity, layer_idx, valid, gate, out, latent, lse, stream):
        """``FlatDsaCore.run`` checks the operand layouts; nothing here is accepted and answered
        wrongly beyond a mailbox the decodes of this step did not fill, which the kernel waits
        for (bounded) rather than reads -- see the note."""
        return None


dsa_dcp_value = DsaDcpValue()
