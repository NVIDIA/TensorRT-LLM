# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_sparse_dcp_decode entry, 16 Helix ranks emulated on one GPU (a
real FlatDsaCore per rank): each rank's pushes, as every owner's mailbox holds them, against the FP32
attention over that rank's kept rows; empty rows push only the empty LSE; the q mailbox re-armed."""

import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_sparse_dcp_decode import (  # noqa: E501
    dsa_sparse_dcp_decode as op,
)

__extra_import_path__ = ["../../../..", "."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_ull_dsa_core_group import (  # noqa: E402 — needs the path declared above
    LATENT,
    LOCAL,
    PARITY,
    WORLD,
    CoreGroup,
    seed_of,
)
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the sparse DCP decode needs an SM10x GPU",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, monkeypatch) -> None:
    rows, kept = cell.spec["rows"], cell.spec["kept"]
    group = CoreGroup(monkeypatch)
    q, steps = group.step(rows, kept, seed_of(cell.why))
    torch.cuda.synchronize()
    refs = []
    for core, s in zip(group.cores, steps):
        decode = core.kernels[rows][0]
        refs.append(op.reference(core, decode, PARITY, s.kv, s.indices, s.valid, s.latent, s.lse,
                                 None))  # fmt: skip
    stream = group.stream()
    with validating(op):
        for core, s in zip(group.cores, steps):
            decode = core.kernels[rows][0]
            op(core, decode, PARITY, s.kv, s.indices, s.valid, s.latent, s.lse, stream)
    torch.cuda.synchronize()
    empty_lse = group.dcp.EMPTY_LSE
    sent = group.dcp.PAIR_SENT - (1 << 32)
    for src, (ref, ref_lse) in enumerate(refs):
        s = steps[src]
        for owner, mail in enumerate(group.mails):
            heads = slice(LOCAL * owner, LOCAL * (owner + 1))
            got = mail.partial[PARITY, src, :rows].float()
            got_lse = mail.lse[PARITY, src, :rows]
            for b in range(rows):
                if int(s.valid[b]) == 0:
                    assert bool((got_lse[b] == empty_lse).all()), f"src {src} row {b}: LSE"
                    unsent = mail.partial[PARITY, src, b].view(torch.int32) == sent
                    assert bool(unsent.all()), f"src {src} row {b}: an empty row's partial sent"
                    continue
                # The FP32 LSE of the same logits: a few FP32 ulps of its magnitude.
                assert (got_lse[b] - ref_lse[b, heads]).abs().max() <= 1e-3
                # E4M3 P (relative step 2^-4 per probability, the subnormals' absolute step
                # 2^-9 / 448 below the row max) over FP32 sums of the unquantized P, then one
                # BF16 rounding: within 2^-4 of sum p |v| / sum p, the subnormal floor and 2^-8.
                k = s.kv[s.indices[b, : int(s.valid[b])].long()].view(torch.float8_e4m3fn)
                v = k.float()[:, :LATENT]
                qh = q[b, heads].float()
                p = torch.softmax(qh @ k.float().t() * core_scale(group), dim=1)
                mag = p @ v.abs()
                floor = int(s.valid[b]) * 2.0**-9 / 448 * v.abs().max()
                band = 2.0**-4 * mag + 2.0**-8 * ref[b, heads].abs() + floor + 1e-6
                err = (got[b] - ref[b, heads]).abs()
                assert bool((err <= band).all()), f"src {src} owner {owner} row {b}: {err.max()}"
    for core in group.cores:
        assert bool((core.mla_query.mailbox[PARITY, :rows] == 0xFF).all()), "q mailbox not re-armed"


def core_scale(group) -> float:
    return group.cores[0].scale_log2 * math.log(2.0)


def test_is_valid_refuses_what_the_kernel_would_read_wrongly(monkeypatch) -> None:
    group = CoreGroup(monkeypatch)
    _, steps = group.step(4, (300, 0, 2048, 65), seed=5)
    core, s = group.cores[0], steps[0]
    decode = core.kernels[4][0]
    bad = s.indices.clone()
    bad[0, 0] = s.kv.shape[0]
    with pytest.raises(ValueError, match="KV pool"):
        op.is_valid(core, decode, PARITY, s.kv, bad, s.valid, s.latent, s.lse, None)
    over = s.valid.clone()
    over[0] = 4096
    with pytest.raises(ValueError, match="kept counts"):
        op.is_valid(core, decode, PARITY, s.kv, s.indices, over, s.latent, s.lse, None)
    assert WORLD == 16
