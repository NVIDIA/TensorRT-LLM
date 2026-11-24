# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_dcp_value entry, 16 Helix ranks emulated on one GPU (a real
FlatDsaCore per rank) after every rank's dsa_sparse_dcp_decode: each owner's gated rows against the
merge of its own mailbox (FP32, the kernel's rounding points) and, loosely, against FP32 attention
over every rank's kept rows end to end; the mailboxes re-armed and no poll timed out."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_dcp_value import (  # noqa: E501
    dsa_dcp_value as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_sparse_dcp_decode import (  # noqa: E501
    dsa_sparse_dcp_decode,
)

__extra_import_path__ = ["../../../..", "."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_ull_dsa_core_group import (  # noqa: E402 — needs the path declared above
    LAYER,
    PARITY,
    CoreGroup,
    seed_of,
    union_reference,
)
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="DcpValue needs an SM10x GPU",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, monkeypatch) -> None:
    rows, kept = cell.spec["rows"], cell.spec["kept"]
    group = CoreGroup(monkeypatch)
    q, steps = group.step(rows, kept, seed_of(cell.why))
    stream = group.stream()
    # Every rank's decode first (the pushes land before any owner polls: one GPU, one stream).
    for core, s in zip(group.cores, steps):
        dsa_sparse_dcp_decode(core, core.kernels[rows][0], PARITY, s.kv, s.indices, s.valid,
                              s.latent, s.lse, stream)  # fmt: skip
    torch.cuda.synchronize()
    refs = []
    for core, s in zip(group.cores, steps):
        value = core.kernels[rows][1]
        refs.append(op.reference(core, value, PARITY, LAYER, s.valid, s.gate, s.out, s.latent,
                                 s.lse, None))  # fmt: skip
    with validating(op):
        for core, s in zip(group.cores, steps):
            value = core.kernels[rows][1]
            op(core, value, PARITY, LAYER, s.valid, s.gate, s.out, s.latent, s.lse, stream)
    torch.cuda.synchronize()
    ideal = union_reference(group, q, steps, rows)
    for owner, (core, s, ref) in enumerate(zip(group.cores, steps, refs)):
        # The merge of the same pushed BF16 partials in FP32 (another summation order), one BF16
        # rounding, the BF16 MMA's FP32 sums and two more BF16 roundings: a BF16 ulp or two of the
        # row's scale.
        assert_within_ulp(s.out, ref, element_ulp=4.0, rms_ulp=1.0)
        # End to end against FP32 attention over every rank's rows: the decode's E4M3 P (2^-4)
        # dominates; its errors average over the 512 latent columns the projection sums.
        x, want = s.out.float(), ideal[owner]
        rel = (x - want).norm() / want.norm().clamp_min(1e-30)
        assert rel <= 2.0**-5, f"owner {owner}: rel L2 {float(rel):.3e}"
        assert group.mails[owner].armed(group.dcp), f"owner {owner}: mailbox not re-armed"
        assert int(core.mla_query.spins.sum()) == 0, f"owner {owner}: a poll timed out"
    for b, n in enumerate(kept):
        if n == 0:
            assert all(bool((s.out[b] == 0).all()) for s in steps), f"row {b}: empty not zero"
