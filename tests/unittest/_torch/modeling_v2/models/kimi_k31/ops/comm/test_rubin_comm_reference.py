# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the K3.1 Rubin comm entries: the references on small shapes, the guards and the
declarations. No kernel runs here (the four-rank GPU test
``rubin_multirank/test_rubin_comm_entries.py`` drives the cells)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
    _rubin_comm_common as lc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
    rubin_lamport_push as lp,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm import (
    rubin_latent_ar_norm as lat,
)

pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7),
        reason="K3.1 Rubin (sm_107) entries; the CPU stage runs this file",
    ),
]

ENTRIES = {
    "rubin_lamport_push": lp.rubin_lamport_push,
    "rubin_lamport_finish": lp.rubin_lamport_finish,
    "rubin_latent_ar_norm_row": lat.rubin_latent_ar_norm_row,
}


def _partials(world, m, h, seed=0, scale=0.25):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(m, h, generator=g) * scale).to(torch.bfloat16) for _ in range(world)]


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_entry_declares_its_cells(name) -> None:
    entry = ENTRIES[name]
    assert entry.CELLS
    whys = [c.why for c in entry.CELLS]
    assert all(whys) and len(set(whys)) == len(whys)
    specs = [repr(sorted(c.spec.items())) for c in entry.CELLS]
    assert len(set(specs)) == len(specs)
    assert all(c.spec["world"] == lc.WORLD for c in entry.CELLS), "the op test drives 4 ranks"


def test_reduce_chunk_is_the_cpp_kernels_order() -> None:
    assert [lc.reduce_chunk(w, True) for w in (2, 4, 8, 16, 32)] == [2, 4, 8, 8, 8]
    assert [lc.reduce_chunk(w, False) for w in (2, 4, 8, 16, 32)] == [2, 4, 8, 16, 16]


def test_allreduce_reference_sums_in_rank_order_by_chunks() -> None:
    xs = _partials(16, 2, 64, seed=1)
    for chunk in (16, 8):
        acc = torch.zeros(2, 64)
        for c0 in range(0, 16, chunk):
            part = torch.zeros(2, 64)
            for r in range(c0, c0 + chunk):
                part = part + xs[r].float()
            acc = acc + part
        got = lc.lamport_allreduce_reference(xs, chunk)
        assert torch.equal(got, acc.to(torch.bfloat16))
    # The order matters. Rank 8's 1 is lost in 2^24 + 1 when one chunk runs in rank order, and
    # kept when ranks 8..15 form their own partial sum (1 - 2^24 is exact in fp32).
    big = [torch.zeros(1, 8, dtype=torch.bfloat16) for _ in range(16)]
    big[0].fill_(2.0**24)
    big[8].fill_(1.0)
    big[9].fill_(-(2.0**24))
    assert float(lc.lamport_allreduce_reference(big, 16)[0, 0]) == 0.0
    assert float(lc.lamport_allreduce_reference(big, 8)[0, 0]) == 1.0


def test_allreduce_reference_never_returns_negative_zero() -> None:
    xs = [torch.full((1, 8), -0.0, dtype=torch.bfloat16) for _ in range(4)]
    out = lc.lamport_allreduce_reference(xs, 4)
    assert torch.equal(out.view(torch.int16), torch.zeros(1, 8, dtype=torch.int16))
    assert torch.equal(
        lc.sanitize_bf16(xs[0]).view(torch.int16), torch.zeros(1, 8, dtype=torch.int16)
    )


def test_allreduce_reference_adds_the_all_gather_rows() -> None:
    xs = _partials(4, 3, 64, seed=2)
    y = _partials(1, 3, 64, seed=3)[0]
    got = lc.lamport_allreduce_reference(xs, 4, y)
    want = (y.float() + lc.lamport_allreduce_reference(xs, 4).float()).to(torch.bfloat16)
    assert torch.equal(got, want)


def test_finish_references_agree() -> None:
    xs = _partials(4, 5, 64, seed=4)
    want = lc.lamport_allreduce_reference(xs, 4)
    assert torch.equal(lp.rubin_lamport_finish.reference(xs, True, None), want)
    assert torch.equal(lp.rubin_lamport_push.reference(None, xs[0], True, True), xs[0])


@pytest.mark.parametrize("norm", [False, True])
def test_latent_reference_is_the_fp32_norm_of_the_bf16_sum(norm) -> None:
    xs = _partials(4, 3, 128, seed=5, scale=1.0)
    gamma = (torch.rand(128, generator=torch.Generator().manual_seed(6)) + 0.5).to(torch.bfloat16)
    got = lat.latent_ar_norm_reference(xs, gamma, 1e-5, norm)
    s = lc.lamport_allreduce_reference(xs, 8)
    if not norm:
        assert torch.equal(got, s)
        return
    x = s.double()
    want = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * gamma.double()).float()
    assert torch.allclose(got.float(), want, rtol=2.0**-8, atol=0.0)
    out, xq, xsc = lat.rubin_latent_ar_norm_row.reference(xs, gamma, 1e-5, True, True)
    assert torch.equal(out, got) and xq.shape == (3, 128) and xsc.shape == (3, 4)


def test_latent_guard_refuses_a_rank_outside_the_group() -> None:
    x = torch.zeros(39, lat.LATENT, dtype=torch.bfloat16)
    gamma = torch.ones(lat.LATENT, dtype=torch.bfloat16)
    flags = torch.zeros(9, dtype=torch.int32)
    with pytest.raises(AssertionError, match="rank"):
        lat.rubin_latent_ar_norm_row.is_valid(
            x, gamma, 1e-5, x.clone(), 1, 1, flags, 4, 4, True, True, True, None, 0, False, False
        )
