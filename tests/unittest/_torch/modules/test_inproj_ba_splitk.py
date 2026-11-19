# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""in_proj_ba split-K kernel (kernels/blackwell/inproj_ba_splitk.py): bitwise equal to an in-test reference of
its arithmetic (four FP32 partials over K / 4, summed in order, then BF16) on data whose partials are exact in any
summation order while their serial sum rounds; a pairwise sum of the same partials gives other bits in every call."""

import pytest
import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._utils import is_blackwell_datacenter

pytestmark = pytest.mark.skipif(
    not is_blackwell_datacenter(),
    reason="the kernel runs on SM 10.0 / 10.3 (bitwise against cuBLAS there)",
)

K, NB, KC = 4096, 16, 1024


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def _ordered(m, seed):
    """Split 0: +-2^25 at one position plus even-integer noise (every partial sum a multiple of 4 below 2^26), split
    2: its negation, permuted; splits 1 / 3: integers in [-1, 1]. Every split partial is then exact in FP32 whatever
    the order of its sums, while (p0 + p1) and (p2 + p3) round to multiples of 4."""
    g = _gen(seed)

    def ints(shape, bound):
        return torch.randint(-bound, bound + 1, shape, generator=g, device="cuda").float()

    x = torch.empty(m, K, device="cuda")
    w = torch.empty(NB, K, device="cuda")
    x[:, :KC] = 2 * ints((m, KC), 4)
    w[:, :KC] = 2 * ints((NB, KC), 4)
    x[:, 0] = 2.0**13 * torch.where(ints((m,), 1) >= 0, 1.0, -1.0)
    w[:, 0] = 2.0**12
    perm = torch.randperm(KC, generator=g, device="cuda")
    x[:, 2 * KC : 3 * KC] = -x[:, :KC][:, perm]
    w[:, 2 * KC : 3 * KC] = w[:, :KC][:, perm]
    for s in (1, 3):
        x[:, s * KC : (s + 1) * KC] = ints((m, KC), 1)
        w[:, s * KC : (s + 1) * KC] = ints((NB, KC), 1)
    return x.to(torch.bfloat16).contiguous(), w.to(torch.bfloat16).contiguous()


def _bits(y):
    return y.view(torch.int16)


def _split_partials(x, w):
    """The four split partials, exact (FP64 sums of exact products), as FP32."""
    return [
        (x[:, s * KC : (s + 1) * KC].double() @ w[:, s * KC : (s + 1) * KC].double().t()).float()
        for s in range(4)
    ]


def test_plan():
    from tensorrt_llm._torch.kernels.blackwell import inproj_ba_splitk as R

    assert R.plan(56, K, NB) == 2 and R.plan(77, K, NB) == 3 and R.plan(96, K, NB) == 3
    assert R.plan(32, K, NB) == 1 and R.plan(97, K, NB) is None
    # K in whole barrier groups of x boxes per split, N within TMEM sub-partition 0's 16 rows
    assert R.plan(56, 2048, NB) == 2 and R.plan(56, 3072, 8) == 2 and R.plan(56, 1536, NB) is None
    assert R.plan(56, K, 32) is None and R.plan(56, K, 12) is None
    # ~110 KB at the model's shape, so a CTA fits beside the MoE all-reduce's CTAs in the 196 KB shared-memory split
    assert R._smem_bytes(K, NB) <= 120 * 1024, R._smem_bytes(K, NB)


@pytest.mark.parametrize("m", [1, 33, 56, 77, 96])
def test_bitwise_vs_serial_reference(m):
    from tensorrt_llm._torch.kernels.blackwell import inproj_ba_splitk as R

    calls = 4
    diff = 0
    pairwise_differs = 0
    for c in range(calls):
        x, w = _ordered(m, 700 + 10 * c)
        p = _split_partials(x, w)
        serial = (((p[0] + p[1]) + p[2]) + p[3]).to(torch.bfloat16)
        pairwise = ((p[0] + p[1]) + (p[2] + p[3])).to(torch.bfloat16)
        y = R.ba_splitk(x, w)
        torch.cuda.synchronize()
        diff += int(not torch.equal(_bits(y), _bits(serial)))
        pairwise_differs += int(not torch.equal(_bits(serial), _bits(pairwise)))
    assert diff == 0, diff
    assert pairwise_differs == calls, pairwise_differs


def test_supports_only_operands_the_kernel_reads():
    """A planned shape of contiguous BF16 operands at the compiled 128-byte alignment; a misaligned or strided view
    falls back instead of asserting."""
    from tensorrt_llm._torch.kernels.blackwell.inproj_ba_splitk import supports

    buf = torch.zeros((57 * K,), dtype=torch.bfloat16, device="cuda")
    w = torch.zeros((NB, K), dtype=torch.bfloat16, device="cuda")
    x = buf[: 56 * K].view(56, K)
    assert supports(x, w)
    assert supports(buf[K : 57 * K].view(56, K), w)
    assert not supports(buf[8 : 8 + 56 * K].view(56, K), w)  # 16-byte aligned only
    assert not supports(x.t().contiguous().t(), w)
    assert not supports(x.float(), w)
