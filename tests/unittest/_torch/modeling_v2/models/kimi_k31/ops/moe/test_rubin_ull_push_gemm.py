# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 low-latency push GEMM (kernels/moe/rubin_ull_push_gemm.py) on one SM107 GPU.

One GPU has no multicast group, so the kernel runs the two-shot transport into eight local stand-in
peer buffers (token t to peer t % 8). Every pushed slot must equal the same split-K kernel's plain
store at the same tactic, ``-0`` sent as ``+0``, bit for bit; no other word of the buffers may
change; the payload stays within 2^-8 of the FP64 product. The all-gather target runs over the TP
group in rubin_multirank/test_rubin_ull_bf16_ranks.py.
"""

import _rubin_ull_reference as uref  # noqa: E402 -- the test-side references next to this file
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    _rubin_ull_common as c,
)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(not _SM107, reason="certified on SM107 only")

WORLD = 8
SENTINEL = -2147483648
NBYTES = 1 << 20


class _Peers:
    def __init__(self):
        self.bufs = [
            torch.full((3 * NBYTES // 4,), SENTINEL, dtype=torch.int32, device="cuda")
            for _ in range(WORLD)
        ]
        self.ptrs = torch.tensor(
            [b.data_ptr() for b in self.bufs], dtype=torch.int64, device="cuda"
        )
        self.flags = torch.tensor(
            [1, 2, NBYTES, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device="cuda"
        )

    def read(self, m, rank, row, col0, n):
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        for t in range(m):
            owner, off = uref.slot_offset(t, rank, WORLD, row, col0, False, 1, NBYTES)
            out[t] = self.bufs[owner][off // 4 : off // 4 + n // 2].view(torch.bfloat16)
        return out

    def written(self):
        return sum(int((b != SENTINEL).sum()) for b in self.bufs)


def _push(x, w, out, peers, rank, tactic, row, col0):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_ull_push_gemm as k,
    )

    k.run_ull_push(
        x,
        w,
        out,
        peer_ptrs=peers.ptrs.data_ptr(),
        mc_base=0,
        flags=peers.flags.data_ptr(),
        rank=rank,
        world=WORLD,
        tactic=k.tactic_of(tactic),
        target=k.SLOT,
        row=row,
        col0=col0,
        one_shot=False,
        use_pdl=True,
    )


def _plain(x, w, tactic):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_ull_push_gemm as k,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
        rubin_draft_gemm_push as base,
    )

    # The store-only kernel serves 64 rows; past them the push kernel's second N tile of 64 rows
    # is the same tile over the next rows (each output element's sum is its own).
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device="cuda")
    for r0 in range(0, x.shape[0], 64):
        rows = slice(r0, r0 + 64)
        base.run_gemm(x[rows].contiguous(), w, out[rows], k.tactic_of(tactic), pdl=True)
    return out


# (site, n, k, row, col0): the front's latent tiles of TP8 ranks 0 and 5 (rows of the front's
# 512-feature slots), the column-parallel tail's columns of ranks 0 and 1 (feature offsets 0 and
# 896 of the hidden-size rows).
_SHAPES = [
    ("front_r0", 512, 7168, 512, 0),
    ("front_r5", 384, 7168, 512, 0),
    ("cp_r0", 896, 3584, 7168, 0),
    ("cp_r1", 896, 3584, 7168, 896),
]


@pytest.mark.parametrize("m", [1, 8, 17, 32, 33, 54, 66, 72])
@pytest.mark.parametrize("shape", _SHAPES, ids=[s[0] for s in _SHAPES])
def test_push_equals_plain_store(m, shape):
    _, n, k, row, col0 = shape
    g = torch.Generator(device="cuda").manual_seed(m * 131 + n + col0)
    x = torch.randn(m, k, device="cuda", generator=g).bfloat16()
    x[0, :8] = 0.0  # a zero row segment: the product's -0 / +0 go out as +0
    w = (torch.randn(n, k, device="cuda", generator=g) * k**-0.5).bfloat16()
    # Past 32 rows also kernel-N 64: one N tile up to 64 rows, two past them.
    wide = [c._deepest(64, 64, 4, k)] if m > 32 else []
    for tactic in [c.push_tactic(m, n, k), *uref.tactic_candidates(m, n, k)[:2], *wide]:
        peers = _Peers()
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        _push(x, w, out, peers, 3, tactic, row, col0)
        torch.cuda.synchronize()
        got = peers.read(m, 3, row, col0, n)
        expect = c.sanitize_bf16(_plain(x, w, tactic))
        assert torch.equal(got.view(torch.int16), expect.view(torch.int16)), tactic
        assert peers.written() == m * n // 2, "a store outside the slots"
        ref = x.double() @ w.double().t()
        norm = float(ref.abs().max())
        assert float((got.double() - ref).abs().max()) <= 2.0**-8 * norm


def test_refuses_what_it_does_not_serve():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_ull_push_gemm as k,
    )

    t = k.tactic_of(c.push_tactic(8, 896, 3584))
    with pytest.raises(ValueError):
        k.check_push(73, 896, 3584, t, 7168, 0, k.AG, True)
    with pytest.raises(ValueError):
        k.check_push(8, 896, 3584, t, 896, 8, k.SLOT, True)  # past the row
    with pytest.raises(ValueError):
        k.check_push(8, 896, 3584, t, 7168, 4, k.AG, True)  # not 8-aligned
    with pytest.raises(ValueError):
        k.check_push(8, 896, 3584, t, 7168, 0, k.AG, False)  # all-gather rows are one-shot
