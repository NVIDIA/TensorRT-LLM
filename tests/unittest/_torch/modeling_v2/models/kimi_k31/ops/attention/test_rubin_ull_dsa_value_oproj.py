# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's dsa_value_oproj entry (``DsaOprojFused``, served by
``FlatDsaOproj.bind_value``): 16 Helix ranks emulated on one GPU (a real FlatDsaCore per rank,
``_k31_ull_dsa_core_group``) after every rank's dsa_sparse_dcp_decode, and per tested owner MR8's
fused o-projection over a one-device multicast mailbox whose other 15 TP slots are pre-filled.
Against the two kernels it replaces on the same pushed partials -- dsa_dcp_value, then
``OprojFused(x_shared)`` on its rows -- the polled x rows, the MoE input and the updated AttnRes
prefix agree within a rel-L2 band; the partial mailbox, the o-projection slots and the other x
parity are re-armed, and no poll timed out. One cell launches the kernel before any partial
lands (pushed from another stream while it polls)."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_dcp_value import (  # noqa: E501
    dsa_dcp_value,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_sparse_dcp_decode import (  # noqa: E501
    dsa_sparse_dcp_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_ull_dsa_value_oproj import (  # noqa: E501
    dsa_value_oproj as op,
)

__extra_import_path__ = ["../../../..", ".", "../gemm"]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _k31_ull_dsa_core_group import (  # noqa: E402 — needs the path declared above
    LAYER,
    LOCAL,
    PACKED_PITCH,
    PARITY,
    WORLD,
    CoreGroup,
    seed_of,
)
from _multicast import MulticastBuffer  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="DsaOprojFused needs an SM10x GPU",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)

HIDDEN, K, CAP, EPS = 7168, 768, 64, 1e-6
#: Owners whose o-projection runs (different head blocks).
OWNERS = (0, 9)
#: A word of two BF16 halves (+0, -0): an unwritten Lamport word (int32 view).
PAIR_SENT = -(1 << 31)

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"


def _modules():
    fused = importlib.import_module(_FAMILY + ".kernels.attention.rubin_ull_dsa_oproj_fused")
    mr8 = importlib.import_module(_FAMILY + ".kernels.vendor.vllm_mr8")
    trap = importlib.import_module(_FAMILY + ".kernels.common.rubin_ull_spin_trap")
    return fused, mr8.module("cutedsl.oproj.prepared"), mr8.module("cutedsl.common.host"), trap


def _rel(out: torch.Tensor, ref: torch.Tensor) -> float:
    out, ref = out.float(), ref.float()
    return float((out - ref).norm() / ref.norm().clamp_min(1e-30))


def _differ(out: torch.Tensor, ref: torch.Tensor) -> int:
    """BF16 elements whose bits differ, a negative zero counted as +0."""
    a, b = out.view(torch.int16), ref.view(torch.int16)
    neg_zero = -(1 << 15)
    a = torch.where(a == neg_zero, torch.zeros_like(a), a)
    b = torch.where(b == neg_zero, torch.zeros_like(b), b)
    return int((a != b).sum())


class _Oproj:
    """One owner's o-projection operands: BF16 weight and AttnRes weights, the AttnRes rows
    (``blocks`` stored [nb, rows, hidden], read through the [rows, nb, hidden] view as the target
    passes it) and the [16, 64, 7168] Lamport mailbox on a one-device multicast buffer, whose
    slots of the other 15 TP ranks hold fixed partials (rows < ``rows``); ``reset`` restores the
    mailbox, the prefix and the output before a run."""

    def __init__(self, owner: int, rows: int, nb: int, seed: int, host) -> None:
        import cutlass

        g = torch.Generator(device="cuda").manual_seed(seed)

        def randn(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g, device="cuda") * scale).bfloat16()

        self.rows = rows
        self.weight = randn(HIDDEN, K, scale=0.03)
        self.norms = (
            randn(HIDDEN, scale=0.1) + 1,
            randn(HIDDEN, scale=0.05),
            randn(HIDDEN, scale=0.1) + 1,
        )
        self.prefix0 = randn(rows, HIDDEN)
        self.blocks = randn(nb, rows, HIDDEN)
        self.prefix = torch.empty_like(self.prefix0)
        self.out = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
        nbytes = WORLD * CAP * HIDDEN * 2
        self.mc = MulticastBuffer(nbytes)
        self.box = self.mc.uc[:nbytes].view(torch.bfloat16).view(WORLD, CAP, HIDDEN)
        filled = randn(WORLD, rows, HIDDEN, scale=0.5)
        filled = torch.where(filled == 0, torch.zeros_like(filled), filled)  # no -0 half
        self.armed = torch.empty_like(self.box)
        self.armed.view(torch.int32).fill_(PAIR_SENT)
        self.armed[:, :rows] = filled
        self.armed[owner].view(torch.int32).fill_(PAIR_SENT)  # this rank's producer fills it
        to_cute = host.to_cute
        self.prepared = SimpleNamespace(
            wview=to_cute(self.weight),
            head=(to_cute(self.box), cutlass.Int64(self.mc.mc), cutlass.Int32(owner)),
            views=tuple(to_cute(w.reshape(-1)) for w in self.norms),
            eps=(cutlass.Float32(EPS), cutlass.Float32(EPS)),
        )

    def blocks_view(self) -> torch.Tensor:
        return self.blocks.permute(1, 0, 2)

    def reset(self) -> None:
        self.box.copy_(self.armed)
        self.prefix.copy_(self.prefix0)
        self.out.fill_(float("nan"))

    def slots_armed(self) -> bool:
        return bool((self.box[:, : self.rows].view(torch.int32) == PAIR_SENT).all())


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, monkeypatch) -> None:
    rows, kept = cell.spec["rows"], cell.spec["kept"]
    nb, delta, late = cell.spec["nb"], cell.spec["delta"], cell.spec["late"]
    fused, oproj, host, trap = _modules()
    group = CoreGroup(monkeypatch)
    q, steps = group.step(rows, kept, seed_of(cell.why))
    stream = group.stream()
    # Every rank's decode first (the pushes land before any owner polls: one GPU, one stream).
    for core, s in zip(group.cores, steps):
        dsa_sparse_dcp_decode(core, core.kernels[rows][0], PARITY, s.kv, s.indices, s.valid,
                              s.latent, s.lse, stream)  # fmt: skip
    torch.cuda.synchronize()
    device = torch.device("cuda", torch.cuda.current_device())
    index = device.index
    drain, vote = oproj.drain(WORLD, rows, device), oproj.oproj_vote(WORLD, rows, device)
    reference = oproj.compile_oproj(
        "fused", K, rows, WORLD, nb, delta, False, cap=CAP, x_shared=True, drain=drain, vote=vote
    )
    for owner in OWNERS:
        core, s, mail = group.cores[owner], steps[owner], group.mails[owner]
        bench = _Oproj(owner, rows, nb, seed_of(cell.why) + owner, host)
        try:
            pushed = (mail.partial[PARITY].clone(), mail.lse[PARITY].clone())
            ideal = op.reference(None, core, bench.prepared, LAYER, s.out, None, s.gate, None,
                                 None, None, None)  # fmt: skip
            # The replaced pair: DcpValue's gated rows, MR8's fused o-projection on them.
            bench.reset()
            dsa_dcp_value(core, core.kernels[rows][1], PARITY, LAYER, s.valid, s.gate, s.out,
                          s.latent, s.lse, stream)  # fmt: skip
            reference(
                host.to_cute(s.out), bench.prepared.wview, *bench.prepared.head,
                host.dynamic_view(bench.prefix), host.dynamic_view(bench.blocks_view()),
                *bench.prepared.views, host.dynamic_view(bench.out), None, None,
                *bench.prepared.eps, stream,
            )  # fmt: skip
            torch.cuda.synchronize()
            want = (s.out.clone(), bench.out.clone(), bench.prefix.clone())
            assert bench.slots_armed(), f"owner {owner}: OprojFused left a slot written"
            # The fused kernel on the same pushes.
            mail.partial[PARITY].copy_(pushed[0])
            mail.lse[PARITY].copy_(pushed[1])
            bench.reset()
            uv = core.w_uv[LAYER]
            code = fused.compile_value_oproj(
                rows, nb, delta, (tuple(uv.shape), tuple(uv.stride()), uv.dtype), PACKED_PITCH,
                WORLD, LOCAL, core.cap, drain, vote, trap.record_address(), index,
            )  # fmt: skip
            x_box = fused.arm(torch.empty((2, fused.ROWS, K), dtype=torch.bfloat16, device="cuda"))
            x_box[1 - PARITY].normal_()  # written by an earlier call: the kernel re-arms it
            x, x_other = x_box[PARITY, :rows], x_box[1 - PARITY]
            args = (code, core, bench.prepared, LAYER, x, x_other, s.gate, bench.prefix,
                    bench.blocks_view(), bench.out, stream)  # fmt: skip
            if not late:
                with validating(op):
                    op(*args)
            else:
                # Nothing pushed yet: the kernel polls, the pushes land from another stream.
                mail.partial[PARITY].view(torch.int32).fill_(group.dcp.PAIR_SENT - (1 << 32))
                mail.lse[PARITY].view(torch.int32).fill_(-1)
                host.resolve_kernel(code, index)  # loaded now, not at its launch
                torch.cuda._sleep(1)  # _sleep's module loaded before the polling kernel
                torch.cuda.synchronize()
                main = torch.cuda.current_stream()
                side = torch.cuda.Stream()
                side.wait_stream(main)
                # The pushes are enqueued first: no launch may follow the polling kernel on the
                # host (a lazily loaded module would wait for the resident poller: a deadlock).
                with torch.cuda.stream(side):
                    torch.cuda._sleep(20_000_000)  # about 10 ms: the kernel polls well before
                    mail.lse[PARITY].copy_(pushed[1])
                    mail.partial[PARITY].copy_(pushed[0])
                with validating(op):
                    op(*args)
            torch.cuda.synchronize()
            got = (x, bench.out, bench.prefix)
            # Same operations and rounding points as the pair (module doc): x may differ only in
            # the sign of a zero; the band admits an FP32 summation-order change of the merge or
            # projection (a BF16 ulp on a few elements), not a misplaced head, row or column
            # (rel-L2 ~1).
            for name, a, b, band in zip(("x", "out", "prefix"), got, want, (2**-8, 2**-7, 2**-7)):
                rel = _rel(a, b)
                assert rel <= band, (
                    f"owner {owner} {name}: rel L2 {rel:.3e} vs the pair "
                    f"({_differ(a, b)} of {b.numel()} BF16 values differ)"
                )
            # The merge of the pushed BF16 partials in FP32 against torch (dsa_dcp_value's band).
            assert_within_ulp(x, ideal, element_ulp=4.0, rms_ulp=1.0)
            for b, n in enumerate(kept):
                if n == 0:
                    assert bool((x[b].float() == 0).all()), f"owner {owner} row {b}: not zero"
            assert mail.armed(group.dcp), f"owner {owner}: partial mailbox not re-armed"
            assert bench.slots_armed(), f"owner {owner}: an o-projection slot left written"
            assert bool((x_other.view(torch.int32) == PAIR_SENT).all()), (
                f"owner {owner}: the other x parity not re-armed"
            )
            assert int(core.mla_query.spins.sum()) == 0, f"owner {owner}: a poll timed out"
        finally:
            bench.mc.close()
