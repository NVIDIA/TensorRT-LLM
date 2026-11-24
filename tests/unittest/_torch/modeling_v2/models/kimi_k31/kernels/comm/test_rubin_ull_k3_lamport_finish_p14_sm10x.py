# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On SM10x: the finish kernel of ``kernels/comm/rubin_ull_k3_lamport_ar_pdlfix.py`` does not
store into ``out`` under a grid still running ahead of it (one GPU).

The reproducer: a grid releases its dependents at entry, spins, then reads row 0
of ``out`` late, while the finish (every row it polls already in place) runs behind it; the reader
must still see the bytes it started with, and the finish must still deliver its result and advance
the rotation state.

The workspace is plain device memory seen by one rank (no multicast). The CPU half (the source,
and the kernels' program order on stand-ins) is ``test_rubin_ull_k3_lamport_finish_p14.py``.
"""

import importlib

import pytest
import torch

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm"
SHARED, P14 = "rubin_ull_k3_lamport_ar", "rubin_ull_k3_lamport_ar_pdlfix"


def _gpu_ok() -> bool:
    try:
        from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
    except Exception:  # noqa: BLE001 (a host without the engine's build)
        return False
    return (
        IS_CUTLASS_DSL_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
    )


pytestmark = pytest.mark.skipif(not _gpu_ok(), reason="needs an SM10x GPU and the CuTe DSL")

HIDDEN = 7168
COLS = HIDDEN // 8 // 128
SPIN_NS = 2_000_000
GUARD = 1024.0
BYTES_PER_BUFFER = 4 << 20


def _module(name: str):
    return importlib.import_module(f"{_PKG}.{name}")


def _read_after_release(lar):
    """The reproducer's early-releasing grid: ``COLS`` CTAs of 128 threads release their
    dependents at entry, spin ``ns`` ns on the global timer, then copy row 0 of ``guarded`` into
    ``observed``, 16 bytes a thread, read past L1."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    @dsl_user_op
    def globaltimer(*, loc=None, ip=None):
        return Int64(
            llvm.inline_asm(
                T.i64(),
                [],
                "mov.u64 $0, %globaltimer;",
                "=l",
                has_side_effects=True,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    class ReadAfterRelease:
        @cute.jit
        def __call__(self, guarded: Int64, observed: Int64, ns: Int64, stream: cuda.CUstream):
            self.kernel(guarded, observed, ns).launch(
                grid=(COLS, 1, 1), block=(128, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(self, guarded: Int64, observed: Int64, ns: Int64):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            lar.griddepcontrol_launch_dependents()
            if tidx == 0:
                start = globaltimer()
                now = start
                while now - start < ns:
                    now = globaltimer()
            cute.arch.barrier()
            offset = Int64(bidx * 128 + tidx) * Int64(16)
            w0, w1, w2, w3 = lar.ld_volatile_v4(guarded + offset)
            lar.st_v4(observed + offset, w0, w1, w2, w3)

    return ReadAfterRelease()


def _bf16_rows(n: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(n, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    x[x == 0] = 0  # no -0: a Lamport row never holds one
    return x


def _early_stores(world, rank, one_shot, ag, rows) -> list:
    """The reproducer on the pdlfix module's ``run_lamport_finish`` (the shared module's workspace
    and helpers): the early words seen by the reader per attempt, after checking the result and the
    rotation state."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int64

    lar, finish = _module(SHARED), _module(P14)
    gen = torch.Generator(device="cuda").manual_seed(16 * rows + world)
    uc = torch.empty(lar.NUM_BUFFERS * BYTES_PER_BUFFER // 2, dtype=torch.bfloat16, device="cuda")
    flags = torch.zeros(9, dtype=torch.int32, device="cuda")
    ws = lar.LamportWorkspace(
        world=world,
        rank=rank,
        uc_local=uc.data_ptr(),
        mc_base=0,
        peer_ptrs=torch.full((world,), uc.data_ptr(), dtype=torch.int64, device="cuda"),
        flags=flags,
        bytes_per_buffer=BYTES_PER_BUFFER,
    )
    if one_shot:
        slots = _bf16_rows(rows * world, gen)
        y = _bf16_rows(rows, gen) if ag else None
        call = torch.cat([slots, y]) if ag else slots
        start = 0
        want = (y.float() + slots.float()).to(torch.bfloat16) if ag else slots
    else:
        call = _bf16_rows(rows, gen)
        start = BYTES_PER_BUFFER // 2 // 2
        want = call
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    observed = torch.empty(HIDDEN // 2, dtype=torch.int32, device="cuda")
    guard = torch.full((HIDDEN,), GUARD, dtype=torch.bfloat16, device="cuda").view(torch.int32)

    def arm():
        uc.view(torch.int32).fill_(lar.SENTINEL)
        uc[start : start + call.numel()].copy_(call.flatten())
        flags.zero_()
        flags[2] = BYTES_PER_BUFFER
        out.fill_(GUARD)
        observed.fill_(-1)
        torch.cuda.synchronize()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    reader = cute.compile(_read_after_release(lar), Int64(0), Int64(0), Int64(0), stream)

    def run():
        reader(Int64(out.data_ptr()), Int64(observed.data_ptr()), Int64(SPIN_NS), stream)
        finish.run_lamport_finish(ws, out, one_shot, use_pdl=True, wait_first=False, ag=ag)
        torch.cuda.synchronize()

    arm()
    run()  # module loads
    early = []
    for attempt in range(3):
        arm()
        run()
        early.append(int((observed != guard).sum()))
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), attempt
        assert flags[:2].tolist() == [1, 0] and int(flags[8]) == 0, flags.tolist()
    return early


CELLS = [
    pytest.param(1, 0, True, True, 4, id="one-shot, all-gather rows, 4 rows"),
    pytest.param(1, 0, True, False, 8, id="one-shot, 8 rows (L0 dense, the drafter)"),
    pytest.param(16, 15, False, False, 7, id="two-shot gather, rows another rank owns"),
]


@pytest.mark.parametrize("world, rank, one_shot, ag, rows", CELLS)
@torch.inference_mode()
def test_the_p14_finish_does_not_store_under_a_running_grid(world, rank, one_shot, ag, rows):
    early = _early_stores(world, rank, one_shot, ag, rows)
    assert early == [0, 0, 0], f"words of out's row 0 changed under a running grid: {early}"
