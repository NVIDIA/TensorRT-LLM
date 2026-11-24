# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pushed all-reduce's finish kernel must not store into ``out`` before the grids it overlaps
have completed (one GPU).

``K3LamportFinishKernel`` (``kernels/comm/rubin_lamport_ar.py``) is a programmatic dependent
launch whose workers poll their Lamport rows before ``griddepcontrol.wait``. ``out`` is a fresh
caching-allocator block, which stream order may hand out while an earlier grid that released its
dependents still reads the same memory: helix8's ``moe_add`` finishes into
``torch.empty_like(prefix)`` behind the cp tail's push GEMM, which loads ``latent`` after its
release. The test runs such a grid (it releases its dependents, spins, then reads row 0 of
``out``) and launches the finish next with every row it polls already in place: the grid must
still read the bytes it started with, and the finish must still deliver its result and advance
the rotation state.

The workspace is plain device memory seen by one rank (no multicast): the one-shot transport at
one rank, with and without the all-gather rows, and the two-shot gather on the last of eight
ranks, for rows another rank owns (it polls the broadcast stage and copies).
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

HIDDEN = 7168
#: The finish's CTAs per row (one 16-byte vector per thread, 128 threads); the reader's grid.
COLS = HIDDEN // 8 // 128
#: The early-releasing grid reads ``out`` this long after it starts.
SPIN_NS = 2_000_000
#: ``out``'s contents before the call; no input or result comes near it.
GUARD = 1024.0
BYTES_PER_BUFFER = 4 << 20

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM10x GPU and the CuTe DSL",
)

_LAR = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.comm.rubin_lamport_ar"


def _lar():
    import importlib

    return importlib.import_module(_LAR)


def _read_after_release():
    """``COLS`` CTAs of 128 threads: release the dependents at entry, spin ``ns`` nanoseconds on
    the global timer, then copy row 0 of ``guarded`` (BF16 ``[m, HIDDEN]``) into ``observed``,
    16 bytes per thread, read past L1."""
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import T, dsl_user_op

    lar = _lar()

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
    x[x == 0] = 0  # no -0: a Lamport row never holds one, so no word is the empty-slot sentinel
    return x


@pytest.mark.parametrize(
    "world, rank, one_shot, ag, rows",
    [
        pytest.param(1, 0, True, True, 4, id="one-shot, all-gather rows (moe_add), 4 rows"),
        pytest.param(1, 0, True, True, 54, id="one-shot, all-gather rows (moe_add), 54 rows"),
        pytest.param(1, 0, True, False, 8, id="one-shot, 8 rows"),
        pytest.param(8, 7, False, False, 7, id="two-shot gather, rows another rank owns"),
    ],
)
@torch.inference_mode()
def test_finish_waits_before_storing_into_reused_memory(world, rank, one_shot, ag, rows):
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int64

    lar = _lar()
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
        # Buffer 0: slot (t, r) is row t * world + r, the all-gather row of t row rows * world + t.
        slots = _bf16_rows(rows * world, gen)
        y = _bf16_rows(rows, gen) if ag else None
        call = torch.cat([slots, y]) if ag else slots
        start = 0
        want = (y.float() + slots.float()).to(torch.bfloat16) if ag else slots
    else:
        # Buffer 0's broadcast stage (its second half): the rows their owners multicast.
        call = _bf16_rows(rows, gen)
        start = BYTES_PER_BUFFER // 2 // 2
        want = call
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    observed = torch.empty(HIDDEN // 2, dtype=torch.int32, device="cuda")
    guard_words = torch.full((HIDDEN,), GUARD, dtype=torch.bfloat16, device="cuda").view(
        torch.int32
    )

    def arm():
        """The call's rows in place, every other word the sentinel, the state of a fresh
        workspace (buffer 0 current, nothing to re-arm), ``out`` holding the guard."""
        uc.view(torch.int32).fill_(lar.SENTINEL)
        uc[start : start + call.numel()].copy_(call.flatten())
        flags.zero_()
        flags[2] = BYTES_PER_BUFFER
        out.fill_(GUARD)
        observed.fill_(-1)
        torch.cuda.synchronize()

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    reader = cute.compile(_read_after_release(), Int64(0), Int64(0), Int64(0), stream)

    def run():
        reader(Int64(out.data_ptr()), Int64(observed.data_ptr()), Int64(SPIN_NS), stream)
        lar.run_lamport_finish(ws, out, one_shot, use_pdl=True, wait_first=False, ag=ag)
        torch.cuda.synchronize()

    # Warm-up: the first launch of each kernel loads its module, which would serialize them.
    arm()
    run()
    for attempt in range(3):
        arm()
        run()
        early = int((observed != guard_words).sum())
        assert early == 0, (
            f"attempt {attempt}: {early}/{observed.numel()} words of out's row 0 changed under a "
            f"running grid that released its dependents: the finish stored before its "
            f"griddepcontrol.wait"
        )
        assert torch.equal(out.view(torch.int16), want.view(torch.int16)), f"attempt {attempt}"
        # The updater advanced the rotation (buffer 1 next, buffer 0 dirty) and reset the count.
        assert flags[:2].tolist() == [1, 0] and int(flags[8]) == 0, flags.tolist()
