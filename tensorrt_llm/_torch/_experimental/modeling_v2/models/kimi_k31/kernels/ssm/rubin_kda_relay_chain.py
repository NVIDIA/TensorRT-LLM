# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The owner chains of the cep sequence-split KDA in one launch (CuTe DSL).

One CTA per (request, head, ``BV`` value rows), one thread per state column ``k``: the tile's state
starts from the request's pool row (cached) or zero and walks the request's segments in order. A
fresh first segment sets the state to its ``N``; any other segment hands its start state to the
segment's rank (BF16, round to nearest even, into the send buffer of the all-to-all C), stores it
as the snapshot when the segment begins at the snapshot point, and moves on to ``bf16(X) @ M + N``:
each thread accumulates its column from ``N`` over ``j = 0 .. 127`` in FP32 (BF16 x BF16 products are
exact in FP32), reading the rounded rows of ``X`` from shared memory and column ``k`` of ``M``
(coalesced over the CTA). The end state goes to the request's pool row. Every value row is an
independent chain (``X @ M`` acts on rows).

BF16 values travel as their int16 bit patterns (``send`` and ``M``).
"""

from typing import Optional

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

# Columns of a chain step: kind (0: X = N, 1: X = X @ M + N), N's (rank, slot), M's (rank, slot),
# the snapshot pool row (-1: none).
STEP_COLS = 6
_K = 128  # state columns (head dim), one thread each
_BV = 8  # value rows per CTA

if IS_CUTLASS_DSL_AVAILABLE:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import from_dlpack

    @cute.jit
    def _bf16_bits(x: cutlass.Float32) -> Int32:
        """The BF16 bits (low 16) of ``x``, rounded to nearest even."""
        u = x.bitcast(Int32)
        return (u + (((u >> 16) & 1) + 0x7FFF)) >> 16

    class _RelayChainKernel:
        def __init__(self, heads: int):
            self.heads = heads

            @cute.struct
            class SharedStorage:
                xb: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _BV * _K], 16]

            self.shared_storage = SharedStorage

        @cute.jit
        def __call__(
            self,
            steps: cute.Tensor,
            counts: cute.Tensor,
            init: cute.Tensor,
            slots: cute.Tensor,
            nf: cute.Tensor,
            mb: cute.Tensor,
            pool: cute.Tensor,
            send: cute.Tensor,
            row_words: Int64,
            m_words: Int64,
            send_slots: Int64,
            max_steps: Int32,
            num: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(
                steps,
                counts,
                init,
                slots,
                nf,
                mb,
                pool,
                send,
                row_words,
                m_words,
                send_slots,
                max_steps,
            ).launch(
                grid=[num, self.heads, _K // _BV],
                block=[_K, 1, 1],
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
            )

        @cute.kernel
        def kernel(
            self,
            steps: cute.Tensor,
            counts: cute.Tensor,
            init: cute.Tensor,
            slots: cute.Tensor,
            nf: cute.Tensor,
            mb: cute.Tensor,
            pool: cute.Tensor,
            send: cute.Tensor,
            row_words: Int64,
            m_words: Int64,
            send_slots: Int64,
            max_steps: Int32,
        ):
            k, _, _ = cute.arch.thread_idx()
            r, h, vb = cute.arch.block_idx()
            st = utils.SmemAllocator().allocate(self.shared_storage)
            sx = st.xb.get_tensor(cute.make_layout((_BV * _K,)))
            elems = Int64(self.heads * _K * _K)
            # This thread's element of value row v of the tile: tile + v * K.
            tile = Int64(h) * Int64(_K * _K) + Int64(vb * _BV) * Int64(_K) + Int64(k)
            slot = Int64(slots[r])
            acc = cute.make_rmem_tensor(cute.make_layout((_BV,)), cutlass.Float32)
            for v in cutlass.range_constexpr(_BV):
                acc[v] = cutlass.Float32(0.0)
            if init[r] != 0:
                for v in cutlass.range_constexpr(_BV):
                    acc[v] = pool[slot, tile + Int64(v * _K)]
            for d in cutlass.range(counts[r], unroll=1):
                base = (r * max_steps + d) * STEP_COLS
                kind = steps[base]
                n0 = Int64(steps[base + 1]) * row_words + Int64(steps[base + 2]) * elems + tile
                if kind == 0:
                    for v in cutlass.range_constexpr(_BV):
                        acc[v] = nf[n0 + Int64(v * _K)]
                else:
                    mq = Int64(steps[base + 3])
                    ml = Int64(steps[base + 4])
                    snap = steps[base + 5]
                    s0 = (mq * send_slots + ml) * elems + tile
                    for v in cutlass.range_constexpr(_BV):
                        bits = _bf16_bits(acc[v])
                        send[s0 + Int64(v * _K)] = cutlass.Int16(bits)
                        sx[v * _K + k] = (bits << 16).bitcast(cutlass.Float32)
                    if snap >= 0:
                        for v in cutlass.range_constexpr(_BV):
                            pool[Int64(snap), tile + Int64(v * _K)] = acc[v]
                    cute.arch.barrier()
                    for v in cutlass.range_constexpr(_BV):
                        acc[v] = nf[n0 + Int64(v * _K)]
                    m0 = mq * Int64(2) * row_words + Int64(2) * m_words + ml * elems
                    m0 = m0 + Int64(h) * Int64(_K * _K) + Int64(k)
                    for j in cutlass.range(_K, unroll=16):
                        mjk = (Int32(mb[m0 + Int64(j * _K)]) << 16).bitcast(cutlass.Float32)
                        for v in cutlass.range_constexpr(_BV):
                            acc[v] = acc[v] + sx[v * _K + j] * mjk
                    cute.arch.barrier()
            for v in cutlass.range_constexpr(_BV):
                pool[slot, tile + Int64(v * _K)] = acc[v]

    def _arg(t: torch.Tensor):
        return from_dlpack(t, assumed_align=4).mark_layout_dynamic(t.dim() - 1)


_compiled: dict = {}


def relay_chain(
    steps: torch.Tensor,
    counts: torch.Tensor,
    init: torch.Tensor,
    slots: torch.Tensor,
    recv: torch.Tensor,
    m_words: int,
    heads: int,
    pool: torch.Tensor,
    send: Optional[torch.Tensor],
) -> None:
    """Run every request's chain (``steps`` ``[R, S, 6]`` int32, ``counts`` / ``init`` ``[R]``
    int32, ``slots`` ``[R]`` int64) over ``recv`` (``[world, CB]`` FP32 words: N at slot ``l`` of row
    ``q`` at word ``l * elems``, M in BF16 from word ``m_words``), storing into the FP32 ``pool``
    (``[slots, heads, 128, 128]``, any slot stride) and the BF16 ``send``
    (``[world, ms, heads, 128, 128]``)."""
    assert IS_CUTLASS_DSL_AVAILABLE, "needs the CUTLASS DSL"
    num, max_steps = steps.shape[0], steps.shape[1]
    if num == 0:
        return
    assert pool.dtype == torch.float32 and pool.stride()[1:] == (_K * _K, _K, 1)
    assert recv.dtype == torch.float32 and recv.is_contiguous()
    flat_send = (
        send.view(torch.int16).reshape(-1)
        if send is not None
        else torch.empty(1, dtype=torch.int16, device=recv.device)
    )
    args = [
        _arg(steps.reshape(-1)),
        _arg(counts),
        _arg(init),
        _arg(slots),
        _arg(recv.reshape(-1)),
        _arg(recv.view(torch.int16).reshape(-1)),
        _arg(pool.flatten(1)),
        _arg(flat_send),
    ]
    scalars = (
        recv.shape[1],
        m_words,
        send.shape[1] if send is not None else 0,
        max_steps,
        num,
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if heads not in _compiled:
        _compiled[heads] = cute.compile(_RelayChainKernel(heads), *args, *scalars, stream)
    _compiled[heads](*args, *scalars, stream)
