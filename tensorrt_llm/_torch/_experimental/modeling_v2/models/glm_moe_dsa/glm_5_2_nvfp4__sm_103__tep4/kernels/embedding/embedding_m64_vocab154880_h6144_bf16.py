# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 replicated Embedding replacement for B300.

TRT-LLM contract
----------------
The GLM-5.2 build constructs this module without a TP mapping, so
each rank owns the complete table and no embedding collective follows it::

    input_ids: contiguous int32 [M]
    weight:    contiguous BF16  [154880, 6144]
    output:    contiguous BF16  [M, 6144]

The checkpoint fixes vocabulary size, hidden size, table dtype and layout.
``M`` is runtime metadata: decode CUDA Graph buckets are 1, 2, 4, 8, 16, 24,
32 and 64, while packed context calls range through 8192 tokens.

The staged TP1 ``Embedding.forward`` materializes nine GPU kernels to compute
the range mask, safe IDs, ``F.embedding`` and invalid-ID zero fill. This file
fuses those operations into one CuTe DSL kernel. Valid IDs copy one table row;
negative IDs and IDs at or above 154880 write an all-zero row.

Production schedules
--------------------
The adapter selects precompiled geometry from tensor shape only. ``M`` remains
symbolic in every compiled callable, so there is no per-M JIT compilation or
host/device synchronization::

    M              algorithm  threads  CTA/row  rows/CTA  BF16/load
    1,2            direct        128       2         1          8
    4,16           direct         96       4         1         16
    8              direct        192       2         1         16
    24             direct        128       1         1         16
    32             direct        128       3         1         16
    33..4096       direct        256       1         4         16
    4097..8192     bulk           64       -         2     whole row

The direct path emits coalesced 128- or 256-bit vector global loads and
stores. The large-M bulk path assigns one warp to each 12-KiB row and copies
global -> shared -> global with ``cp.async.bulk``. PDL is enabled by default:
kernels wait immediately before reading ``input_ids`` and launch dependents only after all output writes
are visible at the CTA synchronization point.

B300 performance
----------------
With CUDA Graph replay and PDL enabled, this kernel is within 2% of the single
``F.embedding`` kernel at most ``M`` and 1.1x to 1.7x faster at ``M`` = 8, 16
and 4096..8192. That baseline handles valid IDs only, so the comparison is
conservative: this kernel also implements the invalid-ID zero fill.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.experimental import primitives as prims

VOCAB_SIZE = 154880
HIDDEN_SIZE = 6144
MAX_PREFILL_M = 8192
ZERO_VECTOR_ELEMENTS = 8
ROW_BYTES = HIDDEN_SIZE * 2
BULK_ROWS_PER_CTA = 2

CompiledKernel = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]

_LDG_COMPILE_CACHE: dict[tuple[int, int, int, int, bool], CompiledKernel] = {}
_BULK_COMPILE_CACHE: dict[bool, CompiledKernel] = {}


@dataclass(frozen=True)
class _LdgSchedule:
    """Compile-time geometry for one direct-load specialization."""

    threads: int
    ctas_per_row: int
    rows_per_cta: int
    vector_elements: int


_LDG_SCHEDULES: dict[str, _LdgSchedule] = {
    "decode_m1_m2": _LdgSchedule(128, 2, 1, 8),
    "decode_m4_m16": _LdgSchedule(96, 4, 1, 16),
    "decode_m8": _LdgSchedule(192, 2, 1, 16),
    "decode_m24": _LdgSchedule(128, 1, 1, 16),
    "decode_m32": _LdgSchedule(128, 3, 1, 16),
    "context_through_m4096": _LdgSchedule(256, 1, 4, 16),
}


@cute.kernel
def ldg_kernel(
    input_ids: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
    active_m: cutlass.Int32,
    threads: cutlass.Constexpr[int],
    ctas_per_row: cutlass.Constexpr[int],
    rows_per_cta: cutlass.Constexpr[int],
    vector_elements: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    token_id_smem = cutlass.Array(
        cutlass.Int32,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=4,
    )
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // cutlass.Int32(32)
    lane = tidx % cutlass.Int32(32)
    row_cta = bidx % cutlass.Int32(ctas_per_row)
    row_group = bidx // cutlass.Int32(ctas_per_row)
    warps_per_row: cutlass.Constexpr = (threads // 32) // rows_per_cta
    row_in_cta = warp // cutlass.Int32(warps_per_row)
    warp_in_row = warp % cutlass.Int32(warps_per_row)
    row = row_group * cutlass.Int32(rows_per_cta) + row_in_cta

    # input_ids is produced by the graph predecessor; table weights are
    # persistent, but the row address cannot be known before the ID is ready.
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    token_id = cutlass.Int32(-1)
    row_is_active = row < active_m
    if cutlass.const_expr(rows_per_cta == 1):
        # The grid is exact for the single-row schedule.  One ID load plus a
        # CTA broadcast is cheaper than four duplicate warp loads in decode.
        row_is_active = cutlass.Boolean(True)
        if tidx == 0:
            token_id_smem.subview(0).store(
                prims.load_ext(
                    input_ids.iterator.raw_ptr() + row,
                    dtype=cutlass.Int32,
                )
            )
        prims.barrier_cta_sync(0)
        token_id = token_id_smem.load()
    else:
        # Every warp works on exactly one row.  Lane 0 loads the token ID and
        # broadcasts it without synchronizing unrelated rows in the CTA.
        if row_is_active and lane == 0:
            token_id = prims.load_ext(
                input_ids.iterator.raw_ptr() + row,
                dtype=cutlass.Int32,
            )
        token_id = cute.arch.shuffle_sync(token_id, 0)

    if row_is_active:
        is_valid = (token_id >= cutlass.Int32(0)) & (token_id < cutlass.Int32(VOCAB_SIZE))
        threads_per_row: cutlass.Constexpr = warps_per_row * 32
        logical_thread = (
            row_cta * cutlass.Int32(threads_per_row) + warp_in_row * cutlass.Int32(32) + lane
        )
        weight_ptr = weight.iterator.raw_ptr()
        output_ptr = output.iterator.raw_ptr()

        vector_index = logical_thread
        while vector_index < cutlass.Int32(HIDDEN_SIZE // vector_elements):
            col = vector_index * cutlass.Int32(vector_elements)
            output_address = output_ptr + row * cutlass.Int32(HIDDEN_SIZE) + col
            if is_valid:
                weight_address = weight_ptr + token_id * cutlass.Int32(HIDDEN_SIZE) + col
                if cutlass.const_expr(vector_elements == 16):
                    # Express 16 BF16 elements as eight 32-bit words because
                    # the primitive vector count is capped at eight lanes.
                    values_256b = prims.load_ext(
                        weight_address,
                        dtype=cutlass.Int32,
                        count=8,
                    )
                    prims.store_ext(values_256b, output_address)
                else:
                    values_128b = prims.load_ext(
                        weight_address,
                        dtype=cutlass.Int16,
                        count=8,
                    )
                    prims.store_ext(values_128b, output_address)
            else:
                # Invalid IDs are warp-uniform. Int64 stores avoid constructing
                # a register vector solely for zeros.
                for zero_store in cutlass.range_constexpr(vector_elements // 4):
                    prims.store_ext(
                        cutlass.Int64(0),
                        output_address + cutlass.Int32(zero_store * 4),
                    )
            vector_index += cutlass.Int32(threads_per_row * ctas_per_row)

    if cutlass.const_expr(use_pdl):
        prims.barrier_cta_sync(0)
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def ldg_launch(
    input_ids: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
    active_m: cutlass.Int32,
    threads: cutlass.Constexpr[int],
    ctas_per_row: cutlass.Constexpr[int],
    rows_per_cta: cutlass.Constexpr[int],
    vector_elements: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    grid_rows = active_m
    if cutlass.const_expr(rows_per_cta > 1):
        grid_rows = (active_m + rows_per_cta - 1) // rows_per_cta
    ldg_kernel(
        input_ids,
        weight,
        output,
        active_m,
        threads,
        ctas_per_row,
        rows_per_cta,
        vector_elements,
        use_pdl,
    ).launch(
        grid=(grid_rows * ctas_per_row, 1, 1),
        block=(threads, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _compile_ldg(
    *,
    threads: int,
    ctas_per_row: int,
    rows_per_cta: int,
    vector_elements: int,
    use_pdl: bool,
) -> CompiledKernel:
    """Compile and cache one production direct schedule with symbolic M."""
    key = (threads, ctas_per_row, rows_per_cta, vector_elements, bool(use_pdl))
    if key in _LDG_COMPILE_CACHE:
        return _LDG_COMPILE_CACHE[key]

    make_fake = cute.runtime.make_fake_compact_tensor
    sym_m = cute.sym_int32(symbol="glm_embedding_active_m")
    ids_fake = make_fake(
        cutlass.Int32,
        (sym_m,),
        stride_order=(0,),
        assumed_align=16,
    )
    weight_fake = make_fake(
        cutlass.BFloat16,
        (VOCAB_SIZE, HIDDEN_SIZE),
        stride_order=(1, 0),
        assumed_align=128,
    )
    output_fake = make_fake(
        cutlass.BFloat16,
        (sym_m, HIDDEN_SIZE),
        stride_order=(1, 0),
        assumed_align=128,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    raw = cute.compile(
        ldg_launch,
        ids_fake,
        weight_fake,
        output_fake,
        fake_stream,
        cutlass.Int32(0),
        threads=threads,
        ctas_per_row=ctas_per_row,
        rows_per_cta=rows_per_cta,
        vector_elements=vector_elements,
        use_pdl=use_pdl,
        options="--enable-tvm-ffi",
    )

    def compiled(
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> object:
        return raw(input_ids, weight, output, input_ids.numel())

    _LDG_COMPILE_CACHE[key] = compiled
    return compiled


@cute.kernel
def bulk_kernel(
    input_ids: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
    active_m: cutlass.Int32,
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """Copy two independent embedding rows through SMEM, one warp per row."""
    row_smem = cutlass.Array(
        cutlass.Int16,
        HIDDEN_SIZE * BULK_ROWS_PER_CTA,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    full_bar = cutlass.Array(
        cutlass.Int64,
        BULK_ROWS_PER_CTA,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    warp = tidx // cutlass.Int32(32)
    lane = tidx % cutlass.Int32(32)
    row = bidx * cutlass.Int32(BULK_ROWS_PER_CTA) + warp
    row_smem_offset = warp * cutlass.Int32(HIDDEN_SIZE)
    row_bar = full_bar.subview(warp)

    token_id = cutlass.Int32(-1)
    row_is_active = row < active_m
    if lane == 0:
        prims.mbarrier_init(row_bar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    # Barrier initialization is independent of predecessors. Wait only before
    # reading input_ids so the setup can overlap a programmatic predecessor.
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
    if row_is_active and lane == 0:
        token_id = prims.load_ext(
            input_ids.iterator.raw_ptr() + row,
            dtype=cutlass.Int32,
        )
    token_id = cute.arch.shuffle_sync(token_id, 0)

    if row_is_active:
        is_valid = (token_id >= cutlass.Int32(0)) & (token_id < cutlass.Int32(VOCAB_SIZE))
        output_address = output.iterator.raw_ptr() + row * cutlass.Int32(HIDDEN_SIZE)
        if is_valid:
            if lane == 0:
                prims.mbarrier_arrive_expect_tx(row_bar, ROW_BYTES)
                prims.cp_async_bulk_shared_cluster_global(
                    row_smem.subview(row_smem_offset),
                    weight.iterator.raw_ptr() + token_id * cutlass.Int32(HIDDEN_SIZE),
                    row_bar,
                    ROW_BYTES,
                )
            while not prims.mbarrier_try_wait_parity(
                row_bar,
                0,
                time_limit=10_000_000,
            ):
                pass
            if lane == 0:
                prims.cp_async_bulk_global_shared_cta(
                    output_address,
                    row_smem.subview(row_smem_offset),
                    ROW_BYTES,
                )
                prims.cp_async_bulk_commit_group()
                prims.cp_async_bulk_wait_group(0)
        else:
            vector_index = lane
            while vector_index < cutlass.Int32(HIDDEN_SIZE // ZERO_VECTOR_ELEMENTS):
                address = output_address + vector_index * cutlass.Int32(ZERO_VECTOR_ELEMENTS)
                prims.store_ext(cutlass.Int64(0), address)
                prims.store_ext(cutlass.Int64(0), address + cutlass.Int32(4))
                vector_index += cutlass.Int32(32)

    prims.barrier_cta_sync(0)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def bulk_launch(
    input_ids: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
    stream: cuda.CUstream,
    active_m: cutlass.Int32,
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    bulk_kernel(
        input_ids,
        weight,
        output,
        active_m,
        use_pdl,
    ).launch(
        grid=((active_m + BULK_ROWS_PER_CTA - 1) // BULK_ROWS_PER_CTA, 1, 1),
        block=(32 * BULK_ROWS_PER_CTA, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _compile_bulk(*, use_pdl: bool = True) -> CompiledKernel:
    """Compile the production whole-row bulk-copy schedule with symbolic M."""
    key = bool(use_pdl)
    if key in _BULK_COMPILE_CACHE:
        return _BULK_COMPILE_CACHE[key]

    make_fake = cute.runtime.make_fake_compact_tensor
    sym_m = cute.sym_int32(symbol="glm_embedding_bulk_active_m")
    ids_fake = make_fake(
        cutlass.Int32,
        (sym_m,),
        stride_order=(0,),
        assumed_align=16,
    )
    weight_fake = make_fake(
        cutlass.BFloat16,
        (VOCAB_SIZE, HIDDEN_SIZE),
        stride_order=(1, 0),
        assumed_align=128,
    )
    output_fake = make_fake(
        cutlass.BFloat16,
        (sym_m, HIDDEN_SIZE),
        stride_order=(1, 0),
        assumed_align=128,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    raw = cute.compile(
        bulk_launch,
        ids_fake,
        weight_fake,
        output_fake,
        fake_stream,
        cutlass.Int32(0),
        use_pdl=use_pdl,
        options="--enable-tvm-ffi",
    )

    def compiled(
        input_ids: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
    ) -> object:
        return raw(input_ids, weight, output, input_ids.numel())

    _BULK_COMPILE_CACHE[key] = compiled
    return compiled


class GLMEmbeddingAdapter(torch.nn.Module):
    """Match the staged TRT-LLM ``Embedding.forward(input_ids)`` ABI.

    Compilation and schedule construction happen during module setup, before
    CUDA Graph capture.  ``forward`` only reads M from tensor metadata,
    allocates the returned tensor, selects a precompiled schedule, and launches
    the three-pointer device kernel.  No device value is read on the host.

    The schedule thresholds are specific to the GLM-5.2 B300 workload.  Small
    graph buckets split rows across CTAs, medium context batches group rows in
    bulk-copy CTAs, and larger batches switch between the direct and bulk-copy
    mappings.
    """

    def __init__(self, weight: torch.Tensor, *, use_pdl: bool = True) -> None:
        super().__init__()
        if (
            weight.shape != (VOCAB_SIZE, HIDDEN_SIZE)
            or weight.dtype != torch.bfloat16
            or not weight.is_cuda
            or not weight.is_contiguous()
        ):
            raise ValueError(
                "GLM embedding weight must be contiguous CUDA BF16 "
                f"[{VOCAB_SIZE},{HIDDEN_SIZE}], got shape={weight.shape} "
                f"dtype={weight.dtype} device={weight.device} "
                f"contiguous={weight.is_contiguous()}"
            )
        self.weight = weight
        self._use_pdl = bool(use_pdl)
        self._kernels = {
            name: _compile_ldg(
                threads=schedule.threads,
                ctas_per_row=schedule.ctas_per_row,
                rows_per_cta=schedule.rows_per_cta,
                vector_elements=schedule.vector_elements,
                use_pdl=self._use_pdl,
            )
            for name, schedule in _LDG_SCHEDULES.items()
        }
        self._kernels["context_through_m8192"] = _compile_bulk(use_pdl=self._use_pdl)

    @staticmethod
    def schedule_name(active_m: int) -> str:
        """Return the B300 schedule selected for runtime token count ``M``."""
        if active_m < 1 or active_m > MAX_PREFILL_M:
            raise ValueError(f"embedding M must be in [1,{MAX_PREFILL_M}], got {active_m}")
        if active_m <= 2:
            return "decode_m1_m2"
        if active_m in (4, 16):
            return "decode_m4_m16"
        if active_m == 8:
            return "decode_m8"
        if active_m == 24:
            return "decode_m24"
        if active_m <= 32:
            return "decode_m32"
        if active_m <= 4096:
            return "context_through_m4096"
        return "context_through_m8192"

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return contiguous BF16 embeddings with staged invalid-ID semantics."""
        active_m = input_ids.numel()
        if input_ids.shape != (active_m,) or input_ids.dtype != torch.int32:
            raise ValueError(
                f"input_ids must be contiguous int32 [M], got "
                f"shape={input_ids.shape} dtype={input_ids.dtype}"
            )
        if not input_ids.is_cuda or not input_ids.is_contiguous():
            raise ValueError("input_ids must be a contiguous CUDA tensor")
        if input_ids.device != self.weight.device:
            raise ValueError(
                f"input_ids and weight must share a CUDA device, got "
                f"{input_ids.device} and {self.weight.device}"
            )

        output = self.weight.new_empty((active_m, HIDDEN_SIZE))
        schedule = self.schedule_name(active_m)
        self._kernels[schedule](input_ids, self.weight, output)
        return output


def make_trtllm_adapter(weight: torch.Tensor, *, use_pdl: bool = True) -> GLMEmbeddingAdapter:
    """Construct the callsite-compatible GLM Embedding replacement module."""
    return GLMEmbeddingAdapter(weight, use_pdl=use_pdl)
