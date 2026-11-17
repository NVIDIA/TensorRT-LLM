# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL prefill indexer top-k for GLM.

TRT-LLM's ``indexer_topk_prefill`` accepts row-local visibility windows:

    logits [num_rows, num_columns]
    row_starts [num_rows]
    row_ends [num_rows]
    output_indices [num_rows, top_k]

For GLM5.2's target prefill shape, ``ISL=1024`` and ``top_k=2048``. Every
causal row has ``row_ends[row] - row_starts[row] <= top_k``. The TRT-LLM C++
kernel has a fast path for this case: it writes local indices
``0..row_len-1`` and fills the remaining slots with ``-1`` without reading
logits. This file provides the same prefill API in CuTe DSL for that
short-window case.

For long windows, such as the OSL=8192 serving warmup path, the op must compute
a real top-k over each row's visible range. The long-window path below reuses
TRT-LLM's shared CuTe DSL radix/filter top-k utility and adds a tiny CuTe fix-up
kernel that converts absolute indices back to TRT-LLM's row-local prefill
contract.
"""

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.memory import SmemAllocator

_THREADS_PER_BLOCK = 512
_DEFAULT_TOP_K = 2048
_ALLOWED_ELEMENTS_PER_THREAD = (1, 2, 4, 8)
_COMPILE_CACHE: dict[tuple[int, int, int], Callable] = {}
_GENERAL_COMPILE_CACHE: dict[
    tuple[int, str, int, int, int, int, int], tuple[Callable, Callable]
] = {}


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _torch_dtype_to_cutlass(torch, dtype):
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    raise TypeError(f"unsupported logits dtype for CuTe prefill top-k: {dtype}")


@cute.kernel
def indexer_topk_prefill_short_window(
    logits: cute.Tensor,
    row_starts: cute.Tensor,
    row_ends: cute.Tensor,
    output_indices: cute.Tensor,
    TOP_K: cutlass.Constexpr[int],
    ELEMENTS_PER_THREAD: cutlass.Constexpr[int],
) -> None:
    """Write local prefill indices for rows whose visible length is <= top_k."""

    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    num_rows = cute.size(logits, mode=[0])

    if row < num_rows:
        row_start = (row_starts.iterator + row).load()
        row_end = (row_ends.iterator + row).load()
        row_len = row_end - row_start
        row_base = row * TOP_K

        for j in cutlass.range_constexpr(ELEMENTS_PER_THREAD):
            col = tidx + j * _THREADS_PER_BLOCK
            if col < TOP_K:
                value = cutlass.Int32(-1)
                if col < row_len:
                    value = col
                (output_indices.iterator + row_base + col).store(value)


def _make_prefill_varlen_topk_class():
    from tensorrt_llm._torch.kernels.blackwell.utils import (
        TRTLLM_ENABLE_PDL,
        griddepcontrol_launch_dependents,
        griddepcontrol_wait,
    )

    from .filtered_top_k_varlen_util import FilteredTopKKernelVarlen, half_as_ushort

    class IndexerTopKPrefillVarlen(FilteredTopKKernelVarlen):
        """Prefill row-start/row-end wrapper around TRT-LLM's CuTe top-k utility."""

        def __init__(
            self,
            dtype: cutlass.Numeric,
            max_num_cols: int,
            top_k: int,
            num_copy_bits: int = 256,
            coarse_radix_bits: int = 8,
            num_threads_per_cta: int = 0,
        ):
            super().__init__(
                dtype=dtype,
                max_num_cols=max_num_cols,
                top_k=top_k,
                num_copy_bits=num_copy_bits,
                return_val=False,
                enable_multi_cta=False,
            )
            self.coarse_radix_bits = coarse_radix_bits
            self.radix = 1 << coarse_radix_bits
            if num_threads_per_cta:
                self.num_threads_per_cta = num_threads_per_cta
            # The inherited histogram clear writes one bin per thread, so a
            # radix wider than the CTA would leave bins uninitialized.
            if self.radix > self.num_threads_per_cta:
                raise ValueError(
                    f"coarse_radix_bits={coarse_radix_bits} needs {self.radix} histogram "
                    f"bins, but num_threads_per_cta is {self.num_threads_per_cta}"
                )

        @cute.jit
        def to_coarse_key(self, x):
            if cutlass.const_expr(self.dtype == cutlass.Float32):
                h = x.to(cutlass.Float16)
                bits = half_as_ushort(h)

                key = cutlass.Uint16(0)
                if bits & 0x8000:
                    key = cutlass.Uint16(bits)
                else:
                    key = (bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF)

                if cutlass.const_expr(self.coarse_radix_bits == 11):
                    return cute.Uint16((key >> 5) & 0x7FF)
                if cutlass.const_expr(self.coarse_radix_bits == 10):
                    return cute.Uint16((key >> 6) & 0x3FF)
                return cute.Uint8((key >> 8) & 0xFF)

            if cutlass.const_expr(self.dtype == cutlass.Float16):
                bits = half_as_ushort(x)
            else:
                bits = half_as_ushort(x)

            key = cute.Uint16(0)
            if bits & 0x8000:
                key = cutlass.Uint16(bits)
            else:
                key = (bits ^ cutlass.Uint16(0xFFFF)) & cute.Uint16(0x7FFF)
            return cute.Uint8((key >> 8) & 0xFF)

        @cute.kernel
        def filtered_topk_prefill_kernel(
            self,
            logits: cute.Tensor,
            extra_buffer: cute.Tensor,
            row_starts: cute.Tensor,
            row_ends: cute.Tensor,
            output_indices: cute.Tensor,
            tiler_mn: cute.Shape,
            copy_atom: cute.CopyAtom,
            tiled_copy: cute.TiledCopy,
        ):
            griddepcontrol_wait()

            smem = SmemAllocator()
            s_histogram = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((self.radix + 1,), order=(0,)),
                byte_alignment=128,
            )
            s_counter = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            s_threshold_bin_id = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            s_num_input = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((2,), order=(0,)),
                byte_alignment=128,
            )
            if cutlass.const_expr(self.enable_gmem_store):
                g_num_input = smem.allocate_tensor(
                    element_type=cutlass.Int32,
                    layout=cute.make_ordered_layout((2,), order=(0,)),
                    byte_alignment=128,
                )
            else:
                g_num_input = None
            s_indices = smem.allocate_tensor(
                element_type=self.index_type,
                layout=cute.make_ordered_layout((self.filtered_topk_max_k,), order=(0,)),
                byte_alignment=128,
            )
            s_input_idx = smem.allocate_tensor(
                element_type=self.index_type,
                layout=cute.make_ordered_layout(
                    (
                        self.num_buffer_smem_input_idx,
                        self.filtered_topk_smem_input_size,
                    ),
                    order=(1, 0),
                ),
                byte_alignment=128,
            )
            s_last_remain = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((1,), order=(0,)),
                byte_alignment=128,
            )
            num_warps = cutlass.const_expr(
                min(self.radix, self.num_threads_per_cta) // cutlass.Int32(32)
            )
            s_warp_sums = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=cute.make_ordered_layout((num_warps,), order=(0,)),
                byte_alignment=128,
            )

            row, _, _ = cute.arch.block_idx()
            row_start = row_starts[row]
            row_end = row_ends[row]
            length = row_end - row_start

            self.filtered_topk_kernel_per_row(
                logits,
                None,
                extra_buffer,
                output_indices,
                None,
                tiler_mn,
                copy_atom,
                tiled_copy,
                row_start,
                length,
                row,
                s_histogram,
                s_counter,
                s_threshold_bin_id,
                s_num_input,
                g_num_input,
                s_indices,
                s_input_idx,
                s_last_remain,
                num_warps,
                s_warp_sums,
            )
            griddepcontrol_launch_dependents()

        @cute.jit
        def __call__(
            self,
            logits: cute.Tensor,
            extra_buffer: cute.Tensor,
            row_starts: cute.Tensor,
            row_ends: cute.Tensor,
            output_indices: cute.Tensor,
            stream: cuda.CUstream,
        ):
            copy_atom, tiled_copy, tiler_mn = self._get_tiled_copy()
            rows = cute.size(logits, mode=[0])
            self.filtered_topk_prefill_kernel(
                logits,
                extra_buffer,
                row_starts,
                row_ends,
                output_indices,
                tiler_mn,
                copy_atom,
                tiled_copy,
            ).launch(
                grid=(rows, 1, 1),
                block=(tiled_copy.size, 1, 1),
                stream=stream,
                use_pdl=TRTLLM_ENABLE_PDL,
            )

    return IndexerTopKPrefillVarlen


@cute.kernel
def _localize_prefill_indices(
    row_starts: cute.Tensor,
    row_ends: cute.Tensor,
    output_indices: cute.Tensor,
    TOP_K: cutlass.Constexpr[int],
) -> None:
    row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row_start = row_starts[row]
    row_end = row_ends[row]
    row_base = row * TOP_K
    # The shared top-k utility writes ABSOLUTE columns for a row longer than
    # TOP_K and window-LOCAL 0..len-1 for a row that fits (its short-row fast
    # path). Only the long rows need localising; a short row is already
    # window-local, and subtracting row_start from it would make its indices
    # negative.
    if row_end - row_start > TOP_K:
        for col in range(tidx, TOP_K, _THREADS_PER_BLOCK):
            value = (output_indices.iterator + row_base + col).load()
            if value >= 0:
                value = value - row_start
                (output_indices.iterator + row_base + col).store(value)


@cute.jit
def _launch_localize_prefill_indices(
    row_starts: cute.Tensor,
    row_ends: cute.Tensor,
    output_indices: cute.Tensor,
    stream: cuda.CUstream,
    TOP_K: cutlass.Constexpr[int],
) -> None:
    rows = cute.size(row_starts, mode=[0])
    _localize_prefill_indices(row_starts, row_ends, output_indices, TOP_K).launch(
        grid=(rows, 1, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_indexer_topk_prefill_short_window(
    logits: cute.Tensor,
    row_starts: cute.Tensor,
    row_ends: cute.Tensor,
    output_indices: cute.Tensor,
    stream: cuda.CUstream,
    TOP_K: cutlass.Constexpr[int],
    ELEMENTS_PER_THREAD: cutlass.Constexpr[int],
) -> None:
    rows = cute.size(logits, mode=[0])

    indexer_topk_prefill_short_window(
        logits,
        row_starts,
        row_ends,
        output_indices,
        TOP_K,
        ELEMENTS_PER_THREAD,
    ).launch(
        grid=(rows, 1, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
    )


def compile_indexer_topk_prefill_short_window(
    device_id: int = 0,
    top_k: int = _DEFAULT_TOP_K,
    elements_per_thread: int = 4,
) -> Callable:
    """Compile the short-window prefill top-k native CuTe kernel."""

    if elements_per_thread not in _ALLOWED_ELEMENTS_PER_THREAD:
        raise ValueError(
            f"elements_per_thread must be one of {_ALLOWED_ELEMENTS_PER_THREAD}, "
            f"got {elements_per_thread}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if _THREADS_PER_BLOCK * elements_per_thread < top_k:
        raise ValueError(
            "elements_per_thread does not cover top_k with one CTA per row: "
            f"{_THREADS_PER_BLOCK} * {elements_per_thread} < {top_k}"
        )

    key = (device_id, top_k, elements_per_thread)
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    cutlass.cuda.initialize_cuda_context(device_id)
    sym_num_rows = cute.sym_int32()
    sym_num_columns = cute.sym_int32()
    fake_logits = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_num_rows, sym_num_columns),
        stride_order=(1, 0),
        assumed_align=16,
    )
    fake_row_starts = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows,),
        assumed_align=4,
    )
    fake_row_ends = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows,),
        assumed_align=4,
    )
    fake_output = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )

    compiled = cute.compile(
        _launch_indexer_topk_prefill_short_window,
        fake_logits,
        fake_row_starts,
        fake_row_ends,
        fake_output,
        make_fake_stream(),
        top_k,
        elements_per_thread,
        options="--enable-tvm-ffi",
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def compile_indexer_topk_prefill_general(
    device_id: int = 0,
    logits_dtype=None,
    max_num_columns: int = 8192,
    top_k: int = _DEFAULT_TOP_K,
    num_copy_bits: int = 256,
    coarse_radix_bits: int = 8,
    num_threads_per_cta: int = 0,
) -> tuple[Callable, Callable]:
    """Compile the general long-window prefill top-k path.

    The top-k kernel writes absolute column indices. The localize kernel then
    converts them to TRT-LLM's prefill contract: row-local indices relative to
    ``row_starts[row]``.
    """

    import torch

    if logits_dtype is None:
        logits_dtype = torch.float32
    dtype = _torch_dtype_to_cutlass(torch, logits_dtype)
    if max_num_columns <= top_k:
        raise ValueError(
            "compile_indexer_topk_prefill_general is for long rows; got "
            f"max_num_columns={max_num_columns}, top_k={top_k}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if coarse_radix_bits not in (8, 10, 11):
        raise ValueError(f"coarse_radix_bits must be 8, 10, or 11, got {coarse_radix_bits}")
    if num_threads_per_cta not in (0, 256, 512, 1024):
        raise ValueError(
            f"num_threads_per_cta must be 0, 256, 512, or 1024, got {num_threads_per_cta}"
        )

    bucketed_num_columns = _next_power_of_2(max_num_columns)
    key = (
        device_id,
        str(logits_dtype),
        bucketed_num_columns,
        top_k,
        num_copy_bits,
        coarse_radix_bits,
        num_threads_per_cta,
    )
    if key in _GENERAL_COMPILE_CACHE:
        return _GENERAL_COMPILE_CACHE[key]

    cutlass.cuda.initialize_cuda_context(device_id)
    kernel_cls = _make_prefill_varlen_topk_class()
    topk_kernel = kernel_cls(
        dtype=dtype,
        max_num_cols=bucketed_num_columns,
        top_k=top_k,
        num_copy_bits=num_copy_bits,
        coarse_radix_bits=coarse_radix_bits,
        num_threads_per_cta=num_threads_per_cta,
    )

    sym_num_rows = cute.sym_int32()
    sym_num_columns = cute.sym_int32()
    fake_logits = make_fake_compact_tensor(
        dtype,
        (sym_num_rows, sym_num_columns),
        stride_order=(1, 0),
        assumed_align=32,
    )
    buffer_numbers = 2 if dtype == cutlass.Float32 else 1
    fake_extra_buffer = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows, buffer_numbers, sym_num_columns),
        stride_order=(2, 1, 0),
        assumed_align=32,
    )
    fake_row_starts = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows,),
        assumed_align=4,
    )
    fake_row_ends = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows,),
        assumed_align=4,
    )
    fake_output = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows, top_k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_topk = cute.compile(
        topk_kernel,
        fake_logits,
        fake_extra_buffer,
        fake_row_starts,
        fake_row_ends,
        fake_output,
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )
    compiled_localize = cute.compile(
        _launch_localize_prefill_indices,
        fake_row_starts,
        fake_row_ends,
        fake_output,
        make_fake_stream(),
        top_k,
        options="--enable-tvm-ffi",
    )
    _GENERAL_COMPILE_CACHE[key] = (compiled_topk, compiled_localize)
    return compiled_topk, compiled_localize


def _current_cuda_stream(torch, device) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device=device).cuda_stream)


def _require_cuda_contiguous(torch, tensor, name: str, dtype) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_same_cuda_device(logits, *named_tensors) -> int:
    device_id = logits.get_device()
    for name, tensor in named_tensors:
        if tensor.get_device() != device_id:
            raise ValueError(
                f"{name} must be on the same CUDA device as logits: "
                f"cuda:{device_id}, got cuda:{tensor.get_device()}"
            )
    return device_id


def indexer_topk_prefill_short_window_run(
    logits,
    row_starts,
    row_ends,
    output_indices,
    top_k: int = _DEFAULT_TOP_K,
    elements_per_thread: int = 4,
    compiled_fn: Callable | None = None,
):
    """Run native CuTe prefill top-k for rows with ``row_len <= top_k``.

    This matches TRT-LLM ``indexer_topk_prefill`` exactly for short windows.
    The caller must guarantee ``0 <= row_starts <= row_ends`` and
    ``row_ends - row_starts <= top_k`` for every row. Use
    ``indexer_topk_prefill_general_run`` when the shape may require the
    general long-window path.
    """

    import torch

    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be {torch.float32}, got {logits.dtype}")
    if not logits.is_cuda:
        raise TypeError("logits must be a CUDA tensor")
    if not logits.is_contiguous():
        logits = logits.contiguous()
    _require_cuda_contiguous(torch, logits, "logits", torch.float32)
    _require_cuda_contiguous(torch, row_starts, "row_starts", torch.int32)
    _require_cuda_contiguous(torch, row_ends, "row_ends", torch.int32)
    _require_cuda_contiguous(torch, output_indices, "output_indices", torch.int32)
    device_id = _require_same_cuda_device(
        logits,
        ("row_starts", row_starts),
        ("row_ends", row_ends),
        ("output_indices", output_indices),
    )
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if row_starts.shape != (logits.shape[0],) or row_ends.shape != (logits.shape[0],):
        raise ValueError("row_starts and row_ends must have one entry per logits row")
    if output_indices.shape != (logits.shape[0], top_k):
        raise ValueError(
            f"output_indices must have shape [{logits.shape[0]},{top_k}], "
            f"got {tuple(output_indices.shape)}"
        )

    if compiled_fn is None:
        compiled_fn = compile_indexer_topk_prefill_short_window(
            device_id=device_id,
            top_k=top_k,
            elements_per_thread=elements_per_thread,
        )

    compiled_fn(
        logits,
        row_starts,
        row_ends,
        output_indices,
        _current_cuda_stream(torch, logits.device),
    )
    return output_indices


def indexer_topk_prefill_general_run(
    logits,
    row_starts,
    row_ends,
    output_indices,
    top_k: int = _DEFAULT_TOP_K,
    num_copy_bits: int = 256,
    coarse_radix_bits: int = 8,
    num_threads_per_cta: int = 0,
    compiled_fns: tuple[Callable, Callable] | None = None,
    extra_buffer=None,
    localize_indices: bool = True,
):
    """Run native CuTe prefill top-k for rows whose visible length may exceed top_k."""

    import torch

    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be {torch.float32}, got {logits.dtype}")
    if not logits.is_cuda:
        raise TypeError("logits must be a CUDA tensor")
    if not logits.is_contiguous():
        logits = logits.contiguous()
    _require_cuda_contiguous(torch, logits, "logits", torch.float32)
    _require_cuda_contiguous(torch, row_starts, "row_starts", torch.int32)
    _require_cuda_contiguous(torch, row_ends, "row_ends", torch.int32)
    _require_cuda_contiguous(torch, output_indices, "output_indices", torch.int32)
    device_id = _require_same_cuda_device(
        logits,
        ("row_starts", row_starts),
        ("row_ends", row_ends),
        ("output_indices", output_indices),
    )
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if row_starts.shape != (logits.shape[0],) or row_ends.shape != (logits.shape[0],):
        raise ValueError("row_starts and row_ends must have one entry per logits row")
    if output_indices.shape != (logits.shape[0], top_k):
        raise ValueError(
            f"output_indices must have shape [{logits.shape[0]},{top_k}], "
            f"got {tuple(output_indices.shape)}"
        )

    if compiled_fns is None:
        compiled_fns = compile_indexer_topk_prefill_general(
            device_id=device_id,
            logits_dtype=logits.dtype,
            max_num_columns=logits.shape[1],
            top_k=top_k,
            num_copy_bits=num_copy_bits,
            coarse_radix_bits=coarse_radix_bits,
            num_threads_per_cta=num_threads_per_cta,
        )

    compiled_topk, compiled_localize = compiled_fns
    buffer_numbers = 2
    expected_buffer_shape = (logits.shape[0], buffer_numbers, logits.shape[1])
    if extra_buffer is None:
        extra_buffer = torch.empty(
            expected_buffer_shape,
            dtype=torch.int32,
            device=logits.device,
        )
    else:
        _require_cuda_contiguous(torch, extra_buffer, "extra_buffer", torch.int32)
        _require_same_cuda_device(logits, ("extra_buffer", extra_buffer))
        if extra_buffer.shape != expected_buffer_shape:
            raise ValueError(
                f"extra_buffer must have shape {expected_buffer_shape}, "
                f"got {tuple(extra_buffer.shape)}"
            )
    stream = _current_cuda_stream(torch, logits.device)
    # The TRT-LLM top-k utility is compiled with use_tvm_ffi_env_stream=True,
    # so its TVM-FFI callable takes only tensor arguments and uses the active
    # stream from the runtime environment.
    compiled_topk(logits, extra_buffer, row_starts, row_ends, output_indices)
    if localize_indices:
        compiled_localize(row_starts, row_ends, output_indices, stream)
    return output_indices
