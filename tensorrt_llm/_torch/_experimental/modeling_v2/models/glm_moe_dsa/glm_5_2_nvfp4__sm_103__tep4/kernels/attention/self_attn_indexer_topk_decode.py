# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL GLM decode indexer top-k.

TRT-LLM's ``indexer_topk_decode`` selects top-k sparse-indexer rows from
decode logits:

    logits         float32 [num_rows, num_columns]
    seq_lens       int32   [num_rows / next_n]
    output_indices int32   [num_rows, top_k]

For speculative decode, each request owns ``next_n`` rows. Row ``r`` scans the
visible prefix
``seq_lens[r // next_n] - next_n + (r % next_n) + 1``. The GLM5.2 steady
decode target is ``num_rows=64``, ``num_columns=8192``, ``next_n=2``, and
``top_k=2048``.

The histogram/threshold/select stage and the final sort remain two logical
stages, but this implementation runs them inside one TopK kernel. It reads the
logits with the row pitch of the preceding MQA logits kernel, so no copy kernel
runs between the two.

The selected index set per row is the exact top-k under the ordering (ordered
fp32 bits descending, column ascending). Threshold bins with more than 512
candidates are resolved by a bounded loop of 9-bit radix rounds in shared
memory instead of the O(n^2) rank sort on large bins; smaller
bins keep the original rank sort with the same tie rule.

Programmatic Dependent Launch (opt-in, ``pdl=True`` on
``compile_indexer_topk_decode``): the launch carries the programmatic stream
serialization attribute so the grid may start while the predecessor (the fp8
paged MQA logits kernel) drains.  Inside the kernel ``griddepcontrol.wait`` is issued
after the ``seq_lens`` read and the shared-memory carve-up and immediately
before the first ``logits`` read / first ``output_indices`` store; ``seq_lens``
is host metadata that predates that kernel, so reading it pre-wait is legal and hides
its latency.  ``griddepcontrol.launch_dependents`` is issued by thread 0 after
a CTA barrier that follows the last global store.  The default (``pdl=False``)
emits neither instruction and launches without the attribute, i.e. the
existing kernel.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor
from cutlass.memory import SmemAllocator

from . import self_attn_indexer_topk_histogram as _histogram

_DEFAULT_TOP_K = 2048
_DEFAULT_NEXT_N = 2
RawTopKLauncher = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, cuda.CUstream], object]


@dataclass(frozen=True)
class CompiledIndexerTopKDecode:
    """A compiled TopK launcher bound to its serving specialization."""

    _raw: RawTopKLauncher
    device_id: int
    logits_dtype: torch.dtype
    max_num_columns: int
    top_k: int
    next_n: int
    pdl: bool = False


_COMPILE_CACHE: dict[tuple[int, str, int, int, int, bool, int], CompiledIndexerTopKDecode] = {}


def _make_decode_histogram_topk_class():
    prefill_cls = _histogram._make_prefill_histogram_topk_class()

    class IndexerTopKDecodeHistogram(prefill_cls):
        """Single-CTA GLM decode top-k: histogram fast path + bounded SMEM refine."""

        # Programmatic Dependent Launch.  Set once by the compile entry; it is a
        # compile-time constant inside the kernel and the launch.
        pdl: bool = False

        @cute.kernel
        def histogram_topk_decode_kernel(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            output_indices: cute.Tensor,
        ):
            row, _, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()
            next_n = cutlass.const_expr(self.next_n)

            req = row // cutlass.Int32(next_n)
            nn = row % cutlass.Int32(next_n)
            seq_len = seq_lens[req]
            length = seq_len - cutlass.Int32(next_n) + nn + cutlass.Int32(1)
            num_columns = cute.size(logits, mode=[1])
            if length > num_columns:
                length = num_columns
            if length < cutlass.Int32(0):
                length = cutlass.Int32(0)

            score = logits[row, None]
            dst = output_indices[row, None]

            smem = SmemAllocator()
            (
                s_histogram,
                s_output,
                s_cand_vals,
                s_cand_idx,
                s_threshold_bin,
                s_cand_count,
                s_final_bin_size,
                s_found_topk,
                s_found_threshold,
                s_pattern,
                s_warp_sums,
                s_sort_keys,
            ) = self.allocate_row_smem(smem)

            if cutlass.const_expr(self.pdl):
                # The predecessor grid (MQA logits) writes ``logits``.  ``seq_lens``
                # above is host metadata that predates it and the SMEM
                # carve-up is CTA-local, so this is the latest legal point:
                # before the first ``logits`` read and the first ``dst`` store,
                # both inside select_topk_row.
                cute.arch.griddepcontrol_wait()

            self.select_topk_row(
                score,
                cutlass.Int32(0),
                length,
                dst,
                s_histogram,
                s_output,
                s_cand_vals,
                s_cand_idx,
                s_threshold_bin,
                s_cand_count,
                s_final_bin_size,
                s_found_topk,
                s_found_threshold,
                s_pattern,
                s_warp_sums,
                s_sort_keys,
                tidx,
            )

            if cutlass.const_expr(self.pdl):
                # Every ``dst`` store of this CTA precedes the barrier; the
                # dependent's own wait provides global visibility.
                cute.arch.barrier()
                if tidx == 0:
                    cute.arch.griddepcontrol_launch_dependents()

        @cute.jit
        def __call__(
            self,
            logits: cute.Tensor,
            seq_lens: cute.Tensor,
            output_indices: cute.Tensor,
            stream: cuda.CUstream,
        ):
            rows = cute.size(logits, mode=[0])
            self.histogram_topk_decode_kernel(
                logits,
                seq_lens,
                output_indices,
            ).launch(
                grid=(rows, 1, 1),
                block=(self.num_threads_per_cta, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )

    return IndexerTopKDecodeHistogram


def compile_indexer_topk_decode(
    device_id: int = 0,
    logits_dtype: torch.dtype | None = None,
    max_num_columns: int = 8192,
    top_k: int = _DEFAULT_TOP_K,
    next_n: int = _DEFAULT_NEXT_N,
    pdl: bool = False,
    refine_cutoff: int | None = None,
) -> CompiledIndexerTopKDecode:
    """Compile the specialized GLM decode top-k path.

    ``pdl=True`` builds the Programmatic-Dependent-Launch variant (launch
    attribute + ``griddepcontrol`` wait/trigger, see the module docstring).  It
    is only useful when the kernel is launched right after the producer of
    ``logits``; the default variant is unchanged.

    ``refine_cutoff`` overrides the helper's ``_REFINE_CUTOFF`` (largest
    threshold bin the rank sort handles before the radix refine takes over);
    ``None`` keeps the helper default.  It is part of the compile cache key.
    """

    if logits_dtype is None:
        logits_dtype = torch.float32
    if logits_dtype != torch.float32:
        raise TypeError(f"decode top-k expects torch.float32 logits, got {logits_dtype}")
    if max_num_columns <= top_k:
        raise ValueError(
            "compile_indexer_topk_decode expects rows longer than top_k; got "
            f"max_num_columns={max_num_columns}, top_k={top_k}"
        )
    if top_k != _DEFAULT_TOP_K:
        raise ValueError(f"decode top-k expects top_k=2048, got {top_k}")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if refine_cutoff is None:
        refine_cutoff = _histogram._REFINE_CUTOFF
    if refine_cutoff < 0:
        raise ValueError(f"refine_cutoff must be non-negative, got {refine_cutoff}")

    # The generated function keeps the physical row pitch symbolic.  The typed
    # handle nevertheless binds its warmed logical capacity and all other
    # specialization fields, so a caller cannot pair a 2048-wide kernel with a
    # smaller output or a different speculative-decode grouping.
    key = (
        device_id,
        str(logits_dtype),
        max_num_columns,
        top_k,
        next_n,
        bool(pdl),
        int(refine_cutoff),
    )
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]

    kernel_cls = _make_decode_histogram_topk_class()
    topk_kernel = kernel_cls(top_k=top_k, refine_cutoff=int(refine_cutoff))
    topk_kernel.next_n = next_n
    topk_kernel.pdl = bool(pdl)

    sym_num_rows = cute.sym_int32()
    sym_num_columns = cute.sym_int32()
    sym_batch = cute.sym_int32()
    sym_logits_stride = cute.sym_int64()
    fake_logits = make_fake_tensor(
        cutlass.Float32,
        (sym_num_rows, sym_num_columns),
        stride=(sym_logits_stride, 1),
        assumed_align=4,
    )
    fake_seq_lens = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=4,
    )
    fake_output = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_rows, top_k),
        stride_order=(1, 0),
        assumed_align=4,
    )

    compiled = cute.compile(
        topk_kernel,
        fake_logits,
        fake_seq_lens,
        fake_output,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
    launcher = CompiledIndexerTopKDecode(
        _raw=compiled,
        device_id=device_id,
        logits_dtype=logits_dtype,
        max_num_columns=max_num_columns,
        top_k=top_k,
        next_n=next_n,
        pdl=bool(pdl),
    )
    _COMPILE_CACHE[key] = launcher
    return launcher


def warmup_indexer_topk_decode(
    max_num_columns: int = 8192,
    top_k: int = _DEFAULT_TOP_K,
    next_n: int = _DEFAULT_NEXT_N,
    device_id: int | None = None,
    pdl: bool = False,
) -> CompiledIndexerTopKDecode:
    """Compile and launch once before TRT-LLM CUDA graph capture starts."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to warm up indexer_topk_decode")
    if device_id is None:
        device_id = torch.cuda.current_device()
    with torch.cuda.device(device_id):
        compiled = compile_indexer_topk_decode(
            device_id=device_id,
            logits_dtype=torch.float32,
            max_num_columns=max_num_columns,
            top_k=top_k,
            next_n=next_n,
            pdl=pdl,
        )

        rows = max(next_n, 1)
        logits = torch.zeros((rows, max_num_columns), dtype=torch.float32, device="cuda")
        seq_lens = torch.full((rows // next_n,), max_num_columns, dtype=torch.int32, device="cuda")
        output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
        indexer_topk_decode_run(
            logits,
            seq_lens,
            output,
            top_k=top_k,
            next_n=next_n,
            compiled_fn=compiled,
        )
        torch.cuda.synchronize()
    return compiled


def _current_cuda_stream(torch, device: torch.device) -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _require_cuda_contiguous(torch, tensor: torch.Tensor, name: str, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def indexer_topk_decode_run(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    output_indices: torch.Tensor,
    compiled_fn: CompiledIndexerTopKDecode,
    top_k: int = _DEFAULT_TOP_K,
    next_n: int = _DEFAULT_NEXT_N,
) -> torch.Tensor:
    """Run native CuTe decode top-k for GLM sparse indexer logits."""

    if not isinstance(compiled_fn, CompiledIndexerTopKDecode):
        raise TypeError("compiled_fn must be a CompiledIndexerTopKDecode handle")
    if compiled_fn.top_k != top_k or compiled_fn.next_n != next_n:
        raise ValueError(
            "compiled TopK specialization mismatch: "
            f"handle=(top_k={compiled_fn.top_k}, next_n={compiled_fn.next_n}), "
            f"call=(top_k={top_k}, next_n={next_n})"
        )
    if compiled_fn.logits_dtype != logits.dtype:
        raise TypeError(
            "compiled TopK logits dtype mismatch: "
            f"handle={compiled_fn.logits_dtype}, call={logits.dtype}"
        )
    if compiled_fn.device_id != logits.device.index:
        raise ValueError(
            "compiled TopK device mismatch: "
            f"handle=cuda:{compiled_fn.device_id}, call={logits.device}"
        )
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be {torch.float32}, got {logits.dtype}")
    if not logits.is_cuda:
        raise TypeError("logits must be a CUDA tensor")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if logits.shape[1] <= 0:
        raise ValueError("logits must have a positive logical column count")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if logits.stride(1) != 1 or logits.stride(0) < logits.shape[1]:
        raise ValueError(
            "logits must have unit column stride and a row stride no smaller "
            f"than the logical column count, got shape={tuple(logits.shape)} "
            f"stride={tuple(logits.stride())}"
        )
    _require_cuda_contiguous(torch, seq_lens, "seq_lens", torch.int32)
    _require_cuda_contiguous(torch, output_indices, "output_indices", torch.int32)
    if logits.device != seq_lens.device or logits.device != output_indices.device:
        raise ValueError("logits, seq_lens, and output_indices must share one device")
    if logits.shape[0] % next_n != 0:
        raise ValueError(f"logits rows must be divisible by next_n={next_n}, got {logits.shape[0]}")
    batch = logits.shape[0] // next_n
    if seq_lens.shape != (batch,):
        raise ValueError(f"seq_lens must have shape [{batch}], got {tuple(seq_lens.shape)}")
    if output_indices.shape != (logits.shape[0], top_k):
        raise ValueError(
            f"output_indices must have shape [{logits.shape[0]},{top_k}], "
            f"got {tuple(output_indices.shape)}"
        )
    storage_identities = {
        logits.untyped_storage().data_ptr(),
        seq_lens.untyped_storage().data_ptr(),
        output_indices.untyped_storage().data_ptr(),
    }
    if len(storage_identities) != 3:
        raise ValueError("logits, seq_lens, and output_indices need disjoint storage")
    if logits.shape[0] == 0:
        return output_indices

    if logits.shape[1] > compiled_fn.max_num_columns:
        raise ValueError(
            "logits columns exceed the compiled TopK capacity: "
            f"columns={logits.shape[1]}, capacity={compiled_fn.max_num_columns}"
        )

    compiled_fn._raw(
        logits,
        seq_lens,
        output_indices,
        _current_cuda_stream(torch, logits.device),
    )
    return output_indices
