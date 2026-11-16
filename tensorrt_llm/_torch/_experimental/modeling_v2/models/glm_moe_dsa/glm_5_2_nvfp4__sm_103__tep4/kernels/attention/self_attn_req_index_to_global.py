# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM sparse-attention request-index conversion: local top-k token index -> global paged-cache pool row.

Mirrors TRT-LLM's ``convertReqIndexToGlobalKernel`` (block size 32): for token
row ``t`` and top-k slot ``s`` with ``idx = token_indices[t, s]``,
``out[t, s] = block_table[req_id[t], idx >> 5] * stride_factor + (idx & 31)
+ layer_id * 32``, and ``-1`` for an invalid slot (negative index, block beyond
the table row, or unmapped block), as the CUDA backend does.

The shared decode chain (owner ``req_index_to_global``) runs the raw-pointer
kernel: ``compile_req_index_rawptr`` / ``convert_req_index_to_global_rawptr``.
Top-k columns and blocks-per-request are compile-time, one element per thread,
raw-pointer loads, ``use_pdl`` (default on: wait before the first request read,
CTA barrier and dependent launch after the store).

The raw-pointer kernel has one compile-time output variant, ``base_rows``:
``compile_req_index_base_rawptr`` / ``convert_req_index_to_base_rawptr``
write the layer-independent base row ``block_table[req_id[t], idx >> 5] *
stride_factor + (idx & 31)`` (the ``layer_id * 32`` term is dropped at compile
time; ``-1`` for invalid slots as above). The shared route runs it once per
three consecutive shared layers and the sparse MLA attention kernel adds its own
``local_layer_idx * 32`` (``index_kind=INDEX_KIND_BASE_WITH_LAYER_OFFSET``),
so ``base_row + local_layer_idx * 32 == global_row`` for every valid slot.
``base_rows=False`` (the default of ``compile_req_index_rawptr``) is the
unchanged global mode.
"""

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

_THREADS_PER_BLOCK = 256


_INT32_ALIGNMENT_BYTES = 4


_REQ_INDEX_BLOCK_SIZE = 32


_REQ_INDEX_BLOCK_SHIFT = 5


_REQ_INDEX_BLOCK_MASK = _REQ_INDEX_BLOCK_SIZE - 1


_REQ_INDEX_RAWPTR_COMPILE_CACHE: dict[tuple[int, int, int, bool, bool], Callable] = {}


@cute.kernel
def self_attn_req_index_to_global_rawptr(
    req_id: cute.Tensor,
    block_table: cute.Tensor,
    token_indices: cute.Tensor,
    output: cute.Tensor,
    stride_factor: cutlass.Int32,
    layer_id: cutlass.Int32,
    NUM_TOPK_TOKENS: cutlass.Constexpr[int],
    MAX_BLOCKS_PER_REQ: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    base_rows: cutlass.Constexpr[bool],
) -> None:
    token_id, tile_col, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    col = tile_col * _THREADS_PER_BLOCK + tidx
    flat_offset = token_id * NUM_TOPK_TOKENS + col

    req_ptr = req_id.iterator
    block_table_ptr = block_table.iterator
    token_indices_ptr = token_indices.iterator
    output_ptr = output.iterator

    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    req = req_ptr[token_id]
    tok = token_indices_ptr[flat_offset]
    mapped = cutlass.Int32(-1)

    if tok >= 0:
        block_id = tok >> _REQ_INDEX_BLOCK_SHIFT
        if block_id < MAX_BLOCKS_PER_REQ:
            base = block_table_ptr[req * MAX_BLOCKS_PER_REQ + block_id]
            if base >= 0:
                if cutlass.const_expr(base_rows):
                    # Base-row variant: layer-independent row; the attention
                    # kernel adds local_layer_idx * 32. ``layer_id`` is unused here.
                    inblock_offset = tok & _REQ_INDEX_BLOCK_MASK
                else:
                    inblock_offset = (
                        tok & _REQ_INDEX_BLOCK_MASK
                    ) + layer_id * _REQ_INDEX_BLOCK_SIZE
                mapped = base * stride_factor + inblock_offset

    output_ptr[flat_offset] = mapped

    cute.arch.barrier(barrier_id=0, number_of_threads=_THREADS_PER_BLOCK)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def _launch_req_index_to_global_rawptr(
    req_id: cute.Tensor,
    block_table: cute.Tensor,
    token_indices: cute.Tensor,
    output: cute.Tensor,
    stride_factor: cutlass.Int32,
    layer_id: cutlass.Int32,
    stream: cuda.CUstream,
    NUM_TOPK_TOKENS: cutlass.Constexpr[int],
    MAX_BLOCKS_PER_REQ: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    base_rows: cutlass.Constexpr[bool],
) -> None:
    num_tokens = cute.size(req_id, mode=[0])
    tiles_per_row = cute.ceil_div(NUM_TOPK_TOKENS, _THREADS_PER_BLOCK)

    self_attn_req_index_to_global_rawptr(
        req_id,
        block_table,
        token_indices,
        output,
        stride_factor,
        layer_id,
        NUM_TOPK_TOKENS,
        MAX_BLOCKS_PER_REQ,
        use_pdl,
        base_rows,
    ).launch(
        grid=(num_tokens, tiles_per_row, 1),
        block=(_THREADS_PER_BLOCK, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def compile_req_index_rawptr(
    device_id: int = 0,
    num_topk_tokens: int = 2048,
    max_blocks_per_req: int = 192,
    *,
    use_pdl: bool = True,
    base_rows: bool = False,
) -> Callable:
    """Compile the raw-pointer req-index conversion kernel for GLM shapes.

    ``base_rows=True`` compiles the base-row variant (layer term dropped);
    the compiled callable keeps the same argument list and ignores ``layer_id``.
    """
    if num_topk_tokens % _THREADS_PER_BLOCK != 0:
        raise ValueError(
            f"num_topk_tokens must be divisible by {_THREADS_PER_BLOCK}, got {num_topk_tokens}"
        )
    if max_blocks_per_req <= 0:
        raise ValueError(f"max_blocks_per_req must be positive, got {max_blocks_per_req}")

    key = (
        device_id,
        num_topk_tokens,
        max_blocks_per_req,
        bool(use_pdl),
        bool(base_rows),
    )
    if key in _REQ_INDEX_RAWPTR_COMPILE_CACHE:
        return _REQ_INDEX_RAWPTR_COMPILE_CACHE[key]

    cutlass.cuda.initialize_cuda_context(device_id)

    sym_num_tokens = cute.sym_int32()
    sym_num_requests = cute.sym_int32()

    fake_req_id = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_tokens,),
        assumed_align=_INT32_ALIGNMENT_BYTES,
    )
    fake_block_table = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_requests, max_blocks_per_req),
        stride_order=(1, 0),
        assumed_align=_INT32_ALIGNMENT_BYTES,
    )
    fake_token_indices = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_tokens, num_topk_tokens),
        stride_order=(1, 0),
        assumed_align=_INT32_ALIGNMENT_BYTES,
    )
    fake_output = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_tokens, num_topk_tokens),
        stride_order=(1, 0),
        assumed_align=_INT32_ALIGNMENT_BYTES,
    )

    compiled = cute.compile(
        _launch_req_index_to_global_rawptr,
        fake_req_id,
        fake_block_table,
        fake_token_indices,
        fake_output,
        cutlass.Int32(0),
        cutlass.Int32(0),
        make_fake_stream(),
        num_topk_tokens,
        max_blocks_per_req,
        bool(use_pdl),
        bool(base_rows),
        options="--enable-tvm-ffi",
    )
    _REQ_INDEX_RAWPTR_COMPILE_CACHE[key] = compiled
    return compiled


def compile_req_index_base_rawptr(
    device_id: int = 0,
    num_topk_tokens: int = 2048,
    max_blocks_per_req: int = 192,
    *,
    use_pdl: bool = True,
) -> Callable:
    """Compile the raw-pointer layer-independent base-row kernel."""
    return compile_req_index_rawptr(
        device_id,
        num_topk_tokens,
        max_blocks_per_req,
        use_pdl=use_pdl,
        base_rows=True,
    )


def _require_int32_cuda_contiguous(torch, tensor, name: str) -> None:
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _current_cuda_stream(torch):
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def _validate_rawptr_inputs(
    torch,
    req_id,
    block_table,
    token_indices,
    block_size: int,
    output,
):
    """Shared validation of the raw-pointer modes; returns the output tensor."""
    _require_int32_cuda_contiguous(torch, req_id, "req_id")
    _require_int32_cuda_contiguous(torch, block_table, "block_table")
    _require_int32_cuda_contiguous(torch, token_indices, "token_indices")
    if req_id.ndim != 1:
        raise ValueError(f"req_id must be 1D, got shape {tuple(req_id.shape)}")
    if block_table.ndim != 2:
        raise ValueError(f"block_table must be 2D, got shape {tuple(block_table.shape)}")
    if token_indices.ndim != 2:
        raise ValueError(f"token_indices must be 2D, got shape {tuple(token_indices.shape)}")
    if req_id.shape[0] != token_indices.shape[0]:
        raise ValueError(
            "req_id length must match token_indices rows, "
            f"got {req_id.shape[0]} and {token_indices.shape[0]}"
        )
    if int(block_size) != _REQ_INDEX_BLOCK_SIZE:
        raise ValueError(
            f"raw-pointer req-index candidate specializes GLM block_size="
            f"{_REQ_INDEX_BLOCK_SIZE}, got {block_size}"
        )
    if token_indices.shape[1] % _THREADS_PER_BLOCK != 0:
        raise ValueError(
            f"token_indices columns must be divisible by {_THREADS_PER_BLOCK}, "
            f"got {token_indices.shape[1]}"
        )

    if output is None:
        output = torch.empty_like(token_indices)
    else:
        _require_int32_cuda_contiguous(torch, output, "output")
        if output.shape != token_indices.shape:
            raise ValueError(
                f"output shape must match token_indices, got {tuple(output.shape)} "
                f"and {tuple(token_indices.shape)}"
            )
    return output


def convert_req_index_to_global_rawptr(
    req_id,
    block_table,
    token_indices,
    block_size: int,
    stride_factor: int,
    layer_id: int,
    compiled_fn: Callable | None = None,
    output=None,
):
    """Run the raw-pointer CuTe req-index conversion kernel.

    This variant specializes the GLM serving shape and mirrors the CUDA backend's
    one-element-per-thread control flow. It intentionally rejects non-compact
    shapes instead of preserving the fully generic TRT-LLM op contract.
    """
    import torch

    output = _validate_rawptr_inputs(torch, req_id, block_table, token_indices, block_size, output)
    if compiled_fn is None:
        compiled_fn = compile_req_index_rawptr(
            device_id=torch.cuda.current_device(),
            num_topk_tokens=int(token_indices.shape[1]),
            max_blocks_per_req=int(block_table.shape[1]),
        )

    compiled_fn(
        req_id,
        block_table,
        token_indices,
        output,
        int(stride_factor),
        int(layer_id),
        _current_cuda_stream(torch),
    )
    return output


def convert_req_index_to_base_rawptr(
    req_id,
    block_table,
    token_indices,
    block_size: int,
    stride_factor: int,
    compiled_fn: Callable | None = None,
    output=None,
):
    """Run the raw-pointer layer-independent base-row kernel.

    Same inputs and validation as ``convert_req_index_to_global_rawptr`` minus
    ``layer_id``: the output holds base rows, and the attention kernel adds
    ``local_layer_idx * 32`` (``index_kind=INDEX_KIND_BASE_WITH_LAYER_OFFSET``).
    """
    import torch

    output = _validate_rawptr_inputs(torch, req_id, block_table, token_indices, block_size, output)
    if compiled_fn is None:
        compiled_fn = compile_req_index_base_rawptr(
            device_id=torch.cuda.current_device(),
            num_topk_tokens=int(token_indices.shape[1]),
            max_blocks_per_req=int(block_table.shape[1]),
        )

    compiled_fn(
        req_id,
        block_table,
        token_indices,
        output,
        int(stride_factor),
        0,  # layer_id: unused by the base_rows specialization
        _current_cuda_stream(torch),
    )
    return output
