# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's key and head-weight projection, written to two buffers in one launch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm import self_attn_indexer_pre_indexer_proj_m64_n160_k6144_tf32 as _kernel
from ...kernels.gemm import self_attn_prefill_projections_bf16 as _prefill

#: Decode row counts the target compiles this entry for.
ROWS = (1, 2, 4, 8, 16, 32, 48, 64)

#: Hidden 6144; 128 key columns, then 32 head-weight columns.
_K, _KEY, _HEADS = _kernel.K, _kernel.INDEXER_K_DIM, _kernel.INDEXER_WEIGHT_DIM
#: The prefill GEMM's entry for this projection: bf16 operands, fp32 `[M, 160]` out.
_PREFILL_ENTRY = "indexer_wk_weights"
#: The prefill handle serves every row count 1..PREFILL_MAX_ROWS.
PREFILL_MAX_ROWS = _prefill.MAX_ROWS


@dataclass(frozen=True)
class PrefillSplitHandle:
    """The prefill GEMM's `indexer_wk_weights` entry, any row count 1..`max_rows`.
    `launch(x, weight, out)` writes the fp32 `[M, 160]` product into `out`."""

    max_rows: int
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]


def compile(device_id: int, rows: int) -> _kernel.CompiledSplitOutputs:
    """Compile the bf16-operand, split-output kernel for exactly `rows` rows.

    Returns the handle the op takes as `compiled`. Costs one CuTe DSL compile
    (cached by the kernel module per row count) and one launch on zeros, so
    the module is loaded before any CUDA graph capture. Raises if the compile
    fails.
    """
    dev = torch.device("cuda", device_id)
    rows = int(rows)
    x = torch.zeros([rows, _K], dtype=torch.bfloat16, device=dev)
    weight = torch.zeros([_KEY + _HEADS, _K], dtype=torch.bfloat16, device=dev)
    key = torch.empty([rows, _KEY], dtype=torch.bfloat16, device=dev)
    head_weights = torch.empty([rows, _HEADS], dtype=torch.float32, device=dev)
    with torch.cuda.device(device_id):
        compiled = _kernel.compile_split_outputs(rows, "bf16")
        _kernel.run_split_outputs(compiled, x, weight, key, head_weights)
    return compiled


def build_prefill(weight: torch.Tensor) -> PrefillSplitHandle:
    """Build the context-row handle against `weight`, the bf16 `[160, 6144]` operand the
    decode handles take, of any one indexer layer. Call at load.

    The handle takes the weight per call, so one serves every indexer layer. Costs
    one CuTe DSL compile (one schedule serves every row count) and one launch on
    zeros. Raises if the vendored kernel has no such entry or the build fails.
    """
    dev = weight.device
    with torch.cuda.device(dev):
        launch = _prefill.build(_PREFILL_ENTRY, weight)
        x = torch.zeros([1, _K], dtype=torch.bfloat16, device=dev)
        launch(x, weight, torch.empty([1, _KEY + _HEADS], dtype=torch.float32, device=dev))
    return PrefillSplitHandle(PREFILL_MAX_ROWS, launch)


def compile_all(
    device_id: int, rows: tuple[int, ...] = ROWS
) -> dict[int, _kernel.CompiledSplitOutputs]:
    """`compile` at every row count in `rows`; `{rows: handle}` for those that
    compiled. A row count whose compile raises is left out, and the target
    sends it to the prefill handle or the fp32 `ix_fused` projection instead."""
    handles = {}
    for m in rows:
        try:
            handles[int(m)] = compile(device_id, m)
        except Exception:  # noqa: BLE001 -- that row count takes the fp32 projection
            continue
    return handles


class CuteDslIndexerPreProjSplit(OpWrapper):
    """`x @ weight.T` split by column into the indexer's two consumers.

    `x` is the bf16 hidden state `[M, 6144]`; `weight` the bf16 `[160, 6144]`
    row-major concatenation `[W_k (128 rows) | W_weights (32 rows)]`. Writes
    columns 0-127 to `key` (bf16 `[M, 128]`, the pre-norm indexer keys) and
    columns 128-159 to `head_weights` (fp32 `[M, 32]`, the per-head weights
    before the q-side scale). Products are exact in fp32 and accumulate in
    fp32 across an 8-CTA split-K cluster. Returns `(key, head_weights)`.

    With a `PrefillSplitHandle` the same product runs on the prefill GEMM (one
    fp32 accumulation per tile, no split-K) into an fp32 `[M, 160]` scratch,
    then two copies: the keys rounded to bf16, the head weights as they are.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The decode path's only caller: one indexer layer's projection per step,
    # at the step's row count (up to 32 sequences x MTP 2).
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row", spec=dict(rows=1)),
        Cell(why="MTP width 2, one sequence", spec=dict(rows=2)),
        Cell(why="a mid decode row count, a quarter of the tile", spec=dict(rows=16)),
        Cell(why="the full decode batch, one whole 64-row tile", spec=dict(rows=64)),
        Cell(
            why="prefill, 65 rows: the first context row count, one partial 128-row tile",
            spec=dict(rows=65, prefill=True),
        ),
        Cell(
            why="prefill, a 1099-row chunk that ends mid-tile",
            spec=dict(rows=1099, prefill=True),
        ),
        Cell(why="prefill at max_num_tokens, 8192 rows", spec=dict(rows=8192, prefill=True)),
    )

    note = """
    The target compiles one decode handle per row count in `ROWS` (1, 2, 4, 8,
    16, 32, 48, 64) and one prefill handle, which it uses above 64 rows. If the
    prefill build failed, those rows go to the fp32 `cute_dsl_projection`
    "ix_fused" kernel (or cuBLAS past it) at the target's call site.

    The decode kernel validates every call (dtypes, shapes, contiguity, 16-byte
    alignment, no aliasing between any two of the four tensors, row count equal
    to the handle's) and raises on a mismatch. The prefill GEMM checks `x` on
    every call, but the weight only the first time it meets its address, and
    the two copies cast to whatever `key` and `head_weights` are: `is_valid`
    covers that. Neither can check the weight's row order: keys first, head
    weights second.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        compiled: _kernel.CompiledSplitOutputs | PrefillSplitHandle,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(compiled, PrefillSplitHandle):
            full = torch.empty([x.shape[0], _KEY + _HEADS], dtype=torch.float32, device=x.device)
            compiled.launch(x, weight, full)
            key.copy_(full[:, :_KEY])
            head_weights.copy_(full[:, _KEY:])
        else:
            _kernel.run_split_outputs(compiled, x, weight, key, head_weights)
        return key, head_weights

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        compiled: _kernel.CompiledSplitOutputs | PrefillSplitHandle,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(key, head_weights)` as new tensors: the product in fp64, rounded
        once to each buffer's dtype. Neither buffer is written; `compiled` is
        ignored."""
        full = x.double() @ weight.double().t()
        return full[:, :_KEY].to(key.dtype), full[:, _KEY:].to(head_weights.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        key: torch.Tensor,
        head_weights: torch.Tensor,
        compiled: _kernel.CompiledSplitOutputs | PrefillSplitHandle,
    ) -> None:
        if not isinstance(compiled, PrefillSplitHandle):
            return  # the decode kernel checks every operand itself
        rows = int(x.shape[0])
        assert 1 <= rows <= compiled.max_rows, (
            f"the prefill handle serves 1..{compiled.max_rows} rows, got {rows}"
        )
        # A weight at an address the GEMM has met runs on its cached operand view.
        assert (
            tuple(weight.shape) == (_KEY + _HEADS, _K)
            and weight.is_contiguous()
            and weight.dtype == torch.bfloat16
        ), f"weight must be contiguous bf16 [{_KEY + _HEADS}, {_K}], got {tuple(weight.shape)}"
        # The copies cast silently; the scorers read `head_weights` densely as fp32.
        assert key.dtype == torch.bfloat16 and tuple(key.shape) == (rows, _KEY), (
            f"key must be bf16 [{rows}, {_KEY}], got {key.dtype} {tuple(key.shape)}"
        )
        assert (
            head_weights.dtype == torch.float32
            and tuple(head_weights.shape) == (rows, _HEADS)
            and head_weights.is_contiguous()
        ), (
            f"head_weights must be contiguous fp32 [{rows}, {_HEADS}], got "
            f"{head_weights.dtype} {tuple(head_weights.shape)}"
        )


cute_dsl_indexer_pre_proj_split = CuteDslIndexerPreProjSplit()
