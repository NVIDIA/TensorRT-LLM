# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 DSA prefill scoring over FP8 queries and gathered FP8 keys, on the tensor cores."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

from .._op import Arch, Cell, OpWrapper
from . import _fp8_mqa_logits_prefill_kernel as _kernel
from .fp8_mqa_logits import fp8_mqa_logits

#: The fewest key blocks (of 128) one work item scores: an item reloads its 4 query rows
#: (16 KiB of queries), so shorter splits spend more on queries than on keys.
_MIN_SPLIT_BLOCKS = 4
#: An item's fixed cost (its queries' load and the pipeline's turn), in key blocks.
_ITEM_COST_BLOCKS = 2
#: The most work items per CTA a split is chosen among.
_MAX_ITEMS_PER_CTA = 32


@dataclass(frozen=True)
class CompiledFp8MqaLogits:
    """A compiled launcher with the device facts its call needs."""

    fn: object
    device_id: int
    sm_count: int


_COMPILED: dict[tuple, CompiledFp8MqaLogits] = {}


@functools.lru_cache(maxsize=4096)
def kv_blocks_per_split(num_q_blocks: int, num_kv_blocks: int, sm_count: int) -> int:
    """Key blocks per work item. The CTAs take items round robin, so the busiest one runs
    `ceil(items / sm_count)` of them; the split minimizes that count times an item's cost
    (its key blocks plus `_ITEM_COST_BLOCKS`), as if every item scored its whole split, over
    splits of at least `_MIN_SPLIT_BLOCKS` and at most `_MAX_ITEMS_PER_CTA` items per CTA. A
    call's tile shapes repeat over the layers of a step, hence the cache."""
    best, best_cost = num_kv_blocks, None
    splits = 1
    while splits <= num_kv_blocks:
        per = -(-num_kv_blocks // splits)
        if splits > 1 and per < _MIN_SPLIT_BLOCKS:
            break
        items = num_q_blocks * splits
        if splits > 1 and items > sm_count * _MAX_ITEMS_PER_CTA:
            break
        cost = -(-items // sm_count) * (per + _ITEM_COST_BLOCKS)
        if best_cost is None or cost < best_cost:
            best, best_cost = per, cost
        # The next split count that shortens the split.
        splits = -(-num_kv_blocks // (per - 1)) if per > 1 else num_kv_blocks + 1
    return best


def build(device_id: int, **options) -> CompiledFp8MqaLogits:
    """Compile the kernel and launch it once, ahead of any served step.

    Every extent is symbolic (rows, keys, the logits row stride), so one handle serves every
    call on the device. A few seconds of compile, cached per process. `options` are the
    kernel's constructor arguments; the defaults are the measured best."""
    device_id = int(device_id)
    key = (device_id, tuple(sorted(options.items())))
    if key in _COMPILED:
        return _COMPILED[key]
    with torch.cuda.device(device_id):
        sym = cute.sym_int
        fake_k = make_fake_compact_tensor(
            cutlass.Uint8, (sym(), _kernel.HEAD_DIM), stride_order=(1, 0), assumed_align=16
        )
        fake_q = make_fake_compact_tensor(
            cutlass.Uint8, (sym(), _kernel.HEAD_DIM), stride_order=(1, 0), assumed_align=16
        )
        fake_w = make_fake_compact_tensor(cutlass.Float32, (sym(),), assumed_align=16)
        fake_s = make_fake_compact_tensor(cutlass.Float32, (sym(),), assumed_align=16)
        fake_ks = make_fake_compact_tensor(cutlass.Int32, (sym(),), assumed_align=4)
        fake_ke = make_fake_compact_tensor(cutlass.Int32, (sym(),), assumed_align=4)
        fake_out = make_fake_tensor(
            cutlass.Float32, (sym(), sym()), stride=(cute.sym_int64(), 1), assumed_align=4
        )
        fn = cute.compile(
            _kernel.Fp8MqaLogitsPrefillKernel(**options),
            fake_k,
            fake_q,
            fake_w,
            fake_s,
            fake_ks,
            fake_ke,
            fake_out,
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        sm_count = torch.cuda.get_device_properties(device_id).multi_processor_count
        handle = CompiledFp8MqaLogits(fn, device_id, int(sm_count))
        q = torch.zeros((4, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
        k = torch.zeros((128, 128), dtype=torch.float8_e4m3fn, device="cuda")
        scale = torch.ones(128, dtype=torch.float32, device="cuda")
        weights = torch.ones((4, 32), dtype=torch.float32, device="cuda")
        ks = torch.zeros(4, dtype=torch.int32, device="cuda")
        ke = torch.full((4,), 128, dtype=torch.int32, device="cuda")
        cute_dsl_fp8_mqa_logits(q, (k, scale), weights, ks, ke, compiled=handle)
        torch.cuda.synchronize()
    _COMPILED[key] = handle
    return handle


class CuteDslFp8MqaLogits(OpWrapper):
    """FP32 sum of weighted, rectified FP8 dot products, masked by each query's window.

    `triton_fp8_mqa_logits`' contract, on the tensor cores:

        logits[r, j] = scale[j] * sum_h weights[r, h] * relu(q[r, h, :] . k[j, :])

    for `cu_seq_len_k_start[r] <= j < cu_seq_len_k_end[r]`, and -inf at every other `j < n_k`.
    The products are exact and accumulate in FP32 in tensor memory; the key's FP32 scale is
    applied after the head reduction. Returns the `[rows, n_k]` view of a buffer whose rows are
    padded to 16 bytes for the prefill top-k consumer.

    The kernel scores, per 4 query rows, only the 128-key blocks that meet the union of the
    rows' windows and stores -inf over the rest, so a causal chunk over a prefix, or a block
    holding several requests' rows, costs MMAs for its windows only.

    Inputs: `q` `[rows, 32, 128]` e4m3, `k` `[n_k, 128]` e4m3, `scale` `[n_k]` fp32, `weights`
    `[rows, 32]` fp32 (the q-side scale folded in), windows `[rows]` int32. Non-contiguous
    weights, scales and windows are copied; `q` and `k` must hold contiguous rows. `compiled` is
    `build(device)`'s handle.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The SM107 targets' calls (`_select_ctx` of the dep4, tep4 and tp4 targets): one per query
    # tile of an indexer prefill block, at most the logits budget (2**26) per call. Shapes are
    # `m` query rows over `n` keys; `causal` gives row r of a chunk of `m` rows after a prefix
    # of `n - m` keys the window [0, n - m + r + 1), `packed` splits the rows and keys into two
    # requests' causal chunks, and otherwise the windows are the scattered ones of the
    # Triton scorer's cells.
    CELLS = tuple(
        Cell(f"SM107 indexer rows={m}, keys={n}", dict(m=m, n=n))
        for m in (1, 17, 129, 2048)
        for n in (33, 2051, 4096)
    ) + (
        Cell("SM107 strided scales and windows", dict(m=17, n=2051, strided=True)),
        Cell("SM107 32K context tile", dict(m=512, n=32768)),
        Cell("SM107 1M context tile", dict(m=64, n=1048576)),
        Cell("SM107 66-row chunk on a 100K prefix", dict(m=66, n=100066, causal=True)),
        Cell("SM107 515-row chunk on a 100K prefix", dict(m=515, n=100515, causal=True)),
        Cell("SM107 2000-row chunk on a 100K prefix, one tile", dict(m=657, n=102000, causal=True)),
        Cell("SM107 16K-row chunk on a 36K prefix, one tile", dict(m=1290, n=52116, causal=True)),
        Cell("SM107 two requests in one block", dict(m=1031, n=40000, packed=True)),
    )

    def __call__(
        self,
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        *,
        compiled: CompiledFp8MqaLogits,
    ) -> torch.Tensor:
        k, scale = kv
        if q.shape[1:] != (32, 128) or k.ndim != 2 or k.shape[1] != 128:
            raise ValueError("SM107 MQA scorer requires 32 query heads and head dimension 128")
        if q.dtype != torch.float8_e4m3fn or k.dtype != torch.float8_e4m3fn:
            raise ValueError("SM107 MQA scorer requires FP8 E4M3 queries and keys")
        rows, keys = q.shape[0], k.shape[0]
        padded = -(-keys // 4) * 4
        output = torch.empty((rows, padded), dtype=torch.float32, device=q.device)[:, :keys]
        if not output.numel():
            return output
        if q.stride() != (4096, 128, 1) or k.stride() != (128, 1):
            raise ValueError("SM107 MQA scorer requires contiguous query and key rows")
        weights = weights.contiguous()
        scale = scale.contiguous()
        if (
            q.data_ptr() % 16
            or k.data_ptr() % 16
            or weights.data_ptr() % 16
            or scale.data_ptr() % 16
        ):
            raise ValueError(
                "SM107 MQA scorer requires 16-byte aligned queries, keys, weights, scales"
            )
        q_blocks = -(-rows // _kernel.BLOCK_Q)
        kv_blocks = -(-keys // _kernel.BLOCK_KV)
        per_split = kv_blocks_per_split(q_blocks, kv_blocks, compiled.sm_count)
        splits = -(-kv_blocks // per_split)
        compiled.fn(
            k.view(torch.uint8),
            q.view(torch.uint8).view(rows * 32, 128),
            weights.view(-1),
            scale,
            cu_seq_len_k_start.contiguous(),
            cu_seq_len_k_end.contiguous(),
            output,
            rows,
            keys,
            per_split,
            splits,
            min(q_blocks * splits, compiled.sm_count),
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        kv: tuple[torch.Tensor, torch.Tensor],
        weights: torch.Tensor,
        cu_seq_len_k_start: torch.Tensor,
        cu_seq_len_k_end: torch.Tensor,
        *,
        compiled: CompiledFp8MqaLogits | None = None,
    ) -> torch.Tensor:
        """`fp8_mqa_logits.reference`: fp64, rounded once to fp32, -inf outside the windows.
        `compiled` is not read."""
        return fp8_mqa_logits.reference(q, kv, weights, cu_seq_len_k_start, cu_seq_len_k_end)


cute_dsl_fp8_mqa_logits = CuteDslFp8MqaLogits()
