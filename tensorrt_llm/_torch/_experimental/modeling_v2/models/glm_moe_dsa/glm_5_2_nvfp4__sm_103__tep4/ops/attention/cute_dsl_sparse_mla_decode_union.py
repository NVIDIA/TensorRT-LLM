# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP3 / MTP5 decode sparse MLA over the union of a query-row group's selections, top-k 2048,
on the CuTe DSL union kernel; the per-row op is `cute_dsl_sparse_mla_decode`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import (
    self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16 as _kernel,
)
from ...kernels.attention import (
    self_attn_fmha_sparse_mla_union_h16_qk576_v512_topk2048_fp8_bf16 as _union_kernel,
)
from .cute_dsl_req_index_to_global import pool_rows
from .cute_dsl_sparse_mla_decode import (
    INDEX_KIND_GLOBAL,
    LAYER_ROW_STRIDE,
    NUM_HEADS,
    TOP_K,
    cute_dsl_sparse_mla_decode,
)

#: Query rows per group of the union kernel, by handle width, for a batch `UNION_SERVED` does
#: not list: one CTA multiplies each Kv128 tile of the group's union against 16 x rows query
#: columns. (4-row groups no longer spill since the register split and chunked
#: correction; they pay where the rows' selections overlap little, see `note`.)
UNION_ROWS_PER_GROUP: Mapping[int, int] = {4: 2, 6: 3}
#: The kernel's own limit: above it a group's selections are taken back to back instead of
#: deduplicated (correct, but no fewer gathered rows than the per-row kernel).
UNION_MASK_CAP = _union_kernel.MASK_CAP


def _served(rows_per_group: int, batches: range) -> dict[int, int]:
    return dict.fromkeys(batches, rows_per_group)


#: The batches (sequences per step) the target sends to the union kernel and the rows per
#: group each takes, keyed by (SM count, max_sq); every other batch, and every batch on a
#: device without an entry, keeps the per-row kernel. From the per-batch sweep in `note`: a
#: batch is served where its best rows-per-group's union / per-row time ratio has a geometric
#: mean <= 0.95 over the five input regimes and is <= 1.2 in each.
UNION_SERVED: Mapping[tuple[int, int], Mapping[int, int]] = {
    (148, 6): {**_served(2, range(13, 16)), 24: 2, **_served(3, range(25, 33))},
    (148, 4): {
        **_served(2, range(19, 23)),
        **_served(4, range(23, 27)),
        **_served(2, range(30, 33)),
    },
}
#: The served batches whose plan keeps per-thread running softmax sums
#: (`UnionSpecialization.thread_sums`: the same band, another fp32 summation order), keyed as
#: `UNION_SERVED`: the batches added with the register split, measured with it. The batches served
#: before keep their kernel's arithmetic bit for bit.
UNION_THREAD_SUMS: Mapping[tuple[int, int], frozenset[int]] = {
    (148, 4): frozenset([*range(23, 27), *range(30, 33)]),
}


def union_batches(device_index: int, max_sq: int, max_batch_size: int) -> frozenset[int]:
    """The batches up to `max_batch_size` the union kernel serves at `max_sq` on this device."""
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    table = UNION_SERVED.get((sms, int(max_sq)), {})
    return frozenset(b for b in table if b <= int(max_batch_size))


@dataclass(frozen=True)
class CompiledSparseMlaDecodeUnion:
    """The union kernel's (rows per group, cluster chunks) pick for every batch of at most
    `max_batch_size` sequences at `max_sq` rows per sequence that takes it (`union_batches`
    unless the caller named the batches), each pick compiled. A batch without a plan takes the
    per-row kernel."""

    device_index: int
    max_batch_size: int
    max_sq: int
    plans: Mapping[int, tuple[int, int, bool]]


def union_plan(batch_size: int, *, device_index: int, max_sq: int) -> tuple[int, int, bool]:
    """`(rows_per_group, num_chunks, thread_sums)` for `batch_size` sequences of `max_sq` rows.

    Groups of the rows `UNION_SERVED` gives the batch on this device (else
    `UNION_ROWS_PER_GROUP[max_sq]`); each group's union is split over a cluster of
    `num_chunks` CTAs, the most (up to the kernel's 8) that keeps every group's cluster
    resident in one wave on this device (`batch x groups` clusters, one CTA per SM), and at
    least 2 for 4-row groups (SMEM). `thread_sums` for the batches `UNION_THREAD_SUMS` lists.
    Host only; reads the device's SM count and cluster occupancy, both cached.
    """
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    served = UNION_SERVED.get((sms, int(max_sq)), {})
    rg = served.get(int(batch_size), UNION_ROWS_PER_GROUP[int(max_sq)])
    groups = (int(max_sq) + rg - 1) // rg
    c = max(1, min(_union_kernel.MAX_CHUNKS, sms // (int(batch_size) * groups)))
    while c > 1 and int(batch_size) * groups > _kernel._max_active_clusters(device_index, c):
        c -= 1
    if rg == 4:
        c = max(c, 2)
    ts = int(batch_size) in UNION_THREAD_SUMS.get((sms, int(max_sq)), frozenset())
    return rg, c, ts


def compile_union(
    *,
    max_batch_size: int,
    device_index: int,
    max_sq: int,
    batches: Iterable[int] | None = None,
) -> CompiledSparseMlaDecodeUnion:
    """Compile the union pick of every batch of 1..`max_batch_size` sequences at `max_sq`
    (4 or 6) rows per sequence that the target serves (`union_batches`), or of `batches`
    when given (the kernel answers any batch; tests certify them all); return the handle.
    Call once, outside any CUDA-graph capture."""
    if int(max_sq) not in UNION_ROWS_PER_GROUP:
        raise ValueError(
            f"the union kernel serves max_sq {sorted(UNION_ROWS_PER_GROUP)}, got {max_sq}"
        )
    if batches is None:
        batches = union_batches(int(device_index), int(max_sq), int(max_batch_size))
    plans = {
        b: union_plan(b, device_index=int(device_index), max_sq=int(max_sq))
        for b in sorted(int(v) for v in batches)
        if 1 <= b <= int(max_batch_size)
    }
    for rg, c, ts in sorted(set(plans.values())):
        _union_kernel.compile_union(
            _union_kernel.UnionSpecialization(rows_per_group=rg, num_chunks=c, thread_sums=ts),
            int(device_index),
        )
    return CompiledSparseMlaDecodeUnion(
        device_index=int(device_index),
        max_batch_size=int(max_batch_size),
        max_sq=int(max_sq),
        plans=plans,
    )


def _union_rows(
    topk_positions: torch.Tensor,
    block_table: torch.Tensor,
    req_idx: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    sequence_length: torch.Tensor,
    layer_id: int,
    stride_factor: int,
) -> torch.Tensor:
    """The global pool rows the union kernel attends per query row: entry j of row t (the
    t-th of its sequence's `sq` rows, KV length n) when j < clamp(n - sq + t + 1, 0, 2048)
    and the position is non-negative, else -1."""
    rows = pool_rows(
        req_idx[: topk_positions.shape[0]], block_table, topk_positions,
        LAYER_ROW_STRIDE, int(stride_factor), int(layer_id) * LAYER_ROW_STRIDE,
    ).long()  # fmt: skip
    cu = (cu_q_seqlens.long() // NUM_HEADS).tolist()
    lens = sequence_length.long().tolist()
    prefix = torch.zeros(topk_positions.shape[0], dtype=torch.int64)
    for b, n in enumerate(lens):
        sq = cu[b + 1] - cu[b]
        for t in range(sq):
            prefix[cu[b] + t] = min(max(n - sq + t + 1, 0), TOP_K)
    keep = torch.arange(TOP_K)[None, :] < prefix[:, None]
    return torch.where(keep.to(rows.device), rows, torch.full_like(rows, -1))


class CuteDslSparseMlaDecodeUnion(OpWrapper):
    """Absorbed sparse MLA for MTP3 / MTP5 decode rows, over each query-row group's union.

    The same attention as `cute_dsl_sparse_mla_decode`, given the indexer's request-local
    top-k positions instead of pool rows: `topk_positions` `[T, 2048]` int32, `block_table`
    `[requests, blocks]` int32, `req_idx` `[T]` int32 (row t's block-table row), the pool
    layer `layer_id` and the pool rows per block `stride_factor`. Query row t (the t-th of
    its sequence's `sq` rows, KV length n) attends the pool rows

        block_table[req_idx[t], p // 32] * stride_factor + p % 32 + layer_id * 32

    of the positions p among its first clamp(n - sq + t + 1, 0, 2048) entries:

        output[t, h] = softmax_r(q[t, h] . kv[r] / 16) @ kv[r, :512],  r in those rows

    The kernel gathers each pool row a group of `UNION_ROWS_PER_GROUP` rows selected once
    (positions deduplicated in SMEM, in position order) and masks it per row; the rows' own
    summation order is not kept, so outputs match the per-row kernel within the same band,
    not bitwise (nor across `thread_sums`, which `UNION_THREAD_SUMS` sets per batch). Returns
    `output`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP5 trunk decode, 32 sequences x 6, lengths around the census window "
            "(~3000), the 3-row groups' 2-chunk clusters",
            spec=dict(p=6, lens=[2850 + 7 * i for i in range(32)], layer=1),
        ),
        Cell(
            why="MTP3, 32 sequences x 4, lengths 3000..6007",
            spec=dict(p=4, lens=[3000 + 97 * i for i in range(32)], layer=2),
        ),
        Cell(
            why="MTP5 trunk decode, 14 sequences x 6, lengths crossing top_k: the 2-row "
            "groups' 3-chunk clusters the target serves at 13..15",
            spec=dict(p=6, lens=[2049 + 211 * i for i in range(13)] + [7], layer=0),
        ),
        Cell(
            why="MTP5, lengths crossing top_k both ways and below the group width: 6, 1..5, "
            "2045..2053, 6000",
            spec=dict(
                p=6, lens=[6, 1, 2, 5, 2045, 2047, 2048, 2049, 2053, 6000, 100, 4097], layer=0
            ),
        ),
        Cell(
            why="MTP3, lengths crossing top_k, 13 sequences (a wider cluster split)",
            spec=dict(
                p=4,
                lens=[4, 3, 2047, 2048, 2049, 2050, 5000] + [700 * i + 9 for i in range(6)],
                layer=1,
            ),
        ),
        Cell(
            why="MTP5 at 1 sequence: the widest cluster split (8 chunks per group)",
            spec=dict(p=6, lens=[5000], layer=2),
        ),
        Cell(
            why="MTP5 at 3 sequences",
            spec=dict(p=6, lens=[9, 2050, 3333], layer=1),
        ),
        Cell(
            why="MTP5, a runtime partial batch: sequences of 6, 1, 3, 6, 4, 2 rows",
            spec=dict(p=6, lens=[3000, 50, 2049, 7000, 6, 3], widths=[6, 1, 3, 6, 4, 2], layer=1),
        ),
        Cell(
            why="MTP5, a context longer than the kernel's position map: the selections back "
            "to back instead of deduplicated",
            spec=dict(p=6, lens=[UNION_MASK_CAP + 37, 3000], layer=0),
        ),
        Cell(
            why="MTP5, 25 and 28 sequences",
            spec=dict(p=6, lens=[6 + 307 * i for i in range(25)], layer=2),
        ),
        Cell(
            why="MTP3 trunk decode, 21 sequences (served at 19..22)",
            spec=dict(p=4, lens=[5 + (509 * i) % 4400 for i in range(21)], layer=0),
        ),
        Cell(
            why="MTP3 at 24 sequences, the served 4-row groups on 5-chunk clusters with "
            "per-thread sums: lengths 1..6, 2045..2053, 6000 and past the position map",
            spec=dict(
                p=4,
                lens=[1, 2, 3, 4, 5, 6, *range(2045, 2054), 6000, UNION_MASK_CAP + 37]
                + [2900 + 13 * i for i in range(7)],
                layer=2,
            ),
        ),
        Cell(
            why="MTP3 at 32 sequences, the served 2-row groups with per-thread sums: a runtime "
            "partial batch (sequences of 1..3 rows) and lengths crossing top_k",
            spec=dict(
                p=4,
                lens=[2045 + i for i in range(9)]
                + [3, 4, 6000]
                + [100 + 211 * i for i in range(20)],
                widths=[4, 1, 2, 3, 4, 4, 2, 4, 4, 3, 4, 4] + [4] * 20,
                layer=1,
            ),
        ),
        Cell(
            why="MTP3 at 16 sequences, the CUDA-graph batch the target keeps on the per-row "
            "kernel: the union kernel's default plan (2-row groups, 4 chunks)",
            spec=dict(p=4, lens=[16 + 431 * i for i in range(16)], layer=0),
        ),
    )

    note = """
    The target builds the handle (`compile_union`) next to the MTP3 / MTP5 per-row handle
    and passes it on the trunk's pure decode steps at 4 / 6 rows per sequence and a batch
    `union_batches` serves: the indexer's positions go straight to this op, so those steps
    skip the base-row conversion. Other batches, the MTP layer, mixed steps and the 1-row
    draft steps keep the per-row kernels.

    Measured (B300, CUDA-graph replay of 8 calls on 8 pool layers, per call, against the
    per-row V-reuse kernel on its own pick, split-K 2 clusters at MTP5 25..32; two
    sweep jobs). Lengths per regime: A 2850..3067 (the served benchmark's census), B
    1500..4073, C/D/E 1024..8960; selections random over each row's causal prefix,
    independent per row (A, B, C) or with 50 % (D) / 80 % (E) of a row's 2048 drawn from one
    per-sequence set (consecutive draft rows' overlap). Union / per-row time:

        batches            rows/group    A          B          C          D          E
        MTP5 25..32 (*)    3             .67-.70    .76-.81    .92-1.19   .79-1.02   .65-.84
        MTP5 24 (*)        2             .95        .98        1.07       .92        .83
        MTP5 13..15 (*)    2             .80        .79        .99        .89-.90    .84-.85
        MTP5 16..23        3 at 16, 2    .86-.97    .85-.99    1.12-1.22  .97-1.06   .88-.96
        MTP3 19..22 (*)    2             .82-.83    .78-.82    .97-.99    .87-.89    .82-.83
        MTP3 23..32        2             .95-.96    .97-.99    1.13-1.21  1.01-1.05  .91-.94
        (*) served. MTP5 at 32: 49.4 -> 33.3 us per call in A, 53.8 -> 49.2 in C.

    MTP5 below 13 and MTP3 below 19 (one-wave per-row grids) measured 0.89 to 2.0 (same
    sweeps), never faster in A: not served. See the kernel module for what bounds each.

    A later rework of the kernel (register split, chunked correction, max.f32 maxima,
    -inf masking: bit for bit the same outputs at every plan above) and added per-thread sums
    (`thread_sums`). The per-batch sweep on it (same regimes and method),
    MTP3, union / per-row, geomean over A..E and the worst regime:

        batches         plan                       A          C            geomean   max
        MTP3 16..18     2 rows, 3-4 chunks, sums   1.02-1.16  1.27-1.39    1.10-1.20 1.27-1.39
        MTP3 19..22 (*) 2 rows, 3 chunks           .78-.79    .94-.95      .82       .94-.95
        MTP3 23..26 (*) 4 rows, 5 chunks, sums     .83        1.08-1.13    .91-.92   1.08-1.13
        MTP3 27..29     2 rows, 2 chunks, sums     .89-.90    1.10-1.11    .952-.956 1.10-1.11
        MTP3 30..32 (*) 2 rows, 2 chunks, sums     .89        1.08-1.10    .939-.946 1.08-1.10
        (*) served; 19..22 kept the plan (and the arithmetic) they had. MTP3 at 32 in A:
        28.9 -> 25.6 us per call; at 24, 27.4 -> 22.8. 4-row groups at 27..32 (4 chunks)
        reach .84-.86 in A but 1.17-1.21 in C (geomean .98-.99); the CUDA-graph batch 16
        stays on the per-row kernel, whose split-K 2 fits one wave there.

    Not certified: 8 or 32 heads, `max_sq` other than 4 and 6, tokens per block other than
    32, a selection with repeated positions in a row (the union counts a position once).
    """

    def __call__(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        topk_positions: torch.Tensor,
        block_table: torch.Tensor,
        req_idx: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        sequence_length: torch.Tensor,
        layer_id: int,
        stride_factor: int,
        compiled: CompiledSparseMlaDecodeUnion,
    ) -> torch.Tensor:
        batch = int(sequence_length.shape[0])
        if (
            batch not in compiled.plans
            or int(quant_q_buffer.shape[1]) != NUM_HEADS
            or quant_q_buffer.device.index != compiled.device_index
        ):
            raise ValueError(
                f"no union kernel compiled for {batch} sequences, {quant_q_buffer.shape[1]} "
                f"heads on {quant_q_buffer.device}"
            )
        rg, chunks, thread_sums = compiled.plans[batch]
        _union_kernel.prepare_union_runner(
            quant_q_buffer,
            kv_pool_view,
            topk_positions,
            block_table,
            req_idx,
            output,
            cu_q_seqlens,
            sequence_length,
            layer_offset=int(layer_id) * LAYER_ROW_STRIDE,
            stride_factor=int(stride_factor),
            max_sq=compiled.max_sq,
            rows_per_group=rg,
            num_chunks=chunks,
            thread_sums=thread_sums,
        )()
        return output

    def reference(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        topk_positions: torch.Tensor,
        block_table: torch.Tensor,
        req_idx: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        sequence_length: torch.Tensor,
        layer_id: int,
        stride_factor: int,
        compiled: CompiledSparseMlaDecodeUnion,
    ) -> torch.Tensor:
        """The per-row op's fp64 reference over the rows `_union_rows` names."""
        rows = _union_rows(
            topk_positions, block_table, req_idx, cu_q_seqlens, sequence_length, layer_id,
            stride_factor,
        )  # fmt: skip
        return cute_dsl_sparse_mla_decode.reference(
            quant_q_buffer, kv_pool_view, rows.to(torch.int32), output, cu_q_seqlens,
            sequence_length, None, None, INDEX_KIND_GLOBAL, None, None,
        )  # fmt: skip

    def is_valid(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        topk_positions: torch.Tensor,
        block_table: torch.Tensor,
        req_idx: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        sequence_length: torch.Tensor,
        layer_id: int,
        stride_factor: int,
        compiled: CompiledSparseMlaDecodeUnion,
    ) -> None:
        """The inputs the kernel takes and answers wrongly. Reads the selection back to the
        host, which is why it is never on the serving path."""
        total_q, heads = int(quant_q_buffer.shape[0]), int(quant_q_buffer.shape[1])
        cu = cu_q_seqlens.tolist()
        lens = sequence_length.tolist()
        assert cu[0] == 0 and cu[-1] == heads * total_q, "cu_q_seqlens must run 0..heads x rows"
        sq = [(b - a) // heads for a, b in zip(cu, cu[1:])]
        assert all(0 <= n <= compiled.max_sq for n in sq), (
            f"a sequence carries more rows than the handle's max_sq {compiled.max_sq}"
        )
        pos = topk_positions.long().cpu()
        req = req_idx.long().cpu()
        for b, (start, n) in enumerate(zip(cu, sq)):
            first = start // heads
            assert bool((req[first : first + n] == req[first]).all()), (
                "the rows of one sequence must share one block-table row"
            )
            for t in range(n):
                valid = min(max(lens[b] - n + t + 1, 0), TOP_K)
                row = pos[first + t]
                head, tail = row[:valid], row[valid:]
                assert bool((tail == -1).all()), (
                    "a position after the row's valid prefix; the kernel ignores it"
                )
                assert bool(((head >= 0) & (head < lens[b])).all()), (
                    "a position outside [0, length) inside the valid prefix"
                )
                assert head.unique().numel() == head.numel(), (
                    "a repeated position in a row; the union counts it once"
                )
        assert (
            int(block_table.max()) * int(stride_factor) + (int(layer_id) + 1) * LAYER_ROW_STRIDE
            < 2**31
        ), "pool rows past int32; the kernel computes them in int32"


cute_dsl_sparse_mla_decode_union = CuteDslSparseMlaDecodeUnion()
