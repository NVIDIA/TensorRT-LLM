# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode sparse MLA over the fp8 latent pool, top-k 2048, on the CuTe DSL kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import (
    self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16 as _kernel,
)

#: `sparse_attn_indices` holds global pool rows, used as they are.
INDEX_KIND_GLOBAL = _kernel.INDEX_KIND_GLOBAL
#: `sparse_attn_indices` holds layer-independent base rows; the kernel adds
#: `local_layer_idx * 32` to every non-negative entry.
INDEX_KIND_BASE_WITH_LAYER_OFFSET = _kernel.INDEX_KIND_BASE_WITH_LAYER_OFFSET
#: Rows between consecutive layers of one page in the pool.
LAYER_ROW_STRIDE = _kernel.KV_POOL_TOKENS_PER_BLOCK
TOP_K = _kernel.TOPK
NUM_HEADS = _kernel.NUM_HEADS
KV_LORA_RANK = _kernel.L
LATENT_DIM = _kernel.KV_STRIDE
#: The per-sequence query count the kernel's grid is sized for: MTP 2.
MAX_SQ = 2
#: The softmax scale the kernel compiles in: 1 / (q_scaling * sqrt(nope + rope))
#: with q_scaling 1 and GLM's 192 + 64.
SOFTMAX_SCALE = 1.0 / math.sqrt(_kernel.QK_HEAD_DIM)


@dataclass(frozen=True)
class CompiledSparseMlaDecode:
    """Every split-K variant a batch of 1..`max_batch_size` sequences reaches,
    per index kind, keyed by the specialization the kernel selects."""

    device_index: int
    max_batch_size: int
    max_sq: int
    all_sparse_indices_valid: bool
    kernels: Mapping[_kernel.MlaFmhaSpecialization, object]

    @property
    def index_kinds(self) -> frozenset[str]:
        return frozenset(spec.index_kind for spec in self.kernels)


def compile(  # noqa: A001 -- the kernel module's own name for this step
    *,
    max_batch_size: int,
    device_index: int,
    index_kinds: tuple[str, ...] = (INDEX_KIND_GLOBAL, INDEX_KIND_BASE_WITH_LAYER_OFFSET),
    max_sq: int = MAX_SQ,
    all_sparse_indices_valid: bool = False,
) -> CompiledSparseMlaDecode:
    """Compile every variant a decode batch can select; return the handle.

    The kernel picks its split-K CTA count (1, 2, 4 or 8) and its reducer from
    the batch size, `max_sq` and the SM count (`split_k_variant`), and
    compiles each pick separately. This walks batch sizes 1..`max_batch_size`
    for each index kind and builds every distinct pick, so no batch up to the
    bound compiles later. The walk is host arithmetic; the cost is the
    compiles, at most seven per index kind whatever the bound (splits 8, 4 and
    2 with or without the global-memory reducer, and the single-CTA split).
    Call once, outside any CUDA-graph capture, with the engine's largest
    per-step sequence count.
    """
    kernels = {}
    for kind in index_kinds:
        specs = _kernel.precompile_for_serving(
            max_batch_size=int(max_batch_size),
            all_sparse_indices_valid=bool(all_sparse_indices_valid),
            max_sq=int(max_sq),
            device_index=int(device_index),
            index_kind=kind,
        )
        for spec in specs:
            # Already built by the walk above; this returns the cached kernel.
            kernels[spec] = _kernel.compile(
                all_sparse_indices_valid=spec.all_sparse_indices_valid,
                num_heads=spec.num_heads,
                q_scaling=spec.q_scaling,
                num_ctas_kv=spec.num_ctas_kv,
                use_gmem_reduction=spec.use_gmem_reduction,
                device_index=int(device_index),
                index_kind=spec.index_kind,
            )
    return CompiledSparseMlaDecode(
        device_index=int(device_index),
        max_batch_size=int(max_batch_size),
        max_sq=int(max_sq),
        all_sparse_indices_valid=bool(all_sparse_indices_valid),
        kernels=kernels,
    )


def split_k_variant(
    batch_size: int, *, device_index: int, max_sq: int = MAX_SQ
) -> tuple[int, bool]:
    """`(num_ctas_kv, use_gmem_reduction)`: the split-K CTA count and reducer the
    kernel selects for `batch_size` sequences on this device.

    The same rule `compile`'s walk and the per-call runner apply, with the KV
    length it reads when no host lengths are passed (top_k). Host only: it
    reads the device's SM count and cluster occupancy, both cached.
    """
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    ctas = _kernel._select_num_ctas_kv(
        batch_size=int(batch_size),
        max_sq=int(max_sq),
        max_seq_len_kv=TOP_K,
        multiprocessor_count=sms,
    )
    gmem = ctas > 1 and _kernel._select_gmem_reduction(
        batch_size=int(batch_size),
        max_sq=int(max_sq),
        max_active_clusters=_kernel._max_active_clusters(int(device_index), ctas),
    )
    return ctas, bool(gmem)


def _pool_bytes(kv_pool_view: torch.Tensor) -> torch.Tensor:
    """The pool as `[rows, 576]` e4m3 bytes, whatever its own shape. Bytes, so
    that rows can be gathered; view the gathered rows as e4m3."""
    return kv_pool_view.reshape(-1).view(torch.uint8).reshape(-1, LATENT_DIM)


class CuteDslSparseMlaDecode(OpWrapper):
    """Absorbed sparse MLA for the generation rows, into `output`.

    `quant_q_buffer` is `[T, 16, 576]` e4m3 (`[q_nope_absorbed | q_rope]`),
    `kv_pool_view` the fp8 latent pool (any contiguous shape, read as
    `[rows, 576]`), `sparse_attn_indices` `[T, 2048]` int32 pool rows,
    `output` `T * 16 * 512` bf16 (any contiguous shape). Sequence `b` owns
    query rows `cu_q_seqlens[b] / 16 .. cu_q_seqlens[b+1] / 16` (cumulative
    head rows, as `mla_rope_generation` writes them) and `sequence_length[b]`
    KV tokens. For query row `q` with selected rows `R` (its non-negative
    entries, plus `local_layer_idx * 32` under the base index kind):

        output[q, h] = softmax_r(s * q[q, h] . kv[r]) @ kv[r, :512] * bmm2,  r in R

    with `s = mla_bmm1_scale[0]` and `bmm2 = mla_bmm2_scale[0]`. Returns
    `output`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this for every generation row: on a pure decode step
    # with base rows at MTP 2 and global rows at MTP 1, in the MTP draft layer
    # and in a mixed batch with global rows. Lengths cross top_k both ways, so
    # some query rows have -1 tails and the two tokens of an MTP sequence can
    # have different valid prefixes. The batch sizes walk the kernel's own
    # split-K picks on the device SM count (`split_k_variant`): the widest
    # split for 2-3 sequences, mid and narrow splits at 12 and 32, the
    # global-memory reducer at the smallest batch that selects it for each
    # clustered split, and the single-CTA split of a batch whose query CTAs
    # exceed the SM count. A `gmem_split` cell names no lengths: its test finds
    # the batch on the device it runs on, and skips when no batch of that
    # split leaves one occupancy wave.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP 1, global rows, 2 sequences: one longer than top_k, one with a "
            "-1 tail; the widest split-K",
            spec=dict(p=1, lens=[3000, 700], index_kind="global", layer=1),
        ),
        Cell(
            why="MTP 2, global rows: lengths 2049, 2048 and 5, so the two tokens of a "
            "sequence see 2048/2048, 2047/2048 and 4/5 valid rows",
            spec=dict(p=2, lens=[2049, 2048, 5], index_kind="global", layer=0),
        ),
        Cell(
            why="MTP 2, base rows plus layer 2's offset: the pure-decode trunk path, "
            "12 sequences, a mid split-K",
            spec=dict(
                p=2,
                lens=[33 + 347 * i for i in range(12)],
                index_kind="base_with_layer_offset",
                layer=2,
            ),
        ),
        Cell(
            why="32 sequences x MTP 2, base rows: the recipe's max_batch_size, a narrow split-K",
            spec=dict(
                p=2,
                lens=[100 + 131 * i for i in range(32)],
                index_kind="base_with_layer_offset",
                layer=1,
            ),
        ),
        Cell(
            why="the global-memory split-K reducer at the 8-CTA split: the smallest "
            "batch whose 8-CTA clusters exceed one occupancy wave, MTP 2, base rows",
            spec=dict(p=2, gmem_split=8, index_kind="base_with_layer_offset", layer=2),
        ),
        Cell(
            why="the global-memory split-K reducer at the 4-CTA split: the smallest "
            "batch whose 4-CTA clusters exceed one occupancy wave, MTP 2, global rows",
            spec=dict(p=2, gmem_split=4, index_kind="global", layer=0),
        ),
        Cell(
            why="the global-memory split-K reducer at the 2-CTA split: the smallest "
            "batch whose 2-CTA clusters exceed one occupancy wave, MTP 2, base rows",
            spec=dict(p=2, gmem_split=2, index_kind="base_with_layer_offset", layer=1),
        ),
        Cell(
            why="128 sequences x MTP 1, global rows: a batch above the recipe's, "
            "single-CTA split once 2 x 128 query CTAs exceed the SM count",
            spec=dict(
                p=1,
                lens=[1 + (389 * i) % 4100 for i in range(128)],
                index_kind="global",
                layer=2,
            ),
        ),
        Cell(
            why="256 sequences x MTP 2, base rows: a large decode batch, 512 query "
            "rows on the single-CTA split",
            spec=dict(
                p=2,
                lens=[2 + (347 * i) % 2600 for i in range(256)],
                index_kind="base_with_layer_offset",
                layer=1,
            ),
        ),
    )

    note = """
    The target builds one handle for both index kinds and 1..max_batch_size
    sequences, read from the attention metadata's `max_num_requests` on its
    first forward, which is eager and ahead of any CUDA-graph capture. The
    engine schedules at most that many sequences per step and pads a
    CUDA-graph batch only up to a captured size no larger, so every reachable
    batch has its kernel in the handle. There is no stock fallback: every
    generation row of every step takes this kernel. A batch above the
    handle's `max_batch_size`, an index kind it was not built for, or a head
    count other than 16 raises here (unreachable from the target); it never
    compiles.

    The runner is prepared per call (host validation, no device sync). When a
    batch's split-K clusters exceed one occupancy wave the kernel switches to
    a global-memory split-K reducer and allocates its partial workspace and
    zero-filled completion counters on each call. That zero-fill is a memset
    between the preceding launch and this one, so on those batches the
    programmatic launch edge into this kernel becomes a full stream
    dependency. Which batches pick the reducer depends on the device's
    cluster occupancy; the `gmem_split` cells find them on the device.

    The kernel reads neither scale array: its softmax scale is compiled in as
    `1/16` and its output scale is 1. `is_valid` holds the arrays to those
    values.

    Not certified: `all_sparse_indices_valid=True` (the all-2048-valid
    specialization), `max_sq` above 2 (an MTP draft length above 1: the
    per-sequence query rows then exceed the grid and the runner raises),
    8 or 32 heads.
    """

    def __call__(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        sparse_attn_indices: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        sequence_length: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        index_kind: str,
        local_layer_idx: Optional[int],
        compiled: CompiledSparseMlaDecode,
    ) -> torch.Tensor:
        # Inside these bounds the kernel's own pick is one the handle built,
        # so preparing the runner never compiles.
        batch = int(sequence_length.shape[0])
        if (
            not 1 <= batch <= compiled.max_batch_size
            or index_kind not in compiled.index_kinds
            or int(quant_q_buffer.shape[1]) != NUM_HEADS
            or quant_q_buffer.device.index != compiled.device_index
        ):
            raise ValueError(
                f"no kernel compiled for {batch} sequences, index kind {index_kind!r}, "
                f"{quant_q_buffer.shape[1]} heads on {quant_q_buffer.device}"
            )
        runner = _kernel.prepare_runner(
            quant_q_buffer,
            kv_pool_view,
            sparse_attn_indices,
            output,
            cu_q_seqlens,
            sequence_length,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            all_sparse_indices_valid=compiled.all_sparse_indices_valid,
            max_sq=compiled.max_sq,
            index_kind=index_kind,
            local_layer_idx=local_layer_idx,
        )
        runner.compiled = compiled.kernels[runner.specialization]
        runner()
        return output

    def reference(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        sparse_attn_indices: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        sequence_length: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        index_kind: str,
        local_layer_idx: Optional[int],
        compiled: CompiledSparseMlaDecode,
    ) -> torch.Tensor:
        """Softmax attention over each query's non-negative rows, in fp64,
        rounded to bf16 once; a new tensor shaped like `output`.

        A missing scale array means the kernel's own value: `1/16` and 1.
        `cu_q_seqlens`, `sequence_length` and the handle are not read: which
        rows count is said by the indices alone, and `is_valid` holds the
        kernel's positional reading of them to the same set.
        """
        scale = SOFTMAX_SCALE if mla_bmm1_scale is None else float(mla_bmm1_scale[0])
        out_scale = 1.0 if mla_bmm2_scale is None else float(mla_bmm2_scale[0])
        rows = sparse_attn_indices.to(torch.int64)
        if index_kind == INDEX_KIND_BASE_WITH_LAYER_OFFSET:
            shift = int(local_layer_idx) * LAYER_ROW_STRIDE
            rows = torch.where(rows >= 0, rows + shift, rows)
        q = quant_q_buffer.view(torch.uint8).view(torch.float8_e4m3fn).float().double()
        kv = _pool_bytes(kv_pool_view)
        out = torch.zeros(
            q.shape[0], q.shape[1], KV_LORA_RANK, dtype=torch.float64, device=q.device
        )
        for lo in range(0, q.shape[0], 8):
            hi = min(lo + 8, q.shape[0])
            r = rows[lo:hi]
            valid = r >= 0
            k = kv[r.clamp(min=0)].view(torch.float8_e4m3fn).float().double()
            s = torch.einsum("thd,tkd->thk", q[lo:hi], k) * scale
            s = s.masked_fill(~valid[:, None, :], float("-inf"))
            p = torch.softmax(s, dim=-1).nan_to_num(0.0)
            out[lo:hi] = torch.einsum("thk,tkd->thd", p, k[..., :KV_LORA_RANK])
        return (out * out_scale).to(torch.bfloat16).reshape(output.shape)

    def is_valid(
        self,
        quant_q_buffer: torch.Tensor,
        kv_pool_view: torch.Tensor,
        sparse_attn_indices: torch.Tensor,
        output: torch.Tensor,
        cu_q_seqlens: torch.Tensor,
        sequence_length: torch.Tensor,
        mla_bmm1_scale: Optional[torch.Tensor],
        mla_bmm2_scale: Optional[torch.Tensor],
        index_kind: str,
        local_layer_idx: Optional[int],
        compiled: CompiledSparseMlaDecode,
    ) -> None:
        """The inputs the kernel takes and answers wrongly. Reads every index
        back to the host, which is why it is never on the serving path."""
        total_q, heads = int(quant_q_buffer.shape[0]), int(quant_q_buffer.shape[1])
        cu = cu_q_seqlens.tolist()
        lens = sequence_length.tolist()
        # A query row no sequence owns is never written, and a sequence with
        # more rows than the grid's `max_sq` has its extra rows skipped. Only
        # the aggregate `total_q <= batch * max_sq` is checked on the host.
        assert cu[0] == 0 and cu[-1] == heads * total_q, (
            f"cu_q_seqlens must run from 0 to {heads * total_q} (heads x rows); got "
            f"{cu[0]}..{cu[-1]}"
        )
        sq = [(b - a) // heads for a, b in zip(cu, cu[1:])]
        assert all((b - a) % heads == 0 for a, b in zip(cu, cu[1:])), (
            "cu_q_seqlens must step in whole query rows of `heads` entries"
        )
        assert all(0 <= n <= compiled.max_sq for n in sq), (
            f"a sequence carries {max(sq)} query rows; the kernel's grid covers "
            f"max_sq = {compiled.max_sq} and skips the rest"
        )

        # The kernel does not look at which entries are -1: it reads the first
        # `clamp(len - sq + t + 1, 0, 2048)` entries of query `t` as rows and
        # masks the rest. A -1 inside that prefix is gathered as a row; a row
        # after it is dropped. Both return a plausible softmax.
        valid = torch.empty(total_q, dtype=torch.int64)
        for b, (start, n) in enumerate(zip(cu, sq)):
            first = start // heads
            for t in range(n):
                valid[first + t] = min(max(lens[b] - n + t + 1, 0), TOP_K)
        rows = sparse_attn_indices.to(torch.int64).cpu()
        if index_kind == INDEX_KIND_BASE_WITH_LAYER_OFFSET:
            rows = torch.where(rows >= 0, rows + int(local_layer_idx) * LAYER_ROW_STRIDE, rows)
        prefix = torch.arange(TOP_K)[None, :] < valid[:, None]
        pool_rows = kv_pool_view.numel() * kv_pool_view.element_size() // LATENT_DIM
        assert bool(((rows >= 0) | ~prefix).all()), (
            "a -1 inside a query's valid prefix (the first min(len - sq + t + 1, 2048) "
            "entries); the kernel gathers it as a row"
        )
        assert bool(((rows < pool_rows) | ~prefix).all()), (
            f"an index in a valid prefix is past the pool's {pool_rows} rows"
        )
        assert bool(((rows == -1) | prefix).all()), (
            "a row after a query's valid prefix, in the -1 tail; the kernel masks it, "
            "so sequence_length and the selection disagree"
        )

        # The kernel never loads the scale arrays. Arrays saying anything but
        # its compiled values (a q_scaling or KV read scale other than 1)
        # describe a softmax it does not compute.
        if mla_bmm1_scale is not None:
            got = mla_bmm1_scale.tolist()
            want = (SOFTMAX_SCALE, SOFTMAX_SCALE * math.log2(math.e))
            assert all(math.isclose(g, w, rel_tol=1e-6) for g, w in zip(got, want)), (
                f"mla_bmm1_scale {got} is not the kernel's compiled softmax scale {want}; "
                "the kernel ignores the array"
            )
        if mla_bmm2_scale is not None:
            assert float(mla_bmm2_scale[0]) == 1.0, (
                f"mla_bmm2_scale {float(mla_bmm2_scale[0])} is not 1; the kernel ignores it"
            )


cute_dsl_sparse_mla_decode = CuteDslSparseMlaDecode()
