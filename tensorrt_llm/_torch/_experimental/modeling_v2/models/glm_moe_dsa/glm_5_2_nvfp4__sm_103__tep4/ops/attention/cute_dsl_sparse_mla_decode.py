# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode sparse MLA over the fp8 latent pool, top-k 2048, on the CuTe DSL kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import (
    self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16 as _kernel,
)
from ...kernels.attention import (
    self_attn_fmha_sparse_mla_union_h16_qk576_v512_topk2048_fp8_bf16 as _union_kernel,
)
from ...kernels.attention import (
    self_attn_fmha_sparse_mla_wide_h16_qk576_v512_topk2048_fp8_bf16 as _wide_kernel,
)
from ...kernels.common.cuda_stream import DIET as _HOST_DIET
from .cute_dsl_req_index_to_global import pool_rows

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
#: The per-sequence query count the kernel's grid is sized for by default: MTP 2.
#: MTP3 / MTP5 build a second handle at 4 / 6 (`compile(max_sq=...)`).
MAX_SQ = 2
#: The softmax scale the kernel compiles in: 1 / (q_scaling * sqrt(nope + rope))
#: with q_scaling 1 and GLM's 192 + 64.
SOFTMAX_SCALE = 1.0 / math.sqrt(_kernel.QK_HEAD_DIM)
#: Batches at which a handle wider than `MAX_SQ` runs split-K 2 with the cluster
#: reducer instead of the kernel's own pick (the single CTA there), keyed by
#: (SM count, max_sq): the batches where that measured, with the wide kernel, on
#: B300 (148 SMs) no slower in CUDA-graph replay in each of 3 KV-length regimes
#: and faster on their mean by more than 1.5 us per call. They are the MTP5
#: batches whose single-CTA grid needs a second wave (6 x 25 = 150 query CTAs
#: and up); an MTP3 grid up to 32 sequences fits one wave. Any other device,
#: width or batch, and the width-2 handle always, keep the kernel's pick.
SPLIT2_CLUSTER_BATCHES: Mapping[tuple[int, int], frozenset[int]] = {
    (148, 6): frozenset(range(25, 33)),
}


@dataclass(frozen=True)
class CompiledSparseMlaDecode:
    """Every split-K variant a batch of 1..`max_batch_size` sequences reaches,
    per index kind, keyed by the specialization the kernel selects.
    `split2_cluster_batches` are the batches this handle runs on split-K 2 with
    the cluster reducer instead (`SPLIT2_CLUSTER_BATCHES`)."""

    device_index: int
    max_batch_size: int
    max_sq: int
    all_sparse_indices_valid: bool
    kernels: Mapping[_kernel.MlaFmhaSpecialization, object]
    split2_cluster_batches: frozenset[int] = frozenset()

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
    A wide handle's `SPLIT2_CLUSTER_BATCHES` up to the bound add split-K 2 with
    the cluster reducer, one of those seven (the kernel's own pick for some
    smaller batch of the same width). Call once, outside any CUDA-graph
    capture, with the engine's largest per-step sequence count.
    """
    split2 = _split2_cluster_batches(int(device_index), int(max_sq), int(max_batch_size))
    kernels = {}
    for kind in index_kinds:
        specs = list(
            _wide_kernel.precompile_for_serving(
                max_batch_size=int(max_batch_size),
                all_sparse_indices_valid=bool(all_sparse_indices_valid),
                max_sq=int(max_sq),
                device_index=int(device_index),
                index_kind=kind,
            )
        )
        if split2:
            specs.append(
                _kernel.MlaFmhaSpecialization(
                    all_sparse_indices_valid=bool(all_sparse_indices_valid),
                    num_ctas_kv=2,
                    use_gmem_reduction=False,
                    index_kind=kind,
                )
            )
        for spec in specs:
            # Already built by the walk above (or a cache hit); this returns the kernel.
            kernels[spec] = _wide_kernel.compile(
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
        split2_cluster_batches=split2,
    )


def _split2_cluster_batches(device_index: int, max_sq: int, max_batch_size: int) -> frozenset[int]:
    """The batches up to `max_batch_size` a handle of `max_sq` rows per sequence on
    this device runs on split-K 2 with the cluster reducer; none at `MAX_SQ`."""
    if max_sq <= MAX_SQ:
        return frozenset()
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    batches = SPLIT2_CLUSTER_BATCHES.get((sms, max_sq), frozenset())
    return frozenset(b for b in batches if b <= max_batch_size)


def split_k_variant(
    batch_size: int, *, device_index: int, max_sq: int = MAX_SQ
) -> tuple[int, bool]:
    """`(num_ctas_kv, use_gmem_reduction)`: the split-K CTA count and reducer a
    handle of `max_sq` rows per sequence runs `batch_size` sequences on, on
    this device.

    The kernel's own rule, which `compile`'s walk and the per-call runner
    apply, with the KV length it reads when no host lengths are passed
    (top_k); except at a wide handle's `SPLIT2_CLUSTER_BATCHES`, split-K 2 with
    the cluster reducer. Host only: it reads the device's SM count and cluster
    occupancy, both cached.
    """
    if int(batch_size) in _split2_cluster_batches(int(device_index), int(max_sq), int(batch_size)):
        return 2, False
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
    # split leaves one occupancy wave. MTP3 / MTP5 verify and draft step 0 put
    # 4 / 6 rows on each of up to 32 sequences, on a handle built at that
    # `max_sq` (the width-2 cells run on one built at 2), in both index kinds:
    # base rows on the trunk, global rows in the MTP layer and on a mixed step.
    # The wide grid moves every split-K pick, so its cells walk them again:
    # small batches on the wide clustered splits, the full batch on the
    # single-CTA split, and the global-memory reducer where this device selects
    # it. On B300 the MTP5 handle runs 25..32 sequences on split-K 2 with the
    # cluster reducer instead (`SPLIT2_CLUSTER_BATCHES`): the MTP5 cells at 32
    # and 31 sequences and cells at 25 and 28, both index kinds between them;
    # an MTP3 cell at 21 sequences keeps the single CTA. Their draft steps
    # 1..D-1 put 1 row on each of 32 sequences, on the width-2 handle.
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
        Cell(
            why="MTP3, base rows: 32 sequences x 4 on the max_sq 4 grid; a sequence of "
            "4 tokens and three whose 4 tokens straddle top_k",
            spec=dict(
                p=4,
                lens=[4, 2047, 2049, 2051] + [100 + 137 * i for i in range(28)],
                index_kind="base_with_layer_offset",
                layer=2,
            ),
        ),
        Cell(
            why="MTP3, global rows: 32 sequences x 4, the MTP layer's and a mixed "
            "step's index kind",
            spec=dict(
                p=4,
                lens=[5 + (389 * i) % 4100 for i in range(32)],
                index_kind="global",
                layer=1,
            ),
        ),
        Cell(
            why="MTP5, base rows: 32 sequences x 6 on the max_sq 6 grid, 192 query "
            "CTAs; a sequence of 6 tokens and three straddling top_k",
            spec=dict(
                p=6,
                lens=[6, 2045, 2048, 2053] + [300 + 131 * i for i in range(28)],
                index_kind="base_with_layer_offset",
                layer=1,
            ),
        ),
        Cell(
            why="MTP5, global rows: 32 sequences x 6, the MTP layer's and a mixed "
            "step's index kind",
            spec=dict(
                p=6,
                lens=[6 + (347 * i) % 4000 for i in range(32)],
                index_kind="global",
                layer=0,
            ),
        ),
        Cell(
            why="MTP3 at 3 sequences: 12 query CTAs on the max_sq 4 grid, a wide clustered split-K",
            spec=dict(p=4, lens=[9, 2050, 3333], index_kind="base_with_layer_offset", layer=0),
        ),
        Cell(
            why="MTP5 at 11 sequences: 66 query CTAs on the max_sq 6 grid, the 2-CTA "
            "clustered split",
            spec=dict(
                p=6,
                lens=[7 + 389 * i for i in range(11)],
                index_kind="global",
                layer=2,
            ),
        ),
        Cell(
            why="MTP3, the global-memory reducer at the 8-CTA split on the max_sq 4 grid",
            spec=dict(p=4, gmem_split=8, index_kind="base_with_layer_offset", layer=1),
        ),
        Cell(
            why="MTP3, the global-memory reducer at the 4-CTA split on the max_sq 4 grid",
            spec=dict(p=4, gmem_split=4, index_kind="global", layer=0),
        ),
        Cell(
            why="MTP5, the global-memory reducer at the 8-CTA split on the max_sq 6 grid",
            spec=dict(p=6, gmem_split=8, index_kind="global", layer=2),
        ),
        Cell(
            why="MTP5, the global-memory reducer at the 4-CTA split on the max_sq 6 grid",
            spec=dict(p=6, gmem_split=4, index_kind="base_with_layer_offset", layer=0),
        ),
        Cell(
            why="MTP5 at 31 sequences on split-K 2 with the cluster reducer (the kernel "
            "would take one CTA), global rows, lengths crossing top_k",
            spec=dict(
                p=6,
                lens=[6, 2047, 2048, 2049, 2053] + [11 + (433 * i) % 6000 for i in range(26)],
                index_kind="global",
                layer=1,
            ),
        ),
        Cell(
            why="MTP5 at 25 sequences, the smallest batch on split-K 2 with the cluster "
            "reducer (150 single-CTA query CTAs would exceed one wave), global rows",
            spec=dict(p=6, lens=[6 + 307 * i for i in range(25)], index_kind="global", layer=0),
        ),
        Cell(
            why="MTP5 at 28 sequences on split-K 2 with the cluster reducer, base rows",
            spec=dict(
                p=6,
                lens=[6, 2050] + [13 + 263 * i for i in range(26)],
                index_kind="base_with_layer_offset",
                layer=1,
            ),
        ),
        Cell(
            why="MTP3 at 21 sequences, global rows: 84 query CTAs on the single-CTA split",
            spec=dict(
                p=4, lens=[5 + (509 * i) % 4400 for i in range(21)], index_kind="global", layer=2
            ),
        ),
        Cell(
            why="MTP3 / MTP5 draft steps 1..D-1: 32 sequences x 1 row on the width-2 "
            "handle, global rows",
            spec=dict(
                p=1,
                lens=[1 + (577 * i) % 4500 for i in range(32)],
                index_kind="global",
                layer=1,
            ),
        ),
    )

    note = """
    The target builds one handle for both index kinds and 1..max_batch_size
    sequences, read from the attention metadata's `max_num_requests` on its
    first forward, which is eager and ahead of any CUDA-graph capture; at
    MTP3 / MTP5 a second one at `max_sq` 4 / 6, which it passes only for a step
    with more than 2 rows per sequence (the 1-row draft steps keep the
    width-2 handle and its wider split-K). The
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

    The handle builds its kernels from the module
    `self_attn_fmha_sparse_mla_wide_...`: the base kernel with PV reading V
    from the tile's own K latent slots instead of gathering the latent a second
    time (H16 only). Same split-K picks, reducers and ABI, bitwise the same
    outputs as the base kernel (pinned by a test until that kernel, which no
    served path ran, was removed), about half the gathered bytes: on B300 at
    32 sequences x MTP 2, 2048 valid rows per query, 28.2 -> 20.7 us per call
    in CUDA-graph replay.

    The kernel's rule gives one CTA per query row from 13 sequences up at
    max_sq 6 and from 19 at max_sq 4: 512 threads at 128 registers and 203 KB
    (wide kernel) of shared memory, one CTA per SM, so 32 x 6 = 192 CTAs take
    1.3 waves on 148 SMs. On B300 the wide kernel's split-K 2 with the cluster
    reducer (and split-K 3, 4 and 2 with the global-memory reducer, never
    faster) was compared with it at 13, 14 and 24..32 sequences (max_sq 6) and
    19, 21 and 32 (max_sq 4), CUDA-graph replay of 8 calls on 8 pool layers,
    per call, in 3 KV-length regimes (lengths 3000..6007, all 2048 rows valid;
    1500..4073; 1024..9216). At max_sq 6 from 25 sequences, where the single
    CTA needs a second wave, split-K 2 is faster or tied in every regime, by
    1.7 to 2.6 us on their mean (32 sequences: 55.3 -> 53.0, 51.0 -> 48.1,
    55.3 -> 53.8 us); at 24 and below, and at max_sq 4 (one wave up to 32
    sequences), it is 1.3 us faster to 6.4 us slower, slower on the mean. So
    only a wide handle on a 148-SM device runs those batches on it
    (`SPLIT2_CLUSTER_BATCHES`); the width-2 handle's picks are the kernel's
    own.

    The kernel reads neither scale array: its softmax scale is compiled in as
    `1/16` and its output scale is 1. `is_valid` holds the arrays to those
    values.

    Not certified: `all_sparse_indices_valid=True` (the all-2048-valid
    specialization), `max_sq` other than 2, 4 and 6 (a step wider than its
    handle's `max_sq` exceeds the grid and the runner raises), 8 or 32 heads.
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
        # Inside these bounds the handle's pick is one it built, so preparing
        # the runner never compiles.
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
        # None: the kernel's own pick (always on the width-2 handle).
        split2 = batch in compiled.split2_cluster_batches
        signature = None
        if _HOST_DIET and not split2 and not torch.cuda.is_current_stream_capturing():
            # CTX-HOST-DIET: a call signature `prepare_runner` passed before launches the same
            # arguments without its contract walk (`run_accepted`). The signature does not name
            # a wide handle's split-K 2 override, so its record is the kernel's own pick and
            # the override's batches always prepare their runner.
            signature = _wide_kernel.run_accepted(
                compiled.kernels, quant_q_buffer, kv_pool_view, sparse_attn_indices, output,
                cu_q_seqlens, sequence_length, mla_bmm1_scale, mla_bmm2_scale,
                all_sparse_indices_valid=compiled.all_sparse_indices_valid,
                max_sq=compiled.max_sq, index_kind=index_kind, local_layer_idx=local_layer_idx,
            )  # fmt: skip
            if signature is None:
                return output
        runner = _wide_kernel.prepare_runner(
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
            num_ctas_kv=2 if split2 else None,
            use_gmem_reduction=False if split2 else None,
            index_kind=index_kind,
            local_layer_idx=local_layer_idx,
        )
        runner.compiled = compiled.kernels[runner.specialization]
        runner()
        if signature is not None:
            _wide_kernel.accept(signature, runner)
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


# ---------------------------------------------------------------------------
# MTP3 / MTP5 decode over the union of a query-row group's selections
# ---------------------------------------------------------------------------
#: Query rows per group of the union kernel, by handle width, for a batch `UNION_SERVED` does
#: not list: one CTA multiplies each Kv128 tile of the group's union against 16 x rows query
#: columns. (4-row groups spill: MTP3 at 32 sequences, 33.0 us on 4-chunk clusters against
#: 27.1 us at 2 rows; MTP35 track M job u6.)
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
    (148, 4): _served(2, range(19, 23)),
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
    plans: Mapping[int, tuple[int, int]]


def union_plan(batch_size: int, *, device_index: int, max_sq: int) -> tuple[int, int]:
    """`(rows_per_group, num_chunks)` for `batch_size` sequences of `max_sq` rows.

    Groups of the rows `UNION_SERVED` gives the batch on this device (else
    `UNION_ROWS_PER_GROUP[max_sq]`); each group's union is split over a cluster of
    `num_chunks` CTAs, the most (up to the kernel's 8) that keeps every group's cluster
    resident in one wave on this device (`batch x groups` clusters, one CTA per SM).
    Host only; reads the device's SM count and cluster occupancy, both cached.
    """
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    served = UNION_SERVED.get((sms, int(max_sq)), {})
    rg = served.get(int(batch_size), UNION_ROWS_PER_GROUP[int(max_sq)])
    groups = (int(max_sq) + rg - 1) // rg
    c = max(1, min(_union_kernel.MAX_CHUNKS, sms // (int(batch_size) * groups)))
    while c > 1 and int(batch_size) * groups > _kernel._max_active_clusters(device_index, c):
        c -= 1
    return rg, c


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
    for rg, c in sorted(set(plans.values())):
        _union_kernel.compile_union(
            _union_kernel.UnionSpecialization(rows_per_group=rg, num_chunks=c), int(device_index)
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
    not bitwise. Returns `output`.
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
    )

    note = """
    The target builds the handle (`compile_union`) next to the MTP3 / MTP5 per-row handle
    and passes it on the trunk's pure decode steps at 4 / 6 rows per sequence and a batch
    `union_batches` serves: the indexer's positions go straight to this op, so those steps
    skip the base-row conversion. Other batches, the MTP layer, mixed steps and the 1-row
    draft steps keep the per-row kernels.

    Measured (B300, CUDA-graph replay of 8 calls on 8 pool layers, per call, against the
    per-row V-reuse kernel on its own pick, split-K 2 clusters at MTP5 25..32; MTP35 track M
    jobs u10 / u11). Lengths per regime: A 2850..3067 (the served benchmark's census), B
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

    MTP5 below 13 and MTP3 below 19 (one-wave per-row grids) measured 0.89 to 2.0 (u10,
    u11), never faster in A: not served. See the kernel module for what bounds each.

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
        rg, chunks = compiled.plans[batch]
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
