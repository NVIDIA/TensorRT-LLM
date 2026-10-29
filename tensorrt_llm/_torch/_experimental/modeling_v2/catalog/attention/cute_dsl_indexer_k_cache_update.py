# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Norm, rope, fp8-pack and append the DSA indexer keys to the paged K cache, in one launch."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from .._op import Arch, Cell, OpWrapper
from . import _indexer_k_cache_update_kernel as _kernel

_HEAD_DIM = 128
_ROPE_DIM = 64
_HALF = _ROPE_DIM // 2
_SCALE_BYTES = 4
#: The kernel's LayerNorm epsilon, a compile-time constant (not a parameter).
NORM_EPS = 1.0e-6
_E4M3_MAX = 448.0
_MIN_AMAX = 1.0e-12

#: The two launch shapes the kernel's own wrapper picks between: 2 rows per CTA
#: up to `_SMALL_TOKEN_THRESHOLD` tokens, 8 above.
_ROWS_PER_BLOCK = (_kernel._SMALL_TOKEN_ROWS_PER_BLOCK, _kernel._FUSED_CAT_SCATTER_ROWS_PER_BLOCK)


@dataclass(frozen=True)
class CompiledIndexerKCacheUpdate:
    """One launcher per (rows per CTA, vectorised input loads), and the flags they share."""

    fns: Mapping[tuple[int, bool], Callable]
    is_neox: bool
    cache_contiguous: bool


def build(device_id: int, *, is_neox: bool, cache_contiguous: bool) -> CompiledIndexerKCacheUpdate:
    """Compile the four launchers a step can need: 2 or 8 rows per CTA, scalar or 8-byte loads.

    `cache_contiguous` must describe every cache view the launchers will write:
    the contiguous variant addresses a strided per-layer view as if it were
    dense. Four compiles of a few seconds each, cached per process by the kernel
    module; allocates nothing. ue8m0 scales and the PDL launch are fixed.
    """
    fns = {}
    for rows_per_block in _ROWS_PER_BLOCK:
        for vectorized in (False, True):
            fns[(rows_per_block, vectorized)] = (
                _kernel.compile_indexer_k_norm_rope_fp8_cache_update(
                    device_id=device_id,
                    use_ue8m0=True,
                    is_neox=bool(is_neox),
                    rows_per_block=rows_per_block,
                    vectorized_loads=vectorized,
                    cache_contiguous=bool(cache_contiguous),
                )
            )
    return CompiledIndexerKCacheUpdate(
        fns=fns, is_neox=bool(is_neox), cache_contiguous=bool(cache_contiguous)
    )


#: What every Mistral Large 4 cell shares: interleaved rope over its YaRN + LeRoPE
#: table of max_seq_len 9216 rows, a 44-layer pool of 64 blocks (layer 3 written),
#: 8-byte-aligned keys, no padding rows.
_ML4 = dict(pool_layers=44, blocks=64, pad_rows=0, misalign=False, is_neox=False, table="yarn")
#: The Mistral Large 4 NVFP4 target's: the same, over its YaRN factor 2048 table of
#: max_seq_len 32768 rows. `layout` says which view the cache is: absent, the
#: strided per-layer view above (the V1 DSA cache manager's); "v2", the view
#: the engine's default `DSACacheManagerV2` hands out, rebuilt: contiguous
#: `[blocks * 44 - layer, 32, 1, 132]` from the layer's first page, a block's
#: page at `block * 44` (see `note`); "manager_v2", that view and the slot
#: mappings taken from a real manager whose quota is `blocks` pages, with one
#: request per `requests` entry (that many new tokens after `cached` cached
#: ones) and the view of model layer `model_layer`.
_ML4_32K = dict(_ML4, max_seq_len=32768, yarn_factor=2048.0)
_ML4_32K_V2 = dict(_ML4_32K, layout="v2")
_ML4_32K_MGR = dict(_ML4_32K, layout="manager_v2", blocks=1100)
#: The largest pool one rank of the NVFP4 target can hold, a deliberate
#: overestimate: 276.62 GiB per GB300 less 143.24 GiB of weights leaves room
#: for 94659 blocks of 32 tokens at the 47280 bytes a token takes in the 72
#: latent and 44 indexer rows; the V2 manager's slot range runs about 5% past
#: its quota, and 110592 blocks clear both.
_ML4_NVFP4_MAX_BLOCKS = 110592
#: The same bound for one rank of the NVFP4 eight-GPU targets, again an
#: overestimate: a tep8 or tp8 rank holds 73.61 GiB of weights and a dep8 rank
#: 89.29 GiB (computed from the shard headers), so 203.01 and 187.33 GiB of a
#: GB300 are left, device memory for 144075 and 132949 V2 slots at 1512960 bytes
#: a slot (32 tokens of 47280 bytes). The manager backs every slot of its range,
#: past its quota or not, with device memory, so 147456 blocks clear any pool a
#: tep8, tp8 or dep8 rank can allocate.
_ML4_NVFP4_8GPU_MAX_BLOCKS = 147456
#: Max_batch_size 16 requests packed to 32768 new tokens, lengths on either side
#: of the page (32), the 2-row launcher's last count (1024) and index_topk (2048).
_ML4_32K_PACKED = (1, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 2047, 2049, 4095, 4097, 8191, 8928)
#: The tp4 target at the checkpoint's own context: max_seq_len 1048576 with chunked
#: prefill on and max_num_tokens 32768 (the chunk), over its YaRN factor 2048 table of
#: 1048576 rows with its values rounded to bf16 (`table_dtype`), as the target's rope
#: tables hold them. `requests` and `cached` place a step's rows on every layout:
#: request i's `requests[i]` rows at positions `cached[i]..cached[i] + requests[i] - 1`.
#: The manager cells build it at max_num_tokens 32768 apart from max_seq_len.
#: Named `_TP4` (tp4's `_ML4_1M*`; only tp4's cells use them): tep4's and dep4's `_ML4_1M`,
#: `_ML4_1M_V2` and `_ML4_1M_MGR` hold no `table_dtype` (an fp32 table) and no
#: `max_num_tokens` key, and tep4's `_ML4_1M_PACKED*` and `_ML4_1M_MIXED_CACHED` are other
#: steps (`_ML4_1M_MIXED_TP4` equals tep4's `_ML4_1M_MIXED`, and keeps its partner's suffix).
_ML4_1M_TP4 = dict(_ML4, max_seq_len=1048576, yarn_factor=2048.0, table_dtype="bfloat16")
_ML4_1M_V2_TP4 = dict(_ML4_1M_TP4, layout="v2")
_ML4_1M_MGR_TP4 = dict(_ML4_1M_TP4, layout="manager_v2", max_num_tokens=32768)
#: One chunk packing four requests: the last 16384 rows of a 1048576-token request,
#: 8192 rows past a 524288-token prefix, a first chunk and a short one past 2048.
_ML4_1M_PACKED_TP4 = (16384, 8192, 4096, 4096)
_ML4_1M_PACKED_CACHED_TP4 = (1032192, 524288, 0, 2048)
#: A mixed chunk step of max_batch_size 16: three context chunks (one ending at position
#: 1048575) ahead of 13 decode rows across the 1M range.
_ML4_1M_MIXED_TP4 = (20000, 9000, 3755, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
_ML4_1M_MIXED_CACHED_TP4 = (
    1028576,
    524288,
    0,
    1048575,
    1048544,
    1015808,
    1015807,
    524288,
    262144,
    131072,
    65536,
    32768,
    32767,
    16384,
    2048,
    1,
)
#: The NVFP4 target at the checkpoint's own context: max_seq_len 1048576 with
#: chunked prefill (max_num_tokens 32768), over the 1048576-row YaRN factor 2048
#: table (or the 1048576-row identity table). On a "v2" view as on the manager,
#: `requests` and `cached` put each request's new rows at positions
#: cached..cached+new-1. tep4's cells use these `_ML4_1M*`, and dep4's `_ML4_1M`,
#: `_ML4_1M_V2` and `_ML4_1M_MGR` (tp4's are the `_TP4` ones above: a bf16-rounded
#: table, the manager's max_num_tokens named in the spec).
_ML4_1M = dict(_ML4, max_seq_len=1048576, yarn_factor=2048.0)
_ML4_1M_V2 = dict(_ML4_1M, layout="v2")
_ML4_1M_MGR = dict(_ML4_1M, layout="manager_v2")
#: Max_batch_size 16 chunk tails packed to 32768 rows (the new tokens, then each
#: one's cached prefix): a decode-sized tail at position 1048575, lengths across
#: the page, launcher and index_topk edges, prefixes across 32768 and 2^16..2^19.
_ML4_1M_PACKED = (1, 7, 31, 32, 33, 100, 255, 257, 513, 1023, 1025, 2047, 2049, 8191, 8193, 9011)
_ML4_1M_PACKED_CACHED = (
    1048575, 0, 1, 31, 32, 33, 2047, 2048, 2049, 32767, 65535, 131072, 262143, 500000, 0, 0,
)  # fmt: skip
#: A mixed step at 1M: three chunks (32755 rows) ahead of 13 decode rows whose
#: prefixes reach 1048575.
_ML4_1M_MIXED = (20000, 9000, 3755) + (1,) * 13
_ML4_1M_MIXED_CACHED = (
    1028576, 500000, 0, 1048575, 1048544, 524287, 131071, 32768, 32767, 2048, 2047, 64, 33, 32,
    2, 1,
)  # fmt: skip
#: The dep4 target's mixed step at 1M: three chunks (32755 rows; one fresh, two
#: behind prefixes of 100000 and 65536) ahead of 13 decode rows whose prefixes
#: run from 1048575 down across 2^19..2^15, index_topk and the page to 1.
_ML4_1M_DEP4_MIXED = (20000, 9000, 3755) + (1,) * 13
_ML4_1M_DEP4_MIXED_CACHED = (
    100000, 0, 65536, 1048575, 524288, 262144, 131072, 65536, 32768, 32767, 2048, 2047, 33, 32,
    2, 1,
)  # fmt: skip
#: The dep4 target's long-family decode graphs at 1M (`boot_capture` cells). The
#: engine captures each batch b of 1/2/4/8/16 on its CUDA-graph warm-up batch:
#: one dummy of max_seq_len - 1 = 1048575 keys (its new key at position
#: 1048574) ahead of b - 1 one-key fillers, each a request of its own (position
#: 0 in a page of its own). Attention DP then pads every rank to the largest
#: rank's batch with one padding dummy repeated (`pad_rows` rows of one request:
#: position 0, one slot pair, one key), so a rank replays the graph with its own
#: live rows ahead of them, an idle rank with its one-key attention-DP dummy.
#: The slot mappings are the metadata's max_num_tokens buffers.
_ML4_1M_DEP4_BOOT = dict(
    _ML4_1M_V2, blocks=1100, graph=True, top_position=True, slot_rows=32768, boot_capture=True
)


class CuteDslIndexerKCacheUpdate(OpWrapper):
    """Turn this step's pre-norm indexer keys into indexer K cache entries, in place.

    For each of the first `num_tokens` rows of `indexer_k` (`[T, 128]` bf16):

    1. LayerNorm over the row (`eps = 1e-6`, fp32 `norm_weight` / `norm_bias`),
       rounded to bf16;
    2. rope on the first 64 dims at `position_ids[t]`, from
       `rotary_cos_sin[pos] = [[cos(32)], [sin(32)]]`, rounded to bf16: NeoX
       halves (`is_neox=True`: dims `i` and `i + 32` rotate by column `i`) or
       interleaved pairs (`is_neox=False`: dims `2i` and `2i + 1` rotate by
       column `i`). The last 64 dims are not rotated;
    3. `scale = max(amax, 1e-12) / 448` rounded up to a power of two, and the
       e4m3 codes `row / scale`;
    4. the 128 code bytes go to flat byte `slot_mapping_fp8[t]` of `k_cache`
       and the fp32 scale to flat byte `slot_mapping_scale[t]`, where "flat"
       is the logical row-major index of `k_cache`'s `[blocks, page, 1, D]`
       shape, mapped through its strides. A row with either slot negative is
       skipped.

    `k_cache` is written in place and returned. Nothing else is written.

    The call is `indexer_k_norm_rope_fp8_cache_update`, a CuTe DSL kernel's
    Python launcher (no `torch.ops` registration), launched once. It lives in
    the private catalog module `_indexer_k_cache_update_kernel`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The Mistral Large 4 FP8 target writes its 44 full indexer layers: decode
    # batches 1..16 (captured at 1, 2, 4, 8, 16), prefill up to its max_num_tokens
    # 9216 at positions 0..9215, interleaved rope, and its 18 NoPE indexer layers
    # with an identity table. The NVFP4 target (`_ML4_32K` cells) is the same forward at
    # max_seq_len = max_num_tokens = 32768 with chunked prefill off: a context
    # step of up to 32768 rows (one request, or up to 16), decode rows at
    # positions up to 32767, graphs 1/2/4/8/16 padded. Its tep8, tp8 and dep8
    # targets make the same call on every rank (the indexer is whole on every
    # rank: replicated at tep8 and tp8, data parallel at dep8), on a larger
    # pool: a tep8 or tp8 rank holds 73.61 GiB of weights, a dep8 rank 89.29
    # GiB, a tep4 rank 143.24 GiB. In the cells without a
    # `layout` the K pool is one block-major allocation of every indexer layer,
    # so a layer's cache is a strided view whose blocks are dense; the engine's
    # default manager hands out a contiguous view instead (`note`), and the
    # target picks the launchers by `k_cache.is_contiguous()`, so its calls
    # run the contiguous launchers. 32 tokens per page, 132 bytes per token.
    # The NVFP4 tp4 target at max_seq_len 1048576 (`_ML4_1M_TP4` cells) is that forward
    # with chunked prefill on: a step still holds at most 32768 rows
    # (max_num_tokens, the chunk), but a context chunk's rows sit at positions
    # cached..cached+new-1 after a cached prefix of up to 1048576 - new (one
    # request, or up to 16 packed, decode rows behind them), decode rows sit
    # anywhere up to 1048575, and the tables have 1048576 rows. Its keys and
    # its table rows are the same per row at every length, so the `_ML4_32K`
    # cells carry over what does not depend on the position.
    # The tep4 and tep8 NVFP4 targets at the checkpoint's own context (`_ML4_1M`
    # cells; the indexer is replicated at both, so the call is the same) run
    # max_seq_len 1048576 with chunked prefill on and max_num_tokens 32768:
    # a context step is still at most 32768 rows, but every chunk after a
    # request's first sits behind its cached prefix (rows at positions
    # cached..cached+new-1, up to 1048575), decode rows reach position 1048575,
    # and both tables have 1048576 rows. Its manager cells hold a whole 1M
    # request, so their pools pass the pre-merge budget and are `large`; the
    # `v2` cells at the same positions are their pre-merge share.
    # The dep4 NVFP4 target at the checkpoint's own context (`_ML4_1M` cells)
    # runs max_seq_len 1048576 with chunked prefill on and max_num_tokens 32768
    # per rank, attention data parallel (each rank appends its own requests'
    # keys to its own pool): a context step is still at most 32768 rows, but
    # every chunk after a request's first sits behind its cached prefix (rows
    # at positions cached..cached+new-1, up to 1048575), decode rows reach
    # position 1048575, and both tables have 1048576 rows. Its decode graphs
    # are captured at boot on the warm-up batch and replayed behind attention
    # DP's padding dummy (`boot_capture`, `_ML4_1M_DEP4_BOOT`). Its manager cells
    # hold a whole 1M request, so their pools pass the pre-merge budget and are
    # `large` (device peaks 58.42 GiB at a 33792-page quota, 124.95 GiB at
    # 73728); the `v2` cells at the same positions are their pre-merge share.
    # A cell's spec is one of the bases above with its own keys over it.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="interleaved rope, decode batch 1 (captured): one row in a 2-row CTA",
            spec=dict(_ML4, tokens=1, graph=True),
        ),
        Cell(
            why="interleaved rope, decode batch 2 (captured)",
            spec=dict(_ML4, tokens=2, graph=True),
        ),
        Cell(
            why="interleaved rope, decode batch 3: an odd batch of the unpadded 1..16 captures",
            spec=dict(_ML4, tokens=3, graph=True),
        ),
        Cell(
            why="interleaved rope, decode batch 4 (captured)",
            spec=dict(_ML4, tokens=4, graph=True),
        ),
        Cell(
            why="interleaved rope, decode batch 8 (captured), slot buffers sized for batch 16",
            spec=dict(_ML4, tokens=8, graph=True, slot_rows=16),
        ),
        Cell(
            why="interleaved rope, decode batch 16 (captured), its last 3 rows padding at slot -1",
            spec=dict(_ML4, tokens=16, graph=True, pad_rows=3),
        ),
        Cell(
            why="a NoPE indexer layer at decode batch 16 (captured): identity table, no rotation",
            spec=dict(_ML4, tokens=16, graph=True, table="identity"),
        ),
        Cell(
            why="interleaved prefill at 1024 rows: the last count of the 2-row-per-CTA launcher",
            spec=dict(_ML4, tokens=1024),
        ),
        Cell(
            why="interleaved prefill at 1025 rows: the first count of the 8-row-per-CTA launcher",
            spec=dict(_ML4, tokens=1025),
        ),
        Cell(
            why="interleaved prefill at max_num_tokens 9216 rows, positions 0..9215",
            spec=dict(_ML4, tokens=9216, blocks=320),
        ),
        Cell(
            why="a NoPE indexer layer at 9216 rows: identity table over positions 0..9215",
            spec=dict(_ML4, tokens=9216, blocks=320, table="identity"),
        ),
        Cell(
            why="interleaved keys not 8-byte aligned: the scalar-load launcher",
            spec=dict(_ML4, tokens=33, misalign=True),
        ),
        Cell(
            why="a 72-layer pool (no indexer layer mask), rows in its last layer",
            spec=dict(_ML4, tokens=16, pool_layers=72, layer=71),
        ),
        Cell(
            why="the largest 44-layer pool a rank can hold (18432 blocks, 3.4 GB; 9.6 GiB peak): "
            "rows in the last block of the last layer, past 2^31 bytes",
            spec=dict(_ML4, tokens=16, blocks=18432, layer=43, far_end=True),
        ),
        Cell(
            why="32768-row prefill on the strided view, positions 0..32767, YaRN factor 2048: "
            "the strided 8-row launcher at the NVFP4 target's max_num_tokens",
            spec=dict(_ML4_32K, tokens=32768, blocks=1100),
        ),
        Cell(
            why="32768-row NoPE layer on the strided view: identity table over positions 0..32767",
            spec=dict(_ML4_32K, tokens=32768, blocks=1100, table="identity"),
        ),
        Cell(
            why="32768 rows, engine V2 manager, layer 68 (pool row 42): one request at positions "
            "0..32767, the contiguous 8-row launcher",
            spec=dict(_ML4_32K_MGR, tokens=32768, requests=(32768,), model_layer=68),
        ),
        Cell(
            why="32768 rows, engine V2 manager, NoPE layer 71 (the last pool row, 43): identity "
            "table over positions 0..32767",
            spec=dict(
                _ML4_32K_MGR, tokens=32768, requests=(32768,), model_layer=71, table="identity"
            ),
        ),
        Cell(
            why="32768 rows, engine V2 manager, layer 0: max_batch_size 16 requests packed, "
            "lengths across page, launcher and index_topk edges",
            spec=dict(_ML4_32K_MGR, tokens=32768, requests=_ML4_32K_PACKED, model_layer=0),
        ),
        Cell(
            why="32768 rows, engine V2 manager, layer 20: a mixed step, 3 contexts (32755 rows) "
            "ahead of 13 decode rows at positions up to 32767",
            spec=dict(
                _ML4_32K_MGR,
                tokens=32768,
                blocks=4096,
                requests=(20000, 9000, 3755) + (1,) * 13,
                cached=(0, 0, 0, 32767, 32736, 16383, 9215, 2048, 2047, 1024, 64, 33, 32, 31, 2, 1),
                model_layer=20,
            ),
        ),
        Cell(
            why="32768-table decode batch 1 (captured), V2 view, last row: position 32767, "
            "the contiguous 2-row launcher",
            spec=dict(_ML4_32K_V2, tokens=1, graph=True, top_position=True, layer=43),
        ),
        Cell(
            why="32768-table decode batch 2 (captured), V2 view, first row",
            spec=dict(_ML4_32K_V2, tokens=2, graph=True, top_position=True, layer=0),
        ),
        Cell(
            why="32768-table decode batch 4 (captured), V2 view",
            spec=dict(_ML4_32K_V2, tokens=4, graph=True, top_position=True, layer=17),
        ),
        Cell(
            why="32768-table decode batch 8 (captured), V2 view",
            spec=dict(_ML4_32K_V2, tokens=8, graph=True, top_position=True, layer=30),
        ),
        Cell(
            why="32768-table decode batch 16 (captured), V2 view, slot buffers of max_num_tokens "
            "32768, its last 3 rows padding at slot -1",
            spec=dict(
                _ML4_32K_V2,
                tokens=16,
                blocks=1100,
                graph=True,
                top_position=True,
                pad_rows=3,
                slot_rows=32768,
                layer=42,
            ),
        ),
        Cell(
            why="32768-table NoPE decode batch 16 (captured), V2 view: identity table",
            spec=dict(
                _ML4_32K_V2, tokens=16, graph=True, top_position=True, table="identity", layer=43
            ),
        ),
        Cell(
            why="32768 rows over a 24576-block V2 view (4.6 GB): slots on both "
            "sides of 2^31 and 2^32 bytes",
            spec=dict(_ML4_32K_V2, tokens=32768, blocks=24576, layer=43),
        ),
        Cell(
            why="32768 rows in the top blocks of the largest V2 pool a rank can hold (110592 "
            "blocks, 20.6 GB; 39.6 GiB peak), descending as V2 hands them out: past 2^34 bytes",
            spec=dict(
                _ML4_32K_V2,
                tokens=32768,
                blocks=_ML4_NVFP4_MAX_BLOCKS,
                layer=43,
                far_end=True,
                large=True,
            ),
        ),
        Cell(
            why="a tep8, tp8 or dep8 rank's largest V2 pool, 147456 blocks (27.4 GB; 51.4 GiB "
            "peak): 32768 rows in its top blocks, descending as V2 hands them out, at pool row 43",
            spec=dict(
                _ML4_32K_V2,
                tokens=32768,
                blocks=_ML4_NVFP4_8GPU_MAX_BLOCKS,
                layer=43,
                far_end=True,
                large=True,
            ),
        ),
        Cell(
            why="1M manager chunk, layer 68 (pool row 42): one request's last 32768 rows at "
            "positions 1015808..1048575 over a 33900-block V2 quota, the contiguous 8-row "
            "launcher (large: the quota's pool)",
            spec=dict(
                _ML4_1M_MGR_TP4,
                tokens=32768,
                blocks=33900,
                requests=(32768,),
                cached=(1015808,),
                model_layer=68,
                large=True,
            ),
        ),
        Cell(
            why="1M manager chunk, NoPE layer 71 (pool row 43): identity table over positions "
            "1015808..1048575 (large: the quota's pool)",
            spec=dict(
                _ML4_1M_MGR_TP4,
                tokens=32768,
                blocks=33900,
                requests=(32768,),
                cached=(1015808,),
                model_layer=71,
                table="identity",
                large=True,
            ),
        ),
        Cell(
            why="1M manager packed chunk, layer 0: 4 requests, to positions 1048575 and 532479 "
            "beside a first chunk and a short one, over a 53000-block quota (large: its pool)",
            spec=dict(
                _ML4_1M_MGR_TP4,
                tokens=32768,
                blocks=53000,
                requests=_ML4_1M_PACKED_TP4,
                cached=_ML4_1M_PACKED_CACHED_TP4,
                model_layer=0,
                large=True,
            ),
        ),
        Cell(
            why="1M chunk on a V2 view, layer 42: 32768 rows at positions 1015808..1048575, the "
            "contiguous 8-row launcher",
            spec=dict(
                _ML4_1M_V2_TP4,
                tokens=32768,
                blocks=1100,
                requests=(32768,),
                cached=(1015808,),
                layer=42,
            ),
        ),
        Cell(
            why="1M chunk on a V2 view, NoPE layer 43: identity table over positions "
            "1015808..1048575",
            spec=dict(
                _ML4_1M_V2_TP4,
                tokens=32768,
                blocks=1100,
                requests=(32768,),
                cached=(1015808,),
                table="identity",
                layer=43,
            ),
        ),
        Cell(
            why="1M packed chunk on a V2 view, layer 0: the manager cell's 4 requests at their "
            "positions, slots drawn by the test",
            spec=dict(
                _ML4_1M_V2_TP4,
                tokens=32768,
                blocks=1100,
                requests=_ML4_1M_PACKED_TP4,
                cached=_ML4_1M_PACKED_CACHED_TP4,
                layer=0,
            ),
        ),
        Cell(
            why="1M mixed step on a V2 view, layer 20: 3 context chunks (one ending at 1048575) "
            "ahead of 13 decode rows across the 1M range",
            spec=dict(
                _ML4_1M_V2_TP4,
                tokens=32768,
                blocks=1100,
                requests=_ML4_1M_MIXED_TP4,
                cached=_ML4_1M_MIXED_CACHED_TP4,
                layer=20,
            ),
        ),
        Cell(
            why="1M chunk on the strided view: 32768 rows at positions 1015808..1048575, the "
            "strided 8-row launcher",
            spec=dict(_ML4_1M_TP4, tokens=32768, blocks=1100, requests=(32768,), cached=(1015808,)),
        ),
        Cell(
            why="1M-table decode batch 1 (captured), V2 view: position 1048575, the table's "
            "last row",
            spec=dict(_ML4_1M_V2_TP4, tokens=1, graph=True, top_position=True, layer=43),
        ),
        Cell(
            why="1M-table decode batch 2 (captured), V2 view, first row",
            spec=dict(_ML4_1M_V2_TP4, tokens=2, graph=True, top_position=True, layer=0),
        ),
        Cell(
            why="1M-table decode batch 4 (captured), V2 view",
            spec=dict(_ML4_1M_V2_TP4, tokens=4, graph=True, top_position=True, layer=17),
        ),
        Cell(
            why="1M-table decode batch 8 (captured), V2 view, its last 3 rows padding at slot -1",
            spec=dict(
                _ML4_1M_V2_TP4, tokens=8, graph=True, top_position=True, pad_rows=3, layer=30
            ),
        ),
        Cell(
            why="1M-table decode batch 16 (captured), V2 view, slot buffers of max_num_tokens "
            "32768 over a 64-block view",
            spec=dict(_ML4_1M_V2_TP4, tokens=16, graph=True, top_position=True, slot_rows=32768),
        ),
        Cell(
            why="1M-table NoPE decode batch 16 (captured), V2 view: identity table",
            spec=dict(
                _ML4_1M_V2_TP4, tokens=16, graph=True, top_position=True, table="identity", layer=43
            ),
        ),
        Cell(
            why="1M chunk on the engine V2 manager, layer 68 (pool row 42): the last 32768 rows "
            "of a 1048576-token prompt, positions 1015808..1048575 behind its cached prefix "
            "(a 33024-page quota, 57.1 GiB device peak: large)",
            spec=dict(
                _ML4_1M_MGR,
                blocks=33024,
                tokens=32768,
                requests=(32768,),
                cached=(1015808,),
                model_layer=68,
                large=True,
            ),
        ),
        Cell(
            why="1M chunk on the engine V2 manager, NoPE layer 71 (pool row 43): identity table "
            "at positions 1015808..1048575 (57.1 GiB device peak: large)",
            spec=dict(
                _ML4_1M_MGR,
                blocks=33024,
                table="identity",
                tokens=32768,
                requests=(32768,),
                cached=(1015808,),
                model_layer=71,
                large=True,
            ),
        ),
        Cell(
            why="1M packed chunk tails on the engine V2 manager, layer 0: 16 requests in 32768 "
            "rows behind prefixes up to 1048575 (a 65234-page quota, 110.7 GiB device peak: "
            "large)",
            spec=dict(
                _ML4_1M_MGR,
                blocks=65234,
                tokens=32768,
                requests=_ML4_1M_PACKED,
                cached=_ML4_1M_PACKED_CACHED,
                model_layer=0,
                large=True,
            ),
        ),
        Cell(
            why="1M mixed step on the engine V2 manager, layer 20 (pool row 18): 3 chunks behind "
            "prefixes up to 1028576 ahead of 13 decode rows at positions up to 1048575 (a "
            "137252-page quota, 144472 slots, slots past 2^34 bytes; 230.7 GiB device peak: "
            "large)",
            spec=dict(
                _ML4_1M_MGR,
                blocks=137252,
                tokens=32768,
                requests=_ML4_1M_MIXED,
                cached=_ML4_1M_MIXED_CACHED,
                model_layer=20,
                large=True,
            ),
        ),
        Cell(
            why="1M-table decode batch 1 (captured), V2 view, last row: position 1048575",
            spec=dict(_ML4_1M_V2, tokens=1, graph=True, top_position=True, layer=43),
        ),
        Cell(
            why="1M-table decode batch 2 (captured), V2 view, first row: positions over the "
            "whole table",
            spec=dict(_ML4_1M_V2, tokens=2, graph=True, top_position=True, layer=0),
        ),
        Cell(
            why="1M-table decode batch 4 (captured), V2 view",
            spec=dict(_ML4_1M_V2, tokens=4, graph=True, top_position=True, layer=17),
        ),
        Cell(
            why="1M-table decode batch 8 (captured), V2 view",
            spec=dict(_ML4_1M_V2, tokens=8, graph=True, top_position=True, layer=30),
        ),
        Cell(
            why="1M-table decode batch 16 (captured), V2 view",
            spec=dict(_ML4_1M_V2, tokens=16, graph=True, top_position=True, layer=42),
        ),
        Cell(
            why="1M-table NoPE decode batch 16 (captured), V2 view: identity table",
            spec=dict(
                _ML4_1M_V2, tokens=16, graph=True, top_position=True, table="identity", layer=43
            ),
        ),
        Cell(
            why="1M chunk on a V2 view (pool row 42): positions 1015808..1048575, the large "
            "manager cell's rows in pre-merge",
            spec=dict(
                _ML4_1M_V2,
                blocks=1100,
                tokens=32768,
                requests=(32768,),
                cached=(1015808,),
                layer=42,
            ),
        ),
        Cell(
            why="1M NoPE chunk on a V2 view (pool row 43): identity table at positions "
            "1015808..1048575",
            spec=dict(
                _ML4_1M_V2,
                blocks=1100,
                table="identity",
                tokens=32768,
                requests=(32768,),
                cached=(1015808,),
                layer=43,
            ),
        ),
        Cell(
            why="1M packed chunk tails on a V2 view (pool row 0): the large manager cell's 16 "
            "requests and prefixes in pre-merge",
            spec=dict(
                _ML4_1M_V2,
                blocks=1100,
                tokens=32768,
                requests=_ML4_1M_PACKED,
                cached=_ML4_1M_PACKED_CACHED,
                layer=0,
            ),
        ),
        Cell(
            why="1M mixed step on a V2 view (pool row 18): the large manager cell's 3 chunks and "
            "13 decode rows in pre-merge",
            spec=dict(
                _ML4_1M_V2,
                blocks=1100,
                tokens=32768,
                requests=_ML4_1M_MIXED,
                cached=_ML4_1M_MIXED_CACHED,
                layer=18,
            ),
        ),
        Cell(
            why="1M every position: 32 chunk calls of 32768 rows on a V2 view read table rows "
            "0..1048575, each at its own position",
            spec=dict(_ML4_1M_V2, blocks=1100, tokens=32768, layer=41, position_sweep=(0, 1048576)),
        ),
        Cell(
            why="dep4 1M boot capture, batch 1 (V2 view): captured on the warm-up dummy's key at "
            "position 1048574, replayed with the live row at 1048575, with attention DP's idle "
            "dummy at 0 and at a drawn position",
            spec=dict(_ML4_1M_DEP4_BOOT, tokens=1, layer=43),
        ),
        Cell(
            why="dep4 1M boot capture, batch 2 (V2 view): the warm-up key at 1048574 ahead of a "
            "one-key filler, replayed with one live row at 1048575 ahead of a padding-dummy row",
            spec=dict(_ML4_1M_DEP4_BOOT, tokens=2, pad_rows=1, layer=0),
        ),
        Cell(
            why="dep4 1M boot capture, batch 4 (V2 view): the warm-up key at 1048574 ahead of 3 "
            "one-key fillers, replayed with one live row at 1048575 ahead of 3 padding-dummy rows",
            spec=dict(_ML4_1M_DEP4_BOOT, tokens=4, pad_rows=3, layer=17),
        ),
        Cell(
            why="dep4 1M boot capture, batch 8 (V2 view): the warm-up key at 1048574 ahead of 7 "
            "one-key fillers, replayed with one live row at 1048575 ahead of 7 padding-dummy rows",
            spec=dict(_ML4_1M_DEP4_BOOT, tokens=8, pad_rows=7, layer=30),
        ),
        Cell(
            why="dep4 1M boot capture, batch 16 (V2 view), slot buffers of 32768: the warm-up key "
            "at 1048574 ahead of 15 one-key fillers, replayed with attention DP's one live row "
            "at 1048575 ahead of 15 padding-dummy rows on one slot",
            spec=dict(_ML4_1M_DEP4_BOOT, tokens=16, pad_rows=15, layer=42),
        ),
        Cell(
            why="dep4 1M mixed step on a V2 view (pool row 18): the large manager cell's 3 "
            "chunks and 13 decode rows in pre-merge",
            spec=dict(
                _ML4_1M_V2,
                blocks=1100,
                tokens=32768,
                requests=_ML4_1M_DEP4_MIXED,
                cached=_ML4_1M_DEP4_MIXED_CACHED,
                layer=18,
            ),
        ),
        Cell(
            why="dep4 1M chunk on the engine V2 manager, layer 68 (pool row 42): the last 32768 "
            "rows of a 1048576-token prompt, positions 1015808..1048575 behind its cached "
            "prefix (a 33792-page quota: large)",
            spec=dict(
                _ML4_1M_MGR,
                blocks=33792,
                tokens=32768,
                requests=(32768,),
                cached=(1015808,),
                model_layer=68,
                large=True,
            ),
        ),
        Cell(
            why="dep4 1M chunk on the engine V2 manager, NoPE layer 71 (pool row 43): identity "
            "table at positions 1015808..1048575 (a 33792-page quota: large)",
            spec=dict(
                _ML4_1M_MGR,
                blocks=33792,
                table="identity",
                tokens=32768,
                requests=(32768,),
                cached=(1015808,),
                model_layer=71,
                large=True,
            ),
        ),
        Cell(
            why="dep4 1M mixed step on the engine V2 manager, layer 20 (pool row 18): 3 chunks "
            "(one fresh, two behind 100000 and 65536 cached) ahead of 13 decode rows at "
            "positions up to 1048575 (a 73728-page quota: large)",
            spec=dict(
                _ML4_1M_MGR,
                blocks=73728,
                tokens=32768,
                requests=_ML4_1M_DEP4_MIXED,
                cached=_ML4_1M_DEP4_MIXED_CACHED,
                model_layer=20,
                large=True,
            ),
        ),
    )

    note = """
    `build(device_id, is_neox=..., cache_contiguous=...)` compiles the four
    launchers once; a target builds them on its first forward (the pool
    layout is an engine value) and passes the result as `compiled`. `__call__`
    picks the launcher on the host: 2 rows per CTA up to 1024 tokens, 8 above,
    and 8-byte loads when `indexer_k` is 8-byte aligned. There is no stock-op
    fallback; every row count runs the kernel. Metadata consumed: none beyond
    the arguments (the slot mappings and positions are the caller's).

    `validate=True` makes the kernel's wrapper check slot and position ranges
    on the device, which is a host sync and breaks a CUDA-graph capture: pass
    it on an eager call only. With it off those ranges are unchecked and
    `is_valid` states them. Captured with interleaved rope (`validate=False`)
    at decode batches 1, 2, 3, 4, 8 and 16 on a strided view, and at 1, 2, 4,
    8 and 16 on a V2 view (the contiguous launchers) at positions up to 32767
    over a 32768-row table and up to 1048575 over a 1048576-row one, a replay
    on new data writes the same bytes as the eager call; a NeoX capture is not
    driven. The dep4 engine's long-family graphs (`boot_capture` cells), each
    batch captured on its warm-up batch (a key at position 1048574 ahead of
    one-key fillers at 0) and replayed with attention DP's one live row at
    1048575, an idle rank's dummy at 0 or a batch at drawn positions, ahead of
    the padding dummy repeated (b - 1 rows writing one slot pair with one key),
    write the eager call's bytes; every code of those steps equals the
    reference's.

    Pool views. `get_indexer_k_cache_buffers(layer)` hands out, at
    Mistral Large 4's geometry (page 32, 132 bytes a token, 44 full indexer
    layers; measured at max_seq_len = max_num_tokens 32768, and at
    max_seq_len 1048576 with max_num_tokens 32768):
      DSACacheManagerV2 (the engine's default): contiguous
        `[slots * 44 - row, 32, 1, 132]`, starting `row * 4224` bytes into the
        indexer pool and running to its end, where `row` is the layer's rank
        among the full indexer layers. The page scale is 44: block b is page
        `b * 44` of every row's view. The slot range runs past the quota
        (2155 slots for a 2048-page quota; at max_seq_len 1048576, 34765 for
        33024, 68671 for 65234 and 144472 for 137252, and 35566 for 33792 and
        77621 for 73728), and a request's pages take descending slots, so the
        last chunk of a 1M request alone in its pool writes slots 0..1023. At
        max_seq_len 1048576 the block table
        is 32772 columns wide and positions up to 1048575 address columns up to
        32767 of it. A caller therefore runs the contiguous launchers.
        The pages of every later layer lie inside the view, so only slot
        mappings built with the page scale keep a call in its own layer; the
        op cannot see a mapping built without it.
      DSACacheManager (V1): `[blocks, 32, 1, 132]` with stride(0) `44 * 4224`;
        page scale 1, the strided launchers.
    `is_valid` and `validate=True` bound the slots by the view, not by the
    layer. At tep8, tp8 and dep8 a rank's weights leave room for about 144075
    and 132949 V2 slots (computed from the checkpoint's shard headers); no boot
    has measured those pools. V2 views are certified to 147456 blocks (the
    eight-GPU cell), rebuilt as above, not taken from a manager of that quota.
    At max_seq_len 1048576 with max_num_tokens 32768 (measured): a 33900-block
    quota gives 35675 slots and a 53000-block one 55786 (50.28 and 78.63 GiB of
    device memory with the 72 latent rows, outside torch's allocator; the
    manager cells read 57.97 and 89.79 GiB in use after their compare), and the
    block table is 32772 columns wide. A 1048576-token request takes 32768
    pages in descending slots, so its last 32768-token chunk lands in the
    lowest 1024 of them; the slot mappings are the engine's own arithmetic
    over block-table columns up to 32767.

    Rope modes: interleaved (Mistral Large 4) is certified; NeoX rope is
    accepted by the op but not certified here. The mode is compiled into the
    launchers, so `is_neox` must match `compiled.is_neox`. The kernel has no
    no-rope mode: a layer that takes no rope (Mistral Large 4's NoPE indexer
    layers) passes an identity table, cos 1 and sin 0 in every row it reads,
    so the rotation `x * 1 - y * 0` is exact and the keys are LayerNorm and
    quantization only. The kernel's
    LayerNorm sums in a different order and uses an approximate rsqrt, so a
    normed element now and then rounds to the other bf16 neighbour of the
    reference's (a handful of codes in 4194304 differ at 32768 rows). Rope
    carries that ulp into the rotated pair, and where the pair cancels the
    small result can land several e4m3 steps away (measured: 3 steps on one
    code of a 32768-row step). The test bounds each code by that chain.

    Refused by the compiled launcher itself, before any launch and with
    `validate` off (measured): `indexer_k` that is not row-contiguous, such as
    the first 128 columns of a `[T, 160]` wk|weights_proj output, raises
    `ValueError: Mismatched indexer_k.strides[0] on argument #0 ... expected to
    be 128` -- copy the keys contiguous first; fp32 `norm_weight`, `norm_bias`
    and `rotary_cos_sin`, int32 `position_ids` and int64 slot mappings are
    enforced (`Mismatched Tensor on argument #N ... expected dtype=...`);
    `position_ids` must have exactly `indexer_k`'s rows (`Mismatched
    position_ids.shape[0] on argument #3 ... expected to match
    indexer_k.shape[0]`), and so must the slot mappings after the wrapper
    slices them to those rows (shorter raises the same way; longer is
    accepted and certified).

    Cost: the call is bound by the host launch at every row count the targets
    pass.

    Not certified: pools of other than 44 and 72 layers (the layer
    count reaches the kernel only as the Int64 block stride, or as a V2
    view's start and length); interleaved positions past 1048575, or a table
    of more than 1048576 rows; NeoX rope; more than 32768
    rows; strided pools past 18432 blocks; rebuilt V2 views past 147456
    blocks, and manager views past 144472 slots (dep4's manager cells reach
    77621); the scalar-load launchers on a V2 view (the target's keys are a
    fresh contiguous copy, 8-byte aligned).

    Hard limits (sm_103)
      slot offset  contiguous launchers: int64 slot + int64 column, added to
                   the view's base pointer   boundary 2^31 / 2^32 bytes
                   probed at a 24576-block 44-row V2 view (4567415424 bytes),
                   slots on both sides of 2^31 and 2^32   passed
                   and at a 110592-block V2 view (20554005120 bytes), rows in
                   its top 1024 blocks up to byte 20554004991 (past 2^34)   passed
                   and at a 147456-block V2 view (27405400704 bytes, an
                   eight-GPU rank's largest pool), rows in its top 1024 blocks
                   up to byte 27405400575, scales to the view's last byte   passed
                   and on the engine manager's view at a 137252-page quota
                   (144472 slots; pool row 18's view 26850912000 bytes), slots
                   up to byte 25461342848   passed
                   and on the engine manager's view at max_seq_len 1048576 and a
                   73728-page quota (77621 slots; pool row 18's view
                   14426252544 bytes), code slots up to byte 13358028416   passed
      pool offset  strided launchers: block * block stride, Int64
                   boundary 2^31 bytes
                   probed at an 18432-block 44-layer pool (3425697792 bytes),
                   rows in the last block of the last layer   passed
      view pages   k_cache dim 0, Int32 in the compiled launchers
                   boundary 2^31 pages = 48806446 blocks of 44 rows
                   the largest view certified has 6488021 pages (tp4's 1M
                   manager cells' 1569658 and 2454584); not reached
      key row      token * 128 elements, Int32   boundary 2^31 / 128 = 16777216 rows
                   not reached by the envelope (32768 rows: element 4194303),
                   not probed
      position     table row * 64, Int64         no 32-bit product; position_ids
                   are int32, the envelope reaches 1048575 (row offset 67108800
                   elements, 268435200 bytes: under 2^31 either way); probed at
                   1048575, the last row of a 1048576-row table (elements up to
                   67108863)   passed; the 2^31 boundary not probed; no index
                   passes through fp32 (and 1048575 < 2^24)
      table rows   rotary_cos_sin dim 0, Int32 in the compiled launchers
                   boundary 2^31 rows; 1048576 certified; not reached
      slot         int64 slot mappings, Int64    no 32-bit product; tp4's 1M
                   manager cells' engine-built slots reach byte 9230170496
    """

    def __call__(
        self,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        *,
        is_neox: bool,
        compiled: CompiledIndexerKCacheUpdate,
        validate: bool = False,
    ) -> torch.Tensor:
        rows_per_block = _kernel._select_rows_per_block(int(num_tokens))
        vectorized = _kernel._can_vectorize_loads(indexer_k)
        _kernel.indexer_k_norm_rope_fp8_cache_update(
            indexer_k,
            norm_weight,
            norm_bias,
            position_ids,
            rotary_cos_sin,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            num_tokens,
            use_ue8m0=True,
            is_neox=is_neox,
            compiled_fn=compiled.fns[(rows_per_block, vectorized)],
            validate=validate,
        )
        return k_cache

    def reference(
        self,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        *,
        is_neox: bool,
        compiled: CompiledIndexerKCacheUpdate,
        validate: bool = False,
    ) -> torch.Tensor:
        """The cache image after the write, as a fresh contiguous tensor shaped like `k_cache`.

        `k_cache` is read (the bytes not written keep their value), not modified.
        LayerNorm and rope run in fp64 and are rounded to bf16 at the two points
        the kernel materialises bf16; the scale and codes are then exact.
        `compiled` and `validate` are not read.
        """
        n = int(num_tokens)
        dev = indexer_k.device
        x = indexer_k[:n].double()
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = (x - mean) * torch.rsqrt(
            var + NORM_EPS
        ) * norm_weight.double() + norm_bias.double()
        normed = normed.float().to(torch.bfloat16).double()

        pe, nope = normed[:, :_ROPE_DIM], normed[:, _ROPE_DIM:]
        table = rotary_cos_sin[position_ids[:n].long()].double()  # [n, 2, 32]
        cos, sin = table[:, 0], table[:, 1]
        if is_neox:
            partner = torch.cat([pe[:, _HALF:], pe[:, :_HALF]], dim=-1)
            cos_e, sin_e = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
            sign = torch.cat([-torch.ones(_HALF), torch.ones(_HALF)])
        else:
            partner = pe.reshape(n, _HALF, 2).flip(-1).reshape(n, _ROPE_DIM)
            cos_e, sin_e = cos.repeat_interleave(2, -1), sin.repeat_interleave(2, -1)
            sign = torch.tensor([-1.0, 1.0]).repeat(_HALF)
        sign = sign.to(device=dev, dtype=torch.float64)
        pe = (pe * cos_e + sign * partner * sin_e).float().to(torch.bfloat16)
        row = torch.cat([pe.float(), nope.float()], dim=-1)  # [n, 128] fp32, bf16 values

        amax = row.abs().amax(dim=-1)
        floor = torch.tensor(_MIN_AMAX, dtype=torch.float32, device=dev)
        inv_max = torch.tensor(1.0 / _E4M3_MAX, dtype=torch.float32, device=dev)
        bits = (torch.maximum(amax, floor) * inv_max).contiguous().view(torch.int32)
        exp_bits = bits & 0x7F800000
        exp_bits = torch.where((bits & 0x007FFFFF) != 0, exp_bits + 0x00800000, exp_bits)
        scale = exp_bits.view(torch.float32)
        inv_scale = (0x7F000000 - exp_bits).view(torch.float32)
        codes = (row * inv_scale[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
        scale_bytes = scale.contiguous().view(torch.uint8).reshape(n, _SCALE_BYTES)

        out = k_cache.clone(memory_format=torch.contiguous_format)
        flat = out.view(-1)
        fp8_slot = slot_mapping_fp8[:n].long()
        scale_slot = slot_mapping_scale[:n].long()
        live = (fp8_slot >= 0) & (scale_slot >= 0)
        cols = torch.arange(_HEAD_DIM, device=dev)
        flat[(fp8_slot[live].unsqueeze(1) + cols).reshape(-1)] = codes[live].reshape(-1)
        cols = torch.arange(_SCALE_BYTES, device=dev)
        flat[(scale_slot[live].unsqueeze(1) + cols).reshape(-1)] = scale_bytes[live].reshape(-1)
        return out

    def is_valid(
        self,
        indexer_k: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
        *,
        is_neox: bool,
        compiled: CompiledIndexerKCacheUpdate,
        validate: bool = False,
    ) -> None:
        # Both are compile-time constants the wrapper does not check against the
        # call: the other rope mode writes plausible wrong keys, and contiguous
        # addressing on a strided per-layer view writes into other layers.
        assert compiled.is_neox == bool(is_neox), (
            f"compiled for is_neox={compiled.is_neox}, called with is_neox={is_neox}"
        )
        assert not compiled.cache_contiguous or k_cache.is_contiguous(), (
            "compiled with contiguous cache addressing but k_cache is a strided view "
            f"(stride {tuple(k_cache.stride())}); the kernel writes outside it"
        )
        # Everything below is what `validate=True` checks on the device; with it
        # off (every call inside a CUDA graph) nothing does.
        n = int(num_tokens)
        assert 0 <= n <= indexer_k.shape[0], (
            f"num_tokens {n} exceeds the {indexer_k.shape[0]} rows of indexer_k"
        )
        if n == 0:
            return
        pos = position_ids[:n]
        assert int(pos.min()) >= 0 and int(pos.max()) < rotary_cos_sin.shape[0], (
            f"a position is outside the {rotary_cos_sin.shape[0]}-row rope table; "
            "the kernel reads past it"
        )
        fp8_slot, scale_slot = slot_mapping_fp8[:n], slot_mapping_scale[:n]
        assert bool(((fp8_slot >= 0) == (scale_slot >= 0)).all()), (
            "slot_mapping_fp8 and slot_mapping_scale disagree on which rows are written"
        )
        size = k_cache.numel()
        live = fp8_slot >= 0
        if bool(live.any()):
            f, s = fp8_slot[live], scale_slot[live]
            assert bool(((f % 4 == 0) & (s % 4 == 0)).all()), (
                "slots must be 4-byte aligned: the kernel stores 32-bit words"
            )
            assert int(f.max()) + _HEAD_DIM <= size and int(s.max()) + _SCALE_BYTES <= size, (
                f"a slot writes past the {size}-byte cache view"
            )


cute_dsl_indexer_k_cache_update = CuteDslIndexerKCacheUpdate()
