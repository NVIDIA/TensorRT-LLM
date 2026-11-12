# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The non-absorbed context attention's K/V source rows over a cached prefix: the prefix's
latent rows from the fp8 paged pool, widened to bf16, and the step's new rows, packed per
request in one CuTe DSL launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_prefill_mha_prefix_gather_fp8_bf16 as _kernel

#: One latent row: kv_lora 512 + rope 64.
LATENT_DIM = _kernel.LAT

CompiledMhaPrefixGather = _kernel.CompiledKernel


def compile(device_index: int) -> CompiledMhaPrefixGather:  # noqa: A001 -- the kernel's own name
    """Compile the kernel for `device_index`; return the handle the op takes as `compiled`.
    Call once at load. Row counts are runtime values, so one compile serves every step.
    Costs one CuTe DSL compile and one launch on a one-row case, so no context step meets
    a first module load."""
    fn = _kernel.compile_gather(int(device_index))
    dev = torch.device("cuda", int(device_index))
    with torch.cuda.device(dev):
        _kernel.gather(
            fn,
            torch.zeros((4, LATENT_DIM), dtype=torch.uint8, device=dev),
            torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=dev),
            1,
            torch.zeros(1, dtype=torch.int32, device=dev),
            torch.zeros((1, 1), dtype=torch.int32, device=dev),
            torch.zeros((1, LATENT_DIM), dtype=torch.bfloat16, device=dev),
            torch.zeros((2, LATENT_DIM), dtype=torch.bfloat16, device=dev),
            1,
            2,
            2,
        )
        torch.cuda.synchronize(dev)
    return fn


class CuteDslMhaPrefixGather(OpWrapper):
    """Write the K/V source rows `[cached prefix | new rows]` of every context request
    into `lat`, packed back to back, and return `lat`.

    `idx` is int64 `[3C + T]`, `[dst_c | tok_c | pos_c | dst_n]` with `C =
    num_cached`, built on the host once per step. Cached work row `w < C` reads the
    pool row `block_table[req_idx[tok_c[w]], pos_c[w] // tpb] * stride_factor +
    pos_c[w] % tpb + layer_offset` -- where the absorbed dense-causal attention reads
    that position -- widens its 576 e4m3 bytes exactly to bf16 and stores them at
    row `dst_c[w]`; new row `t` copies `latent_new[t]` to row `dst_n[t]`.
    `pool_rows` is the paged latent pool as `[P, 576]` uint8, `req_idx` int32 per
    context token, `block_table` int32 `[B, NB]`, `latent_new` `[T, 576]` bf16 and
    `lat` `[T_k, 576]` bf16 (row pitches multiples of 8 elements, rows 16-byte
    aligned). Rows of `lat` that no work row names are left untouched; e4m3 NaN
    encodings become bf16's canonical NaN, as torch converts.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target's call: page 32, a three-layer pool, the new rows read in place from
    # the projection row (stride 2624). The cells vary the prefix: a partial block, a
    # whole block, several blocks, and a step mixing cached and uncached requests.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one request, a 17-token prefix ending mid-block, 64 new rows",
            spec=dict(prefixes=[17], news=[64]),
        ),
        Cell(
            why="prefixes of one token, one whole block and 300 tokens next to an uncached request",
            spec=dict(prefixes=[1, 32, 300, 0], news=[5, 40, 129, 77]),
        ),
        Cell(
            why="a 4100-row step over 8 requests with 25-token prefixes, the shape a "
            "served realtext step takes",
            spec=dict(prefixes=[25] * 8, news=[512] * 7 + [516]),
        ),
    )

    note = """
    The target takes this op on a dense-causal context step whose requests have a
    cached prefix (KV block reuse), once per trunk layer, before the kv_b expansion
    GEMM; a step without one expands the new rows in place. The kernel checks
    dtypes, shapes, devices, strides and alignment on every call and raises. What
    it cannot check is in `is_valid`: index vectors that name rows past `lat` or
    past a request's blocks, and a block table entry that is unallocated (-1).
    """

    def __call__(
        self,
        pool_rows: torch.Tensor,
        idx: torch.Tensor,
        num_cached: int,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        latent_new: torch.Tensor,
        lat: torch.Tensor,
        layer_offset: int,
        stride_factor: int,
        tokens_per_block: int,
        compiled: CompiledMhaPrefixGather,
    ) -> torch.Tensor:
        return _kernel.gather(
            compiled,
            pool_rows,
            idx,
            num_cached,
            req_idx,
            block_table,
            latent_new,
            lat,
            layer_offset,
            stride_factor,
            tokens_per_block,
        )

    def reference(
        self,
        pool_rows: torch.Tensor,
        idx: torch.Tensor,
        num_cached: int,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        latent_new: torch.Tensor,
        lat: torch.Tensor,
        layer_offset: int,
        stride_factor: int,
        tokens_per_block: int,
        compiled: CompiledMhaPrefixGather,
    ) -> torch.Tensor:
        """`lat` as the op leaves it, as a new tensor (`lat` itself is not written):
        the torch sequence the kernel replaces -- `index_select` of the pool rows,
        e4m3 -> bf16, two row scatters. `compiled` is ignored."""
        return _kernel.reference(
            pool_rows,
            idx,
            num_cached,
            req_idx,
            block_table,
            latent_new,
            lat.clone(),
            layer_offset,
            stride_factor,
            tokens_per_block,
        )

    def is_valid(
        self,
        pool_rows: torch.Tensor,
        idx: torch.Tensor,
        num_cached: int,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        latent_new: torch.Tensor,
        lat: torch.Tensor,
        layer_offset: int,
        stride_factor: int,
        tokens_per_block: int,
        compiled: CompiledMhaPrefixGather,
    ) -> None:
        """Reads the index vectors back to the host: never on the serving path."""
        c, t, tpb = int(num_cached), int(latent_new.shape[0]), int(tokens_per_block)
        assert idx.numel() == 3 * c + t, f"idx must hold 3 * {c} + {t} entries"
        dst_c, tok_c, pos_c, dst_n = (s.cpu() for s in torch.split(idx, [c, c, c, t], 0))
        rows = int(lat.shape[0])
        for name, dst in (("dst_c", dst_c), ("dst_n", dst_n)):
            assert dst.numel() == 0 or (int(dst.min()) >= 0 and int(dst.max()) < rows), (
                f"{name} names a row outside lat's {rows}"
            )
        if c == 0:
            return
        assert int(tok_c.min()) >= 0 and int(tok_c.max()) < req_idx.numel(), (
            "tok_c names a token outside req_idx"
        )
        slots = req_idx.cpu().long()[tok_c]
        blocks = block_table.cpu().long()
        assert int(slots.max()) < blocks.shape[0], "a request slot is past the block table"
        page = pos_c // tpb
        assert int(page.max()) < blocks.shape[1], "a prefix position is past the block table"
        block = blocks[slots, page]
        assert bool((block >= 0).all()), "a prefix position's block is unallocated (-1)"
        last = int(block.max()) * int(stride_factor) + tpb - 1 + int(layer_offset)
        assert last < pool_rows.shape[0], f"a pool row past the pool's {pool_rows.shape[0]}"


cute_dsl_mha_prefix_gather = CuteDslMhaPrefixGather()
