# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 non-absorbed context attention's K/V source rows as MXF8: each context request's
`[cached prefix | new rows]` latent rows, the cached ones as their paged-pool e4m3 bytes and
the new ones block-quantized, written as the A operands of the MXF8 K and V expansion GEMMs in
one CuTe DSL launch."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_prefill_mha_kv_gather_mxf8 as _kernel

#: One latent row: kv_lora 512 + rope 64.
LATENT_DIM = _kernel.LAT
#: The V expansion's A operand: the latent's c_kv part.
KV_LORA = _kernel.KV_LORA

CompiledMhaKvGatherMxf8 = _kernel.CompiledKernel
sf_bytes_k = _kernel.sf_bytes_k
sf_bytes_v = _kernel.sf_bytes_v


def buffers(rows: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A step's operands for `rows` K/V source rows: `(a_k [rows, 576], a_v [rows, 512], sfa_k,
    sfa_v)`, uint8 e4m3 values and blocked e8m0 scales."""
    return (
        torch.empty(rows, LATENT_DIM, dtype=torch.uint8, device=device),
        torch.empty(rows, KV_LORA, dtype=torch.uint8, device=device),
        torch.empty(sf_bytes_k(rows), dtype=torch.uint8, device=device),
        torch.empty(sf_bytes_v(rows), dtype=torch.uint8, device=device),
    )


def compile(device_index: int) -> CompiledMhaKvGatherMxf8:  # noqa: A001 -- the kernel's own name
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
            *buffers(2, dev),
            1,
            2,
            2,
            True,
        )
        torch.cuda.synchronize(dev)
    return fn


class CuteDslMhaKvGatherMxf8(OpWrapper):
    """Write the K/V source rows `[cached prefix | new rows]` of every context request into
    the MXF8 operands `(a_k, sfa_k)` (576 columns) and `(a_v, sfa_v)` (the first 512).

    `idx` is int64 `[3C + T]`, `[dst_c | tok_c | pos_c | dst_n]` with `C = num_cached`, as
    `cute_dsl_mha_prefix_gather` takes it. Cached work row `w < C` reads the pool row
    `block_table[req_idx[tok_c[w]], pos_c[w] // tpb] * stride_factor + pos_c[w] % tpb +
    layer_offset` and stores its 576 e4m3 bytes unchanged at row `dst_c[w]`, block scale 2^0
    (the pool holds the latent at the checkpoint's KV scale 1.0). New row `t` is the roped bf16
    latent `latent_new[t]`, block-quantized with the activation quant's rule (per 32 elements a
    power-of-two scale from the amax, e4m3 values), at row `dst_n[t]` -- or at row `t` with
    `new_dst_from_idx=False` on a step without a cached prefix (`idx` then unused). The scale
    bytes of the padding rows up to the next multiple of 128, and of the K operand's two
    padding columns past 576, are 2^0.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="no cached prefix: 7300 new rows in order, a partial last 128-row block",
            spec=dict(prefixes=[0], news=[7300]),
        ),
        Cell(
            why="one request, a 17-token prefix ending mid-block, 64 new rows",
            spec=dict(prefixes=[17], news=[64]),
        ),
        Cell(
            why="prefixes of one token, one whole block and 300 tokens next to an uncached request",
            spec=dict(prefixes=[1, 32, 300, 0], news=[5, 40, 129, 77]),
        ),
        Cell(
            why="8 requests with 25-token prefixes, the shape a served realtext step takes",
            spec=dict(prefixes=[25] * 8, news=[912] * 7 + [916]),
        ),
    )

    note = """
    The target takes this op on a dense-causal context step of the fp8 context attention,
    once per trunk layer before the K and V expansion GEMMs (`cute_dsl_projection_mxf8`
    `kv_b_k8` / `kv_b_v8`); with a cached prefix (KV block reuse) it gathers the prefix rows
    from the pool, without one it quantizes the new rows in order. The kernel checks dtypes,
    shapes, devices, strides and alignment on every call and raises; `is_valid` checks the
    index vectors, which the kernel cannot.
    """

    def __call__(
        self,
        pool_rows: torch.Tensor,
        idx: torch.Tensor,
        num_cached: int,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        latent_new: torch.Tensor,
        a8: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        layer_offset: int,
        stride_factor: int,
        tokens_per_block: int,
        new_dst_from_idx: bool,
        compiled: CompiledMhaKvGatherMxf8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _kernel.gather(
            compiled, pool_rows, idx, num_cached, req_idx, block_table, latent_new, *a8,
            layer_offset, stride_factor, tokens_per_block, new_dst_from_idx,
        )  # fmt: skip
        return a8

    def reference(
        self,
        pool_rows: torch.Tensor,
        idx: torch.Tensor,
        num_cached: int,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        latent_new: torch.Tensor,
        a8: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        layer_offset: int,
        stride_factor: int,
        tokens_per_block: int,
        new_dst_from_idx: bool,
        compiled: CompiledMhaKvGatherMxf8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """`a8` as the op leaves it, as new tensors (`a8` itself is not written): the torch
        sequence the kernel replaces -- index_select of the pool rows, the activation quant's
        torch twin on the new rows, two row scatters, the scale bytes. `compiled` is
        ignored."""
        out = tuple(t.clone() for t in a8)
        _kernel.reference(
            pool_rows, idx, num_cached, req_idx, block_table, latent_new, *out,
            layer_offset, stride_factor, tokens_per_block, new_dst_from_idx,
        )  # fmt: skip
        return out

    def is_valid(
        self,
        pool_rows: torch.Tensor,
        idx: torch.Tensor,
        num_cached: int,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        latent_new: torch.Tensor,
        a8: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        layer_offset: int,
        stride_factor: int,
        tokens_per_block: int,
        new_dst_from_idx: bool,
        compiled: CompiledMhaKvGatherMxf8,
    ) -> None:
        """Reads the index vectors back to the host: never on the serving path."""
        c, t, tpb = int(num_cached), int(latent_new.shape[0]), int(tokens_per_block)
        rows = int(a8[0].shape[0])
        assert rows == c + t, f"the operands hold {rows} rows, not {c} cached + {t} new"
        if not new_dst_from_idx:
            assert c == 0, "a cached prefix needs the new rows' destinations from idx"
            return
        assert idx.numel() == 3 * c + t, f"idx must hold 3 * {c} + {t} entries"
        dst_c, tok_c, pos_c, dst_n = (s.cpu() for s in torch.split(idx, [c, c, c, t], 0))
        dst = torch.cat([dst_c, dst_n])
        assert int(dst.min()) >= 0 and int(dst.max()) < rows, "a destination is outside a8"
        assert torch.unique(dst).numel() == rows, "every row of a8 must be written exactly once"
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


cute_dsl_mha_kv_gather_mxf8 = CuteDslMhaKvGatherMxf8()
