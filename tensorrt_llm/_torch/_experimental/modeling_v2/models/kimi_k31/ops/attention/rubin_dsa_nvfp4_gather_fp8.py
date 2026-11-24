# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA over an NVFP4 latent cache, generation rows: the attended latent rows into an FP8 scratch.

The sparse MLA reads E4M3 latent rows on sm_107. Over an NVFP4 cache a generation step therefore
dequantizes the rows its Top-K selected, after the remap to latent-pool rows, into a scratch that
the attention op reads through ``aux_kv_cache_pool_ptr``, at compact indices instead of pool rows.

The NVFP4 latent row, shared with the context gather (``rubin_dsa_nvfp4_ctx_gather_fp8``): 576
values stored as 288 bytes of E2M1 codes (two per byte, the even element in the low nibble, bit 3
of a code its sign) and 36 E4M3 block scales (one per 16 values) in a region of their own, one row
per pool token. The KV cache manager's host tables address both regions: ``host_pool_pointers[p,
0]`` holds pool ``p``'s code and block-scale bases, ``host_pool_mapping[layer, 0]`` a layer's pool.
"""

from __future__ import annotations

import functools

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Values of a latent row (kv_lora 512 + rope 64).
LATENT_DIM = 576
#: Values per E4M3 block scale.
SCALE_BLOCK = 16
#: Bytes of E2M1 codes and E4M3 block scales of one latent row.
CODE_BYTES = LATENT_DIM // 2
SCALE_BYTES = LATENT_DIM // SCALE_BLOCK
#: Magnitudes of the E2M1 codes 0-7; bit 3 of a code is its sign.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
#: The largest finite E4M3 magnitude: the gathers saturate to it.
E4M3_MAX = 448.0

# A global dequantization scale of amax 100 over 448 x 6, the engine DSA backend's default: with
# it the gathers take their scaled branch, with 1.0 the unit one.
_AMAX100 = 100.0 / (448.0 * 6.0)
# Rows per decode chunk of ``nvfp4_rows_to_e4m3``.
_DECODE_ROWS = 1 << 14


@functools.cache
def _trtllm_ops():
    """``torch.ops.trtllm`` with the compiled ops registered, on the first call."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*

    return torch.ops.trtllm


def nvfp4_rows_to_e4m3(
    codes: torch.Tensor, scales: torch.Tensor, global_dequant_scale: torch.Tensor
) -> torch.Tensor:
    """NVFP4 latent rows in E4M3, rounded as the gather kernels round them.

    ``codes`` uint8 ``[N, 288]``, ``scales`` float8_e4m3fn ``[N, 36]``, ``global_dequant_scale``
    float32 (element 0 is read); returns float8_e4m3fn ``[N, 576]`` on ``codes``' device.

    A global scale of exactly 1.0 makes a value code x block scale, exact in fp16, rounded once to
    nearest-even E4M3 and saturated at 448. Any other global scale is folded into the block scale
    first (an fp32 product rounded to fp16), and the product with the code is rounded to fp16
    before the same E4M3 conversion. Rows are decoded in chunks to bound the working memory.
    """
    rows = codes.shape[0]
    device = codes.device
    scale = global_dequant_scale.reshape(-1)[:1].to(device=device, dtype=torch.float32)
    unit = float(scale) == 1.0
    magnitudes = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    out = torch.empty((rows, LATENT_DIM), dtype=torch.uint8, device=device)
    for start in range(0, rows, _DECODE_ROWS):
        chunk = codes[start : start + _DECODE_ROWS].view(torch.uint8)
        nibbles = torch.stack((chunk & 0x0F, chunk >> 4), dim=-1).reshape(-1, LATENT_DIM).long()
        values = magnitudes[nibbles & 7]
        values = torch.where(nibbles >= 8, -values, values)
        block = scales[start : start + _DECODE_ROWS].view(torch.float8_e4m3fn).float()
        block = block.repeat_interleave(SCALE_BLOCK, dim=-1)
        if unit:
            value = values * block
        else:
            value = (values.half() * (block * scale).half()).float()
        e4m3 = value.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        out[start : start + _DECODE_ROWS] = e4m3.view(torch.uint8)
    return out.view(torch.float8_e4m3fn)


def _spec(rows, **overrides) -> dict:
    spec = dict(
        rows=rows,
        topk=2048,
        pool_tokens=1 << 16,
        invalid=0,
        scale=1.0,
        scales="normal",
        pools=1,
    )
    spec.update(overrides)
    return spec


class RubinDsaNvfp4GatherFp8(OpWrapper):
    """``out[r, c]`` = pool row ``global_indices[r, c]`` of layer ``layer_idx``'s NVFP4 pool in E4M3
    (``nvfp4_rows_to_e4m3``) and ``compact_indices[r, c] = r * topk + c``, for every pair whose pool
    row lies in ``[0, num_pool_tokens)``; ``compact_indices[r, c] = -1`` and ``out[r, c]`` untouched
    elsewhere. Returns ``(out, compact_indices)``; the sparse MLA reads ``out`` through its
    ``aux_kv_cache_pool_ptr`` at the compact indices.

    ``host_pool_pointers`` int64 ``[pools, 2, 2]`` and ``host_pool_mapping`` int32 ``[layers, 2]`` on
    the CPU (the KV cache manager's tables), ``global_indices`` int32 ``[rows, topk]`` (the Top-K
    remap's pool rows, -1 for none), ``out`` float8_e4m3fn ``[rows, topk, 576]`` and
    ``compact_indices`` int32 ``[rows, topk]``, all three contiguous on the GPU;
    ``global_dequant_scale`` float32 on the GPU, element 0 read; ``layer_idx`` the layer's row of
    ``host_pool_mapping``; ``residual_dim`` 0; ``num_pool_tokens`` the rows of the pool.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode row, a unit global scale", spec=_spec(1)),
        Cell(
            why="DFlash2 k7 verify of 8 requests at the amax-100 global scale: 64 rows",
            spec=_spec(64, scale=_AMAX100),
        ),
        Cell(
            why="short requests: -1 Top-K slots stay -1 and their scratch rows untouched",
            spec=_spec(8, invalid=1500, scale=_AMAX100),
        ),
        Cell(
            why="block scales over the whole finite E4M3 range: E4M3 subnormals and saturation",
            spec=_spec(16, scales="wide"),
        ),
        Cell(
            why="block scales over the whole finite E4M3 range at the amax-100 scale",
            spec=_spec(16, scales="wide", scale=_AMAX100),
        ),
        Cell(
            why="two pools, the layer mapped to the second: the op follows host_pool_mapping",
            spec=_spec(8, pools=2),
        ),
        Cell(
            why="the k7 verify of 64 requests: 512 rows, a 604 MB scratch",
            spec=_spec(512, scale=_AMAX100),
        ),
    )
    note = """
    The family entry over the engine's ``trtllm::nvfp4_mla_kv_cache_gather``, the op the engine
    DSA backend runs for an NVFP4 cache's generation rows; every launch argument is the caller's.
    One warp per (row, column) pair, no reduction: the scratch is a function of the pool and the
    indices, so the cells gate bit for bit. Graph-safe: no allocation and no host sync; the op
    reads the two host tables on the host, so a captured graph keeps the pool addresses, which
    are fixed for the cache manager's life, and its launch shape depends on ``rows`` only.

    An FP8 layer beside an NVFP4 cache (the manager's FP8 tail layers) maps to a pool with no
    block-scale region; the op refuses it ("pool pointers must be non-null"). Such a layer keeps
    the FP8 gather-free path.
    """

    def __call__(
        self,
        host_pool_pointers: torch.Tensor,
        host_pool_mapping: torch.Tensor,
        global_indices: torch.Tensor,
        out: torch.Tensor,
        compact_indices: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _trtllm_ops().nvfp4_mla_kv_cache_gather(
            host_pool_pointers,
            host_pool_mapping,
            global_indices,
            out,
            compact_indices,
            global_dequant_scale,
            layer_idx,
            residual_dim,
            num_pool_tokens,
        )
        return out, compact_indices

    def reference(
        self,
        data_pool: torch.Tensor,
        scale_pool: torch.Tensor,
        global_indices: torch.Tensor,
        out: torch.Tensor,
        compact_indices: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """New ``(out, compact_indices)``: ``out``'s bytes with the gathered rows written.

        The one departure from mirroring, as in the catalog's paged-cache entries: the op reads
        the pool through raw addresses, so the reference takes the pool those address for
        ``layer_idx`` in place of the two host tables: ``data_pool`` uint8 ``[num_pool_tokens,
        288]`` (codes) and ``scale_pool`` float8_e4m3fn ``[num_pool_tokens, 36]`` (block scales).
        ``compact_indices`` (written, never read), ``layer_idx`` and ``residual_dim`` (0) are not
        used.
        """
        index = global_indices.long()
        rows, topk = index.shape
        valid = (index >= 0) & (index < num_pool_tokens)
        pairs = torch.arange(rows * topk, dtype=torch.long, device=index.device).view(rows, topk)
        compact = torch.where(valid, pairs, torch.full_like(pairs, -1)).to(torch.int32)
        written = out.view(torch.uint8).clone()
        picked = index[valid]
        codes = data_pool.view(torch.uint8)[picked]
        scales = scale_pool.view(torch.uint8)[picked].view(torch.float8_e4m3fn)
        written[valid] = nvfp4_rows_to_e4m3(codes, scales, global_dequant_scale).view(torch.uint8)
        return written.view(torch.float8_e4m3fn), compact

    def is_valid(
        self,
        host_pool_pointers: torch.Tensor,
        host_pool_mapping: torch.Tensor,
        global_indices: torch.Tensor,
        out: torch.Tensor,
        compact_indices: torch.Tensor,
        global_dequant_scale: torch.Tensor,
        layer_idx: int,
        residual_dim: int,
        num_pool_tokens: int,
    ) -> None:
        # The family cache stores no residual groups (mla_kv_cache_residual_dim 0): another value
        # reads a row layout that is not there.
        assert residual_dim == 0, (
            f"residual_dim must be 0 for the family NVFP4 latent cache; got {residual_dim}"
        )
        # The op takes any multiple of 16 and reads that many values per pool row: a narrower
        # scratch reads the latent rows at the wrong pitch.
        assert out.shape[-1] == LATENT_DIM, (
            f"the scratch holds {LATENT_DIM}-wide latent rows; got {out.shape[-1]}"
        )
        # A pool row at or past num_pool_tokens reads as -1 without a word. The remap never yields
        # one, so a bound below the pool drops keys the attention needs.
        if global_indices.numel():
            assert int(global_indices.max()) < num_pool_tokens, (
                "a pool row at or past num_pool_tokens: the bound is not the pool's size"
            )


rubin_dsa_nvfp4_gather_fp8 = RubinDsaNvfp4GatherFp8()
