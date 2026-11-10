# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H3 mixed-modality VSA routing for TRTLLM and CuTe DSL sparse attention.

The FastH3 inference contract uses 64-token (4, 4, 4) video tiles, dense
text/audio segments, and a learned additive compression gate.
"""

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseBackendForwardArgs,
)


@dataclass
class H3VSAMetadata:
    source: torch.Tensor
    inverse: torch.Tensor
    sizes: torch.Tensor
    valid_bits: torch.Tensor
    prefix_tiles: int
    sparsity: float


_context = ContextVar("h3_vsa", default=None)


@contextmanager
def set_h3_vsa_context(metadata):
    token = _context.set(metadata)
    try:
        yield
    finally:
        _context.reset(token)


def get_h3_vsa_context():
    metadata = _context.get()
    if metadata is None:
        raise RuntimeError("MiniMax-H3 VSA requires packed-sequence metadata.")
    return metadata


def build_h3_vsa_metadata(segments, sparsity, device):
    if not 0 <= sparsity < 1:
        raise ValueError("VSA sparsity must be in [0, 1).")
    tiles = []
    cursor = 0
    prefix_tiles = 0
    for segment in segments:
        if isinstance(segment, int):
            for start in range(0, segment, 64):
                tiles.append(torch.arange(cursor + start, cursor + min(start + 64, segment)))
                prefix_tiles += 1
            cursor += segment
        else:
            t, h, w = segment
            grid = torch.arange(cursor, cursor + t * h * w).reshape(t, h, w)
            for ti in range(0, t, 4):
                for hi in range(0, h, 4):
                    for wi in range(0, w, 4):
                        tiles.append(grid[ti : ti + 4, hi : hi + 4, wi : wi + 4].flatten())
            cursor += t * h * w
    source = torch.full((len(tiles), 64), -1, dtype=torch.long)
    sizes = torch.tensor([tile.numel() for tile in tiles])
    for index, tile in enumerate(tiles):
        source[index, : tile.numel()] = tile
    source = source.flatten()
    valid = source >= 0
    inverse = torch.empty(cursor, dtype=torch.long)
    inverse[source[valid]] = torch.arange(source.numel())[valid]
    words = (valid.reshape(-1, 32).long() * (1 << torch.arange(32))).sum(-1)
    return H3VSAMetadata(
        source.to(device),
        inverse.to(device),
        sizes.to(device),
        words.to(device=device, dtype=torch.uint32),
        prefix_tiles,
        sparsity,
    )


def _cutedsl_fine(q, k, v, selected, metadata):
    from ...cute_dsl_kernels.blackwell.video_sparse_attention.interface import (
        block_sparse_attn_from_indices_cute,
        is_cute_supported,
    )

    if not is_cute_supported(q) or q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "H3 CUTEDSL VSA requires matching FP16/BF16 tensors with head_dim=128 on Blackwell."
        )
    batch, _, heads, _ = q.shape
    ntiles = metadata.sizes.numel()
    prefix = metadata.prefix_tiles
    # Prefix queries are dense; video queries retain the prefix plus selected video tiles.
    indices = torch.zeros((batch, heads, ntiles, ntiles), device=q.device, dtype=torch.int32)
    indices[:, :, :prefix] = torch.arange(ntiles, device=q.device, dtype=torch.int32)
    indices[:, :, prefix:, :prefix] = torch.arange(prefix, device=q.device, dtype=torch.int32)
    indices[:, :, prefix:, prefix : prefix + selected.shape[-1]] = selected[:, :, prefix:]
    counts = torch.full(
        (batch, heads, ntiles), prefix + selected.shape[-1], device=q.device, dtype=torch.int32
    )
    counts[:, :, :prefix] = ntiles
    # The kernel shares a route count across each pair of query tiles. Duplicate
    # boundary queries so dense and sparse groups each have an even tile count.
    query_tiles = list(range(prefix))
    if prefix % 2:
        query_tiles.append(prefix - 1)
    query_tiles.extend(range(prefix, ntiles))
    if (ntiles - prefix) % 2:
        query_tiles.append(ntiles - 1)
    query_tiles = torch.tensor(query_tiles, device=q.device)
    restore = torch.arange(ntiles, device=q.device)
    restore[prefix:] += prefix % 2
    queries = q.view(batch, ntiles, 64, heads, -1).index_select(1, query_tiles).flatten(1, 2)
    fine, _ = block_sparse_attn_from_indices_cute(
        *[x.transpose(1, 2).contiguous() for x in (queries, k, v)],
        q2k_idx=indices.index_select(2, query_tiles),
        q2k_num=counts.index_select(2, query_tiles),
        variable_block_sizes=metadata.sizes.to(torch.int32),
    )
    return (
        fine.transpose(1, 2)
        .reshape(batch, -1, 64, heads, q.shape[-1])
        .index_select(1, restore)
        .flatten(1, 2)
    )


def h3_vsa_attention(q, k, v, gate, metadata, attention, *, backend="TRTLLM"):
    """Execute trained H3 selection, sparse fine attention and coarse merge."""
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or gate.shape != q.shape:
        raise ValueError("H3 VSA requires matching BSHD Q/K/V and compression gate.")
    if q.shape[1] != metadata.inverse.numel():
        raise ValueError("H3 VSA metadata does not match the packed sequence.")
    if backend not in ("TRTLLM", "CUTEDSL"):
        raise ValueError(f"Unsupported H3 VSA backend: {backend}")
    batch, _, heads, dim = q.shape
    ntiles = metadata.sizes.numel()
    valid = (metadata.source >= 0).view(1, -1, 1, 1)

    def tile_pool(x):
        tiled = x.index_select(1, metadata.source.clamp_min(0)).masked_fill(~valid, 0)
        pooled = tiled.view(batch, ntiles, 64, heads, dim).sum(2, dtype=torch.float32)
        pooled = pooled / metadata.sizes.view(1, -1, 1, 1)
        return tiled, pooled.transpose(1, 2)

    qt, qp = tile_pool(q)
    kt, kp = tile_pool(k)
    vt, vp = tile_pool(v)
    scores = torch.matmul(qp, kp.transpose(-2, -1)) * dim**-0.5
    coarse = torch.matmul(scores.softmax(-1), vp).transpose(1, 2).to(q.dtype)
    prefix = metadata.prefix_tiles
    topk = max(1, math.ceil((1 - metadata.sparsity) * (ntiles - prefix)))
    selected = scores[..., prefix:].topk(topk, dim=-1).indices + prefix
    if backend == "CUTEDSL":
        fine = _cutedsl_fine(qt, kt, vt, selected, metadata)
        fine = fine.index_select(1, metadata.inverse)
        coarse = coarse.index_select(1, metadata.inverse // 64)
        return torch.addcmul(fine, coarse, gate)
    mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)
    mask[..., :prefix] = True
    mask[:, :, :prefix, :] = True
    counts = mask.sum(-1, dtype=torch.int32)
    ends = counts.flatten().cumsum(0, dtype=torch.int32).view(batch, heads, -1)
    starts = ends - counts
    indptr = torch.cat((starts, ends[..., -1:]), dim=-1).contiguous()
    indices = mask.nonzero()[:, -1].to(torch.int32).contiguous()
    routes = BlockSparseForwardInputs(
        q_block_size=64,
        kv_block_size=64,
        max_blocks_per_row=ntiles,
        block_indptr=indptr,
        block_indices=indices,
        kv_valid_bits=metadata.valid_bits[None].expand(batch, -1).contiguous(),
    )
    fine = attention(
        qt.flatten(2),
        kt.flatten(2),
        vt.flatten(2),
        sparse_backend_args=SparseBackendForwardArgs(block_sparse_inputs=routes),
    ).view(batch, ntiles * 64, heads, dim)
    fine = fine.index_select(1, metadata.inverse)
    coarse = coarse.index_select(1, metadata.inverse // 64)
    return torch.addcmul(fine, coarse, gate)
