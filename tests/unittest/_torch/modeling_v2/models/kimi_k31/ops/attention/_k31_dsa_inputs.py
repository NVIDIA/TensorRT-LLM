# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the Kimi K3.1 DSA indexer entries' tests, built from a cell's spec.

Two data modes. ``int``: queries and keys are small integers (exact in E4M3), key scales powers
of two and weights small integers times 2^-4, so every product and partial sum of the logits is
an exactly representable fp32 value whatever the accumulation order and the kernel must equal the
fp64 reference bit for bit. ``randn``: Gaussian queries, UE8M0-quantized Gaussian keys and
Gaussian weights, where the accumulation order is visible and a derived band applies.
"""

from __future__ import annotations

import zlib

import torch

__extra_import_path__ = ["../.."]
from _k31_utils import on_sm

HEAD_DIM = 128
NUM_INDEX_HEADS = 32
#: Bytes per index-key slot of a page: 128 E4M3 key bytes, then (after all P keys) one fp32 scale.
BYTES_PER_TOKEN = HEAD_DIM + 4
#: block_kv the K3.1 metadata passes to DeepGEMM's schedule builder (256-key splits).
SCHEDULE_BLOCK_KV = 64

on_sm103 = on_sm(103, 107)


def seed_of(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


def num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def queries(rows: int, data: str, gen: torch.Generator) -> torch.Tensor:
    """E4M3 [rows, 32, 128] on the GPU."""
    if data == "int":
        q = torch.randint(-3, 4, (rows, NUM_INDEX_HEADS, HEAD_DIM), generator=gen).float()
    else:
        q = torch.randn(rows, NUM_INDEX_HEADS, HEAD_DIM, generator=gen)
    return q.to(torch.float8_e4m3fn).cuda()


def weights(rows: int, data: str, gen: torch.Generator) -> torch.Tensor:
    """fp32 [rows, 32] on the GPU."""
    if data == "int":
        return (torch.randint(-3, 4, (rows, NUM_INDEX_HEADS), generator=gen).float() / 16).cuda()
    return torch.randn(rows, NUM_INDEX_HEADS, generator=gen).cuda()


def keys(count: int, data: str, gen: torch.Generator):
    """(E4M3 keys [count, 128], fp32 scales [count]) on the CPU."""
    if data == "int":
        k = torch.randint(-3, 4, (count, HEAD_DIM), generator=gen).float()
        scales = torch.pow(2.0, torch.randint(-6, 2, (count,), generator=gen).float())
        return k.to(torch.float8_e4m3fn), scales
    raw = torch.randn(count, HEAD_DIM, generator=gen)
    amax = raw.abs().amax(dim=1).clamp_min(1e-4)
    scales = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    return (raw / scales[:, None]).to(torch.float8_e4m3fn), scales


def paged_cache(context_lens, page: int, data: str, gen: torch.Generator, page_stride_pad: int = 0):
    """An index-key cache holding each request's keys on random physical pages.

    Returns ``(kv_cache, block_table)``: the [pages, P, 1, 132] uint8 page view (a view with a
    padded page stride when ``page_stride_pad``) and the int32 [B, max_pages] table (unused
    entries 0). Unused pages hold random bytes.
    """
    pages_per_req = [-(-int(c) // page) for c in context_lens]
    num_pages = sum(pages_per_req) + 8
    page_bytes = page * BYTES_PER_TOKEN
    base = torch.randint(0, 256, (num_pages, page_bytes + page_stride_pad), generator=gen)
    base = base.to(torch.uint8)
    perm = torch.randperm(num_pages, generator=gen)
    table = torch.zeros(len(context_lens), max(pages_per_req), dtype=torch.int32)
    used = 0
    for b, (ctx, npages) in enumerate(zip(context_lens, pages_per_req)):
        ids = perm[used : used + npages]
        used += npages
        table[b, :npages] = ids.to(torch.int32)
        k, s = keys(npages * page, data, gen)
        for i, pid in enumerate(ids.tolist()):
            key_bytes = k[i * page : (i + 1) * page].view(torch.uint8).reshape(-1)
            base[pid, : page * HEAD_DIM] = key_bytes
            base[pid, page * HEAD_DIM : page_bytes] = (
                s[i * page : (i + 1) * page].contiguous().view(torch.uint8)
            )
    base = base.cuda()
    view = base[:, :page_bytes].view(num_pages, page, 1, BYTES_PER_TOKEN)
    return view, table.cuda()


def atom_ctx_offsets(next_n: int, num_atoms: int) -> list[int]:
    """Per-atom causal stagger of an in-kernel next_n split (atom 0 is the oldest)."""
    base, rem = divmod(next_n, num_atoms)
    return [
        next_n - (i * base + min(i, rem) + base + (1 if i < rem else 0)) for i in range(num_atoms)
    ]


def chunk_windows(requests, q_rows=None):
    """Row windows of a prefill chunk.

    ``requests``: (cached keys, new rows) per request; the chunk's key matrix holds each request's
    cached and new keys back to back. Returns ``(ks, ke, num_keys)`` (int32 CUDA [rows]) for the
    rows ``q_rows`` = (begin, end) of the chunk, or all rows.
    """
    ks, ke, start = [], [], 0
    for cached, new in requests:
        for i in range(new):
            ks.append(start)
            ke.append(start + cached + i + 1)
        start += cached + new
    ks = torch.tensor(ks, dtype=torch.int32)
    ke = torch.tensor(ke, dtype=torch.int32)
    if q_rows is not None:
        ks, ke = ks[q_rows[0] : q_rows[1]], ke[q_rows[0] : q_rows[1]]
    return ks.cuda(), ke.cuda(), start


def accumulation_band(q, kf, w, rows_cols_mask) -> torch.Tensor:
    """Per-element bound on an fp32 kernel's distance from the fp64 logits.

    A sum of n terms computed in fp32 in any order is within ``n * u * sum |terms|`` of the
    exact sum (u = 2^-24). The logit is a 128-term dot product per head, then a 32-term weighted
    sum of the ReLUs, so n = 128 + 32 bounds both stages; ``sum |terms|`` is
    ``sum_h |w_h| sum_d |q_hd| |k_d|``. ``kf`` holds the dequantized keys.
    """
    acc = torch.zeros(q.shape[0], kf.shape[0], dtype=torch.float64, device=q.device)
    for h in range(q.shape[1]):
        acc += (q[:, h].double().abs() @ kf.abs().t()) * w[:, h : h + 1].double().abs()
    band = (HEAD_DIM + NUM_INDEX_HEADS) * 2.0**-24 * acc
    return band.masked_fill(~rows_cols_mask, 0)


def dequantized_keys(kv_cache, block_table, ctx: int, b: int) -> torch.Tensor:
    """fp64 [ctx, 128]: request ``b``'s keys times their scales, read back from the cache."""
    page = kv_cache.shape[1]
    pages = block_table[b, : -(-ctx // page)].long()
    raw = kv_cache[pages].reshape(pages.shape[0], -1)
    k = raw[:, : page * HEAD_DIM].contiguous().view(torch.float8_e4m3fn)
    scales = raw[:, page * HEAD_DIM :].contiguous().view(torch.float32)
    return k.reshape(-1, HEAD_DIM)[:ctx].double() * scales.reshape(-1)[:ctx, None]


def check_paged_logits(out, ref, args, next_n_of_q: int, context_lens) -> None:
    """Bit-exact on integer data is checked by the caller; this applies the accumulation band
    request by request to real-valued data."""
    batch = args["q"].shape[0]
    causal = torch.isfinite(ref)
    q = args["q"].reshape(batch * next_n_of_q, NUM_INDEX_HEADS, HEAD_DIM)
    for b in range(batch):
        kf = dequantized_keys(args["kv_cache"], args["block_table"], int(context_lens[b]), b)
        rows = slice(b * next_n_of_q, (b + 1) * next_n_of_q)
        window = causal[rows, : kf.shape[0]]
        band = accumulation_band(q[rows], kf, args["weights"][rows], window)
        err = (out[rows, : kf.shape[0]].double() - ref[rows, : kf.shape[0]].double()).abs()
        err = err.masked_fill(~window, 0)
        assert (err <= band).all(), f"request {b}: max excess {float((err - band).max()):.3e}"


def raw_index_cache(num_pages: int, page: int, gen: torch.Generator, page_stride_pad: int = 0):
    """A random-byte index-key cache: ``(base, view)`` with ``view`` the [pages, P, 1, 132] uint8 page
    view the indexer passes (a view with a padded page stride when ``page_stride_pad``)."""
    page_bytes = page * BYTES_PER_TOKEN
    base = torch.randint(0, 256, (num_pages, page_bytes + page_stride_pad), generator=gen)
    base = base.to(torch.uint8).cuda()
    return base, base[:, :page_bytes].view(num_pages, page, 1, BYTES_PER_TOKEN)


def slot_mappings(num_tokens: int, num_pages: int, page: int, gen: torch.Generator):
    """``(fp8, scale)`` int64 CUDA slot mappings of ``num_tokens`` distinct token slots, as the DSA
    metadata computes them: a token at slot ``s`` of page ``p`` has its key bytes at
    ``p * P * 132 + s * 128`` and its scale at ``p * P * 132 + P * 128 + s * 4``."""
    slots = torch.randperm(num_pages * page, generator=gen)[:num_tokens]
    pages, pos = slots // page, slots % page
    block = pages * page * BYTES_PER_TOKEN
    fp8 = block + pos * HEAD_DIM
    scale = block + page * HEAD_DIM + pos * 4
    return fp8.to(torch.int64).cuda(), scale.to(torch.int64).cuda()
