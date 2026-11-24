# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs and expected logits of the K3.1 Rubin FP4 MQA logits entries' tests.

Two data modes. ``int``: E2M1 codes with query exponents 2^-1..2^0, key exponents 2^-1..2^1 and
weights in multiples of 2^-4 up to 3/16, so every partial sum of a logit fits fp32's 24 bits and
the kernel must equal the fp64 logits bit for bit. ``randn``: Gaussian rows quantized by the MXFP4
reference and Gaussian weights, where the accumulation order is visible: each logit must lie within
n u sum|terms| of the fp64 one (n = 128 + 32: the per-head dot product, then the weighted head
sum). The rows decode with ``mxfp4_values``, which the quantize entry's test checks against an
independent derivation of the format.
"""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_fp4_quantize import (  # noqa: E501
    mxfp4_values,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.rubin_idx_fp4_quantize import (  # noqa: E501
    rubin_idx_fp4_quantize as quantize,
)

HEADS, DIM, DATA = 32, 128, 64
#: Bytes of one index key in a page: 64 key bytes, then (after all P keys) its exponent word.
TOKEN = DATA + 4
#: block_kv the DSA metadata passes to DeepGEMM's paged schedule builder.
SCHEDULE_BLOCK_KV = 64
QUERY_EXPONENTS, KEY_EXPONENTS = (126, 127), (126, 127, 128)
#: n u of the accumulation band.
_BAND = (DIM + HEADS) * 2.0**-24


def fp4_rows(count: int, data: str, gen: torch.Generator, exponents) -> tuple:
    """(packed uint8 [count, 64], exponent words int32 [count]) on the CPU."""
    if data == "int":
        packed = torch.randint(0, 256, (count, DATA), generator=gen).to(torch.uint8)
        pick = torch.randint(0, len(exponents), (count, 4), generator=gen)
        words = torch.tensor(exponents, dtype=torch.uint8)[pick].contiguous().view(torch.int32)
        return packed, words.view(count)
    x = torch.randn(count, DIM, generator=gen).bfloat16()
    packed, words = quantize.reference(x[:, :64], x[:, 64:])
    return packed.view(torch.uint8), words.view(count)


def weights(rows: int, data: str, gen: torch.Generator) -> torch.Tensor:
    """fp32 [rows, 32] on the CPU."""
    if data == "int":
        return torch.randint(-3, 4, (rows, HEADS), generator=gen).float() / 16
    return torch.randn(rows, HEADS, generator=gen)


def paged_cache(lens, page: int, data: str, gen: torch.Generator, pad: int = 0, device="cpu"):
    """Each request's keys on random physical pages of an FP4 cache: (the [pages, P, 1, 68] view,
    the int32 [B, max_pages] table, the base tensor). Unused pages hold random bytes; ``pad`` bytes
    widen the page stride (the V2 per-layer view)."""
    pages_per_req = [-(-int(c) // page) for c in lens]
    num_pages = sum(pages_per_req) + 8
    base = torch.randint(0, 256, (num_pages, page * TOKEN + pad), generator=gen).to(torch.uint8)
    perm = torch.randperm(num_pages, generator=gen)
    table = torch.zeros(len(lens), max(pages_per_req), dtype=torch.int32)
    used = 0
    for b, npages in enumerate(pages_per_req):
        ids = perm[used : used + npages]
        used += npages
        table[b, :npages] = ids.to(torch.int32)
        packed, words = fp4_rows(npages * page, data, gen, KEY_EXPONENTS)
        base[ids, : page * DATA] = packed.reshape(npages, page * DATA)
        base[ids, page * DATA : page * TOKEN] = words.view(torch.uint8).reshape(npages, page * 4)
    base = base.to(device)
    return base[:, : page * TOKEN].view(num_pages, page, 1, TOKEN), table.to(device), base


def _request_keys(cache: torch.Tensor, table: torch.Tensor, b: int, ctx: int) -> torch.Tensor:
    """fp64 [ctx, 128]: the first ``ctx`` keys of request ``b``, read through its block table."""
    page = cache.shape[1]
    ids = table[b, : -(-ctx // page)].long()
    raw = cache[ids].reshape(len(ids), -1)
    words = raw[:, page * DATA :].contiguous().view(torch.int32).reshape(-1)
    return mxfp4_values(raw[:, : page * DATA].reshape(-1, DATA), words)[:ctx].double()


def _paged_rows(args: dict, lens_2d: torch.Tensor):
    """Per causal row of a paged call: (row, its queries [32, 128], its keys [end, 128] and its
    fp64 weights), all fp64 on the CPU. Row ``b * n + t`` sees the keys before
    ``lens_2d[b, t] - n + t + 1``."""
    batch, next_n = args["q"].shape[:2]
    queries = mxfp4_values(args["q"].cpu(), args["sf_q"].cpu()).double()
    w = args["weights"].double().cpu()
    cache, table, lens = args["kv_cache"].cpu(), args["block_table"].cpu(), lens_2d.cpu()
    for b in range(batch):
        ctx = int(lens[b].max())
        if ctx == 0:
            continue
        keys = _request_keys(cache, table, b, ctx)
        for t in range(next_n):
            end = int(lens[b, t]) - next_n + t + 1
            if end > 0:
                yield b * next_n + t, queries[b, t], keys[:end], w[b * next_n + t]


def paged_logits(args: dict, lens_2d: torch.Tensor) -> torch.Tensor:
    """The logits of a paged call written out from each request's dense keys (``lens_2d``:
    int32 [B, n], each row's request length), -inf past each row's causal end."""
    batch, next_n = args["q"].shape[:2]
    out = torch.full((batch * next_n, args["max_context_len"]), float("-inf"))
    for row, queries, keys, w in _paged_rows(args, lens_2d):
        scores = torch.einsum("hd,jd->hj", queries, keys).relu()
        out[row, : keys.shape[0]] = torch.einsum("h,hj->j", w, scores).float()
    return out


def assert_paged_band(out: torch.Tensor, ref: torch.Tensor, args: dict, lens_2d) -> None:
    """Real-valued data: each causal logit of a paged call within the band of the fp64 one."""
    out, ref = out.cpu(), ref.cpu()
    for row, queries, keys, w in _paged_rows(args, lens_2d):
        end = keys.shape[0]
        terms = torch.einsum("hd,jd->hj", queries.abs(), keys.abs())
        band = _BAND * torch.einsum("h,hj->j", w.abs(), terms)
        err = (out[row, :end].double() - ref[row, :end].double()).abs()
        assert bool((err <= band).all()), f"row {row}: max excess {float((err - band).max()):.3e}"


def dense_band(args: dict, window: torch.Tensor) -> torch.Tensor:
    """Per-logit band of a dense call (``q`` [M, 32, 64], ``k`` [N, 64]): fp64 [M, N], 0 outside
    ``window``."""
    queries = mxfp4_values(args["q"], args["sf_q"]).double().abs()
    keys = mxfp4_values(args["k"], args["sf_k"]).double().abs()
    acc = torch.zeros(queries.shape[0], keys.shape[0], dtype=torch.float64, device=keys.device)
    for h in range(HEADS):
        acc += (queries[:, h] @ keys.t()) * args["weights"][:, h : h + 1].double().abs()
    return (_BAND * acc).masked_fill(~window, 0)
