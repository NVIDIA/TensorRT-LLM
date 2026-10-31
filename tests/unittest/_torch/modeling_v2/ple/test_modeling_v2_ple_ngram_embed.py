# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ple_ngram_embed catalog entry.

Integer bookkeeping plus a one-rounding dequantize, so gated bitwise -- the
embedding and the whole published pool -- against the entry's reference, which
is the scalar references of `ngram_token_history` and `ngram_hash` and a
float64 gather-and-scale. Beyond the cells: the `query_start_loc` length guard
and graph capture.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch
from cuda.bindings import runtime as cudart

from tensorrt_llm._torch._experimental.modeling_v2.catalog.embedding.ngram_hash import (  # noqa: E501
    CKPT_HEAD_OFFSETS,
    CKPT_HEAD_VOCAB_SIZES,
    CKPT_MULTIPLIERS,
    ngram_hash,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ple.ngram_token_history import (  # noqa: E501
    ngram_token_history,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ple.ple_ngram_embed import (  # noqa: E501
    CKPT_EOS_TOKEN_ID,
    CKPT_HEAD_DIM,
    CKPT_SCALE,
    CKPT_TABLE_ROWS,
    ENGINE_SLOTS,
    TEP4_SLOTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.ple.ple_ngram_embed import (  # noqa: E501
    ple_ngram_embed as op,
)
from tensorrt_llm._utils import CUASSERT

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "ple_ngram_embed requires a CUDA device"

DEV = "cuda"
EOS = CKPT_EOS_TOKEN_ID
VOCAB = 248320
FP8 = torch.float8_e4m3fn
#: Keeps the hosts alive for as long as the tables they back.
_HOSTS = []


def _host_mapped(src: torch.Tensor) -> torch.Tensor:
    """A device view of `src`'s bytes in page-locked host memory.

    The stock engine's placement of the n-gram table under its PLE host offload
    (`Qwen4ExpPinnedHostEmbedding` in `tensorrt_llm/_torch/modules/qwen4_exp/ple.py`):
    a pinned CPU allocation reached through `cudaHostGetDevicePointer`, so the
    kernel gathers rows over the host link and no byte of the table is in device
    memory -- which the pointer's own attributes are asked to confirm.
    """
    rows, cols = src.shape
    host = torch.empty(rows * cols * src.element_size(), dtype=torch.uint8, pin_memory=True)
    host.copy_(src.contiguous().view(torch.uint8).reshape(-1))
    _HOSTS.append(host)
    (pointer,) = CUASSERT(cudart.cudaHostGetDevicePointer(host.data_ptr(), 0))
    # No public torch API views a pointer the caching allocator did not hand out;
    # `|u1` because the array interface has no float8 typestr.
    interface = SimpleNamespace(
        __cuda_array_interface__={
            "shape": (rows, cols * src.element_size()),
            "typestr": "|u1",
            "data": (int(pointer), False),
            "version": 3,
            "strides": None,
        }
    )
    table = torch.as_tensor(interface, device=DEV).view(src.dtype)
    assert table.data_ptr() == int(pointer), "torch.as_tensor copied the host buffer"
    (attributes,) = CUASSERT(cudart.cudaPointerGetAttributes(table.data_ptr()))
    assert attributes.type == cudart.cudaMemoryType.cudaMemoryTypeHost, attributes.type
    assert table.shape == src.shape and table.dtype == src.dtype
    return table


def _scattered(n: int, pool_slots: int, g) -> list:
    """`n` distinct slots over the whole pool, both ends included; a lone
    sequence takes the last slot, the farthest row of the pool."""
    middle = (torch.randperm(pool_slots - 2, generator=g) + 1).tolist()
    slots = ([pool_slots - 1, 0] + middle)[:n]
    assert len(set(slots)) == n and max(slots) < pool_slots
    return slots


def _geometry(spec, g):
    """`(multipliers, vocab, offsets)` for the cell: the checkpoint's own over
    the full-height table, the checkpoint's multipliers over small stand-in head
    vocabularies otherwise."""
    if spec.get("table") == "ckpt":
        return CKPT_MULTIPLIERS, CKPT_HEAD_VOCAB_SIZES, CKPT_HEAD_OFFSETS
    mult = CKPT_MULTIPLIERS
    vocab = tuple(
        1000 + 7 * h + int(x)
        for h, x in enumerate(torch.randint(0, 50, (len(CKPT_HEAD_VOCAB_SIZES),), generator=g))
    )
    offsets, acc = [], 0
    for v in vocab:
        offsets.append(acc)
        acc += v
    return mult, vocab, tuple(offsets)


def _table(kind: str, rows: int, touched: torch.Tensor, g) -> torch.Tensor:
    if kind == "ckpt":
        table = torch.empty(CKPT_TABLE_ROWS, CKPT_HEAD_DIM, dtype=FP8, device=DEV)
        table[touched] = torch.randn(touched.numel(), CKPT_HEAD_DIM, generator=g).to(FP8).to(DEV)
        return table
    src = torch.randn(rows, CKPT_HEAD_DIM, generator=g).to(FP8)
    if kind == "host":
        return _host_mapped(src)
    return src.to(DEV)


def _with_eos(x: torch.Tensor, density: float, g) -> torch.Tensor:
    if density > 0.0:
        x = torch.where(torch.rand(x.shape, generator=g) < density, torch.full_like(x, EOS), x)
    return x


def _build(spec, seed):
    g = torch.Generator().manual_seed(seed)
    lengths = spec["lengths"]
    n = len(CKPT_MULTIPLIERS)
    num_seqs = len(lengths)
    pool_slots = spec.get("pool_slots", num_seqs + 2)
    density = spec.get("eos_density", 0.0)
    ids = _with_eos(torch.randint(0, VOCAB, (sum(lengths),), generator=g), density, g)
    ids = ids.to(torch.int32).to(DEV)
    pool = _with_eos(torch.randint(0, VOCAB, (pool_slots, n - 1), generator=g), density, g).to(DEV)
    slots = _scattered(num_seqs, pool_slots, g) if pool_slots >= 64 else list(range(num_seqs))
    slots = torch.tensor(slots, dtype=torch.int32, device=DEV)
    carries = spec.get("carries")
    carries = None if carries is None else torch.tensor(carries, dtype=torch.bool, device=DEV)
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    qsl = torch.tensor(starts, dtype=torch.int64, device=DEV)
    mult, vocab, offsets = _geometry(spec, g)
    mult = torch.tensor(mult, dtype=torch.int64, device=DEV)
    vocab = torch.tensor(vocab, dtype=torch.int64, device=DEV)
    offsets = torch.tensor(offsets, dtype=torch.int64, device=DEV)
    # The rows this batch touches, from the two scalar references, so that the
    # full-height table needs bytes only where they are read.
    seeded = torch.ones(num_seqs, dtype=torch.bool, device=DEV) if carries is None else carries
    window, _ = ngram_token_history.reference(ids, pool, slots, seeded, qsl, EOS, n)
    touched = ngram_hash.reference(window, mult, vocab, offsets, EOS, 1).reshape(-1)
    rows = int(offsets[-1] + vocab[-1])
    table = _table(spec.get("table", "small"), rows, touched, g)
    scale = torch.tensor([CKPT_SCALE], dtype=torch.bfloat16, device=DEV).reshape(1, 1, 1)
    max_rows = spec.get("max_rows", max(lengths))
    return [ids, pool, slots, carries, qsl, table, scale, mult, vocab, offsets, EOS, max_rows]


def _compare(out, ref) -> None:
    (emb, pool), (ref_emb, ref_pool) = out, ref
    assert emb.dtype == ref_emb.dtype and emb.shape == ref_emb.shape
    assert emb.is_contiguous()
    assert torch.equal(emb, ref_emb), (
        f"embedding differs at {(emb != ref_emb).any(-1).nonzero().flatten()[:8].tolist()}"
    )
    assert torch.equal(pool, ref_pool), "published pool differs"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    ref = op.reference(*args)
    pool_before = args[1].clone()
    out = op(*args)
    _compare((out, args[1]), ref)
    if cell.spec.get("table") == "host":
        assert args[1].data_ptr() != 0 and not torch.equal(args[1], pool_before)


def test_guard_refuses_short_query_start_loc() -> None:
    args = _build(dict(lengths=(1, 1, 1)), seed=3)
    args[4] = args[4][:-1]
    with validating(op), pytest.raises(AssertionError):
        op(*args)


#: Captured decode shapes: tp1's, and the unified target's 128-request decode and MTP7
#: verify (the packed-row form's two launches) at a tep4 rank's pool.
GRAPH_SPECS = (
    dict(lengths=(1,) * 16, pool_slots=ENGINE_SLOTS, eos_density=0.1),
    dict(lengths=(1,) * 128, pool_slots=TEP4_SLOTS, table="host", eos_density=0.1),
    dict(lengths=(8,) * 128, pool_slots=TEP4_SLOTS, table="host", eos_density=0.1),
)


@pytest.mark.parametrize(
    "spec",
    GRAPH_SPECS,
    ids=[f"{len(s['lengths'])}x{s['lengths'][0]}@{s['pool_slots']}" for s in GRAPH_SPECS],
)
def test_cuda_graph_capture_and_replay(spec) -> None:
    args = _build(spec, seed=11)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        op(*args)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(*args)
    fresh = _build(spec, seed=12)
    args[0].copy_(fresh[0])
    pool_before = args[1].clone()
    want, want_pool = op.reference(*args)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_out, want)
    assert torch.equal(args[1], want_pool)
    assert not torch.equal(args[1], pool_before)
