# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the k31_dflash2_shard_top32 and k31_dflash2_merge_top32 entries: every cell
bitwise against torch.topk, and the pair against the engine worker's ``_dflash2_global_top_k``
over an emulated TP group; the Lamport-mode pairs against the Lamport-gathered logits' top-32."""

import zlib
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm._torch.distributed.ops as dist_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.dflash2_top32 import (  # noqa: E501
    _pitched_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_allgather import (  # noqa: E501
    lamport_allgather,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_vocab_gather import (  # noqa: E501
    lamport_vocab_gather,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (  # noqa: E501
    k31_dflash2_merge_top32,
    k31_dflash2_shard_index_pairs,
    k31_dflash2_shard_top32,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the K3.1 DFlash2 top-32 entries are certified on sm_103 and sm_107",
)

#: K3.1 drafter vocabulary.
VOCAB = 163840


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Bit patterns, so NaN payloads compare too."""
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def _assert_bitwise(got, ref) -> None:
    got = got if isinstance(got, tuple) else (got,)
    ref = ref if isinstance(ref, tuple) else (ref,)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.dtype == r.dtype and g.shape == r.shape
        assert torch.equal(_bits(g), _bits(r))


def _signed_zeros(n, slots, width, dtype, g, positives=12, zeros=40):
    """Rows whose 32nd largest is a zero unless ``positives`` >= 32: ``positives`` positive values,
    ``zeros`` zeros (BF16: words of an even +0.0 and an odd -0.0, the Lamport sentinel word, and stray
    +0.0 / -0.0 on either parity; FP32: +0.0 and -0.0), every other value negative."""
    rows, count = n * slots, positives + zeros
    x = (-1.0 - torch.rand(rows, width, generator=g, device="cuda")).to(dtype)
    bits = x.view(torch.int16 if dtype == torch.bfloat16 else torch.int32)
    neg_zero = -(2**15) if dtype == torch.bfloat16 else -(2**31)
    noise = torch.rand(rows, width // 2, generator=g, device="cuda")
    even = noise.argsort(dim=-1)[:, :count] * 2  # distinct word starts
    j = torch.arange(count, device="cuda").expand(rows, count)
    odd = j & 1
    x.scatter_(1, even[:, :positives] + odd[:, :positives], (1.0 + j[:, :positives]).to(dtype))
    zj, ze, zo = j[:, positives:], even[:, positives:], odd[:, positives:]
    kind = zj % 4
    if dtype != torch.bfloat16:
        kind = torch.where(kind == 0, torch.full_like(kind, 3), kind)
    neg = torch.full_like(ze, neg_zero, dtype=bits.dtype)
    pos = torch.zeros_like(ze, dtype=bits.dtype)
    # kind 0: the sentinel word (+0.0 even, -0.0 odd); 1: -0.0 odd; 2: -0.0 even; 3: +0.0 on either parity.
    lo, hi = bits.gather(1, ze), bits.gather(1, ze + 1)
    lo = torch.where(kind == 0, pos, torch.where(kind == 2, neg, lo))
    lo = torch.where((kind == 3) & (zo == 0), pos, lo)
    hi = torch.where((kind == 0) | (kind == 1), neg, hi)
    hi = torch.where((kind == 3) & (zo == 1), pos, hi)
    bits.scatter_(1, ze, lo)
    bits.scatter_(1, ze + 1, hi)
    return x.view(n, slots, width)


def _logits(n, slots, width, dtype, values, g):
    if values == "zeros":
        return _signed_zeros(n, slots, width, dtype, g)
    if values == "ties":
        x = 0.25 * torch.randint(-3, 9, (n, slots, width), generator=g, device="cuda").float()
    else:
        x = 10.0 + 3.0 * torch.randn(n, slots, width, generator=g, device="cuda")
    if values == "special":
        cols = torch.randint(0, width, (6,), generator=g, device="cuda").tolist()
        x[..., cols[0]] = float("nan")
        x[..., cols[1]] = float("inf")
        x[..., cols[2]] = float("-inf")
        x[..., cols[3]] = 0.0
        x[..., cols[4]] = -0.0
        x[..., cols[5]] = float("nan")
    return x.to(dtype)


@pytest.mark.parametrize(
    "cell", k31_dflash2_shard_top32.CELLS, ids=[c.why[:48] for c in k31_dflash2_shard_top32.CELLS]
)
def test_shard_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    width = VOCAB // spec["tp"]
    logits = _logits(spec["n"], spec["slots"], width, spec["dtype"], spec["values"], g)
    pairs = spec["tp"] > 1
    id_offset = (spec["tp"] - 1) * width
    if spec["in_pitch"]:
        _assert_pitched_rows(logits, id_offset, spec, certifying)
        return
    if spec["out_width"]:
        _assert_pairs_into_row(logits, id_offset, spec["out_width"], certifying)
        return
    certifying(k31_dflash2_shard_top32)
    got = k31_dflash2_shard_top32(logits, 32, id_offset, pairs)
    _assert_bitwise(got, k31_dflash2_shard_top32.reference(logits, 32, id_offset, pairs))


def _assert_pitched_rows(logits, id_offset, spec, certifying) -> None:  # noqa: F811
    """Logits as the leading columns of ``in_pitch``-wide rows whose other columns hold +inf and
    NaN (what a read past the shard would select): the pairs written into the exchange row equal
    the packed logits' bit for bit, and the call does not copy the rows."""
    n, slots, width = logits.shape
    rows, pitch = n * slots, spec["in_pitch"]
    buffer = torch.full((rows, pitch), float("inf"), dtype=logits.dtype, device="cuda")
    buffer[:, width + 1 :: 2] = float("nan")
    buffer[:, :width] = logits.view(rows, width)
    pitched = buffer[:, :width].view(n, slots, width)
    assert not pitched.is_contiguous()
    exchange = torch.full((rows, spec["out_width"]), -7.25, dtype=torch.float32, device="cuda")
    out = exchange[:, spec["out_width"] - 64 :]
    certifying(k31_dflash2_shard_top32)
    got = k31_dflash2_shard_top32(pitched, 32, id_offset, True, out=out)
    want = k31_dflash2_shard_top32(logits, 32, id_offset, True)
    _assert_bitwise(got.reshape(rows, 64).contiguous(), want.reshape(rows, 64))
    _assert_bitwise(got, k31_dflash2_shard_top32.reference(logits, 32, id_offset, True))
    src = _pitched_rows(pitched.view(rows, width))
    assert src.data_ptr() == buffer.data_ptr() and src.shape == (rows, pitch), "rows were copied"


def _assert_pairs_into_row(logits, id_offset, row_width, certifying) -> None:  # noqa: F811
    """Pairs written into the last 64 columns of a ``row_width``-wide FP32 buffer: those columns
    are the contiguous result's bits, every other column keeps its sentinel, and the call returns
    the view of the written columns."""
    n, slots, _ = logits.shape
    rows = n * slots
    sentinel = torch.full((rows, row_width), -7.25, dtype=torch.float32, device="cuda")
    buffer = sentinel.clone()
    out = buffer[:, row_width - 64 :]
    certifying(k31_dflash2_shard_top32)
    got = k31_dflash2_shard_top32(logits, 32, id_offset, True, out=out)
    want = k31_dflash2_shard_top32(logits, 32, id_offset, True)
    assert got.data_ptr() == out.data_ptr() and got.shape == (n, slots, 64)
    _assert_bitwise(got.reshape(rows, 64).contiguous(), want.reshape(rows, 64))
    _assert_bitwise(got, k31_dflash2_shard_top32.reference(logits, 32, id_offset, True))
    assert torch.equal(_bits(buffer[:, : row_width - 64]), _bits(sentinel[:, : row_width - 64]))


def _gathered(n, slots, tp, values, g, lamport=False):
    width = VOCAB // tp
    parts = []
    for rank in range(tp):
        shard = _logits(n, slots, width, torch.bfloat16, values, g)
        if lamport:
            parts.append(k31_dflash2_shard_index_pairs.reference(shard, rank * width, True))
        else:
            parts.append(k31_dflash2_shard_top32.reference(shard, 32, rank * width, True))
    if lamport:
        # What the Lamport all-gather delivers.
        return lamport_allgather.reference(parts, None, None).contiguous()
    return torch.cat(parts, dim=-1).contiguous()


@pytest.mark.parametrize(
    "cell", k31_dflash2_merge_top32.CELLS, ids=[c.why[:48] for c in k31_dflash2_merge_top32.CELLS]
)
def test_merge_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    lamport = spec["lamport"]
    gathered = _gathered(spec["n"], spec["slots"], spec["tp"], spec["values"], g, lamport)
    certifying(k31_dflash2_merge_top32)
    got = k31_dflash2_merge_top32(gathered, 32, lamport)
    _assert_bitwise(got, k31_dflash2_merge_top32.reference(gathered, 32, lamport))


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("n, slots", [(1, 7), (38, 4)])
def test_pair_matches_the_engine_worker(tp, n, slots, monkeypatch) -> None:
    """Shard top-32, the all-gather in rank order, merge: the engine worker's
    ``_dflash2_global_top_k`` on every rank of an emulated group, bit for bit."""
    g = torch.Generator(device="cuda").manual_seed(tp * 1000 + n)
    width = VOCAB // tp
    shards = [_logits(n, slots, width, torch.bfloat16, "randn", g) for _ in range(tp)]

    recorded = []

    def record(tensor, mapping, dim=-1, **kwargs):
        recorded.append(tensor.clone())
        return torch.zeros((*tensor.shape[:-1], tensor.shape[-1] * tp), device=tensor.device)

    def gather(tensor, mapping, dim=-1, **kwargs):
        return torch.cat(recorded, dim=-1)

    def engine(rank):
        worker = SimpleNamespace(
            mapping=SimpleNamespace(tp_size=tp, tp_rank=rank),
            _draft_logits_are_sharded=lambda logits, metadata: True,
        )
        return DFlashWorker._dflash2_global_top_k(worker, shards[rank], None, 32, VOCAB)

    if tp == 1:
        ids, unary, _ = engine(0)
        values, got_ids = k31_dflash2_shard_top32(shards[0], 32, 0, False)
        _assert_bitwise((values, got_ids), (unary, ids))
        return
    monkeypatch.setattr(dist_ops, "allgather", record)
    for rank in range(tp):
        engine(rank)
    monkeypatch.setattr(dist_ops, "allgather", gather)
    expected = [engine(rank) for rank in range(tp)]
    pairs = [k31_dflash2_shard_top32(shards[r], 32, r * width, True) for r in range(tp)]
    for rank in range(tp):
        _assert_bitwise(pairs[rank], recorded[rank])
    unary, ids = k31_dflash2_merge_top32(torch.cat(pairs, dim=-1), 32)
    for ids_ref, unary_ref, _ in expected:
        _assert_bitwise((unary, ids), (unary_ref, ids_ref))


def test_cuda_graph_replay() -> None:
    g = torch.Generator(device="cuda").manual_seed(5)
    logits = _logits(8, 7, VOCAB, torch.bfloat16, "randn", g)
    k31_dflash2_shard_top32(logits, 32, 0, False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            values, ids = k31_dflash2_shard_top32(logits, 32, 0, False)
    torch.cuda.current_stream().wait_stream(stream)
    logits.copy_(_logits(8, 7, VOCAB, torch.bfloat16, "ties", g))
    graph.replay()
    expected = torch.topk(logits, 32, dim=-1)
    torch.cuda.synchronize()
    _assert_bitwise((values, ids), (expected.values, expected.indices))


def test_guard_refuses_another_top_k() -> None:
    logits = torch.zeros(1, 4, 5120, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="selects the top 32"):
        with validating(k31_dflash2_shard_top32):
            k31_dflash2_shard_top32(logits, 16, 0, True)


def test_guard_refuses_a_strided_gather() -> None:
    gathered = torch.zeros(1, 4, 256, device="cuda")[..., :128]
    with pytest.raises(AssertionError, match="contiguous rows"):
        with validating(k31_dflash2_merge_top32):
            k31_dflash2_merge_top32(gathered, 32)


def test_merge_cuda_graph_replay() -> None:
    g = torch.Generator(device="cuda").manual_seed(6)
    gathered = _gathered(8, 7, 8, "randn", g)
    k31_dflash2_merge_top32(gathered, 32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            unary, ids = k31_dflash2_merge_top32(gathered, 32)
    torch.cuda.current_stream().wait_stream(stream)
    gathered.copy_(_gathered(8, 7, 8, "ties", g))
    graph.replay()
    expected = k31_dflash2_merge_top32.reference(gathered, 32)
    torch.cuda.synchronize()
    _assert_bitwise((unary, ids), expected)


@pytest.mark.parametrize(
    "cell",
    k31_dflash2_shard_index_pairs.CELLS,
    ids=[c.why[:48] for c in k31_dflash2_shard_index_pairs.CELLS],
)
def test_index_pairs_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    width = VOCAB // spec["tp"]
    logits = _logits(spec["n"], spec["slots"], width, spec["dtype"], spec["values"], g)
    id_offset = (spec["tp"] - 1) * width
    certifying(k31_dflash2_shard_index_pairs)
    got = k31_dflash2_shard_index_pairs(logits, id_offset, spec["lamport"])
    _assert_bitwise(
        got, k31_dflash2_shard_index_pairs.reference(logits, id_offset, spec["lamport"])
    )


@pytest.mark.parametrize("values", ["randn", "ties", "special"])
@pytest.mark.parametrize(
    "tp, n, slots", [(32, 1, 7), (32, 4, 7), (16, 2, 4), (16, 24, 4), (8, 3, 7)]
)
def test_index_pairs_merge_is_the_gathered_logits_top32(tp, n, slots, values) -> None:
    """Every rank's index-ordered pairs, all-gathered in rank order and merged, equal the top-32
    of the all-gathered FP32 logits (the drafter's gathered-logits path) bit for bit: values, ids
    and their order."""
    g = torch.Generator(device="cuda").manual_seed(tp * 100 + n)
    full = _logits(n, slots, VOCAB, torch.bfloat16, values, g)
    width = VOCAB // tp
    pairs = [
        k31_dflash2_shard_index_pairs(
            full[..., r * width : (r + 1) * width].contiguous(), r * width
        )
        for r in range(tp)
    ]
    got = k31_dflash2_merge_top32(torch.cat(pairs, dim=-1).contiguous(), 32)
    expected = k31_dflash2_shard_top32(full.float(), 32, 0, False)
    _assert_bitwise(got, expected)
    ref = torch.topk(full.float(), 32, dim=-1)
    _assert_bitwise(got, (ref.values, ref.indices))


def _full_signed_zeros(n, slots, g, positives):
    """Full-vocab BF16 rows: ``positives`` positive values, ~200 zeros of every kind (sentinel words,
    stray +0.0 / -0.0 on either parity) spread over all shards, every other value negative."""
    return _signed_zeros(n, slots, VOCAB, torch.bfloat16, g, positives=positives, zeros=200)


@pytest.mark.parametrize("values", ["randn", "ties", "special", "zeros31", "zeros12", "zeros40"])
@pytest.mark.parametrize("tp, n, slots", [(32, 2, 7), (16, 16, 4), (8, 7, 4), (4, 3, 4), (2, 2, 4)])
def test_lamport_pairs_are_the_lamport_gathered_logits_top32(tp, n, slots, values) -> None:
    """The sampled drafter's shard path, every transport emulated by its reference: Lamport-mode
    index pairs of every rank's BF16 shard, the Lamport all-gather (-0.0 words arrive as +0.0), the
    Lamport-mode merge, equal the gathered path bit for bit: the Lamport vocab gather (sentinel words
    arrive as two +0.0) and the top-32 of the FP32 rows. Zero rows (``zeros<P>``: P positive values,
    200 signed zeros and sentinel words, the rest negative) put the 32nd largest deep among the zeros
    (12), on the first zero (31) or on a positive value with the zeros left out (40)."""
    g = torch.Generator(device="cuda").manual_seed(tp * 1000 + n * 10 + len(values))
    if values.startswith("zeros"):
        full = _full_signed_zeros(n, slots, g, positives=int(values[5:]))
    else:
        full = _logits(n, slots, VOCAB, torch.bfloat16, values, g)
    width = VOCAB // tp
    shards = [full[..., r * width : (r + 1) * width].contiguous() for r in range(tp)]
    gathered_logits = lamport_vocab_gather.reference(
        [s.view(n * slots, width) for s in shards], None, None
    ).view(n, slots, VOCAB)
    expected = k31_dflash2_shard_top32(gathered_logits, 32, 0, False)
    pairs = [k31_dflash2_shard_index_pairs(shards[r], r * width, True) for r in range(tp)]
    for r in range(tp):
        _assert_bitwise(
            pairs[r], k31_dflash2_shard_index_pairs.reference(shards[r], r * width, True)
        )
    delivered = lamport_allgather.reference(pairs, None, None).contiguous()
    got = k31_dflash2_merge_top32(delivered, 32, True)
    _assert_bitwise(got, expected)
    ref = torch.topk(gathered_logits.float(), 32, dim=-1)
    _assert_bitwise(got, (ref.values, ref.indices))


def test_plain_pairs_lose_the_signed_zero_at_the_32nd() -> None:
    """Why the sampled path takes the Lamport mode: without it, a 32nd largest among signed zeros
    selects or orders differently from the gathered logits once the exchange turns -0.0 into +0.0
    (the selection keys -0.0 below +0.0)."""
    g = torch.Generator(device="cuda").manual_seed(11)
    tp, n, slots = 8, 4, 4
    full = _full_signed_zeros(n, slots, g, positives=12)
    width = VOCAB // tp
    shards = [full[..., r * width : (r + 1) * width].contiguous() for r in range(tp)]
    gathered_logits = lamport_vocab_gather.reference(
        [s.view(n * slots, width) for s in shards], None, None
    ).view(n, slots, VOCAB)
    expected = k31_dflash2_shard_top32(gathered_logits, 32, 0, False)
    pairs = [k31_dflash2_shard_index_pairs(shards[r], r * width) for r in range(tp)]
    delivered = lamport_allgather.reference(pairs, None, None).contiguous()
    got = k31_dflash2_merge_top32(delivered, 32)
    same = torch.equal(got[0].view(torch.int32), expected[0].view(torch.int32)) and torch.equal(
        got[1], expected[1]
    )
    assert not same
