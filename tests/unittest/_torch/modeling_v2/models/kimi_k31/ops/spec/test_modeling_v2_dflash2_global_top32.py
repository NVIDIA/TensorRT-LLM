# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the k31_dflash2_shard_top32, k31_dflash2_merge_top32 and
k31_dflash2_shard_index_pairs entries: a sample of each entry's cells bitwise against its
reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.comm.lamport_allgather import (  # noqa: E501
    lamport_allgather,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (  # noqa: E501
    k31_dflash2_merge_top32,
    k31_dflash2_shard_index_pairs,
    k31_dflash2_shard_top32,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: F401 — needs the path declared above

pytestmark = on_sm(
    103, 107, reason="the K3.1 DFlash2 top-32 entries are certified on sm_103 and sm_107"
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


SHARD_CELLS = sample_cells(k31_dflash2_shard_top32.CELLS, size_key="n")[:2]


@pytest.mark.parametrize("cell", SHARD_CELLS, ids=[c.why[:48] for c in SHARD_CELLS])
def test_shard_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    width = VOCAB // spec["tp"]
    logits = _logits(spec["n"], spec["slots"], width, spec["dtype"], spec["values"], g)
    pairs = spec["tp"] > 1
    id_offset = (spec["tp"] - 1) * width
    certifying(k31_dflash2_shard_top32)
    got = k31_dflash2_shard_top32(logits, 32, id_offset, pairs)
    _assert_bitwise(got, k31_dflash2_shard_top32.reference(logits, 32, id_offset, pairs))


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


MERGE_CELLS = sample_cells(k31_dflash2_merge_top32.CELLS, size_key="n")[:2]


@pytest.mark.parametrize("cell", MERGE_CELLS, ids=[c.why[:48] for c in MERGE_CELLS])
def test_merge_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    lamport = spec["lamport"]
    gathered = _gathered(spec["n"], spec["slots"], spec["tp"], spec["values"], g, lamport)
    certifying(k31_dflash2_merge_top32)
    got = k31_dflash2_merge_top32(gathered, 32, lamport)
    _assert_bitwise(got, k31_dflash2_merge_top32.reference(gathered, 32, lamport))


INDEX_PAIRS_CELLS = sample_cells(k31_dflash2_shard_index_pairs.CELLS, size_key="n")[:2]


@pytest.mark.parametrize(
    "cell",
    INDEX_PAIRS_CELLS,
    ids=[c.why[:48] for c in INDEX_PAIRS_CELLS],
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
