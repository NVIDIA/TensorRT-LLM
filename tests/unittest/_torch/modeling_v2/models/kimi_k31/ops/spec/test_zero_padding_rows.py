# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k31_zero_padding_rows entry: every cell bitwise against the engine's
``SpecWorkerBase._zero_padding_rows`` (in place: the same tensor, real rows untouched), CUDA-graph replay with a
changing padding set, and the DFlash2 verify's call of it."""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec import dflash2_verify
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.zero_padding_rows import (  # noqa: E501
    k31_zero_padding_rows as op,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the K3.1 padding-rows entry is certified on sm_103"
)

DUMMY = 9


def _padding(kind: str, n: int) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool)
    if kind == "last":
        mask[-1] = True
    elif kind == "all":
        mask[:] = True
    elif kind == "ends":
        mask[0] = mask[-1] = True
    elif kind == "some":
        mask[1::5] = True
    return mask


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, rpr, vocab = spec["n"], spec["rows_per_request"], spec["vocab"]
    logits = 8.0 * torch.randn(n * rpr, vocab, generator=g, device="cuda")
    mask = _padding(spec["padding"], n).cuda()
    if spec["special"]:
        logits[:, ::7] = float("nan")
        logits[:, 3::11] = float("inf")
        logits[:, 5::13] = -0.0
    real = torch.randperm(DUMMY + 4 * n, generator=g, device="cuda")
    real = real[real != DUMMY][:n]
    slot_ids = torch.where(mask, torch.full_like(real, DUMMY), real).to(spec["ids"])
    return logits, slot_ids, DUMMY, rpr


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32)


def _engine(logits, slot_ids, dummy, rpr):
    meta = SimpleNamespace(batch_slot_ids=slot_ids, dummy_slot_row=dummy, is_ragged_verify=False)
    return SpecWorkerBase._zero_padding_rows(
        logits, meta, 0, slot_ids.shape[0], rpr, packed_verify=True
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    logits, slot_ids, dummy, rpr = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    original = logits.clone()
    engine = _engine(original, slot_ids, dummy, rpr)
    certifying(op)
    out = op(logits, slot_ids, dummy, rpr)
    expected = op.reference(original, slot_ids, dummy, rpr)
    assert out.data_ptr() == logits.data_ptr()
    assert torch.equal(_bits(out), _bits(expected))
    assert torch.equal(_bits(out), _bits(engine))
    keep = ~(slot_ids == dummy).repeat_interleave(rpr)
    assert torch.equal(_bits(out[keep]), _bits(original[keep]))


def test_cuda_graph_replay() -> None:
    """One captured launch; the padding set changes between replays through the slot-id buffer."""
    spec = op.CELLS[2].spec
    logits, slot_ids, dummy, rpr = _build(spec, seed=21)
    source = logits.clone()
    op(logits, slot_ids, dummy, rpr)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            op(logits, slot_ids, dummy, rpr)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.Generator(device="cuda").manual_seed(22)
    for _ in range(3):
        mask = torch.rand(slot_ids.shape[0], generator=g, device="cuda") < 0.3
        slot_ids.copy_(torch.where(mask, torch.full_like(slot_ids, dummy), slot_ids + 100))
        logits.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(logits), _bits(_engine(source, slot_ids, dummy, rpr)))


class _Worker:
    """A worker with the engine's padding-rows method (what the K3.1 workers inherit)."""

    _zero_padding_rows = staticmethod(SpecWorkerBase._zero_padding_rows)


def test_verify_call_site_is_the_engine_method() -> None:
    """dflash2_verify's helper: in place where the entry applies, the engine method where it does not."""
    logits, slot_ids, dummy, rpr = _build(op.CELLS[1].spec, seed=31)
    n = slot_ids.shape[0]
    original = logits.clone()
    worker = _Worker()
    meta = SimpleNamespace(batch_slot_ids=slot_ids, dummy_slot_row=dummy, is_ragged_verify=False)
    out = dflash2_verify._zero_padding_rows(worker, logits, meta, 0, n, rpr)
    assert out.data_ptr() == logits.data_ptr()
    assert torch.equal(_bits(out), _bits(_engine(original, slot_ids, dummy, rpr)))
    # No padding marker yet (dummy_slot_row 0): the engine method's no-op.
    meta0 = SimpleNamespace(batch_slot_ids=slot_ids, dummy_slot_row=0, is_ragged_verify=False)
    same = original.clone()
    assert dflash2_verify._zero_padding_rows(worker, same, meta0, 0, n, rpr) is same
    # A non-uniform layout keeps the engine method (its own shape check).
    odd = original[:-1].clone()
    assert dflash2_verify._zero_padding_rows(worker, odd, meta, 0, n, rpr) is odd


def test_guard_refuses_a_strided_tensor() -> None:
    logits = torch.zeros(10, 1024, device="cuda")[:, :512]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(logits, torch.zeros(2, dtype=torch.int64, device="cuda"), DUMMY, 5)
