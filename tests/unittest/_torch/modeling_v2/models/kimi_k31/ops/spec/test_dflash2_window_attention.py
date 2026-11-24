# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 window-attention entry (TRTLLM-Gen sparse MLA).

The index lists come from the window-rows entry over random page tables, so every query row
attends its exact window (short contexts floor it at position 0). The kernel rounds the
probabilities and the output to BF16 where the FP32 reference does not, so the gate is a band
in units of the output's BF16 ulp, derived below.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_attention import (  # noqa: E501
    dflash2_window_attention as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_attention import (  # noqa: E501
    dflash2_window_attention_buffers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_rows import (  # noqa: E501
    dflash2_window_rows,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="the TRTLLM-Gen sparse MLA decode runs on SM100/SM103",
)

#: Bands in units of 2**-8 of each row's largest output (the scale `assert_within_ulp` uses).
#: Both sides round the output to BF16 once; the kernel also rounds the probabilities to BF16
#: (2**-9 relative each) before P.V, so its output lands on a neighbouring BF16 value of the
#: reference's about as often as not. Measured on the cells: at most 2 units per element and
#: 0.55 .. 0.82 units in relative RMS -- the size of one output rounding.
_BF16_BAND = dict(element_ulp=4.0, rms_ulp=1.0)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch, width, heads, page_size = spec["batch"], spec["width"], spec["heads"], spec["page_size"]
    dim = spec["latent"] + spec["rope"]
    ctx_len = torch.randint(0, 3000, (batch,), generator=g, device="cuda")
    ctx_len[0] = 9
    pages = (3000 + width) // page_size + 2
    num_pages = 2 * batch * pages
    tables = torch.randperm(num_pages, generator=g, device="cuda")[: batch * pages]
    tables = tables.view(batch, pages)
    rows, lengths = dflash2_window_rows(
        ctx_len, tables, width, page_size, spec["window"], spec["window"]
    )
    base = 0.5 * torch.randn(num_pages, spec["layers"], page_size, dim, generator=g, device="cuda")
    pool = base[:, spec["layers"] - 1].bfloat16()
    query = (0.5 * torch.randn(rows.shape[0], heads, dim, generator=g, device="cuda")).bfloat16()
    workspace, counter = dflash2_window_attention_buffers(rows.shape[0], heads, query.dtype, "cuda")
    return query, pool, rows, lengths, spec["scale"], spec["latent"], workspace, counter


def _check(out, args, spec) -> None:
    with true_fp32_matmul():
        ref = op.reference(*args)
    assert out.shape == ref.shape and out.dtype == ref.dtype, (out.shape, ref.shape)
    assert_within_ulp(out, ref, **_BF16_BAND)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    _check(op(*args), args, cell.spec)


@pytest.mark.parametrize("cell", op.CELLS[:2], ids=["96 heads", "48 heads"])
def test_launches_per_call(cell) -> None:
    """Device launches of the op and of the reference (printed): the kernel, plus the head
    padding at 96 heads."""
    args = _build(cell.spec, seed=1)
    op(*args), op.reference(*args)
    torch.cuda.synchronize()
    counts = {}
    for name, fn in (("op", op), ("reference", op.reference)):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            fn(*args)
            torch.cuda.synchronize()
        counts[name] = [
            e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
    print(
        f"device launches per call: { {k: len(v) for k, v in counts.items()} } op: {counts['op']}"
    )
    assert len(counts["op"]) <= 4, counts["op"]


def test_cuda_graph_replay() -> None:
    """Captured after one eager call, replayed on new queries: the eager values."""
    spec = op.CELLS[1].spec
    args = _build(spec, seed=2)
    op(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    for seed in (3, 4):
        args[0].copy_(_build(spec, seed=seed)[0])
        graph.replay()
        torch.cuda.synchronize()
        eager = op(*args)
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
        _check(out, args, spec)


def test_guard_refuses_unpacked_lengths() -> None:
    args = list(_build(op.CELLS[3].spec, seed=5))
    args[3] = torch.zeros_like(args[3])
    with validating(op), pytest.raises(AssertionError, match="lengths"):
        op(*args)
