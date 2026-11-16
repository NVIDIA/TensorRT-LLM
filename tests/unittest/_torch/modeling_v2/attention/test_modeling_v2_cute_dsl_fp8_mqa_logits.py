# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local VR200 parity checks for the tensor-core DSA prefill scorer."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_fp8_mqa_logits import (
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_fp8_mqa_logits import (
    cute_dsl_fp8_mqa_logits as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.cute_dsl_gvr_topk_prefill import (
    cute_dsl_gvr_topk_prefill,
    gvr_workspace,
)

__extra_import_path__ = [".."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="requires a VR200/SM107 device",
)

#: The FP32 band, in ulps of each logit's magnitude (sum over heads of |w| |q| . |k|, times
#: the key scale): the Triton scorer's, which differs from this kernel only in the order its
#: FP32 sums run.
_ULPS = 8


@pytest.fixture(scope="module")
def compiled():
    return build(torch.cuda.current_device())


def _inputs(spec, seed: int):
    """A cell's queries, keys, scales, head weights and windows. `causal`: one request's chunk
    of m rows after n - m cached keys. `packed`: two such requests, the first 401 rows over the
    first 15000 keys and the rest over the remaining keys, so one query block straddles them.
    Otherwise the Triton scorer's scattered windows, one of them empty and some past n."""
    m, n = spec["m"], spec["n"]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(m, 32, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    k = torch.randn(n, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand(n, device="cuda", generator=generator) + 0.01
    weight = torch.randn(m, 64, device="cuda", generator=generator)[:, ::2]
    rows = torch.arange(m, device="cuda", dtype=torch.int32)
    if spec.get("causal"):
        start = torch.zeros(m, device="cuda", dtype=torch.int32)
        end = n - m + rows + 1
    elif spec.get("packed"):
        m1, n1 = 401, 15000
        start = torch.where(rows < m1, 0, n1).to(torch.int32)
        end = torch.where(rows < m1, n1 - m1 + rows + 1, n - (m - 1) + rows).to(torch.int32)
    else:
        start = rows % max(n // 2, 1)
        end = (start + rows * 71 + 1).clamp(max=n + 17)
        if m > 1:
            end[0] = start[0]
    if spec.get("strided"):

        def strided_view(tensor: torch.Tensor) -> torch.Tensor:
            storage = torch.zeros(tensor.numel() * 2, dtype=tensor.dtype, device=tensor.device)
            storage[::2].copy_(tensor)
            return storage[::2]

        scale, start, end = (strided_view(tensor) for tensor in (scale, start, end))
    return q, (k, scale), weight, start, end


def _magnitude(q, kv, weight) -> torch.Tensor:
    k, scale = kv
    magnitude = torch.zeros(q.shape[0], k.shape[0], device="cuda")
    with true_fp32_matmul():
        for head in range(32):
            magnitude += (q[:, head].float().abs() @ k.float().abs().t()) * weight[
                :, head : head + 1
            ].abs()
    return magnitude * scale[None, :]


@pytest.mark.parametrize("cell", op.CELLS, ids=[cell.why for cell in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — pytest fixture
    certifying(op)
    args = _inputs(cell.spec, cell.seed_base())
    actual = op(*args, compiled=compiled)
    assert actual.shape == (args[0].shape[0], args[1][0].shape[0])
    assert actual.stride(0) % 4 == 0
    reference = op.reference(*args)
    torch.testing.assert_close(torch.isneginf(actual), torch.isneginf(reference))
    live = torch.isfinite(reference)
    assert bool(torch.isfinite(actual[live]).all())
    error = (actual[live].double() - reference[live].double()).abs()
    magnitude = _magnitude(args[0], args[1], args[2])[live].double()
    assert float((error / magnitude.clamp_min(1e-30)).max()) <= _ULPS * 2**-23
    relative_rms = error.square().sum().sqrt() / reference[
        live
    ].double().square().sum().sqrt().clamp_min(1e-30)
    assert float(relative_rms) <= _ULPS * 2**-23


_TOPK_CELLS = [cell for cell in op.CELLS if cell.spec.get("causal") or cell.spec.get("packed")]


@pytest.mark.parametrize("cell", _TOPK_CELLS, ids=[cell.why for cell in _TOPK_CELLS])
def test_topk_agrees_with_reference(cell, compiled) -> None:
    """The prefill top-k of these logits selects the reference logits' selection, up to keys
    whose reference score lies within the band of the row's k-th score (a near-tie the FP32
    summation order may break either way)."""
    top_k = 2048
    q, kv, weight, start, end = _inputs(cell.spec, cell.seed_base())
    actual = op(q, kv, weight, start, end, compiled=compiled)
    reference = op.reference(q, kv, weight, start, end)
    # The reference on the scorer's 16-byte padded rows, as the top-k consumer reads it.
    padded = torch.empty((q.shape[0], actual.stride(0)), device="cuda")[:, : actual.shape[1]]
    padded.copy_(reference)
    workspace = gvr_workspace(q.device)
    selected = torch.full((q.shape[0], top_k), -7, device="cuda", dtype=torch.int32)
    expected = torch.full_like(selected, -7)
    cute_dsl_gvr_topk_prefill(actual, start, end, selected, top_k, workspace)
    cute_dsl_gvr_topk_prefill(padded, start, end, expected, top_k, workspace)
    band = _ULPS * 2**-23 * _magnitude(q, kv, weight)
    differing = 0
    for row in range(q.shape[0]):
        lo, hi = int(start[row]), int(end[row])
        if hi - lo <= top_k:
            torch.testing.assert_close(selected[row], expected[row], rtol=0, atol=0)
            continue
        ours = set(selected[row].tolist())
        theirs = set(expected[row].tolist())
        assert len(ours) == len(theirs) == top_k and -1 not in ours
        if ours == theirs:
            continue
        window = reference[row, lo:hi]
        kth = window.topk(top_k).values[-1]
        swapped = torch.tensor(sorted(ours ^ theirs), device="cuda", dtype=torch.long)
        gap = (window[swapped] - kth).abs()
        assert bool((gap <= band[row, lo + swapped]).all()), (row, swapped.tolist())
        differing += 1
    assert differing <= max(1, q.shape[0] // 20)


@pytest.mark.parametrize("n", [2051, 2530, 2533])
def test_prefill_topk_consumer(n: int, compiled) -> None:
    m, top_k = 17, 2048
    generator = torch.Generator(device="cuda").manual_seed(n)
    q = torch.randn(m, 32, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    k = torch.randn(n, 128, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand(n, device="cuda", generator=generator) + 0.01
    weight = torch.randn(m, 32, device="cuda", generator=generator)
    start = torch.zeros(m, device="cuda", dtype=torch.int32)
    end = torch.full_like(start, n)
    end[:3] = torch.tensor([0, 31, top_k], device="cuda", dtype=torch.int32)
    start[-1] = 32
    logits = op(q, (k, scale), weight, start, end, compiled=compiled)
    output = torch.full((m, top_k), -7, device="cuda", dtype=torch.int32)
    cute_dsl_gvr_topk_prefill(logits, start, end, output, top_k, gvr_workspace(q.device))
    for row in range(m):
        lo, hi = int(start[row]), int(end[row])
        if hi - lo <= top_k:
            expected = torch.full_like(output[row], -1)
            expected[: hi - lo] = torch.arange(hi - lo, device="cuda", dtype=torch.int32)
            torch.testing.assert_close(output[row], expected, rtol=0, atol=0)
        else:
            expected = torch.argsort(logits[row, lo:hi], descending=True, stable=True)[:top_k]
            torch.testing.assert_close(
                output[row].long().sort().values, expected.sort().values, rtol=0, atol=0
            )


def test_empty_calls(compiled) -> None:
    """No rows or no keys: an empty view, no launch."""
    q = torch.zeros(0, 32, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros(5, 128, device="cuda", dtype=torch.float8_e4m3fn)
    windows = torch.zeros(0, device="cuda", dtype=torch.int32)
    scale = torch.ones(5, device="cuda")
    out = op(q, (k, scale), torch.zeros(0, 32, device="cuda"), windows, windows, compiled=compiled)
    assert out.shape == (0, 5)
    q = torch.zeros(3, 32, 128, device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.zeros(0, 128, device="cuda", dtype=torch.float8_e4m3fn)
    windows = torch.zeros(3, device="cuda", dtype=torch.int32)
    out = op(
        q, (k, torch.ones(0, device="cuda")), torch.zeros(3, 32, device="cuda"), windows, windows,
        compiled=compiled,
    )  # fmt: skip
    assert out.shape == (3, 0)
