# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_router_tf32_partials entry on SM107: every cell's 8 K-slice partials are cuBLAS's own
(the TF32 split-K GEMM ``F.linear`` runs, on the K-masked input), their sum in slice order is
``F.linear``'s logits bit for bit, and each partial is within the FP32 accumulation band of an
FP64 product; the reference and the guard on the CPU."""

import contextlib
import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_router_tf32_partials import (  # noqa: E501
    rubin_router_tf32_partials,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

SPLITS = 8
SLICE = HIDDEN // SPLITS


def _operands(rows, seed, device):
    """K3.1's router operands: BF16 rows widened to FP32 (rubin_upcast_fp32's bits) and a BF16
    weight widened to FP32 (the checkpoint's, as dep16's generation server loads it)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, HIDDEN, generator=g) * 2.0 ** torch.randint(-4, 5, (rows, 1), generator=g)
    w = torch.randn(NUM_EXPERTS, HIDDEN, generator=g) * HIDDEN**-0.5
    return x.to(torch.bfloat16).float().to(device), w.to(torch.bfloat16).float().to(device)


@contextlib.contextmanager
def _tf32():
    """torch's FP32 matmul on TF32 tensor cores (the image's TORCH_ALLOW_TF32_CUBLAS_OVERRIDE)."""
    m = torch.backends.cuda.matmul
    prev = m.fp32_precision
    m.fp32_precision = "tf32"
    try:
        yield
    finally:
        m.fp32_precision = prev


def _fold(parts):
    acc = parts[0].clone()
    for part in parts[1:]:
        acc = acc + part
    return acc


@gpu
@pytest.mark.parametrize(
    "cell",
    rubin_router_tf32_partials.CELLS,
    ids=[c.why[:40] for c in rubin_router_tf32_partials.CELLS],
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_router_tf32_partials)
    rows = cell.spec["rows"]
    x, w = _operands(rows, zlib.crc32(cell.why.encode()), "cuda")
    parts = rubin_router_tf32_partials(x, w)
    assert parts.shape == (SPLITS, rows, NUM_EXPERTS) and parts.dtype == torch.float32
    assert torch.equal(parts, rubin_router_tf32_partials(x, w)), "two calls disagree"
    ref = rubin_router_tf32_partials.reference(x, w)
    # TF32-exact operands: the products are exact and only the FP32 accumulation rounds. A lost
    # K tile would be off by about 1/28 of the row's largest partial; the band is 2^-12 of it.
    band = ref.abs().amax(dim=-1, keepdim=True) * 2.0**-12
    assert bool(((parts - ref).abs() <= band).all()), "a partial is off the FP64 product"
    with _tf32():
        for s in range(SPLITS):
            masked = torch.zeros_like(x)
            masked[:, s * SLICE : (s + 1) * SLICE] = x[:, s * SLICE : (s + 1) * SLICE]
            cublas = F.linear(masked, w)
            assert torch.equal(parts[s].view(torch.int32), cublas.view(torch.int32)), (
                f"slice {s} differs from cuBLAS's partial"
            )
        logits = F.linear(x, w)
    assert torch.equal(_fold(parts).view(torch.int32), logits.view(torch.int32)), (
        "the partials' sum in slice order differs from F.linear's logits"
    )


@gpu
def test_cuda_graph_replay_on_new_rows():
    """Captured after one eager call; each replay on new rows writes their partials."""
    x, w = _operands(96, 1, "cuda")
    static = x.clone()
    rubin_router_tf32_partials(static, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rubin_router_tf32_partials(static, w)
    for seed in (2, 3):
        static.copy_(_operands(96, seed, "cuda")[0])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, rubin_router_tf32_partials(static, w))
        with _tf32():
            logits = F.linear(static, w)
        assert torch.equal(_fold(out).view(torch.int32), logits.view(torch.int32))


@pytest.mark.cpu_only
def test_the_reference_sums_to_the_product():
    x, w = _operands(5, 4, "cpu")
    ref = rubin_router_tf32_partials.reference(x, w)
    assert ref.shape == (SPLITS, 5, NUM_EXPERTS) and ref.dtype == torch.float32
    torch.testing.assert_close(ref.double().sum(0), x.double() @ w.double().t(), rtol=0, atol=1e-5)


@pytest.mark.cpu_only
def test_guard_admits_the_dep16_call():
    x, w = _operands(96, 5, "cpu")
    rubin_router_tf32_partials.is_valid(x, w)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case", ["no_rows", "short_k", "bf16_x", "bf16_weight", "strided_x", "weight_rows", "1d"]
)
def test_guard_refuses_what_the_launcher_misreads(case):
    x, w = _operands(8, 6, "cpu")
    if case == "no_rows":
        x = x[:0]
    elif case == "short_k":
        x = x[:, : HIDDEN // 2].contiguous()
    elif case == "bf16_x":
        x = x.to(torch.bfloat16)
    elif case == "bf16_weight":
        w = w.to(torch.bfloat16)
    elif case == "strided_x":
        x = torch.zeros(8, 2 * HIDDEN)[:, :HIDDEN]
    elif case == "weight_rows":
        w = w[:448]
    elif case == "1d":
        x = x[0]
    with pytest.raises(AssertionError):
        rubin_router_tf32_partials.is_valid(x, w)
