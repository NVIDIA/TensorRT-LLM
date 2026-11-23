# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP2CTX_ROUTER_BF16S (not bitwise) numerics on the GPU: the BF16 N-stacked split router GEMM
(minimax_m3/ops/router_bf16_split, ``trtllm::cublas_mm``) against the stock TF32 ``F.linear`` on the FP32 rows.

- At 1024 / 16384 rows the split logits are closer to the FP64 product than TF32 (max |error|), agree
  with the FP64 top-4 (sigmoid + correction bias) at least as often, and repeat bit for bit.
- The gate with the item on returns the split logits; off, the stock TF32 ``F.linear``.
- Reported, not asserted: p99.99 |error| per arm and top-4 agreement against the stock router.

The boundary's case-5 flavor (BF16 norm, no FP32 copy) is covered bit for bit by
test_modeling_v2_m31_dep2ctx_normq_v2_gpu.py (every write_norm / write_f32 / fp4 flavor vs the served op).
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401
import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._utils import get_sm_version

H = 6144
E = 128
TOPK = 4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)


@pytest.fixture(scope="module")
def mods():
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (  # noqa: E501
        modeling,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import router_bf16_split

    return modeling, router_bf16_split


@pytest.fixture
def tf32_on():
    """Today's served precision: torch's fp32 matmul as TF32."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _inputs(rows, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(E, H, generator=g, device="cuda") * 0.02
    scale = 2.0 ** torch.randint(-2, 3, (rows, 1), generator=g, device="cuda").float()
    x = (torch.randn(rows, H, generator=g, device="cuda") * scale).to(torch.bfloat16)
    bias = torch.randn(E, generator=g, device="cuda") * 0.01
    return w, x, bias


def _top(logits, bias):
    return torch.topk(torch.sigmoid(logits) + bias.double(), TOPK, dim=-1).indices.sort(-1).values


@pytest.mark.parametrize("terms", [2, 3])
@pytest.mark.parametrize("rows", [1024, 16384])
def test_split_router_closer_to_fp64_than_tf32(mods, tf32_on, rows, terms):
    _, rbs = mods
    w, x, bias = _inputs(rows)
    wn = rbs.n_stacked_split(w, terms)
    new = rbs.logits(x, wn, terms)
    assert new.shape == (rows, E) and new.dtype == torch.float32 and new.is_contiguous()
    assert torch.equal(new, rbs.logits(x, wn, terms)), "run to run"
    old = torch.nn.functional.linear(x.float(), w)
    ref = x.double() @ w.double().t()
    err_new = (new.double() - ref).abs()
    err_old = (old.double() - ref).abs()
    top_ref = _top(ref, bias)
    agree_new = int((_top(new.double(), bias) == top_ref).all(-1).sum())
    agree_old = int((_top(old.double(), bias) == top_ref).all(-1).sum())
    flips = int((_top(new.double(), bias) != _top(old.double(), bias)).any(-1).sum())
    print(
        f"[ROUTER_BF16S] rows {rows} terms {terms}: max|err| {float(err_new.max()):.3e} vs TF32 "
        f"{float(err_old.max()):.3e}; p99.99 {float(err_new.flatten().float().quantile(0.9999)):.3e} vs "
        f"{float(err_old.flatten().float().quantile(0.9999)):.3e}; top-4 vs fp64 {agree_new} vs {agree_old} "
        f"/ {rows}; rows whose set changes vs today {flips}"
    )
    assert float(err_new.max()) <= float(err_old.max())
    assert agree_new >= agree_old
