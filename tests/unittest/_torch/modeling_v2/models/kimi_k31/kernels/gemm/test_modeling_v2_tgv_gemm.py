# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the family's TGV GEMM (FlashInfer's TGV with a pair-synchronized 2-CTA teardown): at the low-M plan
point of the first one-CTA and the first 2-CTA TGV config (the fork changes only the teardown, so the bits depend on
the config, not on the row), the output is the stock FlashInfer kernel's bit for bit."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.gemm.tgv_gemm import run_tgv
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.low_m_plans import K31_LOW_M_PLANS

__extra_import_path__ = ["../.."]
from _k31_utils import on_sm

pytestmark = on_sm(103)

_STOCK = {}


def _stock_tgv(a: torch.Tensor, w_t: torch.Tensor, out: torch.Tensor, cfg) -> torch.Tensor:
    """FlashInfer's stock TGV kernel class at ``cfg`` (the same compile path, without the pair teardown)."""
    import cuda.bindings.driver as cuda
    from flashinfer.gemm.kernels import tgv_gemm_cute_ext as tgv

    a_, b_, c_, _, layout = tgv._to_cute_swap(a, w_t, out, None)
    key = (cfg, layout)
    if key not in _STOCK:
        cta_m, cta_n, stages, two = cfg
        gemm = tgv.TgvGemmCuteExtKernel(
            acc_dtype=tgv.cutlass.Float32,
            cta_m=cta_m,
            cta_n=cta_n,
            cta_k=tgv._TGV_CUTE_EXT_CTA_K,
            num_ab_stage=stages,
            use_2cta=two,
            use_pdl=True,
            has_bias=False,
        )
        ra, rb, rc, _ = tgv._make_compile_repr_tensors(torch.bfloat16, False, *layout)
        _STOCK[key] = tgv.cute_ext.compile(
            tgv._bmm_no_bias, gemm, ra, rb, rc, tgv.make_fake_stream()
        )
    _STOCK[key](a_, b_, c_, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return out


def _tgv_points():
    """(layout, n, k, rows, config) of the first TGV plan row of the first one-CTA and the first 2-CTA config, at its
    last row."""
    seen = set()
    for layout, plan in K31_LOW_M_PLANS.items():
        for (n, k), rows in plan.items():
            for bound, choice in rows:
                if choice[0] == "tgv":
                    cfg = (choice[1], choice[2], choice[3], bool(choice[4]))
                    if cfg[3] not in seen:
                        seen.add(cfg[3])
                        yield layout, n, k, bound, cfg


POINTS = list(_tgv_points())


@pytest.mark.parametrize(
    "layout,n,k,m,cfg", POINTS, ids=[f"{p[0]}-{p[1]}x{p[2]}-m{p[3]}-{p[4]}" for p in POINTS]
)
def test_bitwise_to_stock_tgv(layout, n, k, m, cfg) -> None:
    gen = torch.Generator(device="cuda").manual_seed(m * 7919 + n + k)
    a = torch.randn(m, k, generator=gen, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, generator=gen, device="cuda") / k**0.5).to(torch.bfloat16)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    ref = torch.empty_like(out)
    run_tgv(a, w.t(), out, True, cfg)
    _stock_tgv(a, w.t(), ref, cfg)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (layout, n, k, m, cfg)
