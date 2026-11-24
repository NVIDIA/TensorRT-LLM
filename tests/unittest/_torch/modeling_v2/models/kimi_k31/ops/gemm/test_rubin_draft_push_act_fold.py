# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one GPU) for the push GEMM's ``act`` mode (``kernels/spec/rubin_draft_gemm_push``):
``silu(x[:, :K]) * x[:, K:]`` computed in the activation load is bit for bit FlashInfer's
``silu_and_mul`` followed by the plain kernel. The weight's first K rows are the identity, so the
first K output columns are the activation itself (a product with one nonzero term is exact, but a
``-0`` comes out ``+0``: the accumulator starts at ``+0``); the other rows check the GEMM. Eager,
and in a CUDA graph behind the gate / up projection that produces ``x`` (the served PDL chain). Above
8 rows each token CTA folds its own 8-row tile: 9-16 rows (two token CTAs) are checked the same way
and against the FP64 product of ``silu_and_mul`` within one BF16 rounding of FP32 sums."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="the push GEMM is an sm_103 / sm_107 kernel",
)

N, K = 7168, 896  # the helix16 drafter's down / 16
TACTIC = (128, 8, 1, 6)  # the drafter's _DRAFT_PUSH_TACTIC without its "splitk" tag


def _kernel():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
        rubin_draft_gemm_push as base,
    )

    return base


def _weight(seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device="cuda") * K**-0.5).bfloat16()
    w[:K] = torch.eye(K, device="cuda", dtype=torch.bfloat16)
    return w


def _gate_up(rows: int, seed: int, fill: str) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * K, generator=g, device="cuda")
    if fill == "wide":
        # Gates far into both tails (exp under- and overflow), exact zeros, signed zeros.
        x[:, :K] *= torch.logspace(-3, 2, K, device="cuda")
        x[:, 0:8] = 0.0
        x[:, 8:16] = -0.0
        x[:, 16:24] = torch.tensor([-200.0, -90.0, -88.7, -87.0, 87.0, 88.7, 90.0, 200.0])
    return x.bfloat16()


def _pair(x: torch.Tensor, w: torch.Tensor):
    base = _kernel()
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

    tactic = SplitKTactic(*TACTIC)
    rows = x.shape[0]
    folded = base.run_gemm(
        x, w, torch.empty(rows, N, dtype=torch.bfloat16, device="cuda"), tactic, act=True
    )
    act = flashinfer_silu_and_mul(x)
    plain = base.run_gemm(act, w, torch.empty(rows, N, dtype=torch.bfloat16, device="cuda"), tactic)
    return folded, plain, act


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _through_identity(act: torch.Tensor) -> torch.Tensor:
    """``act`` as the identity rows of the GEMM return it: ``-0`` becomes ``+0``."""
    return torch.where(act == 0, torch.zeros_like(act), act)


@pytest.mark.parametrize("fill", ["randn", "wide"])
@pytest.mark.parametrize("rows", [1, 3, 8, 9, 13, 16])
def test_act_fold_bitwise_eager(rows: int, fill: str) -> None:
    w = _weight(rows)
    x = _gate_up(rows, 100 + rows, fill)
    folded, plain, act = _pair(x, w)
    torch.cuda.synchronize()
    # The identity rows: the folded activation itself, bit for bit FlashInfer's.
    bad = int((_bits(folded[:, :K]) != _bits(_through_identity(act))).sum())
    assert bad == 0, f"{bad} of {act.numel()} activation values differ from silu_and_mul"
    assert torch.equal(_bits(folded), _bits(plain)), "the product differs from act_and_mul + GEMM"
    cublas = act.double() @ w.double().t()
    err = ((folded.double() - cublas).abs().max() / cublas.abs().max()).item()
    assert err < 1e-2, f"relative max error {err} against the FP64 product of silu_and_mul"


@pytest.mark.parametrize("rows", [8, 16])
def test_act_fold_graph_behind_producer(rows: int) -> None:
    """In one graph: a producer kernel writes ``x`` in place (as the gate / up GEMM does), then the
    folded GEMM reads it after its PDL wait. Replayed with new inputs."""
    base = _kernel()
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

    tactic = SplitKTactic(*TACTIC)
    w = _weight(11)
    src = _gate_up(rows, 12, "randn")
    x = torch.empty_like(src)
    out = torch.empty(rows, N, dtype=torch.bfloat16, device="cuda")
    base.run_gemm(src, w, out, tactic, act=True)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x.copy_(src)
        base.run_gemm(x, w, out, tactic, act=True)
    for seed in range(4):
        src.copy_(_gate_up(rows, 200 + seed, "wide" if seed % 2 else "randn"))
        graph.replay()
        torch.cuda.synchronize()
        _, plain, _ = _pair(src, w)
        assert torch.equal(_bits(out), _bits(plain)), f"replay {seed}"


def test_act_fold_rejects_unsupported() -> None:
    base = _kernel()
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import SplitKTactic

    with pytest.raises(ValueError):
        base.check_act_tactic(SplitKTactic(128, 16, 1, 6), K)
    with pytest.raises(ValueError):
        base.check_act_tactic(SplitKTactic(128, 8, 2, 6), K)
    with pytest.raises(ValueError):
        base.check_act_tactic(SplitKTactic(128, 8, 1, 6), 900)


def test_act_fold_division_variants(monkeypatch) -> None:
    """Diagnostic: which PTX division reproduces silu_and_mul (counts only; the bitwise gate is the
    served variant's, above)."""
    base = _kernel()
    w = _weight(5)
    x = _gate_up(8, 6, "wide")
    counts = {}
    for div in ("div.full.ftz.f32", "div.approx.ftz.f32", "div.rn.ftz.f32", "div.rn.f32"):
        monkeypatch.setattr(base, "SILU_DIV", div)
        base._compiled.cache_clear()
        folded, _, act = _pair(x, w)
        torch.cuda.synchronize()
        counts[div] = int((_bits(folded[:, :K]) != _bits(_through_identity(act))).sum())
    base._compiled.cache_clear()
    print(f"K31_ACT_FOLD_DIV mismatches of {x.shape[0] * K}: {counts}")
