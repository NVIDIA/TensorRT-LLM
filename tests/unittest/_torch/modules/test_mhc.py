# Tests for Multi-Head Hyper-Connection (mHC) module
from collections import defaultdict

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch.modules.mhc.hyper_connection import HCHead, mHC

BENCH_WARMUP = 50
BENCH_ITERS = 200

timing_stats = defaultdict(dict)


# ---------------------------------------------------------------------------
# Vanilla (PyTorch) reference implementations for correctness testing
# ---------------------------------------------------------------------------


def _sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def vanilla_pre_mapping(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    mult: int,
    norm_eps: float,
    eps: float,
    sinkhorn_eps: float,
    post_mult_value: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference pre_mapping implementation in pure PyTorch."""
    assert mult == x.shape[-2]
    residual_flat = x.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + norm_eps).rsqrt()
    scale_expanded = torch.cat(
        [
            scale[0].expand(mult),
            scale[1].expand(mult),
            scale[2].expand(mult * mult),
        ],
    )
    mixes = mixes * scale_expanded + base
    pre_mix = mixes[:, :mult].sigmoid().unsqueeze(-1) + eps
    post_mix = (mixes[:, mult : 2 * mult].sigmoid() * post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * mult :].view(-1, mult, mult)
    res_mix = _sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_iters, eps=sinkhorn_eps)
    layer_input = (x * pre_mix).sum(-2).bfloat16()
    return post_mix, res_mix, layer_input


def vanilla_post_mapping(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    """Reference post_mapping implementation in pure PyTorch."""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def vanilla_hc_head(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    eps: float,
) -> torch.Tensor:
    """Reference HCHead forward implementation in pure PyTorch."""
    shape, dtype = x.size(), x.dtype
    x = x.flatten(-2, -1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


# ---------------------------------------------------------------------------
# Profiling helpers (from bench_dg_vs_fma_nsys.py)
# ---------------------------------------------------------------------------


def profile_fn(fn, warmup=BENCH_WARMUP, iters=BENCH_ITERS):
    """Return dict of {kernel_name: avg_us} for all CUDA kernels."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    result = {}
    for evt in prof.key_averages():
        if evt.self_device_time_total > 0:
            result[evt.key] = evt.self_device_time_total / evt.count
    return result


def profile_fn_total(fn, warmup: int = BENCH_WARMUP, iters: int = BENCH_ITERS) -> float:
    """Return average per-iter kernel time (us) via torch.profiler.

    Sums `self_device_time_total` (microseconds) across every CUDA event
    recorded between start() and stop() and divides by `iters`. This
    captures the true per-iter kernel time and excludes host-side gaps
    between launches (e.g. between post_mapping and pre_mapping in the
    unfused path), regardless of how kernel counts differ across paths.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = sum(evt.self_device_time_total for evt in prof.key_averages())
    return total_us / iters


def sum_kernel_times(timings, filters):
    """Sum times for kernel names matching any filter substring."""
    total = 0.0
    for name, us in timings.items():
        if any(f in name for f in filters):
            total += us
    return total


def sum_all_kernel_times(timings):
    """Sum all GPU kernel times."""
    return sum(timings.values())


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------


def generate_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> dict[str, torch.Tensor | float]:
    """Generate test data for big fuse operator."""
    torch.random.manual_seed(42)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = "cuda"

    residual = (
        (torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1

    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def generate_realistic_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> dict[str, torch.Tensor | float]:
    """Generate real-scale mHC data to catch RMS denominator regressions."""
    torch.random.manual_seed(123)

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    device = "cuda"

    residual = torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device).bfloat16()
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device) * 0.05
    ).flatten(1, 2)
    hc_scale = torch.tensor([0.10, 0.10, 0.30], dtype=torch.float, device=device)
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 2.0

    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": rms_eps,
        "hc_pre_eps": hc_pre_eps,
        "hc_sinkhorn_eps": hc_sinkhorn_eps,
        "hc_post_mult_value": hc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
    }


def generate_post_data(
    n: int,
    hidden_size: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual = torch.randn((n, hc_mult, hidden_size), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device)

    return {
        "x": x,
        "residual": residual,
        "post_layer_mix": post_layer_mix,
        "comb_res_mix": comb_res_mix,
    }


def generate_head_data(
    m: int,
    hidden_size: int,
    hc_mult: int,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Generate test data for post operator."""
    torch.random.manual_seed(42)

    x = torch.randn((m, hc_mult, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    hc_fn = torch.randn((hc_mult, hc_mult * hidden_size), dtype=torch.float32, device=device)
    hc_base = torch.randn((hc_mult,), dtype=torch.float32, device=device)
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device)

    return {
        "x": x,
        "hc_fn": hc_fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
    }


# ---------------------------------------------------------------------------
# Correctness + profiling tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 32, 64, 128, 256, 512, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_mapping(n: int, hidden_size: int, hc_mult: int):
    test_data = generate_pre_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None,
        eps=test_data["hc_pre_eps"],
        norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"],
    ).cuda()
    test_module.fn.copy_(test_data["fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    residual = test_data["residual"]

    t = profile_fn(lambda: test_module.pre_mapping(residual))
    total_us = sum_all_kernel_times(t)
    timing_stats[("pre_mapping", n, hidden_size)]["cuda"] = total_us

    post_mix_cuda, comb_mix_cuda, layer_input_cuda = test_module.pre_mapping(residual)
    post_mix_ref, comb_mix_ref, layer_input_ref = vanilla_pre_mapping(
        residual,
        test_data["fn"],
        test_data["hc_scale"],
        test_data["hc_base"],
        hc_mult,
        test_data["rms_eps"],
        test_data["hc_pre_eps"],
        test_data["hc_sinkhorn_eps"],
        test_data["hc_post_mult_value"],
        test_data["sinkhorn_repeat"],
    )
    torch.testing.assert_close(post_mix_ref, post_mix_cuda, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb_mix_ref, comb_mix_cuda, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(layer_input_ref, layer_input_cuda, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("n", [64])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_pre_mapping_pro_hidden_size(n: int, hidden_size: int, hc_mult: int):
    test_data = generate_pre_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=test_data["sinkhorn_repeat"],
        dtype=None,
        eps=test_data["hc_pre_eps"],
        norm_eps=test_data["rms_eps"],
        post_mult_value=test_data["hc_post_mult_value"],
    ).cuda()
    test_module.fn.copy_(test_data["fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    residual = test_data["residual"]

    post_mix_cuda, comb_mix_cuda, layer_input_cuda = test_module.pre_mapping(residual)
    post_mix_ref, comb_mix_ref, layer_input_ref = vanilla_pre_mapping(
        residual,
        test_data["fn"],
        test_data["hc_scale"],
        test_data["hc_base"],
        hc_mult,
        test_data["rms_eps"],
        test_data["hc_pre_eps"],
        test_data["hc_sinkhorn_eps"],
        test_data["hc_post_mult_value"],
        test_data["sinkhorn_repeat"],
    )
    torch.testing.assert_close(post_mix_ref, post_mix_cuda, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(comb_mix_ref, comb_mix_cuda, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(layer_input_ref, layer_input_cuda, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("n", [64, 128, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_post_mapping(n: int, hidden_size: int, hc_mult: int):
    test_data = generate_post_data(
        n=n,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = mHC(mult=hc_mult, hidden_size=hidden_size, sinkhorn_iters=10)

    t = profile_fn(lambda: test_module.post_mapping(**test_data))
    total_us = sum_all_kernel_times(t)
    timing_stats[("post_mapping", n, hidden_size)]["cuda"] = total_us

    output_cuda = test_module.post_mapping(**test_data)
    output_ref = vanilla_post_mapping(**test_data)
    torch.testing.assert_close(output_ref, output_cuda, rtol=1e-2, atol=0.1)


@pytest.mark.parametrize("n", [1, 32, 128, 512, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_hc(n: int, hidden_size: int, hc_mult: int):
    """Correctness test for mHC.fused_hc.

    fused_hc(x_prev, residual_prev, post_mix_prev, comb_mix_prev) must be
    numerically equivalent to:
        residual_cur = post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
        post_mix_cur, comb_mix_cur, layer_input_cur = pre_mapping(residual_cur)

    Uses two distinct mHC modules so that the 'prev' and 'cur' blocks have
    different weights — mirroring the real decoder layer boundary.
    """
    # Generate parameters for the 'current' mHC (consumed by pre_mapping part).
    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    # Generate the incoming (residual_prev, x_prev, post_mix_prev, comb_mix_prev)
    # that the 'previous' block would have emitted.
    torch.random.manual_seed(7)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    # --- fused_hc path ---
    (
        residual_cur_f,
        post_mix_cur_f,
        comb_mix_cur_f,
        layer_input_cur_f,
    ) = cur_module.fused_hc(x_prev, residual_prev, post_mix_prev, comb_mix_prev)

    # --- two-step reference (post_mapping then pre_mapping via the same module) ---
    residual_cur_ref = cur_module.post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
    post_mix_cur_ref, comb_mix_cur_ref, layer_input_cur_ref = cur_module.pre_mapping(
        residual_cur_ref
    )

    # Timing: fused_hc vs separate (post_mapping + pre_mapping).
    # Both paths sum every CUDA event's self_device_time_total via
    # torch.profiler and divide by the iteration count, so host-side gaps
    # between post_mapping and pre_mapping in the unfused path are excluded.
    def _unfused():
        residual_cur = cur_module.post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
        cur_module.pre_mapping(residual_cur)

    fused_us = profile_fn_total(
        lambda: cur_module.fused_hc(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
    )
    unfused_us = profile_fn_total(_unfused)

    timing_stats[("fused_hc", n, hidden_size)]["cuda"] = fused_us
    timing_stats[("fused_hc", n, hidden_size)]["cuda_unfused"] = unfused_us
    speedup = (unfused_us / fused_us) if fused_us > 0 else 0.0
    print(
        f"[fused_hc benchmark] n={n} hidden={hidden_size}  "
        f"fused={fused_us:7.2f}us  unfused={unfused_us:7.2f}us  "
        f"speedup={speedup:.2f}x"
    )

    # fused_hc is a Python-level chain of the same kernels that pre_mapping and
    # post_mapping use (mhc_post_mapping then the bigfuse pre_mapping pipeline).
    # Tolerance matches the baseline post_mapping test (residuals are bf16).
    torch.testing.assert_close(residual_cur_ref, residual_cur_f, rtol=1e-2, atol=0.1)
    torch.testing.assert_close(post_mix_cur_ref, post_mix_cur_f, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(comb_mix_cur_ref, comb_mix_cur_f, rtol=1e-3, atol=5e-3)
    torch.testing.assert_close(layer_input_cur_ref, layer_input_cur_f, rtol=1e-3, atol=5e-3)


# Explicit backend coverage. The autotuner picks one tactic per M-bucket at
# warmup; to actually exercise every backend across CI we force each tactic.
# Tactic format mirrors MhcFusedHcRunner: (backend, tile_n, num_k_splits,
# bigfuse_bs, tile_m).
#
# FMA tactics intentionally sweep both ks>1 (cross-CTA atomicAdd into y_acc /
# r_acc) and tile_m>1 (Path F only; multi-token per CTA, which reshapes how
# the atomic accumulation buckets tokens). Keeping ks=1,tm=1 only would leave
# the cross-CTA atomic path uncovered.
_BACKEND_TACTICS_BY_M = {
    64: [
        ("fused_half_mma", 0, 8, 256, 1),
        ("fused_all_mma", 0, 1, 0, 1),
        ("fused_half_fma", 2, 2, 256, 1),  # FMA cross-CTA atomic (ks=2)
        ("fused_half_fma", 2, 4, 256, 1),  # FMA deeper cross-CTA atomic (ks=4)
        ("fused_all_fma", 2, 1, 0, 1),
        ("fused_all_fma", 2, 2, 0, 1),  # Path F ks=2 atomic
        ("fused_all_fma", 2, 1, 0, 2),  # Path F tile_m=2 (multi-token CTA)
    ],
    256: [
        ("fused_half_mma", 0, 4, 256, 1),
        ("fused_all_mma", 0, 1, 0, 1),
        ("fused_half_fma", 4, 1, 256, 1),
        ("fused_half_fma", 2, 2, 256, 1),  # ks=2 atomic
        ("fused_all_fma", 4, 1, 0, 1),
        ("fused_all_fma", 2, 2, 0, 1),  # Path F ks=2 atomic (tn=2 required for ks>1)
        ("fused_all_fma", 4, 1, 0, 2),  # Path F tile_m=2
    ],
    # fused_half_fma is intentionally omitted at M=2048: the runner guards the
    # FMA 2-kernel path to M <= 512 (it stops scaling past that M).
    2048: [
        ("fused_half_mma", 0, 2, 128, 1),
        ("fused_all_mma", 0, 1, 0, 1),
        ("fused_all_fma", 4, 1, 0, 1),
        ("fused_all_fma", 2, 2, 0, 1),  # Path F ks=2 atomic (tn=2 required for ks>1)
        ("fused_all_fma", 4, 1, 0, 4),  # Path F tile_m=4
    ],
}


def test_mhc_fused_hc_mma_tactic_filter_hidden_sizes():
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
        _FUSED_HC_HALF_MMA_KS,
        _fused_hc_mma_ks_supported,
    )

    supported_by_hidden_size = {
        hidden_size: {
            ks for ks in _FUSED_HC_HALF_MMA_KS if _fused_hc_mma_ks_supported(hidden_size, ks)
        }
        for hidden_size in (4096, 7168, 8192)
    }

    assert supported_by_hidden_size[4096] == {1, 2, 4, 8, 16, 32, 64}
    assert supported_by_hidden_size[7168] == {1, 2, 4, 8, 16}
    assert supported_by_hidden_size[8192] == set()


@pytest.mark.parametrize("n", list(_BACKEND_TACTICS_BY_M.keys()))
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_hc_backends(n: int, hidden_size: int, hc_mult: int):
    """Every wired fused_hc backend sees bit-identical input and is checked
    against one shared torch reference and one golden backend output.

    Paths D (fused_all_mma) and F (fused_all_fma) are single-kernel
    all-in-one variants; fused_half_mma (Path B) and fused_half_fma (Path E)
    are the 2-kernel baselines. Each is forced by calling
    MhcFusedHcRunner.forward directly with an explicit tactic, bypassing the
    autotuner.

    Path C (bigfuse tcgen05) is not covered: its kernel emits (D_next,
    sqr_sum_next, layer_input) as the layer-to-layer state carrier and does
    not produce post_mix_cur / comb_mix_cur, so it cannot be dropped in
    behind the current mhc_fused_hc API without a kernel-side modification
    that adds post_mix_out / comb_mix_out stores.
    """
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcFusedHcRunner

    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(13)
    device = "cuda"
    # Canonical input tensors — generated once, then deep-cloned per consumer
    # so the torch ref and each backend each get an independent byte-identical
    # copy. Protects the test from any hypothetical in-place mutation inside
    # a kernel launcher or a contiguous() call.
    x_prev_ref = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev_ref = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev_ref = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev_ref = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    # Torch ground-truth — computed from clones so the ref path cannot
    # perturb the canonical input tensors either.
    residual_cur_ref = cur_module.post_mapping(
        x_prev_ref.clone(),
        residual_prev_ref.clone(),
        post_mix_prev_ref.clone(),
        comb_mix_prev_ref.clone(),
    )
    post_mix_ref, comb_mix_ref, layer_input_ref = cur_module.pre_mapping(residual_cur_ref.clone())

    runner = MhcFusedHcRunner(
        n=hc_mult,
        hidden_size=hidden_size,
        rms_eps=pre_data["rms_eps"],
        hc_pre_eps=pre_data["hc_pre_eps"],
        hc_sinkhorn_eps=pre_data["hc_sinkhorn_eps"],
        hc_post_mult_value=pre_data["hc_post_mult_value"],
        sinkhorn_repeat=pre_data["sinkhorn_repeat"],
    )

    def make_runner_inputs():
        return [
            x_prev_ref.clone(),
            residual_prev_ref.reshape(n, hc_mult, hidden_size).clone().contiguous(),
            post_mix_prev_ref.reshape(n, hc_mult).clone().contiguous(),
            comb_mix_prev_ref.reshape(n, hc_mult, hc_mult).clone().contiguous(),
            cur_module.fn.detach().clone().contiguous(),
            cur_module.scale.detach().clone(),
            cur_module.base.detach().clone(),
        ]

    tactic_outputs = {}
    for tactic in _BACKEND_TACTICS_BY_M[n]:
        residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = runner(
            inputs=make_runner_inputs(), tactic=tactic
        )
        tactic_outputs[tactic] = (
            residual_cur,
            post_mix_cur.view(n, hc_mult, 1),
            comb_mix_cur.view(n, hc_mult, hc_mult),
            layer_input_cur,
        )

    # Tolerances: bf16 has a 7-bit mantissa so 1 ulp ~ 7.8e-3. For outputs
    # near unit scale with fp32-accumulated reductions, rtol=1e-2 atol=1e-2
    # is the expected bf16 parity — tighter than test_mhc_post_mapping's
    # atol=0.1 which compared against a pure-bf16 vanilla reference.
    bf16_tol = dict(rtol=1e-2, atol=1e-2)
    fp32_tol = dict(rtol=1e-3, atol=5e-3)

    # (1) Every backend must match the torch reference.
    for tactic, (
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
    ) in tactic_outputs.items():
        torch.testing.assert_close(
            residual_cur_ref,
            residual_cur,
            **bf16_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} residual mismatch",
        )
        torch.testing.assert_close(
            post_mix_ref,
            post_mix_cur,
            **fp32_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} post_mix mismatch",
        )
        torch.testing.assert_close(
            comb_mix_ref,
            comb_mix_cur,
            **fp32_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} comb_mix mismatch",
        )
        torch.testing.assert_close(
            layer_input_ref,
            layer_input_cur,
            **bf16_tol,
            msg=f"[vs torch-ref] tactic={tactic} n={n} hidden={hidden_size} layer_input mismatch",
        )

    # (2) All backends must agree with one golden backend at the same tolerance
    # as vs the torch ref. Different backends vary only in tile shape and
    # reduction order, so cross-backend divergence would indicate a kernel
    # correctness bug rather than expected rounding drift.
    gold = next(
        (tactic for tactic in tactic_outputs if tactic[0] == "fused_half_mma"),
        next(iter(tactic_outputs)),
    )
    gr, gpm, gcm, gli = tactic_outputs[gold]
    for tactic, (
        residual_cur,
        post_mix_cur,
        comb_mix_cur,
        layer_input_cur,
    ) in tactic_outputs.items():
        if tactic == gold:
            continue
        torch.testing.assert_close(
            gr,
            residual_cur,
            **bf16_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} residual mismatch",
        )
        torch.testing.assert_close(
            gpm,
            post_mix_cur,
            **fp32_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} post_mix mismatch",
        )
        torch.testing.assert_close(
            gcm,
            comb_mix_cur,
            **fp32_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} comb_mix mismatch",
        )
        torch.testing.assert_close(
            gli,
            layer_input_cur,
            **bf16_tol,
            msg=f"[vs {gold}] tactic={tactic} n={n} hidden={hidden_size} layer_input mismatch",
        )


@pytest.mark.parametrize(
    "tactic",
    [
        ("fused_half_mma", 0, 1, 256, 1),
        ("fused_half_fma", 2, 1, 256, 1),
        ("fused_all_mma", 0, 1, 0, 1),
        ("fused_all_fma", 2, 1, 0, 1),
    ],
)
@pytest.mark.parametrize("hidden_size", [4096, 7168])
def test_mhc_fused_hc_realistic_scale_regression(tactic, hidden_size: int):
    """Real-scale mHC data catches fused_hc RMS normalization regressions."""
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcFusedHcRunner

    n = 16
    hc_mult = 4
    pre_data = generate_realistic_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(17)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.float, device=device).bfloat16()
    residual_prev = torch.randn(
        (n, hc_mult, hidden_size), dtype=torch.float, device=device
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    residual_cur_ref = cur_module.post_mapping(x_prev, residual_prev, post_mix_prev, comb_mix_prev)
    post_mix_ref, comb_mix_ref, layer_input_ref = cur_module.pre_mapping(residual_cur_ref)

    runner = MhcFusedHcRunner(
        n=hc_mult,
        hidden_size=hidden_size,
        rms_eps=pre_data["rms_eps"],
        hc_pre_eps=pre_data["hc_pre_eps"],
        hc_sinkhorn_eps=pre_data["hc_sinkhorn_eps"],
        hc_post_mult_value=pre_data["hc_post_mult_value"],
        sinkhorn_repeat=pre_data["sinkhorn_repeat"],
    )

    residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur = runner(
        inputs=[
            x_prev.contiguous(),
            residual_prev.contiguous(),
            post_mix_prev.view(n, hc_mult).contiguous(),
            comb_mix_prev.contiguous(),
            cur_module.fn.detach().contiguous(),
            cur_module.scale.detach().contiguous(),
            cur_module.base.detach().contiguous(),
        ],
        tactic=tactic,
    )

    torch.testing.assert_close(residual_cur_ref, residual_cur, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(post_mix_ref, post_mix_cur.view(n, hc_mult, 1), rtol=3e-3, atol=5e-3)
    torch.testing.assert_close(
        comb_mix_ref, comb_mix_cur.view(n, hc_mult, hc_mult), rtol=3e-3, atol=5e-3
    )
    torch.testing.assert_close(layer_input_ref, layer_input_cur, rtol=1e-2, atol=2e-2)


@pytest.mark.parametrize("n", [128, 2048])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_mhc_fused_hc_cuda_graph(n: int, hidden_size: int, hc_mult: int):
    """CUDA-graph capture/replay of mHC.fused_hc.

    The decoder uses fused_hc at every non-first layer boundary; the whole
    decoder is expected to be traced into a single CUDA graph. This test
    verifies that (a) fused_hc can be captured without host syncs, and
    (b) replay produces bit-exact results to eager.

    To keep the bit-exact assertion structurally valid, we drive the runner
    with an explicit ``num_k_splits=1`` tactic (Path B, fused_half_mma).
    That disables split-K atomic accumulation entirely, so none of the four
    outputs depend on the non-deterministic FP ordering that pickKSplits(M)
    would otherwise introduce (it picks ks=16 at M=128 and ks=4 at M=2048
    for the autotuner fallback — atomics active, not deterministic across
    replays).
    """
    from tensorrt_llm._torch.modules.mhc.mhc_cuda import MhcFusedHcRunner

    pre_data = generate_pre_data(n=n, hc_mult=hc_mult, hidden_size=hidden_size)

    torch.random.manual_seed(11)
    device = "cuda"
    x_prev = torch.randn((n, hidden_size), dtype=torch.bfloat16, device=device) / hidden_size
    residual_prev = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device) / hidden_size
    ).bfloat16()
    post_mix_prev = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device) * 0.1
    comb_mix_prev = torch.randn((n, hc_mult, hc_mult), dtype=torch.float32, device=device) * 0.1

    cur_module = mHC(
        mult=hc_mult,
        hidden_size=hidden_size,
        sinkhorn_iters=pre_data["sinkhorn_repeat"],
        dtype=None,
        eps=pre_data["hc_pre_eps"],
        norm_eps=pre_data["rms_eps"],
        post_mult_value=pre_data["hc_post_mult_value"],
    ).cuda()
    cur_module.fn.copy_(pre_data["fn"])
    cur_module.scale.copy_(pre_data["hc_scale"])
    cur_module.base.copy_(pre_data["hc_base"])

    runner = MhcFusedHcRunner(
        n=hc_mult,
        hidden_size=hidden_size,
        rms_eps=pre_data["rms_eps"],
        hc_pre_eps=pre_data["hc_pre_eps"],
        hc_sinkhorn_eps=pre_data["hc_sinkhorn_eps"],
        hc_post_mult_value=pre_data["hc_post_mult_value"],
        sinkhorn_repeat=pre_data["sinkhorn_repeat"],
    )
    # Pin tactic to Path B with num_k_splits=1 → no atomic accumulation on
    # any output. Tactic tuple matches MhcFusedHcRunner.get_tactics().
    tactic = ("fused_half_mma", 0, 1, 128, 1)
    assert tactic[2] == 1, "bit-exact assertion requires num_k_splits=1"

    def _inputs():
        return [
            x_prev,
            residual_prev.reshape(n, hc_mult, hidden_size).contiguous(),
            post_mix_prev.reshape(n, hc_mult).contiguous(),
            comb_mix_prev.reshape(n, hc_mult, hc_mult).contiguous(),
            cur_module.fn,
            cur_module.scale,
            cur_module.base,
        ]

    # Eager reference — runner's workspace cache reuses output tensors across
    # calls with matching shape, so eager_out and graph_out alias the same
    # storage. Clone eager_out so we can compare after the graph replay
    # overwrites the workspace.
    eager_raw = runner(inputs=_inputs(), tactic=tactic)
    eager_out = tuple(t.clone() for t in eager_raw)

    # Warm up on a side stream — required for CUDA graph capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            runner(inputs=_inputs(), tactic=tactic)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Capture.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_out = runner(inputs=_inputs(), tactic=tactic)

    # Replay — outputs should update in place.
    g.replay()
    torch.cuda.synchronize()

    # With ks=1 the kernel has no atomic accumulation anywhere, so replay must
    # be bit-exact against eager on all four outputs.
    for ge, ee, name in zip(
        graph_out, eager_out, ["residual", "post_mix", "comb_mix", "layer_input"]
    ):
        torch.testing.assert_close(
            ge, ee, rtol=0, atol=0, msg=f"fused_hc CUDA-graph mismatch in {name}"
        )

    # Mutate inputs in-place and replay; result should follow — proves the graph
    # is parameterised by input storage, not cached constants.
    x_prev.mul_(1.001)
    residual_prev.mul_(1.001)
    post_mix_prev.mul_(1.001)
    comb_mix_prev.mul_(1.001)
    eager_raw2 = runner(inputs=_inputs(), tactic=tactic)
    eager_out2 = tuple(t.clone() for t in eager_raw2)
    g.replay()
    torch.cuda.synchronize()
    for ge, ee, name in zip(
        graph_out, eager_out2, ["residual", "post_mix", "comb_mix", "layer_input"]
    ):
        torch.testing.assert_close(
            ge, ee, rtol=0, atol=0, msg=f"fused_hc CUDA-graph replay mismatch in {name}"
        )


@pytest.mark.parametrize("m", [64, 128, 4096, 8192])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("hc_mult", [4])
def test_hc_head(m: int, hidden_size: int, hc_mult: int):
    test_data = generate_head_data(
        m=m,
        hc_mult=hc_mult,
        hidden_size=hidden_size,
    )

    test_module = HCHead(mult=hc_mult, hidden_size=hidden_size).cuda()
    test_module.fn.copy_(test_data["hc_fn"])
    test_module.scale.copy_(test_data["hc_scale"])
    test_module.base.copy_(test_data["hc_base"])

    t = profile_fn(lambda: test_module(test_data["x"]))
    total_us = sum_all_kernel_times(t)
    timing_stats[("hc_head", m, hidden_size)]["cuda"] = total_us

    output_cuda = test_module(test_data["x"])
    output_ref = vanilla_hc_head(
        test_data["x"],
        test_data["hc_fn"],
        test_data["hc_scale"],
        test_data["hc_base"],
        norm_eps=1e-6,
        eps=1e-6,
    )
    torch.testing.assert_close(output_ref, output_cuda, rtol=1e-2, atol=0.1)


# ---------------------------------------------------------------------------
# Low-level pre_mapping pipeline benchmark: DG / DG-s16 / FMA
# ---------------------------------------------------------------------------

HC_MULT = 4
HIDDEN_SIZE = 4096
_N = HC_MULT * (HC_MULT + 1 + 1)  # 24
_K = HC_MULT * HIDDEN_SIZE  # 16384
_NUM_SPLITS = 16
_SINKHORN_REPEAT = 20


def _try_import_backends():
    """Return (tf32_hc_prenorm_gemm|None, mhc_gemm_rms_fma_cuda|None,
    mhc_big_fuse_cuda|None)."""
    tf32_hc_prenorm_gemm = None
    try:
        from deep_gemm import tf32_hc_prenorm_gemm
    except ImportError:
        try:
            from tensorrt_llm.deep_gemm import tf32_hc_prenorm_gemm
        except ImportError:
            pass

    mhc_gemm_rms_fma_cuda = mhc_big_fuse_cuda = None
    try:
        from tensorrt_llm._torch.modules.mhc.mhc_cuda import (
            mhc_big_fuse_cuda,
            mhc_gemm_rms_fma_cuda,
        )
    except Exception:
        pass

    return (
        tf32_hc_prenorm_gemm,
        mhc_gemm_rms_fma_cuda,
        mhc_big_fuse_cuda,
    )


def run_bench_pre_mapping(M: int) -> dict:
    """Low-level kernel benchmark for one M: profiles GEMM + BigFuse per backend.
    Returns dict like {"DG": (gemm_us, fuse_us), "FMA": (...), ...}.
    """
    device = "cuda"
    (
        tf32_hc_prenorm_gemm,
        mhc_gemm_rms_fma_cuda,
        mhc_big_fuse_cuda,
    ) = _try_import_backends()

    w_nk = torch.randn(_N, _K, dtype=torch.float32, device=device) * 0.01
    hc_scale = torch.randn(3, dtype=torch.float32, device=device)
    hc_base = torch.randn(_N, dtype=torch.float32, device=device)
    x = (torch.randn(M, _K, dtype=torch.float32, device=device) * 0.01).bfloat16()
    residual = (
        torch.randn(M, HC_MULT, HIDDEN_SIZE, dtype=torch.float32, device=device) / HIDDEN_SIZE
    ).bfloat16()

    times = {}

    if tf32_hc_prenorm_gemm is not None and mhc_big_fuse_cuda is not None:
        y = torch.empty(M, _N, dtype=torch.float32, device=device)
        r = torch.empty(M, dtype=torch.float32, device=device)
        pm = torch.empty(M, HC_MULT, dtype=torch.float32, device=device)
        cm = torch.empty(M, HC_MULT * HC_MULT, dtype=torch.float32, device=device)
        li = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        def dg_fn():
            tf32_hc_prenorm_gemm(x, w_nk, y, r)
            mhc_big_fuse_cuda(
                y,
                r,
                residual,
                hc_scale,
                hc_base,
                pm,
                cm,
                li,
                M,
                _K,
                HIDDEN_SIZE,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                _SINKHORN_REPEAT,
                num_splits=1,
            )

        t = profile_fn(dg_fn)
        times["DG"] = (sum_kernel_times(t, ["hc_prenorm_gemm"]), sum_kernel_times(t, ["BigFuse"]))

        y_s = torch.empty(_NUM_SPLITS, M, _N, dtype=torch.float32, device=device)
        r_s = torch.empty(_NUM_SPLITS, M, dtype=torch.float32, device=device)
        pm_s = torch.empty(M, HC_MULT, dtype=torch.float32, device=device)
        cm_s = torch.empty(M, HC_MULT * HC_MULT, dtype=torch.float32, device=device)
        li_s = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        def dg_s16_fn():
            tf32_hc_prenorm_gemm(x, w_nk, y_s, r_s, num_splits=_NUM_SPLITS)
            mhc_big_fuse_cuda(
                y_s,
                r_s,
                residual,
                hc_scale,
                hc_base,
                pm_s,
                cm_s,
                li_s,
                M,
                _K,
                HIDDEN_SIZE,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                _SINKHORN_REPEAT,
                num_splits=_NUM_SPLITS,
            )

        t = profile_fn(dg_s16_fn)
        times["DG-s16"] = (
            sum_kernel_times(t, ["hc_prenorm_gemm"]),
            sum_kernel_times(t, ["BigFuse"]),
        )

    if mhc_gemm_rms_fma_cuda is not None and mhc_big_fuse_cuda is not None:
        pm_f = torch.empty(M, HC_MULT, dtype=torch.float32, device=device)
        cm_f = torch.empty(M, HC_MULT * HC_MULT, dtype=torch.float32, device=device)
        li_f = torch.empty(M, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)

        def fma_fn():
            y_f, r_f = mhc_gemm_rms_fma_cuda(x, None, M, _N, _K, w_t=w_nk)
            mhc_big_fuse_cuda(
                y_f,
                r_f,
                residual,
                hc_scale,
                hc_base,
                pm_f,
                cm_f,
                li_f,
                M,
                _K,
                HIDDEN_SIZE,
                1e-6,
                1e-6,
                1e-6,
                1.0,
                _SINKHORN_REPEAT,
                num_splits=1,
            )

        t = profile_fn(fma_fn)
        times["FMA"] = (sum_kernel_times(t, ["GemmSqrsumFma"]), sum_kernel_times(t, ["BigFuse"]))

    return times


def _print_bench_timing_table(bench_entries: dict):
    """Print the pre_mapping pipeline (GEMM + BigFuse) benchmark table."""
    if not bench_entries:
        return
    all_cols = []
    for v in bench_entries.values():
        for c in v:
            if c not in all_cols:
                all_cols.append(c)
    print("\nPRE_MAPPING PIPELINE (GEMM + BigFuse)")
    header = f"  {'M':>6s}"
    for c in all_cols:
        header += f"  {c:>16s}"
    header += f"  {'best':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in sorted(bench_entries):
        _, M, _ = key
        times = bench_entries[key]
        totals = {c: times[c][0] + times[c][1] for c in times}
        best = min(totals, key=totals.get) if totals else "N/A"
        row = f"  {M:6d}"
        for c in all_cols:
            if c in times:
                g, f = times[c]
                row += f"  {g + f:8.1f}({g:4.1f}+{f:4.1f})"
            else:
                row += f"  {'N/A':>16s}"
        row += f"  {best:>8s}"
        print(row)


# ---------------------------------------------------------------------------
# Session-scoped fixture: print timing table at end (pytest only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def print_timing_stats():
    """Print collected GPU profiler timings at end of session."""
    yield

    if not timing_stats:
        return

    print("\n" + "=" * 90)
    print("GPU Kernel Timing (torch.profiler, microseconds)")
    print("=" * 90)

    # --- Per-backend correctness/perf tests (pre_mapping, post_mapping, hc_head) ---
    for test_type in ("pre_mapping", "post_mapping", "fused_hc", "hc_head"):
        entries = {
            k: v for k, v in timing_stats.items() if isinstance(k, tuple) and k[0] == test_type
        }
        if not entries:
            continue

        dim_label = "m" if test_type == "hc_head" else "n"
        print(f"\n{test_type.upper()}")

        all_backends = sorted({b for d in entries.values() for b in d})
        header = f"  {dim_label:>6s}  hidden"
        for b in all_backends:
            header += f"  {b:>10s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for key in sorted(entries):
            _, dim_val, hidden = key
            row = f"  {dim_val:6d}  {hidden:6d}"
            for b in all_backends:
                us = entries[key].get(b)
                row += f"  {us:10.1f}" if us is not None else f"  {'N/A':>10s}"
            print(row)

    # --- Low-level pipeline bench table (only populated when run via main()) ---
    bench_entries = {
        k: v for k, v in timing_stats.items() if isinstance(k, tuple) and k[0] == "bench_pre"
    }
    _print_bench_timing_table(bench_entries)

    print("\n" + "=" * 90)


def main():
    """Run pre_mapping pipeline benchmark (GEMM + BigFuse) for various M.
    Invoked when running: python test_mhc.py
    """
    torch.manual_seed(42)
    bench_M = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    bench_stats = {}
    for M in bench_M:
        bench_stats[("bench_pre", M, HIDDEN_SIZE)] = run_bench_pre_mapping(M)

    print("\n" + "=" * 90)
    print("GPU Kernel Timing (torch.profiler, microseconds) — benchmark only")
    print("=" * 90)
    _print_bench_timing_table(bench_stats)
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
