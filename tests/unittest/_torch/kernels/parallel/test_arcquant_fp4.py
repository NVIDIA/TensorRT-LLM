import pytest
import torch
import torch.nn.functional as F
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401


def _benchmark(fn, warmup=10, iters=100):
    """Benchmark a callable using CUDA events. Returns elapsed time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _effective_bytes(M, K, KE, input_dtype=torch.bfloat16):
    """Compute effective bytes moved (read + write) for a quantization kernel.

    Read:  M * K * sizeof(input_dtype)           -- input tensor
    Write: M * (K + KE) / 2                      -- packed e2m1 output
           + M * (K + KE) / 16                   -- scale factors (one fp8 per 16 elements)
    """
    elem_bytes = 2 if input_dtype == torch.bfloat16 else 1
    read_bytes = M * K * elem_bytes
    out_K = K + KE
    write_bytes = M * out_K // 2 + M * out_K // 16
    return read_bytes + write_bytes


def _gb_per_sec(total_bytes, time_ms):
    """Convert total bytes and time in ms to GB/s."""
    return total_bytes / time_ms / 1e6 if time_ms > 0 else float("inf")


@skip_pre_blackwell
@pytest.mark.parametrize(
    "mnk",
    [
        (39, 6144, 4096),
        (46, 4096, 4096),
        (155, 1024, 4096),
        (232, 12288, 4096),
        (1357, 4096, 12288),
    ],
)
@pytest.mark.parametrize("input_type", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_arcquant_fp4(mnk, input_type):
    M, N, K = mnk
    step = 256
    for i in range(4096 // step + 1):
        KE = step * i
        torch.manual_seed(45510)
        X = torch.rand(M, K, dtype=torch.bfloat16, device="cuda") - 0.5
        W = torch.rand(N, K, dtype=torch.bfloat16, device="cuda") - 0.5
        # reorder_index = torch.arange(K, dtype=torch.int16, device="cuda")
        reorder_index = torch.randperm(K, dtype=torch.int16, device="cuda")

        scale_x = (448.0 * 6.0) / torch.max(X.abs()).float()
        scale_w = (448.0 * 6.0) / torch.max(W.abs()).float()
        if input_type == torch.float8_e4m3fn:
            FP8_Scale = 448.0 / torch.max(X.abs()).float()
            X_fp8 = (X * FP8_Scale).to(torch.float8_e4m3fn)

            A, SFA = torch.ops.trtllm.fp4_quantize_with_reorder_residual(
                X_fp8, scale_x, reorder_index, KE, is_act=True
            )
        else:
            A, SFA = torch.ops.trtllm.fp4_quantize_with_reorder_residual(
                X, scale_x, reorder_index, KE, is_act=True
            )
        B, SFB = torch.ops.trtllm.fp4_quantize_with_reorder_residual(
            W, scale_w, reorder_index, KE, is_act=False
        )

        C = torch.ops.trtllm.nvfp4_gemm(
            A, B, SFA, SFB, 1.0 / (scale_x * scale_w).float(), torch.bfloat16
        )
        D = F.linear(X, W)
        assert F.cosine_similarity(C.flatten(), D.flatten(), dim=0).item() > 0.98


@skip_pre_blackwell
@pytest.mark.parametrize(
    "mnk",
    [
        (39, 6144, 4096),
        (46, 4096, 4096),
        (155, 1024, 4096),
        (232, 12288, 4096),
        (1357, 4096, 12288),
    ],
)
@pytest.mark.parametrize("input_type", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_arcquant_fp4_with_residual(mnk, input_type):
    """Test nvfp4_quantize_residual_with_block_size kernel (no reorder, block-size loop pattern).

    Verifies end-to-end GEMM quality (cosine similarity > 0.98).
    """
    M, N, K = mnk
    step = 256
    for i in range(4096 // step + 1):
        KE = step * i
        torch.manual_seed(45510)
        X = torch.rand(M, K, dtype=torch.bfloat16, device="cuda") - 0.5
        W = torch.rand(N, K, dtype=torch.bfloat16, device="cuda") - 0.5

        scale_x = (448.0 * 6.0) / torch.max(X.abs()).float()
        scale_w = (448.0 * 6.0) / torch.max(W.abs()).float()

        if input_type == torch.float8_e4m3fn:
            FP8_Scale = 448.0 / torch.max(X.abs()).float()
            X_fp8 = (X * FP8_Scale).to(torch.float8_e4m3fn)

            # New kernel (no reorder)
            A, SFA = torch.ops.trtllm.fp4_quantize_with_residual(X_fp8, scale_x, KE, is_act=True)
        else:
            # New kernel (no reorder)
            A, SFA = torch.ops.trtllm.fp4_quantize_with_residual(X, scale_x, KE, is_act=True)

        # Weight path (bf16 only).
        B, SFB = torch.ops.trtllm.fp4_quantize_with_residual(W, scale_w, KE, is_act=False)

        # End-to-end GEMM quality check.
        C = torch.ops.trtllm.nvfp4_gemm(
            A, B, SFA, SFB, 1.0 / (scale_x * scale_w).float(), torch.bfloat16
        )
        D = F.linear(X, W)
        assert F.cosine_similarity(C.flatten(), D.flatten(), dim=0).item() > 0.98


@pytest.mark.skip(
    reason="Manual perf benchmark — run explicitly with pytest -s -k perf_no_residual"
)
@skip_pre_blackwell
@pytest.mark.parametrize(
    "mk",
    [
        (128, 4096),
        (512, 4096),
        (1024, 7168),
        (2048, 12288),
        (4096, 4096),
    ],
)
def test_arcquant_fp4_perf_no_residual(mk):
    """Perf comparison at KE=0: old arcquant vs new arcquant vs fp4_quantize.

    KE=0 means no residual, so all three kernels do equivalent work.
    Run manually: pytest -s test_arcquant_fp4.py::test_arcquant_fp4_perf_no_residual --no-header -rN
    """
    M, K = mk

    torch.manual_seed(45510)
    X = torch.rand(M, K, dtype=torch.bfloat16, device="cuda") - 0.5
    identity_index = torch.arange(K, dtype=torch.int16, device="cuda")
    scale_x = (448.0 * 6.0) / torch.max(X.abs()).float()

    t_old = _benchmark(
        lambda: torch.ops.trtllm.fp4_quantize_with_reorder_residual(
            X, scale_x, identity_index, 0, is_act=True
        )
    )

    t_new = _benchmark(
        lambda: torch.ops.trtllm.fp4_quantize_with_residual(X, scale_x, 0, is_act=True)
    )

    t_std = _benchmark(lambda: torch.ops.trtllm.fp4_quantize(X, scale_x, 16, False, True))

    eff_bytes = _effective_bytes(M, K, KE=0)

    print(
        f"\n  M={M:5d} K={K:5d} KE=0    | "
        f"old_arcquant: {t_old:.4f} ms ({_gb_per_sec(eff_bytes, t_old):7.1f} GB/s) | "
        f"new_arcquant: {t_new:.4f} ms ({_gb_per_sec(eff_bytes, t_new):7.1f} GB/s) | "
        f"fp4_quantize: {t_std:.4f} ms ({_gb_per_sec(eff_bytes, t_std):7.1f} GB/s) | "
        f"new/std: {t_new / t_std:.2f}x"
    )


@pytest.mark.skip(
    reason="Manual perf benchmark — run explicitly with pytest -s -k perf_with_residual"
)
@skip_pre_blackwell
@pytest.mark.parametrize(
    "mk",
    [
        (128, 4096),
        (512, 4096),
        (1024, 7168),
        (2048, 12288),
        (4096, 4096),
    ],
)
@pytest.mark.parametrize("KE", [512, 1024])
def test_arcquant_fp4_perf_with_residual(mk, KE):
    """Perf comparison at KE>0: old arcquant vs new arcquant (both with residual).

    fp4_quantize is excluded since it does not support residual quantization.
    Run manually: pytest -s test_arcquant_fp4.py::test_arcquant_fp4_perf_with_residual --no-header -rN
    """
    M, K = mk
    if KE > K:
        pytest.skip(f"KE={KE} > K={K}")

    torch.manual_seed(45510)
    X = torch.rand(M, K, dtype=torch.bfloat16, device="cuda") - 0.5
    identity_index = torch.arange(K, dtype=torch.int16, device="cuda")
    scale_x = (448.0 * 6.0) / torch.max(X.abs()).float()

    t_old = _benchmark(
        lambda: torch.ops.trtllm.fp4_quantize_with_reorder_residual(
            X, scale_x, identity_index, KE, is_act=True
        )
    )

    t_new = _benchmark(
        lambda: torch.ops.trtllm.fp4_quantize_with_residual(X, scale_x, KE, is_act=True)
    )

    speedup = t_old / t_new if t_new > 0 else float("inf")
    eff_bytes = _effective_bytes(M, K, KE)

    print(
        f"\n  M={M:5d} K={K:5d} KE={KE:4d} | "
        f"with reorder: {t_old:.4f} ms ({_gb_per_sec(eff_bytes, t_old):7.1f} GB/s) | "
        f"without reorder: {t_new:.4f} ms ({_gb_per_sec(eff_bytes, t_new):7.1f} GB/s) | "
        f"speedup(old/new): {speedup:.2f}x"
    )
