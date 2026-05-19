# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Micro-benchmark: fused cross-head QK Norm + RoPE vs split (separate norm + RoPE).

Three baselines are compared:
    1. Eager:   PyTorch eager-mode split norm + RoPE (many small CUDA kernels
                launched via PyTorch dispatch).
    2. Compile: `torch.compile`'d version of (1), where TorchInductor fuses the
                elementwise + norm ops into a few large Triton kernels.
    3. Fused:   Our single hand-written CUDA kernel that does cross-head RMSNorm
                + RoPE in a single pass.

Usage:
    python benchmarks/bench_fused_dit_cross_head_qk_norm_rope.py
"""

import torch
import torch._dynamo
import torch.nn.functional as F

import tensorrt_llm  # noqa: F401  # required to register torch.ops.trtllm.*

# We benchmark many shape/mode combinations with a single torch.compile()'d
# function; each unique input shape + static arg combo forces Dynamo to
# recompile. Raise the cache limit so we don't silently fall back to eager.
torch._dynamo.config.cache_size_limit = 64


def _apply_interleaved_rope(x, cos, sin):
    x_rot = torch.empty_like(x, dtype=torch.float32)
    x_rot[:, 0::2] = -x[:, 1::2].float()
    x_rot[:, 1::2] = x[:, 0::2].float()
    return (x.float() * cos + x_rot * sin).to(x.dtype)


def _apply_rotate_half_rope(x, cos, sin, head_dim, num_heads):
    """rotate_half RoPE: pair (x[i], x[i + head_dim/2]) within each head."""
    T = x.shape[0]
    x_4d = x.view(T, num_heads, head_dim).float()
    half = head_dim // 2
    x1 = x_4d[..., :half]
    x2 = x_4d[..., half:]
    cos_4d = cos.view(T, num_heads, head_dim)
    sin_4d = sin.view(T, num_heads, head_dim)
    out = torch.empty_like(x_4d)
    out[..., :half] = x1 * cos_4d[..., :half] - x2 * sin_4d[..., :half]
    out[..., half:] = x2 * cos_4d[..., half:] + x1 * sin_4d[..., half:]
    return out.reshape(T, -1).to(x.dtype)


def split_norm_rope(
    qkv, num_heads, head_dim, eps, q_weight, k_weight, cos_emb, sin_emb, interleave
):
    """Baseline: separate RMSNorm + RoPE (current WAN path)."""
    q_size = num_heads * head_dim
    q = qkv[:, :q_size]
    k = qkv[:, q_size : 2 * q_size]

    q = F.rms_norm(q.float(), (q_size,), q_weight.float(), eps).to(qkv.dtype)
    k = F.rms_norm(k.float(), (q_size,), k_weight.float(), eps).to(qkv.dtype)

    num_tokens = qkv.shape[0]
    cos_q = cos_emb.unsqueeze(1).expand(-1, num_heads, -1).reshape(num_tokens, q_size)
    sin_q = sin_emb.unsqueeze(1).expand(-1, num_heads, -1).reshape(num_tokens, q_size)

    if interleave:
        q = _apply_interleaved_rope(q, cos_q, sin_q)
        k = _apply_interleaved_rope(k, cos_q, sin_q)
    else:
        q = _apply_rotate_half_rope(q, cos_q, sin_q, head_dim, num_heads)
        k = _apply_rotate_half_rope(k, cos_q, sin_q, head_dim, num_heads)

    qkv[:, :q_size] = q
    qkv[:, q_size : 2 * q_size] = k


def fused_norm_rope(
    qkv, num_heads, head_dim, eps, q_weight, k_weight, cos_emb, sin_emb, interleave
):
    """Fused cross-head QK Norm + RoPE kernel."""
    torch.ops.trtllm.fused_dit_cross_head_qk_norm_rope(
        qkv,
        num_heads,
        num_heads,
        num_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_emb,
        sin_emb,
        interleave,
    )


def benchmark(fn, warmup=50, iters=200):
    """Time GPU-side kernel work only using CUDA events.

    fn() should perform the kernel call(s) on pre-allocated buffers; caller is
    responsible for ensuring any in-place mutation does not affect timing
    semantics (we do iters back-to-back calls on the same buffer).
    """
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


def main():
    device = "cuda"
    configs = [
        {"name": "WAN_1.3B_256tok", "num_heads": 12, "head_dim": 128, "num_tokens": 256},
        {"name": "WAN_1.3B_4096tok", "num_heads": 12, "head_dim": 128, "num_tokens": 4096},
        {"name": "WAN_14B_256tok", "num_heads": 40, "head_dim": 128, "num_tokens": 256},
        {"name": "WAN_14B_4096tok", "num_heads": 40, "head_dim": 128, "num_tokens": 4096},
        {"name": "WAN_14B_16384tok", "num_heads": 40, "head_dim": 128, "num_tokens": 16384},
    ]

    # TorchInductor Triton-fused version of the eager split path.
    # Separate torch.compile() call per shape avoids graph-break costs being
    # counted in-between shapes (Inductor caches compiled kernels by shape).
    split_compiled = torch.compile(split_norm_rope, mode="reduce-overhead", dynamic=False)

    # interleave=True  -> WAN-1 family (interleaved RoPE)
    # interleave=False -> WAN-2 family (rotate_half RoPE)
    for mode_name, interleave in [("interleave (WAN-1)", True), ("rotate_half (WAN-2)", False)]:
        print(f"\n=== RoPE mode: {mode_name} ===")
        header = (
            f"{'Config':<22} {'Eager (ms)':>12} {'Compile (ms)':>14} {'Fused (ms)':>12} "
            f"{'Fused/Eager':>12} {'Fused/Compile':>14}"
        )
        print(header)
        print("-" * len(header))

        for cfg in configs:
            num_heads = cfg["num_heads"]
            head_dim = cfg["head_dim"]
            num_tokens = cfg["num_tokens"]
            q_dim = num_heads * head_dim
            hidden_size = 3 * q_dim

            torch.random.manual_seed(42)
            qkv_seed = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
            q_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device)
            k_weight = torch.randn(q_dim, dtype=torch.bfloat16, device=device)

            half_dim = head_dim // 2
            freqs = torch.randn(num_tokens, half_dim, device=device, dtype=torch.float32)
            freqs = freqs.repeat_interleave(2, dim=-1)
            cos_emb = freqs.cos()
            sin_emb = freqs.sin()

            # Pre-allocate one working buffer per path; benchmark() runs the
            # kernel back-to-back on the same tensor (timing is independent of
            # values).
            qkv_eager_buf = qkv_seed.clone()
            qkv_compile_buf = qkv_seed.clone()
            qkv_fused_buf = qkv_seed.clone()

            eager_ms = benchmark(
                lambda: split_norm_rope(
                    qkv_eager_buf,
                    num_heads,
                    head_dim,
                    1e-6,
                    q_weight,
                    k_weight,
                    cos_emb,
                    sin_emb,
                    interleave,
                )
            )
            compile_ms = benchmark(
                lambda: split_compiled(
                    qkv_compile_buf,
                    num_heads,
                    head_dim,
                    1e-6,
                    q_weight,
                    k_weight,
                    cos_emb,
                    sin_emb,
                    interleave,
                ),
                warmup=100,  # first few calls trigger Inductor compilation
            )
            fused_ms = benchmark(
                lambda: fused_norm_rope(
                    qkv_fused_buf,
                    num_heads,
                    head_dim,
                    1e-6,
                    q_weight,
                    k_weight,
                    cos_emb,
                    sin_emb,
                    interleave,
                )
            )

            speedup_eager = eager_ms / fused_ms
            speedup_compile = compile_ms / fused_ms
            print(
                f"{cfg['name']:<22} {eager_ms:>12.4f} {compile_ms:>14.4f} {fused_ms:>12.4f} "
                f"{speedup_eager:>11.2f}x {speedup_compile:>13.2f}x"
            )


if __name__ == "__main__":
    main()
