# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Byte-exact check of the MoE-entry add-RMSNorm with NVFP4 output (TRTLLM_MOE_ENTRY_NORM_NVFP4).

``torch.ops.trtllm.flashinfer_fused_add_rmsnorm_nvfp4`` (one kernel) against today's two kernels,
``torch.ops.trtllm.flashinfer_fused_add_rmsnorm`` (FlashInfer's CuTe DSL fused add-RMSNorm, in place) followed by
``fp4_quantize(x, scale, 16, False, False)`` (the C++ op and the Triton quantizer): the normalized BF16 output,
the residual, the packed E2M1 bytes and the linear scale factors must all match, eagerly and under CUDA-graph
replay. Also reports per-call timing of both (CUDA graph of --reps calls).

    python check_dsr1_moe_entry_norm_nvfp4.py --tokens 1 7 128 1536 4096
"""

import argparse

import torch


def main():
    import flashinfer.norm as flashinfer_norm

    import tensorrt_llm  # noqa: F401  (loads the trtllm ops)
    from tensorrt_llm._torch.custom_ops.flashinfer_custom_ops import (
        flashinfer_fused_add_rmsnorm,
        flashinfer_fused_add_rmsnorm_nvfp4,
    )
    from tensorrt_llm._torch.cute_dsl_kernels.flashinfer_fused_add_rmsnorm_nvfp4 import (
        fused_add_rmsnorm_nvfp4_supported,
    )
    from tensorrt_llm._torch.modules.fused_ops.nvfp4_quantize_triton import nvfp4_quantize_triton

    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 7, 128, 1536, 4096])
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--reps", type=int, default=20)
    args = p.parse_args()
    device = torch.device("cuda")
    use_cuda_norm = flashinfer_norm._use_cuda_norm(device)
    print(
        f"MOE_ENTRY flashinfer CUDA-JIT norm fallback: {use_cuda_norm} (False = CuTe DSL reference)",
        flush=True,
    )
    assert not use_cuda_norm, "the reference must be FlashInfer's CuTe DSL norm"
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    eps = 1e-6
    h = args.hidden
    weight = (1.0 + 0.3 * torch.randn(h, generator=gen, device=device)).to(torch.bfloat16)
    total = 0
    for spiky in (False, True):
        for m in args.tokens:
            x = (torch.randn(m, h, generator=gen, device=device) * 0.5).to(torch.bfloat16)
            r = (torch.randn(m, h, generator=gen, device=device) * 8.0).to(torch.bfloat16)
            if spiky:
                x[:, :32] *= 300.0  # massive-activation channels
                r[:, 100:116] = 0.0
                if m > 1:
                    x[0] = 0.0
                    r[0] = 0.0  # an all-zero row: rstd = rsqrt(eps), zero scale factors
            scale = torch.tensor(
                448.0 * 6.0 / float(torch.empty(1).uniform_(5.0, 50.0)), device=device
            )
            assert fused_add_rmsnorm_nvfp4_supported(x, r, weight, scale)
            x_ref, r_ref = x.clone(), r.clone()
            flashinfer_fused_add_rmsnorm(x_ref, r_ref, weight, eps)
            q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x_ref, scale, 16, False, False)
            q_tri, sf_tri = nvfp4_quantize_triton(x_ref, scale, False)
            x_new, r_new = x.clone(), r.clone()
            q, sf = flashinfer_fused_add_rmsnorm_nvfp4(x_new, r_new, weight, scale, eps)
            torch.cuda.synchronize()
            bad = {
                "norm_out": int((x_new.view(torch.int16) != x_ref.view(torch.int16)).sum()),
                "residual": int((r_new.view(torch.int16) != r_ref.view(torch.int16)).sum()),
                "fp4": int((q.view(torch.uint8) != q_ref.view(torch.uint8)).sum()),
                "sf": int(
                    (sf.view(torch.uint8).reshape(-1) != sf_ref.view(torch.uint8).reshape(-1)).sum()
                ),
            }
            tri = int((q_tri.view(torch.uint8) != q_ref.view(torch.uint8)).sum()) + int(
                (sf_tri.view(torch.uint8).reshape(-1) != sf_ref.view(torch.uint8).reshape(-1)).sum()
            )
            # CUDA graph replay of the fused op on fresh copies of the inputs
            xg, rg = x.clone(), r.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            g = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(g, stream=stream):
                    qg, sfg = flashinfer_fused_add_rmsnorm_nvfp4(xg, rg, weight, scale, eps)
            torch.cuda.current_stream().wait_stream(stream)
            xg.copy_(x)
            rg.copy_(r)
            g.replay()
            torch.cuda.synchronize()
            gbad = (
                int((xg.view(torch.int16) != x_ref.view(torch.int16)).sum())
                + int((rg.view(torch.int16) != r_ref.view(torch.int16)).sum())
                + int((qg.view(torch.uint8) != q_ref.view(torch.uint8)).sum())
                + int(
                    (
                        sfg.view(torch.uint8).reshape(-1) != sf_ref.view(torch.uint8).reshape(-1)
                    ).sum()
                )
            )

            # timing: today's two kernels vs the fused one (graphs of --reps calls on scratch copies)
            def timed(fn):
                s2 = torch.cuda.Stream()
                s2.wait_stream(torch.cuda.current_stream())
                g2 = torch.cuda.CUDAGraph()
                with torch.cuda.stream(s2):
                    with torch.cuda.graph(g2, stream=s2):
                        for _ in range(args.reps):
                            fn()
                torch.cuda.current_stream().wait_stream(s2)
                for _ in range(3):
                    g2.replay()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(20):
                    g2.replay()
                end.record()
                end.synchronize()
                return start.elapsed_time(end) * 1000.0 / (20 * args.reps)

            xs, rs = x.clone(), r.clone()

            def today():
                flashinfer_fused_add_rmsnorm(xs, rs, weight, eps)
                nvfp4_quantize_triton(xs, scale, False)

            def fused():
                flashinfer_fused_add_rmsnorm_nvfp4(xs, rs, weight, scale, eps)

            def norm_only():
                flashinfer_fused_add_rmsnorm(xs, rs, weight, eps)

            t_today, t_fused, t_norm = timed(today), timed(fused), timed(norm_only)
            print(
                f"MOE_ENTRY M={m} spiky={spiky} mismatches={bad} triton_vs_cpp={tri} graph={gbad} "
                f"us/call: norm+quant(triton)={t_today:.1f} norm_only={t_norm:.1f} fused={t_fused:.1f}",
                flush=True,
            )
            total += sum(bad.values()) + gbad
    print(f"MOE_ENTRY_DONE ok={total == 0} mismatches={total}", flush=True)


if __name__ == "__main__":
    main()
