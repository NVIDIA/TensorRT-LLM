# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Hash every FP4 MLA cache pool after context + generation writes (A/B regression aid).

Prints one SHA-256 per pool (K pages, K scales, HP rows, V scales, V-pack) after
the production context writer and N one-token generation steps, so two builds
can be compared for bit-identical writer output::

    python check_dsr1_fp4_mla_writer_hash.py --backend cutedsl --steps 40
"""

import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_dsr1_mla_decode as bench  # noqa: E402


def _digest(t):
    if t is None:
        return "none"
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4

    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="cutedsl")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[1, 17, 33, 100, 129, 300, 1000])
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--heads", type=int, default=128)
    args = p.parse_args()
    torch.manual_seed(5)
    device = torch.device("cuda")
    ctx = list(args.seq_lens)
    cap = [L + args.steps for L in ctx]
    B = len(ctx)
    manager, md, _ = bench.build_fp4_case(cap, args.heads, args.backend, skip_fill=True)
    try:
        stride = md.fp4_mla_page_table_stride
        md.high_precision_kv_pool.zero_()
        lat = [
            (torch.randn(L, bench.HEAD_DIM_QK, device=device) * 0.5).to(torch.bfloat16) for L in cap
        ]
        lens = torch.tensor(ctx, dtype=torch.int64, device=device)
        md.num_contexts, md.num_ctx_tokens = B, int(sum(ctx))
        md.num_context_blocks, md.num_generation_blocks = B * stride, 0
        md.batch_indices = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32, device=device), lens
        )
        md.positions = torch.cat([torch.arange(L, dtype=torch.int32, device=device) for L in ctx])
        md.kv_lens_cuda_runtime = lens.to(torch.int32)
        md.prompt_lens_cuda_runtime = lens.to(torch.int32)
        md.prompt_lens_cpu_runtime = torch.tensor(ctx, dtype=torch.int32)
        fp4.scatter_fp4_mla_kv_cache(
            md,
            torch.cat([x[:L] for x, L in zip(lat, ctx)]).contiguous(),
            layer_idx=0,
            token_offset=0,
            phase="context",
            local_layer=0,
            v_head_dim=512,
        )
        md.num_contexts = md.num_ctx_tokens = md.num_context_blocks = 0
        md.num_generation_blocks = B * stride
        md.prompt_lens_cuda_runtime = torch.ones(B, dtype=torch.int32, device=device)
        md.prompt_lens_cpu_runtime = torch.ones(B, dtype=torch.int32)
        md.batch_indices = torch.arange(B, dtype=torch.int32, device=device)
        cos_sin = torch.zeros(max(cap) + 1, 64, 2, dtype=torch.float32, device=device)
        cos_sin[..., 0] = 1.0
        qg = (torch.randn(B, args.heads, bench.HEAD_DIM_QK, device=device) * 0.25).to(
            torch.bfloat16
        )
        for step in range(args.steps):
            cur = [L + step for L in ctx]
            md.kv_lens_cuda_runtime = torch.tensor(
                [c + 1 for c in cur], dtype=torch.int32, device=device
            )
            md.positions = torch.tensor(cur, dtype=torch.int32, device=device)
            md.fp4_mla_generation_lengths_num_tokens = -1
            fp4.scatter_fp4_mla_kv_cache(
                md,
                torch.stack([lat[i][cur[i]] for i in range(B)]).contiguous(),
                layer_idx=0,
                token_offset=0,
                phase="generation",
                local_layer=0,
                v_head_dim=512,
                rotary_cos_sin=cos_sin,
                q_pe=qg[..., 512:],
                q_rope_out=qg[..., 512:],
                q_quant_input=qg,
            )
        torch.cuda.synchronize()
        kv, sf = manager.get_fp4_mla_cache_buffers(0)
        vp = manager.get_mla_v_packed_pool_base()
        print(
            f"WRITER_HASH backend={args.backend} K={_digest(kv)} KSF={_digest(sf)} "
            f"HP={_digest(md.high_precision_kv_pool)} VSF={_digest(md.fp4_mla_v_scale_pool)} "
            f"VPACK={_digest(vp)} Q={_digest(md._fp4_mla_prequantized_q[: B * args.heads])}",
            flush=True,
        )
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
