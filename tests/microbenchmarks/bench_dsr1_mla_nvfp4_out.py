# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Per-layer timing of the NVFP4-out MLA decode epilogues (TRTLLM_MLA_NVFP4_EPILOGUE) on one Rubin GPU.

Q chain (bmm_q -> fused RoPE/Q/KV writer) and O chain (bmm_o -> o_proj input quantizer), today against the
NVFP4-out GEMMs, each as a CUDA graph of --reps back-to-back calls replayed for --seconds, rounds interleaved:

* q_base:    bmm_out into fused_q[..., :512], then the writer with its Q-prefix programs (serial, as today)
* q_serial:  bmm_q NVFP4-out (2x1 cluster), then the writer without Q-prefix programs
* q_overlap: bmm_q NVFP4-out (2x2 cluster) with that writer on a second stream (bit 4)
* o_base:    bmm_out into [T, 16384] BF16, then fp4_quantize (Triton, swizzled)
* o_fused:   bmm_o NVFP4-out

    python bench_dsr1_mla_nvfp4_out.py --batches 256 992 1536 --ctx-mean 5000
"""

import argparse
import os
import sys

os.environ.setdefault("TRTLLM_FP4_MLA_MXFP4_2D", "1")
os.environ.setdefault("TRTLLM_FP4_MLA_ATTENTION_BACKEND", "cutedsl")
os.environ.setdefault("TRTLLM_NVFP4_QUANTIZE_TRITON", "1")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_dsr1_mla_decode as bench  # noqa: E402

HEADS = 128


def _time_graph(fn, reps, seconds, side=None):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    fn()  # eager warmup (compiles)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            for _ in range(reps):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    n = 0
    total = 0.0
    while total < seconds * 1000.0:
        start.record()
        for _ in range(10):
            g.replay()
        end.record()
        end.synchronize()
        total += start.elapsed_time(end)
        n += 10
    return total * 1000.0 / (n * reps)  # us per call


def main():
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.dense_gemm_nvfp4_out import (
        MODE_O,
        Sm107Nvfp4OutBmm,
    )
    from tensorrt_llm._torch.modules.fused_ops.nvfp4_quantize_triton import fp4_quantize
    from tensorrt_llm.math_utils import pad_up

    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, nargs="+", default=[256, 992, 1536])
    p.add_argument("--ctx-mean", type=int, default=5000)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--rounds", type=int, default=2)
    args = p.parse_args()
    device = torch.device("cuda")
    gen = torch.Generator()
    gen.manual_seed(5)
    for b in args.batches:
        seq_lens = [
            int(v)
            for v in torch.randint(
                args.ctx_mean // 2, args.ctx_mean * 3 // 2, (b,), generator=gen
            ).tolist()
        ]
        manager, metadata, _ = bench.build_fp4_case(seq_lens, HEADS, "cutedsl", skip_fill=True)
        try:
            w_uk = (torch.randn(HEADS, 512, 128, device=device) * 0.05).to(torch.bfloat16)
            w_uv = (torch.randn(HEADS, 128, 512, device=device) * 0.05).to(torch.bfloat16)
            q = (torch.randn(b, HEADS, 192, device=device)).to(torch.bfloat16)
            q_nope_t = q[..., :128].transpose(0, 1)
            q_pe = q[..., 128:]
            latent = (torch.randn(b, bench.HEAD_DIM_QK, device=device) * 0.25).to(torch.bfloat16)
            cos_sin = torch.randn(max(seq_lens) + 2, 64, 2, device=device)
            metadata.kv_lens_cuda_runtime = torch.tensor(
                [s for s in seq_lens], dtype=torch.int32, device=device
            )
            metadata.positions = torch.tensor(
                [s - 1 for s in seq_lens], dtype=torch.int32, device=device
            )
            metadata.batch_indices = torch.arange(b, dtype=torch.int32, device=device)
            # all-generation batch, one token per sequence (as build_fp4_case does after its fill)
            metadata.num_contexts = 0
            metadata.num_ctx_tokens = 0
            metadata.num_context_blocks = 0
            metadata.num_generation_blocks = b * metadata.fp4_mla_page_table_stride
            metadata.prompt_lens_cuda_runtime = torch.ones(b, dtype=torch.int32, device=device)
            metadata.prompt_lens_cpu_runtime = torch.ones(b, dtype=torch.int32)
            metadata.fp4_mla_generation_lengths_num_tokens = -1
            fused_q = torch.empty(b, HEADS, 576, dtype=torch.bfloat16, device=device)
            side = torch.cuda.Stream()

            def writer(external):
                fp4.scatter_fp4_mla_kv_cache(
                    metadata,
                    latent,
                    layer_idx=0,
                    token_offset=0,
                    phase="generation",
                    local_layer=0,
                    v_head_dim=512,
                    rotary_cos_sin=cos_sin,
                    q_pe=q_pe,
                    q_rope_out=fused_q[..., 512:],
                    q_quant_input=fused_q,
                    q_prefix_external=external,
                )

            def q_base():
                torch.ops.trtllm.bmm_out(
                    q_nope_t, w_uk.transpose(1, 2), fused_q[..., :512].transpose(0, 1)
                )
                writer(False)

            def q_serial():
                fp4.fp4_mla_bmm_q_fp4_prefix(metadata, q_nope_t, w_uk, overlap=False)
                writer(True)

            def q_overlap():
                side.wait_stream(torch.cuda.current_stream())
                fp4.fp4_mla_bmm_q_fp4_prefix(metadata, q_nope_t, w_uk, overlap=True)
                with torch.cuda.stream(side):
                    writer(True)
                torch.cuda.current_stream().wait_stream(side)

            latent_o = (torch.randn(b, HEADS, 512, device=device)).to(torch.bfloat16)
            o_scale = torch.tensor([100.0], device=device)
            o_bf16 = torch.empty(b, HEADS * 128, dtype=torch.bfloat16, device=device)
            o_fp4 = torch.empty(b, HEADS * 64, dtype=torch.uint8, device=device)
            o_sf = torch.empty(pad_up(b, 128) * 1024, dtype=torch.uint8, device=device)

            def o_base():
                torch.ops.trtllm.bmm_out(
                    latent_o.transpose(0, 1),
                    w_uv.transpose(1, 2),
                    o_bf16.view(b, HEADS, 128).transpose(0, 1),
                )
                fp4_quantize(o_bf16, o_scale, 16, False, True)

            def o_fused():
                Sm107Nvfp4OutBmm.run(
                    MODE_O,
                    latent_o.transpose(0, 1),
                    w_uv,
                    o_fp4,
                    o_sf,
                    o_scale,
                    out_stride_m=HEADS * 64,
                    out_stride_l=64,
                    sf_row_m=1,
                    sf_row_l=0,
                    sf_col_l=8,
                    sf_cols_padded=1024,
                    fp4_rows=b,
                    sf_rows=pad_up(b, 128),
                )

            def bmm_q_only():
                torch.ops.trtllm.bmm_out(
                    q_nope_t, w_uk.transpose(1, 2), fused_q[..., :512].transpose(0, 1)
                )

            def writer_full():
                writer(False)

            def writer_ext():
                writer(True)

            variants = dict(
                q_base=q_base,
                q_serial=q_serial,
                q_overlap=q_overlap,
                bmm_q=bmm_q_only,
                writer_full=writer_full,
                writer_ext=writer_ext,
                o_base=o_base,
                o_fused=o_fused,
            )
            res = {k: [] for k in variants}
            for _ in range(args.rounds):
                for name, fn in variants.items():
                    res[name].append(_time_graph(fn, args.reps, args.seconds))
            line = " ".join(f"{k}={min(v):.1f}" for k, v in res.items())
            q_save = min(res["q_base"]) - min(res["q_overlap"])
            o_save = min(res["o_base"]) - min(res["o_fused"])
            print(
                f"NVFP4OUT_BENCH B={b} ctx_mean={args.ctx_mean} us/call: {line} | saved/layer "
                f"q(overlap)={q_save:.1f} q(serial)={min(res['q_base']) - min(res['q_serial']):.1f} "
                f"o={o_save:.1f}",
                flush=True,
            )
        finally:
            manager.shutdown()


if __name__ == "__main__":
    main()
