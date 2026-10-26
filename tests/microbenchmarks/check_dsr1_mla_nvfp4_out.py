# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Byte-exact check of the NVFP4-out MLA decode epilogues (TRTLLM_MLA_NVFP4_EPILOGUE) on Rubin SM107.

Part O (bit 1): the CuTe DSL bmm_o with o_proj's NVFP4 quantizer in its epilogue against today's path,
``torch.ops.trtllm.bmm_out`` (BF16) + ``fp4_quantize`` (the C++ op and the Triton kernel), byte for byte:
packed E2M1 and the whole 128x4-swizzled scale buffer including its padding rows.

Part Q (bit 2): the CuTe DSL bmm_q with the FP4 MLA Q-prefix quantizer in its epilogue followed by the real fused
generation RoPE/Q/KV writer with ``q_prefix_external=True``, against today's path (``bmm_out`` into
``fused_q[..., :512]`` + the same writer with its Q-prefix programs), on a real FP4 MLA (MXFP4-2D) cache: every
active Q byte and Q-scale byte must match. Also checks that the GEMM alone writes only the prefix bytes and scale
columns 0-31, the co-run tactic (writer on a second stream), and CUDA-graph replay.

Run (sets TRTLLM_FP4_MLA_MXFP4_2D=1 before import)::

    python check_dsr1_mla_nvfp4_out.py --tokens 1 16 64 129 992 1536
"""

import argparse
import os
import sys

os.environ.setdefault("TRTLLM_FP4_MLA_MXFP4_2D", "1")
os.environ.setdefault("TRTLLM_FP4_MLA_ATTENTION_BACKEND", "cutedsl")

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_dsr1_mla_decode as bench  # noqa: E402

HEADS = 128
V_HEAD_DIM = 128
NOPE = 128
KV_LORA = 512


def _activations(shape, gen, device, spiky: bool) -> torch.Tensor:
    x = torch.randn(shape, generator=gen, device=device)
    if spiky:
        # rows over many decades, zero blocks, tiny rows and saturating rows
        row_scale = torch.pow(
            10.0, torch.randint(-6, 3, (shape[0], 1, 1), generator=gen, device=device)
        )
        x = x * row_scale
        x[:, :, :16] = 0.0
        if shape[0] > 2:
            x[1] = 1e-30
            x[2] = 1e4
    return x.to(torch.bfloat16)


def check_o(tokens, gen, device, graph: bool) -> int:
    from tensorrt_llm._torch.cute_dsl_kernels.rubin.dense_gemm_nvfp4_out import (
        MODE_O,
        Sm107Nvfp4OutBmm,
    )
    from tensorrt_llm._torch.modules.fused_ops.nvfp4_quantize_triton import nvfp4_quantize_triton
    from tensorrt_llm.math_utils import pad_up

    hidden = HEADS * V_HEAD_DIM
    w_uv = (torch.randn(HEADS, V_HEAD_DIM, KV_LORA, generator=gen, device=device) * 0.05).to(
        torch.bfloat16
    )
    bad_total = 0
    for spiky in (False, True):
        for t in tokens:
            latent = _activations((t, HEADS, KV_LORA), gen, device, spiky)
            scale = torch.tensor([float(torch.empty(1).uniform_(20.0, 400.0))], device=device)
            ref = torch.empty(t, hidden, dtype=torch.bfloat16, device=device)
            torch.ops.trtllm.bmm_out(
                latent.transpose(0, 1),
                w_uv.transpose(1, 2),
                ref.view(t, HEADS, V_HEAD_DIM).transpose(0, 1),
            )
            ref_q, ref_sf = torch.ops.trtllm.fp4_quantize(ref, scale, 16, False, True)
            tq, tsf = nvfp4_quantize_triton(ref, scale, True)
            sf_bytes = pad_up(t, 128) * pad_up(hidden // 16, 4)
            out = torch.full((t, hidden // 2), 0xA5, dtype=torch.uint8, device=device)
            out_sf = torch.full((sf_bytes,), 0xA5, dtype=torch.uint8, device=device)

            def run():
                return Sm107Nvfp4OutBmm.run(
                    MODE_O,
                    latent.transpose(0, 1),
                    w_uv,
                    out,
                    out_sf,
                    scale,
                    out_stride_m=hidden // 2,
                    out_stride_l=V_HEAD_DIM // 2,
                    sf_row_m=1,
                    sf_row_l=0,
                    sf_col_l=V_HEAD_DIM // 16,
                    sf_cols_padded=pad_up(hidden // 16, 4),
                    fp4_rows=t,
                    sf_rows=pad_up(t, 128),
                )

            tactic = run()
            torch.cuda.synchronize()
            bad = int((out != ref_q.view(torch.uint8)).sum()) + int(
                (out_sf != ref_sf.view(torch.uint8).reshape(-1)[:sf_bytes]).sum()
            )
            tri = int((tq.view(torch.uint8) != ref_q.view(torch.uint8)).sum()) + int(
                (tsf.view(torch.uint8) != ref_sf.view(torch.uint8).reshape(-1)).sum()
            )
            gbad = 0
            if graph:
                eager_out, eager_sf = out.clone(), out_sf.clone()
                out.fill_(0x5A)
                out_sf.fill_(0x5A)
                g = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(g, stream=stream):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                gbad = int((out != eager_out).sum()) + int((out_sf != eager_sf).sum())
            print(
                f"NVFP4OUT O tokens={t} spiky={spiky} tactic={tactic} mismatches={bad} "
                f"triton_vs_cpp={tri} graph_vs_eager={gbad}",
                flush=True,
            )
            bad_total += bad + gbad
    return bad_total


def check_q(batch, gen, cpu_gen, device, overlap: bool, graph: bool) -> int:
    import tensorrt_llm._torch.attention_backend.fp4_mla as fp4

    seq_lens = [int(v) for v in torch.randint(40, 400, (batch,), generator=cpu_gen).tolist()]
    manager, metadata, _ = bench.build_fp4_case(seq_lens, HEADS, "cutedsl")
    try:
        metadata.fp4_mla_generation_lengths_num_tokens = -1
        b = len(seq_lens)
        w_uk = (torch.randn(HEADS, KV_LORA, NOPE, generator=gen, device=device) * 0.05).to(
            torch.bfloat16
        )
        q = _activations((b, HEADS, NOPE + 64), gen, device, spiky=False)
        q_nope_t = q[..., :NOPE].transpose(0, 1)
        q_pe = q[..., NOPE:]
        latent = (torch.randn(b, bench.HEAD_DIM_QK, generator=gen, device=device) * 0.25).to(
            torch.bfloat16
        )
        cos_sin = torch.randn(max(seq_lens) + 2, 64, 2, generator=gen, device=device)
        kv = torch.tensor([s + 1 for s in seq_lens], dtype=torch.int32, device=device)
        metadata.kv_lens_cuda_runtime = kv
        metadata.positions = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        metadata.batch_indices = torch.arange(b, dtype=torch.int32, device=device)

        def writer(fused_q, external):
            metadata.fp4_mla_generation_lengths_num_tokens = -1
            fp4.scatter_fp4_mla_kv_cache(
                metadata,
                latent,
                layer_idx=0,
                token_offset=0,
                phase="generation",
                local_layer=0,
                v_head_dim=KV_LORA,
                rotary_cos_sin=cos_sin,
                q_pe=q_pe,
                q_rope_out=fused_q[..., KV_LORA:],
                q_quant_input=fused_q,
                q_prefix_external=external,
            )

        rows = b * HEADS
        nsf = fp4._get_fp4_mla_swizzled_scale_size(rows, fp4.FP4_MLA_Q_LOGICAL_DIM)
        # Today's path.
        fused_q = torch.empty(b, HEADS, KV_LORA + 64, dtype=torch.bfloat16, device=device)
        torch.ops.trtllm.bmm_out(
            q_nope_t, w_uk.transpose(1, 2), fused_q[..., :KV_LORA].transpose(0, 1)
        )
        writer(fused_q, False)
        torch.cuda.synchronize()
        ref_q = metadata._fp4_mla_prequantized_q[:rows].clone()
        ref_sf = metadata._fp4_mla_prequantized_q_sf.view(torch.uint8)[:nsf].clone()
        q_buf, q_sf_buf, _ = fp4._prepare_fp4_mla_q_buffers(metadata, b, HEADS, device)

        # The GEMM alone: prefix bytes + scale columns 0-31 only.
        q_buf.fill_(0xA5)
        q_sf_buf.view(torch.uint8).fill_(0xA5)
        tactic = fp4.fp4_mla_bmm_q_fp4_prefix(metadata, q_nope_t, w_uk, overlap=overlap)
        torch.cuda.synchronize()
        offsets = torch.arange(nsf, device=device)
        logical_col = offsets % 4 + ((offsets % (128 * 40)) // 512) * 4
        prefix_cols = logical_col < 32
        got_q, got_sf = q_buf[:rows], q_sf_buf.view(torch.uint8)[:nsf]
        bad_prefix = int((got_q[:, :256] != ref_q[:, :256]).sum()) + int(
            (got_sf[prefix_cols] != ref_sf[prefix_cols]).sum()
        )
        untouched = int((got_q[:, 256:] != 0xA5).sum()) + int((got_sf[~prefix_cols] != 0xA5).sum())

        # The fused path end to end (fused_q's BF16 prefix left as garbage).
        def fused_path():
            fused_q2 = torch.full_like(fused_q, float("nan"))
            if overlap:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                fp4.fp4_mla_bmm_q_fp4_prefix(metadata, q_nope_t, w_uk, overlap=True)
                with torch.cuda.stream(side):
                    writer(fused_q2, True)
                torch.cuda.current_stream().wait_stream(side)
            else:
                fp4.fp4_mla_bmm_q_fp4_prefix(metadata, q_nope_t, w_uk, overlap=False)
                writer(fused_q2, True)
            return fused_q2

        q_buf.fill_(0xA5)
        q_sf_buf.view(torch.uint8).fill_(0xA5)
        fused_q2 = fused_path()
        torch.cuda.synchronize()
        bad_e2e = int((q_buf[:rows] != ref_q).sum()) + int(
            (q_sf_buf.view(torch.uint8)[:nsf] != ref_sf).sum()
        )
        rope_tail_bad = int(
            (
                fused_q2[..., KV_LORA:].view(torch.int16)
                != fused_q[..., KV_LORA:].view(torch.int16)
            ).sum()
        )
        gbad = 0
        if graph and not overlap:
            q_buf.fill_(0x5A)
            q_sf_buf.view(torch.uint8).fill_(0x5A)
            fused_q3 = torch.full_like(fused_q, float("nan"))
            g = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(g, stream=stream):
                    fp4.fp4_mla_bmm_q_fp4_prefix(metadata, q_nope_t, w_uk, overlap=False)
                    writer(fused_q3, True)
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(3):
                g.replay()
            torch.cuda.synchronize()
            gbad = int((q_buf[:rows] != ref_q).sum()) + int(
                (q_sf_buf.view(torch.uint8)[:nsf] != ref_sf).sum()
            )
        print(
            f"NVFP4OUT Q batch={b} overlap={overlap} tactic={tactic} gemm_prefix_mismatches={bad_prefix} "
            f"gemm_wrote_outside_prefix={untouched} e2e_q_mismatches={bad_e2e} rope_tail_mismatches={rope_tail_bad} "
            f"graph_mismatches={gbad}",
            flush=True,
        )
        return bad_prefix + untouched + bad_e2e + rope_tail_bad + gbad
    finally:
        manager.shutdown()


def check_verify_counter(device) -> int:
    """VERIFY=2 counter: eager adds and CUDA-graph replays accumulate on the device and reach pinned host memory."""
    from tensorrt_llm._torch.modules.mla import VerifyMismatchCounter

    counter = VerifyMismatchCounter("[check]", ("x", "y"), period=3600.0)
    x = torch.zeros(1000, dtype=torch.uint8, device=device)
    y = torch.zeros(1000, dtype=torch.uint8, device=device)
    y[:7] = 1

    def add():
        counter.add({"x": (x != y).sum(), "y": (x == x).sum() * 0})

    add()  # eager
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(g, stream=stream):
            add()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    got = counter.host.tolist()
    # The capture itself does not execute kernels: 1 eager add + 5 replays = 6 checks, 6 x 7 mismatches.
    want = [6 * 7, 0, 6]
    print(
        f"NVFP4OUT V counter host={got} expected={want} pinned={counter.host.is_pinned()}",
        flush=True,
    )
    return int(got != want)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 64, 129, 992, 1536])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--parts", default="OQV")
    p.add_argument("--no-graph", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda")
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    total = 0
    if "O" in args.parts:
        total += check_o(args.tokens, gen, device, graph=not args.no_graph)
    if "Q" in args.parts:
        for batch in args.tokens:
            for overlap in (False, True):
                total += check_q(batch, gen, cpu_gen, device, overlap, graph=not args.no_graph)
    if "V" in args.parts:
        total += check_verify_counter(device)
    print(f"NVFP4OUT_DONE ok={total == 0} mismatches={total}", flush=True)


if __name__ == "__main__":
    main()
