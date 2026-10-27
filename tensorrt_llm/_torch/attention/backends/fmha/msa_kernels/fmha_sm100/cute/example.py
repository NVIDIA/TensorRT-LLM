# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""End-to-end sparse attention example with NVTX ranges.

This file demonstrates two customer-facing flows:

A. Sparse attention (prefill) via ``sparse_atten_func``:

    1. Build k2q CSR metadata and the fused attention schedule.
    2. Pass CSR + schedule into the sparse attention forward.

B. Paged FP8 decode via ``SparseDecodePagedAttentionWrapper``:

    1. plan() once (builds split-KV schedule on GPU).
    2. run() per decode step (3 kernels: schedule was prebuilt, attn fwd,
       and combine when split_kv is True).

For Nsight Systems, run with ``--profile`` and capture the CUDA profiler range:

    # Sparse (prefill):
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
      --stop-on-range-end=true -o nsys_reports/sparse_e2e_example \
      python example.py --case both --warmup 2 --iters 1 --profile

    # Decode:
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
      --stop-on-range-end=true -o nsys_reports/decode_example \
      python example.py --case decode_long --warmup 3 --iters 5 --profile
"""

from __future__ import annotations

import argparse
import contextlib
import math
from dataclasses import dataclass
from typing import Iterable

import torch

from interface import sparse_atten_func, SparseDecodePagedAttentionWrapper
from src.sm100.prepare_k2q_csr import SparseK2qCsrBuilderSm100


@dataclass(frozen=True)
class SparseExampleCase:
    name: str
    batch: int
    seqlen_q: int
    seqlen_k: int
    head_kv: int
    qhead_per_kv: int


CASES: dict[str, SparseExampleCase] = {
    "small": SparseExampleCase(
        name="small_sink_smoke",
        batch=1,
        seqlen_q=4096,
        seqlen_k=4096,
        head_kv=1,
        qhead_per_kv=2,
    ),
    "ring48k": SparseExampleCase(
        name="bs1_nhq16_hkv1_seq48k_ring_attn",
        batch=1,
        seqlen_q=48 * 1024,
        seqlen_k=48 * 1024,
        head_kv=1,
        qhead_per_kv=16,
    ),
    "ulysses384k": SparseExampleCase(
        name="bs1_nhq2_hkv1_seq384k_ulysses",
        batch=1,
        seqlen_q=384 * 1024,
        seqlen_k=384 * 1024,
        head_kv=1,
        qhead_per_kv=2,
    ),
}


# ---------------------------------------------------------------------------
# Paged FP8 decode
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodeExampleCase:
    name: str
    batch: int
    seqlen_q: int       # q tokens per request (8 for our packed-GQA layout)
    seqlen_k: int       # KV context length per request
    head_kv: int
    qhead_per_kv: int   # must be 16
    page_size: int = 128


DECODE_CASES: dict[str, DecodeExampleCase] = {
    "decode_short": DecodeExampleCase(
        name="decode_b32_sq8_kv4k",
        batch=32, seqlen_q=8, seqlen_k=4096,
        head_kv=4, qhead_per_kv=16,
    ),
    "decode_mid": DecodeExampleCase(
        name="decode_b32_sq8_kv16k",
        batch=32, seqlen_q=8, seqlen_k=16384,
        head_kv=4, qhead_per_kv=16,
    ),
    "decode_long": DecodeExampleCase(
        name="decode_b32_sq8_kv131k",
        batch=32, seqlen_q=8, seqlen_k=131072,
        head_kv=4, qhead_per_kv=16,
    ),
    # Smaller batch + long kv: split_kv=True so combine kernel also runs.
    "decode_split": DecodeExampleCase(
        name="decode_b4_sq8_kv131k_split",
        batch=4, seqlen_q=8, seqlen_k=131072,
        head_kv=4, qhead_per_kv=16,
    ),
}


@contextlib.contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def make_cu_seqlens(lengths: Iterable[int], *, device: torch.device) -> torch.Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int32, device=device)


def make_sink_q2k(
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    *,
    head_kv: int,
    topk: int,
    blk_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a sink-token sparse pattern in q2k format.

    q2k has shape [head_kv, total_q, topK]. Values are batch-local KV block
    indices. Slot 0 always targets KV block 0, creating the sink load pattern.
    """
    total_q = sum(q_lens)
    q2k = torch.full((head_kv, total_q, topk), -1, dtype=torch.int32, device=device)
    head_hash = torch.arange(head_kv, dtype=torch.int64, device=device).view(-1, 1)
    q_cursor = 0

    for seqlen_q, seqlen_k in zip(q_lens, k_lens):
        num_kv_blocks = (seqlen_k + blk_kv - 1) // blk_kv
        if num_kv_blocks < 1:
            raise ValueError("each sequence must have at least one KV block")

        q_local = torch.arange(seqlen_q, dtype=torch.int32, device=device)
        budget = min(topk, num_kv_blocks)
        q_slice = q2k[:, q_cursor : q_cursor + seqlen_q, :]

        if budget >= 1:
            q_slice[:, :, 0] = 0
        if budget >= 2:
            q_slice[:, :, 1] = num_kv_blocks - 1
        if budget > 2:
            q_hash = q_local.to(torch.int64).view(1, -1)
            # Slots 2..topK-1 avoid the sink block and the last block.
            candidate_span = max(num_kv_blocks - 2, 1)
            for slot in range(2, budget):
                candidate = 1 + torch.remainder(
                    q_hash * 1103515245 + head_hash * 12345 + slot * 2654435761,
                    candidate_span,
                )
                q_slice[:, :, slot] = candidate.to(torch.int32)

        q_cursor += seqlen_q

    return q2k.contiguous()


def build_inputs(
    case: SparseExampleCase,
    *,
    dtype: torch.dtype,
    topk: int,
    blk_kv: int,
    device: torch.device,
) -> dict[str, object]:
    q_lens = (case.seqlen_q,) * case.batch
    k_lens = (case.seqlen_k,) * case.batch
    total_q = sum(q_lens)
    total_k = sum(k_lens)
    head_q = case.head_kv * case.qhead_per_kv

    q = torch.randn(total_q, head_q, 128, dtype=dtype, device=device)
    k = torch.randn(total_k, case.head_kv, 128, dtype=dtype, device=device)
    v = torch.randn(total_k, case.head_kv, 128, dtype=dtype, device=device)
    q2k_indices = make_sink_q2k(
        q_lens,
        k_lens,
        head_kv=case.head_kv,
        topk=topk,
        blk_kv=blk_kv,
        device=device,
    )
    cu_seqlens_q = make_cu_seqlens(q_lens, device=device)
    cu_seqlens_k = make_cu_seqlens(k_lens, device=device)

    return {
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": q2k_indices,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "q_lens": q_lens,
        "k_lens": k_lens,
        "total_rows": sum((length + blk_kv - 1) // blk_kv for length in k_lens),
    }


def run_e2e_once(
    *,
    case: SparseExampleCase,
    inputs: dict[str, object],
    csr_builder: SparseK2qCsrBuilderSm100,
    topk: int,
    blk_kv: int,
    partial_dtype: torch.dtype,
) -> None:
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]
    cu_seqlens_q = inputs["cu_seqlens_q"]
    cu_seqlens_k = inputs["cu_seqlens_k"]
    q_lens = inputs["q_lens"]
    k_lens = inputs["k_lens"]

    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(v, torch.Tensor)
    assert isinstance(cu_seqlens_q, torch.Tensor)
    assert isinstance(cu_seqlens_k, torch.Tensor)

    with nvtx_range(f"{case.name}__build_k2q_csr_with_schedule"):
        k2q_row_ptr, k2q_q_indices, schedule = csr_builder(
            inputs["q2k_indices"],
            cu_seqlens_q,
            cu_seqlens_k,
            total_k=int(k.shape[0]),
            blk_kv=blk_kv,
            max_seqlen_k=max(k_lens),
            max_seqlen_q=max(q_lens),
            total_rows=int(inputs["total_rows"]),
            qhead_per_kv=case.qhead_per_kv,
            return_schedule=True,
        )

    with nvtx_range(f"{case.name}__sparse_attention_fwd"):
        sparse_atten_func(
            q,
            k,
            v,
            k2q_row_ptr,
            k2q_q_indices,
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(k_lens),
            blk_kv=blk_kv,
            causal=False,
            softmax_scale=1.0 / math.sqrt(q.shape[-1]),
            partial_dtype=partial_dtype,
            return_softmax_lse=False,
            schedule=schedule,
        )


def build_decode_inputs(case: DecodeExampleCase, *, device: torch.device) -> dict:
    """Build paged FP8 decode inputs.

    Layout follows the SparseDecodePagedAttentionWrapper contract:
      - q: [B*Sq, Hq, D] fp8
      - k_paged, v_paged: [num_pages, Hkv, page_size, D] fp8
      - page_table: [B, max_pages_per_seq] i32
      - seqused_k: [B] i32
    """
    B, sq, kv = case.batch, case.seqlen_q, case.seqlen_k
    head_q = case.head_kv * case.qhead_per_kv
    dim = 128
    pages_per_b = (kv + case.page_size - 1) // case.page_size
    total_pages = B * pages_per_b

    q = torch.randn(B * sq, head_q, dim, device=device, dtype=torch.float16
                    ).to(torch.float8_e4m3fn)
    k = torch.randn(total_pages, case.head_kv, case.page_size, dim, device=device,
                    dtype=torch.float16).to(torch.float8_e4m3fn)
    v = torch.randn(total_pages, case.head_kv, case.page_size, dim, device=device,
                    dtype=torch.float16).to(torch.float8_e4m3fn)
    page_table = torch.arange(total_pages, device=device, dtype=torch.int32
                              ).view(B, pages_per_b)
    seqused_k = torch.full((B,), kv, device=device, dtype=torch.int32)
    return {
        "q": q, "k": k, "v": v,
        "page_table": page_table, "seqused_k": seqused_k,
        "max_seqlen_k": kv,
    }


def run_decode_once(
    *,
    case: DecodeExampleCase,
    wrapper: SparseDecodePagedAttentionWrapper,
    inputs: dict,
    out_buffer: torch.Tensor,
) -> None:
    """Single decode step.

    plan() is called once outside; here we just exercise run() which is
    the per-step hot path.  Three CUDA kernels expected in the profile:
      1. build_decode_schedule_gpu_kernel (~17us, runs in plan() — NOT here)
      2. decode_attention kernel (the main UMMA attn fwd)
      3. SparseDecodeForwardCombine (only when split_kv=True)

    Anything else visible in nsys is overhead.
    """
    q = inputs["q"]; k = inputs["k"]; v = inputs["v"]
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    with nvtx_range(f"{case.name}__run"):
        wrapper.run(q, k, v, softmax_scale=softmax_scale,
                    return_softmax_lse=False, out=out_buffer)


def resolve_cases(raw_case: str) -> tuple[SparseExampleCase, ...]:
    if raw_case == "both":
        return (CASES["ring48k"], CASES["ulysses384k"])
    return (CASES[raw_case],)


def resolve_decode_cases(raw_case: str) -> tuple[DecodeExampleCase, ...]:
    if raw_case == "decode_all":
        return tuple(DECODE_CASES.values())
    return (DECODE_CASES[raw_case],)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=(
            "small", "ring48k", "ulysses384k", "both",
            "decode_short", "decode_mid", "decode_long", "decode_split", "decode_all",
        ),
        default="small",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--blk-kv", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true", help="Wrap measured iterations in cudaProfilerStart/Stop")
    return parser.parse_args()


def _is_decode_case(raw_case: str) -> bool:
    return raw_case.startswith("decode")


def main_decode(args: argparse.Namespace) -> None:
    """Decode-only flow: plan() once, run() N iters under profile range."""
    device = torch.device("cuda")
    cases = resolve_decode_cases(args.case)

    # plan() each case once, outside the profile range, so the profile
    # captures the *per-step* run() hot path (what production sees).
    contexts = []
    for case in cases:
        inputs = build_decode_inputs(case, device=device)
        wrapper = SparseDecodePagedAttentionWrapper(
            blk_kv=case.page_size, causal=True,
        )
        with nvtx_range(f"{case.name}__plan"):
            wrapper.plan(
                page_table=inputs["page_table"],
                seqused_k=inputs["seqused_k"],
                seqlen_q=case.seqlen_q,
                max_seqlen_k=inputs["max_seqlen_k"],
                num_qo_heads=case.head_kv * case.qhead_per_kv,
                num_kv_heads=case.head_kv,
                head_dim=128,
            )
        out_buffer = torch.empty(
            inputs["q"].shape, dtype=torch.bfloat16, device=device,
        )
        contexts.append((case, wrapper, inputs, out_buffer))

    # Warmup: compile the attn kernel + combine on first call so the
    # profile range doesn't include JIT.
    for _ in range(args.warmup):
        for case, wrapper, inputs, out_buffer in contexts:
            with nvtx_range(f"{case.name}__warmup"):
                run_decode_once(
                    case=case, wrapper=wrapper,
                    inputs=inputs, out_buffer=out_buffer,
                )
    torch.cuda.synchronize()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        with nvtx_range("decode_e2e_profile"):
            for iteration in range(args.iters):
                for case, wrapper, inputs, out_buffer in contexts:
                    with nvtx_range(f"{case.name}__iter_{iteration}"):
                        run_decode_once(
                            case=case, wrapper=wrapper,
                            inputs=inputs, out_buffer=out_buffer,
                        )
        torch.cuda.synchronize()
    finally:
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()

    print(
        f"completed decode case={args.case} warmup={args.warmup} "
        f"iters={args.iters}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("example.py requires CUDA")

    torch.manual_seed(args.seed)

    if _is_decode_case(args.case):
        main_decode(args)
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16
    partial_dtype = torch.float32
    csr_builder = SparseK2qCsrBuilderSm100()
    cases = resolve_cases(args.case)
    contexts = [
        (case, build_inputs(case, dtype=dtype, topk=args.topk, blk_kv=args.blk_kv, device=device))
        for case in cases
    ]

    for _ in range(args.warmup):
        for case, inputs in contexts:
            with nvtx_range(f"{case.name}__warmup"):
                run_e2e_once(
                    case=case,
                    inputs=inputs,
                    csr_builder=csr_builder,
                    topk=args.topk,
                    blk_kv=args.blk_kv,
                    partial_dtype=partial_dtype,
                )
    torch.cuda.synchronize()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        with nvtx_range("sparse_attention_e2e_profile"):
            for iteration in range(args.iters):
                for case, inputs in contexts:
                    with nvtx_range(f"{case.name}__iter_{iteration}"):
                        run_e2e_once(
                            case=case,
                            inputs=inputs,
                            csr_builder=csr_builder,
                            topk=args.topk,
                            blk_kv=args.blk_kv,
                            partial_dtype=partial_dtype,
                        )
        torch.cuda.synchronize()
    finally:
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()

    print(
        "completed "
        f"case={args.case} warmup={args.warmup} iters={args.iters} "
        f"topk={args.topk} blk_kv={args.blk_kv}"
    )


if __name__ == "__main__":
    main()
