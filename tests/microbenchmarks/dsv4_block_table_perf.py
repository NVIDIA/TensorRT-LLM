# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from collections.abc import Callable

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401

BAD_PAGE_INDEX = -1


def _torch_sliding_block_tables(
    block_offsets: torch.Tensor,
    copy_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    valid_pool: torch.Tensor,
    scales: torch.Tensor,
    layer_offsets: torch.Tensor,
    output: torch.Tensor,
) -> None:
    base = block_offsets[pool_ids[:, :, None], copy_idx[None, None, :], 0, :]
    scaled_base = torch.where(
        (base == BAD_PAGE_INDEX) | ~(valid_pool[:, :, None, None]),
        BAD_PAGE_INDEX,
        base * scales[:, :, None, None] + layer_offsets[:, :, None, None],
    )
    output.copy_(scaled_base)


def _torch_sliding_block_tables_with_scratch(
    block_offsets: torch.Tensor,
    copy_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    valid_pool: torch.Tensor,
    scales: torch.Tensor,
    layer_offsets: torch.Tensor,
    block_positions: torch.Tensor,
    scratch_pages: torch.Tensor,
    scratch_begs: torch.Tensor,
    scratch_ends: torch.Tensor,
    scratch_slots: torch.Tensor,
    num_contexts: torch.Tensor,
    output: torch.Tensor,
) -> None:
    base = block_offsets[pool_ids[:, :, None], copy_idx[None, None, :], 0, :]
    scaled_base = torch.where(
        (base == BAD_PAGE_INDEX) | ~(valid_pool[:, :, None, None]),
        BAD_PAGE_INDEX,
        base * scales[:, :, None, None] + layer_offsets[:, :, None, None],
    )
    output.copy_(scaled_base)

    context_positions = torch.arange(
        scratch_begs.shape[1],
        dtype=torch.int32,
        device=scratch_begs.device,
    )
    active_context = context_positions < num_contexts
    mask = (
        (block_positions >= scratch_begs[:, :, None])
        & (block_positions < scratch_ends[:, :, None])
        & active_context[None, :, None]
    )
    range_index = torch.where(mask, block_positions - scratch_begs[:, :, None], 0)
    total_offset = range_index[pool_ids] * scratch_pages[:, :, None, None]
    slot_idx = (total_offset // scales[:, :, None, None]).clamp(max=scratch_slots.shape[-1] - 1)
    slot_id = scratch_slots[pool_ids].gather(-1, slot_idx.long())
    offset = total_offset % scales[:, :, None, None]
    scratch_index = (
        slot_id * scales[:, :, None, None]
        + (offset + layer_offsets[:, :, None, None]) % scales[:, :, None, None]
    )
    scratch_capacity = scratch_begs.shape[1]
    scratch_rows = scaled_base[:, :, :scratch_capacity, :]
    mask = mask[pool_ids] & valid_pool[:, :, None, None]
    output[:, :, :scratch_capacity, :].copy_(torch.where(mask, scratch_index, scratch_rows))


def _custom_sliding_block_tables(*args) -> None:
    torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables(*args)


def _custom_sliding_block_tables_with_scratch(
    block_offsets: torch.Tensor,
    copy_idx: torch.Tensor,
    pool_ids: torch.Tensor,
    valid_pool: torch.Tensor,
    scales: torch.Tensor,
    layer_offsets: torch.Tensor,
    block_positions: torch.Tensor,
    scratch_pages: torch.Tensor,
    scratch_begs: torch.Tensor,
    scratch_ends: torch.Tensor,
    scratch_slots: torch.Tensor,
    num_contexts: torch.Tensor,
    output: torch.Tensor,
) -> None:
    del block_positions
    torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables_with_scratch(
        block_offsets,
        copy_idx,
        pool_ids,
        valid_pool,
        scales,
        layer_offsets,
        scratch_pages,
        scratch_begs,
        scratch_ends,
        scratch_slots,
        num_contexts,
        output,
    )


def _make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    device = "cuda"
    block_offsets = torch.randint(
        0,
        1_000_000,
        (args.num_pools, args.table_capacity, 2, args.max_blocks),
        dtype=torch.int32,
        device=device,
    )
    block_offsets.masked_fill_(
        torch.rand(block_offsets.shape, device=device) < args.bad_ratio, BAD_PAGE_INDEX
    )
    copy_idx = (
        torch.randperm(args.table_capacity, device=device)[: args.num_tables]
        .to(torch.int32)
        .contiguous()
    )
    pool_ids = torch.randint(
        0,
        args.num_pools,
        (args.num_layers, args.num_attn_types),
        dtype=torch.int64,
        device=device,
    )
    valid_pool = torch.ones((args.num_layers, args.num_attn_types), dtype=torch.bool, device=device)
    if args.num_layers > 0 and args.num_attn_types > 0:
        pool_ids[0, -1] = -1
        valid_pool[0, -1] = False
        valid_pool[-1, 0] = False
    scales = torch.randint(1, args.max_scale + 1, pool_ids.shape, dtype=torch.int32, device=device)
    layer_offsets = torch.randint(
        0, args.max_scale, pool_ids.shape, dtype=torch.int32, device=device
    )
    block_positions = torch.arange(args.max_blocks, dtype=torch.int32, device=device)

    scratch_begs = torch.randint(
        0,
        max(1, args.max_blocks // 2),
        (args.num_pools, args.num_tables),
        dtype=torch.int32,
        device=device,
    )
    scratch_widths = torch.randint(
        1,
        max(2, args.max_blocks // 4),
        (args.num_pools, args.num_tables),
        dtype=torch.int32,
        device=device,
    )
    scratch_ends = torch.minimum(
        scratch_begs + scratch_widths,
        torch.tensor(args.max_blocks, dtype=torch.int32, device=device),
    )
    scratch_slots = torch.randint(
        0,
        1_000_000,
        (args.num_pools, args.num_tables, args.max_scratch_slots),
        dtype=torch.int32,
        device=device,
    )
    scratch_pages = torch.randint(
        1, args.max_scratch_pages + 1, pool_ids.shape, dtype=torch.int32, device=device
    )
    num_contexts = torch.tensor(args.num_contexts, dtype=torch.int32, device=device)
    output_shape = (args.num_layers, args.num_attn_types, args.num_tables, args.max_blocks)
    return {
        "block_offsets": block_offsets,
        "copy_idx": copy_idx,
        "pool_ids": pool_ids,
        "valid_pool": valid_pool,
        "scales": scales,
        "layer_offsets": layer_offsets,
        "block_positions": block_positions,
        "scratch_pages": scratch_pages,
        "scratch_begs": scratch_begs,
        "scratch_ends": scratch_ends,
        "scratch_slots": scratch_slots,
        "num_contexts": num_contexts,
        "reference_output": torch.empty(output_shape, dtype=torch.int32, device=device),
        "custom_output": torch.empty(output_shape, dtype=torch.int32, device=device),
    }


def _time_cuda(fn: Callable, fn_args: tuple, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*fn_args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*fn_args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _maybe_compile(fn: Callable, reference: str) -> Callable:
    if reference == "eager":
        return fn
    return torch.compile(fn, options={"max-autotune": True})


def _run_case(
    name: str, reference_fn: Callable, custom_fn: Callable, fn_args: tuple, args: argparse.Namespace
) -> None:
    reference_fn = _maybe_compile(reference_fn, args.reference)
    reference_output = fn_args[-1]
    custom_output = args.inputs["custom_output"]
    custom_args = (*fn_args[:-1], custom_output)

    reference_fn(*fn_args)
    custom_fn(*custom_args)
    torch.cuda.synchronize()
    if not torch.equal(custom_output, reference_output):
        mismatch = (custom_output != reference_output).nonzero()
        first = mismatch[0].tolist() if mismatch.numel() > 0 else []
        raise AssertionError(f"{name}: custom output differs from reference at {first}")

    reference_ms = _time_cuda(reference_fn, fn_args, args.warmup, args.iters)
    custom_ms = _time_cuda(custom_fn, custom_args, args.warmup, args.iters)
    print(
        f"{name}: reference={reference_ms:.4f} ms  custom={custom_ms:.4f} ms  "
        f"speedup={reference_ms / custom_ms:.3f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", choices=["compile", "eager"], default="compile")
    parser.add_argument("--case", choices=["all", "no-scratch", "scratch"], default="all")
    parser.add_argument("--num-pools", type=int, default=8)
    parser.add_argument("--table-capacity", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=61)
    parser.add_argument("--num-attn-types", type=int, default=4)
    parser.add_argument("--num-tables", type=int, default=256)
    parser.add_argument("--max-blocks", type=int, default=1024)
    parser.add_argument("--max-scale", type=int, default=128)
    parser.add_argument("--max-scratch-pages", type=int, default=16)
    parser.add_argument("--max-scratch-slots", type=int, default=64)
    parser.add_argument("--num-contexts", type=int, default=256)
    parser.add_argument("--bad-ratio", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_contexts > args.num_tables:
        raise ValueError("--num-contexts must be <= --num-tables")

    args.inputs = _make_inputs(args)
    common_args = (
        args.inputs["block_offsets"],
        args.inputs["copy_idx"],
        args.inputs["pool_ids"],
        args.inputs["valid_pool"],
        args.inputs["scales"],
        args.inputs["layer_offsets"],
    )
    scratch_args = (
        args.inputs["block_positions"],
        args.inputs["scratch_pages"],
        args.inputs["scratch_begs"],
        args.inputs["scratch_ends"],
        args.inputs["scratch_slots"],
        args.inputs["num_contexts"],
    )

    shape = tuple(args.inputs["reference_output"].shape)
    print(f"shape={shape} reference={args.reference} warmup={args.warmup} iters={args.iters}")
    if args.case in ("all", "no-scratch"):
        _run_case(
            "no-scratch",
            _torch_sliding_block_tables,
            _custom_sliding_block_tables,
            (*common_args, args.inputs["reference_output"]),
            args,
        )
    if args.case in ("all", "scratch"):
        _run_case(
            "scratch",
            _torch_sliding_block_tables_with_scratch,
            _custom_sliding_block_tables_with_scratch,
            (*common_args, *scratch_args, args.inputs["reference_output"]),
            args,
        )


if __name__ == "__main__":
    main()
