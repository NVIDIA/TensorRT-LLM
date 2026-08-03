# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared harness for the KV-cache compaction tests."""

import torch


def encode_block_offsets(page_ids: torch.Tensor) -> torch.Tensor:
    """Native V2 [pool, request, K/V, block] layout: K = 2*page, V = K+1."""
    if page_ids.ndim == 2:
        page_ids = page_ids.unsqueeze(0)
    encoded = torch.empty(
        page_ids.shape[0],
        page_ids.shape[1],
        2,
        page_ids.shape[2],
        dtype=torch.int32,
        device=page_ids.device,
    )
    encoded[:, :, 0] = page_ids.to(torch.int32) * 2
    encoded[:, :, 1] = encoded[:, :, 0] + 1
    return encoded


def _write_move_offsets(compaction, offsets, moves_per_request):
    cumulative = [0]
    for count in moves_per_request:
        cumulative.append(cumulative[-1] + count)
    # Rows past the cohort are padding and contribute no moves.
    cumulative.extend(cumulative[-1:] * (compaction["request_count"] - len(moves_per_request)))
    offsets.copy_(torch.tensor(cumulative, dtype=torch.int32), non_blocking=True)


def set_protected_tails(compaction, tail_lengths, draft_tail_lengths=None):
    """Load per-request protected tails into the caller-owned move offsets."""
    if len(tail_lengths) > compaction["request_count"]:
        raise ValueError("the cohort exceeds the compaction request capacity")
    if any(tail < 0 or tail > compaction["protected_tail_capacity"] for tail in tail_lengths):
        raise ValueError("a protected tail exceeds the configured capacity")
    _write_move_offsets(
        compaction,
        compaction["dense_move_offsets"],
        [compaction["decode_keep_count"] + int(tail) for tail in tail_lengths],
    )
    if compaction["has_swa"]:
        _write_move_offsets(
            compaction,
            compaction["swa_move_offsets"],
            [compaction["swa_window"] + int(tail) for tail in tail_lengths],
        )
    if compaction["draft_move_offsets"] is not None:
        if draft_tail_lengths is None:
            draft_tail_lengths = [0] * len(tail_lengths)
        if len(draft_tail_lengths) != len(tail_lengths):
            raise ValueError("draft protected tails must match the cohort")
        if any(
            tail < 0 or tail > compaction["draft_protected_tail_capacity"]
            for tail in draft_tail_lengths
        ):
            raise ValueError("a draft protected tail exceeds the configured capacity")
        _write_move_offsets(
            compaction,
            compaction["draft_move_offsets"],
            [compaction["decode_keep_count"] + int(tail) for tail in draft_tail_lengths],
        )


def make_ramp_pools(
    count,
    *,
    num_kv_heads=2,
    pages=6,
    tokens_per_block=32,
    head_dim=64,
    layer_stride=37,
    base=0,
    device=None,
):
    """bf16 pools with a shifted ``arange % 251`` ramp: every wrong move
    lands on a different byte pattern (supported geometry defaults)."""
    return [
        (
            (
                torch.arange(
                    pages * 2 * num_kv_heads * tokens_per_block * head_dim,
                    dtype=torch.int32,
                    device=device,
                )
                + base
                + layer * layer_stride
            )
            % 251
        )
        .view(pages, 2, num_kv_heads, tokens_per_block, head_dim)
        .to(torch.bfloat16)
        for layer in range(count)
    ]


def build_compaction(**overrides):
    """``build_compaction_params`` with the suite's 2-layer defaults:
    allocates the caller-owned move-offset rows (capacity cumsum) and SWA
    destination bases, and hands the test's pre-settled
    ``kept_token_ordinals`` in as the decision rows. Returns the opaque
    ``params`` plus a test-side mirror of the caller-owned inputs."""
    from tensorrt_llm._torch.kv_cache_compression.compaction import build_compaction_params

    args = dict(
        eviction_mode="union",
        dense_layers=[0, 1],
        swa_layers=[],
        layer_group_representative={0: 0, 1: 1},
        layer_pool_ids=[0, 0],
        request_count=2,
        decode_keep_count=4,
        swa_window=None,
    )
    args.update(overrides)
    args.pop("eviction_mode")
    kept = args.pop("kept_token_ordinals")
    request_count = args["request_count"]
    keep_count = args["decode_keep_count"]
    tail = int(args.get("protected_tail_capacity", 0))
    draft_tail = int(args.get("draft_protected_tail_capacity") or 0)
    has_draft = bool(args.get("draft_layers"))
    has_swa = bool(args["swa_layers"])
    device = args["layer_pools"][args["dense_layers"][0]].device
    swa_window = int(args["swa_window"] or 0) if has_swa else 0
    swa_destination_bases = torch.empty_like(args["prompt_offsets"]) if has_swa else None

    def capacity_offsets(count):
        return torch.arange(0, (request_count + 1) * count, count, dtype=torch.int32, device=device)

    args.setdefault("dense_move_offsets", capacity_offsets(keep_count + tail))
    args.setdefault("swa_move_offsets", capacity_offsets(swa_window + tail) if has_swa else None)
    if has_draft:
        args.setdefault("draft_move_offsets", capacity_offsets(keep_count + draft_tail))
    params_list = [
        build_compaction_params(
            dict(
                layer_pools=args["layer_pools"],
                dense_layers=args["dense_layers"],
                swa_layers=args["swa_layers"],
                swa_window=args["swa_window"],
                layer_pool_ids=args["layer_pool_ids"],
            ),
            block_offsets=args["kv_block_offsets"],
            kept_ordinals=kept.reshape(-1, keep_count),
            source_lengths=args["valid_sequence_lengths"],
            dense_destination_bases=args["prompt_offsets"],
            dense_move_offsets=args["dense_move_offsets"],
            protected_tail_capacity=tail,
            swa_move_offsets=args["swa_move_offsets"],
            swa_destination_bases=swa_destination_bases,
        )
    ]
    if has_draft:
        params_list.append(
            build_compaction_params(
                dict(
                    layer_pools=args["draft_layer_pools"],
                    dense_layers=args["draft_layers"],
                    swa_layers=[],
                    layer_pool_ids=args["draft_layer_pool_ids"],
                ),
                block_offsets=args["draft_kv_block_offsets"],
                kept_ordinals=kept.reshape(-1, keep_count),
                source_lengths=args["valid_sequence_lengths"],
                dense_destination_bases=args["prompt_offsets"],
                dense_move_offsets=args["draft_move_offsets"],
                protected_tail_capacity=draft_tail,
            )
        )
    params = tuple(params_list)
    # Opaque plans plus a test-side mirror of the caller-owned construction
    # inputs (production binds the same values as manager attributes); the
    # standalone helpers here need the move-offset rows and SWA staging back.
    return dict(
        params=params,
        prompt_offsets=args["prompt_offsets"],
        request_count=request_count,
        decode_keep_count=keep_count,
        protected_tail_capacity=tail,
        draft_protected_tail_capacity=draft_tail if has_draft else 0,
        dense_move_offsets=args["dense_move_offsets"],
        swa_move_offsets=args["swa_move_offsets"],
        draft_move_offsets=args["draft_move_offsets"] if has_draft else None,
        has_swa=has_swa,
        swa_window=swa_window,
        swa_destination_bases=swa_destination_bases,
        swa_rebase_delta=keep_count - swa_window,
    )


def run_compaction(compaction):
    """Replica of the round's move stage in production order: SWA
    destination rebase, then ``compact`` loops the opaque params (each packs
    its decision rows into move sources and fires its native moves)."""
    from tensorrt_llm._torch.kv_cache_compression.compaction import compact

    if compaction["swa_destination_bases"] is not None:
        torch.add(
            compaction["prompt_offsets"],
            compaction["swa_rebase_delta"],
            out=compaction["swa_destination_bases"],
        )
    compact(compaction["params"], compaction["request_count"])
