# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert selection and ungated BF16 or block-scaled routed experts."""

import torch
from torch.nn import functional as F


def routing_weights(logits, bias, topk, scale):
    """Select on corrected logits and weight the selected experts by their unbiased scores."""
    ids = torch.topk(logits + bias, topk, dim=-1, sorted=False).indices
    weights = torch.sigmoid(logits).gather(1, ids)
    if topk > 1:
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    return weights * scale, ids.int()


def clamped_relu2(x, clamp):
    limited = (clamp * torch.tanh(x.float() / clamp)).to(x.dtype)
    return limited.relu().square()


def grouped_experts(x, expert_ids, scores, up, down, ep_rank, clamp):
    """Apply the routing weight before the down projection and sum slots in FP32."""
    num_local = up.shape[0]
    local = expert_ids.flatten() - ep_rank * num_local
    # Remote experts sort after the last local group. Keep a fixed number of rows
    # so graph replay can change every token's routing decision.
    local = torch.where((local >= 0) & (local < num_local), local, num_local)
    ordered, perm = local.sort(stable=True)
    groups = torch.arange(num_local, device=x.device, dtype=ordered.dtype)
    ends = torch.searchsorted(ordered, groups, right=True, out_int32=True)
    tokens = perm // expert_ids.shape[1]
    # grouped_mm permits an unused suffix. Retain one spare row even on EP1.
    rows = F.pad(x[tokens], (0, 0, 0, 1))
    hidden = F.grouped_mm(rows, up.transpose(1, 2), offs=ends)
    scale = F.pad(scores.flatten()[perm], (0, 1))
    hidden = (clamped_relu2(hidden, clamp).float() * scale[:, None]).to(x.dtype)
    result = F.grouped_mm(hidden, down.transpose(1, 2), offs=ends)[:-1]
    result = torch.where((ordered < num_local)[:, None], result, 0)
    slots = torch.empty_like(result).index_copy_(0, perm, result)
    return slots.view(x.shape[0], expert_ids.shape[1], x.shape[1]).float().sum(1)


def grouped_quantized_experts(x, expert_ids, scores, up, down, ep_rank, clamp):
    """Run two block-scaled projections around the slot-weighted expert activation."""
    ntokens, topk = expert_ids.shape
    if ntokens == 0:
        return x.float()
    num_local = up.weight.shape[0]
    local = expert_ids.flatten() - ep_rank * num_local
    local = torch.where((local >= 0) & (local < num_local), local, num_local)
    ordered, perm = local.sort(stable=True)
    boundaries = torch.searchsorted(
        ordered, torch.arange(num_local + 1, device=x.device, dtype=ordered.dtype)
    )
    counts = boundaries.diff().to(torch.int32)
    # Every token chooses distinct experts. No expert can have more than ntokens
    # assignments. Up to 1024 padded rows fit in the per-pair activation workspace.
    rows = min(ntokens, 128)
    # The NVFP4 GEMM requires a 16-byte aligned FP32 alpha pointer at the start
    # of each consecutive expert batch. Slice at multiples of four experts.
    per_batch = min(num_local, (1024 // rows // 4) * 4)
    # For large eager prefill a single count read gives the number of tiles in
    # each group. Graph replay must permit routes to change after capture.
    eager_counts = None
    if ntokens > rows and not torch.cuda.is_current_stream_capturing():
        eager_counts = counts.tolist()

    sink = ntokens * topk
    slots = x.new_zeros((sink + 1, x.shape[-1]))
    row = torch.arange(rows, device=x.device)
    for first in range(0, num_local, per_batch):
        last = min(first + per_batch, num_local)
        max_count = ntokens if eager_counts is None else max(eager_counts[first:last])
        for base in range(0, max_count, rows):
            mask = (counts[first:last] - base).clamp(0, rows)
            indices = (boundaries[first:last, None] + base + row).clamp_max(sink - 1)
            routed_slots = perm[indices]
            inputs = x[routed_slots // topk]
            hidden = up.apply_experts(inputs, mask, first, last)
            weighted = (
                clamped_relu2(hidden, clamp).float() * scores.flatten()[routed_slots, None]
            ).to(x.dtype)
            result = down.apply_experts(weighted.contiguous(), mask, first, last)
            # Each active slot has exactly one writer. All padding writes land on
            # the discarded final row. Non-local slots retain zero for the EP sum.
            dest = torch.where(row < mask[:, None], routed_slots, sink)
            slots.index_copy_(0, dest.flatten(), result.reshape(-1, x.shape[-1]))
    return slots[:sink].view(ntokens, topk, -1).float().sum(1)
