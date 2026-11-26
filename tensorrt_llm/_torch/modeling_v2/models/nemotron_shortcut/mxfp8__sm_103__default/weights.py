# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load logical E4M3 and UE8M0 tensor windows before interleaving rank-local scales."""

import torch

from ..manifest import (
    block_columns,
    check_checkpoint_keys,
    copy_checked,
    dynamic_placements,
    expert_ids,
    load_unquantized,
    output_head_entry,
    pad_expert_weight,
)


def _load_projection(weights, entries, target, scales, shape):
    rows, cols = shape
    padded_n, padded_k = target.shape[-2:]
    target[..., rows:, :].zero_()
    target[..., :, cols:].zero_()
    linear_scales = torch.zeros((padded_n, padded_k // 32), dtype=torch.uint8, device=target.device)
    if target.ndim == 3:
        # Stream one local expert at a time from the fused on-disk bank.
        ((key, (experts, *axes), _, _),) = entries
        windows = [(key, tuple(axes), None, None)]
        local_experts = expert_ids(experts)
    else:
        windows = entries
        local_experts = (None,)
    for local, expert in enumerate(local_experts):
        matrix = target if expert is None else target[local]
        prefix = () if expert is None else (expert,)
        for key, window, dest, _ in windows:
            source_rows = window or (slice(None),)
            dest_rows = dest[0] if dest is not None else slice(0, rows)
            copy_checked(matrix[dest_rows, :cols], weights[key][prefix + source_rows], key)
            scale_key = key + "_scale"
            copy_checked(
                linear_scales[dest_rows, : cols // 32],
                weights[scale_key][prefix + block_columns(source_rows, 32)],
                scale_key,
            )
        dest_scales = scales if expert is None else scales[local]
        dest_scales.copy_(torch.ops.trtllm.block_scale_interleave(linear_scales).reshape(-1))


@torch.no_grad()
def load(model, weights):
    core = model.model
    manifest = core.weight_manifest
    head = model.lm_head.weight
    if head.dtype != torch.bfloat16:
        raise ValueError("The MXFP8 text decoder requires a BF16 output head")
    head_entry = output_head_entry(core, head)
    needed = {"lm_head.weight"}
    for name, entries in manifest.items():
        for key, *_ in entries:
            needed.add(key)
            if name in core.mxfp8_shapes:
                needed.add(key + "_scale")
    check_checkpoint_keys(weights, needed, mtp_enabled=core.mtp_enabled)
    for name, entries in manifest.items():
        if name in core.mxfp8_shapes:
            _load_projection(
                weights, entries, core.w[name], core.w[name + "_scale"], core.mxfp8_shapes[name]
            )
        else:
            load_unquantized(core.w[name], entries, weights)
    for prefix, placement in dynamic_placements(core):
        host = {}
        for plane in ("up", "down"):
            name = f"{prefix}experts_{plane}"
            ((key, (_, *axes), _, _),) = manifest[name]
            tensors = weights[key][(slice(None), *axes)].cpu()
            host[plane] = pad_expert_weight(tensors, core.w[name].shape[-2:])
            scales = weights[key + "_scale"][(slice(None), *block_columns(axes, 32))].cpu()
            interleaved = torch.ops.trtllm.block_scale_interleave(scales.contiguous())
            host[plane + "_scale"] = interleaved.view(core.num_experts, -1)
        placement.bind_weights(core.w, prefix, host)
    load_unquantized(head, head_entry, weights)
    if head.is_cuda:
        torch.cuda.current_stream(head.device).synchronize()
