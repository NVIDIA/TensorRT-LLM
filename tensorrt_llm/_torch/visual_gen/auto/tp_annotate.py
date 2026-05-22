# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotate Diffusers `nn.Linear`s with TP roles (pre-export).

For TP we need to know which Linears are attention Q/K/V projections (output-
split — TensorParallelMode.COLUMN) vs output projections / MLP-down (input-
split with all-reduce — TensorParallelMode.ROW). The auto path's structural
fusion passes already detect these roles in the captured FX graph (e.g.,
`auto/fusion.py:_fuse_same_input_linears` finds Q/K/V via "same input + feed
attention reshape"), but `replace_linear_with_trtllm` runs *before* capture
on the original `nn.Module` tree — so we need a pre-capture annotator.

Strategy: walk the module tree, find container classes whose names indicate
attention or feed-forward role (`*Attention*`, `*Attn*`, `*FeedForward*`,
`*FFN*`, `*MLP*`), and tag their direct-child `nn.Linear`s by attribute
name (Diffusers naming is highly stable for these: `to_q/k/v`, `to_out`,
`add_q_proj`, `to_qkv_mlp_proj`, `linear_in/out`, ...).

Why this isn't fragile: every DiT-family Diffusers transformer follows the
same `Attention(...).to_q/.to_out` and `FeedForward(...).linear_in/.linear_out`
conventions — they're enforced by Diffusers' `AttentionModuleMixin`. Adding
a new family means at most a few extra attribute names in the table below.

The annotator also **patches `attn.heads` to `H // tp_size`** on each attention
module. Diffusers' attention reshape is `query.unflatten(-1, (attn.heads, -1))`
— with the patched count, the unflatten produces `(B, S, H/P, head_dim)` on
each rank when the TP-sharded Linear emits `(B, S, hidden/P)`. No FX-graph
rewrite of the reshape is needed.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch.nn as nn

from tensorrt_llm.logger import logger

# Tagged via `module._tp_role`. The keys correspond to TensorParallelMode:
#   "qkv", "ff_in"  → COLUMN (output-split, no collective)
#   "out_proj", "ff_out" → ROW (input-split, all-reduce after)
_QKV_NAMES = frozenset(
    {
        "to_q",
        "to_k",
        "to_v",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "qkv_proj",
        "to_qkv",
        "add_qkv_proj",
        "to_qkv_mlp_proj",  # FLUX.2 single-stream fused
    }
)
_OUT_NAMES = frozenset(
    {
        "to_out",
        "to_add_out",
        "out_proj",
    }
)
_FF_IN_NAMES = frozenset(
    {
        "linear_in",
        "linear_1",
        "fc1",
        "gate_proj",
        "up_proj",
        "proj_in",
    }
)
_FF_OUT_NAMES = frozenset(
    {
        "linear_out",
        "linear_2",
        "fc2",
        "down_proj",
    }
)


def _is_attention_module(module: nn.Module) -> bool:
    name = type(module).__name__
    return "Attention" in name or name.endswith("Attn")


def _is_ffn_module(module: nn.Module) -> bool:
    name = type(module).__name__
    upper = name.upper()
    return (
        "FEEDFORWARD" in upper
        or "FFN" in upper
        or (upper.endswith("MLP") and "FORMER" not in upper)
    )


def _tag_linear(module: nn.Linear, role: str, counts: defaultdict) -> None:
    module._tp_role = role
    counts[role] += 1


def _tag_attention_children(attn: nn.Module, counts: defaultdict) -> None:
    """Walk one attention module's direct children, tag Q/K/V/out by name."""
    for child_name, child in attn.named_children():
        if isinstance(child, nn.Linear):
            if child_name in _QKV_NAMES:
                _tag_linear(child, "qkv", counts)
            elif child_name in _OUT_NAMES:
                _tag_linear(child, "out_proj", counts)
        elif isinstance(child, nn.ModuleList):
            # to_out is sometimes ModuleList([Linear, Dropout])
            if child_name in _OUT_NAMES:
                for sub in child:
                    if isinstance(sub, nn.Linear):
                        _tag_linear(sub, "out_proj", counts)


def _tag_ffn_children(ffn: nn.Module, counts: defaultdict) -> None:
    for child_name, child in ffn.named_children():
        if not isinstance(child, nn.Linear):
            continue
        if child_name in _FF_IN_NAMES:
            _tag_linear(child, "ff_in", counts)
        elif child_name in _FF_OUT_NAMES:
            _tag_linear(child, "ff_out", counts)


def annotate_tp_roles(transformer: nn.Module, tp_size: int) -> dict[str, Any]:
    """Tag every Linear in attention/FFN containers with a TP role and patch
    `attn.heads = num_heads // tp_size` on each attention module.

    Returns a summary dict with role counts and patched-attention count.
    Untagged Linears (embedders, modulation projections, output projection of
    the transformer, etc.) stay replicated — no TP, no collective.
    """
    if tp_size <= 1:
        return {"qkv": 0, "out_proj": 0, "ff_in": 0, "ff_out": 0, "attn_patched": 0}

    counts: defaultdict = defaultdict(int)
    n_attn_patched = 0
    for name, module in transformer.named_modules():
        if _is_attention_module(module):
            _tag_attention_children(module, counts)
            # Patch the head count so `unflatten(-1, (attn.heads, -1))` produces
            # the sharded head count at capture time.
            if hasattr(module, "heads"):
                if module.heads % tp_size != 0:
                    raise ValueError(
                        f"TP annotate: {name}.heads={module.heads} not divisible by "
                        f"tp_size={tp_size}"
                    )
                module.heads = module.heads // tp_size
                n_attn_patched += 1
        elif _is_ffn_module(module):
            _tag_ffn_children(module, counts)
    summary = dict(counts)
    summary["attn_patched"] = n_attn_patched
    logger.info(f"VisGen-Auto TP annotate: {dict(summary)} (tp_size={tp_size})")
    return summary
