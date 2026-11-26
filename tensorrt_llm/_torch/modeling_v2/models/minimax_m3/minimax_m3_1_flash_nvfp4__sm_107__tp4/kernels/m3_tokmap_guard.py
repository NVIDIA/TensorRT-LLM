# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The token map of ``thop.trtllm_gen_generation_preprocess`` for one request with several tokens.

For multi-token generation the op's QKV preprocessing places every row by the workspace token map (``tokensInfo``,
(request, token) per row), but builds that map only when ``isBuildDecoderInfoKernelNeeded()`` holds, which at one
request with the default RoPE it does not. The op then applies whatever map the last multi-request call left in the
workspace: a row can land on another request's length and pages, and its RoPE position and K/V slot are wrong. This
module wraps the op so that, for one request and several tokens, the map the build would write, (0, t) for each row t,
is written into the op's own map slice of the workspace (the offset from the op's layout function) just before the
call. Every other call passes through unchanged. The rows' positions still come from the request's generation length,
exactly as with a freshly built map.

The write is a device copy from a map prebuilt at install (no allocation, no host sync), so it is captured into the
decode graphs with the call it precedes.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger

#: Argument names of ``thop.trtllm_gen_generation_preprocess`` in order (the nanobind binding).
_ARG_NAMES = (
    "qkv_input",
    "workspace",
    "sequence_lengths",
    "spec_decoding_generation_lengths",
    "spec_decoding_position_offsets",
    "kv_cache_block_offsets",
    "host_kv_cache_pool_pointers",
    "host_kv_cache_pool_mapping",
    "kv_scale_orig_quant",
    "kv_scale_quant_orig",
    "attention_output_orig_quant",
    "rotary_inv_freq",
    "rotary_cos_sin",
    "mrope_position_deltas",
    "layer_idx",
    "seq_offset",
    "num_heads",
    "num_kv_heads",
    "head_size",
    "tokens_per_block",
    "kv_cache_quant_mode",
    "max_attention_window_size",
    "cyclic_attention_window_size",
    "num_tokens",
    "batch_beam",
    "input_seq_length",
    "max_past_kv_length",
    "rotary_embedding_dim",
    "rotary_embedding_base",
    "rotary_embedding_scale_type",
    "rotary_embedding_scale",
    "rotary_embedding_max_positions",
    "position_embedding_type",
    "bmm1_scale",
    "bmm2_scale",
    "fp8_context_fmha",
    "predicted_tokens_per_seq",
    "attention_chunk_size",
    "multi_processor_count",
    "total_num_blocks",
    "kv_factor",
    "need_build_kv_cache_metadata",
    "cross_attention",
    "skip_fmha_workspace",
)
_INT2_BYTES = 8  # capacity: fixed: an element size

_original: Optional[Callable] = None
_max_rows = 0
_maps: Dict[torch.device, torch.Tensor] = {}
_offsets: Dict[Tuple, int] = {}


def _no_taken(rows: int) -> None:
    pass


def _no_missed(rows: int, reason: str) -> None:
    pass


_on_taken: Callable[[int], None] = _no_taken
_on_missed: Callable[[int, str], None] = _no_missed


def _indexed(device: torch.device) -> torch.device:
    """``device`` with its index (an unindexed ``cuda`` means the current device), the key ``_maps`` is built under."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _one_request_map(device: torch.device, rows: int) -> torch.Tensor:
    """The prebuilt (0, t) map's first ``rows`` rows on ``device``."""
    device = _indexed(device)
    m = _maps.get(device)
    if m is None or m.shape[0] < rows:
        reason = f"no {rows}-row token map was prebuilt on {device} (install() builds it before the warmup)"
        _on_missed(rows, reason)
        raise RuntimeError(f"m3_tokmap_guard: {reason}")
    return m[:rows]


def _map_offset(args: dict, rows: int) -> int:
    x = args["qkv_input"]
    skip = bool(args.get("skip_fmha_workspace", False))
    key = (
        x.dtype,
        rows,
        int(args["num_heads"]),
        int(args["head_size"]),
        int(args["rotary_embedding_dim"]),
        int(args["num_kv_heads"]),
        skip,
    )
    off = _offsets.get(key)
    if off is None:
        layout = thop.get_trtllm_gen_generation_workspace_layout(
            x.dtype, 1, rows, key[2], key[3], key[4], key[5], 0, False, skip
        )
        off, size = int(layout["tokens_info_offset"]), int(layout["tokens_info_size"])
        if off < 0 or off % 4 != 0 or size < _INT2_BYTES * rows:
            raise RuntimeError(
                f"m3_tokmap_guard: the op's generation layout has no {rows}-row token map "
                f"(offset {off}, {size} bytes): the layout changed under this guard"
            )
        _offsets[key] = off
    return off


def _guarded(*args, **kwargs):
    a = dict(zip(_ARG_NAMES, args))
    a.update(kwargs)
    rows = int(a["num_tokens"])
    if (
        a.get("spec_decoding_generation_lengths") is not None
        and int(a["predicted_tokens_per_seq"]) > 1
        and int(a["batch_beam"]) == 1
        and rows > 1
        and not bool(a.get("cross_attention", False))
    ):
        ws = a["workspace"]
        src = _one_request_map(ws.device, rows)
        off = _map_offset(a, rows)
        # view (never reshape: a copy would leave the op's own workspace unwritten); raises on a strided workspace.
        dst = ws.view(-1).view(torch.uint8)[off : off + _INT2_BYTES * rows]
        dst.view(torch.int32).view(rows, 2).copy_(src)
        _on_taken(rows)
    return _original(*args, **kwargs)


def install(
    device: torch.device,
    max_rows: int,
    on_taken: Callable[[int], None],
    on_missed: Callable[[int, str], None] = _no_missed,
) -> bool:
    """Wrap ``thop.trtllm_gen_generation_preprocess`` (once per process) and prebuild the (0, t) map for up to
    ``max_rows`` rows on ``device`` (a one-request call past it is ``on_missed`` and raises). Returns whether this
    call installed the wrapper."""
    global _original, _max_rows, _on_taken, _on_missed
    device = _indexed(device)
    if device not in _maps or _maps[device].shape[0] < max_rows:
        t = torch.arange(max_rows, dtype=torch.int32, device=device)
        _maps[device] = torch.stack([torch.zeros_like(t), t], dim=1).contiguous()
    _max_rows = max(_max_rows, max_rows)
    _on_taken = on_taken
    _on_missed = on_missed
    if _original is not None:
        return False
    _original = thop.trtllm_gen_generation_preprocess
    thop.trtllm_gen_generation_preprocess = _guarded
    logger.info(
        "[M3 prebuild] m3_tokmap_guard installed: one-request multi-token generations get their (0, t) token map "
        f"(up to {max_rows} rows)"
    )
    return True
