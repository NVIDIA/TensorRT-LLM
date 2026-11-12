# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-map preprocessing using each target's live state and callback bindings."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, cast

import torch


class _PreprocessOps(Protocol):
    get_trtllm_gen_generation_workspace_layout: Callable[..., dict[str, int]]
    trtllm_gen_generation_preprocess: Callable[..., object]


class _LogSink(Protocol):
    def info(self, message: str) -> None: ...


class TokenMapTarget(Protocol):
    """State and callback fields owned by one SM107 target module."""

    _original: Optional[Callable[..., object]]
    _max_rows: int
    _maps: dict[torch.device, torch.Tensor]
    _offsets: dict[tuple, int]
    _on_taken: Callable[[int], None]
    _on_missed: Callable[[int, str], None]
    _ARG_NAMES: tuple[str, ...]
    _INT2_BYTES: int
    _indexed: Callable[[torch.device], torch.device]
    _one_request_map: Callable[[torch.device, int], torch.Tensor]
    _map_offset: Callable[[dict, int], int]
    _guarded: Callable[..., object]
    thop: _PreprocessOps
    logger: _LogSink


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

_INT2_BYTES = 8


def _indexed(device: torch.device) -> torch.device:
    """``device`` with its index (an unindexed ``cuda`` means the current device), the key ``_maps`` is built under."""
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _one_request_map(state: TokenMapTarget, /, device: torch.device, rows: int) -> torch.Tensor:
    """The prebuilt (0, t) map's first ``rows`` rows on ``device``."""
    device = state._indexed(device)
    m = state._maps.get(device)
    if m is None or m.shape[0] < rows:
        reason = f"no {rows}-row token map was prebuilt on {device} (install() builds it before the warmup)"
        state._on_missed(rows, reason)
        raise RuntimeError(f"m3_tokmap_guard: {reason}")
    return m[:rows]


def _map_offset(state: TokenMapTarget, /, args: dict, rows: int) -> int:
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
    off = state._offsets.get(key)
    if off is None:
        layout = state.thop.get_trtllm_gen_generation_workspace_layout(
            x.dtype, 1, rows, key[2], key[3], key[4], key[5], 0, False, skip
        )
        off, size = (int(layout["tokens_info_offset"]), int(layout["tokens_info_size"]))
        if off < 0 or off % 4 != 0 or size < state._INT2_BYTES * rows:
            raise RuntimeError(
                f"m3_tokmap_guard: the op's generation layout has no {rows}-row token map "
                f"(offset {off}, {size} bytes): the layout changed under this guard"
            )
        state._offsets[key] = off
    return off


def _guarded(state: TokenMapTarget, /, *args: Any, **kwargs: Any) -> object:
    a = dict(zip(state._ARG_NAMES, args))
    a.update(kwargs)
    rows = int(a["num_tokens"])
    if (
        a.get("spec_decoding_generation_lengths") is not None
        and int(a["predicted_tokens_per_seq"]) > 1
        and (int(a["batch_beam"]) == 1)
        and (rows > 1)
        and (not bool(a.get("cross_attention", False)))
    ):
        ws = a["workspace"]
        src = state._one_request_map(ws.device, rows)
        off = state._map_offset(a, rows)
        dst = ws.view(-1).view(torch.uint8)[off : off + state._INT2_BYTES * rows]
        dst.view(torch.int32).view(rows, 2).copy_(src)
        state._on_taken(rows)
    return cast(Callable[..., object], state._original)(*args, **kwargs)


def install(
    state: TokenMapTarget,
    /,
    device: torch.device,
    max_rows: int,
    on_taken: Callable[[int], None],
    on_missed: Callable[[int, str], None],
) -> bool:
    """Wrap ``thop.trtllm_gen_generation_preprocess`` (once per process) and prebuild the (0, t) map for up to
    ``max_rows`` rows on ``device`` (a one-request call past it is ``on_missed`` and raises). Returns whether this
    call installed the wrapper."""
    device = state._indexed(device)
    if device not in state._maps or state._maps[device].shape[0] < max_rows:
        t = torch.arange(max_rows, dtype=torch.int32, device=device)
        state._maps[device] = torch.stack([torch.zeros_like(t), t], dim=1).contiguous()
    state._max_rows = max(state._max_rows, max_rows)
    state._on_taken = on_taken
    state._on_missed = on_missed
    if state._original is not None:
        return False
    state._original = state.thop.trtllm_gen_generation_preprocess
    state.thop.trtllm_gen_generation_preprocess = state._guarded
    state.logger.info(
        "[M3 prebuild] m3_tokmap_guard installed: one-request multi-token generations get their (0, t) token map "
        f"(up to {max_rows} rows)"
    )
    return True
