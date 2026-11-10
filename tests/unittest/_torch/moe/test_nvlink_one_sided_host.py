# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-only NVLinkOneSided tests: workspace layouts, sentinel row geometry and limits, settings
and the per-step transport choice. No GPU or MPI workers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
    SENTINEL_MAX_TOKENS_ENV,
    SYNC_PROTOCOL_ENV,
    NVLinkOneSided,
    NVLinkOneSidedConfigError,
    default_sentinel_max_tokens,
    get_sentinel_max_tokens,
    get_sync_protocol,
    sentinel_payload_format,
    sentinel_row,
)

pytestmark = pytest.mark.cpu_only


def test_workspace_layout() -> None:
    from tensorrt_llm.bindings import internal as _tllm_internal

    thop = _tllm_internal.thop

    def field(layout: torch.Tensor, name: str) -> int:
        return int(layout[getattr(thop, f"MOE_A2A_{name}")])

    ep_size, top_k, capacity, hidden_size = 8, 6, 128, 7168
    layout = NVLinkOneSided._make_workspace_layout(
        ep_size, top_k, capacity, hidden_size, torch.bfloat16, None, 0
    )
    dispatch_start = field(layout, "DISPATCH_PAYLOAD_OFFSET_INDEX")
    dispatch_end = dispatch_start + field(layout, "DISPATCH_PAYLOAD_SIZE_INDEX")
    combine_start = field(layout, "COMBINE_INPUT_OFFSET_INDEX")
    combine_end = combine_start + field(layout, "COMBINE_INPUT_SIZE_INDEX")
    total = field(layout, "WORKSPACE_SIZE_INDEX")
    assert field(layout, "TOPK_TARGET_INDICES_OFFSET_INDEX") < dispatch_start
    assert dispatch_end <= field(layout, "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX") < combine_start
    assert field(layout, "COMBINE_INPUT_SIZE_INDEX") == ep_size * capacity * hidden_size * 2
    assert dispatch_start % 256 == combine_start % 256 == total % 256 == 0
    assert combine_end == total

    # Routing metadata reserves the configured top-k, not the maximum supported top-k.
    routes_start = field(layout, "TOPK_TARGET_RANKS_OFFSET_INDEX")
    indices_start = field(layout, "TOPK_TARGET_INDICES_OFFSET_INDEX")
    assert indices_start - routes_start == capacity * top_k * 4
    with pytest.raises(RuntimeError, match="capacity"):
        torch.ops.trtllm.moe_a2a_get_workspace_layout(ep_size, capacity, top_k, -1, 256)
    with pytest.raises(RuntimeError, match="overflows"):
        torch.ops.trtllm.moe_a2a_get_workspace_layout(
            ep_size, capacity, top_k, (1 << 63) - 512, 1024
        )


@pytest.mark.parametrize("payload_format", ["BF16", "MXFP8", "FP8_BLOCK128", "NVFP4"])
def test_sentinel_workspace_layout(payload_format: str) -> None:
    from tensorrt_llm.bindings import internal as _tllm_internal

    thop = _tllm_internal.thop

    def field(layout: torch.Tensor, name: str) -> int:
        return int(layout[getattr(thop, f"MOE_A2A_{name}")])

    ep_size, top_k, capacity, hidden_size, sentinel_tokens = 8, 8, 256, 7168, 160
    base = NVLinkOneSided._make_workspace_layout(
        ep_size, top_k, capacity, hidden_size, torch.bfloat16, None, 0
    )
    for name in ("SENTINEL_MAX_NUM_TOKENS_INDEX", "SENTINEL_RECV_SIZE_INDEX"):
        assert field(base, name) == 0
    layout = base.clone()
    torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
        layout,
        sentinel_tokens,
        int(getattr(thop, f"MOE_A2A_SENTINEL_FORMAT_{payload_format}")),
        hidden_size,
    )
    # The other regions keep their offsets; the sentinel regions follow them in order.
    num_base_fields = int(thop.MOE_A2A_SENTINEL_MAX_NUM_TOKENS_INDEX)
    assert torch.equal(
        layout[: int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX)],
        base[: int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX)],
    )
    assert num_base_fields == int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX) + 1
    starts = [
        field(layout, f"SENTINEL_{name}_OFFSET_INDEX")
        for name in ("STATE", "LANDING", "COMBINE_LANDING", "RECV")
    ]
    assert field(base, "WORKSPACE_SIZE_INDEX") <= starts[0]
    assert starts == sorted(starts) and all(start % 256 == 0 for start in starts)
    row = sentinel_row(payload_format, hidden_size, top_k)
    rows = ep_size * sentinel_tokens
    # One bf16 partial row per (source rank, token), padded to the workspace alignment.
    assert starts[3] - starts[2] >= rows * hidden_size * 2
    assert field(layout, "SENTINEL_RECV_SIZE_INDEX") >= rows * (
        row.hidden_bytes + row.scale_bytes + 2 * top_k * 4
    )
    # The regions are the per-row bytes the default limit divides by, times the rows, plus
    # bounded padding: the round state (a cache line and one mask word per token slot), one
    # alignment per region and landing row group, and one cache line per receive buffer.
    alignment = int(thop.MOE_A2A_WORKSPACE_ALIGNMENT)
    padding_bound = (
        (128 + 8 * sentinel_tokens) + 8 * alignment + int(thop.MOE_A2A_MAX_PAYLOADS) * 128
    )
    sentinel_bytes = field(layout, "WORKSPACE_SIZE_INDEX") - starts[0]
    assert 0 <= sentinel_bytes - rows * row.bytes_per_row <= padding_bound
    assert field(layout, "WORKSPACE_SIZE_INDEX") == starts[3] + field(
        layout, "SENTINEL_RECV_SIZE_INDEX"
    )
    assert field(layout, "SENTINEL_MAX_NUM_TOKENS_INDEX") == sentinel_tokens
    assert field(layout, "SENTINEL_HIDDEN_SIZE_INDEX") == hidden_size

    # Only a layout fresh from moe_a2a_get_workspace_layout can be extended, once.
    with pytest.raises(RuntimeError, match="must be a layout from moe_a2a_get_workspace_layout"):
        torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
            layout, sentinel_tokens, int(thop.MOE_A2A_SENTINEL_FORMAT_BF16), hidden_size
        )
    with pytest.raises(RuntimeError, match="token capacity"):
        torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
            base.clone(), capacity + 1, int(thop.MOE_A2A_SENTINEL_FORMAT_BF16), hidden_size
        )


def test_sentinel_row_geometry() -> None:
    hidden_size, top_k = 7168, 8
    # (hidden-state bytes, scale bytes, dispatch blocks per token) per row.
    assert sentinel_row("BF16", hidden_size, top_k)[:3] == (2 * hidden_size, 0, 7)
    assert sentinel_row("MXFP8", hidden_size, top_k)[:3] == (hidden_size, hidden_size // 32, 4)
    assert sentinel_row("FP8_BLOCK128", hidden_size, top_k)[:3] == (
        hidden_size,
        hidden_size // 32,
        4,
    )
    assert sentinel_row("NVFP4", hidden_size, top_k)[:3] == (hidden_size // 2, hidden_size // 16, 2)
    # The payload format must split into whole 16-byte vectors.
    with pytest.raises(RuntimeError, match="hidden_size % 256"):
        sentinel_row("NVFP4", 2880, top_k)
    with pytest.raises(RuntimeError, match="hidden_size % 8"):
        sentinel_row("BF16", 4100, top_k)


@pytest.mark.parametrize(
    ("payload_format", "hidden_size", "top_k", "ep_size", "expected"),
    [
        # Limits FlashInfer's sentinel transport reports on GB300 (152 SMs).
        # One wave holds fewer than 159 tokens: the floor.
        pytest.param("BF16", 7168, 8, 8, 159, id="dsv3-bf16-ep8"),
        # The 64 MiB bound on the sentinel regions.
        pytest.param("NVFP4", 7168, 8, 8, 371, id="dsv3-nvfp4-ep8"),
        # The 64 MiB bound below the floor: the floor.
        pytest.param("NVFP4", 7168, 8, 64, 159, id="dsv3-nvfp4-ep64"),
        # The one-wave bound.
        pytest.param("MXFP8", 3584, 16, 8, 455, id="kimi-k3-mxfp8-ep8"),
    ],
)
def test_default_sentinel_max_tokens(
    payload_format: str, hidden_size: int, top_k: int, ep_size: int, expected: int
) -> None:
    row = sentinel_row(payload_format, hidden_size, top_k)
    assert default_sentinel_max_tokens(ep_size, 152, row) == expected


def test_sentinel_payload_format() -> None:
    from tensorrt_llm.quantization.mode import QuantAlgo

    assert sentinel_payload_format(QuantAlgo.NVFP4) == "NVFP4"
    assert sentinel_payload_format(QuantAlgo.NVFP4_AWQ) == "NVFP4"  # a calibration of NVFP4
    assert sentinel_payload_format(QuantAlgo.W4A8_MXFP4_MXFP8) == "MXFP8"
    assert sentinel_payload_format(QuantAlgo.MXFP8) == "MXFP8"
    # FP8 block-scale backends quantize after dispatch.
    assert sentinel_payload_format(QuantAlgo.FP8_BLOCK_SCALES) == "BF16"
    assert sentinel_payload_format(None) == "BF16"
    # FP8 per-tensor rows carry no block scales: no sentinel format.
    assert sentinel_payload_format(QuantAlgo.FP8) is None


def test_sync_protocol_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SYNC_PROTOCOL_ENV, raising=False)
    monkeypatch.delenv(SENTINEL_MAX_TOKENS_ENV, raising=False)
    assert get_sync_protocol() == "auto"
    assert get_sentinel_max_tokens() is None
    # The removed CFT pin keeps selecting fence for one release.
    monkeypatch.setenv("TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT", "0")
    assert get_sync_protocol() == "fence"
    monkeypatch.setenv(SYNC_PROTOCOL_ENV, " Sentinel ")
    assert get_sync_protocol() == "sentinel"  # an explicit protocol wins
    for name, value, parse in (
        (SYNC_PROTOCOL_ENV, "off", get_sync_protocol),
        (SENTINEL_MAX_TOKENS_ENV, "0", get_sentinel_max_tokens),
        (SENTINEL_MAX_TOKENS_ENV, "many", get_sentinel_max_tokens),
    ):
        monkeypatch.setenv(name, value)
        with pytest.raises(NVLinkOneSidedConfigError):
            parse()


# Dispatch payloads of each sentinel format: (hidden states, scales) of `tokens` tokens.
def _format_payload(
    payload_format: str, tokens: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if payload_format == "BF16":
        return torch.empty((tokens, hidden_size), dtype=torch.bfloat16), None
    if payload_format == "MXFP8":  # e4m3 values, UE8M0 scale bytes per 32
        return (
            torch.empty((tokens, hidden_size), dtype=torch.uint8),
            torch.empty((tokens, hidden_size // 32), dtype=torch.uint8),
        )
    if payload_format == "FP8_BLOCK128":  # e4m3 values, fp32 scales per 128
        return (
            torch.empty((tokens, hidden_size), dtype=torch.uint8),
            torch.empty((tokens, hidden_size // 128), dtype=torch.float32),
        )
    assert payload_format == "NVFP4"  # two e2m1 values per byte, e4m3 scale bytes per 16
    return (
        torch.empty((tokens, hidden_size // 2), dtype=torch.uint8),
        torch.empty((tokens, hidden_size // 16), dtype=torch.uint8),
    )


@pytest.mark.parametrize("payload_format", ["BF16", "MXFP8", "FP8_BLOCK128", "NVFP4"])
def test_sentinel_step_selection(payload_format: str) -> None:
    hidden_size, top_k, tokens = 7168, 8, 4
    comm = SimpleNamespace(
        _sentinel_row=sentinel_row(payload_format, hidden_size, top_k),
        _sentinel_format=payload_format,
        sentinel_max_tokens=160,
        top_k=top_k,
    )
    hidden, sf = _format_payload(payload_format, tokens, hidden_size)
    weights = torch.empty((tokens, top_k), dtype=torch.float32)

    def carries(
        runtime_max_tokens_per_rank: int,
        hidden_states: torch.Tensor,
        hidden_states_sf: torch.Tensor | None,
        token_final_scales: torch.Tensor | None,
        eplb_local_stats: torch.Tensor | None,
    ) -> bool:
        return NVLinkOneSided.use_sentinel_for_dispatch(
            comm,
            runtime_max_tokens_per_rank,
            hidden_states,
            hidden_states_sf,
            token_final_scales,
            eplb_local_stats,
        )

    assert carries(160, hidden, sf, weights, None)
    assert carries(160, hidden, sf, None, None)
    assert not carries(161, hidden, sf, weights, None)  # above the token limit
    assert not carries(1, hidden, sf, weights, torch.zeros(4))  # EPLB statistics
    # Payloads of every other format do not match this one's rows.
    # (MXFP8 and FP8_BLOCK128 rows have the same bytes; a layer dispatches only its own format.)
    same_rows = {"MXFP8": "FP8_BLOCK128", "FP8_BLOCK128": "MXFP8"}.get(payload_format)
    for other in ("BF16", "MXFP8", "FP8_BLOCK128", "NVFP4"):
        if other not in (payload_format, same_rows):
            other_hidden, other_sf = _format_payload(other, tokens, hidden_size)
            assert not carries(1, other_hidden, other_sf, weights, None), other
    if payload_format == "BF16":
        # Same row bytes, but not bf16 values: the sentinel rewrite would not preserve them.
        assert not carries(1, hidden.view(torch.float16), None, weights, None)
    else:
        assert not carries(1, hidden, None, weights, None)  # no scales
    assert carries(1, hidden, sf, weights.bfloat16(), None)  # bf16 weights (TRTLLM-Gen)
    assert not carries(1, hidden, sf, weights.half(), None)  # other 2-byte weights
    comm._sentinel_row, comm.sentinel_max_tokens = None, 0
    assert not carries(1, hidden, sf, weights, None)
