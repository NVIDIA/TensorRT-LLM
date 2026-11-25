# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for prompt-embedding validation, identity and chunk placement."""

import dataclasses
import hashlib
import pickle
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _augment_tokens_with_prompt_embeds,
)
from tensorrt_llm.inputs.prompt_embeds import (
    PROMPT_EMBEDS_IDENTITY_VERSION,
    PromptEmbeds,
    build_prompt_embeds,
    multimodal_item_spans,
    multimodal_run_spans,
    prebuild_prompt_embeds,
    prompt_embeds_block_key,
    prompt_embeds_cache_salt,
    remap_prompt_embeds,
    value_range,
)

pytestmark = pytest.mark.cpu_only

HIDDEN = 8


def _rows(n: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Return ``n`` distinct embedding rows."""
    return torch.arange(n * HIDDEN, dtype=torch.float32).reshape(n, HIDDEN).to(dtype)


@pytest.mark.parametrize(
    "ids, mask, scheduled, positions",
    [
        (None, None, [0, 0, 0], [0, 1, 2]),
        ([7, 8, 9], None, [7, 8, 9], [0, 1, 2]),
        ([1, 2, 3, 4], [False, True, True, False], [1, 2, 3, 4], [1, 2]),
        ([1, 2, 3, 4], torch.tensor([False, True, True, False]), [1, 2, 3, 4], [1, 2]),
        ([1, 2, 3], torch.tensor([False, True, False]).to_sparse(), [1, 2, 3], [1]),
    ],
    ids=["no_ids", "no_mask", "mask", "mask_tensor", "mask_sparse"],
)
def test_build_positions(
    ids: list[int] | None,
    mask: list[bool] | torch.Tensor | None,
    scheduled: list[int],
    positions: list[int],
) -> None:
    out_ids, pe = build_prompt_embeds(ids, _rows(len(positions)), mask)
    assert out_ids == scheduled
    assert pe.positions.tolist() == positions
    assert pe.hidden_size == HIDDEN


@pytest.mark.parametrize(
    "ids, embeds, mask, error, match",
    [
        ([1, 2, 3], _rows(2), None, ValueError, "rows for"),
        ([1, 2, 3], _rows(2), [True, False], ValueError, "length"),
        (
            [1, 2, 3],
            _rows(1),
            torch.sparse_coo_tensor([[1]], [True], (1 << 40,)),
            ValueError,
            "length",
        ),
        ([1, 2, 3], _rows(1), [True, True, False], ValueError, "selects"),
        ([1, 2, 3], _rows(2), [1, 1, 0], TypeError, "only bools"),
        ([1, 2, 3], _rows(2), torch.tensor([1, 1, 0]), TypeError, "torch.bool"),
        (None, _rows(2), [True, True], ValueError, "requires prompt_token_ids"),
        ([1, 2], _rows(2).to(torch.int32), None, TypeError, "dtype"),
        ([1, 2], _rows(2)[None], None, ValueError, "shape"),
        ([1, 2], [[0.0] * HIDDEN] * 2, None, TypeError, "torch.Tensor"),
        ([], torch.empty(0, HIDDEN), None, ValueError, "at least one row"),
        (
            [1, 2],
            _rows(2).index_fill(1, torch.tensor([3]), float("nan")),
            None,
            ValueError,
            "non-finite",
        ),
    ],
)
def test_rejects_malformed(
    ids: list[int] | None,
    embeds: object,
    mask: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        build_prompt_embeds(ids, embeds, mask)


def test_snapshot_is_owned() -> None:
    rows = _rows(2)
    _, pe = build_prompt_embeds([1, 2], rows)
    digest = pe.digest
    rows.fill_(123)
    assert pe.embeds[0, 1].item() == 1.0
    assert pe.digest == digest
    _, again = build_prompt_embeds([1, 2], _rows(2))
    assert again.digest == digest


def test_digest_covers_values_positions_dtype_and_shape() -> None:
    _, base = build_prompt_embeds([1, 2, 3], _rows(2), [True, True, False])
    _, moved = build_prompt_embeds([1, 2, 3], _rows(2), [True, False, True])
    changed_rows = _rows(2)
    changed_rows[1, 0] = -1
    _, changed = build_prompt_embeds([1, 2, 3], changed_rows, [True, True, False])
    _, bf16 = build_prompt_embeds([1, 2, 3], _rows(2, torch.bfloat16), [True, True, False])
    # Same bytes as two fp16 rows, read as one fp32 row.
    _, alias = build_prompt_embeds(None, _rows(2).view(torch.float32).reshape(1, HIDDEN).clone())
    _, fp16 = build_prompt_embeds(None, _rows(2))
    digests = {base.digest, moved.digest, changed.digest, bf16.digest, alias.digest}
    assert len(digests) == 5 and fp16.digest != alias.digest


def test_cache_salt() -> None:
    _, pe = build_prompt_embeds([1, 2], _rows(2))
    salt = prompt_embeds_cache_salt(None, pe.digest)
    assert salt == prompt_embeds_cache_salt(None, pe.digest)
    assert salt != prompt_embeds_cache_salt("user", pe.digest)
    assert prompt_embeds_cache_salt("user", pe.digest) != prompt_embeds_cache_salt(
        "other", pe.digest
    )
    assert len(prompt_embeds_cache_salt("x" * 256, pe.digest).encode()) <= 256


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 16])
def test_chunk_rows_cover_every_position_once(chunk: int) -> None:
    mask = [False, True, True, False, True, False, False, True, True, True]
    _, pe = build_prompt_embeds(list(range(10)), _rows(6), mask)
    seen = []
    for begin in range(0, 10, chunk):
        end = min(begin + chunk, 10)
        lo, hi = pe.chunk_rows(begin, end)
        for row in range(lo, hi):
            position = int(pe.positions[row])
            assert begin <= position < end
            seen.append(row)
    assert seen == list(range(6))
    assert [p for p in range(10) if pe.covers(p)] == [1, 2, 4, 7, 8, 9]


def test_copies_without_rows() -> None:
    _, pe = build_prompt_embeds([1, 2], _rows(2, torch.float16))
    ident, peer = pe.identity_only(), pe.for_peer_ranks()
    for copy in (ident, peer):
        assert copy.embeds is None
        assert (copy.digest, copy.row_digests) == (pe.digest, pe.row_digests)
        assert (copy.dtype, copy.hidden_size) == (torch.float16, pe.hidden_size)
        assert torch.equal(copy.positions, pe.positions)
    # Peer ranks still run the prefill; a disaggregated generation request cannot.
    assert (pe.prefillable, peer.prefillable, ident.prefillable) == (True, True, False)


def test_block_keys_cover_row_and_placeholder_only() -> None:
    rows = _rows(3)
    _, pe = build_prompt_embeds([1, 2, 3, 4], rows, [True, True, False, True])
    # Position is carried by the block chain, not the key.
    _, same = build_prompt_embeds([9, 9], rows[1:])
    assert (same.row_digest(0), same.row_digest(1)) == (pe.row_digest(1), pe.row_digest(2))
    assert len({pe.row_digest(i) for i in range(3)}) == 3
    _, bf16 = build_prompt_embeds([9], rows[:1].to(torch.bfloat16))
    assert bf16.row_digest(0) != pe.row_digest(0)
    # Models that still read ids at embedding rows must not share KV across ids.
    key = prompt_embeds_block_key(pe.row_digest(0), 1)
    assert len(key) == 32 and key != prompt_embeds_block_key(pe.row_digest(0), 2)


@pytest.mark.parametrize("start, end", [(0, 6), (0, 3), (2, 5), (4, 6)])
def test_block_reuse_keys(start: int, end: int) -> None:
    ids = [10, 11, 12, 13, 14, 15]
    mask = [False, True, False, True, True, False]
    _, pe = build_prompt_embeds(ids, _rows(3), mask)
    keys = _augment_tokens_with_prompt_embeds(
        list(ids[start:end]), pe, start, pe.chunk_rows(start, end)
    )
    expected = []
    for p in range(start, end):
        row = sum(mask[:p])
        expected.append(prompt_embeds_block_key(pe.row_digest(row), ids[p]) if mask[p] else ids[p])
    assert keys == expected


def test_remap_through_placeholder_expansion() -> None:
    # 50 = vision start, 51 = image placeholder (expanded to 3 rows), 52 = end.
    original = [1, 50, 51, 52, 2, 3]
    expanded = [1, 50, 51, 51, 51, 52, 2, 3]
    spans = [(2, 5)]
    _, pe = build_prompt_embeds(original, _rows(2), [True, False, False, False, False, True])
    moved = remap_prompt_embeds(pe, original, expanded, spans)
    assert moved.positions.tolist() == [0, 7]
    assert moved.row_digests == pe.row_digests
    assert moved.digest != pe.digest
    _, on_image = build_prompt_embeds(original, _rows(1), [False, False, True, False, False, False])
    with pytest.raises(ValueError, match="multimodal placeholder"):
        remap_prompt_embeds(on_image, original, expanded, spans)
    with pytest.raises(ValueError, match="did not survive"):
        remap_prompt_embeds(pe, original, [1, 50, 51, 51, 51, 52, 2, 9, 3], spans)
    # Llama4-style expansion: the placeholder becomes rows of another id.
    with pytest.raises(ValueError, match="different token ids"):
        remap_prompt_embeds(pe, original, [1, 50, 70, 70, 70, 52, 2, 3], spans)


def test_positions_stay_on_cpu_under_default_device() -> None:
    original, expanded = [1, 50, 51, 52, 2, 3], [1, 50, 51, 51, 51, 52, 2, 3]
    rows = _rows(2)
    with torch.device("meta"):
        _, pe = build_prompt_embeds(original, rows, [True, False, False, False, False, True])
        moved = remap_prompt_embeds(pe, original, expanded, [(2, 5)])
    assert pe.positions.device.type == moved.positions.device.type == "cpu"


# Video notation: 50/52 = vision start/end, 53 = video placeholder rows,
# 60/61 = timestamp text inserted before each frame.


def test_remap_through_framed_video_expansion() -> None:
    original = [1, 50, 53, 52, 2, 3]
    expanded = [1, 60, 50, 53, 53, 52, 61, 50, 53, 53, 52, 2, 3]
    _, pe = build_prompt_embeds(original, _rows(2), [True, False, False, False, True, False])
    assert remap_prompt_embeds(pe, original, expanded, [(3, 10)]).positions.tolist() == [0, 11]
    # Text after the last item must re-tokenize to itself.
    with pytest.raises(ValueError, match="did not survive"):
        remap_prompt_embeds(pe, original, expanded[:-1] + [9, 3], [(3, 10)])


def test_remap_text_merged_with_framing() -> None:
    # Qwen3-VL: "?" before the video merges with the timestamp's "<" into one token (62).
    original = [1, 2, 50, 53, 52, 3]
    expanded = [1, 62, 60, 50, 53, 53, 52, 3]
    _, pe = build_prompt_embeds(original, _rows(2), [True] + [False] * 4 + [True])
    assert remap_prompt_embeds(pe, original, expanded, [(4, 6)]).positions.tolist() == [0, 7]
    _, merged = build_prompt_embeds(original, _rows(1), [False, True] + [False] * 4)
    with pytest.raises(ValueError, match="merged"):
        remap_prompt_embeds(merged, original, expanded, [(4, 6)])
    # The framing (62, 2) holds the text id 2: matching from the end could land on it.
    repeated = [1, 62, 2, 50, 53, 53, 52, 3]
    with pytest.raises(ValueError, match="ambiguous"):
        remap_prompt_embeds(merged, original, repeated, [(4, 6)])
    # "</task>" before a video: ">" (2) merges into "><" (62) and the timestamp ends in ">".
    original = [1, 7, 2, 50, 53, 52, 3]
    closing = [1, 7, 62, 63, 2, 50, 53, 53, 52, 3]
    _, tag = build_prompt_embeds(original, _rows(1), [False, False, True] + [False] * 4)
    with pytest.raises(ValueError, match="ambiguous"):
        remap_prompt_embeds(tag, original, closing, [(6, 8)])
    # Framing that merges without growing the gap: "7" and "<" become 62, the timestamp ends in 2.
    same_length = [1, 62, 2, 50, 53, 53, 52, 3]
    with pytest.raises(ValueError, match="ambiguous"):
        remap_prompt_embeds(tag, original, same_length, [(4, 6)])


def test_remap_text_after_video_matching_timestamp() -> None:
    original = [1, 50, 53, 52, 61, 2]
    expanded = [1, 60, 50, 53, 53, 52, 61, 50, 53, 53, 52, 61, 2]
    _, pe = build_prompt_embeds(original, _rows(1), [False] * 4 + [True, False])
    assert remap_prompt_embeds(pe, original, expanded, [(3, 10)]).positions.tolist() == [11]


def test_item_spans_from_runs_feed_remap() -> None:
    # Two videos, two frame runs each (rows 3-4, 8-9 and 14-15, 19-20).
    runs = SimpleNamespace(
        multimodal_item_run_cu_offsets=[0, 2, 4],
        multimodal_run_positions=[3, 8, 14, 19],
        multimodal_run_lengths=[2, 2, 2, 2],
        multimodal_positions=[3, 14],
        multimodal_lengths=[4, 4],
    )
    spans = multimodal_item_spans(runs)
    assert spans == [(3, 10), (14, 21)]
    original = [1, 50, 53, 52, 61, 50, 53, 52, 2]
    expanded = [1, 60, 50, 53, 53, 52, 61, 50, 53, 53, 52, 61,
                60, 50, 53, 53, 52, 61, 50, 53, 53, 52, 2]  # fmt: skip
    # The text between the videos (61) equals a timestamp token inside each video.
    _, pe = build_prompt_embeds(
        original, _rows(3), [True] + [False] * 3 + [True] + [False] * 3 + [True]
    )
    assert remap_prompt_embeds(pe, original, expanded, spans).positions.tolist() == [0, 11, 22]

    # Contiguous fallback: one image of three rows.
    contiguous = SimpleNamespace(
        multimodal_item_run_cu_offsets=None,
        multimodal_run_positions=None,
        multimodal_run_lengths=None,
        multimodal_positions=[2],
        multimodal_lengths=[3],
    )
    assert multimodal_item_spans(contiguous) == [(2, 5)]


def test_remap_keeps_already_expanded_ids() -> None:
    # A disaggregated encoder handoff schedules the ids it received.
    ids = [1, 50, 51, 51, 52, 2]
    _, pe = build_prompt_embeds(ids, _rows(1), [False] * 5 + [True])
    assert remap_prompt_embeds(pe, ids, list(ids), [(2, 4)]) is pe
    _, on_image = build_prompt_embeds(ids, _rows(1), [False, False, False, True, False, False])
    with pytest.raises(ValueError, match="multimodal placeholder"):
        remap_prompt_embeds(on_image, ids, list(ids), [(2, 4)])

    # Timestamp text between two frames of one video is not a multimodal row.
    video = [1, 60, 50, 53, 53, 52, 61, 50, 53, 53, 52, 2]
    runs = SimpleNamespace(
        multimodal_item_run_cu_offsets=[0, 2],
        multimodal_run_positions=[3, 8],
        multimodal_run_lengths=[2, 2],
    )
    spans = (multimodal_item_spans(runs), multimodal_run_spans(runs))
    _, between = build_prompt_embeds(video, _rows(1), [p == 6 for p in range(len(video))])
    assert remap_prompt_embeds(between, video, list(video), *spans) is between
    _, on_frame = build_prompt_embeds(video, _rows(1), [p == 8 for p in range(len(video))])
    with pytest.raises(ValueError, match="multimodal placeholder"):
        remap_prompt_embeds(on_frame, video, list(video), *spans)


def _split_prompt_embeds(inputs: Dict[str, Any]) -> Any:
    """Call ``BaseLLM._split_prompt_embeds`` on a minimal decoder-only stand-in."""
    from tensorrt_llm.llmapi.llm import BaseLLM

    llm = SimpleNamespace(
        _PROMPT_EMBEDS_MULTIMODAL_KEYS=BaseLLM._PROMPT_EMBEDS_MULTIMODAL_KEYS,
        _PROMPT_EMBEDS_ALLOWED_KEYS=BaseLLM._PROMPT_EMBEDS_ALLOWED_KEYS,
        _normalize_token_ids=BaseLLM._normalize_token_ids,
        _is_encoder_decoder_model=lambda: False,
        args=SimpleNamespace(backend="pytorch", mm_encoder_only=False),
        _encode_only=False,
    )
    return BaseLLM._split_prompt_embeds(llm, inputs)


def test_llm_takes_prebuilt_snapshot() -> None:
    """The LLM API takes a prebuilt snapshot as is and refuses one that does not fit its prompt."""
    ids, snapshot = prebuild_prompt_embeds(
        [5, 0, 0, 6], torch.ones(2, 4), [False, True, True, False]
    )
    prompt, taken = _split_prompt_embeds({"prompt_token_ids": ids, "prompt_embeds": snapshot})
    assert taken is snapshot and prompt["prompt_token_ids"] == ids
    # A snapshot built for a caller to keep is not trusted: only tensors are.
    _, kept = build_prompt_embeds(ids, torch.ones(2, 4), [False, True, True, False])
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        _split_prompt_embeds({"prompt_token_ids": ids, "prompt_embeds": kept})
    bad = [
        {"prompt_token_ids": ids[:2], "prompt_embeds": snapshot},
        {"prompt_token_ids": ids, "prompt_embeds": snapshot.identity_only()},
        {
            "prompt_token_ids": ids,
            "prompt_embeds": dataclasses.replace(snapshot, positions=torch.tensor([2, 1])),
        },
        {
            "prompt_token_ids": ids,
            "prompt_embeds": dataclasses.replace(snapshot, positions=torch.tensor([-1, 1])),
        },
        {
            "prompt_token_ids": ids,
            "prompt_embeds": dataclasses.replace(snapshot, embeds=torch.ones(3, 4)),
        },
    ]
    for inputs in bad:
        with pytest.raises(ValueError, match="PrebuiltPromptEmbeds needs"):
            _split_prompt_embeds(inputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("index", [0, 4097, 64 * 256 - 1])
def test_rejects_non_finite_anywhere(dtype: torch.dtype, value: float, index: int) -> None:
    # Large enough for the vectorized reduction paths, with the bad value at either end or inside.
    rows = torch.randn(64, 256).to(dtype)
    rows.view(-1)[index] = value
    with pytest.raises(ValueError, match="non-finite"):
        build_prompt_embeds(None, rows)


def test_value_range() -> None:
    assert value_range(torch.tensor([[1.0, -70000.0], [3.0, 5.0]])) == (-70000.0, 5.0)


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_pickle_round_trip(protocol: int, dtype: torch.dtype) -> None:
    _, pe = build_prompt_embeds([1, 2, 3, 4], _rows(3, dtype), [True, False, True, True])
    for snapshot in (pe, pe.for_peer_ranks(), pe.identity_only()):
        loaded = pickle.loads(pickle.dumps(snapshot, protocol))
        assert type(loaded) is PromptEmbeds
        for field in dataclasses.fields(PromptEmbeds):
            a, b = getattr(snapshot, field.name), getattr(loaded, field.name)
            if isinstance(a, torch.Tensor):
                assert a.dtype == b.dtype and torch.equal(a, b)
            else:
                assert a == b
    loaded = pickle.loads(pickle.dumps(pe, protocol))
    # The worker may write into the rows it owns (e.g. pinned staging reuses them).
    loaded.embeds.add_(1)
    assert loaded.embeds.is_contiguous()


@pytest.mark.parametrize("num_rows", [3, 1201])
def test_row_digests_are_sha256_per_row(num_rows: int) -> None:
    rows = torch.randn(num_rows, 4096).to(torch.float16)
    _, pe = build_prompt_embeds(None, rows)
    prefix = f"prompt_embeds.v{PROMPT_EMBEDS_IDENTITY_VERSION}|torch.float16|4096|".encode()
    for row in (0, num_rows // 2, num_rows - 1):
        expected = hashlib.sha256(prefix + rows[row].view(torch.uint8).numpy().tobytes()).digest()
        assert pe.row_digest(row) == expected
    assert len(pe.row_digests) == 32 * num_rows
