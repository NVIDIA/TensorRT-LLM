# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for prompt-embedding validation, identity and chunk placement."""

import pickle

import pytest
import torch

from tensorrt_llm.inputs.prompt_embeds import (
    PromptEmbeds,
    build_prompt_embeds,
    prompt_embeds_cache_salt,
)

pytestmark = pytest.mark.cpu_only

HIDDEN = 8


def _rows(n: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Return ``n`` distinct embedding rows."""
    return torch.arange(n * HIDDEN, dtype=torch.float32).reshape(n, HIDDEN).to(dtype)


def test_whole_prompt_without_ids() -> None:
    ids, pe = build_prompt_embeds(None, _rows(5))
    assert ids == [0] * 5
    assert pe.positions.tolist() == [0, 1, 2, 3, 4]
    assert pe.hidden_size == HIDDEN
    assert pe.num_rows == 5


def test_whole_prompt_with_ids_and_no_mask() -> None:
    ids, pe = build_prompt_embeds([7, 8, 9], _rows(3))
    assert ids == [7, 8, 9]
    assert pe.positions.tolist() == [0, 1, 2]


def test_interleaved_mask() -> None:
    mask = [False, True, True, False, True, False]
    ids, pe = build_prompt_embeds([1, 2, 3, 4, 5, 6], _rows(3), mask)
    assert ids == [1, 2, 3, 4, 5, 6]
    assert pe.positions.tolist() == [1, 2, 4]
    ids_t, pe_t = build_prompt_embeds([1, 2, 3, 4, 5, 6], _rows(3), torch.tensor(mask))
    assert pe_t.digest == pe.digest


@pytest.mark.parametrize(
    "ids, embeds, mask, error",
    [
        ([1, 2, 3], _rows(2), None, ValueError),
        ([1, 2, 3], _rows(2), [True, False], ValueError),
        ([1, 2, 3], _rows(1), [True, True, False], ValueError),
        ([1, 2, 3], _rows(2), [1, 1, 0], TypeError),
        ([1, 2, 3], _rows(2), torch.tensor([1, 1, 0]), TypeError),
        (None, _rows(2), [True, True], ValueError),
        ([1, 2], _rows(2).to(torch.int32), None, TypeError),
        ([1, 2], _rows(2)[None], None, ValueError),
        ([1, 2], [[0.0] * HIDDEN] * 2, None, TypeError),
        ([], torch.empty(0, HIDDEN), None, ValueError),
    ],
)
def test_rejects_malformed(ids, embeds, mask, error) -> None:
    with pytest.raises(error):
        build_prompt_embeds(ids, embeds, mask)


def test_rejects_non_finite() -> None:
    rows = _rows(2)
    rows[1, 3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_prompt_embeds([1, 2], rows)


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
    digests = {base.digest, moved.digest, changed.digest, bf16.digest}
    assert len(digests) == 4


def test_digest_separates_byte_aliases() -> None:
    # Two fp16 rows and one fp32 row of the same byte length and content.
    raw = torch.arange(2 * HIDDEN, dtype=torch.float16).reshape(2, HIDDEN)
    as_fp32 = raw.view(torch.float32).reshape(1, HIDDEN)
    _, a = build_prompt_embeds(None, raw)
    _, b = build_prompt_embeds(None, as_fp32.clone())
    assert a.digest != b.digest


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


def test_identity_only() -> None:
    _, pe = build_prompt_embeds([1, 2], _rows(2, torch.float32))
    ident = pe.identity_only()
    assert ident.embeds is None
    assert ident.digest == pe.digest
    assert ident.row_digests == pe.row_digests
    assert torch.equal(ident.positions, pe.positions)


def test_row_digests() -> None:
    rows = _rows(3)
    _, pe = build_prompt_embeds([1, 2, 3, 4], rows, [True, True, False, True])
    assert len(pe.row_digests) == 3 * 32
    # A row's key depends on its values and dtype only; the block chain
    # carries its position.
    _, same = build_prompt_embeds([9, 9], rows[1:])
    assert same.row_digest(0) == pe.row_digest(1)
    assert same.row_digest(1) == pe.row_digest(2)
    assert len({pe.row_digest(i) for i in range(3)}) == 3
    _, bf16 = build_prompt_embeds([9], rows[:1].to(torch.bfloat16))
    assert bf16.row_digest(0) != pe.row_digest(0)


@pytest.mark.parametrize("start, end", [(0, 6), (0, 3), (2, 5), (4, 6)])
def test_block_reuse_keys(start: int, end: int) -> None:
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
        _augment_tokens_with_prompt_embeds,
    )

    ids = [10, 11, 12, 13, 14, 15]
    mask = [False, True, False, True, True, False]
    _, pe = build_prompt_embeds(ids, _rows(3), mask)
    keys = _augment_tokens_with_prompt_embeds(list(ids[start:end]), pe, start, end)
    expected = []
    for p in range(start, end):
        expected.append(pe.row_digest(sum(mask[:p])) if mask[p] else ids[p])
    assert keys == expected


def test_pickle_round_trip() -> None:
    _, pe = build_prompt_embeds([1, 2, 3], _rows(2), [False, True, True])
    restored = pickle.loads(pickle.dumps(pe))
    assert isinstance(restored, PromptEmbeds)
    assert torch.equal(restored.embeds, pe.embeds)
    assert torch.equal(restored.positions, pe.positions)
    assert restored.digest == pe.digest
