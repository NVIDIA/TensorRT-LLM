# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""H2DSkipCache(changed_span=True): a changed packed 2-D upload stages only each row's changed span."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import h2d_cache, step_inputs
from tensorrt_llm._torch.pyexecutor.h2d_cache import H2DSkipCache, copy_h2d_if_changed
from tensorrt_llm._torch.pyexecutor.step_inputs import StepInputPack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

ROWS, WIDTH = 4, 1024


@pytest.fixture
def pack():
    return StepInputPack(torch.device("cuda"), parallel_scatter=True)


def _table(seed, rows=ROWS, width=WIDTH):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 1 << 20, (rows, width), generator=generator, dtype=torch.int32
    ).pin_memory()


def _upload(pack, cache, dst, src):
    """One step: open the pack, route the copy through the cache, issue the pack; returns the staged copies."""
    with h2d_cache.active(cache), step_inputs.active(pack):
        step_inputs.begin()
        copied = copy_h2d_if_changed(dst, src, pack=True)
        staged = list(pack.h2d)
        step_inputs.flush()
    torch.cuda.synchronize()
    return copied, staged


@pytest.mark.parametrize("changed_span", [False, True])
def test_device_rows_equal_the_source_after_every_step(pack, changed_span):
    """Bitwise: whatever is staged, the device view holds the source after each step."""
    cache = H2DSkipCache(changed_span=changed_span)
    buf = torch.full((8, 2 * WIDTH), -7, dtype=torch.int32, device="cuda")
    dst = buf[:ROWS, :WIDTH]
    src = _table(0)
    generator = torch.Generator().manual_seed(1)
    for step in range(40):
        src = src.clone().pin_memory()
        # A request crosses into a new page: one entry of one row changes (sometimes several, in several rows).
        for _ in range(1 + step % 3):
            row = int(torch.randint(0, ROWS, (1,), generator=generator))
            col = int(torch.randint(0, WIDTH, (1,), generator=generator))
            src[row, col] = step
        _upload(pack, cache, dst, src)
        assert torch.equal(dst.cpu(), src)
    # Rows and columns outside the view were never written.
    assert bool((buf[ROWS:] == -7).all()) and bool((buf[:, WIDTH:] == -7).all())


def test_one_changed_entry_stages_one_element(pack):
    cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(2)
    _upload(pack, cache, dst, src)

    src = src.clone().pin_memory()
    src[2, 517] += 1
    copied, staged = _upload(pack, cache, dst, src)

    assert copied is True
    assert staged == [(dst[2, 517].data_ptr(), staged[0][1], 4)]
    assert torch.equal(dst.cpu(), src)


def test_each_changed_row_stages_its_first_to_last_changed_column(pack):
    cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(3)
    _upload(pack, cache, dst, src)

    src = src.clone().pin_memory()
    src[0, 10] += 1
    src[0, 20] += 1
    src[3, 1000] += 1
    _, staged = _upload(pack, cache, dst, src)

    assert [(ptr, nbytes) for ptr, _, nbytes in staged] == [
        (dst[0, 10].data_ptr(), 11 * 4),
        (dst[3, 1000].data_ptr(), 4),
    ]
    assert torch.equal(dst.cpu(), src)


def test_wider_view_uploads_the_columns_the_shadow_does_not_cover(pack):
    cache = H2DSkipCache(changed_span=True)
    buf = torch.full((ROWS, WIDTH), -1, dtype=torch.int32, device="cuda")
    src = _table(4)
    _upload(pack, cache, buf[:, : WIDTH // 2], src[:, : WIDTH // 2].contiguous().pin_memory())

    _, staged = _upload(
        pack, cache, buf[:, : WIDTH // 2 + 3], src[:, : WIDTH // 2 + 3].contiguous().pin_memory()
    )

    assert [nbytes for _, _, nbytes in staged] == [3 * 4] * ROWS
    assert torch.equal(buf[:, : WIDTH // 2 + 3].cpu(), src[:, : WIDTH // 2 + 3])


def test_a_mostly_changed_table_is_staged_whole(pack):
    cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    _upload(pack, cache, dst, _table(5))

    src = _table(6)
    _, staged = _upload(pack, cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]
    assert torch.equal(dst.cpu(), src)


def test_off_by_default_the_whole_view_is_staged(pack):
    cache = H2DSkipCache()
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(7)
    _upload(pack, cache, dst, src)

    src = src.clone().pin_memory()
    src[1, 1] += 1
    _, staged = _upload(pack, cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]


def test_an_unpacked_copy_copies_the_whole_view(pack):
    cache = H2DSkipCache(changed_span=True)
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(8)
    with h2d_cache.active(cache):
        copy_h2d_if_changed(dst, src)
        src = src.clone().pin_memory()
        src[1, 1] += 1
        assert copy_h2d_if_changed(dst, src) is True
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), src)


def test_a_view_with_another_row_stride_is_staged_whole(pack):
    """The shadow describes other device bytes: nothing of it may be trusted."""
    cache = H2DSkipCache(changed_span=True)
    buf = torch.zeros((2 * ROWS, WIDTH), dtype=torch.int32, device="cuda")
    src = _table(9)
    _upload(pack, cache, buf[:ROWS], src)

    _, staged = _upload(pack, cache, buf[::2], src)

    assert [nbytes for _, _, nbytes in staged] == [WIDTH * 4] * ROWS
    assert torch.equal(buf[::2].cpu(), src)
