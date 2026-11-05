# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeviceCopyBuffer.update(..., changed_span=True) through a site in the plan's changed_span: a changed merged 2-D
upload stages only each row's changed span."""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import device_copy
from tensorrt_llm._torch.pyexecutor.device_copy import (
    CopyPlan,
    CopySiteEnum,
    DeviceCopier,
    DeviceCopyBuffer,
    Direction,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

ROWS, WIDTH = 4, 1024


class _Site(CopySiteEnum):
    TABLE = ("test_changed_span.table", Direction.H2D)


@pytest.fixture(autouse=True)
def _no_check_mode(monkeypatch):
    monkeypatch.setattr(device_copy, "CHECK_MODE", False)


def _copier(changed_span: bool = True) -> DeviceCopier:
    plan = CopyPlan(
        merge={_Site.TABLE},
        changed_span={_Site.TABLE} if changed_span else set(),
        parallel_scatter=True,
    )
    return DeviceCopier(plan, torch.device("cuda", torch.cuda.current_device()))


@pytest.fixture
def pack():
    return _copier()


def _record(buf: torch.Tensor) -> DeviceCopyBuffer:
    return DeviceCopyBuffer(buf, site=_Site.TABLE)


def _table(seed, rows=ROWS, width=WIDTH):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 1 << 20, (rows, width), generator=generator, dtype=torch.int32
    ).pin_memory()


def _upload(copier, record, dst, src):
    """One step: open the pack, upload through the record, issue the pack; returns the staged copies."""
    with device_copy.active(copier):
        device_copy.begin()
        copied = record.update(src, view=dst, changed_span=True)
        staged = list(copier._pack.h2d)
        device_copy.flush()
    torch.cuda.synchronize()
    return copied, staged


@pytest.mark.parametrize("changed_span", [False, True])
def test_device_rows_equal_the_source_after_every_step(changed_span):
    """Bitwise: whatever is staged, the device view holds the source after each step."""
    pack = _copier(changed_span)
    buf = torch.full((8, 2 * WIDTH), -7, dtype=torch.int32, device="cuda")
    cache = _record(buf)
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
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(dst)
    src = _table(2)
    _upload(pack, cache, dst, src)

    src = src.clone().pin_memory()
    src[2, 517] += 1
    copied, staged = _upload(pack, cache, dst, src)

    assert copied is True
    assert staged == [(dst[2, 517].data_ptr(), staged[0][1], 4)]
    assert torch.equal(dst.cpu(), src)


def test_each_changed_row_stages_its_first_to_last_changed_column(pack):
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(dst)
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
    buf = torch.full((ROWS, WIDTH), -1, dtype=torch.int32, device="cuda")
    cache = _record(buf)
    src = _table(4)
    _upload(pack, cache, buf[:, : WIDTH // 2], src[:, : WIDTH // 2].contiguous().pin_memory())

    _, staged = _upload(
        pack, cache, buf[:, : WIDTH // 2 + 3], src[:, : WIDTH // 2 + 3].contiguous().pin_memory()
    )

    assert [nbytes for _, _, nbytes in staged] == [3 * 4] * ROWS
    assert torch.equal(buf[:, : WIDTH // 2 + 3].cpu(), src[:, : WIDTH // 2 + 3])


def test_a_mostly_changed_table_is_staged_whole(pack):
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(dst)
    _upload(pack, cache, dst, _table(5))

    src = _table(6)
    _, staged = _upload(pack, cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]
    assert torch.equal(dst.cpu(), src)


def test_a_site_outside_the_plans_changed_span_stages_the_whole_view():
    pack = _copier(changed_span=False)
    assert not CopyPlan().changed_span
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(dst)
    src = _table(7)
    _upload(pack, cache, dst, src)

    src = src.clone().pin_memory()
    src[1, 1] += 1
    _, staged = _upload(pack, cache, dst, src)

    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]


def test_an_unmerged_copy_copies_the_whole_view(pack, monkeypatch):
    """No open pack: one eager copy of the view, not one per row span."""
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(dst)
    src = _table(8)
    with device_copy.active(pack):
        cache.update(src, view=dst, changed_span=True)
        src = src.clone().pin_memory()
        src[1, 1] += 1
        copies = []
        real = device_copy.copy
        monkeypatch.setattr(
            device_copy, "copy", lambda *a, **k: copies.append(tuple(a[1].shape)) or real(*a, **k)
        )
        assert cache.update(src, view=dst, changed_span=True) is True
    torch.cuda.synchronize()
    assert copies == [(ROWS, WIDTH)]
    assert torch.equal(dst.cpu(), src)


def test_a_view_with_another_row_stride_is_staged_whole(pack):
    """The shadow describes other device bytes: nothing of it may be trusted."""
    buf = torch.zeros((2 * ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(buf)
    src = _table(9)
    _upload(pack, cache, buf[:ROWS], src)

    _, staged = _upload(pack, cache, buf[::2], src)

    assert [nbytes for _, _, nbytes in staged] == [WIDTH * 4] * ROWS
    assert torch.equal(buf[::2].cpu(), src)


def test_a_discarded_span_upload_drops_the_record(pack):
    """A merged span copy whose pack is discarded never lands: the record is dropped, so the next step sends all."""
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(dst)
    src = _table(10)
    _upload(pack, cache, dst, src)
    changed = src.clone().pin_memory()
    changed[0, 3] += 1
    with device_copy.active(pack):
        device_copy.begin()
        cache.update(changed, view=dst, changed_span=True)
        device_copy.discard()
    assert cache.host_values is None
    _, staged = _upload(pack, cache, dst, changed)
    assert [nbytes for _, _, nbytes in staged] == [ROWS * WIDTH * 4]
    assert torch.equal(dst.cpu(), changed)


def test_a_barrier_site_stages_the_whole_view_with_one_barrier():
    """Per-row spans through a flush_after site would each flush on their own: the whole view goes instead."""
    plan = CopyPlan(
        merge={_Site.TABLE},
        flush_after={_Site.TABLE},
        changed_span={_Site.TABLE},
        parallel_scatter=True,
    )
    copier = DeviceCopier(plan, torch.device("cuda", torch.cuda.current_device()))
    dst = torch.zeros((ROWS, WIDTH), dtype=torch.int32, device="cuda")
    cache = _record(dst)
    src = _table(11)
    with device_copy.active(copier):
        device_copy.begin()
        cache.update(src, view=dst, changed_span=True)
        changed = src.clone().pin_memory()
        changed[1, 7] += 1
        segments_before = copier._pack.seg
        cache.update(changed, view=dst, changed_span=True)
        # One more segment (the barrier after the whole view), not one per span.
        assert copier._pack.seg > segments_before
        assert not copier._pack.h2d
        device_copy.flush()
    torch.cuda.synchronize()
    assert torch.equal(dst.cpu(), changed)
