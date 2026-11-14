# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NORM_L2PF, family ``minimax_m3/ops/add_rmsnorm`` on CPU: the band rule (``l2pf_ranges``: 65..128 rows, the
first quarter of each range, 16-byte multiples, at most two ranges, nothing without the op's prefetch arguments) and
``add_rmsnorm_mxfp8``'s hand-over of the ranges to the op. The op is a recorder; the bitwise checks are the GPU tests.
CPU only."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import (
    add_rmsnorm as native,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def with_prefetch(monkeypatch):
    monkeypatch.setattr(native, "takes_prefetch", lambda: True)


def _weight():
    return torch.zeros(9856, 640, dtype=torch.uint8)  # 6,307,840 bytes


def _scale():
    return torch.zeros(1000, dtype=torch.uint8)


@pytest.mark.parametrize(
    "rows, taken",
    [(1, False), (64, False), (65, True), (88, True), (128, True), (129, False), (384, False)],
)
def test_the_band_is_65_to_128_rows(with_prefetch, rows, taken):
    weight, scale = _weight(), _scale()
    ranges = native.l2pf_ranges(rows, weight, scale)
    if not taken:
        assert ranges is None
        return
    assert len(ranges) == 2 and ranges[0][0] is weight and ranges[1][0] is scale
    assert [n for _, n in ranges] == [6307840 // 4, (1000 // 4) & ~15]
    assert all(n % 16 == 0 for _, n in ranges)


def test_no_prefetch_without_the_ops_arguments(monkeypatch):
    monkeypatch.setattr(native, "takes_prefetch", lambda: False)
    assert native.l2pf_ranges(88, _weight(), _scale()) is None


def test_one_or_two_ranges_only(with_prefetch):
    assert len(native.l2pf_ranges(88, _weight())) == 1
    assert native.l2pf_ranges(88) is None
    assert native.l2pf_ranges(88, _weight(), _scale(), _scale()) is None


def test_the_band_constants():
    assert (native.L2PF_MIN_ROWS, native.L2PF_MAX_ROWS, native.L2PF_DIVISOR) == (65, 128, 4)


@pytest.fixture
def op_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "takes", lambda *a: True)
    monkeypatch.setattr(native, "add_rmsnorm_op", lambda: lambda *args: calls.append(args))
    return calls


def test_the_mxfp8_form_hands_the_ranges_to_the_op(op_calls):
    x, r, w, q, sf = (torch.zeros(1) for _ in range(5))
    weight, scale = _weight(), _scale()
    native.add_rmsnorm_mxfp8(
        x, r, w, 1e-6, q, sf, enable_pdl=True, prefetch=((weight, 32), (scale, 16))
    )
    native.add_rmsnorm_mxfp8(x, r, w, 1e-6, q, sf, enable_pdl=False, prefetch=((weight, 48),))
    native.add_rmsnorm_mxfp8(x, r, w, 1e-6, q, sf, enable_pdl=True)
    native.add_rmsnorm_mxfp8(x, r, w, 1e-6, q, sf, enable_pdl=True, prefetch=None)
    first, second, third, fourth = op_calls
    assert first[7:] == (True, weight, 32, scale, 16) and first[6] is None
    assert second[7:] == (False, weight, 48, None, 0)
    assert len(third) == 8 and third[7] is True and len(fourth) == 8


def test_the_cute_fallback_ignores_the_ranges(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "takes", lambda *a: False)
    monkeypatch.setattr(
        native,
        "_cute",
        lambda: type(
            "C", (), {"add_rmsnorm_mxfp8": staticmethod(lambda *a, **kw: calls.append((a, kw)))}
        ),
    )
    native.add_rmsnorm_mxfp8(1, 2, 3, 1e-6, 4, 5, enable_pdl=True, prefetch=((_weight(), 32),))
    ((args, kwargs),) = calls
    assert args == (1, 2, 3, 1e-6, 4, 5) and kwargs == {"enable_pdl": True}
