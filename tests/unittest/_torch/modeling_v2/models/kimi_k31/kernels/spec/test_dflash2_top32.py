# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 top-32 kernels against torch.topk, bit for bit: single-CTA and chunked rows, BF16
and FP32, ties, specials, the interleaved pair output and the merge of gathered pairs."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.dflash2_top32 import (  # noqa: E501
    merge_top32,
    top32,
    torch_topk_sorts_stably,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() not in (103, 107),
    reason="the K3.1 DFlash2 top-32 kernels are certified on sm_103 and sm_107",
)


def _bits(t):
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def _make(rows, width, dtype, kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "ties":
        x = 0.25 * torch.randint(-3, 6, (rows, width), generator=g, device="cuda").float()
    elif kind == "hot":
        x = torch.randn(rows, width, generator=g, device="cuda")
        hot = torch.randint(0, width, (rows, 60), generator=g, device="cuda")
        burst = 0.25 * torch.randint(0, 6, (rows, 60), generator=g, device="cuda").float() + 30
        x.scatter_(1, hot, burst)
    elif kind == "equal":
        # Every value tied: more survivors of the floor than the shared candidate buffer.
        x = torch.full((rows, width), 10.0, device="cuda")
        x[:, ::97] = 12.0
    else:
        x = 10.0 + 3.0 * torch.randn(rows, width, generator=g, device="cuda")
    if kind == "special":
        for col, value in ((5, "nan"), (17, "inf"), (7, "-inf"), (3, "0.0"), (11, "-0.0")):
            x[:, col] = float(value)
        x[:, width - 2] = float("nan")
        x[:, width - 1] = 50.0
    return x.to(dtype)


@pytest.mark.parametrize("width", [163840, 81920, 40960, 20480, 10240, 5120, 4096, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [4, 7, 56, 448])
@pytest.mark.parametrize("kind", ["randn", "ties", "hot", "special", "equal"])
def test_top32_is_torch_topk(width, dtype, rows, kind) -> None:
    if width >= 81920 and rows == 448 and kind != "hot":
        pytest.skip("covered by the narrower widths; keeps the sweep short")
    x = _make(rows, width, dtype, kind, seed=width + rows)
    values, ids = top32(x)
    ref = torch.topk(x, 32, dim=-1)
    assert torch.equal(ids, ref.indices)
    assert torch.equal(_bits(values), _bits(ref.values))


@pytest.mark.parametrize("width", [81920, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_pairs_are_the_exchange_layout(width, dtype) -> None:
    x = _make(56, width, dtype, "special", seed=3)
    offset = 7 * width
    got = top32(x, id_offset=offset, pairs=True)
    ref = torch.topk(x, 32, dim=-1)
    expected = torch.stack([(ref.indices + offset).float(), ref.values.float()], -1).flatten(-2)
    assert torch.equal(_bits(got), _bits(expected))


def test_stable_shape_follows_its_rule() -> None:
    """The one shape torch 2.14 sorts stably on sm_103 (contiguous FP32 [>=152, 256]): the kernel
    returns its ties in index order there and torch's network order below it. torch's own tie order
    for that shape depends on the GPU (sm_107 keeps the network), so there the selection and the
    values are compared with torch, and the order with the rule."""
    for rows in (151, 152, 448):
        x = _make(rows, 256, torch.float32, "ties", seed=rows)
        stable = torch_topk_sorts_stably(rows, 256, torch.float32, contiguous=True)
        assert stable == (rows >= 152)
        values, ids = top32(x)
        ref = torch.topk(x, 32, dim=-1)
        assert torch.equal(_bits(values), _bits(ref.values))
        assert torch.equal(ids.sort(dim=-1).values, ref.indices.sort(dim=-1).values)
        if stable:
            tied = values[:, 1:] == values[:, :-1]
            assert bool((ids[:, 1:] > ids[:, :-1])[tied].all()), "ties not in index order"
        else:
            assert torch.equal(ids, ref.indices)


@pytest.mark.parametrize("tp", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("rows", [4, 56, 151, 152, 448])
@pytest.mark.parametrize("kind", ["randn", "ties", "special"])
def test_merge_is_torch_topk_of_the_value_column(tp, rows, kind) -> None:
    g = torch.Generator(device="cuda").manual_seed(tp * 7 + rows)
    shard = 163840 // tp
    ids = torch.stack(
        [torch.randperm(shard, generator=g, device="cuda")[:32] + r * shard for r in range(tp)]
    )
    ids = ids[:, None].expand(tp, rows, 32)
    vals = _make(tp * rows, 32, torch.float32, kind, seed=tp + rows).view(tp, rows, 32)
    if kind == "ties":
        vals = vals.sort(-1, descending=True).values
    gathered = torch.cat(list(torch.stack([ids.float(), vals], -1).flatten(-2)), dim=-1)
    gathered = gathered.contiguous()
    values, cids = merge_top32(gathered)
    unary, sel = torch.topk(gathered[..., 1::2], 32, dim=-1)
    assert torch.equal(cids, gathered[..., 0::2].gather(-1, sel).long())
    assert torch.equal(_bits(values), _bits(unary))
