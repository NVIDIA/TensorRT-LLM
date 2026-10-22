# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test: `qsa_rollback_gather` / `qsa_rollback_restore` against torch's advanced indexing,
bitwise, over planes that are strided views of one per-page buffer (as an NVFP4 KV cache's
manager packs the side planes) and over contiguous planes."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_rollback_planes import (  # noqa: E501
    ROW,
    plane_table,
    qsa_rollback_gather,
    qsa_rollback_restore,
)

assert torch.cuda.is_available(), "qsa_rollback_planes requires a CUDA device"

DEV = "cuda"
TPB = 32
PAGES = 600


def _planes(count, strided, g):
    if strided:
        page = torch.zeros(PAGES, count * TPB * ROW * 2 + 384, dtype=torch.uint8, device=DEV)
        planes = [
            page[:, i * TPB * ROW * 2 : (i + 1) * TPB * ROW * 2]
            .view(torch.bfloat16)
            .view(PAGES, TPB, 1, ROW)
            for i in range(count)
        ]
    else:
        planes = [
            torch.empty(PAGES, TPB, 1, ROW, dtype=torch.bfloat16, device=DEV) for _ in range(count)
        ]
    for t in planes:
        t.copy_(torch.randn(t.shape, generator=g, device=DEV).to(torch.bfloat16))
    return planes


def _slots(rows, g):
    pages = torch.randperm(PAGES, generator=torch.Generator().manual_seed(rows))[:rows].to(DEV)
    within = torch.randint(0, TPB, (rows,), generator=g, device=DEV)
    return pages.to(torch.int64), within.to(torch.int64)


@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "shared_page"])
@pytest.mark.parametrize(
    "cell", qsa_rollback_gather.CELLS, ids=[c.why[:40] for c in qsa_rollback_gather.CELLS]
)
def test_gather_cells(cell, strided):
    g = torch.Generator(device=DEV).manual_seed(7)
    planes = _planes(cell.spec["planes"], strided, g)
    pages, within = _slots(cell.spec["rows"], g)
    ptrs, strides = plane_table(planes)
    got = qsa_rollback_gather(ptrs, strides, pages, within, len(planes))
    want = torch.stack([t[pages, within].reshape(-1, ROW) for t in planes])
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "shared_page"])
@pytest.mark.parametrize(
    "cell", qsa_rollback_restore.CELLS, ids=[c.why[:40] for c in qsa_rollback_restore.CELLS]
)
def test_restore_cells(cell, strided):
    g = torch.Generator(device=DEV).manual_seed(9)
    planes = _planes(cell.spec["planes"], strided, g)
    pages, within = _slots(cell.spec["rows"], g)
    ptrs, strides = plane_table(planes)
    old = qsa_rollback_gather(ptrs, strides, pages, within, len(planes))
    # The step overwrites every slot; the restore puts back the rejected ones only.
    for t in planes:
        t[pages, within] = torch.randn(t[pages, within].shape, generator=g, device=DEV).to(t.dtype)
    after_step = [t.clone() for t in planes]
    rejected = (torch.rand(pages.shape[0], generator=g, device=DEV) < 0.5).to(torch.int32)
    qsa_rollback_restore(ptrs, strides, pages, within, rejected, old)
    for i, (t, step) in enumerate(zip(planes, after_step)):
        want = step.clone()
        want[pages, within] = torch.where(
            rejected.bool()[:, None, None], old[i].view(-1, 1, ROW), step[pages, within]
        )
        assert torch.equal(t.view(torch.int16), want.view(torch.int16)), f"plane {i}"
