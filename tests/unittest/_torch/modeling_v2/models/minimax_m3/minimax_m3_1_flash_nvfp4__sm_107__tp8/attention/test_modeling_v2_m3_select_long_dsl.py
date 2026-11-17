# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the tp8 (Rubin) m3_select_long_dsl entry: the long-row top-16 select from one 384-thread CTA per row,
against the reference and bit for bit against the 1024-thread m3_select_long on every cell."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.attention.m3_select_long import (  # noqa: E501
    TOPK,
    m3_select_long,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.attention.m3_select_long_dsl import (  # noqa: E501
    m3_select_long_dsl as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 TP8 target runs on SM107 only", allow_module_level=True)

# Valid-block counts that land a row on each of the kernel's paths: <= 16 (all picked), the 4-, 11- and 24-key
# widths and their edges, past the row (clamped), and zero (a CUDA-graph padding row).
_MIXED_VALID = (0, 5, 16, 17, 200, 1536, 1537, 4224, 4225, 8193, 9000, 1, 300, 6000, 129, 3000)


def _scores(spec, g):
    nb, tq = spec["num_blocks"], spec["total_q"]
    s = torch.randn(spec["num_kv_heads"], nb, tq, generator=g, device="cuda", dtype=torch.float32)
    kind = spec["scores"]
    if kind == "ties":
        # Even tokens: a flat row with ten winners, so thousands of keys tie at the 16th-largest score (the
        # whole-row re-rank). Odd tokens: scores on a coarse grid, so ties are broken by block index.
        s[:, :, 0::2] = 0.25
        winners = torch.randint(0, nb, (spec["num_kv_heads"], 10, tq), generator=g, device="cuda")
        s[:, :, 0::2].scatter_(1, winners[:, :, 0::2], 3.0)
        s[:, :, 1::2] = (s[:, :, 1::2] * 2).round() / 2
    elif kind == "ties1500":
        s = torch.full_like(s, -1.0)
        s[:, :1500, :] = 0.25
        winners = torch.randint(
            1500, nb, (spec["num_kv_heads"], 10, tq), generator=g, device="cuda"
        )
        s.scatter_(1, winners, 3.0)
    elif kind == "nonfinite":
        s[:, 7::50, :] = float("inf")
        s[:, 11::70, :] = float("nan")
        s[:, 3::40, :] = -0.0
        s[:, 4::40, :] = 0.0
        s[:, :, 1::2] = float("-inf")
        s[:, 0:5, 1::2] = 1.0
    return s


def _valid(spec, g):
    tq, nb = spec["total_q"], spec["num_blocks"]
    if spec["valid"] == "mixed":
        vals = [_MIXED_VALID[i % len(_MIXED_VALID)] for i in range(tq)]
    else:
        vals = (nb - torch.randint(0, 12, (tq,), generator=g, device="cuda")).tolist()
        vals[0] = nb
    return torch.tensor(vals, dtype=torch.int32, device="cuda")


def _case(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = _scores(spec, g)
    nvb = _valid(spec, g)
    out = torch.empty(
        spec["num_kv_heads"], spec["total_q"], TOPK, dtype=torch.int32, device="cuda"
    ).permute(1, 0, 2)
    return scores, nvb, out, spec["init_blocks"], spec["local_blocks"], False


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    args = _case(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    out = op(*args)
    ref = op.reference(*args)
    assert out.dtype is torch.int32
    assert torch.equal(out, ref), "selected blocks differ from the reference"
    # Every row, mass ties included, is the 1024-thread kernel's answer.
    scores, nvb, _, init_blocks, local_blocks, _ = args
    long_out = torch.empty_like(out)
    m3_select_long(scores, nvb, long_out, init_blocks, local_blocks, False)
    assert torch.equal(out, long_out), "selected blocks differ from m3_select_long"


@pytest.mark.parametrize("layout", ["kv_blocks_q", "kv_q_blocks"])
@pytest.mark.parametrize("valid", [17, 1294, 2524, 3907, 9216])
def test_layouts_and_widths_match_the_1024_thread_select(layout, valid) -> None:
    """Both score layouts the scorer can hand over, at a valid count in each key width."""
    g = torch.Generator(device="cuda").manual_seed(valid)
    nb = max(valid, 129)
    s = torch.randn(1, nb, 8, generator=g, device="cuda")
    if layout == "kv_q_blocks":
        s = s.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    nvb = torch.tensor([valid - (t % 2) for t in range(8)], dtype=torch.int32, device="cuda")
    out = torch.empty(1, 8, TOPK, dtype=torch.int32, device="cuda").permute(1, 0, 2)
    ref = torch.empty_like(out)
    op(s, nvb, out, 0, 1, False)
    m3_select_long(s, nvb, ref, 0, 1, False)
    assert torch.equal(out, ref)


def test_op_refuses_a_row_past_its_width() -> None:
    scores = torch.zeros(1, 9217, 1, device="cuda")
    nvb = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.empty(1, 1, TOPK, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="at most 9216"):
        op(scores, nvb, out, 0, 1, False)
