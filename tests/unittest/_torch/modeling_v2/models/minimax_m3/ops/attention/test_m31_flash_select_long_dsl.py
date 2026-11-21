# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_select_long_dsl entry of the MiniMax-M3.1-Flash sm_107 tp2 and tp8 targets: the long-row top-16
select from one 384-thread CTA per row, against the reference and bit for bit against the 1024-thread m3_select_long
the target pairs it with, on every cell. On tp2 also at the two KV heads per rank the target runs, plus the K7 rows."""

import importlib
import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402, F401 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

_PACKAGE = "tensorrt_llm._torch.modeling_v2.models.minimax_m3"
# Target -> the module of the 1024-thread m3_select_long it pairs the DSL select with.
_LONG_SELECT = {
    "minimax_m3_1_flash_nvfp4__sm_107__tp2": f"{_PACKAGE}.ops.attention.m3_select_long",
    "minimax_m3_1_flash_nvfp4__sm_107__tp8": (
        f"{_PACKAGE}.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.attention.m3_select_long"
    ),
}
_TP2 = "minimax_m3_1_flash_nvfp4__sm_107__tp2"
_TP8 = "minimax_m3_1_flash_nvfp4__sm_107__tp8"


def _entries(target: str):
    """The target's entry and its 1024-thread select, or None when this build does not carry the target."""
    try:
        op = importlib.import_module(
            f"{_PACKAGE}.{target}.ops.attention.m3_select_long_dsl"
        ).m3_select_long_dsl
        long = importlib.import_module(_LONG_SELECT[target])
    except NoTarget:
        return None
    return SimpleNamespace(op=op, m3_select_long=long.m3_select_long, topk=long.TOPK)


ENTRIES = {target: _entries(target) for target in _LONG_SELECT}


def _params(target: str, values, ids):
    entry = ENTRIES[target]
    if entry is None:
        return [pytest.param(None, marks=pytest.mark.skip(reason=f"{target} is not built"))]
    return [pytest.param(v, id=f"{target.rsplit('__', 1)[1]}-{i}") for v, i in zip(values, ids)]


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


def _case(spec, topk, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scores = _scores(spec, g)
    nvb = _valid(spec, g)
    out = torch.empty(
        spec["num_kv_heads"], spec["total_q"], topk, dtype=torch.int32, device="cuda"
    ).permute(1, 0, 2)
    return scores, nvb, out, spec["init_blocks"], spec["local_blocks"], False


def _check_against_both(entry, spec, seed, certify=None) -> None:
    args = _case(spec, entry.topk, seed)
    if certify is not None:
        certify(entry.op)
    out = entry.op(*args)
    assert out.dtype is torch.int32
    assert torch.equal(out, entry.op.reference(*args)), "selected blocks differ from the reference"
    # Every row, mass ties included, is the 1024-thread kernel's answer.
    scores, nvb, _, init_blocks, local_blocks, _ = args
    long_out = torch.empty_like(out)
    entry.m3_select_long(scores, nvb, long_out, init_blocks, local_blocks, False)
    assert torch.equal(out, long_out), "selected blocks differ from m3_select_long"


_CELLS = [
    p
    for target, entry in ENTRIES.items()
    for p in _params(
        target,
        [(target, c) for c in (entry.op.CELLS if entry else ())],
        [c.why[:44] for c in (entry.op.CELLS if entry else ())],
    )
]


@pytest.mark.parametrize("target_cell", _CELLS)
def test_certified_cells(target_cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    target, cell = target_cell
    _check_against_both(
        ENTRIES[target], cell.spec, zlib.crc32(cell.why.encode()) % 2**31, certifying
    )


def _tp2_specs():
    """The TP2 geometry the target runs this op at: two KV heads per rank (each head selects its own blocks), every
    cell's step again, and the K7 verify rows of c1..c4 over a 165521-token context (1294 blocks)."""
    entry = ENTRIES[_TP2]
    cells = entry.op.CELLS if entry else ()
    return [dict(c.spec, num_kv_heads=2) for c in cells] + [
        dict(
            num_blocks=1294,
            total_q=tq,
            valid="near_full",
            scores="random",
            num_kv_heads=2,
            init_blocks=0,
            local_blocks=1,
        )
        for tq in (8, 16, 24, 32)
    ]


_TP2_SPECS = _tp2_specs()


@pytest.mark.parametrize(
    "spec",
    _params(
        _TP2,
        _TP2_SPECS,
        [f"kv2_{s['num_blocks']}x{s['total_q']}_{s['scores']}" for s in _TP2_SPECS],
    ),
)
def test_two_kv_heads(spec) -> None:
    """At two KV heads: the reference's blocks and the 1024-thread kernel's, bit for bit, on every row."""
    _check_against_both(ENTRIES[_TP2], spec, zlib.crc32(repr(spec).encode()) % 2**31)


_LAYOUTS = [
    p
    for target in ENTRIES
    for p in _params(
        target,
        [(target, layout, valid) for layout in ("kv_blocks_q", "kv_q_blocks")
         for valid in (17, 1294, 2524, 3907, 9216)],
        [f"{layout}-{valid}" for layout in ("kv_blocks_q", "kv_q_blocks")
         for valid in (17, 1294, 2524, 3907, 9216)],
    )
]  # fmt: skip


@pytest.mark.parametrize("case", _LAYOUTS)
def test_layouts_and_widths_match_the_1024_thread_select(case) -> None:
    """Both score layouts the scorer can hand over, at a valid count in each key width."""
    target, layout, valid = case
    entry = ENTRIES[target]
    g = torch.Generator(device="cuda").manual_seed(valid)
    nb = max(valid, 129)
    s = torch.randn(1, nb, 8, generator=g, device="cuda")
    if layout == "kv_q_blocks":
        s = s.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    nvb = torch.tensor([valid - (t % 2) for t in range(8)], dtype=torch.int32, device="cuda")
    out = torch.empty(1, 8, entry.topk, dtype=torch.int32, device="cuda").permute(1, 0, 2)
    ref = torch.empty_like(out)
    entry.op(s, nvb, out, 0, 1, False)
    entry.m3_select_long(s, nvb, ref, 0, 1, False)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("target", _params(_TP8, [_TP8], ["width"]))
def test_op_refuses_a_row_past_its_width(target) -> None:
    entry = ENTRIES[target]
    scores = torch.zeros(1, 9217, 1, device="cuda")
    nvb = torch.ones(1, dtype=torch.int32, device="cuda")
    out = torch.empty(1, 1, entry.topk, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="at most 9216"):
        entry.op(scores, nvb, out, 0, 1, False)
