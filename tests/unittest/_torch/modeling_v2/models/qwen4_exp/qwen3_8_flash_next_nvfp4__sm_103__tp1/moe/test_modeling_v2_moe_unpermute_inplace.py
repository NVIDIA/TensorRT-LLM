# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe_unpermute_inplace entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file lays a cell's FC2 rows out the way the
CuteDSL FC2 hands them over -- placed by moe_sort, every row the op must not read
filled with NaN, the output pre-filled with NaN -- and holds the claims no cell
carries: CUDA-graph replay at the largest captured row count, bitwise equal to
eager, and the contiguity guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.moe_unpermute_inplace import (  # noqa: E501
    moe_unpermute_inplace as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "moe_unpermute_inplace requires a CUDA device"

DEV = "cuda"
E, K, H = 512, 10, 2560
NAN = float("nan")

#: The largest decode capture: 32 requests at MTP verify width 4.
_CAPTURED_ROWS = 128


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise.

    `reference` is the computation the op performs, in its order and with its
    roundings, so no band is needed and none is given. The gate sees the
    accumulation order: the same sum taken in reverse slot order, with each
    product rounded before its add, as the float64 sum rounded once, or
    accumulated in bf16, differs in many elements and is rejected. A failure in a
    few isolated one-ulp elements points at the kernel's rounding (a lost
    multiply-add contraction, say); a wrong row or scale moves whole rows.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (tuple(out.shape), tuple(ref.shape))
    bits = {2: torch.int16, 4: torch.int32}[out.element_size()]
    differ = out.view(bits) != ref.view(bits)
    n = int(differ.sum())
    if n:
        first = torch.nonzero(differ)[0].tolist()
        raise AssertionError(
            f"{n} of {out.numel()} elements differ bitwise from the slot-order fp32 "
            f"sum; first at {first}: {out[tuple(first)].item()} vs "
            f"{ref[tuple(first)].item()}"
        )


def _spec(tokens, tile_tokens_dim=128, values="normal"):
    """The fields `_problem` reads; the same names as a cell's spec."""
    return dict(tokens=tokens, tile_tokens_dim=tile_tokens_dim, values=values)


def _ids(tokens: int, seed: int) -> torch.Tensor:
    """Expert ids, `[tokens, K]` int32: each row's top-10 of uniform scores."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    scores = torch.rand(tokens, E, device=DEV, generator=g)
    return scores.topk(K, dim=-1).indices.to(torch.int32)


def _scales(tokens: int, seed: int) -> torch.Tensor:
    """fp32 routing weights, a softmax over each row's ten slots."""
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    return torch.softmax(torch.randn(tokens, K, device=DEV, generator=g), dim=-1)


def _rows(tokens: int, seed: int, values: str = "normal") -> torch.Tensor:
    """The FC2 output row of each (token, slot) pair, `[tokens, K, H]` bf16."""
    g = torch.Generator(device=DEV).manual_seed(seed + 2)
    v = torch.randn(tokens, K, H, device=DEV, generator=g)
    if values == "wide":
        v = v * torch.exp2(torch.randint(-30, 31, v.shape, device=DEV, generator=g).float())
    elif values != "normal":
        raise ValueError(values)
    return v.to(torch.bfloat16)


def _sort(ids: torch.Tensor, scales: torch.Tensor, tile: int):
    """moe_sort as the CuteDSL MoE calls it: (expanded_idx_to_permuted_idx, padded rows)."""
    out = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales,
        num_experts=E,
        top_k=K,
        local_expert_offset=0,
        local_num_experts=E,
        tile_tokens_dim=tile,
    )
    return out[2], out[3].numel()


def _place(rows: torch.Tensor, e2p: torch.Tensor, num_rows: int) -> torch.Tensor:
    """The FC2 output as the op receives it: each live pair's row at its permuted
    index, NaN in every row the op has no business reading."""
    perm = torch.full((num_rows, H), NAN, dtype=rows.dtype, device=DEV)
    flat = e2p.reshape(-1)
    live = flat >= 0
    perm[flat[live].long()] = rows.reshape(-1, H)[live]
    return perm


def _output(tokens: int) -> torch.Tensor:
    return torch.full((tokens, H), NAN, dtype=torch.bfloat16, device=DEV)


def _problem(spec: dict, seed: int):
    """A cell's arguments, plus the per-pair rows and ids they were built from."""
    tokens = spec["tokens"]
    ids = _ids(tokens, seed)
    scales = _scales(tokens, seed)
    e2p, num_rows = _sort(ids, scales, spec["tile_tokens_dim"])
    rows = _rows(tokens, seed, spec["values"])
    args = dict(
        permuted_input=_place(rows, e2p, num_rows),
        output=_output(tokens),
        expanded_idx_to_permuted_idx=e2p,
        topk_scales=scales,
    )
    return args


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _cell_id(cell):
    s = cell.spec
    return f"t{s['tile_tokens_dim']}-T{s['tokens']}-{s['values']}"


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[_cell_id(c) for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    assert (s["num_experts"], s["top_k"], s["hidden_size"]) == (E, K, H)
    assert (s["dtype"], s["scales_dtype"]) == (torch.bfloat16, torch.float32)
    args = _problem(s, zlib.crc32(cell.why.encode()))
    before = {
        n: args[n].clone()
        for n in ("permuted_input", "expanded_idx_to_permuted_idx", "topk_scales")
    }
    out = op(**args)
    assert out is args["output"], "the op must return the tensor it wrote"
    _compare(out, op.reference(**args))
    for name, snapshot in before.items():
        assert _same_bits(args[name], snapshot), f"{name} was written"


# ── CUDA graphs: captured row counts, on new data ─────────────────────────


def _capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static = fn()
    return graph, static


def test_captured_rows_replay_on_new_data(certifying):  # noqa: F811
    """The sort, the FC2 rows' placement and this op are captured together, as the
    target runs them; the graph is replayed on two fresh sets of ids, scales and
    rows. The permuted buffer is not cleared between replays, so the rows the new
    sort does not use hold the previous placement's rows. Each replay is gated
    against the reference, and is bitwise equal to an eager call on the same data
    through its own sort. At the target's tile, 128.
    """
    certifying(op)
    tile = 128
    stale_rows = 0
    tokens = _CAPTURED_ROWS
    ids = _ids(tokens, seed=tokens)
    scales = _scales(tokens, seed=tokens)
    rows = _rows(tokens, seed=tokens)
    num_rows = _sort(ids, scales, tile)[1]
    perm = torch.full((num_rows, H), NAN, dtype=torch.bfloat16, device=DEV)
    out = _output(tokens)

    def step(ids=ids, scales=scales, rows=rows, perm=perm, out=out, tile=tile):
        e2p = _sort(ids, scales, tile)[0]
        perm.index_copy_(0, e2p.reshape(-1).long(), rows.reshape(-1, H))
        op(
            permuted_input=perm,
            output=out,
            expanded_idx_to_permuted_idx=e2p,
            topk_scales=scales,
        )
        return e2p

    graph, e2p = _capture(step)
    for trial in (1, 2):
        seed = 1000 * trial + tokens
        ids.copy_(_ids(tokens, seed))
        scales.copy_(_scales(tokens, seed))
        rows.copy_(_rows(tokens, seed))
        out.fill_(NAN)
        graph.replay()
        args = dict(
            permuted_input=perm,
            output=out,
            expanded_idx_to_permuted_idx=e2p,
            topk_scales=scales,
        )
        _compare(out, op.reference(**args))
        used = torch.zeros(num_rows, dtype=torch.bool, device=DEV)
        used[e2p.reshape(-1).long()] = True
        stale_rows += int((~used & ~perm.isnan().any(1)).sum())

        e2p_eager, n = _sort(ids.clone(), scales.clone(), tile)
        eager = op(
            permuted_input=_place(rows.clone(), e2p_eager, n),
            output=_output(tokens),
            expanded_idx_to_permuted_idx=e2p_eager,
            topk_scales=scales.clone(),
        )
        assert _same_bits(out, eager), trial
    del graph, e2p
    torch.cuda.synchronize()
    assert stale_rows > 0, "no replay ran over a stale row: the stale-row claim is untested"


# ── the guards ────────────────────────────────────────────────────────────


def _strided(t: torch.Tensor, pad: int) -> torch.Tensor:
    """The same values in a view whose rows are `pad` elements wider apart."""
    wide = torch.zeros(t.shape[0], t.shape[1] + pad, dtype=t.dtype, device=t.device)
    wide[:, : t.shape[1]] = t
    return wide[:, : t.shape[1]]


def test_noncontiguous_operand_answers_wrongly_and_is_refused():
    """The FC2 rows the op reads; the guard checks all four operands the same way."""
    name = "permuted_input"
    args = _problem(_spec(300), seed=61)
    ref = op.reference(**args)
    bad = dict(args, **{name: _strided(args[name], 64)})
    out = op(**bad)
    with pytest.raises(AssertionError, match="differ bitwise"):
        _compare(out.contiguous(), ref)
    with validating(op):
        with pytest.raises(AssertionError, match=f"{name} must be contiguous"):
            op(**bad)
        out = op(**dict(args, output=_output(300)))
    _compare(out, ref)
