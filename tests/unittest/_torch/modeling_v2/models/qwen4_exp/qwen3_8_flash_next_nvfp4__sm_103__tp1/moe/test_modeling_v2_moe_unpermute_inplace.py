# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the moe_unpermute_inplace entry of the Qwen3.8-Flash-Next target.

The entry owns the contract. This file lays a cell's FC2 rows out the way the
CuteDSL FC2 hands them over -- placed by moe_sort, every row the op must not read
filled with NaN, the output pre-filled with NaN -- and holds the claims no cell
carries: CUDA-graph replay at the captured row counts, the eager prefix sweep,
run-to-run and placement independence, the gate's discrimination, T = 0, the loud
rejections and the guards.
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

#: Row counts of a decode capture (requests x MTP verify width 1..4, up to 32
#: requests), and the eager prefix lengths: both sides of the power-of-two edges up
#: to the 8192-row ceiling. A subset of the prefixes is also re-sorted.
_CAPTURED_ROWS = (1, 2, 3, 4, 5, 8, 32, 96, 128)
_PREFIXES = (
    *(1, 2, 3, 4, 5, 8, 31, 32, 33, 96, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025),
    *(2047, 2048, 2049, 4095, 4096, 4097, 8191),
)
_RESORT = (1, 2, 5, 32, 129, 1024, 8192)


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


def _spec(tokens, tile_tokens_dim=128, dist="router", values="normal", non_local_from=None):
    """The fields `_problem` reads; the same names as a cell's spec."""
    return dict(
        tokens=tokens,
        tile_tokens_dim=tile_tokens_dim,
        dist=dist,
        values=values,
        non_local_from=non_local_from,
    )


def _ids(dist: str, tokens: int, seed: int) -> torch.Tensor:
    """Expert ids, `[tokens, K]` int32, every row naming 10 distinct experts."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    if dist == "router":
        scores = torch.rand(tokens, E, device=DEV, generator=g)
        return scores.topk(K, dim=-1).indices.to(torch.int32)
    if dist == "hot":
        hot = torch.randperm(E, device=DEV, generator=g)[:K]
        slots = torch.rand(tokens, K, device=DEV, generator=g).argsort(dim=-1)
        return hot[slots].to(torch.int32)
    if dist == "round_robin":
        base = torch.arange(tokens, device=DEV).unsqueeze(1) * K
        return ((base + torch.arange(K, device=DEV)) % E).to(torch.int32)
    raise ValueError(dist)


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


def _output(tokens: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.full((tokens, H), NAN, dtype=dtype, device=DEV)


def _problem(spec: dict, seed: int):
    """A cell's arguments, plus the per-pair rows and ids they were built from."""
    tokens = spec["tokens"]
    ids = _ids(spec["dist"], tokens, seed)
    scales = _scales(tokens, seed)
    e2p, num_rows = _sort(ids, scales, spec["tile_tokens_dim"])
    if spec["non_local_from"] is not None:
        e2p = torch.where(ids >= spec["non_local_from"], -1, e2p).to(torch.int32)
        e2p[0] = -1
    rows = _rows(tokens, seed, spec["values"])
    args = dict(
        permuted_input=_place(rows, e2p, num_rows),
        output=_output(tokens),
        expanded_idx_to_permuted_idx=e2p,
        topk_scales=scales,
    )
    return args, rows, ids


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(_bits(a), _bits(b))


def _cell_id(cell):
    s = cell.spec
    tag = f"t{s['tile_tokens_dim']}-T{s['tokens']}-{s['dist']}-{s['values']}"
    return tag if s["non_local_from"] is None else f"{tag}-nonlocal{s['non_local_from']}"


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[_cell_id(c) for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    assert (s["num_experts"], s["top_k"], s["hidden_size"]) == (E, K, H)
    assert (s["dtype"], s["scales_dtype"]) == (torch.bfloat16, torch.float32)
    args, _, _ = _problem(s, zlib.crc32(cell.why.encode()))
    before = {
        n: args[n].clone()
        for n in ("permuted_input", "expanded_idx_to_permuted_idx", "topk_scales")
    }
    out = op(**args)
    assert out is args["output"], "the op must return the tensor it wrote"
    _compare(out, op.reference(**args))
    for name, snapshot in before.items():
        assert _same_bits(args[name], snapshot), f"{name} was written"
    if s["non_local_from"] is not None:
        assert int((args["expanded_idx_to_permuted_idx"] < 0).sum()) > s["tokens"]
        assert bool((_bits(out[0]) == 0).all()), "a token with every slot -1 is not +0"


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


@pytest.mark.parametrize("tile", (128, 256))
def test_captured_rows_replay_on_new_data(tile, certifying):  # noqa: F811
    """The sort, the FC2 rows' placement and this op are captured together, as the
    target runs them; each graph is replayed on two fresh sets of ids, scales and
    rows. The permuted buffer is not cleared between replays, so the rows the new
    sort does not use hold the previous placement's rows. Each replay is gated
    against the reference, and is bitwise equal to an eager call on the same data
    through its own sort.
    """
    certifying(op)
    stale_rows = 0
    for tokens in _CAPTURED_ROWS:
        ids = _ids("router", tokens, seed=tokens)
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
            ids.copy_(_ids("router", tokens, seed))
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
            assert _same_bits(out, eager), (tokens, trial)
        del graph, e2p
    torch.cuda.synchronize()
    assert stale_rows > 0, "no replay ran over a stale row: the stale-row claim is untested"


# ── eager: prefixes of the largest batch ──────────────────────────────────


@pytest.mark.parametrize("tile", (128, 256))
def test_eager_rows_prefix_bitwise(tile, certifying):  # noqa: F811
    """One block per token makes the grid the only thing T sizes: the 8192-row call
    is gated against the reference, and the first T rows of every row count in the
    sweep are bitwise those of the 8192-row call -- over the same placement, and
    (a subset) over a fresh sort of the first T tokens alone, which places their
    rows elsewhere. A token's output does not depend on its batch.
    """
    certifying(op)
    args, rows, ids = _problem(_spec(8192, tile_tokens_dim=tile), seed=8192 + tile)
    full = op(**args)
    _compare(full, op.reference(**args))
    perm, e2p, scales = (
        args["permuted_input"],
        args["expanded_idx_to_permuted_idx"],
        args["topk_scales"],
    )
    for tokens in _PREFIXES:
        out = op(
            permuted_input=perm,
            output=_output(tokens),
            expanded_idx_to_permuted_idx=e2p[:tokens],
            topk_scales=scales[:tokens],
        )
        assert _same_bits(out, full[:tokens]), tokens
    for tokens in _RESORT:
        e2p_t, n = _sort(ids[:tokens], scales[:tokens], tile)
        out = op(
            permuted_input=_place(rows[:tokens], e2p_t, n),
            output=_output(tokens),
            expanded_idx_to_permuted_idx=e2p_t,
            topk_scales=scales[:tokens],
        )
        assert _same_bits(out, full[:tokens]), tokens


# ── run to run, and where the rows sit ────────────────────────────────────


@pytest.mark.parametrize("tokens", (1, 32, 128, 8192))
def test_repeats_and_placements_are_bitwise(tokens, certifying):  # noqa: F811
    """What the fused finalize does not give: 8 identical calls, a second sort
    of the same ids (the sort orders an expert's rows by atomics), a random
    permutation of every permuted row, and a shuffle of the tokens all give the
    same bits.
    """
    certifying(op)
    args, rows, ids = _problem(_spec(tokens), seed=tokens + 7)
    first = op(**args).clone()
    _compare(first, op.reference(**args))
    for _ in range(7):
        assert _same_bits(op(**dict(args, output=_output(tokens))), first)

    perm, e2p, scales = (
        args["permuted_input"],
        args["expanded_idx_to_permuted_idx"],
        args["topk_scales"],
    )
    e2p2, n = _sort(ids, scales, 128)
    if tokens == 8192:
        assert int((e2p2 != e2p).sum()) > 0, "the second sort placed every row where the first did"
    out = op(
        permuted_input=_place(rows, e2p2, n),
        output=_output(tokens),
        expanded_idx_to_permuted_idx=e2p2,
        topk_scales=scales,
    )
    assert _same_bits(out, first), "a second sort of the same ids changed the result"

    g = torch.Generator(device=DEV).manual_seed(tokens)
    p = torch.randperm(perm.shape[0], device=DEV, generator=g)
    moved = torch.empty_like(perm)
    moved[p] = perm
    out = op(
        permuted_input=moved,
        output=_output(tokens),
        expanded_idx_to_permuted_idx=p[e2p.long()].to(torch.int32),
        topk_scales=scales,
    )
    assert _same_bits(out, first), "a random placement of the rows changed the result"

    order = torch.randperm(tokens, device=DEV, generator=g)
    out = op(
        permuted_input=perm,
        output=_output(tokens),
        expanded_idx_to_permuted_idx=e2p[order].contiguous(),
        topk_scales=scales[order].contiguous(),
    )
    assert _same_bits(out, first[order]), "shuffling the tokens changed a token's result"


# ── what the bitwise gate can see ─────────────────────────────────────────


def test_the_gate_sees_slot_order_and_rounding(certifying):  # noqa: F811
    """`_compare` is bitwise; a sum taken in another order or with other roundings
    is rejected. The reverse-order sum is the entry's own reference over the
    slots reversed.
    """
    certifying(op)
    args, rows, _ = _problem(_spec(8192), seed=99)
    out = op(**args)
    _compare(out, op.reference(**args))
    scales = args["topk_scales"]
    reverse = op.reference(
        permuted_input=args["permuted_input"],
        output=args["output"],
        expanded_idx_to_permuted_idx=args["expanded_idx_to_permuted_idx"].flip(1).contiguous(),
        topk_scales=scales.flip(1).contiguous(),
    )
    unfused = torch.zeros(8192, H, device=DEV)
    for k in range(K):
        unfused = unfused + rows[:, k].float() * scales[:, k : k + 1]
    exact = (rows.double() * scales.double().unsqueeze(-1)).sum(1)
    bf16_sum = torch.zeros(8192, H, dtype=torch.bfloat16, device=DEV)
    for k in range(K):
        bf16_sum = bf16_sum + (rows[:, k].float() * scales[:, k : k + 1]).to(torch.bfloat16)
    for wrong in (
        reverse,  # reverse slot order
        unfused.to(torch.bfloat16),  # product rounded before the add
        exact.to(torch.bfloat16),  # float64 sum rounded once
        bf16_sum,  # bf16 accumulation
    ):
        with pytest.raises(AssertionError, match="differ bitwise"):
            _compare(out, wrong)


# ── T = 0 and the loud rejections ─────────────────────────────────────────


def test_zero_tokens_returns_and_leaves_no_error():
    args, _, _ = _problem(_spec(8), seed=71)
    empty = op(
        permuted_input=args["permuted_input"],
        output=torch.empty(0, H, dtype=torch.bfloat16, device=DEV),
        expanded_idx_to_permuted_idx=args["expanded_idx_to_permuted_idx"][:0],
        topk_scales=args["topk_scales"][:0],
    )
    torch.cuda.synchronize()
    assert tuple(empty.shape) == (0, H)
    assert float((torch.ones(1000, device=DEV) + 1).sum()) == 2000.0
    out = op(**args)
    _compare(out, op.reference(**args))


def test_op_rejects_loudly():
    """Rejected by the op itself; none of these needs a guard."""
    tokens = 64
    args, _, _ = _problem(_spec(tokens), seed=81)
    p, e, s = (
        args["permuted_input"],
        args["expanded_idx_to_permuted_idx"],
        args["topk_scales"],
    )
    narrow = 2556
    cases = [
        ("int64 indices", dict(expanded_idx_to_permuted_idx=e.long()), "expected scalar type Int"),
        (
            "hidden not a multiple of 8",
            dict(
                permuted_input=p[:, :narrow].contiguous(),
                output=torch.empty(tokens, narrow, dtype=torch.bfloat16, device=DEV),
            ),
            "hidden_size must be divisible by 8",
        ),
        (
            "fp32 rows",
            dict(permuted_input=p.float(), output=_output(tokens, torch.float32)),
            "Unsupported input dtype",
        ),
        ("fp16 scales, bf16 rows", dict(topk_scales=s.half()), "Unsupported input dtype"),
        (
            "fewer than T * top_k rows",
            dict(permuted_input=p[: tokens * K - 1]),
            r"greater than or equal to num_tokens \* top_k",
        ),
        ("scales rows", dict(topk_scales=s[:-1]), r"topk_scales.size\(0\) must be num_tokens"),
        (
            "scales width",
            dict(topk_scales=s[:, :9].contiguous()),
            r"topk_scales.size\(1\) must be top_k",
        ),
        (
            "index rows",
            dict(expanded_idx_to_permuted_idx=e[:-1]),
            r"expanded_idx_to_permuted_idx.size\(0\) must be num_tokens",
        ),
        (
            "output width",
            dict(output=torch.empty(tokens, H - 8, dtype=torch.bfloat16, device=DEV)),
            r"output.size\(1\) must be hidden_size",
        ),
    ]
    for tag, change, match in cases:
        with pytest.raises(RuntimeError, match=match):
            op(**dict(args, **change))
    torch.cuda.synchronize()
    out = op(**args)
    _compare(out, op.reference(**args))


# ── the guards ────────────────────────────────────────────────────────────

_OPERANDS = ("permuted_input", "output", "expanded_idx_to_permuted_idx", "topk_scales")


def _strided(t: torch.Tensor, pad: int) -> torch.Tensor:
    """The same values in a view whose rows are `pad` elements wider apart."""
    wide = torch.zeros(t.shape[0], t.shape[1] + pad, dtype=t.dtype, device=t.device)
    wide[:, : t.shape[1]] = t
    return wide[:, : t.shape[1]]


@pytest.mark.parametrize("name", _OPERANDS)
def test_noncontiguous_operand_answers_wrongly_and_is_refused(name):
    args, _, _ = _problem(_spec(300), seed=61)
    ref = op.reference(**args)
    bad = dict(args, **{name: _strided(args[name], 64 if args[name].shape[1] == H else 6)})
    out = op(**bad)
    with pytest.raises(AssertionError, match="differ bitwise"):
        _compare(out.contiguous(), ref)
    with validating(op):
        with pytest.raises(AssertionError, match=f"{name} must be contiguous"):
            op(**bad)
        out = op(**dict(args, output=_output(300)))
    _compare(out, ref)


def test_output_dtype_mismatch_answers_wrongly_and_is_refused():
    """An fp32 output receives the bf16 result's bit patterns, packed into the
    first half of its bytes.
    """
    args, _, _ = _problem(_spec(300), seed=62)
    ref = op.reference(**args)
    out32 = op(**dict(args, output=_output(300, torch.float32)))
    assert _same_bits(out32.view(torch.bfloat16).reshape(-1)[: 300 * H].view(300, H), ref)
    with pytest.raises(AssertionError):
        _compare(out32, ref)
    with validating(op):
        with pytest.raises(AssertionError, match="output dtype"):
            op(**dict(args, output=_output(300, torch.float32)))


def test_tokens_past_int32_are_refused():
    """The guard reads only `output.size(0)` first, so a meta tensor stands in for
    the 2^31-row output the op would silently leave unwritten."""
    args, _, _ = _problem(_spec(8), seed=63)
    huge = torch.empty((2**31, 8), dtype=torch.bfloat16, device="meta")
    with validating(op), pytest.raises(AssertionError, match=r">= 2\^31 rows"):
        op(**dict(args, output=huge))
