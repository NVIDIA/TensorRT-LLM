# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_paged_sparse_gqa_nvfp4 entry.

The paged state is real: one NVFP4 pool as the QSA cache manager serves it --
e2m1 data `[pages, 2, kv, tpb, D/2]` and e4m3 block scales `[pages, 2, kv, tpb,
D/16]` -- sized from the pages its requests map, and the engine's slot table,
a row per request slot, more slots than requests, assigned by a seeded
permutation, `-1` past each sequence. Selection rows are built the way QSA's
top-k + expansion lay them out, packed at the front.

Beside `_compare` against the reference, every cell runs QSA's bf16 sparse GQA
over a bf16 pool that holds the dequantized values and holds the two outputs to
the same band: the NVFP4 kernels are the bf16 ones with a dequantizing load.
"""

import math
import re

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4 as _entry,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.attention.qsa_paged_sparse_gqa_nvfp4 import (  # noqa: E501
    qsa_paged_sparse_gqa_nvfp4 as op,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import triton_qsa_paged_sparse_gqa

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "qsa_paged_sparse_gqa_nvfp4 requires a CUDA device"

DEV = "cuda"
ENGINE_MAX_SEQ_LEN = 8192
#: The e4m3fn NaN bit pattern.
E4M3_NAN = 0x7F

_KERNELS = {
    "_nvfp4_sparse_gqa_prefill_kernel": "prefill",
    "_nvfp4_sparse_gqa_splitk_kernel": "splitk",
    "_nvfp4_merge_splitk_kernel": "merge",
}


class _LaunchRecorder:
    """Records a Triton kernel's grid indexing, then delegates to the kernel."""

    def __init__(self, inner, name, log):
        self._inner, self._name, self._log = inner, name, log

    def __getitem__(self, grid):
        self._log.append((self._name, tuple(grid)))
        return self._inner[grid]


@pytest.fixture
def launches(monkeypatch):
    """Which branch kernels the next call launches, in order, with grids."""
    log: list = []
    for name, tag in _KERNELS.items():
        monkeypatch.setattr(_entry, name, _LaunchRecorder(getattr(_entry, name), tag, log))
    return log


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each (row, head)'s magnitude, 2 ulp relative RMS, in q's dtype.

    After the dequantize the kernels are the bf16 sparse GQA kernels, whose band
    this is (they round the online-softmax probabilities to the value dtype
    before P.V); the dequantize itself is exact at unit scale and rounded once,
    as `reference` rounds it, otherwise.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.numel() == 0:
        return
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _segments(spec):
    """`(first position, rows)` per request in row order, and a generation mask."""
    if "mixed" not in spec:
        segments = list(spec["segments"])
        return segments, torch.zeros(sum(n for _, n in segments), dtype=torch.bool, device=DEV)
    requests, context_lens = spec["mixed"]
    segments = [
        (0 if i == 0 else (907 * i) % (ENGINE_MAX_SEQ_LEN - n + 1), n)
        for i, n in enumerate(context_lens)
    ]
    segments += [(100 + (499 * g) % 8000, 2) for g in range(requests - len(context_lens))]
    generation = torch.cat(
        [torch.full((n,), i >= len(context_lens), device=DEV) for i, (_, n) in enumerate(segments)]
    )
    return segments, generation


def _selected_tokens(positions, seed, compress_ratio, token_topk):
    """QSA's expansion layout: chosen complete groups, the trailing partial group, -1."""
    block_topk = token_topk // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = positions.numel()
    visible = positions.long() + 1
    available = visible // compress_ratio
    taken = available.clamp(max=block_topk)
    width = max(block_topk, int(available.max().item()) if rows else 0)
    scores = torch.rand(rows, width, generator=gen, device=DEV)
    group_ids = torch.arange(width, device=DEV)[None, :]
    scores = torch.where(group_ids < available[:, None], scores, torch.inf)
    chosen = scores.argsort(dim=1)[:, :block_topk]
    column = torch.arange(final_topk, device=DEV)[None, :]
    group_of_column = (column // compress_ratio).clamp(max=block_topk - 1).expand(rows, -1)
    body = torch.gather(chosen, 1, group_of_column) * compress_ratio + column % compress_ratio
    in_body = column < (taken * compress_ratio)[:, None]
    tail_column = column - (taken * compress_ratio)[:, None]
    tail = (available * compress_ratio)[:, None] + tail_column
    in_tail = ~in_body & (tail_column >= 0) & (tail < visible[:, None])
    selected = torch.where(in_body, body, torch.where(in_tail, tail, torch.full_like(body, -1)))
    return selected.to(torch.int32)


def _random_pool(pages, kv, tpb, d, gen):
    """Pool bytes: any e2m1 nibble, and positive e4m3 scales from 2^-6 to 0.5."""
    data = torch.randint(0, 256, (pages, 2, kv, tpb, d // 2), generator=gen, device=DEV)
    codes = torch.randint(0x08, 0x31, (pages, 2, kv, tpb, d // 16), generator=gen, device=DEV)
    return data.to(torch.uint8), codes.to(torch.uint8)


def _build(spec, seed):
    """Real paged state for one spec, keyed by the op's parameter names."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tpb, kv, d, q_heads = (spec[k] for k in ("tokens_per_block", "kv_heads", "head_dim", "q_heads"))
    segments, generation = _segments(spec)
    positions = torch.cat([torch.arange(s, s + n, device=DEV) for s, n in segments])
    req = torch.cat([torch.full((n,), i, device=DEV) for i, (_, n) in enumerate(segments)])
    rows, slots = positions.numel(), spec["slots"]
    assert len(segments) <= slots
    slot_of = torch.randperm(slots, generator=torch.Generator().manual_seed(seed))[: len(segments)]
    mapped = [math.ceil((s + n) / tpb) for s, n in segments]
    assert not mapped or max(mapped) <= spec["pages"]
    needed = sum(mapped) + 7
    slot_pages, layer = spec["slot_pages"], spec["layer"]
    planes = slice(2 * layer, 2 * layer + 2)
    # The requests sit on the pool's last `needed` pages, and every scale outside them is
    # NaN, so a page address that strayed could not come back finite.
    pool_pages = spec.get("pool_pages", needed)
    assert pool_pages >= needed
    first = pool_pages - needed
    page_ids = first + torch.randperm(needed, generator=gen, device=DEV).to(torch.int32)
    data = torch.zeros(pool_pages, slot_pages, kv, tpb, d // 2, dtype=torch.uint8, device=DEV)
    scales = torch.full(
        (pool_pages, slot_pages, kv, tpb, d // 16), E4M3_NAN, dtype=torch.uint8, device=DEV
    )
    data[first:, planes], scales[first:, planes] = _random_pool(needed, kv, tpb, d, gen)
    block_table = torch.full((slots, spec["pages"]), -1, dtype=torch.int32, device=DEV)
    cursor = 0
    for slot, count in zip(slot_of.tolist(), mapped):
        block_table[slot, :count] = page_ids[cursor : cursor + count]
        cursor += count

    q_dtype = spec["q_dtype"]
    if spec["q_layout"] == "qkv_slice":
        qkv = torch.randn(rows, q_heads + 2 * kv, d, generator=gen, device=DEV).to(q_dtype)
        q = qkv[:, :q_heads]
    else:
        q = torch.randn(rows, q_heads, d, generator=gen, device=DEV).to(q_dtype)
    selected = _selected_tokens(positions, seed + 3, spec["compress_ratio"], spec["token_topk"])
    if "width" in spec:
        selected = selected[:, : spec["width"]].contiguous()
    request_indices = slot_of.to(DEV)[req].to(spec["request_dtype"])
    if spec.get("holes"):
        selected[1] = -1
        request_indices[2] = slots + 67
        block_table[:, 0] = -1

    args = dict(
        q=q,
        k_cache=data[:, 2 * layer],
        v_cache=data[:, 2 * layer + 1],
        k_scale_cache=scales[:, 2 * layer],
        v_scale_cache=scales[:, 2 * layer + 1],
        kv_scale_quant_orig=torch.tensor(
            [1.0, spec["global_k"], spec["global_v"]], dtype=torch.float32, device=DEV
        ),
        block_table=block_table,
        selected_tokens=selected,
        request_indices=request_indices,
        tokens_per_block=tpb,
        softmax_scale=d**-0.5,
    )
    if spec["prefill_pair"]:
        args.update(query_positions=positions, compress_ratio=spec["compress_ratio"])
    return args, generation


def _bf16_twin(args):
    """QSA's bf16 sparse GQA over a bf16 pool holding the dequantized values."""
    pages = args["k_cache"].shape[0]
    all_pages = torch.arange(pages, device=DEV)[:, None].expand(-1, args["tokens_per_block"])
    slots = torch.arange(args["tokens_per_block"], device=DEV)[None, :].expand(pages, -1)
    planes = []
    for data, scales, index, grouped in (
        (args["k_cache"], args["k_scale_cache"], 1, False),
        (args["v_cache"], args["v_scale_cache"], 2, True),
    ):
        values = _entry._dequantize_tokens(
            data, scales, all_pages, slots, args["kv_scale_quant_orig"][index], grouped
        )  # [pages, tpb, H, D]
        planes.append(values.to(args["q"].dtype).transpose(1, 2).contiguous())
    twin = {
        name: args[name]
        for name in ("q", "block_table", "selected_tokens", "request_indices", "tokens_per_block")
    }
    twin.update(softmax_scale=args["softmax_scale"], k_cache=planes[0], v_cache=planes[1])
    if "query_positions" in args:
        twin.update(query_positions=args["query_positions"], compress_ratio=args["compress_ratio"])
    return triton_qsa_paged_sparse_gqa(**twin)


def _bound(positions, spec):
    cr = spec["compress_ratio"]
    visible = positions.long() + 1
    return (visible // cr).clamp(max=spec["token_topk"] // cr) * cr + visible % cr


def _is_prefix_packed(table):
    valid = table >= 0
    columns = torch.arange(table.shape[1], device=table.device)[None, :]
    return (valid == (columns < valid.sum(1)[:, None])).all(1)


_OPERANDS = (
    "q",
    "k_cache",
    "v_cache",
    "k_scale_cache",
    "v_scale_cache",
    "block_table",
    "selected_tokens",
    "request_indices",
)


def _capture_sweep(spec, seed, launches):
    """The decode site under a CUDA graph, at each G for each rows-per-request:
    capture, write a second state's operands in place, replay, and require the
    eager call on that second state bitwise."""
    sweep = spec["capture"]
    for per_request in sweep["rows_per_request"]:
        for groups in sweep["groups"]:
            _capture_once(
                _decode_step(spec, groups, per_request),
                seed + 100 * per_request + groups,
                launches,
                where=f"G={groups} x {per_request} rows",
            )


def _start(request: int, rows: int) -> int:
    """A request's first position, across the budget."""
    return min(100 + (499 * request) % 8000, ENGINE_MAX_SEQ_LEN - rows)


def _decode_step(spec, groups, per_request):
    """The decode site's step: `groups` requests x `per_request` rows."""
    segments = tuple((_start(g, per_request), per_request) for g in range(groups))
    return {**spec, "segments": segments}


def _capture_once(shape, seed, launches, where) -> None:
    """One call of `shape` captured, a second state's operands written in place,
    replayed: the replay must be bitwise the eager call on the second state."""
    where = f"{where}, [{shape['slots']}, {shape['pages']}] table"
    args, _ = _build(shape, seed)
    fresh, _ = _build(shape, seed + 50)
    launches.clear()
    op(**args)
    assert [name for name, _ in launches] == list(shape["launches"]), where
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = op(**args)
        for name in _OPERANDS:
            args[name].copy_(fresh[name])
        graph.replay()
        torch.cuda.synchronize()
        eager = op(**fresh)
        assert torch.equal(captured, eager), f"replay differs from eager at {where}"
        _compare(captured, op.reference(**fresh))
    finally:
        del graph
        torch.cuda.synchronize()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying, launches) -> None:  # noqa: F811
    certifying(op)
    spec = cell.spec
    if "capture" in spec:
        _capture_sweep(spec, seed=sum(map(ord, cell.why)), launches=launches)
        return
    args, _ = _build(spec, seed=sum(map(ord, cell.why)))
    rows = args["q"].shape[0]
    if spec["prefill_pair"] and not spec.get("holes"):
        selected = args["selected_tokens"]
        assert bool(_is_prefix_packed(selected).all())
        valid = (selected >= 0).sum(1)
        assert torch.equal(valid, _bound(args["query_positions"], spec))

    out = op(**args)
    assert [name for name, _ in launches] == list(spec["launches"])
    assert launches[0][1][:2] == (rows, spec["kv_heads"])
    assert out.shape == (rows, spec["q_heads"], spec["head_dim"])
    assert out.dtype == args["q"].dtype and out.is_contiguous()
    assert not bool(out.isnan().any())
    _compare(out, op.reference(**args))
    # QSA's bf16 kernels over the same values: inside the same band, not bitwise
    # (the dequantizing load changes the compiled kernel).
    _compare(out, _bf16_twin(args))


CKPT = op.CELLS[0].spec


def _with(**overrides):
    return {**CKPT, **overrides}


def test_gate_discriminates_wrong_variants() -> None:
    """References wrong one way each -- V's scales read token-major like K's,
    the nibbles swapped, the second-level scale ignored, K and V swapped -- are
    all rejected by `compare`."""
    args, _ = _build(_with(segments=((2032, 272),), prefill_pair=True, global_v=2.5), seed=9000)
    out = op(**args)
    _compare(out, op.reference(**args))
    v_scales = args["v_scale_cache"]
    pages, kv, tpb, groups = v_scales.shape
    token_major = (
        v_scales.reshape(pages, kv, tpb // 4, 4, groups)
        .permute(0, 1, 2, 4, 3)
        .reshape(v_scales.shape)
    )
    nibbles = (args["v_cache"] >> 4) | (args["v_cache"] << 4)
    variants = {
        "v_scales_token_major": dict(v_scale_cache=token_major),
        "nibbles_swapped": dict(v_cache=nibbles),
        "no_second_level_scale": dict(
            kv_scale_quant_orig=torch.ones(3, dtype=torch.float32, device=DEV)
        ),
        "k_and_v_swapped": dict(
            k_cache=args["v_cache"],
            v_cache=args["k_cache"],
            k_scale_cache=args["v_scale_cache"],
            v_scale_cache=args["k_scale_cache"],
        ),
    }
    for name, override in variants.items():
        wrong = op.reference(**{**args, **override})
        with pytest.raises(AssertionError):
            _compare(out, wrong)
        assert (out.float() - wrong.float()).abs().max().item() > 0.05, name


def test_calls_are_bitwise_deterministic() -> None:
    args, _ = _build(_with(segments=tuple((4000 + 37 * i, 2) for i in range(4))), seed=8800)
    assert torch.equal(op(**args), op(**args))


def test_zero_rows_return_an_empty_output() -> None:
    args, _ = _build(_with(segments=((900, 1),) * 4), seed=8400)
    for name in ("q", "selected_tokens", "request_indices"):
        args[name] = args[name][:0]
    out = op(**args)
    assert out.shape == (0, CKPT["q_heads"], CKPT["head_dim"]) and out.dtype == torch.bfloat16
    _compare(out, op.reference(**args))


_REJECTIONS = {
    "float32_query": (lambda a: {**a, "q": a["q"].float()}, "q must be BF16 or FP16"),
    "bf16_pool": (
        lambda a: {**a, "k_cache": a["k_cache"].view(torch.int8)},
        "k_cache must be uint8",
    ),
    "scales_of_another_width": (
        lambda a: {**a, "v_scale_cache": a["v_scale_cache"][..., :8]},
        "v_scale_cache must be uint8",
    ),
    "single_scale": (
        lambda a: {**a, "kv_scale_quant_orig": a["kv_scale_quant_orig"][:1]},
        "kv_scale_quant_orig must be a contiguous fp32 [3]",
    ),
    "int64_block_table": (
        lambda a: {**a, "block_table": a["block_table"].long()},
        "block_table must be a row-contiguous 2-D int32 tensor",
    ),
    "positions_without_ratio": (
        lambda a: {**a, "query_positions": torch.zeros(4, dtype=torch.int64, device=DEV)},
        "pass both query_positions and compress_ratio, or neither",
    ),
    "host_query": (lambda a: {**a, "q": a["q"].cpu()}, "every operand must be on q's CUDA device"),
}


@pytest.mark.parametrize("case", list(_REJECTIONS), ids=list(_REJECTIONS))
def test_rejects(case) -> None:
    """The op's own loud refusals -- the reason the entry carries no `is_valid`."""
    args, _ = _build(_with(segments=((600, 1),) * 4), seed=9100)
    args["selected_tokens"] = args["selected_tokens"][:, :64].contiguous()
    mutate, message = _REJECTIONS[case]
    with pytest.raises(ValueError, match=re.escape(message)):
        with validating(op):
            op(**mutate(args))
