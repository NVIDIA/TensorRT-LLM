# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_indexer_k_cache_update entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel is built for sm_103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_k_cache_update import (  # noqa: E402, E501
    build,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.attention.cute_dsl_indexer_k_cache_update import (  # noqa: E402, E501
    cute_dsl_indexer_k_cache_update as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

HEAD_DIM, ROPE_DIM = 128, 64
#: Page size and per-token bytes of the indexer K pool (128 codes + one fp32 scale).
PAGE, TOKEN_BYTES = 32, 132
#: The layer of the pool each cell writes: neither the first nor the last, so a
#: stray write on either side shows.
LAYER = 3
POISON = 0xA7
THETA = 8.0e6  # the checkpoint's `rope_theta`

_COMPILED: dict = {}


def _compiled(cache_contiguous: bool):
    if cache_contiguous not in _COMPILED:
        _COMPILED[cache_contiguous] = build(
            torch.cuda.current_device(), is_neox=False, cache_contiguous=cache_contiguous
        )
    return _COMPILED[cache_contiguous]


def _rope_table(positions: int) -> torch.Tensor:
    """The target's indexer table `[positions, cos(32) | sin(32)]`, viewed `[positions, 2, 32]`."""
    half = ROPE_DIM // 2
    inv = 1.0 / (THETA ** (torch.arange(half, dtype=torch.float64) * 2.0 / ROPE_DIM))
    ang = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    table = torch.cat([ang.cos(), ang.sin()], dim=1).float().cuda().contiguous()
    return table.view(-1, 2, half)


def _build(spec, seed):
    """`(args, pool)` for one cell; `args` is the op's positional argument list.

    The pool is `[blocks, layers, page * 132]` uint8, block-major over layers
    as TRT-LLM allocates the indexer K cache, and the cache is one layer's
    `[blocks, page, 1, 132]` view of it. Each token gets a distinct (page, slot)
    and the TRT-LLM slot pair: codes at `block * page * 132 + slot * 128`, the
    scale at `block * page * 132 + page * 128 + slot * 4`. Padding rows get -1.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    cpu = torch.Generator().manual_seed(seed)
    tokens, blocks, layers = spec["tokens"], spec["blocks"], spec["pool_layers"]
    pool = torch.full((blocks, layers, PAGE * TOKEN_BYTES), POISON, dtype=torch.uint8).cuda()
    layer = min(LAYER, layers - 1)
    k_cache = pool[:, layer, :].view(blocks, PAGE, 1, TOKEN_BYTES)

    keys = torch.randn(tokens, HEAD_DIM, generator=g, device="cuda") * 3.0 + 0.5
    if spec["misalign"]:
        storage = torch.empty(tokens * HEAD_DIM + 1, dtype=torch.bfloat16, device="cuda")
        indexer_k = storage[1:].view(tokens, HEAD_DIM)
        indexer_k.copy_(keys)
        assert indexer_k.data_ptr() % 8 != 0
    else:
        indexer_k = keys.to(torch.bfloat16)
    norm_weight = 1.0 + 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda")
    norm_bias = 0.1 * torch.randn(HEAD_DIM, generator=g, device="cuda")
    max_pos = max(8192, tokens)
    if tokens > 64:
        pos = torch.arange(tokens, device="cuda")
    else:
        pos = torch.randint(0, max_pos, (tokens,), generator=g, device="cuda")

    flat = torch.randperm(blocks * PAGE, generator=cpu)[:tokens]
    block, slot = flat // PAGE, flat % PAGE
    fp8_slot = block * PAGE * TOKEN_BYTES + slot * HEAD_DIM
    scale_slot = block * PAGE * TOKEN_BYTES + PAGE * HEAD_DIM + slot * 4
    if spec["pad_rows"]:
        fp8_slot[-spec["pad_rows"] :] = -1
        scale_slot[-spec["pad_rows"] :] = -1
    args = (
        indexer_k,
        norm_weight,
        norm_bias,
        pos.to(torch.int32),
        _rope_table(max_pos),
        k_cache,
        fp8_slot.cuda(),
        scale_slot.cuda(),
        tokens,
    )
    return args, pool, layer


def _compare(k_cache, expected, fp8_slot, scale_slot, num_tokens) -> None:
    """Bytes the op must not write: exact. Scales: exact. Codes: one e4m3 step, rarely.

    The kernel normalises with its own summation order and an approximate
    rsqrt; the reference normalises in fp64. The fp32 results differ by a few
    ulps, which moves a bf16 rounding about once in 10**4 values and an e4m3
    code far less often, and then by exactly one step. So: every differing code
    is one step from the reference with the same sign, and at most 2**-10 of
    the codes (at least one) differ. A wrong rope mode, table or norm moves
    most codes by many steps. The scale is a power of two from the row's
    amax; one step on the amax only moves it when the amax sits on a
    power-of-two boundary, which these inputs do not hit, so scales are exact.
    """
    got = k_cache.contiguous().view(-1)
    want = expected.view(-1)
    live = fp8_slot[:num_tokens] >= 0
    code_at = (
        fp8_slot[:num_tokens][live].unsqueeze(1) + torch.arange(HEAD_DIM, device="cuda")
    ).flatten()
    scale_at = (
        scale_slot[:num_tokens][live].unsqueeze(1) + torch.arange(4, device="cuda")
    ).flatten()
    written = torch.zeros_like(got, dtype=torch.bool)
    written[code_at] = True
    written[scale_at] = True
    assert torch.equal(got[~written], want[~written]), "the op wrote outside its slots"
    assert torch.equal(got[scale_at], want[scale_at]), "a row scale differs"
    a, b = got[code_at].int(), want[code_at].int()
    off = a != b
    if bool(off.any()):
        same_sign = (a[off] & 0x80) == (b[off] & 0x80)
        one_step = ((a[off] & 0x7F) - (b[off] & 0x7F)).abs() <= 1
        assert bool((same_sign & one_step).all()), "a code is more than one e4m3 step off"
    allowed = max(1, code_at.numel() >> 10)
    assert int(off.sum()) <= allowed, (
        f"{int(off.sum())} of {code_at.numel()} codes differ; at most {allowed} may"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args, pool, layer = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    k_cache = args[5]
    compiled = _compiled(k_cache.is_contiguous())
    # The reference reads the bytes it does not write, so it runs first.
    expected = op.reference(*args, is_neox=False, compiled=compiled)
    pool_before = pool.clone()
    with validating(op):
        # The target's first, eager call validates on the device; later ones do not.
        out = op(*args, is_neox=False, compiled=compiled, validate=True)
    assert out is k_cache, "the op writes the caller's cache and returns it"
    others = [i for i in range(pool.shape[1]) if i != layer]
    assert torch.equal(pool[:, others], pool_before[:, others]), "another layer was written"
    _compare(k_cache, expected, args[6], args[7], args[8])


def test_a_capturable_call_matches_the_validated_one() -> None:
    """`validate=False` (every call inside a CUDA graph) writes the same bytes."""
    spec = dict(tokens=16, pool_layers=4, blocks=8, pad_rows=2, misalign=False)
    args, _, _ = _build(spec, seed=41)
    again, _, _ = _build(spec, seed=41)
    compiled = _compiled(False)
    op(*args, is_neox=False, compiled=compiled, validate=True)
    op(*again, is_neox=False, compiled=compiled, validate=False)
    assert torch.equal(args[5], again[5])


def _small():
    return _build(dict(tokens=8, pool_layers=4, blocks=8, pad_rows=0, misalign=False), seed=43)[0]


def test_guard_refuses_a_rope_mode_the_launchers_were_not_built_for() -> None:
    with pytest.raises(AssertionError, match="compiled for is_neox"):
        with validating(op):
            op(*_small(), is_neox=True, compiled=_compiled(False))


def test_guard_refuses_contiguous_addressing_on_a_strided_view() -> None:
    args = _small()
    assert not args[5].is_contiguous()
    with pytest.raises(AssertionError, match="strided view"):
        with validating(op):
            op(*args, is_neox=False, compiled=_compiled(True))


def test_guard_refuses_a_position_past_the_rope_table() -> None:
    args = list(_small())
    args[3] = args[3].clone()
    args[3][0] = args[4].shape[0]
    with pytest.raises(AssertionError, match="rope table"):
        with validating(op):
            op(*args, is_neox=False, compiled=_compiled(False))


def test_guard_refuses_a_misaligned_slot() -> None:
    args = list(_small())
    args[6] = args[6].clone()
    args[6][1] += 2
    with pytest.raises(AssertionError, match="4-byte aligned"):
        with validating(op):
            op(*args, is_neox=False, compiled=_compiled(False))


def test_guard_refuses_slots_that_disagree_on_validity() -> None:
    args = list(_small())
    args[7] = args[7].clone()
    args[7][2] = -1
    with pytest.raises(AssertionError, match="disagree"):
        with validating(op):
            op(*args, is_neox=False, compiled=_compiled(False))
