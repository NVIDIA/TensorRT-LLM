# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gemm/m3_target_head entry: the target lm_head shard GEMV over the lossless 12-bit coding of its
BF16 weight, against its reference, the in-tree BF16 head GEMM (``F.linear``) and fp64.

* The coding round-trips bit for bit (code values and exception lists), and a corrupted coding fails the check.
* Integer-valued data (fp32 sums exact in any order): the kernel's logits equal the in-tree GEMM's bit for bit.
* Random data: the kernel's error against fp64 is the in-tree GEMM's (the same products, another fp32 order, one
  BF16 rounding), its argmax is the in-tree one and its result is the same run to run.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops import (  # noqa: E501
    m3_target_head as target_head_kernels,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.m3_target_head import (  # noqa: E501
    CTA_WARPS,
    HIDDEN,
    TARGET_SHARD,
    decode,
    encode_chunked,
    round_trip_ok,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.m3_target_head import (  # noqa: E501
    m3_target_head as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401 — needs the path declared above

assert torch.cuda.is_available(), "m3_target_head requires a CUDA device"


def _seed(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _weight(n: int, k: int, seed: int, exceptions: bool = False) -> torch.Tensor:
    """A [n, k] head of trained-weight scale; with ``exceptions``, values on both sides of the exponent window, exact
    zeros and a large outlier in many tiles."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    if exceptions:
        w[::97, ::131] = 1e-9
        w[::89, 7] = 0.0
        w[3, 5] = 7.5
        w[n - 1, k - 1] = -3.0
    return w.to(torch.bfloat16)


_CODED: dict = {}


def _coded(n: int, k: int, seed: int, exceptions: bool):
    """(weight, encode_chunked(weight)), built once per shape."""
    key = (n, k, seed, exceptions)
    if key not in _CODED:
        _CODED.clear()
        w = _weight(n, k, seed, exceptions)
        _CODED[key] = (w, encode_chunked(w))
    return _CODED[key]


def _args(coded, x: torch.Tensor, n: int, grid: int):
    packed, base, exc_off, exc_pos, exc_w = coded
    out = torch.full((x.shape[0], n), float("nan"), dtype=torch.bfloat16, device="cuda")
    part = torch.zeros(grid * CTA_WARPS * 256, dtype=torch.float32, device="cuda")
    counter = torch.zeros((n + 15) // 16, dtype=torch.int32, device="cuda")
    return (packed, x, out, part, counter, exc_off, exc_pos, exc_w.view(-1), base, n, grid, True)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """The kernel against its reference and the in-tree BF16 GEMM: the error against fp64 is the in-tree GEMM's
    (both are one BF16 rounding of an fp32 sum of the same exact products), the argmax is the in-tree one, nearly
    every logit is the in-tree one bit for bit, and a second call gives the same bits."""
    spec = cell.spec
    m, n, k = spec["m"], spec["n"], spec["k"]
    exceptions = spec.get("exceptions", False)
    w, coded = _coded(n, k, 1, exceptions)
    if exceptions:
        assert coded[3].numel() > 0, "the cell must exercise the exception lists"
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x = torch.randn(m, k, device="cuda", generator=g).to(torch.bfloat16)
    args = _args(coded, x, n, _num_sms())
    certifying(op)
    ref = op.reference(*args)
    out = op(*args).clone()
    again = op(*_args(coded, x, n, _num_sms())).clone()
    torch.cuda.synchronize()
    assert not bool(args[4].any()), "the split-tile counters are not re-armed"
    assert torch.equal(out, again), "the split-tile reduction is not deterministic"
    stock = torch.nn.functional.linear(x, w)
    exact = x.double() @ w.double().t()
    # Each side: one BF16 rounding (<= half an ulp of the logit) of an fp32 sum whose order error is far below it.
    err_out = (out.double() - exact).abs().max()
    err_stock = (stock.double() - exact).abs().max()
    err_ref = (ref.double() - exact).abs().max()
    assert err_out <= 1.01 * max(err_stock, err_ref) + 1e-6, (err_out, err_stock, err_ref)
    assert (out == stock).float().mean() > 0.99
    assert (out == ref).float().mean() > 0.99
    assert torch.equal(out.float().argmax(-1), stock.float().argmax(-1))


@pytest.mark.parametrize("m", [1, 4, 8])
def test_integer_data_is_the_bf16_head_bit_for_bit(m) -> None:
    """Small-integer weights and activations: every fp32 sum is exact in any order, so the coded head's logits are
    the in-tree BF16 GEMM's bit for bit (exceptions included: zeros and integers outside the exponent window)."""
    g = torch.Generator(device="cuda").manual_seed(_seed(f"integer {m}"))
    n, k = 4096, HIDDEN
    w = torch.randint(-3, 4, (n, k), generator=g, device="cuda").to(torch.bfloat16)
    w[::7, ::5] = 96.0  # far above the window of the {1, 2, 3} exponents: exceptions
    x = torch.randint(-2, 3, (m, k), generator=g, device="cuda").to(torch.bfloat16)
    coded = encode_chunked(w)
    assert coded[3].numel() > 0
    assert round_trip_ok(*coded, w)
    with validating(op):
        out = op(*_args(coded, x, n, _num_sms()))
    assert torch.equal(out, torch.nn.functional.linear(x, w))


def test_the_head_writes_only_the_shards_columns() -> None:
    """Integer data over a 4000-row shard: every logit is the in-tree GEMM's bit for bit, and the columns past n of a
    wider output are left untouched."""
    n, k, m = 4000, 1024, 2
    g = torch.Generator(device="cuda").manual_seed(_seed("partial last tile"))
    # Small integers: every fp32 sum is exact in any order, so the logits are comparable bit for bit.
    w = torch.randint(-3, 4, (n, k), generator=g, device="cuda").to(torch.bfloat16)
    w[::7, ::5] = 96.0  # exceptions, including in the partial tile
    x = torch.randint(-2, 3, (m, k), generator=g, device="cuda").to(torch.bfloat16)
    coded = encode_chunked(w)
    assert round_trip_ok(*coded, w)
    args = list(_args(coded, x, n, _num_sms()))
    args[2] = torch.full((m, n + 16), float("nan"), dtype=torch.bfloat16, device="cuda")
    with validating(op):
        op(*args)
    out = args[2]
    assert torch.isnan(out[:, n:]).all(), "the head wrote past the shard's last column"
    stock = torch.nn.functional.linear(x, w)
    assert torch.equal(out[:, :n], stock)


def test_a_shard_of_partial_tiles_keeps_the_bf16_head() -> None:
    """The kernel streams whole 16-row tiles: a shard whose rows are not a multiple of 16 is not taken (the fused
    verify keeps the in-tree GEMM), and the op refuses it."""
    w = torch.zeros(4004, 1024, dtype=torch.bfloat16, device="cuda")
    assert not target_head_kernels.head_supported(SimpleNamespace(weight=w, bias=None))
    assert target_head_kernels.head_supported(SimpleNamespace(weight=w[:4000], bias=None))


@pytest.mark.parametrize("n,k", [(TARGET_SHARD, HIDDEN), (4004, 1024)])
def test_round_trip_is_bitwise(n, k) -> None:
    w, coded = _coded(n, k, 1, True)
    assert coded[3].numel() > 0
    assert round_trip_ok(*coded, w)
    assert torch.equal(decode(*coded, n, k).view(torch.int16), w.view(torch.int16))


def test_a_corrupted_coding_fails_closed() -> None:
    """A flipped sign/mantissa bit, a wrong exception value or a wrong base each fail the bitwise round trip."""
    w = _weight(4004, 1024, 2, exceptions=True)
    packed, base, exc_off, exc_pos, exc_w = encode_chunked(w)
    assert round_trip_ok(packed, base, exc_off, exc_pos, exc_w, w)
    bad = packed.clone()
    bad[100] ^= 1
    assert not round_trip_ok(bad, base, exc_off, exc_pos, exc_w, w)
    bad_w = exc_w.clone()
    bad_w[0, 0] += 1e-3
    assert not round_trip_ok(packed, base, exc_off, exc_pos, bad_w, w)
    assert not round_trip_ok(packed, base + 1, exc_off, exc_pos, exc_w, w)


def test_prepare_head_keeps_only_a_bitwise_coding(monkeypatch) -> None:
    """prepare_head keeps the coded shard when it round-trips and runs it as the entry; a coding that does not
    round-trip is dropped (the in-tree GEMM stays the path)."""
    w = _weight(4000, 1024, 3, exceptions=True)
    head = target_head_kernels.prepare_head(SimpleNamespace(weight=w, bias=None))
    assert head.coded and head.weight is w
    x = torch.randn(4, 1024, device="cuda").to(torch.bfloat16)
    out = target_head_kernels.logits(head, x)
    stock = torch.nn.functional.linear(x, w)
    assert torch.equal(out.float().argmax(-1), stock.float().argmax(-1))
    assert (out == stock).float().mean() > 0.99

    def corrupt(weight, *args, **kwargs):
        packed, base, exc_off, exc_pos, exc_w = encode_chunked(weight)
        packed = packed.clone()
        packed[7] ^= 0x40
        return packed, base, exc_off, exc_pos, exc_w

    monkeypatch.setattr(target_head_kernels, "encode_chunked", corrupt)
    dropped = target_head_kernels.prepare_head(SimpleNamespace(weight=w, bias=None))
    assert not dropped.coded and dropped.w is None


def test_guard_refuses_a_dirty_counter() -> None:
    w, coded = _coded(4000, 1024, 1, True)
    args = list(_args(coded, torch.zeros(2, 1024, dtype=torch.bfloat16, device="cuda"), 4000, 8))
    args[4][3] = 1
    with pytest.raises(AssertionError, match="counter must be zero"):
        with validating(op):
            op(*args)


def _tiny_exceptions_head(n: int = 4096, k: int = 1024) -> torch.Tensor:
    """Weights of 1 (exponent window base 113) with column 0 zero except 2^-30 / 2^-29 on rows 0 / 1: exceptions far
    below the window whose exact products are the only non-zero logits."""
    w = torch.ones((n, k), dtype=torch.bfloat16, device="cuda")
    w[:, 0] = 0.0
    w[0, 0] = 2.0**-30
    w[1, 0] = 2.0**-29
    return w


@pytest.mark.parametrize("m", [1, 4, 8])
def test_tiny_exception_products_are_exact(m) -> None:
    """An exception adds exactly x * w: with a one-hot activation on column 0 the logits are 0, 2^-30 and 2^-29 bit
    for bit as the BF16 head's, and row 1 wins (a correction against a non-zero placeholder cancels them to 0)."""
    w = _tiny_exceptions_head()
    coded = encode_chunked(w)
    assert coded[1] == 113 and round_trip_ok(*coded, w)
    x = torch.zeros((m, w.shape[1]), dtype=torch.bfloat16, device="cuda")
    x[:, 0] = 1.0
    with validating(op):
        out = op(*_args(coded, x, w.shape[0], _num_sms()))
    stock = torch.nn.functional.linear(x, w)
    assert torch.equal(out.view(torch.int16), stock.view(torch.int16))
    assert out.float().argmax(-1).tolist() == [1] * m


def test_a_shard_smaller_than_the_grid_is_stored() -> None:
    """A shard with fewer coded units than warps of the requested grid: the launch narrows the grid so every warp owns
    a unit, and every logit is stored."""
    w = _tiny_exceptions_head(n=256)
    coded = encode_chunked(w)
    x = torch.zeros((2, w.shape[1]), dtype=torch.bfloat16, device="cuda")
    x[:, 0] = 1.0
    out = op(*_args(coded, x, w.shape[0], _num_sms()))
    assert torch.equal(out.view(torch.int16), torch.nn.functional.linear(x, w).view(torch.int16))
