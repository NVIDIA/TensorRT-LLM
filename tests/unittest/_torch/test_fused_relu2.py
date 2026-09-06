# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``relu2`` must stay bit-identical to ``torch.square(F.relu(x))``.

The fused kernel is only a memory-traffic optimisation, so every dispatch
decision it makes has to be invisible in the result. That means checking both
that the kernel matches eager exactly on the shapes it claims, and that the
ineligible cases actually fall back rather than producing something close.
"""

import os
from collections.abc import Callable, Iterator
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

import tensorrt_llm._torch.fused_relu2_triton as kernel_mod
from tensorrt_llm._torch.fused_relu2_triton import fused_relu2, is_eligible
from tensorrt_llm._torch.utils import _fused_relu2_impl, relu2


def eager_relu2(x: torch.Tensor) -> torch.Tensor:
    return torch.square(F.relu(x))


@pytest.fixture(autouse=True)
def _clear_impl_cache() -> Iterator[None]:
    """``_fused_relu2_impl`` memoises the env lookup, so reset it per test."""
    _fused_relu2_impl.cache_clear()
    yield
    _fused_relu2_impl.cache_clear()


@pytest.fixture
def fused_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """Shapes the Triton kernel was actually dispatched with.

    Comparing against eager is not enough on its own: if the resolver returned
    None the eager fallback would satisfy the comparison exactly, and the test
    would still pass while covering nothing. Recording the dispatches lets a
    test assert the fused path is the one that produced the result.
    """
    seen: list[tuple[int, ...]] = []
    real = kernel_mod.fused_relu2

    def spy(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        seen.append(tuple(x.shape))
        return real(x, out)

    monkeypatch.setattr(kernel_mod, "fused_relu2", spy)
    # relu2() resolves the kernel through a cached import, so the cache has to
    # be dropped after patching for the spy to be picked up.
    _fused_relu2_impl.cache_clear()
    return seen


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (4095,),  # not a multiple of the kernel's block
        (4096,),
        (4097,),
        (32768, 3712),  # a large relu2 MLP at ISL 32k
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_matches_eager_bitwise(
    shape: tuple[int, ...], dtype: torch.dtype, fused_calls: list[tuple[int, ...]]
) -> None:
    assert _fused_relu2_impl() is not None, "fused kernel did not resolve"
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).to(dtype)
    # randn alone barely reaches the negative saturation region
    x[..., ::7] *= -20.0
    assert is_eligible(x)
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)
    assert fused_calls == [shape]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_negative_and_zero_inputs(fused_calls: list[tuple[int, ...]]) -> None:
    """relu2 is flat at zero on the whole negative half, including -0.0."""
    x = torch.tensor(
        [-1e30, -1.0, -1e-30, -0.0, 0.0, 1e-30, 1.0], device="cuda", dtype=torch.float32
    )
    got = relu2(x)
    torch.testing.assert_close(got, eager_relu2(x), rtol=0, atol=0)
    assert not got[:5].any()
    assert fused_calls == [(7,)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_non_finite_inputs(dtype: torch.dtype, fused_calls: list[tuple[int, ...]]) -> None:
    """NaN must survive the relu.

    tl.maximum defaults to PropagateNan.NONE, which quietly turns a NaN into 0.0
    where torch.relu keeps it -- so the fused path would disagree with eager
    exactly where a NaN is the signal someone is looking for.
    """
    x = torch.tensor(
        [float("nan"), float("-nan"), float("inf"), float("-inf"), -1.0, 0.0, 2.0],
        device="cuda",
        dtype=dtype,
    )
    got, expected = relu2(x), eager_relu2(x)
    assert fused_calls == [(7,)]
    torch.testing.assert_close(got, expected, rtol=0, atol=0, equal_nan=True)
    assert got[:2].isnan().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_non_contiguous_falls_back(fused_calls: list[tuple[int, ...]]) -> None:
    base = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    x = base.t()
    assert not x.is_contiguous()
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)
    assert not fused_calls


def test_cpu_falls_back(fused_calls: list[tuple[int, ...]]) -> None:
    x = torch.randn(1024, dtype=torch.float32)
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)
    assert not fused_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_unsupported_dtype_falls_back(fused_calls: list[tuple[int, ...]]) -> None:
    x = torch.randn(1024, device="cuda", dtype=torch.float64)
    torch.testing.assert_close(relu2(x), eager_relu2(x), rtol=0, atol=0)
    assert not fused_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_empty_tensor(fused_calls: list[tuple[int, ...]]) -> None:
    x = torch.empty(0, device="cuda", dtype=torch.bfloat16)
    assert relu2(x).shape == x.shape
    assert not fused_calls  # numel() == 0 is ineligible; a zero-block launch is not


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_env_var_disables_fusion(fused_calls: list[tuple[int, ...]]) -> None:
    x = torch.randn(8192, device="cuda", dtype=torch.bfloat16)
    expected = eager_relu2(x)
    with mock.patch.dict(os.environ, {"TRTLLM_FUSED_RELU2": "0"}):
        _fused_relu2_impl.cache_clear()
        assert _fused_relu2_impl() is None
        torch.testing.assert_close(relu2(x), expected, rtol=0, atol=0)
    assert not fused_calls


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_explicit_out_is_written() -> None:
    x = torch.randn(4097, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    got = fused_relu2(x, out=out)
    assert got.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, eager_relu2(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(
    "make_out",
    [
        # too small: the kernel would write past the allocation
        pytest.param(
            lambda x: torch.empty(x.numel() // 2, device=x.device, dtype=x.dtype), id="too_small"
        ),
        pytest.param(
            lambda x: torch.empty(x.numel() * 2, device=x.device, dtype=x.dtype), id="too_large"
        ),
        # right shape but strided: values would land in the wrong positions
        pytest.param(
            lambda x: torch.empty(x.numel() * 2, device=x.device, dtype=x.dtype)[::2],
            id="non_contiguous",
        ),
        pytest.param(lambda x: torch.empty_like(x, dtype=torch.float32), id="wrong_dtype"),
        pytest.param(lambda x: torch.empty(x.shape, dtype=x.dtype), id="wrong_device"),
    ],
)
def test_invalid_out_is_rejected(make_out: Callable[[torch.Tensor], torch.Tensor]) -> None:
    x = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    out = make_out(x)
    with pytest.raises(ValueError, match="fused_relu2 out"):
        fused_relu2(x, out=out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_exact_aliasing_is_allowed() -> None:
    """Writing back over the input is safe: thread i reads and writes index i."""
    x = torch.randn(kernel_mod._BLOCK * 3, device="cuda", dtype=torch.bfloat16)
    expected = eager_relu2(x)
    got = fused_relu2(x, out=x)
    assert got.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shift", [1, kernel_mod._BLOCK - 1, kernel_mod._BLOCK])
def test_partially_overlapping_out_is_rejected(shift: int) -> None:
    """A shifted view of the same storage races across blocks, so reject it.

    Shape, dtype, device and contiguity all match here, so nothing but an
    overlap check can catch it. The span covers several blocks in each
    direction so the racing pair is not confined to one program.
    """
    base = torch.randn(kernel_mod._BLOCK * 3 + shift, device="cuda", dtype=torch.bfloat16)
    x = base[:-shift]
    out = base[shift:]
    assert x.shape == out.shape and x.is_contiguous() and out.is_contiguous()
    with pytest.raises(ValueError, match="overlaps"):
        fused_relu2(x, out=out)
    # ...and the other direction
    with pytest.raises(ValueError, match="overlaps"):
        fused_relu2(out, out=x)
