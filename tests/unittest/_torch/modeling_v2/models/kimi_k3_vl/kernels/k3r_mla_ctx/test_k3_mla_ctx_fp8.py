# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA context: the Triton static FP8 quantization the split-K context path uses
(``ctx_split_k``) against the C++ one, and the ``static_quantize_e4m3`` entry on its cells."""

import pytest
import torch

__extra_import_path__ = ["../../../.."]

NOPE, ROPE, V, LORA = 128, 64, 128, 512

requires_fp8_cvt = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="E4M3 conversion instructions require SM89+",
)


# ---------------------------------------------------------------------------------------------
# GPU: Triton FP8 kernels against the C++ static FP8 quantization
# ---------------------------------------------------------------------------------------------


def _cpp_quant(x, scale):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x.contiguous(), scale)[0]


def _quant_test_input(rows, cols, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, device="cuda") * 64
    flat = x.view(-1)
    special = [
        float("inf"),
        float("-inf"),
        float("nan"),
        1e30,
        -1e30,
        1e-30,
        -0.0,
        0.0,
        447.0,
        448.0,
        456.0,
        464.0,
        465.0,
        480.0,
        -464.0,
        -465.0,
        2.0**-9,
        2.0**-10,
    ]
    flat[: len(special)] = torch.tensor(special, device="cuda")
    n = min(flat.numel() - len(special), 8192)
    flat[len(special) : len(special) + n] = torch.logspace(-12, 12, n, device="cuda") * torch.sign(
        torch.randn(n, generator=g, device="cuda")
    )
    return x.to(torch.bfloat16)


@requires_fp8_cvt
@pytest.mark.parametrize("shape", [(1, 18432), (257, 18432), (4096, 192), (3, 6)])
@pytest.mark.parametrize("scale", [1.0, 0.5, 3.0, 0.37])
def test_triton_static_quant_matches_cpp(shape, scale):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_ctx import (
        fp8_static_quant,
    )

    x = _quant_test_input(*shape, seed=shape[0])
    s = torch.tensor([scale], dtype=torch.float32, device="cuda")
    y = fp8_static_quant.static_quantize_e4m3(x, s)
    assert y.shape == x.shape and y.dtype == torch.float8_e4m3fn
    assert torch.equal(y.view(torch.uint8), _cpp_quant(x, s).view(torch.uint8))


@requires_fp8_cvt
@pytest.mark.parametrize("rows", [1, 63, 64, 1000, 16384])
@pytest.mark.parametrize("scale", [1.0, 0.37])
def test_triton_static_quant_rows_matches_cpp(rows, scale):
    """Row-strided views (the K-pe columns of latent_cache) quantize to the C++ bytes."""
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3r_mla_ctx import (
        fp8_static_quant,
    )

    latent = _quant_test_input(rows, LORA + ROPE, seed=rows)
    s = torch.tensor([scale], dtype=torch.float32, device="cuda")
    for view in (latent[:, LORA:], latent[:, LORA:].contiguous(), latent[:, :ROPE]):
        assert fp8_static_quant.supports_rows(view)
        y = fp8_static_quant.static_quantize_e4m3_rows(view, s)
        assert y.is_contiguous() and y.shape == view.shape
        assert torch.equal(y.view(torch.uint8), _cpp_quant(view, s).view(torch.uint8))
    # Odd storage offsets, non-power-of-two widths and non-BF16 inputs are not supported.
    for view in (latent[:, 511:575], latent[:, LORA : LORA + 48], latent[:, LORA:].float()):
        assert not fp8_static_quant.supports_rows(view)
        with pytest.raises(ValueError):
            fp8_static_quant.static_quantize_e4m3_rows(view, s)


# ---------------------------------------------------------------------------------------------
# GPU: the static_quantize_e4m3 entry on its cells
# ---------------------------------------------------------------------------------------------


def _entry():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.static_quantize_e4m3 import (  # noqa: E501
        static_quantize_e4m3,
    )

    return static_quantize_e4m3


@requires_fp8_cvt
@pytest.mark.parametrize("cell", _entry().CELLS, ids=[c.why[:48] for c in _entry().CELLS])
def test_static_quantize_e4m3_cells(cell):
    """The entry's ``.CELLS`` under ``validating``: bitwise its reference (one IEEE FP32
    reciprocal, saturating round-to-nearest-even E4M3) and the C++ quantizer."""
    from _validating import validating

    op = _entry()
    spec = cell.spec
    tokens = spec["tokens"]
    width = {"q": 96 * (NOPE + ROPE), "latent_k_pe": LORA + ROPE, "k_pe": ROPE}[spec["source"]]
    x = _quant_test_input(tokens, width, seed=tokens + width)
    if spec["source"] == "latent_k_pe":
        x = x[:, LORA:]
    scale = torch.ones(1, dtype=torch.float32, device="cuda")
    with validating(op):
        y = op(x, scale, rows=spec["rows"])
    assert y.dtype == torch.float8_e4m3fn and y.shape == x.shape and y.is_contiguous()
    expected = op.reference(x, scale, rows=spec["rows"])
    assert torch.equal(y.view(torch.uint8), expected.view(torch.uint8)), cell.why
    assert torch.equal(y.view(torch.uint8), _cpp_quant(x, scale).view(torch.uint8)), cell.why
