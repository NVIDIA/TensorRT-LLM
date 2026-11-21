# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 QA of the static-FP8 scenario's fused ops against the operators they replace.

Each fused op is compared with the same math built from the installed operators: the native
static FP8 quantize, the cuBLAS scaled GEMM and the family AdaLN ops. The sequence lengths cover
one row, lengths that are not a multiple of any tile, and the 1344x768x124 request lengths; the
kernels are compiled once and serve all of them.
"""

import pytest
import torch
import torch.nn.functional as F

D, QK, UP = 5376, 7168, 14336
BF16 = torch.bfloat16
LENGTHS = (1, 300, 1000, 4097, 37712, 38169)
EPS = 1e-6


def _quantize(x, scale):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, scale)[0]


def _gemm(a, w, scale_a, scale_w):
    return torch.ops.trtllm.cublas_scaled_mm(
        a, w.t(), scale_a=scale_a, scale_b=scale_w, bias=None, out_dtype=BF16
    )


def _same_fraction(got, ref):
    assert got.shape == ref.shape and got.dtype == ref.dtype
    width = got.element_size()
    same = got.reshape(-1).view(torch.uint8) == ref.reshape(-1).view(torch.uint8)
    return float(same.view(-1, width).all(dim=1).double().mean())


def _relative_l2(got, ref):
    got, ref = got.float(), ref.float()
    return float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))


@pytest.fixture(scope="module")
def scenario():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("The fused static-FP8 kernels require an SM107 GPU")
    if torch.cuda.get_device_properties(0).multi_processor_count != 212:
        pytest.skip("The fused static-FP8 kernels require the qualified 212-SM GPU")
    import tensorrt_llm  # noqa: F401  -- registers the trtllm operators
    from tensorrt_llm._torch.visual_gen.models.minimax_h3 import fused_adaln
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1 import ops
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1.kernels import (
        conversion,
        gate_up_swiglu,
        projection_residual,
    )

    # These ops read no per-process scenario setting, so nothing is selected here.
    device = torch.cuda.current_device()
    # As at model load: the ops only use kernels that are already compiled.
    for width, gate in (
        (QK, projection_residual.ATTENTION_OUTPUT_GATE),
        (UP, projection_residual.DOWN_GATE),
    ):
        projection_residual.compile_kernel(device, D, width, gate, BF16, torch.int64)
    # The conversions run on the installed operators here; test_fp8_conversions.py covers the
    # optional compiled ones, which produce the same bytes.
    previous = conversion._MODULES.get(device)
    conversion._MODULES[device] = (None, None)
    yield ops, fused_adaln, gate_up_swiglu, projection_residual
    if previous is None:
        del conversion._MODULES[device]
    else:
        conversion._MODULES[device] = previous


def _weight(n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn((n, k), generator=gen, device="cuda") * 0.02).to(BF16)
    scale = (weight.float().abs().max() / 448).reshape(1)
    return _quantize(weight, scale).contiguous(), scale


@pytest.fixture(scope="module")
def weights():
    return {
        "out": _weight(D, QK, 11),
        "gate_up": _weight(2 * UP, D, 12),
        "down": _weight(D, UP, 13),
    }


def _inputs(length, rows):
    gen = torch.Generator(device="cuda").manual_seed(100 + length)

    def randn(*shape, std=1.0):
        return torch.randn(shape, generator=gen, device="cuda") * std

    return {
        "hidden": randn(1, length, D).to(BF16),
        "modulation": randn(rows, 6 * D, std=0.2).to(BF16),
        "indices": torch.randint(0, rows, (length,), generator=gen, device="cuda"),
        "norm_weight": (1 + randn(D, std=0.05)).to(BF16),
        "randn": randn,
    }


def _fused(scenario, activation, weight, x, gate):
    """True when the op will run the fused projection kernel for these tensors."""
    projection_residual = scenario[3]
    return projection_residual.supports(
        activation,
        weight,
        x["hidden"],
        x["modulation"],
        x["indices"],
        getattr(projection_residual, gate),
    )


def _scale(value):
    scale = torch.tensor([value], device="cuda")
    return scale, torch.reciprocal(scale.cpu()).cuda()


@pytest.mark.parametrize("length", LENGTHS)
def test_adaln_to_fp8_is_bit_identical(scenario, length):
    ops, fused_adaln, _, _ = scenario
    x = _inputs(length, rows=3)
    scale, reciprocal = _scale(0.031)
    codes = ops.norm(x["hidden"], x["norm_weight"], x["modulation"], x["indices"], reciprocal, EPS)
    normed = fused_adaln.h3_norm_mod(
        x["hidden"], x["norm_weight"], x["modulation"], x["indices"], EPS, 1, 0
    )
    assert torch.equal(codes.view(torch.uint8), _quantize(normed, scale).view(torch.uint8))


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rows", [3, 6])
def test_attention_projection_residual_and_codes(scenario, weights, length, rows):
    ops, fused_adaln, _, _ = scenario
    x = _inputs(length, rows)
    weight, weight_scale = weights["out"]
    raw = x["randn"](length, QK, std=1.5).to(BF16).reshape(1, length, QK)
    input_scale, input_reciprocal = _scale(0.02)
    up_scale, up_reciprocal = _scale(0.027)
    activation = _quantize(raw.reshape(length, QK), input_scale)
    assert _fused(scenario, activation, weight, x, "ATTENTION_OUTPUT_GATE")
    residual, codes = ops.attention_projection(
        raw,
        weight,
        input_scale,
        weight_scale,
        x["hidden"],
        x["norm_weight"],
        x["modulation"],
        x["indices"],
        up_reciprocal,
        input_reciprocal,
        EPS,
    )
    projected = _gemm(
        _quantize(raw.reshape(length, QK), input_scale), weight, input_scale, weight_scale
    )
    expected_residual, expected_norm = fused_adaln.h3_gate_res_norm_mod(
        x["hidden"],
        projected.reshape_as(x["hidden"]),
        x["norm_weight"],
        x["modulation"],
        x["indices"],
        EPS,
        2,
        4,
        3,
    )
    # cuBLAS orders the accumulation differently at small row counts, so a few last-bit
    # differences are expected below about 1000 rows; from there the residual is bit-identical.
    assert _relative_l2(residual, expected_residual) < 1e-4
    assert _same_fraction(residual, expected_residual) > 0.999
    assert _same_fraction(codes, _quantize(expected_norm, up_scale)) > 0.999
    if length >= 1000:
        assert torch.equal(residual.view(torch.uint8), expected_residual.view(torch.uint8))
        assert torch.equal(
            codes.view(torch.uint8), _quantize(expected_norm, up_scale).view(torch.uint8)
        )


@pytest.mark.parametrize("length", LENGTHS)
def test_gate_up_swiglu_to_fp8(scenario, weights, length):
    ops, _, gate_up_swiglu, _ = scenario
    x = _inputs(length, rows=3)
    weight, weight_scale = weights["gate_up"]
    up_scale, _ = _scale(0.027)
    down_scale, _ = _scale(0.05)
    activation = _quantize(x["randn"](length, D).to(BF16), up_scale)
    codes = ops.gate_up(
        activation, gate_up_swiglu.pack_weight(weight), up_scale, weight_scale, down_scale
    )
    projected = _gemm(activation, weight, up_scale, weight_scale)
    expected = (F.silu(projected[:, :UP].float()) * projected[:, UP:].float()).to(BF16)
    # The fused kernel applies SwiGLU to the FP32 accumulator; the reference rounds the GEMM
    # output to BF16 first, so about 1.5 % of the codes differ by one FP8 step.
    reference = _quantize(expected, down_scale)
    assert _same_fraction(codes, reference) > 0.97
    assert _relative_l2(codes, reference) < 0.02
    # No element is off by more than one FP8 step, and no row is wrong as a whole.
    got, ref = codes.view(torch.uint8).int(), reference.view(torch.uint8).int()
    magnitude = ((got & 0x7F) - (ref & 0x7F)).abs()
    same_sign = (got & 0x80) == (ref & 0x80)
    around_zero = ((got & 0x7F) <= 1) & ((ref & 0x7F) <= 1)
    assert bool(((magnitude <= 1) & (same_sign | around_zero)).all())
    assert float((got == ref).double().mean(dim=1).min()) > 0.9


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rows", [3, 6])
def test_down_gated_residual(scenario, weights, length, rows):
    ops = scenario[0]
    x = _inputs(length, rows)
    weight, weight_scale = weights["down"]
    down_scale, _ = _scale(0.05)
    activation = _quantize(x["randn"](length, UP, std=2.0).to(BF16), down_scale)
    assert _fused(scenario, activation, weight, x, "DOWN_GATE")
    residual = ops.down(
        activation,
        weight,
        down_scale,
        weight_scale,
        x["hidden"],
        x["modulation"],
        x["indices"],
    )
    projected = _gemm(activation, weight, down_scale, weight_scale).reshape_as(x["hidden"])
    gate = x["modulation"][:, 5 * D : 6 * D].index_select(0, x["indices"])
    expected = torch.addcmul(x["hidden"].float(), gate.float(), projected.float()).to(BF16)
    assert _relative_l2(residual, expected) < 1e-4
    assert _same_fraction(residual, expected) > 0.999
    if length >= 1000:
        assert torch.equal(residual.view(torch.uint8), expected.view(torch.uint8))


def test_one_projection_kernel_serves_every_length(scenario, weights):
    ops, _, _, projection_residual = scenario
    weight, weight_scale = weights["down"]
    down_scale, _ = _scale(0.05)
    before = dict(projection_residual._compiled)
    for length in (7, 513, 2049):
        x = _inputs(length, rows=3)
        activation = _quantize(x["randn"](length, UP).to(BF16), down_scale)
        assert _fused(scenario, activation, weight, x, "DOWN_GATE")
        ops.down(
            activation,
            weight,
            down_scale,
            weight_scale,
            x["hidden"],
            x["modulation"],
            x["indices"],
        )
    assert projection_residual._compiled == before


def _misaligned(tensor):
    """The same values in contiguous storage that starts 2 bytes off any 256-byte boundary."""
    storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    view = storage[1:].view(tensor.shape)
    view.copy_(tensor)
    assert view.is_contiguous() and view.data_ptr() % 256 != 0
    return view


@pytest.mark.parametrize("length", (300, 4097))
@pytest.mark.parametrize("case", ("misaligned_hidden", "int32_indices"))
def test_down_outside_the_kernel_contract_takes_the_unfused_path(scenario, weights, length, case):
    ops, _, _, projection_residual = scenario
    x = _inputs(length, rows=3)
    weight, weight_scale = weights["down"]
    down_scale, _ = _scale(0.05)
    activation = _quantize(x["randn"](length, UP, std=2.0).to(BF16), down_scale)
    hidden, indices = x["hidden"], x["indices"]
    if case == "misaligned_hidden":
        hidden = _misaligned(hidden)
    else:
        indices = indices.int()
    assert not projection_residual.supports(
        activation, weight, hidden, x["modulation"], indices, projection_residual.DOWN_GATE
    )
    before = dict(projection_residual._compiled)
    residual = ops.down(
        activation, weight, down_scale, weight_scale, hidden, x["modulation"], indices
    )
    assert projection_residual._compiled == before  # nothing compiles at run time
    projected = _gemm(activation, weight, down_scale, weight_scale).reshape_as(hidden)
    gate = x["modulation"][:, 5 * D : 6 * D].index_select(0, x["indices"])
    expected = torch.addcmul(hidden.float(), gate.float(), projected.float()).to(BF16)
    assert torch.equal(residual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("length", (300, 4097))
def test_attention_projection_outside_the_kernel_contract_takes_the_unfused_path(
    scenario, weights, length
):
    ops, fused_adaln, _, projection_residual = scenario
    x = _inputs(length, rows=3)
    weight, weight_scale = weights["out"]
    raw = x["randn"](length, QK, std=1.5).to(BF16).reshape(1, length, QK)
    input_scale, input_reciprocal = _scale(0.02)
    up_scale, up_reciprocal = _scale(0.027)
    hidden = _misaligned(x["hidden"])
    activation = _quantize(raw.reshape(length, QK), input_scale)
    assert not projection_residual.supports(
        activation,
        weight,
        hidden,
        x["modulation"],
        x["indices"],
        projection_residual.ATTENTION_OUTPUT_GATE,
    )
    residual, codes = ops.attention_projection(
        raw,
        weight,
        input_scale,
        weight_scale,
        hidden,
        x["norm_weight"],
        x["modulation"],
        x["indices"],
        up_reciprocal,
        input_reciprocal,
        EPS,
    )
    projected = _gemm(activation, weight, input_scale, weight_scale)
    expected_residual, expected_norm = fused_adaln.h3_gate_res_norm_mod(
        x["hidden"],
        projected.reshape_as(x["hidden"]),
        x["norm_weight"],
        x["modulation"],
        x["indices"],
        EPS,
        2,
        4,
        3,
    )
    assert _relative_l2(residual, expected_residual) < 1e-4
    assert _same_fraction(residual, expected_residual) > 0.999
    assert _same_fraction(codes, _quantize(expected_norm, up_scale)) > 0.999


def test_non_finite_values_saturate_in_adaln(scenario):
    """Where the normalized value is NaN the fused kernel writes a saturated code, not the NaN code.

    Row 3 holds a NaN, so its norm and every element of the row are NaN. Row 5 holds an Inf, so
    its norm is Inf: that element is Inf / Inf = NaN and the rest of the row is finite."""
    ops, fused_adaln, _, _ = scenario
    x = _inputs(64, rows=3)
    hidden = x["hidden"].clone()
    hidden[0, 3, 7] = float("nan")
    hidden[0, 5, 9] = float("inf")
    scale, reciprocal = _scale(0.031)
    codes = ops.norm(hidden, x["norm_weight"], x["modulation"], x["indices"], reciprocal, EPS)
    normed = fused_adaln.h3_norm_mod(
        hidden, x["norm_weight"], x["modulation"], x["indices"], EPS, 1, 0
    )
    got, ref = codes.view(torch.uint8)[0], _quantize(normed, scale).view(torch.uint8)[0]
    nan_code = (ref & 0x7F) == 0x7F
    assert bool(nan_code[3].all()) and int(nan_code[5].sum()) == 1 and bool(nan_code[5, 9])
    assert int(nan_code.sum()) == D + 1
    assert torch.equal(got[~nan_code], ref[~nan_code])
    assert bool(((got[nan_code] & 0x7F) == 0x7E).all())


def test_family_rope_table_repeats_its_frequencies():
    """The contract behind ``rope_halves_equal=True`` in the FP8 attention."""
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.transformer_minimax_h3 import (
        MiniMaxH3RotaryPosEmbed,
    )

    positions = torch.stack(
        [torch.arange(40) % 5, torch.arange(40) % 7, torch.arange(40) % 11], dim=1
    )
    cos, sin = MiniMaxH3RotaryPosEmbed()(positions)
    half = cos.shape[-1] // 2
    assert torch.equal(cos[..., :half], cos[..., half:])
    assert torch.equal(sin[..., :half], sin[..., half:])
