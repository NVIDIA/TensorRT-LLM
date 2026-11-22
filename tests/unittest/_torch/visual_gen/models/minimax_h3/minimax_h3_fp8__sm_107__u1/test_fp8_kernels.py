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


@pytest.fixture
def branch(scenario, monkeypatch):
    """Counts the projection ops' calls into the fused kernel and into the unfused cuBLAS GEMM."""
    ops, _, _, projection_residual = scenario
    counts = {"fused": 0, "unfused": 0}
    run, gemm = projection_residual.run, ops._cublas_scaled_mm

    def fused(*args, **kwargs):
        counts["fused"] += 1
        return run(*args, **kwargs)

    def unfused(*args, **kwargs):
        counts["unfused"] += 1
        return gemm(*args, **kwargs)

    monkeypatch.setattr(projection_residual, "run", fused)
    monkeypatch.setattr(ops, "_cublas_scaled_mm", unfused)
    return counts


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
def test_attention_projection_residual_and_codes(scenario, weights, branch, length, rows):
    ops, fused_adaln, _, _ = scenario
    x = _inputs(length, rows)
    weight, weight_scale = weights["out"]
    raw = x["randn"](length, QK, std=1.5).to(BF16).reshape(1, length, QK)
    input_scale, input_reciprocal = _scale(0.02)
    up_scale, up_reciprocal = _scale(0.027)
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
    assert branch == {"fused": 1, "unfused": 0}
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
    got_m, ref_m = got & 0x7F, ref & 0x7F
    assert bool((got_m != 0x7F).all()) and bool((ref_m != 0x7F).all())  # no NaN code
    same_sign = (got & 0x80) == (ref & 0x80)
    one_step = ((got_m - ref_m).abs() <= 1) & same_sign
    zero_and_smallest = (torch.minimum(got_m, ref_m) == 0) & (torch.maximum(got_m, ref_m) <= 1)
    assert bool((one_step | zero_and_smallest).all())
    if length > 1:
        assert float((got == ref).double().mean(dim=1).min()) > 0.9


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("rows", [3, 6])
def test_down_gated_residual(scenario, weights, branch, length, rows):
    ops = scenario[0]
    x = _inputs(length, rows)
    weight, weight_scale = weights["down"]
    down_scale, _ = _scale(0.05)
    activation = _quantize(x["randn"](length, UP, std=2.0).to(BF16), down_scale)
    residual = ops.down(
        activation,
        weight,
        down_scale,
        weight_scale,
        x["hidden"],
        x["modulation"],
        x["indices"],
    )
    assert branch == {"fused": 1, "unfused": 0}
    projected = _gemm(activation, weight, down_scale, weight_scale).reshape_as(x["hidden"])
    gate = x["modulation"][:, 5 * D : 6 * D].index_select(0, x["indices"])
    expected = torch.addcmul(x["hidden"].float(), gate.float(), projected.float()).to(BF16)
    assert _relative_l2(residual, expected) < 1e-4
    assert _same_fraction(residual, expected) > 0.999
    if length >= 1000:
        assert torch.equal(residual.view(torch.uint8), expected.view(torch.uint8))


def test_one_projection_kernel_serves_every_length(scenario, weights, branch):
    ops, _, _, projection_residual = scenario
    weight, weight_scale = weights["down"]
    down_scale, _ = _scale(0.05)
    before = dict(projection_residual._compiled)
    for length in (7, 513, 2049):
        x = _inputs(length, rows=3)
        activation = _quantize(x["randn"](length, UP).to(BF16), down_scale)
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
    assert branch == {"fused": 3, "unfused": 0}


@pytest.mark.parametrize("padded", (False, True), ids=("full", "padded"))
@pytest.mark.parametrize("length", (1, 300, 4097, 38169))
def test_bf16_block_attention_cast_matches_the_family_route(scenario, monkeypatch, length, padded):
    """A block the checkpoint keeps in BF16 feeds its BF16 QKV projection to the scenario's
    QK-norm, RoPE and FP8 cast; the codes and descales are those of the family route's op."""
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1 import modeling
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
        fused_fp8_qk,
    )

    # The scenario's kernel settings, pinned here because no scenario is selected in this process.
    monkeypatch.setattr(fused_fp8_qk, "_MODE", "qkstatic")
    monkeypatch.setattr(fused_fp8_qk, "_V2", True)
    monkeypatch.setattr(fused_fp8_qk, "_VAMAX_FOLD", False)
    monkeypatch.setattr(fused_fp8_qk, "_split_launch", lambda: True)
    monkeypatch.setattr(fused_fp8_qk, "_REDUCE_GROUP", None)
    monkeypatch.setattr(fused_fp8_qk, "_launch_config", lambda: (8, 2))

    heads, head_dim, rot = 56, 128, 96
    randn = _inputs(length, 1)["randn"]
    qkv = randn(1, length, 3 * QK, std=2.0).to(BF16)
    weight_q = (1 + randn(head_dim, std=0.1)).to(BF16)
    weight_k = (1 + randn(head_dim, std=0.1)).to(BF16)
    position = torch.arange(length, dtype=torch.float32, device="cuda")[:, None]
    frequency = 1.0 / (10000.0 ** (torch.arange(0, rot, 2, device="cuda").float() / rot))
    angle = torch.cat([position * frequency, position * frequency], dim=-1)  # the family table
    descale = fused_fp8_qk.qk_static_descale(head_dim, qkv.device)
    valid_len = None
    if padded:
        valid_len = torch.tensor([max(length - 7, 1)], dtype=torch.int32, device="cuda")
    arguments = (qkv, weight_q, weight_k, angle.cos(), angle.sin(), descale, EPS)
    arguments += (heads, head_dim)
    block_id = torch.tensor(0, dtype=torch.int64, device="cpu")

    got = modeling.qk_norm_rope(*arguments, block_id, valid_len)
    ref = torch.ops.h3_lever8.qk_norm_rope_fp8(*arguments, 0, valid_len)
    for name, a, b in zip(("q", "k", "v", "descale"), got, ref):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        assert _same_fraction(a, b) == 1.0, name


def _misaligned(tensor):
    """The same values in contiguous storage that starts 2 bytes off any 256-byte boundary."""
    storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    view = storage[1:].view(tensor.shape)
    view.copy_(tensor)
    assert view.is_contiguous() and view.data_ptr() % 256 != 0
    return view


@pytest.mark.parametrize("case", ("other_shape", "custom_swiglu", "not_precompiled"))
def test_bf16_gate_up_preserves_framework_fallback(scenario, monkeypatch, case):
    ops, _, gate_up_swiglu, _ = scenario
    if case == "other_shape":
        activation = torch.ones((3, 16), dtype=BF16, device="cuda")
        weight = torch.full((32, 16), 0.25, dtype=BF16, device="cuda")
    else:
        activation = torch.ones((1, D), dtype=BF16, device="cuda")
        weight = torch.full((2 * UP, D), 0.002, dtype=BF16, device="cuda")
    settings = (2.0, 1.5, 0.25) if case == "custom_swiglu" else (None, None, None)

    def unexpected_launch(*_):
        pytest.fail("A BF16 kernel was launched outside its contract")

    def precompiled(_):
        return None if case == "not_precompiled" else unexpected_launch

    monkeypatch.setattr(gate_up_swiglu, "precompiled_bf16", precompiled)
    monkeypatch.setattr(gate_up_swiglu, "_get_compiled", unexpected_launch)
    unit = torch.ones(1, device="cuda", dtype=torch.float32)
    out = ops.gate_up_bf16(activation, weight, unit, *settings)
    ref = ops.swiglu(
        F.linear(activation, weight),
        swiglu_limit=settings[0],
        swiglu_alpha=settings[1],
        swiglu_beta=settings[2],
    )
    assert torch.equal(out, ref)


def test_bf16_attention_projection_rejects_a_reshape_that_changes_rows(scenario):
    ops = scenario[0]
    tensors = _inputs(2, rows=1)
    raw = torch.empty((1, 4, QK // 2), device="cuda", dtype=BF16)
    weight = torch.empty((D, QK), device="cuda", dtype=BF16)
    unit = torch.ones(1, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected matching attention"):
        ops.bf16_attention_projection(
            raw,
            weight,
            unit,
            tensors["hidden"],
            tensors["norm_weight"],
            tensors["modulation"],
            tensors["indices"],
            EPS,
        )


@pytest.mark.parametrize("projection", ("down", "attention"))
def test_bf16_projection_rejects_fp8_operands(scenario, projection):
    ops = scenario[0]
    tensors = _inputs(2, rows=1)
    unit = torch.ones(1, device="cuda", dtype=torch.float32)
    fp8 = torch.float8_e4m3fn
    if projection == "down":
        activation = torch.empty((2, UP), device="cuda", dtype=fp8)
        weight = torch.empty((D, UP), device="cuda", dtype=fp8)
        operation = ops.bf16_down
        args = (
            activation,
            weight,
            unit,
            tensors["hidden"],
            tensors["modulation"],
            tensors["indices"],
        )
    else:
        activation = torch.empty((1, 2, QK), device="cuda", dtype=fp8)
        weight = torch.empty((D, QK), device="cuda", dtype=fp8)
        operation = ops.bf16_attention_projection
        args = (
            activation,
            weight,
            unit,
            tensors["hidden"],
            tensors["norm_weight"],
            tensors["modulation"],
            tensors["indices"],
            EPS,
        )
    with pytest.raises(ValueError, match="BF16 projection requires BF16 activation and weight"):
        operation(*args)


def test_fp8_down_does_not_use_a_bf16_compiled_kernel(scenario, monkeypatch):
    ops, _, _, projection_residual = scenario
    hidden = _inputs(2, rows=1)
    activation = torch.empty((2, UP), device="cuda", dtype=BF16)
    weight = torch.empty((D, UP), device="cuda", dtype=BF16)
    unit = torch.ones(1, device="cuda", dtype=torch.float32)

    def expected_fallback(*_):
        raise ValueError("Reached the existing scaled GEMM fallback")

    monkeypatch.setattr(projection_residual, "supports", lambda *_: True)
    monkeypatch.setattr(ops, "_cublas_scaled_mm", expected_fallback)
    with pytest.raises(ValueError, match="Reached the existing scaled GEMM fallback"):
        ops.down(
            activation,
            weight,
            unit,
            unit,
            hidden["hidden"],
            hidden["modulation"],
            hidden["indices"],
        )


@pytest.mark.parametrize("length", (300, 4097))
@pytest.mark.parametrize("case", ("misaligned_hidden", "int32_indices", "fp32_modulation"))
def test_down_outside_the_kernel_contract_takes_the_unfused_path(
    scenario, weights, branch, length, case
):
    """Inputs the compiled kernel does not serve give the fused kernel's result through cuBLAS."""
    ops, _, _, projection_residual = scenario
    x = _inputs(length, rows=3)
    weight, weight_scale = weights["down"]
    down_scale, _ = _scale(0.05)
    activation = _quantize(x["randn"](length, UP, std=2.0).to(BF16), down_scale)
    hidden, modulation, indices = x["hidden"], x["modulation"], x["indices"]
    fused = ops.down(activation, weight, down_scale, weight_scale, hidden, modulation, indices)
    assert branch == {"fused": 1, "unfused": 0}
    if case == "misaligned_hidden":
        hidden = _misaligned(hidden)
    elif case == "int32_indices":
        indices = indices.int()
    else:
        modulation = modulation.float()
    before = dict(projection_residual._compiled)
    unfused = ops.down(activation, weight, down_scale, weight_scale, hidden, modulation, indices)
    assert branch == {"fused": 1, "unfused": 1}
    assert projection_residual._compiled == before  # no projection kernel compiles at run time
    assert unfused.shape == fused.shape and unfused.dtype == fused.dtype
    assert _relative_l2(unfused, fused) < 1e-4
    assert _same_fraction(unfused, fused) > 0.999
    if length >= 1000:
        assert torch.equal(unfused.view(torch.uint8), fused.view(torch.uint8))


@pytest.mark.parametrize("length", (300, 4097))
def test_attention_projection_outside_the_kernel_contract_takes_the_unfused_path(
    scenario, weights, branch, length
):
    ops, fused_adaln, _, _ = scenario
    x = _inputs(length, rows=3)
    weight, weight_scale = weights["out"]
    raw = x["randn"](length, QK, std=1.5).to(BF16).reshape(1, length, QK)
    input_scale, input_reciprocal = _scale(0.02)
    up_scale, up_reciprocal = _scale(0.027)

    def run(hidden):
        return ops.attention_projection(
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

    fused_residual, fused_codes = run(x["hidden"])
    assert branch == {"fused": 1, "unfused": 0}
    residual, codes = run(_misaligned(x["hidden"]))
    assert branch == {"fused": 1, "unfused": 1}
    assert _relative_l2(residual, fused_residual) < 1e-4
    assert _same_fraction(residual, fused_residual) > 0.999
    assert _same_fraction(codes, fused_codes) > 0.999
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
    assert _relative_l2(residual, expected_residual) < 1e-4
    assert _same_fraction(codes, _quantize(expected_norm, up_scale)) > 0.999


def test_a_weight_the_kernel_cannot_tile_is_rejected_at_compile(scenario):
    projection_residual = scenario[3]
    before = dict(projection_residual._compiled)
    with pytest.raises(ValueError, match="does not tile"):
        projection_residual.compile_kernel(
            torch.cuda.current_device(), 5000, UP, projection_residual.DOWN_GATE, BF16, torch.int64
        )
    assert projection_residual._compiled == before


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


def test_the_unfused_path_is_reported_once(scenario, weights, branch, monkeypatch):
    ops = scenario[0]
    warnings = []
    monkeypatch.setattr(ops, "_UNFUSED_REPORTED", set())
    monkeypatch.setattr(ops.logger, "warning", warnings.append)
    x = _inputs(300, rows=3)
    weight, weight_scale = weights["down"]
    down_scale, _ = _scale(0.05)
    activation = _quantize(x["randn"](300, UP, std=2.0).to(BF16), down_scale)
    arguments = (activation, weight, down_scale, weight_scale)
    ops.down(*arguments, x["hidden"], x["modulation"], x["indices"])
    assert warnings == []
    for _ in range(3):
        ops.down(*arguments, _misaligned(x["hidden"]), x["modulation"], x["indices"])
    assert branch == {"fused": 1, "unfused": 3}
    assert len(warnings) == 1 and "Down projection" in warnings[0]
