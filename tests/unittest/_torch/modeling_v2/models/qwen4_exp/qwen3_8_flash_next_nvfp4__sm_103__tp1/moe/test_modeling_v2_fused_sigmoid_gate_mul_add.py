# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fused_sigmoid_gate_mul_add catalog entry.

The entry owns the contract. This file turns a cell into operands, locks both
sides of the in-place write, and holds the claims no cell carries: gate layouts
bitwise interchangeable, aliasing, the empty batch, the wrong-variant
discrimination and the loud rejections.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fused_sigmoid_gate_mul_add import (  # noqa: E501
    fused_sigmoid_gate_mul_add as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fused_sigmoid_gate_mul_add requires a CUDA device"

DEV = "cuda"
HIDDEN = 2560
PAD = 64


def _operands(spec, seed):
    """`(final, gate, shared, output, buffers)`; `buffers` holds the padded
    allocations a `dest="padded"` cell writes a window of."""
    t, h, dt = spec["tokens"], spec["hidden"], spec["dtype"]
    g = torch.Generator(device=DEV).manual_seed(seed)

    def rand(*shape, dtype=dt):
        return torch.randn(*shape, device=DEV, generator=g).to(dtype)

    buffers = {}
    if spec["dest"] == "padded":
        buffers = {k: rand(t + 2, h + PAD) for k in ("final", "shared", "output")}
        final, shared, output = (buffers[k][:t, :h] for k in ("final", "shared", "output"))
        assert output.stride() == (h + PAD, 1) and not output.is_contiguous()
    else:
        final, shared = rand(t, h), rand(t, h)
        output = torch.empty_like(final) if spec["dest"] == "output" else None
    gate = rand(t, 1, dtype=spec["gate_dtype"])
    layout = spec["gate_layout"]
    if layout == "flat":
        gate = gate.reshape(-1)
    elif layout == "row":
        gate = gate.reshape(1, t)
    elif layout == "strided":
        wide = rand(t, 4, dtype=spec["gate_dtype"])
        wide[:, 2:3] = gate
        gate = wide[:, 2:3]
        assert not gate.is_contiguous()
    return final, gate, shared, output, buffers


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    final, gate, shared, output, buffers = _operands(cell.spec, zlib.crc32(cell.why.encode()))
    before = {k: v.clone() for k, v in dict(final=final, gate=gate, shared=shared).items()}
    snapshots = {k: v.clone() for k, v in buffers.items()}
    ref = op.reference(before["final"], gate, shared, output)

    out = op(final, gate, shared, output)

    destination = final if output is None else output
    assert out.data_ptr() == destination.data_ptr(), "the op returns what it wrote"
    assert out.dtype == final.dtype and out.shape == final.shape
    torch.testing.assert_close(out, ref)
    assert torch.equal(gate, before["gate"]) and torch.equal(shared, before["shared"])
    if output is not None:
        assert torch.equal(final, before["final"]), "an output buffer leaves the input alone"
    if buffers:
        t, h = cell.spec["tokens"], cell.spec["hidden"]
        o, o0 = buffers["output"], snapshots["output"]
        assert torch.equal(o[:t, h:], o0[:t, h:]), "wrote into the row padding"
        assert torch.equal(o[t:], o0[t:]), "wrote past num_tokens"
        assert torch.equal(buffers["final"], snapshots["final"])
        assert torch.equal(buffers["shared"], snapshots["shared"])


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_gate_layouts_are_the_same_call_bit_for_bit():
    spec = dict(
        tokens=33,
        hidden=HIDDEN,
        dtype=torch.bfloat16,
        gate_dtype=torch.bfloat16,
        gate_layout="column",
        dest="output",
    )
    final, gate, shared, _, _ = _operands(spec, 3)
    wide = torch.zeros(33, 4, dtype=torch.bfloat16, device=DEV)
    wide[:, 2:3] = gate
    results = [
        op(final.clone(), g, shared)
        for g in (gate, gate.reshape(-1), gate.reshape(1, 33), wide[:, 2:3])
    ]
    for r in results[1:]:
        assert torch.equal(r, results[0])


def test_the_output_buffer_may_alias_the_shared_expert_output():
    spec = dict(
        tokens=17,
        hidden=HIDDEN,
        dtype=torch.bfloat16,
        gate_dtype=torch.bfloat16,
        gate_layout="column",
        dest="inplace",
    )
    final, gate, shared, _, _ = _operands(spec, 8)
    expected = op.reference(final, gate, shared)
    returned = op(final, gate, shared, output=shared)
    assert returned.data_ptr() == shared.data_ptr()
    torch.testing.assert_close(shared, expected)


def test_zero_tokens_is_a_no_op():
    final = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=DEV)
    shared = torch.empty(0, HIDDEN, dtype=torch.bfloat16, device=DEV)
    gate = torch.empty(0, 1, dtype=torch.bfloat16, device=DEV)
    returned = op(final, gate, shared)
    assert returned.shape == (0, HIDDEN) and returned.data_ptr() == final.data_ptr()


def test_wrong_variants_land_far_outside_the_default_gate():
    """The default bf16 band (atol 1e-5, rtol 1.6e-2) separates the correct
    combine from every plausible wrong one by more than 100x."""
    spec = dict(
        tokens=17,
        hidden=HIDDEN,
        dtype=torch.bfloat16,
        gate_dtype=torch.bfloat16,
        gate_layout="column",
        dest="inplace",
    )
    final, gate, shared, _, _ = _operands(spec, 9)
    correct = op.reference(final, gate, shared)
    actual = op(final.clone(), gate, shared)
    width = 1e-5 + 1.6e-2 * correct.float().abs()
    s = torch.sigmoid(gate.reshape(-1, 1).float())
    f, sh = final.float(), shared.float()

    def ratio(wrong: torch.Tensor) -> float:
        return ((wrong.float() - correct.float()).abs() / width).max().item()

    assert ratio(actual) <= 1.0
    for name, wrong in {
        "no sigmoid, raw gate": f + gate.float().reshape(-1, 1) * sh,
        "gate applied to the routed half": s * f + sh,
        "one gate for the whole batch": f + s.reshape(-1)[0] * sh,
        "multiply instead of add": f * s * sh,
        "gate dropped, plain add": f + sh,
    }.items():
        r = ratio(wrong.to(torch.bfloat16))
        print(f"  {name:36s} {r:10.1f}x the gate")
        assert r > 100.0, f"{name} is indistinguishable from the correct result"


@pytest.mark.parametrize(
    "case",
    [
        "final_3d",
        "shared_shape_mismatch",
        "shared_dtype_mismatch",
        "gate_numel_too_small",
        "gate_numel_too_large",
        "output_shape_mismatch",
        "output_dtype_mismatch",
        "output_last_stride",
        "final_last_stride_without_an_output",
    ],
)
def test_metadata_misuse_raises(case: str) -> None:
    """Every metadata violation is loud, which is why the entry has no guard."""
    bf = dict(dtype=torch.bfloat16, device=DEV)
    final, shared, gate = (
        torch.randn(4, 128, **bf),
        torch.randn(4, 128, **bf),
        torch.randn(4, 1, **bf),
    )
    output = None
    if case == "final_3d":
        final = torch.randn(2, 2, 128, **bf)
    elif case == "shared_shape_mismatch":
        shared = torch.randn(4, 64, **bf)
    elif case == "shared_dtype_mismatch":
        shared = torch.randn(4, 128, dtype=torch.float16, device=DEV)
    elif case == "gate_numel_too_small":
        gate = torch.randn(3, 1, **bf)
    elif case == "gate_numel_too_large":
        gate = torch.randn(5, 1, **bf)
    elif case == "output_shape_mismatch":
        output = torch.empty(4, 64, **bf)
    elif case == "output_dtype_mismatch":
        output = torch.empty(4, 128, dtype=torch.float16, device=DEV)
    elif case == "output_last_stride":
        output = torch.empty(4, 256, **bf)[:, ::2]
    elif case == "final_last_stride_without_an_output":
        final = torch.randn(4, 256, **bf)[:, ::2]
    with pytest.raises(AssertionError):
        op(final, gate, shared, output=output)


def test_cpu_operands_are_rejected_by_triton() -> None:
    final = torch.randn(4, 128, dtype=torch.bfloat16)
    shared = torch.randn(4, 128, dtype=torch.bfloat16)
    gate = torch.randn(4, 1, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Pointer argument cannot be accessed from Triton"):
        op(final, gate, shared)
