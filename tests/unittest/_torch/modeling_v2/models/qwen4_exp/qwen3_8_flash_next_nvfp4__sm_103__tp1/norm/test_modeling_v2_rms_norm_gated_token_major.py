# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rms_norm_gated_token_major entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.norm.rms_norm_gated_token_major import (  # noqa: E501
    rms_norm_gated_token_major as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "rms_norm_gated_token_major requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """fp8 output: byte-exact, as measured. Otherwise torch's default band."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    if out.dtype == torch.float8_e4m3fn:
        assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8)), (
            "fp8 output is byte-identical to the unfused reference"
        )
        return
    torch.testing.assert_close(out, ref)


#: The target's concatenated GDN input projection: [q k v | z | a | b] = 10240 | 6144 | 48 | 48.
_IN_PROJ_WIDTH, _Z_OFFSET = 16480, 10240


#: The checkpoint's GDN epilogue geometry over a packed `z`, for the non-cell tests.
_PACKED = dict(op.CELLS[0].spec, tokens=17, z="packed")


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, n, dtype = spec["tokens"], spec["heads"], spec["width"], spec["dtype"]

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    x = randn(t * heads, n)

    layout = spec.get("z", "packed")
    if layout == "in_proj":
        z = randn(t, _IN_PROJ_WIDTH)[:, _Z_OFFSET : _Z_OFFSET + heads * n].view(t, heads, n)
    else:
        assert layout == "packed", layout
        z = randn(t, heads, n)

    weight = randn(n)
    fp8_scale = None
    if "fp8_scale" in spec:
        fp8_scale = torch.tensor([spec["fp8_scale"]], dtype=torch.float32, device="cuda")
    return x, z, weight, spec["eps"], fp8_scale, spec["gate_activation"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell.why))
    x, z, weight, _, fp8_scale, _ = args
    if cell.spec.get("z") == "in_proj" and cell.spec["tokens"] > 1:
        assert z.stride() == (_IN_PROJ_WIDTH, cell.spec["width"], 1) and not z.is_contiguous()
    before = [x.clone(), z.clone(), weight.clone()]

    out = op(*args)

    assert out.shape == x.shape and out.is_contiguous()
    assert out.dtype == (torch.float8_e4m3fn if fp8_scale is not None else x.dtype)
    for t, b in zip((x, z, weight), before):
        assert torch.equal(t, b), "inputs are read-only"
    if fp8_scale is not None:
        # ... and no byte is the e4m3 NaN pattern, so the byte gate compares numbers.
        assert torch.isfinite(out.float()).all()
    _compare(out, op.reference(*args))


def test_fp8_byte_gate_discriminates() -> None:
    """Each wrong variant of the reference misses the byte gate on most elements."""
    spec = dict(op.CELLS[-1].spec, fp8_scale=0.0234375)
    x, z, weight, eps, fp8_scale, act = _build(spec, seed=8)
    out = op(x, z, weight, eps, fp8_scale, act)
    _compare(out, op.reference(x, z, weight, eps, fp8_scale, act))

    gated_x = x * torch.sigmoid(z.reshape(x.shape).float()).to(x.dtype)
    wrong = {
        "sigmoid gate instead of silu": op.reference(x, z, weight, eps, fp8_scale, "sigmoid"),
        "gate applied before the norm": op.reference(gated_x, z, weight, eps, fp8_scale, act),
        "weight omitted": op.reference(x, z, torch.ones_like(weight), eps, fp8_scale, act),
    }
    wrong["scale multiplied instead of divided"] = (
        op.reference(x, z, weight, eps, None, act).float() * fp8_scale
    ).to(torch.float8_e4m3fn)
    for name, other in wrong.items():
        matched = (out.view(torch.uint8) == other.view(torch.uint8)).float().mean().item()
        assert matched < 0.5, f"{name}: byte-identical on {matched:.4f} of elements"


def test_the_two_gates_differ() -> None:
    """A kernel ignoring `gate_activation` would pass the sigmoid cells; the two gates differ."""
    x, z, weight, eps, _, _ = _build(_PACKED, seed=3)
    assert not torch.equal(
        op(x, z, weight, eps, None, "silu"), op(x, z, weight, eps, None, "sigmoid")
    )


def test_guard_refuses_a_weight_of_the_wrong_length() -> None:
    """Unguarded, a length-192 weight is silently truncated to weight[:128]."""
    x, z, weight, eps, _, act = _build(dict(_PACKED, tokens=4), seed=12)
    long_weight = torch.cat([weight, torch.randn(64, dtype=weight.dtype, device="cuda")])
    assert torch.equal(
        torch.ops.trtllm.rms_norm_gated_token_major(x, z, long_weight, eps, None, act),
        torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, None, act),
    ), "the hazard this guard covers is gone"

    for bad in (long_weight, weight[:64]):
        with pytest.raises(AssertionError, match="weight must be 1-D of length"):
            with validating(op):
                op(x, z, bad, eps, None, act)


def test_guard_refuses_a_malformed_fp8_scale() -> None:
    """Unguarded, a two-element scale quantizes against element 0 and a bf16
    scale is accepted -- neither raises."""
    x, z, weight, eps, _, act = _build(dict(_PACKED, tokens=4), seed=12)
    one = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    two = torch.tensor([0.5, 0.25], dtype=torch.float32, device="cuda")
    half = torch.tensor([0.5], dtype=torch.bfloat16, device="cuda")
    by_two = torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, two, act)
    by_one = torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, one, act)
    assert torch.equal(by_two.view(torch.uint8), by_one.view(torch.uint8))
    by_half = torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, half, act)
    assert by_half.dtype == torch.float8_e4m3fn

    with pytest.raises(AssertionError, match="fp8_scale must hold exactly one element"):
        with validating(op):
            op(x, z, weight, eps, two, act)
    with pytest.raises(AssertionError, match="fp8_scale must be float32"):
        with validating(op):
            op(x, z, weight, eps, half, act)


def test_rejects_unsupported_gate_activation() -> None:
    x, z, weight, eps, _, _ = _build(dict(_PACKED, tokens=4), seed=10)
    with pytest.raises(ValueError, match="gate_activation must be 'silu' or 'sigmoid'"):
        op(x, z, weight, eps, None, "gelu")


def test_rejects_z_that_does_not_cover_x() -> None:
    x, z, weight, eps, _, act = _build(dict(_PACKED, tokens=4, heads=8), seed=11)
    with pytest.raises(AssertionError, match="does not match x shape"):
        op(x, z[:, :7, :].contiguous(), weight, eps, None, act)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        op(x, z.reshape(32, 128), weight, eps, None, act)


def test_prefill_launch_rejects_z_that_does_not_cover_x() -> None:
    """From 32768 rows the op runs its own multi-row launch; a `z` one token short
    still reaches the registered op's check instead of being read past its end."""
    x, z, weight, eps, _, act = _build(dict(_PACKED, tokens=683), seed=12)
    assert x.shape[0] >= 32768, x.shape
    with pytest.raises(AssertionError, match="does not match x shape"):
        op(x, z[:-1], weight, eps, None, act)
