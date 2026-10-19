# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rms_norm_gated_grouped entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.rms_norm_gated_grouped import (  # noqa: E501
    PROJ_WIDTH,
    WIDTH,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.rms_norm_gated_grouped import (  # noqa: E501
    rms_norm_gated_grouped as op,
)
from tensorrt_llm._torch.modules.mamba.layernorm_gated import _layer_norm_fwd

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "rms_norm_gated_grouped requires a CUDA device"

#: The band, in `assert_within_ulp`'s unit: 2^-8 of each row's largest
#: magnitude. Kernel and reference both form the whole product in fp32 and
#: round once, so they can differ only where the reduction order moves the
#: fp32 value across a rounding boundary: by one output ulp. One bf16 ulp of an
#: element in the row maximum's binade is up to 2^-7 of that maximum, i.e. 2
#: units -- ELEMENT_ULP. Measured over every cell and seed tried: at most 1.04
#: units, on at most 1.5e-5 of the elements; relative RMS at most 0.0045
#: units. RMS_ULP is ten times that, and thirteen times below the 0.64-0.72
#: units HF's bf16-after-cast weight multiply lands at (its element error
#: reaches the same 2-unit bound, so the RMS is the gate that tells the two
#: conventions apart).
ELEMENT_ULP = 2.0
RMS_ULP = 0.05


def _backing(t: torch.Tensor) -> torch.Tensor:
    return t if t._base is None else t._base


def _build(spec, seed):
    """`__call__`'s arguments for one spec; z a view of the projection when the spec says so."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, width, group_size = spec["tokens"], spec["width"], spec["group_size"]

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda")

    x = randn(tokens, width) * spec.get("x_scale", 1.0)
    if "group_scales" in spec:
        scales = torch.tensor(spec["group_scales"], device="cuda")
        x = x * scales.repeat_interleave(group_size)
    x = x.to(torch.bfloat16)
    if spec["z"] == "proj":
        proj_width = spec["proj_width"]
        z = randn(tokens, proj_width).to(torch.bfloat16)[:, :width]
        assert z.stride() == (proj_width, 1)
    else:
        z = randn(tokens, width).to(torch.bfloat16)
    # The checkpoint's gated-norm weights reach 13.25; this spans that.
    weight = (randn(width) * 4).to(torch.bfloat16)
    return [x, z, weight, spec["eps"], group_size]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    x_before, z_buffer_before = args[0].clone(), _backing(args[1]).clone()
    out = op(*args)
    assert out.shape == args[0].shape and out.dtype == args[0].dtype and out.is_contiguous()
    assert torch.equal(args[0], x_before), "x is read-only"
    assert torch.equal(_backing(args[1]), z_buffer_before), (
        "z and the projection it views are read-only"
    )
    assert_within_ulp(out, op.reference(*args), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def _hf_gated_norm(x, z, weight, eps, group_size):
    """HF's NemotronH / Zamba2 gated norm: the normalized value is cast to the
    activation dtype, then multiplied by the bf16 weight in bf16."""
    tokens, width = x.shape
    zf = z.float()
    g = (x.float() * (zf * torch.sigmoid(zf))).view(tokens, width // group_size, group_size)
    g = g * torch.rsqrt(g.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * g.view(tokens, width).to(x.dtype)


def _spec(prefix: str):
    return next(c.spec for c in op.CELLS if c.why.startswith(prefix))


def test_the_weight_multiply_is_fp32_not_hf_bf16() -> None:
    """Discrimination: the kernel multiplies by the weight in fp32 before its
    one rounding (`reference`), not after a cast to bf16 as HF does -- and the
    band is tight enough that the HF convention fails it."""
    args = _build(_spec("prefill 4101"), seed=41)
    out = op(*args)
    assert_within_ulp(out, op.reference(*args), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)
    with pytest.raises(AssertionError):
        assert_within_ulp(out, _hf_gated_norm(*args), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def test_groups_are_normalized_independently() -> None:
    """Discrimination: one statistic over the whole 4096-wide row (group_size
    = N) misses the grouped reference."""
    args = _build(_spec("group scales 100x apart"), seed=42)
    whole_row = op(*args[:4], WIDTH)
    with pytest.raises(AssertionError):
        assert_within_ulp(whole_row, op.reference(*args), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def test_the_gate_applies_before_the_norm() -> None:
    """Discrimination: the same kernel with the gate after the norm (its other
    mode, GDN's) misses the reference by orders of magnitude."""
    args = _build(_spec("decode batch 32"), seed=43)
    x, z, weight, eps, group_size = args
    gate_after, _, _ = _layer_norm_fwd(
        x, weight, None, eps, z=z, group_size=group_size, norm_before_gate=True, is_rms_norm=True
    )
    with pytest.raises(AssertionError):
        assert_within_ulp(gate_after, op.reference(*args), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


def test_cuda_graph_replay_reads_inputs_on_device() -> None:
    """Decode runs captured: the replayed kernel reads x and z where they live
    at replay time and writes the output the capture allocated."""
    args = _build(_spec("decode batch 32"), seed=44)
    x, z = args[0], args[1]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args)  # first launch (and Triton compile) outside capture
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    g = torch.Generator(device="cuda").manual_seed(45)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, generator=g, device="cuda").to(x.dtype))
        z.copy_(torch.randn(z.shape, generator=g, device="cuda").to(z.dtype))
        graph.replay()
        torch.cuda.synchronize()
        assert_within_ulp(out, op.reference(*args), element_ulp=ELEMENT_ULP, rms_ulp=RMS_ULP)


#: case -> (what the op raises, a fragment of its message). The shape and
#: stride checks are bare asserts in `_layer_norm_fwd`: no text to match.
_REJECTIONS = {
    "x_column_stride": (AssertionError, None),
    "z_column_stride": (AssertionError, None),
    "z_shape": (AssertionError, None),
    "weight_shape": (AssertionError, None),
    "weight_stride": (AssertionError, None),
    "group_does_not_divide": (AssertionError, None),
    "group_wider_than_64kb": (RuntimeError, "64KB"),
}


@pytest.mark.parametrize("case", sorted(_REJECTIONS))
def test_op_rejects_every_metadata_violation_loudly(case: str) -> None:
    """Why the entry has no `is_valid`: every metadata violation probed raises."""
    spec = dict(tokens=8, width=WIDTH, group_size=2048, eps=1e-5, z="proj", proj_width=PROJ_WIDTH)
    if case == "group_wider_than_64kb":
        spec.update(width=65536, group_size=65536, z="contiguous")
    args = _build(spec, seed=46)
    x, z, weight = args[0], args[1], args[2]
    if case == "x_column_stride":
        args[0] = torch.repeat_interleave(x, 2, dim=1)[:, ::2]
    elif case == "z_column_stride":
        args[1] = torch.repeat_interleave(z, 2, dim=1)[:, ::2]
    elif case == "z_shape":
        args[1] = z[:, : WIDTH // 2]
    elif case == "weight_shape":
        args[2] = weight[: WIDTH // 2]
    elif case == "weight_stride":
        args[2] = torch.repeat_interleave(weight, 2)[::2]
    elif case == "group_does_not_divide":
        args[4] = 1000
    error, fragment = _REJECTIONS[case]
    with pytest.raises(error, match=fragment):
        with validating(op):
            op(*args)
