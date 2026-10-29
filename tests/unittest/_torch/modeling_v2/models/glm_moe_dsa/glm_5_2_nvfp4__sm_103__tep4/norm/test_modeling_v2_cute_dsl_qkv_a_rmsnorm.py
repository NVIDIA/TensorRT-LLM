# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 target's cute_dsl_qkv_a_rmsnorm entry."""

import zlib

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
    pytest.skip("the vendored CuTe DSL kernel needs an sm_103 GPU", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.ops.norm import (  # noqa: E402, E501
    cute_dsl_qkv_a_rmsnorm as entry,
)

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

op = entry.cute_dsl_qkv_a_rmsnorm
_Q, _KV, _ROW = 2048, 512, 2624


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """2 ulp per element against the row scale, 0.5 ulp relative RMS.

    Both sides round to bf16 once. The kernel's fp32 path differs from the
    fp64 reference by its sum-of-squares order and an approximate rsqrt, a few
    fp32 ulps in all, so the two bf16 results differ only where the fp32 value
    sits that close to a bf16 rounding midpoint: one bf16 ulp of that element.
    One ulp of an element at the row's largest magnitude reads as up to 2
    units of `ULP[bf16]` against the row scale, hence 2. Such flips are rare
    (on the order of 1e-3 of elements), so their RMS sits far under half an
    ulp; a wrong eps, weight or row statistic moves every element by much
    more.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


def _build(spec, seed):
    """A `[rows, 2624]` projection output and its q / kv views, as the target slices it."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["rows"]
    parent = torch.randn(rows, _ROW, generator=g, device="cuda").to(torch.bfloat16)
    q_w = (1.0 + 0.25 * torch.randn(_Q, generator=g, device="cuda")).to(torch.bfloat16)
    kv_w = (1.0 + 0.25 * torch.randn(_KV, generator=g, device="cuda")).to(torch.bfloat16)
    q_raw, ckv_raw, _ = torch.split(parent, [_Q, _KV, _ROW - _Q - _KV], -1)
    return parent, q_raw, q_w, ckv_raw, kv_w


@pytest.fixture(scope="module")
def compiled():
    return entry.compile(torch.cuda.current_device(), 1e-5)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, compiled, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    eps = cell.spec["eps"]
    parent, q_raw, q_w, ckv_raw, kv_w = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    before = parent.clone()
    # The reference first: the op overwrites ckv_raw.
    ref_q, ref_kv = op.reference(q_raw, q_w, ckv_raw, kv_w, eps, compiled)

    q_a, ckv = op(q_raw, q_w, ckv_raw, kv_w, eps, compiled)

    assert ckv is ckv_raw, "the kv result must be the caller's view, normalised in place"
    assert q_a.is_contiguous() and q_a.shape == (cell.spec["rows"], _Q)
    assert q_a.data_ptr() != parent.data_ptr(), "q_a is a new tensor, not the projection row"
    _compare(q_a, ref_q)
    _compare(parent[:, _Q : _Q + _KV], ref_kv)
    # q_raw and k_pe are read, never written: the latent's rope tail must be the
    # projection's, bit for bit.
    assert torch.equal(parent[:, :_Q], before[:, :_Q]), "q_raw was written"
    assert torch.equal(parent[:, _Q + _KV :], before[:, _Q + _KV :]), "k_pe was written"


def test_guard_refuses_an_eps_the_handle_was_not_compiled_with(compiled) -> None:
    """The kernel uses the eps baked into the handle and ignores the one passed."""
    _, q_raw, q_w, ckv_raw, kv_w = _build(dict(rows=4), seed=1)
    with pytest.raises(AssertionError, match="baked in"):
        with validating(op):
            op(q_raw, q_w, ckv_raw, kv_w, 1e-6, compiled)


def test_guard_refuses_a_compact_destination_handle() -> None:
    """Compiled for compact outputs, the kernel writes kv at row pitch 512 into
    a buffer whose pitch is 2624: the wrong elements, no error."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tep4.kernels.norm import (  # noqa: E501
        self_attn_qkv_a_rmsnorm as kernel,
    )

    compact = kernel.compile_qkv_a_rmsnorm(torch.cuda.current_device(), 1e-5, inplace_kv_view=False)
    _, q_raw, q_w, ckv_raw, kv_w = _build(dict(rows=4), seed=2)
    with pytest.raises(AssertionError, match="compact outputs"):
        with validating(op):
            op(q_raw, q_w, ckv_raw, kv_w, 1e-5, compact)


def test_guard_refuses_a_foreign_handle() -> None:
    _, q_raw, q_w, ckv_raw, kv_w = _build(dict(rows=4), seed=3)
    with pytest.raises(AssertionError, match="this entry's compile"):
        with validating(op):
            op(q_raw, q_w, ckv_raw, kv_w, 1e-5, lambda *a: None)
