# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pre-QKV prenorm ordering switch (MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL).

The switch selects between two well-defined numerics orderings at the
pre-QKV boundary of later layers: the historical one rounds the updated FP32
residual to BF16 before the norm reads it; the reference one computes the
norm on the FP32 residual and rounds to BF16 once, on the output. The two
coincide exactly when the residual is BF16-representable (layer 0) and
differ at BF16 grain otherwise. These tests pin the flag's default, the arm
selection in all three implementations (torch fallback, fused Triton,
CuteDSL cluster boundary), and the ordering properties themselves.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenGatedResidual,
    _preqkv_norm_fp32_residual_enabled,
    _rmsnorm_fp32_in,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _gemma_norm_reference(x_fp32: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 gemma RMSNorm chain with one final BF16 round (reference ordering)."""
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    normed = x_fp32 * torch.rsqrt(variance + eps)
    return (normed * (weight.to(torch.float32) + 1.0)).to(torch.bfloat16)


def _cast_first_reference(x_fp32: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Historical ordering: round the residual to BF16 before the norm."""
    return _gemma_norm_reference(x_fp32.to(torch.bfloat16).to(torch.float32), weight, eps)


def _nonrepresentable_residual(tokens: int, hidden: int, device: str) -> torch.Tensor:
    """FP32 residual with a component BF16 cannot carry, like the stream after
    one or more FP32 gated adds."""
    torch.manual_seed(20260915)
    base = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16).to(torch.float32)
    noise = torch.randn(tokens, hidden, device=device, dtype=torch.float32) * 2e-3
    return base + noise


def _mismatch_fraction(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a != b).float().mean().item()


# ---------------------------------------------------------------------------
# Flag contract (host-side, no GPU)
# ---------------------------------------------------------------------------


def test_flag_defaults_off(monkeypatch):
    """Unset or falsy keeps the historical ordering: the default serving
    numerics do not change until the reference ordering is adopted."""
    monkeypatch.delenv("MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL", raising=False)
    assert not _preqkv_norm_fp32_residual_enabled()
    for value in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL", value)
        assert not _preqkv_norm_fp32_residual_enabled()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " TRUE "])
def test_flag_opt_in(monkeypatch, value):
    monkeypatch.setenv("MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL", value)
    assert _preqkv_norm_fp32_residual_enabled()


def test_flag_resolved_at_construction(monkeypatch):
    """The ordering is a property of the constructed engine, not the call-time
    environment (CUDA-graph capture would otherwise pin one arm silently)."""
    monkeypatch.setenv("MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL", "1")
    gate = MuseSparkOpenGatedResidual(64, torch.bfloat16)
    monkeypatch.setenv("MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL", "0")
    assert gate._preqkv_norm_fp32
    assert not MuseSparkOpenGatedResidual(64, torch.bfloat16)._preqkv_norm_fp32


# ---------------------------------------------------------------------------
# Ordering properties (pure torch, CPU-runnable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_orderings_differ_only_on_nonrepresentable_residual(device):
    """The audited difference pattern: the two orderings disagree at BF16
    grain on a residual carrying an FP32-only component, and coincide bitwise
    when the residual is BF16-representable (the layer-0 equivalence)."""
    hidden, tokens, eps = 8192, 64, 1e-6
    torch.manual_seed(7)
    weight = torch.randn(hidden, device=device, dtype=torch.bfloat16) * 0.02

    residual = _nonrepresentable_residual(tokens, hidden, device)
    norm_first = _gemma_norm_reference(residual, weight, eps)
    cast_first = _cast_first_reference(residual, weight, eps)
    fraction = _mismatch_fraction(norm_first, cast_first)
    # The audit probe saw ~25% of outputs move at 0.6-6% noise and ~15% at
    # 0.06%; anything clearly nonzero pins that the orderings are distinct.
    assert fraction > 0.01
    rel = (norm_first.float() - cast_first.float()).norm() / cast_first.float().norm()
    assert rel < 0.01  # BF16-grain perturbation, not a different function

    representable = residual.to(torch.bfloat16).to(torch.float32)
    assert torch.equal(
        _gemma_norm_reference(representable, weight, eps),
        _cast_first_reference(representable, weight, eps),
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda_only)])
def test_rmsnorm_fp32_in_matches_reference_chain(device):
    hidden, tokens = 512, 8
    norm = RMSNorm(hidden_size=hidden, eps=1e-6, dtype=torch.bfloat16, use_gemma=True).to(device)
    with torch.no_grad():
        norm.weight.normal_(0, 0.02)
    residual = _nonrepresentable_residual(tokens, hidden, device)
    out = _rmsnorm_fp32_in(residual, norm)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, _gemma_norm_reference(residual, norm.weight, norm.variance_epsilon))


# ---------------------------------------------------------------------------
# Arm selection: torch fallback (unfused) and fused Triton
# ---------------------------------------------------------------------------


def _case(tokens, hidden, monkeypatch, norm_fp32: bool, contiguous_branch: bool = True):
    monkeypatch.setenv("MUSESPARK_OPEN_PREQKV_NORM_FP32_RESIDUAL", "1" if norm_fp32 else "0")
    torch.manual_seed(719)
    residual = _nonrepresentable_residual(tokens, hidden, "cuda")
    branch = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    if not contiguous_branch:
        # Same values in a layout the kernels do not take, which sends the
        # join to the torch composition.
        branch = branch.t().contiguous().t()
    gate = MuseSparkOpenGatedResidual(hidden, torch.bfloat16).cuda()
    norm = RMSNorm(hidden_size=hidden, eps=1e-5, dtype=torch.bfloat16, use_gemma=True).cuda()
    with torch.no_grad():
        gate.gate.normal_(0, 0.5)
        norm.weight.normal_(0, 0.02)
    gate.cache_derived_state()
    return residual, branch, gate, norm


@cuda_only
@pytest.mark.parametrize("norm_fp32", [False, True])
def test_unfused_arm_selects_ordering(monkeypatch, norm_fp32):
    """A branch the kernels do not take: the composition must take exactly the
    selected ordering, bitwise."""
    with torch.inference_mode():
        residual, branch, gate, norm = _case(
            16, 512, monkeypatch, norm_fp32=norm_fp32, contiguous_branch=False
        )
        assert not branch.is_contiguous()
        updated, next_input = gate._add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        expected_updated = gate.add_after_rmsnorm(residual, branch, 1e-5)
        assert torch.equal(updated, expected_updated)
        if norm_fp32:
            assert torch.equal(next_input, _rmsnorm_fp32_in(updated, norm))
        else:
            assert torch.equal(next_input, norm(updated.to(torch.bfloat16)))


@cuda_only
@pytest.mark.parametrize("tokens,hidden", [(64, 8192), (16, 512)])
def test_fused_arm_residual_invariant_and_norm_switched(monkeypatch, tokens, hidden):
    """Fused Triton path (token counts/widths that dodge the cluster kernel):
    the residual stream is bitwise identical across the two orderings; the
    fused next-norm output moves with the switch and matches the matching
    torch reference orderings elementwise at BF16 grain."""
    with torch.inference_mode():
        residual, branch, gate_off, norm = _case(tokens, hidden, monkeypatch, norm_fp32=False)
        updated_off, next_off = gate_off._add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)
        _, _, gate_on, _ = _case(tokens, hidden, monkeypatch, norm_fp32=True)
        with torch.no_grad():
            gate_on.gate.copy_(gate_off.gate)
        gate_on.cache_derived_state()
        updated_on, next_on = gate_on._add_after_rmsnorm_and_norm(residual, branch, 1e-5, norm)

        assert torch.equal(updated_on, updated_off)
        # The switched output sits on its own reference ordering: closer to it
        # (in exact-match terms) than to the other ordering's reference, which
        # is the audited difference pattern. Reduction order may still flip
        # rare single BF16 ulps against the torch chain, so the assertion is
        # on the pattern, not blanket bitwise equality.
        ref_norm_first = _gemma_norm_reference(updated_off, norm.weight, norm.variance_epsilon)
        ref_cast_first = _cast_first_reference(updated_off, norm.weight, norm.variance_epsilon)
        assert _mismatch_fraction(next_on, ref_norm_first) < 0.001
        assert _mismatch_fraction(next_off, ref_cast_first) < 0.001
        assert _mismatch_fraction(next_on, ref_cast_first) > 0.01
        assert _mismatch_fraction(next_off, ref_norm_first) > 0.01


@cuda_only
def test_fused_arm_layer0_equivalence(monkeypatch):
    """With a BF16-representable updated residual (alpha=1, beta=0 fed to the
    kernel directly), the two orderings are the same function: outputs must be
    bitwise identical."""
    import triton

    from tensorrt_llm._torch.models.modeling_musespark_open import (
        _RESIDUAL_NUM_WARPS,
        _gated_residual_after_rmsnorm_kernel,
    )

    tokens, hidden = 8, 512
    torch.manual_seed(11)
    residual = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16).to(torch.float32)
    branch = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    alpha = torch.ones(hidden, device="cuda", dtype=torch.float32)
    beta = torch.zeros(hidden, device="cuda", dtype=torch.float32)
    weight = (torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()

    outputs = {}
    for norm_fp32 in (False, True):
        updated = torch.empty_like(residual)
        next_input = torch.empty_like(branch)
        _gated_residual_after_rmsnorm_kernel[(tokens,)](
            updated,
            residual,
            branch,
            alpha,
            alpha,
            beta,
            1e-5,
            next_input,
            weight,
            1e-6,
            HIDDEN=hidden,
            BLOCK_H=triton.next_power_of_2(hidden),
            HAS_GAIN=False,
            FUSE_NEXT_NORM=True,
            NORM_FP32_RESIDUAL=norm_fp32,
            num_warps=_RESIDUAL_NUM_WARPS,
        )
        outputs[norm_fp32] = (updated, next_input)
        assert torch.equal(updated, residual)  # alpha=1, beta=0
    assert torch.equal(outputs[True][1], outputs[False][1])


# ---------------------------------------------------------------------------
# CuteDSL cluster boundary
# ---------------------------------------------------------------------------


def _cluster_case(tokens):
    from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary_available

    if not cluster_boundary_available(torch.device("cuda", torch.cuda.current_device())):
        pytest.skip("CuteDSL cluster boundary requires GB300")
    hidden = 8192
    torch.manual_seed(31)
    residual = _nonrepresentable_residual(tokens, hidden, "cuda")
    branch = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    alpha = (torch.rand(hidden, device="cuda", dtype=torch.float32) * 0.2 + 0.8).contiguous()
    beta = (torch.rand(hidden, device="cuda", dtype=torch.float32) * 0.2).contiguous()
    weight = (torch.randn(hidden, device="cuda", dtype=torch.bfloat16) * 0.02).contiguous()
    return residual, branch, alpha, beta, weight


@cuda_only
@pytest.mark.parametrize("tokens", [1, 16])
def test_cluster_boundary_norm_fp32_switch(tokens):
    from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary

    with torch.inference_mode():
        residual, branch, alpha, beta, weight = _cluster_case(tokens)
        updated_off, next_off = cluster_boundary(
            residual, branch, None, alpha, beta, weight, 1e-5, 1e-6, norm_fp32_residual=False
        )
        updated_on, next_on = cluster_boundary(
            residual, branch, None, alpha, beta, weight, 1e-5, 1e-6, norm_fp32_residual=True
        )
        assert torch.equal(updated_on, updated_off)
        ref_norm_first = _gemma_norm_reference(updated_off, weight, 1e-6)
        ref_cast_first = _cast_first_reference(updated_off, weight, 1e-6)
        assert _mismatch_fraction(next_on, ref_norm_first) < 0.001
        assert _mismatch_fraction(next_off, ref_cast_first) < 0.001
        assert _mismatch_fraction(next_on, ref_cast_first) > 0.01
        assert _mismatch_fraction(next_off, ref_norm_first) > 0.01


@cuda_only
def test_cluster_boundary_layer0_equivalence():
    """alpha=1, beta=0: the updated residual is BF16-representable when the
    input residual is, and the two orderings must agree bitwise."""
    from tensorrt_llm._torch.models.musespark_open_boundary import cluster_boundary

    with torch.inference_mode():
        residual, branch, _, _, weight = _cluster_case(4)
        residual = residual.to(torch.bfloat16).to(torch.float32)
        alpha = torch.ones(8192, device="cuda", dtype=torch.float32)
        beta = torch.zeros(8192, device="cuda", dtype=torch.float32)
        results = {
            arm: cluster_boundary(
                residual, branch, None, alpha, beta, weight, 1e-5, 1e-6, norm_fp32_residual=arm
            )
            for arm in (False, True)
        }
        assert torch.equal(results[True][0], residual)
        assert torch.equal(results[True][1], results[False][1])
