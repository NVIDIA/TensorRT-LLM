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
"""B12x MoE backend gating and dispatch tests."""

import sys
import types
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x import (
    CuteDslB12xFusedMoE,
    CuteDslB12xNvfp4FusedMoE,
    CuteDslB12xW4a16Nvfp4FusedMoE,
    _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    canonical_quant,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep, _probe_cutedsl_cuda13
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import (
    backend_family_of,
    impl_class_for,
    resolve_moe_impl,
)
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CuteDslB12xFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.utils import ActivationType
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

pytestmark = pytest.mark.cpu_only


# Spelled out rather than read from _SUPPORTED_SM_VERSIONS: deriving the input
# from the value under test hides a narrowing of that set, which is the
# direction that silently drops hardware support.
SUPPORTED_SM = [120, 121]

# (leaf, the one quantization format it admits). Both leaves run the mixin's
# ``can_implement``, so the gates other than the format are tested on one leaf.
B12X_LEAVES = [
    (CuteDslB12xNvfp4FusedMoE, QuantAlgo.NVFP4),
    (CuteDslB12xW4a16Nvfp4FusedMoE, QuantAlgo.W4A16_NVFP4),
]
LEAF_IDS = [leaf.__name__ for leaf, _ in B12X_LEAVES]
NVFP4_LEAF = CuteDslB12xNvfp4FusedMoE


def _deployment(
    sm: int,
    *,
    flashinfer: bool = True,
    cutedsl_cuda13: bool = True,
    ep_size: int = 1,
    tp_size: int = 1,
    use_dp: bool = False,
    parallel_size: Optional[int] = None,
    eplb: bool = False,
) -> MoEDeployment:
    """Declare the machine rather than patching the probes that read it."""
    deps = []
    if flashinfer:
        deps.append(MoEDep.FLASHINFER.value)
    if cutedsl_cuda13:
        deps.append(MoEDep.CUTEDSL_CUDA13.value)
    return MoEDeployment(
        ep_size=ep_size,
        tp_size=tp_size,
        parallel_size=ep_size * tp_size if parallel_size is None else parallel_size,
        use_dp=use_dp,
        num_slots=8,
        eplb_enabled=eplb,
        env=MoEEnvironment(
            sm=sm,
            available_deps=tuple(deps),
        ),
    )


def _problem(
    quant_algo=QuantAlgo.NVFP4,
    dtype=torch.bfloat16,
    swiglu_gptoss_style=None,
    intermediate_size: int = 2048,
) -> MoEProblem:
    return MoEProblem(
        quant=canonical_quant(quant_algo),
        dtype_act=dtype,
        hidden_size=2048,
        intermediate_size=intermediate_size,
        num_experts=8,
        top_k=2,
        swiglu_gptoss_style=swiglu_gptoss_style,
    )


@pytest.mark.parametrize("sm_version", [80, 89, 90, 100, 103])
def test_can_implement_rejects_unsupported_sm(sm_version):
    verdict = NVFP4_LEAF.can_implement(_problem(), _deployment(sm_version))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.SM_UNSUPPORTED
    assert f"SM{sm_version}" in verdict.detail


@pytest.mark.parametrize("leaf,quant_algo", B12X_LEAVES, ids=LEAF_IDS)
@pytest.mark.parametrize("sm_version", SUPPORTED_SM)
def test_can_implement_accepts_supported_sm(leaf, quant_algo, sm_version):
    verdict = leaf.can_implement(_problem(quant_algo), _deployment(sm_version))
    assert verdict.eligible
    assert verdict.reject_reason is None


@pytest.mark.parametrize(
    "leaf,rejected_algo",
    [
        (CuteDslB12xNvfp4FusedMoE, None),
        (CuteDslB12xNvfp4FusedMoE, QuantAlgo.FP8_BLOCK_SCALES),
        (CuteDslB12xNvfp4FusedMoE, QuantAlgo.W4A16_NVFP4),
        (CuteDslB12xW4a16Nvfp4FusedMoE, QuantAlgo.NVFP4),
    ],
)
def test_can_implement_rejects_foreign_quant(leaf, rejected_algo):
    """Each leaf admits one format -- neither answers for the other's."""
    verdict = leaf.can_implement(_problem(rejected_algo), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.QUANT_UNSUPPORTED


def test_family_name_is_not_an_implementation():
    """``CuteDslB12xFusedMoE`` names both leaves; asking it gives a verdict that
    points at them instead of failing on the format it does not carry."""
    verdict = CuteDslB12xFusedMoE.can_implement(_problem(), _deployment(120))
    assert verdict.reject_reason is MoERejectReason.NOT_AN_IMPL
    assert CuteDslB12xNvfp4FusedMoE.__name__ in verdict.detail


@pytest.mark.parametrize("alias", [QuantAlgo.NVFP4_AWQ, QuantAlgo.NVFP4_ARC])
def test_nvfp4_leaf_admits_the_calibration_aliases(alias):
    """The format gate compares identities, so an uncanonicalized alias is
    admitted, as the CUTLASS NVFP4 leaf admits it, and shape-gated as NVFP4 is."""
    problem = MoEProblem(quant=alias.value, dtype_act=torch.bfloat16, intermediate_size=2048)
    assert NVFP4_LEAF.can_implement(problem, _deployment(120)).eligible
    # FC1 rows 2 * 1000 are not a multiple of 128.
    unaligned = MoEProblem(quant=alias.value, dtype_act=torch.bfloat16, intermediate_size=1000)
    verdict = NVFP4_LEAF.can_implement(unaligned, _deployment(120))
    assert verdict.reject_reason is MoERejectReason.SHAPE_UNALIGNED


def test_can_implement_rejects_swiglu_gptoss_style():
    verdict = NVFP4_LEAF.can_implement(_problem(swiglu_gptoss_style=True), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.ACTIVATION_UNSUPPORTED


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_can_implement_rejects_unsupported_activation_dtype(dtype):
    verdict = NVFP4_LEAF.can_implement(_problem(dtype=dtype), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DTYPE_UNSUPPORTED


def test_can_implement_rejects_missing_flashinfer():
    verdict = NVFP4_LEAF.can_implement(_problem(), _deployment(120, flashinfer=False))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DEP_MISSING


@pytest.mark.parametrize("sm_version", SUPPORTED_SM)
def test_can_implement_rejects_nvfp4_without_cuda13_cute_dsl(sm_version):
    """CUDA 12.x CuTe DSL emits PTX that ptxas rejects for the NVFP4 MMA."""
    verdict = NVFP4_LEAF.can_implement(
        _problem(), _deployment(sm_version, cutedsl_cuda13=False)
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DEP_MISSING
    assert "CUDA 13 native payload" in verdict.detail
    assert "Unexpected instruction types" in verdict.detail


@pytest.mark.parametrize("sm_version", SUPPORTED_SM)
def test_can_implement_cuda13_gate_is_limited_to_pure_nvfp4(sm_version):
    """Only the pure-NVFP4 decode kernel instantiates the NVFP4 MMA atom."""
    verdict = CuteDslB12xW4a16Nvfp4FusedMoE.can_implement(
        _problem(QuantAlgo.W4A16_NVFP4), _deployment(sm_version, cutedsl_cuda13=False)
    )
    assert verdict.eligible


def _cutedsl_model_config() -> ModelConfig:
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    return cfg


@pytest.mark.parametrize("sm_version", SUPPORTED_SM)
def test_cutedsl_degrades_to_cutlass_without_cuda13_cute_dsl(sm_version):
    """CUTEDSL + NVFP4 + SM12x avoids the unsupported CuTe DSL JIT path."""
    report = resolve_moe_impl(
        _cutedsl_model_config(),
        problem=_problem(),
        deployment=_deployment(sm_version, cutedsl_cuda13=False),
    )
    assert backend_family_of(impl_class_for(report)) == "CUTLASS"
    assert report.degraded
    b12x_rejections = [r for r in report.rejected if r.legacy_backend == NVFP4_LEAF.__name__]
    assert len(b12x_rejections) == 1
    assert b12x_rejections[0].reason is MoERejectReason.DEP_MISSING
    assert "CUDA 13 native payload" in b12x_rejections[0].detail


@pytest.mark.parametrize("sm_version", SUPPORTED_SM)
def test_cutedsl_selects_b12x_with_cuda13_cute_dsl(sm_version):
    report = resolve_moe_impl(
        _cutedsl_model_config(),
        problem=_problem(),
        deployment=_deployment(sm_version),
    )
    assert impl_class_for(report) is NVFP4_LEAF
    assert not report.degraded


def test_probe_cutedsl_cuda13_reports_missing_cutlass(monkeypatch):
    # A ``None`` entry in ``sys.modules`` makes ``import cutlass`` raise
    # ImportError without touching an installed package.
    monkeypatch.setitem(sys.modules, "cutlass", None)
    available, detail = _probe_cutedsl_cuda13()
    assert not available
    assert "import cutlass failed" in detail


@pytest.mark.parametrize(
    ("cuda_version", "expected"),
    [
        (None, False),
        (types.SimpleNamespace(), False),
        (types.SimpleNamespace(major="13"), False),
        (types.SimpleNamespace(major=12), False),
        (types.SimpleNamespace(major=13), True),
        (types.SimpleNamespace(major=14), True),
    ],
)
def test_probe_cutedsl_cuda13_reads_public_cutlass_api(monkeypatch, cuda_version, expected):
    cutlass_module = types.ModuleType("cutlass")
    if cuda_version is not None:
        cutlass_module.CUDA_VERSION = cuda_version
    monkeypatch.setitem(sys.modules, "cutlass", cutlass_module)
    available, detail = _probe_cutedsl_cuda13()
    assert available is expected
    assert bool(detail) is not expected


def test_can_implement_rejects_eplb():
    """EPLB is its own reject class, not a topology one: the machine is fine."""
    verdict = NVFP4_LEAF.can_implement(_problem(), _deployment(120, eplb=True))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.EPLB_UNSUPPORTED


def test_can_implement_rejects_expert_parallelism():
    verdict = NVFP4_LEAF.can_implement(_problem(), _deployment(120, ep_size=2))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.TOPOLOGY_UNSUPPORTED


def test_can_implement_rejects_attention_dp_without_expert_parallelism():
    """moe_tp == tp leaves ep_size at 1, so the EP gate alone would let this in."""
    verdict = NVFP4_LEAF.can_implement(
        _problem(), _deployment(120, ep_size=1, use_dp=True, parallel_size=2)
    )
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.TOPOLOGY_UNSUPPORTED
    assert "attention-DP" in verdict.detail


@pytest.mark.parametrize("leaf,quant_algo", B12X_LEAVES, ids=LEAF_IDS)
@pytest.mark.parametrize(
    "intermediate_size,eligible",
    [pytest.param(256, True, id="aligned"), pytest.param(192, False, id="unaligned")],
)
def test_can_implement_rejects_unaligned_nvfp4_fc1_shard(
    leaf, quant_algo, intermediate_size, eligible
):
    """Both leaves consume the FC1 buffer the CUTLASS NVFP4 loader pads, which
    splits gate/up at the wrong row for an unaligned shard."""
    verdict = leaf.can_implement(
        _problem(quant_algo, intermediate_size=intermediate_size), _deployment(120, tp_size=2)
    )
    assert verdict.eligible is eligible
    if not eligible:
        assert verdict.reject_reason is MoERejectReason.SHAPE_UNALIGNED


def test_nvfp4_decode_applies_the_awq_pre_quant_scale():
    """b12x has no operand for an NVFP4_AWQ pre-quant scale, so a decode chunk
    is scaled before it is handed over, as the CUTLASS prefill path does."""
    moe = object.__new__(CuteDslB12xNvfp4FusedMoE)
    x = torch.ones(CuteDslB12xNvfp4FusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD - 1, 4)

    moe.fc31_act_scale = None
    out, out_sf = moe.quantize_input(x)
    assert out is x
    assert out_sf is None

    moe.fc31_act_scale = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out, out_sf = moe.quantize_input(x)
    assert torch.equal(out, x * moe.fc31_act_scale)
    assert out_sf is None


@pytest.mark.parametrize("has_pre_quant_scale", [False, True], ids=["nvfp4", "nvfp4_awq"])
def test_w4a16_applies_the_awq_pre_quant_scale(has_pre_quant_scale):
    """This leaf never quantizes the activation, but an AWQ checkpoint still
    needs its pre-quant scale: it stores ``W / s``, so the weights the b12x
    W4A16 kernel dequantizes expect ``x * s``. ``load_quant_scales`` fills
    ``fc31_act_scale`` from the ``*.pre_quant_scale`` keys without consulting
    ``quant_algo``, so a plain NVFP4 checkpoint leaves it None and the input is
    handed over untouched.

    ``__new__`` without ``__init__``: the real constructor wants a GPU, while
    ``quantize_input`` reads one parameter. Registered the way
    ``create_weights`` does -- the slot as ``None``, then a ``(1, hidden)``
    parameter of the module dtype -- so the ``nn.Module`` parameter lookup and
    the broadcast shape are the production ones.
    """
    hidden_size = 8
    moe = CuteDslB12xW4a16Nvfp4FusedMoE.__new__(CuteDslB12xW4a16Nvfp4FusedMoE)
    torch.nn.Module.__init__(moe)
    moe.register_parameter("fc31_act_scale", None)
    pre_quant_scale = None
    if has_pre_quant_scale:
        pre_quant_scale = torch.linspace(0.25, 4.0, hidden_size, dtype=torch.bfloat16).unsqueeze(0)
        moe.register_parameter(
            "fc31_act_scale", torch.nn.Parameter(pre_quant_scale, requires_grad=False)
        )
    x = torch.randn(3, hidden_size, dtype=torch.bfloat16)

    out, out_sf = moe.quantize_input(x)

    assert out_sf is None
    if has_pre_quant_scale:
        torch.testing.assert_close(out, x * pre_quant_scale, rtol=0, atol=0)
    else:
        assert out is x


def test_w4a16_nvfp4_post_load_uses_modelopt_scale_contract(monkeypatch):
    class _RoutingMethod:
        experts_per_token = 4

    class _FakeB12xWrapper:
        calls = []

        def __init__(self, **kwargs):
            self._moe_output = None
            self.calls.append(kwargs)

    def _convert_sf_to_mma_layout(scales, *, m, k, num_groups):
        return scales

    flashinfer = types.ModuleType("flashinfer")
    flashinfer.B12xMoEWrapper = _FakeB12xWrapper
    cute_dsl = types.ModuleType("flashinfer.cute_dsl")
    utils = types.ModuleType("flashinfer.cute_dsl.utils")
    utils.convert_sf_to_mma_layout = _convert_sf_to_mma_layout
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl", cute_dsl)
    monkeypatch.setitem(sys.modules, "flashinfer.cute_dsl.utils", utils)

    num_experts = 2
    hidden_size = 128
    logical_intermediate_size = 1856
    padded_intermediate_size = 1920
    module = torch.nn.Module()
    module.num_experts = num_experts
    module.hidden_size = hidden_size
    module.intermediate_size_per_partition = logical_intermediate_size
    module.moe_max_num_tokens = 8
    module.routing_method = _RoutingMethod()
    module.activation_type = ActivationType.Swiglu
    module.quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_NVFP4)
    module.w3_w1_weight = torch.empty(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.int64,
    )
    module.w2_weight = torch.empty(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.int64,
    )
    w3_w1_weight_scale = torch.ones(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    w2_weight_scale = torch.ones(
        num_experts,
        hidden_size,
        padded_intermediate_size // 16,
        dtype=torch.float8_e4m3fn,
    )
    module.w3_w1_weight_scale = w3_w1_weight_scale.clone()
    module.w2_weight_scale = w2_weight_scale.clone()
    module.fc31_alpha = torch.tensor([0.25, 0.5], dtype=torch.float32)
    module.fc2_alpha = torch.tensor([0.125, 0.25], dtype=torch.float32)
    module.fc31_input_scale = torch.tensor(2.0, dtype=torch.float32)
    module.fc2_input_scale = torch.tensor(4.0, dtype=torch.float32)

    with patch.object(NVFP4CutlassFusedMoEMethod, "transform_weights", return_value=None):
        NVFP4CuteDslB12xFusedMoEMethod().transform_weights(module)

    assert _FakeB12xWrapper.calls
    wrapper_kwargs = _FakeB12xWrapper.calls[0]
    assert wrapper_kwargs.get("quant_mode") == "w4a16", wrapper_kwargs
    assert wrapper_kwargs["intermediate_size"] == padded_intermediate_size
    assert module._b12x_weights["fc2_input_scale"] is None
    assert torch.equal(
        module._b12x_weights["w1_weight_sf"].float(),
        w3_w1_weight_scale.float(),
    )
    assert torch.equal(
        module._b12x_weights["w2_weight_sf"].float(),
        w2_weight_scale.float(),
    )
    assert torch.allclose(
        module._b12x_weights["w1_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )
    assert torch.allclose(
        module._b12x_weights["w2_alpha"],
        torch.tensor([0.5, 1.0], dtype=torch.float32),
    )


def test_dispatch_rejects_non_tensor():
    """Non-tensor inputs (e.g. Fp4QuantizedTensor) stay on the b12x path
    so the existing ValueError surfaces in quantize_input."""
    moe = object.__new__(CuteDslB12xNvfp4FusedMoE)
    assert moe._route_to_cutlass(object()) is False


# ---------------------------------------------------------------------------
# flashinfer W4A16 ultra-wide FC2 tile re-validation shim (nvbug 6721561)
# ---------------------------------------------------------------------------

# FC2 problem of Qwen3.6-35B-A3B (N=hidden_size, K=moe_intermediate_size) and
# the RTX PRO 6000 opt-in shared memory minus flashinfer's 512-byte reserve:
# the exact geometry the post-merge failure re-pinned.
_QWEN36_FC2 = dict(problem_n=2048, problem_k=512)
_RTX_PRO_6000_SMEM = 101376 - 512
_ULTRA_FC2 = dict(tile_n=512, tile_k=32, cta_threads=256)


def _flashinfer_w4a16_kernel():
    kernel = pytest.importorskip("flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel")
    if hasattr(kernel, "_ultra_wide_fc2_tile_fits"):
        pytest.skip("installed flashinfer ships the ultra-wide FC2 tile fix; drop the shim")
    return kernel


def _reset_to_stock_validator(kernel, monkeypatch) -> None:
    """Start every test from flashinfer's own validator, whichever test or
    backend construction installed the shim earlier in this process."""
    fits = kernel._candidate_tile_fits
    monkeypatch.setattr(
        kernel, "_candidate_tile_fits", getattr(fits, "_trtllm_upstream_fits", fits)
    )


def _fits(kernel, **overrides) -> bool:
    kwargs = dict(cta_m_blocks=1, max_shared_mem=_RTX_PRO_6000_SMEM, **_QWEN36_FC2, **_ULTRA_FC2)
    kwargs.update(overrides)
    return kernel._candidate_tile_fits(**kwargs)


def test_w4a16_ultra_wide_fc2_shim_accepts_repinned_ultra_tile(monkeypatch):
    kernel = _flashinfer_w4a16_kernel()
    _reset_to_stock_validator(kernel, monkeypatch)
    assert not _fits(kernel), (
        "flashinfer's stock validator accepts the ultra-wide FC2 tile; the shim in "
        "fused_moe_cute_dsl_b12x.py is no longer needed"
    )
    stock_wide_fc2 = _fits(kernel, tile_n=256, tile_k=64)

    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is True

    # The failing re-pin now passes ...
    assert _fits(kernel)
    assert _fits(kernel, scale_format="e8m0_k32")
    # ... under the override's own rule: exact N/K tiling and shared-memory fit.
    assert not _fits(kernel, problem_n=2048 + 256)
    assert not _fits(kernel, problem_k=512 + 16)
    assert not _fits(kernel, max_shared_mem=16 * 1024)
    # Everything that is not the ultra-wide FC2 tile still goes to flashinfer.
    assert _fits(kernel, tile_n=256, tile_k=64) is stock_wide_fc2
    assert not _fits(kernel, tile_n=256, tile_k=32, cta_threads=128)
    assert not _fits(kernel, tile_n=512, tile_k=32, cta_threads=128)


def test_w4a16_ultra_wide_fc2_shim_is_idempotent(monkeypatch):
    kernel = _flashinfer_w4a16_kernel()
    _reset_to_stock_validator(kernel, monkeypatch)
    stock = kernel._candidate_tile_fits

    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is True
    installed = kernel._candidate_tile_fits
    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is True

    assert kernel._candidate_tile_fits is installed
    assert installed._trtllm_upstream_fits is stock


def test_w4a16_ultra_wide_fc2_shim_yields_to_upstream_fix(monkeypatch):
    kernel = _flashinfer_w4a16_kernel()
    _reset_to_stock_validator(kernel, monkeypatch)
    stock = kernel._candidate_tile_fits
    monkeypatch.setattr(kernel, "_ultra_wide_fc2_tile_fits", lambda **_: True, raising=False)

    assert _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation() is False
    assert kernel._candidate_tile_fits is stock
