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

import dataclasses
import sys
import types
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.cutlass import TrtllmCutlassNvfp4Impl
from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_b12x import (
    CuteDslB12xFusedMoE,
    CuteDslB12xNvfp4FusedMoE,
    CuteDslB12xW4a16Nvfp4FusedMoE,
    _patch_flashinfer_w4a16_ultra_wide_fc2_tile_validation,
)
from tensorrt_llm._torch.moe.fused_moe.impl_contract import (
    MoECommPlan,
    MoEDeployment,
    MoEEnvironment,
    MoEProblem,
    MoERejectReason,
    MoERunContext,
    canonical_quant,
)
from tensorrt_llm._torch.moe.fused_moe.impl_environment import MoEDep
from tensorrt_llm._torch.moe.fused_moe.quantization import (
    NVFP4CuteDslB12xFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.peft.lora.layer import LoraModuleType
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
    ep_size: int = 1,
    tp_size: int = 1,
    use_dp: bool = False,
    parallel_size: Optional[int] = None,
    eplb: bool = False,
    force_dynamic_quantization: bool = False,
    oss_cutlass_moe: bool = True,
) -> MoEDeployment:
    """Declare the machine rather than patching the probes that read it."""
    return MoEDeployment(
        ep_size=ep_size,
        tp_size=tp_size,
        parallel_size=ep_size * tp_size if parallel_size is None else parallel_size,
        use_dp=use_dp,
        num_slots=8,
        eplb_enabled=eplb,
        force_dynamic_quantization=force_dynamic_quantization,
        env=MoEEnvironment(
            sm=sm,
            oss_cutlass_moe=oss_cutlass_moe,
            available_deps=(MoEDep.FLASHINFER.value,) if flashinfer else (),
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize("leaf,quant_algo", B12X_LEAVES, ids=LEAF_IDS)
def test_can_implement_rejects_unsupported_activation_dtype(leaf, quant_algo, dtype):
    """``B12xMoEWrapper`` reads and writes bfloat16 only, so float16 is turned
    down too, on both leaves: the W4A16 kernel has a float16 path of its own,
    but the wrapper hands it a bfloat16 output buffer."""
    verdict = leaf.can_implement(_problem(quant_algo, dtype=dtype), _deployment(120))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DTYPE_UNSUPPORTED


def test_can_implement_rejects_missing_flashinfer():
    verdict = NVFP4_LEAF.can_implement(_problem(), _deployment(120, flashinfer=False))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.DEP_MISSING


def test_can_implement_rejects_eplb():
    """EPLB is its own reject class, not a topology one: the machine is fine."""
    verdict = NVFP4_LEAF.can_implement(_problem(), _deployment(120, eplb=True))
    assert not verdict.eligible
    assert verdict.reject_reason is MoERejectReason.EPLB_UNSUPPORTED


@pytest.mark.parametrize("oss_cutlass_moe", [True, False], ids=["oss", "internal"])
@pytest.mark.parametrize(
    "leaf,quant_algo,eligible",
    [
        pytest.param(CuteDslB12xNvfp4FusedMoE, QuantAlgo.NVFP4, False, id="nvfp4"),
        pytest.param(CuteDslB12xW4a16Nvfp4FusedMoE, QuantAlgo.W4A16_NVFP4, True, id="w4a16"),
    ],
)
def test_can_implement_declines_forced_dynamic_quantization(
    leaf, quant_algo, eligible, oss_cutlass_moe
):
    """The NVFP4 decode kernel quantizes against load-time scales, so per-call
    scales would reach only the CUTLASS prefill chunks. Declined on every build,
    for that reason rather than the prefill build gate's. The W4A16 kernel
    quantizes no activation, so the setting asks nothing of it."""
    deployment = _deployment(120, force_dynamic_quantization=True, oss_cutlass_moe=oss_cutlass_moe)
    verdict = leaf.can_implement(_problem(quant_algo), deployment)
    assert verdict.eligible is eligible
    if not eligible:
        assert verdict.reject_reason is MoERejectReason.DYNAMIC_QUANT_UNSUPPORTED
        assert "force_dynamic_quantization" in verdict.detail


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
# run_moe call boundary: each MoERunContext field is honoured or refused
# ---------------------------------------------------------------------------

_DECODE_TOKENS = CuteDslB12xNvfp4FusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD - 1
_PREFILL_TOKENS = CuteDslB12xNvfp4FusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD

# (leaf, chunk size, the kernel that chunk reaches): both routes of the NVFP4
# leaf, and the W4A16 leaf, which has only the b12x one.
B12X_ROUTES = [
    pytest.param(CuteDslB12xNvfp4FusedMoE, _DECODE_TOKENS, "b12x", id="nvfp4-decode"),
    pytest.param(CuteDslB12xNvfp4FusedMoE, _PREFILL_TOKENS, "cutlass", id="nvfp4-prefill"),
    pytest.param(CuteDslB12xW4a16Nvfp4FusedMoE, _PREFILL_TOKENS, "b12x", id="w4a16"),
]
B12X_KERNEL_ROUTES = [route for route in B12X_ROUTES if route.values[2] == "b12x"]

_LAYER_IDX = 3


def _moe_lora_params(layer_idx: int) -> dict:
    """``lora_params`` carrying a routed-expert MoE LoRA module for
    ``layer_idx``."""
    module_id = int(LoraModuleType.from_string("moe_h_to_4h"))
    return {
        "num_seqs": 1,
        layer_idx: {module_id: {"adapter_size": None, "weight_pointers": None}},
    }


class _B12xWrapperRecorder:
    """Stands in for ``B12xMoEWrapper``, writing bfloat16 as its kernels do."""

    def __init__(self, log: list) -> None:
        self.log = log
        self.output: Optional[torch.Tensor] = None

    def run(self, *, x, token_selected_experts, token_final_scales, **weights) -> torch.Tensor:
        self.log.append("b12x")
        self.output = torch.zeros(x.shape, dtype=torch.bfloat16)
        return self.output


class _RecordingEvent:
    def __init__(self, log: list) -> None:
        self.log = log

    def wait(self, stream=None) -> None:
        self.log.append("wait")


def _leaf_with_recorded_kernels(leaf_cls: type, log: list, monkeypatch) -> CuteDslB12xFusedMoE:
    """A leaf whose two kernels -- b12x and the inherited CUTLASS NVFP4
    ``run_moe`` -- only record that they ran.

    ``__new__`` without ``__init__``: the real constructor wants a GPU, while
    ``run_moe`` reads only the layer index, the wrapper and its weights.
    """

    def cutlass_run_moe(self, ctx: MoERunContext, *, workspace=None) -> torch.Tensor:
        log.append("cutlass")
        return torch.zeros(ctx.x.shape, dtype=ctx.output_dtype)

    monkeypatch.setattr(TrtllmCutlassNvfp4Impl, "run_moe", cutlass_run_moe)
    leaf = leaf_cls.__new__(leaf_cls)
    leaf.layer_idx = _LAYER_IDX
    leaf.b12x_wrapper = _B12xWrapperRecorder(log)
    leaf._b12x_weights = {}
    return leaf


def _run_context(num_tokens: int, **overrides) -> MoERunContext:
    """What the scheduler hands a served B12x layer: routing already applied,
    no communication strategy, bfloat16 activations."""
    ctx = MoERunContext(
        token_selected_experts=torch.zeros((num_tokens, 2), dtype=torch.int32),
        token_final_scales=torch.ones((num_tokens, 2), dtype=torch.float32),
        x=torch.zeros((num_tokens, 8), dtype=torch.bfloat16),
        x_sf=None,
        output_dtype=torch.bfloat16,
        comm_plan=MoECommPlan(
            input_sf_swizzled=True,
            enable_alltoall=False,
            moe_output=None,
            payload_in_workspace=False,
        ),
    )
    return dataclasses.replace(ctx, **overrides)


@pytest.mark.parametrize("leaf_cls,num_tokens,kernel", B12X_ROUTES)
def test_run_moe_hands_back_the_kernel_output_untouched(leaf_cls, num_tokens, kernel, monkeypatch):
    """The served configuration reaches its kernel, and the b12x output comes
    back as the very tensor the wrapper wrote."""
    log = []
    leaf = _leaf_with_recorded_kernels(leaf_cls, log, monkeypatch)

    out = leaf.run_moe(_run_context(num_tokens, all_rank_num_tokens=[num_tokens]))

    assert log == [kernel]
    if kernel == "b12x":
        assert out is leaf.b12x_wrapper.output


@pytest.mark.parametrize(
    "overrides,error,match",
    [
        pytest.param(
            dict(do_finalize=False), NotImplementedError, "do_finalize=False", id="unfinalized"
        ),
        pytest.param(
            dict(lora_params=_moe_lora_params(_LAYER_IDX)), NotImplementedError, "LoRA", id="lora"
        ),
        pytest.param(
            dict(
                token_selected_experts=None,
                token_final_scales=None,
                router_logits=torch.zeros(1, 8),
            ),
            ValueError,
            "router_logits",
            id="router-logits-only",
        ),
        pytest.param(
            dict(
                comm_plan=MoECommPlan(
                    input_sf_swizzled=True,
                    enable_alltoall=True,
                    moe_output=None,
                    payload_in_workspace=False,
                )
            ),
            NotImplementedError,
            "all-to-all",
            id="alltoall",
        ),
    ],
)
@pytest.mark.parametrize("leaf_cls,num_tokens,kernel", B12X_ROUTES)
def test_run_moe_refuses_what_neither_route_honours(
    leaf_cls, num_tokens, kernel, overrides, error, match, monkeypatch
):
    """Neither kernel the leaf runs returns the unfinalized triple, fuses LoRA,
    routes from logits or holds only a slice of the experts, so asking for any
    of these fails before a kernel launches instead of being ignored."""
    log = []
    leaf = _leaf_with_recorded_kernels(leaf_cls, log, monkeypatch)

    with pytest.raises(error, match=match):
        leaf.run_moe(_run_context(num_tokens, **overrides))
    assert log == []


@pytest.mark.parametrize(
    "lora_params",
    [
        pytest.param({"num_seqs": 1}, id="no-layer-adapters"),
        pytest.param(_moe_lora_params(_LAYER_IDX + 1), id="moe-lora-on-another-layer"),
    ],
)
@pytest.mark.parametrize("leaf_cls,num_tokens,kernel", B12X_ROUTES)
def test_run_moe_admits_lora_params_without_moe_lora_on_its_layer(
    leaf_cls, num_tokens, kernel, lora_params, monkeypatch
):
    """``lora_params`` describes every adapter of the forward. Only
    routed-expert LoRA on this layer asks the leaf for something it cannot
    fuse; adapters elsewhere leave its output as it is."""
    log = []
    leaf = _leaf_with_recorded_kernels(leaf_cls, log, monkeypatch)

    leaf.run_moe(_run_context(num_tokens, lora_params=lora_params))

    assert log == [kernel]


@pytest.mark.parametrize(
    "lora_params",
    [
        pytest.param(_moe_lora_params(_LAYER_IDX), id="moe-lora"),
        pytest.param({"num_seqs": 1}, id="no-layer-adapters"),
    ],
)
@pytest.mark.parametrize("leaf_cls,num_tokens,kernel", B12X_ROUTES)
def test_run_moe_refuses_lora_params_when_it_has_no_layer_index(
    leaf_cls, num_tokens, kernel, lora_params, monkeypatch
):
    """``layer_idx`` is what picks this layer's adapters out of ``lora_params``,
    and it defaults to None. A leaf without one cannot tell MoE LoRA from the
    adapters it may ignore, so it refuses every ``lora_params`` -- including the
    one the same leaf admits once it knows which layer it is."""
    log = []
    leaf = _leaf_with_recorded_kernels(leaf_cls, log, monkeypatch)
    leaf.layer_idx = None

    with pytest.raises(NotImplementedError, match="layer_idx"):
        leaf.run_moe(_run_context(num_tokens, lora_params=lora_params))
    assert log == []


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("leaf_cls,num_tokens,kernel", B12X_KERNEL_ROUTES)
def test_b12x_route_refuses_an_output_dtype_it_cannot_write(
    leaf_cls, num_tokens, kernel, output_dtype, monkeypatch
):
    """``B12xMoEWrapper`` writes bfloat16 and refuses any other
    ``output_dtype`` at construction, so a chunk asking for another one fails
    rather than receiving bfloat16."""
    log = []
    leaf = _leaf_with_recorded_kernels(leaf_cls, log, monkeypatch)

    with pytest.raises(ValueError, match="bfloat16"):
        leaf.run_moe(_run_context(num_tokens, output_dtype=output_dtype))
    assert log == []


def test_cutlass_route_writes_the_requested_output_dtype(monkeypatch):
    """The NVFP4 prefill route is the CUTLASS op, which writes any
    high-precision dtype; only the b12x route is bound to bfloat16."""
    log = []
    leaf = _leaf_with_recorded_kernels(CuteDslB12xNvfp4FusedMoE, log, monkeypatch)

    out = leaf.run_moe(_run_context(_PREFILL_TOKENS, output_dtype=torch.float16))

    assert log == ["cutlass"]
    assert out.dtype == torch.float16


@pytest.mark.parametrize("leaf_cls,num_tokens,kernel", B12X_ROUTES)
def test_run_moe_joins_the_input_ready_event_before_its_kernel(
    leaf_cls, num_tokens, kernel, monkeypatch
):
    """``x`` may still be written on another stream; no kernel reads it before
    the event the scheduler handed over."""
    log = []
    leaf = _leaf_with_recorded_kernels(leaf_cls, log, monkeypatch)

    leaf.run_moe(_run_context(num_tokens, x_ready_event=_RecordingEvent(log)))

    assert log == ["wait", kernel]


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
