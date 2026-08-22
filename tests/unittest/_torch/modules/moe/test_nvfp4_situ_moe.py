# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 + TRTLLM-Gen SiTU MoE tests.

SiTU on NVFP4 is served by the fused ``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*`` FC1
cubins (group-16 block scales); the MXFP4 variant is covered separately by
``test_kimi_k3_situ_moe.py``. The two paths differ in one way that is easy to
get wrong: NVFP4 carries *global* scale scalars into the kernel
(``output1_scale_scalar`` / ``output1_scale_gate_scalar``) while MXFP4 relies
purely on block scales.

Because SiTuGlu is not linear in x0 (x0 feeds ``tanh(x0 / beta)``), trtllm-gen
requires the kernel's scaleC to be ``quantScaleC`` alone -- ``dequantScaleAb``
is folded into the tanh/sigmoid arguments via scaleGate instead. Passing the
SwiGlu-style combined ``dequantScaleAb * quantScaleC`` multiplies the FC1
output by ~6e-8, which pushes the per-block E4M3 scale factors below their
smallest subnormal, so the NVFP4 intermediate quantizes to *exactly zero* and
the MoE returns all zeros.

That failure mode is the reason these tests assert a relative L2 error and a
nonzero-element count rather than only ``check_accuracy``'s tolerance band: an
all-zero output has ``relL2 == 1.0`` but still *passes* a
``rtol=0.1, atol=0.15, percent=0.97`` band, because these MoE outputs have a
standard deviation (~0.04) well below ``atol``.
"""

import os

import pytest
import torch
from _torch.modules.moe.quantize_utils import get_test_quant_params
from transformers import PretrainedConfig
from utils.util import getSMVersion

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm._torch.modules.fused_moe.impl_contract import MoECommPlan, MoERunContext
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEMethod
from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.kimi_k3_moe._mlp import SituAndMul
from tensorrt_llm._torch.utils import ActivationType, ActType_TrtllmGen
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.quantization.mode import QuantAlgo

situ_supported = pytest.mark.skipif(
    getSMVersion() not in (100, 103),
    reason="The NVFP4 SiTU cubins only support SM100/SM103. Current SM is %d." % getSMVersion(),
)

# Kimi K3 defaults. ``trtllm_gen_activation_alpha`` is the gate-side SiTU beta
# (the cubin's ``alpha``); ``trtllm_gen_activation_beta`` is the linear-side
# SiTU beta (the cubin's ``beta``). See modeling_kimi_linear.py.
GATE_ALPHA = 4.0
LINEAR_BETA = 25.0


def _build_backend(
    *, num_experts, top_k, hidden_size, intermediate_size, quantize_util, quant_config, situ
):
    """A single-rank TRTLLM-Gen NVFP4 MoE, with or without the SiTU override."""
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=quant_config,
        mapping=Mapping(world_size=1, tp_size=1, rank=0),
        moe_backend="TRTLLM",
    )
    situ_kwargs = {}
    if situ:
        situ_kwargs = dict(
            trtllm_gen_activation_type=ActType_TrtllmGen.SiTu,
            trtllm_gen_activation_alpha=GATE_ALPHA,
            trtllm_gen_activation_beta=LINEAR_BETA,
        )
    return TRTLLMGenFusedMoE(
        routing_method=RenormalizeMoeRoutingMethod(top_k=top_k),
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=model_config,
        init_load_balancer=False,
        weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        # SiTU rides the generic SwiGLU geometry; the fused activation is
        # selected by trtllm_gen_activation_type, not by activation_type.
        activation_type=ActivationType.Swiglu,
        **situ_kwargs,
    )


def _make_case(num_experts, top_k, hidden_size, intermediate_size, seq_len, situ, seed=0):
    """Backend + reference module sharing one set of NVFP4 weights."""
    torch.manual_seed(seed)
    device = "cuda"
    x = torch.randn((seq_len, hidden_size), dtype=torch.bfloat16, device=device) * 0.5
    router_logits = torch.randn((seq_len, num_experts), dtype=torch.bfloat16, device=device)

    util_cls, quant_config, quant_kwargs = get_test_quant_params(QuantAlgo.NVFP4, x, "TRTLLM")
    quant_kwargs.pop("ref_cls", None)
    quantize_util = util_cls(
        num_experts=num_experts,
        dtype=torch.bfloat16,
        intermediate_size=intermediate_size,
        hidden_size=hidden_size,
        quant_config=quant_config,
        bias=False,
        activation_type=ActivationType.Swiglu,
    )
    weights = quantize_util.create_weights(**quant_kwargs)

    backend = _build_backend(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        quantize_util=quantize_util,
        quant_config=quant_config,
        situ=situ,
    )
    backend.load_weights([weights])
    backend.post_load_weights()
    backend.cuda()
    return backend, quantize_util, weights, x, router_logits


def _run_backend(backend, x, router_logits, top_k):
    routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
    with torch.inference_mode():
        selected, scales = routing_method.apply(router_logits)
        x_quantized, x_sf = backend.quantize_input(x, post_quant_comm=False)
        return backend.run_moe(
            MoERunContext(
                x=x_quantized,
                x_sf=x_sf,
                token_selected_experts=selected.to(torch.int32),
                token_final_scales=scales.to(torch.bfloat16),
                # Single GPU, no comm strategy: quantize_input ran locally and
                # left the scale factors swizzled.
                comm_plan=MoECommPlan(
                    input_sf_swizzled=True,
                    enable_alltoall=False,
                    moe_output=None,
                    payload_in_workspace=False,
                ),
            )
        )


def _reference(quantize_util, weights, x, router_logits, top_k, situ):
    ref = quantize_util.create_ref_module(RenormalizeMoeRoutingMethod(top_k=top_k))
    if situ:
        # The in-tree SiTU activation is the source of truth: its ``beta`` is
        # the gate side and ``linear_beta`` the linear side, matching how
        # modeling_kimi_linear.py maps them onto the cubin's alpha/beta.
        activation = SituAndMul(beta=GATE_ALPHA, linear_beta=LINEAR_BETA)
        for expert in ref.experts:
            expert.activation = activation
    ref.load_weights([weights])
    ref.cuda()
    with torch.inference_mode():
        return ref.forward(x, router_logits)


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """||actual - reference|| / ||reference||. An all-zero actual gives 1.0."""
    return ((actual.float() - reference.float()).norm() / reference.float().norm()).item()


@situ_supported
@pytest.mark.parametrize("num_tokens", [1, 8, 512])
def test_nvfp4_situ_runner_has_valid_configs(num_tokens):
    """The NVFP4 SiTU cubins must be reachable through tactic selection."""
    runner = torch.classes.trtllm.FP4BlockScaleMoERunner(int(ActType_TrtllmGen.SiTu))
    # (topK, hiddenSize, intermediateSize, numLocalExperts, numTokens)
    tactics = runner.get_valid_configs(2, 512, 256, 8, num_tokens)
    assert tactics, f"no valid NVFP4 SiTU tactic for num_tokens={num_tokens}"


@situ_supported
def test_nvfp4_situ_fc31_scale_c_drops_dequant_scale():
    """SiTuGlu is nonlinear in x0, so scaleC must be quantScaleC alone.

    Pins the trtllm-gen rule (``!isLinearInX0(mActType) && mFusedAct`` in
    ``BatchedGemm/BatchedGemmTestUtils.h``) at the level where TRT-LLM
    computes it, without needing to run a kernel. SwiGLU is the control: it
    *is* linear in x0 and keeps the combined ``quantScaleC * dequantScaleAb``.
    """
    shape = dict(num_experts=8, top_k=2, hidden_size=512, intermediate_size=256, seq_len=8)
    situ_backend, *_ = _make_case(**shape, situ=True)
    swiglu_backend, *_ = _make_case(**shape, situ=False)

    # fc2_input_scale is quantScaleC; fc31_alpha is dequantScaleAb (the
    # scaleGate the kernel already folds into the tanh/sigmoid arguments).
    torch.testing.assert_close(
        situ_backend.fc31_scale_c.data.float(),
        situ_backend.fc2_input_scale.data.float().expand_as(situ_backend.fc31_scale_c.data),
        msg="SiTU fc31_scale_c must not carry the dequantScaleAb factor",
    )
    torch.testing.assert_close(
        swiglu_backend.fc31_scale_c.data.float(),
        (swiglu_backend.fc2_input_scale.data * swiglu_backend.fc31_alpha.data).float(),
        msg="SwiGLU fc31_scale_c must keep the combined scale",
    )
    # The two conventions must actually differ, otherwise this test would pass
    # vacuously on a build where dequantScaleAb happens to be 1.
    assert not torch.allclose(
        situ_backend.fc31_scale_c.data.float(), swiglu_backend.fc31_scale_c.data.float()
    )


@situ_supported
@pytest.mark.parametrize(
    "num_experts,top_k,hidden_size,intermediate_size",
    [
        pytest.param(8, 2, 512, 256, id="e8_k2_h512_i256"),
        pytest.param(8, 2, 2048, 1024, id="e8_k2_h2048_i1024"),
        pytest.param(256, 8, 2048, 1024, id="e256_k8_h2048_i1024"),
    ],
)
@pytest.mark.parametrize("seq_len", [1, 16, 256], ids=lambda n: f"tokens{n}")
def test_nvfp4_situ_matches_reference(num_experts, top_k, hidden_size, intermediate_size, seq_len):
    backend, quantize_util, weights, x, router_logits = _make_case(
        num_experts, top_k, hidden_size, intermediate_size, seq_len, situ=True
    )
    output = _run_backend(backend, x, router_logits, top_k)
    reference = _reference(quantize_util, weights, x, router_logits, top_k, situ=True)

    # Regression guard for the all-zero FC1 output: check_accuracy's tolerance
    # band alone passes on a zero tensor here (ref std ~0.04 << atol 0.15).
    nonzero = int((output != 0).sum())
    assert nonzero > 0.99 * output.numel(), (
        f"NVFP4 SiTU produced a near-empty output ({nonzero}/{output.numel()} "
        "nonzero) -- the FC1 intermediate most likely underflowed to zero"
    )

    rel_l2 = _rel_l2(output, reference)
    assert rel_l2 < 0.1, (
        f"NVFP4 SiTU relative L2 error {rel_l2:.4f} is too large "
        "(an output uncorrelated with the reference gives ~1.0)"
    )

    quantize_util.create_ref_module(RenormalizeMoeRoutingMethod(top_k=top_k)).check_accuracy(
        output, reference
    )


_LAUNCH_EVIDENCE_SCRIPT = r"""
import torch
from test_nvfp4_situ_moe import _make_case, _run_backend, _rel_l2, _reference

backend, quantize_util, weights, x, router_logits = _make_case(
    8, 2, 512, 256, 16, situ=True)
out = _run_backend(backend, x, router_logits, 2)
ref = _reference(quantize_util, weights, x, router_logits, 2, situ=True)
torch.cuda.synchronize()
assert int((out != 0).sum()) > 0, "all-zero output"
assert _rel_l2(out, ref) < 0.1, "reference mismatch"
"""


@situ_supported
def test_nvfp4_situ_launches_situ_cubin():
    """The FC1 kernel actually selected must be an NVFP4 siTuGlu cubin.

    Runs in a subprocess because the C++ logger level is fixed at process
    start (TLLM_LOG_LEVEL) and TLLM_BATCHED_GEMM_PRINT_NAME logs at INFO.
    """
    import subprocess
    import sys

    env = dict(os.environ)
    env["TLLM_BATCHED_GEMM_PRINT_NAME"] = "1"
    env["TLLM_LOG_LEVEL"] = "INFO"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    unittest_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    env["PYTHONPATH"] = os.pathsep.join([this_dir, unittest_root, env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", _LAUNCH_EVIDENCE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=this_dir,
        timeout=600,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, f"fused forward failed:\n{log[-4000:]}"
    assert "siTuGlu" in log, (
        "expected the FC1 launch log to name a siTuGlu kernel; got:\n" + log[-4000:]
    )
    # E2m1 in/out is what distinguishes the NVFP4 family from the MXFP4 one.
    assert "E2m1_E2m1E2m1" in log or "e2m1_e2m1e2m1" in log.lower(), (
        "expected the NVFP4 (E2m1_E2m1E2m1) SiTU cubin family; got:\n" + log[-4000:]
    )


@situ_supported
def test_nvfp4_situ_padded_quant_method_is_selected():
    """SiTU must pick the padded NVFP4 method regardless of call ordering.

    ``_get_quant_method`` keys off ``is_situ_activation`` rather than
    ``swiglu_alpha is not None`` precisely because create_weights fills the
    alpha/beta slots afterwards.
    """
    backend, *_ = _make_case(8, 2, 512, 256, 8, situ=True)
    assert isinstance(backend.quant_method, NVFP4TRTLLMGenFusedMoEMethod)
    assert backend.is_situ_activation
    assert backend.scaling_vector_size == 16
