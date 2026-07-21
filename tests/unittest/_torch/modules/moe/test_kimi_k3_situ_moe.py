# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 native TRTLLM-Gen SiTU MoE tests.

Covers the acceptance criteria of the SiTU cubin integration plan
(`tensorrt_llm/_torch/modules/kimi_k3_moe/SITU_CUBIN_INTEGRATION_PLAN.md`):

* runner-local ActType numeric stability (SwiGlu/Relu2/Silu unchanged,
  SiTu appended);
* the native SiTU runner returns valid tactics and actually launches a
  kernel whose name contains ``siTuGlu``;
* fused output matches the Python ``SituAndMul`` reference within
  MXFP8/MXFP4 quantization tolerance, for the default AND asymmetric
  non-default ``activation_situ_beta`` / ``activation_situ_linear_beta``;
* swapping the FC1 halves (gate-first packing) breaks accuracy
  (mutation test — proves the w3-first convention is load-bearing);
* module-level semantics (latent projections, RMSNorm, shared experts)
  survive the fused path;
* the fused path fails loudly without loaded weights (no silent
  random-weight fallback).
"""

import dataclasses
import os

import pytest
import torch
from utils.util import check_accuracy

from tensorrt_llm._torch.modules.kimi_k3_moe import KimiK3SparseMoeBlock
from tensorrt_llm._torch.modules.kimi_k3_moe._moe_kernels import (
    is_native_situ_supported,
    make_situ_alpha_beta,
    padded_fused_shapes,
)
from tensorrt_llm._torch.utils import ActType_TrtllmGen

situ_supported = pytest.mark.skipif(
    not is_native_situ_supported(),
    reason="native SiTU cubins require SM100/SM103 (Blackwell)",
)


@dataclasses.dataclass
class _K3Config:
    """Minimal config carrying the fields KimiK3SparseMoeBlock reads."""

    hidden_size: int = 512
    num_experts: int = 8
    num_experts_per_token: int = 2
    moe_intermediate_size: int = 256
    moe_renormalize: bool = True
    moe_router_activation_func: str = "sigmoid"
    routed_scaling_factor: float = 1.0
    num_expert_group: int = 1
    topk_group: int = 1
    num_shared_experts: int = None
    routed_expert_hidden_size: int = None
    latent_moe_use_norm: bool = False
    rms_norm_eps: float = 1e-5
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0


def _init_block_weights(block: KimiK3SparseMoeBlock, seed: int = 1234):
    """Fill gate/latent/shared weights and the MXFP4 expert bank."""
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def randn_like_param(p, scale):
        return torch.randn(p.shape, generator=gen, dtype=torch.float32) * scale

    with torch.no_grad():
        block.gate.weight.copy_(randn_like_param(block.gate.weight, 0.05))
        block.gate.e_score_correction_bias.copy_(
            randn_like_param(block.gate.e_score_correction_bias, 0.1)
        )
        if block.use_latent_moe:
            block.routed_expert_down_proj.weight.copy_(
                randn_like_param(block.routed_expert_down_proj.weight, 0.05).to(
                    block.routed_expert_down_proj.weight.dtype
                )
            )
            block.routed_expert_up_proj.weight.copy_(
                randn_like_param(block.routed_expert_up_proj.weight, 0.05).to(
                    block.routed_expert_up_proj.weight.dtype
                )
            )
            if block.routed_expert_norm is not None:
                block.routed_expert_norm.weight.fill_(1.0)
        if block.shared_experts is not None:
            block.shared_experts.gate_up_proj.weight.copy_(
                randn_like_param(block.shared_experts.gate_up_proj.weight, 0.05).to(
                    block.shared_experts.gate_up_proj.weight.dtype
                )
            )
            block.shared_experts.down_proj.weight.copy_(
                randn_like_param(block.shared_experts.down_proj.weight, 0.05).to(
                    block.shared_experts.down_proj.weight.dtype
                )
            )

    isize, hsize = block.expert_bank.intermediate_size, block.expert_bank.hidden_size
    for e in range(block.num_experts):
        w1 = torch.randn(isize, hsize, generator=gen, dtype=torch.float32) * 0.1
        w2 = torch.randn(hsize, isize, generator=gen, dtype=torch.float32) * 0.1
        w3 = torch.randn(isize, hsize, generator=gen, dtype=torch.float32) * 0.1
        block.expert_bank.store_expert(e, w1, w2, w3)


def _make_block_pair(config, device):
    """Fused block + reference block sharing identical weights."""
    fused = KimiK3SparseMoeBlock(
        config, use_fused_cubin=True, dtype=torch.bfloat16, device=device
    ).to(device)
    ref = KimiK3SparseMoeBlock(
        config, use_fused_cubin=False, dtype=torch.bfloat16, device=device
    ).to(device)
    _init_block_weights(fused)
    ref.load_state_dict(fused.state_dict())
    fused.build_fused_weights()
    return fused, ref


def test_act_type_enum_values_stable():
    assert int(ActType_TrtllmGen.SwiGlu) == 0
    assert int(ActType_TrtllmGen.Relu2) == 1
    assert int(ActType_TrtllmGen.Silu) == 2
    assert int(ActType_TrtllmGen.SiTu) == 3


def test_padded_fused_shapes():
    assert padded_fused_shapes(512, 256) == (512, 512, 256)
    assert padded_fused_shapes(128, 256) == (512, 128, 256)
    assert padded_fused_shapes(2880, 96) == (3072, 2944, 128)


@situ_supported
def test_make_situ_alpha_beta_contract():
    alpha, beta = make_situ_alpha_beta(
        local_num_experts=8,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        device=torch.device("cuda"),
    )
    for buf, val in ((alpha, 4.0), (beta, 25.0)):
        assert buf.is_cuda and buf.dtype == torch.float32 and buf.is_contiguous()
        assert buf.shape == (8,)
        assert torch.all(buf == val)
    with pytest.raises(RuntimeError, match="must be > 0"):
        make_situ_alpha_beta(
            local_num_experts=8,
            situ_beta=-1.0,
            situ_linear_beta=25.0,
            device=torch.device("cuda"),
        )


@situ_supported
def test_situ_runner_returns_valid_tactics():
    runner = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(int(ActType_TrtllmGen.SiTu), True)
    # Representative low-latency and throughput shapes (topK, hidden,
    # intermediate, localExperts, numTokens, validHidden, validIntermediate).
    for num_tokens in (1, 8, 512):
        tactics = runner.get_valid_configs(2, 512, 256, 8, num_tokens, 512, 256)
        assert len(tactics) > 0, f"no valid SiTU tactic for num_tokens={num_tokens}"


_LAUNCH_EVIDENCE_SCRIPT = r"""
import torch
from test_kimi_k3_situ_moe import _K3Config, _make_block_pair

device = torch.device("cuda")
config = _K3Config()
fused, _ = _make_block_pair(config, device)
x = torch.randn(1, 16, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5
fused(x)
torch.cuda.synchronize()
assert fused._cubin_call_count == 1
"""


@situ_supported
def test_fused_forward_launches_situ_kernel():
    """Launch evidence: the FC1 kernel actually selected must be a siTuGlu cubin.

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
        cwd=os.path.dirname(os.path.abspath(__file__)),
        timeout=600,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, f"fused forward failed:\n{log[-4000:]}"
    assert "siTuGlu" in log, (
        "expected the FC1 launch log to name a siTuGlu kernel; got:\n" + log[-4000:]
    )


@situ_supported
@pytest.mark.parametrize(
    "situ_beta,situ_linear_beta",
    [
        (4.0, 25.0),  # Kimi K3 defaults
        (2.5, 7.0),  # asymmetric non-defaults — catches alpha/beta swaps
    ],
    ids=["default_alpha_beta", "asymmetric_alpha_beta"],
)
@pytest.mark.parametrize("num_tokens", [1, 15, 256], ids=lambda n: f"tokens{n}")
def test_fused_matches_reference(num_tokens, situ_beta, situ_linear_beta):
    device = torch.device("cuda")
    config = _K3Config(
        activation_situ_beta=situ_beta,
        activation_situ_linear_beta=situ_linear_beta,
    )
    fused, ref = _make_block_pair(config, device)

    torch.manual_seed(7)
    x = torch.randn(1, num_tokens, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    out_fused = fused(x)
    out_ref = ref(x)

    # Error budget: MXFP8 activation quantization (FC1 input and FC1->FC2
    # intermediate) on top of shared canonical MXFP4 weights.
    check_accuracy(out_fused, out_ref, atol=0.1, rtol=0.15, percent=0.95)


@situ_supported
def test_fused_matches_reference_with_latent_and_shared_experts():
    device = torch.device("cuda")
    config = _K3Config(
        hidden_size=512,
        routed_expert_hidden_size=256,
        latent_moe_use_norm=True,
        num_shared_experts=2,
    )
    fused, ref = _make_block_pair(config, device)

    torch.manual_seed(11)
    x = torch.randn(2, 33, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    out_fused = fused(x)
    out_ref = ref(x)
    assert out_fused.shape == x.shape and out_fused.dtype == x.dtype
    check_accuracy(out_fused, out_ref, atol=0.1, rtol=0.15, percent=0.95)


@situ_supported
def test_fc1_swap_mutation_breaks_accuracy():
    """Packing gate-first (HF order) must break the numerics.

    Guards the w3-first packing convention: if the kernel accepted either
    order, this mutation would silently pass and the convention assert in
    the docs would be untestable.
    """
    device = torch.device("cuda")
    config = _K3Config()
    fused, ref = _make_block_pair(config, device)

    # Rebuild the fused buffers with w1/w3 swapped.
    from tensorrt_llm._torch.modules.kimi_k3_moe._moe_kernels import pack_routed_expert_weights

    swapped = pack_routed_expert_weights(
        w1_packed=fused.expert_bank.w3_packed,
        w1_scales=fused.expert_bank.w3_scales,
        w3_packed=fused.expert_bank.w1_packed,
        w3_scales=fused.expert_bank.w1_scales,
        w2_packed=fused.expert_bank.w2_packed,
        w2_scales=fused.expert_bank.w2_scales,
        device=device,
    )
    fused.gemm1_weights = swapped["gemm1_weights"]
    fused.gemm1_weights_scale = swapped["gemm1_weights_scale"]

    torch.manual_seed(13)
    x = torch.randn(1, 64, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5
    out_fused = fused(x)
    out_ref = ref(x)
    with pytest.raises(Exception, match="Mismatch percentage"):
        check_accuracy(out_fused, out_ref, atol=0.1, rtol=0.15, percent=0.95)


@situ_supported
def test_swiglu_act_mutation_breaks_accuracy():
    """Running the same weights through SwiGlu kernels must not match SiTU."""
    device = torch.device("cuda")
    config = _K3Config()
    fused, ref = _make_block_pair(config, device)

    import tensorrt_llm._torch.modules.kimi_k3_moe._moe_kernels as mk

    torch.manual_seed(17)
    x = torch.randn(1, 64, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5

    orig = mk.invoke_native_situ_moe

    def swiglu_invoke(**kwargs):
        kwargs["act_type"] = int(ActType_TrtllmGen.SwiGlu)
        return orig(**kwargs)

    from tensorrt_llm._torch.modules import kimi_k3_moe

    kimi_k3_moe.kimi_k3_moe_block.invoke_native_situ_moe = swiglu_invoke
    try:
        out_fused = fused(x)
    finally:
        kimi_k3_moe.kimi_k3_moe_block.invoke_native_situ_moe = orig

    out_ref = ref(x)
    with pytest.raises(Exception, match="Mismatch percentage"):
        check_accuracy(out_fused, out_ref, atol=0.1, rtol=0.15, percent=0.95)


@situ_supported
def test_fused_forward_without_weights_raises():
    device = torch.device("cuda")
    config = _K3Config()
    block = KimiK3SparseMoeBlock(
        config, use_fused_cubin=True, dtype=torch.bfloat16, device=device
    ).to(device)
    _init_block_weights(block)  # bank filled, but fused buffers NOT built
    x = torch.randn(1, 4, config.hidden_size, dtype=torch.bfloat16, device=device)
    with pytest.raises(RuntimeError, match="fused weights were never built"):
        block(x)
