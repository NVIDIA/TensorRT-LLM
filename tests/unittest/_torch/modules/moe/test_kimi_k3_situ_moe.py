# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 native TRTLLM-Gen SiTU MoE tests.

Covers the SiTU cubin integration behavior:

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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from _torch.modules.moe.kimi_k3_ref_moe._moe_kernels import (
    is_native_situ_supported,
    make_situ_alpha_beta,
    padded_fused_shapes,
)
from _torch.modules.moe.kimi_k3_ref_moe.kimi_k3_moe_block import KimiK3SparseMoeBlock
from utils.util import check_accuracy

import tensorrt_llm._torch.models.modeling_kimi_linear as modeling_kimi_linear
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import (
    _K3_MOE_EP_ENV,
    _K3_MOE_TP_ENV,
    KimiK3MoEGate,
    KimiK3MoERuntime,
)
from tensorrt_llm._torch.modules.fused_moe.communication import CommunicationFactory
from tensorrt_llm._torch.modules.fused_moe.mega_moe.mega_moe_deepgemm import (
    _MEGA_MOE_SYMM_BUFFER_CACHE,
)
from tensorrt_llm._torch.utils import ActType_TrtllmGen
from tensorrt_llm._utils import get_free_port
from tensorrt_llm.mapping import Mapping

situ_supported = pytest.mark.skipif(
    not is_native_situ_supported(),
    reason="native SiTU cubins require SM100/SM103 (Blackwell)",
)


@pytest.fixture
def _single_rank_nccl_process_group(monkeypatch):
    if dist.is_initialized():
        yield
        return
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(get_free_port()))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    try:
        yield
    finally:
        _MEGA_MOE_SYMM_BUFFER_CACHE.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


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


def test_communication_factory_accepts_model_selected_method(monkeypatch):
    mapping = SimpleNamespace(
        enable_attention_dp=True,
        dp_size=16,
        moe_tp_size=1,
        moe_ep_size=16,
    )
    model_config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(hidden_size=3584),
        torch_dtype=torch.bfloat16,
        quant_config=None,
        max_num_tokens=4096,
        moe_max_num_tokens=65536,
        use_cuda_graph=False,
        use_low_precision_moe_combine=False,
    )
    selected = object()
    method = None

    def create_forced_method(force_method, *args, **kwargs):
        nonlocal method
        method = force_method
        return selected

    monkeypatch.delenv("TRTLLM_FORCE_COMM_METHOD", raising=False)
    monkeypatch.setattr(
        CommunicationFactory,
        "_create_forced_method",
        staticmethod(create_forced_method),
    )
    actual = CommunicationFactory.create_strategy(
        model_config=model_config,
        num_experts=896,
        num_slots=896,
        top_k=16,
        expert_size_per_partition=56,
        hidden_size=3584,
        communication_method="ALLGATHER",
    )

    assert method == "ALLGATHER"
    assert actual is selected


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
    with pytest.raises(RuntimeError, match="must be > 0"):
        make_situ_alpha_beta(
            local_num_experts=8,
            situ_beta=4.0,
            situ_linear_beta=0.0,
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
    from _torch.modules.moe.kimi_k3_ref_moe._moe_kernels import pack_routed_expert_weights

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

    import _torch.modules.moe.kimi_k3_ref_moe._moe_kernels as mk

    torch.manual_seed(17)
    x = torch.randn(1, 64, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5

    orig = mk.invoke_native_situ_moe

    def swiglu_invoke(**kwargs):
        kwargs["act_type"] = int(ActType_TrtllmGen.SwiGlu)
        return orig(**kwargs)

    from _torch.modules.moe.kimi_k3_ref_moe import kimi_k3_moe_block

    kimi_k3_moe_block.invoke_native_situ_moe = swiglu_invoke
    try:
        out_fused = fused(x)
    finally:
        kimi_k3_moe_block.invoke_native_situ_moe = orig

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


# ---------------------------------------------------------------------------
# Routed MoE TP/EP split selection (CPU-only).
# ---------------------------------------------------------------------------


def test_mapping_records_moe_tp_ep_user_specified():
    # Auto default: -1 sentinels resolve to (moe_tp=tp, moe_ep=1) but must
    # NOT be flagged as a user request.
    auto = Mapping(world_size=8, tp_size=8)
    assert auto.moe_tp_size == 8 and auto.moe_ep_size == 1
    assert not auto.moe_tp_ep_user_specified

    tp = Mapping(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert tp.moe_tp_ep_user_specified

    # Setting only one side still counts as explicit.
    ep = Mapping(world_size=8, tp_size=8, moe_ep_size=8)
    assert ep.moe_tp_ep_user_specified
    assert ep.moe_tp_size == 1 and ep.moe_ep_size == 8


def test_kimi_k3_moe_split_selection(monkeypatch):
    monkeypatch.delenv(_K3_MOE_TP_ENV, raising=False)
    monkeypatch.delenv(_K3_MOE_EP_ENV, raising=False)

    # Auto mapping default stays EP-only (the historical K3 layout), even
    # though the resolved mapping says moe_tp=8.
    auto = Mapping(world_size=8, tp_size=8)
    assert KimiK3MoERuntime._select_moe_tp_ep(auto) == (1, 8)

    # Explicit pure-TP and hybrid requests are honored.
    tp = Mapping(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert KimiK3MoERuntime._select_moe_tp_ep(tp) == (8, 1)
    tep = Mapping(world_size=8, tp_size=8, moe_tp_size=4, moe_ep_size=2)
    assert KimiK3MoERuntime._select_moe_tp_ep(tep) == (4, 2)

    # Env override wins; a single side derives the other from tp_size.
    monkeypatch.setenv(_K3_MOE_TP_ENV, "4")
    assert KimiK3MoERuntime._select_moe_tp_ep(auto) == (4, 2)
    monkeypatch.delenv(_K3_MOE_TP_ENV)
    monkeypatch.setenv(_K3_MOE_EP_ENV, "2")
    assert KimiK3MoERuntime._select_moe_tp_ep(auto) == (4, 2)


@pytest.mark.parametrize("backend", ["TRTLLM", "MEGAMOE_DEEPGEMM"])
def test_kimi_k3_routed_config_preserves_explicit_backend(backend):
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        moe_backend=backend,
    )

    routed_model_config = KimiK3MoERuntime._routed_moe_model_config(model_config)

    assert routed_model_config.moe_backend == backend
    assert model_config.moe_backend == backend


@pytest.mark.parametrize(
    "backend,expected_moe_max_num_tokens",
    [
        pytest.param("TRTLLM", 33024, id="trtllm"),
        pytest.param("MEGAMOE_DEEPGEMM", 131072, id="megamoe"),
    ],
)
def test_kimi_k3_routed_config_scopes_megamoe_capacity(backend, expected_moe_max_num_tokens):
    model_config = ModelConfig(
        mapping=Mapping(
            world_size=16,
            rank=0,
            tp_size=16,
            moe_ep_size=16,
            enable_attention_dp=True,
        ),
        max_num_tokens=8192,
        moe_max_num_tokens=33024,
        moe_backend=backend,
    )

    routed_model_config = KimiK3MoERuntime._routed_moe_model_config(model_config)

    assert routed_model_config.moe_max_num_tokens == expected_moe_max_num_tokens
    assert model_config.moe_max_num_tokens == 33024


def test_kimi_k3_routed_config_logs_megamoe_capacity_override(monkeypatch):
    info_once = MagicMock()
    monkeypatch.setattr(modeling_kimi_linear.logger, "info_once", info_once)
    model_config = ModelConfig(
        mapping=Mapping(
            world_size=16,
            rank=0,
            tp_size=16,
            moe_ep_size=16,
            enable_attention_dp=True,
        ),
        max_num_tokens=8192,
        moe_max_num_tokens=33024,
        moe_backend="MEGAMOE_DEEPGEMM",
    )

    KimiK3MoERuntime._routed_moe_model_config(model_config)

    info_once.assert_any_call(
        "Kimi K3 MegaMoE raises moe_max_num_tokens from 33024 to 131072 "
        "because the global DP SymmBuffer requires capacity for "
        "max_num_tokens * dp_size.",
        key="kimi_k3_megamoe_capacity_override_33024_131072",
    )


def test_kimi_k3_routed_config_rejects_backend_without_situ_support():
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        moe_backend="CUTLASS",
    )

    with pytest.raises(ValueError, match="SiTU routed experts only support"):
        KimiK3MoERuntime._routed_moe_model_config(model_config)

    assert model_config.moe_backend == "CUTLASS"


@pytest.mark.parametrize(
    "architecture", ["KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"]
)
def test_kimi_k3_moe_auto_backend_defaults_to_trtllm(architecture):
    assert ModelConfig.resolve_moe_backend("AUTO", architecture) == "TRTLLM"


# ---------------------------------------------------------------------------
# MoE tensor-parallel shard parity (ConfigurableMoE / TRTLLM-Gen, GPU).
#
# Production K3 TP8 geometry per rank: ALL experts, intermediate 3072/8=384,
# latent hidden 3584, group-32 packed MXFP4 weights column-sharded (w1/w3)
# and row-sharded (w2) by the stock TRTLLM-Gen quant-method loaders. These
# tests run the identical shard shapes on ONE GPU by loading each simulated
# rank through the real `load_packed_mxfp4_expert` path with a proxy module
# exposing (tp_size, tp_rank), then summing the per-rank partial outputs.
# ---------------------------------------------------------------------------

_TP_HIDDEN = 3584
_TP_INTERMEDIATE = 3072
_TP_EXPERTS = 8
_TP_TOPK = 2


def _make_packed_expert_bank(num_experts, intermediate, hidden, seed=101):
    """Random group-32 packed MXFP4 tensors in checkpoint layout (uint8)."""
    gen = torch.Generator().manual_seed(seed)

    def nibbles(*shape):
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    def scales(*shape):
        # UE8M0 exponents 2^-9..2^-4 keep bf16 outputs well-conditioned.
        return torch.randint(118, 124, shape, generator=gen, dtype=torch.uint8)

    bank = []
    for _ in range(num_experts):
        bank.append(
            {
                "w1": nibbles(intermediate, hidden // 2),
                "w1_sf": scales(intermediate, hidden // 32),
                "w3": nibbles(intermediate, hidden // 2),
                "w3_sf": scales(intermediate, hidden // 32),
                "w2": nibbles(hidden, intermediate // 2),
                "w2_sf": scales(hidden, intermediate // 32),
            }
        )
    return bank


def _make_test_gate(num_experts=_TP_EXPERTS, top_k=_TP_TOPK, seed=71):
    """One deterministically-initialized gate SHARED by all modules under
    comparison: the fused routing kernel applies the gate's
    e_score_correction_bias per module, so a per-module `torch.empty`
    (garbage) bias would silently route the shard and whole-expert modules
    to different experts."""
    cfg = _K3Config(
        hidden_size=_TP_HIDDEN,
        num_experts=num_experts,
        num_experts_per_token=min(top_k, num_experts),
    )
    gate = KimiK3MoEGate(cfg)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        gate.weight.copy_(torch.randn(gate.weight.shape, generator=gen, dtype=torch.float32) * 0.05)
        gate.e_score_correction_bias.copy_(
            torch.randn(gate.e_score_correction_bias.shape, generator=gen, dtype=torch.float32)
            * 0.1
        )
    return gate.cuda()


def _make_routed_moe(
    intermediate_size,
    gate,
    num_experts=_TP_EXPERTS,
    moe_backend="TRTLLM",
):
    """Mirror KimiK3MoERuntime's create_moe call on a single-rank mapping."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.modules.fused_moe import ConfigurableMoE, create_moe
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = _TP_HIDDEN
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16
    pretrained_config.activation_situ_beta = 4.0
    pretrained_config.activation_situ_linear_beta = 25.0
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(),
        moe_backend=moe_backend,
    )
    moe_kwargs = dict(
        routing_method=gate.routing_method,
        num_experts=num_experts,
        hidden_size=_TP_HIDDEN,
        intermediate_size=intermediate_size,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=model_config,
        override_quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
        layer_idx=0,
        communication_method=None,
    )
    if moe_backend == "TRTLLM":
        moe_kwargs.update(
            trtllm_gen_activation_type=ActType_TrtllmGen.SiTu,
            trtllm_gen_activation_alpha=4.0,
            trtllm_gen_activation_beta=25.0,
        )
    else:
        moe_kwargs.update(
            activation="situ",
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
    moe = create_moe(**moe_kwargs).cuda()
    assert isinstance(moe, ConfigurableMoE)
    return moe


def _load_bank(moe, bank, tp_size=1, tp_rank=0):
    """Load packed experts through the production per-expert adapter.

    ``tp_size > 1`` simulates one MoE-TP rank: the proxy exposes the shard
    coordinates so the stock `load_weight_shard` slicing runs exactly as it
    would on a real multi-rank mapping, while the backing module holds the
    shard-sized parameters.
    """
    backend = moe.backend
    if hasattr(backend.quant_method, "load_packed_mxfp4_expert"):
        loader_module = backend
        if hasattr(backend, "scaling_vector_size"):
            loader_module = SimpleNamespace(
                expert_size_per_partition=backend.expert_size_per_partition,
                initial_local_expert_ids=backend.initial_local_expert_ids,
                scaling_vector_size=backend.scaling_vector_size,
                tp_size=tp_size,
                tp_rank=tp_rank,
                w3_w1_weight=backend.w3_w1_weight,
                w2_weight=backend.w2_weight,
                w3_w1_weight_scale=backend.w3_w1_weight_scale,
                w2_weight_scale=backend.w2_weight_scale,
            )
        for expert_id, tensors in enumerate(bank):
            backend.quant_method.load_packed_mxfp4_expert(
                loader_module,
                global_expert_id=expert_id,
                local_slot_id=expert_id,
                w1_weight=tensors["w1"],
                w1_weight_scale=tensors["w1_sf"],
                w2_weight=tensors["w2"],
                w2_weight_scale=tensors["w2_sf"],
                w3_weight=tensors["w3"],
                w3_weight_scale=tensors["w3_sf"],
            )
    else:
        weights = {}
        for expert_id, tensors in enumerate(bank):
            prefix = f"{expert_id}."
            weights.update(
                {
                    prefix + "w1.weight": tensors["w1"],
                    prefix + "w1.weight_scale": tensors["w1_sf"],
                    prefix + "w2.weight": tensors["w2"],
                    prefix + "w2.weight_scale": tensors["w2_sf"],
                    prefix + "w3.weight": tensors["w3"],
                    prefix + "w3.weight_scale": tensors["w3_sf"],
                }
            )
        backend.load_weights([weights])
    backend._weights_transformed = False
    moe.post_load_weights()
    return moe


@situ_supported
@pytest.mark.parametrize("tp_size", [2, 8], ids=lambda n: f"tp{n}")
def test_tp_shard_loader_matches_manual_slice(tp_size):
    """The stock shard loaders must equal a manual contiguous slice.

    Guards the group-32 packed-byte / scale slicing assumptions: w1/w3
    column-shard along intermediate rows, w2 row-shard along the packed
    intermediate bytes and per-32-group scales.
    """
    ipp = _TP_INTERMEDIATE // tp_size
    num_experts = 2
    bank = _make_packed_expert_bank(num_experts, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate(num_experts=num_experts)

    for tp_rank in (0, tp_size - 1):
        via_shard = _make_routed_moe(ipp, gate, num_experts=num_experts)
        _load_bank(via_shard, bank, tp_size=tp_size, tp_rank=tp_rank)

        rows = slice(tp_rank * ipp, (tp_rank + 1) * ipp)
        cols_packed = slice(tp_rank * (ipp // 2), (tp_rank + 1) * (ipp // 2))
        cols_sf = slice(tp_rank * (ipp // 32), (tp_rank + 1) * (ipp // 32))
        manual_bank = [
            {
                "w1": e["w1"][rows].contiguous(),
                "w1_sf": e["w1_sf"][rows].contiguous(),
                "w3": e["w3"][rows].contiguous(),
                "w3_sf": e["w3_sf"][rows].contiguous(),
                "w2": e["w2"][:, cols_packed].contiguous(),
                "w2_sf": e["w2_sf"][:, cols_sf].contiguous(),
            }
            for e in bank
        ]
        via_manual = _make_routed_moe(ipp, gate, num_experts=num_experts)
        _load_bank(via_manual, manual_bank)

        for name in ("w3_w1_weight", "w2_weight", "w3_w1_weight_scale", "w2_weight_scale"):
            a = getattr(via_shard.backend, name).data
            b = getattr(via_manual.backend, name).data
            assert torch.equal(a, b), f"{name} mismatch for tp_size={tp_size} tp_rank={tp_rank}"


@situ_supported
@pytest.mark.parametrize("num_tokens", [1, 16], ids=lambda n: f"tokens{n}")
def test_tp8_sharded_forward_matches_whole_expert(num_tokens):
    """Sum of 8 TP-shard partial outputs == whole-expert reference.

    Per-element MXFP4/MXFP8 numerics are identical between the two layouts
    (group-32 boundaries align: 384 % 32 == 0), so the only expected error
    is bf16 rounding of the per-shard FC2 partial sums.
    """
    tp_size = 8
    ipp = _TP_INTERMEDIATE // tp_size  # 384 — the production TP8 shard size
    bank = _make_packed_expert_bank(_TP_EXPERTS, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate()

    whole = _make_routed_moe(_TP_INTERMEDIATE, gate)
    _load_bank(whole, bank)

    torch.manual_seed(3)
    x = torch.randn(num_tokens, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    router_logits = gate.compute_logits(x)

    out_whole = whole.forward(x, router_logits, all_rank_num_tokens=None)

    partial_sum = torch.zeros(num_tokens, _TP_HIDDEN, dtype=torch.float32, device="cuda")
    for tp_rank in range(tp_size):
        shard = _make_routed_moe(ipp, gate)
        _load_bank(shard, bank, tp_size=tp_size, tp_rank=tp_rank)
        out_shard = shard.forward(x, router_logits, all_rank_num_tokens=None)
        partial_sum += out_shard.float()
        del shard
        torch.cuda.empty_cache()

    check_accuracy(partial_sum.to(torch.bfloat16), out_whole, atol=0.08, rtol=0.08, percent=0.98)


@situ_supported
@pytest.mark.parametrize("num_tokens", [1, 16, 128], ids=lambda n: f"tokens{n}")
@pytest.mark.parametrize(
    "num_experts,top_k",
    [
        pytest.param(8, 1, id="experts8-top1"),
        pytest.param(8, 2, id="experts8-top2"),
        pytest.param(32, 16, id="experts32-top16"),
    ],
)
def test_megamoe_deepgemm_situ_matches_trtllm_gen(
    num_tokens, num_experts, top_k, _single_rank_nccl_process_group
):
    """Compare SiTU kernels with identical packed MXFP4 weights and routing.

    MegaMoE folds routing weights into the FC1 activation before its MXFP8
    requantization, while TRTLLM-Gen combines after the expert output. The
    quantized graphs therefore need semantic, rather than elementwise,
    parity: high cosine similarity and bounded relative L2 error.
    """
    bank = _make_packed_expert_bank(num_experts, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate(num_experts=num_experts, top_k=top_k)

    trtllm_gen = _load_bank(
        _make_routed_moe(_TP_INTERMEDIATE, gate, num_experts=num_experts),
        bank,
    )
    mega_moe = _load_bank(
        _make_routed_moe(
            _TP_INTERMEDIATE,
            gate,
            num_experts=num_experts,
            moe_backend="MEGAMOE_DEEPGEMM",
        ),
        bank,
    )

    torch.manual_seed(37)
    x = torch.randn(num_tokens, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    router_logits = gate.compute_logits(x)

    with torch.inference_mode():
        trtllm_gen_output = trtllm_gen(x, router_logits)
        mega_moe_output = mega_moe(x, router_logits)

    assert torch.isfinite(mega_moe_output).all()
    diff = (mega_moe_output.float() - trtllm_gen_output.float()).abs()
    ref = trtllm_gen_output.float()
    cosine = torch.nn.functional.cosine_similarity(
        mega_moe_output.float().flatten(),
        ref.flatten(),
        dim=0,
    )
    relative_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)
    print(
        f"M={num_tokens}, experts={num_experts}, top_k={top_k}: "
        f"cosine={cosine.item():.8f}, "
        f"relative_l2={relative_l2.item():.8f}, "
        f"mean_abs={diff.mean().item():.8f}, "
        f"p99_abs={torch.quantile(diff, 0.99).item():.8f}, "
        f"max_abs={diff.max().item():.8f}"
    )
    assert cosine > 0.998
    assert relative_l2 < 0.06
