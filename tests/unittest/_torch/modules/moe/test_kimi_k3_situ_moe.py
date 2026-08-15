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
from types import SimpleNamespace

import pytest
import torch
from _torch.modules.moe.kimi_k3_ref_moe._moe_kernels import (
    is_native_situ_supported,
    make_situ_alpha_beta,
    padded_fused_shapes,
)
from _torch.modules.moe.kimi_k3_ref_moe.kimi_k3_moe_block import KimiK3SparseMoeBlock
from utils.util import check_accuracy

from tensorrt_llm._torch.modules.fused_moe.communication import CommunicationFactory
from tensorrt_llm._torch.modules.kimi_k3_moe.kimi_k3_moe_gate import KimiK3MoEGate
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


def test_kimi_gate_reuses_deepseek_v3_routing():
    config = _K3Config(num_experts=16, num_experts_per_token=4)
    gate = KimiK3MoEGate(config)
    torch.manual_seed(23)
    with torch.no_grad():
        gate.weight.normal_(std=0.1)
        gate.e_score_correction_bias.normal_(std=0.05)
    hidden_states = torch.randn(2, 7, config.hidden_size)

    expected_ids, expected_weights = gate(hidden_states)
    routing_method = gate.routing_method
    # Exercise the portable PyTorch short path; the production path keeps
    # is_fused=True and uses the same routing contract.
    routing_method.routing_impl.is_fused = False
    actual_ids, actual_weights = routing_method.apply(gate.compute_logits(hidden_states))

    expected_order = expected_ids.argsort(dim=-1)
    actual_order = actual_ids.argsort(dim=-1)
    assert actual_ids.dtype == torch.int32
    torch.testing.assert_close(
        expected_ids.gather(1, expected_order).to(actual_ids.dtype),
        actual_ids.gather(1, actual_order),
    )
    torch.testing.assert_close(
        expected_weights.gather(1, expected_order),
        actual_weights.gather(1, actual_order),
    )


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
    from tensorrt_llm.mapping import Mapping

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
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _K3_MOE_EP_ENV,
        _K3_MOE_TP_ENV,
        KimiK3MoERuntime,
    )
    from tensorrt_llm.mapping import Mapping

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


def _make_test_gate(num_experts=_TP_EXPERTS, seed=71):
    """One deterministically-initialized gate SHARED by all modules under
    comparison: the fused routing kernel applies the gate's
    e_score_correction_bias per module, so a per-module `torch.empty`
    (garbage) bias would silently route the shard and whole-expert modules
    to different experts."""
    cfg = _K3Config(
        hidden_size=_TP_HIDDEN,
        num_experts=num_experts,
        num_experts_per_token=min(_TP_TOPK, num_experts),
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


def _make_routed_moe(intermediate_size, gate, num_experts=_TP_EXPERTS):
    """Mirror KimiK3MoERuntime's create_moe call on a single-rank mapping."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.fused_moe import ConfigurableMoE, create_moe
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = _TP_HIDDEN
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(),
        moe_backend="TRTLLM",
    )
    moe = create_moe(
        routing_method=gate.routing_method,
        num_experts=num_experts,
        hidden_size=_TP_HIDDEN,
        intermediate_size=intermediate_size,
        dtype=torch.bfloat16,
        reduce_results=True,
        model_config=model_config,
        override_quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
        layer_idx=0,
        trtllm_gen_activation_type=ActType_TrtllmGen.SiTu,
        trtllm_gen_activation_alpha=4.0,
        trtllm_gen_activation_beta=25.0,
        communication_method=None,
    ).cuda()
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
    proxy = SimpleNamespace(
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
            proxy,
            global_expert_id=expert_id,
            local_slot_id=expert_id,
            w1_weight=tensors["w1"],
            w1_weight_scale=tensors["w1_sf"],
            w2_weight=tensors["w2"],
            w2_weight_scale=tensors["w2_sf"],
            w3_weight=tensors["w3"],
            w3_weight_scale=tensors["w3_sf"],
        )
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
