# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 native TRTLLM-Gen SiTU MoE tests.

Covers the SiTU cubin integration behavior:

* runner-local ActType numeric stability (SwiGlu/Relu2/Silu unchanged,
  SiTu appended);
* the native SiTU runner actually launches a kernel whose name contains
  ``siTuGlu``, and from the expected cubin family -- ``MxE4m3_MxE2m1`` for
  the MXFP4 drop, ``E2m1_E2m1E2m1`` for the NVFP4 one (tactic availability
  for both families is pinned at the op level in
  tests/unittest/_torch/thop/serial/test_moe.py);
* fused output matches the Python ``SituAndMul`` reference within
  MXFP8/MXFP4 quantization tolerance, for the default AND asymmetric
  non-default ``activation_situ_beta`` / ``activation_situ_linear_beta``;
* swapping the FC1 halves (gate-first packing) breaks accuracy
  (mutation test — proves the w3-first convention is load-bearing);
* module-level semantics (latent projections, RMSNorm, shared experts)
  survive the fused path;
* the fused path fails loudly without loaded weights (no silent
  random-weight fallback).

The NVFP4 half of SiTU lives here too, on both FP4 backends. Both take it as
one ``SiTuActivation`` carrier; they differ only in what serves it -- CUTLASS
an ``ActivationType`` its kernels branch on, TRTLLM-Gen the fused
``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*`` FC1 cubins. Tests that apply to both are
parametrized over ``moe_backend`` rather than duplicated.
"""

import dataclasses
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from _torch.moe.kimi_k3_ref_moe._moe_kernels import (
    is_native_situ_supported,
    make_situ_alpha_beta,
    padded_fused_shapes,
)
from _torch.moe.kimi_k3_ref_moe.kimi_k3_moe_block import KimiK3SparseMoeBlock
from utils.util import check_accuracy

import tensorrt_llm._torch.models.modeling_kimi_linear as modeling_kimi_linear
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_kimi_linear import KimiK3MoEGate, KimiK3MoERuntime
from tensorrt_llm._torch.moe.fused_moe.communication import CommunicationFactory
from tensorrt_llm._torch.moe.fused_moe.mega_moe.mega_moe_deepgemm import _MEGA_MOE_SYMM_BUFFER_CACHE
from tensorrt_llm._torch.utils import ActType_TrtllmGen
from tensorrt_llm._utils import get_free_port, get_sm_version
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

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


def test_kimi_situ_betas_require_linear_beta():
    cfg = SimpleNamespace(activation_situ_beta=4.0, activation_situ_linear_beta=None)

    with pytest.raises(ValueError, match="require activation_situ_linear_beta"):
        modeling_kimi_linear._resolve_kimi_situ_betas(cfg)


@pytest.mark.parametrize(
    "situ_beta,situ_linear_beta",
    [(0.0, 25.0), (4.0, 0.0), (-1.0, 25.0), (4.0, -1.0)],
)
def test_kimi_situ_betas_must_be_positive(situ_beta, situ_linear_beta):
    cfg = SimpleNamespace(
        activation_situ_beta=situ_beta,
        activation_situ_linear_beta=situ_linear_beta,
    )

    with pytest.raises(ValueError, match="must be positive"):
        modeling_kimi_linear._resolve_kimi_situ_betas(cfg)


def test_clear_checkpoint_fp8_pairs_releases_unconsumed_stashes():
    linear = torch.nn.Linear(2, 2, bias=False)
    setattr(linear.weight, modeling_kimi_linear._K3_CKPT_FP8_ATTR, (torch.ones(1), torch.ones(1)))

    assert modeling_kimi_linear._clear_checkpoint_fp8_pairs(linear) == 1
    assert not hasattr(linear.weight, modeling_kimi_linear._K3_CKPT_FP8_ATTR)


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
        has_cp_helix=lambda: False,
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


_LAUNCH_EVIDENCE_SCRIPTS = {
    # MXFP4: the K3 fused block (MXFP8 activations x group-32 MXFP4 weights).
    "mxfp4": r"""
import torch
from test_kimi_k3_situ_moe import _K3Config, _make_block_pair

device = torch.device("cuda")
config = _K3Config()
fused, _ = _make_block_pair(config, device)
x = torch.randn(1, 16, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5
fused(x)
torch.cuda.synchronize()
assert fused._cubin_call_count == 1
""",
    # NVFP4: the routed MoE on the TRTLLM-Gen backend (group-16 scales).
    "nvfp4": r"""
import torch
from test_kimi_k3_situ_moe import (
    _TP_EXPERTS,
    _TP_HIDDEN,
    _TP_INTERMEDIATE,
    _load_nvfp4_bank_for,
    _make_nvfp4_expert_bank,
    _make_nvfp4_moe,
    _make_test_gate,
)

gate = _make_test_gate()
moe = _make_nvfp4_moe(gate, moe_backend="TRTLLM")
bank = _make_nvfp4_expert_bank(_TP_EXPERTS, _TP_INTERMEDIATE, _TP_HIDDEN)
_load_nvfp4_bank_for(moe, bank, "TRTLLM")

torch.manual_seed(5)
x = torch.randn(16, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
out = moe.forward(x, gate.compute_logits(x), all_rank_num_tokens=None)
torch.cuda.synchronize()
assert torch.isfinite(out).all()
""",
}

# What separates the two fused SiTu FC1 cubin families in the launch log.
_SITU_CUBIN_FAMILY = {"mxfp4": "MxE4m3_MxE2m1", "nvfp4": "E2m1_E2m1E2m1"}


@situ_supported
@pytest.mark.parametrize("fmt", ["mxfp4", "nvfp4"])
def test_fused_forward_launches_situ_kernel(fmt):
    """Launch evidence: the FC1 kernel selected must be a siTuGlu cubin from
    the expected quantization family.

    ``siTuGlu`` alone only proves that *some* SiTu kernel ran. The family
    substring is what separates the MXFP4 drop from the NVFP4 one, so a path
    silently resolving to the other family still fails here.

    Runs in a subprocess because the C++ logger level is fixed at process
    start (TLLM_LOG_LEVEL) and TLLM_BATCHED_GEMM_PRINT_NAME logs at INFO.
    """
    import subprocess
    import sys

    env = dict(os.environ)
    env["TLLM_BATCHED_GEMM_PRINT_NAME"] = "1"
    env["TLLM_LOG_LEVEL"] = "INFO"
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # The child process imports ``_torch.moe.kimi_k3_ref_moe``, so the root it
    # needs on PYTHONPATH is tests/unittest, not the repo's tests/ directory.
    unittest_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    env["PYTHONPATH"] = os.pathsep.join([this_dir, unittest_root, env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", _LAUNCH_EVIDENCE_SCRIPTS[fmt]],
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
    family = _SITU_CUBIN_FAMILY[fmt]
    assert family.lower() in log.lower(), (
        f"expected a {fmt.upper()} ({family}) SiTu cubin; got:\n" + log[-4000:]
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
    from _torch.moe.kimi_k3_ref_moe._moe_kernels import pack_routed_expert_weights

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

    import _torch.moe.kimi_k3_ref_moe._moe_kernels as mk

    torch.manual_seed(17)
    x = torch.randn(1, 64, config.hidden_size, dtype=torch.bfloat16, device=device) * 0.5

    orig = mk.invoke_native_situ_moe

    def swiglu_invoke(**kwargs):
        kwargs["act_type"] = int(ActType_TrtllmGen.SwiGlu)
        return orig(**kwargs)

    from _torch.moe.kimi_k3_ref_moe import kimi_k3_moe_block

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


def test_kimi_k3_moe_split_selection() -> None:
    # Auto mapping default stays EP-only (the historical K3 layout), even
    # though the resolved mapping says moe_tp=8.
    auto = Mapping(world_size=8, tp_size=8)
    assert KimiK3MoERuntime._select_moe_tp_ep(auto) == (1, 8)

    # Explicit pure-TP and hybrid requests are honored. The MoE split is only
    # ever set through the config; there is no env override.
    tp = Mapping(world_size=8, tp_size=8, moe_tp_size=8, moe_ep_size=1)
    assert KimiK3MoERuntime._select_moe_tp_ep(tp) == (8, 1)
    tep = Mapping(world_size=8, tp_size=8, moe_tp_size=4, moe_ep_size=2)
    assert KimiK3MoERuntime._select_moe_tp_ep(tep) == (4, 2)


@pytest.mark.parametrize("backend", ["CUTLASS", "TRTLLM", "MEGAMOE_DEEPGEMM", "MEGAMOE_CUTEDSL"])
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
        pytest.param("MEGAMOE_DEEPGEMM", 131072, id="megamoe-deepgemm"),
        pytest.param("MEGAMOE_CUTEDSL", 131072, id="megamoe-cutedsl"),
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
        moe_backend="TRITON",
    )

    with pytest.raises(ValueError, match="SiTU routed experts only support"):
        KimiK3MoERuntime._routed_moe_model_config(model_config)

    assert model_config.moe_backend == "TRITON"


@pytest.mark.parametrize(
    "architecture", ["KimiK3ForConditionalGeneration", "KimiLinearForCausalLM"]
)
def test_kimi_k3_moe_auto_backend_defaults_to_trtllm(architecture):
    assert ModelConfig.resolve_moe_backend("AUTO", architecture) == "TRTLLM"


# ---------------------------------------------------------------------------
# The K3 model layer and TRTLLMGenFusedMoE both have to know which routed-expert
# formats trtllm-gen has a fused SiTu FC1 cubin for. They disagreed once: the
# model's copy was written when MXFP4 was the only drop (#17865) and #17940 then
# shipped the NVFP4 cubins and updated only the backend, so for a week an NVFP4
# K3 checkpoint raised at construction on a path that the kernels supported.
#
# It survived because every other SiTu test calls ``create_moe`` directly and so
# never reaches the model-layer guard -- the kernel path was green throughout.
# These two tests enter through the guard instead.
# ---------------------------------------------------------------------------


def test_kimi_k3_trtllm_situ_admits_every_backend_supported_quant():
    """The model must not narrow what the backend says it can serve.

    Asserting agreement rather than a literal set is the point: a new fused
    SiTu cubin family should require no edit here, and removing one should
    fail loudly rather than leave a stale allow-list behind.
    """
    from tensorrt_llm._torch.moe.fused_moe import TRTLLMGenFusedMoE

    supported = TRTLLMGenFusedMoE.situ_supported_quant_algos()
    assert QuantAlgo.NVFP4 in supported, (
        "trtllm-gen has shipped group-16 Bmm_E2m1_E2m1E2m1_..._siTuGlu_* cubins "
        "since #17940; if this fails the backend regressed, not the model."
    )
    for algo in supported:
        KimiK3MoERuntime._check_trtllm_situ_quant("TRTLLM", algo)


@pytest.mark.parametrize("quant_algo", [QuantAlgo.FP8_BLOCK_SCALES, QuantAlgo.W4A16_MXFP4, None])
def test_kimi_k3_trtllm_situ_rejects_quant_without_fused_cubin(quant_algo):
    """...and must still reject the formats that have no fused SiTu cubin.

    ``resolve_moe_backend`` sends every K3 architecture to TRTLLM, including
    the generic FP8_BLOCK_SCALES fallback, so this rejection is reachable
    without anyone asking for TRTLLM by name. It has to name the fix.
    """
    from tensorrt_llm._torch.moe.fused_moe import TRTLLMGenFusedMoE

    assert quant_algo not in TRTLLMGenFusedMoE.situ_supported_quant_algos()
    with pytest.raises(ValueError, match="fused SiTu cubins exist only for"):
        KimiK3MoERuntime._check_trtllm_situ_quant("TRTLLM", quant_algo)

    # Any other backend owns its own SiTu translation and is not this guard's
    # business -- gating it here is how CUTLASS would get blocked by a
    # trtllm-gen cubin inventory.
    KimiK3MoERuntime._check_trtllm_situ_quant("CUTLASS", quant_algo)


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
    routed_quant_config=None,
):
    """Mirror KimiK3MoERuntime's create_moe call on a single-rank mapping."""
    from transformers.configuration_utils import PretrainedConfig

    from tensorrt_llm._torch.moe.fused_moe import ConfigurableMoE, SiTuActivation, create_moe
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
        override_quant_config=(
            routed_quant_config
            if routed_quant_config is not None
            else QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
        ),
        layer_idx=0,
        communication_method=None,
        # Mirror KimiK3MoERuntime exactly: one activation for every backend,
        # naming the two soft-caps rather than the ABI registers they land in.
        activation=SiTuActivation(gate_softcap=4.0, linear_softcap=25.0),
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


# ---------------------------------------------------------------------------
# NVFP4 streaming expert loading (nvidia/Kimi-K3-NVFP4).
#
# K3 never calls the MoE backend's whole-checkpoint ``load_weights``: the
# checkpoint is 1.5 TB, so experts are streamed one at a time through a
# per-expert adapter. For NVFP4 that adapter is
# ``NVFP4FusedMoEMethod.load_streaming_nvfp4_expert``, and the property that
# matters is that it lands the model in exactly the state ``load_weights``
# would have. Nothing about a wrong [w1 | w3] order or a lost slot raises —
# shapes still match and only accuracy moves — so it is asserted here.
# ---------------------------------------------------------------------------

_NVFP4_GROUP_SIZE = 16

nvfp4_moe_supported = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="NVFP4 Cutlass MoE requires Blackwell (SM100+)",
)


def _make_nvfp4_expert_bank(num_experts, intermediate, hidden, seed=907):
    """Random NVFP4 tensors in ``nvidia/Kimi-K3-NVFP4`` checkpoint layout."""
    gen = torch.Generator().manual_seed(seed)

    def nibbles(*shape):
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8)

    def block_scales(*shape):
        # Modest exponents keep the dequantized weights well-conditioned.
        return (
            torch.randint(120, 132, shape, generator=gen, dtype=torch.int32)
            .to(torch.float32)
            .div(126.0)
            .to(torch.float8_e4m3fn)
        )

    bank = []
    for _ in range(num_experts):
        expert = {
            "w1.weight": nibbles(intermediate, hidden // 2),
            "w1.weight_scale": block_scales(intermediate, hidden // _NVFP4_GROUP_SIZE),
            "w3.weight": nibbles(intermediate, hidden // 2),
            "w3.weight_scale": block_scales(intermediate, hidden // _NVFP4_GROUP_SIZE),
            "w2.weight": nibbles(hidden, intermediate // 2),
            "w2.weight_scale": block_scales(hidden, intermediate // _NVFP4_GROUP_SIZE),
        }
        for w in ("w1", "w2", "w3"):
            # The real checkpoint stores one global weight scale per tensor and
            # a static activation scale of 1.0.
            expert[f"{w}.weight_scale_2"] = torch.tensor(0.00012207, dtype=torch.float32)
            expert[f"{w}.input_scale"] = torch.tensor(1.0, dtype=torch.float32)
        bank.append(expert)
    return bank


def _make_nvfp4_moe(gate, num_experts=_TP_EXPERTS, moe_backend="CUTLASS"):
    """NVFP4 + SiTU routed MoE on either FP4 backend.

    Both take the same ``SiTuActivation`` carrier; CUTLASS serves it as an
    ``ActivationType`` its kernels branch on, TRTLLM-Gen with the fused
    ``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*`` FC1 cubins (group-16 block scales).
    ``_make_routed_moe`` already mirrors both of KimiK3MoERuntime's branches,
    so the backend is the only variable.
    """
    from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

    return _make_routed_moe(
        _TP_INTERMEDIATE,
        gate,
        num_experts=num_experts,
        moe_backend=moe_backend,
        routed_quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=_NVFP4_GROUP_SIZE),
    )


def _stream_nvfp4_bank(moe, bank, swap_w1_w3=False, num_threads=1):
    """Load a bank through the per-expert streaming adapter K3 uses."""
    backend = moe.backend
    quant_method = backend.quant_method
    quant_method.prepare_streaming_expert_load(backend)

    def load_one(expert_id):
        tensors = dict(bank[expert_id])
        if swap_w1_w3:
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
                tensors[f"w1.{kind}"], tensors[f"w3.{kind}"] = (
                    tensors[f"w3.{kind}"],
                    tensors[f"w1.{kind}"],
                )
        quant_method.load_streaming_nvfp4_expert(
            backend,
            global_expert_id=expert_id,
            local_slot_id=expert_id,
            **{
                f"{w}_{kind}": tensors[f"{w}.{kind}"]
                for w in ("w1", "w2", "w3")
                for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
            },
        )

    if num_threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(pool.map(load_one, range(len(bank))))
    else:
        for expert_id in range(len(bank)):
            load_one(expert_id)

    backend.process_weights_after_loading()
    return backend


def _load_nvfp4_bank_whole(moe, bank):
    """Load the same bank through the stock whole-checkpoint path."""
    weights = {}
    for expert_id, tensors in enumerate(bank):
        for key, value in tensors.items():
            weights[f"{expert_id}.{key}"] = value
    moe.backend.load_weights([weights])
    return moe.backend


def _load_nvfp4_bank_for(moe, bank, moe_backend):
    """Load a bank and run the post-load transform, ready for ``forward``.

    CUTLASS goes through K3's per-expert streaming adapter -- the path the real
    checkpoint loader takes, and the one the buffer-equality tests above pin.
    TRTLLM-Gen's NVFP4 quant method does not override
    ``prepare_streaming_expert_load``, so it takes the stock whole-checkpoint
    path instead; the loader is not what these tests are about, the kernel is.
    """
    if moe_backend == "CUTLASS":
        backend = _stream_nvfp4_bank(moe, bank)
        # _stream_nvfp4_bank already ran process_weights_after_loading, which
        # the whole-checkpoint path leaves to post_load_weights below.
        backend._weights_transformed = False
    else:
        backend = _load_nvfp4_bank_whole(moe, bank)
    moe.post_load_weights()
    return backend


_NVFP4_LOADED_STATE = (
    "w3_w1_weight",
    "w2_weight",
    "w3_w1_weight_scale",
    "w2_weight_scale",
    "fc31_alpha",
    "fc2_alpha",
    "fc31_input_scale",
    "fc2_input_scale",
)


def _bitwise_equal(actual, expected):
    """Exact equality, including for the packed / float8 buffers.

    ``torch.equal`` refuses float8 operands, so those are compared through a
    uint8 reinterpretation — which in turn needs a non-0-dim tensor, hence the
    ``reshape(-1)`` for the scalar input scales.
    """
    if actual.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        actual = actual.reshape(-1).view(torch.uint8)
        expected = expected.reshape(-1).view(torch.uint8)
    return torch.equal(actual, expected)


@nvfp4_moe_supported
@pytest.mark.parametrize("num_threads", [1, 4], ids=lambda n: f"threads{n}")
def test_nvfp4_streaming_expert_load_matches_whole_checkpoint(num_threads):
    """Streaming per-expert loading == stock whole-checkpoint loading.

    ``num_threads=4`` is the concurrency K3's file-grouped loader actually
    uses; the staging containers are pre-created for exactly this reason, and
    a lost slot would show up here as a mismatching buffer rather than as an
    error.
    """
    bank = _make_nvfp4_expert_bank(_TP_EXPERTS, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate()

    reference = _load_nvfp4_bank_whole(_make_nvfp4_moe(gate), bank)
    streamed = _stream_nvfp4_bank(_make_nvfp4_moe(gate), bank, num_threads=num_threads)

    assert streamed._streamed_expert_slots == set(range(_TP_EXPERTS))
    for name in _NVFP4_LOADED_STATE:
        expected = getattr(reference, name).data
        actual = getattr(streamed, name).data
        assert actual.shape == expected.shape, name
        assert _bitwise_equal(actual, expected), name


@nvfp4_moe_supported
def test_nvfp4_streaming_expert_load_w1_w3_order_is_load_bearing():
    """Mutation test for the trap that does not raise.

    ``w3_w1_weight`` holds the two halves in a fixed order. Feeding w1 and w3
    the wrong way round keeps every shape valid and every load silent, so
    without this assertion the only detector would be an accuracy run.
    """
    bank = _make_nvfp4_expert_bank(_TP_EXPERTS, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate()

    correct = _stream_nvfp4_bank(_make_nvfp4_moe(gate), bank)
    swapped = _stream_nvfp4_bank(_make_nvfp4_moe(gate), bank, swap_w1_w3=True)

    assert not _bitwise_equal(swapped.w3_w1_weight.data, correct.w3_w1_weight.data)


@nvfp4_moe_supported
def test_nvfp4_streaming_expert_load_rejects_duplicate_slot():
    bank = _make_nvfp4_expert_bank(2, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate()
    moe = _make_nvfp4_moe(gate)
    backend = moe.backend
    backend.quant_method.prepare_streaming_expert_load(backend)

    kwargs = dict(
        global_expert_id=0,
        local_slot_id=0,
        **{
            f"{w}_{kind}": bank[0][f"{w}.{kind}"]
            for w in ("w1", "w2", "w3")
            for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
        },
    )
    backend.quant_method.load_streaming_nvfp4_expert(backend, **kwargs)
    with pytest.raises(ValueError, match="loaded twice"):
        backend.quant_method.load_streaming_nvfp4_expert(backend, **kwargs)


def test_kimi_k3_expert_ckpt_spec_selection():
    """The three loader call sites share one layout decision."""
    from tensorrt_llm._torch.models.modeling_kimi_linear import _k3_expert_ckpt_spec

    mxfp4 = _k3_expert_ckpt_spec(QuantAlgo.W4A8_MXFP4_MXFP8)
    assert mxfp4.kinds == ("weight_packed", "weight_scale")
    assert not mxfp4.needs_layer_finalize

    nvfp4 = _k3_expert_ckpt_spec(QuantAlgo.NVFP4)
    assert nvfp4.kinds == ("weight", "weight_scale", "weight_scale_2", "input_scale")
    assert nvfp4.needs_layer_finalize

    with pytest.raises(NotImplementedError, match="no per-expert checkpoint layout"):
        _k3_expert_ckpt_spec(QuantAlgo.FP8)


def test_materialize_handles_scalar_lazy_safetensors(tmp_path):
    """The NVFP4 checkpoint stores weight_scale_2 / input_scale as 0-dim.

    ``_materialize`` realizes a lazy slice with ``[:]``, which a 0-dim entry
    rejects — so the fallback (non-file-grouped) expert load died on the very
    first NVFP4 scalar it touched.
    """
    import safetensors.torch
    from safetensors import safe_open

    from tensorrt_llm._torch.models.modeling_kimi_linear import _materialize

    scalar = torch.tensor(0.00012207, dtype=torch.float32)
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file({"scalar": scalar, "matrix": matrix}, path)

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert torch.equal(_materialize(handle.get_slice("scalar")), scalar)
        assert torch.equal(_materialize(handle.get_slice("matrix")), matrix)
    assert torch.equal(_materialize(scalar), scalar)


@nvfp4_moe_supported
def test_nvfp4_streamed_experts_forward_runs():
    """End-to-end guard for the geometry the loader writes into.

    The buffer-equality tests compare two loaders against each other, so they
    stay green even when both write into a wrongly-shaped destination. Only
    running the kernel catches that.
    """
    bank = _make_nvfp4_expert_bank(_TP_EXPERTS, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate()
    moe = _make_nvfp4_moe(gate)

    backend = _stream_nvfp4_bank(moe, bank)
    assert backend.w3_w1_weight.shape[1] == 2 * _TP_INTERMEDIATE
    backend._weights_transformed = False
    moe.post_load_weights()

    torch.manual_seed(5)
    x = torch.randn(16, _TP_HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
    out = moe.forward(x, gate.compute_logits(x), all_rank_num_tokens=None)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def _situ_reference_moe(x, router_logits, routing_method, w1, w2, w3, beta, linear_beta):
    """Golden K3 routed-MoE, straight from the SituAndMul definition.

    ``out = [beta*tanh(g/beta)*sigmoid(g)] * [linear_beta*tanh(u/linear_beta)]``
    with g = w1(x) (gate_proj) and u = w3(x) (up_proj).
    """
    ids, weights = routing_method.apply(router_logits)
    out = torch.zeros_like(x, dtype=torch.float32)
    xf = x.float()
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[token, slot])
            g = xf[token] @ w1[e].float().t()
            u = xf[token] @ w3[e].float().t()
            situ = beta * torch.tanh(g / beta) * torch.sigmoid(g)
            u = linear_beta * torch.tanh(u / linear_beta)
            out[token] += float(weights[token, slot]) * ((situ * u) @ w2[e].float().t())
    return out


@nvfp4_moe_supported
def test_cutlass_situ_bf16_matches_reference():
    """CUTLASS + SiTU in BF16, against the golden activation. No quantization.

    This is the unambiguous form of the question the NVFP4 GSM8K collapse
    raised. With unquantized weights there is no 4-bit noise to hide behind,
    so a wrong FC1 half assignment or a swapped alpha/beta cannot pass, and
    nothing about the NVFP4 loader is involved.
    """
    from tensorrt_llm.models.modeling_utils import QuantConfig

    num_experts, hidden, inter = _TP_EXPERTS, _TP_HIDDEN, 512
    gate = _make_test_gate(num_experts=num_experts)
    moe = _make_routed_moe(
        inter,
        gate,
        num_experts=num_experts,
        moe_backend="CUTLASS",
        routed_quant_config=QuantConfig(),
    )

    torch.manual_seed(77)
    w1 = [
        torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    w3 = [
        torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    w2 = [
        torch.randn(hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    weights = {}
    for e in range(num_experts):
        weights[f"{e}.w1.weight"] = w1[e]
        weights[f"{e}.w2.weight"] = w2[e]
        weights[f"{e}.w3.weight"] = w3[e]
    moe.backend.load_weights([weights])
    moe.backend._weights_transformed = False
    moe.post_load_weights()

    x = torch.randn(8, hidden, dtype=torch.bfloat16, device="cuda") * 0.5
    router_logits = gate.compute_logits(x)
    actual = moe.forward(x, router_logits, all_rank_num_tokens=None).float()
    expected = _situ_reference_moe(
        x, router_logits, gate.routing_method, w1, w2, w3, beta=4.0, linear_beta=25.0
    )

    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    rel_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    print(f"cutlass-situ-bf16 vs reference: cosine={cosine.item():.6f} rel_l2={rel_l2.item():.6f}")
    assert cosine > 0.999, f"cosine={cosine.item()}, rel_l2={rel_l2.item()}"


def _quantize_expert_to_nvfp4(w1, w2, w3, input_scale):
    """One expert's checkpoint tensors, in nvidia/Kimi-K3-NVFP4's layout.

    w1 and w3 must share a global scale: the loader asserts their
    weight_scale_2 match when it folds them into a single fc31 alpha.
    """
    sv = _NVFP4_GROUP_SIZE
    w13_global = torch.min((448 * 6) / w1.abs().max().float(), (448 * 6) / w3.abs().max().float())
    w2_global = (448 * 6) / w2.abs().max().float()

    def q(w, g):
        packed, sf = torch.ops.trtllm.fp4_quantize(w, g, sv, False)
        # The checkpoint stores plain per-block scales; the swizzle is applied
        # by the backend at load time.
        return packed.cpu(), torch.ops.trtllm.block_scale_interleave_reverse(
            sf.cpu().view(w.shape[0], -1)
        )

    out = {}
    for name, w, g in (("w1", w1, w13_global), ("w3", w3, w13_global), ("w2", w2, w2_global)):
        packed, sf = q(w, g)
        out[f"{name}.weight"] = packed
        out[f"{name}.weight_scale"] = sf
        out[f"{name}.weight_scale_2"] = (1.0 / g).cpu().float()
        out[f"{name}.input_scale"] = torch.tensor(input_scale, dtype=torch.float32)
    return out


@nvfp4_moe_supported
@pytest.mark.parametrize(
    "moe_backend,input_scale",
    [
        pytest.param("CUTLASS", 1.0, id="CUTLASS-static_1.0"),
        pytest.param("TRTLLM", 1.0, id="TRTLLM-static_1.0"),
        # derived_act_scale is a CUTLASS-path finding (see the reason below),
        # so it stays pinned to CUTLASS rather than becoming a strict xfail the
        # TRTLLM-Gen path would have to reproduce.
        pytest.param(
            "CUTLASS",
            None,
            id="CUTLASS-derived_act_scale",
            marks=pytest.mark.xfail(
                strict=True,
                reason="OPEN: the conventional NVFP4 activation global scale is the WORSE "
                "of the two. Measured cosine/rel_l2 vs the golden activation: "
                "static 1.0 -> 0.9727/0.234, derived (448*6)/amax -> 0.8757/0.538. "
                "Correct NVFP4 scaling maps amax onto the FP4xE4M3 range and should "
                "reduce error, not double it, so this points at the activation scale "
                "being applied more than once somewhere on the CUTLASS SiTU path -- "
                "which a checkpoint shipping input_scale=1.0 cannot expose, because "
                "applying 1.0 twice is still 1.0. nvidia/Kimi-K3-NVFP4 ships exactly "
                "that, so this does not affect the current bring-up (GSM8K 96.40 "
                "confirms the shipped configuration). strict=True so that fixing it "
                "reports here instead of passing silently.",
            ),
        ),
    ],
)
def test_nvfp4_experts_match_situ_reference(moe_backend, input_scale):
    """NVFP4 SiTU experts against the golden activation, on both FP4 backends.

    The streamed-vs-whole-checkpoint tests compare two loaders that share all
    the scale handling, so they agree whether or not that handling is right.
    This compares against the definition instead.

    ``moe_backend`` is the discriminating dimension: CUTLASS runs SiTU as an
    ``ActivationType``, TRTLLM-Gen runs the fused ``E2m1_E2m1E2m1 siTuGlu``
    FC1 cubins. The latter is the one that folds ``dequantScaleAb`` into the
    tanh/sigmoid arguments instead of into scaleC; getting that wrong drives
    the FC1 block scales below their smallest subnormal and the MoE returns
    exact zeros, which the ``cosine > 0.95`` bound below rejects (an all-zero
    output scores 0).

    ``static_1.0`` is what nvidia/Kimi-K3-NVFP4 actually ships for
    ``input_scale``; ``derived_act_scale`` is the conventional value computed
    from the activations. Splitting them separates "the loader is wrong" from
    "this checkpoint's static activation scale is being interpreted wrongly".
    """
    num_experts, hidden, inter = _TP_EXPERTS, _TP_HIDDEN, _TP_INTERMEDIATE
    gate = _make_test_gate(num_experts=num_experts)

    torch.manual_seed(91)
    x = torch.randn(8, hidden, dtype=torch.bfloat16, device="cuda") * 0.5
    act_scale = float(x.abs().max().float() / (448 * 6)) if input_scale is None else input_scale

    w1 = [
        torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    w3 = [
        torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    w2 = [
        torch.randn(hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    bank = [_quantize_expert_to_nvfp4(w1[e], w2[e], w3[e], act_scale) for e in range(num_experts)]

    moe = _make_nvfp4_moe(gate, num_experts=num_experts, moe_backend=moe_backend)
    _load_nvfp4_bank_for(moe, bank, moe_backend)

    router_logits = gate.compute_logits(x)
    actual = moe.forward(x, router_logits, all_rank_num_tokens=None).float()
    expected = _situ_reference_moe(
        x, router_logits, gate.routing_method, w1, w2, w3, beta=4.0, linear_beta=25.0
    )

    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    rel_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    print(
        f"nvfp4[{moe_backend}][{input_scale}] vs reference: "
        f"cosine={cosine.item():.6f} rel_l2={rel_l2.item():.6f}"
    )
    # The shipped configuration measures 0.9727 / 0.234. That residual is
    # activation quantization: this path takes the ACTIVATIONS to FP4 as well
    # as the weights, on random Gaussian data whose per-block dynamic range is
    # a worst case real activations do not have. 0.95 leaves room for that
    # while still catching a gross scale error, which lands far below it.
    #
    # An earlier version of this comment justified the bound by claiming both
    # parameterizations landed identically, "the signature of a noise floor".
    # That observation was an artifact of the harness truncating pytest output
    # to the last failure; the two in fact differ a lot (see the xfail above),
    # and the claim was never measured. The structural guards do not rest on
    # this number at all: they are the BF16 comparison against the same golden
    # reference and the SiTU-vs-SwiGLU discriminator, both tolerance-free. The
    # accuracy gate for the real checkpoint is GSM8K.
    assert cosine > 0.95, f"cosine={cosine.item()}, rel_l2={rel_l2.item()}"


def _swiglu_reference_moe(x, router_logits, routing_method, w1, w2, w3, alpha, beta):
    """Same routing/geometry as the SiTU reference, but CUTLASS's SwigluBias.

    ``gate*sigmoid(gate*alpha)*(linear+beta)`` -- what the FC1 epilogue would
    compute if the activation enum did not resolve to SiTu.
    """
    ids, weights = routing_method.apply(router_logits)
    out = torch.zeros_like(x, dtype=torch.float32)
    xf = x.float()
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            e = int(ids[token, slot])
            g = xf[token] @ w1[e].float().t()
            u = xf[token] @ w3[e].float().t()
            h = g * torch.sigmoid(g * alpha) * (u + beta)
            out[token] += float(weights[token, slot]) * (h @ w2[e].float().t())
    return out


@nvfp4_moe_supported
@pytest.mark.parametrize("moe_backend", ["CUTLASS", "TRTLLM"])
def test_nvfp4_kernel_actually_applies_situ(moe_backend):
    """Which activation does the QUANTIZED kernel actually run?

    BF16 + SiTU already matches the golden reference, so the activation, the
    FC1 half assignment and the alpha/beta mapping are all correct in the
    unquantized path. If the NVFP4 path silently resolved to a different
    gated activation -- e.g. because SiTu is not instantiated for the FP4
    epilogue and something falls back -- the output would be structurally
    wrong in exactly the way the GSM8K collapse showed, while every
    shape-and-buffer check stayed green.

    Run for both FP4 backends: CUTLASS resolves SiTU through the activation
    enum, TRTLLM-Gen through a distinct fused-cubin family
    (``Bmm_E2m1_E2m1E2m1_..._siTuGlu_*``). A silent fallback is a different
    failure on each, and this comparison is tolerance-free, so it catches
    both -- including the degenerate all-zero FC1 output, which scores 0
    against both references and so fails the assertion below.

    Reported rather than merely asserted: which reference the kernel is
    closer to is the diagnosis.
    """
    num_experts, hidden, inter = _TP_EXPERTS, _TP_HIDDEN, _TP_INTERMEDIATE
    gate = _make_test_gate(num_experts=num_experts)

    torch.manual_seed(91)
    x = torch.randn(8, hidden, dtype=torch.bfloat16, device="cuda") * 0.5
    act_scale = float(x.abs().max().float() / (448 * 6))
    w1 = [
        torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    w3 = [
        torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    w2 = [
        torch.randn(hidden, inter, dtype=torch.bfloat16, device="cuda") * 0.05
        for _ in range(num_experts)
    ]
    bank = [_quantize_expert_to_nvfp4(w1[e], w2[e], w3[e], act_scale) for e in range(num_experts)]

    moe = _make_nvfp4_moe(gate, num_experts=num_experts, moe_backend=moe_backend)
    _load_nvfp4_bank_for(moe, bank, moe_backend)

    router_logits = gate.compute_logits(x)
    actual = moe.forward(x, router_logits, all_rank_num_tokens=None).float()

    situ = _situ_reference_moe(
        x, router_logits, gate.routing_method, w1, w2, w3, beta=4.0, linear_beta=25.0
    )
    swiglu = _swiglu_reference_moe(
        x, router_logits, gate.routing_method, w1, w2, w3, alpha=4.0, beta=25.0
    )

    def score(ref):
        cos = torch.nn.functional.cosine_similarity(actual.flatten(), ref.flatten(), dim=0)
        l2 = torch.linalg.vector_norm(actual - ref) / torch.linalg.vector_norm(ref)
        return cos.item(), l2.item()

    situ_cos, situ_l2 = score(situ)
    swiglu_cos, swiglu_l2 = score(swiglu)
    print(
        f"NVFP4[{moe_backend}] kernel vs SiTU ref:   "
        f"cosine={situ_cos:.6f} rel_l2={situ_l2:.6f}\n"
        f"NVFP4[{moe_backend}] kernel vs SwiGLU ref: "
        f"cosine={swiglu_cos:.6f} rel_l2={swiglu_l2:.6f}"
    )
    assert situ_cos > swiglu_cos, (
        f"the NVFP4 {moe_backend} kernel matches a SwiGLU reference better than "
        f"the SiTU one (situ={situ_cos:.6f}, swiglu={swiglu_cos:.6f}): the "
        f"quantized path is not applying SiTU"
    )


def test_fp8_block_scaled_dequantization():
    """FP8_PB_WO attention weights must be dequantized, never reinterpreted.

    nvidia/Kimi-K3-NVFP4 stores the attention projections as FP8 E4M3 plus a
    per-128x128-block FP32 scale; moonshotai/Kimi-K3 stores plain BF16 at the
    SAME shape. Nothing downstream can tell them apart, so before this the
    loader's shape check passed and src.to(param.dtype) reinterpreted the
    quantized values as real ones, silently dropping the scale.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import _dequantize_fp8_block_scaled

    n, k = 256, 384
    torch.manual_seed(3)
    quantized = (torch.randn(n, k) * 2).to(torch.float8_e4m3fn)
    scale = torch.rand(n // 128, 1, k // 128, 1) * 3 + 0.5
    weights = {"w.weight": quantized, "w.weight_scale": scale}

    got = _dequantize_fp8_block_scaled("w.weight", quantized, weights)
    assert got.dtype == torch.bfloat16
    expected = quantized.float() * scale.reshape(n // 128, k // 128).repeat_interleave(
        128, 0
    ).repeat_interleave(128, 1)
    torch.testing.assert_close(got.float(), expected.to(torch.bfloat16).float())

    # A BF16 checkpoint (the MXFP4 original) must pass through untouched.
    bf16 = torch.randn(8, 8, dtype=torch.bfloat16)
    assert _dequantize_fp8_block_scaled("w.weight", bf16, {}) is bf16

    # FP8 without a scale is the silent-corruption case; it must raise.
    with pytest.raises(KeyError, match="refusing"):
        _dequantize_fp8_block_scaled("w.weight", quantized, {"w.weight": quantized})


@nvfp4_moe_supported
def test_nvfp4_streaming_drains_staging_per_expert():
    """The staged halves must not accumulate across the load.

    Cutlass defers the cat+pad of w3_w1 to process_weights_after_loading by
    staging both halves per expert, which is a second copy of the
    routed-expert weights. Draining per LAYER does not bound that for Kimi
    K3, whose loader groups work by shard FILE: a layer completes only when
    the last of its slots happens to land, so many layers stage at once.
    Draining per EXPERT does bound it.
    """
    bank = _make_nvfp4_expert_bank(_TP_EXPERTS, _TP_INTERMEDIATE, _TP_HIDDEN)
    gate = _make_test_gate()
    moe = _make_nvfp4_moe(gate)
    backend = moe.backend
    quant_method = backend.quant_method
    quant_method.prepare_streaming_expert_load(backend)

    for expert_id in range(_TP_EXPERTS):
        quant_method.load_streaming_nvfp4_expert(
            backend,
            global_expert_id=expert_id,
            local_slot_id=expert_id,
            **{
                f"{w}_{kind}": bank[expert_id][f"{w}.{kind}"]
                for w in ("w1", "w2", "w3")
                for kind in ("weight", "weight_scale", "weight_scale_2", "input_scale")
            },
        )
        # Nothing may still be staged for a slot that has been loaded.
        assert not backend.tmp_cutlass_w3_w1_weights, (
            f"{len(backend.tmp_cutlass_w3_w1_weights)} staged weight entries "
            f"still held after loading slot {expert_id}"
        )
        assert not backend.tmp_cutlass_w3_w1_weight_scales, (
            f"{len(backend.tmp_cutlass_w3_w1_weight_scales)} staged scale entries "
            f"still held after loading slot {expert_id}"
        )

    backend.process_weights_after_loading()
    # And the result must still equal the whole-checkpoint path.
    reference = _load_nvfp4_bank_whole(_make_nvfp4_moe(gate), bank)
    for name in _NVFP4_LOADED_STATE:
        assert _bitwise_equal(getattr(backend, name).data, getattr(reference, name).data), name


def _checkpoint_scale_2d(scale: "torch.Tensor") -> "torch.Tensor":
    """Mirror the loader's normalization of the checkpoint's 4-D block scale."""
    return scale.reshape(scale.shape[0], scale.shape[2]) if scale.dim() == 4 else scale


@nvfp4_moe_supported
def test_fp8_checkpoint_scale_bridge_matches_the_roundtrip_path():
    """FP8_PB_WO straight from the checkpoint == the BF16 round trip.

    The shipping FP8 weight-read path starts from a BF16 weight and runs
    per_block_cast_to_fp8 -> resmooth_to_fp8_e8m0 -> deep_gemm layout. A
    checkpoint that already stores FP8_PB_WO has done the first step, so
    reading it resident should be the same two remaining steps.

    That equality is the whole premise of keeping attention FP8 on device
    rather than expanding it to BF16, and it is not obvious: fp8_swap_ab_gemm
    runs with disable_ue8m0_cast=True, so it consumes a pre-formatted UE8M0
    scale and would misread the plain FP32 block scale the checkpoint ships.
    This asserts the bridge, on the same values, before any loader is rewired
    to depend on it.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8BlockScaleWeightReadLinear as FP8Linear,
    )
    from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

    out_features, in_features = 256, 512
    torch.manual_seed(4)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda") * 0.05

    # Path A: what production does today, from a BF16 weight.
    w_a, s_a = FP8Linear.quantize_weight(weight)

    # Path B: what a checkpoint hands us -- FP8 + FP32 128x128 block scale --
    # fed straight into the scale bridge.
    ckpt_fp8, ckpt_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
    # nvidia/Kimi-K3-NVFP4 stores the block scale 4-D as
    # [ceil(N/128), 1, ceil(K/128), 1]. Feed that exact shape, not the 2-D one
    # per_block_cast_to_fp8 happens to return: an earlier version of this test
    # used the 2-D form, passed, and the real checkpoint then tripped
    # transform_sf_into_required_layout's rank assert on a four-node run.
    nb_m, nb_k = ckpt_scale.shape
    ckpt_scale_4d = ckpt_scale.float().reshape(nb_m, 1, nb_k, 1)
    w_b, s_b = FP8Linear.prepare_checkpoint_scale(ckpt_fp8, _checkpoint_scale_2d(ckpt_scale_4d))

    assert w_a.shape == w_b.shape and s_a.shape == s_b.shape
    # _bitwise_equal reinterprets only float8 (which torch.equal refuses); the
    # prepared scale is int32 in deep_gemm's packed layout, whose stride makes a
    # uint8 view invalid anyway.
    assert _bitwise_equal(w_a, w_b), "FP8 weights differ"
    assert _bitwise_equal(s_a, s_b), "prepared scales differ"

    # And the GEMM they drive agrees.
    x = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda") * 0.5
    lin_a = FP8Linear(w_a, s_a, out_features)
    lin_b = FP8Linear(w_b, s_b, out_features)
    torch.testing.assert_close(lin_a.forward(x), lin_b.forward(x))


@nvfp4_moe_supported
def test_fp8_weight_read_prefers_the_checkpoint_pair():
    """``from_linear`` must use a stashed FP8_PB_WO pair when one is present.

    Three call sites convert attention projections to the FP8 weight-read
    Linear, all through ``from_linear``, so that is the one place the
    checkpoint-direct route has to be honoured. Without this the loader would
    stash the pair and nothing would read it -- silently falling back to the
    BF16 round trip, which is exactly what this is meant to avoid.
    """
    import torch.nn as nn

    from tensorrt_llm._torch.models.modeling_kimi_linear import _K3_CKPT_FP8_ATTR
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8BlockScaleWeightReadLinear as FP8Linear,
    )
    from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

    out_features, in_features = 256, 512
    torch.manual_seed(6)
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16, device="cuda") * 0.05
    ckpt_fp8, ckpt_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)

    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        # Deliberately NOT the weight the pair came from: if from_linear
        # re-quantized linear.weight instead of using the pair, the result
        # would follow this garbage and the assert below would catch it.
        linear.weight.copy_(torch.zeros_like(weight))
    setattr(linear.weight, _K3_CKPT_FP8_ATTR, (ckpt_fp8, ckpt_scale.float()))

    converted = FP8Linear.from_linear(linear)
    assert not hasattr(linear.weight, _K3_CKPT_FP8_ATTR)
    expected_w, expected_s = FP8Linear.prepare_checkpoint_scale(ckpt_fp8, ckpt_scale.float())
    assert _bitwise_equal(converted.weight, expected_w)
    assert _bitwise_equal(converted.weight_scale, expected_s)
    # float8 supports almost no arithmetic (no abs_cuda), so probe the bytes.
    assert converted.weight.view(torch.uint8).any().item(), (
        "fell back to re-quantizing the zeroed BF16 weight"
    )


@nvfp4_moe_supported
def test_fused_fp8_from_checkpoint_slices_matches_the_bf16_concat():
    """Fusing checkpoint FP8 slices == quantizing the BF16 concatenation.

    The KDA conversion fuses q/k/v/g into one qkvg_proj by concatenating their
    BF16 weights and quantizing the result. Constructing attention as FP8
    requires building that fused weight from the per-projection checkpoint
    FP8 instead, with no BF16 anywhere -- which is only valid because every
    out dim is a multiple of 128, so no 128x128 block straddles a boundary.

    That is the load-bearing assumption behind 3a, so it is checked directly
    rather than inferred from the docstring that states it.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8BlockScaleWeightReadLinear as FP8Linear,
    )
    from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

    in_features = 512
    outs = [256, 256, 128, 384]  # all multiples of 128, mixed sizes like q/k/v/g
    torch.manual_seed(9)
    parts = [torch.randn(o, in_features, dtype=torch.bfloat16, device="cuda") * 0.05 for o in outs]

    # What production does today: concatenate in BF16, then quantize.
    fused_ref = FP8Linear.from_linear(
        type(
            "L",
            (),
            {
                "weight": type("W", (), {"data": torch.cat(parts, dim=0)})(),
                "bias": None,
                "out_features": sum(outs),
            },
        )()
    )

    # 3a's route: each projection arrives already FP8 from the checkpoint.
    pairs = []
    for part in parts:
        w, sc = per_block_cast_to_fp8(part, use_ue8m0=False)
        pairs.append((w, sc.float()))
    fused_new = FP8Linear.fuse_checkpoint_fp8(pairs)

    assert _bitwise_equal(fused_new.weight, fused_ref.weight), "fused FP8 weights differ"
    assert _bitwise_equal(fused_new.weight_scale, fused_ref.weight_scale), "fused scales differ"

    x = torch.randn(8, in_features, dtype=torch.bfloat16, device="cuda") * 0.5
    torch.testing.assert_close(fused_new.forward(x), fused_ref.forward(x))

    # And a non-128 out dim must be rejected, not silently mis-fused.
    bad = torch.randn(96, in_features, dtype=torch.bfloat16, device="cuda")
    w, sc = per_block_cast_to_fp8(bad, use_ue8m0=False)
    with pytest.raises(ValueError, match="multiple of 128"):
        FP8Linear.fuse_checkpoint_fp8([(w, sc.float())])


@nvfp4_moe_supported
def test_fp8_placeholder_fill_contract():
    """The contract construction-time FP8 dispatch has to satisfy.

    A placeholder must cost nothing until filled, must produce exactly what
    the eager constructors produce once filled, must refuse to run while
    empty, and must refuse a second fill. The first property is the whole
    point - DEP8's OOM was during module construction - and the last three
    are what stop that saving from turning into silent garbage, which is how
    every other mistake on this path has presented.
    """
    from tensorrt_llm._torch.models.modeling_kimi_linear import (
        _Fp8BlockScaleWeightReadLinear as FP8Linear,
    )
    from tensorrt_llm.deep_gemm.utils.math import per_block_cast_to_fp8

    in_features, outs = 512, [256, 128]
    torch.manual_seed(11)
    parts = [torch.randn(o, in_features, dtype=torch.bfloat16, device="cuda") * 0.05 for o in outs]
    pairs = []
    for part in parts:
        w, sc = per_block_cast_to_fp8(part, use_ue8m0=False)
        pairs.append((w, sc.float()))

    ph = FP8Linear.empty_placeholder(sum(outs), in_features)
    assert ph.is_placeholder and ph.weight.numel() == 0

    x = torch.randn(4, in_features, dtype=torch.bfloat16, device="cuda") * 0.5
    with pytest.raises(RuntimeError, match="never filled"):
        ph.forward(x)

    ph.load_checkpoint_pair(pairs)
    assert not ph.is_placeholder

    expected = FP8Linear.fuse_checkpoint_fp8(pairs)
    assert _bitwise_equal(ph.weight, expected.weight)
    assert _bitwise_equal(ph.weight_scale, expected.weight_scale)
    torch.testing.assert_close(ph.forward(x), expected.forward(x))

    with pytest.raises(RuntimeError, match="filled twice"):
        ph.load_checkpoint_pair(pairs)

    # The single-projection case takes the same route.
    single = FP8Linear.empty_placeholder(outs[0], in_features)
    single.load_checkpoint_pair([pairs[0]])
    ref = FP8Linear.from_checkpoint_fp8(pairs[0][0], pairs[0][1], outs[0])
    assert _bitwise_equal(single.weight, ref.weight)


def test_megamoe_streamed_coverage_survives_per_expert_drain() -> None:
    """MegaMoE's coverage check must not be defeated by draining the stashes.

    ``finalize_streamed_expert`` drains each slot's staged w3_w1 halves as soon
    as it lands, so the second copy of the routed-expert weights stays bounded
    by the experts in flight. That is what makes EP8 fit: at EP16 the unbounded
    staging survived on 56 rank-local experts, at EP8 it is 112 and the DEP8
    disagg gen worker OOM'd inside setup_engine with the whole card free
    beforehand.

    The drain was previously blocked because ``_streamed_coverage`` decided
    which slots a load had populated by COUNTING staged dict entries -- so
    draining would report 0/N and turn a complete load into a spurious
    "partially covered" error. This asserts the two are now disentangled:
    coverage comes from ``_streamed_expert_slots`` for a streamed load, which
    records the same fact and survives draining.

    Pure bookkeeping, so it needs no GPU and no EP rendezvous.
    """
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    n_slots = 4
    method = NVFP4MegaMoECuteDslMethod.__new__(NVFP4MegaMoECuteDslMethod)

    def _fake_module() -> SimpleNamespace:
        m = SimpleNamespace()
        m.w3_w1_weight = SimpleNamespace(data=torch.zeros(n_slots, 8, 4, dtype=torch.uint8))
        m.w3_w1_weight_scale = SimpleNamespace(data=torch.zeros(n_slots, 8, 4, dtype=torch.uint8))
        m.w2_weight = SimpleNamespace(data=torch.zeros(n_slots, 4, 4, dtype=torch.uint8))
        m.w2_weight_scale = SimpleNamespace(data=torch.zeros(n_slots, 4, 4, dtype=torch.uint8))
        # w2 is written through directly; its coverage is row-pointer based and
        # is unaffected by this change. Mark every row covered.
        m._streamed_w2_covered = {m.w2_weight.data[i].data_ptr() for i in range(n_slots)}
        m._streamed_w2_scale_covered = {
            m.w2_weight_scale.data[i].data_ptr() for i in range(n_slots)
        }
        return m

    # ---- streamed load, stashes fully drained: must report FULL coverage.
    drained = _fake_module()
    drained.tmp_cutlass_w3_w1_weights = {}
    drained.tmp_cutlass_w3_w1_weight_scales = {}
    drained._streamed_expert_slots = set(range(n_slots))
    cov = method._streamed_coverage(drained)
    assert cov == {
        "w3_w1_weight": n_slots,
        "w3_w1_weight_scale": n_slots,
        "w2_weight": n_slots,
        "w2_weight_scale": n_slots,
    }, f"drained streamed load must read as fully covered, got {cov}"

    # ---- a genuinely partial streamed load must still read as partial, so the
    # drain does not turn the check into a rubber stamp.
    partial = _fake_module()
    partial.tmp_cutlass_w3_w1_weights = {}
    partial.tmp_cutlass_w3_w1_weight_scales = {}
    partial._streamed_expert_slots = {0, 1}
    cov = method._streamed_coverage(partial)
    assert cov["w3_w1_weight"] == 2 and cov["w3_w1_weight_scale"] == 2, cov

    # ---- whole-checkpoint load (never streams, never drains): the stash count
    # is still the source of truth, and a half-staged entry still does not count.
    whole = _fake_module()
    whole._streamed_expert_slots = set()
    wbase = whole.w3_w1_weight.data.storage().data_ptr()
    sbase = whole.w3_w1_weight_scale.data.storage().data_ptr()
    whole.tmp_cutlass_w3_w1_weights = {
        (wbase, i): {"w1": 1, "w3": 1, "dst": None} for i in range(n_slots)
    }
    whole.tmp_cutlass_w3_w1_weight_scales = {
        (sbase, i): {"w1": 1, "w3": 1, "dst": None} for i in range(n_slots - 1)
    }
    whole.tmp_cutlass_w3_w1_weight_scales[(sbase, n_slots - 1)] = {"w1": 1, "dst": None}
    cov = method._streamed_coverage(whole)
    assert cov["w3_w1_weight"] == n_slots, cov
    assert cov["w3_w1_weight_scale"] == n_slots - 1, (
        "a stash entry missing its w3 half must not count as covered",
        cov,
    )


def test_megamoe_overrides_finalize_streamed_expert() -> None:
    """MegaMoE must actually drain, and must not interleave when it does.

    Two things a reader could get wrong by copying the Cutlass sibling: it is a
    SIBLING, not a parent, so nothing is inherited; and its scale resolver must
    NOT call block_scale_interleave -- MegaMoE's kernel does its own 16-atom
    gate/up interleave and to_blocked swizzle in _build_mega_format_weights, so
    interleaving here would apply it twice.
    """
    import inspect

    from tensorrt_llm._torch.moe.fused_moe.quantization import (
        NVFP4CutlassFusedMoEMethod,
        NVFP4FusedMoEMethod,
    )
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod as MegaMoE

    assert not issubclass(MegaMoE, NVFP4CutlassFusedMoEMethod), (
        "MegaMoE is a sibling of the Cutlass method, not a child; if this ever "
        "changes, re-check which staging behaviour it inherits"
    )
    assert issubclass(MegaMoE, NVFP4FusedMoEMethod)
    assert "_finalize_staged_w3_w1_expert" in NVFP4FusedMoEMethod.__dict__

    for name in (
        "finalize_streamed_expert",
        "_resolve_staged_w3_w1_weight",
        "_resolve_staged_w3_w1_weight_scale",
    ):
        assert name in MegaMoE.__dict__, f"MegaMoE must define its own {name}"

    for method in (NVFP4CutlassFusedMoEMethod, MegaMoE):
        finalize_src = inspect.getsource(method.finalize_streamed_expert)
        assert "self._finalize_staged_w3_w1_expert" in finalize_src, (
            f"{method.__name__} must delegate staged draining to the shared helper"
        )

    scale_src = inspect.getsource(MegaMoE._resolve_staged_w3_w1_weight_scale)
    assert "_interleave_w3_w1_weight_scale" not in scale_src, (
        "MegaMoE's scale resolver must not interleave; the kernel does that itself"
    )
    # The Cutlass sibling's does, which is what makes the distinction load-bearing.
    assert "_interleave_w3_w1_weight_scale" in inspect.getsource(
        NVFP4CutlassFusedMoEMethod._resolve_staged_w3_w1_weight_scale
    )


@nvfp4_moe_supported
def test_mega_format_transform_is_slot_blockwise() -> None:
    """Chunking the MegaMoE-format transform must be exactly equivalent.

    ``_build_mega_format_weights`` runs the transform in slot chunks so its
    transient stays bounded: the transform materializes several contiguous
    copies of whatever it is handed, and at EP8 a whole layer's routed weights
    made that peak overflow the card (job 489557 asked for 1.15 GiB with ~50
    MiB free). Chunking only helps if slot i's output depends on slot i's input
    and nothing else.

    That property is asserted here BITWISE rather than argued from the code,
    because the transform's 16-atom gate/up interleave is exactly the kind of
    layout operation whose errors are silent -- shapes stay right, values go
    wrong, and only an accuracy run notices.

    Includes an uneven final chunk (a prime slot count against the chunk size),
    which is the case a divisible-only test would miss.
    """
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4MegaMoECuteDslMethod as MegaMoE

    method = MegaMoE.__new__(MegaMoE)

    hidden, intermediate = 256, 64
    expand_intermediate = 2 * intermediate
    num_slots = 7  # deliberately not a multiple of the chunk size below
    h_bytes = hidden // 2

    torch.manual_seed(31)
    raw_w3_w1 = torch.randint(
        0, 255, (num_slots, expand_intermediate, h_bytes), dtype=torch.uint8, device="cuda"
    )
    raw_w2 = torch.randint(
        0, 255, (num_slots, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
    )
    raw_w3_w1_sf = torch.randint(
        0, 255, (num_slots, expand_intermediate, hidden // 16), dtype=torch.uint8, device="cuda"
    )
    raw_w2_sf = torch.randint(
        0, 255, (num_slots, hidden, intermediate // 16), dtype=torch.uint8, device="cuda"
    )

    def _run(lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return method._build_mega_format_buffers(
            raw_w3_w1=raw_w3_w1[lo:hi],
            raw_w3_w1_sf=raw_w3_w1_sf[lo:hi],
            raw_w2=raw_w2[lo:hi],
            raw_w2_sf=raw_w2_sf[lo:hi],
            num_slots=hi - lo,
            intermediate=intermediate,
            hidden=hidden,
            expand_intermediate=expand_intermediate,
        )

    whole = _run(0, num_slots)

    for chunk in (1, 3, 16):  # 16 > num_slots: the single-call degenerate case
        pieces = [_run(lo, min(lo + chunk, num_slots)) for lo in range(0, num_slots, chunk)]
        for i, name in enumerate(("mega_fc1", "mega_fc1_sf", "mega_fc2", "mega_fc2_sf")):
            stitched = torch.cat([p[i] for p in pieces], dim=0)
            assert stitched.shape == whole[i].shape, (
                f"chunk={chunk} {name}: {stitched.shape} vs {whole[i].shape}"
            )
            assert _bitwise_equal(stitched, whole[i]), (
                f"chunk={chunk} {name} differs from the whole-layer transform"
            )
