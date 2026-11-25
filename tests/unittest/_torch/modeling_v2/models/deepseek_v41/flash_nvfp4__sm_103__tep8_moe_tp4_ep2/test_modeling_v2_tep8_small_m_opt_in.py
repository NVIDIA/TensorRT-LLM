# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every DeepSeek-V4.1 target opts its shell's `Linear` modules into the FP8 small-M kernel race as stock's
`DeepseekV4ForCausalLM.__init__` does (`use_fp8_small_m_gemm = True` on every `Linear` once the shell, and with it the
engine's DSpark draft model, is built).

Proved: each target's `__init__` sets the flag on every `Linear` of the built model, the draft model's included; and the
flag is what gives a draft's FP8 block-scale shared expert the FP32 128x128 scales (`weight_scale_block128`) that
stock's decode path needs to run this rank's piece of it (`DeepseekV4MoE._full_tp_shared` ->
`full_tp_shared_expert_forward`): without it the shared expert falls back to the replicated FP8 shard (`fp8_block_scale`
GEMMs and their fused SwiGLU-quantize), the regression the flag's absence caused in the targets' DSpark draft on GB300.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4MoE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils.fp8_utils import resmooth_to_fp8_e8m0

_PKG = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
TARGETS = ("flash_nvfp4__sm_103__tep8_moe_tp4_ep2",)
DEV = "cuda"
# The draft's shared expert as one rank holds it at tep8: the TP2 shard (gcd(2304 / 128, 8) = 2) of the 2304-wide
# intermediate, of which this rank runs one of tp / 2 = 4 pieces.
H, SHARD, PIECES = 5120, 1152, 4


def _modeling(target):
    return importlib.import_module(f"{_PKG}.{target}.modeling")


def _target_class(mod):
    (cls,) = [
        v
        for k, v in vars(mod).items()
        if k.startswith("ModelingV2DeepseekV41") and isinstance(v, type)
    ]
    return cls


@pytest.mark.parametrize("target", TARGETS)
def test_every_linear_opts_in(target, monkeypatch):
    """The target's `__init__` opts in every `Linear` its shell built: the draft model's and the core's alike."""
    mod = _modeling(target)
    cls = _target_class(mod)
    shell = cls.__mro__[1]

    def shell_init(self, model, *args, **kwargs):
        nn.Module.__init__(self)
        self.model = model
        self.draft_model = nn.Sequential(
            Linear(64, 64, bias=False, dtype=torch.bfloat16), nn.ReLU()
        )

    monkeypatch.setattr(shell, "__init__", shell_init)
    for hook in ("__post_init__", "__pp_init__"):  # the shell's quant-config and pipeline passes
        monkeypatch.setattr(cls, hook, lambda self: None, raising=False)
    monkeypatch.setattr(
        mod, "ModelingV2Core", lambda cfg: nn.Sequential(Linear(64, 32, bias=False))
    )
    monkeypatch.setattr(mod, "text_model_config", lambda cfg: cfg)
    cfg = SimpleNamespace(
        pretrained_config=SimpleNamespace(
            hidden_size=64, vocab_size=32, candidate_source_layer_id=2
        ),
        spec_config=None,
    )
    model = cls(cfg)
    linears = [m for m in model.modules() if isinstance(m, Linear)]
    assert len(linears) == 2
    assert all(getattr(m, "use_fp8_small_m_gemm", False) for m in linears)


def _fp8_linear(n, k, gen, opted):
    lin = Linear(
        k,
        n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    )
    lin = lin.to(DEV)
    lin.weight.data.copy_(
        (torch.randn(n, k, generator=gen, device=DEV) * 2).to(torch.float8_e4m3fn)
    )
    # power-of-two block scales, as the checkpoint's 128x128 requantization writes them
    exps = torch.randint(-8, -4, (n // 128, k // 128), generator=gen, device=DEV)
    lin.weight_scale.data.copy_(torch.pow(2.0, exps.float()))
    if opted:
        lin.use_fp8_small_m_gemm = True
    return lin


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="sm_103 (GB300)",
)
@pytest.mark.parametrize("target", TARGETS)
def test_draft_shared_expert_takes_stocks_piece_path(target):
    """With the target's opt-in, a draft's FP8 shared expert keeps `weight_scale_block128` (its UE8M0-resmoothed block
    scales, bit for bit) after its weight transform, and stock's decode path then serves a decode-sized step from this
    rank's piece; without it there are no FP32 block scales and stock runs the replicated shard."""
    opt_in = importlib.import_module(f"{_PKG}.common").opt_into_small_m_gemm
    x = torch.randn(6, H, device=DEV, dtype=torch.bfloat16)
    for opted in (True, False):
        gen = torch.Generator(device=DEV).manual_seed(7)
        gate_up, down = _fp8_linear(2 * SHARD, H, gen, False), _fp8_linear(H, SHARD, gen, False)
        mlp = nn.ModuleDict(dict(gate_up_proj=gate_up, down_proj=down))
        if opted:
            opt_in(mlp)
        scales = [
            resmooth_to_fp8_e8m0(m.weight.detach().clone(), m.weight_scale.detach().clone())[1]
            for m in (gate_up, down)
        ]
        gate_up.post_load_weights()
        down.post_load_weights()
        blocks = [getattr(m, "weight_scale_block128", None) for m in (gate_up, down)]
        moe = SimpleNamespace(
            _full_tp_shared_piece=(1, PIECES), _full_tp_shared_weights=None, shared_experts=mlp
        )
        piece = DeepseekV4MoE._full_tp_shared(moe, x)
        if opted:
            for block, scale in zip(blocks, scales):
                assert block is not None and torch.equal(block, scale)
            w13, s13, w2, s2 = piece
            assert w13.shape == (2 * 320, H) and w2.shape == (
                H,
                320,
            )  # the 288-wide piece, padded to the K step
        else:
            assert blocks == [None, None] and piece is None
