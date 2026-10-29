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
"""Dense DWDP on real Kimi-K3 modules: outputs are bit-exact after the weights move.

End to end on one GPU.  The model is a ``KimiLinearForCausalLM`` with the text shapes of
the Kimi-K3 NVFP4 checkpoint (hidden 7168, KDA 96 x 128 heads, MLA q_lora 1536 / kv_lora 512 /
rope 64, latent 3584, shared expert 2 x 3072, 896 experts, fp32 router under attention DP) and the
checkpoint's first nine layers: layer 0 (KDA, dense MLP, never covered), layers 1, 2, 4, 5, 6 and
8 (KDA MoE) and layers 3 and 7 (MLA MoE).  Only the vocabulary, layer 0's dense-MLP width and the
RoPE table length are cut down; none of them is under a dense root.

Weights are random, in the layout of the checkpoint (FP8 codes with fp32 128x128 block scales for
the attention projections, BF16 elsewhere), and go through the real load path: ``load_weights``
with ``_finalize_weight_load`` (KDA QKVG fusion and decode constants, the FP8 weight-read swap,
the ``KIMI_K3_FP8_WEIGHT_READ_MOE_MLP`` conversion of the shared and latent MLPs) and then the
model loader's post-load walk, so every moved tensor has its deployed dtype, layout and aliasing.
The routed experts belong to expert DWDP and are never a dense root: a stand-in without tensors
replaces them (created through ``create_moe``, as ``test_kimi_k3_mlp.py`` does), which keeps the
rest of the MoE block real, its forward included, and their checkpoint slices are not loaded.

Dense DWDP is set up by the real ``setup_dense_dwdp`` (P1-P6 with Phase V and the P5 self-check)
for one rank of a 16-rank group held in this process: the group's collectives hand every rank
this rank's report, and the loopback store holds all 16 rows.  The deployed configuration moves
both segments over the TMA SM gather; the module test also runs the attention-only canary, the
copy engine's batched copies (``contention_opt``) over a store in several chunks, and per-slice
copies.
``strict`` turns a demand fetch, a fence fallback, a leaked original, an unknown large attribute
or a shared storage into an error; ``debug=1`` fills the slots with NaN at commit, and the
decoder test also runs ``debug=2``, whose slot watchdog checks every slot at each H1 against the
layer's P1 checksum at Kimi-K3 blob sizes.

``test_k3_modules_are_bit_exact_after_dense_dwdp`` runs every module that reads a moved weight on
fixed inputs before setup and again afterwards inside its read window of a forward driven through
the runtime hooks (priming, then ``wait_layer`` / ``attention_done`` / ``layer_done`` per covered
layer, with the copies of later layers in flight), once per token count.  Every output must equal
its pre-setup output bit for bit.  The first attention slot holds layers 1, 3, 5 and 7 in turn
and the second layers 2, 4, 6 and 8; the MoE-dense slots hold layers 1, 4 and 7, 2, 5 and 8, and 3
and 6.  So each layer reads its weights through the same addresses as the layers sharing its
slots: KDA and MLA layers alike, and the two MLA layers 3 and 7 read every moved MLA weight at the
same address with the same shape, as Kimi-K3's MLA layers 3, 7, ..., 91 do.  Kimi-K3's last MLA
layer, 92, shares the second attention slot with KDA layers, which no MLA layer built here does.
Read outside its window, a layer sees another layer's weights.  Covered readers:

* KDA: ``_project_packed_conv_input`` (fused QKVG MXFP8 GEMM), the gate inputs as prefill
  projects them (BF16 ``[f_a | b]`` and ``f_b``), ``_output_gate_and_proj`` (the output gate,
  ``o_norm`` and ``o_proj``) and the separate q/k/v/g projections, which read row views of the
  fused weight with their own scales;
* MLA: ``kv_a_proj_with_mqa``, ``q_b_proj``, ``kv_b_proj``, ``g_proj``, ``o_proj``, the output
  gate with ``o_proj`` (``_apply_output_gate_and_o_proj``), both absorbed BMMs (the deployment's
  BF16 ``*_dequant`` copies) and the FP8-output GEMM that the FP4 context path
  (``forward_context_with_chunked_prefill_fp4``) runs on ``kv_b_proj``'s weight and scale;
* MoE-dense: the fp32 router logits, the latent down and up projections, the shared expert and
  the whole MoE block.

``test_k3_decoder_forward_is_bit_exact_through_the_dense_hooks`` runs ``KimiLinearModel.forward``
itself, with H0-H3 at their call sites, real KDA prefill and MLA context attention over fresh
context requests, and checks every layer's output and the final hidden state, also after a
forward that aborted midway and on an execution stream the executor loop primed for.  Both tests
delay the copies of one forward and the read windows of another with spin kernels, so that a
read or a copy that did not wait for its event would see another layer's bytes.  The KV cache is
BF16 (``helix_test_utils`` builds a v1 ``KVCacheManager``), so the MLA layers run the BF16
context path.  The deployment gives every MLA layer but 91 and 92 an NVFP4 KV cache, whose context
path for requests with a cached prefix reads ``kv_b_proj`` through the FP8-output GEMM; the module
test covers that read, but no FP4 MLA attention kernel runs here.  Those kernels read the KV cache
and no weight, and they too run inside the attention window: the FP4 cache update runs on a side
stream that forks after H1 and joins before the attention returns.

Tactics are fixed.  ``TLLM_AUTOTUNER_CACHE_PATH`` (a scratch copy of the file is loaded, never
written) supplies the deployment's MXFP8 GEMM tactics, keyed with the gather's ``clc_dynamic``
scheduler, and every MXFP8 GEMM with BF16 output must then hit it: the autotuner warmups run only
fresh context requests, so they never tune the FP8-output GEMM of the FP4 context path.  Without
the cache every GEMM runs its fallback tactic.  Nothing is tuned, and each call's tactic is
recorded and must be the same before and after setup.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import json
import operator
import os
import shutil
import time
import weakref
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from torch import nn

pytest.importorskip("fla")

from _torch.modules.helix_test_utils import CACHE_TYPE_SELFKONLY, setup_kv_and_metadata
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata, KVCacheParams
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.autotuner import AutoTuner, AutoTunerProfilingCache
from tensorrt_llm._torch.configs.kimi_linear import KimiLinearConfig
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models import modeling_kimi_linear as mkl
from tensorrt_llm._torch.modules.dwdp import fabric_gather
from tensorrt_llm._torch.modules.dwdp.dense import setup as dense_setup
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.plan import DenseCopyLauncher, select_copy_mode
from tensorrt_llm._torch.modules.dwdp.dense.runtime import DenseDWDP
from tensorrt_llm._torch.modules.dwdp.dense.store import LoopbackShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import CopyMode, DenseLayout, SegmentKind
from tensorrt_llm._torch.modules.fused_moe import ConfigurableMoE, CuteDslFusedMoE
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KDA_PREFILL_CHUNK_SIZE
from tensorrt_llm._torch.modules.mla import fp8_block_scaling_bmm_out
from tensorrt_llm._torch.pyexecutor.dwdp import DwdpManager
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.utils import model_extra_attrs
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

ATTN = SegmentKind.ATTN
MOE = SegmentKind.MOE

# ---------------------------------------------------------------------------------------------
# The Kimi-K3 configuration
# ---------------------------------------------------------------------------------------------

#: ``text_config`` of the Kimi-K3 NVFP4 checkpoint, except the fields this test cuts down
#: (``_CUT_DOWN``) and the per-layer lists (``_KDA_LAYERS``, ``_MLA_LAYERS``).
_K3_TEXT_CONFIG = {
    "hidden_size": 7168,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "rms_norm_eps": 1e-5,
    "moe_intermediate_size": 3072,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "num_experts": 896,
    "num_experts_per_token": 16,
    "num_shared_experts": 2,
    "routed_scaling_factor": 1.0,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "use_grouped_topk": True,
    "num_expert_group": 1,
    "topk_group": 1,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "attn_res_block_size": 12,
    "latent_moe_use_norm": True,
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "routed_expert_hidden_size": 3584,
    "topk_method": "noaux_tc",
    "hidden_act": "situ",
    "tie_word_embeddings": False,
}

#: ``linear_attn_config`` of the checkpoint without its layer lists.
_K3_LINEAR_ATTN = {
    "num_heads": 96,
    "head_dim": 128,
    "short_conv_kernel_size": 4,
    "use_full_rank_gate": True,
    "gate_lower_bound": -5.0,
}

#: Decoder layers built: the checkpoint's first nine.
_NUM_LAYERS = 9

#: The checkpoint's 1-indexed ``kda_layers`` and ``full_attn_layers`` among the layers built:
#: layers 0, 1, 2, 4, 5, 6 and 8 are KDA, layers 3 and 7 are MLA (both at even positions among
#: the covered layers, so they share the first attention slot).
_KDA_LAYERS = (1, 2, 3, 5, 6, 7, 9)
_MLA_LAYERS = (4, 8)

#: Fields cut down from the checkpoint; nothing under a dense root depends on them.
_CUT_DOWN = {
    "vocab_size": 256,
    # Layer 0's dense MLP only; the MoE layers use moe_intermediate_size.
    "intermediate_size": 512,
    # RoPE table length of the MLA backends (K3 is NoPE; the table holds identity rotations).
    "max_position_embeddings": 4096,
}

#: The layers dense DWDP covers: every MoE layer.
_COVERED = tuple(range(1, _NUM_LAYERS))

_KDA_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
_MLA_PROJECTIONS = ("q_a_proj", "kv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "g_proj", "o_proj")

#: Per-segment content of one layer as the collector counts it: ``(moved bytes, storages)``.
_KDA_ATTN = (471_531_520, 10)
_MLA_ATTN = (275_447_808, 11)
#: The MoE segment's router weight is BF16 (with ``KIMI_K3_ROUTER_BF16`` unset the router GEMM is
#: BF16 under attention DP too); the rest of the segment is 187_629_568 bytes.
_ROUTER_WEIGHT_BYTES = _K3_TEXT_CONFIG["num_experts"] * _K3_TEXT_CONFIG["hidden_size"] * 2
_MOE_DENSE = (187_629_568 + _ROUTER_WEIGHT_BYTES, 7)

#: Token counts of the forwards, in order (each runs every covered layer): a short and a long
#: context, then one token, the size of the generation request of the attention warmup.
_TOKEN_COUNTS = (16, 3000, 1)

#: The long forward's token count.
_LONG = _TOKEN_COUNTS[1]

_NUM_RANKS = 16

#: Environment of the deployed context workers (profile ``rubin-dwdp-ra-dense``) that the
#: modules and the gather read; the scheduler and SM reserve are what the SM gather sets on its
#: compute side (``fabric_gather.apply_compute_side_defaults``).
_DEPLOYMENT_ENV = {
    "KIMI_K3_FP8_WEIGHT_READ_MOE_MLP": "1",
    "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER": "clc_dynamic",
    "TRTLLM_CUTEDSL_SM_RESERVE": str(fabric_gather.TARGET_BLOCKS),
    "TRTLLM_DWDP_GATHER_KERNEL": "tma",
}

#: Variables that would change what this test builds: the RoPE table length, the router dtype.
_UNSET_ENV = ("KIMI_K3_MLA_MAX_POSITIONS", "KIMI_K3_ROUTER_BF16")

_MXFP8_GEMM = "trtllm::cute_dsl_mxfp8_gemm_rubin"

_INT_VIEWS = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _k3_config() -> KimiLinearConfig:
    linear_attn = dict(
        _K3_LINEAR_ATTN, kda_layers=list(_KDA_LAYERS), full_attn_layers=list(_MLA_LAYERS)
    )
    return KimiLinearConfig(
        num_hidden_layers=_NUM_LAYERS,
        linear_attn_config=linear_attn,
        dtype=torch.bfloat16,
        **_CUT_DOWN,
        **_K3_TEXT_CONFIG,
    )


def _model_config(cfg: KimiLinearConfig) -> ModelConfig:
    """The deployed context worker: tp1 with attention DP, FP8 block-scaled attention."""
    fp8 = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES, group_size=128)
    declared = {}
    for layer in range(_NUM_LAYERS):
        names = _KDA_PROJECTIONS if layer + 1 in _KDA_LAYERS else _MLA_PROJECTIONS
        declared.update({f"model.layers.{layer}.self_attn.{name}": fp8 for name in names})
    return ModelConfig(
        pretrained_config=cfg,
        mapping=Mapping(world_size=1, tp_size=1, rank=0, enable_attention_dp=True),
        quant_config=QuantConfig(),
        quant_config_dict=declared,
        moe_backend="CUTEDSL",
    )


# ---------------------------------------------------------------------------------------------
# Model, weights and loading
# ---------------------------------------------------------------------------------------------


class _RoutedExpertsStandIn(ConfigurableMoE):
    """Stands in for the routed experts of one MoE layer, which expert DWDP owns.

    Holds no tensors, so it adds nothing to the dense roots or to the checkpoint plan's
    parameters.  Its forward scales the latent input by the mean router probability of each
    token, so the MoE block's output depends on the router weight as well as on the latent
    projections and the shared expert.  Setting ``fail`` makes the forward raise, which aborts a
    model forward inside the layer's MoE block.
    """

    def __init__(self, **kwargs) -> None:
        nn.Module.__init__(self)
        self.fail = False
        # KimiK3MoERuntime requires exactly the CuteDslFusedMoE backend a CUTEDSL request builds.
        # This one skips its constructor, so it is kept out of the module tree: the walks over it
        # (parameters, the loader's post-load hooks) would call into the missing state.
        backend = object.__new__(CuteDslFusedMoE)
        backend.initial_local_expert_ids = [0]
        object.__setattr__(self, "backend", backend)
        self.comm = None
        self.layer_load_balancer = None
        self.all_reduce = None
        self.use_dp = True
        self.parallel_size = 1
        self.moe_backend = kwargs["model_config"].moe_backend
        # ``KimiK3MoERuntime`` reads the routing method off its routed experts (routing tail). It
        # is an ``nn.Module``; kept out of the module tree like the backend.
        object.__setattr__(self, "routing_method", kwargs["routing_method"])

    def create_weights(self) -> None:
        pass

    def load_weights(self, weights, allow_partial_loading: bool = False) -> None:
        pass

    def transform_weights(self) -> None:
        pass

    def cache_derived_state(self) -> None:
        pass

    def post_load_weights(self) -> None:
        pass

    def process_weights_after_loading(self) -> None:
        pass

    def forward(
        self, x: torch.Tensor, router_logits: torch.Tensor, *, all_rank_num_tokens=None, **kwargs
    ) -> torch.Tensor:
        if self.fail:
            raise RuntimeError("injected routed-expert failure")
        probabilities = torch.sigmoid(router_logits).mean(dim=-1, keepdim=True)
        return x * probabilities.to(x.dtype)


def _build_model(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> mkl.KimiLinearForCausalLM:
    """The nine-layer K3 model with random weights loaded and post-processed as deployed."""
    monkeypatch.setattr(mkl, "create_moe", lambda **kwargs: _RoutedExpertsStandIn(**kwargs))
    expert_jobs = []
    # The routed experts are the stand-in's; their checkpoint slices are not loaded.
    monkeypatch.setattr(
        mkl.KimiLinearForCausalLM,
        "_load_expert_slices",
        lambda self, weights, jobs: expert_jobs.extend(jobs),
    )
    model_config = _model_config(_k3_config())
    with torch.device(device):
        model = mkl.KimiLinearForCausalLM(model_config).eval()
    generator = torch.Generator(device=device).manual_seed(20261016)
    weights = _checkpoint(model, generator, device)
    model.load_weights(weights)
    del weights
    ModelLoader._walk_full_post_load(model)
    torch.cuda.synchronize(device)
    assert [layer for layer, _, _ in expert_jobs] == list(_COVERED)
    return model


def _checkpoint(
    model: mkl.KimiLinearForCausalLM, generator: torch.Generator, device: torch.device
) -> dict[str, torch.Tensor]:
    """Random tensors for every key ``load_weights`` expects, in the checkpoint's layout.

    Trunk parameters are BF16 (the router's too: the checkpoint stores it in BF16 and the loader
    widens it); the FP8 attention projections are E4M3 codes in [-64, 64] with fp32 scales in
    ModelOpt's ``[N/128, 1, K/128, 1]`` layout; the routed-expert keys get placeholders.
    """
    name_map, expected_keys, _ = model.checkpoint_name_plan("")
    params = dict(model.named_parameters())
    weights = {}
    for name, key in name_map.items():
        param = params[name]
        if name.endswith(mkl._GATE_UP_FUSED_SUFFIX):
            rows = param.shape[0] // 2
            for half in mkl._gate_up_ckpt_keys(key):
                weights[half] = _normal(generator, (rows, param.shape[1]), 0.02, device)
        elif name.endswith("norm.weight"):
            weights[key] = 1 + _normal(generator, tuple(param.shape), 0.05, device)
        else:
            weights[key] = _normal(generator, tuple(param.shape), 0.02, device)
    for key in sorted(expected_keys - weights.keys()):
        if ".block_sparse_moe.experts." in key:
            weights[key] = torch.zeros(1, dtype=torch.uint8)
        elif key.endswith(".weight"):
            weights[key], weights[mkl._fp8_block_scale_key(key)] = _fp8_pair(
                generator, _fp8_checkpoint_shape(key), device
            )
    missing = sorted(expected_keys - weights.keys())
    assert not missing, f"no synthetic tensor for {missing[:8]}"
    return weights


def _fp8_checkpoint_shape(key: str) -> tuple[int, int]:
    """``[out, in]`` of an FP8 attention projection as the checkpoint stores it."""
    layer = int(key.split(".")[2])
    projection = key.split(".")[-2]
    cfg = SimpleNamespace(**_K3_TEXT_CONFIG)
    hidden = cfg.hidden_size
    if layer + 1 in _KDA_LAYERS:
        width = _K3_LINEAR_ATTN["num_heads"] * _K3_LINEAR_ATTN["head_dim"]
        return (hidden, width) if projection == "o_proj" else (width, hidden)
    heads = cfg.num_attention_heads
    return {
        "q_a_proj": (cfg.q_lora_rank, hidden),
        "kv_a_proj_with_mqa": (cfg.kv_lora_rank + cfg.qk_rope_head_dim, hidden),
        "q_b_proj": (heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), cfg.q_lora_rank),
        "kv_b_proj": (heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
        "g_proj": (heads * cfg.v_head_dim, hidden),
        "o_proj": (hidden, heads * cfg.v_head_dim),
    }[projection]


def _normal(
    generator: torch.Generator, shape: tuple[int, ...], std: float, device: torch.device
) -> torch.Tensor:
    values = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    return (values * std).to(torch.bfloat16)


def _fp8_pair(
    generator: torch.Generator, shape: tuple[int, int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    codes = torch.randint(-64, 65, shape, generator=generator, device=device)
    grid = tuple(-(-dim // 128) for dim in shape)
    scale = (1 + torch.rand(grid, generator=generator, device=device)) * 1e-3
    return codes.to(torch.float8_e4m3fn), scale[:, None, :, None]


# ---------------------------------------------------------------------------------------------
# The modules under test
# ---------------------------------------------------------------------------------------------


class _Inputs(NamedTuple):
    """Fixed activations of one token count ``T`` (BF16)."""

    hidden: torch.Tensor
    """``[T, hidden]``: decoder-layer activations (KDA, MLA and MoE inputs)."""

    kda_core: torch.Tensor
    """``[T, heads, head_dim]``: the KDA core that ``o_proj`` projects."""

    q_latent: torch.Tensor
    """``[T, q_lora_rank]``: the MLA ``q_b_proj`` input."""

    kv_latent: torch.Tensor
    """``[T, kv_lora_rank]``: the MLA ``kv_b_proj`` input."""

    mla_heads: torch.Tensor
    """``[T, heads * v_head_dim]``: the MLA attention output (``o_proj`` and gate input)."""

    q_nope: torch.Tensor
    """``[T, heads, qk_nope_head_dim]``: the absorbed-K BMM input."""

    attn_latent: torch.Tensor
    """``[T, heads, kv_lora_rank]``: the absorbed-V BMM input."""


def _make_inputs(tokens: int, device: torch.device) -> _Inputs:
    generator = torch.Generator(device=device).manual_seed(tokens)
    cfg = SimpleNamespace(**_K3_TEXT_CONFIG)
    heads = cfg.num_attention_heads

    def normal(*shape: int) -> torch.Tensor:
        return _normal(generator, (tokens, *shape), 1.0, device)

    return _Inputs(
        hidden=normal(cfg.hidden_size),
        kda_core=normal(_K3_LINEAR_ATTN["num_heads"], _K3_LINEAR_ATTN["head_dim"]),
        q_latent=normal(cfg.q_lora_rank),
        kv_latent=normal(cfg.kv_lora_rank),
        mla_heads=normal(heads * cfg.v_head_dim),
        q_nope=normal(heads, cfg.qk_nope_head_dim),
        attn_latent=normal(heads, cfg.kv_lora_rank),
    )


def _attention_outputs(layer: mkl.KimiLinearDecoderLayer, inputs: _Inputs) -> dict:
    """Every reader of the layer's attention weights (segment A) on ``inputs``."""
    if layer.is_kda:
        return _kda_outputs(layer.linear_attn, inputs)
    return _mla_outputs(layer.self_attn.mixer, inputs)


def _kda_outputs(attention: nn.Module, inputs: _Inputs) -> dict:
    x2d = inputs.hidden
    x = x2d.unsqueeze(0)
    packed_conv, output_gate = attention._project_packed_conv_input(x, x2d)
    assert output_gate is not None, "K3's full-rank output gate rides in the fused QKVG GEMM"
    # The gate inputs as forward_prefill projects them: beta and f_a from the fused [f_a | b]
    # GEMM, then f_b.
    head_dim, heads = attention.head_dim, attention.num_heads
    bfa = nn.functional.linear(x, attention._bfa_proj_weight)
    outputs = {
        "linear_attn.qkvg.packed_conv": packed_conv,
        "linear_attn.qkvg.output_gate": output_gate,
        "linear_attn.gate_inputs.beta": bfa[..., head_dim : head_dim + heads],
        "linear_attn.gate_inputs.forget_gate": attention.f_b_proj(bfa[..., :head_dim]),
        "linear_attn.output_gate_and_proj": attention._output_gate_and_proj(
            x, inputs.kda_core, output_gate
        ),
    }
    for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
        outputs[f"linear_attn.{name}"] = getattr(attention, name)(x2d)
    return outputs


def _mla_outputs(mixer: nn.Module, inputs: _Inputs) -> dict:
    tokens = inputs.hidden.shape[0]
    absorbed_k = inputs.hidden.new_empty((mixer.num_heads_tp, tokens, mixer.kv_lora_rank))
    fp8_block_scaling_bmm_out(
        inputs.q_nope,
        mixer.k_b_proj_trans,
        mixer.k_b_proj_trans_scale,
        absorbed_k,
        mixer.k_b_proj_trans_dequant,
        mixer.use_cute_dsl_blockscaling_bmm,
    )
    absorbed_v = inputs.hidden.new_empty((mixer.num_heads_tp_cp, tokens, mixer.v_head_dim))
    fp8_block_scaling_bmm_out(
        inputs.attn_latent,
        mixer.v_b_proj,
        mixer.v_b_proj_scale,
        absorbed_v,
        mixer.v_b_proj_dequant,
        mixer.use_cute_dsl_blockscaling_bmm,
    )
    return {
        "self_attn.kv_a_proj_with_mqa": mixer.kv_a_proj_with_mqa(inputs.hidden),
        "self_attn.q_b_proj": mixer.q_b_proj(inputs.q_latent),
        "self_attn.kv_b_proj": mixer.kv_b_proj(inputs.kv_latent),
        "self_attn.kv_b_proj_fp8_out": _kv_b_proj_fp8_output(mixer, inputs.kv_latent),
        "self_attn.g_proj": mixer.g_proj(inputs.hidden),
        "self_attn.output_gate_and_o_proj": mixer._apply_output_gate_and_o_proj(
            inputs.hidden, inputs.mla_heads
        ),
        "self_attn.o_proj": mixer.o_proj(inputs.mla_heads),
        "self_attn.absorbed_k": absorbed_k,
        "self_attn.absorbed_v": absorbed_v,
    }


def _kv_b_proj_fp8_output(mixer: nn.Module, kv_latent: torch.Tensor) -> torch.Tensor:
    """``kv_b_proj`` as the FP4 context path reads it: the FP8-output GEMM on its weight and scale.

    ``forward_context_with_chunked_prefill_fp4`` (NVFP4 KV cache, requests with a cached prefix)
    runs this GEMM on the module's tensors instead of calling the module; the call is the one
    ``test_k3_ctx_fp8_output.py`` checks against the module.
    """
    activation, activation_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(kv_latent)
    return torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
        activation,
        mixer.kv_b_proj.weight,
        activation_scale,
        mixer.kv_b_proj.weight_scale,
        output_dtype=torch.float8_e4m3fn,
    )


def _moe_dense_outputs(layer: mkl.KimiLinearDecoderLayer, inputs: _Inputs) -> dict:
    """Every reader of the layer's MoE-dense weights (segment D) on ``inputs``."""
    moe = layer.block_sparse_moe
    hidden = inputs.hidden
    latent = moe.routed_expert_down_proj(hidden)
    return {
        "moe.gate.logits": moe.gate.compute_logits(hidden),
        "moe.routed_expert_down_proj": latent,
        "moe.routed_expert_up_proj": moe.routed_expert_up_proj(moe.routed_expert_norm(latent)),
        "moe.shared_experts": moe.shared_experts(hidden),
        "moe.block": moe(hidden),
    }


def _layer_outputs(
    layer_idx: int, layer: mkl.KimiLinearDecoderLayer, inputs: _Inputs, segment: SegmentKind
) -> dict[str, torch.Tensor]:
    """The readers of one segment of one layer, keyed ``layers.<index>.<reader>``."""
    readers = _attention_outputs if segment is ATTN else _moe_dense_outputs
    outputs = readers(layer, inputs)
    return {f"layers.{layer_idx}.{name}": value.clone() for name, value in outputs.items()}


def _reference_outputs(layers: nn.ModuleList, inputs: _Inputs) -> dict[str, torch.Tensor]:
    """The outputs of every covered layer's readers, in the order a forward runs them."""
    outputs = {}
    with torch.no_grad():
        for layer_idx in _COVERED:
            for segment in (ATTN, MOE):
                outputs.update(_layer_outputs(layer_idx, layers[layer_idx], inputs, segment))
    return outputs


def _hooked_forward(
    runtime: DenseDWDP,
    layers: nn.ModuleList,
    inputs: _Inputs,
    *,
    prime: bool = True,
    spin_cycles: int = 0,
) -> dict[str, torch.Tensor]:
    """One forward over the covered layers through the dense hooks, as the model drives them.

    Priming enqueues the first two layers as ``DwdpManager`` does; every layer then waits for
    its slots (H1), runs its attention readers, releases the attention slot and enqueues the
    layer two ahead (H2), runs its MoE-dense readers and releases its MoE-dense slot (H3).
    Nothing synchronizes the host: the copies of later layers run while earlier layers compute.

    Args:
        runtime: The committed runtime.
        layers: The decoder layers.
        inputs: The activations of this forward.
        prime: Prime the first two layers; without it their H1 runs a fenced demand fetch.
        spin_cycles: Spin the compute stream this many clock cycles at the start of every read
            window, so the copies enqueued meanwhile run while the window's reads are pending.
    """
    if prime:
        for layer_idx in runtime.layers[:2]:
            runtime.prefetch(layer_idx)
        runtime.mark_primed()
    outputs = {}
    with torch.no_grad():
        for layer_idx in runtime.layers:
            layer = layers[layer_idx]
            runtime.wait_layer(layer_idx)
            _spin(spin_cycles)
            outputs.update(_layer_outputs(layer_idx, layer, inputs, ATTN))
            runtime.attention_done(layer_idx)
            _spin(spin_cycles)
            outputs.update(_layer_outputs(layer_idx, layer, inputs, MOE))
            runtime.layer_done(layer_idx)
    return outputs


def _spin(cycles: int) -> None:
    """Keep the current stream busy for ``cycles`` GPU clock cycles."""
    if cycles:
        torch.cuda._sleep(cycles)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's bytes as integers, so NaN payloads and signed zeros compare too."""
    return tensor.contiguous().view(_INT_VIEWS[tensor.element_size()])


def _differences(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> list[str]:
    if expected.keys() != actual.keys():
        return [f"outputs {sorted(expected.keys() ^ actual.keys())} are missing on one side"]
    differences = []
    for name, reference in expected.items():
        value = actual[name]
        if value.dtype != reference.dtype or value.shape != reference.shape:
            differences.append(
                f"{name}: {value.dtype} {tuple(value.shape)} instead of {reference.dtype} "
                f"{tuple(reference.shape)}"
            )
        elif not torch.equal(_bits(value), _bits(reference)):
            changed = (_bits(value) != _bits(reference)).sum().item()
            differences.append(f"{name}: {changed} of {reference.numel()} elements differ")
    return differences


# ---------------------------------------------------------------------------------------------
# Dense DWDP plumbing
# ---------------------------------------------------------------------------------------------


class _LoopbackGroup:
    """A DWDP group of ``size`` ranks inside this process.

    With the loopback store every rank's rows come from this process's weights, so this rank's
    report in each collective stands for every rank's.
    """

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size
        self.allgathers = 0

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._size

    def allgather(self, obj: object) -> list[object]:
        self.allgathers += 1
        return [obj] * self._size


class _Variant(NamedTuple):
    """One dense DWDP configuration of this rank of the group."""

    segments: tuple[SegmentKind, ...]
    """``TRTLLM_DWDP_DENSE_SEGMENTS``."""

    rank: int
    """This rank in the 16-rank group."""

    transfer: CopyMode = CopyMode.GATHER
    """The expert DWDP transfer mode, which dense DWDP follows."""

    store_chunk_mb: int = 0
    """``TRTLLM_DWDP_DENSE_STORE_CHUNK_MB``."""

    strict: bool = True
    """``TRTLLM_DWDP_DENSE_STRICT``; a lenient run starts with a forward that is not primed."""

    debug: int = 1
    """``TRTLLM_DWDP_DENSE_DEBUG``: 1 fills the slots with NaN at commit; 2 also runs the slot
    watchdog."""

    @property
    def name(self) -> str:
        chunks = f"-chunk{self.store_chunk_mb}mb" if self.store_chunk_mb else ""
        lenient = "" if self.strict else "-lenient"
        debug = "" if self.debug == 1 else f"-debug{self.debug}"
        segments = "_".join(kind.value for kind in self.segments)
        return f"{segments}-{self.transfer.value}-rank{self.rank}{chunks}{lenient}{debug}"


#: The deployment (profile ``rubin-dwdp-ra-dense``): both segments over the SM gather.
_DEPLOYED = _Variant((ATTN, MOE), rank=5)

#: The module test's configurations: the deployment; the attention-only canary
#: (``TRTLLM_DWDP_DENSE_SEGMENTS=attn``); the copy engine's batched copies (``contention_opt``)
#: with a chunked store; the per-slice copies of a DWDP run without either transfer option; a
#: lenient deployment whose first forward runs the fenced demand fetches of a forward nobody
#: primed.
_VARIANTS = (
    _DEPLOYED,
    _Variant((ATTN,), rank=15),
    _Variant((ATTN, MOE), rank=0, transfer=CopyMode.BATCHED, store_chunk_mb=64),
    _Variant((ATTN, MOE), rank=10, transfer=CopyMode.PER_SLICE),
    _Variant((ATTN, MOE), rank=3, strict=False),
)

#: GPU clock cycles of a spin kernel: long enough that a read or a
#: copy that does not wait for its event runs into the other stream's work.
_SPIN_CYCLES = 20_000_000


class _SpinningLauncher:
    """The production copy launcher behind an optional spin kernel on the copy stream.

    With ``spin_cycles`` set every copy lands long after compute has reached the H1 that waits
    for it, so a read that did not wait would see the slot's previous holder or the NaN fill.
    """

    def __init__(self, launcher: DenseCopyLauncher) -> None:
        self._launcher = launcher
        self.spin_cycles = 0

    def __call__(self, layer_idx: int) -> None:
        _spin(self.spin_cycles)
        self._launcher(layer_idx)

    def release(self) -> None:
        self._launcher.release()


class _DenseRun(NamedTuple):
    """A committed dense DWDP setup and what it showed."""

    runtime: DenseDWDP
    launcher: _SpinningLauncher
    layout: DenseLayout
    allgathers: int
    seconds: float
    allocated_before: int
    allocated_after: int


def _set_up_dense(
    model: mkl.KimiLinearForCausalLM,
    variant: _Variant,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> _DenseRun:
    """``setup_dense_dwdp`` as ``DwdpManager.setup`` calls it, for one rank of a loopback group.

    The weight manager stand-in supplies the copy stream and the variant's transfer mode; the
    launcher is the production one for that mode, behind a ``_SpinningLauncher``.
    """
    layouts = []
    collect = dense_setup.collect_dense_layout

    def recording_collect(*args, **kwargs) -> DenseLayout:
        layout = collect(*args, **kwargs)
        layouts.append(layout)
        return layout

    launchers = []

    def spinning_launcher(plans, weight_manager, device) -> _SpinningLauncher:
        mode = select_copy_mode(
            use_gather=weight_manager.use_gather, use_batched=weight_manager.use_batched_prefetch
        )
        launchers.append(_SpinningLauncher(DenseCopyLauncher(plans, mode, device=device)))
        return launchers[-1]

    monkeypatch.setattr(dense_setup, "collect_dense_layout", recording_collect)
    # What makes DwdpManager.setup run dense setup for this model at all.
    assert dense_setup.dense_roots_available(model)
    group = _LoopbackGroup(variant.rank, _NUM_RANKS)
    weight_manager = SimpleNamespace(
        copy_stream=torch.cuda.Stream(device),
        use_gather=variant.transfer is CopyMode.GATHER,
        use_batched_prefetch=variant.transfer is CopyMode.BATCHED,
    )
    config = DenseConfig(
        enabled=True,
        segments=variant.segments,
        strict=variant.strict,
        debug=variant.debug,
        store_chunk_mb=variant.store_chunk_mb,
    )
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    started = time.monotonic()
    runtime = dense_setup.setup_dense_dwdp(
        model,
        weight_manager,
        group,
        dwdp_rank=variant.rank,
        dwdp_size=_NUM_RANKS,
        device_id=device.index,
        moe_layers=list(_COVERED),
        config=config,
        store_factory=LoopbackShardStore,
        launcher_factory=spinning_launcher,
    )
    seconds = time.monotonic() - started
    assert runtime is not None, "dense DWDP setup returned no runtime"
    (layout,) = layouts
    (launcher,) = launchers
    return _DenseRun(
        runtime,
        launcher,
        layout,
        group.allgathers,
        seconds,
        allocated_before,
        torch.cuda.memory_allocated(device),
    )


class _TacticLog(NamedTuple):
    """Every autotuned op call: ``(op, input shapes, cache hit, tactic, output dtype)``, in call
    order; the output dtype is the runner's, ``None`` for a runner without one."""

    cache_loaded: bool
    calls: list


@pytest.fixture(scope="module")
def device() -> Iterator[torch.device]:
    """A Rubin GPU with the deployment's environment and the gather op; skips otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the Kimi-K3 MXFP8 weight-read GEMM runs on SM107 (Rubin) only")
    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("the CuTe DSL Rubin kernels are not available")
    with pytest.MonkeyPatch.context() as patch:
        for name, value in _DEPLOYMENT_ENV.items():
            patch.setenv(name, value)
        for name in _UNSET_ENV:
            patch.delenv(name, raising=False)
        cuda = torch.device("cuda", torch.cuda.current_device())
        _require_gather(cuda)
        yield cuda


def _require_gather(device: torch.device) -> None:
    """Skip unless the gather op is registered and has a kernel for this GPU."""
    op = fabric_gather.gather_op()
    if op is None:
        pytest.skip("trtllm::dwdp_fabric_gather is not registered by this TensorRT-LLM build")
    src = torch.arange(fabric_gather.LINE_BYTES, dtype=torch.uint8, device=device)
    dst = torch.zeros_like(src)

    def int64(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device)

    try:
        op(int64([dst.data_ptr()]), int64([src.data_ptr()]), int64([1]), int64([0, 1]), 1, 32)
        torch.cuda.synchronize(device)
    except RuntimeError as exc:
        if "no kernel image" in str(exc):
            pytest.skip(f"this build has no gather kernel for this GPU: {exc}")
        raise
    assert torch.equal(dst, src), "a one-line probe gather did not copy its 16 bytes"


@pytest.fixture
def tactics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _TacticLog:
    """No tuning; the deployment's cache if ``TLLM_AUTOTUNER_CACHE_PATH`` names one.

    The cache is loaded from a scratch copy into a fresh profiling cache, so neither the file nor
    the process-wide autotuner state outlives the test.
    """
    tuner = AutoTuner.get()
    cache = AutoTunerProfilingCache()
    source = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH")
    loaded = bool(source) and os.path.isfile(source)
    if loaded:
        scratch = tmp_path / "autotuner_cache.json"
        shutil.copyfile(source, scratch)
        cache.load_cache(scratch, tuner.mapping.rank)
    monkeypatch.setattr(tuner, "profiling_cache", cache)
    monkeypatch.setattr(tuner, "is_tuning_mode", False)
    log = _TacticLog(loaded, [])
    choose_one = tuner.choose_one

    def recording_choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
        shapes = tuple(
            value.size() if isinstance(value, torch.Tensor) else torch.Size((0,))
            for value in inputs
        )
        hit = cache.search_cache(custom_op, runners, shapes, tuning_config)[0]
        runner, tactic = choose_one(custom_op, runners, tuning_config, inputs, **kwargs)
        output_dtype = getattr(runner, "output_dtype", None)
        log.calls.append((custom_op, shapes, hit, str(tactic), output_dtype))
        return runner, tactic

    monkeypatch.setattr(tuner, "choose_one", recording_choose_one)
    return log


def _check_tactics(log: _TacticLog, reference_calls: list) -> int:
    """The calls since the references repeat them exactly; MXFP8 GEMMs with BF16 output hit a
    loaded cache (no warmup tunes the FP8-output one).

    Returns:
        The number of MXFP8 weight-read GEMMs in the calls.
    """
    assert log.calls == reference_calls, "the GEMM calls or their tactics changed after setup"
    mxfp8 = [call for call in log.calls if call[0] == _MXFP8_GEMM]
    assert mxfp8, "no MXFP8 weight-read GEMM ran"
    if log.cache_loaded:
        missed = sorted(
            {
                tuple(tuple(shape) for shape in call[1][:2])
                for call in mxfp8
                if not call[2] and call[4] == torch.bfloat16
            }
        )
        assert not missed, f"MXFP8 GEMMs missed the autotuner cache: {missed}"
    return len(mxfp8)


def _expected_segment(layer_idx: int, segment: SegmentKind) -> tuple[int, int]:
    if segment is MOE:
        return _MOE_DENSE
    return _KDA_ATTN if layer_idx + 1 in _KDA_LAYERS else _MLA_ATTN


def _check_setup(model: mkl.KimiLinearForCausalLM, run: _DenseRun, variant: _Variant) -> None:
    """Five collectives; the model's anchors; slots shared as the rings say; loads refused."""
    segments = variant.segments
    assert run.allgathers == 5
    assert model._dense_dwdp_active and model.model._dense_dwdp_active
    layout = run.layout
    assert tuple(layout.layers) == _COVERED
    assert list(layout.warnings) == []
    assert (len(layout.chunks) > 1) == bool(variant.store_chunk_mb)
    found = {
        key: (segment.moved_bytes, len(segment.anchors)) for key, segment in layout.segments.items()
    }
    expected = {
        (layer_idx, kind): _expected_segment(layer_idx, kind)
        for layer_idx in _COVERED
        for kind in segments
    }
    assert found == expected
    _check_slot_sharing(model.model.layers, segments)
    with pytest.raises(RuntimeError, match="dense DWDP is active"):
        model.load_weights({})


def _slot_storage(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


#: Moved MLA weights that two MLA layers sharing a slot read at one address with one shape.
_MLA_SLOT_WEIGHTS = (
    "q_b_proj.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
    "k_b_proj_trans",
    "v_b_proj_dequant",
)


def _attention_weight(layer: mkl.KimiLinearDecoderLayer) -> torch.Tensor:
    """A moved attention weight of ``layer``: the KDA or the MLA ``o_proj`` weight."""
    if layer.is_kda:
        return layer.linear_attn.o_proj.weight
    return layer.self_attn.mixer.o_proj.weight


def _check_slot_sharing(layers: nn.ModuleList, segments: Sequence[SegmentKind]) -> None:
    """Layers an even number of positions apart (attention) or a multiple of three (MoE-dense)
    read through one slot; the two MLA layers read every moved MLA weight at the same address.
    """
    positions = {layer_idx: position for position, layer_idx in enumerate(_COVERED)}
    for first, second in itertools.combinations(_COVERED, 2):
        apart = positions[second] - positions[first]
        same_attention = _slot_storage(_attention_weight(layers[first])) == _slot_storage(
            _attention_weight(layers[second])
        )
        assert same_attention == (apart % 2 == 0), (first, second)
        gates = [layers[index].block_sparse_moe.gate.weight for index in (first, second)]
        same_gate = gates[0].data_ptr() == gates[1].data_ptr()
        assert same_gate == (MOE in segments and apart % 3 == 0), (first, second)
    assert (
        layers[2].linear_attn.qkvg_proj.weight.data_ptr()
        == layers[4].linear_attn.qkvg_proj.weight.data_ptr()
    )
    mla = [index - 1 for index in _MLA_LAYERS]
    assert len(mla) == 2 and (positions[mla[1]] - positions[mla[0]]) % 2 == 0
    for name in _MLA_SLOT_WEIGHTS:
        weights = [operator.attrgetter(name)(layers[index].self_attn.mixer) for index in mla]
        placements = [(weight.data_ptr(), weight.shape, weight.stride()) for weight in weights]
        assert placements[0] == placements[1], name


def _check_reads_follow_the_slots(
    runtime: DenseDWDP,
    layers: nn.ModuleList,
    inputs: _Inputs,
    references: dict[str, torch.Tensor],
    segments: Sequence[SegmentKind],
) -> None:
    """Outside its windows a layer reads whatever its slots hold now.

    After a forward each slot holds the last covered layer that maps to it: the attention slots
    layers 7 (MLA) and 8 (KDA), the first MoE-dense slot layer 7.  So layer 3's MLA readers see
    layer 7's MLA weights, layer 2's KDA readers layer 8's, and layer 1's MoE-dense readers
    layer 7's; MoE-dense weights that stay in place still give layer 1's outputs.  This is what
    makes the bit-exact comparisons able to catch a slot that holds the wrong layer, MLA layers
    included.
    """
    assert runtime.slot_tag(ATTN, runtime.slot_index(ATTN, 3)) == 7
    assert runtime.slot_tag(ATTN, runtime.slot_index(ATTN, 2)) == 8
    if MOE in segments:
        assert runtime.slot_tag(MOE, runtime.slot_index(MOE, 1)) == 7
    with torch.no_grad():
        stale = _layer_outputs(3, layers[3], inputs, ATTN)
        stale.update(_layer_outputs(2, layers[2], inputs, ATTN))
        stale.update(_layer_outputs(1, layers[1], inputs, MOE))
    torch.cuda.synchronize()
    changed = {
        name
        for name, value in stale.items()
        if not torch.equal(_bits(value), _bits(references[name]))
    }
    expected = {name for name in stale if not name.startswith("layers.1.") or MOE in segments}
    assert changed == expected


def _check_detached(layers: nn.ModuleList, segments: Sequence[SegmentKind]) -> None:
    """After release every moved binding is a zero-size tensor; what stayed in place is intact."""
    for layer_idx in _COVERED:
        layer = layers[layer_idx]
        if layer.is_kda:
            attention = layer.linear_attn
            moved = [
                attention.qkvg_proj.weight,
                attention.q_proj.weight,
                attention.o_proj.weight_scale,
                attention._bfa_proj_weight,
                attention.f_a_proj.weight,
                attention.f_b_proj.weight,
            ]
        else:
            mixer = layer.self_attn.mixer
            moved = [mixer.kv_b_proj.weight, mixer.v_b_proj, mixer.k_b_proj_trans_dequant]
        moe = layer.block_sparse_moe
        kept = [moe.routed_expert_norm.weight, moe.gate.e_score_correction_bias]
        moe_dense = [moe.gate.weight, moe.shared_experts.gate_up_proj.weight]
        if MOE in segments:
            moved += moe_dense
        else:
            kept += moe_dense
        assert all(tensor.numel() == 0 for tensor in moved), f"layer {layer_idx}"
        assert all(tensor.numel() > 0 for tensor in kept), f"layer {layer_idx}"


def _report(name: str, run: _DenseRun, details: str) -> None:
    """Print what setup measured (shown with ``pytest -s``)."""
    layout = run.layout
    print(
        f"\n[{name}] setup {run.seconds:.2f}s, moved {layout.moved_bytes / 1e9:.3f} GB, "
        f"store/rank {layout.store_bytes / 1e9:.3f} GB, slots {layout.slot_bytes / 1e9:.3f} GB, "
        f"memory_allocated {run.allocated_before / 1e9:.3f} -> "
        f"{run.allocated_after / 1e9:.3f} GB; {details}"
    )


# ---------------------------------------------------------------------------------------------
# Module readers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", _VARIANTS, ids=[variant.name for variant in _VARIANTS])
def test_k3_modules_are_bit_exact_after_dense_dwdp(
    variant: _Variant,
    device: torch.device,
    tactics: _TacticLog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every dense reader of eight real K3 layers gives bit-identical outputs after setup.

    With only the attention segment (the attention-only canary) the MoE-dense weights stay in
    place.
    """
    segments = variant.segments
    model = _build_model(monkeypatch, device)
    layers = model.model.layers
    inputs = {tokens: _make_inputs(tokens, device) for tokens in _TOKEN_COUNTS}

    references = {tokens: _reference_outputs(layers, inputs[tokens]) for tokens in _TOKEN_COUNTS}
    reference_calls = list(tactics.calls)
    for tokens in _TOKEN_COUNTS:
        repeated = _reference_outputs(layers, inputs[tokens])
        assert not _differences(references[tokens], repeated), "the modules are not deterministic"
        # Widened first: torch has no isfinite for FP8.
        assert all(torch.isfinite(value.float()).all() for value in references[tokens].values()), (
            "the synthetic weights give non-finite outputs"
        )
    del repeated
    tactics.calls.clear()

    run = _set_up_dense(model, variant, device, monkeypatch)
    runtime = run.runtime
    try:
        _check_setup(model, run, variant)
        # The first forward delays every copy (each read must wait for its gather), the second
        # every read window (each gather must wait for the reads of the slot's previous holder),
        # the third neither.
        for forward, tokens in enumerate(_TOKEN_COUNTS):
            run.launcher.spin_cycles = _SPIN_CYCLES if forward == 0 else 0
            outputs = _hooked_forward(
                runtime,
                layers,
                inputs[tokens],
                prime=variant.strict or forward > 0,
                spin_cycles=_SPIN_CYCLES if forward == 1 else 0,
            )
            torch.cuda.synchronize(device)
            differences = _differences(references[tokens], outputs)
            assert not differences, f"T={tokens}: " + "; ".join(differences)
            del outputs
        run.launcher.spin_cycles = 0
        # Unprimed, the first two layers' H1 each fetch on demand; the rest was enqueued at H2.
        assert runtime.demand_fetches == (0 if variant.strict else 2)
        assert runtime.war_fences == 0
        assert not runtime.forward_open
        gemms = _check_tactics(tactics, reference_calls)
        tokens = _TOKEN_COUNTS[0]
        _check_reads_follow_the_slots(runtime, layers, inputs[tokens], references[tokens], segments)
        _report(
            f"dense k3 modules {variant.name}",
            run,
            f"{sum(len(value) for value in references.values())} outputs bit-exact over "
            f"T={list(_TOKEN_COUNTS)}, {gemms} MXFP8 GEMMs "
            f"({'cached' if tactics.cache_loaded else 'fallback'} tactics), "
            f"demand fetches {runtime.demand_fetches}, fences {runtime.war_fences}",
        )
    finally:
        runtime.release()
    _check_detached(layers, segments)


# ---------------------------------------------------------------------------------------------
# The decoder forward through the real hooks
# ---------------------------------------------------------------------------------------------


class _ExpertPrefetchRecorder:
    """The expert DWDP calls of ``DwdpManager`` priming and recovery, recorded instead of run.

    The routed experts are stand-ins, so there is nothing to copy for them.
    """

    def __init__(self, layers: Sequence[int]) -> None:
        self._layers = tuple(layers)
        self.calls: list[tuple[str, int]] = []

    def first_moe_layer(self) -> int:
        return self._layers[0]

    def next_moe_layer(self, layer_idx: int) -> int | None:
        position = self._layers.index(layer_idx) + 1
        return self._layers[position] if position < len(self._layers) else None

    def prefetch_layer(self, layer_idx: int) -> None:
        self.calls.append(("prefetch", layer_idx))

    def record_compute(self, layer_idx: int) -> None:
        self.calls.append(("record_compute", layer_idx))


def _dwdp_manager(runtime: DenseDWDP, experts: _ExpertPrefetchRecorder) -> DwdpManager:
    """A ``DwdpManager`` holding ``runtime`` and the expert recorder, without its DWDP group.

    Its constructor needs the expert DWDP group; ``dense`` and ``ensure_primed`` read only the
    dense runtime and the weight manager, which are attached directly.
    """
    manager = DwdpManager.__new__(DwdpManager)
    manager._dense = runtime
    manager._weight_manager = experts
    return manager


def _attach_dwdp(model: mkl.KimiLinearForCausalLM, manager: DwdpManager | None) -> None:
    """Hand the model the manager it captures at construction when DWDP is on (M0)."""
    model.model._dwdp = manager
    for layer in model.model.layers:
        layer._dwdp = manager


#: The executor's token budget per forward in the decoder test, the same for every forward as in
#: an executor: no request is longer.
_MAX_NUM_TOKENS = max(_TOKEN_COUNTS)


def _kda_request_metadata(tokens: int, device: torch.device) -> SimpleNamespace:
    """``mamba_metadata`` of one fresh context request, as ``KimiK3MambaMetadata`` prepares it."""
    chunks = -(-tokens // KDA_PREFILL_CHUNK_SIZE)
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.long, device=device)
    chunk_indices = torch.stack(
        (torch.zeros(chunks, dtype=torch.long), torch.arange(chunks, dtype=torch.long)), dim=1
    )
    return SimpleNamespace(
        state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        query_start_loc=cu_seqlens.to(torch.int32),
        query_start_loc_long=cu_seqlens,
        has_initial_states=torch.zeros(1, dtype=torch.bool, device=device),
        use_initial_states=False,
        kda_chunk_indices=chunk_indices.to(device),
        kda_varlen_is_aligned=tokens % KDA_PREFILL_CHUNK_SIZE == 0,
        kda_single_sequence_length=tokens,
        _arange_buffer=torch.arange(2, dtype=torch.int32, device=device),
    )


def _kda_layer_cache(device: torch.device) -> SimpleNamespace:
    """One request's KDA caches: ``[1, 3 * D, W - 1]`` BF16 convolution, ``[1, H, V, K]`` fp32 state."""
    heads = _K3_LINEAR_ATTN["num_heads"]
    head_dim = _K3_LINEAR_ATTN["head_dim"]
    width = _K3_LINEAR_ATTN["short_conv_kernel_size"]
    return SimpleNamespace(
        conv=torch.zeros(1, 3 * heads * head_dim, width - 1, dtype=torch.bfloat16, device=device),
        temporal=torch.zeros(1, heads, head_dim, head_dim, dtype=torch.float32, device=device),
        kda_k_cache=None,
        kda_g_cache=None,
    )


class _RequestSlot:
    """The caches of one context request, reused by every forward of the decoder test.

    The MLA layer reads the TRTLLM backend's metadata over a paged KV cache.  The KDA layers read
    ``mamba_metadata`` and their state caches through ``kv_cache_manager.mamba_layer_cache``; in
    the deployment the hybrid cache manager provides both, here stand-ins do (as in
    ``test_kimi_kda_verify_parity.py``).  Every forward runs one fresh request without initial
    state in the slot, which rewrites the slot's KV from position 0.

    Args:
        model: The model whose layers the requests run through.
        tokens: Capacity of the slot in tokens.
        device: The CUDA device.
    """

    def __init__(self, model: mkl.KimiLinearForCausalLM, tokens: int, device: torch.device) -> None:
        cfg = SimpleNamespace(**_K3_TEXT_CONFIG)
        scenario = SimpleNamespace(
            ctx_len=tokens,
            batch=1,
            kv_cache_tokens_per_block=64,
            num_layers=_NUM_LAYERS,
            kv_cache_dtype=torch.bfloat16,
        )
        self._kv_cache_manager, _ = setup_kv_and_metadata(
            scenario,
            model.model_config.mapping,
            cache_type=CACHE_TYPE_SELFKONLY,
            num_kv_heads=1,
            head_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
        )
        caches = {
            index: _kda_layer_cache(device)
            for index, layer in enumerate(model.model.layers)
            if layer.is_kda
        }
        self._kv_cache_manager.mamba_layer_cache = caches.__getitem__
        self._mapping = model.model_config.mapping
        self._device = device

    def request_metadata(self, tokens: int) -> AttentionMetadata:
        """Attention metadata of one fresh context request of ``tokens`` tokens."""
        metadata = get_attention_backend("TRTLLM").Metadata(
            seq_lens=torch.tensor([tokens], dtype=torch.int),
            request_ids=[0],
            max_num_requests=1,
            num_contexts=1,
            prompt_lens=[tokens],
            max_num_tokens=_MAX_NUM_TOKENS,
            kv_cache_manager=self._kv_cache_manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
            mapping=self._mapping,
        )
        metadata.prepare()
        metadata.mamba_metadata = _kda_request_metadata(tokens, self._device)
        return metadata

    def close(self) -> None:
        torch.cuda.synchronize(self._device)
        self._kv_cache_manager.shutdown()


def _capture_output(
    outputs: dict[str, torch.Tensor], name: str, module: nn.Module, args: tuple, output: tuple
) -> None:
    outputs[name] = output[0].clone()


def _model_forward(
    model: mkl.KimiLinearForCausalLM, slot: _RequestSlot, tokens: int
) -> dict[str, torch.Tensor]:
    """``KimiLinearModel.forward`` over one context request of ``tokens`` tokens in ``slot``.

    Returns:
        Each decoder layer's output (the running prefix sum) and the model's final hidden state.
    """
    device = model.model.embed_tokens.weight.device
    generator = torch.Generator(device=device).manual_seed(tokens)
    input_ids = torch.randint(
        0,
        _CUT_DOWN["vocab_size"],
        (tokens,),
        generator=generator,
        device=device,
        dtype=torch.int32,
    )
    metadata = slot.request_metadata(tokens)
    outputs = {}
    handles = [
        layer.register_forward_hook(functools.partial(_capture_output, outputs, f"layers.{index}"))
        for index, layer in enumerate(model.model.layers)
    ]
    extra_attrs = dict(model.model_config.extra_attrs, attention_metadata=weakref.ref(metadata))
    try:
        with model_extra_attrs(extra_attrs), torch.no_grad():
            outputs["final"] = model.model(metadata, input_ids=input_ids).clone()
    finally:
        for handle in handles:
            handle.remove()
    return outputs


@contextlib.contextmanager
def _slow_windows(model: mkl.KimiLinearForCausalLM, cycles: int) -> Iterator[None]:
    """Spin the compute stream before every covered layer's attention and MoE block."""
    handles = []
    if cycles:
        for layer_idx in _COVERED:
            layer = model.model.layers[layer_idx]
            attention = layer.linear_attn if layer.is_kda else layer.self_attn
            for module in (attention, layer.block_sparse_moe):
                handles.append(module.register_forward_pre_hook(lambda module, args: _spin(cycles)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("debug", [1, 2], ids=["debug1", "debug2"])
def test_k3_decoder_forward_is_bit_exact_through_the_dense_hooks(
    debug: int, device: torch.device, tactics: _TacticLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model's own forward over layers 0-8, with H0-H3 at their call sites, is bit-exact.

    ``KimiLinearModel.forward`` runs one context request per forward: real KDA prefill and MLA
    context attention, the MoE blocks with the routed-expert stand-in, layer 0 uncovered.  Dense
    DWDP reaches the hooks through a ``DwdpManager``, whose ``ensure_primed`` is H0.  Every
    layer's output and the final hidden state must equal the outputs without dense DWDP bit for
    bit, the first forward with delayed copies and the second with delayed read windows.  A
    forward that then fails inside layer 3's MoE block leaves the forward open; the next forward
    recovers through the WAR barrier and must still be bit-exact.  The last forward runs on its
    own execution stream after ``DwdpManager.prefetch_first_layers`` primed from the default
    stream, as the executor loop does.  With ``TRTLLM_DWDP_DENSE_DEBUG=2`` (the validation
    run) the slot watchdog also checks every slot each H1 waits for against the layer's P1
    checksum, at Kimi-K3 blob sizes, and must find nothing.
    """
    model = _build_model(monkeypatch, device)
    slot = _RequestSlot(model, _MAX_NUM_TOKENS, device)
    try:
        variant = _DEPLOYED._replace(debug=debug)
        _check_decoder_forwards(model, slot, variant, device, tactics, monkeypatch)
    finally:
        slot.close()


@pytest.mark.parametrize("debug", [1])
def test_k3_decoder_forward_with_ctx_fused_moe_residual_is_bit_exact(
    debug: int, device: torch.device, tactics: _TacticLog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX`` under dense DWDP, through the model's own forward.

    The references run without the flag. With it, every MoE layer of the 3000-token forwards adds
    its residual in the fused kernel and none of the 16- and 1-token forwards does, and every
    layer and final output of the decoder test's forwards still equals the references bit for bit.
    """
    model = _build_model(monkeypatch, device)
    slot = _RequestSlot(model, _MAX_NUM_TOKENS, device)
    try:
        variant = _DEPLOYED._replace(debug=debug)
        _check_decoder_forwards(
            model, slot, variant, device, tactics, monkeypatch, ctx_fused_residual=True
        )
    finally:
        slot.close()


def _fuse_ctx_moe_residual(monkeypatch: pytest.MonkeyPatch) -> list:
    """Turns ``TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX`` on; returns the list of the fused kernel's grids."""
    from tensorrt_llm._torch.modules import kimi_k3_moe_residual as residual_mod

    launches = []
    kernel = residual_mod._moe_residual_add_kernel

    class _LaunchSpy:
        def __getitem__(self, grid):
            launches.append(grid)
            return kernel[grid]

    monkeypatch.setattr(residual_mod, "_moe_residual_add_kernel", _LaunchSpy())
    monkeypatch.setattr(mkl, "_K3_FUSE_MOE_RESIDUAL_CTX", True)
    return launches


def _check_decoder_forwards(
    model: mkl.KimiLinearForCausalLM,
    slot: _RequestSlot,
    variant: _Variant,
    device: torch.device,
    tactics: _TacticLog,
    monkeypatch: pytest.MonkeyPatch,
    ctx_fused_residual: bool = False,
) -> None:
    """The body of the decoder test, with the request slot open.

    With ``ctx_fused_residual`` the references run without ``TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX``
    and every forward after them with it.
    """
    references = {tokens: _model_forward(model, slot, tokens) for tokens in _TOKEN_COUNTS}
    reference_calls = list(tactics.calls)
    for tokens in _TOKEN_COUNTS:
        repeated = _model_forward(model, slot, tokens)
        assert not _differences(references[tokens], repeated), "the forward is not deterministic"
        assert all(torch.isfinite(value).all() for value in references[tokens].values())
    del repeated
    tactics.calls.clear()
    launches = _fuse_ctx_moe_residual(monkeypatch) if ctx_fused_residual else None
    moe_layers = sum(layer.is_moe for layer in model.model.layers)
    long_forwards = sum(tokens > mkl._MOE_RESIDUAL_MAX_FUSED_ROWS for tokens in _TOKEN_COUNTS)

    run = _set_up_dense(model, variant, device, monkeypatch)
    runtime = run.runtime
    experts = _ExpertPrefetchRecorder(_COVERED)
    _attach_dwdp(model, _dwdp_manager(runtime, experts))
    try:
        _check_setup(model, run, variant)
        # As in the module test: delayed copies, then delayed read windows, then neither.
        for forward, tokens in enumerate(_TOKEN_COUNTS):
            run.launcher.spin_cycles = _SPIN_CYCLES if forward == 0 else 0
            with _slow_windows(model, _SPIN_CYCLES if forward == 1 else 0):
                outputs = _model_forward(model, slot, tokens)
            differences = _differences(references[tokens], outputs)
            assert not differences, f"T={tokens}: " + "; ".join(differences)
        run.launcher.spin_cycles = 0
        assert experts.calls == [("prefetch", 1), ("prefetch", 2)] * len(_TOKEN_COUNTS)
        if launches is not None:
            assert long_forwards == 1 and len(launches) == moe_layers, (moe_layers, launches)
        gemms = _check_tactics(tactics, reference_calls)

        routed_experts = model.model.layers[3].block_sparse_moe.routed_experts
        routed_experts.fail = True
        with pytest.raises(RuntimeError, match="injected"):
            _model_forward(model, slot, _TOKEN_COUNTS[0])
        routed_experts.fail = False
        assert runtime.forward_open
        experts.calls.clear()
        tokens = _LONG
        differences = _differences(references[tokens], _model_forward(model, slot, tokens))
        assert not differences, f"T={tokens} after an aborted forward: " + "; ".join(differences)
        if launches is not None:
            # The aborted 16-token forward added nothing; the 3000-token one every MoE layer.
            assert len(launches) == 2 * moe_layers, (moe_layers, launches)
        assert experts.calls == [
            ("record_compute", 1),
            ("record_compute", 2),
            ("prefetch", 1),
            ("prefetch", 2),
        ]

        # The executor loop primes from its own stream before the forward runs on the execution
        # stream; H0 then finds the slots primed and enqueues nothing.
        experts.calls.clear()
        manager = model.model._dwdp
        manager.prefetch_first_layers()
        assert runtime.primed
        execution_stream = torch.cuda.Stream(device)
        execution_stream.wait_stream(torch.cuda.current_stream(device))
        run.launcher.spin_cycles = _SPIN_CYCLES
        tokens = _TOKEN_COUNTS[0]
        with torch.cuda.stream(execution_stream):
            outputs = _model_forward(model, slot, tokens)
        torch.cuda.synchronize(device)
        run.launcher.spin_cycles = 0
        differences = _differences(references[tokens], outputs)
        assert not differences, f"T={tokens} on the execution stream: " + "; ".join(differences)
        if launches is not None:
            assert len(launches) == 2 * moe_layers, (moe_layers, launches)
        assert experts.calls == [("prefetch", 1), ("prefetch", 2)]
        assert (runtime.demand_fetches, runtime.war_fences) == (0, 0)
        assert not runtime.forward_open
        # Five forwards ran every covered layer's H1; the aborted one ran H1 of layers 1-3.
        checks = (5 * len(_COVERED) + _COVERED.index(3) + 1) * len(variant.segments)
        if variant.debug < 2:
            checks = 0
        assert (runtime.watchdog_checks, runtime.watchdog_mismatches) == (checks, 0)
        _report(
            f"dense k3 decoder forward debug={variant.debug}"
            + (" ctx-fused-moe-residual" if ctx_fused_residual else ""),
            run,
            f"{len(references[tokens])} layer and final outputs per forward bit-exact over "
            f"T={list(_TOKEN_COUNTS)}, after an aborted forward and on an execution stream the "
            f"executor loop primed for; {gemms} MXFP8 GEMMs "
            f"({'cached' if tactics.cache_loaded else 'fallback'} tactics); slot watchdog "
            f"{runtime.watchdog_checks} checks, {runtime.watchdog_mismatches} mismatches",
        )
    finally:
        _attach_dwdp(model, None)
        runtime.release()
    assert runtime.watchdog_mismatches == 0
    _check_detached(model.model.layers, variant.segments)


@pytest.mark.cpu_only
def test_k3_shapes_match_the_checkpoint() -> None:
    """The shapes this file builds are the Kimi-K3 NVFP4 checkpoint's (when it is available)."""
    root = llm_models_root()
    config_path = None if root is None else root / "Kimi-K3-NVFP4" / "config.json"
    if config_path is None or not config_path.is_file():
        pytest.skip("the Kimi-K3-NVFP4 checkpoint is not under LLM_MODELS_ROOT")
    with open(config_path) as file:
        text = json.load(file)["text_config"]
    for name, value in _K3_TEXT_CONFIG.items():
        assert text[name] == value, name
    linear_attn = text["linear_attn_config"]
    for name, value in _K3_LINEAR_ATTN.items():
        assert linear_attn[name] == value, name
    built = set(range(1, _NUM_LAYERS + 1))
    assert built & set(linear_attn["kda_layers"]) == set(_KDA_LAYERS)
    assert built & set(linear_attn["full_attn_layers"]) == set(_MLA_LAYERS)
