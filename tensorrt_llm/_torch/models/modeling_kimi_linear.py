# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KimiLinearForCausalLM — Kimi K3 text model, PyTorch backend.

Runtime integration of the Kimi K3 hybrid architecture for the standard
TRT-LLM PyTorch-backend flow (``LLM(model=<ckpt>) -> generate``):

* 93 decoder layers: 69 KDA (Kimi Delta Attention, linear attention) layers
  and 24 MLA (absorbed-MQA, NoPE) layers, per the 1-indexed
  ``linear_attn_config.kda_layers`` / ``full_attn_layers`` schedule.
* Layer 0 uses a dense SiTU MLP; layers 1..92 use the 896-expert latent MoE
  (top-16, sigmoid + e_score_correction_bias routing, MXFP4 routed experts,
  2 shared experts, latent down/norm/up projections).
* The attention-residual ("attn_res") scheme from the HF reference
  ``modeling_kimi.py`` is applied per token: snapshot mixing before
  ``input_layernorm`` / ``post_attention_layernorm`` and at the model output,
  with a new snapshot appended whenever ``layer_idx % attn_res_block_size == 0``.

Caching
-------
KDA states live on the mamba side of a ``MixedMambaHybridCacheManager``
(wired in ``pyexecutor/_util.py``): per layer, a short-conv slot of
``[3 * num_heads * head_dim, W]`` bf16 (the full FLA ``ShortConvolution``
cache window, sections ``[q | k | v]``) and a delta-rule recurrent slot of
``[num_heads, head_dim, head_dim]`` fp32 (``[H, V, K]``, the
``state_v_first`` FLA layout). MLA layers use the paged-KV side with
``num_kv_heads=1`` and ``head_dim = kv_lora_rank + qk_rope_head_dim`` (576),
SELFKONLY, exactly like DeepSeek MLA.

MLA prefill routing
-------------------
The parity-tested in-tree ``KimiK3MLAAttention`` routes prefill through the
MLA *generation* FMHA (the context FMHA cubin was found numerically wrong for
the K3 head configuration; see the module docstring). That path requires
generation-shaped metadata with one request, so the model builds one derived
``TrtllmAttentionMetadata`` per context request per forward step (B=1 each)
and runs the decode path on the whole generation batch at once using the
executor-provided metadata with ``AttentionInputType.generation_only``.

Parallelism
-----------
EP-only: every rank holds the full bf16 non-expert model replicated (no TP on
any ``nn.Linear``); each MoE layer holds only a contiguous
``num_experts / ep_size`` slice of the MXFP4 expert bank, where
``ep_size == mapping.tp_size`` (launch with ``tensor_parallel_size=N``).
Routing is computed replicated; the routed partial sums are all-reduced in
the latent space (before ``routed_expert_norm`` / ``routed_expert_up_proj``,
which are nonlinear/linear layers applied to the full sum). ``lm_head`` uses
the stock ``LMHead`` (vocab-sharded + gather), so logits are identical on all
ranks.

Speculative decoding: SA (suffix automaton, one-engine, draft-weight-free);
the KDA/MLA runtimes implement multi-token verification with deferred
state promotion.

Chunked prefill is supported: continuation chunks feed the previous KDA
conv/recurrent state back into the FLA kernels (``use_initial_states``)
and the MLA prefill path natively attends over the cached latent prefix
(``kv_len = cached + q_len``). KV-cache block reuse is supported as an
opt-in via ``kv_cache_config.enable_block_reuse=true``, which routes to
the unified-pool ``CppMambaHybridCacheManager`` (per-block KDA state
snapshots every ``mamba_state_cache_interval`` tokens, FORCE_CHUNK
context chunking).

Not supported: pipeline parallelism, draft-head spec-dec modes
(MTP/Eagle — no draft-head checkpoint exists). SA speculative decoding
is validated only without block reuse (Mixed cache manager).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import os
import torch
from torch import nn

from ...logger import logger
from ..attention_backend import AttentionMetadata, TrtllmAttentionMetadata
from ..distributed import AllReduce
from ..metadata import KVCacheParams
from ..model_config import ModelConfig
from ..modules.kimi_k3_moe._mlp import KimiK3MLP, KimiK3RMSNorm, SituAndMul
from ..modules.kimi_k3_moe.kimi_k3_moe_block import KimiK3RoutedExpertBank
from ..modules.kimi_k3_moe.kimi_k3_moe_gate import KimiK3MoEGate
from ..modules.rms_norm import RMSNorm
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ..modules.kimi_k3_mla import (KimiK3MLAAttention,
                                       KimiK3MLARuntimeInputs)
    from ..modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention

# Identity-RoPE table positions for the MLA backends. K3 is NoPE (the table
# holds cos=1/sin=0 and the rope kernels are skipped), but the backend still
# allocates a table of this size per instance; the checkpoint's 1M
# max_position_embeddings would cost ~512MB per backend, so cap it.
_KIMI_K3_MLA_MAX_POSITIONS_ENV = "KIMI_K3_MLA_MAX_POSITIONS"
_KIMI_K3_MLA_MAX_POSITIONS_DEFAULT = 65536


# ---------------------------------------------------------------------------
# Config helpers.
# ---------------------------------------------------------------------------


def _get_text_config(pretrained_config: "PretrainedConfig"):
    """Return the Kimi text config, unwrapping a composite kimi_k3 config."""
    if getattr(pretrained_config, "model_type", None) == "kimi_k3" or (
            not hasattr(pretrained_config, "linear_attn_config")
            and hasattr(pretrained_config, "text_config")):
        return pretrained_config.text_config
    return pretrained_config


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


# ---------------------------------------------------------------------------
# attn_res: per-token snapshot mixing (HF `_apply_attn_res`).
# ---------------------------------------------------------------------------


KIMI_K3_FUSED_ATTN_RES_ENV = "KIMI_K3_FUSED_ATTN_RES"
"""Set to ``0`` to disable the in-tree fused Torch op
``trtllm::attn_res_fwd`` (Blackwell only). Default: fused with fallback."""

_FUSED_ATTN_RES_ENABLED = os.environ.get(KIMI_K3_FUSED_ATTN_RES_ENV, "1") == "1"


def _apply_attn_res_fused(prefix_sum: torch.Tensor,
                          block_residual: torch.Tensor, proj: nn.Linear,
                          norm: KimiK3RMSNorm) -> Optional[torch.Tensor]:
    """Fused attn_res via the in-tree ``trtllm::attn_res_fwd`` op.

    Returns ``None`` when the call falls outside the fused kernel's
    contract (dtype/shape/arch) so the caller can use the exact fp32
    reference instead. Candidate order matches the reference: snapshots
    first, the running prefix sum last.
    """
    if prefix_sum.dtype is not torch.bfloat16:
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[1])
    if K + 1 > 12 or M > 16384 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    try:
        attn_res_op = torch.ops.trtllm.attn_res_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.transpose(0, 1).reshape(K, M, 1,
                                                          H).contiguous()
    output, _rsigma, _probs, _logits = attn_res_op(
        layer_kernel, block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(), float(norm.eps))
    return output.reshape(M, H)


def _apply_attn_res(prefix_sum: torch.Tensor, block_residual: torch.Tensor,
                    proj: nn.Linear, norm: KimiK3RMSNorm) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_tokens, num_snapshots, hidden_size]``

    Unless ``KIMI_K3_FUSED_ATTN_RES=0``, inputs fitting the fused kernel's
    contract dispatch to the in-tree ``trtllm::attn_res_fwd`` op.
    """
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
        if fused is not None:
            return fused
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    variance = v_float.pow(2).mean(-1, keepdim=True)
    k = v_float * torch.rsqrt(variance + norm.eps)
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    scores = (k * score_weight).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    hidden_states = torch.matmul(probs, v_float).squeeze(1)
    return hidden_states.to(v.dtype)


# ---------------------------------------------------------------------------
# Dense / shared-expert MLP: fused [gate | up] layout (``KimiK3MLP``).
#
# The HF checkpoint stores separate ``gate_proj`` / ``up_proj`` tensors;
# ``load_weights`` row-concatenates them into ``gate_up_proj`` (see
# ``_gate_up_ckpt_keys``), replacing two GEMMs + torch.cat with one GEMM.
# ---------------------------------------------------------------------------


_GATE_UP_FUSED_SUFFIX = ".gate_up_proj.weight"


def _gate_up_ckpt_keys(fused_key: str) -> Tuple[str, str]:
    """Checkpoint ``(gate_proj, up_proj)`` keys whose row-concat loads the
    fused ``gate_up_proj`` parameter named by ``fused_key``."""
    return (fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".gate_proj.weight"),
            fused_key.replace(_GATE_UP_FUSED_SUFFIX, ".up_proj.weight"))


# ---------------------------------------------------------------------------
# MoE block with EP-sharded MXFP4 expert bank.
# ---------------------------------------------------------------------------


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 sparse MoE block for the runtime flow (EP-only sharding).

    The routed-expert compute is delegated to :class:`KimiK3RoutedExpertBank`
    (``forward_expert(local_idx, tokens)``); a fused MoE backend can be
    swapped in behind the same bank interface by replacing
    :meth:`_moe_infer_local`.
    """

    def __init__(self, model_config: ModelConfig, cfg, layer_idx: int):
        super().__init__()
        mapping = model_config.mapping
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        assert self.moe_hidden_size is not None, \
            "Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)"

        if mapping.enable_attention_dp and mapping.tp_size > 1:
            # DEP: attention runs data-parallel, experts stay sharded across
            # the EP group; tokens are exchanged via dispatch/combine below.
            self.ep_size = mapping.moe_ep_size
            self.ep_rank = mapping.moe_ep_rank
        else:
            self.ep_size = (mapping.tp_size
                            if not mapping.enable_attention_dp else 1)
            self.ep_rank = mapping.tp_rank if self.ep_size > 1 else 0
        assert self.num_experts % self.ep_size == 0, (
            f"num_experts={self.num_experts} not divisible by "
            f"ep_size={self.ep_size}")
        self.experts_per_rank = self.num_experts // self.ep_size
        self.expert_lo = self.ep_rank * self.experts_per_rank
        self.expert_hi = self.expert_lo + self.experts_per_rank

        situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
        situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
        dtype = torch.bfloat16

        # Routing params stay fp32 (scores are computed in fp32).
        self.gate = KimiK3MoEGate(cfg)

        self.expert_bank = KimiK3RoutedExpertBank(
            num_experts=self.experts_per_rank,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            activation=SituAndMul(beta=situ_beta,
                                  linear_beta=situ_linear_beta),
        )

        self.shared_experts = KimiK3MLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.moe_intermediate_size *
            cfg.num_shared_experts,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            use_fused_activation=True,
            dtype=dtype,
        )
        self.routed_expert_down_proj = nn.Linear(cfg.hidden_size,
                                                 self.moe_hidden_size,
                                                 bias=False,
                                                 dtype=dtype)
        self.routed_expert_up_proj = nn.Linear(self.moe_hidden_size,
                                               cfg.hidden_size,
                                               bias=False,
                                               dtype=dtype)
        assert getattr(cfg, "latent_moe_use_norm", False), \
            "Kimi K3 runtime expects latent_moe_use_norm=True"
        # Stock fused RMSNorm (flashinfer kernel; the no-flashinfer
        # fallback is the same fp32-variance eager math as KimiK3RMSNorm).
        self.routed_expert_norm = RMSNorm(hidden_size=self.moe_hidden_size,
                                          eps=cfg.rms_norm_eps,
                                          dtype=dtype)

        import os as _os
        self.moe = None
        if (torch.cuda.get_device_capability() in ((10, 0), (10, 3))
                and (_os.environ.get("KIMI_K3_CMOE") or "1") != "0"):
            # Middle tier: host the native SiTU op under ConfigurableMoE.
            # The stock wrapper contributes the ExternalComm scheduler
            # (chunking via moe_max_num_tokens), the CommunicationFactory
            # lane selection with per-workload fallback, and the EPLB attr
            # plumbing; the kernel and the lazily packed bank weights stay
            # K3-owned. KIMI_K3_CMOE=0 restores the direct path below.
            self.moe = self._build_configurable_moe(model_config, cfg,
                                                    layer_idx, dtype)
        self.dep_comm = None
        if (self.moe is None and mapping.enable_attention_dp
                and mapping.tp_size > 1):
            # KIMI_K3_CMOE=0 escape hatch only: the ConfigurableMoE path
            # above owns comm-lane selection (stock CommunicationFactory +
            # scheduler fallback); the direct path stays on the merged
            # upstream AG/RS form.
            from ..modules.fused_moe.communication.allgather_reducescatter \
                import AllGatherReduceScatter
            self.dep_comm = AllGatherReduceScatter(mapping)
        self.allreduce = (AllReduce(mapping=mapping,
                                    strategy=model_config.allreduce_strategy,
                                    dtype=dtype)
                          if (self.moe is None and self.ep_size > 1
                              and self.dep_comm is None) else None)

        # Fused SiTU MoE path selection is automatic: Blackwell
        # (SM100/SM103) uses the in-tree TRTLLM-Gen SiTU op (W4A8
        # MXFP4xMXFP8, torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner);
        # any other arch falls back to the Python per-expert dequant
        # reference. An explicit ``moe_config.backend="VANILLA"`` forces the
        # reference on any arch — the bit-parity oracle for tests. In the
        # fused mode the kernel-layout weights are prepared lazily on the
        # first forward (after load_weights has populated the bank); the
        # bank's packed buffers are freed afterwards to avoid holding the
        # expert slice twice.
        from ..modules.kimi_k3_moe._moe_kernels import get_moe_sm_version
        self.use_native_fused_moe = (
            str(model_config.moe_backend).upper() != "VANILLA"
            and get_moe_sm_version() in (100, 103))
        self._situ_alpha = float(situ_beta)
        self._situ_linear_beta = float(situ_linear_beta
                                       if situ_linear_beta is not None else 1.0)
        self._native_fused_weights = None

    def _build_configurable_moe(self, model_config: ModelConfig, cfg,
                                layer_idx: int, dtype: torch.dtype):
        """Host the native SiTU op under ``ConfigurableMoE`` (thin shell)."""
        import copy as _copy

        from ..modules.fused_moe.routing import DeepSeekV3MoeRoutingMethod
        from ..modules.kimi_k3_moe.configurable_moe_shell import \
            KimiK3ConfigurableMoE

        gate = self.gate
        assert (gate.moe_router_activation_func == "sigmoid"
                and gate.moe_renormalize and gate.num_expert_group == 1
                and gate.topk_group == 1), (
                    "DeepSeekV3MoeRoutingMethod mirrors the K3 gate only for "
                    "sigmoid scoring + renormalize + ungrouped top-k")
        # K3 is EP-only: the non-attention-DP mode reinterprets the TP width
        # as the EP width, so hand the wrapper a mapping whose moe_ep/moe_tp
        # split says the same thing. Attention modules keep the original.
        mapping = model_config.mapping
        moe_model_config = model_config
        if mapping.moe_ep_size != self.ep_size:
            moe_mapping = _copy.deepcopy(mapping)
            moe_mapping.moe_ep_size = self.ep_size
            moe_mapping.moe_tp_size = mapping.tp_size // self.ep_size
            moe_model_config = _copy.copy(model_config)
            moe_model_config._frozen = False
            moe_model_config.mapping = moe_mapping
            moe_model_config._frozen = True
        routing_method = DeepSeekV3MoeRoutingMethod(
            top_k=self.top_k,
            n_group=1,
            topk_group=1,
            routed_scaling_factor=cfg.routed_scaling_factor,
            callable_e_score_correction_bias=lambda:
            gate.e_score_correction_bias,
            is_fused=False,
        )
        moe = KimiK3ConfigurableMoE(
            ensure_weights_fn=self._ensure_native_fused_weights,
            get_weights_fn=lambda: self._native_fused_weights,
            routing_method=routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            dtype=dtype,
            reduce_results=True,
            model_config=moe_model_config,
            layer_idx=layer_idx,
        )
        assert (moe.slot_start == self.expert_lo
                and moe.expert_size_per_partition == self.experts_per_rank), (
                    f"wrapper expert partition [{moe.slot_start}, "
                    f"{moe.slot_start + moe.expert_size_per_partition}) "
                    f"diverges from the bank slice "
                    f"[{self.expert_lo}, {self.expert_hi})")
        return moe

    def forward(self, hidden_states: torch.Tensor,
                all_rank_num_tokens=None) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16."""
        identity = hidden_states
        if self.moe is not None:
            # ConfigurableMoE path: the scheduler owns routing (recomputed
            # per chunk from the fp32 gate logits), dispatch/combine (or the
            # non-DP EP allreduce) and chunking; the partial latent sums are
            # still completed BEFORE the nonlinear latent norm, as below.
            router_logits = self.gate.compute_logits(hidden_states)
            routed_in = self.routed_expert_down_proj(hidden_states)
            y = self.moe(routed_in, router_logits,
                         all_rank_num_tokens=all_rank_num_tokens,
                         use_dp_padding=False)
            y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)
            return y + self.shared_experts(identity)
        # Gate expects a 3D layout (HF contract).
        topk_idx, topk_weight = self.gate(hidden_states.unsqueeze(0))

        routed_in = self.routed_expert_down_proj(hidden_states)
        use_dep_comm = (self.dep_comm is not None
                        and all_rank_num_tokens is not None)
        if use_dep_comm:
            # DEP: exchange tokens in the latent space (half the payload of
            # hidden). Gate ids are GLOBAL expert ids, so gathered tokens go
            # through the same local-expert kernel unchanged.
            routed_in, _, topk_idx, topk_weight = self.dep_comm.dispatch(
                routed_in, None, topk_idx, topk_weight,
                all_rank_num_tokens, use_dp_padding=False)
        y = self._moe_infer_local(routed_in, topk_idx, topk_weight)
        # EP: each rank holds partial routed sums (its experts only) in the
        # latent space; sum them BEFORE the (nonlinear) latent norm.
        if use_dep_comm:
            # Varsize reduce-scatter: this rank gets its own tokens' completed
            # latent sums (same addends as the allreduce below).
            y = self.dep_comm.combine(y)
        elif self.allreduce is not None:
            y = self.allreduce(y)
        y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)
        # Shared experts are replicated: computed once per rank, added after
        # the allreduce so they are not double counted.
        return y + self.shared_experts(identity)

    def _ensure_native_fused_weights(self) -> None:
        """Lazily pack the bank's MXFP4 buffers for the in-tree SiTU op."""
        if self._native_fused_weights is not None:
            return
        from ..modules.kimi_k3_moe._moe_kernels import (
            assert_native_situ_supported, make_situ_alpha_beta,
            pack_routed_expert_weights)
        assert_native_situ_supported(
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.expert_bank.intermediate_size)
        bank = self.expert_bank
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        packed = pack_routed_expert_weights(
            w1_packed=bank.w1_packed,
            w1_scales=bank.w1_scales,
            w3_packed=bank.w3_packed,
            w3_scales=bank.w3_scales,
            w2_packed=bank.w2_packed,
            w2_scales=bank.w2_scales,
            device=device,
        )
        packed["gemm1_alpha"], packed["gemm1_beta"] = make_situ_alpha_beta(
            local_num_experts=self.experts_per_rank,
            situ_beta=self._situ_alpha,
            situ_linear_beta=self._situ_linear_beta,
            device=device,
        )
        self._native_fused_weights = packed
        logger.info(
            f"[KimiK3MoERuntime] layer {self.layer_idx}: prepared native "
            f"TRTLLM-Gen SiTU weights for {self.experts_per_rank} local "
            f"experts (offset {self.expert_lo})")
        # Free the (now redundant) bank copies of the expert slice.
        empty = torch.empty(0, dtype=torch.uint8,
                            device=bank.w1_packed.device)
        for name in ("w1_packed", "w1_scales", "w3_packed", "w3_scales",
                     "w2_packed", "w2_scales"):
            bank.register_buffer(name, empty.clone())

    def _moe_infer_local(self, x: torch.Tensor, topk_ids: torch.Tensor,
                         topk_weight: torch.Tensor) -> torch.Tensor:
        """HF ``KimiSparseMoeBlock.moe_infer`` restricted to local experts.

        Tokens routed to non-local experts contribute zeros; the cross-rank
        allreduce in :meth:`forward` completes the sum.
        """
        if self.use_native_fused_moe:
            if x.shape[0] == 0:
                return torch.zeros_like(x)
            from ..modules.kimi_k3_moe._moe_kernels import \
                invoke_native_situ_moe
            self._ensure_native_fused_weights()
            fw = self._native_fused_weights
            return invoke_native_situ_moe(
                hidden_states=x,
                topk_ids=topk_ids,
                topk_weights=topk_weight,
                gemm1_weights=fw["gemm1_weights"],
                gemm1_weights_scale=fw["gemm1_weights_scale"],
                gemm2_weights=fw["gemm2_weights"],
                gemm2_weights_scale=fw["gemm2_weights_scale"],
                gemm1_alpha=fw["gemm1_alpha"],
                gemm1_beta=fw["gemm1_beta"],
                num_experts=self.num_experts,
                top_k=self.top_k,
                valid_hidden_size=self.moe_hidden_size,
                valid_intermediate_size=self.expert_bank.intermediate_size,
                local_expert_offset=self.expert_lo,
                local_num_experts=self.experts_per_rank,
            )
        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.num_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().tolist()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        outs = torch.zeros_like(sorted_tokens)
        start = 0
        for i, n_tokens in enumerate(tokens_per_expert):
            end = start + int(n_tokens)
            if n_tokens and self.expert_lo <= i < self.expert_hi:
                outs[start:end] = self.expert_bank.forward_expert(
                    i - self.expert_lo, sorted_tokens[start:end])
            start = end

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (new_x.view(*topk_ids.shape, -1).type(
            topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(
                dim=1).type(new_x.dtype))
        return final_out


# ---------------------------------------------------------------------------
# KDA runtime (pool-backed prefill / decode via the FLA kernels).
# ---------------------------------------------------------------------------


def _kda_split_conv_sections(
        cs: torch.Tensor, d: int) -> Tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor]:
    """Split a gathered ``[N, 3D, W]`` conv-cache into contiguous q/k/v."""
    return (cs[:, :d].contiguous(), cs[:, d:2 * d].contiguous(),
            cs[:, 2 * d:].contiguous())


class KimiKDARuntime(nn.Module):
    """Wraps the parity-tested ``KimiKDALinearAttention`` parameters with a
    cache-pool-aware forward for the executor flow.

    Parameter names mirror the HF checkpoint 1:1 (the wrapped mixer is
    registered under the layer as ``self_attn``, so e.g.
    ``model.layers.N.self_attn.q_proj.weight`` maps identically).
    """

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        # Lazy import: pulls in fla/einops.
        from ..modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
        lin = cfg.linear_attn_config
        self.layer_idx = layer_idx
        self.mixer = KimiKDALinearAttention(
            hidden_size=cfg.hidden_size,
            num_heads=lin["num_heads"],
            head_dim=lin["head_dim"],
            conv_kernel_size=lin["short_conv_kernel_size"],
            use_full_rank_gate=lin.get("use_full_rank_gate", True),
            gate_lower_bound=lin.get("gate_lower_bound", None),
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            # Prefill defaults to the FLA path while two issues in the
            # in-tree trtllm::kda_prefill kernels are being resolved: an
            # illegal-address crash at the autotuner-warmup shape (partial
            # final chunk reads past exactly-sized buffers) and stalls at
            # the first eval-scale batches (root cause under
            # investigation). TLLM_KDA_ENABLE_OPT_PREFILL=1 opts back in
            # for kernel development.
            use_optimized_prefill=os.getenv("TLLM_KDA_ENABLE_OPT_PREFILL",
                                            "0") == "1",
            use_optimized_decode=True,
        )
        self.proj_size = lin["num_heads"] * lin["head_dim"]

    def forward(self, hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """``hidden_states``: flattened ``[num_tokens, hidden]`` (ctx tokens
        first, then one token per generation request)."""
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        num_decodes = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(
            self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32
        # index_copy_/index_select need int64 indices.
        state_indices = mamba_metadata.state_indices[:batch_size].long()

        outputs: List[torch.Tensor] = []
        if num_prefills > 0:
            outputs.append(
                self._forward_prefill(
                    hidden_states[:num_ctx_tokens],
                    mamba_metadata.query_start_loc_long[:num_prefills + 1],
                    mamba_metadata,
                    num_prefills,
                    conv_pool,
                    ssm_pool,
                    state_indices[:num_prefills],
                    layer_cache,
                ))
        if num_decodes > 0:
            decode_rows = hidden_states.shape[0] - num_ctx_tokens
            if decode_rows == num_decodes:
                outputs.append(
                    self._forward_decode(
                        hidden_states[num_ctx_tokens:],
                        conv_pool,
                        ssm_pool,
                        state_indices[num_prefills:],
                        layer_cache,
                    ))
            else:
                # Speculative verification: each generation request carries
                # 1 + draft_len tokens (drafts are padded to the static max,
                # so T is uniform). Per-step states go to the manager's
                # SpeculativeState scratch buffers — never the live pools —
                # and kv_cache_manager.update_mamba_states() promotes the
                # accepted step after sampling.
                assert decode_rows % num_decodes == 0, (
                    f"ragged generation batch: {decode_rows} tokens for "
                    f"{num_decodes} requests")
                outputs.append(
                    self._forward_verify(
                        hidden_states[num_ctx_tokens:],
                        decode_rows // num_decodes,
                        layer_cache,
                        conv_pool,
                        ssm_pool,
                        state_indices[num_prefills:],
                    ))
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    def _has_kda_replay_caches(self, layer_cache) -> bool:
        """True when the manager allocated the fused-verify replay caches."""
        return getattr(layer_cache, "kda_qkg_cache", None) is not None

    def _sync_kda_replay_conv_window(self, layer_cache, slot_indices,
                                     conv_q, conv_k, conv_v) -> None:
        """Seed the replay conv caches' committed window from FLA windows.

        The fused verify kernel keeps its own extended fp32 dim-contiguous
        conv caches; their committed window (columns ``[0, W-1)``) must hold
        the last ``W-1`` raw conv inputs whenever another path (prefill,
        plain decode) advances the base conv pool. The FLA window's oldest
        column drops out of every future convolution, so columns ``[1, W)``
        of the FLA cache map 1:1 onto the committed window.
        """
        if not self._has_kda_replay_caches(layer_cache):
            return
        w = self.mixer.conv_size
        for cache, window in ((layer_cache.kda_conv_q, conv_q),
                              (layer_cache.kda_conv_k, conv_k),
                              (layer_cache.kda_conv_v, conv_v)):
            cache[:, :, :w - 1].index_copy_(
                0, slot_indices, window[:, :, 1:].to(cache.dtype))

    def _forward_prefill(self, x2d, cu_seqlens, mamba_metadata, num_prefills,
                         conv_pool, ssm_pool, slot_indices,
                         layer_cache=None) -> torch.Tensor:
        from einops import rearrange

        mixer = self.mixer
        d = self.proj_size
        x = x2d.unsqueeze(0)  # [1, T, hidden]

        q_proj_states = mixer.q_proj(x)
        k_proj_states = mixer.k_proj(x)
        v_proj_states = mixer.v_proj(x)

        # Initial states: present for continuation chunks (chunked prefill)
        # and for prefix-cache hits (block reuse), where the previous
        # conv/recurrent state was onboarded into this request's slot.
        conv_q_in = conv_k_in = conv_v_in = None
        recurrent_in = None
        if mamba_metadata.use_initial_states:
            has_init = mamba_metadata.has_initial_states[:num_prefills]
            cs = conv_pool.index_select(0, slot_indices)
            cs[~has_init] = 0
            conv_q_in, conv_k_in, conv_v_in = _kda_split_conv_sections(cs, d)
            recurrent_in = ssm_pool.index_select(0, slot_indices)
            recurrent_in[~has_init] = 0

        q, conv_q = mixer.q_conv1d(q_proj_states,
                                   cache=conv_q_in,
                                   output_final_state=True,
                                   cu_seqlens=cu_seqlens)
        k, conv_k = mixer.k_conv1d(k_proj_states,
                                   cache=conv_k_in,
                                   output_final_state=True,
                                   cu_seqlens=cu_seqlens)
        v, conv_v = mixer.v_conv1d(v_proj_states,
                                   cache=conv_v_in,
                                   output_final_state=True,
                                   cu_seqlens=cu_seqlens)

        g = mixer.f_b_proj(mixer.f_a_proj(x))
        g = rearrange(g, "... (h d) -> ... h d", d=mixer.head_dim)
        beta = mixer.b_proj(x).float()

        q = rearrange(q, "... (h d) -> ... h d", d=mixer.head_k_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=mixer.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=mixer.head_dim)

        # Kernel dispatch (in-tree trtllm::kda_prefill or FLA chunk_kda).
        # Both paths exchange states in the pool's V-first [N, H, V, K]
        # layout, so recurrent_in / final_state map to ssm_pool 1:1.
        lower_bound = mixer.gate_lower_bound
        o, final_state = mixer.prefill_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=mixer.A_log,
            dt_bias=mixer.dt_bias,
            scale=mixer.head_k_dim**-0.5,
            initial_state=recurrent_in,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
        )

        # Persist per-request states into the pools.
        conv_pool.index_copy_(
            0, slot_indices,
            torch.cat([conv_q, conv_k, conv_v], dim=1).to(conv_pool.dtype))
        ssm_pool.index_copy_(0, slot_indices, final_state.to(ssm_pool.dtype))
        # Fused-verify replay caches: seed the committed conv window so the
        # first verify round convolves the correct history (pending drafts
        # are zero for a fresh request, so the tail columns are unused).
        self._sync_kda_replay_conv_window(layer_cache, slot_indices, conv_q,
                                          conv_k, conv_v)

        return self._output_gate_and_proj(x, o)

    def _forward_decode(self, x2d, conv_pool, ssm_pool, slot_indices,
                        layer_cache=None) -> torch.Tensor:
        from ..modules.kimi_kda.kimi_kda_mixer import KimiKDACachedState

        mixer = self.mixer
        d = self.proj_size
        x = x2d.unsqueeze(1)  # [B, 1, hidden]

        cs = conv_pool.index_select(0, slot_indices)
        conv_q, conv_k, conv_v = _kda_split_conv_sections(cs, d)
        cache = KimiKDACachedState(
            conv_state_q=conv_q,
            conv_state_k=conv_k,
            conv_state_v=conv_v,
            recurrent_state=ssm_pool.index_select(0, slot_indices),
        )
        out, new_cache = mixer.forward_decode(x, cache)

        conv_pool.index_copy_(
            0,
            slot_indices,
            torch.cat(
                [
                    new_cache.conv_state_q,
                    new_cache.conv_state_k,
                    new_cache.conv_state_v,
                ],
                dim=1,
            ).to(conv_pool.dtype),
        )
        ssm_pool.index_copy_(
            0, slot_indices, new_cache.recurrent_state.to(ssm_pool.dtype))
        # Fused-verify replay caches: keep the committed conv window in
        # sync with the plain-decode advance. NOTE: this path is only
        # correct for requests with no pending accepted drafts
        # (prev_num_accepted_tokens == 0); with drafts pending, the live
        # pools lag by the pending prefix and only the fused verify kernel
        # can advance them. The spec workers pad drafts to the static max,
        # so drafted batches always take the verify path.
        self._sync_kda_replay_conv_window(layer_cache, slot_indices,
                                          new_cache.conv_state_q,
                                          new_cache.conv_state_k,
                                          new_cache.conv_state_v)

        return out.squeeze(1)

    def _forward_verify(self, x2d, num_steps, layer_cache, conv_pool,
                        ssm_pool, slot_indices) -> torch.Tensor:
        """Speculative verification: advance each request ``num_steps``
        tokens (1 golden + ``num_steps - 1`` padded drafts).

        Two paths:

        * Fused (``trtllm::kda_mtp_decode``, when the manager allocated the
          KDA replay caches): one kernel launch replays the previous
          round's accepted drafts from the per-slot replay caches, then
          processes the new tokens, committing the recurrent state and conv
          windows **in place** after the golden token and caching the new
          drafts. ``update_mamba_states()`` afterwards only records the
          accepted count for the next round's replay.
        * Legacy (sequential per-step FLA): per-step states go to the
          manager's batch-row-indexed intermediate scratch buffers and
          ``update_mamba_states()`` promotes the accepted step's state
          after sampling.
        """
        if self._has_kda_replay_caches(layer_cache):
            assert self.mixer.verify_kernel_path == "optimized", (
                "KDA replay caches are allocated but the fused verify "
                "kernel is unavailable; the legacy intermediate buffers "
                "were not allocated so there is no fallback")
            return self._forward_verify_fused(x2d, num_steps, layer_cache,
                                              ssm_pool, slot_indices)
        return self._forward_verify_sequential(x2d, num_steps, layer_cache,
                                               conv_pool, ssm_pool,
                                               slot_indices)

    def _forward_verify_fused(self, x2d, num_steps, layer_cache, ssm_pool,
                              slot_indices) -> torch.Tensor:
        """Fused multi-token verify via ``trtllm::kda_mtp_decode``.

        Token layout: the kernel indexes each request's new tokens at
        ``cu_seqlens[n] + num_accepted[n] + i``. The runtime packs the
        ``num_steps`` new tokens per request contiguously, so we pass
        ``cu_seqlens[n] = n * num_steps - num_accepted[n]`` — the shift
        lands the kernel's reads/writes exactly on the packed rows. A
        negative entry for request 0 is fine: ``bos`` is only ever used
        additively with a token offset ``>= num_accepted``.
        """
        mixer = self.mixer
        num_decodes = x2d.shape[0] // num_steps
        num_spec = num_steps - 1
        H = mixer.num_heads
        K = mixer.head_k_dim
        x = x2d.view(num_decodes, num_steps, -1)  # [B, T, hidden]
        T_total = num_decodes * num_steps

        x_q = mixer.q_proj(x).view(1, T_total, H, K)
        x_k = mixer.k_proj(x).view(1, T_total, H, K)
        x_v = mixer.v_proj(x).view(1, T_total, H, mixer.head_dim)
        # Raw gate / beta: the kernel applies dt_bias, A_log, the
        # lower-bound sigmoid gate, and the beta sigmoid itself.
        g = mixer.f_b_proj(mixer.f_a_proj(x)).view(1, T_total, H, K)
        beta = mixer.b_proj(x).view(1, T_total, H)

        w_q, w_k, w_v = self._get_mtp_conv_weights()
        lower_bound = (mixer.gate_lower_bound_override
                       if mixer.gate_lower_bound_override is not None else
                       mixer.gate_lower_bound)

        pending = layer_cache.prev_num_accepted_tokens[slot_indices].to(
            torch.int32)  # accepted drafts of the previous round, per req
        cu_seqlens = torch.arange(0, (num_decodes + 1) * num_steps,
                                  num_steps,
                                  dtype=torch.int32,
                                  device=x2d.device)
        cu_seqlens[:num_decodes].sub_(pending)

        out = mixer._dispatch.mtp_verify(
            x_q=x_q,
            x_k=x_k,
            x_v=x_v,
            w_q=w_q,
            w_k=w_k,
            w_v=w_v,
            cs_q=layer_cache.kda_conv_q,
            cs_k=layer_cache.kda_conv_k,
            cs_v=layer_cache.kda_conv_v,
            g=g,
            beta=beta,
            # .detach(): the CuTe DSL DLPack bridge rejects grad-tracking
            # tensors.
            A_log=mixer.A_log.detach(),
            dt_bias=mixer.dt_bias.detach(),
            recurrent_state=ssm_pool,
            qkg_cache=layer_cache.kda_qkg_cache,
            v_cache=layer_cache.kda_v_cache,
            beta_cache=layer_cache.kda_beta_cache,
            ssm_state_indices=slot_indices.to(torch.int32),
            cu_seqlens=cu_seqlens,
            num_spec=num_spec,
            num_accepted_tokens=pending,
            lower_bound=lower_bound,
            scale=mixer.head_k_dim**-0.5,
        )
        o = out.view(num_decodes, num_steps, H, mixer.head_dim)
        return self._output_gate_and_proj(x, o)

    def _get_mtp_conv_weights(self):
        """fp32 ``[dim, W]`` conv weights for the fused verify kernel,
        computed once per runtime instance."""
        cached = getattr(self, "_mtp_conv_weights", None)
        if cached is None:
            mixer = self.mixer
            cached = tuple(
                conv.weight.detach().squeeze(1).float().contiguous()
                for conv in (mixer.q_conv1d, mixer.k_conv1d, mixer.v_conv1d))
            self._mtp_conv_weights = cached
        return cached

    def _forward_verify_sequential(self, x2d, num_steps, layer_cache,
                                   conv_pool, ssm_pool,
                                   slot_indices) -> torch.Tensor:
        """Sequential per-step FLA verification (legacy intermediate-buffer
        path). Live pools are read-only here; ``update_mamba_states()``
        commits the accepted step's state after sampling.
        """
        from einops import rearrange
        from fla.ops.kda import fused_recurrent_kda

        intermediate_conv = layer_cache.intermediate_conv_window
        intermediate_ssm = layer_cache.intermediate_ssm
        assert intermediate_conv is not None and intermediate_ssm is not None, (
            "speculative verification requires the cache manager's "
            "SpeculativeState (legacy intermediate-buffer path)")

        mixer = self.mixer
        d = self.proj_size
        num_decodes = x2d.shape[0] // num_steps
        x = x2d.view(num_decodes, num_steps, -1)  # [B, T, hidden]

        q_proj_states = mixer.q_proj(x)
        k_proj_states = mixer.k_proj(x)
        v_proj_states = mixer.v_proj(x)
        g = mixer.f_b_proj(mixer.f_a_proj(x))
        g = rearrange(g, "... (h d) -> ... h d", d=mixer.head_dim)
        beta = mixer.b_proj(x).float()

        # Gathered copies — mutated across steps, never written back to the
        # live pools.
        cs = conv_pool.index_select(0, slot_indices)
        conv_q, conv_k, conv_v = _kda_split_conv_sections(cs, d)
        state = ssm_pool.index_select(0, slot_indices)

        step_outputs: List[torch.Tensor] = []
        for t in range(num_steps):
            # ShortConvolution.step updates the (gathered) caches in place.
            q_t, conv_q = mixer.q_conv1d(q_proj_states[:, t:t + 1],
                                         cache=conv_q,
                                         output_final_state=True)
            k_t, conv_k = mixer.k_conv1d(k_proj_states[:, t:t + 1],
                                         cache=conv_k,
                                         output_final_state=True)
            v_t, conv_v = mixer.v_conv1d(v_proj_states[:, t:t + 1],
                                         cache=conv_v,
                                         output_final_state=True)

            q_t = rearrange(q_t, "... (h d) -> ... h d", d=mixer.head_k_dim)
            k_t = rearrange(k_t, "... (h d) -> ... h d", d=mixer.head_k_dim)
            v_t = rearrange(v_t, "... (h d) -> ... h d", d=mixer.head_dim)

            o_t, state = fused_recurrent_kda(
                q=q_t,
                k=k_t,
                v=v_t,
                g=g[:, t:t + 1],
                beta=beta[:, t:t + 1],
                A_log=mixer.A_log,
                dt_bias=mixer.dt_bias,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=mixer.gate_lower_bound,
                state_v_first=True,
            )
            step_outputs.append(o_t)

            # Batch-row indexed ([:num_decodes] prefix), matching
            # update_mamba_states()'s intermediate_state_indices.
            intermediate_conv[:num_decodes, t] = torch.cat(
                [conv_q, conv_k, conv_v], dim=1).to(intermediate_conv.dtype)
            intermediate_ssm[:num_decodes, t] = state.to(
                intermediate_ssm.dtype)

        o = torch.cat(step_outputs, dim=1)  # [B, T, H, V]
        return self._output_gate_and_proj(x, o)

    def _output_gate_and_proj(self, x: torch.Tensor,
                              o: torch.Tensor) -> torch.Tensor:
        from einops import rearrange
        mixer = self.mixer
        if mixer.use_full_rank_gate:
            g_out = mixer.g_proj(x)
        else:
            g_out = mixer.g_b_proj(mixer.g_a_proj(x))
        g_out = rearrange(g_out, "... (h d) -> ... h d", d=mixer.head_dim)
        o = mixer.o_norm(o, g_out)
        o = rearrange(o, "b t h d -> (b t) (h d)")
        return mixer.o_proj(o)


# ---------------------------------------------------------------------------
# MLA per-step runtime bundle.
# ---------------------------------------------------------------------------


@dataclass
class _MLAStepRuntime:
    """Per-forward MLA runtime inputs shared by all MLA layers."""
    ctx_rts: List["KimiK3MLARuntimeInputs"] = field(default_factory=list)
    ctx_lens: List[int] = field(default_factory=list)
    gen_rt: Optional["KimiK3MLARuntimeInputs"] = None


def _build_mla_step_runtime(
        attn_metadata: AttentionMetadata) -> _MLAStepRuntime:
    from ..modules.kimi_k3_mla import KimiK3MLARuntimeInputs

    rt = _MLAStepRuntime()
    num_contexts = attn_metadata.num_contexts
    batch_size = attn_metadata.seq_lens.shape[0]
    num_cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq

    if num_contexts > 0:
        seq_lens = attn_metadata.seq_lens[:num_contexts].tolist()
        for i in range(num_contexts):
            ctx_len = int(seq_lens[i])
            cached = int(num_cached[i])
            # Present this context request as a single "generation" request
            # with q_len = ctx_len: the MLA generation FMHA under a causal
            # mask with kv_len == cached + q_len reproduces causal-prefill
            # semantics exactly (see KimiK3MLAAttention.forward_prefill).
            md = TrtllmAttentionMetadata(
                seq_lens=torch.tensor([ctx_len], dtype=torch.int),
                request_ids=[attn_metadata.request_ids[i]],
                max_num_requests=1,
                max_num_sequences=1,
                num_contexts=0,
                prompt_lens=[cached + ctx_len],
                max_num_tokens=ctx_len,
                kv_cache_manager=attn_metadata.kv_cache_manager,
                kv_cache_params=KVCacheParams(
                    use_cache=True,
                    num_cached_tokens_per_seq=[cached],
                ),
                mapping=attn_metadata.mapping,
                # KDA layers use the outer metadata's mamba_metadata; skip
                # re-preparing it for the derived MLA-only metadata.
                mamba_metadata=False,
                enable_flash_mla=getattr(attn_metadata, "enable_flash_mla",
                                         False),
            )
            md.prepare()
            rt.ctx_rts.append(
                KimiK3MLARuntimeInputs(
                    metadata=md,
                    request_ids=[attn_metadata.request_ids[i]],
                    seq_lens=[ctx_len],
                    num_cached_tokens_per_seq=[cached],
                ))
            rt.ctx_lens.append(ctx_len)

    if batch_size - num_contexts > 0:
        rt.gen_rt = KimiK3MLARuntimeInputs(
            metadata=attn_metadata,
            request_ids=list(attn_metadata.request_ids[num_contexts:]),
            seq_lens=attn_metadata.seq_lens[num_contexts:].tolist(),
            num_cached_tokens_per_seq=list(num_cached[num_contexts:]),
        )
    return rt


class KimiMLARuntime(nn.Module):
    """Wraps ``KimiK3MLAAttention`` with the mixed-batch executor dispatch."""

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        import os

        from ..modules.kimi_k3_mla import KimiK3MLAAttention
        max_positions = min(
            cfg.max_position_embeddings,
            int(
                os.environ.get(_KIMI_K3_MLA_MAX_POSITIONS_ENV,
                               _KIMI_K3_MLA_MAX_POSITIONS_DEFAULT)))
        self.layer_idx = layer_idx
        # The trtllm-gen MLA generation kernels group query heads per CTA and
        # require numHeadsQ divisible by the group size (a power of two, up
        # to 128). K3's 96 query heads are unsupported, so pad to the next
        # power of two (128) with zero weights: padded heads produce exactly
        # zero output (their kv_b_proj "v_absorb" rows are zero), so the
        # numerics are unchanged at ~33% extra MLA-layer q compute.
        self.num_real_heads = cfg.num_attention_heads
        padded_heads = 1
        while padded_heads < self.num_real_heads:
            padded_heads *= 2
        self.num_padded_heads = padded_heads
        self.mixer = KimiK3MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=padded_heads,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            use_output_gate=cfg.mla_use_output_gate,
            max_position_embeddings=max_positions,
        )

    def forward(self, hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata,
                mla_rt: _MLAStepRuntime) -> torch.Tensor:
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        outputs: List[torch.Tensor] = []
        offset = 0
        for ctx_rt, ctx_len in zip(mla_rt.ctx_rts, mla_rt.ctx_lens):
            outputs.append(
                self.mixer.forward_prefill(
                    hidden_states[offset:offset + ctx_len], ctx_rt))
            offset += ctx_len
        assert offset == num_ctx_tokens, (
            f"MLA context split mismatch: {offset} != {num_ctx_tokens}")
        if hidden_states.shape[0] > num_ctx_tokens:
            outputs.append(
                self.mixer.forward_decode(hidden_states[num_ctx_tokens:],
                                          mla_rt.gen_rt))
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiLinearDecoderLayer(nn.Module):

    def __init__(self, model_config: ModelConfig, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        dtype = torch.bfloat16

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(
                f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            self.self_attn = KimiKDARuntime(cfg, layer_idx)
        else:
            self.self_attn = KimiMLARuntime(cfg, layer_idx)

        self.is_moe = (cfg.num_experts is not None
                       and layer_idx >= cfg.first_k_dense_replace
                       and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0)
        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(model_config, cfg,
                                                     layer_idx)
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta",
                                       None)
            self.mlp = KimiK3MLP(hidden_size=cfg.hidden_size,
                                 intermediate_size=cfg.intermediate_size,
                                 situ_beta=situ_beta,
                                 situ_linear_beta=situ_linear_beta,
                                 use_fused_activation=True,
                                 dtype=dtype)

        # Stock fused RMSNorm for the plain (whole-tensor) norms; numerics
        # are drop-in for KimiK3RMSNorm (fp32 variance, weight applied
        # after downcast, use_gemma=False).
        self.input_layernorm = RMSNorm(hidden_size=cfg.hidden_size,
                                       eps=cfg.rms_norm_eps,
                                       dtype=dtype)
        self.post_attention_layernorm = RMSNorm(hidden_size=cfg.hidden_size,
                                                eps=cfg.rms_norm_eps,
                                                dtype=dtype)

        # Attention residual scheme (always on for K3). The res norms stay
        # KimiK3RMSNorm: they are consumed field-wise (.weight/.eps) by
        # _apply_attn_res and the fused attn_res op, never called as
        # modules.
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, \
            "Kimi K3 runtime expects attn_res_block_size to be set"
        self.self_attention_res_norm = KimiK3RMSNorm(cfg.hidden_size,
                                                     eps=cfg.rms_norm_eps,
                                                     dtype=dtype)
        self.mlp_res_norm = KimiK3RMSNorm(cfg.hidden_size,
                                          eps=cfg.rms_norm_eps,
                                          dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size,
                                                 1,
                                                 bias=False,
                                                 dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size,
                                      1,
                                      bias=False,
                                      dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
        mla_rt: Optional[_MLAStepRuntime],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        Returns ``(prefix_sum, block_residual)``; the running prefix sum is
        the hidden state handed to the next layer.
        """
        prefix_sum = hidden_states

        if block_residual.shape[1] > 0:
            hidden_states = _apply_attn_res(prefix_sum, block_residual,
                                            self.self_attention_res_proj,
                                            self.self_attention_res_norm)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.unsqueeze(1)], dim=1)
            prefix_sum = None

        hidden_states = self.input_layernorm(hidden_states)
        if self.is_kda:
            hidden_states = self.self_attn(hidden_states, attn_metadata)
        else:
            hidden_states = self.self_attn(hidden_states, attn_metadata,
                                           mla_rt)

        if prefix_sum is not None:
            prefix_sum = prefix_sum + hidden_states
        else:
            prefix_sum = hidden_states

        hidden_states = _apply_attn_res(prefix_sum, block_residual,
                                        self.mlp_res_proj, self.mlp_res_norm)

        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            hidden_states = self.block_sparse_moe(
                hidden_states,
                getattr(attn_metadata, "all_rank_num_tokens", None))
        else:
            hidden_states = self.mlp(hidden_states)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, block_residual


# ---------------------------------------------------------------------------
# Model.
# ---------------------------------------------------------------------------


class KimiLinearModel(DecoderModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        self.embed_tokens = nn.Embedding(cfg.vocab_size,
                                         cfg.hidden_size,
                                         dtype=dtype)
        self.layers = nn.ModuleList([
            KimiLinearDecoderLayer(model_config, cfg, layer_idx)
            for layer_idx in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(hidden_size=cfg.hidden_size,
                            eps=cfg.rms_norm_eps,
                            dtype=dtype)

        # KimiK3RMSNorm (not RMSNorm): consumed field-wise (.weight/.eps)
        # by _apply_attn_res and the fused attn_res op.
        self.output_attn_res_norm = KimiK3RMSNorm(cfg.hidden_size,
                                                  eps=cfg.rms_norm_eps,
                                                  dtype=dtype)
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size,
                                              1,
                                              bias=False,
                                              dtype=dtype)

        self.has_mla = any(not layer.is_kda for layer in self.layers)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        num_tokens = attn_metadata.num_tokens
        assert hidden_states.shape[0] == num_tokens, (
            f"Kimi K3 does not support padded batches "
            f"(got {hidden_states.shape[0]} rows, metadata says {num_tokens} "
            "tokens); disable CUDA graphs and the overlap scheduler.")

        mla_rt = (_build_mla_step_runtime(attn_metadata)
                  if self.has_mla else None)

        block_residual = hidden_states.new_zeros(hidden_states.shape[0], 0,
                                                 hidden_states.shape[1])
        for layer in self.layers:
            hidden_states, block_residual = layer(hidden_states,
                                                  block_residual,
                                                  attn_metadata, mla_rt)

        hidden_states = _apply_attn_res(hidden_states, block_residual,
                                        self.output_attn_res_proj,
                                        self.output_attn_res_norm)
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Causal LM wrapper + weight loading.
# ---------------------------------------------------------------------------


def _materialize(value) -> torch.Tensor:
    """Materialize a (possibly lazy safetensors slice) weight value."""
    if isinstance(value, torch.Tensor):
        return value
    return value[:]


@register_auto_model("KimiK3ForConditionalGeneration")
@register_auto_model("KimiLinearForCausalLM")
class KimiLinearForCausalLM(SpecDecOneEngineForCausalLM[KimiLinearModel,
                                                        Any]):
    """Kimi K3 text model (the vision tower is ignored; text-only serving)."""

    def __init__(self, model_config: ModelConfig):
        cfg = _get_text_config(model_config.pretrained_config)
        assert model_config.mapping.pp_size == 1, \
            "Kimi K3 does not support pipeline parallelism"
        spec_config = getattr(model_config, "spec_config", None)
        # SA (suffix automaton) is the supported spec-dec mode: one-engine
        # in-forward drafting, no draft weights; the KDA/MLA verify paths
        # below implement multi-token verification for it. Modes needing
        # draft heads (MTP/Eagle) are blocked until a draft-head
        # checkpoint exists.
        assert spec_config is None or spec_config.spec_dec_mode.is_sa(), \
            "Kimi K3 supports speculative decoding only with SA"
        super().__init__(KimiLinearModel(model_config),
                         model_config,
                         hidden_size=cfg.hidden_size,
                         vocab_size=cfg.vocab_size)

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        # - enable_block_reuse defaults off: reuse is supported as an
        #   explicit opt-in (routes to CppMambaHybridCacheManager with
        #   per-block KDA state snapshots); the default stays on the
        #   Mixed manager, which SA speculative decoding requires.
        # - tokens_per_block=64: with 32, the flashinfer trtllm-gen FMHA lib
        #   rejects the MLA (576, 512) generation kernel (marked slower) and
        #   the fallback C++ path requires num_heads % 64 == 0, which K3's
        #   96 query heads violate.
        return {
            "kv_cache_config": {
                "enable_block_reuse": False,
                "tokens_per_block": 64,
            }
        }

    # ------------------------------------------------------------------
    # Weight loading (streams the 1.5TB checkpoint; only the rank-local
    # expert slice of each MoE layer is read).
    # ------------------------------------------------------------------

    def checkpoint_name_plan(self, prefix: str):
        """Return ``(name_map, expected_keys, bank_jobs)``.

        ``name_map`` maps every model parameter name to its checkpoint key
        (for fused ``gate_up_proj`` parameters the mapped key is virtual;
        the two real per-half keys come from ``_gate_up_ckpt_keys``);
        ``expected_keys`` additionally covers the rank-local per-expert MXFP4
        tensors; ``bank_jobs`` lists ``(layer_idx, moe_module, key_base)``
        for the expert-bank copies. Exposed separately so the weight-name
        mapping can be dry-run without touching any tensor data.
        """
        params = dict(self.named_parameters())
        expected_keys = set()
        name_map: Dict[str, str] = {}
        for name in params:
            if name == "lm_head.weight":
                ckpt_key = prefix + "lm_head.weight"
            else:
                # Runtime wrapper modules hold the parity-tested mixers as a
                # "mixer" submodule; the checkpoint names have no such scope.
                ckpt_key = prefix + name.replace(".self_attn.mixer.",
                                                 ".self_attn.")
            name_map[name] = ckpt_key
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Fused [gate | up] MLP layout (dense mlp / shared_experts):
                # the checkpoint stores two separate tensors.
                expected_keys.update(_gate_up_ckpt_keys(ckpt_key))
            else:
                expected_keys.add(ckpt_key)

        # Expert-bank buffers (per-expert checkpoint tensors, EP-sliced).
        bank_jobs = []
        for layer_idx, layer in enumerate(self.model.layers):
            if not getattr(layer, "is_moe", False):
                continue
            moe = layer.block_sparse_moe
            base = f"{prefix}model.layers.{layer_idx}.block_sparse_moe.experts"
            for local_idx in range(moe.experts_per_rank):
                expert_idx = moe.expert_lo + local_idx
                for w in ("w1", "w2", "w3"):
                    expected_keys.add(f"{base}.{expert_idx}.{w}.weight_packed")
                    expected_keys.add(f"{base}.{expert_idx}.{w}.weight_scale")
            bank_jobs.append((layer_idx, moe, base))
        return name_map, expected_keys, bank_jobs

    def load_weights(self, weights: Dict):
        from .modeling_utils import run_concurrently

        prefix = ("language_model."
                  if any(k.startswith("language_model.") for k in weights)
                  else "")

        params = dict(self.named_parameters())
        name_map, expected_keys, bank_jobs = self.checkpoint_name_plan(prefix)

        # ---- key-set validation (both directions) ----
        ckpt_keys = set(weights.keys())
        relevant_ckpt_keys = {
            k
            for k in ckpt_keys
            if not (k.startswith("vision_tower.")
                    or k.startswith("mm_projector."))
        }
        missing = sorted(expected_keys - ckpt_keys)
        if missing:
            raise KeyError(
                f"Kimi K3 load_weights: {len(missing)} expected checkpoint "
                f"keys are missing, e.g. {missing[:10]}")
        unexpected = relevant_ckpt_keys - expected_keys
        # Non-local experts and (in layer-truncated debug mode) extra layers
        # are expected leftovers.
        surprising = sorted(
            k for k in unexpected
            if ".block_sparse_moe.experts." not in k
            and not k.startswith(f"{prefix}model.layers."))
        if surprising:
            logger.warning(
                f"Kimi K3 load_weights: {len(surprising)} unmatched "
                f"checkpoint keys, e.g. {surprising[:10]}")

        device = next(self.parameters()).device

        def load_param(name: str, param: torch.nn.Parameter):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            if name.endswith(_GATE_UP_FUSED_SUFFIX):
                # Row-concat the checkpoint's separate gate_proj / up_proj
                # tensors into the fused [gate | up] parameter.
                gate_key, up_key = _gate_up_ckpt_keys(name_map[name])
                gate = _materialize(weights[gate_key])
                up = _materialize(weights[up_key])
                inter = param.shape[0] // 2
                if (gate.shape != (inter, param.shape[1])
                        or up.shape != gate.shape):
                    raise ValueError(
                        f"{name}: checkpoint gate/up shapes "
                        f"{tuple(gate.shape)} / {tuple(up.shape)} do not "
                        f"concat to param shape {tuple(param.shape)}")
                param.data[:inter].copy_(gate.to(param.dtype))
                param.data[inter:].copy_(up.to(param.dtype))
                return
            src = _materialize(weights[name_map[name]])
            if name == "lm_head.weight":
                # LMHead is vocab-sharded (TP column) + gathered; its
                # load_weights shards the full checkpoint tensor.
                self.lm_head.load_weights(weights=[{"weight": src}])
                return
            if name.endswith(".A_log") and src.numel() != param.numel():
                # The checkpoint pads A_log from [num_heads] to [head_dim]
                # (e.g. [96] -> [128]); the tail must be zeros.
                assert src.numel() > param.numel(), (name, src.shape)
                tail = src[param.numel():]
                if tail.abs().max().item() != 0.0:
                    raise ValueError(
                        f"{name}: expected zero padding beyond "
                        f"{param.numel()} entries, got nonzero tail")
                src = src[:param.numel()]
            if src.shape != param.shape:
                # MLA head padding (96 -> 128 query heads, see
                # KimiMLARuntime): pad the head-major output rows
                # (q_b_proj / kv_b_proj / g_proj) or the head-major input
                # columns (o_proj) with zeros. KDA layers' identically named
                # projections match exactly and never take this path.
                if (".self_attn." in name and name.endswith(
                    (".q_b_proj.weight", ".kv_b_proj.weight", ".g_proj.weight"))
                        and src.shape[1:] == param.shape[1:]
                        and src.shape[0] < param.shape[0]):
                    param.data.zero_()
                    param.data[:src.shape[0]].copy_(src.to(param.dtype))
                    return
                if (".self_attn." in name and name.endswith(".o_proj.weight")
                        and src.shape[0] == param.shape[0]
                        and src.shape[1] < param.shape[1]):
                    param.data.zero_()
                    param.data[:, :src.shape[1]].copy_(src.to(param.dtype))
                    return
                raise ValueError(f"{name}: checkpoint shape "
                                 f"{tuple(src.shape)} != param shape "
                                 f"{tuple(param.shape)}")
            param.data.copy_(src.to(param.dtype))

        def load_bank(layer_idx: int, moe: KimiK3MoERuntime, base: str):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            bank = moe.expert_bank
            for local_idx in range(moe.experts_per_rank):
                expert_idx = moe.expert_lo + local_idx
                for w, packed_buf, scales_buf in (
                    ("w1", bank.w1_packed, bank.w1_scales),
                    ("w2", bank.w2_packed, bank.w2_scales),
                    ("w3", bank.w3_packed, bank.w3_scales),
                ):
                    packed = _materialize(
                        weights[f"{base}.{expert_idx}.{w}.weight_packed"])
                    scales = _materialize(
                        weights[f"{base}.{expert_idx}.{w}.weight_scale"])
                    packed_buf[local_idx].copy_(packed)
                    scales_buf[local_idx].copy_(scales)

        param_jobs = [(name, param) for name, param in params.items()]
        run_concurrently(load_param, param_jobs, num_workers=8)

        # ---- expert banks: file-grouped streaming ----
        # The shared lazy ``weights`` dict keeps every shard mmapped for the
        # whole load, so pages it touches cannot be dropped until the load
        # ends (fadvise skips mapped pages). The expert slices are ~90 GB of
        # DISTINCT pages per rank — with 4 ranks/node that overruns the job
        # cgroup and OOM-kills the step (observed repeatedly on GB300
        # trays). Instead, group the rank-local expert tensors by shard file
        # and stream each file through a short-lived handle:
        # open -> copy -> close (unmap) -> fadvise(DONTNEED).
        ckpt_dir = getattr(self.model_config.pretrained_config,
                           "_name_or_path", None)
        index_path = os.path.join(ckpt_dir or "",
                                  "model.safetensors.index.json")
        if bank_jobs and os.path.isfile(index_path):
            import json as _json

            from safetensors import safe_open
            with open(index_path) as f:
                weight_map = _json.load(f)["weight_map"]
            per_file: Dict[str, list] = {}
            for layer_idx, moe, base in bank_jobs:
                bank = moe.expert_bank
                bufs = (("w1", bank.w1_packed, bank.w1_scales),
                        ("w2", bank.w2_packed, bank.w2_scales),
                        ("w3", bank.w3_packed, bank.w3_scales))
                for local_idx in range(moe.experts_per_rank):
                    expert_idx = moe.expert_lo + local_idx
                    for w, packed_buf, scales_buf in bufs:
                        for kind, buf in (("weight_packed", packed_buf),
                                          ("weight_scale", scales_buf)):
                            key = f"{base}.{expert_idx}.{w}.{kind}"
                            per_file.setdefault(weight_map[key], []).append(
                                (key, buf, local_idx))

            def load_bank_file(file_name: str, entries: list):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                path = os.path.join(ckpt_dir, file_name)
                with safe_open(path, framework="pt", device="cpu") as fh:
                    for key, buf, local_idx in entries:
                        buf[local_idx].copy_(fh.get_tensor(key))
                # Handle closed -> pages unmapped -> the drop takes effect.
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass

            run_concurrently(load_bank_file, sorted(per_file.items()),
                             num_workers=4)
        else:
            run_concurrently(load_bank, bank_jobs, num_workers=4)
        logger.info(
            f"Kimi K3: loaded {len(param_jobs)} parameters and the expert "
            f"slices of {len(bank_jobs)} MoE layers")
