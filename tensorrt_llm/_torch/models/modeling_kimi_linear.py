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

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import copy
import torch
from torch import nn

from ...logger import logger
from ...mapping import Mapping
from ...models.modeling_utils import QuantAlgo, QuantConfig
from ..attention_backend import AttentionMetadata, TrtllmAttentionMetadata
from ..distributed import AllReduce
from ..metadata import KVCacheParams
from ..model_config import ModelConfig
from ..modules.fused_moe import ConfigurableMoE, create_moe
from ..modules.kimi_k3_moe._mlp import KimiK3MLP, KimiK3RMSNorm
from ..modules.kimi_k3_moe.kimi_k3_moe_gate import KimiK3MoEGate
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..utils import ActType_TrtllmGen
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ..modules.kimi_k3_mla import KimiK3MLARuntimeInputs

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
    contract (dtype/shape/arch) so the caller can use the exact fp32 reference
    instead. ``block_residual`` is kept in the kernel-native ``[K, M, H]``
    layout. Candidate order matches the reference: snapshots first, the
    running prefix sum last.
    """
    if prefix_sum.dtype is not torch.bfloat16:
        return None
    M, H = prefix_sum.shape
    K = int(block_residual.shape[0])
    if K + 1 > 12 or M > 16384 or not (4096 <= H <= 8192 and H % 1024 == 0):
        return None
    try:
        attn_res_op = torch.ops.trtllm.attn_res_fwd
    except (AttributeError, RuntimeError):
        return None
    layer_kernel = prefix_sum.reshape(M, 1, H).contiguous()
    block_kernel = block_residual.reshape(K, M, 1, H).contiguous()
    output, _rsigma, _probs, _logits = attn_res_op(
        layer_kernel, block_kernel,
        proj.weight.reshape(-1).to(torch.bfloat16).contiguous(),
        norm.weight.to(torch.bfloat16).contiguous(), float(norm.eps))
    return output.reshape(M, H)


def _apply_attn_res(prefix_sum: torch.Tensor, block_residual: torch.Tensor,
                    proj: nn.Linear, norm: KimiK3RMSNorm) -> torch.Tensor:
    """Exact port of HF ``modeling_kimi._apply_attn_res`` (fp32 math).

    prefix_sum:     ``[num_tokens, hidden_size]``
    block_residual: ``[num_snapshots, num_tokens, hidden_size]``

    Unless ``KIMI_K3_FUSED_ATTN_RES=0``, inputs fitting the fused kernel's
    contract dispatch directly to the in-tree ``trtllm::attn_res_fwd`` op.
    Only the fallback boundary restores the HF ``[M, K, H]`` layout.
    """
    if _FUSED_ATTN_RES_ENABLED:
        fused = _apply_attn_res_fused(prefix_sum, block_residual, proj, norm)
        if fused is not None:
            return fused
    block_residual_hf = block_residual.transpose(0, 1)
    v = torch.cat((block_residual_hf, prefix_sum.unsqueeze(1)), dim=1)
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
# Latent MoE block using the unified ConfigurableMoE stack.
# ---------------------------------------------------------------------------


class KimiK3MoERuntime(nn.Module):
    """Kimi K3 latent MoE block backed by ConfigurableMoE/TRTLLM-Gen."""

    def __init__(self,
                 model_config: ModelConfig,
                 cfg,
                 layer_idx: int,
                 aux_stream: Optional[torch.cuda.Stream] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_experts = cfg.num_experts
        self.top_k = cfg.num_experts_per_token
        self.moe_hidden_size = cfg.routed_expert_hidden_size
        assert self.moe_hidden_size is not None, \
            "Kimi K3 runtime expects the latent MoE (routed_expert_hidden_size)"

        situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
        situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
        dtype = torch.bfloat16

        # Routing params stay fp32 (scores are computed in fp32).
        self.gate = KimiK3MoEGate(cfg)

        routed_moe_model_config = self._routed_moe_model_config(model_config)
        routed_quant_config = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
        self.routed_experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=cfg.moe_intermediate_size,
            dtype=dtype,
            reduce_results=True,
            model_config=routed_moe_model_config,
            override_quant_config=routed_quant_config,
            layer_idx=layer_idx,
            trtllm_gen_activation_type=ActType_TrtllmGen.SiTu,
            # Cubin alpha is the gate-side SiTU beta; cubin beta is the
            # linear-side SiTU beta.
            trtllm_gen_activation_alpha=float(situ_beta),
            trtllm_gen_activation_beta=float(
                situ_linear_beta if situ_linear_beta is not None else 1.0
            ),
            # Let CommunicationFactory select the best available strategy.
            communication_method=None,
        )
        if not isinstance(self.routed_experts, ConfigurableMoE):
            raise RuntimeError(
                "Kimi K3 requires ConfigurableMoE; ENABLE_CONFIGURABLE_MOE must not be disabled."
            )
        if self.routed_experts.layer_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "dynamic EPLB or replicated expert slots."
            )
        local_expert_ids = list(self.routed_experts.backend.initial_local_expert_ids)
        if local_expert_ids != list(
            range(local_expert_ids[0], local_expert_ids[0] + len(local_expert_ids))
        ):
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming currently requires a "
                "contiguous static expert partition."
            )
        self.local_expert_ids = tuple(local_expert_ids)
        self.experts_per_rank = len(local_expert_ids)
        self.expert_lo = local_expert_ids[0]
        self.expert_hi = self.expert_lo + self.experts_per_rank

        # Shared experts stay replicated (DeepSeek's attention-DP
        # semantics): ConfigurableMoE owns its own reduction, so there is
        # no existing collective for column-shard partial sums to ride —
        # the direct-path shared-expert TP (partials joining the MoE
        # combine RS / routed allreduce) needs a partial-carry hook in the
        # wrapper before it can be ported (follow-up). Instead their cost
        # is hidden by running them on the aux stream, overlapped with the
        # routed dispatch/expert/combine chain (see forward()).
        shared_intermediate = cfg.moe_intermediate_size * cfg.num_shared_experts
        self.shared_experts = KimiK3MLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=shared_intermediate,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            use_fused_activation=True,
            dtype=dtype,
        )
        # Side stream (+ fork/join events) for overlapping the replicated
        # shared-expert compute with the routed dispatch/expert/combine
        # chain; see forward(). Only engaged when multi-stream is active
        # (CUDA graphs on) and aux_stream is set, otherwise both run in
        # order on the default stream.
        self.aux_stream = aux_stream
        self.moe_main_event = torch.cuda.Event()
        self.moe_shared_event = torch.cuda.Event()
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

    @staticmethod
    def _routed_moe_model_config(model_config: ModelConfig) -> ModelConfig:
        """Build a private EP-only mapping without mutating the shared config."""
        if model_config.moe_load_balancer is not None:
            raise NotImplementedError(
                "Kimi K3 packed-checkpoint streaming does not yet support "
                "EPLB or replicated expert slots."
            )
        mapping = model_config.mapping
        if getattr(mapping, "_dwdp_size", 0) > 1:
            raise NotImplementedError("Kimi K3 packed-checkpoint streaming does not support DWDP.")

        mapping_dict = mapping.to_dict()
        mapping_dict["moe_cluster_size"] = 1
        mapping_dict["moe_tp_size"] = 1
        mapping_dict["moe_ep_size"] = mapping.tp_size
        routed_mapping = Mapping.from_dict(mapping_dict)

        routed_model_config = copy.copy(model_config)
        routed_model_config._frozen = False
        routed_model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        routed_model_config.mapping = routed_mapping
        routed_model_config.moe_backend = "TRTLLM"
        routed_model_config._frozen = True
        return routed_model_config

    def forward(self, hidden_states: torch.Tensor,
                all_rank_num_tokens=None) -> torch.Tensor:
        """``hidden_states``: ``[num_tokens, hidden_size]`` bf16."""
        identity = hidden_states
        router_logits = self.gate.compute_logits(hidden_states)

        def _routed_output():
            routed_in = self.routed_expert_down_proj(hidden_states)
            y = self.routed_experts(
                routed_in,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            # EP partial latent sums are completed by the wrapper's own
            # reduction BEFORE the (nonlinear) latent norm.
            y = self.routed_expert_norm(y)
            return self.routed_expert_up_proj(y)

        # Shared experts are replicated (computed once per rank) and depend
        # only on the block input, not on the routed dispatch/expert/combine
        # chain -- so run them on the aux stream to overlap with the serial
        # EP dispatch/combine collectives. Multi-stream engages only under
        # CUDA graphs; otherwise both run in order on the default stream
        # with an identical result. Added after the routed combine so the
        # replicated shared output is not double counted.
        routed_out, shared_out = maybe_execute_in_parallel(
            _routed_output,
            lambda: self.shared_experts(identity),
            self.moe_main_event,
            self.moe_shared_event,
            self.aux_stream,
            disable_on_compile=True)
        return routed_out + shared_out


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

    def __init__(self, cfg, layer_idx: int, mapping=None):
        super().__init__()
        # Lazy import: pulls in fla/einops.
        from ..modules.kimi_kda.kimi_kda_mixer import KimiKDALinearAttention
        lin = cfg.linear_attn_config
        self.layer_idx = layer_idx
        # Attention-family TP semantics (Qwen3-Next GatedDeltaNet pattern,
        # gdn_mixer.py): replicated under attention-DP — each rank runs
        # its own batch with the full head set — and head-sharded across
        # mapping.tp_size otherwise: every rank holds the same batch, runs
        # its 1/tp head slice, and the row-sharded o_proj partials are
        # all-reduced at the end of forward().
        if (mapping is not None and mapping.tp_size > 1
                and not mapping.enable_attention_dp):
            self._kda_tp_size = mapping.tp_size
        else:
            self._kda_tp_size = 1
        self._kda_tp_rank = (mapping.tp_rank
                             if self._kda_tp_size > 1 else 0)
        self._o_allreduce = (AllReduce(mapping=mapping,
                                       dtype=torch.bfloat16)
                             if self._kda_tp_size > 1 else None)
        num_heads = lin["num_heads"]
        assert num_heads % self._kda_tp_size == 0, (
            f"KDA num_heads {num_heads} not divisible by "
            f"tp_size {self._kda_tp_size}")
        self.mixer = KimiKDALinearAttention(
            hidden_size=cfg.hidden_size,
            num_heads=num_heads // self._kda_tp_size,
            head_dim=lin["head_dim"],
            conv_kernel_size=lin["short_conv_kernel_size"],
            use_full_rank_gate=lin.get("use_full_rank_gate", True),
            gate_lower_bound=lin.get("gate_lower_bound", None),
            rms_norm_eps=cfg.rms_norm_eps,
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            # Use TLLM_KDA_ENABLE_OPT_PREFILL=0 to opt out of the optimized
            # prefill kernel.
            use_optimized_prefill=os.getenv("TLLM_KDA_ENABLE_OPT_PREFILL",
                                            "1") == "1",
            use_optimized_decode=True,
        )
        self.proj_size = (num_heads // self._kda_tp_size) * lin["head_dim"]

    def forward(self, hidden_states: torch.Tensor,
                attn_metadata: AttentionMetadata) -> torch.Tensor:
        """``hidden_states``: flattened ``[num_tokens, hidden]`` (ctx tokens
        first, then one token per generation request)."""
        mamba_metadata = attn_metadata.mamba_metadata
        num_prefills = attn_metadata.num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        batch_size = attn_metadata.seq_lens.shape[0]
        # index_copy_/index_select need int64 indices.
        state_indices = mamba_metadata.state_indices[:batch_size].long()
        cu_seqlens = mamba_metadata.query_start_loc_long[:num_prefills + 1]
        num_decodes = batch_size - num_prefills

        layer_cache = attn_metadata.kv_cache_manager.mamba_layer_cache(
            self.layer_idx)
        conv_pool = layer_cache.conv  # [slots, 3D, W] bf16
        ssm_pool = layer_cache.temporal  # [slots, H, V, K] fp32

        outputs: List[torch.Tensor] = []
        if num_prefills > 0:
            outputs.append(
                self._forward_prefill(
                    hidden_states[:num_ctx_tokens],
                    cu_seqlens,
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
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        if self._o_allreduce is not None:
            # Head-sharded TP: every rank ran its head shard on the same
            # local batch; sum the row-sharded o_proj partials.
            out = self._o_allreduce(out)
        return out

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

    def __init__(self, cfg, layer_idx: int, mapping=None):
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
        # Attention-family TP semantics (DeepSeek MLA pattern, mla.py):
        # replicated under attention-DP, head-sharded otherwise. Pad
        # FIRST, then divide, so every rank gets a power-of-two head
        # count (128/16 = 8; sharding the real 96 would give 6/rank and
        # break the generation-FMHA per-CTA head grouping). q_b/kv_b/g
        # column-shard and o_proj row-shards by padded head range; ranks
        # holding only padded heads contribute exact zeros. The latent KV
        # cache and both *_a_proj down-projections stay replicated — with
        # a single latent KV head the TP ranks hold duplicated KV cache,
        # exactly like DeepSeek MLA under TP (attention-DP dedups it).
        self._mla_tp_size = (
            mapping.tp_size if (mapping is not None
                                and not mapping.enable_attention_dp
                                and mapping.tp_size > 1)
            else 1)
        self._mla_tp_rank = (mapping.tp_rank
                             if self._mla_tp_size > 1 else 0)
        assert padded_heads % self._mla_tp_size == 0, (
            f"padded MLA heads {padded_heads} not divisible by "
            f"tp_size {self._mla_tp_size}")
        self._o_allreduce = (AllReduce(mapping=mapping,
                                       dtype=torch.bfloat16)
                             if self._mla_tp_size > 1 else None)
        self.mixer = KimiK3MLAAttention(
            hidden_size=cfg.hidden_size,
            num_heads=padded_heads // self._mla_tp_size,
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
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        if self._o_allreduce is not None:
            # Head-sharded TP: sum the row-sharded o_proj partials across
            # the head-shard group.
            out = self._o_allreduce(out)
        return out


# ---------------------------------------------------------------------------
# Decoder layer.
# ---------------------------------------------------------------------------


class KimiLinearDecoderLayer(nn.Module):

    def __init__(self,
                 model_config: ModelConfig,
                 cfg,
                 layer_idx: int,
                 aux_stream: Optional[torch.cuda.Stream] = None):
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
            self.self_attn = KimiKDARuntime(cfg,
                                            layer_idx,
                                            mapping=model_config.mapping)
        else:
            self.self_attn = KimiMLARuntime(cfg,
                                            layer_idx,
                                            mapping=model_config.mapping)

        self.is_moe = (cfg.num_experts is not None
                       and layer_idx >= cfg.first_k_dense_replace
                       and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0)
        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(model_config, cfg,
                                                     layer_idx, aux_stream)
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta",
                                       None)
            # Dense-MLP TP semantics (DeepSeek _compute_mlp_tp_size
            # pattern): replicated under attention-DP — each rank runs
            # only its own tokens, so a weight shard would need an extra
            # gather/scatter — and sharded like the shared experts
            # otherwise (column gate_up, row down); the partial sums are
            # all-reduced right after the call in forward().
            self._mlp_tp_size = (
                model_config.mapping.tp_size
                if (not model_config.mapping.enable_attention_dp
                    and model_config.mapping.tp_size > 1
                    and cfg.intermediate_size % model_config.mapping.tp_size
                    == 0)
                else 1)
            self.mlp = KimiK3MLP(
                hidden_size=cfg.hidden_size,
                intermediate_size=cfg.intermediate_size // self._mlp_tp_size,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                use_fused_activation=True,
                dtype=dtype)
            self._mlp_allreduce = (AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
                dtype=dtype) if self._mlp_tp_size > 1 else None)

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

        Returns ``(prefix_sum, block_residual)`` with the snapshot stack in
        kernel-native ``[K, M, H]`` layout; the running prefix sum is the
        hidden state handed to the next layer.
        """
        prefix_sum = hidden_states

        if block_residual.shape[0] > 0:
            hidden_states = _apply_attn_res(prefix_sum, block_residual,
                                            self.self_attention_res_proj,
                                            self.self_attention_res_norm)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual = torch.cat(
                (block_residual, prefix_sum.unsqueeze(0)), dim=0)
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
            if getattr(self, "_mlp_allreduce", None) is not None:
                # TEP-sharded dense MLP: sum the row-parallel partials.
                hidden_states = self._mlp_allreduce(hidden_states)

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

        # One side stream shared across all layers, used by KimiK3MoERuntime
        # to overlap the replicated shared-expert compute with the routed EP
        # dispatch/combine collectives.
        self.aux_stream = torch.cuda.Stream()

        self.embed_tokens = nn.Embedding(cfg.vocab_size,
                                         cfg.hidden_size,
                                         dtype=dtype)
        self.layers = nn.ModuleList([
            KimiLinearDecoderLayer(model_config, cfg, layer_idx,
                                   self.aux_stream)
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

        block_residual = hidden_states.new_zeros(0, hidden_states.shape[0],
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
        """Return ``(name_map, expected_keys, expert_jobs)``.

        ``name_map`` maps every model parameter name to its checkpoint key
        (for fused ``gate_up_proj`` parameters the mapped key is virtual;
        the two real per-half keys come from ``_gate_up_ckpt_keys``);
        ``expected_keys`` additionally covers the rank-local per-expert MXFP4
        tensors; ``expert_jobs`` lists ``(layer_idx, moe_module, key_base)``
        for backend-owned expert slots. Exposed separately so the weight-name
        mapping can be dry-run without touching any tensor data.
        """
        params = dict(self.named_parameters())
        expected_keys = set()
        name_map: Dict[str, str] = {}
        for name in params:
            # ConfigurableMoE's backend owns already-packed runtime weights,
            # generated zero biases, and SiTU constants. They do not have
            # one-to-one checkpoint parameter names.
            if ".routed_experts.backend." in name:
                continue
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

        # Backend-owned expert slots (per-expert checkpoint tensors, EP-sliced).
        expert_jobs = []
        for layer_idx, layer in enumerate(self.model.layers):
            if not getattr(layer, "is_moe", False):
                continue
            moe = layer.block_sparse_moe
            base = f"{prefix}model.layers.{layer_idx}.block_sparse_moe.experts"
            for expert_idx in moe.local_expert_ids:
                for w in ("w1", "w2", "w3"):
                    expected_keys.add(f"{base}.{expert_idx}.{w}.weight_packed")
                    expected_keys.add(f"{base}.{expert_idx}.{w}.weight_scale")
            expert_jobs.append((layer_idx, moe, base))
        return name_map, expected_keys, expert_jobs

    def load_weights(self, weights: Dict):
        from .modeling_utils import run_concurrently

        prefix = ("language_model."
                  if any(k.startswith("language_model.") for k in weights)
                  else "")

        params = dict(self.named_parameters())
        name_map, expected_keys, expert_jobs = self.checkpoint_name_plan(prefix)

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

        # MLP TP shard index (used only when a param's checkpoint shape is a
        # tp_size multiple of the param shape — the dense L0 MLP with
        # attention-DP off; shapes match and no slicing runs otherwise).
        # Every mode that shards these fused-MLP tensors shards by tp_rank.
        shared_tp_rank = self.model_config.mapping.tp_rank
        # KDA head-shard (attention-DP off): rank r loads head rows/cols
        # [r*local : (r+1)*local] of every head-major KDA tensor.
        kda_tp_size, kda_tp_rank = 1, 0
        for layer in self.model.layers:
            if getattr(layer, "is_kda", False):
                kda_tp_size = layer.self_attn._kda_tp_size
                kda_tp_rank = layer.self_attn._kda_tp_rank
                break
        # MLA head-shard (attention-DP off): rank slice of the zero-padded
        # 128-head layout (pad before divide).
        mla_tp_size, mla_tp_rank = 1, 0
        for layer in self.model.layers:
            if not getattr(layer, "is_kda", True):
                mla_tp_size = layer.self_attn._mla_tp_size
                mla_tp_rank = layer.self_attn._mla_tp_rank
                break

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
                if gate.shape[0] != inter and gate.shape[0] % inter == 0:
                    # TP-sharded fused MLP (shared experts on the direct
                    # MoE path, dense MLP with attention-DP off): take
                    # this rank's MATCHING row block from each half so
                    # the SiTU gate/up pairs stay aligned. shared_tp_rank
                    # == tp_rank in every mode that shards these.
                    lo = shared_tp_rank * inter
                    gate = gate[lo:lo + inter]
                    up = up[lo:lo + inter]
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
                # (e.g. [96] -> [128]); the tail must be zeros. Under KDA
                # head-shard TP the param holds this rank's head range
                # instead of the full [num_heads].
                assert src.numel() > param.numel(), (name, src.shape)
                if kda_tp_size > 1:
                    lo = kda_tp_rank * param.numel()
                    src = src[lo:lo + param.numel()]
                else:
                    tail = src[param.numel():]
                    if tail.abs().max().item() != 0.0:
                        raise ValueError(
                            f"{name}: expected zero padding beyond "
                            f"{param.numel()} entries, got nonzero tail")
                    src = src[:param.numel()]
            if src.shape != param.shape:
                # KDA head-shard (attention-DP off): every mismatching KDA
                # tensor is head-major with the checkpoint exactly
                # kda_tp_size times larger on one axis — q/k/v/g/f_b
                # projections, b_proj, dt_bias, and the depthwise conv
                # weights on dim 0 (rows), o_proj on dim 1 (columns).
                # MLA layers never produce a x-tp_size ratio (their
                # mismatches are the padding branches below), so shape
                # ratios alone identify the KDA slices.
                if kda_tp_size > 1 and ".self_attn." in name:
                    if (src.shape[0] == param.shape[0] * kda_tp_size
                            and src.shape[1:] == param.shape[1:]):
                        s = param.shape[0]
                        lo = kda_tp_rank * s
                        param.data.copy_(src[lo:lo + s].to(param.dtype))
                        return
                    if (src.dim() == 2 and src.shape[0] == param.shape[0]
                            and src.shape[1] == param.shape[1] * kda_tp_size):
                        s = param.shape[1]
                        lo = kda_tp_rank * s
                        param.data.copy_(src[:, lo:lo + s].to(param.dtype))
                        return
                # MLA head-shard (attention-DP off): the checkpoint holds
                # the real 96 heads; the param holds this rank's slice of
                # the zero-PADDED 128-head layout (pad before divide, so
                # per-rank counts stay a power of two). Head-major output
                # rows for q_b/kv_b/g_proj, input columns for o_proj; a
                # rank whose padded range lies beyond the real heads gets
                # zeros (its v_absorb rows are zero -> exact zero output).
                # KDA layers never reach here: their identically named
                # g/o projections match the exact-ratio branch above.
                if mla_tp_size > 1 and ".self_attn." in name:
                    if (name.endswith((".q_b_proj.weight",
                                       ".kv_b_proj.weight",
                                       ".g_proj.weight"))
                            and src.shape[1:] == param.shape[1:]
                            and src.shape[0] < param.shape[0] * mla_tp_size):
                        s = param.shape[0]
                        lo = mla_tp_rank * s
                        param.data.zero_()
                        n = max(0, min(src.shape[0] - lo, s))
                        if n > 0:
                            param.data[:n].copy_(src[lo:lo + n].to(
                                param.dtype))
                        return
                    if (name.endswith(".o_proj.weight") and src.dim() == 2
                            and src.shape[0] == param.shape[0]
                            and src.shape[1] < param.shape[1] * mla_tp_size):
                        s = param.shape[1]
                        lo = mla_tp_rank * s
                        param.data.zero_()
                        n = max(0, min(src.shape[1] - lo, s))
                        if n > 0:
                            param.data[:, :n].copy_(src[:, lo:lo + n].to(
                                param.dtype))
                        return
                # Shared-expert TP (direct MoE path): the module holds a
                # 1/tp shard of the FFN dim — column shard for gate/up
                # (output rows), row shard for down (input columns).
                if ".shared_experts." in name or ".mlp." in name:
                    # Shared experts (direct MoE path) and the dense L0
                    # MLP (attention-DP off): the fused gate_up_proj is
                    # sliced in its dedicated branch above; here the
                    # unfused halves (if ever configured) and down_proj.
                    if (name.endswith((".gate_proj.weight",
                                       ".up_proj.weight"))
                            and src.shape[0] % param.shape[0] == 0
                            and src.shape[1:] == param.shape[1:]):
                        lo = shared_tp_rank * param.shape[0]
                        param.data.copy_(
                            src[lo:lo + param.shape[0]].to(param.dtype))
                        return
                    if (name.endswith(".down_proj.weight")
                            and src.shape[1] % param.shape[1] == 0
                            and src.shape[0] == param.shape[0]):
                        lo = shared_tp_rank * param.shape[1]
                        param.data.copy_(
                            src[:, lo:lo + param.shape[1]].to(param.dtype))
                        return
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

        def load_expert(
            moe: KimiK3MoERuntime, base: str, local_slot_id: int, expert_idx: int, get_tensor
        ):
            if device.type == "cuda":
                torch.cuda.set_device(device)
            backend = moe.routed_experts.backend
            backend.quant_method.load_packed_mxfp4_expert(
                backend,
                global_expert_id=expert_idx,
                local_slot_id=local_slot_id,
                w1_weight=get_tensor(f"{base}.{expert_idx}.w1.weight_packed"),
                w1_weight_scale=get_tensor(f"{base}.{expert_idx}.w1.weight_scale"),
                w2_weight=get_tensor(f"{base}.{expert_idx}.w2.weight_packed"),
                w2_weight_scale=get_tensor(f"{base}.{expert_idx}.w2.weight_scale"),
                w3_weight=get_tensor(f"{base}.{expert_idx}.w3.weight_packed"),
                w3_weight_scale=get_tensor(f"{base}.{expert_idx}.w3.weight_scale"),
            )

        def load_experts_from_weights(layer_idx: int, moe: KimiK3MoERuntime, base: str):
            del layer_idx
            for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
                load_expert(
                    moe,
                    base,
                    local_slot_id,
                    expert_idx,
                    lambda key: _materialize(weights[key]),
                )

        param_jobs = [(name, params[name]) for name in name_map]
        run_concurrently(load_param, param_jobs, num_workers=8)

        # ---- backend expert slots: file-grouped streaming ----
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
        if expert_jobs and os.path.isfile(index_path):
            import json as _json
            from contextlib import ExitStack

            from safetensors import safe_open
            with open(index_path) as f:
                weight_map = _json.load(f)["weight_map"]
            per_file: Dict[str, list] = {}
            split_file_jobs = []
            for layer_idx, moe, base in expert_jobs:
                del layer_idx
                for local_slot_id, expert_idx in enumerate(moe.local_expert_ids):
                    keys = [
                        f"{base}.{expert_idx}.{w}.{kind}"
                        for w in ("w1", "w2", "w3")
                        for kind in ("weight_packed", "weight_scale")
                    ]
                    files = {weight_map[key] for key in keys}
                    job = (moe, base, local_slot_id, expert_idx)
                    if len(files) == 1:
                        per_file.setdefault(files.pop(), []).append(job)
                    else:
                        split_file_jobs.append((job, files))

            def drop_file_pages(file_name: str):
                path = os.path.join(ckpt_dir, file_name)
                try:
                    fd = os.open(path, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    finally:
                        os.close(fd)
                except OSError:
                    pass

            def load_expert_file(file_name: str, jobs: list):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                path = os.path.join(ckpt_dir, file_name)
                with safe_open(path, framework="pt", device="cpu") as fh:
                    for moe, base, local_slot_id, expert_idx in jobs:
                        load_expert(moe, base, local_slot_id, expert_idx, fh.get_tensor)
                # Handle closed -> pages unmapped -> the drop takes effect.
                drop_file_pages(file_name)

            def load_split_file_expert(job, files):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                with ExitStack() as stack:
                    handles = {
                        file_name: stack.enter_context(
                            safe_open(
                                os.path.join(ckpt_dir, file_name), framework="pt", device="cpu"
                            )
                        )
                        for file_name in files
                    }

                    def get_tensor(key):
                        return handles[weight_map[key]].get_tensor(key)

                    load_expert(*job, get_tensor)
                for file_name in files:
                    drop_file_pages(file_name)

            run_concurrently(load_expert_file, sorted(per_file.items()),
                             num_workers=4)
            run_concurrently(load_split_file_expert, split_file_jobs, num_workers=4)
        else:
            run_concurrently(load_experts_from_weights, expert_jobs, num_workers=4)

        for _, moe, _ in expert_jobs:
            backend = moe.routed_experts.backend
            loaded_slots = getattr(backend, "_packed_mxfp4_loaded_slots", set())
            expected_slots = set(range(backend.expert_size_per_partition))
            if loaded_slots != expected_slots:
                missing_slots = sorted(expected_slots - loaded_slots)
                raise RuntimeError(
                    "Kimi K3 packed expert loading did not fill all backend "
                    f"slots; missing {missing_slots[:10]}."
                )
            backend._weights_transformed = False
        logger.info(
            f"Kimi K3: loaded {len(param_jobs)} parameters and the expert "
            f"slices of {len(expert_jobs)} MoE layers")
