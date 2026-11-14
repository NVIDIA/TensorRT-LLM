# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: Kimi K3 (MXFP4) / sm_100 / tp16 attention, routed experts moe_tp 4 x moe_ep 4.

Kimi K3's language model: 93 layers, hidden 7168. Every fourth layer (3, 7, ..., 91) is MLA attention with 96
query heads; the others are Kimi Delta Attention (KDA), a gated linear-attention recurrence with per-request state.
Layer 0's MLP is dense. Every other layer's MLP is a latent MoE: 896 routed experts, top-16, expert width 3584.
Attention residuals mix each sublayer's input from a bank of earlier outputs. The routed experts are MXFP4 (the
checkpoint's compressed-tensors default, read as W4A8_MXFP4_MXFP8); everything else is bf16, and so is the KV pool.

`tp16_moetp4ep4` is `tensor_parallel_size: 16` with `moe_tensor_parallel_size: 4` and
`moe_expert_parallel_size: 4`, no attention data parallelism, all 16 GPUs in one NVLink domain. Attention is
head-split (6 MLA query heads per rank), and every rank holds a quarter of the width of a quarter of the experts.

**Each step is classified once, on the host, from its shape** (`decode_step`, a `DecodeStep` or None), and the
classification decides which kernels each module runs:

* **small**: at most 8 tokens (`DECODE_MAX_TOKENS`, one token tile), context requests included. The token-count
  kernels take it: decode GEMVs, the MoE front and routed experts, the sandwiches, the embedding and residual
  epilogues.
* **decode**: a pure decode step of R <= 8 generation requests with the same T <= 8 tokens each (one token without
  speculation, 1 + 7 drafts with DSpark) and no context request, so at most 64 tokens. The request-aware kernels
  take it: MLA attention and its KV store, the KDA verify, the drafter's attention.
* **wide**: a decode step of more than one token tile (DSpark verify of several requests). It keeps the decode
  layout's MoE head and tail, on M-general ops.

Every other step (prefill, mixed steps, decode steps above those bounds) runs the **generic path**: this target's
text model (`KimiLinearModel` below: decoder layers, attention residuals, and the MLA / KDA / MoE modules of `mla.py`,
`kda.py` and `moe.py`), on catalog entries and this family's ops, with the operations and the order of the built-in
Kimi K3 text model. One weight differs: where the MoE decode path takes a layer, its latent norm's weight is folded
into the latent up projection (the same function, rounded differently).

The text model hands each step's classification to its attention modules, which run a **decode step** on the K3
decode kernels' ops:

* KDA (`K3DecodeKDA`): one token per request, the fused input projection and the plain decode in one
  `ssm/k3_kda_decode_attn` launch. Verify tokens (DSpark at an even verify width up to 8): the cache manager
  then keeps the last verify's draft records (per draft the row innovations, beta * k and the decay), and every
  verify of the layer, on any step, runs the kernels that write them and replay the accepted drafts from them:
  `ssm/k3_kda_attn` for one request of 8 tokens (the projection fused in), else `ssm/k3_kda_verify`.
* MLA (`K3DecodeMLA`): `attention/k3_mla_qkv` (the query path and the step's latent KV rows into the paged cache),
  then `attention/k3_mla_attn_vb_out` (the attention, v_b and the output gate in one launch).

The projections around them run on the decode GEMV sites of `decode_gemv.py` (the [W_a; W_g] and KDA verify-row
projections, `o_proj` on every classified step), as do the LM head, the embedding and layer 0's dense MLP. A
classified step's attention-residual epilogues (the selection and the RMSNorm after it) take the fused kernels up to
one token tile, 32 tokens on a wide decode step. On any step of at most 16 tokens but a wide one, the post-attention
all-reduce carries the residual update (`decode_comm.py`): at most 8 tokens on an attention decode branch, o_proj, the
all-reduce and the update are one `comm/k3_sandwich_oproj` kernel; otherwise the attention's unreduced o_proj output
goes through `comm/mnnvl_allreduce_attn_res`. The stock MNNVL all-reduces send one-shot up to 4 MiB, a wide decode
step's up to the stock 1 MiB. The state those kernels share (the KDA projection's Lamport buffers, the MLA attention
workspace, the decode GEMVs' state, the TP group's MNNVL and sandwich workspaces, the MoE path's) lives in typed
objects this target creates in `post_load_weights`, before any graph capture.

A MoE layer on a step of at most 8 tokens runs `decode_moe.py`: `moe/k3_moe_front` (this rank's head slice, its
all-gather, the routing, the MXFP8 latent and the shared experts' gate_up + SiTU in one kernel), `moe/k3_moe`, the
latent all-reduce (on a pushing step, `DecodeStep.latent_push`, `moe/k3_moe`'s push form plus `comm/k3_latent_reduce`),
then the row-parallel tail, which the next layer's pre-attention step (the final norm's, after the
last layer) runs with its all-reduce and residual update as one `comm/k3_sandwich_tail` kernel. A wide decode step's
MoE keeps the sharded head and the row-parallel tail, on `moe/k3_route_quant`, `moe/k3_moe` and M-general ops.

**What this target asserts rather than adapts**: SM 10.0; the topology above; the MXFP4 checkpoint's quantization
(W4A16_MXFP4 with no per-layer declarations, so the routed experts run the W4A8_MXFP4_MXFP8 default and the excluded
modules stay bf16); bf16 weights and a bf16 KV pool;
tokens_per_block 64 (the MLA generation kernels K3's 96 heads reach exist only at 64); the V2 hybrid KV / state
manager, which holds the KDA states, with block reuse off and fp32 recurrent states; an all-reduce strategy of AUTO or
MNNVL. The construction-time ones fail in `__init__`, the per-engine ones on the first forward, each naming the
setting. A layer the decode kernels do not take fails the weight load.

**Text only.** The checkpoint is the vision-language wrapper. This target builds and loads no vision tower (its
weights are a predicted non-load, `weights.py`), and a step carrying multimodal input raises.

**Speculative decoding** is built and run by `spec_shell.py`'s functions: SA, or DSpark with a
standalone drafter checkpoint. The DSpark drafter is this target's: `K3DSparkDrafter` for a GQA drafter,
`K3MLADSparkDrafter` (`drafter_mla.py`) for an MLA one. The GQA drafter runs a decode step's block on the drafter
entries (`attention/k3_drafter_attn_qknorm` and the drafter's decode GEMV sites). Where every attention all-reduce
runs over MNNVL, it also runs on the TP group's collective state: its context projection's `fc` is split by input
feature over the group, `hidden_norm` applied in the all-reduce of the partial products
(`comm/mnnvl_fusion_allreduce`), and a block's residual adds and RMSNorms run in its all-reduces
(`comm/k3_sandwich_plain` with the projection up to 8 rows, else `comm/mnnvl_fusion_allreduce`). The DSpark worker is
this target's (`spec_worker.py`): the speculative runtime's worker with the decode kernels `trtllm::k3_spec_accept`,
`k3_ctx_kv` and `k3_markov` behind its `k3_decode`.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    NoTarget,
    step_contract_enabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.attention.k3_mla_attn_workspace import (
    K3MlaAttnWorkspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
    K3AllReduce,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_buffers import (
    K3KdaBuffers,
)
from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
from tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata import KimiK3MambaMetadata
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

from . import decode_comm as _decode_comm
from . import decode_gemv as _decode_gemv
from . import decode_moe as _decode_moe
from . import spec_shell as _spec_shell
from . import spec_worker as _spec_worker
from . import weights as _weights
from .attn_res import (
    _AUX_ATTN_RES_STREAM_ENABLED,
    KIMI_K3_AUX_ATTN_RES_STREAM_ENV,
    KimiK3RMSNorm,
    _apply_attn_res,
    _apply_attn_res_add_and_rmsnorm,
    _apply_attn_res_and_rmsnorm,
    _attn_res_max_tokens,
)
from .drafter import K3DSparkDrafter
from .kda import K3DecodeKDA
from .mla import KimiMLARuntime, resolve_attention_quant_config
from .moe import KimiK3GatedMLP, KimiK3MoERuntime
from .norm import RMSNorm
from .step import DecodeStep, decode_step, latent_push

# The GPU architecture this target IS. Routing will not send another one here, but a direct instantiation could,
# and the certification is per arch.
_SM = (10, 0)

#: The engine surface the first forward checks, under TRTLLM_MODELING_V2_VALIDATE, before this target relies on it:
#: per object, the attributes read. A renamed field upstream then fails here, naming it.
REQUIRED_ENGINE_FIELDS = {
    "attn_metadata": (
        "num_contexts",
        "num_generations",
        "num_seqs",
        "num_tokens",
        "seq_lens",
        "tokens_per_block",
        "kv_cache_manager",
        "mamba_metadata",
    ),
    "kv_cache_manager": ("enable_block_reuse", "mamba_layer_cache"),
}

# The MLA generation kernels for K3's 96 query heads exist only at a 64-token page (the built-in model's own
# get_model_defaults sets it for the same reason).
_TOKENS_PER_BLOCK = 64

_LANG_PREFIX = "language_model."


def _get_text_config(pretrained_config: "PretrainedConfig"):
    """Return the Kimi text config, unwrapping a composite kimi_k3 config."""
    if getattr(pretrained_config, "model_type", None) == "kimi_k3" or (
        not hasattr(pretrained_config, "linear_attn_config")
        and hasattr(pretrained_config, "text_config")
    ):
        return pretrained_config.text_config
    return pretrained_config


def _is_kda_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["kda_layers"]


def _is_mla_layer(cfg, layer_idx: int) -> bool:
    return (layer_idx + 1) in cfg.linear_attn_config["full_attn_layers"]


class KimiLinearDecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        cfg,
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        dtype = torch.bfloat16

        self.is_kda = _is_kda_layer(cfg, layer_idx)
        is_mla = _is_mla_layer(cfg, layer_idx)
        if self.is_kda == is_mla:
            raise ValueError(f"Kimi K3 layer {layer_idx} must be exactly one of KDA/MLA")

        if self.is_kda:
            projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "o_proj")
            attention_config = copy.copy(model_config)
            attention_config._frozen = False
            attention_config.quant_config_dict = {
                name: resolve_attention_quant_config(model_config, layer_idx, name)
                for name in projection_names
            }
            attention_config._frozen = model_config._frozen
            self.linear_attn = K3DecodeKDA(
                cfg,
                layer_idx,
                mapping=model_config.mapping,
                allreduce_strategy=model_config.allreduce_strategy,
                aux_stream=aux_stream_dict[AuxStreamType.Attention],
                model_config=attention_config,
            )
        else:
            self.self_attn = KimiMLARuntime(
                cfg,
                layer_idx,
                model_config=model_config,
                aux_stream_dict=aux_stream_dict,
            )

        self.is_moe = (
            cfg.num_experts is not None
            and layer_idx >= cfg.first_k_dense_replace
            and layer_idx % getattr(cfg, "moe_layer_freq", 1) == 0
        )
        if self.is_moe:
            self.block_sparse_moe = KimiK3MoERuntime(model_config, cfg, layer_idx, aux_stream_dict)
        else:
            situ_beta = getattr(cfg, "activation_situ_beta", None) or 1.0
            situ_linear_beta = getattr(cfg, "activation_situ_linear_beta", None)
            mapping = model_config.mapping
            self.mlp_tp_size = math.gcd(cfg.intermediate_size, mapping.tp_size)
            # Over MNNVL (one NVLink domain across the nodes, where a cross-node all-reduce costs what a node's
            # does) the MLP stays split over the whole TP group, the per-rank shapes its decode GEMVs take
            # (decode_gemv.SITES); otherwise it stays within one node.
            spans_nodes = self._mnnvl_allreduce() is not None
            if self.mlp_tp_size > mapping.gpus_per_node and not spans_nodes:
                self.mlp_tp_size = math.gcd(self.mlp_tp_size, mapping.gpus_per_node)
            # The MLP's own TP group: mlp_tp_size ranks, the all-reduce within it (the remaining factor of the
            # attention TP group goes to pp_size, which only groups the ranks).
            groups = mapping.pp_size * mapping.tp_size // self.mlp_tp_size
            mlp_mapping = Mapping(
                world_size=self.mlp_tp_size * groups,
                rank=mapping.rank,
                gpus_per_node=mapping.gpus_per_node,
                tp_size=self.mlp_tp_size,
                pp_size=groups,
                enable_attention_dp=mapping.enable_attention_dp,
            )
            # K3's dense layer is bf16.
            self.mlp = KimiK3GatedMLP(
                cfg.hidden_size,
                cfg.intermediate_size,
                (situ_beta, situ_linear_beta),
                mlp_mapping,
                (
                    K3AllReduce(mlp_mapping, model_config.allreduce_strategy, dtype)
                    if self.mlp_tp_size > 1
                    else None
                ),
                layer_idx,
                dtype=dtype,
            )
            self._situ = (situ_beta, situ_linear_beta)
            # The decode GEMVs' state (decode_gemv.py), set by the target's cache_derived_state.
            self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

        # The fused RMSNorm (norm.py) for the plain (whole-tensor) norms; numerics are drop-in for KimiK3RMSNorm
        # (fp32 variance, weight applied after downcast).
        self.input_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )

        # Attention residual scheme (always on for K3). The res norms stay
        # KimiK3RMSNorm: they are consumed field-wise (.weight/.eps) by
        # _apply_attn_res and the fused attn_res op, never called as
        # modules.
        self.attn_res_block_size = cfg.attn_res_block_size
        assert self.attn_res_block_size is not None, (
            "Kimi K3 runtime expects attn_res_block_size to be set"
        )
        self.self_attention_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.mlp_res_norm = KimiK3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.self_attention_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.mlp_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        # The decode path's collectives (decode_comm.py), and whether their sandwich takes this layer's o_proj; set
        # by the target's post_load_weights.
        self.decode_comm: Optional[_decode_comm.K3DecodeComm] = None
        self.sandwich_oproj = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        step: DecodeStep,
        capture: Optional[Tuple[Any, int]] = None,
        prenormed: bool = False,
        pending_moe_partial: Optional[Union[torch.Tensor, _decode_comm.PendingTail]] = None,
        defer_moe_tail: bool = False,
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Any]]:
        """Port of HF ``KimiDecoderLayer._forward_attn_residual`` (per token).

        ``block_residual`` is a preallocated snapshot bank in kernel-native
        ``[K_max, M, H]`` layout. Returns the running prefix sum and the
        number of valid bank rows.

        ``capture`` is ``(spec_metadata, layer_id)`` and taps the DSpark aux
        stream for the layer BEFORE this one: the aggregated stream for layer j
        is by definition what its next consumer sees, so the mixture computed
        below already is it. Reading it here beats recomputing it, and is only
        possible because K3 asserts pp_size == 1 -- layer j+1 is always local.
        PP support would need a recompute at the rank boundary.

        ``step`` is the step's classification (``decode_step``), handed to the
        attention module.

        ``prenormed`` (layer 0 on a decode step): ``hidden_states`` already is
        this layer's input norm, and the layer's input, the step's embedding,
        already is in ``block_residual[0]`` (``K3DecodeGemvs.embed_norm``).

        The post-attention step of at most ``AR_ATTN_RES_MAX_TOKENS`` tokens
        runs the attention's all-reduce and the residual update in one
        collective once the target has built its ``decode_comm``: with o_proj
        (``K3DecodeComm.sandwich_oproj``) where the sandwich takes the layer,
        the step and the attention's decode branch, else on the attention's
        unreduced o_proj output (``K3DecodeComm.allreduce_attn_res``).

        ``pending_moe_partial`` (only where ``accepts_moe_partial`` held): the
        previous layer's MoE output, unreduced (``defer_moe_tail``); then
        ``hidden_states`` is the prefix sum without it, and this layer's
        pre-attention step reduces and adds it: a ``PendingTail`` in one
        ``K3DecodeComm.sandwich_tail`` call, a wide decode step's tensor by its
        all-reduce and the fused add + attn_res + RMSNorm. On a tapped layer
        the ``sandwich_tail`` call also stores the tap into the capture slot
        where the speculative metadata exposes it (``capture_view``); other
        tapped layers take the tap after the step.

        ``defer_moe_tail``: return ``(prefix_sum, num_snapshots, partial)``
        instead, ``partial`` this layer's MoE output unreduced, for the next
        consumer to reduce.
        """
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]
        attn_res_max_tokens = _attn_res_max_tokens(step)
        tail = (
            pending_moe_partial
            if isinstance(pending_moe_partial, _decode_comm.PendingTail)
            else None
        )
        # A snapshot layer whose pre-attention step is the sandwich tail has the kernel store the running prefix sum
        # straight into the bank row it snapshots.
        snapshot_row = None
        if tail is not None and self.layer_idx % self.attn_res_block_size == 0:
            snapshot_row = block_residual[num_snapshots]
        # A tapped layer whose pre-attention step is the sandwich tail has the kernel store the tap straight into the
        # layer's capture slot, where the speculative metadata exposes that slot as a view (``capture_view``).
        tap_view = None
        if tail is not None and capture is not None:
            tap_view = _spec_worker.capture_view(capture[0], capture[1], prefix_sum.shape[0])

        if prenormed:
            assert num_snapshots == 0 and self.layer_idx % self.attn_res_block_size == 0
            assert pending_moe_partial is None and capture is None
        elif tail is not None:
            hidden_states, prefix_sum = self.decode_comm.sandwich_tail(
                tail,
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                updated_out=snapshot_row,
                tap=tap_view,
                tap_updated=not _AUX_ATTN_RES_STREAM_ENABLED,
            )
        elif pending_moe_partial is not None:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                _decode_comm.wide_all_reduce(self._o_allreduce(), pending_moe_partial),
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                attn_res_max_tokens,
            )
        elif capture is not None:
            # The mixture tap needs the PRE-norm value, which the fused
            # attn-res + RMSNorm kernel does not expose. Keep the two steps
            # split on captured layers only and fuse everywhere else.
            if num_snapshots > 0:
                hidden_states = _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            # A property of the DRAFTER checkpoint, not a knob: a mismatch only lowers
            # acceptance, silently. hidden_states is the pre-norm attn_res mixture;
            # prefix_only wants the running prefix, already in hand as prefix_sum.
            tapped = hidden_states if _AUX_ATTN_RES_STREAM_ENABLED else prefix_sum
            capture[0].maybe_capture_hidden_states(capture[1], tapped, None)
            hidden_states = self.input_layernorm(hidden_states)
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
                attn_res_max_tokens,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if capture is not None and pending_moe_partial is not None and tap_view is None:
            # The tapped layer handed its MoE output on and no kernel wrote the tap: the step above reduced the output
            # into prefix_sum. Tap that value's pre-norm attn_res mixture, what the split path captures.
            tapped = (
                _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
                if _AUX_ATTN_RES_STREAM_ENABLED
                else prefix_sum
            )
            capture[0].maybe_capture_hidden_states(capture[1], tapped, None)

        if self.layer_idx % self.attn_res_block_size == 0:
            if not prenormed and snapshot_row is None:
                block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None
        attention = self.linear_attn if self.is_kda else self.self_attn
        comm = self.decode_comm
        if comm is not None and comm.takes_post_attention(hidden_states, step):
            if (
                self.sandwich_oproj
                and step.num_tokens <= _decode_comm.SANDWICH_MAX_TOKENS
                and attention.will_run_decode_branch(attn_metadata, step)
            ):
                core = attention(hidden_states, attn_metadata, step=step, project_output=False)
                hidden_states, prefix_sum = comm.sandwich_oproj(
                    core,
                    self._o_proj().weight,
                    prefix_sum,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                )
            else:
                partial = attention(hidden_states, attn_metadata, step=step, reduce_output=False)
                hidden_states, prefix_sum = comm.allreduce_attn_res(
                    partial,
                    prefix_sum,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                )
        else:
            if comm is not None and step.wide:
                partial = attention(hidden_states, attn_metadata, step=step, reduce_output=False)
                attention_output = _decode_comm.wide_all_reduce(self._o_allreduce(), partial)
            else:
                attention_output = attention(hidden_states, attn_metadata, step=step)
            if prefix_sum is None:
                prefix_sum = attention_output
                hidden_states = _apply_attn_res_and_rmsnorm(
                    prefix_sum,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                    attn_res_max_tokens,
                )
            else:
                prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                    prefix_sum,
                    attention_output,
                    valid_block_residual,
                    self.mlp_res_proj,
                    self.mlp_res_norm,
                    self.post_attention_layernorm,
                    attn_res_max_tokens,
                )
        all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
        if self.is_moe and defer_moe_tail:
            partial = self.block_sparse_moe(
                hidden_states, all_rank_num_tokens, partial_tail=True, step=step
            )
            return prefix_sum, num_snapshots, partial
        if self.is_moe:
            hidden_states = self.block_sparse_moe(hidden_states, all_rank_num_tokens, step=step)
        else:
            hidden_states = self._dense_mlp(hidden_states, step)

        prefix_sum = prefix_sum + hidden_states
        return prefix_sum, num_snapshots

    def forward_generic(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        attn_metadata: AttentionMetadata,
        capture: Optional[Tuple[Any, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        prefix_sum = hidden_states
        valid_block_residual = block_residual[:num_snapshots]
        if capture is not None:
            if num_snapshots > 0:
                hidden_states = _apply_attn_res(
                    prefix_sum,
                    valid_block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            tapped = hidden_states if _AUX_ATTN_RES_STREAM_ENABLED else prefix_sum
            capture[0].maybe_capture_hidden_states(capture[1], tapped, None)
            hidden_states = self.input_layernorm(hidden_states)
        elif num_snapshots > 0:
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.input_layernorm,
            )
        else:
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_idx % self.attn_res_block_size == 0:
            block_residual[num_snapshots].copy_(prefix_sum)
            num_snapshots += 1
            valid_block_residual = block_residual[:num_snapshots]
            prefix_sum = None
        attention = self.linear_attn if self.is_kda else self.self_attn
        comm = self.decode_comm
        if comm is not None and comm.takes_post_attention(hidden_states, None):
            partial = attention.forward_generic(hidden_states, attn_metadata, reduce_output=False)
            hidden_states, prefix_sum = comm.allreduce_attn_res(
                partial,
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        elif prefix_sum is None:
            prefix_sum = attention.forward_generic(hidden_states, attn_metadata)
            hidden_states = _apply_attn_res_and_rmsnorm(
                prefix_sum,
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        else:
            prefix_sum, hidden_states = _apply_attn_res_add_and_rmsnorm(
                prefix_sum,
                attention.forward_generic(hidden_states, attn_metadata),
                valid_block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                self.post_attention_layernorm,
            )
        if self.is_moe:
            all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
            hidden_states = self.block_sparse_moe.forward_generic(
                hidden_states, all_rank_num_tokens
            )
        else:
            hidden_states = self.mlp(hidden_states)
        return prefix_sum + hidden_states, num_snapshots

    def _dense_mlp(self, hidden_states: torch.Tensor, step: DecodeStep) -> torch.Tensor:
        """The dense MLP: on a step of at most DECODE_MAX_TOKENS tokens, its GEMVs and activation on the decode
        kernels (``K3DecodeGemvs.dense_mlp``), then the down projection's all-reduce; else the module."""
        gemvs = self.decode_gemvs
        if gemvs is not None and step.small:
            out = gemvs.dense_mlp(
                hidden_states,
                self.mlp.gate_up_proj.weight,
                self.mlp.down_proj.weight,
                *self._situ,
            )
            if out is not None:
                return self.mlp.down_proj.all_reduce(out) if self.mlp_tp_size > 1 else out
        return self.mlp(hidden_states)

    def _mnnvl_allreduce(self):
        """The MNNVL all-reduce of this layer's attention output, or None."""
        attention = self.linear_attn if self.is_kda else self.self_attn
        return getattr(getattr(attention, "_o_allreduce", None), "mnnvl_allreduce", None)

    def _o_proj(self) -> nn.Module:
        """This layer's attention output projection (row parallel)."""
        return self.linear_attn.o_proj if self.is_kda else self.self_attn.mixer.o_proj

    def _o_allreduce(self) -> K3AllReduce:
        """The all-reduce module of this layer's attention output; it also reduces a wide step's MoE partial."""
        return (self.linear_attn if self.is_kda else self.self_attn)._o_allreduce

    def accepts_moe_partial(self, num_snapshots: int) -> bool:
        """Whether this layer's pre-attention step can reduce the previous layer's unreduced MoE output: the target
        built its collective state, and the snapshot bank is not empty."""
        return num_snapshots > 0 and self.decode_comm is not None

    def skip_forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """No-op stand-in for ``forward``, matching ``DecoderLayer.skip_forward``.

        ``modeling_utils.skip_forward()`` only drops a module's weights when it
        finds this attribute, so without it the layer-wise benchmarks would
        allocate all 93 layers instead of the profiled slice.
        """
        return hidden_states, block_residual


class KimiLinearModel(DecoderModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        cfg = _get_text_config(model_config.pretrained_config)
        self._text_cfg = cfg
        dtype = torch.bfloat16

        # Attention and MoE phases are sequential, so their branch-overlap
        # roles share one stream; MoE-internal overlap roles remain separate.
        aux_stream_list = [torch.cuda.Stream() for _ in range(4)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
            AuxStreamType.MoeOutputMemset: aux_stream_list[3],
        }

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(model_config, cfg, layer_idx, self.aux_stream_dict)
                for layer_idx in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size=cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)

        # KimiK3RMSNorm (not RMSNorm): consumed field-wise (.weight/.eps)
        # by _apply_attn_res and the fused attn_res op.
        self.output_attn_res_norm = KimiK3RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype
        )
        self.output_attn_res_proj = nn.Linear(cfg.hidden_size, 1, bias=False, dtype=dtype)
        self.num_attn_res_snapshots = (
            cfg.num_hidden_layers + cfg.attn_res_block_size - 1
        ) // cfg.attn_res_block_size
        # The decode path's GEMVs and embedding (decode_gemv.py), built by the target's cache_derived_state once the
        # weights are final. None: every step embeds on the generic path.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None

        # Which convention the drafter tap is on is not recoverable from the
        # served output -- a mismatch only lowers acceptance -- so state it once
        # at construction rather than leaving it to be inferred from an AL.
        logger.info_once(
            "Kimi K3 aux hidden capture: mode="
            f"{'attn_res_stream' if _AUX_ATTN_RES_STREAM_ENABLED else 'prefix_only'} "
            f"({KIMI_K3_AUX_ATTN_RES_STREAM_ENV}={int(_AUX_ATTN_RES_STREAM_ENABLED)})",
            key="kimi_k3_aux_capture_mode",
        )

    @property
    def kda_draft_records(self) -> bool:
        """Whether the hybrid cache manager keeps the last verify's draft records (``kda_state_tok``), from which
        ``ssm/k3_kda_verify`` and ``ssm/k3_kda_attn`` replay the accepted drafts. The engine reads it once the weights
        are loaded, to build the manager: DSpark drafts of an even verify width up to 8, every KDA layer
        taking the K3 kernels. Otherwise the KDA verify replays the accepted drafts from the replay caches
        (`ssm/kda_mtp_decode`), or runs the recurrence token by token."""
        spec_config = getattr(self.model_config, "spec_config", None)
        return bool(
            spec_config is not None
            and spec_config.spec_dec_mode.is_dspark()
            and spec_config.tokens_per_gen_step in (2, 4, 6, 8)
            and all(layer.linear_attn.takes_k3_kernels for layer in self.layers if layer.is_kda)
        )

    def _latent_push(self, attn_metadata: AttentionMetadata, step: DecodeStep) -> bool:
        """Whether the MoE layers push on ``step`` (``latent_push``): read only while a CUDA graph is being captured,
        from what the graph's key fixes (the step's shape, the attention metadata the decode kernels read) and from
        load-time state, so every rank and every replay of the graph decides alike."""
        if not torch.cuda.is_current_stream_capturing():
            return False
        kda = [layer.linear_attn for layer in self.layers if layer.is_kda]
        mla = [layer.self_attn for layer in self.layers if not layer.is_kda]
        return latent_push(
            step,
            capturing=True,
            breakable=is_in_breakable_cuda_graph(),
            kda_draft_records=self.kda_draft_records,
            kda_decode_kernels=all(m.takes_k3_kernels and m.k3_buffers is not None for m in kda),
            mla_decode_branch=all(m.will_run_decode_branch(attn_metadata, step) for m in mla[:1]),
        )

    def _defer_moe_tail(
        self,
        i: int,
        hidden_states: torch.Tensor,
        num_snapshots: int,
        spec_metadata,
        capture_set,
        step: DecodeStep,
    ) -> bool:
        """Whether layer ``i`` hands its MoE output on unreduced (the row-parallel tail, ``decode_moe.py``): its MoE
        takes the step with that tail, and its consumer, the next layer's pre-attention step or the final norm after
        the last layer, accepts it. A layer DSpark taps defers too: the next layer's step reduces the output, then taps
        its pre-norm mixture. An unknown capture set (every layer tapped) and a tapped last layer keep the replicated
        tail."""
        layer = self.layers[i]
        if not (layer.is_moe and layer.block_sparse_moe.tail_rp_eligible(hidden_states, step)):
            return False
        if spec_metadata is not None and (
            capture_set is None or (layer.layer_idx in capture_set and i == len(self.layers) - 1)
        ):
            return False
        return self.layers[min(i + 1, len(self.layers) - 1)].accepts_moe_partial(num_snapshots)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata=None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        num_tokens = (input_ids if inputs_embeds is None else inputs_embeds).shape[0]
        step = decode_step(attn_metadata, num_tokens)
        if step is None:
            return self._generic_forward(attn_metadata, input_ids, inputs_embeds, spec_metadata)
        if self._latent_push(attn_metadata, step):
            step = replace(step, latent_push=True)
        # A decode step embeds and norms for layer 0 in one launch, the embedding written as layer 0's first snapshot.
        prenormed = None
        if (
            inputs_embeds is None
            and self.decode_gemvs is not None
            and len(self.layers) > 0
            and self.num_attn_res_snapshots > 0
        ):
            table = self.embed_tokens.weight
            block_residual = table.new_empty(
                self.num_attn_res_snapshots, num_tokens, table.shape[1]
            )
            prenormed = self.decode_gemvs.embed_norm(
                input_ids, table, self.layers[0].input_layernorm, block_residual
            )
        if prenormed is not None:
            hidden_states = prenormed
        else:
            hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            block_residual = hidden_states.new_empty(
                self.num_attn_res_snapshots,
                hidden_states.shape[0],
                hidden_states.shape[1],
            )
        num_snapshots = 0
        capture_set = getattr(spec_metadata, "_capture_layer_set", None)
        # A MoE layer's output handed on unreduced, which the next layer's pre-attention step (or the final norm)
        # reduces: a decode_comm.PendingTail, or a wide decode step's tensor.
        pending_moe_partial = None
        for i, layer in enumerate(self.layers):
            defer_moe_tail = self._defer_moe_tail(
                i, hidden_states, num_snapshots, spec_metadata, capture_set, step
            )
            outputs = layer(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                step=step,
                capture=self._capture(spec_metadata, capture_set, i),
                prenormed=i == 0 and prenormed is not None,
                pending_moe_partial=pending_moe_partial,
                defer_moe_tail=defer_moe_tail,
            )
            if defer_moe_tail:
                hidden_states, num_snapshots, pending_moe_partial = outputs
            else:
                (hidden_states, num_snapshots), pending_moe_partial = outputs, None

        if isinstance(pending_moe_partial, _decode_comm.PendingTail):
            normed, _ = self.layers[-1].decode_comm.sandwich_tail(
                pending_moe_partial,
                hidden_states,
                block_residual[:num_snapshots],
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                self.norm,
            )
            return normed
        if pending_moe_partial is not None:
            _, normed = _apply_attn_res_add_and_rmsnorm(
                hidden_states,
                _decode_comm.wide_all_reduce(self.layers[-1]._o_allreduce(), pending_moe_partial),
                block_residual[:num_snapshots],
                self.output_attn_res_proj,
                self.output_attn_res_norm,
                self.norm,
                _attn_res_max_tokens(step),
            )
            return normed
        return self._final_norm(
            hidden_states,
            block_residual,
            num_snapshots,
            spec_metadata,
            capture_set,
            _attn_res_max_tokens(step),
        )

    def _generic_forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor],
        inputs_embeds: Optional[torch.FloatTensor],
        spec_metadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        block_residual = hidden_states.new_empty(
            self.num_attn_res_snapshots,
            hidden_states.shape[0],
            hidden_states.shape[1],
        )
        num_snapshots = 0
        capture_set = getattr(spec_metadata, "_capture_layer_set", None)
        for i, layer in enumerate(self.layers):
            hidden_states, num_snapshots = layer.forward_generic(
                hidden_states,
                block_residual,
                num_snapshots,
                attn_metadata,
                capture=self._capture(spec_metadata, capture_set, i),
            )
        return self._final_norm(
            hidden_states, block_residual, num_snapshots, spec_metadata, capture_set
        )

    def _capture(self, spec_metadata, capture_set, i: int) -> Optional[Tuple[Any, int]]:
        # DSpark hidden-state capture. The drafter is distilled on
        # the aggregated stream value -- the pre-norm softmax mixture its
        # next consumer sees -- not on the raw prefix sum a layer returns,
        # which is SGLang's fallback for models without the
        # attention-residual scheme. Capturing the prefix sum costs 4.5pt
        # of draft acceptance on K3 + RadixArk DSpark (AR 66.9% -> 71.4%).
        # The tap fires inside layer i+1, which computes that tensor
        # anyway; see its forward docstring. Ground truth: SGLang
        # kimi_k3.py:2697 _dspark_capture_stream, attn_residual.py:313
        # aggregate_stream_torch.
        if (
            spec_metadata is not None
            and i > 0
            and (capture_set is None or self.layers[i - 1].layer_idx in capture_set)
        ):
            return (spec_metadata, self.layers[i - 1].layer_idx)
        return None

    def _final_norm(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        num_snapshots: int,
        spec_metadata,
        capture_set,
        max_fused_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        # The last layer has no successor, so this one recompute is
        # unavoidable -- output-side score weights, matching SGLang's
        # layer_idx + 1 >= end_layer branch. Unreachable for K3's capture set
        # against 93 layers; kept so a set that does include the final layer
        # gets the right tensor rather than the raw prefix sum.
        if spec_metadata is not None and len(self.layers) > 0:
            last = self.layers[-1]
            if capture_set is None or last.layer_idx in capture_set:
                tail = (
                    _apply_attn_res(
                        hidden_states,
                        block_residual[:num_snapshots],
                        self.output_attn_res_proj,
                        self.output_attn_res_norm,
                    )
                    if num_snapshots > 0 and _AUX_ATTN_RES_STREAM_ENABLED
                    else hidden_states
                )
                spec_metadata.maybe_capture_hidden_states(last.layer_idx, tail, None)

        return _apply_attn_res_and_rmsnorm(
            hidden_states,
            block_residual[:num_snapshots],
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.norm,
            max_fused_tokens,
        )


# ----------------------------------------------------------------------------------------------------------------------
# The target: step classification, the construction checks and the class route returns.
# ----------------------------------------------------------------------------------------------------------------------


def _text_model_config(model_config: ModelConfig) -> ModelConfig:
    """The language model's ModelConfig: the checkpoint's text_config, with quant exclusions renamed to match.

    The checkpoint names its language-model modules `language_model.<x>`; the text model's are `<x>`, with
    `layers.*` under `model.`.
    """
    config = model_config.pretrained_config
    text = copy.copy(model_config)
    text._frozen = False
    text.pretrained_config = config.text_config
    excluded = text.quant_config.exclude_modules
    if excluded:
        text.quant_config = copy.copy(text.quant_config)
        renamed = []
        for name in excluded:
            if name.startswith(_LANG_PREFIX):
                name = name[len(_LANG_PREFIX) :]
                if name.startswith("layers."):
                    name = "model." + name
            renamed.append(name)
        text.quant_config.exclude_modules = renamed
    text.skip_create_weights_in_init = True
    text._frozen = True
    return text


def _check_construction(model_config: ModelConfig) -> None:
    """The settings this target is built for that are fixed before the first step."""
    capability = torch.cuda.get_device_capability()
    if capability != _SM:
        raise NoTarget(
            f"this target is certified on sm_{_SM[0]}{_SM[1]}, running on sm_{capability[0]}{capability[1]}"
        )
    mapping = model_config.mapping
    topology = (
        mapping.world_size,
        mapping.tp_size,
        mapping.pp_size,
        mapping.moe_tp_size,
        mapping.moe_ep_size,
        mapping.enable_attention_dp,
    )
    if topology != (16, 16, 1, 4, 4, False):
        raise NoTarget(
            "the tp16_moetp4ep4 target needs world_size 16, tensor_parallel_size 16, pipeline_parallel_size 1, "
            "moe_tensor_parallel_size 4, moe_expert_parallel_size 4 and enable_attention_dp false; the engine built "
            f"(world, tp, pp, moe_tp, moe_ep, attention_dp) = {topology}"
        )
    if model_config.torch_dtype != torch.bfloat16:
        raise NoTarget(
            f"this target computes in bf16; the engine resolved dtype {model_config.torch_dtype}"
        )
    kv_algo = model_config.quant_config.kv_cache_quant_algo
    if kv_algo is not None:
        raise NoTarget(
            f"this target's MLA kernels read a bf16 KV pool; kv_cache_config.dtype resolved to {kv_algo}"
        )
    quant_algo = model_config.quant_config.quant_algo
    if quant_algo != QuantAlgo.W4A16_MXFP4 or model_config.quant_config_dict:
        raise NoTarget(
            "this target loads the MXFP4 checkpoint, whose compressed-tensors config the model config reads as "
            "W4A16_MXFP4 with no per-layer declarations (its routed experts run the W4A8_MXFP4_MXFP8 default); the "
            f"engine read {quant_algo} with {len(model_config.quant_config_dict or {})} per-layer declarations"
        )
    strategy = model_config.allreduce_strategy
    if strategy not in (AllReduceStrategy.AUTO, AllReduceStrategy.MNNVL):
        raise NoTarget(
            f"this target takes allreduce_strategy AUTO or MNNVL; allreduce_strategy is {strategy.name}"
        )


class ModelingV2KimiK3Mxfp4Sm100Tp16Moetp4ep4(
    DecoderModelForCausalLM[KimiLinearModel, PretrainedConfig]
):
    """The class ``route`` returns: this target's text model (`KimiLinearModel` above) behind its checks, with
    one-engine speculative decoding (the functions of `spec_shell.py`). The checkpoint load is this target's manifest
    loader (`weights.py`)."""

    # The engine builds the KDA layers' per-step metadata from this class (the hybrid cache manager's replay slots).
    mamba_metadata_cls = KimiK3MambaMetadata
    # The one-engine speculative fast paths (vocabulary-sharded sampling, hoisted positions): not taken.
    spec_decode_fast_paths = False

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """The V2 hybrid manager holds the KDA states; the step contract requires it."""
        return "V2"

    def __init__(self, model_config: ModelConfig):
        config = model_config.pretrained_config
        if getattr(config, "text_config", None) is None:
            raise NoTarget(
                "this target loads the KimiK3ForConditionalGeneration checkpoint, whose language model is its "
                "text_config"
            )
        _check_construction(model_config)
        text = _text_model_config(model_config)
        cfg = text.pretrained_config
        super().__init__(
            KimiLinearModel(text),
            config=text,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
        )
        # The speculative fields the engine reads, off until spec_shell.build sets them.
        self.spec_config = None
        self.draft_config = None
        self.draft_model = None
        self.spec_worker = None
        self.use_separate_draft_kv_cache = False
        # The capture slot of the final hidden states (DSpark capture layer -1).
        self.layer_idx = -1
        _spec_shell.build(self, text)
        self._step_checked = False
        # Off unless TRTLLM_MODELING_V2_VALIDATE asks for it; read once here rather than per forward.
        self._validate_step_contract = step_contract_enabled()
        # The LM head on gemm/k3_head_gemv at decode size, once cache_derived_state has built its state. It is the
        # processor the speculative worker's target logits and a parallel drafter's logits call too.
        stock_logits_processor = self.logits_processor
        self.logits_processor = _decode_gemv.K3LogitsProcessor(stock_logits_processor)
        draft_model = getattr(self, "draft_model", None)
        if getattr(draft_model, "logits_processor", None) is stock_logits_processor:
            draft_model.logits_processor = self.logits_processor
        # The executor reads generation settings (eos_token_id, ...) off the model config the engine holds, which
        # must therefore be the text config, as the built-in wrapper leaves it.
        model_config._frozen = False
        model_config.pretrained_config = self.config
        model_config._frozen = True

    def load_weights(self, weights, *args, **kwargs):
        _weights.load(self, weights)

    def cache_derived_state(self) -> None:
        """Build the decode GEMVs' state once the weights are final: the LM head's workspace, and one eager call of
        every decode GEMV kernel at its site's shape (decode_gemv.SITES), so none compiles under capture. Built once:
        a later call keeps it, since CUDA graphs captured in between hold its workspace. The DSpark drafter's sites
        are among them, and the drafter gets the state too."""
        super().cache_derived_state()
        if self.model.decode_gemvs is not None:
            return
        gemvs = _decode_gemv.K3DecodeGemvs.create(self.lm_head)
        self.model.decode_gemvs = gemvs
        self.logits_processor.gemvs = gemvs
        for layer in self.model.layers:
            if not layer.is_moe:
                layer.decode_gemvs = gemvs
        if isinstance(self.draft_model, K3DSparkDrafter):
            self.draft_model.decode_gemvs = gemvs

    def post_load_weights(self) -> None:
        """The state the decode kernels share, built once per device before any CUDA-graph capture and handed to
        every layer of its kind: the KDA projection's Lamport buffers and the MLA decode attention's workspace; the
        decode GEMVs' state (built by ``cache_derived_state``) handed to every attention module; and, where every
        attention all-reduce runs over MNNVL, the TP group's collective state (``K3DecodeComm``, collective: every
        rank builds it here) handed to every layer, with whether its sandwich takes the layer's o_proj, the decode
        path's one-shot ceiling on every stock MNNVL all-reduce (``use_decode_one_shot``), and the MoE decode path
        (``_build_decode_moe``); then the speculative worker's decode kernels (``_gate_spec_worker_kernels``) and the
        DSpark drafter's collectives (``_gate_drafter_comm``)."""
        super().post_load_weights()
        kda = [layer.linear_attn for layer in self.model.layers if layer.is_kda]
        mla = [layer.self_attn.mixer for layer in self.model.layers if not layer.is_kda]
        if kda[0].k3_buffers is not None:
            return  # built by an earlier call; CUDA graphs captured since hold it
        device = self.model.embed_tokens.weight.device
        buffers = K3KdaBuffers.create(device)
        workspace = K3MlaAttnWorkspace.create(device, mla[0].num_heads_tp // 6)
        for module in kda:
            module.k3_buffers = buffers
        for module in mla:
            module.k3_workspace = workspace
        for module in kda + mla:
            module.decode_gemvs = self.model.decode_gemvs
        layers = self.model.layers
        comm = None
        if all(layer._mnnvl_allreduce() is not None for layer in layers):
            for layer in layers:
                layer.sandwich_oproj = _decode_comm.K3DecodeComm.takes_oproj(
                    layer._o_proj(), self.model.num_attn_res_snapshots
                )
            oproj = next((layer._o_proj().weight for layer in layers if layer.sandwich_oproj), None)
            comm = _decode_comm.K3DecodeComm.create(self.model_config.mapping, oproj)
            for layer in layers:
                layer.decode_comm = comm
            _decode_comm.use_decode_one_shot(self)
        moe_layers = 0 if comm is None else self._build_decode_moe(comm)
        logger.info(
            "Kimi K3 decode kernels: KDA on k3_kda_decode_attn, k3_kda_attn and k3_kda_verify "
            f"({sum(m.takes_k3_kernels for m in kda)} / {len(kda)} layers take them), MLA on k3_mla_qkv and "
            f"k3_mla_attn_vb_out ({len(mla)} layers); the post-attention all-reduce of at most "
            f"{_decode_comm.AR_ATTN_RES_MAX_TOKENS} tokens "
            + (
                "with the residual update: k3_sandwich_oproj with o_proj at most "
                f"{_decode_comm.SANDWICH_MAX_TOKENS} decode tokens "
                f"({sum(layer.sandwich_oproj for layer in layers)} / {len(layers)} layers), else "
                "mnnvl_allreduce_attn_res"
                if comm is not None
                else "unfused (an attention all-reduce does not run over MNNVL)"
            )
            + f"; MoE on k3_moe_front, k3_moe and the row-parallel tail ({moe_layers} layers)"
        )
        self._gate_spec_worker_kernels(comm)
        self._gate_drafter_comm(comm)

    def _build_decode_moe(self, comm: _decode_comm.K3DecodeComm) -> int:
        """The MoE decode path (``decode_moe.py``) on every MoE layer it takes: the shared state (collective: the
        front's head workspace), each layer's decode weights with its latent norm folded into the latent up
        projection, the decode GEMVs' state; then one call of every kernel of the path (collective: the front and
        the sandwich tail). Every rank builds it here. Returns the number of MoE layers it takes."""
        mapping = self.model_config.mapping
        moes = [layer.block_sparse_moe for layer in self.model.layers if layer.is_moe]
        gaps = {
            id(moe): _decode_moe.layout_gaps(
                moe, mapping.tp_size, self.model.num_attn_res_snapshots
            )
            for moe in moes
        }
        takes = [moe for moe in moes if not gaps[id(moe)]]
        for reason in sorted({gap for moe in moes for gap in gaps[id(moe)]}):
            logger.info_once(
                f"Kimi K3 MoE decode path off on some layers: {reason}",
                key=f"k3_decode_moe_off_{reason}",
            )
        if not takes:
            return 0
        backend = takes[0].routed_experts.backend
        state = _decode_moe.K3DecodeMoe.create(
            mapping,
            backend.w3_w1_weight.device,
            backend.w3_w1_weight.shape[1] // 2,
            backend.expert_size_per_partition,
            comm.mnnvl,
        )
        for moe in takes:
            _decode_moe.fold_latent_norm(moe)
            moe.decode_moe = _decode_moe.K3DecodeMoeLayer.create(
                moe, state, mapping.tp_rank, mapping.tp_size
            )
            moe.decode_gemvs = self.model.decode_gemvs
        first = takes[0].decode_moe
        first.warm_up(takes[0])
        logger.info(
            "Kimi K3 MoE decode path: the latent all-reduce at <= 8 tokens on "
            + (
                "the routed experts' push form and k3_latent_reduce on pushing steps (a pure decode step captured "
                "into a CUDA graph on the decode kernels), else the routed experts' all-reduce"
                if state.exchange is not None
                else f"the routed experts' all-reduce (no latent exchange at TP {mapping.tp_size})"
            )
        )
        comm.compile_tail(takes[0].moe_hidden_size, first.shared_cols, first.tail_weight)
        return len(takes)

    def _gate_spec_worker_kernels(self, comm: Optional[_decode_comm.K3DecodeComm]) -> bool:
        """Turn the DSpark worker's Kimi K3 decode kernels (its ``k3_decode``: ``trtllm::k3_spec_accept``,
        ``k3_ctx_kv`` and ``k3_markov``, on target and draft logits kept vocabulary-sharded) on only when every input
        that path needs exists: the TP group's collective state over MNNVL (``comm``) and the LM head's
        ``gemm/k3_head_gemv`` workspace. ``K3LogitsProcessor.lm_head_shard`` needs that workspace to produce the
        vocabulary shard, so the workspace is a precondition of the path, not a policy choice. The worker still checks
        each step's own conditions. Returns the setting (False without such a worker)."""
        worker = getattr(self, "spec_worker", None)
        if not hasattr(worker, "k3_decode"):
            return False
        gemvs = self.model.decode_gemvs
        worker.k3_decode = (
            comm is not None and gemvs is not None and gemvs.head_workspace is not None
        )
        logger.info(
            f"Kimi K3 decode kernels: {type(worker).__name__} k3_spec_accept, k3_ctx_kv and k3_markov "
            + (
                "on"
                if worker.k3_decode
                else "off (no MNNVL decode state or no k3_head_gemv LM head)"
            )
        )
        return worker.k3_decode

    def _gate_drafter_comm(self, comm: Optional[_decode_comm.K3DecodeComm]) -> bool:
        """Hand the DSpark drafter (`K3DSparkDrafter`) the TP group's collective state ``comm`` where this target built
        it (every attention all-reduce over MNNVL; TP16 is a construction assert): the drafter then runs its context
        projection split over the group, ``hidden_norm`` in the all-reduce, and its blocks' residual adds and RMSNorms
        in their all-reduces (``K3DSparkDrafter.use_decode_comm``, collective: it compiles the drafter's sandwich on
        the group's workspace). Without ``comm`` it keeps the stock replicated ``fc``, all-reduces and norms. The gate
        requires exactly what this path uses: ``comm`` and nothing else, so not the LM head's ``k3_head_gemv``
        workspace, which only the speculative worker's path (`_gate_spec_worker_kernels`) reads. Returns whether the
        drafter took the state (False without such a drafter)."""
        drafter = getattr(self, "draft_model", None)
        if comm is None or not isinstance(drafter, K3DSparkDrafter):
            return False
        drafter.use_decode_comm(comm)
        return True

    def _check_step_contract(self, attn_metadata: AttentionMetadata) -> None:
        """First-forward checks. The engine fields this target reads (``REQUIRED_ENGINE_FIELDS``) are checked only
        when TRTLLM_MODELING_V2_VALIDATE asks for it; a missing one otherwise fails where it is read. The engine
        settings this target cannot run with are checked always: each raises naming the setting."""
        if self._validate_step_contract:
            objects = {
                "attn_metadata": attn_metadata,
                "kv_cache_manager": attn_metadata.kv_cache_manager,
            }
            missing = [
                f"{owner}.{name}"
                for owner, names in REQUIRED_ENGINE_FIELDS.items()
                for name in names
                if not hasattr(objects[owner], name)
            ]
            if missing:
                raise AssertionError(f"engine fields this target reads are missing: {missing}")
        # Always: decode_step reads seq_lens with a default, and without it would classify a step of unequal
        # generation lengths as a uniform decode step.
        if not hasattr(attn_metadata, "seq_lens"):
            raise RuntimeError(
                "the attention metadata has no seq_lens; this target's decode_step needs it"
            )
        if attn_metadata.tokens_per_block != _TOKENS_PER_BLOCK:
            raise ValueError(
                f"this target needs kv_cache_config.tokens_per_block {_TOKENS_PER_BLOCK}; the engine built "
                f"{attn_metadata.tokens_per_block}"
            )
        manager = attn_metadata.kv_cache_manager
        if not isinstance(manager, KimiK3HybridCacheManager):
            raise ValueError(
                "this target needs the Kimi K3 hybrid KV / state cache manager "
                "(kv_cache_config.use_kv_cache_manager_v2); the engine built "
                f"{type(manager).__name__}"
            )
        if manager.enable_block_reuse:
            raise ValueError(
                "this target runs with kv_cache_config.enable_block_reuse false; the engine enabled it"
            )
        kda_layer = next(layer.layer_idx for layer in self.model.layers if layer.is_kda)
        state_dtype = manager.mamba_layer_cache(kda_layer).temporal.dtype
        if state_dtype != torch.float32:
            raise ValueError(
                "this target's KDA kernels keep fp32 recurrent states (kv_cache_config.mamba_ssm_cache_dtype float32 "
                f"or auto); the engine built a {state_dtype} state pool"
            )
        self._step_checked = True

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.pop("multimodal_params", None):
            raise ValueError(
                "this Kimi K3 target is text only: it loads no vision tower, and a request carried image input"
            )
        if not self._step_checked:
            self._check_step_contract(attn_metadata)
        if self.spec_worker is not None:
            return _spec_shell.forward(
                self,
                attn_metadata,
                input_ids,
                position_ids,
                inputs_embeds,
                spec_metadata,
                resource_manager,
                **kwargs,
            )
        hidden_states = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        if attn_metadata.padded_num_tokens is not None:
            hidden_states = hidden_states[: attn_metadata.num_tokens]
        return self.logits_processor.forward(
            hidden_states, self.lm_head, attn_metadata, return_context_logits
        )

    def load_draft_weights(self, weights: Dict, weight_mapper=None) -> None:
        """The drafter's checkpoint (the model loader's second pass), then the shared embedding / LM head."""
        _spec_shell.load_draft_weights(self, weights, weight_mapper)

    def set_guided_decoder(self, guided_decoder) -> bool:
        return _spec_shell.set_guided_decoder(self, guided_decoder)
