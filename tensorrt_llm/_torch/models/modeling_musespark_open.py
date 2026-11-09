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
"""Muse Spark Open text decoder for the PyTorch backend.

The decoder handles text only. `MuseSparkForConditionalGeneration` in
`modeling_musespark_open_omni` wraps it with the vision and audio towers;
this module never constructs them.

Shape summary for the released checkpoint: 62 MoE decoder layers, hidden 8192,
128 query heads / 16 KV heads of dim 64, 256 experts with top-8 routing, expert
state 4096 wide (`compression_factor=2`) and expert FFN intermediate 4096.
Layers `l` with `(num_layers - l - 1) % 4 == 0` are NoPE + full causal; the
rest are interleaved-RoPE with a 2048-key sliding window.

The residual stream is FP32. Each branch reads a BF16 cast of it, and its
output rejoins through a norm-preserving gated residual add.
"""

import contextlib
import copy
import os
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.configs.musespark_open import SLIDING_ATTENTION, MuseSparkOpenConfig
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.functional import AllReduceStrategy, PositionEmbeddingType, RotaryScalingType

from ...models.modeling_utils import QuantConfig
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..attention.qk_norm_attention import QKNormRoPEAttention
from ..distributed import AllReduce
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.linear import Linear, TensorParallelMode
from ..modules.logits_processor import LogitsProcessor, scale_and_soft_cap_logits
from ..modules.rms_norm import RMSNorm
from ..moe.fused_moe import MiniMaxM2MoeRoutingMethod, create_moe
from ..utils import AuxStreamType
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel

# Perception subtrees in the checkpoint. The text decoder never builds these,
# and their tensors are hundreds of GiB, so the loader skips them unread.
PERCEPTION_KEY_PREFIXES = ("vision_model.", "modalities.", "audio_model.")

# Per-layer, per-op NVTX ranges, on when `TLLM_LLMAPI_ENABLE_NVTX` or
# `TLLM_NVTX_DEBUG` is set (the variables `nvtx_range_debug` reads). The
# decision is made once per module at construction, so with profiling off a
# forward enters one shared no-op context per range and formats no strings.
# The ranges are host-side: under CUDA graphs they annotate the capture pass,
# so attribute kernels from an eager profile.
NVTX_DOMAIN = "MuseSparkOpen"
_NVTX_OFF = contextlib.nullcontext()


def nvtx_layer_ranges_enabled() -> bool:
    return (
        os.getenv("TLLM_LLMAPI_ENABLE_NVTX", "0") == "1" or os.getenv("TLLM_NVTX_DEBUG", "0") == "1"
    )


class LayerNvtx:
    """Prebuilt `<module>.L<idx>.<op>` NVTX ranges for one layer."""

    def __init__(self, module: str, layer_idx: int, ops: Tuple[str, ...]):
        self.enabled = nvtx_layer_ranges_enabled()
        self.names = {op: f"{module}.L{layer_idx:03d}.{op}" for op in ops} if self.enabled else {}

    def __call__(self, op: str):
        if not self.enabled:
            return _NVTX_OFF
        return nvtx_range(self.names[op], domain=NVTX_DOMAIN)


def normalize_exclude_modules(patterns: List[str]) -> List[str]:
    """Rewrite the checkpoint's bare-substring exclusions as fnmatch globs.

    `hf_quant_config.json` lists `["self_attn", "lm_head", "layers.0.",
    "layers.61."]`. TRT-LLM matches an exclusion against the module name and
    its ancestors with `fnmatch`, so a bare substring like `layers.0.`
    matches nothing and the BF16 layers would be built as NVFP4. Anchor each
    substring with `*` unless the producer already wrote a glob or regex.
    """
    normalized = []
    for pattern in patterns:
        if pattern.startswith("re:") or "*" in pattern or "?" in pattern:
            normalized.append(pattern)
        elif pattern == "lm_head":
            normalized.append(pattern)
        else:
            normalized.append(
                f"*{pattern.strip('.')}.*" if pattern.endswith(".") else f"*{pattern}*"
            )
    return normalized


def post_expert_norm_rows(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weight-first RMSNorm of each selected expert's output, before combining.

    Returns `(rows, valid)` with `rows` shaped `[num_tokens, top_k,
    hidden_size]` in FP32 and `valid` marking the slots whose expert is
    local to this rank.

    The per-expert rows are kept separate from the combine below so a parity
    check can compare them individually: against their sum alone, normalizing
    after the sum and normalizing before it are indistinguishable.

    `expanded_idx_to_permuted_idx` holds `-1` for slots whose expert is not
    local to this rank (or is padding). The MoE kernel never writes those rows
    of `gemm2_output`, so they may hold NaN; they are selected out rather
    than multiplied by zero.
    """
    index = expanded_idx_to_permuted_idx.reshape(num_tokens, top_k)
    valid = index >= 0
    safe_index = torch.where(valid, index, torch.zeros_like(index))
    gathered = gemm2_output.index_select(0, safe_index.reshape(-1).int())
    gathered = gathered[:, :hidden_size].reshape(num_tokens, top_k, hidden_size)
    gathered = torch.where(
        valid.unsqueeze(-1),
        gathered.to(torch.float32),
        torch.zeros_like(gathered, dtype=torch.float32),
    )

    scaled = gathered * (norm_weight.to(torch.float32) + 1.0)
    variance = scaled.pow(2).mean(-1, keepdim=True)
    normed = scaled * torch.rsqrt(variance + eps)
    # Invalid slots stay exactly zero: their gathered rows were replaced by
    # zeros, and rsqrt(0 + eps) would otherwise scale them to a finite value.
    return torch.where(valid.unsqueeze(-1), normed, torch.zeros_like(normed)), valid


def post_expert_norm_combine(
    gemm2_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    routing_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    *,
    num_tokens: int,
    top_k: int,
    hidden_size: int,
) -> torch.Tensor:
    """Weight-first RMSNorm each selected expert output, then combine, in FP32.

    The MoE kernel's own finalize cannot do this: RMSNorm is nonlinear, so
    normalizing after the top-k sum is a different function.
    """
    normed, _ = post_expert_norm_rows(
        gemm2_output,
        expanded_idx_to_permuted_idx,
        norm_weight,
        eps,
        num_tokens=num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
    )
    weights = routing_weights.reshape(num_tokens, top_k, 1).to(torch.float32)
    return (normed * weights).sum(dim=1)


def _routing_weights(
    scores: torch.Tensor,
    topk_idx: torch.Tensor,
    eps: float,
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather and renormalize the unbiased scores of the selected experts."""
    weights = scores.gather(1, topk_idx)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    return topk_idx.to(torch.int32), weights.to(output_dtype)


class MuseSparkOpenRoutingMethod(MiniMaxM2MoeRoutingMethod):
    """sigmoid -> add expert bias (selection only) -> top-k -> renormalize.

    Same algorithm as MiniMax-M2, with the checkpoint's renormalization epsilon
    (`router_score_division_eps`) instead of the MiniMax constant. Routing
    runs here rather than inside the MoE kernel, so the kernel and the
    post-expert-norm finalize use exactly these top-k ids and weights; routing
    again inside the kernel would apply a different epsilon.
    """

    def __init__(
        self, *, top_k: int, num_experts: int, callable_e_score_correction_bias, renorm_eps: float
    ):
        super().__init__(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=callable_e_score_correction_bias,
        )
        self.renorm_eps = renorm_eps

    @property
    def requires_separated_routing(self) -> bool:
        return True

    def apply(self, router_logits: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        scores, scores_with_bias = self.get_scores(router_logits, self.e_score_correction_bias)
        _, topk_idx = torch.topk(scores_with_bias, k=self.top_k, dim=-1, sorted=False)
        # The weights come from the unbiased scores; the bias only steers which
        # experts are picked.
        return _routing_weights(scores, topk_idx, self.renorm_eps, self.output_dtype)


class MuseSparkOpenRMSNormWeightFirst(nn.Module):
    """`rmsnorm((1 + w) * x, eps)` with no output gain.

    The checkpoint's "input scale" norm form, used for `post_expert_norm` and
    `post_ffn_norm`. It is not the same function as
    `RMSNorm(use_gemma=True)`, which applies `(1 + w)` *after* normalizing.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: Optional[torch.dtype]):
        super().__init__()
        # Zero-centered gamma: the stored value is an offset from 1.0.
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype), requires_grad=False)
        self.variance_epsilon = eps
        # `weight + 1`, derived from the frozen parameter once the weights
        # are loaded. Non-persistent, so it never enters a checkpoint.
        self.register_buffer("_weight_plus_one", None, persistent=False)

    def cache_derived_state(self) -> None:
        """Cache `weight + 1`; the loader calls this after the weights land."""
        if self.weight.is_meta:
            return
        self._weight_plus_one = self.weight + 1

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def gain(self) -> torch.Tensor:
        """`1 + w`, from the cache when it has been filled."""
        if self._weight_plus_one is None:
            return self.weight + 1
        return self._weight_plus_one

    def _apply(self, *args, **kwargs):
        # .to()/.cuda()/.float()/.half() all funnel through _apply and would
        # carry the cache along. Casting `w + 1` is not the same as
        # computing `cast(w) + 1`, so drop the cache on any move; the forward
        # recomputes until cache_derived_state() runs again.
        out = super()._apply(*args, **kwargs)
        self._weight_plus_one = None
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # The scaled input is rounded to the activation dtype before the
        # variance, as in the checkpoint's reference implementation.
        scaled = self.gain() * hidden_states
        scaled_fp32 = scaled.to(torch.float32)
        variance = scaled_fp32.pow(2).mean(-1, keepdim=True)
        return (scaled_fp32 * torch.rsqrt(variance + self.variance_epsilon)).to(input_dtype)


class MuseSparkOpenGatedResidual(nn.Module):
    """Norm-preserving gated residual add, evaluated in FP32.

    `h' = alpha * h + beta * y` with `beta = sigmoid(g / 0.3)` and
    `alpha = sqrt(max(sigmoid(-g / 0.3) * (1 + beta), 1e-3))`. Both branches
    are gated (this is not `h + g * y`), which keeps `alpha^2 + beta^2 == 1`
    away from the clamp.
    """

    TEMPERATURE = 0.3
    EPS = 1e-3

    def __init__(self, hidden_size: int, dtype: Optional[torch.dtype]):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(hidden_size, dtype=dtype), requires_grad=False)
        # alpha and beta depend only on the frozen gate; they are cached once
        # the weights are loaded. Non-persistent, so never checkpointed.
        self.register_buffer("_alpha", None, persistent=False)
        self.register_buffer("_beta", None, persistent=False)

    def _compute_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        scaled = self.gate.to(torch.float32) / self.TEMPERATURE
        beta = torch.sigmoid(scaled)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-scaled) * (1.0 + beta), min=self.EPS))
        return alpha, beta

    def cache_derived_state(self) -> None:
        """Cache alpha and beta from the loaded gate.

        `ModelLoader` calls `cache_derived_state` on every module that
        defines it, including on readers that import weights another rank
        committed and never run the weight mapper; `post_load_weights`
        delegates here so the ordinary load path fills the cache too. Skips a
        meta parameter, which has no values to cache, and recomputes from the
        live parameter on every call.
        """
        if self.gate.is_meta:
            return
        self._alpha, self._beta = self._compute_coefficients()

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    def _apply(self, *args, **kwargs):
        # alpha and beta are FP32 whatever the module dtype; a .to(bfloat16)
        # after the cache was filled would round them, and coefficients() would
        # then differ from the uncached path. Drop the cache on any move.
        out = super()._apply(*args, **kwargs)
        self._alpha = None
        self._beta = None
        return out

    def coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._alpha is None:
            return self._compute_coefficients()
        return self._alpha, self._beta

    def forward(self, residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.coefficients()
        return alpha * residual + beta * branch.to(torch.float32)

    def add_after_rmsnorm(
        self,
        residual: torch.Tensor,
        branch: torch.Tensor,
        eps: float,
        gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`self(residual, rmsnorm(branch))`: one residual join of a layer.

        `gain` selects the norm: `1 + w` for the weight-first FFN norm,
        `None` for the weightless attention one. The gain product and the
        normalized branch are each rounded to the activation dtype, as the
        modules they stand for do.
        """
        alpha, beta = self.coefficients()
        scaled = (gain * branch if gain is not None else branch).to(torch.float32)
        normed = (scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)).to(branch.dtype)
        return alpha * residual + beta * normed.to(torch.float32)


class MuseSparkOpenLogitsProcessor(LogitsProcessor):
    """LM head epilogue: metap multiplier then a tanh soft cap, both FP32."""

    def __init__(self, logit_multiplier: float, soft_cap: float):
        super().__init__()
        self.logit_multiplier = logit_multiplier
        self.soft_cap = soft_cap

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head,
        attn_metadata,
        return_context_logits: bool = False,
    ) -> torch.Tensor:
        logits = super().forward(hidden_states, lm_head, attn_metadata, return_context_logits)
        return scale_and_soft_cap_logits(logits, self.logit_multiplier, self.soft_cap)


def _allreduce_strategy(model_config: ModelConfig) -> AllReduceStrategy:
    """The configured all-reduce strategy, with NCCL in place of AUTO.

    AUTO does not work for this model at TP4 on one node:

    - The attention output projection reduces a tiny message per decode step,
      and AUTO sends tiny messages down the NCCL symmetric-memory path, which
      corrupts the reduce and returns non-finite layer outputs.
    - The MoE all-reduce goes through `tunable_allreduce`, whose autotuner
      profiles each tactic inside a CUDA-graph capture that the ranks do not
      enter as one collective, so engine warmup hangs.

    An explicit NCCL_SYMMETRIC selects that symmetric-memory path, so it is
    rejected. Any other strategy is used as configured.
    """
    strategy = model_config.allreduce_strategy
    if strategy == AllReduceStrategy.NCCL_SYMMETRIC:
        raise ValueError(
            "MuseSparkOpen does not support allreduce_strategy=NCCL_SYMMETRIC: the NCCL "
            "symmetric-memory all-reduce corrupts the attention output projection's small "
            "decode-step reduces at TP4. Use NCCL or leave the default (AUTO)."
        )
    return AllReduceStrategy.NCCL if strategy == AllReduceStrategy.AUTO else strategy


class MuseSparkOpenAttention(QKNormRoPEAttention):
    """Attention with weightless per-head Q/K norms, a query pre-scale, an
    output norm+gate fused into the QKV projection, and a per-layer scope.

    `QKNormRoPEAttention(is_qk_norm=False)` gives exactly the checkpoint's
    weightless per-head RMSNorm over `head_dim`; the query pre-scale and the
    output norm are layered on top.
    """

    def __init__(self, model_config: ModelConfig[MuseSparkOpenConfig], layer_idx: int):
        config = model_config.pretrained_config
        self.is_sliding = config.layer_types[layer_idx] == SLIDING_ATTENTION
        self.uses_rope = config.use_rope_per_layer[layer_idx]
        self.attention_window_size = config.sliding_window if self.is_sliding else None
        self.scale_query_by = config.scale_query_by

        if self.uses_rope:
            rope_params = RopeParams(
                dim=config.head_dim,
                theta=config.rope_theta,
                scale_type=RotaryScalingType.none,
                scale=1.0,
                max_positions=config.max_position_embeddings,
            )
            # GPT-J / interleaved rotation over adjacent channel pairs.
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=rope_params,
                is_neox=False,
            )
        else:
            pos_embd_params = None

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            skip_rope=not self.uses_rope,
            # The query pre-scale sits between the norm and the rotation, which
            # the fused qk-norm + RoPE op cannot express.
            fuse_qk_norm_rope=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            # The kernel keeps its plain 1/sqrt(head_dim); the checkpoint's extra
            # query multiplier is applied explicitly in apply_qk_norm.
            q_scaling=1.0,
            attn_output_gate=True,
            is_qk_norm=False,
        )
        # Weightless per-head norm on the attention output, applied before the
        # output gate.
        self.o_head_norm = RMSNorm(
            hidden_size=self.head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )
        # `o_proj` builds its all-reduce with the configured strategy as is;
        # rebuild it so AUTO becomes NCCL (see _allreduce_strategy).
        if self.o_proj.all_reduce is not None:
            self.o_proj.all_reduce = AllReduce(
                mapping=self.o_proj.mapping,
                strategy=_allreduce_strategy(model_config),
                dtype=self.o_proj.dtype,
            )

    def apply_qk_norm(self, q, k):
        q, k = super().apply_qk_norm(q, k)
        q = q * self.scale_query_by
        return q, k

    def apply_output_gate(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Weightless per-head RMSNorm on the attention result, then the gate.

        Order matters: the checkpoint normalizes each head of the attention
        output before multiplying by `sigmoid(o_gate)`, and only then
        projects.
        """
        normed = self.o_head_norm(attention_output.reshape(-1, self.head_dim)).reshape(
            attention_output.shape
        )
        return super().apply_output_gate(normed, gate)

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.CAUSAL,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


class MuseSparkOpenMoE(nn.Module):
    """Router + pooled experts + per-expert post-norm combine + unpooler.

    Order (the checkpoint's `feed_forward`): route on the 8192-wide pre-FFN
    activation, pool it to 4096, pre-expert norm, run the top-8 experts, apply
    the weight-first post-expert norm to *each* expert output before the
    weighted sum, all-reduce the partial sums across expert-parallel ranks,
    then unpool back to 8192.

    The per-expert norm is why the MoE runs with `do_finalize=False` and
    this module's `finalize` registered as its `unfinalized_combine_fn`:
    RMSNorm is nonlinear, so normalizing the already-combined result would be a
    different model.
    """

    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.expert_hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.moe_topk
        self.metap_multiplier = config.metap_multiplier
        self.mapping = model_config.mapping
        # Expert parallelism reduces the normalized local experts. Pure MoE
        # tensor parallelism must instead reduce each expert's FC2 partials
        # before its nonlinear normalization (see finalize).
        pure_ep = self.mapping.moe_tp_size == 1 and (
            self.mapping.moe_ep_size == self.mapping.tp_size
        )
        pure_tp = (
            self.mapping.moe_ep_size == 1
            and (self.mapping.moe_tp_size == self.mapping.tp_size)
            and not self.mapping.enable_attention_dp
        )
        if not (pure_ep or pure_tp):
            raise ValueError(
                "MuseSparkOpen requires MoE EP == attention TP with MoE TP == 1, "
                "or pure MoE TP without attention DP, got "
                f"moe_tp_size={self.mapping.moe_tp_size}, "
                f"moe_ep_size={self.mapping.moe_ep_size}, "
                f"tp_size={self.mapping.tp_size}, "
                f"enable_attention_dp={self.mapping.enable_attention_dp}"
            )

        # Router: unquantized FP32 linear, replicated on every rank.
        self.router = Linear(
            self.hidden_size, self.num_experts, bias=False, dtype=torch.float32, quant_config=None
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32), requires_grad=False
        )

        # Pooler and unpooler stay replicated and unquantized: the checkpoint
        # keeps them in original precision, and replication avoids an allgather.
        self.pooler = Linear(
            self.hidden_size,
            self.expert_hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=None,
        )
        self.unpooler = Linear(
            self.expert_hidden_size,
            self.hidden_size,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=None,
        )

        self.pre_expert_norm = RMSNorm(
            hidden_size=self.expert_hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.post_expert_norm_eps = config.rms_post_norm_eps
        # Weight-first post-expert norm, applied per selected expert output.
        self.post_expert_norm_weight = nn.Parameter(
            torch.zeros(self.expert_hidden_size, dtype=config.torch_dtype), requires_grad=False
        )

        aux_stream_dict = (
            {AuxStreamType.MoeChunkingOverlap: aux_stream} if aux_stream is not None else None
        )
        # The checkpoint quant-excludes whole layers ("layers.0." in
        # exclude_modules), leaving their experts BF16. Backend resolution only
        # sees the global quant config, so give it the format the layer actually
        # has; it then resolves the TRTLLM-Gen BF16 implementation, which
        # returns the unfinalized output the per-expert norm needs. The MegaMoE
        # families have no BF16 implementation, so such a layer falls back to
        # the TRTLLM backend.
        layer_model_config = model_config
        layer_quant_config = None
        global_quant_config = model_config.quant_config
        if (
            global_quant_config is not None
            and global_quant_config.exclude_modules
            and global_quant_config.is_module_excluded_from_quantization(
                f"model.layers.{layer_idx}"
            )
        ):
            layer_quant_config = QuantConfig(
                kv_cache_quant_algo=global_quant_config.kv_cache_quant_algo
            )
            if str(getattr(model_config, "moe_backend", "")).upper() in (
                "MEGAMOE_CUTEDSL",
                "MEGAMOE_DEEPGEMM",
            ):
                layer_model_config = copy.copy(model_config)
                layer_model_config._frozen = False
                layer_model_config.moe_backend = "TRTLLM"
                layer_model_config._frozen = True
        self.experts = create_moe(
            routing_method=MuseSparkOpenRoutingMethod(
                top_k=self.top_k,
                num_experts=self.num_experts,
                callable_e_score_correction_bias=lambda: self.expert_bias,
                renorm_eps=config.router_score_division_eps,
            ),
            num_experts=self.num_experts,
            hidden_size=self.expert_hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=config.torch_dtype,
            # The per-expert post-norm has to run before the combine, so the
            # partial sums are reduced by this module, not inside the backend.
            reduce_results=False,
            model_config=layer_model_config,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
            override_quant_config=layer_quant_config,
            unfinalized_combine_fn=self.finalize,
        )
        # NCCL in place of AUTO keeps warmup off the tunable all-reduce
        # (see _allreduce_strategy).
        self.all_reduce = AllReduce(
            mapping=self.mapping, strategy=_allreduce_strategy(model_config)
        )

        self.nvtx = LayerNvtx(
            "moe",
            layer_idx,
            ("router", "pool", "experts", "finalize", "all_reduce", "unpool"),
        )

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """FP32 router logits, scaled by the metap multiplier before routing."""
        return self.router(hidden_states.to(torch.float32)) * self.metap_multiplier

    def finalize(
        self,
        *,
        gemm2_output: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        token_selected_slots: Optional[torch.Tensor],
        routing_weights: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """The MoE's `unfinalized_combine_fn`: per-expert norm, then combine.

        The scheduler calls this on the backend's unfinalized output with the
        routing weights it dispatched with, before any communication combine.
        Each output row depends on one `(token, expert)` row at a time and
        non-local slots contribute exact zeros, so combining the per-rank
        results equals finalizing after the combine.
        """
        # The post-expert norm is the same for every expert.
        del token_selected_slots
        with self.nvtx("finalize"):
            if self.mapping.moe_tp_size > 1:
                # Backend row permutations are rank-local, so reduce in canonical
                # (token, selected-expert) order rather than in GEMM scratch
                # order. FP32 keeps small residuals between opposing FC2 shards
                # before the norm can amplify them. This is the only MoE reduce
                # under pure TP; every rank then holds complete expert outputs.
                index = expanded_idx_to_permuted_idx.reshape(-1)
                valid = (index >= 0) & (index < gemm2_output.shape[0])
                safe_index = torch.where(valid, index, torch.zeros_like(index))
                rows = gemm2_output.index_select(0, safe_index.int())[:, : self.expert_hidden_size]
                rows = torch.where(valid[:, None], rows.float(), 0.0)
                gemm2_output = self.all_reduce(rows.contiguous())
                expanded_idx_to_permuted_idx = torch.arange(
                    num_tokens * self.top_k, dtype=torch.int32, device=gemm2_output.device
                )
            return post_expert_norm_combine(
                gemm2_output,
                expanded_idx_to_permuted_idx,
                routing_weights,
                self.post_expert_norm_weight,
                self.post_expert_norm_eps,
                num_tokens=num_tokens,
                top_k=self.top_k,
                hidden_size=self.expert_hidden_size,
            )

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        with self.nvtx("router"):
            router_logits = self.router_logits(hidden_states)
        with self.nvtx("pool"):
            pooled = self.pre_expert_norm(self.pooler(hidden_states))
        with self.nvtx("experts"):
            # With the finalize registered, the MoE returns this rank's tokens
            # already normalized, weighted and summed over the local experts.
            combined = self.experts(
                pooled,
                router_logits,
                do_finalize=False,
                output_dtype=hidden_states.dtype,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens
                if attn_metadata is not None
                else None,
                use_dp_padding=False,
            )
        # Cast before the reduce: with expert parallelism the partial sums cross
        # the wire at the activation width, and they are cast to it for the
        # unpooler either way.
        combined = combined.to(hidden_states.dtype)
        # Expert parallelism leaves a partial sum on each rank; reduce it over
        # the TP group. Not under attention DP, where the ranks hold different
        # tokens and the MoE's own combine has already reduced the partials, and
        # not under pure MoE TP, where finalize has already reduced each expert.
        if (
            not self.mapping.enable_attention_dp
            and self.mapping.has_tp()
            and self.mapping.moe_tp_size == 1
        ):
            with self.nvtx("all_reduce"):
                combined = self.all_reduce(combined)
        with self.nvtx("unpool"):
            return self.unpooler(combined)


class MuseSparkOpenDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[MuseSparkOpenConfig],
        layer_idx: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping

        self.self_attn = MuseSparkOpenAttention(model_config, layer_idx)
        self.feed_forward = MuseSparkOpenMoE(model_config, layer_idx, aux_stream)

        self.pre_qkv_proj_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        # post_norm_affine=false: the post-attention norm has no weight.
        self.post_attn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_post_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )
        self.pre_ffn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.post_ffn_norm = MuseSparkOpenRMSNormWeightFirst(
            hidden_size=config.hidden_size, eps=config.rms_post_norm_eps, dtype=config.torch_dtype
        )

        self.add_attn = MuseSparkOpenGatedResidual(config.hidden_size, config.torch_dtype)
        self.add_ffn = MuseSparkOpenGatedResidual(config.hidden_size, config.torch_dtype)
        self.dtype = config.torch_dtype

        self.nvtx = LayerNvtx(
            "layer",
            layer_idx,
            ("pre_qkv_proj_norm", "attn", "norm_add_attn", "moe", "norm_add_ffn"),
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # hidden_states is the FP32 residual stream; each branch reads a cast of
        # it to the activation dtype and writes back through the gated add.
        with self.nvtx("pre_qkv_proj_norm"):
            attn_input = self.pre_qkv_proj_norm(hidden_states.to(self.dtype))
        # The attention projection and its TP reduction stay in this range.
        with self.nvtx("attn"):
            attn_out = self.self_attn(
                position_ids=position_ids,
                hidden_states=attn_input,
                attn_metadata=attn_metadata,
                **kwargs,
            )
        with self.nvtx("norm_add_attn"):
            hidden_states = self.add_attn.add_after_rmsnorm(
                hidden_states, attn_out, self.post_attn_norm.variance_epsilon
            )
        with self.nvtx("moe"):
            ffn_out = self.feed_forward(
                self.pre_ffn_norm(hidden_states.to(self.dtype)), attn_metadata
            )
        with self.nvtx("norm_add_ffn"):
            hidden_states = self.add_ffn.add_after_rmsnorm(
                hidden_states,
                ffn_out,
                self.post_ffn_norm.variance_epsilon,
                self.post_ffn_norm.gain(),
            )
        # Speculative drafters that read target hidden states capture the FP32
        # residual stream.
        spec_metadata = kwargs.get("spec_metadata")
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, None)
        return hidden_states


class MuseSparkOpenModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[MuseSparkOpenConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.aux_stream = torch.cuda.Stream()
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        # Weightless RMSNorm on the token embeddings, then the m_emb scale.
        self.embed_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )
        self.m_emb = config.m_emb
        self.layers = nn.ModuleList(
            [
                MuseSparkOpenDecoderLayer(model_config, layer_idx, self.aux_stream)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # The final norm uses the stored weight as is (gain centered at 0).
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.dtype = config.torch_dtype

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        embeds = self.embed_norm(inputs_embeds.to(self.dtype))
        if self.m_emb != 1.0:
            embeds = embeds * self.m_emb
        # The FP32 residual stream, entered once.
        hidden_states = embeds.to(torch.float32)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                **kwargs,
            )
        return self.norm(hidden_states.to(self.dtype))


def validate_spec_config(spec_config: Any) -> None:
    """Admit only the speculative modes this decoder can serve.

    External drafters (NGram, user-provided) run outside the model and need
    nothing from it. The suffix automaton is one-engine and weight-free, so
    `SpecDecOneEngineForCausalLM` runs its worker inside this forward.
    DFlash is one-engine but its drafter weights come from the separate
    `speculative_model` checkpoint; the decoder layers feed it through
    `spec_metadata.maybe_capture_hidden_states`. The remaining one-engine
    modes (MTP, Eagle3, PARD, DSpark) need draft weights the target checkpoint
    does not carry.
    """
    if spec_config is None:
        return
    mode = spec_config.spec_dec_mode
    if mode.use_one_engine() and not (mode.is_sa() or mode.is_dflash()):
        raise ValueError(
            "MuseSparkOpen supports speculative decoding only with external "
            "drafters (NGram, user-provided), the suffix automaton, or DFlash; "
            f"{mode.name} needs draft weights the checkpoint does not carry"
        )


# Not registered under an architecture name: the checkpoint's architecture,
# MuseSparkForConditionalGeneration, is the multimodal wrapper in
# modeling_musespark_open_omni, which builds this decoder. Registration is
# first-wins and the wrapper imports this module, so a registration here would
# take the slot and the perception towers would never be built.
class MuseSparkOpenForCausalLM(
    SpecDecOneEngineForCausalLM[MuseSparkOpenModel, MuseSparkOpenConfig]
):
    def __init__(self, model_config: ModelConfig[MuseSparkOpenConfig]):
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.exclude_modules:
            quant_config.exclude_modules = normalize_exclude_modules(quant_config.exclude_modules)
        config = model_config.pretrained_config
        if model_config.mapping.pp_size != 1:
            raise ValueError("MuseSparkOpen does not support pipeline parallelism")
        validate_spec_config(getattr(model_config, "spec_config", None))
        super().__init__(
            MuseSparkOpenModel(model_config),
            model_config,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )
        self.logits_processor = MuseSparkOpenLogitsProcessor(
            logit_multiplier=config.metap_multiplier, soft_cap=config.output_soft_cap_temp
        )

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """V2 sizes the 2048-key sliding-window pool and the full-context pool
        independently instead of splitting one budget statically."""
        return "V2"

    @classmethod
    def requires_kv_cache_manager_v2(cls, pretrained_config: Any = None) -> bool:
        """The V1 manager is not a supported route for this model.

        Routes that cannot keep V2 (disaggregated serving outside NIXL with
        the Python transceiver) are refused at configuration time rather than
        demoted to V1.
        """
        return True

    def load_weights(
        self, weights: Dict[str, Any], weight_mapper: Optional[Any] = None, **kwargs
    ) -> Dict[str, int]:
        """Load the checkpoint and return the key-accounting summary.

        The generic module walk cannot express this checkpoint's stacked
        experts, per-half expert scales, fused query/output-gate projection, or
        its out-of-range layer keys, so loading is fully owned here.
        """
        from .checkpoints.hf.musespark_open_weight_mapper import load_musespark_open_weights

        return load_musespark_open_weights(self, weights)

    def infer_max_seq_len(self) -> int:
        return self.config.max_position_embeddings


def head_interleaved_q_gate(
    q_weight: torch.Tensor, gate_weight: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Interleave the query and output-gate projections head by head.

    `Attention.preprocess_qkv` reads the first `2 * q_size` columns as
    `[tokens, num_heads, 2 * head_dim]` and splits the last dim in half, so
    each head's query rows must be immediately followed by that head's gate
    rows. Column-parallel sharding then keeps each rank's heads (and their
    gates) contiguous.
    """
    hidden = q_weight.shape[-1]
    q_heads = q_weight.reshape(num_heads, head_dim, hidden)
    gate_heads = gate_weight.reshape(num_heads, head_dim, hidden)
    return torch.stack([q_heads, gate_heads], dim=1).reshape(num_heads * 2 * head_dim, hidden)
