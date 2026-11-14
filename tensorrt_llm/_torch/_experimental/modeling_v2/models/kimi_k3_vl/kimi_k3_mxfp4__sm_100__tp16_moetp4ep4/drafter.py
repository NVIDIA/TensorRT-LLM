# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter from a standalone GQA checkpoint (a Qwen3-shaped backbone, e.g. RadixArk-Kimi-K3-DSpark), built
from operations.

Per layer: RMSNorm, the fused q / k / v projection, the q / k RMSNorm with NeoX RoPE, block attention over the request's
paged context and its own block, the output projection and its TP all-reduce, RMSNorm, the SiLU-and-mul MLP and its
TP all-reduce; then the final norm. The context projection ``fc`` with ``hidden_norm``, the Markov and confidence heads
and the mask row come from the draft checkpoint; the embedding and the LM head are the target's.

`K3DSparkDrafter` keeps the attribute set the speculative worker reads off a DFlash / DSpark drafter (`WORKER_FIELDS`).
The worker that calls it is this target's ``KimiK3DSparkWorker`` (``spec_worker.py``).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.fused_qk_norm_rope import (
    fused_qk_norm_rope,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import (
    drafter_config as _drafter_config,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.drafter_config import (
    head_config,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.comm.k3_allreduce import (
    K3AllReduce,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._utils import get_hf_rope_theta
from tensorrt_llm.functional import RopeEmbeddingUtils, RotaryScalingType
from tensorrt_llm.logger import logger

from . import decode_comm as _decode_comm
from . import decode_gemv as _decode_gemv
from .norm import RMSNorm
from .ops.activation.k3_draft_silu_and_mul import k3_draft_silu_and_mul
from .ops.attention import k3_draft_rope as _rope_op
from .ops.attention import k3_dspark_block_attn as _block_attn_op
from .ops.attention import k3_dspark_dense_block_attn as _dense_attn_op
from .ops.attention.k3_draft_rope import k3_draft_rope
from .ops.attention.k3_drafter_attn_qknorm import k3_drafter_attn_qknorm
from .ops.attention.k3_dspark_block_attn import k3_dspark_block_attn
from .ops.attention.k3_dspark_dense_block_attn import k3_dspark_dense_block_attn
from .ops.spec.k3_dspark_markov_chain import k3_dspark_markov_chain
from .step import MAX_REQUESTS, MAX_TOKENS_PER_REQUEST

#: The (requests, block tokens) splits attention/k3_drafter_attn_qknorm certifies; R x 7 is DSpark's block at
#: max_draft_len 7.
DRAFTER_ATTN_SPLITS = frozenset(
    {(1, 1), (1, 8), (2, 4), (4, 2), (8, 1), (3, 1), (2, 8), (8, 8)} | {(r, 7) for r in range(1, 9)}
)

# The drafter layout the decode entries certify: 6 query heads and 1 KV head of 64 per rank, context pages of 64
# rows, the q / k RMSNorm's epsilon and the NeoX RoPE's base.
_DRAFTER_HEADS = (6, 1)


_DRAFTER_HEAD_DIM = 64


_DRAFTER_PAGE = 64


_DRAFTER_EPS = 1e-5


_DRAFTER_ROPE_BASE = 10000.0

#: Every name the speculative runtime reads off a DFlash / DSpark drafter it is handed (``speculative/dflash.py``,
#: ``speculative/dspark.py``, ``pyexecutor/_util.py``, ``pyexecutor/model_loader.py``) or that this target's worker
#: reads (``spec_worker.py``). The worker reads them duck-typed; the shell test checks the drafter has each.
WORKER_FIELDS = (
    # Identity, config and heads.
    "config",
    "model_config",
    "model",
    "draft_model_full",
    "lm_head",
    "logits_processor",
    "block_size",
    "mask_token_id",
    "mask_token_embedding",
    "target_layer_ids",
    "logit_multiplier",
    "logit_softcap",
    "apply_draft_logit_transform",
    "_dspark_shift_label",
    "has_markov_head",
    "markov_w1",
    "markov_w2",
    "apply_markov_chain_logits",
    # Which optional DFlash features it has (none).
    "is_dflash2",
    "has_block_conv",
    "has_candidate_selector",
    "candidate_selector",
    "lilicorr",
    "_uses_lilicorr",
    # The attention backend and the context cache's geometry.
    "dflash_attention_backend",
    "_uses_worker_attention_backend",
    "_paged_ctx_cache",
    "_kv_factor",
    "_runtime_position_ceiling",
    "validate_block_attention_windows",
    "_build_fused_kv_buffers",
    "_num_attn_layers",
    "_num_heads",
    "_num_kv_heads",
    "_head_dim",
    # The context projection and the context K / V (the K3 worker's k3_ctx_kv reads the fused weights).
    "fc",
    "hidden_norm",
    "project_target_hidden",
    "precompute_context_kv",
    "_fused_kv_weight",
    "_fused_kv_bias",
    "_k_norm_stacked",
    "_k_norm_eps",
    "_input_ln_eps",
    "_is_neox",
    "_get_cos_sin_cache",
    # The block forward and the weight load.
    "dflash_forward",
    "load_weights",
    "load_weights_from_target_model",
)

# The DSpark head tensors and the names they ship under.
_HEAD_WEIGHT_ALIASES = {
    "markov_w1.weight": ("markov_head.markov_w1.weight", "markov_w1.weight"),
    "markov_w2.weight": ("markov_head.markov_w2.weight", "markov_w2.weight"),
    "confidence_proj.weight": ("confidence_head.proj.weight", "confidence_proj.weight"),
    "confidence_proj.bias": ("confidence_head.proj.bias", "confidence_proj.bias"),
}

# The wrapper-owned tensors every draft checkpoint ships.
_WRAPPER_OWNED_WEIGHTS = ("fc.weight", "hidden_norm.weight")


def resolve_attention_backend(requested: str) -> str:
    """The drafter's block attention for ``attention_backend`` ``requested``, resolved as the built-in GQA DSpark
    drafter resolves it: TRTLLM (flashinfer TRTLLM-Gen on the manager's paged context) or VANILLA (flash-attn on the
    worker's private arena); AUTO is TRTLLM where this build runs it, else VANILLA. ``route`` checks the same refusals
    first (``drafter_config.gqa_attention_backend_refusal``), except VANILLA's flash-attn.

    Raises:
        ValueError: a name the drafter does not take (CUTEDSL among them), or an explicit TRTLLM this build cannot
            run.
        NoTarget: FA4 (``drafter_config.gqa_attention_backend_declines``).
        RuntimeError: VANILLA without flash-attn.
    """
    reason = _drafter_config.gqa_unknown_attention_backend(requested)
    if reason is not None:
        raise ValueError(reason)
    reason = _drafter_config.gqa_attention_backend_declines(requested)
    if reason is not None:
        raise NoTarget(f"the Kimi K3 DSpark drafter does not build this drafter: {reason}")
    if requested in ("AUTO", "TRTLLM"):
        reason = _block_attn_op.unavailability_reason()
        if reason is None:
            return "TRTLLM"
        if requested == "TRTLLM":
            raise ValueError(
                _drafter_config.requested_backend_unavailable("K3DSparkDrafter", requested, reason)
            )
        logger.info(
            f"K3DSparkDrafter prefers the TRTLLM attention backend but it is unavailable ({reason}); falling back "
            "to VANILLA."
        )
    _dense_attn_op.flash_attention()
    return "VANILLA"


class DraftRope:
    """The drafter's RoPE parameters, read from its config the way the engine's attention backends read a model's
    (``attention.backends.interface.RopeParams.from_config``; the test compares the two), and their cos / sin
    table: ``scale_type``, ``theta``, ``max_positions`` and the table's other inputs."""

    def __init__(self, config) -> None:
        self.dim = 0
        self.theta = 10000.0
        self.alpha = 1.0
        self.scale_type = RotaryScalingType.none
        self.scale = 1.0
        self.low_freq_factor = 1.0
        self.high_freq_factor = 4.0
        self.max_positions = 1024
        self.original_max_positions = 1024
        self.beta_fast = 32
        self.beta_slow = 1
        self.mscale = 1.0
        self.mscale_all_dim = 0.0
        self.short_factor = None
        self.long_factor = None
        self.max_seq_len = None
        self.duplicate_data = False
        self._read(config)

    def _read(self, config) -> None:
        hf_rope_parameters = getattr(config, "rope_parameters", None)
        normalized_rope_parameters = hf_rope_parameters
        if hf_rope_parameters is not None:
            try:
                from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
            except ImportError:
                from transformers.configuration_utils import (
                    ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES,
                )
            if set(hf_rope_parameters.keys()).issubset(ALLOWED_ATTENTION_LAYER_TYPES):
                # Per-layer-type RoPE parameters: the full-attention layers' (every drafter layer is one).
                if "full_attention" in hf_rope_parameters:
                    normalized_rope_parameters = hf_rope_parameters["full_attention"]
                else:
                    fallback_key = next(iter(hf_rope_parameters))
                    logger.warning(
                        f"Per-layer-type rope_parameters has no 'full_attention' entry; falling back to "
                        f"'{fallback_key}'. Available layer types: {list(hf_rope_parameters.keys())}."
                    )
                    normalized_rope_parameters = hf_rope_parameters[fallback_key]
            config.update(normalized_rope_parameters)

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = hidden_size // num_attention_heads
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None and normalized_rope_parameters is not None:
            rope_scaling = normalized_rope_parameters
        self.max_positions = config.max_position_embeddings
        self.theta = get_hf_rope_theta(config, 10000.0)
        rope_percentage = (
            getattr(config, "rotary_pct", None)
            or getattr(config, "partial_rotary_factor", None)
            or 1.0
        )
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", None)
        self.dim = (
            getattr(config, "rotary_dim", None)
            or getattr(config, "rotary_emb_base", None)
            or qk_rope_head_dim
            or int(head_dim * rope_percentage)
        )
        if qk_rope_head_dim is not None:
            self.duplicate_data = True
        self.scale_type = RotaryScalingType.none
        self.scale = 1.0
        if rope_scaling is not None:
            self.alpha = rope_scaling.get("alpha", 1.0)
            rotary_scaling_type = rope_scaling.get("type", None) or rope_scaling.get("rope_type")
            self.scale_type = RotaryScalingType.from_string(rotary_scaling_type)
            self.scale = rope_scaling.get("factor", 1.0)
            self.low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
            self.high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
            self.original_max_positions = (
                getattr(config, "original_max_position_embeddings", None)
                or rope_scaling.get("original_max_position_embeddings", None)
                or 1024
            )
            self.beta_fast = rope_scaling.get("beta_fast", 32)
            self.beta_slow = rope_scaling.get("beta_slow", 1)
            self.mscale = rope_scaling.get("mscale", 1.0)
            self.mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
            if "short_factor" in rope_scaling:
                self.short_factor = tuple(rope_scaling["short_factor"])
            if "long_factor" in rope_scaling:
                self.long_factor = tuple(rope_scaling["long_factor"])
        elif config.model_type == "deepseek_v3":
            self.scale_type = RotaryScalingType.yarn
        self.max_seq_len = getattr(config, "max_seq_len", None)

    def cos_sin(self, device="cuda") -> torch.Tensor:
        """The non-interleaved cos / sin table, ``[1, max_positions * dim]`` fp32 on ``device``: per position the
        ``dim / 2`` cosines, then the ``dim / 2`` sines."""
        if self.scale_type == RotaryScalingType.yarn:
            _, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                self.max_positions,
                self.dim,
                self.theta,
                self.scale,
                self.original_max_positions,
                self.beta_fast,
                self.beta_slow,
                self.mscale,
                self.mscale_all_dim,
                self.duplicate_data,
            )
        elif self.scale_type == RotaryScalingType.longrope:
            _, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_long_rope(
                num_pos=self.max_positions,
                dim=self.dim,
                theta=self.theta,
                original_max_pos=self.original_max_positions,
                short_factor=self.short_factor,
                long_factor=self.long_factor,
                max_seq_len=self.max_seq_len,
                duplicate_data=self.duplicate_data,
            )
        else:
            _, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                self.max_positions,
                self.dim,
                self.theta,
                self.scale,
                self.scale_type,
                rope_scaling_config={
                    "factor": self.scale,
                    "alpha": self.alpha,
                    "low_freq_factor": self.low_freq_factor,
                    "high_freq_factor": self.high_freq_factor,
                    "original_max_position_embeddings": self.original_max_positions,
                },
                duplicate_data=self.duplicate_data,
            )
        rope_cos_sin = (
            rope_cos_sin.reshape(self.max_positions, -1, 2)[:, : self.dim // 2, :]
            .transpose(0, 2, 1)
            .reshape(1, -1)
        )
        return torch.tensor(rope_cos_sin, dtype=torch.float32, device=device)


def fc_columns(in_features: int, tp_size: int, tp_rank: int) -> Optional[Tuple[int, int]]:
    """Rank ``tp_rank``'s input columns ``[start, end)`` of the drafter's context projection ``fc`` split by input
    feature over ``tp_size`` ranks: equal contiguous blocks in rank order. None where they do not split evenly."""
    if tp_size < 1 or not 0 <= tp_rank < tp_size or in_features <= 0 or in_features % tp_size:
        return None
    width = in_features // tp_size
    return tp_rank * width, (tp_rank + 1) * width


class K3FcSlice(nn.Module):
    """This rank's block of the DSpark drafter's context projection ``fc`` (`fc_columns`): ``weight`` is
    ``fc.weight[:, start:end]``, contiguous. Its output is this rank's partial product; the sum over the TP group's
    ranks is ``fc``'s output."""

    def __init__(self, weight: torch.Tensor, start: int, end: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.start = start
        self.end = end

    @classmethod
    def of(cls, fc_weight: torch.Tensor, tp_size: int, tp_rank: int) -> Optional["K3FcSlice"]:
        """Rank ``tp_rank``'s block of the full ``fc_weight`` ``[out_features, in_features]``: a copy of its columns
        (on one rank, the weight itself); None where the input columns do not split evenly over ``tp_size`` ranks."""
        columns = fc_columns(fc_weight.shape[1], tp_size, tp_rank)
        if columns is None:
            return None
        start, end = columns
        return cls(fc_weight.detach()[:, start:end].contiguous(), start, end)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_states[:, start:end] @ weight.T`` for the full features ``hidden_states`` ``[N, in_features]``: the
        GEMM reads the column block in place, through the rows' stride."""
        return torch.nn.functional.linear(hidden_states[:, self.start : self.end], self.weight)


# ----------------------------------------------------------------------------------------------------------------------
# The backbone's parameters: plain holders, read by the block forward below.
# ----------------------------------------------------------------------------------------------------------------------


class DraftLinear(nn.Module):
    """A bias-free projection: ``weight`` [out, in], this rank's shard. A row-parallel one has its TP ``all_reduce``
    (None on one rank). Calling it is the projection then the all-reduce."""

    bias = None

    def __init__(self, out_features: int, in_features: int, dtype: torch.dtype, all_reduce=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype), requires_grad=False
        )
        self.all_reduce = all_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(F.linear(x, self.weight))

    def reduce(self, partial: torch.Tensor) -> torch.Tensor:
        """``partial``, this rank's projection output, summed over the TP group (unchanged without an all-reduce)."""
        return partial if self.all_reduce is None else self.all_reduce(partial)


class DraftAttention(nn.Module):
    """One layer's attention parameters: the fused ``qkv_proj`` (column shard), ``o_proj`` (row shard), the per-head
    ``q_norm`` / ``k_norm``, and this rank's head counts."""

    def __init__(self, cfg, tp_size: int, dtype: torch.dtype, all_reduce):
        super().__init__()
        self.head_dim = int(
            getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        )
        self.num_heads = cfg.num_attention_heads // tp_size
        self.num_key_value_heads = cfg.num_key_value_heads // tp_size
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        hidden = cfg.hidden_size
        self.qkv_proj = DraftLinear(self.q_size + 2 * self.kv_size, hidden, dtype)
        self.o_proj = DraftLinear(hidden, self.q_size, dtype, all_reduce)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps, dtype=dtype)


class DraftMLP(nn.Module):
    """One layer's SiLU MLP parameters: the fused ``gate_up_proj`` (column shard, ``[gate; up]``) and ``down_proj``
    (row shard)."""

    def __init__(self, cfg, tp_size: int, dtype: torch.dtype, all_reduce):
        super().__init__()
        self.intermediate_size = cfg.intermediate_size // tp_size
        self.gate_up_proj = DraftLinear(2 * self.intermediate_size, cfg.hidden_size, dtype)
        self.down_proj = DraftLinear(cfg.hidden_size, self.intermediate_size, dtype, all_reduce)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(k3_draft_silu_and_mul(F.linear(x, self.gate_up_proj.weight)))


class DraftLayer(nn.Module):
    def __init__(self, cfg, tp_size: int, dtype: torch.dtype, make_all_reduce):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.self_attn = DraftAttention(cfg, tp_size, dtype, make_all_reduce())
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.mlp = DraftMLP(cfg, tp_size, dtype, make_all_reduce())


class DraftBackbone(nn.Module):
    """The drafter's layers and final norm. ``embed_tokens`` is the target's (`load_weights_from_target_model`)."""

    def __init__(self, cfg, tp_size: int, dtype: torch.dtype, make_all_reduce):
        super().__init__()
        self.layers = nn.ModuleList(
            DraftLayer(cfg, tp_size, dtype, make_all_reduce) for _ in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.embed_tokens = None
        # No draft-to-target vocabulary map: the drafter shares the target's vocabulary.
        self.d2t = None


class DraftModelFull(nn.Module):
    """Where the speculative worker looks for the embedding (``model.embed_tokens``) and the LM head."""

    def __init__(self, model: DraftBackbone):
        super().__init__()
        self.model = model
        self.lm_head = None


# ----------------------------------------------------------------------------------------------------------------------
# The drafter.
# ----------------------------------------------------------------------------------------------------------------------


class K3DSparkDrafter(nn.Module):
    """Kimi K3's DSpark drafter from a standalone Qwen3-shaped GQA checkpoint.

    The block forward (``dflash_forward``) runs one of two paths:

    * **The drafter entries**, a decode step's block where they take it: per layer the fused q / k / v projection on
      the ``drafter_qkv`` decode GEMV site; ``attention/k3_drafter_attn_qknorm`` (the q / k RMSNorm, the NeoX RoPE,
      and each request's block attending to its paged context and its own k / v in one launch, the block's k / v not
      stored); the output projection on ``drafter_o`` and its all-reduce; gate / up on ``drafter_gate_up`` and the
      down projection with the SiLU-and-mul on ``drafter_down``, then its all-reduce. A projection whose site does not
      take its rows runs ``F.linear``. They take a block of a split `DRAFTER_ATTN_SPLITS` lists, on 6 / 1 heads of 64
      with plain NeoX RoPE of base 1e4, bf16 64-row pages (``attention_backend`` TRTLLM only), and under
      CUDA-graph capture only a compile key of the attention that ran eagerly.
    * **The generic block**, everything else: per layer RMSNorm, the q / k / v projection, the q / k RMSNorm and RoPE
      (``catalog fused_qk_norm_rope`` for plain RoPE, else the per-head RMSNorm and ``k3_draft_rope`` on the drafter's
      cos / sin table, YaRN included), the block attention (the block's k / v stored after the context, then every
      row attending to context and block: ``k3_dspark_block_attn``, TRTLLM-Gen over the manager's pages, on
      ``attention_backend`` TRTLLM; ``k3_dspark_dense_block_attn``, flash-attn over the worker's arena, on VANILLA),
      the output projection and its all-reduce, the fused residual RMSNorm, the SiLU-and-mul MLP and its all-reduce;
      then the final norm. These are the ops of the built-in DFlash drafter's block forward on its TRTLLM and VANILLA
      backends, in its order.

    Where the target hands the drafter the TP group's collective state (``use_decode_comm``):

    * the context projection's ``fc`` holds only this rank's block of input columns (`K3FcSlice`), and
      ``project_target_hidden`` sums the ranks' partial products with ``hidden_norm`` in one all-reduce;
    * a block the entries take runs each residual add and RMSNorm in the all-reduce before it (o_proj's with the
      post-attention norm, the MLP's with the next layer's input norm, the last one's with the final norm), the
      block's input serving as the first residual, uncopied. Up to 8 rows ``comm/k3_sandwich_plain`` runs the
      projection, its all-reduce and the norm in one launch (o_proj, and with the SiLU-and-mul the down projection
      after ``drafter_gate_up``); otherwise the projection (its site, or ``F.linear``) is followed by
      ``comm/mnnvl_fusion_allreduce``. A block whose rows the MNNVL workspace does not hold keeps the module
      all-reduces and norms.
    """

    #: Attention backends' fields the DFlash worker reads.
    _uses_worker_attention_backend = True
    _paged_ctx_cache = False
    _uses_lilicorr = False
    # Cache halves per token per layer: K and V.
    _kv_factor = 2

    def __init__(
        self, draft_config: ModelConfig, *, dflash_attention_backend: str = "AUTO"
    ) -> None:
        super().__init__()
        cfg = draft_config.pretrained_config
        mapping = draft_config.mapping
        reason = _drafter_config.gqa_declines(cfg) or _drafter_config.gqa_tp_declines(cfg, mapping)
        if reason is not None:
            raise NoTarget(f"the Kimi K3 DSpark drafter does not build this drafter: {reason}")
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        self.model_config = draft_config
        self.config = cfg
        self.dflash_attention_backend = resolve_attention_backend(dflash_attention_backend)
        dtype = torch.bfloat16

        def make_all_reduce():
            # A row-parallel projection's TP all-reduce (the family's K3AllReduce: the MNNVL all-reduce, which
            # use_decode_one_shot gives the decode ceiling; the catalog all-reduce off MNNVL), on the drafter's mapping.
            if tp_size == 1:
                return None
            return K3AllReduce(mapping, draft_config.allreduce_strategy, dtype)

        self.model = DraftBackbone(cfg, tp_size, dtype, make_all_reduce)
        self.draft_model_full = DraftModelFull(self.model)
        self.lm_head = None
        self.logits_processor = None  # set by the shell

        dflash_config = getattr(cfg, "dflash_config", None) or {}
        self.mask_token_id = dflash_config.get(
            "mask_token_id", getattr(cfg, "mask_token_id", cfg.vocab_size)
        )
        self.mask_token_embedding: Optional[torch.Tensor] = None
        self.target_layer_ids = dflash_config.get("target_layer_ids", None)
        self.block_size = dflash_config.get("block_size", getattr(cfg, "block_size", None))
        self.logit_multiplier, self.logit_softcap = _drafter_config.logit_transform(dflash_config)

        # The DSpark heads: shift_label slots on by default, the vanilla Markov head, the confidence head (loaded,
        # unused: confidence-scheduled verification is not implemented).
        shift_label = head_config(cfg, "shift_label")
        self._dspark_shift_label = True if shift_label is None else bool(shift_label)
        self._dspark_markov_rank = int(head_config(cfg, "markov_rank") or 0)
        self._dspark_use_confidence_head = bool(head_config(cfg, "use_confidence_head") or False)
        self.markov_w1 = None
        self.markov_w2 = None
        self.confidence_proj_weight = None
        self.confidence_proj_bias = None
        if self._dspark_use_confidence_head:
            logger.warning(
                "DSpark drafter declares use_confidence_head; confidence-scheduled verification is not implemented "
                "yet (confidence_proj weights are loaded but unused, drafting always proposes the full K tokens)."
            )

        # The DFlash features this drafter does not have.
        self.is_dflash2 = False
        self.has_block_conv = False
        self.has_candidate_selector = False
        self.candidate_selector = None
        self.lilicorr = None
        # Positions the worker serves, published by its lazy buffer init.
        self._runtime_position_ceiling = None

        # Built from the checkpoint by load_weights.
        self.fc: Optional[nn.Module] = None
        self.hidden_norm: Optional[nn.RMSNorm] = None

        # RoPE: NeoX, full rotary, the drafter config's parameters (plain or YaRN).
        self._rope = DraftRope(cfg)
        self._is_neox = True
        self._rotary_cos_sin: Optional[torch.Tensor] = None
        self._cos_sin_cache_fp32: Optional[torch.Tensor] = None
        self._rope_dummy_q: Optional[torch.Tensor] = None
        # The fused q / k norm + RoPE kernel derives frequencies itself: only for plain RoPE.
        self._use_fused_qk_norm_rope = getattr(self._rope.scale_type, "name", "none") == "none"

        # The layers' K / V weights stacked for the context K / V (built after the load, _build_fused_kv_buffers).
        self._fused_kv_weight: Optional[torch.Tensor] = None
        self._fused_kv_bias = None
        self._k_norm_stacked: Optional[torch.Tensor] = None
        self._k_norm_eps: Optional[float] = None
        self._input_ln_eps = None
        attn0 = self.model.layers[0].self_attn
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads
        self._num_kv_heads = attn0.num_key_value_heads
        self._head_dim = attn0.head_dim

        # TRTLLM-Gen's scratch and the block's append indices, allocated eagerly before any capture.
        self._dflash_trtllm_gen_workspace: Optional[torch.Tensor] = None
        self._dflash_trtllm_gen_counters: Optional[torch.Tensor] = None
        self._dflash_trtllm_gen_device = None
        self._dflash_trtllm_gen_sm_count = None
        self._dflash_batch_indices: Optional[torch.Tensor] = None
        self._dflash_block_offsets: Optional[torch.Tensor] = None

        # The drafter entries' state. The decode GEMVs' state (decode_gemv.py), shared with the target's layers.
        self.decode_gemvs: Optional[_decode_gemv.K3DecodeGemvs] = None
        # Whether every layer is one the drafter entries take; decided on the first block, the weights loaded.
        self._k3_take: Optional[bool] = None
        # The attention's compile keys that ran eagerly ((more than one request, page stride)).
        self._k3_attn_ran: set = set()
        # The TP group's collective state (decode_comm.py), set by the target (use_decode_comm).
        self.decode_comm: Optional[_decode_comm.K3DecodeComm] = None
        self._k3_zero_rows: Optional[torch.Tensor] = None
        self._k3_norms_fuse = False
        self._k3_sandwich_forms: frozenset = frozenset()
        logger.info(
            f"Kimi K3 DSpark drafter: {self._num_attn_layers} layers, {self._num_heads} / {self._num_kv_heads} heads "
            f"of {self._head_dim} per rank, mask_token_id {self.mask_token_id}, target_layer_ids "
            f"{self.target_layer_ids}, block_size {self.block_size}, RoPE {self._rope.scale_type}"
        )

    # ------------------------------------------------------------------------------------------------------------------
    # The heads and the shared embedding / LM head.
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def has_markov_head(self) -> bool:
        return self._dspark_markov_rank > 0 and self.markov_w1 is not None

    def apply_markov_chain_logits(
        self, base_logits: torch.Tensor, first_prev_tokens: torch.Tensor, tp_exchange=None
    ) -> torch.Tensor:
        """The vanilla-Markov intra-block bias on fp32 block logits (the full vocabulary, or this rank's shard with the
        TP group's ``tp_exchange``); ``base_logits`` unchanged without a Markov head."""
        if not self.has_markov_head:
            return base_logits
        return k3_dspark_markov_chain(
            base_logits, first_prev_tokens, self.markov_w1, self.markov_w2, tp_exchange
        )

    def apply_draft_logit_transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale and soft-cap draft logits the way the drafter was trained, in fp32 (in place when fp32)."""
        if self.logit_multiplier == 1.0 and self.logit_softcap is None:
            return logits
        logits = logits.float()
        if self.logit_multiplier != 1.0:
            logits.mul_(self.logit_multiplier)
        if self.logit_softcap is not None:
            logits.div_(self.logit_softcap).tanh_().mul_(self.logit_softcap)
        return logits

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Share the target's embedding and LM head."""
        self.model.embed_tokens = target_model.model.embed_tokens
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head

    def validate_block_attention_windows(self) -> list:
        """On TRTLLM every draft layer (non-causal, unwindowed) runs the decode kernel with the block's rows folded into
        the query heads, as the worker's TRTLLM checks expect; empty on VANILLA, as the built-in DSpark drafter
        returns."""
        if self.dflash_attention_backend != "TRTLLM":
            return []
        return [_block_attn_op.LOWERING] * self._num_attn_layers

    # ------------------------------------------------------------------------------------------------------------------
    # The weight load.
    # ------------------------------------------------------------------------------------------------------------------

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs) -> None:
        """Load the draft checkpoint (keys ``layers.{i}.*``, ``norm.weight``, ``fc.weight``, ``hidden_norm.weight``,
        the DSpark heads, and optionally the embedding, of which only the mask token's trained row is kept): this
        rank's shard of every projection, cut as the built-in DSpark drafter's loader cuts them. ``weight_mapper`` is
        not used. On the TP group's
        collective state ``fc`` is split again (`_k3_split_fc`)."""
        del weight_mapper, kwargs
        weights = self._take_head_weights(weights, self.model.norm.weight.device)
        mask_row = self._trained_mask_row(weights)
        weights = self._split_fused_qkv(weights)
        remapped = {}
        for key, value in weights.items():
            if key.startswith("model."):
                key = key[len("model.") :]
            remapped[key] = value
        missing = [k for k in _WRAPPER_OWNED_WEIGHTS if k not in remapped]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint is missing {missing}, which this wrapper owns and builds "
                "from the checkpoint."
            )
        # The drafter's own parameters' device (CUDA in the engine).
        device = self.model.norm.weight.device
        fc_weight = _tensor(remapped["fc.weight"])
        self.fc = nn.Linear(
            fc_weight.shape[1], fc_weight.shape[0], bias=False, device=device, dtype=fc_weight.dtype
        )
        self.fc.weight.data.copy_(fc_weight)
        norm_weight = _tensor(remapped["hidden_norm.weight"])
        self.hidden_norm = nn.RMSNorm(
            norm_weight.shape[0],
            eps=getattr(self.config, "rms_norm_eps", 1e-6),
            device=device,
            elementwise_affine=True,
            dtype=norm_weight.dtype,
        )
        self.hidden_norm.weight.data.copy_(norm_weight)
        self._load_backbone(remapped)
        if mask_row is not None:
            self.mask_token_embedding = mask_row.to(
                device=self.fc.weight.device, dtype=self.model_config.torch_dtype
            )
        self._k3_split_fc()

    def _load_backbone(self, weights: Dict) -> None:
        """This rank's shard of every layer tensor and the final norm; every one must be in ``weights``."""
        mapping = self.model_config.mapping
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        rank = 0 if tp_size == 1 else mapping.tp_rank
        missing = []

        def take(name: str):
            if name not in weights:
                missing.append(name)
                return None
            return weights[name]

        def rows(name: str):
            t = take(name)
            return None if t is None else _shard(t, 0, rank, tp_size)

        def cols(name: str):
            t = take(name)
            return None if t is None else _shard(t, 1, rank, tp_size)

        def full(name: str):
            t = take(name)
            return None if t is None else _tensor(t)

        loads = []
        for i, layer in enumerate(self.model.layers):
            p = f"layers.{i}."
            attn, mlp = layer.self_attn, layer.mlp
            q, k, v = (rows(p + f"self_attn.{n}_proj.weight") for n in "qkv")
            gate, up = rows(p + "mlp.gate_proj.weight"), rows(p + "mlp.up_proj.weight")
            loads += [
                (attn.qkv_proj.weight, None if None in (q, k, v) else torch.cat((q, k, v))),
                (attn.o_proj.weight, cols(p + "self_attn.o_proj.weight")),
                (attn.q_norm.weight, full(p + "self_attn.q_norm.weight")),
                (attn.k_norm.weight, full(p + "self_attn.k_norm.weight")),
                (layer.input_layernorm.weight, full(p + "input_layernorm.weight")),
                (
                    layer.post_attention_layernorm.weight,
                    full(p + "post_attention_layernorm.weight"),
                ),
                (mlp.gate_up_proj.weight, None if None in (gate, up) else torch.cat((gate, up))),
                (mlp.down_proj.weight, cols(p + "mlp.down_proj.weight")),
            ]
        loads.append((self.model.norm.weight, full("norm.weight")))
        if missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint provides no weights for {missing[:8]}"
                f"{' ...' if len(missing) > 8 else ''}; loading would leave them uninitialized."
            )
        with torch.no_grad():
            for param, value in loads:
                if tuple(param.shape) != tuple(value.shape):
                    raise ValueError(
                        f"{type(self).__name__}: a checkpoint tensor of shape {tuple(value.shape)} for a parameter "
                        f"of shape {tuple(param.shape)}"
                    )
                param.data.copy_(value)

    def _split_fused_qkv(self, weights: Dict) -> Dict:
        """A checkpoint that ships a fused ``self_attn.qkv_proj`` split into q / k / v by the global head counts, as
        the built-in DSpark drafter's loader splits it."""
        if not any(k.endswith("self_attn.qkv_proj.weight") for k in weights):
            return weights
        cfg = self.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        q = cfg.num_attention_heads * head_dim
        kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * head_dim
        split = {}
        for k, v in weights.items():
            if k.endswith("self_attn.qkv_proj.weight"):
                b = k[: -len("qkv_proj.weight")]
                split[b + "q_proj.weight"] = v[:q]
                split[b + "k_proj.weight"] = v[q : q + kv]
                split[b + "v_proj.weight"] = v[q + kv :]
            else:
                split[k] = v
        return split

    def _take_head_weights(self, weights: Dict, device: torch.device) -> Dict:
        """Load the DSpark head tensors onto self; return the other weights."""
        heads, consumed = {}, set()
        for canonical, aliases in _HEAD_WEIGHT_ALIASES.items():
            for name in aliases:
                if name in weights:
                    heads[canonical] = weights[name]
                    consumed.add(name)
                    break
        if consumed:
            weights = {k: v for k, v in weights.items() if k not in consumed}
        if self._dspark_markov_rank <= 0 and "markov_w1.weight" in heads:
            raise ValueError(
                "DSpark drafter ships markov_w1/markov_w2 but markov_rank resolved to 0. The checkpoint's head "
                "switches were not found in dspark_config, dflash_config, or at the top level; loading it would "
                "drop the Markov head silently."
            )
        if self._dspark_markov_rank > 0:
            vocab, rank = self.config.vocab_size, self._dspark_markov_rank
            for k in ("markov_w1.weight", "markov_w2.weight"):
                if k not in heads:
                    raise ValueError(
                        f"DSpark drafter declares markov_rank={rank} but the checkpoint is missing {k}."
                    )
                if _shape(heads[k]) != (vocab, rank):
                    raise ValueError(
                        f"DSpark {k} has shape {_shape(heads[k])}, expected [vocab, markov_rank] = "
                        f"({vocab}, {rank})."
                    )
            self.markov_w1 = _tensor(heads["markov_w1.weight"]).to(device)
            self.markov_w2 = _tensor(heads["markov_w2.weight"]).to(device)
        if "confidence_proj.weight" in heads:
            self.confidence_proj_weight = _tensor(heads["confidence_proj.weight"]).to(device)
        if "confidence_proj.bias" in heads:
            self.confidence_proj_bias = _tensor(heads["confidence_proj.bias"]).to(device)
        return weights

    def _trained_mask_row(self, weights: Dict) -> Optional[torch.Tensor]:
        """The checkpoint's own embedding row for the mask token, [hidden_size]; None if it ships no embedding.

        Raises:
            ValueError: the embedding is not ``[rows, hidden_size]``, or the mask id is not one of its rows.
        """
        names = [k for k in ("embed_tokens.weight", "model.embed_tokens.weight") if k in weights]
        if not names:
            return None
        name = names[0]
        weight = weights[name]
        shape = _shape(weight)
        hidden_size = self.config.hidden_size
        if len(shape) != 2 or shape[1] != hidden_size:
            raise ValueError(
                f"{type(self).__name__}: the checkpoint's {name} has shape {list(shape)}, "
                f"expected [rows, hidden_size={hidden_size}]."
            )
        if 0 <= self.mask_token_id < shape[0]:
            return weight[self.mask_token_id]
        if self.mask_token_id == self.config.vocab_size:
            logger.warning(
                f"{type(self).__name__}: the checkpoint ships {name}, but mask_token_id is DFlash's default "
                f"({self.mask_token_id} = vocab_size, outside its {shape[0]} rows): no trained mask row is "
                "kept, and the masked slots use the shared embedding's row."
            )
            return None
        raise ValueError(
            f"{type(self).__name__}: mask_token_id {self.mask_token_id} is not a row of the checkpoint's {name} "
            f"({shape[0]} rows)."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # The context projection and the context K / V.
    # ------------------------------------------------------------------------------------------------------------------

    def use_decode_comm(self, comm: _decode_comm.K3DecodeComm) -> None:
        """Run the drafter's collectives on ``comm``, the TP group's decode state (``decode_comm.K3DecodeComm``) the
        target builds where every attention all-reduce runs over MNNVL. The target calls it once the weights are
        loaded, before any CUDA-graph capture. Collective: every rank of the group calls it at the same point.

        * The context projection's ``fc`` keeps only this rank's block of input columns (`K3FcSlice`), where they
          split evenly over the group and the drafter's layers have their TP all-reduce: ``project_target_hidden``
          sums the ranks' partial products and applies ``hidden_norm`` in one ``comm/mnnvl_fusion_allreduce`` call.
        * Where the layers' norms and projections take it (`_k3_norms_take_comm`), a block's residual adds and
          RMSNorms run in its all-reduces (``_k3_block_forward``). Each form of ``comm/k3_sandwich_plain`` whose
          shape every layer shares compiles here, with one call on a zero row of a zero weight; a form that does not
          compile is not used."""
        self.decode_comm = comm
        self._k3_split_fc()
        self._k3_norms_fuse = self._k3_norms_take_comm()
        forms = set()
        if self._k3_norms_fuse:
            layers = self.model.layers
            for form, swiglu, weights in (
                ("o_proj", False, [layer.self_attn.o_proj.weight for layer in layers]),
                ("down", True, [layer.mlp.down_proj.weight for layer in layers]),
            ):
                if len({tuple(w.shape) for w in weights}) == 1 and comm.compile_plain(
                    weights[0], swiglu=swiglu
                ):
                    forms.add(form)
        self._k3_sandwich_forms = frozenset(forms)
        logger.info(
            "Kimi K3 DSpark drafter: residual adds and RMSNorms "
            + (
                f"in the all-reduces (k3_sandwich_plain: {sorted(forms) or 'none'}, else "
                "mnnvl_fusion_allreduce)"
                if self._k3_norms_fuse
                else "on the module norms (a layer norm or projection the fused all-reduces do not reproduce)"
            )
        )

    def _k3_norms_take_comm(self) -> bool:
        """Whether a block's residual adds and RMSNorms can run in its all-reduces: every norm is a bf16 RMSNorm of
        the hidden width and every output and down projection has its TP all-reduce."""
        hidden = self.config.hidden_size

        def plain(norm: nn.Module) -> bool:
            weight = getattr(norm, "weight", None)
            return (
                isinstance(weight, torch.Tensor)
                and weight.dtype == torch.bfloat16
                and tuple(weight.shape) == (hidden,)
                and hasattr(norm, "variance_epsilon")
            )

        return plain(self.model.norm) and all(
            plain(layer.input_layernorm)
            and plain(layer.post_attention_layernorm)
            and layer.self_attn.o_proj.all_reduce is not None
            and layer.mlp.down_proj.all_reduce is not None
            for layer in self.model.layers
        )

    def _k3_tp_all_reduce(self) -> Optional[nn.Module]:
        """The drafter's own TP all-reduce (its first output projection's), or None."""
        return self.model.layers[0].self_attn.o_proj.all_reduce

    def _k3_split_fc(self) -> None:
        """Replace the replicated ``fc`` with this rank's block of its input columns (`K3FcSlice`) and size the zero
        residual of the projection's all-reduce, where the drafter runs on the TP group's collective state, ``fc`` is
        a bias-free bf16 projection whose input columns split evenly over the group, and the drafter has its TP
        all-reduce (for rows the MNNVL workspace does not hold). Otherwise ``fc`` stays replicated."""
        fc = getattr(self, "fc", None)
        if self.decode_comm is None or fc is None or isinstance(fc, K3FcSlice):
            return
        mapping = self.model_config.mapping
        weight = getattr(fc, "weight", None)
        sliced = None
        if (
            isinstance(weight, torch.Tensor)
            and weight.dim() == 2
            and weight.dtype == torch.bfloat16
            and getattr(fc, "bias", None) is None
            and self._k3_tp_all_reduce() is not None
        ):
            sliced = K3FcSlice.of(weight, mapping.tp_size, mapping.tp_rank)
        if sliced is None:
            logger.info(
                "Kimi K3 DSpark drafter: fc stays replicated (it is not a bias-free bf16 projection whose input "
                f"columns split evenly over TP{mapping.tp_size}, or the layers have no TP all-reduce)"
            )
            return
        self.fc = sliced
        self._k3_zero_rows = weight.new_zeros(
            MAX_REQUESTS * MAX_TOKENS_PER_REQUEST, weight.shape[0]
        )
        logger.info(
            f"Kimi K3 DSpark drafter: fc split by input feature over TP{mapping.tp_size}: rank {mapping.tp_rank} "
            f"holds columns [{sliced.start}, {sliced.end}), hidden_norm in the all-reduce"
        )

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(hidden_states))`` of the captured target features ``[N, in_features]``.

        With ``fc`` split (`K3FcSlice`): this rank's partial product of its columns, then the sum over the TP group
        with ``hidden_norm`` applied in one ``comm/mnnvl_fusion_allreduce`` call (a zero residual) up to a decode
        step's rows the MNNVL workspace holds; more rows (a prefill) go through the drafter's TP all-reduce, then
        ``hidden_norm``. Otherwise the replicated projection."""
        fc = self.fc
        if not isinstance(fc, K3FcSlice):
            return self.hidden_norm(self.fc(hidden_states.to(self.fc.weight.dtype)))
        partial = fc(hidden_states.to(fc.weight.dtype))
        rows = partial.shape[0]
        zeros = self._k3_zero_rows
        if rows <= zeros.shape[0] and self.decode_comm.takes_allreduce_norm(rows, partial.shape[1]):
            normed, _ = self.decode_comm.allreduce_norm(partial, zeros[:rows], self.hidden_norm)
            return normed
        if rows == 0:
            return self.hidden_norm(partial)
        return self.hidden_norm(self._k3_tp_all_reduce()(partial))

    def _init_rope(self) -> None:
        """The cos / sin table of the drafter's RoPE, [max_positions, 2, head_dim / 2] (`DraftRope.cos_sin`)."""
        rope = self._rope
        self._rotary_cos_sin = rope.cos_sin().reshape(rope.max_positions, 2, -1)

    def _get_cos_sin_cache(self) -> torch.Tensor:
        """The flashinfer cos / sin cache, [max_positions, head_dim] fp32."""
        if self._cos_sin_cache_fp32 is None:
            if self._rotary_cos_sin is None:
                self._init_rope()
            max_pos = self._rotary_cos_sin.shape[0]
            self._cos_sin_cache_fp32 = (
                self._rotary_cos_sin.view(max_pos, -1).to(torch.float32).contiguous()
            )
        return self._cos_sin_cache_fp32

    def _build_fused_kv_buffers(self) -> None:
        """Stack the layers' K / V projection rows and k_norm weights for one context-K / V GEMM; after the load."""
        if self._fused_kv_weight is not None:
            return
        attns = [layer.self_attn for layer in self.model.layers]
        q_size, kv_size = attns[0].q_size, attns[0].kv_size
        self._fused_kv_weight = torch.cat(
            [a.qkv_proj.weight[q_size : q_size + 2 * kv_size] for a in attns], dim=0
        ).contiguous()
        self._k_norm_stacked = torch.stack([a.k_norm.weight.data for a in attns])
        self._k_norm_eps = attns[0].k_norm.variance_epsilon
        self._get_cos_sin_cache()

    def precompute_context_kv(
        self, projected_hidden: torch.Tensor, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Every layer's context K (k_norm, RoPE) and V of ``projected_hidden`` [N, hidden] (already
        ``hidden_norm(fc(.))``) at ``positions`` [N]: ``k``, ``v`` [N, L, KV, head_dim]."""
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        layers, nkv, hd = self._num_attn_layers, self._num_kv_heads, self._head_dim
        if projected_hidden.dtype != self._fused_kv_weight.dtype:
            projected_hidden = projected_hidden.to(self._fused_kv_weight.dtype)
        kv = F.linear(projected_hidden, self._fused_kv_weight).view(n, layers, 2, nkv, hd)
        k = kv[:, :, 0].contiguous()
        v = kv[:, :, 1].contiguous()
        k = F.rms_norm(k, (hd,), eps=self._k_norm_eps)
        k = k * self._k_norm_stacked.view(1, layers, 1, hd)
        positions_i32 = positions.view(-1).to(torch.int32)
        if layers > 1:
            positions_i32 = positions_i32.repeat_interleave(layers)
        k_flat = k.view(n * layers, nkv * hd)
        # flashinfer needs a query: a one-head scratch.
        dummy_q = self._rope_dummy_q
        if dummy_q is None or dummy_q.dtype != k_flat.dtype or dummy_q.shape[0] < k_flat.shape[0]:
            dummy_q = k_flat.new_empty(k_flat.shape[0], hd)
            self._rope_dummy_q = dummy_q
        k3_draft_rope(
            positions_i32, dummy_q[: k_flat.shape[0]], k_flat, hd, self._get_cos_sin_cache(), True
        )
        return k, v

    # ------------------------------------------------------------------------------------------------------------------
    # The block forward.
    # ------------------------------------------------------------------------------------------------------------------

    def dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache: Optional[torch.Tensor],
        ctx_v_cache: Optional[torch.Tensor],
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache: Optional[torch.Tensor] = None,
        ctx_page_table: Optional[torch.Tensor] = None,
        ctx_rows_start: Optional[int] = None,
        block_placement=None,
    ) -> torch.Tensor:
        """The block's hidden states ``[B * block, hidden]`` (``noise_embedding`` [B, block, hidden], positions
        ``query_positions`` [B, block], ``num_ctx_per_req[b]`` context rows of request b): on the drafter entries where
        they take it (the class docstring), else the generic block. The context is, on TRTLLM, the manager's pages
        ``ctx_kv_cache`` (per layer [pages, 2, KV, page, head_dim]) through rows ``ctx_cache_batch_idx`` of
        ``ctx_page_table``; on VANILLA, the worker's arena ``ctx_k_cache`` / ``ctx_v_cache`` ([slots, layers,
        columns, KV, head_dim]) at slots ``ctx_cache_batch_idx``. ``ctx_rows_start``: where ``ctx_cache_batch_idx`` is
        the contiguous rows ``[ctx_rows_start, ctx_rows_start + B)`` of the page table, their start; the entries then
        read those rows as a view. ``block_placement`` is not read: this drafter appends its block itself."""
        del block_placement
        if self.dflash_attention_backend == "VANILLA":
            if ctx_k_cache is None or ctx_v_cache is None:
                raise RuntimeError(
                    "K3DSparkDrafter on VANILLA requires the context arena (ctx_k_cache, ctx_v_cache)."
                )
        elif ctx_kv_cache is None or ctx_page_table is None:
            raise RuntimeError(
                f"DFlash {self.dflash_attention_backend} requires a paged context cache and page table."
            )
        elif any(cache.dtype != noise_embedding.dtype for cache in ctx_kv_cache):
            # Before any buffer or page is touched, as the built-in DFlash drafter refuses it.
            raise ValueError(
                "DFlash paged KV cache must use the draft activation dtype; "
                "quantized context-cache writes are not supported."
            )
        keys = self._k3_block_keys(noise_embedding, ctx_kv_cache, ctx_page_table)
        if keys is None:
            return self._generic_block(
                noise_embedding,
                query_positions,
                num_ctx_per_req,
                ctx_k_cache,
                ctx_v_cache,
                ctx_cache_batch_idx,
                ctx_kv_cache,
                ctx_page_table,
            )
        out = self._k3_block_forward(
            noise_embedding,
            query_positions,
            num_ctx_per_req,
            ctx_cache_batch_idx,
            ctx_kv_cache,
            ctx_page_table,
            ctx_rows_start,
        )
        if not torch.cuda.is_current_stream_capturing():
            self._k3_attn_ran |= keys
        return out

    def _prepare_trtllm_gen_buffers(
        self, device: torch.device, max_batch_size: int, block_size: int
    ) -> None:
        """TRTLLM-Gen's workspace and counters and the block's append indices at ``max_batch_size`` requests of
        ``block_size`` rows; allocated eagerly (raises under capture where they would have to grow). The workspace and
        counters cover both shapes the built-in DFlash drafter sizes them for: ``max_batch_size * block_size`` tokens
        of the drafter's heads, and ``max_batch_size`` tokens of the heads a folded decode launch carries
        (``k3_dspark_block_attn``)."""
        ops = _block_attn_op.trtllm_gen_ops()
        heads, kv_heads, head_dim = self._num_heads, self._num_kv_heads, self._head_dim
        chunk = _block_attn_op.folded_decode_chunk_size(block_size, heads, kv_heads)
        shapes = ((max_batch_size * block_size, heads), (max_batch_size, heads * chunk))
        workspace_bytes = max(
            ops.get_workspace_size(
                dtype=torch.bfloat16,
                num_tokens=tokens,
                num_gen_tokens=tokens,
                num_heads=launch_heads,
                num_kv_heads=kv_heads,
                head_size=head_dim,
                max_num_requests=max_batch_size,
                rotary_embedding_dim=0,
                fp8_context_fmha=False,
            )
            for tokens, launch_heads in shapes
        )
        device = torch.device(device)
        capturing = torch.cuda.is_current_stream_capturing()
        if self._dflash_trtllm_gen_device != device:
            if capturing:
                raise RuntimeError(
                    "DFlash TRTLLM-Gen buffers must be prepared on the current device before CUDA graph capture."
                )
            self._dflash_trtllm_gen_device = device
            self._dflash_trtllm_gen_sm_count = torch.cuda.get_device_properties(
                device
            ).multi_processor_count
        workspace = self._dflash_trtllm_gen_workspace
        if (
            workspace is None
            or workspace.device != device
            or workspace.numel() * workspace.element_size() < workspace_bytes
        ):
            if capturing:
                raise RuntimeError(
                    "The DFlash TRTLLM-Gen workspace must be allocated at the required size before CUDA graph "
                    "capture."
                )
            self._dflash_trtllm_gen_workspace = torch.empty(
                workspace_bytes, dtype=torch.uint8, device=device
            )
        counter_bytes = ops.get_multi_ctas_kv_counter_size(
            max(launch_heads for _, launch_heads in shapes),
            max_batch_size,
            self._dflash_trtllm_gen_sm_count,
        )
        counters = self._dflash_trtllm_gen_counters
        if (
            counters is None
            or counters.device != device
            or counters.numel() * counters.element_size() < counter_bytes
        ):
            if capturing:
                raise RuntimeError(
                    "The DFlash TRTLLM-Gen counter buffer must be allocated at the required size before CUDA "
                    "graph capture."
                )
            self._dflash_trtllm_gen_counters = torch.zeros(
                counter_bytes, dtype=torch.uint8, device=device
            )
        self._prepare_index_buffers(device, max_batch_size, block_size)

    def _prepare_index_buffers(
        self,
        device: torch.device,
        max_batch_size: int,
        block_size: int,
        need_batch_indices: bool = True,
    ) -> None:
        """The block's column offsets and (paged context only) its per-request row indices, allocated eagerly
        (raises under capture where they would have to grow)."""
        device = torch.device(device)
        offsets, indices = self._dflash_block_offsets, self._dflash_batch_indices
        need_offsets = offsets is None or offsets.device != device or offsets.numel() != block_size
        need_indices = need_batch_indices and (
            indices is None
            or indices.device != device
            or indices.size(0) < max_batch_size
            or indices.size(1) != block_size
        )
        if need_offsets or need_indices:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DFlash index buffers must be allocated at the required size before CUDA graph capture."
                )
            if need_offsets:
                self._dflash_block_offsets = torch.arange(
                    block_size, dtype=torch.int32, device=device
                )
            if need_indices:
                self._dflash_batch_indices = (
                    torch.arange(max_batch_size, dtype=torch.int32, device=device)
                    .view(-1, 1)
                    .expand(-1, block_size)
                    .contiguous()
                )

    def _generic_block(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache: Optional[torch.Tensor],
        ctx_v_cache: Optional[torch.Tensor],
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache,
        ctx_page_table: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """The generic block (the class docstring): the ops of the built-in DFlash drafter's block forward on its TRTLLM
        backend (the manager's pages) or its VANILLA backend (the worker's arena) for a non-causal unwindowed drafter,
        on the same tensors."""
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        if noise_embedding.dtype != torch.bfloat16:
            raise RuntimeError(f"K3DSparkDrafter computes in bf16, got {noise_embedding.dtype}")
        use_fused_qk_norm_rope = self._use_fused_qk_norm_rope
        if not (use_fused_qk_norm_rope or _rope_op.available()):
            raise RuntimeError("K3DSparkDrafter needs flashinfer's RoPE for a non-plain RoPE")
        heads, kv_heads, head_dim = self._num_heads, self._num_kv_heads, self._head_dim
        q_size, kv_size = heads * head_dim, kv_heads * head_dim
        batch, block = noise_embedding.shape[:2]
        device = noise_embedding.device

        cache_seqlens = num_ctx_per_req[:batch].to(torch.int32)
        cache_batch_idx = ctx_cache_batch_idx.to(torch.int32)
        seq_lens_after = cache_seqlens + block
        paged = self.dflash_attention_backend == "TRTLLM"
        if paged:
            self._prepare_trtllm_gen_buffers(device, ctx_page_table.size(0), block)
            block_tables = ctx_page_table.index_select(0, cache_batch_idx.long())
            pages_per_slot = block_tables.size(1)
            page_size = ctx_kv_cache[0].size(-2)
            kv_indices = block_tables.flatten()
            kv_indptr = torch.arange(
                0, (batch + 1) * pages_per_slot, pages_per_slot, dtype=torch.int32, device=device
            )
            kv_last_page_len = ((seq_lens_after - 1) % page_size) + 1
            append_batch_indices = self._dflash_batch_indices[:batch].reshape(-1)
            append_positions = (
                (cache_seqlens.view(-1, 1) + self._dflash_block_offsets).reshape(-1).contiguous()
            )
        else:
            # The arena slot and column of each block K / V row.
            self._prepare_index_buffers(
                ctx_k_cache.device, ctx_k_cache.size(0), block, need_batch_indices=False
            )
            append_batch_indices = cache_batch_idx.view(-1, 1).expand(-1, block).reshape(-1).long()
            append_positions = (
                (cache_seqlens.view(-1, 1) + self._dflash_block_offsets).reshape(-1).long()
            )
            # Pack the query heads that share a KV head into the rows: exact for non-causal unwindowed layers.
            pack_gqa = heads // kv_heads > 1
        positions = query_positions.reshape(-1).to(torch.int32)
        cos_sin = None if use_fused_qk_norm_rope else self._get_cos_sin_cache()

        hidden_states = noise_embedding
        residual = None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if residual is None:
                residual = hidden_states.clone()
                normed = layer.input_layernorm(hs_flat)
            else:
                res_flat = residual.reshape(-1, residual.shape[-1])
                normed, res_flat = layer.input_layernorm(hs_flat, res_flat)
                residual = res_flat.reshape(batch, block, -1)

            qkv = F.linear(normed, attn.qkv_proj.weight)
            if use_fused_qk_norm_rope:
                fused_qk_norm_rope(
                    qkv,
                    heads,
                    kv_heads,
                    kv_heads,
                    head_dim,
                    head_dim,
                    attn.q_norm.variance_epsilon,
                    attn.q_norm.weight,
                    attn.k_norm.weight,
                    self._rope.theta,
                    True,
                    positions,
                    1.0,
                    0,
                    0,
                    1.0,
                    True,
                    False,
                    False,
                    0,
                    0,
                )
                q_bshd = qkv[:, :q_size].reshape(batch, block, heads, head_dim)
                k_bshd = qkv[:, q_size : q_size + kv_size].reshape(batch, block, kv_heads, head_dim)
                v_bshd = qkv[:, q_size + kv_size :].reshape(batch, block, kv_heads, head_dim)
            else:
                q = attn.q_norm(qkv[:, :q_size].reshape(-1, head_dim)).view(-1, q_size)
                k = attn.k_norm(qkv[:, q_size : q_size + kv_size].reshape(-1, head_dim)).view(
                    -1, kv_size
                )
                k3_draft_rope(positions, q, k, head_dim, cos_sin, True)
                q_bshd = q.view(batch, block, heads, head_dim)
                k_bshd = k.view(batch, block, kv_heads, head_dim)
                v_bshd = qkv[:, q_size + kv_size :].reshape(batch, block, kv_heads, head_dim)

            if paged:
                out = torch.empty_like(q_bshd)
                k3_dspark_block_attn(
                    q_bshd.reshape(-1, heads, head_dim),
                    k_bshd.reshape(-1, kv_heads, head_dim).contiguous(),
                    v_bshd.reshape(-1, kv_heads, head_dim).contiguous(),
                    ctx_kv_cache[layer_idx],
                    block_tables,
                    append_batch_indices,
                    append_positions,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                    seq_lens_after,
                    pages_per_slot * page_size,
                    self._dflash_trtllm_gen_workspace,
                    self._dflash_trtllm_gen_counters,
                    out.reshape(-1, heads, head_dim),
                )
            else:
                out = k3_dspark_dense_block_attn(
                    q_bshd,
                    k_bshd.reshape(-1, kv_heads, head_dim),
                    v_bshd.reshape(-1, kv_heads, head_dim),
                    ctx_k_cache,
                    ctx_v_cache,
                    layer_idx,
                    append_batch_indices,
                    append_positions,
                    seq_lens_after,
                    cache_batch_idx,
                    pack_gqa,
                )
            hidden_out = attn.o_proj(out.reshape(batch * block, q_size))
            res_flat = residual.reshape(-1, residual.shape[-1])
            hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)
            hidden_out = layer.mlp(hidden_out)
            hidden_states = hidden_out.reshape(batch, block, -1)
            residual = res_flat.reshape(batch, block, -1)

        out, _ = self.model.norm(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            residual.reshape(-1, residual.shape[-1]),
        )
        return out

    # ------------------------------------------------------------------------------------------------------------------
    # The drafter entries.
    # ------------------------------------------------------------------------------------------------------------------

    def _k3_layers_take(self) -> bool:
        """Whether every layer is one the drafter entries take: plain NeoX RoPE of base 1e4 with the q / k RMSNorm at
        the head layout and epsilon the attention certifies, bf16 norm weights."""
        if self._k3_take is None:
            take = self._use_fused_qk_norm_rope and self._rope.theta == _DRAFTER_ROPE_BASE
            for layer in self.model.layers:
                attn = layer.self_attn
                take = take and (
                    (attn.num_heads, attn.num_key_value_heads) == _DRAFTER_HEADS
                    and attn.head_dim == _DRAFTER_HEAD_DIM
                    and all(
                        norm.variance_epsilon == _DRAFTER_EPS
                        and norm.weight.dtype == torch.bfloat16
                        and tuple(norm.weight.shape) == (_DRAFTER_HEAD_DIM,)
                        for norm in (attn.q_norm, attn.k_norm)
                    )
                )
            self._k3_take = bool(take)
            logger.info(
                f"Kimi K3 DSpark drafter: {len(self.model.layers)} layers, decode blocks "
                + (
                    "on k3_drafter_attn_qknorm and the drafter GEMV sites"
                    if self._k3_take
                    else "on the generic block (a layer the drafter entries do not take)"
                )
            )
        return self._k3_take

    def _k3_block_keys(
        self,
        noise_embedding: torch.Tensor,
        ctx_kv_cache,
        ctx_page_table: Optional[torch.Tensor],
    ) -> Optional[frozenset]:
        """The attention's compile keys for this block when the drafter entries take it, else None."""
        if (
            self.dflash_attention_backend != "TRTLLM"
            or ctx_kv_cache is None
            or ctx_page_table is None
            or noise_embedding.dtype != torch.bfloat16
            or tuple(noise_embedding.shape[:2]) not in DRAFTER_ATTN_SPLITS
            or is_in_breakable_cuda_graph()
            or not self._k3_layers_take()
        ):
            return None
        num_kv_heads = self._num_kv_heads
        page = (2, num_kv_heads, _DRAFTER_PAGE, _DRAFTER_HEAD_DIM)
        dense = (
            num_kv_heads * _DRAFTER_PAGE * _DRAFTER_HEAD_DIM,
            _DRAFTER_PAGE * _DRAFTER_HEAD_DIM,
        )
        strides = set()
        for layer_idx in range(len(self.model.layers)):
            cache = ctx_kv_cache[layer_idx]
            if (
                cache.dtype != torch.bfloat16
                or cache.dim() != 5
                or tuple(cache.shape[1:]) != page
                or cache.stride()[1:] != (*dense, _DRAFTER_HEAD_DIM, 1)
                or cache.stride(0) % 8
            ):
                return None
            strides.add(cache.stride(0))
        keys = frozenset((noise_embedding.shape[0] > 1, stride) for stride in strides)
        if torch.cuda.is_current_stream_capturing() and not keys <= self._k3_attn_ran:
            return None
        return keys

    def _k3_block_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache,
        ctx_page_table: torch.Tensor,
        ctx_rows_start: Optional[int] = None,
    ) -> torch.Tensor:
        batch, block = noise_embedding.shape[:2]
        rows = batch * block
        ctx_len = num_ctx_per_req[:batch].to(torch.int32)
        if ctx_rows_start is None:
            page_table = ctx_page_table.index_select(0, ctx_cache_batch_idx.to(torch.long))
        else:
            # The batch's page-table rows are one contiguous run: a view, no gather.
            page_table = ctx_page_table[ctx_rows_start : ctx_rows_start + batch]
        positions = query_positions.reshape(-1).contiguous()
        hidden = noise_embedding.reshape(rows, -1)
        layers = self.model.layers
        comm = self._k3_fused_comm(hidden)
        residual = None
        if comm is not None:
            # The fused all-reduces read the residual and return the updated one: the block's input serves as the
            # first residual, uncopied.
            residual = hidden
            normed = layers[0].input_layernorm(hidden)
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            if comm is None:
                if residual is None:
                    residual = hidden.clone()
                    normed = layer.input_layernorm(hidden)
                else:
                    normed, residual = layer.input_layernorm(hidden, residual)
            qkv = self._k3_project("drafter_qkv", normed, attn.qkv_proj)
            out = torch.empty(rows, attn.q_size, dtype=torch.bfloat16, device=qkv.device)
            k3_drafter_attn_qknorm(
                qkv,
                attn.q_norm.weight,
                attn.k_norm.weight,
                positions,
                attn.q_norm.variance_epsilon,
                self._rope.theta,
                ctx_kv_cache[layer_idx],
                page_table,
                ctx_len,
                attn.num_heads,
                attn.num_key_value_heads,
                out,
            )
            if comm is not None:
                # o_proj's all-reduce applies the post-attention norm; the MLP's the next layer's input norm, after
                # the last layer the final norm.
                next_norm = (
                    layers[layer_idx + 1].input_layernorm
                    if layer_idx + 1 < len(layers)
                    else self.model.norm
                )
                normed, residual = self._k3_project_norm(
                    comm, out, attn.o_proj, residual, layer.post_attention_layernorm
                )
                normed, residual = self._k3_mlp_norm(comm, layer.mlp, normed, residual, next_norm)
                continue
            hidden = self._k3_project("drafter_o", out, attn.o_proj)
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = self._k3_mlp(layer.mlp, hidden)
        if comm is not None:
            return normed
        out, _ = self.model.norm(hidden, residual)
        return out

    def _k3_fused_comm(self, hidden: torch.Tensor) -> Optional[_decode_comm.K3DecodeComm]:
        """The TP group's collective state where a block of rows ``hidden`` runs its residual adds and RMSNorms in its
        all-reduces: the drafter runs on it (``use_decode_comm``) with layers that take it, and its MNNVL workspace
        holds an all-reduce of the block's rows; else None (the module all-reduces and norms)."""
        comm = self.decode_comm
        if comm is None or not self._k3_norms_fuse or not comm.takes_allreduce_norm(*hidden.shape):
            return None
        return comm

    def _k3_project_norm(
        self,
        comm: _decode_comm.K3DecodeComm,
        x: torch.Tensor,
        linear: DraftLinear,
        residual: torch.Tensor,
        norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm(updated), updated)``, ``updated = residual + linear(x)`` for the row-parallel output projection,
        the residual add and ``norm`` in its all-reduce: ``comm/k3_sandwich_plain`` where it takes the rows (the
        projection in the ``drafter_o`` site's arithmetic, in the same launch), else the projection on ``drafter_o``
        where that takes the rows, or ``F.linear``, then ``comm/mnnvl_fusion_allreduce``."""
        if "o_proj" in self._k3_sandwich_forms and comm.takes_plain(
            x, linear.weight, residual, norm
        ):
            return comm.sandwich_plain(x, linear.weight, residual, norm)
        gemvs = self.decode_gemvs
        partial = None if gemvs is None else gemvs.project("drafter_o", x, linear.weight)
        if partial is None:
            partial = F.linear(x, linear.weight)
        return comm.allreduce_norm(partial, residual, norm)

    def _k3_mlp_norm(
        self,
        comm: _decode_comm.K3DecodeComm,
        mlp: DraftMLP,
        x: torch.Tensor,
        residual: torch.Tensor,
        norm: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(norm(updated), updated)``, ``updated = residual + mlp(x)``, the residual add and ``norm`` in the down
        projection's all-reduce: gate / up on ``drafter_gate_up``, then ``comm/k3_sandwich_plain``'s SiLU-and-mul
        form where it takes the rows (the ``drafter_down`` site's arithmetic), else the SiLU-and-mul and down
        projection on ``drafter_down``; where those sites do not take the rows, the MLP without its all-reduce; then
        ``comm/mnnvl_fusion_allreduce``."""
        gemvs = self.decode_gemvs
        gate_up = (
            None if gemvs is None else gemvs.project("drafter_gate_up", x, mlp.gate_up_proj.weight)
        )
        partial = None
        if gate_up is not None:
            if "down" in self._k3_sandwich_forms and comm.takes_plain(
                gate_up, mlp.down_proj.weight, residual, norm, swiglu=True
            ):
                return comm.sandwich_plain(
                    gate_up, mlp.down_proj.weight, residual, norm, swiglu=True
                )
            partial = gemvs.project("drafter_down", gate_up, mlp.down_proj.weight)
        if partial is None:
            partial = F.linear(
                k3_draft_silu_and_mul(F.linear(x, mlp.gate_up_proj.weight)), mlp.down_proj.weight
            )
        return comm.allreduce_norm(partial, residual, norm)

    def _k3_project(self, site: str, x: torch.Tensor, linear: DraftLinear) -> torch.Tensor:
        """``linear(x)``, its GEMM on the ``site`` decode GEMV where that takes the rows, then a row-parallel
        projection's all-reduce."""
        y = None if self.decode_gemvs is None else self.decode_gemvs.project(site, x, linear.weight)
        if y is None:
            return linear(x)
        return linear.reduce(y)

    def _k3_mlp(self, mlp: DraftMLP, x: torch.Tensor) -> torch.Tensor:
        """``mlp(x)``: gate / up on ``drafter_gate_up`` and the SiLU-and-mul with the down projection on
        ``drafter_down`` where both take the rows, then the down projection's all-reduce."""
        gemvs = self.decode_gemvs
        gate_up = (
            None if gemvs is None else gemvs.project("drafter_gate_up", x, mlp.gate_up_proj.weight)
        )
        down = (
            None
            if gate_up is None
            else gemvs.project("drafter_down", gate_up, mlp.down_proj.weight)
        )
        if down is None:
            return mlp(x)
        return mlp.down_proj.reduce(down)


def _shape(t) -> Tuple[int, ...]:
    """A checkpoint tensor's shape (a torch tensor or a lazy safetensors slice)."""
    return tuple(t.get_shape()) if hasattr(t, "get_shape") else tuple(t.shape)


def _tensor(t) -> torch.Tensor:
    """A checkpoint tensor, read whole."""
    return t if isinstance(t, torch.Tensor) else t[:]


def _shard(t, dim: int, rank: int, tp_size: int) -> torch.Tensor:
    """Rank ``rank``'s contiguous block of ``t`` along ``dim`` split evenly over ``tp_size`` ranks (the built-in
    loader's cut for these divisible widths), reading only that block from a lazy slice."""
    width = _shape(t)[dim]
    if width % tp_size:
        raise ValueError(
            f"a draft checkpoint dimension of {width} does not split over TP {tp_size}"
        )
    step = width // tp_size
    index = [slice(None)] * len(_shape(t))
    index[dim] = slice(rank * step, (rank + 1) * step)
    return t[tuple(index)].contiguous() if isinstance(t, torch.Tensor) else t[tuple(index)]
