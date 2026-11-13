# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter from a standalone MLA checkpoint (Inferact-Kimi-K3-DSpark: ``k3_dspark``, 5 MLA layers of 64
heads, kv_lora_rank 512 + qk_rope_head_dim 64, YaRN), built from operations.

The context cache holds one MLA latent (``kv_lora_rank + qk_rope_head_dim``) per token per layer in the draft KV
manager's pages, no V. Per layer the block decode runs: RMSNorm (the previous MLP's all-reduce with the residual add
fused in, under TP), the q projections (``q_a``, its RMSNorm, ``q_b``), the q rope rotated into the absorbed query
(``k3_dspark_rope_into``), the nope half absorbed by ``k_b`` (``catalog bmm_out``), the block's own latent
(``kv_a``, then ``k3_dspark_rmsnorm_rope``) written into its pages, one ``k3_dspark_mla_decode`` over context and
block with per-token KV bounds (every block row sees the whole block), ``v_b`` un-absorption, ``o_proj`` and its
all-reduce with the residual add and the post-attention RMSNorm, the SiLU-and-mul MLP; then the final norm. These are
the ops of the built-in ``MLADSparkForCausalLM`` on its CUTEDSL backend, in its order. Its other two backends match
the built-in ones too: TRTLLM stores the block's latents the same way, runs ``k3_dspark_mla_paged_decode`` (flashinfer
TRTLLM-Gen, causal over the block) and merges in the block's strict upper triangle, attended eagerly, by
log-sum-exp; VANILLA runs an eager torch attention over the gathered context and the block (the block's latents not
stored).

The drafter ships its own embedding (vocab-sharded) and shares only the target's LM head. Its worker is this target's
``KimiK3DSparkWorker``.
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.bmm_out import bmm_out
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
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams, RopeEmbeddingUtils
from tensorrt_llm.logger import logger

from .drafter import DraftLinear, K3DSparkDrafter, _shard, _tensor
from .norm import RMSNorm
from .ops.activation.k3_draft_silu_and_mul import k3_draft_silu_and_mul
from .ops.attention import k3_dspark_mla as _mla_ops
from .ops.attention.k3_dspark_mla import (
    k3_dspark_mla_decode,
    k3_dspark_mla_paged_decode,
    k3_dspark_rmsnorm_rope,
    k3_dspark_rope_into,
)

# The flashinfer / CuTe DSL scratch the block decode reserves (the built-in MLA DSpark drafter's size).
_MLA_DECODE_WORKSPACE_BYTES = 256 * 1024 * 1024


class _BlockFixup(NamedTuple):
    """Layer-invariant operands of the block decode: where the block's latents go and the kernel's lengths."""

    pages: torch.Tensor
    offsets: torch.Tensor
    valid: (
        torch.Tensor
    )  # [1, block, 1, block]: query i sees block key j > i (the TRTLLM decode's fixup)
    page_tables_i32: torch.Tensor
    seq_lens_i32: torch.Tensor
    kv_bounds_i32: torch.Tensor


# trtllm-gen reports the log-sum-exp in base 2; times this, the natural log the merge takes.
_TRTLLM_GEN_LSE_LN2 = math.log(2.0)


def _block_upper_triangle(block_size: int, device) -> torch.Tensor:
    """``[1, block, 1, block]``: query i sees block key j > i (the part the causal TRTLLM decode leaves out)."""
    idx = torch.arange(block_size, device=device)
    return (idx.view(1, block_size) > idx.view(block_size, 1)).view(1, block_size, 1, block_size)


def _attention_backend_unavailability_reason(backend: str) -> Optional[str]:
    """Why this build cannot run ``backend``'s block decode, or None: CUTEDSL's CuTe DSL decode with KV bounds,
    TRTLLM's flashinfer MLA decode; VANILLA, the eager torch decode, always runs."""
    if backend == "CUTEDSL":
        return _mla_ops.mla_decode_unavailability_reason()
    if backend == "TRTLLM":
        return _mla_ops.mla_paged_decode_unavailability_reason()
    return None


def resolve_attention_backend(requested: str) -> str:
    """The block decode for ``attention_backend`` ``requested``, resolved as the built-in MLA DSpark drafter resolves
    it: AUTO is CUTEDSL (``drafter_config.MLA_DEFAULT_ATTENTION_BACKEND``), with no fallback; a backend must be one of
    ``drafter_config.MLA_ATTENTION_BACKENDS`` and runnable here. ``route`` checks the same refusals first
    (``drafter_config.mla_attention_backend_refusal``), except whether the CuTe DSL decode is registered.

    Raises:
        ValueError: AUTO with CUTEDSL unavailable, an unknown backend, or an explicit backend unavailable here.
    """
    name = "K3MLADSparkDrafter"
    backend = requested
    if requested == "AUTO":
        default = _drafter_config.MLA_DEFAULT_ATTENTION_BACKEND
        reason = _attention_backend_unavailability_reason(default)
        if reason is not None:
            raise ValueError(
                f"{name} has no usable attention backend: its default {default!r} is unavailable "
                f"({reason}) and this family does not degrade."
            )
        backend = default
    reason = _drafter_config.mla_unknown_attention_backend(backend)
    if reason is not None:
        raise ValueError(reason)
    reason = _attention_backend_unavailability_reason(backend)
    if reason is not None:
        raise ValueError(_drafter_config.requested_backend_unavailable(name, backend, reason))
    return backend


def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def rope_params(pretrained_config, model_config) -> Dict:
    """The drafter's YaRN settings (``rope_parameters`` or ``rope_scaling``) and the fallback table length (the
    served length, else ``max_position_embeddings``), as the built-in MLA DSpark drafter resolves them. ``route``
    declines the configs this raises for (``drafter_config.mla_rope_declines``).

    Raises:
        ValueError: a RoPE other than YaRN.
        KeyError: a YaRN without ``factor`` or ``original_max_position_embeddings``.
    """
    scaling = _drafter_config.mla_rope_scaling(pretrained_config)
    rope_type = _drafter_config.mla_rope_type(scaling)
    if rope_type != "yarn":
        raise ValueError(_drafter_config.mla_rope_declines(pretrained_config))
    theta = scaling.get("rope_theta") or getattr(pretrained_config, "rope_theta", 10000.0)
    runtime = getattr(model_config, "max_seq_len", None)
    return {
        "theta": float(theta),
        "scaling_factor": float(scaling["factor"]),
        "original_max_positions": int(scaling["original_max_position_embeddings"]),
        "max_positions": int(
            runtime or getattr(pretrained_config, "max_position_embeddings", 163840)
        ),
        "beta_fast": float(scaling.get("beta_fast", 32.0)),
        "beta_slow": float(scaling.get("beta_slow", 1.0)),
        "mscale": float(scaling.get("mscale", 1.0)),
        "mscale_all_dim": float(scaling.get("mscale_all_dim", 0.0)),
    }


def yarn_freqs_cis(rope: Dict, dim: int, max_positions: int, device) -> torch.Tensor:
    """The adjacent-pair YaRN table ``[max_positions, dim // 2]`` complex64: the duplicated-data numpy table, its first
    half as (cos, sin) (the built-in ``build_dspark_mla_yarn_freqs_cis``)."""
    _, packed = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
        num_pos=max_positions,
        dim=dim,
        base=rope["theta"],
        scaling_factor=rope["scaling_factor"],
        original_max_position_embeddings=rope["original_max_positions"],
        beta_fast=rope["beta_fast"],
        beta_slow=rope["beta_slow"],
        mscale=rope["mscale"],
        mscale_all_dim=rope["mscale_all_dim"],
        duplicate_data=True,
    )
    table = torch.from_numpy(packed).reshape(max_positions, dim, 2)
    cos = table[..., 0].contiguous().to(device)
    sin = table[..., 1].contiguous().to(device)
    half = cos.shape[-1] // 2
    return torch.complex(cos[..., :half], sin[..., :half]).contiguous()


def rmsnorm_rope(
    t: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    rope_dim: int,
    freqs_cis: torch.Tensor,
    *,
    num_heads: int = 1,
    apply_weight: bool = True,
    apply_rmsnorm: bool = True,
    norm_dim: Optional[int] = None,
) -> torch.Tensor:
    """RMSNorm of ``t``'s first ``norm_dim`` columns and adjacent-pair RoPE of its last ``rope_dim`` (the built-in
    ``_rmsnorm_rope_batched``): the CuTe DSL kernel where it takes the operands, else its eager reference."""
    freqs_real = torch.view_as_real(freqs_cis).reshape(-1, freqs_cis.shape[-1], 2)
    args = (t, weight, freqs_real, num_heads, rope_dim, eps, apply_weight, apply_rmsnorm, norm_dim)
    if _mla_ops.fused_rope_supported(t, weight, freqs_real, num_heads, rope_dim, norm_dim):
        return k3_dspark_rmsnorm_rope(*args)
    return k3_dspark_rmsnorm_rope.reference(*args)


class DraftEmbedding(nn.Module):
    """The drafter's own embedding, vocabulary-sharded over the TP group (the layout of an ``Embedding`` in COLUMN
    mode):
    ``weight`` is this rank's rows; a lookup masks the ids outside them and sums over the group."""

    def __init__(self, vocab: int, hidden: int, tp_size: int, tp_rank: int, dtype, all_reduce):
        super().__init__()
        rows = math.ceil(vocab / tp_size)
        self.weight = nn.Parameter(torch.empty(rows, hidden, dtype=dtype), requires_grad=False)
        self.tp_size = tp_size
        self.vocab_start_index = tp_rank * rows if tp_size > 1 else 0
        self.vocab_end_index = min((tp_rank + 1) * rows, vocab) if tp_size > 1 else vocab
        self.all_reduce = all_reduce

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            start, end = self.vocab_start_index, self.vocab_end_index
        else:
            start, end = 0, self.weight.shape[0]
        vocab_mask = (input_ids >= start) & (input_ids < end)
        valid_offset = start * vocab_mask
        ids = vocab_mask * (input_ids - valid_offset)
        output = F.embedding(ids, self.weight)
        output.masked_fill_(~vocab_mask.unsqueeze(-1), 0)
        if self.tp_size > 1:
            output = self.all_reduce(output)
        return output


class MLADraftAttention(nn.Module):
    """One layer's MLA parameters: ``q_a`` / ``kv_a`` (replicated) with their RMSNorms, ``q_b`` / ``kv_b`` (column
    shards by head), ``o_proj`` (row shard) and this rank's head count; ``_k_b_proj`` / ``_v_b_proj``, ``kv_b``
    split for absorption, are built after the load."""

    def __init__(self, cfg, tp_size: int, dtype, all_reduce):
        super().__init__()
        self.num_heads_tp = cfg.num_attention_heads // tp_size
        self.q_lora_rank = cfg.q_lora_rank
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.kv_cache_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        hidden, heads = cfg.hidden_size, self.num_heads_tp
        self.q_a_proj = DraftLinear(self.q_lora_rank, hidden, dtype)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=cfg.rms_norm_eps, dtype=dtype)
        self.q_b_proj = DraftLinear(heads * self.qk_head_dim, self.q_lora_rank, dtype)
        self.kv_a_proj_with_mqa = DraftLinear(self.kv_cache_head_dim, hidden, dtype)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=cfg.rms_norm_eps, dtype=dtype)
        self.kv_b_proj = DraftLinear(
            heads * (self.qk_nope_head_dim + self.v_head_dim), self.kv_lora_rank, dtype
        )
        self.o_proj = DraftLinear(hidden, heads * self.v_head_dim, dtype, all_reduce)
        self._k_b_proj: Optional[torch.Tensor] = None
        self._v_b_proj: Optional[torch.Tensor] = None


class MLADraftMLP(nn.Module):
    def __init__(self, cfg, tp_size: int, dtype, all_reduce):
        super().__init__()
        inter = cfg.intermediate_size // tp_size
        self.gate_up_proj = DraftLinear(2 * inter, cfg.hidden_size, dtype)
        self.down_proj = DraftLinear(cfg.hidden_size, inter, dtype, all_reduce)

    def partial(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's MLP output, before the down projection's all-reduce."""
        return F.linear(
            k3_draft_silu_and_mul(F.linear(x, self.gate_up_proj.weight)), self.down_proj.weight
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.reduce(self.partial(x))


class MLADraftLayer(nn.Module):
    def __init__(self, cfg, tp_size: int, dtype, make_all_reduce):
        super().__init__()
        self.self_attn = MLADraftAttention(cfg, tp_size, dtype, make_all_reduce())
        self.mlp = MLADraftMLP(cfg, tp_size, dtype, make_all_reduce())
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)


class MLADraftBackbone(nn.Module):
    def __init__(self, cfg, tp_size: int, tp_rank: int, dtype, make_all_reduce):
        super().__init__()
        self.embed_tokens = DraftEmbedding(
            cfg.vocab_size, cfg.hidden_size, tp_size, tp_rank, dtype, make_all_reduce()
        )
        self.layers = nn.ModuleList(
            MLADraftLayer(cfg, tp_size, dtype, make_all_reduce)
            for _ in range(cfg.num_hidden_layers)
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.d2t = None


class MLADraftModelFull(nn.Module):
    """Where the speculative worker looks for the embedding (``model.embed_tokens``) and the LM head."""

    def __init__(self, model: MLADraftBackbone):
        super().__init__()
        self.model = model
        self.lm_head = None


def _all_reduce_add_norm(all_reduce, partial: torch.Tensor, residual: torch.Tensor, norm: RMSNorm):
    """``norm(all_reduce(partial) + residual)`` and the new residual, in the all-reduce."""
    return all_reduce(
        partial,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm.weight,
            eps=norm.variance_epsilon,
        ),
    )


class K3MLADSparkDrafter(nn.Module):
    """Kimi K3's DSpark drafter from a standalone MLA checkpoint (the module docstring). Keeps the attribute set the
    speculative worker reads off a drafter (``drafter.WORKER_FIELDS``) with the built-in MLA drafter's values: one
    latent per token per layer (``_kv_factor`` 1, ``_num_kv_heads`` 1, ``_head_dim`` 576), the paged context, a block
    decode of its own (``_uses_worker_attention_backend`` False)."""

    _uses_worker_attention_backend = False
    _paged_ctx_cache = True
    _uses_lilicorr = False
    _kv_factor = 1

    # The heads, the logit transform and the shared-head load are the GQA drafter's.
    has_markov_head = K3DSparkDrafter.has_markov_head
    apply_markov_chain_logits = K3DSparkDrafter.apply_markov_chain_logits
    apply_draft_logit_transform = K3DSparkDrafter.apply_draft_logit_transform
    _take_head_weights = K3DSparkDrafter._take_head_weights

    def __init__(
        self, draft_config: ModelConfig, *, dflash_attention_backend: str = "AUTO"
    ) -> None:
        super().__init__()
        cfg = draft_config.pretrained_config
        # The refusals route checks first (drafter_config.declines), in the same order.
        reason = _drafter_config.mla_declines(cfg)
        if reason is not None:
            raise NoTarget(f"the Kimi K3 MLA DSpark drafter does not build this drafter: {reason}")
        mapping = draft_config.mapping
        reason = _drafter_config.mla_cp_declines(mapping)
        if reason is not None:
            raise NotImplementedError(reason)
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        tp_rank = 0 if tp_size == 1 else mapping.tp_rank
        reason = _drafter_config.mla_tp_declines(cfg, mapping)
        if reason is not None:
            raise ValueError(reason)
        self.model_config = draft_config
        self.config = cfg
        self.dflash_attention_backend = resolve_attention_backend(dflash_attention_backend)
        dtype = torch.bfloat16

        def make_all_reduce():
            if tp_size == 1:
                return None
            return K3AllReduce(mapping, draft_config.allreduce_strategy, dtype)

        self.model = MLADraftBackbone(cfg, tp_size, tp_rank, dtype, make_all_reduce)
        self.draft_model_full = MLADraftModelFull(self.model)
        self.lm_head = None
        self.logits_processor = None  # set by the shell

        dflash_config = getattr(cfg, "dflash_config", None) or {}
        self.mask_token_id = dflash_config.get(
            "mask_token_id", getattr(cfg, "mask_token_id", cfg.vocab_size)
        )
        self.mask_token_embedding = None
        self.target_layer_ids = dflash_config.get("target_layer_ids", None)
        self.block_size = dflash_config.get("block_size", getattr(cfg, "block_size", None))
        self.logit_multiplier, self.logit_softcap = _drafter_config.logit_transform(dflash_config)

        shift_label = head_config(cfg, "shift_label")
        self._dspark_shift_label = True if shift_label is None else bool(shift_label)
        self._dspark_markov_rank = int(head_config(cfg, "markov_rank") or 0)
        reason = _drafter_config.mla_markov_head_declines(
            cfg
        ) or _drafter_config.mla_causal_declines(cfg)
        if reason is not None:
            raise ValueError(reason)
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

        self.is_dflash2 = False
        self.has_block_conv = False
        self.has_candidate_selector = False
        self.candidate_selector = None
        self.lilicorr = None
        self._runtime_position_ceiling = None

        # The TP fusions: fc split by input feature with hidden_norm in its all-reduce, and the residual adds and
        # RMSNorms after o_proj / down_proj in their all-reduces.
        self._tp_split = tp_size > 1
        self._fc_input_slice: Optional[slice] = None
        self.fc: Optional[nn.Linear] = None
        self.hidden_norm: Optional[nn.RMSNorm] = None
        self._fc_all_reduce = None

        attn0 = self.model.layers[0].self_attn
        self.qk_nope_head_dim = attn0.qk_nope_head_dim
        self.qk_rope_head_dim = attn0.qk_rope_head_dim
        self.kv_lora_rank = attn0.kv_lora_rank
        self.v_head_dim = attn0.v_head_dim
        self._num_attn_layers = len(self.model.layers)
        self._num_heads = attn0.num_heads_tp
        self._num_kv_heads = 1
        self._head_dim = attn0.kv_cache_head_dim

        self._mla_rope_params = rope_params(cfg, draft_config)
        self._mla_workspace: Optional[torch.Tensor] = None
        self._mla_freqs: Optional[torch.Tensor] = None
        self._mla_freqs_cap: Optional[int] = None
        self._cute_dsl_checked_shape = None
        mscale = _yarn_get_mscale(
            self._mla_rope_params["scaling_factor"], self._mla_rope_params["mscale_all_dim"]
        )
        self.softmax_scale = (mscale * mscale) / math.sqrt(
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )

        # Built after the load (_build_fused_kv_buffers). No k_norm and no RoPE cos / sin cache: the K3 worker's
        # context-K / V kernel declines this drafter.
        self._fused_kv_weight: Optional[torch.Tensor] = None
        self._fused_kv_bias = None
        self._kv_a_norm_stacked: Optional[torch.Tensor] = None
        self._kv_a_norm_eps: Optional[float] = None
        self._k_norm_stacked = None
        self._k_norm_eps = None
        self._input_ln_eps = None
        self._is_neox = True
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False
        logger.info(
            f"Kimi K3 MLA DSpark drafter: {self._num_attn_layers} layers, {self._num_heads} heads per rank, latent "
            f"{self.kv_lora_rank} + rope {self.qk_rope_head_dim}, block decode on {self.dflash_attention_backend}"
        )

    # ------------------------------------------------------------------------------------------------------------------
    # The worker's interface.
    # ------------------------------------------------------------------------------------------------------------------

    def load_weights_from_target_model(self, target_model: nn.Module) -> None:
        """Share only the target's LM head: the drafter ships its own trained embedding."""
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head

    def validate_block_attention_windows(self) -> list:
        """Not read for this drafter (it brings its own block decode)."""
        return []

    def _get_cos_sin_cache(self) -> torch.Tensor:
        raise RuntimeError(
            "the MLA DSpark drafter rotates only its rope slice; it has no RoPE cos / sin cache"
        )

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(hidden_states))``; under the TP fusions this rank's input columns, summed over the group
        with ``hidden_norm`` in the all-reduce."""
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        if self._fc_input_slice is None:
            return self.hidden_norm(self.fc(hidden_states))
        partial = self.fc(hidden_states[:, self._fc_input_slice])
        return self._fc_all_reduce(
            partial,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RMS_NORM,
                norm_weight=self.hidden_norm.weight,
                eps=self.hidden_norm.eps,
            ),
        )

    def _mla_freqs_cis(self, device) -> torch.Tensor:
        """The adjacent-pair YaRN table, sized from the worker's published position ceiling (else the config's)."""
        cap = self._mla_rope_params["max_positions"]
        if self._runtime_position_ceiling is not None:
            cap = int(self._runtime_position_ceiling)
        if self._mla_freqs is None or self._mla_freqs_cap != cap:
            self._mla_freqs = yarn_freqs_cis(
                self._mla_rope_params, self.qk_rope_head_dim, cap, device
            )
            self._mla_freqs_cap = cap
        return self._mla_freqs

    def _mla_decode_workspace(self, device, min_bytes: int = 0) -> torch.Tensor:
        want = max(_MLA_DECODE_WORKSPACE_BYTES, int(min_bytes))
        if self._mla_workspace is None or self._mla_workspace.numel() < want:
            self._mla_workspace = torch.zeros(want, dtype=torch.uint8, device=device)
        return self._mla_workspace

    def _build_fused_kv_buffers(self) -> None:
        """Stack the layers' latent projections and their RMSNorm weights for the context latents; split ``kv_b``
        into the absorption operands."""
        if self._fused_kv_weight is not None:
            return
        attns = [layer.self_attn for layer in self.model.layers]
        self._fused_kv_weight = torch.cat(
            [a.kv_a_proj_with_mqa.weight for a in attns], dim=0
        ).contiguous()
        self._kv_a_norm_stacked = torch.stack([a.kv_a_layernorm.weight.data for a in attns])
        self._kv_a_norm_eps = attns[0].kv_a_layernorm.variance_epsilon
        heads = attns[0].num_heads_tp
        for a in attns:
            w = a.kv_b_proj.weight.view(heads, self.qk_nope_head_dim + self.v_head_dim, -1)
            a._k_b_proj = w[:, : self.qk_nope_head_dim].contiguous()
            a._v_b_proj = w[:, self.qk_nope_head_dim :].contiguous()

    def precompute_context_kv(self, projected_hidden: torch.Tensor, positions: torch.Tensor):
        """Every layer's context latent (RMSNorm of the latent, RoPE of k_pe) of ``projected_hidden`` [N, hidden] at
        ``positions``: ``([N, L, 1, 576], None)``, no V half."""
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        n = projected_hidden.shape[0]
        layers = self._num_attn_layers
        if projected_hidden.dtype != self._fused_kv_weight.dtype:
            projected_hidden = projected_hidden.to(self._fused_kv_weight.dtype)
        latent = F.linear(projected_hidden, self._fused_kv_weight).view(n, layers, self._head_dim)
        freqs = self._mla_freqs_cis(latent.device)[positions.view(-1).long()]
        out = torch.empty_like(latent)
        for layer_idx in range(layers):
            out[:, layer_idx] = rmsnorm_rope(
                latent[:, layer_idx].unsqueeze(0).contiguous(),
                self._kv_a_norm_stacked[layer_idx],
                self._kv_a_norm_eps,
                self.qk_rope_head_dim,
                freqs.unsqueeze(0),
                norm_dim=self.kv_lora_rank,
            ).squeeze(0)
        return out.unsqueeze(2).contiguous(), None

    # ------------------------------------------------------------------------------------------------------------------
    # The weight load.
    # ------------------------------------------------------------------------------------------------------------------

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs) -> None:
        """Load the draft checkpoint (TorchSpec spellings ``context_proj`` / ``context_norm`` / ``final_norm`` or
        ``fc`` / ``hidden_norm`` / ``norm``; ``layers.{i}.*``; ``embed_tokens``; the DSpark heads): this rank's shard
        of every tensor, cut as the built-in MLA drafter's loader cuts them. ``weight_mapper`` is not used."""
        del weight_mapper, kwargs
        device = self.model.norm.weight.device
        weights = self._take_head_weights(weights, device)
        renames = {
            "context_proj.weight": "fc.weight",
            "context_norm.weight": "hidden_norm.weight",
            "final_norm.weight": "norm.weight",
        }
        weights = {renames.get(k, k): v for k, v in weights.items()}
        weights = {
            (k[len("model.") :] if k.startswith("model.") else k): v for k, v in weights.items()
        }
        missing = [k for k in ("fc.weight", "hidden_norm.weight") if k not in weights]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint is missing {missing}, which this wrapper owns and builds "
                "from the checkpoint."
            )
        mapping = self.model_config.mapping
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        rank = 0 if tp_size == 1 else mapping.tp_rank
        fc_weight = _tensor(weights["fc.weight"])
        if self._tp_split and fc_weight.shape[1] % mapping.tp_size == 0:
            width = fc_weight.shape[1] // mapping.tp_size
            self._fc_input_slice = slice(mapping.tp_rank * width, (mapping.tp_rank + 1) * width)
            self._fc_all_reduce = K3AllReduce(
                mapping, self.model_config.allreduce_strategy, fc_weight.dtype
            )
            fc_weight = fc_weight[:, self._fc_input_slice]
        elif self._tp_split:
            logger.info(
                f"{type(self).__name__}: fc input width {fc_weight.shape[1]} does not split over "
                f"tp_size={mapping.tp_size}; keeping the unsplit context projection."
            )
        self.fc = nn.Linear(
            fc_weight.shape[1], fc_weight.shape[0], bias=False, device=device, dtype=fc_weight.dtype
        )
        self.fc.weight.data.copy_(fc_weight)
        norm_weight = _tensor(weights["hidden_norm.weight"])
        self.hidden_norm = nn.RMSNorm(
            norm_weight.shape[0],
            eps=getattr(self.config, "rms_norm_eps", 1e-6),
            device=device,
            elementwise_affine=True,
            dtype=norm_weight.dtype,
        )
        self.hidden_norm.weight.data.copy_(norm_weight)

        missing = []

        def take(name):
            if name not in weights:
                missing.append(name)
                return None
            return weights[name]

        def rows(name):
            t = take(name)
            return None if t is None else _shard(t, 0, rank, tp_size)

        def cols(name):
            t = take(name)
            return None if t is None else _shard(t, 1, rank, tp_size)

        def full(name):
            t = take(name)
            return None if t is None else _tensor(t)

        loads = [(self.model.embed_tokens.weight, rows("embed_tokens.weight"))]
        for i, layer in enumerate(self.model.layers):
            p = f"layers.{i}."
            attn, mlp = layer.self_attn, layer.mlp
            gate, up = rows(p + "mlp.gate_proj.weight"), rows(p + "mlp.up_proj.weight")
            loads += [
                (attn.q_a_proj.weight, full(p + "self_attn.q_a_proj.weight")),
                (attn.q_a_layernorm.weight, full(p + "self_attn.q_a_layernorm.weight")),
                (attn.q_b_proj.weight, rows(p + "self_attn.q_b_proj.weight")),
                (attn.kv_a_proj_with_mqa.weight, full(p + "self_attn.kv_a_proj_with_mqa.weight")),
                (attn.kv_a_layernorm.weight, full(p + "self_attn.kv_a_layernorm.weight")),
                (attn.kv_b_proj.weight, rows(p + "self_attn.kv_b_proj.weight")),
                (attn.o_proj.weight, cols(p + "self_attn.o_proj.weight")),
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

    # ------------------------------------------------------------------------------------------------------------------
    # The block decode.
    # ------------------------------------------------------------------------------------------------------------------

    def _assert_can_implement(self, batch: int, block: int, page_size: int, dtype) -> None:
        key = (batch, block, page_size, dtype)
        if self._cute_dsl_checked_shape == key:
            return
        if not _mla_ops.mla_decode_can_implement(
            batch,
            block,
            page_size,
            self._num_heads,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            dtype,
        ):
            raise ValueError(
                f"attention_backend='CUTEDSL' cannot serve this shape: batch={batch}, block_size={block}, "
                f"page_size={page_size}, num_heads={self._num_heads}, dtype={dtype}. Use attention_backend='TRTLLM'."
            )
        self._cute_dsl_checked_shape = key

    def _block_decode_variant(self, paged: bool) -> str:
        """The block decode this step runs (the built-in ``_mla_block_decode_variant``): ``eager`` on VANILLA or on the
        private arena (no page table for a kernel); ``cute_dsl`` on CUTEDSL; on TRTLLM ``trtllm_gen`` where
        flashinfer's MLA decode imports, else ``eager``."""
        if not paged or self.dflash_attention_backend == "VANILLA":
            return "eager"
        if self.dflash_attention_backend == "CUTEDSL":
            return "cute_dsl"
        return "trtllm_gen" if _mla_ops.mla_paged_decode_kernel() is not None else "eager"

    def _eager_attention(self, query, ctx, blk_kv, key_valid, batch: int) -> torch.Tensor:
        """Absorbed-MLA attention of ``query`` [B, T, H, 576] over the context latents ``ctx`` [B, C, 576] and the
        block's ``blk_kv`` [B, T, 576], keys masked by ``key_valid`` [B, C + T]: the latent output [B, T, H, 512].
        The built-in VANILLA decode's ops, in its order."""
        kv_full = torch.cat([ctx, blk_kv], dim=1)
        scores = torch.einsum("bshc,btc->bsht", query, kv_full).float() * self.softmax_scale
        scores = scores.masked_fill(~key_valid.view(batch, 1, 1, -1), float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(kv_full.dtype)
        return torch.einsum("bsht,btc->bshc", probs, kv_full[..., : self.kv_lora_rank])

    def _block_attention(
        self,
        variant: str,
        query,
        blk_kv,
        layer_cache,
        fixup: _BlockFixup,
        block_size: int,
        max_kv_len: int,
    ):
        """The block's latents into their pages, then the paged decode over context and block: CuTe DSL in one pass
        (``cute_dsl``), or TRTLLM-Gen causal over the block plus the block's strict upper triangle attended eagerly
        and merged by log-sum-exp (``trtllm_gen``). The built-in ``_mla_paged_attention``'s ops, in its order."""
        pool = layer_cache[:, 0, 0]  # [pages, page_size, head_dim]
        pool[fixup.pages.reshape(-1), fixup.offsets.reshape(-1)] = blk_kv.reshape(
            -1, blk_kv.shape[-1]
        )
        if variant == "trtllm_gen":
            return self._trtllm_gen_attention(
                query, blk_kv, layer_cache, fixup, block_size, max_kv_len
            )
        batch, _, num_heads, _ = query.shape
        page_size = pool.shape[1]
        self._assert_can_implement(batch, block_size, page_size, query.dtype)
        workspace = self._mla_decode_workspace(
            query.device,
            _mla_ops.mla_decode_workspace_bytes(num_heads, block_size, self.kv_lora_rank, batch),
        )
        return k3_dspark_mla_decode(
            query,
            pool,
            fixup.page_tables_i32,
            fixup.seq_lens_i32,
            fixup.kv_bounds_i32,
            self.kv_lora_rank,
            self.softmax_scale,
            workspace,
        )

    def _trtllm_gen_attention(
        self, query, blk_kv, layer_cache, fixup, block_size: int, max_kv_len: int
    ):
        batch, _, num_heads, _ = query.shape
        device = query.device
        out_lower = torch.empty(
            batch, block_size, num_heads, self.kv_lora_rank, dtype=query.dtype, device=device
        )
        lse_lower = torch.empty(batch * block_size, num_heads, dtype=torch.float32, device=device)
        k3_dspark_mla_paged_decode(
            query,
            layer_cache[:, 0],  # [pages, 1, page, head_dim]
            self._mla_decode_workspace(device),
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            fixup.page_tables_i32,
            fixup.seq_lens_i32,
            max_kv_len,
            self.softmax_scale,
            out_lower,
            lse_lower,
        )
        # The block's strict upper triangle, the only keys the kernel's in-block causality hides.
        scores = torch.einsum("bshc,btc->bsht", query, blk_kv).float() * self.softmax_scale
        scores = scores.masked_fill(~fixup.valid, float("-inf"))
        lse_upper = scores.logsumexp(-1)
        probs = torch.softmax(scores, dim=-1).nan_to_num_().to(blk_kv.dtype)
        out_upper = torch.einsum("bsht,btc->bshc", probs, blk_kv[..., : self.kv_lora_rank])
        # The flash-decoding combine: w_u / (w_l + w_u) = sigmoid(lse_u - lse_l), the kernel's LSE in base 2.
        lse_l = lse_lower.view(batch, block_size, num_heads).float() * _TRTLLM_GEN_LSE_LN2
        weight_upper = torch.sigmoid(lse_upper - lse_l).unsqueeze(-1)
        merged = torch.lerp(out_lower.float(), out_upper.float(), weight_upper)
        return merged.to(query.dtype)

    def dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache,
        ctx_v_cache,
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache=None,
        ctx_page_table: Optional[torch.Tensor] = None,
        block_placement=None,
    ) -> torch.Tensor:
        """The block's hidden states ``[B * block, hidden]`` over the latent context: the paged one (``ctx_kv_cache``
        per layer ``[pages, 1, 1, page, 576]``, rows ``ctx_cache_batch_idx`` of ``ctx_page_table``), or without a
        page table the arena ``ctx_k_cache`` ([slots, layers, columns, 1, 576]) at slots ``ctx_cache_batch_idx``.
        ``block_placement``: where the worker already placed the block in the pages.

        The block decode (`_block_decode_variant`): CUTEDSL's or TRTLLM's kernel on the pages; VANILLA's eager torch
        attention over the gathered context and the block (also every backend's on the arena)."""
        del ctx_v_cache
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        batch, block_size = noise_embedding.shape[0], noise_embedding.shape[1]
        device = noise_embedding.device
        num_heads, latent_dim = self._num_heads, self.kv_lora_rank
        nope, rope = self.qk_nope_head_dim, self.qk_rope_head_dim
        blk_freqs = self._mla_freqs_cis(device)[query_positions.long()]
        slots = ctx_cache_batch_idx.long()
        paged = ctx_page_table is not None
        variant = self._block_decode_variant(paged)
        page_tables = None if not paged else ctx_page_table.index_select(0, slots)
        page_size = None if not paged else ctx_kv_cache[0].shape[-2]
        capacity = ctx_k_cache.shape[2] if not paged else ctx_page_table.shape[1] * page_size
        fixup = key_valid = None
        if variant == "eager":
            if paged:
                page_tables = page_tables.long()
            # Context columns below each request's length, then the whole block (non-causal).
            ctx_valid = torch.arange(capacity, device=device).unsqueeze(0) < num_ctx_per_req[
                :batch
            ].view(-1, 1)
            key_valid = torch.cat(
                [ctx_valid, torch.ones(batch, block_size, dtype=torch.bool, device=device)], dim=1
            )
        elif block_placement is not None:
            fixup = _BlockFixup(
                pages=block_placement.pages,
                offsets=block_placement.offsets,
                valid=_block_upper_triangle(block_size, device),
                page_tables_i32=page_tables,
                seq_lens_i32=block_placement.seq_lens,
                kv_bounds_i32=block_placement.kv_bounds,
            )
        else:
            ctx_len = num_ctx_per_req[:batch]
            page_tables_long = page_tables.long()
            pos = ctx_len.view(-1, 1) + torch.arange(block_size, device=device)
            seq_lens = (ctx_len + block_size).to(torch.int32)
            fixup = _BlockFixup(
                pages=page_tables_long.gather(1, pos // page_size),
                offsets=pos % page_size,
                valid=_block_upper_triangle(block_size, device),
                page_tables_i32=page_tables_long.to(torch.int32),
                seq_lens_i32=seq_lens,
                kv_bounds_i32=seq_lens.repeat_interleave(block_size),
            )

        hidden_states = noise_embedding
        residual = None
        partial, partial_all_reduce = None, None
        for layer_idx, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if residual is None:
                residual = hidden_states.clone()
                hs_normed = layer.input_layernorm(hs_flat)
            elif partial is not None:
                hs_normed, res_flat = _all_reduce_add_norm(
                    partial_all_reduce,
                    partial,
                    residual.reshape(-1, residual.shape[-1]),
                    layer.input_layernorm,
                )
                residual = res_flat.reshape(batch, block_size, -1)
            else:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed, res_flat = layer.input_layernorm(hs_flat, res_flat)
                residual = res_flat.reshape(batch, block_size, -1)

            q = F.linear(
                attn.q_a_layernorm(F.linear(hs_normed, attn.q_a_proj.weight)), attn.q_b_proj.weight
            )
            q = q.view(batch, block_size, num_heads, nope + rope)
            query = torch.empty(
                (batch, block_size, num_heads, latent_dim + rope), dtype=q.dtype, device=q.device
            )
            freqs_real = torch.view_as_real(blk_freqs).reshape(-1, blk_freqs.shape[-1], 2)
            if _mla_ops.fused_rope_supported(
                q, None, freqs_real, num_heads, rope, q.shape[-1] - rope
            ):
                k3_dspark_rope_into(q, freqs_real, query, num_heads, rope, latent_dim)
            else:
                q = rmsnorm_rope(
                    q,
                    None,
                    0.0,
                    rope,
                    blk_freqs,
                    num_heads=num_heads,
                    apply_weight=False,
                    apply_rmsnorm=False,
                )
                query[..., latent_dim:] = q[..., nope:]
            q_nope = q[..., :nope]
            bmm_out(
                q_nope.view(batch * block_size, num_heads, nope).transpose(0, 1),
                attn._k_b_proj.to(q.dtype),
                query[..., :latent_dim]
                .view(batch * block_size, num_heads, latent_dim)
                .transpose(0, 1),
            )
            blk_latent = F.linear(hs_normed, attn.kv_a_proj_with_mqa.weight).view(
                batch, block_size, -1
            )
            blk_kv = rmsnorm_rope(
                blk_latent,
                attn.kv_a_layernorm.weight,
                attn.kv_a_layernorm.variance_epsilon,
                rope,
                blk_freqs,
                norm_dim=latent_dim,
            )
            if variant != "eager":
                ctx_out = self._block_attention(
                    variant, query, blk_kv, ctx_kv_cache[layer_idx], fixup, block_size, capacity
                )
            else:
                if not paged:
                    ctx = ctx_k_cache[slots, layer_idx, :, 0, :]
                else:
                    ctx = ctx_kv_cache[layer_idx][:, 0, 0][page_tables].reshape(batch, capacity, -1)
                ctx_out = self._eager_attention(query, ctx, blk_kv, key_valid, batch)
            attn_out = torch.einsum("bshc,hvc->bshv", ctx_out, attn._v_b_proj.to(ctx_out.dtype))

            res_flat = residual.reshape(-1, residual.shape[-1])
            if self._tp_split:
                hidden_out = F.linear(attn_out.reshape(batch * block_size, -1), attn.o_proj.weight)
                hidden_out, res_flat = _all_reduce_add_norm(
                    attn.o_proj.all_reduce, hidden_out, res_flat, layer.post_attention_layernorm
                )
                partial = layer.mlp.partial(hidden_out)
                partial_all_reduce = layer.mlp.down_proj.all_reduce
                hidden_states = partial
            else:
                hidden_out = attn.o_proj(attn_out.reshape(batch * block_size, -1))
                hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)
                hidden_out = layer.mlp(hidden_out)
                hidden_states = hidden_out.reshape(batch, block_size, -1)
            residual = res_flat.reshape(batch, block_size, -1)

        res_flat = residual.reshape(-1, residual.shape[-1])
        if partial is not None:
            out, _ = _all_reduce_add_norm(partial_all_reduce, partial, res_flat, self.model.norm)
            return out
        out, _ = self.model.norm(hidden_states.reshape(-1, hidden_states.shape[-1]), res_flat)
        return out
