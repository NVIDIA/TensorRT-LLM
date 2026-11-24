# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.distributed import AllReduce
from tensorrt_llm._torch.pyexecutor.config_utils import (
    _is_sliding_attention_layer,
    get_layer_attention_window,
)
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceParams
from tensorrt_llm.llmapi.llm_args import is_dflash2_checkpoint
from tensorrt_llm.logger import logger

from .draft_attention import (
    dflash_trtllm_gen_unavailability_reason,
    get_dflash_fa4_fwd,
    get_dflash_flash_attention,
    get_dflash_paged_append,
    get_dflash_trtllm_gen_ops,
)


def dspark_layer_window_size(
    use_swa: bool, swa_window: int, layer_types, layer_idx: int
) -> tuple[int, int]:
    """flash-attn ``window_size`` for one draft layer of the block decode.

    Sliding-window block decode was introduced by the DSpark drafters
    (deepseek-ai/DeepSpec) -- hence the name -- but it is an attention
    configuration, not part of the DSpark head set, so it stays in the DFlash
    base: the block decode below indexes the resolved windows directly and must
    not depend on a subclass-only attribute.

    Those drafters run the draft block through HF attention with
    ``sliding_window`` set on 'sliding_attention' layers and is_causal=False.
    HF's flash path (transformers/modeling_flash_attention_utils.py) translates
    that to ``window_size = (sliding_window - 1, sliding_window - 1)``, i.e.
    each query attends keys within ``swa_window - 1`` KV-index distance on both
    sides. In the DFlash pool layout KV index == token position, so this limits
    draft queries to the most recent ``swa_window`` context tokens plus the
    (nearby) draft block. Full-attention layers and drafters that do not enable
    the window keep flash-attn's default ``(-1, -1)`` (no window).
    """
    if not use_swa:
        return (-1, -1)
    if (
        layer_types is not None
        and layer_idx < len(layer_types)
        and layer_types[layer_idx] != "sliding_attention"
    ):
        return (-1, -1)
    return (swa_window - 1, swa_window - 1)


def dflash2_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Short dynamic depthwise convolution across one drafted block.

    ``Conv(x)_t = sum_i k_{t,i} * x_{t-i}``, with ``k_{t,i} = base_kernel[i] +
    delta[t, i]`` (a per-position correction shared by every ``group_size``
    channels). Taps reaching past a block's start contribute nothing.

    Args:
        hidden_states: ``[B * block_size, hidden_size]``, request-major.
        delta: ``[B * block_size, taps, hidden_size // group_size]``.
        base_kernel: ``[taps, hidden_size]``.
    Returns:
        ``[B * block_size, hidden_size]``
    """
    taps, hidden_size = base_kernel.shape
    num_groups = hidden_size // group_size
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base_kernel.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    if taps > 1:
        position = torch.arange(hidden_states.shape[0], device=hidden_states.device) % block_size
        for tap in range(1, taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


def dflash2_score_edges(
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    gate: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Score every adjacent candidate pair in a drafted block at once.

    ``S_t(a, b) = U_t(b) + <A(a) * H(h_t), B(b)>``: the drafter's logit for
    ``b`` plus a low-rank bilinear match between predecessor ``a`` and
    successor ``b``. Position 0's predecessor is the anchor token.

    Accumulated in fp32; bf16 rounding moves candidate argmaxes often enough to
    change the walk's path.

    Args:
        predecessor_codebook / successor_codebook: ``[vocab_size, rank]``.
        candidate_ids: ``[B, K, top_k]`` per-position candidate token ids.
        unary_logits: ``[B, K, top_k]`` drafter logits for those candidates.
        gate: ``[B, K, rank]`` projected hidden state per block position.
        anchor_token_ids: ``[B]``.
    Returns:
        ``[B, K, top_k, top_k]`` fp32 scores indexed
        ``[batch, position, predecessor, candidate]``.
    """
    top_k = candidate_ids.shape[-1]
    successors = successor_codebook[candidate_ids].float()
    predecessor_ids = torch.cat(
        (anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]), dim=1
    )
    predecessors = predecessor_codebook[predecessor_ids].float()
    return unary_logits.float()[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * gate.float()[:, :, None], successors
    )


def dflash2_walk_candidate_paths(
    candidate_ids: torch.Tensor,
    edge_scores: torch.Tensor,
) -> torch.Tensor:
    """Walk the scored lattice and return the realized score row per position.

    Starting from the anchor, each step keeps the highest-scoring successor and
    conditions the next step on it. Sequential, but only over precomputed
    scores: ``top_k`` values per step, no vocabulary work.

    Args:
        candidate_ids: ``[B, K, top_k]``.
        edge_scores: ``[B, K, top_k, top_k]`` from :func:`dflash2_score_edges`.
    Returns:
        ``[B, K, top_k]`` scores of each position's candidates under the
        predecessor the walk settled on.
    """
    batch, num_steps, top_k, _ = edge_scores.shape
    realized = edge_scores.new_empty(batch, num_steps, top_k)
    # Step 0 replicates the anchor across the predecessor axis, so any row works.
    predecessor = torch.zeros(batch, 1, 1, dtype=torch.long, device=edge_scores.device)
    for step in range(num_steps):
        scores = edge_scores[:, step].gather(1, predecessor.expand(-1, 1, top_k)).squeeze(1)
        realized[:, step] = scores
        predecessor = scores.argmax(-1).view(batch, 1, 1)
    return realized


class DFlashForCausalLM(nn.Module):
    """Draft model wrapper for DFlash speculative decoding.

    DFlash uses cross-attention where Q comes from noise/query tokens and K/V
    come from the concatenation of target hidden states and noise hidden states.
    The target_hidden stays CONSTANT across all layers (no input_layernorm applied).

    Reference: https://arxiv.org/pdf/2602.06036
    """

    context_kv_only = False

    # Whether ``dflash_attention_backend`` drives this drafter's block decode.
    # Subclasses that bring their own attention set this False: neither backend
    # can express every drafter shape, the ops behind them are optional
    # dependencies, and the worker's per-backend shape checks do not apply.
    _uses_worker_attention_backend = True

    # What ``attention_backend="AUTO"`` resolves to, and what the field may say
    # at all. Per drafter family, because the fastest kernel that can express
    # the shape differs: GQA DFlash cross-attention is what the two worker op
    # sets were built for, while an MLA drafter runs its own block decode and
    # has a third implementation they cannot express.
    _default_attention_backend = "VANILLA"
    _uses_lilicorr = False
    _supported_attention_backends = ("VANILLA", "TRTLLM", "FA4")
    # Where AUTO lands when the preferred backend cannot run. None means AUTO
    # propagates the reason instead: a family whose only deployable target can
    # run the fast kernel gains nothing from a silently slower path.
    _auto_fallback_attention_backend = "VANILLA"

    # Whether the drafter's context KV lives in the draft KV cache manager's
    # paged pool rather than a private arena dense in max_seq_len. Orthogonal
    # to the attention backend: paging is about where the KV lives, the backend
    # is about which kernel reads it.
    _paged_ctx_cache = False

    # Whether the drafter runs its TP fusions under plain TP (tp > 1, no
    # attention DP): fc keeps only this rank's input features and one
    # all-reduce sums the partial projections with hidden_norm fused in, and
    # the drafter's own block decode may fuse its all-reduces with the residual
    # add and the RMSNorm after them. They change the drafter's rounding, so a
    # family declares this once its acceptance length has been measured with
    # them; the others keep the unsplit fc and the unfused norms.
    _tp_fusions = False

    # Positions the worker will actually serve, published once by
    # DFlashDrafter._lazy_init_ctx_buffers before the first forward. A drafter
    # that sizes an absolute-position table reads it in place of its
    # config-derived cap; None means no worker has run (direct construction in
    # tests), so the config cap stands.
    _runtime_position_ceiling = None

    @classmethod
    def _resolve_auto_attention_backend(cls) -> str:
        """Turn ``AUTO`` into a concrete backend for this drafter family."""
        want = cls._default_attention_backend
        reason = cls._attention_backend_unavailability_reason(want)
        if reason is None:
            return want
        fallback = cls._auto_fallback_attention_backend
        if fallback is None:
            raise ValueError(
                f"{cls.__name__} has no usable attention backend: its default "
                f"{want!r} is unavailable ({reason}) and this family does not "
                f"degrade."
            )
        logger.info_once(
            f"{cls.__name__} prefers the {want} attention backend but it is "
            f"unavailable ({reason}); falling back to {fallback}.",
            key=f"dflash_auto_backend_fallback_{cls.__name__}",
        )
        return fallback

    @classmethod
    def _attention_backend_unavailability_reason(cls, backend: str) -> Optional[str]:
        """Why this build cannot run ``backend``, or None. Overridden per family.

        The same backend NAME resolves to different kernels per drafter family
        -- GQA TRTLLM is the trtllm-gen FMHA op set, MLA TRTLLM is flashinfer's
        absorbed-MLA paged decode -- so the probe belongs to the class rather
        than to the resolver, which only knows the name.
        """
        if backend == "TRTLLM":
            return dflash_trtllm_gen_unavailability_reason()
        return None

    @classmethod
    def check_valid_attention_backend(cls, backend: str) -> None:
        """Raise unless this build can actually run ``backend``.

        Both halves are fatal on an EXPLICIT request: a typo, and a backend the
        build cannot serve. Silently running something else is what this
        replaces -- the MLA drafter used to fall through to its eager reference,
        correct but orders slower, with nothing raised.
        """
        if backend not in cls._supported_attention_backends:
            raise ValueError(
                f"{cls.__name__} attention backend must be one of "
                f"{list(cls._supported_attention_backends)}, got {backend!r}."
            )
        reason = cls._attention_backend_unavailability_reason(backend)
        if reason is not None:
            raise ValueError(
                f"attention_backend={backend!r} was requested but it is "
                f"unavailable for {cls.__name__}: {reason}."
            )

    def __init__(self, draft_config, *, dflash_attention_backend: str = "AUTO"):
        """Build the draft model, resolving its architecture from the draft config
        (falling back to a model_type-derived name when the checkpoint uses a
        custom DFlash architecture label)."""
        super().__init__()

        from .draft_backbone import K3DsparkForCausalLM

        DraftModelClass = K3DsparkForCausalLM
        if dflash_attention_backend == "AUTO":
            dflash_attention_backend = self._resolve_auto_attention_backend()
        self._backbone_cls_name = DraftModelClass.__name__
        self._helix_mapping = self._shard_context_kv_by_helix(
            draft_config, dflash_attention_backend
        )

        # Remove spec_config to prevent recursive spec-dec initialization
        draft_config_no_spec = replace(draft_config, spec_config=None, lm_head_gather_output=False)
        # ModelConfig.extra_attrs is init=False, so dataclasses.replace() does
        # not preserve the shared custom-op registries.
        draft_config_no_spec.extra_attrs = draft_config.extra_attrs

        # Weights will be loaded later by ModelLoader.load_draft_weights()
        self.draft_model_full = DraftModelClass(draft_config_no_spec)
        self.model = self.draft_model_full.model
        self.lm_head = self.draft_model_full.lm_head

        # Required by weight mappers
        self.model_config = draft_config_no_spec
        self.config = draft_config_no_spec.pretrained_config

        # Get mask_token_id from dflash_config
        pretrained_config = draft_config.pretrained_config
        dflash_config = getattr(pretrained_config, "dflash_config", None) or {}
        self.mask_token_id = dflash_config.get(
            "mask_token_id",
            getattr(pretrained_config, "mask_token_id", pretrained_config.vocab_size),
        )

        self.target_layer_ids = dflash_config.get("target_layer_ids", None)
        # Upstream keeps the dflash_config override; rubin-advance dropped it.
        self.block_size = dflash_config.get(
            "block_size", getattr(pretrained_config, "block_size", None)
        )
        if dflash_attention_backend == "AUTO":
            dflash_attention_backend = self._resolve_auto_attention_backend()
        self.dflash_attention_backend = dflash_attention_backend
        self.check_valid_attention_backend(self.dflash_attention_backend)
        # Each backend loads only its own ops; the rest stay None so the
        # shared paged prologue can read them unconditionally.
        self._dflash_flash_attention = None
        self._dflash_trtllm_gen_ops = None
        self._dflash_fa4_fwd = None
        self._dflash_paged_append = None
        if not self._uses_worker_attention_backend:
            # Still validated above so a typo fails here rather than silently,
            # but no op set is loaded: this drafter calls none of them.
            logger.info_once(
                f"{type(self).__name__} brings its own block decode; "
                f"attention_backend={self.dflash_attention_backend!r} selects "
                f"among its implementations, not the shared DFlash op sets.",
                key=f"dflash_own_attention_{type(self).__name__}",
            )
        elif self.dflash_attention_backend == "VANILLA":
            self._dflash_flash_attention = get_dflash_flash_attention()
        elif self.dflash_attention_backend == "TRTLLM":
            self._dflash_trtllm_gen_ops = get_dflash_trtllm_gen_ops()
        elif self.dflash_attention_backend == "FA4":
            self._dflash_fa4_fwd = get_dflash_fa4_fwd()
            self._dflash_paged_append = get_dflash_paged_append()
        else:
            # Not the user's typo -- check_valid_attention_backend rejected
            # those above. This is a subclass that widened
            # _supported_attention_backends without adding the branch that
            # loads the ops, which a bare else would answer by silently
            # handing it FA4's.
            raise ValueError(
                f"{type(self).__name__} allows attention_backend="
                f"{self.dflash_attention_backend!r} but loads no op set for it."
            )
        self._dflash_trtllm_gen_workspace = None
        self._dflash_trtllm_gen_counters = None
        self.register_buffer("_dflash_batch_indices", None, persistent=False)
        self.register_buffer("_dflash_block_offsets", None, persistent=False)
        self._dflash_trtllm_gen_device = None
        self._dflash_trtllm_gen_sm_count = None
        logger.info(
            f"DFlash draft model initialized with mask_token_id: {self.mask_token_id}, "
            f"target_layer_ids: {self.target_layer_ids}, block_size: {self.block_size}, "
            f"attention_backend: {self.dflash_attention_backend}"
        )

        # Sliding-window block decode (use_swa / swa_window_size). Kept in the
        # base class rather than in the DSpark subclass because the block decode
        # below indexes ``self._layer_windows`` directly: a base-class forward
        # must not reach for an attribute only a subclass defines. Drafters that
        # leave use_swa unset get all-(-1,-1) windows, i.e. a no-op.
        self._use_swa = bool(dflash_config.get("use_swa", False))
        self._swa_window = int(dflash_config.get("swa_window_size", 0) or 0)
        if self._use_swa and self._swa_window < 1:
            raise ValueError(
                "DFlash drafter sets use_swa but swa_window_size="
                f"{dflash_config.get('swa_window_size')} is invalid."
            )
        # Per-layer flash-attn window for the block decode, resolved once.
        num_draft_layers = getattr(pretrained_config, "num_hidden_layers", 0)
        layer_types = getattr(pretrained_config, "layer_types", None)
        self._layer_windows = [
            dspark_layer_window_size(self._use_swa, self._swa_window, layer_types, i)
            for i in range(num_draft_layers)
        ]

        # DFlash 2 (https://inco.ai/blog/dflash2/) keeps the DFlash block decode
        # and adds per-sublayer convolutions plus a pairwise candidate selector.
        # Both are keyed off the draft checkpoint rather than a separate
        # decoding_type, matching how SGLang and vLLM serve DFlash 2 drafters.
        self._dflash2_conv_taps = int(dflash_config.get("conv_kernel_size", 0) or 0)
        self._dflash2_conv_group_size = int(dflash_config.get("conv_group_size", 0) or 0)
        self._dflash2_selector_rank = int(dflash_config.get("selector_rank", 0) or 0)
        self._dflash2_selector_top_k = int(dflash_config.get("selector_top_k", 0) or 0)
        # LiLiCorr supplies its own scorer and permits convolution-free checkpoints.
        self._is_dflash2 = not self._uses_lilicorr and is_dflash2_checkpoint(
            getattr(pretrained_config, "architectures", None), dflash_config
        )
        if self._is_dflash2:
            self._validate_dflash2_config()
            logger.info(
                "DFlash 2 drafter: conv_kernel_size="
                f"{self._dflash2_conv_taps}, conv_group_size="
                f"{self._dflash2_conv_group_size}, selector_rank="
                f"{self._dflash2_selector_rank}, selector_top_k="
                f"{self._dflash2_selector_top_k}"
            )
        # Built in load_weights() from the checkpoint, like fc / hidden_norm.
        self.w = nn.ParameterDict()
        self.attention_convs = None
        self.mlp_convs = None
        self.candidate_selector = None

        self.logits_processor = None  # Set by caller after construction

        # RoPE - lazily initialized from draft model's attention module
        self._rope_initialized = False
        self._rotary_cos_sin = None
        self._is_neox = True

        self._cos_sin_cache_fp32 = None
        self._rope_dummy_q = None

        # Lazy-built after weights load (see _build_fused_kv_buffers).
        self._fused_kv_weight = None
        self._fused_kv_bias = None
        self._k_norm_stacked = None
        self._k_norm_eps = None
        self._num_attn_layers = 0
        self._num_heads = 0
        self._head_dim = 0
        self._num_kv_heads = 0
        # Cache halves per token per layer: K and V for a GQA drafter, one MLA
        # latent (and no V) for an MLA one. The worker sizes its context arena
        # and validates the managed pool against this, so a drafter that stores
        # a single tensor must override it (see MLADSparkForCausalLM).
        self._kv_factor = 2
        # The drafter's TP fusions (see _tp_fusions) run under plain
        # TP: fc keeps only this rank's input features (_project_features),
        # and a drafter's block decode may fuse its all-reduces with the
        # residual add and the following RMSNorm.
        mapping = self.model_config.mapping
        self._tp_split = (
            self._tp_fusions and mapping.tp_size > 1 and not mapping.enable_attention_dp
        )
        # This rank's input features of fc and the all-reduce that sums the
        # partial projections, set by _load_target_projection under _tp_split
        # when the fc input width splits over tp_size.
        self._fc_input_slice: Optional[slice] = None
        self._fc_all_reduce: Optional[AllReduce] = None
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False
        # Laguna-specific draft-layer behaviors, disabled by default so generic
        # DFlash drafters keep the original contract (no context input_layernorm,
        # non-causal block attention). Subclasses opt in.
        self._context_input_layernorm = False
        self._sliding_layers_causal = False
        self._validate_gqa_shape()
        self._warn_inferred_attention_windows()

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project captured target hidden states into the draft hidden space.

        Generic DFlash: fc then hidden_norm. Subclasses (e.g. Laguna) may
        normalize the per-aux features first by overriding this method.
        """
        return self._project_features(hidden_states.to(self.w["fc"].dtype))

    def _project_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``hidden_norm(fc(hidden_states))`` for ``[N, in_features]`` features.

        With the TP fusions (``_tp_split``) every rank holds the same features,
        so each multiplies its slice of them by its slice of the ``fc`` columns
        and one all-reduce sums the partials with ``hidden_norm`` fused in:
        every rank reads ``1 / tp`` of the ``fc`` weight.
        """
        if self._fc_input_slice is None:
            return F.rms_norm(
                F.linear(hidden_states, self.w["fc"]),
                (self.config.hidden_size,),
                self.w["hidden_norm"],
                getattr(self.config, "rms_norm_eps", 1e-6),
            )
        partial = F.linear(hidden_states[:, self._fc_input_slice], self.w["fc"])
        return self._fc_all_reduce(
            partial,
            all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RMS_NORM,
                norm_weight=self.w["hidden_norm"],
                eps=self.config.rms_norm_eps,
            ),
        )

    def _validate_dflash2_config(self) -> None:
        """Require the full DFlash 2 recipe; a partial one would silently
        degrade to DFlash 1 acceptance instead of failing."""
        required = {
            "conv_kernel_size": self._dflash2_conv_taps,
            "conv_group_size": self._dflash2_conv_group_size,
            "selector_rank": self._dflash2_selector_rank,
            "selector_top_k": self._dflash2_selector_top_k,
        }
        missing = sorted(name for name, value in required.items() if value < 1)
        if missing:
            raise ValueError(
                "DFlash 2 drafter is missing required dflash_config entries "
                f"{missing}; got {required}."
            )
        if self._dflash2_selector_top_k < 2:
            raise ValueError(
                f"DFlash 2 selector_top_k={self._dflash2_selector_top_k} leaves "
                "no candidates to choose between; it must be at least 2."
            )

    @property
    def is_dflash2(self) -> bool:
        return self._is_dflash2

    @property
    def has_block_conv(self) -> bool:
        return self.attention_convs is not None

    @property
    def has_candidate_selector(self) -> bool:
        return self.candidate_selector is not None

    def select_candidate_path(
        self,
        block_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        block_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Restrict each block position's logits to the selector's chosen path.

        Writes ``-inf`` into ``block_logits`` everywhere except each position's
        candidates, which carry the score row the walk realized. Argmax
        therefore reproduces the walk, and sampling a row samples the
        selector's own proposal distribution.

        Args:
            block_logits: ``[B, K, vocab_size]`` full-vocab destination.
            candidate_ids: ``[B, K, top_k]`` full-vocab candidate token ids.
            unary_logits: ``[B, K, top_k]`` drafter logits for those candidates.
            block_hidden_states: ``[B, K, hidden_size]`` at the drafted
                positions, after the drafter's final norm.
            anchor_token_ids: ``[B]`` last accepted token per request.
        """
        gate = F.linear(
            block_hidden_states.to(self.w["hidden_projection__weight"].dtype),
            self.w["hidden_projection__weight"],
        )
        edge_scores = dflash2_score_edges(
            self.w["predecessor_codebook"],
            self.w["successor_codebook"],
            candidate_ids,
            unary_logits,
            gate,
            anchor_token_ids,
        )
        realized_scores = dflash2_walk_candidate_paths(candidate_ids, edge_scores)
        block_logits.fill_(float("-inf"))
        block_logits.scatter_(-1, candidate_ids, realized_scores.to(block_logits.dtype))
        return block_logits

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load weights into the DFlash draft model.

        DFlash checkpoints differ from standard HF format:
        - Layer weights lack the 'model.' prefix (e.g., 'layers.0...' not 'model.layers.0...')
        - Extra DFlash-specific weights: 'fc.weight', 'hidden_norm.weight'
        - Missing embed_tokens and lm_head (shared with target model)
        """
        # Taken before the backbone remap, which would prefix these with
        # 'model.' and then silently drop them under allow_partial_loading.
        if self._is_dflash2:
            weights = self._load_dflash2_weights(weights)

        # Remap: add 'model.' prefix where needed, and extract DFlash-specific weights
        remapped = {}
        for key, value in weights.items():
            if key.startswith("fc.") or key == "hidden_norm.weight":
                # DFlash-specific projection weights - store directly
                remapped[key] = value
            elif key == "norm.weight":
                remapped["model.norm.weight"] = value
            elif not key.startswith("model."):
                remapped[f"model.{key}"] = value
            else:
                remapped[key] = value

        # Wrapper-owned and built FROM the checkpoint, so they are never in
        # draft_model_full's module tree for _assert_backbone_complete to walk.
        # Without fc the drafter has no capture projection and drafts from an
        # empty context forever.
        wrapper_missing = [k for k in self.WRAPPER_OWNED_WEIGHTS if k not in remapped]
        if wrapper_missing:
            raise ValueError(
                f"{type(self).__name__}: checkpoint is missing {wrapper_missing}, "
                "which this wrapper owns and builds from the checkpoint."
            )

        # Load DFlash-specific weights directly
        self._load_target_projection(remapped)

        if "hidden_norm.weight" in remapped:
            self.w["hidden_norm"] = nn.Parameter(
                remapped["hidden_norm.weight"].to("cuda"), requires_grad=False
            )
            del remapped["hidden_norm.weight"]

        # allow_partial_loading is what lets the shared modules below be absent.
        # It is also why a truncated checkpoint loads clean, so gate it on the
        # subclass's own declaration of what may legitimately be missing.
        self._assert_backbone_complete(remapped, weight_mapper)
        self.draft_model_full.load_weights(
            weights=remapped, weight_mapper=weight_mapper, allow_partial_loading=True
        )

    def _load_target_projection(self, weights: Dict) -> None:
        if "fc.weight" not in weights:
            return
        weight = weights.pop("fc.weight")
        # A drafter with the TP fusions keeps only its rank's slice of the
        # input features (see _project_features); a width that does not split
        # over tp_size keeps the whole fc.
        mapping = self.model_config.mapping
        if self._tp_split and weight.shape[1] % mapping.tp_size != 0:
            logger.info(
                f"{type(self).__name__}: fc input width {weight.shape[1]} does not split "
                f"over tp_size={mapping.tp_size}; keeping the unsplit context projection."
            )
        elif self._tp_split:
            width = weight.shape[1] // mapping.tp_size
            self._fc_input_slice = slice(mapping.tp_rank * width, (mapping.tp_rank + 1) * width)
            self._fc_all_reduce = AllReduce(
                mapping, strategy=self.model_config.allreduce_strategy, dtype=weight.dtype
            )
            weight = weight[:, self._fc_input_slice]
        self.w["fc"] = nn.Parameter(weight.to("cuda").contiguous(), requires_grad=False)

    def _load_dflash2_weights(self, weights: Dict, *, load_selector: bool = True) -> Dict:
        """Build the DFlash 2 convolutions and candidate selector.

        Returns ``weights`` minus the keys consumed here.
        """
        consumed = set()

        def take(key: str) -> torch.Tensor:
            if key not in weights:
                raise ValueError(
                    f"DFlash 2 drafter is missing checkpoint weight '{key}'. "
                    "The draft checkpoint declares DFlash 2 in its config but "
                    "does not ship the corresponding weights."
                )
            consumed.add(key)
            return weights[key].to("cuda")

        hidden_size = self.config.hidden_size
        attention_convs = []
        mlp_convs = []
        for layer_idx in range(len(self.model.layers)):
            for name, convs in (("attention_conv", attention_convs), ("mlp_conv", mlp_convs)):
                prefix = f"layers.{layer_idx}.{name}."
                base = take(prefix + "base_kernel")
                projection = take(prefix + "kernel_projection.weight")
                taps, group = self._dflash2_conv_taps, self._dflash2_conv_group_size
                if hidden_size % group or base.shape != (2, taps, hidden_size):
                    raise ValueError(f"Invalid draft convolution shape for {prefix}")
                if projection.shape != (2 * taps * (hidden_size // group), hidden_size):
                    raise ValueError(f"Invalid draft convolution projection for {prefix}")
                key = prefix.replace(".", "__")
                self.w[key + "base"] = nn.Parameter(base, requires_grad=False)
                self.w[key + "projection"] = nn.Parameter(projection, requires_grad=False)
                convs.append((key, taps, group))

        self.attention_convs = attention_convs
        self.mlp_convs = mlp_convs
        if load_selector:
            for field in ("predecessor_codebook", "successor_codebook", "hidden_projection.weight"):
                value = take("candidate_selector." + field)
                expected = (
                    (self._dflash2_selector_rank, hidden_size)
                    if field.endswith("weight")
                    else (self.config.vocab_size, self._dflash2_selector_rank)
                )
                if value.shape != expected:
                    raise ValueError(f"Invalid draft selector shape for {field}: {value.shape}")
                self.w[field.replace(".", "__")] = nn.Parameter(value, requires_grad=False)
            self.candidate_selector = True
        return {k: v for k, v in weights.items() if k not in consumed}

    #: Tensors the wrapper itself owns: not in draft_model_full, built from the
    #: checkpoint in load_weights, and required.
    WRAPPER_OWNED_WEIGHTS = ("fc.weight", "hidden_norm.weight")

    #: Parameter-name prefixes this drafter takes from the target instead of its
    #: own checkpoint. Everything else in ``draft_model_full`` must be provided.
    #: GQA DFlash checkpoints ship neither embedding nor head; an MLA drafter
    #: ships its own embedding and overrides this.
    WEIGHTS_SHARED_WITH_TARGET = ("embed_tokens", "lm_head")

    def _assert_backbone_complete(self, weights, weight_mapper=None):
        from ..weights import draft_checkpoint_keys

        missing = draft_checkpoint_keys(self.model) - weights.keys()
        if missing:
            raise ValueError(f"Draft checkpoint is missing {sorted(missing)[:8]}")

    def load_weights_from_target_model(self, target_model: torch.nn.Module) -> None:
        """Share embed_tokens and lm_head from the target model."""
        self.draft_model_full.model.embed_tokens = target_model.model.embed_tokens
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head

    def _warn_inferred_attention_windows(self) -> None:
        """Warn once at initialization when checkpoint metadata enables SWA."""
        if getattr(self.config, "use_sliding_window", None) is not None:
            return

        num_hidden_layers = getattr(self.config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            num_hidden_layers = len(self.model.layers)
        layers_by_window = {}
        for layer_idx in range(num_hidden_layers):
            window = get_layer_attention_window(self.config, layer_idx)
            if window is not None:
                layers_by_window.setdefault(window, []).append(layer_idx)

        for window, layer_indices in layers_by_window.items():
            logger.warning(
                "DFlash inferred pooled-context sliding-window attention from "
                f"checkpoint config for draft layers {layer_indices}: "
                f"window={window}. Context attention is truncated to {window} "
                "tokens for these layers; if the drafter expects full context, "
                "acceptance rate may drop. Set use_sliding_window explicitly "
                "to confirm or disable windowing."
            )

    def _get_attention_mask_args(self, layer_idx):
        """Return FlashAttention causal and local-window arguments for a layer."""
        layer_types = getattr(self.config, "layer_types", None)
        is_sliding_layer = False
        if layer_types:
            layer_type = layer_types[layer_idx % len(layer_types)]
            is_sliding_layer = _is_sliding_attention_layer(layer_type)

        sliding_window = get_layer_attention_window(self.config, layer_idx)
        is_sliding_layer = is_sliding_layer or sliding_window is not None

        # LiLiCorr uses non-causal block attention with symmetric sliding windows.
        # DFlash 2 honors explicit is_causal metadata when present; otherwise,
        # causality follows the layer-type default.
        explicit_causal = getattr(self.config, "is_causal", None) if self._is_dflash2 else None
        if self._uses_lilicorr:
            causal = False
        elif explicit_causal is not None:
            causal = bool(explicit_causal)
        elif not is_sliding_layer:
            return False, (-1, -1)
        else:
            causal = self._sliding_layers_causal or sliding_window is not None

        if sliding_window is None:
            # Legacy drafters without an explicit window preserve their prior
            # non-windowed behavior.
            return causal, (-1, -1)
        # FlashAttention's bounds are inclusive: W tokens are current + W-1 left.
        # A non-causal sliding layer windows both sides, matching the HF flash
        # path these drafters are trained under (see dspark_layer_window_size).
        if not causal:
            return False, (sliding_window - 1, sliding_window - 1)
        return True, (sliding_window - 1, 0)

    def _resolve_block_attention(self, layer_idx: int) -> tuple[bool, tuple[int, int]]:
        causal, window_size = self._get_attention_mask_args(layer_idx)
        swa_window = (
            self._layer_windows[layer_idx] if layer_idx < len(self._layer_windows) else (-1, -1)
        )
        if swa_window != (-1, -1):
            window_size = swa_window
        return causal, window_size

    def validate_block_attention_windows(self) -> None:
        """Refuse windows the TRTLLM block-decode kernel cannot apply.

        TRTLLM-Gen only implements sliding windows for causal masks: flashinfer
        rejects ``causal=False`` with any finite ``window_left`` outright. A
        non-causal windowed layer must therefore run on VANILLA or FA4, which
        take a two-sided window.
        """
        if self.dflash_attention_backend != "TRTLLM":
            return
        num_layers = getattr(self.config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = len(self.model.layers)
        for layer_idx in range(num_layers):
            causal, (window_left, _) = self._resolve_block_attention(layer_idx)
            if causal or window_left < 0:
                continue
            raise ValueError(
                f"DFlash draft layer {layer_idx} attends non-causally within a "
                f"{window_left + 1}-token sliding window. The TRTLLM DFlash "
                "attention backend does not support non-causal sliding-window "
                "attention. Use attention_backend=VANILLA or FA4."
            )
