# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import replace
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm.logger import logger

from ...functional import RotaryScalingType
from ..modules.rotary_embedding import RotaryEmbedding

try:
    from ..custom_ops import flashinfer_apply_rope_with_cos_sin_cache_inplace as _flashinfer_rope
except ImportError:
    _flashinfer_rope = None
from ..pyexecutor.config_utils import _is_sliding_attention_layer, get_layer_attention_window
from ..speculative.dflash_attention import get_dflash_flash_attention, get_dflash_trtllm_gen_ops
from ..speculative.interface import SpeculativeDecodingMode
from .modeling_utils import get_model_architecture, register_draft_model


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


def _is_dflash2_architecture(config: PretrainedConfig) -> bool:
    """Whether a draft checkpoint advertises itself as a DFlash 2 drafter.

    Matches on the "dflash2" stem (released drafters use ``DFlash2DraftModel``)
    so per-target variants of the label are picked up too.
    """
    return any(
        "dflash2" in str(arch).lower().replace("_", "").replace("-", "")
        for arch in (getattr(config, "architectures", None) or [])
    )


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


class DFlash2BlockConv(nn.Module):
    """Two-tap dynamic depthwise convolutions wrapped around one sublayer.

    DFlash 2 (https://inco.ai/blog/dflash2/) moves the within-block mixing into
    a short convolution. ``prepare`` convolves the sublayer's input and
    ``finish`` its output; a single projection of the *input* drives both, so
    ``finish`` takes the coefficients ``prepare`` hands back.
    """

    def __init__(
        self,
        base_kernel: torch.Tensor,
        kernel_projection_weight: torch.Tensor,
        taps: int,
        group_size: int,
        hidden_size: int,
    ):
        super().__init__()
        # Side 0 wraps the sublayer input, side 1 its output.
        if tuple(base_kernel.shape) != (2, taps, hidden_size):
            raise ValueError(
                f"DFlash2 conv base_kernel has shape "
                f"{tuple(base_kernel.shape)}, expected [2, conv_kernel_size, "
                f"hidden_size] = {(2, taps, hidden_size)}."
            )
        if hidden_size % group_size:
            raise ValueError(
                f"DFlash2 conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        expected = (2 * taps * self.num_groups, hidden_size)
        if tuple(kernel_projection_weight.shape) != expected:
            raise ValueError(
                f"DFlash2 conv kernel_projection has shape "
                f"{tuple(kernel_projection_weight.shape)}, expected "
                f"[2 * conv_kernel_size * hidden_size / conv_group_size, "
                f"hidden_size] = {expected}."
            )
        self.base_kernel = nn.Parameter(base_kernel, requires_grad=False)
        self.kernel_projection = nn.Linear(
            hidden_size,
            expected[0],
            bias=False,
            device=kernel_projection_weight.device,
            dtype=kernel_projection_weight.dtype,
        )
        self.kernel_projection.weight.data.copy_(kernel_projection_weight)

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int, block_size: int
    ) -> torch.Tensor:
        return dflash2_grouped_conv(
            hidden_states, delta, self.base_kernel[side], block_size, self.group_size
        )

    def prepare(
        self, hidden_states: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve the sublayer input; also return ``finish``'s coefficients."""
        coefficients = self.kernel_projection(hidden_states).view(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[:, 0], 0, block_size),
            coefficients[:, 1],
        )

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Convolve the sublayer output with the coefficients from ``prepare``."""
        return self._convolve(hidden_states, coefficients, 1, block_size)


class DFlash2CandidateSelector(nn.Module):
    """Pairwise path selector over a DFlash block's top-k candidates.

    Independent per-position picks leave the right token in the candidate list
    far more often than in first place, so DFlash 2 keeps the top
    ``selector_top_k`` per position and scores every adjacent pair (see
    :func:`dflash2_score_edges`) in one shot for the whole block.
    """

    def __init__(
        self,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        hidden_projection_weight: torch.Tensor,
        top_k: int,
        vocab_size: int,
        rank: int,
    ):
        super().__init__()
        for name, table in (
            ("predecessor_codebook", predecessor_codebook),
            ("successor_codebook", successor_codebook),
        ):
            if tuple(table.shape) != (vocab_size, rank):
                raise ValueError(
                    f"DFlash2 {name} has shape {tuple(table.shape)}, expected "
                    f"[vocab_size, selector_rank] = ({vocab_size}, {rank})."
                )
        if tuple(hidden_projection_weight.shape)[0] != rank:
            raise ValueError(
                "DFlash2 candidate_selector.hidden_projection has shape "
                f"{tuple(hidden_projection_weight.shape)}, expected "
                f"selector_rank={rank} output features."
            )
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(predecessor_codebook, requires_grad=False)
        self.successor_codebook = nn.Parameter(successor_codebook, requires_grad=False)
        self.hidden_projection = nn.Linear(
            hidden_projection_weight.shape[1],
            rank,
            bias=False,
            device=hidden_projection_weight.device,
            dtype=hidden_projection_weight.dtype,
        )
        self.hidden_projection.weight.data.copy_(hidden_projection_weight)

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score all adjacent candidate pairs; see :func:`dflash2_score_edges`."""
        gate = self.hidden_projection(hidden_states.to(self.hidden_projection.weight.dtype))
        return dflash2_score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            gate,
            anchor_token_ids,
        )


class DFlashForCausalLM(nn.Module):
    """Draft model wrapper for DFlash speculative decoding.

    DFlash uses cross-attention where Q comes from noise/query tokens and K/V
    come from the concatenation of target hidden states and noise hidden states.
    The target_hidden stays CONSTANT across all layers (no input_layernorm applied).

    Reference: https://arxiv.org/pdf/2602.06036
    """

    def __init__(self, draft_config, *, dflash_attention_backend: str = "VANILLA"):
        """Build the draft model, resolving its architecture from the draft config
        (falling back to a model_type-derived name when the checkpoint uses a
        custom DFlash architecture label)."""
        super().__init__()

        pretrained_cfg = draft_config.pretrained_config
        try:
            DraftModelClass, _ = get_model_architecture(pretrained_cfg)
        except RuntimeError:
            model_type = pretrained_cfg.model_type
            arch_name = "".join(w.capitalize() for w in model_type.split("_")) + "ForCausalLM"
            logger.info(
                f"DFlash: architecture {pretrained_cfg.architectures} not found, "
                f"falling back to {arch_name} based on model_type={model_type}"
            )
            original_archs = pretrained_cfg.architectures
            try:
                pretrained_cfg.architectures = [arch_name]
                DraftModelClass, _ = get_model_architecture(pretrained_cfg)
            finally:
                pretrained_cfg.architectures = original_archs

        # Remove spec_config to prevent recursive spec-dec initialization
        draft_config_no_spec = replace(draft_config, spec_config=None, lm_head_gather_output=False)

        # Weights will be loaded later by ModelLoader.load_draft_weights()
        self.draft_model_full = DraftModelClass(draft_config_no_spec)
        self.model = self.draft_model_full.model
        self.lm_head = self.draft_model_full.lm_head

        # Required by weight mappers
        self.model_config = draft_config_no_spec
        self.config = draft_config_no_spec.pretrained_config

        # Get mask_token_id from dflash_config
        pretrained_config = draft_config.pretrained_config
        dflash_config = getattr(pretrained_config, "dflash_config", {})
        self.mask_token_id = dflash_config.get(
            "mask_token_id",
            getattr(pretrained_config, "mask_token_id", pretrained_config.vocab_size),
        )

        self.target_layer_ids = dflash_config.get("target_layer_ids", None)
        self.block_size = dflash_config.get(
            "block_size", getattr(pretrained_config, "block_size", None)
        )
        self.dflash_attention_backend = dflash_attention_backend
        if self.dflash_attention_backend == "VANILLA":
            self._dflash_flash_attention = get_dflash_flash_attention()
        elif self.dflash_attention_backend == "TRTLLM":
            self._dflash_trtllm_gen_ops = get_dflash_trtllm_gen_ops()
        else:
            raise ValueError(
                "DFlash attention backend must be VANILLA or TRTLLM, got "
                f"{self.dflash_attention_backend!r}."
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
        self._is_dflash2 = (
            self._dflash2_conv_taps > 0
            or self._dflash2_selector_rank > 0
            or _is_dflash2_architecture(pretrained_config)
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
        self._has_qk_norm = False
        self._use_fused_qk_norm_rope = False
        # Laguna-specific draft-layer behaviors, disabled by default so generic
        # DFlash drafters keep the original contract (no context input_layernorm,
        # non-causal block attention). Subclasses opt in.
        self._context_input_layernorm = False
        self._sliding_layers_causal = False
        self._validate_gqa_shape()
        self._warn_inferred_attention_windows()

    def _validate_gqa_shape(self):
        """Reject a backbone the block decode cannot express.

        The hand-written block decode is GQA-shaped throughout: it splits a
        fused ``qkv_proj`` by (num_heads, num_kv_heads, head_dim), fuses K/V
        across layers on one uniform head dim, and shares a single RoPE cache.
        The registry will happily build a backbone that violates any of that --
        an MLA drafter has no per-head K/V at all -- and without this the
        failure surfaces much later inside ``_build_fused_kv_buffers`` or, worse,
        as silently mis-sliced weights. Raise at construction instead.
        """
        layers = getattr(self.model, "layers", None)
        if not layers:
            return
        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None or not hasattr(attn, "qkv_proj"):
                raise ValueError(
                    f"DFlash block decode requires a fused self_attn.qkv_proj on "
                    f"every draft layer, but layer {idx} of draft backbone "
                    f"{type(self.config).__name__} has none. Backbones that do not "
                    "project per-head Q/K/V (e.g. MLA) need their own block decode."
                )
        num_kv_heads = layers[0].self_attn.num_key_value_heads
        mismatched = [
            idx
            for idx, layer in enumerate(layers[1:], start=1)
            if layer.self_attn.num_key_value_heads != num_kv_heads
        ]
        if mismatched:
            raise ValueError(
                "DFlash fuses draft K/V across layers and needs one uniform "
                f"num_key_value_heads, but layers {mismatched} differ from layer 0 "
                f"({num_kv_heads})."
            )

    @staticmethod
    def _rope_signature(attn):
        """Return the effective RoPE configuration used by an attention layer."""
        if attn.rotary_emb is not None:
            return (
                attn.rotary_emb.rope_params,
                attn.rotary_emb.head_dim,
                attn.rotary_emb.is_neox,
            )
        if attn.pos_embd_params is not None:
            return (
                attn.pos_embd_params.rope,
                attn.head_dim,
                attn.pos_embd_params.is_neox,
            )
        return None

    def _validate_uniform_rope(self):
        """Check that all draft layers can safely share one RoPE cache."""
        if len(self.model.layers) == 0:
            raise ValueError("DFlash requires at least one draft model layer.")

        signatures = [self._rope_signature(layer.self_attn) for layer in self.model.layers]

        mismatched_layers = [
            layer_idx
            for layer_idx, signature in enumerate(signatures[1:], start=1)
            if signature != signatures[0]
        ]
        if mismatched_layers:
            layer_types = getattr(self.config, "layer_types", None)
            raise ValueError(
                "DFlash shares one RoPE cache across draft layers, but layers "
                f"{mismatched_layers} have a different effective RoPE "
                f"configuration from layer 0. layer_types={layer_types}."
            )

    def _init_rope(self):
        """Initialize RoPE from the draft model's attention configuration.

        Reuses the existing RotaryEmbedding infrastructure which correctly
        handles all RoPE variants (standard, YaRN, scaled, etc.).
        """
        # The flattened context-KV path shares layer 0's RoPE cache.
        self._validate_uniform_rope()
        attn0 = self.model.layers[0].self_attn

        if attn0.rotary_emb is not None:
            self._rotary_cos_sin = attn0.rotary_emb.rotary_cos_sin
            self._is_neox = attn0.rotary_emb.is_neox
        elif attn0.pos_embd_params is not None:
            rope_emb = RotaryEmbedding(
                attn0.pos_embd_params.rope,
                head_dim=attn0.head_dim,
                is_neox=attn0.pos_embd_params.is_neox,
            )
            self._rotary_cos_sin = rope_emb.rotary_cos_sin
            self._is_neox = rope_emb.is_neox
        else:
            # Fallback: basic NeoX-style RoPE
            config = self.config
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            rope_theta = getattr(config, "rope_theta", 1000000.0)
            max_pos = getattr(config, "max_position_embeddings", 32768)

            inv_freq = 1.0 / (
                rope_theta
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda") / head_dim)
            )
            positions = torch.arange(max_pos, dtype=torch.float32, device="cuda")
            freqs = torch.outer(positions, inv_freq)
            rope_cos = freqs.cos().to(config.torch_dtype)
            rope_sin = freqs.sin().to(config.torch_dtype)
            # [max_pos, 2, rot_dim//2] to match RotaryEmbedding format
            self._rotary_cos_sin = torch.stack([rope_cos, rope_sin], dim=1)
            self._is_neox = True

        self._rope_initialized = True

    def project_target_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project captured target hidden states into the draft hidden space.

        Generic DFlash: fc then hidden_norm. Subclasses (e.g. Laguna) may
        normalize the per-aux features first by overriding this method.
        """
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        return self.hidden_norm(self.fc(hidden_states))

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
        edge_scores = self.candidate_selector(
            candidate_ids, unary_logits, block_hidden_states, anchor_token_ids
        )
        realized_scores = dflash2_walk_candidate_paths(candidate_ids, edge_scores)
        block_logits.fill_(float("-inf"))
        block_logits.scatter_(-1, candidate_ids, realized_scores.to(block_logits.dtype))
        return block_logits

    def _post_attention_gate(self, attn_output, gate_input, attn_mod, num_heads, head_dim):
        """Hook applied to the block-attention output before o_proj.

        No-op for generic DFlash; overridden by drafters that gate (e.g. Laguna).
        """
        return attn_output

    def load_weights(self, weights: Dict, weight_mapper=None, **kwargs):
        """Load weights into the DFlash draft model.

        DFlash checkpoints differ from standard HF format:
        - Layer weights lack the 'model.' prefix (e.g., 'layers.0...' not 'model.layers.0...')
        - Extra DFlash-specific weights: 'fc.weight', 'hidden_norm.weight'
        - Missing embed_tokens and lm_head (shared with target model)
        """
        # Laguna DFlash checkpoints may ship a fused self_attn.qkv_proj; the draft
        # loader expects split q/k/v (a fused key is silently dropped otherwise).
        if any(k.endswith("self_attn.qkv_proj.weight") for k in weights):
            for attr in ("num_attention_heads_per_layer", "num_key_value_heads_per_layer"):
                per_layer = getattr(self.config, attr, None)
                if per_layer is not None and len(set(per_layer)) > 1:
                    raise ValueError(
                        "DFlash load_weights() splits the fused qkv_proj using "
                        "the global head count, but the drafter has heterogeneous "
                        f"{attr} {sorted(set(per_layer))}; per-layer qkv splitting "
                        "is required for this checkpoint."
                    )
            head_dim = getattr(
                self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads
            )
            num_kv_heads = getattr(
                self.config, "num_key_value_heads", self.config.num_attention_heads
            )
            q = self.config.num_attention_heads * head_dim
            kv = num_kv_heads * head_dim
            split = {}
            for k, v in weights.items():
                if k.endswith("self_attn.qkv_proj.weight"):
                    b = k[: -len("qkv_proj.weight")]
                    split[b + "q_proj.weight"] = v[:q]
                    split[b + "k_proj.weight"] = v[q : q + kv]
                    split[b + "v_proj.weight"] = v[q + kv :]
                else:
                    split[k] = v
            weights = split

        # Taken before the backbone remap, which would prefix these with
        # 'model.' and then silently drop them under allow_partial_loading.
        if self._is_dflash2:
            weights = self._load_dflash2_weights(weights)

        # Remap: add 'model.' prefix where needed, and extract DFlash-specific weights
        remapped = {}
        for key, value in weights.items():
            if key in ("fc.weight", "hidden_norm.weight"):
                # DFlash-specific projection weights - store directly
                remapped[key] = value
            elif key == "norm.weight":
                remapped["model.norm.weight"] = value
            elif not key.startswith("model."):
                remapped[f"model.{key}"] = value
            else:
                remapped[key] = value

        # Load DFlash-specific weights directly
        if "fc.weight" in remapped:
            self.fc = nn.Linear(
                remapped["fc.weight"].shape[1],
                remapped["fc.weight"].shape[0],
                bias=False,
                device="cuda",
                dtype=remapped["fc.weight"].dtype,
            )
            self.fc.weight.data.copy_(remapped["fc.weight"])
            del remapped["fc.weight"]

        if "hidden_norm.weight" in remapped:
            rms_norm_eps = getattr(self.config, "rms_norm_eps", 1e-6)
            self.hidden_norm = nn.RMSNorm(
                remapped["hidden_norm.weight"].shape[0],
                eps=rms_norm_eps,
                device="cuda",
                elementwise_affine=True,
                dtype=remapped["hidden_norm.weight"].dtype,
            )
            self.hidden_norm.weight.data.copy_(remapped["hidden_norm.weight"])
            del remapped["hidden_norm.weight"]

        # Load remaining weights into the draft model.
        # DFlash checkpoints don't include embed_tokens or lm_head, so allow partial loading
        # since those modules won't find matching weights.
        self.draft_model_full.load_weights(
            weights=remapped, weight_mapper=weight_mapper, allow_partial_loading=True
        )

    def _load_dflash2_weights(self, weights: Dict) -> Dict:
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
        attention_convs = nn.ModuleList()
        mlp_convs = nn.ModuleList()
        for layer_idx in range(len(self.model.layers)):
            for name, convs in (("attention_conv", attention_convs), ("mlp_conv", mlp_convs)):
                prefix = f"layers.{layer_idx}.{name}."
                convs.append(
                    DFlash2BlockConv(
                        base_kernel=take(prefix + "base_kernel"),
                        kernel_projection_weight=take(prefix + "kernel_projection.weight"),
                        taps=self._dflash2_conv_taps,
                        group_size=self._dflash2_conv_group_size,
                        hidden_size=hidden_size,
                    )
                )

        selector = DFlash2CandidateSelector(
            predecessor_codebook=take("candidate_selector.predecessor_codebook"),
            successor_codebook=take("candidate_selector.successor_codebook"),
            hidden_projection_weight=take("candidate_selector.hidden_projection.weight"),
            top_k=self._dflash2_selector_top_k,
            vocab_size=self.config.vocab_size,
            rank=self._dflash2_selector_rank,
        )

        self.attention_convs = attention_convs
        self.mlp_convs = mlp_convs
        self.candidate_selector = selector
        return {k: v for k, v in weights.items() if k not in consumed}

    def load_weights_from_target_model(self, target_model: torch.nn.Module) -> None:
        """Share embed_tokens and lm_head from the target model."""
        self.draft_model_full.model.embed_tokens = target_model.model.embed_tokens
        self.draft_model_full.lm_head = target_model.lm_head
        self.lm_head = target_model.lm_head

    def precompute_context_kv(
        self,
        projected_hidden: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-norm / post-RoPE K and V for ALL drafter layers in one fused GEMM.

        Args:
            projected_hidden: [N, hidden_size], already fc + hidden_norm'd.
            positions:        [N] int32/64, RoPE positions for each entry.
        Returns:
            k: [N, L, nkv, hd]  post k_norm and RoPE
            v: [N, L, nkv, hd]  post split only
        """
        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()
        N = projected_hidden.shape[0]
        L = self._num_attn_layers
        nkv = self._num_kv_heads
        hd = self._head_dim
        weight_dtype = self._fused_kv_weight.dtype
        if getattr(self, "_input_ln_eps", None) is not None:
            ph = projected_hidden.float()
            ph = ph * torch.rsqrt(ph.pow(2).mean(-1, keepdim=True) + self._input_ln_eps)
            projected_hidden = ph.to(weight_dtype)
        elif projected_hidden.dtype != weight_dtype:
            projected_hidden = projected_hidden.to(weight_dtype)

        kv_flat = F.linear(projected_hidden, self._fused_kv_weight, self._fused_kv_bias)
        # Per-layer layout [L0_K|L0_V|L1_K|L1_V|...] keeps K and V contiguous
        # after the select() splits — no extra copy required.
        kv = kv_flat.view(N, L, 2, nkv, hd)
        k = kv[:, :, 0].contiguous()
        v = kv[:, :, 1].contiguous()

        if self._k_norm_stacked is not None:
            # Fuse L per-layer RMSNorms into one. k is [N, L, nkv, hd];
            # each layer has its own weight ([L, hd]) but shares eps.
            k = F.rms_norm(k, (hd,), eps=self._k_norm_eps)
            k = k * self._k_norm_stacked.view(1, L, 1, hd)

        self._fused_rope_inplace(k.view(N * L, nkv * hd), positions, N, L)
        return k, v

    def _get_cos_sin_cache(self) -> torch.Tensor:
        """Return the flashinfer-style cos/sin cache for the drafter.

        Shape [max_positions, head_dim], fp32 — flashinfer's
        apply_rope_with_cos_sin_cache_inplace requires fp32 regardless of
        the query/key dtype.
        """
        if self._cos_sin_cache_fp32 is not None:
            return self._cos_sin_cache_fp32
        if not self._rope_initialized:
            self._init_rope()
        max_pos = self._rotary_cos_sin.shape[0]
        self._cos_sin_cache_fp32 = (
            self._rotary_cos_sin.view(max_pos, -1).to(torch.float32).contiguous()
        )
        return self._cos_sin_cache_fp32

    def _fused_rope_inplace(
        self,
        k_flat: torch.Tensor,
        positions: torch.Tensor,
        N: int,
        L: int,
    ) -> None:
        """In-place fused RoPE over [N*L, nkv*hd] K values.

        Layout of k_flat: row (i*L + l) holds layer l of position i, so
        positions must be repeat_interleaved by L to match.
        """
        positions_int32 = positions.view(-1).to(torch.int32)
        if L > 1:
            positions_int32 = positions_int32.repeat_interleave(L)

        if _flashinfer_rope is not None:
            # flashinfer requires a non-None query tensor; pass a single-head
            # scratch so the extra rotate is negligible.
            need_rows = k_flat.shape[0]
            dummy_q = self._rope_dummy_q
            if dummy_q is None or dummy_q.dtype != k_flat.dtype or dummy_q.shape[0] < need_rows:
                dummy_q = k_flat.new_empty(need_rows, self._head_dim)
                self._rope_dummy_q = dummy_q
            _flashinfer_rope(
                positions_int32,
                dummy_q[:need_rows],
                k_flat,
                self._head_dim,
                self._get_cos_sin_cache(),
                self._is_neox,
            )
            return

        # Pure-PyTorch fallback (older environments without flashinfer).
        cos, sin = self._get_rope_cos_sin(positions_int32.view(1, -1), dtype=k_flat.dtype)
        k_roped = RotaryEmbedding.apply_rotary_pos_emb(
            k_flat.view(k_flat.shape[0], -1, self._head_dim),
            cos.squeeze(0),
            sin.squeeze(0),
            unsqueeze_dim=1,
            is_neox=self._is_neox,
        )
        k_flat.copy_(k_roped.view_as(k_flat))

    def _build_fused_kv_buffers(self) -> None:
        """Stack per-layer KV projection + k_norm weights for a single fused GEMM.

        Must run after weights are loaded.
        """
        if self._fused_kv_weight is not None:
            return
        layers_attn = [layer.self_attn for layer in self.model.layers]
        attn0 = layers_attn[0]
        q_size = attn0.q_size
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        num_heads = attn0.num_heads
        num_kv_heads = attn0.num_key_value_heads
        # Head counts are read from layer 0 here and in dflash_forward; assert
        # uniformity (the target uses per-layer heads, the drafter does not).
        for a in layers_attn[1:]:
            assert (
                a.q_size == q_size
                and a.kv_size == kv_size
                and a.head_dim == head_dim
                and a.num_heads == num_heads
                and a.num_key_value_heads == num_kv_heads
            ), (
                "DFlash fused KV requires all drafter layers to share "
                "q_size / kv_size / head_dim / num_heads / num_kv_heads."
            )

        has_k_norm = [hasattr(a, "k_norm") for a in layers_attn]
        assert all(has_k_norm) or not any(has_k_norm), (
            "DFlash fused KV requires either all or no drafter layers to have k_norm."
        )

        kv_weights = [a.qkv_proj.weight[q_size : q_size + 2 * kv_size] for a in layers_attn]
        # Fold each drafter layer's input_layernorm weight into its KV projection
        # so context K/V match the query path. vLLM laguna_dflash applies
        # layer.input_layernorm to context states before KV; RMSNorm gives
        # (x_hat * w) @ Wkv.T == x_hat @ (Wkv * w).T, and the shared 1/rms(x) is
        # applied to projected_hidden in precompute_context_kv.
        dlayers = self.model.layers
        if self._context_input_layernorm and all(hasattr(dl, "input_layernorm") for dl in dlayers):
            eps_set = {
                getattr(
                    dl.input_layernorm,
                    "variance_epsilon",
                    getattr(self.config, "rms_norm_eps", 1e-6),
                )
                for dl in dlayers
            }
            assert len(eps_set) == 1, (
                "DFlash fused context input_layernorm needs all drafter layers "
                f"to share variance_epsilon; got {sorted(eps_set)}"
            )
            self._input_ln_eps = eps_set.pop()
            folded = []
            for w, dl in zip(kv_weights, dlayers):
                scale = dl.input_layernorm.weight.data
                if getattr(dl.input_layernorm, "use_gemma", False):
                    scale = scale + 1
                folded.append(w * scale[None, :].to(w.dtype))
            kv_weights = folded
        else:
            self._input_ln_eps = None
        fused_kv_weight = torch.cat(kv_weights, dim=0).contiguous()
        if attn0.qkv_proj.bias is not None:
            kv_biases = [a.qkv_proj.bias[q_size : q_size + 2 * kv_size] for a in layers_attn]
            self._fused_kv_bias = torch.cat(kv_biases, dim=0).contiguous()
        else:
            self._fused_kv_bias = None

        if all(has_k_norm):
            k_norm0 = layers_attn[0].k_norm
            eps = k_norm0.variance_epsilon
            eps_set = {a.k_norm.variance_epsilon for a in layers_attn}
            assert len(eps_set) == 1, (
                f"DFlash fused k_norm requires all drafter layers to share "
                f"variance_epsilon; got {sorted(eps_set)}."
            )
            self._k_norm_stacked = torch.stack([a.k_norm.weight.data for a in layers_attn])
            self._k_norm_eps = eps
        else:
            self._k_norm_stacked = None
            self._k_norm_eps = None
        self._num_attn_layers = len(layers_attn)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._num_kv_heads = num_kv_heads
        self._fused_kv_weight = fused_kv_weight

        # fused_qk_norm_rope derives YaRN / partial-rotary frequencies on
        # the fly, which can disagree with precompute_context_kv's cached
        # cos/sin. Only enable it when the drafter uses plain RoPE.
        self._has_qk_norm = all(has_k_norm) and all(hasattr(a, "q_norm") for a in layers_attn)
        rope_params = getattr(getattr(attn0, "pos_embd_params", None), "rope", None)
        scale_type = getattr(rope_params, "scale_type", None)
        partial_rotary_factor = getattr(
            getattr(attn0, "pretrained_config", None), "partial_rotary_factor", 1.0
        )
        self._use_fused_qk_norm_rope = (
            self._has_qk_norm
            and hasattr(attn0, "apply_qk_norm_rope")
            and rope_params is not None
            and scale_type in (None, RotaryScalingType.none)
            and partial_rotary_factor == 1.0
        )

        logger.debug(
            f"DFlash: fused KV weights built for {self._num_attn_layers} layers "
            f"(fused_kv_weight shape={tuple(self._fused_kv_weight.shape)})"
        )

    def _get_rope_cos_sin(self, positions, dtype=None):
        """Get cos/sin for given positions, suitable for apply_rotary_pos_emb.

        Args:
            positions: [B, seq_len]
            dtype: target dtype for cos/sin (default: keep original)
        Returns:
            rope_cos: [B, seq, rot_dim//2] (broadcastable with unsqueeze_dim=1)
            rope_sin: [B, seq, rot_dim//2]
        """
        if not self._rope_initialized:
            self._init_rope()

        # rotary_cos_sin: [max_pos, 2, rot_dim//2]
        rope_cache = self._rotary_cos_sin[positions]  # [B, seq, 2, rot_dim//2]
        rope_cos = rope_cache[..., 0, :]  # [B, seq, rot_dim//2]
        rope_sin = rope_cache[..., 1, :]
        if dtype is not None:
            rope_cos = rope_cos.to(dtype)
            rope_sin = rope_sin.to(dtype)
        return rope_cos, rope_sin

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

        # A DFlash 2 checkpoint's top-level is_causal wins over the layer-type
        # default (which would read an all-sliding drafter as causal). Only
        # consulted for DFlash 2: is_causal is common enough elsewhere that
        # honoring it everywhere could silently retarget an older drafter.
        explicit_causal = getattr(self.config, "is_causal", None) if self._is_dflash2 else None
        if explicit_causal is not None:
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

    def _prepare_dflash_trtllm_gen_buffers(
        self,
        dtype: torch.dtype,
        device: torch.device,
        max_batch_size: int,
        block_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        trtllm_gen_ops = self._dflash_trtllm_gen_ops
        workspace_bytes = trtllm_gen_ops.get_workspace_size(
            dtype=dtype,
            num_tokens=max_batch_size * block_size,
            num_gen_tokens=max_batch_size * block_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            max_num_requests=max_batch_size,
            rotary_embedding_dim=0,
            fp8_context_fmha=False,
        )
        device = torch.device(device)
        is_capturing = torch.cuda.is_current_stream_capturing()
        if self._dflash_trtllm_gen_device != device:
            if is_capturing:
                raise RuntimeError(
                    "DFlash TRTLLM-Gen buffers must be prepared on the current "
                    "device before CUDA graph capture."
                )
            self._dflash_trtllm_gen_device = device
            self._dflash_trtllm_gen_sm_count = torch.cuda.get_device_properties(
                device
            ).multi_processor_count

        workspace = self._dflash_trtllm_gen_workspace
        workspace_needs_allocation = (
            workspace is None
            or workspace.device != device
            or workspace.numel() * workspace.element_size() < workspace_bytes
        )
        if workspace_needs_allocation:
            if is_capturing:
                raise RuntimeError(
                    "The DFlash TRTLLM-Gen workspace must be allocated at the "
                    "required size before CUDA graph capture."
                )
            self._dflash_trtllm_gen_workspace = torch.empty(
                workspace_bytes, dtype=torch.uint8, device=device
            )

        sm_count = self._dflash_trtllm_gen_sm_count
        counter_bytes = trtllm_gen_ops.get_multi_ctas_kv_counter_size(
            num_heads, max_batch_size, sm_count
        )
        counters = self._dflash_trtllm_gen_counters
        counters_need_allocation = (
            counters is None
            or counters.device != device
            or counters.numel() * counters.element_size() < counter_bytes
        )
        if counters_need_allocation:
            if is_capturing:
                raise RuntimeError(
                    "The DFlash TRTLLM-Gen counter buffer must be allocated at "
                    "the required size before CUDA graph capture."
                )
            self._dflash_trtllm_gen_counters = torch.zeros(
                counter_bytes, dtype=torch.uint8, device=device
            )

        append_batch_indices = self._dflash_batch_indices
        block_offsets = self._dflash_block_offsets
        static_indices_need_allocation = (
            append_batch_indices is None
            or block_offsets is None
            or append_batch_indices.device != device
            or block_offsets.device != device
            or append_batch_indices.size(0) < max_batch_size
            or append_batch_indices.size(1) != block_size
            or block_offsets.numel() != block_size
        )
        if static_indices_need_allocation:
            if is_capturing:
                raise RuntimeError(
                    "DFlash TRTLLM-Gen index buffers must be allocated at the "
                    "required size before CUDA graph capture."
                )
            self._dflash_batch_indices = (
                torch.arange(max_batch_size, dtype=torch.int32, device=device)
                .view(-1, 1)
                .expand(-1, block_size)
                .contiguous()
            )
            self._dflash_block_offsets = torch.arange(block_size, dtype=torch.int32, device=device)

    def dflash_forward(
        self,
        noise_embedding: torch.Tensor,
        query_positions: torch.Tensor,
        num_ctx_per_req: torch.Tensor,
        ctx_k_cache: torch.Tensor,
        ctx_v_cache: torch.Tensor,
        ctx_cache_batch_idx: torch.Tensor,
        ctx_kv_cache: Optional[torch.Tensor] = None,
        ctx_page_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DFlash draft forward with cross-attention over a pooled K/V buffer.

        All shapes are fixed so the forward is CUDA-graph compatible.

        Args:
            noise_embedding: [B, block_size, hidden_size]
            query_positions: [B, block_size]
            num_ctx_per_req: [B] — per-batch context length in the pool
            ctx_k_cache: [pool_batch, L, max_ctx+block_size, nkv, hd]
            ctx_v_cache: [pool_batch, L, max_ctx+block_size, nkv, hd]
            ctx_cache_batch_idx: [B] — slot index into the pool per batch entry
        Returns:
            [B * block_size, hidden_size]
        """
        if self.dflash_attention_backend == "TRTLLM":
            if ctx_kv_cache is None or ctx_page_table is None:
                raise RuntimeError(
                    "DFlash TRTLLM-Gen requires a paged context cache and page table."
                )
            trtllm_gen_ops = self._dflash_trtllm_gen_ops
        elif self.dflash_attention_backend == "VANILLA":
            flash_attention = self._dflash_flash_attention
        else:
            raise ValueError(
                "DFlash attention backend must be VANILLA or TRTLLM, got "
                f"{self.dflash_attention_backend!r}."
            )

        if self._fused_kv_weight is None:
            self._build_fused_kv_buffers()

        layer0 = self.model.layers[0]
        attn0 = layer0.self_attn
        q_size = attn0.q_size
        kv_size = attn0.kv_size
        head_dim = attn0.head_dim
        # Uniformity across layers is asserted in _build_fused_kv_buffers (above).
        num_heads_per_rank = attn0.num_heads
        num_kv_heads_per_rank = attn0.num_key_value_heads
        gqa_group_size = num_heads_per_rank // num_kv_heads_per_rank

        has_qk_norm = self._has_qk_norm
        is_bf16 = noise_embedding.dtype == torch.bfloat16
        use_fused_qk_norm_rope = self._use_fused_qk_norm_rope and is_bf16
        use_fused_rope = (
            _flashinfer_rope is not None and has_qk_norm and is_bf16 and not use_fused_qk_norm_rope
        )

        B = noise_embedding.shape[0]
        block_size = noise_embedding.shape[1]

        hidden_states = noise_embedding  # [B, block_size, hidden]

        # Precompute RoPE cos/sin for the pure-PyTorch fallback path only.
        # The fused flashinfer path reads self._get_cos_sin_cache() inline.
        rope_dtype = hidden_states.dtype
        if not use_fused_rope:
            q_rope_cos, q_rope_sin = self._get_rope_cos_sin(query_positions, dtype=rope_dtype)
        _rope = RotaryEmbedding.apply_rotary_pos_emb

        # cache_seqlens (BEFORE append). flash_attn appends block_size
        # k/v at cache_seqlens[i]..+block_size for batch i.
        cache_seqlens_i32 = num_ctx_per_req[:B].to(torch.int32)
        cache_batch_idx_i32 = ctx_cache_batch_idx.to(torch.int32)

        if self.dflash_attention_backend == "TRTLLM":
            max_batch_size = ctx_page_table.size(0)
            self._prepare_dflash_trtllm_gen_buffers(
                hidden_states.dtype,
                hidden_states.device,
                max_batch_size,
                block_size,
                num_heads_per_rank,
                num_kv_heads_per_rank,
                head_dim,
            )
            block_tables = ctx_page_table.index_select(0, cache_batch_idx_i32.long())
            pages_per_slot = block_tables.size(1)
            page_size = ctx_kv_cache.size(-2)
            kv_indices = block_tables.flatten()
            kv_indptr = torch.arange(
                0,
                (B + 1) * pages_per_slot,
                pages_per_slot,
                dtype=torch.int32,
                device=hidden_states.device,
            )
            seq_lens_after = cache_seqlens_i32 + block_size
            kv_last_page_len = ((seq_lens_after - 1) % page_size) + 1
            batch_indices = self._dflash_batch_indices
            append_batch_indices = batch_indices[:B].reshape(-1)
            append_positions = (
                (cache_seqlens_i32.view(-1, 1) + self._dflash_block_offsets)
                .reshape(-1)
                .contiguous()
            )

        # Flatten query positions once for the fused QK-norm-RoPE kernel.
        query_positions_flat_i32 = query_positions.reshape(-1).to(torch.int32)

        residual = None
        has_block_conv = self.has_block_conv

        for layer_idx, layer in enumerate(self.model.layers):
            attn_mod = layer.self_attn

            # Apply input_layernorm (flatten to 2D for norm, reshape back)
            hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            if residual is None:
                residual = hidden_states.clone()
                hs_normed_flat = layer.input_layernorm(hs_flat)
            else:
                res_flat = residual.reshape(-1, residual.shape[-1])
                hs_normed_flat, res_flat = layer.input_layernorm(hs_flat, res_flat)
                residual = res_flat.reshape(B, block_size, -1)

            # DFlash 2 wraps each sublayer in a block-local convolution; both
            # sides share one projection of the input, hence the coefficients
            # handed forward to finish().
            attn_conv_coefficients = None
            if has_block_conv:
                hs_normed_flat, attn_conv_coefficients = self.attention_convs[layer_idx].prepare(
                    hs_normed_flat, block_size
                )

            # QKV projection on normed query tokens (2D)
            qkv_query = attn_mod.qkv_proj(hs_normed_flat)  # [B*blk, qkv_size]

            if use_fused_qk_norm_rope:
                # One kernel does q_norm + k_norm + RoPE in-place on qkv.
                # Only safe when the drafter's rope params don't use YaRN /
                # long-rope / partial-rotary — otherwise fall back to the
                # shared-cache path below.
                attn_mod.apply_qk_norm_rope(qkv_query, query_positions_flat_i32)
                q_all_2d = qkv_query[:, :q_size]
                k_noise_2d = qkv_query[:, q_size : q_size + kv_size]
                v_noise_2d = qkv_query[:, q_size + kv_size :]
                Q_bshd = q_all_2d.reshape(B, block_size, num_heads_per_rank, head_dim)
                k_noise_bshd = k_noise_2d.reshape(B, block_size, num_kv_heads_per_rank, head_dim)
                v_noise_bshd = v_noise_2d.reshape(B, block_size, num_kv_heads_per_rank, head_dim)
            elif use_fused_rope:
                # Per-head RMSNorm on q/k (returns new contiguous tensors),
                # then flashinfer in-place RoPE sharing the same cos/sin cache
                # as precompute_context_kv.
                q = attn_mod.q_norm(qkv_query[:, :q_size].reshape(-1, head_dim)).view(-1, q_size)
                k = attn_mod.k_norm(
                    qkv_query[:, q_size : q_size + kv_size].reshape(-1, head_dim)
                ).view(-1, kv_size)
                _flashinfer_rope(
                    query_positions_flat_i32,
                    q,
                    k,
                    head_dim,
                    self._get_cos_sin_cache(),
                    self._is_neox,
                )
                Q_bshd = q.view(B, block_size, num_heads_per_rank, head_dim)
                k_noise_bshd = k.view(B, block_size, num_kv_heads_per_rank, head_dim)
                v_noise_bshd = qkv_query[:, q_size + kv_size :].reshape(
                    B, block_size, num_kv_heads_per_rank, head_dim
                )
            else:
                qkv_query_3d = qkv_query.reshape(B, block_size, -1)
                q_all = qkv_query_3d[..., :q_size]
                k_noise_all = qkv_query_3d[..., q_size : q_size + kv_size]
                v_noise_all = qkv_query_3d[..., q_size + kv_size :]
                if has_qk_norm:
                    q_for_rope = attn_mod.q_norm(q_all.reshape(-1, head_dim)).reshape(
                        B, block_size, q_size
                    )
                    k_noise_for_rope = attn_mod.k_norm(k_noise_all.reshape(-1, head_dim)).reshape(
                        B, block_size, kv_size
                    )
                else:
                    q_for_rope = q_all
                    k_noise_for_rope = k_noise_all
                Q = _rope(
                    q_for_rope.reshape(B, block_size, num_heads_per_rank, head_dim).transpose(1, 2),
                    q_rope_cos,
                    q_rope_sin,
                    unsqueeze_dim=1,
                    is_neox=self._is_neox,
                )
                k_noise_rope = _rope(
                    k_noise_for_rope.reshape(
                        B, block_size, num_kv_heads_per_rank, head_dim
                    ).transpose(1, 2),
                    q_rope_cos,
                    q_rope_sin,
                    unsqueeze_dim=1,
                    is_neox=self._is_neox,
                )
                Q_bshd = Q.transpose(1, 2)
                k_noise_bshd = k_noise_rope.transpose(1, 2)
                v_noise_bshd = v_noise_all.reshape(B, block_size, num_kv_heads_per_rank, head_dim)

            # Per-layer view into the pooled ctx cache.
            causal, window_size = self._get_attention_mask_args(layer_idx)
            swa_window = (
                self._layer_windows[layer_idx] if layer_idx < len(self._layer_windows) else (-1, -1)
            )
            if swa_window != (-1, -1):
                window_size = swa_window
            if self.dflash_attention_backend == "TRTLLM":
                layer_cache = ctx_kv_cache[layer_idx]
                trtllm_gen_ops.append_paged_kv_cache(
                    append_key=k_noise_bshd.reshape(
                        -1, num_kv_heads_per_rank, head_dim
                    ).contiguous(),
                    append_value=v_noise_bshd.reshape(
                        -1, num_kv_heads_per_rank, head_dim
                    ).contiguous(),
                    batch_indices=append_batch_indices,
                    positions=append_positions,
                    paged_kv_cache=layer_cache,
                    kv_indices=kv_indices,
                    kv_indptr=kv_indptr,
                    kv_last_page_len=kv_last_page_len,
                    kv_layout="HND",
                )
                out = torch.empty_like(Q_bshd)
                q_flat = Q_bshd.reshape(-1, num_heads_per_rank, head_dim)
                out_flat = out.reshape(-1, num_heads_per_rank, head_dim)
                window_left = window_size[0]
                if causal:
                    trtllm_gen_ops.batch_decode_with_kv_cache(
                        query=q_flat,
                        kv_cache=(layer_cache[:, 0], layer_cache[:, 1]),
                        workspace_buffer=self._dflash_trtllm_gen_workspace,
                        block_tables=block_tables,
                        seq_lens=seq_lens_after,
                        max_seq_len=pages_per_slot * page_size,
                        bmm1_scale=head_dim**-0.5,
                        bmm2_scale=1.0,
                        window_left=window_left,
                        out=out_flat,
                        sinks=None,
                        enable_pdl=False,
                        kv_layout="HND",
                        backend="trtllm-gen",
                        q_len_per_req=block_size,
                        max_q_len=None,
                        cum_seq_lens_q=None,
                        kv_cache_sf=None,
                        uses_shared_paged_kv_idx=True,
                        bmm1_scale_log2=None,
                        multi_ctas_kv_counter_buffer=self._dflash_trtllm_gen_counters,
                    )
                else:
                    cum_seq_lens_q = torch.arange(
                        0,
                        (B + 1) * block_size,
                        block_size,
                        dtype=torch.int32,
                        device=hidden_states.device,
                    )
                    cum_seq_lens_kv = torch.cat(
                        (
                            torch.zeros(1, dtype=torch.int32, device=hidden_states.device),
                            seq_lens_after.cumsum(0, dtype=torch.int32),
                        )
                    )
                    trtllm_gen_ops.batch_context_with_kv_cache(
                        query=q_flat,
                        kv_cache=(layer_cache[:, 0], layer_cache[:, 1]),
                        workspace_buffer=self._dflash_trtllm_gen_workspace,
                        block_tables=block_tables,
                        seq_lens=seq_lens_after,
                        max_q_len=block_size,
                        max_kv_len=pages_per_slot * page_size,
                        bmm1_scale=head_dim**-0.5,
                        bmm2_scale=1.0,
                        batch_size=B,
                        cum_seq_lens_q=cum_seq_lens_q,
                        cum_seq_lens_kv=cum_seq_lens_kv,
                        window_left=window_left,
                        out=out_flat,
                        sinks=None,
                        enable_pdl=False,
                        kv_layout="HND",
                        kv_cache_sf=None,
                        uses_shared_paged_kv_idx=True,
                        causal=False,
                        multi_ctas_kv_counter_buffer=self._dflash_trtllm_gen_counters,
                    )
            else:  # VANILLA, validated before entering the layer loop.
                layer_k_cache = ctx_k_cache[:, layer_idx]
                layer_v_cache = ctx_v_cache[:, layer_idx]

                # Pack gqa_group_size query heads sharing a KV head into the
                # row dimension: [B, blk, h_q, d] -> [B, group*blk, h_kv, d].
                # Each CTA owns a whole query-head group and streams KV head's context once
                # instead of gqa_group_size CTAs each re-reading it.
                # Exact only while every row of the block attends to the same
                # key set, i.e. non-causal, unwindowed layers. Causal or
                # windowed layers mask by row, so they stay unpacked.
                pack_gqa = gqa_group_size > 1 and not causal and window_size == (-1, -1)
                if pack_gqa:
                    q_grouped = Q_bshd.reshape(
                        B, block_size, num_kv_heads_per_rank, gqa_group_size, head_dim
                    )
                    q_packed = q_grouped.permute(0, 3, 1, 2, 4)
                    q_in = q_packed.reshape(
                        B, gqa_group_size * block_size, num_kv_heads_per_rank, head_dim
                    )
                else:
                    q_in = Q_bshd
                out = flash_attention(
                    q=q_in,
                    k_cache=layer_k_cache,
                    v_cache=layer_v_cache,
                    k=k_noise_bshd,
                    v=v_noise_bshd,
                    cache_seqlens=cache_seqlens_i32,
                    cache_batch_idx=cache_batch_idx_i32,
                    causal=causal,
                    window_size=window_size,
                )
                if pack_gqa:
                    # Undo the packing: [B, group*blk, h_kv, d] -> [B, blk, h_q, d].
                    out = out.view(
                        B, gqa_group_size, block_size, num_kv_heads_per_rank, head_dim
                    ).permute(0, 2, 3, 1, 4)

            attn_output = out.reshape(B * block_size, q_size)

            # Per-drafter post-attention gate (no-op for generic DFlash; Laguna
            # applies per-head softplus g_proj gating). gate input is the
            # input_layernorm output (the attention input).
            attn_output = self._post_attention_gate(
                attn_output, hs_normed_flat, attn_mod, num_heads_per_rank, head_dim
            )

            # o_proj (flat 2D, handles all-reduce internally)
            hidden_out = attn_mod.o_proj(attn_output)
            if has_block_conv:
                hidden_out = self.attention_convs[layer_idx].finish(
                    hidden_out, attn_conv_coefficients, block_size
                )

            # Post-attention layernorm + MLP (flat 2D)
            res_flat = residual.reshape(-1, residual.shape[-1])
            hidden_out, res_flat = layer.post_attention_layernorm(hidden_out, res_flat)
            if has_block_conv:
                hidden_out, mlp_conv_coefficients = self.mlp_convs[layer_idx].prepare(
                    hidden_out, block_size
                )
            hidden_out = layer.mlp(hidden_out)
            if has_block_conv:
                hidden_out = self.mlp_convs[layer_idx].finish(
                    hidden_out, mlp_conv_coefficients, block_size
                )

            hidden_states = hidden_out.reshape(B, block_size, -1)
            residual = res_flat.reshape(B, block_size, -1)

        # Final norm
        hidden_states_out, _ = self.model.norm(
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            residual.reshape(-1, residual.shape[-1]),
        )
        return hidden_states_out

    def forward(
        self,
        attn_metadata,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        return_context_logits: bool = False,
        spec_metadata=None,
        hidden_states: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the draft model and return (hidden_states, hidden_states) for the
        speculative-decoding contract."""
        hidden_states_out = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            **kwargs,
        )

        return hidden_states_out, hidden_states_out


class DFlashLagunaForCausalLM(DFlashForCausalLM):
    """Laguna DFlash drafter.

    The generic block decode lives in DFlashForCausalLM; this subclass supplies
    the Laguna draft-layer specifics: per-head g_proj softplus gating and the
    per-aux fc_norm applied to captured target features before fc.
    """

    @staticmethod
    def _normalize_config(config: PretrainedConfig) -> None:
        """Fill TRT-LLM Laguna defaults missing from dense DFlash drafts."""
        if getattr(config, "num_experts", None) is None:
            config.num_experts = 0
        if getattr(config, "mlp_layer_types", None) is None:
            config.mlp_layer_types = ["dense"] * config.num_hidden_layers
        if getattr(config, "block_size", None) is None:
            dflash_config = getattr(config, "dflash_config", {})
            if isinstance(dflash_config, dict):
                config.block_size = dflash_config.get("block_size", None)

    def __init__(self, draft_config, *, dflash_attention_backend: str = "VANILLA"):
        """Pin the Laguna draft-layer class and enable Laguna-specific behaviors
        (context input_layernorm, causal sliding blocks); reject non-per-head
        gating."""
        # The checkpoint labels itself with the vLLM name (model_type "llama");
        # remap to the Laguna architecture so TRT-LLM builds the Laguna layers.
        draft_config.pretrained_config.architectures = ["LagunaForCausalLM"]
        self._normalize_config(draft_config.pretrained_config)
        super().__init__(
            draft_config,
            dflash_attention_backend=dflash_attention_backend,
        )
        self._context_input_layernorm = True
        self._sliding_layers_causal = True
        gating = getattr(self.config, "gating", True)
        if gating not in (True, "per-head"):
            raise NotImplementedError(
                f"Laguna DFlash drafter supports per-head gating only, got gating={gating!r}"
            )

    def load_weights(self, weights, weight_mapper=None, **kwargs):
        """Build the per-aux ``fc_norm`` from the drafter's ``aux_hidden_norms.*``
        weights, then defer the remaining weights to the base loader."""
        aux_keys = sorted(
            (k for k in weights if k.startswith("aux_hidden_norms.")),
            key=lambda k: int(k.split(".")[1]),
        )
        if not aux_keys:
            raise ValueError("Laguna DFlash checkpoint is missing aux_hidden_norms.* weights")
        weights = dict(weights)
        eps = getattr(self.config, "rms_norm_eps", 1e-6)
        norms = []
        for k in aux_keys:
            w = weights.pop(k)
            norm = nn.RMSNorm(
                w.shape[0], eps=eps, device="cuda", elementwise_affine=True, dtype=w.dtype
            )
            norm.weight.data.copy_(w)
            norms.append(norm)
        self.fc_norm = nn.ModuleList(norms)
        super().load_weights(weights, weight_mapper=weight_mapper, **kwargs)

    def project_target_hidden(self, hidden_states):
        """Project captured target features to the draft width: apply the per-aux
        ``fc_norm`` to each hidden chunk, then ``fc`` + ``hidden_norm``."""
        hidden_states = hidden_states.to(self.fc.weight.dtype)
        fc_norm = getattr(self, "fc_norm", None)
        if fc_norm is not None:
            chunks = hidden_states.chunk(len(fc_norm), dim=-1)
            hidden_states = torch.cat([norm(chunk) for norm, chunk in zip(fc_norm, chunks)], dim=-1)
        return self.hidden_norm(self.fc(hidden_states))

    def _post_attention_gate(self, attn_output, gate_input, attn_mod, num_heads, head_dim):
        """Apply Laguna's per-head softplus output gate (``g_proj``) to the
        attention output; a no-op when the layer has no ``g_proj``."""
        g_proj = getattr(attn_mod, "g_proj", None)
        if g_proj is None:
            return attn_output
        gate = F.softplus(g_proj(gate_input).float()).to(attn_output.dtype)
        return (attn_output.unflatten(-1, (num_heads, head_dim)) * gate.unsqueeze(-1)).flatten(-2)


# Published DSpark drafters spell the head switches four different ways, and a
# reader that knows only one of them degrades silently: the heads are skipped,
# their weights are dropped, and nothing raises. Resolution order matches
# ``TorchLlmArgs.validate_speculative_config`` so the model and the user-visible
# spec_config can never disagree about whether a head is on.
_DSPARK_HEAD_KEY_ALIASES = {
    # RadixArk/Kimi-K3-DSpark ships ``enable_confidence_head`` top-level.
    "use_confidence_head": ("use_confidence_head", "enable_confidence_head"),
}


def resolve_dspark_head_config(pretrained_config, key):
    """Resolve one DSpark head switch across every spelling in the wild.

    Looks in ``dspark_config``, then ``dflash_config``, then the top level as
    ``dspark_<key>`` and as ``<key>``, for each accepted alias of ``key``.
    Returns ``None`` when the drafter declares the switch nowhere.
    """
    dspark_cfg = getattr(pretrained_config, "dspark_config", None) or {}
    dflash_cfg = getattr(pretrained_config, "dflash_config", None) or {}
    for name in _DSPARK_HEAD_KEY_ALIASES.get(key, (key,)):
        for value in (
            dspark_cfg.get(name),
            dflash_cfg.get(name),
            getattr(pretrained_config, f"dspark_{name}", None),
            getattr(pretrained_config, name, None),
        ):
            if value is not None:
                return value
    return None


def declares_dspark_heads(pretrained_config) -> bool:
    """True when a drafter config asks for the DSpark head set.

    Resolved here rather than in the DSpark module: this module must not import
    the DSpark drafters, or the ``modeling_dspark -> modeling_dflash``
    inheritance edge would become a cycle.
    """
    return bool(
        str(resolve_dspark_head_config(pretrained_config, "projector_type") or "").lower()
        == "dspark"
        or resolve_dspark_head_config(pretrained_config, "shift_label")
        or resolve_dspark_head_config(pretrained_config, "use_confidence_head")
        or int(resolve_dspark_head_config(pretrained_config, "markov_rank") or 0) > 0
    )


@register_draft_model(SpeculativeDecodingMode.DFLASH)
def _build_dflash_draft(model_config, draft_config, lm_head, model):
    """Build the DFlash drafter.

    Selects the Laguna variant by detecting its architecture in the draft
    checkpoint's own config. A drafter that declares the DSpark heads is
    rejected rather than silently served without them: DFlash no longer
    implements the Markov / confidence / shift_label semantics, so building one
    here would degrade the drafter with no error and show up only as a lower
    acceptance rate.
    """
    if declares_dspark_heads(draft_config.pretrained_config):
        raise ValueError(
            "This drafter checkpoint declares the DSpark head set "
            "(dflash_config with any of projector_type='dspark', shift_label, "
            "use_confidence_head, markov_rank > 0), which decoding_type "
            "'DFlash' does not implement. Set speculative_config.decoding_type "
            "to 'DSpark' to run it."
        )
    draft_arches = getattr(draft_config.pretrained_config, "architectures", None) or []
    dflash_attention_backend = model_config.spec_config.attention_backend
    if any("Laguna" in arch for arch in draft_arches):
        return DFlashLagunaForCausalLM(
            draft_config,
            dflash_attention_backend=dflash_attention_backend,
        )
    return DFlashForCausalLM(
        draft_config,
        dflash_attention_backend=dflash_attention_backend,
    )
