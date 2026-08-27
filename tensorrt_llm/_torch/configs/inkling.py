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
"""Config classes for the Inkling multimodal checkpoint.

Transformers does not ship Inkling, so these classes reconstruct the config from
the checkpoint's nested dicts without a transformers shim. The audio, vision and
MTP sub-configs are kept verbatim as ``PretrainedConfig`` blobs, which the towers
read their geometry off directly.

Field names mirror the checkpoint ``text_config`` and the numeric defaults are the
real checkpoint values, overridable through ``from_dict``.
"""

from transformers.configuration_utils import PretrainedConfig


class InklingTextConfig(PretrainedConfig):
    """Text-tower sub-config (``InklingCausalLLM``).

    A RoPE-free hybrid-attention decoder: per-head q/k RMSNorm, learned
    relative-position bias, four short convolutions per layer, sigmoid-gated MoE
    with two shared experts, muP logit scaling, and an unpadded vocab slice.
    """

    model_type = "inkling_text"

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int = 200058,
        hidden_size: int = 6144,
        num_hidden_layers: int = 66,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        model_max_length: int = 1048576,
        logits_mup_width_multiplier: float = 24.0,
        use_embed_norm: bool = True,
        tie_word_embeddings: bool = False,
        # hybrid attention geometry
        local_layer_ids: list[int] | None = None,
        sliding_window_size: int = 512,
        swa_num_attention_heads: int = 64,
        swa_num_key_value_heads: int = 16,
        swa_head_dim: int = 128,
        # MTP (next-N draft) chain. Depths listed in ``mtp_local_layer_ids``
        # are banded; the rest are global. The SWA geometry defaults to the
        # trunk's, which is what the checkpoint uses unless it says otherwise.
        mtp_local_layer_ids: list[int] | None = None,
        mtp_local_extent: int | None = None,
        mtp_swa_num_attention_heads: int | None = None,
        mtp_swa_num_key_value_heads: int | None = None,
        mtp_swa_head_dim: int | None = None,
        # relative-bias / log-scaling
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int = 128000,
        log_scaling_alpha: float = 0.1,
        # short conv
        use_sconv: bool = True,
        sconv_kernel_size: int = 4,
        # dense MLP / MoE
        dense_mlp_idx: int = 2,
        intermediate_size: int = 3072,
        dense_intermediate_size: int = 24576,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 6,
        n_shared_experts: int = 2,
        shared_expert_sink: bool = True,
        route_scale: float = 8.0,
        use_gate_bias: bool = True,
        gate_activation: str = "sigmoid",
        norm_after_topk: bool = True,
        use_global_scale: bool = True,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.model_max_length = model_max_length
        # `max_position_embeddings` is read by several TRT-LLM code paths
        # (Attention, RopeParams probing); Inkling has no RoPE but keep the 1M
        # context window available so nothing clamps sequence length.
        self.max_position_embeddings = kwargs.get("max_position_embeddings", model_max_length)
        self.logits_mup_width_multiplier = logits_mup_width_multiplier
        self.use_embed_norm = use_embed_norm

        self.local_layer_ids = list(local_layer_ids) if local_layer_ids else []
        self.sliding_window_size = sliding_window_size
        self.swa_num_attention_heads = swa_num_attention_heads
        self.mtp_local_layer_ids = list(mtp_local_layer_ids or [])
        self.mtp_local_extent = mtp_local_extent
        self.mtp_swa_num_attention_heads = (
            mtp_swa_num_attention_heads
            if mtp_swa_num_attention_heads is not None
            else swa_num_attention_heads
        )
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.mtp_swa_num_key_value_heads = (
            mtp_swa_num_key_value_heads
            if mtp_swa_num_key_value_heads is not None
            else swa_num_key_value_heads
        )
        self.mtp_swa_head_dim = mtp_swa_head_dim if mtp_swa_head_dim is not None else swa_head_dim

        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha

        self.use_sconv = use_sconv
        self.sconv_kernel_size = sconv_kernel_size

        self.dense_mlp_idx = dense_mlp_idx
        self.intermediate_size = intermediate_size
        # A config.json may spell this out; PretrainedConfig.__init__ has already
        # stored it from **kwargs, so read it back before defaulting.
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", intermediate_size)
        self.dense_intermediate_size = dense_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.shared_expert_sink = shared_expert_sink
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.gate_activation = gate_activation
        self.norm_after_topk = norm_after_topk
        self.use_global_scale = use_global_scale
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout

    # ---- per-layer classification helpers (single source of truth) ----
    @property
    def _local_ids(self) -> set:
        return set(self.local_layer_ids)

    def is_dense_layer(self, layer_idx: int) -> bool:
        """Dense MLP layers are the ones with index < ``dense_mlp_idx``."""
        return layer_idx < self.dense_mlp_idx

    def is_local_layer(self, layer_idx: int) -> bool:
        """Local (sliding-window) layers are listed in ``local_layer_ids``."""
        return layer_idx in self._local_ids

    def layer_num_kv_heads(self, layer_idx: int) -> int:
        return (
            self.swa_num_key_value_heads
            if self.is_local_layer(layer_idx)
            else self.num_key_value_heads
        )

    def layer_num_heads(self, layer_idx: int) -> int:
        return (
            self.swa_num_attention_heads
            if self.is_local_layer(layer_idx)
            else self.num_attention_heads
        )

    def layer_head_dim(self, layer_idx: int) -> int:
        return self.swa_head_dim if self.is_local_layer(layer_idx) else self.head_dim

    def layer_window(self, layer_idx: int) -> int | None:
        """Sliding-window size for local layers; ``None`` for global layers."""
        return self.sliding_window_size if self.is_local_layer(layer_idx) else None

    # ------------------------------------------------------------------ MTP --
    # The draft chain's depths are NOT trunk layers: a banded depth runs at the
    # HEAD's window, because the checkpoint's rel_logits_proj for that depth was
    # trained at that window. Reusing the trunk geometry would silently apply
    # the wrong relative-position extent.
    def is_mtp_local_depth(self, depth: int) -> bool:
        """Whether MTP depth ``depth`` is a banded (sliding-window) block."""
        return depth in set(self.mtp_local_layer_ids or ())

    def mtp_depth_window(self, depth: int) -> int | None:
        """Sliding-window extent for an MTP depth; ``None`` if global."""
        if not self.is_mtp_local_depth(depth):
            return None
        return (
            self.mtp_local_extent if self.mtp_local_extent is not None else self.sliding_window_size
        )

    def mtp_depth_num_heads(self, depth: int) -> int:
        return (
            self.mtp_swa_num_attention_heads
            if self.is_mtp_local_depth(depth)
            else self.num_attention_heads
        )

    def mtp_depth_num_kv_heads(self, depth: int) -> int:
        return (
            self.mtp_swa_num_key_value_heads
            if self.is_mtp_local_depth(depth)
            else self.num_key_value_heads
        )

    def mtp_depth_head_dim(self, depth: int) -> int:
        return self.mtp_swa_head_dim if self.is_mtp_local_depth(depth) else self.head_dim

    def mtp_num_kv_heads_per_layer(self, num_depths: int) -> list[int]:
        """Per-depth KV-head counts for the DRAFT chain's own KV cache.

        The draft chain gets a separate cache manager, sized from the chain's
        geometry -- not a slice of the trunk's. Slicing would be wrong twice
        over: the list would be the trunk's length (42 or 66) where the manager
        expects one entry per built depth, which is the assertion this exists to
        satisfy; and the trunk's banded layers are not the chain's, so on the
        full checkpoint (banded 16 KV heads, global 8) depths 1 and 3 would be
        allocated 16 heads' worth of pages for an 8-head layer.
        """
        return [self.mtp_depth_num_kv_heads(d) for d in range(num_depths)]

    def mtp_block_config(self, depth: int, layer_idx: int | None = None) -> "InklingTextConfig":
        """A derived config that makes ``InklingDecoderLayer`` build an MTP block.

        The draft block is structurally a DENSE trunk layer whose attention
        geometry comes from the MTP chain, not the trunk. Rather than teach the
        decoder layer about MTP -- which would put a second notion of "which
        layer am I" inside it -- hand it a config where the ordinary questions
        it already asks give the draft answers:

        * ``is_dense_layer(depth)`` is forced True, because every MTP block uses
          the dense MLP (SGLang forces this, and both checkpoints agree: one
          global_scale per depth, no expert tensors);
        * ``is_local_layer(depth)`` follows the CHAIN's banded depths;
        * the window and SWA head geometry come from the chain's overrides.
        """
        import copy

        cfg = copy.copy(self)
        # The decoder layer is built with, and queries this config by, its own
        # layer index. That is the CHAIN depth for geometry but the GLOBAL index
        # (trunk layers + depth) for KV-cache addressing, because the draft
        # manager keys its layer offsets globally. So the config is written to
        # answer for whichever index the layer will actually use.
        idx = depth if layer_idx is None else layer_idx
        # Everything below dense_mlp_idx is dense, so idx+1 makes this layer
        # dense whatever its index.
        cfg.dense_mlp_idx = idx + 1
        # Only this one index is ever queried on this config, so the chain's
        # banded list collapses to "is this layer banded".
        cfg.local_layer_ids = [idx] if self.is_mtp_local_depth(depth) else []
        if self.is_mtp_local_depth(depth):
            cfg.sliding_window_size = self.mtp_depth_window(depth)
            cfg.swa_num_attention_heads = self.mtp_depth_num_heads(depth)
            cfg.swa_num_key_value_heads = self.mtp_depth_num_kv_heads(depth)
            cfg.swa_head_dim = self.mtp_depth_head_dim(depth)
        return cfg

    def num_kv_heads_per_layer(self) -> list[int]:
        """Per-layer KV-head counts for the hybrid attention geometry.

        ``KVCacheManagerV2`` takes this list as ``num_kv_heads`` so the paged pool
        is sized per layer; ``head_dim`` is uniform, so only the count varies.
        """
        return [self.layer_num_kv_heads(i) for i in range(self.num_hidden_layers)]


class InklingConfig(PretrainedConfig):
    """Top-level Inkling multimodal config (``inkling_mm_model``).

    Reconstructs ``text_config`` with :class:`InklingTextConfig`;
    ``audio_config``, ``vision_config`` and ``mtp_config`` are retained as plain
    ``PretrainedConfig`` blobs so the multimodal checkpoint round-trips.
    """

    model_type = "inkling_mm_model"
    sub_configs = {"text_config": InklingTextConfig}

    def __init__(
        self,
        text_config=None,
        audio_config=None,
        vision_config=None,
        mtp_config=None,
        eos_token_id: int = 200006,
        # In-vocab chat-template placeholder ids, omitted from the checkpoint's
        # config.json. They must be in-vocab: the executor rejects out-of-range
        # request token ids.
        image_token_id: int = 200054,
        audio_token_id: int = 200053,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id

        if text_config is None:
            self.text_config = InklingTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = InklingTextConfig(**text_config)
        else:
            self.text_config = text_config

        # Retained verbatim; interpreted only in the Phase-3 multimodal stage.
        self.audio_config = self._as_config(audio_config)
        self.vision_config = self._as_config(vision_config)
        # Retained verbatim so the checkpoint round-trips, and canonicalized
        # onto text_config: the draft-chain geometry is read by the MTP blocks,
        # the KV-pool routing and the weight mapper, and three readers of two
        # sources is how they drift.
        self.mtp_config = self._as_config(mtp_config)
        if self.mtp_config is not None:
            local_ids = getattr(self.mtp_config, "local_layer_ids", None)
            if local_ids:
                self.text_config.mtp_local_layer_ids = list(local_ids)
            extent = getattr(self.mtp_config, "local_extent", None)
            if extent is not None:
                self.text_config.mtp_local_extent = extent
            # The framework's MTPForCausalLM reads the chain depth as
            # ``pretrained_config.num_nextn_predict_layers``; Inkling declares it
            # on mtp_config, so mirror it under the name the framework looks for
            # rather than special-casing Inkling inside the framework.
            #
            # Falls back to the listed depths because that read is a BARE
            # attribute access -- ``checkpoint_mtp_num_layers =
            # model_config.pretrained_config.num_nextn_predict_layers``, no
            # getattr and no default. A checkpoint that describes its chain only
            # by naming the banded depths would otherwise reach it with the
            # attribute absent and die on a bare AttributeError from inside
            # framework code, naming neither Inkling nor the field.
            depths = getattr(self.mtp_config, "num_nextn_predict_layers", None)
            if depths is None:
                # ``local_layer_ids`` names WHICH depths are banded, not how
                # many there are -- ``is_mtp_local_depth`` uses it as a
                # membership set. The shipped small checkpoint declares 8 depths
                # with ids [0, 2, 4, 5, 6, 7]: the length is 6, the last id is
                # 7, and only ``max + 1`` recovers the 8. So the count is
                # derived from the largest index, not from how many are listed.
                ids = getattr(self.mtp_config, "local_layer_ids", None) or ()
                depths = (max(ids) + 1) if ids else None
            if depths is not None:
                self.text_config.num_nextn_predict_layers = int(depths)
                # And on THIS config as well, because two different framework
                # readers look in two different places. MTPForCausalLM gets the
                # TEXT sub-config, but update_spec_config_from_model_config is
                # handed config.pretrained_config -- the top-level object for a
                # multimodal checkpoint -- and reads num_nextn_predict_layers off
                # it directly. Not finding it, it falls back to 1 and resolves to
                # MTP_EAGLE_ONE_MODEL rather than vanilla MTP: one draft block
                # replayed, not Inkling's per-depth chain.
                self.num_nextn_predict_layers = int(depths)

    @staticmethod
    def _as_config(value):
        if value is None or isinstance(value, PretrainedConfig):
            return value
        if isinstance(value, dict):
            cfg = PretrainedConfig()
            for k, v in value.items():
                setattr(cfg, k, v)
            return cfg
        return value
