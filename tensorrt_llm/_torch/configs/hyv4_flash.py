# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HY4-Flash configuration compatibility for Transformers versions before native support.

The checkpoint declares a text decoder and a vision tower in one composite config
(``model_type`` ``hy_v4_flash``); the decoder is gated MLA with two attention kinds
(``deepseek_sparse_attention`` on the indexed layers, ``sliding_attention`` on the
windowed ones), a latent MoE, the iHC residual streams and Engram hash tables, and
the composite also carries the checkpoint-native DFly2 drafting geometry.

Every value here is the checkpoint's own; the defaults only make a config built
from a partial dict readable, and the checkpoint's ``config.json`` always wins.
"""

from transformers import PretrainedConfig


class HYV4FlashTextConfig(PretrainedConfig):
    """The HY4-Flash text decoder (``text_config``)."""

    model_type = "hy_v4_flash_text"
    base_config_key = "text_config"
    attribute_map = {"num_local_experts": "n_routed_experts"}

    #: The checkpoint's name for a sparse-MLA layer that runs the DSA indexer.
    #: Transformers' generic layer-type validator only knows ``full_attention``
    #: (before it learns the native hy_v4_flash names), so construction maps it
    #: to that and restores the checkpoint's name before returning.
    SPARSE_LAYER_TYPE = "deepseek_sparse_attention"
    _GENERIC_LAYER_TYPE = {
        "deepseek_sparse_attention": "full_attention",
        "indexed_attention": "full_attention",
    }

    #: The M-RoPE split and its interleaving, published per layer type inside
    #: ``rope_parameters``; transformers' RoPE validator does not know them.
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    #: Base wavelength per layer type, keyed by the generic names construction
    #: uses: the indexed layers rotate at 1e7 and the windowed ones at 1e4.
    default_theta = {"full_attention": 10_000_000.0, "sliding_attention": 10_000.0}

    def __init__(self, **kwargs):
        fields = {
            "vocab_size": 120832,
            "hidden_size": 3072,
            "intermediate_size": 13824,
            "moe_intermediate_size": 1536,
            "num_hidden_layers": 42,
            "num_attention_heads": 32,
            "q_lora_rank": 1024,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "sliding_window": 512,
            "swa_num_attention_heads": 64,
            "swa_qk_nope_head_dim": 192,
            "swa_qk_rope_head_dim": 64,
            "swa_v_head_dim": 256,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_n_heads": 16,
            "index_kpool": 4,
            "n_routed_experts": 512,
            "n_shared_experts": 1,
            "num_experts_per_tok": 16,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "routed_scaling_factor": 3.998,
            "moe_router_enable_expert_bias": True,
            "moe_router_use_sigmoid": True,
            "routed_expert_hidden_size": 1536,
            "first_k_dense_replace": 1,
            "swiglu_limit": 10.0,
            "hc_mult": 4,
            "hc_magnitude": 2.0,
            "hc_eps": 1e-06,
            "enable_ihc": True,
            "ihc_kernel_backend": "naive",
            "engram_max_ngram_size": 3,
            "engram_n_head_per_ngram": 8,
            "engram_n_embed_per_ngram": 768,
            "engram_pad_id": 120002,
            "engram_fp8_embedding": True,
            "engram_lookup_table_size": 120818,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-05,
            "initializer_range": 0.006,
            "bos_token_id": 120000,
            "eos_token_id": 120025,
            "pad_token_id": 120002,
            "tie_word_embeddings": False,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "qk_norm": True,
            "sandwich_norm": True,
            "num_nextn_predict_layers": 0,
            "hidden_act": "silu",
            "use_cache": True,
            "use_grouped_mm": False,
            "use_dsa": True,
            "use_compressed_indexer": True,
            "use_mla": True,
            "gated_mla": True,
            "gating_type": "elementwise",
            "learnable_sink": True,
            "learnable_sink_init": 0.0,
            "mrope_section": [12, 10, 10],
            "mrope_interleaved": True,
        }
        fields.update(kwargs)
        # The published checkpoint labels the nested text config with the
        # composite's own `model_type`; keep the class's own name instead.
        fields.pop("model_type", None)

        # The checkpoint publishes one compressed-indexer block under its own
        # ``compressed_indexer_*`` names; ``index_kpool`` is the canonical name
        # for the group width it pools each head's keys to.
        compress_ratio = fields.pop("compressed_indexer_compress_ratio", None)
        if compress_ratio is not None:
            fields["index_kpool"] = compress_ratio
        fields.pop("compressed_indexer_head_dim", None)
        fields.pop("compressed_indexer_num_attention_heads", None)
        fields.pop("compressed_indexer_group_topk", None)
        fields.pop("compressed_indexer_candidate_topk", None)
        fields.pop("swa_qk_head_dim", None)

        num_layers = fields["num_hidden_layers"]
        layer_types = fields.pop("layer_types", None)
        if layer_types is None:
            # Two indexed layers first, then one indexed layer per three windowed
            # ones: layers 0, 1, 5, 9, ... of the checkpoint's own pattern.
            layer_types = [
                self.SPARSE_LAYER_TYPE
                if layer_idx < 2 or (layer_idx - 1) % 4 == 0
                else "sliding_attention"
                for layer_idx in range(num_layers)
            ]
        else:
            layer_types = list(layer_types)
            if len(layer_types) != num_layers:
                raise ValueError(
                    f"`layer_types` has {len(layer_types)} entries for `num_hidden_layers` "
                    f"{num_layers}"
                )

        # Transformers' layer-type validator rejects the checkpoint's sparse-layer
        # name, and it only standardizes RoPE per layer type when the rope keys are
        # in `layer_types`. Construct under the generic names, then publish the
        # checkpoint's own names once construction (and its validation) is done.
        generic_of = {name: self._GENERIC_LAYER_TYPE.get(name, name) for name in set(layer_types)}
        self.layer_types = [generic_of[name] for name in layer_types]
        published_of = {generic_of[name]: name for name in layer_types}

        rope_parameters = fields.pop("rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = fields.pop("rope_scaling", None)
        self.rope_parameters = (
            {
                generic_of.get(name, name): dict(params)
                for name, params in rope_parameters.items()
                if isinstance(params, dict)
            }
            if isinstance(rope_parameters, dict)
            else {}
        )

        super().__init__(**fields)

        self.layer_types = layer_types
        self.rope_parameters = {
            published_of.get(name, name): params for name, params in self.rope_parameters.items()
        }

        # MLA expands the latent to one key/value per query head, so keys are never
        # grouped, and RoPE applies to the rope slice only.
        self.num_key_value_heads = self.num_attention_heads
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim

        # The windowed layers run their own head geometry, published as `swa_*`.
        sliding = {
            "num_attention_heads": self.swa_num_attention_heads,
            "num_key_value_heads": self.swa_num_attention_heads,
            "qk_nope_head_dim": self.swa_qk_nope_head_dim,
            "qk_rope_head_dim": self.swa_qk_rope_head_dim,
            "qk_head_dim": self.swa_qk_nope_head_dim + self.swa_qk_rope_head_dim,
            "head_dim": self.swa_qk_rope_head_dim,
            "v_head_dim": self.swa_v_head_dim,
        }
        self.per_layer_config = {
            layer_idx: dict(sliding)
            for layer_idx, layer_type in enumerate(layer_types)
            if layer_type == "sliding_attention"
        }

        # Engram's own ids are 1-based; the decoder's layers are 0-based.
        engram_layer_ids = getattr(self, "engram_layer_ids", None) or []
        self.engram_layer_ids = list(engram_layer_ids)
        self.engram_layer_indices = [layer_id - 1 for layer_id in self.engram_layer_ids]

        # Which layers own a DSA indexer pool: every indexed layer runs one, and a
        # windowed layer runs none. The engine reads this as `index_topk_pattern`
        # ('F' a full per-layer indexer, 'S' no pool of its own).
        self.index_topk_pattern = [
            "F" if layer_type == self.SPARSE_LAYER_TYPE else "S" for layer_type in layer_types
        ]

        self._validate()

    def _validate(self) -> None:
        """Check the geometry whose mismatch would only surface as wrong math later."""
        for layer_type in set(self.layer_types):
            rope_head_dim = (
                self.swa_qk_rope_head_dim
                if layer_type == "sliding_attention"
                else self.qk_rope_head_dim
            )
            mrope_section = self.rope_parameters[layer_type]["mrope_section"]
            if len(mrope_section) != 3 or sum(mrope_section) != rope_head_dim // 2:
                raise ValueError(
                    f"`mrope_section` must hold three entries (time, height, width) summing to "
                    f"half the rotary width; got {mrope_section} for {layer_type!r} with a "
                    f"rotary width of {rope_head_dim}"
                )

        # The per-head sizes are hash moduli: a table of the wrong shape would hash
        # against the wrong moduli instead of failing.
        if self.engram_layer_ids and not self.engram_pad_id < self.engram_lookup_table_size:
            raise ValueError(
                f"`engram_lookup_table_size` ({self.engram_lookup_table_size}) must cover "
                f"`engram_pad_id` ({self.engram_pad_id})"
            )
        num_orders = self.engram_max_ngram_size - 1
        sizes_by_layer = getattr(self, "engram_layer_head_vocab_sizes", None) or {}
        for layer_id in self.engram_layer_ids:
            sizes = sizes_by_layer.get(str(layer_id))
            if (
                not isinstance(sizes, list)
                or len(sizes) != num_orders
                or any(
                    not isinstance(order, list) or len(order) != self.engram_n_head_per_ngram
                    for order in sizes
                )
            ):
                raise ValueError(
                    f"`engram_layer_head_vocab_sizes['{layer_id}']` must hold {num_orders} "
                    f"lists (n-gram orders 2 to {self.engram_max_ngram_size}) of "
                    f"{self.engram_n_head_per_ngram} sizes each; got {sizes}"
                )

    def convert_rope_params_to_dict(self, **kwargs):
        """Complete the checkpoint's per-layer-type RoPE blocks.

        The checkpoint nests one block per layer type, each carrying its own base
        wavelength and the interleaved M-RoPE split, and publishes no top-level
        ``rope_theta``. The base implementation fills a single global theta, which
        would collapse the two bases, so each block is completed here instead.
        """
        rope_scaling = kwargs.pop("rope_scaling", None)
        if rope_scaling is not None:
            self.rope_parameters = rope_scaling
        rope_theta = kwargs.pop("rope_theta", None)
        rope_parameters = {
            layer_type: dict(params)
            for layer_type, params in (self.rope_parameters or {}).items()
            if isinstance(params, dict)
        }
        mrope_section = kwargs.get("mrope_section", getattr(self, "mrope_section", None))
        mrope_interleaved = kwargs.get(
            "mrope_interleaved", getattr(self, "mrope_interleaved", None)
        )
        default_theta = self.default_theta
        for layer_type in set(self.layer_types):
            params = rope_parameters.setdefault(layer_type, {})
            params.setdefault("rope_type", "default")
            theta = default_theta[layer_type] if isinstance(default_theta, dict) else default_theta
            params.setdefault("rope_theta", rope_theta if rope_theta is not None else theta)
            if mrope_section is not None:
                params.setdefault("mrope_section", list(mrope_section))
            if mrope_interleaved is not None:
                params.setdefault("mrope_interleaved", mrope_interleaved)
        self.rope_parameters = rope_parameters
        self.standardize_rope_params()
        return kwargs


class HYV4FlashVisionConfig(PretrainedConfig):
    """The HY4-Flash SigLIP2-like vision tower and its projector (``vision_config``)."""

    model_type = "hy_v4_flash_vision"
    base_config_key = "vision_config"
    attribute_map = {"num_heads": "num_attention_heads"}

    def __init__(self, **kwargs):
        fields = {
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "spatial_patch_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 3072,
            "text_hidden_size": 3072,
            "hidden_act": "gelu",
            "rms_norm_eps": 1e-05,
            "max_image_size": 2048,
            "min_image_size": 512,
            "resize_resolution": 2048,
            "anyres_vit_max_image_size": 2048,
            "img_max_token_num": 4096,
            "max_vit_seq_len": 16384,
            "video_max_image_size": 768,
            "video_min_image_size": 256,
            "packed_seq_alignment": 128,
            "projector_type": "rmsnorm_mlp_adapter",
            "add_patchemb_bias": True,
            "bias_activation_fusion": True,
            "cat_extra_token": 0,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "attention_backend": "hf_align_sdpa",
            "interpolate_mode": "bilinear",
            "learnable_mlp_pooling_size": 0,
            "perceive_post_norm": False,
            "perceive_pre_norm": True,
            "remove_prenorm": True,
            "rope_2d_percent": 1.0,
            "rope_2d_theta": 10000.0,
            "video_spatial_compression": False,
            "video_temporal_compression": False,
            "vision_full_attention": False,
        }
        fields.update(kwargs)
        # The published checkpoint labels the nested tower with the composite's
        # own `model_type`, which would recurse under `AutoConfig` resolution.
        fields.pop("model_type", None)
        super().__init__(**fields)
        self.head_dim = self.hidden_size // self.num_attention_heads


class HYV4FlashConfig(PretrainedConfig):
    """The HY4-Flash composite config: text decoder, vision tower and DFly2 geometry."""

    model_type = "hy_v4_flash"
    sub_configs = {
        "text_config": HYV4FlashTextConfig,
        "vision_config": HYV4FlashVisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        self.text_config = (
            HYV4FlashTextConfig(**text_config) if isinstance(text_config, dict) else text_config
        )
        self.vision_config = (
            HYV4FlashVisionConfig(**vision_config)
            if isinstance(vision_config, dict)
            else vision_config
        )
        fields = {
            "image_start_token_id": 120118,
            "vision_start_token_id": 120118,
            "image_end_token_id": 120119,
            "image_token_id": 120120,
            "image_newline_token_id": 120121,
            "video_start_token_id": 120122,
            "video_end_token_id": 120123,
            "video_token_id": 120124,
            "spatial_merge_size": 2,
            "tie_word_embeddings": False,
            # Checkpoint-native DFly2 drafting: five draft layers over the target's
            # sparse layers 25, 29, 33, 37 and 41 (0-based), each reusing that
            # layer's MLA KV and its top-`dfly_target_index_topk` index picks.
            "dfly_num_hidden_layers": 5,
            "dfly_mask_token_id": 120009,
            "dfly_target_kv_layers": [25, 29, 33, 37, 41],
            "dfly_target_index_layer": [25, 29, 33, 37, 41],
            "dfly_target_index_topk": 2048,
            "dfly_mtp_fusion": "slot0",
            "dfly_enable_hidden_correction": True,
            "dfly_context_local_window": None,
        }
        fields.update(kwargs)
        super().__init__(**fields)
