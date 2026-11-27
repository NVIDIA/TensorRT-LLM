# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 configuration for Transformers versions without native `inkling2` support.

`thinkingmachines/Inkling-2-Small` ships a composite `config.json` (`model_type: inkling2`) with
`text_config`, `vision_config` and `mtp_config` blocks and no `auto_map`. Native
Transformers config classes take precedence when available.

The text config also publishes the attributes the engine reads to size the KV cache:
`layer_types` (`sliding_attention` for the layers in `local_layer_ids`, `full_attention`
for the rest), `sliding_window` and `max_position_embeddings`. Without `local_layer_ids`, every
sixth layer is global, as in the release, for whatever number of layers the config has.
"""

from transformers import PretrainedConfig

#: The fields `layer_types`, `sliding_window` and `max_position_embeddings` are derived from.
_DERIVATION_SOURCES = frozenset(
    {"num_hidden_layers", "local_layer_ids", "sliding_window_size", "model_max_length"}
)


class Inkling2TextConfig(PretrainedConfig):
    """Text decoder of Inkling 2 (`text_config`).

    Defaults mirror the Inkling-2-Small release; values in a checkpoint's `config.json` take
    precedence. The derived attributes follow their source fields when those are set after
    construction, as `from_dict` keyword overrides and `model_kwargs` do.
    """

    model_type = "inkling2_text"
    base_config_key = "text_config"

    def __init__(self, **kwargs):
        defaults = dict(
            vocab_size=201024,
            unpadded_vocab_size=200058,
            hidden_size=4608,
            num_hidden_layers=48,
            num_attention_heads=48,
            num_key_value_heads=8,
            head_dim=128,
            swa_num_attention_heads=48,
            swa_num_key_value_heads=8,
            swa_head_dim=128,
            sliding_window_size=1024,
            local_layer_ids=None,
            d_rel=16,
            rel_extent=1024,
            q_bias=False,
            o_bias=False,
            log_scaling_n_floor=8192,
            log_scaling_alpha=0.1,
            rms_norm_eps=1e-6,
            pre_norm_pre_affine=False,
            final_norm_pre_affine=False,
            use_value_norm_no_gamma=True,
            qk_l2_clip_threshold=22.627416997969522,
            rel_l2_clip_threshold=8.0,
            sconv_kernel_size=4,
            final_logit_softcapping=30.0,
            n_layers_per_attn_res_block=6,
            attn_res_normalize_values=True,
            n_routed_experts=384,
            num_experts_per_tok=8,
            n_shared_experts=0,
            intermediate_size=2304,
            mlp_gate_softcap=4.0,
            mlp_up_softcap=25.0,
            route_scale=8.0,
            use_gate_bias=True,
            model_max_length=1048576,
            tie_word_embeddings=False,
        )
        fields = {**defaults, **kwargs}
        # Derived below; a stale serialized copy must not override the source fields.
        for derived in ("layer_types", "sliding_window", "max_position_embeddings"):
            fields.pop(derived, None)
        super().__init__(**fields)
        self.__dict__.update(self._derived(self.__dict__))

    def __setattr__(self, key, value):
        if key in _DERIVATION_SOURCES and "layer_types" in self.__dict__:
            fields = {**self.__dict__, key: value}
            if key == "num_hidden_layers" and fields["local_layer_ids"] is not None:
                # A smaller model (the `model_kwargs` shrink idiom) keeps the explicit
                # sliding-window layers it still has; an explicit `local_layer_ids` outside the
                # layers is still refused.
                fields["local_layer_ids"] = [i for i in fields["local_layer_ids"] if int(i) < value]
            # Validate against the new values before anything changes, then keep the derived
            # attributes in step with their sources.
            derived = self._derived(fields)
            super().__setattr__(key, value)
            self.__dict__.update(derived)
        else:
            super().__setattr__(key, value)

    @staticmethod
    def _derived(fields) -> dict:
        """The engine-facing attributes for the source values in `fields`.

        Explicit `local_layer_ids` are kept (sorted) and must name existing layers; unset ones stay
        unset, so the released pattern follows a later change of `num_hidden_layers`.
        """
        num_layers = fields["num_hidden_layers"]
        derived = {}
        if fields["local_layer_ids"] is None:
            # The released pattern: five sliding-window layers, then one global layer.
            local = {i for i in range(num_layers) if i % 6 != 5}
        else:
            local_layer_ids = sorted(int(i) for i in fields["local_layer_ids"])
            out_of_range = [i for i in local_layer_ids if not 0 <= i < num_layers]
            if out_of_range:
                raise ValueError(
                    f"local_layer_ids {out_of_range} are outside the {num_layers} decoder layers"
                )
            derived["local_layer_ids"] = local_layer_ids
            local = set(local_layer_ids)
        derived["layer_types"] = [
            "sliding_attention" if i in local else "full_attention" for i in range(num_layers)
        ]
        derived["sliding_window"] = fields["sliding_window_size"]
        derived["max_position_embeddings"] = fields["model_max_length"]
        return derived

    @property
    def global_layer_ids(self) -> list:
        """Decoder layers that attend over the whole context."""
        return [i for i, kind in enumerate(self.layer_types) if kind == "full_attention"]


class Inkling2VisionConfig(PretrainedConfig):
    """Image encoder of Inkling 2 (`vision_config`)."""

    model_type = "inkling2_vision"
    base_config_key = "vision_config"

    def __init__(self, **kwargs):
        defaults = dict(
            temporal_patch_size=2,
            native_patch_size=20,
            adapter_merge_size=2,
            projection_hidden_dim=4096,
            projector_bias=False,
            hidden_size=1024,
            num_hidden_layers=23,
            num_attention_heads=16,
            mlp_ratio=4.0,
            posemb_grid_size=32,
            layer_norm_eps=1e-5,
        )
        super().__init__(**{**defaults, **kwargs})


class Inkling2Config(PretrainedConfig):
    """Inkling 2 (`Inkling2ForConditionalGeneration`): text decoder, image encoder and drafter.

    `mtp_config` describes the bundled DSpark drafter (`mtp.safetensors`) and is kept as the
    plain mapping the checkpoint stores.
    """

    model_type = "inkling2"
    sub_configs = {"text_config": Inkling2TextConfig, "vision_config": Inkling2VisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, text_config=None, vision_config=None, mtp_config=None, **kwargs):
        self.text_config = (
            Inkling2TextConfig(**(text_config if text_config is not None else {}))
            if isinstance(text_config, dict) or text_config is None
            else text_config
        )
        self.vision_config = (
            Inkling2VisionConfig(**(vision_config or {}))
            if isinstance(vision_config, dict) or vision_config is None
            else vision_config
        )
        self.mtp_config = dict(mtp_config) if mtp_config is not None else None
        defaults = dict(
            eos_token_id=200006,
            image_token_id=200054,
            tie_word_embeddings=False,
        )
        super().__init__(**{**defaults, **kwargs})
