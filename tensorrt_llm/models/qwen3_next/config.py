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
from typing import TYPE_CHECKING, List, Optional, Union

from ...layers import MoeConfig
from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig

if TYPE_CHECKING:
    import transformers


def _get_rope_theta(text_cfg: dict, default: float = 10000000.0) -> float:
    """RoPE ``theta`` from a (possibly v5-nested) HF text config dict.

    Transformers v5+ nests ``rope_theta`` under ``rope_parameters``; older
    releases expose ``rope_theta`` directly. Inlined here so this module does
    not depend on ``tensorrt_llm._utils.get_hf_rope_theta``, which is absent in
    older installed wheels.
    """
    theta = text_cfg.get("rope_theta")
    if theta is not None:
        return float(theta)
    rope_params = text_cfg.get("rope_parameters")
    if isinstance(rope_params, dict) and rope_params.get("rope_theta") is not None:
        return float(rope_params["rope_theta"])
    return default


# Layer-type tokens used by the HF ``qwen3_5_moe`` / Qwen3-Next configs.
LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"


class Qwen3NextConfig(PretrainedConfig):
    """Engine-path config for the Qwen3.5/Qwen3.6 hybrid MoE family.

    Parses the nested HF ``config.json`` (``text_config`` + ``vision_config`` +
    ``quantization_config``) into the flat field set the TensorRT-LLM engine
    builder expects, while preserving the hybrid layer pattern (``layer_types``)
    and the Gated DeltaNet ("linear attention") parameters needed to build the
    recurrent layers.

    Only the *text decoder* fields are consumed by the engine model today. The
    vision-tower and MTP fields are parsed and stored so the converter can carry
    those weights, but building/running them is not yet supported on the engine
    path.
    """

    def __init__(
        self,
        *,
        # --- hybrid layer pattern ---
        layer_types: Optional[List[str]] = None,
        full_attention_interval: int = 4,
        # --- full attention ---
        attn_bias: bool = False,
        attn_output_gate: bool = True,
        rotary_base: float = 10000000.0,
        rotary_scaling: Optional[dict] = None,
        partial_rotary_factor: float = 0.25,
        mrope_section: Optional[List[int]] = None,
        mrope_interleaved: bool = True,
        # --- gated deltanet (linear attention) ---
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_value_head_dim: int = 128,
        mamba_ssm_dtype: str = "float32",
        # --- moe ---
        moe: Optional[Union[MoeConfig, dict]] = None,
        moe_intermediate_size: int = 0,
        moe_shared_expert_intermediate_size: int = 0,
        # --- mtp ---
        mtp_num_hidden_layers: int = 0,
        mtp_use_dedicated_embeddings: bool = False,
        # --- multimodal ---
        vision_config: Optional[dict] = None,
        image_token_id: Optional[int] = None,
        video_token_id: Optional[int] = None,
        vision_start_token_id: Optional[int] = None,
        vision_end_token_id: Optional[int] = None,
        **kwargs,
    ):
        # Keep the canonical HF vocabulary internally (``linear_attention`` /
        # ``full_attention``) so the Python build's per-layer dispatch works. But
        # tolerate the C++-runtime vocabulary too: a round-trip rebuild from a
        # saved TRT-LLM checkpoint would otherwise feed back the translated
        # ``recurrent``/``attention`` strings (emitted by ``to_dict`` for the C++
        # ``buildLayerTypes`` parser) and silently build an all-attention model.
        _runtime_to_canonical = {
            "recurrent": LINEAR_ATTENTION,
            "attention": FULL_ATTENTION,
        }
        self.layer_types = [_runtime_to_canonical.get(t, t) for t in (layer_types or [])]
        self.full_attention_interval = full_attention_interval

        self.attn_bias = attn_bias
        self.attn_output_gate = attn_output_gate
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.partial_rotary_factor = partial_rotary_factor
        # ``Attention.create_attention_const_params`` reads the rotary fraction
        # as ``config.rotary_pct`` (the RecurrentGemma/engine convention),
        # defaulting to 1.0 (FULL RoPE) when absent. The HF checkpoint expresses
        # it as ``partial_rotary_factor`` (0.25 -> rotary_dim 64), so expose the
        # alias too — without it the gpt_attention RoPE const params (baked into
        # the engine) use the wrong rotary dimension and the model emits
        # gibberish (Q/K rotated over 256 dims instead of 64).
        self.rotary_pct = partial_rotary_factor
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved

        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_value_head_dim = linear_value_head_dim
        self.mamba_ssm_dtype = mamba_ssm_dtype

        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        if moe is None:
            moe = MoeConfig(
                num_experts=kwargs.pop("moe_num_experts", 0),
                top_k=kwargs.pop("moe_top_k", 0),
                normalization_mode=MoeConfig.ExpertScaleNormalizationMode.NONE,
            )
        elif isinstance(moe, dict):
            moe = MoeConfig.from_dict(moe)
        assert isinstance(moe, MoeConfig)
        self.moe = moe.validate()

        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        super().__init__(**kwargs)

        # Recurrent-state runtime fields consumed by the C++ RnnConfig parse
        # (gptJsonConfig.cpp, gated by isRnnBased()/ModelVariant::kQwen3Next).
        # Derived from the Gated DeltaNet linear_* dims; the same names + meaning
        # Mamba uses (models/mamba/config.py). Set AFTER super().__init__ so they
        # always win over any same-named value leaked in via **kwargs on a
        # checkpoint round-trip. They serialize automatically through the base
        # ``to_dict`` (deepcopy of __dict__) and are inert to the Python build
        # (only the C++ recurrent-state runtime reads them). For GDN layer-0:
        # state_size 128 / conv_kernel 4 / rnn_hidden_size 4096 / rnn_head_size
        # 128 / rnn_conv_dim_size 8192 -> rnn state [L,S,32,128,128], conv state
        # [L,S,3,8192] (matches the engine's past_rnn_state / past_conv_state).
        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        self.state_size = self.linear_key_head_dim
        self.conv_kernel = self.linear_conv_kernel_dim
        self.rnn_hidden_size = value_dim
        self.rnn_head_size = self.linear_value_head_dim
        self.rnn_conv_dim_size = 2 * key_dim + value_dim

    # ------------------------------------------------------------------
    # Per-layer helpers (hybrid pattern)
    # ------------------------------------------------------------------
    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        """Whether ``layer_idx`` is a Gated DeltaNet (linear attention) layer."""
        return self.get_layer_type(layer_idx) == LINEAR_ATTENTION

    def get_layer_type(self, layer_idx: int) -> str:
        if self.layer_types:
            return self.layer_types[layer_idx % len(self.layer_types)]
        # Fall back to the interval rule used by the HF config.
        is_full = (layer_idx + 1) % self.full_attention_interval == 0
        return FULL_ATTENTION if is_full else LINEAR_ATTENTION

    @property
    def has_vision(self) -> bool:
        return self.vision_config is not None

    @property
    def has_mtp(self) -> bool:
        return self.mtp_num_hidden_layers > 0

    def to_dict(self):
        output = super().to_dict()
        # NOTE: 'layer_types' is intentionally NOT in this list — it is emitted
        # below, translated to the C++ runtime vocabulary. The 5 recurrent-state
        # fields (state_size/conv_kernel/rnn_hidden_size/rnn_head_size/
        # rnn_conv_dim_size) are NOT listed either: they ride the base
        # ``to_dict``'s deepcopy of ``__dict__`` (the Mamba pattern), and this
        # explicit loop only overwrites the keys it names, so it never clobbers
        # them.
        for f in [
            "full_attention_interval",
            "attn_bias",
            "attn_output_gate",
            "rotary_base",
            "rotary_scaling",
            "partial_rotary_factor",
            "rotary_pct",
            "mrope_section",
            "mrope_interleaved",
            "linear_conv_kernel_dim",
            "linear_key_head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_value_head_dim",
            "mamba_ssm_dtype",
            "moe_intermediate_size",
            "moe_shared_expert_intermediate_size",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
            "vision_config",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]:
            output[f] = getattr(self, f)
        output["moe"] = self.moe.to_dict()
        # Translate layer_types to the C++ runtime vocabulary. The C++
        # ``buildLayerTypes`` (gptJsonConfig.cpp) only recognizes
        # attention/recurrent/linear/no_op; unknown strings -> WARNING + default
        # attention. The 30 Gated DeltaNet (linear_attention) layers are the
        # recurrent/conv-state path -> 'recurrent'; the 10 full_attention layers
        # use the KV cache -> 'attention'. We translate ONLY here, leaving the
        # in-memory ``self.layer_types`` canonical so the Python build is intact.
        _canonical_to_runtime = {
            LINEAR_ATTENTION: "recurrent",
            FULL_ATTENTION: "attention",
        }
        output["layer_types"] = [_canonical_to_runtime.get(t, t) for t in self.layer_types]
        return output

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: Union[str, "transformers.PretrainedConfig"],
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ) -> "Qwen3NextConfig":
        import json
        import os

        import transformers

        trust_remote_code = kwargs.pop("trust_remote_code", True)

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_dict = hf_config_or_dir.to_dict()
        else:
            # The Qwen3.5/3.6 ``qwen3_5_moe`` arch may be newer than the
            # installed transformers, in which case ``AutoConfig`` raises.
            # The engine path only needs the config dict, so read config.json
            # directly when AutoConfig cannot recognize the model type.
            cfg_dir = str(hf_config_or_dir)
            try:
                hf_config = transformers.AutoConfig.from_pretrained(
                    cfg_dir, trust_remote_code=trust_remote_code
                )
                hf_dict = hf_config.to_dict()
            except (ValueError, KeyError):
                cfg_path = (
                    cfg_dir if cfg_dir.endswith(".json") else os.path.join(cfg_dir, "config.json")
                )
                with open(cfg_path) as f:
                    hf_dict = json.load(f)
        # Top-level multimodal wrapper -> text_config holds the decoder.
        text_cfg = hf_dict.get("text_config", hf_dict)
        vision_cfg = hf_dict.get("vision_config", None)
        hf_quant_cfg = hf_dict.get("quantization_config")

        arch = hf_dict.get("architectures", ["Qwen3_5MoeForConditionalGeneration"])[0]

        head_dim = text_cfg.get(
            "head_dim", text_cfg["hidden_size"] // text_cfg["num_attention_heads"]
        )

        rope_params = text_cfg.get("rope_parameters", {}) or {}
        rotary_base = _get_rope_theta(text_cfg, 10000000.0)
        partial_rotary_factor = rope_params.get("partial_rotary_factor") or text_cfg.get(
            "partial_rotary_factor", 1.0
        )
        mrope_section = rope_params.get("mrope_section")
        mrope_interleaved = rope_params.get("mrope_interleaved", False)
        rotary_scaling = text_cfg.get("rope_scaling", None)

        dtype = infer_dtype(dtype, text_cfg.get("dtype") or text_cfg.get("torch_dtype"))

        moe_config = MoeConfig(
            num_experts=text_cfg.get("num_experts", 0),
            top_k=text_cfg.get("num_experts_per_tok", 0),
            normalization_mode=MoeConfig.ExpertScaleNormalizationMode.NONE,
        )
        moe_config.validate()

        # Mixed-precision (FP8 attention + NVFP4 MoE/lm_head + bf16 MTP/vision)
        # quant ingestion. An explicitly-supplied ``quant_config``
        # takes precedence; otherwise parse the checkpoint's modelopt
        # ``quantization_config`` into a per-layer ``LayerQuantConfig`` so the
        # engine builds real FP8/NVFP4 weights (not a silent bf16 fallback).
        quantization = quant_config
        if (
            quantization is None
            and isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_algo") == "MIXED_PRECISION"
        ):
            from .convert import build_layer_quant_config

            num_layers = text_cfg["num_hidden_layers"]
            layer_types = text_cfg.get("layer_types")
            full_attn_interval = text_cfg.get("full_attention_interval", 4)

            def _is_linear(layer_idx: int) -> bool:
                if layer_types:
                    return layer_types[layer_idx % len(layer_types)] == LINEAR_ATTENTION
                return (layer_idx + 1) % full_attn_interval != 0

            quantization = build_layer_quant_config(hf_quant_cfg, num_layers, _is_linear)

        return cls(
            architecture=arch,
            dtype=dtype,
            # core dims
            num_hidden_layers=text_cfg["num_hidden_layers"],
            num_attention_heads=text_cfg["num_attention_heads"],
            num_key_value_heads=text_cfg.get("num_key_value_heads"),
            hidden_size=text_cfg["hidden_size"],
            intermediate_size=text_cfg.get(
                "intermediate_size", text_cfg.get("moe_intermediate_size")
            ),
            head_size=head_dim,
            vocab_size=text_cfg["vocab_size"],
            max_position_embeddings=text_cfg.get("max_position_embeddings"),
            norm_epsilon=text_cfg.get("rms_norm_eps", 1e-6),
            hidden_act="swiglu",
            tie_word_embeddings=text_cfg.get("tie_word_embeddings", False),
            # hybrid pattern
            layer_types=text_cfg.get("layer_types"),
            full_attention_interval=text_cfg.get("full_attention_interval", 4),
            # full attention
            attn_bias=text_cfg.get("attention_bias", False),
            attn_output_gate=text_cfg.get("attn_output_gate", True),
            position_embedding_type="rope_gpt_neox",
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
            qk_layernorm=True,
            # gated deltanet
            linear_conv_kernel_dim=text_cfg.get("linear_conv_kernel_dim", 4),
            linear_key_head_dim=text_cfg.get("linear_key_head_dim", 128),
            linear_num_key_heads=text_cfg.get("linear_num_key_heads", 16),
            linear_num_value_heads=text_cfg.get("linear_num_value_heads", 32),
            linear_value_head_dim=text_cfg.get("linear_value_head_dim", 128),
            mamba_ssm_dtype=text_cfg.get("mamba_ssm_dtype", "float32"),
            # moe
            moe=moe_config,
            moe_intermediate_size=text_cfg.get("moe_intermediate_size", 0),
            moe_shared_expert_intermediate_size=text_cfg.get("shared_expert_intermediate_size", 0),
            # mtp
            mtp_num_hidden_layers=text_cfg.get("mtp_num_hidden_layers", 0),
            mtp_use_dedicated_embeddings=text_cfg.get("mtp_use_dedicated_embeddings", False),
            # multimodal
            vision_config=vision_cfg,
            image_token_id=hf_dict.get("image_token_id"),
            video_token_id=hf_dict.get("video_token_id"),
            vision_start_token_id=hf_dict.get("vision_start_token_id"),
            vision_end_token_id=hf_dict.get("vision_end_token_id"),
            mapping=mapping,
            quantization=quantization,
            **kwargs,
        )
