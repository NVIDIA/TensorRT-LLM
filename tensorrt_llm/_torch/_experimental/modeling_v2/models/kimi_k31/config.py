# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 config classes.

A K3.1 checkpoint (``model_type: kimi_k31``) has Kimi K3's composite layout: a ``text_config`` (model_type
``kimi_linear``) and a MoonViT ``vision_config`` (model_type ``kimi_k3_vision``). These classes parse it without the
checkpoint's remote code. Next to Kimi K3's text fields, the text config has two of its own:

* ``dsa_config``: a DeepSeek sparse-attention (DSA) indexer on the MLA layers it lists, which selects the
  ``index_topk`` cached tokens each query attends to. The flat ``index_*`` properties name its fields the way the
  sparse-attention config reads them from a DeepSeek-V3.2 config.
* ``kda_fp32_gate_params``: the KDA layers keep ``A_log`` and ``dt_bias`` at the checkpoint's FP32 (always True for
  Kimi K3.1; the KDA layer refuses False).

The text config keeps model_type ``kimi_linear``: the engine's Kimi KDA hybrid-cache paths key on that model_type, so
they serve K3.1 as they serve K3.

It also reads the ModelOpt NVFP4 export's precision per module (``modelopt_layer_quant``) and says whether a
checkpoint is that export as the sm_107 targets read it (``modelopt_nvfp4_mismatch``): routing and the targets' weight
tables ask the same question.

The engine reads these classes through ``architecture.py`` before a target is routed, so this module imports no target
and no kernel.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm.quantization.mode import QuantAlgo

#: model_types of the composite (text + vision) Kimi configs whose ``text_config`` the target reads.
KIMI_K3_COMPOSITE_MODEL_TYPES = ("kimi_k3", "kimi_k31")


def text_config_of(pretrained_config: PretrainedConfig) -> PretrainedConfig:
    """The Kimi text config: the ``text_config`` of a composite config, else ``pretrained_config`` itself."""
    if getattr(pretrained_config, "model_type", None) in KIMI_K3_COMPOSITE_MODEL_TYPES or (
        not hasattr(pretrained_config, "linear_attn_config")
        and hasattr(pretrained_config, "text_config")
    ):
        return pretrained_config.text_config
    return pretrained_config


class KimiK31TextConfig(PretrainedConfig):
    """The text model's config: Kimi K3's ``kimi_linear`` fields plus K3.1's DSA indexer and fp32 KDA gate
    parameters."""

    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        dsa_config: Optional[dict] = None,
        kda_fp32_gate_params: bool = True,
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: Optional[int] = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        mla_use_nope: Optional[bool] = False,
        mla_use_output_gate: Optional[bool] = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: Optional[dict] = None,
        attn_res_block_size: Optional[int] = None,
        latent_moe_use_norm: bool = False,
        activation_situ_beta: Optional[float] = None,
        activation_situ_linear_beta: Optional[float] = None,
        max_position_embeddings: int = 4096,
        routed_expert_hidden_size: Optional[int] = None,
        topk_method: str = "noaux_tc",
        **kwargs,
    ):
        if dsa_config is not None:
            assert dsa_config["dsa_layers"] is not None
            # The reference defaults to an RMSNorm key norm; only LayerNorm is implemented.
            k_norm_type = dsa_config.get("index_k_norm_type", "rmsnorm")
            if k_norm_type != "layernorm":
                raise NotImplementedError(
                    f"dsa_config.index_k_norm_type={k_norm_type!r} is not supported; "
                    "the DSA indexer key norm is a LayerNorm."
                )
        self.dsa_config = dsa_config
        self.kda_fp32_gate_params = kda_fp32_gate_params

        # transformers keys registry lookups off the class attribute, so a ``model_type`` kwarg must not shadow it.
        kwargs.pop("model_type", None)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.attn_res_block_size = attn_res_block_size
        self.latent_moe_use_norm = latent_moe_use_norm
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.max_position_embeddings = max_position_embeddings
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.topk_method = topk_method

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def is_dsa_layer(self, layer_idx: int) -> bool:
        """0-indexed layer check; ``dsa_layers`` is 1-indexed."""
        return self.dsa_config is not None and (layer_idx + 1) in self.dsa_config["dsa_layers"]

    def _dsa_value(self, name: str, default=None):
        if self.dsa_config is None:
            return default
        return self.dsa_config.get(name, default)

    @property
    def index_n_heads(self) -> Optional[int]:
        return self._dsa_value("index_num_heads")

    @property
    def index_head_dim(self) -> Optional[int]:
        return self._dsa_value("index_head_dim")

    @property
    def index_topk(self) -> Optional[int]:
        return self._dsa_value("index_topk")

    @property
    def index_rope_dim(self) -> Optional[int]:
        return self._dsa_value("index_rope_dim")

    @property
    def index_k_norm_eps(self) -> Optional[float]:
        # The indexer key LayerNorm uses the model's norm epsilon.
        return None if self.dsa_config is None else self.rms_norm_eps

    @property
    def index_topk_freq(self) -> int:
        return self._dsa_value("index_topk_freq", 1)

    @property
    def index_skip_topk_offset(self) -> int:
        return self._dsa_value("index_skip_topk_offset", 2)


class KimiK31VisionConfig(PretrainedConfig):
    """The vision tower's config: a MoonViT-3D encoder (``vt_*`` fields) and a ``patchmergerv2`` MLP projector
    (``mm_*`` / ``projector_*`` fields), as Kimi K3 ships them."""

    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple = (2, 2),
        merge_type: str = "sd2_tpool",
        _attn_implementation: str = "flash_attention_2",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: Optional[int] = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        text_hidden_size: int = 7168,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = tuple(merge_kernel_size)
        self.merge_type = merge_type

        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.text_hidden_size = text_hidden_size

        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id

        # transformers v5's PretrainedConfig.__init__ assigns ``attn_implementation`` (default None) over a
        # ``_attn_implementation`` set before it, so the default goes in as the kwarg; a caller's
        # ``attn_implementation`` still wins.
        kwargs.setdefault("attn_implementation", _attn_implementation)
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class KimiK31Config(PretrainedConfig):
    """Top-level composite config of a Kimi K3.1 checkpoint: a :class:`KimiK31TextConfig` text config and a
    :class:`KimiK31VisionConfig` vision config, built from the nested dicts of ``config.json``."""

    model_type = "kimi_k31"

    def __init__(
        self,
        text_config: Optional[Union[dict, KimiK31TextConfig]] = None,
        vision_config: Optional[Union[dict, KimiK31VisionConfig]] = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = KimiK31TextConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = KimiK31VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config

        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        # The routed-expert MXFP4 quantization is on the text config; the engine's quant-config extraction reads it at
        # the top level.
        if getattr(self.text_config, "quantization_config", None) is not None:
            self.quantization_config = self.text_config.quantization_config

        super().__init__(pad_token_id=pad_token_id, **kwargs)

    @property
    def dsa_indexer_config(self) -> Optional[KimiK31TextConfig]:
        """The config that carries the DSA indexer fields (the text config), or None without a ``dsa_config``.

        The engine builds the sparse-attention config from it before any target is routed.
        """
        text_config = self.text_config
        return text_config if getattr(text_config, "dsa_config", None) is not None else None


# The ModelOpt NVFP4 export (``hf_quant_config.json``, MIXED_PRECISION). Its ``quantized_layers`` name each module
# under three prefixes and a MoE layer's experts also under ``mlp.``; the fused names (``fused_qkvg_proj``,
# ``in_proj_qkvgfab``, ``gate_up_proj``) are quantization groups over the same tensors, not tensors of their own.
_MODELOPT_PREFIXES = ("language_model.model.", "model.")
_MODELOPT_MOE_ALIASES = (
    (".mlp.experts", ".block_sparse_moe.experts"),
    (".mlp.shared_experts.", ".block_sparse_moe.shared_experts."),
)
_KDA_MXFP8 = ("q_proj", "k_proj", "v_proj", "g_proj", "b_proj", "f_a_proj", "o_proj")
_KDA_MXFP8_GROUPS = ("fused_qkvg_proj", "in_proj_qkvgfab")
_SHARED_MXFP8 = ("gate_proj", "up_proj", "down_proj")
_SHARED_MXFP8_GROUPS = ("gate_up_proj",)
#: Values per E4M3 scale of the export's NVFP4 routed experts.
MODELOPT_NVFP4_GROUP_SIZE = 16


def _modelopt_module(name: str) -> str:
    """A ``quantized_layers`` name as one spelling: ``layers.{i}.<module>``, MoE modules under ``block_sparse_moe``."""
    for prefix in _MODELOPT_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for alias, module in _MODELOPT_MOE_ALIASES:
        if alias in name:
            return name.replace(alias, module, 1)
    return name


def _field(cfg: Any, name: str) -> Any:
    return cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)


def modelopt_layer_quant(model_config: Any) -> Optional[Dict[str, Tuple[QuantAlgo, Optional[int]]]]:
    """``layers.{i}.<module>`` -> (algo, group size) of a ModelOpt MIXED_PRECISION checkpoint, or None for any other.

    Read from the engine's parse of ``hf_quant_config.json`` (``quant_config_dict``), the export's spellings of one
    module folded into one entry. A module without an entry is BF16. Raises ``ValueError`` when two spellings of a
    module disagree.
    """
    quant = getattr(model_config, "quant_config", None)
    if _field(quant, "quant_algo") != QuantAlgo.MIXED_PRECISION:
        return None
    layers: Dict[str, Tuple[QuantAlgo, Optional[int]]] = {}
    for name, cfg in (getattr(model_config, "quant_config_dict", None) or {}).items():
        module = _modelopt_module(name)
        entry = (QuantAlgo(_field(cfg, "quant_algo")), _field(cfg, "group_size"))
        if layers.setdefault(module, entry) != entry:
            raise ValueError(
                f"the ModelOpt quantized_layers give {module} as {layers[module]} and as {entry}"
            )
    return layers


def _modelopt_nvfp4_modules(text) -> Tuple[Dict[str, tuple], Dict[str, tuple]]:
    """(modules the NVFP4 export quantizes, its quantization groups) of a K3.1 text config, with their precision."""
    nvfp4 = (QuantAlgo.NVFP4, MODELOPT_NVFP4_GROUP_SIZE)
    mxfp8 = (QuantAlgo.MXFP8, None)
    n = text.num_hidden_layers
    modules: Dict[str, tuple] = {}
    groups: Dict[str, tuple] = {}
    if getattr(text, "num_experts", None) is not None:
        for i in range(text.first_k_dense_replace, n):
            prefix = f"layers.{i}.block_sparse_moe."
            modules[prefix + "experts"] = nvfp4
            modules.update({f"{prefix}shared_experts.{m}": mxfp8 for m in _SHARED_MXFP8})
            groups.update({f"{prefix}shared_experts.{m}": mxfp8 for m in _SHARED_MXFP8_GROUPS})
    # linear_attn_config numbers layers from 1.
    for layer in text.linear_attn_config["kda_layers"]:
        if layer <= n:
            prefix = f"layers.{layer - 1}.self_attn."
            modules.update({prefix + m: mxfp8 for m in _KDA_MXFP8})
            groups.update({prefix + m: mxfp8 for m in _KDA_MXFP8_GROUPS})
    return modules, groups


def _precision(entry: tuple) -> str:
    algo, group_size = entry
    return f"{algo.name}" if group_size is None else f"{algo.name} (group {group_size})"


def modelopt_nvfp4_mismatch(model_config: Any, nvfp4_kv_cache: bool = False) -> Optional[str]:
    """Why the checkpoint is not the ModelOpt NVFP4 export as the sm_107 targets read it, or None.

    That export: MIXED_PRECISION; the routed experts NVFP4 with a scale per 16 values; the shared experts and the KDA
    projections (q, k, v, g, b, f_a, o) MXFP8; everything else BF16; no KV-cache quantization, so no KV scales. The
    engine writes the server's ``kv_cache_config`` dtype into ``quant_config`` before the weights load, so the KV cache
    a target runs reads here: FP8 for the specialized targets; FP8 or NVFP4 for the default target
    (``nvfp4_kv_cache``), which sets an NVFP4 cache's scales itself. Neither changes a weight read; any other KV cache
    is refused.
    """
    try:
        layers = modelopt_layer_quant(model_config)
    except ValueError as e:
        return str(e)
    if layers is None:
        algo = _field(getattr(model_config, "quant_config", None), "quant_algo")
        return f"the checkpoint is not a ModelOpt MIXED_PRECISION export (quant_algo {algo})"
    kv_algo = _field(model_config.quant_config, "kv_cache_quant_algo")
    kv_algos = (None, QuantAlgo.FP8, QuantAlgo.NVFP4) if nvfp4_kv_cache else (None, QuantAlgo.FP8)
    if kv_algo not in kv_algos:
        runs = (
            "the sm_107 default target runs an FP8 or NVFP4 KV cache"
            if nvfp4_kv_cache
            else "the sm_107 specialized targets run an FP8 KV cache"
        )
        return f"the KV cache is {kv_algo}; {runs}"
    pretrained = model_config.pretrained_config
    modules, groups = _modelopt_nvfp4_modules(
        getattr(pretrained, "text_config", None) or pretrained
    )
    for module, entry in sorted(layers.items()):
        want = modules.get(module, groups.get(module))
        if want is None:
            return f"{module} is {_precision(entry)}; the NVFP4 export stores it BF16"
        if entry != want:
            return f"{module} is {_precision(entry)}; the NVFP4 export stores it {_precision(want)}"
    missing = sorted(set(modules) - set(layers))
    if missing:
        return f"{len(missing)} module(s) the NVFP4 export quantizes are BF16 here, first {missing[:3]}"
    return None
