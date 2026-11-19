# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi text config the K3.1 targets read.

A K3.1 checkpoint (``model_type: kimi_k31``) has Kimi K3's composite layout: a ``text_config`` (model_type
``kimi_linear``) and a MoonViT ``vision_config``. Its config class is Kimi K3's under K3.1's model_type
(``tensorrt_llm._torch.configs.kimi_k31``); the targets read the configs by attribute and import none of them.

The checkpoints' DSA indexer (``DSA_INDEXER``) is the sparse attention the targets declare in their model defaults
(``dsa_model_defaults``).

It also reads the ModelOpt NVFP4 export's precision per module (``modelopt_layer_quant``) and says whether a
checkpoint is that export as the NVFP4 targets read it (``modelopt_nvfp4_mismatch``): routing and the targets' weight
tables ask the same question.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple

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


#: The DSA indexer of the K3.1 checkpoints, as the text config's ``dsa_config`` states it: 32 index heads of 128
#: channels that select the top 2048 keys per query. Routing refuses a checkpoint that states another.
DSA_INDEXER = MappingProxyType({"index_num_heads": 32, "index_head_dim": 128, "index_topk": 2048})


def dsa_model_defaults() -> Dict[str, Any]:
    """The sparse attention of the targets' ``get_model_defaults``: DSA with :data:`DSA_INDEXER`, under the
    sparse-attention config's field names. The fields the arguments set win over these."""
    return {
        "sparse_attention_config": {
            "algorithm": "dsa",
            "index_n_heads": DSA_INDEXER["index_num_heads"],
            "index_head_dim": DSA_INDEXER["index_head_dim"],
            "index_topk": DSA_INDEXER["index_topk"],
        }
    }


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


def _checkpoint_quantization(pretrained_config: Any) -> dict:
    """The checkpoint's ``quantization_config`` (the composite config's or its text config's) as a dict."""
    quant = getattr(pretrained_config, "quantization_config", None)
    if quant is None:
        quant = getattr(
            getattr(pretrained_config, "text_config", None), "quantization_config", None
        )
    if not isinstance(quant, dict):
        quant = getattr(quant, "to_dict", lambda: {})()
    return quant or {}


def mxfp4_experts(pretrained_config: Any) -> bool:
    """Whether the checkpoint is Moonshot's MXFP4 export (routed experts ``mxfp4-pack-quantized``)."""
    return _checkpoint_quantization(pretrained_config).get("format") == "mxfp4-pack-quantized"


def routes_to_nvfp4_targets(pretrained_config: Any, quant_algo: Any) -> bool:
    """Whether routing sends the checkpoint to the NVFP4 targets: a ModelOpt mixed-precision export
    (``quant_algo``, the engine's reading of its quantization, is MIXED_PRECISION) without Moonshot's MXFP4 experts.
    Those targets then check the export's per-layer quantization (``modelopt_nvfp4_mismatch``) and refuse any other;
    routing and the input processor (its text-only refusal) both decide with this function."""
    algo = str(getattr(quant_algo, "value", quant_algo) or "").upper()
    return algo == QuantAlgo.MIXED_PRECISION.value and not mxfp4_experts(pretrained_config)


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


def modelopt_nvfp4_mismatch(model_config: Any) -> Optional[str]:
    """Why the checkpoint is not the ModelOpt NVFP4 export as the NVFP4 targets read it, or None.

    That export: MIXED_PRECISION; the routed experts NVFP4 with a scale per 16 values; the shared experts and the KDA
    projections (q, k, v, g, b, f_a, o) MXFP8; everything else BF16; no KV-cache quantization, so no KV scales. The
    engine writes the server's ``kv_cache_config`` dtype into ``quant_config`` before the weights load, so the KV cache
    a target runs reads here: FP8, which changes no weight read; any other KV cache is refused.
    """
    try:
        layers = modelopt_layer_quant(model_config)
    except ValueError as e:
        return str(e)
    if layers is None:
        algo = _field(getattr(model_config, "quant_config", None), "quant_algo")
        return f"the checkpoint is not a ModelOpt MIXED_PRECISION export (quant_algo {algo})"
    kv_algo = _field(model_config.quant_config, "kv_cache_quant_algo")
    if kv_algo not in (None, QuantAlgo.FP8):
        return f"the KV cache is {kv_algo}; the NVFP4 targets run an FP8 KV cache"
    pretrained = model_config.pretrained_config
    text = getattr(pretrained, "text_config", None) or pretrained
    # The export's module list is read off the text config's layer layout.
    kinds = getattr(text, "linear_attn_config", None)
    if not isinstance(kinds, dict) or not isinstance(kinds.get("kda_layers"), (list, tuple)):
        return (
            "the text config has no linear_attn_config kda_layers, the KDA layers whose projections "
            "the NVFP4 export quantizes; serve the export's own config.json"
        )
    if not isinstance(getattr(text, "num_hidden_layers", None), int) or (
        getattr(text, "num_experts", None) is not None
        and not isinstance(getattr(text, "first_k_dense_replace", None), int)
    ):
        return (
            "the text config has no num_hidden_layers or first_k_dense_replace, which place the "
            "modules the NVFP4 export quantizes; serve the export's own config.json"
        )
    modules, groups = _modelopt_nvfp4_modules(text)
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
