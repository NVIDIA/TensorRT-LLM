# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1's config classes and the modeling_v2 architecture index that reads them (no GPU)."""

import ast
import importlib
import json
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from transformers import AutoConfig

from tensorrt_llm._torch._experimental.modeling_v2 import _architecture_index
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
    KimiK31Config,
    KimiK31TextConfig,
    KimiK31VisionConfig,
    text_config_of,
)
from tensorrt_llm._torch.configs import KimiK3Config, KimiLinearConfig
from tensorrt_llm._torch.pyexecutor.config_utils import (
    is_kimi_linear,
    load_pretrained_config,
    unwrap_kimi_text_config,
)

pytestmark = pytest.mark.cpu_only

_DSA = {
    "dsa_layers": [2, 4],
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 64,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}


def _text(**extra) -> dict:
    return {
        "model_type": "kimi_linear",
        "hidden_size": 7168,
        "num_attention_heads": 96,
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-5,
        "linear_attn_config": {"kda_layers": [1, 3], "full_attn_layers": [2, 4]},
        **extra,
    }


def _composite(model_type: str, **text_extra) -> dict:
    return {
        "model_type": model_type,
        "architectures": [f"{model_type}-as-published"],
        "text_config": _text(**text_extra),
        "vision_config": {"vt_num_attention_heads": 16, "vt_hidden_size": 1152},
    }


def _load(config: dict):
    with tempfile.TemporaryDirectory() as model_dir:
        (Path(model_dir) / "config.json").write_text(json.dumps(config))
        return load_pretrained_config(model_dir)


def test_k31_checkpoint_loads_the_family_config():
    config = _load(_composite("kimi_k31", dsa_config=_DSA))
    assert type(config) is KimiK31Config
    assert config.architectures == ["KimiK31ForConditionalGeneration"]
    text = config.text_config
    assert type(text) is KimiK31TextConfig
    assert text.model_type == "kimi_linear"
    assert text.kda_fp32_gate_params
    # dsa_layers is 1-indexed; is_dsa_layer takes 0-indexed layers.
    assert [text.is_dsa_layer(i) for i in range(4)] == [False, True, False, True]
    assert (text.index_n_heads, text.index_head_dim, text.index_topk) == (64, 128, 2048)
    assert (text.index_rope_dim, text.index_topk_freq, text.index_skip_topk_offset) == (0, 1, 1)
    assert text.index_k_norm_eps == text.rms_norm_eps
    assert config.vision_config.vt_num_attention_heads == 16


def test_k31_text_only_checkpoint_keeps_the_composite():
    raw = _composite("kimi_k31", dsa_config=_DSA)
    raw["language_model_only"] = True
    config = _load(raw)
    assert type(config) is KimiK31Config
    assert config.architectures == ["KimiK31ForConditionalGeneration"]


def test_k31_rejects_an_rmsnorm_index_key_norm():
    with pytest.raises(NotImplementedError, match="index_k_norm_type"):
        _load(_composite("kimi_k31", dsa_config=dict(_DSA, index_k_norm_type="rmsnorm")))


def test_loading_registers_the_config_class_with_autoconfig():
    _load(_composite("kimi_k31", dsa_config=_DSA))
    assert type(AutoConfig.for_model("kimi_k31")) is KimiK31Config
    # A second load in the same process finds it registered already.
    assert type(_load(_composite("kimi_k31", dsa_config=_DSA))) is KimiK31Config


def test_dsa_indexer_config_is_the_text_config_with_a_dsa_config():
    config = _load(_composite("kimi_k31", dsa_config=_DSA))
    assert config.dsa_indexer_config is config.text_config
    assert KimiK31Config(text_config=_text()).dsa_indexer_config is None
    assert getattr(_load(_composite("kimi_k3")), "dsa_indexer_config", None) is None


@pytest.mark.parametrize(
    "target, name",
    [
        ("kimi_k31_pro_mxfp4__sm_10x__default", "ModelingV2KimiK31ProMxfp4Sm10xDefault"),
        ("kimi_k31_pro_mxfp4__sm_103__tep", "ModelingV2KimiK31ProMxfp4Sm103Tep"),
        ("kimi_k31_pro_mxfp4__sm_103__dep", "ModelingV2KimiK31ProMxfp4Sm103Dep"),
    ],
    ids=["general", "sm103-tp", "sm103-dp"],
)
def test_target_defaults_select_the_trtllm_moe_backend(target, name):
    import importlib

    from tensorrt_llm.llmapi.llm_args import MoeConfig, TorchLlmArgs

    package = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
    Target = getattr(importlib.import_module(f"{package}.{target}.modeling"), name)
    from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args

    args = TorchLlmArgs(model="/tmp/dummy_model", moe_config=MoeConfig(max_num_tokens=1024))
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.moe_config.backend == "TRTLLM"
    assert args.moe_config.max_num_tokens == 1024
    # A backend the user names is kept; the target then refuses it with the reason.
    args = TorchLlmArgs(model="/tmp/dummy_model", moe_config=MoeConfig(backend="CUTLASS"))
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.moe_config.backend == "CUTLASS"


def _get_model_defaults(target: str):
    """The target shell's ``get_model_defaults``, compiled from its source: importing an sm_107
    target needs its GPU kernels."""
    path = Path(sys.modules[KimiK31Config.__module__].__file__).parent / target / "modeling.py"
    (fn,) = [
        n
        for c in ast.parse(path.read_text()).body
        if isinstance(c, ast.ClassDef)
        for n in c.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_model_defaults"
    ]
    fn.decorator_list = []
    ns = {"Any": Any}
    exec(compile(ast.fix_missing_locations(ast.Module([fn], [])), str(path), "exec"), ns)
    return ns["get_model_defaults"]


@pytest.mark.parametrize("combine", [None, True, False], ids=["unset", "true", "false"])
@pytest.mark.parametrize("backend", ["CUTEDSL", "MEGAMOE_CUTEDSL"])
@pytest.mark.parametrize(
    "target",
    ["kimi_k31_pro_nvfp4__sm_107__dep16", "kimi_k31_pro_nvfp4__sm_107__cep"],
    ids=["dep16", "cep"],
)
def test_a_context_all_to_all_combines_in_fp8_unless_the_arguments_say_false(
    target, backend, combine
):
    """The arguments a server YAML hands the workers (its ``moe_config`` through the serve merge,
    pickled as the executor ships them), after the engine applies the target's model defaults:
    ``moe_config.use_low_precision_moe_combine`` is the YAML's value where it names one; where it
    leaves it out, true under ``CUTEDSL`` and the engine default under MegaMoE (dep16's generation
    role, cep's other backend), which takes no model default."""
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import (
        apply_model_defaults_to_llm_args,
        update_llm_args_with_extra_dict,
    )

    moe = {"backend": backend}
    if combine is not None:
        moe["use_low_precision_moe_combine"] = combine
    options = update_llm_args_with_extra_dict({"model": "/tmp/dummy_model"}, {"moe_config": moe})
    args = pickle.loads(pickle.dumps(TorchLlmArgs(**options)))
    apply_model_defaults_to_llm_args(args, _get_model_defaults(target)(None, args))
    want = (backend == "CUTEDSL") if combine is None else combine
    assert args.moe_config.use_low_precision_moe_combine is want


def _k31_model_config(dsa_config=_DSA, **kwargs):
    from tensorrt_llm._torch.model_config import ModelConfig

    with tempfile.TemporaryDirectory() as model_dir:
        (Path(model_dir) / "config.json").write_text(
            json.dumps(_composite("kimi_k31", dsa_config=dsa_config))
        )
        return ModelConfig.from_pretrained(model_dir, trust_remote_code=True, **kwargs)


def test_k31_sparse_config_keeps_the_user_fields():
    """The DSA config built for K3.1 keeps every user-set field and fills in the checkpoint's geometry."""
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    requested = DeepSeekSparseAttentionConfig(
        enable_heuristic_topk=True,
        use_self_sampling_topk=True,
    )
    sparse = _k31_model_config(sparse_attention_config=requested).sparse_attention_config
    assert sparse.enable_heuristic_topk
    assert (sparse.index_n_heads, sparse.index_topk) == (64, 2048)


@pytest.mark.parametrize("indexer_k_dtype", ["fp4", "fp8", None], ids=["fp4", "fp8", "left-out"])
def test_k31_sparse_config_takes_the_fields_of_a_server_yaml(indexer_k_dtype):
    """The sparse_attention_config block of a server YAML reaches the DSA config built for K3.1.
    A block that leaves ``indexer_k_dtype`` out gets FP8 index keys as well, the loader's default,
    but the arguments do not name them: that difference is all the Rubin routing reads
    (``routing._names_indexer_k_dtype``) to run FP8 index keys only where asked."""
    from pydantic import TypeAdapter

    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig

    block = dict(
        algorithm="dsa",
        enable_heuristic_topk=True,
        gvr_deterministic_topk=True,
        use_self_sampling_topk=True,
    )
    if indexer_k_dtype is not None:
        block["indexer_k_dtype"] = indexer_k_dtype
    requested = TypeAdapter(SparseAttentionConfig).validate_python(block)
    assert ("indexer_k_dtype" in requested.model_fields_set) is (indexer_k_dtype is not None)
    sparse = _k31_model_config(sparse_attention_config=requested).sparse_attention_config
    assert sparse.indexer_k_dtype == (indexer_k_dtype or "fp8")
    assert sparse.enable_heuristic_topk and sparse.use_self_sampling_topk
    assert (sparse.index_n_heads, sparse.index_head_dim, sparse.index_topk) == (64, 128, 2048)


def test_k31_indexer_scores_in_fp8_by_default():
    # The K3.1 reference scores the indexer in FP8 like DeepSeek-V3.2; the checkpoint's
    # indexer_use_fp8 flag does not change that.
    sparse = _k31_model_config(dict(_DSA, indexer_use_fp8=False)).sparse_attention_config
    assert sparse.indexer_k_dtype == "fp8"
    assert sparse.to_sparse_params().use_cute_dsl_paged_mqa_logits


def test_k3_checkpoint_keeps_the_built_in_config():
    config = _load(_composite("kimi_k3"))
    assert type(config) is KimiK3Config
    assert config.architectures == ["KimiK3ForConditionalGeneration"]
    assert type(config.text_config) is KimiLinearConfig
    assert getattr(config.text_config, "dsa_config", None) is None


@pytest.mark.parametrize("model_type", ["kimi_k3", "kimi_k31"])
def test_kimi_predicates_read_the_composite_by_its_text_config(model_type):
    text_extra = {"dsa_config": _DSA} if model_type == "kimi_k31" else {}
    config = _load(_composite(model_type, **text_extra))
    assert is_kimi_linear(config)
    assert unwrap_kimi_text_config(config) is config.text_config
    assert is_kimi_linear(config.text_config)
    assert unwrap_kimi_text_config(config.text_config) is config.text_config


def test_predicates_ignore_other_composites():
    other = type("OtherComposite", (), {"model_type": "other", "text_config": object()})()
    assert not is_kimi_linear(other)
    assert unwrap_kimi_text_config(other) is other


def test_input_processor_comes_from_the_architecture_index():
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    k31 = _load(_composite("kimi_k31", dsa_config=_DSA))
    processor = _architecture_index.input_processor_class(k31)
    assert processor.__name__ == "K31InputProcessor"
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder("kimi_k31", "image") == (
        "<|kimi_image_placeholder|>"
    )
    # Kimi K3 has a built-in model class; the index names nothing for it.
    assert _architecture_index.input_processor_class(_load(_composite("kimi_k3"))) is None


def test_index_has_nothing_for_other_model_types():
    assert _architecture_index.architecture_module(None) is None
    assert _architecture_index.architecture_module("kimi_k3") is None
    assert _architecture_index.load_config({"model_type": "llama"}) is None


def test_loading_another_model_imports_no_family_module():
    code = (
        "import json, sys, tempfile, pathlib\n"
        "from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config\n"
        "d = tempfile.mkdtemp()\n"
        "pathlib.Path(d, 'config.json').write_text(json.dumps("
        "{'model_type': 'kimi_linear', 'hidden_size': 64, 'num_attention_heads': 4}))\n"
        "load_pretrained_config(d)\n"
        "family = 'tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31'\n"
        "print(sorted(m for m in sys.modules if m.startswith(family)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"


# The family config classes against the engine-subclass build of the same configs: the same model_type strings, the
# same attributes in the same order, the same serialized config.

_K31_HIDDEN = 7168
_K31_DENSE = 33792
_K31_KDA_HEADS = 96


class _EngineK31TextConfig(KimiLinearConfig):
    """The engine-subclass build of the K3.1 text config: the engine's Kimi K3 text config plus K3.1's fields."""

    def __init__(self, dsa_config=None, kda_fp32_gate_params=True, **kwargs):
        self.dsa_config = dsa_config
        self.kda_fp32_gate_params = kda_fp32_gate_params
        super().__init__(**kwargs)


class _EngineK31Config(KimiK3Config):
    """The engine-subclass build of the K3.1 composite config, on the engine's Kimi K3 composite config."""

    model_type = "kimi_k31"

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = _EngineK31TextConfig(**text_config)
        super().__init__(text_config=text_config, **kwargs)


_LAYERS = 93
_FULL_ATTN_LAYERS = [*range(4, _LAYERS, 4), _LAYERS]


def _k31_config_dict(**text_overrides) -> dict:
    """A K3.1 config.json: the published geometry, as nested dicts."""
    text = {
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"],
        "vocab_size": 163840,
        "hidden_size": _K31_HIDDEN,
        "intermediate_size": _K31_DENSE,
        "moe_intermediate_size": 3072,
        "num_hidden_layers": _LAYERS,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 12,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "first_k_dense_replace": 1,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "latent_moe_use_norm": True,
        "max_position_embeddings": 1048576,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "routed_expert_hidden_size": 3584,
        "rms_norm_eps": 1e-5,
        "dtype": "bfloat16",
        "linear_attn_config": {
            "full_attn_layers": _FULL_ATTN_LAYERS,
            "kda_layers": [i for i in range(1, _LAYERS + 1) if i not in _FULL_ATTN_LAYERS],
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "num_heads": _K31_KDA_HEADS,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
        },
        "dsa_config": {
            "dsa_layers": _FULL_ATTN_LAYERS,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_num_heads": 32,
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "indexer_use_fp8": False,
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
        },
        **text_overrides,
    }
    vision = {
        "_attn_implementation": "flash_attention_2",
        "merge_kernel_size": [2, 2],
        "mm_hidden_size": 1024,
        "qkv_hidden_size": 1536,
        "text_hidden_size": _K31_HIDDEN,
        "vt_hidden_size": 1024,
        "vt_intermediate_size": 4096,
        "vt_num_attention_heads": 12,
        "vt_num_hidden_layers": 27,
    }
    return {
        "model_type": "kimi_k31",
        "architectures": ["KimiK31ForConditionalGeneration"],
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "pad_token_id": 163839,
        "media_placeholder_token_id": 163605,
        "tie_word_embeddings": False,
        "text_config": text,
        "vision_config": vision,
    }


@pytest.mark.parametrize("kda_fp32_gate_params", [None, False])
def test_family_config_is_the_engine_built_config(kda_fp32_gate_params):
    overrides = {} if kda_fp32_gate_params is None else {"kda_fp32_gate_params": False}
    raw = _k31_config_dict(**overrides)
    got = KimiK31Config.from_dict(raw)
    expected = _EngineK31Config.from_dict(raw)

    assert type(got.text_config) is KimiK31TextConfig
    assert type(got.vision_config) is KimiK31VisionConfig
    # The model_type strings the engine dispatches on.
    assert [c.model_type for c in (got, got.text_config, got.vision_config)] == [
        c.model_type for c in (expected, expected.text_config, expected.vision_config)
    ]
    assert (got.text_config.model_type, got.vision_config.model_type) == (
        "kimi_linear",
        "kimi_k3_vision",
    )
    # Every attribute, with the same value, set in the same order.
    for g, e in [
        (got, expected),
        (got.text_config, expected.text_config),
        (got.vision_config, expected.vision_config),
    ]:
        g_vars = {k: v for k, v in vars(g).items() if k not in ("text_config", "vision_config")}
        e_vars = {k: v for k, v in vars(e).items() if k not in ("text_config", "vision_config")}
        assert list(g_vars.items()) == list(e_vars.items())
    assert got.to_dict() == expected.to_dict()
    assert got.to_json_string() == expected.to_json_string()
    assert (
        got.quantization_config
        == expected.quantization_config
        == raw["text_config"]["quantization_config"]
    )

    text = got.text_config
    assert (text.index_n_heads, text.index_head_dim, text.index_topk) == (32, 128, 2048)
    assert (text.index_rope_dim, text.index_k_norm_eps) == (0, 1e-5)
    assert (text.index_topk_freq, text.index_skip_topk_offset) == (1, 1)
    assert [text.is_dsa_layer(i) for i in range(_LAYERS)] == [
        (i + 1) in _FULL_ATTN_LAYERS for i in range(_LAYERS)
    ]
    assert got.dsa_indexer_config is text


def test_text_config_defaults_are_the_engine_defaults():
    got, expected = KimiK31TextConfig(), _EngineK31TextConfig()
    assert list(vars(got).items()) == list(vars(expected).items())
    assert KimiK31Config().to_dict() == _EngineK31Config().to_dict()


def test_text_config_of_unwraps_a_composite_config():
    config = KimiK31Config.from_dict(_k31_config_dict())
    assert text_config_of(config) is config.text_config
    assert text_config_of(config.text_config) is config.text_config
    # A composite of another model_type is read by its text config too, when it has no KDA layout of its own.
    other = SimpleNamespace(model_type="other", text_config=config.text_config)
    assert text_config_of(other) is config.text_config


# --- the ModelOpt NVFP4 export's quantization config ---

# The export's per-module precision on the 4-layer text config above (KDA layers 0 and 2, MLA
# layers 1 and 3, a dense layer 0 and MoE layers 1-3), spelled as the export spells it: every module
# under three prefixes, MoE modules also under ``mlp.``, and the fused quantization groups beside
# their parts.
_NVFP4 = {"quant_algo": "NVFP4", "group_size": 16}
_MXFP8 = {"quant_algo": "MXFP8"}


def _modelopt_modules() -> dict:
    modules = {}
    for i in (1, 2, 3):
        for moe in ("block_sparse_moe", "mlp"):
            modules[f"layers.{i}.{moe}.experts"] = _NVFP4
            for proj in ("gate_proj", "up_proj", "down_proj", "gate_up_proj"):
                modules[f"layers.{i}.{moe}.shared_experts.{proj}"] = _MXFP8
    for i in (0, 2):
        for proj in ("q", "k", "v", "g", "b", "f_a", "o", "fused_qkvg", "in_proj_qkvgfab"):
            name = proj if proj == "in_proj_qkvgfab" else f"{proj}_proj"
            modules[f"layers.{i}.self_attn.{name}"] = _MXFP8
    return modules


def _modelopt_quant_config(modules: dict, kv_cache_quant_algo=None) -> dict:
    return {
        "producer": {"name": "modelopt", "version": "0.49.0"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": kv_cache_quant_algo,
            "group_size": 16,
            "quantized_layers": {
                prefix + name: cfg
                for name, cfg in modules.items()
                for prefix in ("language_model.model.", "model.", "")
            },
            "exclude_modules": [
                "*block_sparse_moe.gate*",
                "*layernorm*",
                "lm_head",
                "vision_tower*",
            ],
        },
    }


def _modelopt_model_config(modules: dict, kv_cache_quant_algo=None):
    from tensorrt_llm._torch.model_config import ModelConfig

    with tempfile.TemporaryDirectory() as model_dir:
        text = dict(num_experts=8, num_shared_experts=2, first_k_dense_replace=1, dsa_config=_DSA)
        (Path(model_dir) / "config.json").write_text(json.dumps(_composite("kimi_k31", **text)))
        (Path(model_dir) / "hf_quant_config.json").write_text(
            json.dumps(_modelopt_quant_config(modules, kv_cache_quant_algo))
        )
        return ModelConfig.from_pretrained(model_dir, trust_remote_code=True)


def test_modelopt_export_parses_per_module():
    """The engine's parse of the export folds into one entry per module, and is the export the
    specialized targets read."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
        modelopt_layer_quant,
        modelopt_nvfp4_mismatch,
    )
    from tensorrt_llm.quantization.mode import QuantAlgo

    model_config = _modelopt_model_config(_modelopt_modules())
    assert model_config.quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    layers = modelopt_layer_quant(model_config)
    nvfp4, mxfp8 = (QuantAlgo.NVFP4, 16), (QuantAlgo.MXFP8, None)
    assert layers["layers.1.block_sparse_moe.experts"] == nvfp4
    assert layers["layers.3.block_sparse_moe.shared_experts.down_proj"] == mxfp8
    assert layers["layers.2.self_attn.f_a_proj"] == mxfp8
    # Three prefixes and the mlp. alias fold: 3 MoE layers x 5 modules, 2 KDA layers x 9.
    assert len(layers) == 3 * 5 + 2 * 9
    assert not any(module.startswith("layers.1.self_attn") for module in layers)
    assert modelopt_nvfp4_mismatch(model_config) is None


def test_modelopt_export_takes_the_servers_fp8_kv_cache():
    """A server with ``kv_cache_config.dtype: fp8`` has FP8 in ``quant_config`` when the weights
    load."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
        modelopt_nvfp4_mismatch,
    )

    assert modelopt_nvfp4_mismatch(_modelopt_model_config(_modelopt_modules(), "FP8")) is None


def test_modelopt_export_takes_the_default_targets_nvfp4_kv_cache():
    """The default target runs an FP8 or NVFP4 KV cache on the export (``nvfp4_kv_cache``): the
    export stores no KV scales, so neither changes a weight read. Another KV cache is refused."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
        modelopt_nvfp4_mismatch,
    )

    for kv in (None, "FP8", "NVFP4"):
        model_config = _modelopt_model_config(_modelopt_modules(), kv)
        assert modelopt_nvfp4_mismatch(model_config, nvfp4_kv_cache=True) is None, kv
    model_config = _modelopt_model_config(_modelopt_modules(), "INT8")
    assert modelopt_nvfp4_mismatch(model_config, nvfp4_kv_cache=True) == (
        "the KV cache is INT8; the sm_107 default target runs an FP8 or NVFP4 KV cache"
    )


_RUBIN_TARGETS = (
    "kimi_k31_pro_nvfp4__sm_107__cep",
    "kimi_k31_pro_nvfp4__sm_107__dep16",
    "kimi_k31_pro_nvfp4__sm_107__helix8",
    "kimi_k31_pro_nvfp4__sm_107__tep8",
)


@pytest.mark.parametrize("kv", [None, "FP8", "NVFP4"], ids=["no-kv-quant", "fp8", "nvfp4"])
@pytest.mark.parametrize("target", _RUBIN_TARGETS)
def test_each_weight_table_takes_the_kv_cache_its_target_runs(target, kv):
    """The loader writes the server's KV cache into the config before the weights load. The default
    target's weight table takes the export under an FP8 or NVFP4 cache; each specialized target's
    takes FP8 and refuses NVFP4."""
    weights = importlib.import_module(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.{target}.weights"
    )
    model_config = _modelopt_model_config(_modelopt_modules(), kv)
    if kv == "NVFP4":
        refused = "the KV cache is NVFP4; the sm_107 specialized targets run an FP8 KV cache"
        with pytest.raises(ValueError, match=re.escape(refused)):
            weights.check_checkpoint(model_config)
    else:
        weights.check_checkpoint(model_config)


def _drop(modules: dict, module: str) -> dict:
    return {k: v for k, v in modules.items() if k != module}


@pytest.mark.parametrize(
    "modules, kv, reason",
    [
        pytest.param(
            {**_modelopt_modules(), "layers.1.self_attn.q_b_proj": _MXFP8},
            None,
            r"layers\.1\.self_attn\.q_b_proj is MXFP8; the NVFP4 export stores it BF16",
            id="mla-projection-quantized",
        ),
        pytest.param(
            _drop(_modelopt_modules(), "layers.2.self_attn.o_proj"),
            None,
            r"1 module\(s\) the NVFP4 export quantizes are BF16 here, first \['layers\.2\.self_attn\.o_proj'\]",
            id="kda-projection-bf16",
        ),
        pytest.param(
            {
                **_modelopt_modules(),
                "layers.3.block_sparse_moe.experts": _MXFP8,
                "layers.3.mlp.experts": _MXFP8,
            },
            None,
            r"layers\.3\.block_sparse_moe\.experts is MXFP8; the NVFP4 export stores it NVFP4 \(group 16\)",
            id="experts-mxfp8",
        ),
        pytest.param(
            {**_modelopt_modules(), "layers.3.mlp.experts": _MXFP8},
            None,
            r"give layers\.3\.block_sparse_moe\.experts as .* and as ",
            id="spellings-disagree",
        ),
        pytest.param(
            _modelopt_modules(),
            "NVFP4",
            r"the KV cache is NVFP4; the sm_107 specialized targets run an FP8 KV cache",
            id="kv-cache-not-fp8",
        ),
    ],
)
def test_modelopt_mismatch_names_the_module(modules, kv, reason):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
        modelopt_nvfp4_mismatch,
    )

    assert re.search(reason, modelopt_nvfp4_mismatch(_modelopt_model_config(modules, kv)))


def test_the_mxfp4_checkpoint_is_not_the_modelopt_export():
    """Moonshot's MXFP4 checkpoint (the default target's) parses to no ModelOpt layers."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import (
        modelopt_layer_quant,
        modelopt_nvfp4_mismatch,
    )
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    model_config = ModelConfig(
        pretrained_config=KimiK31Config(text_config=_text(dsa_config=_DSA)),
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
    )
    assert modelopt_layer_quant(model_config) is None
    assert "not a ModelOpt MIXED_PRECISION export" in modelopt_nvfp4_mismatch(model_config)
