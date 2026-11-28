# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1's config as the targets read it: model defaults, the family's input processor declaration,
the ModelOpt NVFP4 export's precision per module that the sm_107 targets read, and the config-time
DFlash buffer check of an aggregated server (no GPU). The config classes are tested in
``tests/unittest/_torch/test_custom_config_registration.py``."""

import importlib
import json
import re
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

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


def _composite(model_type: str, architecture: str | None = None, **text_extra) -> dict:
    return {
        "model_type": model_type,
        "architectures": [architecture or f"{model_type}-as-published"],
        "text_config": _text(**text_extra),
        "vision_config": {"vt_num_attention_heads": 16, "vt_hidden_size": 1152},
    }


def _load(config: dict):
    with tempfile.TemporaryDirectory() as model_dir:
        (Path(model_dir) / "config.json").write_text(json.dumps(config))
        return load_pretrained_config(model_dir)


def test_target_defaults_admit_the_shortest_uncached_context_first():
    """The general target's model defaults."""
    from tensorrt_llm.llmapi.llm_args import (
        CapacitySchedulerPolicy,
        ContextSchedulingPolicy,
        MoeConfig,
        SchedulerConfig,
        TorchLlmArgs,
    )
    from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args

    package = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"
    Target = importlib.import_module(
        f"{package}.kimi_k31_pro_mxfp4__sm_10x__default.modeling"
    ).ModelingV2KimiK31ProMxfp4Sm10xDefault

    args = TorchLlmArgs(model="/tmp/dummy_model")
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.scheduler_config.context_scheduling_policy == (
        ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST
    )
    # Other scheduler fields the user sets keep the default policy beside them.
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        scheduler_config=SchedulerConfig(
            capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
        ),
    )
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.scheduler_config.capacity_scheduler_policy == (
        CapacitySchedulerPolicy.MAX_UTILIZATION
    )
    assert args.scheduler_config.context_scheduling_policy == (
        ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST
    )
    # A policy the user names is kept, and the merged arguments are valid.
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        enable_chunked_prefill=True,
        scheduler_config=SchedulerConfig(context_scheduling_policy=ContextSchedulingPolicy.FCFS),
    )
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.scheduler_config.context_scheduling_policy == ContextSchedulingPolicy.FCFS
    assert args.scheduler_config.context_starvation_iterations is None
    # A starvation bound the user sets beside the default policy is kept, and valid with it.
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        enable_chunked_prefill=True,
        scheduler_config=SchedulerConfig(
            context_scheduling_policy=ContextSchedulingPolicy.SHORTEST_UNCACHED_FIRST,
            context_starvation_iterations=25,
        ),
    )
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.scheduler_config.context_starvation_iterations == 25
    # The MoE backend defaults to TRTLLM beside the user's other MoE fields; a backend the user
    # names is kept, and the target then refuses it with the reason.
    args = TorchLlmArgs(model="/tmp/dummy_model", moe_config=MoeConfig(max_num_tokens=1024))
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.moe_config.backend == "TRTLLM"
    assert args.moe_config.max_num_tokens == 1024
    args = TorchLlmArgs(model="/tmp/dummy_model", moe_config=MoeConfig(backend="CUTLASS"))
    apply_model_defaults_to_llm_args(args, Target.get_model_defaults(args))
    assert args.moe_config.backend == "CUTLASS"


def test_input_processor_comes_from_the_family_declaration():
    from tensorrt_llm._torch.modeling_v2._router_index import modeling_v2_input_processor
    from tensorrt_llm.inputs.registry import MULTIMODAL_PLACEHOLDER_REGISTRY

    # Every Kimi K3.1 checkpoint states this architecture; the config loads through the registry.
    k31 = _load(_composite("kimi_k31", "KimiK31ForConditionalGeneration", dsa_config=_DSA))
    processor = modeling_v2_input_processor(k31)
    assert processor.__name__ == "K31InputProcessor"
    assert MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder("kimi_k31", "image") == (
        "<|kimi_image_placeholder|>"
    )
    # Kimi K3 has a built-in model class and declares no modeling_v2 input processor.
    assert modeling_v2_input_processor(_load(_composite("kimi_k3"))) is None
    assert modeling_v2_input_processor(SimpleNamespace(architectures=None)) is None


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
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.config import (
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


def test_the_weight_table_takes_the_kv_cache_its_target_runs():
    """The loader writes the server's KV cache into the config before the weights load. The
    specialized tep8 target's weight table refuses the export under an NVFP4 cache."""
    weights = importlib.import_module(
        "tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8.weights"
    )
    model_config = _modelopt_model_config(_modelopt_modules(), "NVFP4")
    refused = "the KV cache is NVFP4; the NVFP4 targets run an FP8 KV cache"
    with pytest.raises(ValueError, match=re.escape(refused)):
        weights.check_checkpoint(model_config)


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
            _modelopt_modules(),
            "INT8",
            r"the KV cache is INT8; the NVFP4 targets run an FP8 KV cache",
            id="kv-cache-int8",
        ),
    ],
)
def test_modelopt_mismatch_names_the_module(modules, kv, reason):
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.config import modelopt_nvfp4_mismatch

    model_config = _modelopt_model_config(modules, kv)
    assert re.search(reason, modelopt_nvfp4_mismatch(model_config))


def test_the_mxfp4_checkpoint_is_not_the_modelopt_export():
    """Moonshot's MXFP4 checkpoint (the default target's) parses to no ModelOpt layers."""
    from tensorrt_llm._torch.configs import KimiK31Config
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.config import (
        modelopt_layer_quant,
        modelopt_nvfp4_mismatch,
    )
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    model_config = ModelConfig(
        pretrained_config=KimiK31Config(text_config=_text(dsa_config=_DSA)),
        quant_config=QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8),
    )
    assert modelopt_layer_quant(model_config) is None
    assert "not a ModelOpt MIXED_PRECISION export" in modelopt_nvfp4_mismatch(model_config)


# The DFlash2 MLA drafter's geometry: five sliding-window MLA layers, one 512 + 64 latent each.
_DRAFT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "block_size": 5,
    "dflash_config": {"attention_mode": "mla", "block_size": 5},
    "dtype": "bfloat16",
    "head_dim": 64,
    "hidden_size": 7168,
    "kv_lora_rank": 512,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576,
    "num_attention_heads": 96,
    "num_hidden_layers": 5,
    "num_key_value_heads": 96,
    "qk_rope_head_dim": 64,
    "sliding_window": 1024,
}


def _aggregated_dflash2_args(tmp_path, skip_budget_check: bool, cache_transceiver_config=None):
    """A TEP8 aggregated K3.1 server with DFlash2 at a 1M-token context."""
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, TorchLlmArgs

    target = tmp_path / "target"
    target.mkdir(exist_ok=True)
    (target / "config.json").write_text(
        json.dumps({"architectures": ["KimiK31ForConditionalGeneration"]})
    )
    drafter = tmp_path / "drafter"
    drafter.mkdir(exist_ok=True)
    (drafter / "config.json").write_text(json.dumps(_DRAFT_CONFIG))
    # model_construct skips model-path validation; only the DFlash budget check is under test.
    return TorchLlmArgs.model_construct(
        model=str(target),
        modeling_v2="require",
        speculative_config=DFlashDecodingConfig(
            max_draft_len=4,
            speculative_model=str(drafter),
            skip_ctx_buffer_budget_check=skip_budget_check,
        ),
        max_batch_size=8,
        max_num_tokens=16384,
        max_seq_len=1048576,
        tensor_parallel_size=8,
        cache_transceiver_config=cache_transceiver_config,
    )


def test_aggregated_dflash2_needs_the_budget_flag(tmp_path):
    """An aggregated DFlash2 server is sized by the engine's dense drafter-context estimate, which
    refuses a K3.1 shape with a 1M-token context; ``skip_ctx_buffer_budget_check`` lets such a
    config through. The drafter arena the worker then allocates is checked in
    ``test_modeling_v2_modeling_dflash_arena.py``."""
    budget = int(84.86 * 1024**3)  # the headroom per GPU the refused serve was estimated to have
    # 2 x 9 slots x 5 layers x (1048576 + 5) x 12 kv heads per rank x 64 x 2 bytes = 135.00 GiB.
    with pytest.raises(ValueError, match=r"need 135\.00 GiB per GPU"):
        _aggregated_dflash2_args(tmp_path, False)._validate_dflash_ctx_budget(
            memory_budget_bytes=budget, uses_paged_context=False
        )
    _aggregated_dflash2_args(tmp_path, True)._validate_dflash_ctx_budget(
        memory_budget_bytes=budget, uses_paged_context=False
    )


@pytest.mark.parametrize(
    "target,backend,cp,manager_v2,shared,paged",
    [
        ("default", None, None, True, False, False),
        ("default", "NIXL", None, True, False, True),
        ("default", None, "HELIX", True, False, True),
        ("default", None, "ULYSSES", True, False, False),
        ("default", "NIXL", None, False, False, False),
        ("default", "NIXL", None, True, True, False),
        ("rubin", None, None, True, False, True),
        ("stock", "NIXL", "HELIX", True, False, False),
        ("off", "NIXL", "HELIX", True, False, False),
    ],
)
def test_dflash_budget_uses_resolved_target(
    tmp_path, monkeypatch, target, backend, cp, manager_v2, shared, paged
):
    from unittest.mock import Mock

    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__tep8.modeling import (
        ModelingV2KimiK31ProNvfp4Sm107Tep8,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.runtime import K31_RUNTIME_HOOKS
    from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, CpConfig, TorchLlmArgs
    from tensorrt_llm.mapping import Mapping

    targets = dict(
        default=SimpleNamespace(runtime_hooks=K31_RUNTIME_HOOKS),
        rubin=ModelingV2KimiK31ProNvfp4Sm107Tep8,
        stock=object,
        off=None,
    )
    args = _aggregated_dflash2_args(tmp_path, False, CacheTransceiverConfig(backend=backend))
    args.kv_cache_config.use_kv_cache_manager_v2 = manager_v2
    args.speculative_config._use_shared_kv_cache = shared
    args.context_parallel_size = 8 if cp else 1
    args.tensor_parallel_size = 1
    args.cp_config = CpConfig(cp_type=cp) if cp else None
    mapping = Mapping(
        world_size=args.context_parallel_size,
        cp_size=args.context_parallel_size,
        cp_config={"cp_type": cp} if cp else None,
    )
    budget = Mock()
    monkeypatch.setattr(TorchLlmArgs, "_validate_dflash_ctx_budget", budget)
    spec = args.speculative_config
    config = SimpleNamespace(modeling_v2_target=targets[target], mapping=mapping, spec_config=spec)
    # The loader can receive its speculative configuration independently of the API args.
    args.speculative_config = None
    args._validate_dflash_target(config)
    budget.assert_called_once_with(uses_paged_context=paged, spec_config=spec)


@pytest.mark.parametrize(
    "family,shared,explicit,tp,attention_dp,supported",
    [
        (True, False, True, 1, False, True),
        (True, True, True, 1, False, False),
        (False, False, True, 1, False, False),
        (False, False, False, 1, False, False),
        (True, False, True, 2, False, False),
        (True, False, True, 1, True, False),
    ],
)
def test_helix_sampling_requires_target_opt_in(
    tmp_path, family, shared, explicit, tp, attention_dp, supported
):
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix8.modeling import (
        ModelingV2KimiK31ProNvfp4Sm107Helix8,
    )
    from tensorrt_llm.llmapi.llm_args import CpConfig

    args = _aggregated_dflash2_args(tmp_path, True)
    args.context_parallel_size = 8
    args.tensor_parallel_size = tp
    args.enable_attention_dp = attention_dp
    args.cp_config = CpConfig(cp_type="HELIX")
    spec = args.speculative_config
    spec._use_shared_kv_cache = shared
    spec.use_rejection_sampling = True
    if not explicit:
        # Exercise a value inherited from model defaults.
        spec.model_fields_set.discard("use_rejection_sampling")
    mapping = SimpleNamespace(has_cp_helix=lambda: True, helix_repurposed_cp_size=1)
    target = ModelingV2KimiK31ProNvfp4Sm107Helix8 if family else None
    with (
        pytest.raises(ValueError, match="context parallelism")
        if explicit and not supported
        else nullcontext()
    ):
        args._validate_dflash_target(
            SimpleNamespace(modeling_v2_target=target, mapping=mapping, spec_config=spec)
        )
    if supported or not explicit:
        assert spec.use_rejection_sampling is supported
