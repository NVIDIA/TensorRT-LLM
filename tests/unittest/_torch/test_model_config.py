# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import types
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.model_config import (
    _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT,
    ModelConfig,
    MoESelection,
)
from tensorrt_llm._torch.moe.fused_moe.moe_resolution import moe_impl_request
from tensorrt_llm._torch.pyexecutor.model_loader import (
    validate_and_set_kv_cache_quant,
    validate_encoder_decoder_kv_cache_config,
)
from tensorrt_llm.llmapi.llm_args import MiniMaxM3SparseAttentionConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig
from tensorrt_llm.quantization.modelopt_config import read_modelopt_quant_config

pytestmark = pytest.mark.cpu_only


def make_pretrained_config(
    *,
    num_attention_heads: int = 16,
    num_key_value_heads=8,
    head_dim: int | None = None,
    num_hidden_layers: int = 1,
    vocab_size: int = 3000,
    is_encoder_decoder: bool = False,
):
    # A minimal config object that provides the attributes used by
    # ModelConfig.get_bindings_model_config().
    hidden_size = head_dim * num_attention_heads
    intermediate_size = hidden_size * 4

    return types.SimpleNamespace(
        architectures=["DummyArchitecture"],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        torch_dtype=torch.float16,
        is_encoder_decoder=is_encoder_decoder,
    )


@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(8, id="kv_heads_scalar"),
        pytest.param([8, 20], id="kv_heads_per_layer_varied"),
    ],
)
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # Same tp/cp sizes, but different ways of setting attention TP:
        # - No explicit attn_tp_size: Mapping infers it.
        # - Explicit attn_tp_size: Mapping uses the provided value.
        dict(world_size=8, tp_size=4, cp_size=2),
        dict(world_size=4, tp_size=2, cp_size=2, attn_tp_size=4),
    ],
)
def test_get_bindings_model_config_attention_dp_attn_tp_override(
    enable_attention_dp, mapping_kwargs, num_key_value_heads
):
    mapping = Mapping(enable_attention_dp=enable_attention_dp, **mapping_kwargs)
    cfg = make_pretrained_config(
        # Keep values consistent:
        # hidden_size = num_attention_heads * head_dim.
        num_attention_heads=16,
        head_dim=4,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=2,
    )
    model_config = ModelConfig(pretrained_config=cfg, mapping=mapping)

    tokens_per_block = 32
    bindings_cfg = model_config.get_bindings_model_config(tokens_per_block=tokens_per_block)

    # bindings hidden_size is sharded by attn_tp_size and attn_cp_size.
    attn_tp_size = mapping.attn_tp_size if not mapping.enable_attention_dp else 1
    attn_cp_size = mapping.attn_cp_size

    def ceil_div(a, b):
        return (a + b - 1) // b

    assert bindings_cfg.num_heads == ceil_div(cfg.num_attention_heads, attn_tp_size * attn_cp_size)
    # bindings hidden_size is sharded by attn_tp_size.
    assert bindings_cfg.hidden_size == ceil_div(cfg.hidden_size, attn_tp_size)
    if isinstance(cfg.num_key_value_heads, (list, tuple)):
        expected_num_kv_heads_per_layer = [
            ceil_div(kv, attn_tp_size * attn_cp_size) for kv in cfg.num_key_value_heads
        ]
        assert list(bindings_cfg.num_kv_heads_per_layer) == expected_num_kv_heads_per_layer
        assert bindings_cfg.num_kv_heads(0) == expected_num_kv_heads_per_layer[0]
    else:
        assert bindings_cfg.num_kv_heads(0) == ceil_div(
            cfg.num_key_value_heads, attn_tp_size * attn_cp_size
        )

    # tp_size-dependent value (uses mapping.tp_size, not attn_tp_size).
    assert bindings_cfg.mlp_hidden_size == ceil_div(cfg.intermediate_size, mapping.tp_size)
    assert bindings_cfg.tokens_per_block == tokens_per_block


def _make_model_config_with_kv_quant(kv_cache_quant_algo):
    return ModelConfig(quant_config=QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo))


def _make_kv_cache_config(
    *, use_kv_cache_manager_v2: bool, cross_kv_cache_fraction: float | None = None
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
        cross_kv_cache_fraction=cross_kv_cache_fraction,
        enable_block_reuse=True,
    )


def test_validate_and_set_kv_cache_quant_auto_uses_checkpoint():
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "auto")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


@pytest.mark.parametrize(
    "kv_cache_dtype,expected_quant",
    [("fp8", QuantAlgo.FP8), ("fp8_ds_mla", QuantAlgo.FP8), ("nvfp4", QuantAlgo.NVFP4)],
)
def test_validate_and_set_kv_cache_quant_explicit_dtype_overrides(
    kv_cache_dtype: str,
    expected_quant: QuantAlgo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Pin a non-SM107 device: on SM107 an explicit NVFP4 request falls back to FP8.
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.model_loader.get_sm_version", lambda: 100)
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, kv_cache_dtype)
    assert model_config.quant_config.kv_cache_quant_algo == expected_quant


@pytest.mark.parametrize("is_encoder_decoder", [False, True])
def test_validate_and_set_kv_cache_quant_rejects_invalid_dtype(is_encoder_decoder):
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    model_config.is_encoder_decoder = is_encoder_decoder
    with pytest.raises(ValueError, match="Accepted types are"):
        validate_and_set_kv_cache_quant(model_config, "invalid_dtype")


@pytest.mark.parametrize(
    "kv_cache_dtype,checkpoint_quant,expected_quant",
    [
        ("fp8", None, QuantAlgo.FP8),
        ("fp8_ds_mla", None, QuantAlgo.FP8),
        ("nvfp4", None, QuantAlgo.NVFP4),
        ("fp8", QuantAlgo.NVFP4, QuantAlgo.FP8),
        ("auto", QuantAlgo.FP8, QuantAlgo.FP8),
        ("auto", QuantAlgo.NVFP4, QuantAlgo.NVFP4),
    ],
)
@pytest.mark.parametrize("sm107", [False, True])
def test_validate_and_set_kv_cache_quant_preserves_encoder_decoder_self_cache(
    kv_cache_dtype: str,
    checkpoint_quant: QuantAlgo | None,
    expected_quant: QuantAlgo,
    sm107: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if sm107:
        _mock_sm107(monkeypatch)
    else:
        monkeypatch.setattr(
            "tensorrt_llm._torch.pyexecutor.model_loader.get_sm_version", lambda: 100
        )
    model_config = ModelConfig(
        is_encoder_decoder=True,
        quant_config=QuantConfig(kv_cache_quant_algo=checkpoint_quant),
    )

    validate_and_set_kv_cache_quant(model_config, kv_cache_dtype)

    if sm107 and expected_quant == QuantAlgo.NVFP4:
        expected_quant = QuantAlgo.FP8
    assert model_config.quant_config.kv_cache_quant_algo == expected_quant


@pytest.mark.parametrize("model_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_encoder_decoder", [False, True])
@pytest.mark.parametrize(
    "weight_quant,checkpoint_quant",
    [(None, None), (QuantAlgo.FP8, None), (None, QuantAlgo.INT8)],
)
def test_validate_and_set_kv_cache_quant_preserves_unaffected_configs(
    model_dtype: torch.dtype,
    is_encoder_decoder: bool,
    weight_quant: QuantAlgo | None,
    checkpoint_quant: QuantAlgo | None,
) -> None:
    pretrained_config = make_pretrained_config(head_dim=4, is_encoder_decoder=is_encoder_decoder)
    pretrained_config.torch_dtype = model_dtype
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        quant_config=QuantConfig(quant_algo=weight_quant, kv_cache_quant_algo=checkpoint_quant),
    )

    validate_and_set_kv_cache_quant(model_config, "auto")

    assert model_config.quant_config.kv_cache_quant_algo == checkpoint_quant
    assert model_config.quant_config.quant_algo == weight_quant
    assert model_config.torch_dtype == model_dtype


def _mock_sm107(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.model_loader.get_sm_version", lambda: 107)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.model_loader.torch.cuda.is_available",
        lambda: True,
    )


def test_validate_and_set_kv_cache_quant_downgrades_nvfp4_on_sm107(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_sm107(monkeypatch)
    model_config = _make_model_config_with_kv_quant(QuantAlgo.FP8)
    validate_and_set_kv_cache_quant(model_config, "nvfp4")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def _make_mixed_precision_model_config():
    """MIXED_PRECISION checkpoint shape: global config plus per-layer entries
    whose kv_cache_quant_algo comes only from hf_quant_config.json (None here)."""
    return ModelConfig(
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=None),
        quant_config_dict={
            "model.layers.0.attention": QuantConfig(quant_algo=QuantAlgo.FP8),
            "model.layers.0.mixer.experts": QuantConfig(quant_algo=QuantAlgo.NVFP4),
        },
    )


def test_validate_and_set_kv_cache_quant_propagates_to_quant_config_dict():
    """Explicit kv_cache_config.dtype must override the per-layer QuantConfigs
    too, otherwise the KV pool (sized from the global config) and attention
    modules (built from per-layer configs) disagree on KV element size."""
    model_config = _make_mixed_precision_model_config()
    validate_and_set_kv_cache_quant(model_config, "fp8")
    assert model_config.quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo == QuantAlgo.FP8


def test_validate_and_set_kv_cache_quant_auto_keeps_quant_config_dict():
    model_config = _make_mixed_precision_model_config()
    validate_and_set_kv_cache_quant(model_config, "auto")
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo is None


@pytest.mark.parametrize("implementation", [None, "triton", "msa"])
@pytest.mark.parametrize("requested_dtype", ["nvfp4", "auto"])
def test_validate_and_set_kv_cache_quant_sm107_downgrade_propagates_to_quant_config_dict(
    monkeypatch: pytest.MonkeyPatch,
    implementation: str | None,
    requested_dtype: str,
) -> None:
    """The SM107 NVFP4->FP8 decision must reach per-layer QuantConfigs too,
    otherwise a mixed-precision checkpoint and the KV pool (sized from the
    global config) disagree. MiniMax-M3 MSA keeps NVFP4, which its hybrid
    cache serves on SM107; every other configuration falls back to FP8."""
    _mock_sm107(monkeypatch)
    model_config = _make_mixed_precision_model_config()
    if implementation is not None:
        model_config.sparse_attention_config = MiniMaxM3SparseAttentionConfig(
            implementation=implementation
        )
    if requested_dtype == "auto":
        model_config.quant_config.kv_cache_quant_algo = QuantAlgo.NVFP4
        for layer_quant_config in model_config.quant_config_dict.values():
            layer_quant_config.kv_cache_quant_algo = QuantAlgo.NVFP4
    validate_and_set_kv_cache_quant(model_config, requested_dtype)
    expected = QuantAlgo.NVFP4 if implementation == "msa" else QuantAlgo.FP8
    assert model_config.quant_config.kv_cache_quant_algo == expected
    for layer_quant_config in model_config.quant_config_dict.values():
        assert layer_quant_config.kv_cache_quant_algo == expected


def _write_safetensors_header(checkpoint_dir, tensor_dtype, tensor_shape):
    shard_name = "model-00001-of-00001.safetensors"
    header = {
        _DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: {
            "dtype": tensor_dtype,
            "shape": tensor_shape,
            "data_offsets": [0, 0],
        }
    }
    encoded_header = json.dumps(header).encode("utf-8")

    with open(checkpoint_dir / shard_name, "wb") as f:
        f.write(struct.pack("<Q", len(encoded_header)))
        f.write(encoded_header)

    with open(checkpoint_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {_DEEPSEEK_V4_ROUTED_EXPERT_WEIGHT: shard_name}}, f)


@pytest.mark.parametrize(
    "tensor_dtype,tensor_shape,expected_layout,expected_is_base",
    [
        pytest.param("I8", [2048, 2048], "mxfp4", False, id="mxfp4"),
        pytest.param("U8", [2048, 2048], "nvfp4", False, id="nvfp4"),
        pytest.param("F8_E4M3", [2048, 4096], None, True, id="base-fp8"),
    ],
)
def test_deepseek_v4_base_checkpoint_detection(
    tmp_path, tensor_dtype, tensor_shape, expected_layout, expected_is_base
):
    _write_safetensors_header(tmp_path, tensor_dtype, tensor_shape)

    assert ModelConfig._detect_deepseek_v4_routed_moe_layout(str(tmp_path)) == expected_layout
    assert ModelConfig._is_deepseek_v4_base_checkpoint(str(tmp_path)) is expected_is_base


def test_nvfp4_plus_override_selects_only_the_cutedsl_moe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under the override AUTO resolves to CUTEDSL, while another explicit backend
    and a checkpoint that names NVFP4_PLUS are rejected as the config loads.

    The config pre-pass in load_config_and_apply_defaults reads the checkpoint
    without the requested backend, so AUTO must pass get_mxfp4_quant_algo.
    """
    monkeypatch.setenv("OVERRIDE_QUANT_ALGO", "NVFP4_PLUS")
    assert ModelConfig.resolve_moe_backend("AUTO", "DeepseekV4ForCausalLM") == "CUTEDSL"
    assert ModelConfig.get_mxfp4_quant_algo("AUTO") == QuantAlgo.NVFP4_PLUS
    with pytest.raises(ValueError, match="moe_config.backend: CUTEDSL"):
        ModelConfig.get_mxfp4_quant_algo("TRTLLM")
    with pytest.raises(ValueError, match="OVERRIDE_QUANT_ALGO=NVFP4_PLUS"):
        read_modelopt_quant_config({"quantization": {"quant_algo": "NVFP4_PLUS"}})


@pytest.mark.parametrize(
    "tactic, names_the_override",
    [("nvfp4_plus", True), ("trtllm.cutlass.grouped_gemm.nvfp4", False)],
    ids=["nvfp4_plus_without_override", "nvfp4"],
)
def test_an_mxfp4_checkpoint_refuses_a_tactic_naming_none_of_its_run_formats(
    monkeypatch: pytest.MonkeyPatch, tactic: str, names_the_override: bool
) -> None:
    """MXFP4 experts run as one of three formats, so a tactic naming none of them fails as the
    checkpoint is read, and an NVFP4_PLUS one is told the override that selects its format."""
    monkeypatch.delenv("OVERRIDE_QUANT_ALGO", raising=False)
    with pytest.raises(ValueError, match="names no implementation of a format MXFP4") as error:
        ModelConfig.get_mxfp4_quant_algo("AUTO", moe_tactic=tactic)
    assert ("OVERRIDE_QUANT_ALGO=NVFP4_PLUS" in str(error.value)) is names_the_override


def test_deepseek_v4_mxfp4_experts_take_the_nvfp4_plus_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An MXFP4 checkpoint's routed experts become NVFP4_PLUS with 32-value blocks."""
    monkeypatch.setenv("OVERRIDE_QUANT_ALGO", "NVFP4_PLUS")
    _write_safetensors_header(tmp_path, "I8", [2048, 2048])
    layer_quant_config = ModelConfig._set_deepseek_v4_routed_moe_quant_config(
        types.SimpleNamespace(num_hidden_layers=2),
        str(tmp_path),
        MoESelection(backend="CUTEDSL").request(),
        None,
    )
    for layer_idx in range(2):
        experts = layer_quant_config[f"model.layers.{layer_idx}.mlp.experts"]
        assert experts.quant_algo == QuantAlgo.NVFP4_PLUS
        assert experts.group_size == 32


@pytest.mark.parametrize("sidecar", [True, False], ids=["modelopt-file", "modelopt-inline"])
def test_deepseek_v41_nvfp4_checkpoint_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sidecar: bool
) -> None:
    """Resolve dense FP8 separately from the checkpoint's NVFP4 expert policy."""
    from tensorrt_llm._torch import model_config as model_config_module
    from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41Config

    quantization = {
        "quant_algo": "MIXED_PRECISION",
        "group_size": 16,
        "kv_cache_quant_algo": "FP8",
        "exclude_modules": ["*.attn.*", "*.ffn.shared_experts.*", "head", "mtp.*"],
        "quantized_layers": {
            "layers.0.ffn.experts": {"quant_algo": "NVFP4", "group_size": 16},
            "model.layers.0.self_attn.o_proj": {
                "quant_algo": "FP8_BLOCK_SCALES",
                "group_size": 128,
            },
        },
    }
    inline = {
        **deepcopy(quantization),
        "producer": {"name": "modelopt"},
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
        "modules_to_not_convert": ["*engram*"],
    }
    pretrained = DeepseekV41Config(
        architectures=["DeepseekV41ForCausalLM"], quantization_config=deepcopy(inline)
    )
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained
    )
    _write_safetensors_header(tmp_path, "U8", [2304, 2560])
    if sidecar:
        # The file's explicit null must win over the inline KV policy.
        quantization["kv_cache_quant_algo"] = None
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps({"producer": inline["producer"], "quantization": quantization})
        )

    model_config = ModelConfig.from_pretrained(str(tmp_path), moe_backend="CUTLASS")
    dense = model_config.quant_config
    assert dense.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert dense.group_size == 128
    assert dense.kv_cache_quant_algo == (None if sidecar else QuantAlgo.FP8)
    assert dense.is_module_excluded_from_quantization("model.layers.1.engram.kv_proj")
    assert not dense.is_module_excluded_from_quantization("model.layers.0.mlp.shared_experts")
    for layer in range(pretrained.num_hidden_layers):
        expert = model_config.quant_config_dict[f"model.layers.{layer}.mlp.experts"]
        assert expert.quant_algo == QuantAlgo.NVFP4
        assert expert.group_size == 16
    explicit = model_config.quant_config_dict["model.layers.0.self_attn.o_proj"]
    assert explicit.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert explicit.group_size == 128
    assert explicit.kv_cache_quant_algo == dense.kv_cache_quant_algo
    assert pretrained.quantization_config == inline


def test_deepseek_v4_missing_compress_ratios_raises(tmp_path, monkeypatch):
    """DeepSeek-V4 load must fail fast with a clear error when neither the
    checkpoint config nor a user ``sparse_attention_config`` provides
    ``compress_ratios``.

    Regression test: previously ``compress_ratios`` could stay ``None`` and the
    internal normalization comprehension raised an opaque
    ``TypeError: 'NoneType' object is not iterable`` mid-load.
    """
    from tensorrt_llm._torch import model_config as model_config_module
    from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config

    pretrained_config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        compress_ratios=None,
        num_hidden_layers=4,
    )

    # Avoid touching the filesystem for the HF config load; the empty tmp_path
    # makes the real ``_is_deepseek_v4_base_checkpoint`` probe return False.
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained_config
    )

    with pytest.raises(ValueError, match="compress_ratios"):
        ModelConfig.from_pretrained(str(tmp_path))


_TRTLLM_FP8_BLOCK_EXCLUDES = ["*kv_b_proj*", "*k_b_proj*", "*eh_proj"]


def _load_with_tactic(
    tmp_path, monkeypatch, tactic, quant_algo, architecture="DummyMoeForCausalLM"
):
    """``from_pretrained`` on a modelopt checkpoint of one global ``quant_algo``."""
    from transformers import PretrainedConfig

    from tensorrt_llm._torch import model_config as model_config_module

    pretrained = PretrainedConfig(architectures=[architecture])
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained
    )
    monkeypatch.setattr(model_config_module, "is_sm_100f", lambda: False)
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"producer": {"name": "modelopt"}, "quantization": {"quant_algo": quant_algo}})
    )
    return ModelConfig.from_pretrained(str(tmp_path), moe_tactic=tactic)


@pytest.mark.parametrize(
    "tactic, backend, exclude_modules",
    [
        pytest.param(None, "CUTLASS", None, id="no_tactic"),
        pytest.param("trtllm_gen", "TRTLLM", _TRTLLM_FP8_BLOCK_EXCLUDES, id="one_family"),
        pytest.param("deepgemm.cuda.grouped_gemm.fp8_block_scales", "DEEPGEMM", None, id="full_id"),
        # TRTLLM-gen and Marlin leaves; the FP8 block-scales format leaves one.
        pytest.param("fused_moe", "TRTLLM", _TRTLLM_FP8_BLOCK_EXCLUDES, id="narrowed_family"),
    ],
)
def test_a_tactic_sets_the_backend_the_checkpoint_is_read_for(
    tmp_path, monkeypatch, tactic, backend, exclude_modules
):
    """Quant parsing and the final literal follow the family a pin settles on.

    The FP8 block-scales exclusions are what parsing adds for TRTLLM, so they
    show which family parsing saw. The architecture-only hint is CUTLASS on a
    non-SM100 machine, so it must not decide for a pin. Without a tactic both
    are AUTO, which resolves to CUTLASS there.
    """
    model_config = _load_with_tactic(tmp_path, monkeypatch, tactic, "FP8_BLOCK_SCALES")

    assert model_config.moe_backend == backend
    assert model_config.moe_backend_requested == "AUTO"
    assert model_config.moe_tactic == tactic
    assert model_config.quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert model_config.quant_config.exclude_modules == exclude_modules


def test_a_tactic_is_narrowed_to_the_checkpoints_format(tmp_path, monkeypatch):
    """``mega_moe`` names a CuteDSL NVFP4 and a DeepGEMM MXFP4 leaf; NVFP4 settles it."""
    from tensorrt_llm._torch.moe.fused_moe.mega_moe import TrtllmCutedslMegaMoeNvfp4Impl
    from tensorrt_llm._torch.moe.fused_moe.moe_resolution import backend_family_of

    model_config = _load_with_tactic(tmp_path, monkeypatch, "mega_moe", "NVFP4")

    assert model_config.moe_backend == backend_family_of(TrtllmCutedslMegaMoeNvfp4Impl)


def test_a_tactic_spanning_families_for_the_checkpoint_fails_at_load(tmp_path, monkeypatch):
    """Model code is built for one family, so a tactic still naming several is refused."""
    with pytest.raises(ValueError, match="several backend families"):
        _load_with_tactic(tmp_path, monkeypatch, "fp8_block_scales", "FP8_BLOCK_SCALES")


_FP8_BLOCK = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
_MXFP4 = QuantConfig(quant_algo=QuantAlgo.W4A8_MXFP4_MXFP8)
_NVFP4 = QuantConfig(quant_algo=QuantAlgo.NVFP4)
_MIXED = QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)


@pytest.mark.parametrize(
    "tactic, quant_config, quant_config_dict, backend",
    [
        # DeepSeek-V4 MXFP4: FP8 block-scale attention, MXFP4 routed experts.
        pytest.param(
            "deepgemm",
            _FP8_BLOCK,
            {"model.layers.0.self_attn.wq_a": _FP8_BLOCK, "model.layers.0.mlp.experts": _MXFP4},
            "MEGAMOE_DEEPGEMM",
            id="experts_entry",
        ),
        # Per-linear expert keys describe the experts module they sit under.
        pytest.param(
            "mega_moe",
            _MIXED,
            {
                "model.layers.0.self_attn.q_proj": _MXFP4,
                "model.layers.0.block_sparse_moe.experts.0.w1": _NVFP4,
                "model.layers.0.block_sparse_moe.experts.0.w2": _NVFP4,
            },
            "MEGAMOE_CUTEDSL",
            id="per_expert_keys",
        ),
        # No experts entry: the experts run as the global format.
        pytest.param(
            "mega_moe",
            _NVFP4,
            {"model.layers.0.self_attn.q_proj": _MXFP4},
            "MEGAMOE_CUTEDSL",
            id="global_format",
        ),
    ],
)
def test_a_tactic_is_narrowed_by_the_expert_formats_only(
    tactic, quant_config, quant_config_dict, backend
):
    """Formats of other modules do not keep a tactic spanning families."""
    assert (
        ModelConfig.pinned_moe_backend(tactic, quant_config, quant_config_dict, architecture="")
        == backend
    )


def test_a_tactic_spanning_families_across_moe_layers_fails():
    """Experts of two formats leave ``mega_moe`` one family per layer, not one per model."""
    quant_config_dict = {
        "model.layers.0.mlp.experts": _NVFP4,
        "model.layers.1.mlp.experts": _MXFP4,
    }
    with pytest.raises(ValueError, match="several backend families"):
        ModelConfig.pinned_moe_backend("mega_moe", _MIXED, quant_config_dict, architecture="")


_KIMI_K3 = "KimiK3ForConditionalGeneration"
# How the compressed-tensors parser reads the original moonshotai/Kimi-K3.
_K3_CHECKPOINT = QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4, group_size=32)


@pytest.mark.parametrize(
    "quant_config_dict, backend",
    [
        pytest.param(None, "MEGAMOE_DEEPGEMM", id="default_format"),
        pytest.param(
            {"model.layers.1.block_sparse_moe.experts": _NVFP4},
            "MEGAMOE_CUTEDSL",
            id="per_layer_entries",
        ),
    ],
)
def test_a_tactic_on_kimi_k3_follows_the_format_its_experts_run_as(quant_config_dict, backend):
    """Kimi K3 runs its routed experts as W4A8_MXFP4_MXFP8 unless the checkpoint lists them."""
    assert (
        ModelConfig.pinned_moe_backend(
            "mega_moe", _K3_CHECKPOINT, quant_config_dict, architecture=_KIMI_K3
        )
        == backend
    )


@pytest.mark.parametrize(
    "quant_config, architecture",
    [
        pytest.param(QuantConfig(), "LlamaForCausalLM", id="dense"),
        pytest.param(_K3_CHECKPOINT, "Qwen3MoeForCausalLM", id="unserved_format"),
    ],
)
def test_a_tactic_serving_no_expert_format_builds_for_auto(monkeypatch, quant_config, architecture):
    """No family follows from ``mega_moe`` there; ``resolve_moe_impl`` refuses its MoE layers."""
    monkeypatch.delenv("OVERRIDE_QUANT_ALGO", raising=False)
    warnings = []
    monkeypatch.setattr(logger, "warning_once", lambda message, key: warnings.append(message))

    backend = ModelConfig.pinned_moe_backend(
        "mega_moe", quant_config, None, architecture=architecture
    )

    assert backend == ModelConfig.resolve_moe_backend("AUTO", architecture, quant_config)
    assert len(warnings) == 1
    assert repr(moe_impl_request(tactic="mega_moe").label) in warnings[0]
    assert f"moe_backend {backend}" in warnings[0]


@pytest.mark.parametrize(
    "tactic, quant_algo, backend",
    [
        (None, QuantAlgo.W4A8_MXFP4_MXFP8, "TRTLLM"),
        ("trtllm.trtllm_gen.fused_moe.w4a16_mxfp4", QuantAlgo.W4A16_MXFP4, "TRTLLM"),
        # No quant segment: the one MXFP4 format among its matches is the format.
        ("deepgemm.mega_moe", QuantAlgo.W4A8_MXFP4_MXFP8, "MEGAMOE_DEEPGEMM"),
    ],
    ids=["no_tactic", "w4a16", "no_quant_segment"],
)
def test_an_mxfp4_checkpoint_runs_as_the_format_the_tactic_names(
    tmp_path, monkeypatch, tactic, quant_algo, backend
):
    """MXFP4 weights run as several formats; the implementations requested pick one."""
    from transformers import PretrainedConfig

    from tensorrt_llm._torch import model_config as model_config_module

    pretrained = PretrainedConfig(
        architectures=["GptOssForCausalLM"], quantization_config={"quant_method": "mxfp4"}
    )
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained
    )
    monkeypatch.setattr(model_config_module, "get_sm_version", lambda: 100)
    monkeypatch.delenv("OVERRIDE_QUANT_ALGO", raising=False)

    model_config = ModelConfig.from_pretrained(str(tmp_path), moe_tactic=tactic)

    assert model_config.quant_config.quant_algo == quant_algo
    assert model_config.moe_backend == backend


@pytest.mark.parametrize(
    "backend, tactic, sm, quant_algo",
    [
        pytest.param("AUTO", None, 100, QuantAlgo.W4A8_MXFP4_MXFP8, id="auto-sm100"),
        pytest.param("AUTO", "trtllm_gen.w4a16_mxfp4", 100, QuantAlgo.W4A16_MXFP4, id="tactic"),
    ],
)
def test_the_mxfp4_format_follows_the_request_before_the_backend_is_resolved(
    monkeypatch, backend, tactic, sm, quant_algo
):
    """How ``llm_utils`` reads an MXFP4 checkpoint, where the literal may still be AUTO."""
    from tensorrt_llm._torch import model_config as model_config_module

    monkeypatch.setattr(model_config_module, "get_sm_version", lambda: sm)
    monkeypatch.delenv("OVERRIDE_QUANT_ALGO", raising=False)

    assert ModelConfig.get_mxfp4_quant_algo(backend, moe_tactic=tactic) is quant_algo


@pytest.mark.parametrize(
    "tactic, backend", [(None, "AUTO"), ("cutedsl.grouped_gemm", "CUTEDSL")], ids=str
)
def test_the_mistral_loader_builds_for_the_family_a_tactic_implies(monkeypatch, tactic, backend):
    """The Mistral loader rewrites ``from_pretrained`` and has to settle the literal too."""
    from transformers import PretrainedConfig

    from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader

    pretrained = PretrainedConfig(
        architectures=["MistralLarge3ForCausalLM"],
        quantization_config={"quant_method": "compressed-tensors", "config_groups": {"NVFP4": {}}},
    )
    monkeypatch.setattr(
        MistralConfigLoader, "_parse_mistral_config", lambda self, checkpoint_dir: ({}, pretrained)
    )

    model_config = MistralConfigLoader().load("/unused", moe_backend="AUTO", moe_tactic=tactic)

    assert model_config.quant_config.quant_algo == QuantAlgo.NVFP4
    assert model_config.moe_backend == backend


def test_a_tactic_the_registry_refuses_fails_at_load(tmp_path, monkeypatch):
    from transformers import PretrainedConfig

    from tensorrt_llm._torch import model_config as model_config_module

    pretrained = PretrainedConfig(architectures=["DummyMoeForCausalLM"])
    monkeypatch.setattr(
        model_config_module, "load_pretrained_config", lambda *args, **kwargs: pretrained
    )

    with pytest.raises(ValueError, match="unknown MoE impl token 'nosuchprovider'"):
        ModelConfig.from_pretrained(str(tmp_path), moe_tactic="nosuchprovider")


def test_model_config_sets_is_encoder_decoder_from_pretrained_config():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    assert model_config.is_encoder_decoder is True


def test_validate_encoder_decoder_kv_cache_config_accepts_v1_enc_dec():
    """Both V1 and V2 are supported as long as ``cross_kv_cache_fraction`` is set."""
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    validate_encoder_decoder_kv_cache_config(
        model_config,
        _make_kv_cache_config(use_kv_cache_manager_v2=False, cross_kv_cache_fraction=0.5),
    )


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
def test_validate_encoder_decoder_kv_cache_config_requires_cross_fraction(
    use_kv_cache_manager_v2: bool,
) -> None:
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    with pytest.raises(ValueError, match="cross_kv_cache_fraction to be set"):
        validate_encoder_decoder_kv_cache_config(
            model_config,
            _make_kv_cache_config(use_kv_cache_manager_v2=use_kv_cache_manager_v2),
        )


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True])
def test_validate_encoder_decoder_kv_cache_config_rejects_cross_fraction_for_decoder_only(
    use_kv_cache_manager_v2: bool,
) -> None:
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=False,
        )
    )

    with pytest.raises(ValueError, match="should only be set for encoder-decoder models"):
        validate_encoder_decoder_kv_cache_config(
            model_config,
            _make_kv_cache_config(
                use_kv_cache_manager_v2=use_kv_cache_manager_v2, cross_kv_cache_fraction=0.5
            ),
        )


def test_validate_encoder_decoder_kv_cache_config_accepts_v2_enc_dec():
    model_config = ModelConfig(
        pretrained_config=make_pretrained_config(
            head_dim=4,
            is_encoder_decoder=True,
        )
    )

    validate_encoder_decoder_kv_cache_config(
        model_config,
        _make_kv_cache_config(use_kv_cache_manager_v2=True, cross_kv_cache_fraction=0.5),
    )


def test_copy_with_changes_the_moe_selection_on_the_copy_only():
    config = ModelConfig()
    config.moe_backend = "CUTLASS"
    config._frozen = True

    copied = config.copy_with(moe_backend="CUTEDSL", moe_tactic="cutedsl.grouped_gemm")

    assert (copied.moe_backend, copied.moe_tactic) == ("CUTEDSL", "cutedsl.grouped_gemm")
    assert (config.moe_backend, config.moe_tactic) == ("CUTLASS", None)
    assert copied._frozen
    assert copied.extra_attrs is config.extra_attrs


@pytest.mark.parametrize(
    "changes",
    [{"moe_backend": "CUTEDSL"}, {"moe_tactic": "cutedsl"}],
    ids=["backend_only", "tactic_only"],
)
def test_copy_with_refuses_half_a_moe_selection(changes):
    """A tactic left next to another backend pins what the copy is not built for."""
    with pytest.raises(ValueError, match="one MoE selection"):
        ModelConfig().copy_with(**changes)


def test_copy_with_refuses_an_attribute_model_config_lacks():
    with pytest.raises(AttributeError, match="no attribute 'quant_cfg_dict'"):
        ModelConfig().copy_with(quant_cfg_dict={})
