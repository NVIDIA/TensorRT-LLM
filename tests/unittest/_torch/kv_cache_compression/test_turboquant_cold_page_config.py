# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration, admission, and factory tests for TurboQuant cold-page compression."""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from transformers import PretrainedConfig

from tensorrt_llm._errors import UnsupportedError
from tensorrt_llm._torch.kv_cache_compression.quantization_for_cold_page.turboquant_quantization import (
    TurboQuantColdPageQuantizationCompression,
)
from tensorrt_llm._torch.pyexecutor import _util as util_mod
from tensorrt_llm._torch.pyexecutor.resource_manager import DataType
from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm.llmapi.llm_args import (
    ColdPageQuantizationCompressionConfig,
    TorchLlmArgs,
    update_llm_args_with_extra_dict,
)
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, BufferConfig

pytestmark = pytest.mark.cpu_only

_SUPPORTED_SETTINGS = [("mse", bits) for bits in (2, 2.5, 3, 3.5, 4)] + [
    ("prod", bits) for bits in (2, 3, 4)
]


@pytest.fixture(autouse=True)
def _sm100_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit both formats whatever GPU, if any, runs the test."""
    monkeypatch.setattr(util_mod, "get_sm_version", lambda: 100)
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: True)


def _turboquant(**options: object) -> ColdPageQuantizationCompressionConfig:
    return ColdPageQuantizationCompressionConfig(quant="turboquant", **options)


def _create(
    config: ColdPageQuantizationCompressionConfig,
    *,
    active_kv_quant: QuantConfig | None = None,
    kv_dtype: str = "auto",
) -> tuple[object, PretrainedConfig]:
    pretrained_config = PretrainedConfig(num_hidden_layers=2, num_key_value_heads=2, head_dim=128)
    pretrained_config.model_type = "llama"
    model_engine = SimpleNamespace(
        mapping=SimpleNamespace(has_cp_helix=lambda: False),
        spec_config=None,
        model=SimpleNamespace(
            model_config=SimpleNamespace(
                quant_config=active_kv_quant, pretrained_config=pretrained_config
            )
        ),
    )
    manager = util_mod.create_kv_cache_compression_manager(
        config,
        model_engine=model_engine,
        kv_cache_config=SimpleNamespace(enable_block_reuse=True, dtype=kv_dtype),
    )
    return manager, pretrained_config


def _admit(
    config: ColdPageQuantizationCompressionConfig,
    spec_dec_mode: SpeculativeDecodingMode | None = None,
) -> None:
    util_mod.validate_kv_cache_compression_compatibility(
        config,
        SimpleNamespace(enable_block_reuse=True),
        None if spec_dec_mode is None else SimpleNamespace(spec_dec_mode=spec_dec_mode),
    )


def test_defaults_and_dumps_keep_each_format_to_its_own_options() -> None:
    turboquant = _turboquant()
    assert (turboquant.turboquant_bits, turboquant.turboquant_mode) == (4.0, "mse")
    assert type(turboquant.turboquant_bits) is float
    assert turboquant.turboquant_seed == 5
    assert not turboquant.skip_rope_quantization
    assert turboquant.supports_block_reuse()
    assert turboquant.supports_speculative_decoding()
    assert turboquant.model_dump() == {
        "algorithm": "quantization_for_cold_page",
        "quant": "turboquant",
        "skip_rope_quantization": False,
        "mixed_precision_first_n": 0,
        "mixed_precision_last_n": 0,
        "turboquant_bits": 4.0,
        "turboquant_mode": "mse",
        "turboquant_seed": 5,
    }
    # NVFP4 stays the default format, and its dump holds only the shared and the NVFP4 options.
    nvfp4 = ColdPageQuantizationCompressionConfig()
    assert nvfp4.model_dump() == {
        "algorithm": "quantization_for_cold_page",
        "quant": "nvfp4",
        "skip_rope_quantization": False,
        "nvfp4_residual_dim": 64,
        "scale_checkpoint_path": None,
        "mixed_precision_first_n": 0,
        "mixed_precision_last_n": 0,
    }
    assert ColdPageQuantizationCompressionConfig.model_validate(nvfp4.model_dump()) == nvfp4


@pytest.mark.parametrize(("mode", "bits"), _SUPPORTED_SETTINGS)
def test_supported_settings_parse_from_python_and_yaml_and_round_trip(mode, bits) -> None:
    yaml_options = yaml.safe_load(f"""
kv_cache_compression_config:
  algorithm: quantization_for_cold_page
  quant: turboquant
  turboquant_bits: {bits}
  turboquant_mode: {mode}
  turboquant_seed: 7
""")
    section = yaml_options["kv_cache_compression_config"]
    # YAML reads 3 as an int and 2.5 as a float; both forms must be accepted.
    assert type(section["turboquant_bits"]) is type(bits)
    serve_args = update_llm_args_with_extra_dict({"model": "/tmp/dummy_model"}, yaml_options)
    configs = (
        _turboquant(turboquant_bits=bits, turboquant_mode=mode, turboquant_seed=7),
        _turboquant(turboquant_bits=float(bits), turboquant_mode=mode, turboquant_seed=7),
        _turboquant(turboquant_bits=np.float32(bits), turboquant_mode=mode, turboquant_seed=7),
        ColdPageQuantizationCompressionConfig.model_validate(section),
        TorchLlmArgs(**serve_args).kv_cache_compression_config,
    )
    for config in configs:
        assert config == configs[0]
        assert (config.quant, config.turboquant_mode, config.turboquant_bits) == (
            "turboquant",
            mode,
            bits,
        )
        assert type(config.turboquant_bits) is float
        assert ColdPageQuantizationCompressionConfig.model_validate(config.model_dump()) == config
        assert (
            ColdPageQuantizationCompressionConfig.model_validate_json(config.model_dump_json())
            == config
        )


def test_model_defaults_revalidate_the_dump_of_either_format() -> None:
    """Applying model defaults rebuilds LLM args from a full dump of every field."""
    from tensorrt_llm.llmapi.llm_utils import apply_model_defaults_to_llm_args

    for config in (
        ColdPageQuantizationCompressionConfig(),
        _turboquant(turboquant_bits=3.5, turboquant_seed=5),
    ):
        args = TorchLlmArgs(model="/tmp/dummy_model", kv_cache_compression_config=config)
        apply_model_defaults_to_llm_args(args, {"max_batch_size": 8})
        assert args.max_batch_size == 8
        assert args.kv_cache_compression_config == config


@pytest.mark.parametrize(
    ("options", "match"),
    [
        *(
            ({"quant": "turboquant", "turboquant_bits": bits}, "turboquant_bits must be one of")
            for bits in (1, 5, 2.25, "x", True)
        ),
        *(
            (
                {"quant": "turboquant", "turboquant_mode": "prod", "turboquant_bits": bits},
                "fractional bit widths require turboquant_mode='mse'",
            )
            for bits in (2.5, 3.5)
        ),
        ({"quant": "turboquant", "turboquant_mode": "lloyd"}, "turboquant_mode"),
        ({"quant": "turboquant", "turboquant_seed": -1}, "turboquant_seed"),
        (
            {"quant": "turboquant", "nvfp4_residual_dim": 64},
            "nvfp4_residual_dim applies only to quant='nvfp4'",
        ),
        (
            {"quant": "turboquant", "scale_checkpoint_path": "/scales"},
            "scale_checkpoint_path applies only to quant='nvfp4'",
        ),
        # Explicitly set TurboQuant options are rejected even at their default values.
        ({"turboquant_bits": 4}, "turboquant_bits applies only to quant='turboquant'"),
        (
            {"quant": "nvfp4", "turboquant_mode": "mse", "turboquant_seed": 0},
            "turboquant_mode, turboquant_seed apply only to quant='turboquant'",
        ),
    ],
)
def test_unsupported_settings_are_rejected_at_construction(options, match) -> None:
    with pytest.raises(ValueError, match=match):
        ColdPageQuantizationCompressionConfig(**options)


@pytest.mark.parametrize(
    ("bits", "mode", "width", "error", "match"),
    [
        (4, "mse", 24, ValueError, "divisible by 16"),  # alignment of integer widths
        (2.5, "mse", 48, ValueError, "divisible by 32"),  # alignment of fractional widths
        (4, "mse", 560, NotImplementedError, "power-of-two rest"),  # 2 x 256 + 48
        (4, "mse", 1024, NotImplementedError, "up to 1008"),  # instance size
        (4, "prod", 752, NotImplementedError, "at most 4 power-of-two pieces"),  # 512+128+64+32+16
    ],
)
def test_codec_state_rejects_unsupported_vector_widths(bits, mode, width, error, match) -> None:
    """The width rules reject a layout before any kernel is compiled (no GPU needed)."""
    provider = TurboQuantColdPageQuantizationCompression(
        _turboquant(turboquant_bits=bits, turboquant_mode=mode),
        pretrained_config=SimpleNamespace(model_type="llama"),
    )
    slot_bytes = 32 * width * 2  # one KV head, 32 tokens, BF16
    cache_config = SimpleNamespace(
        tokens_per_block=32,
        layers=(
            AttentionLayerConfig(
                layer_id=0,
                buffers=[BufferConfig(role=role, size=slot_bytes) for role in ("key", "value")],
            ),
        ),
    )
    with pytest.raises(error, match=match):
        provider.build_codec_state(
            cache_config,
            runtime_dtype=DataType.BF16,
            pp_layers=(0,),
            num_kv_heads_per_layer=(1,),
            head_dim_per_layer=(width,),
        )


def test_factory_builds_the_turboquant_provider() -> None:
    config = _turboquant(turboquant_bits=3)
    manager, pretrained_config = _create(config)
    assert type(manager).__name__ == "TurboQuantColdPageQuantizationCompression"
    assert manager.provides_cold_page_codec is True
    assert manager.config is config
    assert manager.pretrained_config is pretrained_config


def test_factory_rejects_an_active_nvfp4_kv_cache() -> None:
    with pytest.raises(ValueError, match="the active KV cache uses NVFP4"):
        _create(_turboquant(), active_kv_quant=QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4))


@pytest.mark.parametrize("sm_version", (80, 86, 89, 90))
@pytest.mark.parametrize("source", ("kv_cache_config", "checkpoint"))
def test_an_fp8_active_kv_cache_needs_compute_capability_8_9(monkeypatch, sm_version, source):
    monkeypatch.setattr(util_mod, "get_sm_version", lambda: sm_version)
    if source == "kv_cache_config":
        options = dict(kv_dtype="fp8")
    else:
        options = dict(active_kv_quant=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8))
    if sm_version >= 89:
        assert _create(_turboquant(), **options)[0] is not None
    else:
        with pytest.raises(UnsupportedError, match="compute capability 8.9 or newer"):
            _create(_turboquant(), **options)
        # FP16 and BF16 KV caches stay supported there.
        assert _create(_turboquant())[0] is not None


@pytest.mark.parametrize("sm_version", (-1, 75, 80, 90, 103, 120))
def test_admission_needs_compute_capability_8_0_rather_than_sm100(monkeypatch, sm_version) -> None:
    monkeypatch.setattr(util_mod, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(util_mod, "is_sm_100f", lambda: False)
    if sm_version >= 80:
        _admit(_turboquant())
    else:
        with pytest.raises(UnsupportedError, match="compute capability 8.0 or newer"):
            _create(_turboquant())


@pytest.mark.parametrize(
    ("spec_dec_mode", "supported"),
    [
        (SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL, True),
        (SpeculativeDecodingMode.EAGLE3_ONE_MODEL, True),
        (SpeculativeDecodingMode.MTP, False),
        (SpeculativeDecodingMode.DFLASH, False),
        (SpeculativeDecodingMode.NGRAM, False),
    ],
)
def test_speculative_admission_matches_nvfp4(spec_dec_mode, supported) -> None:
    for config in (_turboquant(), ColdPageQuantizationCompressionConfig()):
        if supported:
            _admit(config, spec_dec_mode)
        else:
            with pytest.raises(ValueError, match="use one-model MTP-EAGLE or EAGLE3"):
                _admit(config, spec_dec_mode)
