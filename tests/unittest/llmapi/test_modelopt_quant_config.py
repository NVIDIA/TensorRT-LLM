# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``tensorrt_llm.quantization.modelopt_config``."""

import logging

import pytest

from tensorrt_llm.logger import logger as tllm_logger
from tensorrt_llm.quantization.modelopt_config import (
    ModelOptQuantConfig,
    is_modelopt_quant_config,
    parse_modelopt_quant_config,
    warn_if_inline_quant_config_differs,
)


@pytest.fixture
def warn_caplog(caplog):
    """Capture warnings emitted via the TensorRT-LLM logger.

    The TRT-LLM Logger wraps the stdlib logger named ``"TRT-LLM"`` and sets
    ``propagate = False`` so its messages don't reach pytest's root handler.
    Temporarily re-enable propagation, bind caplog to that named logger, and
    lower the TRT-LLM severity so warnings actually flow through.
    """
    underlying = tllm_logger._logger
    prev_propagate = underlying.propagate
    prev_severity = tllm_logger._min_severity
    underlying.propagate = True
    tllm_logger.set_level("warning")
    caplog.set_level(logging.WARNING, logger="TRT-LLM")
    try:
        yield caplog
    finally:
        underlying.propagate = prev_propagate
        tllm_logger.set_level(prev_severity)


# ---------- Fixtures ---------------------------------------------------------

_QUANTIZED_LAYERS = {
    "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
    "model.layers.0.mlp.down_proj": {
        "quant_algo": "NVFP4",
        "group_size": 16,
    },
}

_EXCLUDES = ["lm_head", "model.embed_tokens"]


@pytest.fixture
def legacy_config():
    """ModelOpt 0.43.x shape: nested under 'quantization'."""
    return {
        "producer": {"name": "modelopt", "version": "0.43.0rc1"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "exclude_modules": list(_EXCLUDES),
            "quantized_layers": dict(_QUANTIZED_LAYERS),
        },
    }


@pytest.fixture
def flat_config():
    """ModelOpt 1.0.x shape: flat at top level with 'ignore' and 'quant_method'."""
    return {
        "producer": {"name": "modelopt", "version": "1.0.0"},
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": list(_EXCLUDES),
        "config_groups": {
            "group_0": {
                "input_activations": {"dynamic": False, "num_bits": 8, "type": "float"},
                "weights": {"dynamic": False, "num_bits": 8, "type": "float"},
                "targets": ["model.layers.0.mlp.up_proj"],
            },
            "group_1": {
                "input_activations": {
                    "dynamic": False,
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                },
                "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["model.layers.0.mlp.down_proj"],
            },
        },
        "quantized_layers": dict(_QUANTIZED_LAYERS),
    }


# ---------- is_modelopt_quant_config -----------------------------------------


def test_is_modelopt_legacy(legacy_config):
    assert is_modelopt_quant_config(legacy_config)


def test_is_modelopt_flat(flat_config):
    assert is_modelopt_quant_config(flat_config)


def test_is_modelopt_compressed_tensors_no_modelopt():
    cfg = {"quant_method": "compressed-tensors", "config_groups": {}, "ignore": []}
    assert not is_modelopt_quant_config(cfg)


def test_is_modelopt_garbage():
    assert not is_modelopt_quant_config({})
    assert not is_modelopt_quant_config({"producer": {}})


# ---------- parse_modelopt_quant_config: parallel readers --------------------


def test_parse_legacy_branch(legacy_config):
    parsed = parse_modelopt_quant_config(legacy_config)
    assert isinstance(parsed, ModelOptQuantConfig)
    assert parsed.source_format == "legacy"
    assert parsed.quant_algo == "MIXED_PRECISION"
    assert parsed.kv_cache_quant_algo is None
    assert parsed.exclude_modules == _EXCLUDES
    assert parsed.quantized_layers == _QUANTIZED_LAYERS


def test_parse_flat_branch(flat_config):
    parsed = parse_modelopt_quant_config(flat_config)
    assert parsed.source_format == "flat"
    assert parsed.quant_algo == "MIXED_PRECISION"
    assert parsed.exclude_modules == _EXCLUDES
    assert parsed.quantized_layers == _QUANTIZED_LAYERS


def test_parse_flat_kv_cache_scheme_to_fp8():
    cfg = {
        "producer": {"name": "modelopt"},
        "quant_method": "modelopt",
        "quant_algo": "FP8",
        "kv_cache_scheme": {"type": "float", "num_bits": 8},
    }
    assert parse_modelopt_quant_config(cfg).kv_cache_quant_algo == "FP8"


def test_parse_flat_kv_cache_scheme_non_fp8_yields_none():
    cfg = {
        "producer": {"name": "modelopt"},
        "quant_method": "modelopt",
        "quant_algo": "FP8",
        "kv_cache_scheme": {"type": "int", "num_bits": 8},
    }
    assert parse_modelopt_quant_config(cfg).kv_cache_quant_algo is None


def test_parse_legacy_and_flat_produce_equal_structs(legacy_config, flat_config):
    """Both branches produce the same struct for equivalent inputs.

    The whole point of parallel readers: both branches produce the same
    struct (modulo source_format) for equivalent inputs.
    """
    legacy_parsed = parse_modelopt_quant_config(legacy_config)
    flat_parsed = parse_modelopt_quant_config(flat_config)
    # Drop source_format so the comparison ignores that one provenance field.
    for p in (legacy_parsed, flat_parsed):
        p.source_format = ""
    assert legacy_parsed == flat_parsed


def test_parse_rejects_non_modelopt():
    cfg = {"quant_method": "compressed-tensors", "config_groups": {}}
    with pytest.raises(ValueError, match="Not a ModelOpt"):
        parse_modelopt_quant_config(cfg)


def test_parse_rejects_non_dict():
    with pytest.raises(ValueError, match="Expected a dict"):
        parse_modelopt_quant_config([])  # type: ignore[arg-type]


# ---------- ModelOptQuantConfig.to_legacy_inner_dict -------------------------


def test_to_legacy_inner_dict_roundtrip(legacy_config):
    parsed = parse_modelopt_quant_config(legacy_config)
    out = parsed.to_legacy_inner_dict()
    assert out["quant_algo"] == "MIXED_PRECISION"
    assert out["exclude_modules"] == _EXCLUDES
    assert out["quantized_layers"] == _QUANTIZED_LAYERS
    # kv_cache_quant_algo was None -> not emitted in the dict.
    assert "kv_cache_quant_algo" not in out


# ---------- warn_if_inline_quant_config_differs ------------------------------


def _captured_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_warn_no_inline_no_warning(legacy_config, warn_caplog):
    warn_if_inline_quant_config_differs(
        parse_modelopt_quant_config(legacy_config), None, source_file="hf_quant_config.json"
    )
    assert _captured_warnings(warn_caplog) == []


def test_warn_legacy_and_flat_inline_match(legacy_config, flat_config, warn_caplog):
    """File is legacy, inline is flat with identical content: no warning."""
    warn_if_inline_quant_config_differs(
        parse_modelopt_quant_config(legacy_config), flat_config, source_file="hf_quant_config.json"
    )
    assert _captured_warnings(warn_caplog) == []


def test_warn_quant_algo_mismatch(legacy_config, warn_caplog):
    inline = {
        "producer": {"name": "modelopt"},
        "quant_method": "modelopt",
        "quant_algo": "FP8",
        "ignore": list(_EXCLUDES),
        "quantized_layers": dict(_QUANTIZED_LAYERS),
    }
    warn_if_inline_quant_config_differs(
        parse_modelopt_quant_config(legacy_config), inline, source_file="hf_quant_config.json"
    )
    msgs = _captured_warnings(warn_caplog)
    assert len(msgs) == 1
    assert "quant_algo" in msgs[0]


def test_warn_exclude_modules_mismatch(legacy_config, warn_caplog):
    inline = {
        "producer": {"name": "modelopt"},
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": _EXCLUDES + ["extra"],
        "quantized_layers": dict(_QUANTIZED_LAYERS),
    }
    warn_if_inline_quant_config_differs(
        parse_modelopt_quant_config(legacy_config), inline, source_file="hf_quant_config.json"
    )
    msgs = _captured_warnings(warn_caplog)
    assert len(msgs) == 1
    assert "exclude_modules" in msgs[0]


def test_warn_quantized_layers_count_mismatch(legacy_config, warn_caplog):
    inline_layers = dict(_QUANTIZED_LAYERS)
    inline_layers["model.layers.1.mlp.up_proj"] = {"quant_algo": "FP8"}
    inline = {
        "producer": {"name": "modelopt"},
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "ignore": list(_EXCLUDES),
        "quantized_layers": inline_layers,
    }
    warn_if_inline_quant_config_differs(
        parse_modelopt_quant_config(legacy_config), inline, source_file="hf_quant_config.json"
    )
    msgs = _captured_warnings(warn_caplog)
    assert len(msgs) == 1
    assert "quantized_layers" in msgs[0]


def test_warn_skips_unparseable_inline(legacy_config, warn_caplog):
    inline = {"quant_method": "compressed-tensors", "config_groups": {}}
    warn_if_inline_quant_config_differs(
        parse_modelopt_quant_config(legacy_config), inline, source_file="hf_quant_config.json"
    )
    assert _captured_warnings(warn_caplog) == []
