# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU routing diagnostics; no model construction or checkpoint allocation."""

import builtins
import sys
from enum import Enum
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2.models.beam import routing
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only
_TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4"


def _config():
    return NS(
        pretrained_config=NS(
            architectures=["BeamForCausalLM"],
            num_hidden_layers=52,
            hidden_size=5120,
            num_experts=256,
            quantization_config={"quant_algo": "NVFP4"},
        ),
        mapping=NS(
            world_size=4,
            tp_size=4,
            pp_size=1,
            moe_tp_size=4,
            moe_ep_size=1,
            enable_attention_dp=False,
        ),
        spec_config=None,
        quant_config=NS(kv_cache_quant_algo=None),
    )


def _dflash(**overrides):
    fields = dict(
        spec_dec_mode=NS(is_dflash=lambda: True),
        speculative_model="draft",
        attention_backend="TRTLLM",
        max_draft_len=7,
        draft_len_schedule=None,
        use_rejection_sampling=False,
    )
    fields.update(overrides)
    return NS(**fields)


@pytest.mark.parametrize("builtin", [None, object])
def test_auto_refusal_warns_only_when_no_builtin_exists(monkeypatch, builtin):
    monkeypatch.setattr(_router_index, "_device_compute_capability", lambda: (10, 3))
    monkeypatch.setattr(AutoModelForCausalLM, "_resolve_class", lambda config: builtin)
    log = Mock()
    monkeypatch.setattr(_router_index, "logger", log)
    reads = Mock(side_effect=AssertionError("unsupported options must not read checkpoints"))
    monkeypatch.setattr(routing, "checkpoint_refusal", reads)
    args = NS(
        modeling_v2="auto",
        kv_cache_config=NS(dtype="auto"),
        cuda_graph_config=object(),
        disable_overlap_scheduler=False,
        max_num_tokens=8193,
    )
    assert _router_index.modeling_v2_resolve(_config(), args) is None
    message = (log.warning if builtin is None else log.info).call_args.args[0]
    for reason in ("CUDA graphs", "overlap scheduler", "max_num_tokens 8193"):
        assert reason in message
    if builtin is None:
        assert "no built-in implementation" in message
        log.info.assert_not_called()
    else:
        assert "using the built-in implementation" in message
        log.warning.assert_not_called()
    reads.assert_not_called()


def test_require_reports_all_engine_and_dflash_refusals(monkeypatch):
    monkeypatch.setattr(_router_index, "_device_compute_capability", lambda: (10, 3))
    config = _config()
    config.spec_config = _dflash(
        speculative_model=None,
        attention_backend="VANILLA",
        max_draft_len=3,
        draft_len_schedule={1: 3},
        use_rejection_sampling=True,
    )
    args = NS(
        modeling_v2="require",
        kv_cache_config=NS(dtype="fp8"),
        cuda_graph_config=object(),
        disable_overlap_scheduler=False,
        max_num_tokens=0,
    )
    with pytest.raises(ValueError) as error:
        _router_index.modeling_v2_resolve(config, args)
    for reason in (
        "drafter checkpoint",
        "only the TRTLLM backend reads",
        "max_draft_len 3",
        "draft_len_schedule",
        "rejection sampling",
        "CUDA graphs",
        "overlap scheduler",
        "max_num_tokens 0",
    ):
        assert reason in str(error.value)


@pytest.mark.parametrize("bad_shape", [False, True])
def test_rejected_hardware_or_shape_does_not_import_target(monkeypatch, bad_shape):
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        if "full_model_nvfp4__sm_103__tp4" in name and name.endswith(
            (".model", ".modeling", ".layout")
        ):
            pytest.fail(f"Target imported before deployment admission: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    config = _config()
    if bad_shape:
        config.pretrained_config.num_hidden_layers = 6
    with pytest.raises(_router_index.NoTarget):
        routing.route(
            config.pretrained_config,
            config.mapping,
            (10, 3) if bad_shape else (9, 0),
            None,
            None,
            llm_args=NS(max_num_tokens=8192),
        )


@pytest.mark.parametrize("spec", [None, _dflash()])
@pytest.mark.parametrize("cache", [None, QuantAlgo.FP8])
@pytest.mark.parametrize("tokens", [1, 8192])
def test_supported_cache_and_dflash_routes_remain_available(monkeypatch, spec, cache, tokens):
    target = type("Target", (), {})
    monkeypatch.setitem(
        sys.modules, _TARGET + ".modeling", NS(ModelingV2BeamFullModelNvfp4Sm103Tp4=target)
    )
    reads = Mock(return_value=None)
    monkeypatch.setattr(routing, "checkpoint_refusal", reads)
    config = _config()
    args = NS(
        cuda_graph_config=None,
        disable_overlap_scheduler=True,
        max_num_tokens=tokens,
        checkpoint_format="beam_nvfp4",
        torch_compile_config=None,
        prefill_cuda_graph_backend="disabled",
    )
    assert (
        routing.route(config.pretrained_config, config.mapping, (10, 3), spec, cache, llm_args=args)
        is target
    )
    reads.assert_called_once_with(config.pretrained_config, spec)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("cuda_graph_config", object(), "CUDA graphs"),
        ("disable_overlap_scheduler", False, "overlap scheduler"),
        ("load_format", "dummy", "load_format must be auto"),
        ("max_beam_width", 2, "max_beam_width must be 1"),
        ("max_num_tokens", None, "max_num_tokens must be explicitly set"),
        ("torch_compile_config", object(), "torch_compile_config must be None"),
        ("prefill_cuda_graph_backend", "piecewise", "prefill_cuda_graph_backend must be disabled"),
        ("prefill_cuda_graph_backend", "breakable", "prefill_cuda_graph_backend must be disabled"),
        (
            "prefill_cuda_graph_backend",
            Enum("PrefillBackend", {"BREAKABLE": "breakable"}).BREAKABLE,
            "prefill_cuda_graph_backend must be disabled",
        ),
    ],
)
@pytest.mark.parametrize("spec", [None, _dflash()])
def test_execution_mode_refused_before_checkpoint_access(monkeypatch, field, value, message, spec):
    reads = Mock(side_effect=AssertionError("unsupported options must not read checkpoints"))
    monkeypatch.setattr(routing, "checkpoint_refusal", reads)
    config = _config()
    args = NS(cuda_graph_config=None, disable_overlap_scheduler=True, max_num_tokens=None)
    setattr(args, field, value)
    with pytest.raises(_router_index.NoTarget, match=message):
        routing.route(config.pretrained_config, config.mapping, (10, 3), spec, None, llm_args=args)
    reads.assert_not_called()


@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize(
    "chunked,reuse,spec",
    [(False, False, None), (True, False, None), (False, True, None), (False, False, _dflash())],
)
def test_paged_context_page_size_admission(monkeypatch, page_size, chunked, reuse, spec):
    target = type("Target", (), {})
    monkeypatch.setitem(
        sys.modules, _TARGET + ".modeling", NS(ModelingV2BeamFullModelNvfp4Sm103Tp4=target)
    )
    reads = Mock(return_value=None)
    monkeypatch.setattr(routing, "checkpoint_refusal", reads)
    config = _config()
    args = NS(
        checkpoint_format="beam_nvfp4",
        max_num_tokens=8192,
        enable_chunked_prefill=chunked,
        kv_cache_config=NS(tokens_per_block=page_size, enable_block_reuse=reuse),
    )
    if page_size != 32 and (chunked or reuse or spec is not None):
        with pytest.raises(
            _router_index.NoTarget, match="kv_cache_config.tokens_per_block must be 32"
        ):
            routing.route(
                config.pretrained_config, config.mapping, (10, 3), spec, None, llm_args=args
            )
        reads.assert_not_called()
    else:
        assert (
            routing.route(
                config.pretrained_config, config.mapping, (10, 3), spec, None, llm_args=args
            )
            is target
        )
        reads.assert_called_once()


def test_loader_admission_precedes_checkpoint_access(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.weights import (
        BeamNvfp4CheckpointLoader,
    )

    target = type("Target", (), {})
    monkeypatch.setitem(
        sys.modules, _TARGET + ".modeling", NS(ModelingV2BeamFullModelNvfp4Sm103Tp4=target)
    )
    reads = Mock(return_value=None)
    monkeypatch.setattr(routing, "checkpoint_refusal", reads)
    config = _config()
    args = NS(max_num_tokens=8192)
    for loader, checkpoint_format in [(None, None), (object(), "beam_nvfp4")]:
        args.checkpoint_loader, args.checkpoint_format = loader, checkpoint_format
        with pytest.raises(
            _router_index.NoTarget, match="checkpoint_loader=BeamNvfp4CheckpointLoader"
        ):
            routing.route(
                config.pretrained_config, config.mapping, (10, 3), None, None, llm_args=args
            )
    reads.assert_not_called()
    for loader, checkpoint_format in [
        (None, "beam_nvfp4"),
        (object.__new__(BeamNvfp4CheckpointLoader), None),
    ]:
        args.checkpoint_loader, args.checkpoint_format = loader, checkpoint_format
        assert (
            routing.route(
                config.pretrained_config, config.mapping, (10, 3), None, None, llm_args=args
            )
            is target
        )
    assert reads.call_count == 2
