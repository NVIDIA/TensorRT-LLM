# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup checks for the executor loops a non-generation model can run in."""

import re
from types import SimpleNamespace

import pytest
from transformers import BertConfig, LlamaConfig

from tensorrt_llm._torch.models.checkpoints.hf.checkpoint_loader import HfCheckpointLoader
from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
from tensorrt_llm._torch.pyexecutor.py_executor_creator import (
    _check_no_kv_execution,
    _check_scheduled_encoder_support,
)
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_PP2 = dict(world_size=2, tp_size=1, pp_size=2)
_ADP = dict(world_size=2, tp_size=2, enable_attention_dp=True)


def _args():
    return SimpleNamespace(enable_chunked_prefill=False)


@pytest.mark.parametrize(
    ("mapping", "error"),
    [(_PP2, "Pipeline parallelism"), (_ADP, "attention DP")],
    ids=["pp", "adp"],
)
def test_non_generation_models_reject_unsupported_loops(mapping, error):
    with pytest.raises(ValueError, match=error):
        _check_no_kv_execution(_args(), Mapping(**mapping), is_generation=False)


@pytest.mark.parametrize("mapping", [_PP2, _ADP], ids=["pp", "adp"])
def test_generation_models_are_not_checked(mapping):
    _check_no_kv_execution(_args(), Mapping(**mapping), is_generation=True)


def _encoder_args(**overrides):
    args = dict(
        enable_chunked_prefill=False,
        lora_config=None,
        speculative_config=None,
        cache_transceiver_config=None,
        kv_connector_config=None,
        dwdp_config=None,
        moe_config=SimpleNamespace(load_balancer=None),
        enable_resource_governor=False,
        sleep_config=None,
        orchestrator_type=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        (_PP2, "Pipeline parallelism"),
        (_ADP, "attention DP"),
        (dict(world_size=2, tp_size=1, cp_size=2), "Context parallelism"),
    ],
    ids=["pp", "adp", "cp"],
)
def test_scheduled_encoder_rejects_unsupported_parallelism(mapping, error):
    with pytest.raises(ValueError, match=error):
        _check_scheduled_encoder_support(_encoder_args(), Mapping(**mapping))


@pytest.mark.parametrize(
    ("overrides", "setting"),
    [
        (dict(lora_config=object()), "lora_config"),
        (dict(speculative_config=object()), "speculative_config"),
        (
            dict(cache_transceiver_config=SimpleNamespace(backend="NIXL")),
            "cache_transceiver_config",
        ),
        (dict(kv_connector_config=object()), "kv_connector_config"),
        (dict(dwdp_config=object()), "dwdp_config"),
        (dict(moe_config=SimpleNamespace(load_balancer=object())), "moe_config.load_balancer"),
        (dict(enable_resource_governor=True), "enable_resource_governor"),
        (dict(sleep_config=object()), "sleep_config"),
        (dict(orchestrator_type="ray"), "orchestrator_type='ray'"),
    ],
)
def test_scheduled_encoder_rejects_settings_by_name(overrides, setting):
    with pytest.raises(ValueError, match=re.escape(setting)):
        _check_scheduled_encoder_support(_encoder_args(**overrides), Mapping())


def test_scheduled_encoder_serves_tensor_parallel_rpc_workers():
    _check_scheduled_encoder_support(
        _encoder_args(
            orchestrator_type="rpc", cache_transceiver_config=SimpleNamespace(backend=None)
        ),
        Mapping(world_size=2, tp_size=2),
    )


@pytest.mark.parametrize(
    ("config", "model_kwargs", "is_generation"),
    [
        (BertConfig(architectures=["BertForSequenceClassification"]), None, False),
        (LlamaConfig(architectures=["LlamaForCausalLM"]), None, True),
        (
            LlamaConfig(architectures=["LlamaForCausalLM"]),
            {"architectures": ["LlamaForTextEmbedding"]},
            False,
        ),
    ],
    ids=["pooling", "generation", "overridden-to-pooling"],
)
def test_dispatch_reads_the_architecture_model_loading_will_use(
    tmp_path, config, model_kwargs, is_generation
):
    config.save_pretrained(tmp_path)
    args = TorchLlmArgs(model=str(tmp_path), model_kwargs=model_kwargs)

    _, loaded = ModelLoader.load_config_with_defaults(str(tmp_path), args, HfCheckpointLoader())

    assert loaded.is_generation is is_generation
