# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the transformers AutoConfig / AutoTokenizer dispatch
for TRT-LLM-only model_types (deepseek_v32, kimi_k2).

Before this registration, transformers >= 5.5 falls back to a bare
PreTrainedConfig that lacks `max_position_embeddings`, and
AutoTokenizer.from_pretrained then raises AttributeError on it. The
broken test that motivated this — perf/test_perf_sanity.py disagg gen_only
on GB200 — only runs in L0_PostMerge because it needs 12 GB200 GPUs across
3 nodes, so a cheap pre-merge unit test is the right place to catch
regressions.
"""

import json

import pytest

import tensorrt_llm  # noqa: F401  triggers AutoConfig registration
from tensorrt_llm._torch.configs import DeepseekV3Config


@pytest.mark.parametrize("model_type", ["deepseek_v32", "kimi_k2"])
def test_custom_model_type_registered_with_autoconfig(model_type):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert model_type in CONFIG_MAPPING
    assert CONFIG_MAPPING[model_type] is DeepseekV3Config


@pytest.mark.parametrize("model_type", ["deepseek_v32", "kimi_k2"])
def test_autoconfig_from_pretrained_resolves_to_local_config(tmp_path, model_type):
    # Mirrors what the benchmark_serving subprocess does under the hood:
    # AutoTokenizer.from_pretrained -> AutoConfig.from_pretrained. Without
    # the registration this fails through to a bare PreTrainedConfig that
    # lacks `max_position_embeddings`.
    from transformers import AutoConfig

    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "max_position_embeddings": 16384,
            }
        )
    )

    cfg = AutoConfig.from_pretrained(str(model_dir))
    assert isinstance(cfg, DeepseekV3Config)
    assert cfg.max_position_embeddings == 16384


@pytest.mark.parametrize("model_type", ["deepseek_v32", "kimi_k2"])
def test_load_hf_model_config_uses_autoconfig_dispatch(tmp_path, model_type):
    # ModelLoader.load_hf_model_config is the llmapi/llm_utils entry point used
    # by trtllm-serve to pre-load HF model configs. On transformers 5.5.x it
    # must dispatch via AutoConfig (which CONFIG_MAPPING.register affects), not
    # directly via PretrainedConfig.from_pretrained — the latter bypasses the
    # mapping and returns a bare PretrainedConfig without
    # `max_position_embeddings`, causing downstream AttributeError on the V3.2
    # disagg gen_only GB200 post-merge perf-sanity test.
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "max_position_embeddings": 16384,
            }
        )
    )

    cfg = ModelLoader.load_hf_model_config(str(model_dir))
    assert cfg is not None
    assert isinstance(cfg, DeepseekV3Config)
    assert cfg.max_position_embeddings == 16384
