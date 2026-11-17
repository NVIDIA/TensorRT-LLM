# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the engine config for generated-data Beam routing cases."""

from types import SimpleNamespace

from tensorrt_llm._torch.modeling_v2.models.beam import routing


def route(pretrained_config, mapping, capability, spec, cache, *, llm_args=None):
    config = SimpleNamespace(
        pretrained_config=pretrained_config,
        mapping=mapping,
        spec_config=spec,
        quant_config=SimpleNamespace(kv_cache_quant_algo=cache, quant_algo=None),
    )
    if llm_args is None:
        llm_args = SimpleNamespace(checkpoint_format="beam_nvfp4", max_num_tokens=8192)
    if not hasattr(llm_args, "checkpoint_format"):
        llm_args.checkpoint_format = "beam_nvfp4"
    if not hasattr(llm_args, "kv_cache_config"):
        llm_args.kv_cache_config = SimpleNamespace(dtype="auto")
    elif not hasattr(llm_args.kv_cache_config, "dtype"):
        llm_args.kv_cache_config.dtype = "auto"
    return routing.route(config, llm_args, capability)
