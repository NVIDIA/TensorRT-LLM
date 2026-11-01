# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from _torch.pyexecutor.engine_test_utils import _AttentionBackend

from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor.engine.input_buffers import InputBuffers
from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.no_kv_cache import NoKVCacheRunnerConfig
from tensorrt_llm._torch.pyexecutor.engine.runners.pooling import PoolingRunner
from tensorrt_llm.llmapi.llm_args import PrefillCudaGraphBackend


def _make_runner(
    runner_type: type[PoolingRunner] | type[MultimodalEncoderRunner],
    model: torch.nn.Module | SimpleNamespace,
    config: NoKVCacheRunnerConfig | None = None,
    *,
    model_caller: Mock | None = None,
) -> PoolingRunner | MultimodalEncoderRunner:
    config = config or _config()
    buffers = InputBuffers(
        input_ids_cuda=torch.empty(config.max_num_tokens, dtype=torch.int),
        position_ids_cuda=torch.empty(config.max_num_tokens, dtype=torch.int),
    )
    kwargs = dict(
        mapping=SimpleNamespace(),
        dist=SimpleNamespace(),
        moe_load_balancer=None,
        attention_runtime_features=AttentionRuntimeFeatures(),
    )
    if issubclass(runner_type, PoolingRunner):
        kwargs["model_caller"] = model_caller or Mock(return_value={"logits": torch.empty(0)})
    with patch.object(InputBuffers, "allocate", return_value=buffers):
        return runner_type(model, config, **kwargs)


def _config() -> NoKVCacheRunnerConfig:
    return NoKVCacheRunnerConfig(
        max_batch_size=4,
        max_num_tokens=16,
        max_seq_len=16,
        max_beam_width=1,
        without_logits=False,
        attention_backend=_AttentionBackend,
        enable_attention_dp=False,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=[],
        mm_encoder_cache_enabled=False,
    )


def _model(*, is_generation: bool, is_encoder_decoder: bool = False) -> SimpleNamespace:
    pretrained_config = SimpleNamespace(
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    return SimpleNamespace(
        extra_attrs={},
        model_config=SimpleNamespace(
            is_generation=is_generation,
            is_encoder_decoder=is_encoder_decoder,
            pretrained_config=pretrained_config,
            sparse_attention_config=None,
            enable_flash_mla=False,
        ),
        config=pretrained_config,
    )
