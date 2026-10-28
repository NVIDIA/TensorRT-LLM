# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.engine.lora import LoraParamBuilder
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.config import DecoderRunnerConfig
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.dummy_requests import DummyRequestFactory
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.graphs import DecoderCudaGraphs
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import (
    DecoderInputBuffers,
    DecoderInputPreparer,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import (
    DecoderWarmup,
    WarmupAgreement,
)
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, PrefillCudaGraphBackend
from tensorrt_llm.mapping import Mapping


def decoder_config(**changes) -> DecoderRunnerConfig:
    config = DecoderRunnerConfig(
        max_batch_size=13,
        max_num_tokens=258,
        max_seq_len=2048,
        max_beam_width=1,
        without_logits=False,
        attention_backend=TrtllmAttention,
        dtype=torch.float16,
        enable_attention_dp=False,
        disable_overlap_scheduler=False,
        is_encode_only=False,
        is_spec_decode=False,
        spec_config=None,
        max_draft_len=0,
        max_total_draft_tokens=0,
        max_draft_loop_tokens=0,
        original_max_draft_len=0,
        original_max_total_draft_tokens=0,
        spec_dec_max_total_draft_tokens=0,
        num_seq_slots=None,
        cuda_graph_config=CudaGraphConfig(
            enable_padding=True, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]
        ),
        cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
        cuda_graph_padding_enabled=True,
        max_cuda_graph_batch_size=128,
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        prefill_cuda_graph_num_tokens=[],
        enable_in_graph_sampling=False,
        torch_compile_enabled=False,
        torch_compile_piecewise_cuda_graph=False,
        torch_compile_prefill_only=False,
        compile_only_piecewise_graphs=False,
        use_mrope=False,
        is_multimodal=False,
        mm_encoder_cache_enabled=False,
        enable_autotuner=True,
        cuda_graph_specialize_lora=False,
    )
    spec = changes.get("spec_config")
    if spec is not None and changes.get("is_spec_decode", True):
        draft_len = spec.max_draft_len
        total_draft_tokens = spec.tokens_per_gen_step - 1
        changes = dict(
            {
                "is_spec_decode": True,
                "max_draft_len": draft_len,
                "max_total_draft_tokens": total_draft_tokens,
                "max_draft_loop_tokens": total_draft_tokens,
                "original_max_draft_len": draft_len,
                "original_max_total_draft_tokens": total_draft_tokens,
                "spec_dec_max_total_draft_tokens": total_draft_tokens,
                **changes,
            }
        )
    return replace(config, **changes)


def input_preparer(
    config: DecoderRunnerConfig,
    *,
    model,
    mapping: Mapping | None = None,
    buffers: DecoderInputBuffers | None = None,
    mm_item_scheduler=None,
) -> DecoderInputPreparer:
    mapping = mapping or Mapping()
    return DecoderInputPreparer(
        buffers=buffers,
        config=config,
        mapping=mapping,
        dist=None,
        model=model,
        get_runtime_tokens_per_gen_step=(
            config.spec_config.get_runtime_tokens_per_gen_step
            if config.is_spec_decode
            else lambda draft_len: draft_len + 1
        ),
        iter_states={},
        lora=LoraParamBuilder(
            attn_backend=config.attention_backend,
            cuda_graph_manager=None,
            spec_config=config.spec_config,
        ),
        use_beam_search=config.max_beam_width > 1,
        steady_gen_cache_supported=True,
        mm_item_scheduler=mm_item_scheduler,
    )


def decoder_graphs(
    config: DecoderRunnerConfig,
    *,
    model=None,
    mapping=None,
    cuda_graph_runner=None,
    dummy_requests=None,
    metadata=None,
    forward_warmup=None,
    lora=None,
) -> DecoderCudaGraphs:
    mapping = mapping or Mapping()
    return DecoderCudaGraphs(
        cuda_graph_runner=cuda_graph_runner,
        breakable_cuda_graph_runner=None,
        config=config,
        mapping=mapping,
        model=model if model is not None else torch.nn.Identity(),
        sparse_attention_config=None,
        metrics={},
        dummy_requests=dummy_requests if dummy_requests is not None else Mock(),
        agreement=WarmupAgreement(None, mapping),
        metadata=metadata if metadata is not None else Mock(),
        get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
        dynamic_draft_len_mapping=None,
        lora=lora if lora is not None else SimpleNamespace(cuda_graph_manager=None),
        forward_warmup=forward_warmup if forward_warmup is not None else Mock(),
        torch_compile_bypass=nullcontext,
        context_promotion_supported=True,
    )


def decoder_warmup(
    config: DecoderRunnerConfig,
    *,
    model=None,
    mapping=None,
    agreement=None,
    dummy_requests=None,
    graphs=None,
    forward_warmup=None,
) -> DecoderWarmup:
    mapping = mapping or Mapping()
    model = model if model is not None else torch.nn.Identity()
    forward_warmup = forward_warmup if forward_warmup is not None else Mock()
    dummy_requests = (
        dummy_requests
        if dummy_requests is not None
        else DummyRequestFactory(
            config=config,
            model=model,
            get_runtime_tokens_per_gen_step=lambda draft_len: draft_len + 1,
            release_request_state=None,
        )
    )
    graphs = (
        graphs
        if graphs is not None
        else decoder_graphs(
            config,
            model=model,
            mapping=mapping,
            dummy_requests=dummy_requests,
            forward_warmup=forward_warmup,
        )
    )
    return DecoderWarmup(
        config=config,
        mapping=mapping,
        dist=agreement.dist if agreement is not None else None,
        model=model,
        sparse_attention_config=None,
        warmup_timer=_WarmupTimer(rank=0),
        metrics={},
        agreement=agreement if agreement is not None else WarmupAgreement(None, mapping),
        metadata=Mock(),
        dummy_requests=dummy_requests,
        graphs=graphs,
        inputs=SimpleNamespace(guided_decoder=None),
        forward_warmup=forward_warmup,
        torch_compile_bypass=nullcontext,
        context_warmups_supported=True,
        eager_workspace_reclaimable=False,
    )
