# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch

import torch

import tensorrt_llm
import tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup as decoder_warmup_module
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import DecoderRunner
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.warmup import WarmupAgreement
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class Config:
    torch_dtype: torch.dtype
    num_key_value_heads: int = 16
    num_attention_heads: int = 16
    hidden_size: int = 256
    architectures: list[str] | None = None
    rope_scaling: dict[str, object] | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class DummyModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype, rope_scaling: dict[str, object] | None = None) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            pretrained_config=Config(torch_dtype=dtype, rope_scaling=rope_scaling)
        )
        self.recorded_position_ids: torch.Tensor | None = None
        self.recorded_kwargs: list[dict[str, Any]] = []

    def infer_max_seq_len(self) -> int:
        return 2048

    @property
    def config(self) -> Config:
        return self.model_config.pretrained_config

    def forward(self, *args: object, **kwargs: Any) -> dict[str, torch.Tensor]:
        input_ids = kwargs["input_ids"]
        self.recorded_position_ids = kwargs["position_ids"]
        # Input buffers are reused across steps, so keep copies.
        if not torch.cuda.is_current_stream_capturing():
            self.recorded_kwargs.append(_clone_tensors(kwargs))
        batch_size = input_ids.size(0)
        return {"logits": torch.randn((batch_size, 10), device="cuda")}


class DummyModelEngine(ModelEngine):
    def __init__(
        self,
        llm_args: TorchLlmArgs,
        dtype: torch.dtype,
        spec_config: DecodingBaseConfig | None = None,
        is_generation: bool = True,
        model: DummyModel | None = None,
    ) -> None:
        self.dtype = dtype
        mapping = Mapping(
            world_size=tensorrt_llm.mpi_world_size(),
            tp_size=tensorrt_llm.mpi_world_size(),
            rank=tensorrt_llm.mpi_rank(),
        )
        if model is None:
            model = DummyModel(self.dtype)
        model.model_config.is_generation = is_generation
        super().__init__(
            model_path="dummy",
            mapping=mapping,
            model=model,
            llm_args=llm_args,
            spec_config=spec_config,
        )


def _create_request(num_tokens: int, req_id: int) -> LlmRequest:
    sampling_params = SamplingParams()
    kwargs = {
        "request_id": req_id,
        "max_new_tokens": 1,
        "input_tokens": [0] * num_tokens,
        "sampling_config": tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        "is_streaming": False,
    }
    result = LlmRequest(**kwargs)
    result.paged_kv_block_ids = []
    return result


def _create_request_with_tokens(tokens: list[int], req_id: int) -> LlmRequest:
    sampling_params = SamplingParams()
    request = LlmRequest(
        request_id=req_id,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()
        ),
        is_streaming=False,
    )
    request.paged_kv_block_ids = []
    return request


def _make_request_stub(req_id: int, prompt_len: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        py_request_id=req_id,
        context_chunk_size=1,
        context_remaining_length=1,
        context_current_position=prompt_len - 1,
        py_prompt_len=prompt_len,
        py_beam_width=1,
        py_draft_tokens=[],
        is_context_only_request=False,
        is_generation_only_request=False,
        py_disaggregated_params=None,
        py_multimodal_data=None,
        py_mm_encoder_event=None,
        py_mrope_position_delta=None,
        py_return_context_logits=False,
        py_batch_idx=None,
        lora_task_id=None,
        is_dummy=False,
        max_beam_num_tokens=prompt_len,
        state="context",
        py_llm_request_type="context_and_generation",
    )


@contextlib.contextmanager
def create_model_engine_and_kvcache(
    llm_args: TorchLlmArgs | None = None,
    execution_stream: torch.cuda.Stream | None = None,
    spec_config: DecodingBaseConfig | None = None,
    model: DummyModel | None = None,
) -> Iterator[tuple[DummyModelEngine, KVCacheManagerV2]]:
    if llm_args is None:
        llm_args = TorchLlmArgs(
            model="dummy",
            max_batch_size=13,
            max_num_tokens=258,
            cuda_graph_config=CudaGraphConfig(
                enable_padding=True, batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128]
            ),
        )
        assert all(size not in llm_args.cuda_graph_config.batch_sizes for size in (5, 13))
        assert all(size in llm_args.cuda_graph_config.batch_sizes for size in (8, 16))
    engine = DummyModelEngine(llm_args, torch.half, spec_config, model=model)
    with kv_cache_manager(model=engine.model, execution_stream=execution_stream) as manager:
        try:
            yield engine, manager
        finally:
            engine.release_executor_resources(release_model_owned_graphs=True)


@contextlib.contextmanager
def kv_cache_manager(
    *,
    model: DummyModel,
    max_tokens: int = 258,
    max_batch_size: int = 13,
    mapping: Mapping | None = None,
    execution_stream: torch.cuda.Stream | None = None,
    config: KvCacheConfig | None = None,
    max_num_tokens: int | None = None,
) -> Iterator[KVCacheManagerV2]:
    options = {} if max_num_tokens is None else {"max_num_tokens": max_num_tokens}
    manager = KVCacheManagerV2(
        config or KvCacheConfig(max_tokens=max_tokens),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        tokens_per_block=1,
        max_seq_len=max_tokens,
        max_batch_size=max_batch_size,
        mapping=mapping or Mapping(world_size=1, tp_size=1, rank=0),
        dtype=tensorrt_llm.bindings.DataType.HALF,
        execution_stream=execution_stream,
        **options,
    )
    try:
        yield manager
    finally:
        manager.shutdown()


def _clone_tensors(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in values.items()
    }


@contextlib.contextmanager
def _capture_tllm_logs() -> Iterator[list[str]]:
    """Capture logger.info calls emitted from the decoder runner module.

    tensorrt_llm.logger.logger is a custom Singleton (not a stdlib
    logging.Logger) and does not route through stdlib logging by default,
    so a logging.Handler attached to logging.getLogger("tensorrt_llm")
    sees nothing. Patch the logger.info bound on the decoder runner module
    directly so we observe exactly the messages warmup() emits.
    """
    records = []

    def _record(*msg: object) -> None:
        records.append(" ".join(str(m) for m in msg))

    with patch.object(decoder_warmup_module.logger, "info", side_effect=_record):
        yield records


def _agreement(
    *,
    world_size: int = 1,
    dwdp_size: int = 0,
    tp_size: int = 1,
    tp_peer_values: tuple[int, ...] = (),
    peer_values: tuple[int, ...] | None = None,
    with_dist: bool = True,
) -> WarmupAgreement:
    """Build the warmup agreement over a stand-in communicator and mapping.

    ``dist`` is ``Optional`` on the real engine -- stub engines such as
    ``DummyModelEngine`` construct without a communicator -- so ``with_dist``
    covers that state too.

    ``tp_allgather`` returns this rank's value followed by ``tp_peer_values``;
    ``allgather`` does the same over ``peer_values`` for the forward group.
    ``peer_values`` defaults to peers that mirror this rank, i.e. a world that
    agrees with whatever this rank reports.
    """

    def _allgather(value: int) -> list[int]:
        peers = [value] * (world_size - 1) if peer_values is None else list(peer_values)
        return [value, *peers]

    dist = (
        SimpleNamespace(
            world_size=world_size,
            rank=3,
            tp_allgather=lambda value: [value, *tp_peer_values],
            allgather=_allgather,
        )
        if with_dist
        else None
    )
    mapping = SimpleNamespace(
        dwdp_enabled=dwdp_size > 1,
        has_cp_helix=lambda: False,
        tp_size=tp_size,
    )
    return WarmupAgreement(dist, mapping)


class _AttentionMetadata:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)
        self.block_ids_per_seq = object()
        self.kv_block_ids_per_seq = object()
        self.on_update_kv_lens = Mock()


class _AttentionBackend:
    Metadata = _AttentionMetadata


def _model_engine_with_runner(
    runner: Mock | None,
    *,
    kv_cache_manager: object | None,
) -> tuple[ModelEngine, Mock]:
    engine = object.__new__(ModelEngine)
    engine._cleanup_done = True
    engine._model_caller = Mock()
    engine.model = SimpleNamespace(extra_attrs={})
    if runner is None:
        runner = object.__new__(DecoderRunner)
        runner._model = engine.model
        runner._trtllm_gen_jit_warmup = False
    engine._runner = runner
    engine._warmup_depth = 0
    engine.moe_load_balancer = None
    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = kv_cache_manager
    return engine, resource_manager
