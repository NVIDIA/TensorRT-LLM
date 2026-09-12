# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder-phase runner for encoder-decoder models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.memory_buffer_utils import with_shared_pool
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range, prefer_pinned

from .common import apply_position_id_offset, get_top_level_model
from .encoder import EncoderConfigMixin, EncoderMixin, EncoderPreparedInputs
from .interface import RunnerConfig, RunnerDeps


@dataclass(frozen=True)
class EncoderDecoderRunnerConfig(EncoderConfigMixin, RunnerConfig):
    """Configuration for the Encoder-Decoder model runner.

    Add ``DecoderConfigMixin`` when the decoder runner is introduced; that
    mixin belongs with the decoder implementation rather than this module.
    """


class EncoderDecoderRunner(EncoderMixin):
    """Run the independent encoder phase of an encoder-decoder model."""

    def __init__(
        self,
        model: nn.Module,
        deps: RunnerDeps,
        config: EncoderDecoderRunnerConfig,
    ) -> None:
        if not config.is_encoder_decoder:
            raise ValueError("EncoderDecoderRunner requires an encoder-decoder model.")
        self._initialize_encoder(model, deps, config)
        self._feature_staging: torch.Tensor | None = None
        self._feature_staging_event: torch.cuda.Event | None = None
        self._feature_copy_stream: torch.cuda.Stream | None = None

    def _build_attention_metadata(
        self,
        sequence_lengths: list[int],
        request_ids: list[int],
    ) -> VanillaAttentionMetadata | TrtllmAttentionMetadata:
        if len(sequence_lengths) != len(request_ids):
            raise ValueError("Encoder sequence lengths and request IDs must have the same length.")
        metadata = self._create_attention_metadata(
            enable_context_mla_with_cached_kv=False,
            num_heads_per_kv=1,
        )
        if not isinstance(metadata, (VanillaAttentionMetadata, TrtllmAttentionMetadata)):
            raise TypeError(
                "Only vanilla and TRT-LLM attention metadata are supported "
                "for encoder-decoder encoder execution."
            )
        metadata.seq_lens = torch.tensor(
            sequence_lengths, dtype=torch.int, pin_memory=prefer_pinned()
        )
        metadata.num_contexts = len(sequence_lengths)
        metadata.max_seq_len = self._config.max_seq_len
        metadata.request_ids = request_ids
        metadata.prepare_encoder_only()
        return metadata

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager | None,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        **model_inputs: Any,
    ) -> EncoderPreparedInputs:
        """Pack one scheduled encoder batch into the model's input contract."""
        if model_inputs:
            raise NotImplementedError(
                "EncoderDecoderRunner does not support additional model inputs. "
                f"Unsupported keys: {sorted(model_inputs)}"
            )
        del cuda_graph_lora_manager, runtime_draft_len
        encoder_requests = scheduled_requests.encoder_requests
        if not encoder_requests:
            raise ValueError("Encoder execution requires at least one request.")

        has_features = [
            request.py_encoder_input_features is not None for request in encoder_requests
        ]
        if any(has_features):
            if not all(has_features):
                raise ValueError(
                    "Feature- and token-driven encoder requests cannot share one batch."
                )
            return self._prepare_feature_inputs(
                encoder_requests,
                resource_manager=resource_manager,
            )
        return self._prepare_token_inputs(
            encoder_requests,
            resource_manager=resource_manager,
        )

    def _prepare_token_inputs(
        self,
        encoder_requests: list[LlmRequest],
        *,
        resource_manager: ResourceManager | None,
    ) -> EncoderPreparedInputs:
        input_ids: list[int] = []
        position_ids: list[int] = []
        sequence_lengths: list[int] = []
        request_ids: list[int] = []
        for request in encoder_requests:
            tokens = request.encoder_tokens
            if tokens is None:
                raise ValueError(f"Encoder request {request.py_request_id} has no encoder tokens.")
            sequence_length = len(tokens)
            input_ids.extend(tokens)
            position_ids.extend(
                apply_position_id_offset(
                    list(range(sequence_length)),
                    model=self._model,
                )
            )
            sequence_lengths.append(sequence_length)
            request_ids.append(request.py_request_id)

        num_tokens = len(input_ids)
        if num_tokens != len(position_ids):
            raise ValueError("Encoder input IDs and position IDs must have the same length.")
        if num_tokens > self._config.max_num_tokens:
            raise ValueError(
                f"Encoder packed length ({num_tokens}) exceeds max_num_tokens "
                f"({self._config.max_num_tokens})."
            )

        return self._prepare_packed_token_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_lengths=sequence_lengths,
            request_ids=request_ids,
            resource_manager=resource_manager,
        )

    def _prepare_feature_inputs(
        self,
        encoder_requests: list[LlmRequest],
        *,
        resource_manager: ResourceManager | None,
    ) -> EncoderPreparedInputs:
        features: list[torch.Tensor] = []
        sequence_lengths: list[int] = []
        request_ids: list[int] = []
        for request in encoder_requests:
            request_features = request.py_encoder_input_features
            if request_features is None:
                raise ValueError(f"Encoder request {request.py_request_id} has no input features.")
            features.append(request_features)
            sequence_lengths.append(int(request.encoder_output_len))
            request_ids.append(request.py_request_id)

        num_tokens = sum(sequence_lengths)
        if num_tokens > self._config.max_num_tokens:
            raise ValueError(
                f"Encoder packed length ({num_tokens}) exceeds max_num_tokens "
                f"({self._config.max_num_tokens})."
            )
        graph_inputs = self._prepare_encoder_feature_graph_inputs(
            features,
            sequence_lengths,
            request_ids,
            self._build_attention_metadata,
        )
        if graph_inputs is not None:
            return graph_inputs
        metadata = self._build_attention_metadata(sequence_lengths, request_ids)
        return EncoderPreparedInputs(
            {
                "input_features": self._pack_features(features),
                "encoder_attn_metadata": metadata,
                "encoder_seq_lens": sequence_lengths,
                "resource_manager": resource_manager,
            },
            sequence_lengths=sequence_lengths,
        )

    def _pack_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        first = features[0]
        uniform = first.device.type == "cpu" and all(
            feature.shape[1:] == first.shape[1:]
            and feature.dtype == first.dtype
            and feature.device.type == "cpu"
            for feature in features
        )
        if not uniform:
            return torch.cat(features, dim=0).to("cuda", non_blocking=True)

        rows = sum(feature.shape[0] for feature in features)
        staging = self._feature_staging
        if (
            staging is None
            or staging.dtype != first.dtype
            or staging.shape[1:] != first.shape[1:]
            or staging.shape[0] < rows
        ):
            if staging is not None:
                assert self._feature_staging_event is not None
                self._feature_staging_event.synchronize()
            staging = torch.empty(
                (rows, *first.shape[1:]),
                dtype=first.dtype,
                pin_memory=prefer_pinned(),
            )
            self._feature_staging = staging
            self._feature_staging_event = torch.cuda.Event()
            if self._feature_copy_stream is None:
                self._feature_copy_stream = torch.cuda.Stream()
        else:
            assert self._feature_staging_event is not None
            self._feature_staging_event.synchronize()

        offset = 0
        for feature in features:
            staging[offset : offset + feature.shape[0]].copy_(feature)
            offset += feature.shape[0]

        assert self._feature_copy_stream is not None
        assert self._feature_staging_event is not None
        consumer_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._feature_copy_stream):
            packed = staging[:rows].to("cuda", non_blocking=True)
            self._feature_staging_event.record()
        consumer_stream.wait_event(self._feature_staging_event)
        packed.record_stream(consumer_stream)
        return packed

    def warmup(self, resource_manager: ResourceManager | None) -> None:
        """Encoder-decoder warmup is performed while capturing graph shapes."""

    def capture_graphs(self, resource_manager: ResourceManager | None) -> None:
        self._capture_encoder_cuda_graphs(
            lambda sequence_lengths: self._prepare_capture_inputs(
                sequence_lengths, resource_manager
            ),
            self._execute_prepared,
            build_metadata=self._build_attention_metadata,
        )

    def _prepare_capture_inputs(
        self, sequence_lengths: list[int], resource_manager: ResourceManager | None
    ) -> EncoderPreparedInputs:
        request_ids = list(range(len(sequence_lengths)))
        position_ids: list[int] = []
        for sequence_length in sequence_lengths:
            position_ids.extend(
                apply_position_id_offset(list(range(sequence_length)), model=self._model)
            )
        return self._prepare_packed_token_inputs(
            input_ids=[0] * sum(sequence_lengths),
            position_ids=position_ids,
            sequence_lengths=sequence_lengths,
            request_ids=request_ids,
            resource_manager=resource_manager,
        )

    def _prepare_packed_token_inputs(
        self,
        *,
        input_ids: list[int],
        position_ids: list[int],
        sequence_lengths: list[int],
        request_ids: list[int],
        resource_manager: ResourceManager | None,
    ) -> EncoderPreparedInputs:
        input_ids_cpu = torch.tensor(
            input_ids,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        position_ids_cpu = torch.tensor(
            position_ids,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        metadata = self._build_attention_metadata(sequence_lengths, request_ids)
        graph_inputs = {
            "input_ids": input_ids_cpu,
            "position_ids": position_ids_cpu,
            "seq_lens": sequence_lengths,
            "resource_manager": resource_manager,
        }
        prepared = self._prepare_encoder_graph_inputs(graph_inputs, metadata)
        if prepared is not None:
            return prepared
        return EncoderPreparedInputs(
            {
                "encoder_input_ids": input_ids_cpu.to("cuda", non_blocking=True),
                "encoder_position_ids": position_ids_cpu.to("cuda", non_blocking=True).unsqueeze(0),
                "encoder_attn_metadata": metadata,
                "encoder_seq_lens": sequence_lengths,
                "resource_manager": resource_manager,
            },
            sequence_lengths=sequence_lengths,
        )

    @torch.inference_mode()
    @nvtx_range("encoder_decoder_forward")
    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager | None,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        gather_context_logits: bool,
        **model_inputs: Any,
    ) -> dict[str, Any]:
        del gather_context_logits
        prepared = self.prepare_inputs(
            scheduled_requests,
            resource_manager=resource_manager,
            cuda_graph_lora_manager=cuda_graph_lora_manager,
            runtime_draft_len=runtime_draft_len,
            **model_inputs,
        )
        hidden_states = self._execute_prepared(prepared)
        return {
            "encoder_hidden_states": hidden_states,
            "encoder_seq_lens": prepared.sequence_lengths,
        }

    def _forward_encoder_stack(self, inputs: dict[str, Any]) -> torch.Tensor:
        encoder = getattr(self._model, "encoder", None)
        if encoder is None:
            inner = getattr(self._model, "model", None)
            encoder = getattr(inner, "encoder", None) if inner is not None else None
        if encoder is None:
            raise AttributeError("Encoder-decoder models must expose `encoder` or `model.encoder`.")

        input_features = inputs.get("input_features")
        if input_features is not None:
            return encoder(
                input_features=input_features,
                attn_metadata=inputs["encoder_attn_metadata"],
            )

        top_level_model = get_top_level_model(self._model)
        embedding = getattr(top_level_model, "shared_embedding", None) or getattr(
            top_level_model, "embed_tokens", None
        )
        input_ids = inputs["encoder_input_ids"]
        if embedding is None:
            hidden_states = input_ids
        else:
            hidden_states = embedding(input_ids)
            embedding_scale = getattr(top_level_model, "embed_scale", None)
            if embedding_scale is not None:
                hidden_states = hidden_states * embedding_scale

        position_ids = inputs.get("encoder_position_ids")
        if position_ids is not None and position_ids.dim() == 2:
            position_ids = position_ids.squeeze(0)
        return encoder(
            hidden_states=hidden_states,
            attn_metadata=inputs["encoder_attn_metadata"],
            position_ids=position_ids,
        )

    def _forward_graph_inputs(self, inputs: dict[str, Any]) -> torch.Tensor:
        return self._forward_encoder_stack(
            {
                "encoder_input_ids": inputs["input_ids"],
                "encoder_position_ids": inputs.get("position_ids"),
                "encoder_attn_metadata": inputs["attn_metadata"],
                "resource_manager": inputs.get("resource_manager"),
            }
        )

    def _execute_prepared(self, prepared: EncoderPreparedInputs) -> torch.Tensor:
        key = prepared.graph_key
        if key is None:
            return self._forward_encoder_stack(prepared.kwargs)

        graph_runner = self._encoder_cuda_graph_runner
        # Feature graphs must not install the token path's shared allocator pool.
        pool_context = (
            nullcontext()
            if graph_runner.feature_mode
            else with_shared_pool(graph_runner.get_graph_pool())
        )
        with pool_context:
            graph_outputs = self._execute_encoder_cuda_graph(
                prepared,
                self._forward_feature_graph_inputs
                if graph_runner.feature_mode
                else self._forward_graph_inputs,
            )
        if not isinstance(graph_outputs, torch.Tensor):
            raise TypeError("Encoder-decoder CUDA Graph replay must return hidden states.")
        if graph_runner.feature_mode:
            return graph_outputs[: sum(prepared.sequence_lengths)].clone()
        return graph_runner.restore_encoder_decoder_output(key, graph_outputs, prepared.kwargs)

    def _forward_feature_graph_inputs(self, inputs: dict[str, Any]) -> torch.Tensor:
        return self._forward_encoder_stack(
            {
                "input_features": inputs["input_features"],
                "encoder_attn_metadata": inputs["attn_metadata"],
                "encoder_seq_lens": inputs["seq_lens"],
            }
        )
