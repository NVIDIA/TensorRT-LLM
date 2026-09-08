# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder-phase runner for encoder-decoder models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tensorrt_llm._torch.memory_buffer_utils import with_shared_pool
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.logger import logger

from .common import apply_position_id_offset, get_top_level_model
from .encoder import EncoderConfigMixin, EncoderMixin
from .interface import PreparedInputs, RunnerConfig, RunnerDeps


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

    def _prepare_encoder_requests(
        self,
        encoder_requests: list[LlmRequest],
        *,
        resource_manager: ResourceManager | None = None,
    ) -> PreparedInputs:
        """Pack one scheduled encoder batch into the model's input contract."""
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

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager | None,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
    ) -> PreparedInputs:
        del cuda_graph_lora_manager, runtime_draft_len
        return self._prepare_encoder_requests(
            scheduled_requests.encoder_requests,
            resource_manager=resource_manager,
        )

    def _prepare_token_inputs(
        self,
        encoder_requests: list[LlmRequest],
        *,
        resource_manager: ResourceManager | None,
    ) -> PreparedInputs:
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
    ) -> PreparedInputs:
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
        return PreparedInputs(
            {
                "input_features_list": features,
                "encoder_attn_metadata": self._build_attention_metadata(
                    sequence_lengths, request_ids
                ),
                "encoder_seq_lens": sequence_lengths,
                "resource_manager": resource_manager,
            }
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
        self._capture_after_warmup(lambda: self._capture_configured_graphs(resource_manager))

    def _capture_configured_graphs(self, resource_manager: ResourceManager | None) -> None:
        runner = self._cuda_graph_runner
        operation = "warmup" if runner.is_warmup_only else "capture"
        num_processed = 0
        logger.info(f"Running encoder-decoder encoder CUDA graph {operation} ...")
        for key in sorted(runner.capture_keys, reverse=True):
            sequence_lengths = runner.get_capture_warmup_sequence_lengths(key)
            if sequence_lengths is None:
                continue
            logger.info(f"Encoder-decoder encoder CUDA graph {operation}: key={key}")
            if runner.feature_mode:
                self._forward_feature_graph(
                    features=[
                        torch.zeros(
                            (1, *runner.config.feature_shape),
                            dtype=runner.config.feature_dtype,
                        )
                        for _ in sequence_lengths
                    ],
                    sequence_lengths=sequence_lengths,
                    request_ids=list(range(len(sequence_lengths))),
                )
            else:
                input_ids = [0] * sum(sequence_lengths)
                position_ids: list[int] = []
                for sequence_length in sequence_lengths:
                    position_ids.extend(
                        apply_position_id_offset(
                            list(range(sequence_length)),
                            model=self._model,
                        )
                    )
                prepared = self._prepare_packed_token_inputs(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    sequence_lengths=sequence_lengths,
                    request_ids=list(range(len(sequence_lengths))),
                    resource_manager=resource_manager,
                )
                self._execute_prepared(prepared.kwargs)
            torch.cuda.synchronize()
            num_processed += 1
        logger.info(
            "Completed encoder-decoder encoder CUDA graph "
            f"{operation} for {num_processed} graph shape(s)."
        )

    def _prepare_packed_token_inputs(
        self,
        *,
        input_ids: list[int],
        position_ids: list[int],
        sequence_lengths: list[int],
        request_ids: list[int],
        resource_manager: ResourceManager | None,
    ) -> PreparedInputs:
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
        runner = self._cuda_graph_runner
        batch_size = len(sequence_lengths)
        use_graph_staging = runner.enabled and (
            batch_size in runner.supported_batch_sizes
            or (runner.padding_enabled and batch_size <= runner.max_supported_batch_size)
        )
        return PreparedInputs(
            {
                "encoder_input_ids": (
                    input_ids_cpu
                    if use_graph_staging
                    else input_ids_cpu.to("cuda", non_blocking=True)
                ),
                "encoder_position_ids": (
                    position_ids_cpu
                    if use_graph_staging
                    else position_ids_cpu.to("cuda", non_blocking=True)
                ).unsqueeze(0),
                "encoder_attn_metadata": self._build_attention_metadata(
                    sequence_lengths, request_ids
                ),
                "encoder_seq_lens": sequence_lengths,
                "encoder_input_ids_host": input_ids_cpu,
                "encoder_position_ids_host": position_ids_cpu,
                "resource_manager": resource_manager,
            }
        )

    @torch.inference_mode()
    @nvtx_range("encoder_decoder_forward")
    def _forward_encoder_requests(
        self,
        encoder_requests: list[LlmRequest],
        *,
        prepared_inputs: PreparedInputs,
    ) -> tuple[torch.Tensor, list[int]]:
        """Execute one scheduled encoder phase and return packed states."""
        graph_result = self._maybe_forward_feature_graph(encoder_requests)
        if graph_result is not None:
            return graph_result

        model_inputs = prepared_inputs.kwargs
        features = model_inputs.get("input_features_list")
        if features is not None:
            model_inputs = dict(model_inputs)
            model_inputs.pop("input_features_list")
            model_inputs["input_features"] = self._pack_features(features)
        hidden_states = self._execute_prepared(model_inputs)
        return hidden_states, model_inputs["encoder_seq_lens"]

    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager | None,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        moe_load_balancer: MoeLoadBalancer | None,
        gather_context_logits: bool,
    ) -> dict[str, Any]:
        del cuda_graph_lora_manager, runtime_draft_len
        del gather_context_logits
        encoder_requests = scheduled_requests.encoder_requests
        prepared_inputs = self._prepare_encoder_requests(
            encoder_requests,
            resource_manager=resource_manager,
        )
        with MoeLoadBalancerIterContext(moe_load_balancer):
            hidden_states, sequence_lengths = self._forward_encoder_requests(
                encoder_requests,
                prepared_inputs=prepared_inputs,
            )
        return {
            "encoder_hidden_states": hidden_states,
            "encoder_seq_lens": sequence_lengths,
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

    def _execute_prepared(self, inputs: dict[str, Any]) -> torch.Tensor:
        input_ids = inputs.get("encoder_input_ids_host")
        position_ids = inputs.get("encoder_position_ids_host")
        sequence_lengths = inputs["encoder_seq_lens"]
        runner = self._cuda_graph_runner
        if input_ids is None or position_ids is None:
            return self._forward_encoder_stack(inputs)

        graph_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "seq_lens": sequence_lengths,
            "resource_manager": inputs.get("resource_manager"),
        }
        with runner.pad_batch(graph_inputs, len(sequence_lengths)) as padded_inputs:
            graph_metadata, key = runner.maybe_get_cuda_graph(
                padded_inputs,
                inputs["encoder_attn_metadata"],
            )
            if key is None:
                eager_inputs = inputs
                if inputs["encoder_input_ids"].device.type == "cpu":
                    eager_inputs = dict(inputs)
                    eager_inputs["encoder_input_ids"] = inputs["encoder_input_ids"].to(
                        "cuda", non_blocking=True
                    )
                    eager_inputs["encoder_position_ids"] = inputs["encoder_position_ids"].to(
                        "cuda", non_blocking=True
                    )
                return self._forward_encoder_stack(eager_inputs)

            runner.retire_staging()
            model_inputs = runner.prepare_encoder_decoder_inputs(
                padded_inputs,
                key,
                sequence_lengths,
            )
            graph_metadata.prepare_encoder_cuda_graph_replay(model_inputs["seq_lens"], key[1])
            model_inputs["attn_metadata"] = graph_metadata

            with with_shared_pool(runner.get_graph_pool()):
                capture_outputs = None
                if runner.needs_capture(key):

                    def capture_forward_fn(
                        capture_inputs: dict[str, Any],
                    ) -> torch.Tensor:
                        return self._forward_graph_inputs(capture_inputs)

                    capture_outputs = runner.capture(
                        key,
                        capture_forward_fn,
                        model_inputs,
                    )

                if runner.is_warmup_only:
                    graph_outputs = capture_outputs
                else:
                    graph_outputs = runner.replay(key, model_inputs)

        if not isinstance(graph_outputs, torch.Tensor):
            raise TypeError("Encoder-decoder CUDA Graph replay must return hidden states.")
        return runner.restore_encoder_decoder_output(
            key,
            graph_outputs,
            model_inputs,
        )

    def _maybe_forward_feature_graph(
        self, encoder_requests: list[LlmRequest]
    ) -> tuple[torch.Tensor, list[int]] | None:
        runner = self._cuda_graph_runner
        if not runner.enabled or not runner.feature_mode:
            return None

        fixed_seq_len = runner.config.fixed_seq_len
        features: list[torch.Tensor] = []
        for request in encoder_requests:
            feature = request.py_encoder_input_features
            if (
                feature is None
                or int(request.encoder_output_len) != fixed_seq_len
                or tuple(feature.shape) != (1, *runner.config.feature_shape)
                or feature.dtype != runner.config.feature_dtype
            ):
                logger.warning_once(
                    "Encoder CUDA Graph request features do not match the "
                    "captured contract; the encoder phase stays eager.",
                    key="encoder_cuda_graph_feature_contract_warning",
                )
                return None
            features.append(feature)

        sequence_lengths = [fixed_seq_len] * len(encoder_requests)
        output = self._forward_feature_graph(
            features=features,
            sequence_lengths=sequence_lengths,
            request_ids=[request.py_request_id for request in encoder_requests],
        )
        if output is None:
            return None
        real_tokens = fixed_seq_len * len(encoder_requests)
        return output[:real_tokens].clone(), sequence_lengths

    def _forward_feature_graph(
        self,
        *,
        features: list[torch.Tensor],
        sequence_lengths: list[int],
        request_ids: list[int],
    ) -> torch.Tensor | None:
        runner = self._cuda_graph_runner
        fixed_seq_len = runner.config.fixed_seq_len
        graph_inputs = {
            "seq_lens": sequence_lengths,
            "input_features": features,
        }
        with runner.pad_batch(graph_inputs, len(sequence_lengths)) as padded_inputs:
            padded_sequence_lengths = padded_inputs["seq_lens"]
            padded_request_ids = list(request_ids) + [
                -(index + 1) for index in range(len(padded_sequence_lengths) - len(request_ids))
            ]
            graph_metadata, key = runner.captured_graph_metadata(padded_inputs)
            if key is None:
                eager_metadata = self._build_attention_metadata(
                    padded_sequence_lengths,
                    padded_request_ids,
                )
                graph_metadata, key = runner.maybe_get_cuda_graph(
                    padded_inputs,
                    eager_metadata,
                )
            if key is None:
                return None
            padded_inputs["attn_metadata"] = graph_metadata

            capture_output = None
            if runner.needs_capture(key):
                padded_batch_size, padded_num_tokens, _ = key
                graph_metadata.prepare_encoder_cuda_graph_replay(
                    [fixed_seq_len] * padded_batch_size,
                    padded_num_tokens,
                )
                capture_output = runner.capture(
                    key,
                    self._forward_feature_graph_inputs,
                    padded_inputs,
                )
            if runner.is_warmup_only:
                return capture_output
            return runner.replay(key, padded_inputs)

    def _forward_feature_graph_inputs(self, inputs: dict[str, Any]) -> torch.Tensor:
        return self._forward_encoder_stack(
            {
                "input_features": inputs["input_features"],
                "encoder_attn_metadata": inputs["attn_metadata"],
                "encoder_seq_lens": inputs["seq_lens"],
            }
        )
