# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model runners for encoder-only and encoder-decoder encoder execution."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import torch
from torch import nn
from typing_extensions import Self

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionBackend,
    AttentionMetadata,
    AttentionRuntimeFeatures,
)
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.memory_buffer_utils import with_shared_pool
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancerIterContext
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    EncoderCUDAGraphRunner,
    EncoderCUDAGraphRunnerConfig,
    EncoderKeyType,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.utils import torch_multi_arange, with_model_extra_attrs
from tensorrt_llm._utils import maybe_pin_memory, nvtx_range, prefer_pinned
from tensorrt_llm.llmapi.llm_args import EncodeCudaGraphConfig, validate_token_encoder_bucket_config
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..cuda_graph import cuda_graph_capture, cuda_graph_disabled, filter_cuda_graph_batch_sizes
from ..metadata import build_attention_metadata
from .interface import PreparedInputs, RunnerConfig, RunnerDeps


@dataclass(frozen=True, kw_only=True)
class EncoderConfigMixin:
    """Encoder-specific fields shared by runner configuration types."""

    enable_autotuner: bool
    cuda_graph_enabled: bool
    cuda_graph_padding_enabled: bool
    cuda_graph_batch_sizes: list[int]
    cuda_graph_num_tokens: list[int]
    cuda_graph_seq_lens: list[int]
    max_cuda_graph_batch_size: int
    max_cuda_graph_num_tokens: int
    is_encoder_decoder: bool
    use_fixed_sequence_slots: bool
    feature_shape: tuple[int, ...] | None = None
    feature_dtype: torch.dtype | None = None
    fixed_seq_len: int | None = None

    @classmethod
    def create(
        cls,
        *,
        model: nn.Module,
        mapping: Mapping,
        graph_config: EncodeCudaGraphConfig | None,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        max_beam_width: int,
        without_logits: bool,
        attention_backend: type[AttentionBackend],
        attention_runtime_features: AttentionRuntimeFeatures,
        enable_autotuner: bool,
        is_encoder_decoder: bool,
        draft_model: bool,
    ) -> Self:
        """Resolve encoder settings and construct the concrete runner config."""
        batch_sizes = list(graph_config.batch_sizes or []) if graph_config is not None else []
        num_tokens = list(graph_config.num_tokens or []) if graph_config is not None else []
        seq_lens = list(graph_config.seq_lens or []) if graph_config is not None else []
        enable_padding = graph_config.enable_padding if graph_config is not None else False

        unwrapped_model = getattr(model, "_orig_mod", model)
        graph_spec_fn = getattr(unwrapped_model, "encoder_graph_spec", None)
        model_graph_spec = graph_spec_fn() if graph_spec_fn is not None else None

        if graph_config is not None and model_graph_spec is None:
            config_error = validate_token_encoder_bucket_config(
                num_tokens,
                seq_lens,
                stays_eager=not is_encoder_decoder,
            )
            if config_error is not None:
                if not is_encoder_decoder:
                    logger.warning(config_error)
                else:
                    raise ValueError(config_error)

        feature_shape = None
        feature_dtype = None
        fixed_seq_len = None
        if (
            graph_config is not None
            and not draft_model
            and is_encoder_decoder
            and model_graph_spec is not None
        ):
            if mapping.tp_size > 1:
                logger.warning(
                    "Feature-mode encoder CUDA graphs require TP=1; the encoder phase stays eager."
                )
            else:
                feature_shape, feature_dtype, fixed_seq_len = model_graph_spec

        filtered_batch_sizes = (
            filter_cuda_graph_batch_sizes(
                batch_sizes,
                max_batch_size,
                max_num_tokens,
                fixed_seq_len or 1,
                enable_padding,
            )
            if batch_sizes
            else []
        )
        filtered_num_tokens = (
            _filter_cuda_graph_num_tokens(
                num_tokens,
                max_num_tokens,
                enable_padding,
            )
            if num_tokens
            else []
        )
        filtered_seq_lens = (
            _filter_cuda_graph_seq_lens(
                seq_lens,
                max_seq_len,
                enable_padding,
            )
            if seq_lens
            else []
        )

        if feature_shape is not None:
            graph_shapes_available = bool(filtered_batch_sizes)
            if not graph_shapes_available:
                logger.warning(
                    "Feature-mode encoder CUDA graphs have no batch size within "
                    f"the {max_num_tokens} token budget; the encoder phase stays eager."
                )
                feature_shape = None
                feature_dtype = None
                fixed_seq_len = None
        elif model_graph_spec is not None:
            graph_shapes_available = False
            if graph_config is not None:
                logger.warning(
                    "This model consumes fixed-shape encoder features, but feature "
                    "CUDA graphs are unavailable; the encoder phase stays eager."
                )
        else:
            graph_shapes_available = bool(filtered_num_tokens and filtered_seq_lens)

        cuda_graph_enabled = bool(graph_config is not None and graph_shapes_available)
        max_graph_batch_size = filtered_batch_sizes[-1] if filtered_batch_sizes else 0
        max_graph_num_tokens = (
            max_graph_batch_size * fixed_seq_len
            if feature_shape is not None
            else (filtered_num_tokens[-1] if filtered_num_tokens else 0)
        )
        return cls(
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            max_beam_width=max_beam_width,
            without_logits=without_logits,
            attention_backend=attention_backend,
            attention_runtime_features=attention_runtime_features,
            enable_autotuner=enable_autotuner,
            cuda_graph_enabled=cuda_graph_enabled,
            cuda_graph_padding_enabled=enable_padding,
            cuda_graph_batch_sizes=filtered_batch_sizes,
            cuda_graph_num_tokens=filtered_num_tokens,
            cuda_graph_seq_lens=filtered_seq_lens,
            max_cuda_graph_batch_size=max_graph_batch_size,
            max_cuda_graph_num_tokens=max_graph_num_tokens,
            is_encoder_decoder=is_encoder_decoder,
            use_fixed_sequence_slots=(
                is_encoder_decoder
                and hasattr(
                    model.model_config.pretrained_config,
                    "relative_attention_num_buckets",
                )
            ),
            feature_shape=feature_shape,
            feature_dtype=feature_dtype,
            fixed_seq_len=fixed_seq_len,
        )


@dataclass(frozen=True)
class EncoderRunnerConfig(EncoderConfigMixin, RunnerConfig):
    """Configuration for the EncoderOnly model runner."""

    @classmethod
    def create(
        cls,
        *,
        model: nn.Module,
        mapping: Mapping,
        graph_config: EncodeCudaGraphConfig | None,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        max_beam_width: int,
        without_logits: bool,
        attention_backend: type[AttentionBackend],
        attention_runtime_features: AttentionRuntimeFeatures,
        enable_autotuner: bool,
        draft_model: bool,
    ) -> Self:
        return super().create(
            model=model,
            mapping=mapping,
            graph_config=graph_config,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            max_beam_width=max_beam_width,
            without_logits=without_logits,
            attention_backend=attention_backend,
            attention_runtime_features=attention_runtime_features,
            enable_autotuner=enable_autotuner,
            is_encoder_decoder=False,
            draft_model=draft_model,
        )


def _filter_cuda_graph_num_tokens(
    values: list[int],
    max_num_tokens: int,
    enable_padding: bool,
) -> list[int]:
    result: list[int] = []
    for index, value in enumerate(values):
        if value <= max_num_tokens:
            result.append(value)
            continue
        if enable_padding and (index == 0 or result[index - 1] != max_num_tokens):
            logger.warning(
                "CUDA graph padding is enabled, but one of the given encoder "
                f"CUDA graph num_tokens ({value}) is larger than max_num_tokens "
                f"({max_num_tokens}). We will pad to {max_num_tokens}."
            )
            result.append(max_num_tokens)
        break
    return result


def _filter_cuda_graph_seq_lens(
    values: list[int],
    max_seq_len: int,
    enable_padding: bool,
) -> list[int]:
    result: list[int] = []
    for index, value in enumerate(values):
        if value <= max_seq_len:
            result.append(value)
            continue
        if enable_padding and (index == 0 or result[index - 1] != max_seq_len):
            logger.warning(
                "CUDA graph padding is enabled, but one of the given encoder "
                f"CUDA graph seq_lens ({value}) is larger than max_seq_len "
                f"({max_seq_len}). We will pad to {max_seq_len}."
            )
            result.append(max_seq_len)
        break
    return result


@dataclass(frozen=True, kw_only=True)
class EncoderPreparedInputs(PreparedInputs):
    """Model arguments, original lengths, and graph selected during preparation."""

    sequence_lengths: list[int]
    graph_key: EncoderKeyType | None = None


def get_encoder_graph_batch_sizes(
    batch_sizes: tuple[int, ...], max_batch_size: int, *, pad_to_limit: bool
) -> tuple[int, ...]:
    """Resolve encoder scheduling targets from immutable startup settings."""
    bounded_sizes = [size for size in batch_sizes if size <= max_batch_size]
    # Token padding can target a limit between buckets; feature padding cannot.
    if (
        pad_to_limit
        and any(size > max_batch_size for size in batch_sizes)
        and max_batch_size > 0
        and (not bounded_sizes or bounded_sizes[-1] != max_batch_size)
    ):
        bounded_sizes.append(max_batch_size)
    return tuple(bounded_sizes)


class EncoderMixin:
    """Shared no-KV-cache encoder mechanics.

    The concrete runners retain their own input and model-call contracts.
    This mixin owns encoder graph initialization, preparation, and capture
    mechanics. Concrete runners keep execution contexts and output semantics.
    """

    def _initialize_encoder(
        self,
        model: nn.Module,
        deps: RunnerDeps,
        config: EncoderConfigMixin,
    ) -> None:
        self._model = model
        self._deps = deps
        self._config = cast(RunnerConfig, config)
        self._encoder_config = config
        self._initialize_encoder_cuda_graph()

    def _initialize_encoder_cuda_graph(self) -> None:
        """Construct the encoder backend from already resolved runner settings."""
        config = self._encoder_config
        self._encoder_cuda_graph_runner = EncoderCUDAGraphRunner(
            EncoderCUDAGraphRunnerConfig(
                use_cuda_graph=config.cuda_graph_enabled,
                cuda_graph_padding_enabled=config.cuda_graph_padding_enabled,
                cuda_graph_batch_sizes=config.cuda_graph_batch_sizes,
                cuda_graph_num_tokens=config.cuda_graph_num_tokens,
                cuda_graph_seq_lens=config.cuda_graph_seq_lens,
                max_cuda_graph_batch_size=config.max_cuda_graph_batch_size,
                max_cuda_graph_num_tokens=config.max_cuda_graph_num_tokens,
                max_num_tokens=self._config.max_num_tokens,
                max_seq_len=self._config.max_seq_len,
                cuda_graph_mem_pool=None,
                is_encoder_decoder=config.is_encoder_decoder,
                use_fixed_sequence_slots=config.use_fixed_sequence_slots,
                feature_shape=config.feature_shape,
                feature_dtype=config.feature_dtype,
                fixed_seq_len=config.fixed_seq_len,
            )
        )
        if config.feature_shape is not None:
            logger.info(
                "Feature-mode encoder CUDA graphs enabled for batch sizes "
                f"{config.cuda_graph_batch_sizes} "
                f"(fixed_seq_len={config.fixed_seq_len}, "
                f"feature_shape={config.feature_shape})."
            )

        graph_runner = self._encoder_cuda_graph_runner
        # Temporary immutable settings for decoder capture and encoder scheduling.
        self._encoder_graph_shapes = (
            frozenset((bs, nt) for bs, nt, _ in graph_runner.capture_keys)
            if graph_runner.enabled
            else frozenset()
        )
        self._encoder_graph_batch_sizes = (
            tuple(graph_runner.supported_batch_sizes) if graph_runner.enabled else ()
        )
        self._encoder_graph_pad_to_limit = (
            graph_runner.padding_enabled and not graph_runner.feature_mode
        )

    def _create_attention_metadata(
        self,
        *,
        enable_context_mla_with_cached_kv: bool | None = None,
        num_heads_per_kv: int | None = None,
    ) -> AttentionMetadata:
        metadata = build_attention_metadata(
            self._model.model_config,
            max_batch_size=self._config.max_batch_size,
            max_num_tokens=self._config.max_num_tokens,
            max_beam_width=self._config.max_beam_width,
            attention_backend=self._config.attention_backend,
            attention_runtime_features=self._config.attention_runtime_features,
            mapping=self._deps.mapping,
            cache_indirection=None,
            kv_cache_manager=None,
            enable_context_mla_with_cached_kv=enable_context_mla_with_cached_kv,
            num_heads_per_kv=num_heads_per_kv,
        )
        metadata.block_ids_per_seq = None
        metadata.kv_block_ids_per_seq = None
        return metadata

    def _is_distributed_forward(self) -> bool:
        dist = self._deps.dist
        if dist is None:
            return False
        return dist.world_size > 1 or self._deps.mapping.dwdp_enabled

    def _prepare_encoder_graph_inputs(
        self,
        inputs: dict[str, Any],
        metadata: AttentionMetadata,
    ) -> EncoderPreparedInputs | None:
        """Select and prepare a token encoder graph, leaving eager fallback to the caller."""
        runner = self._encoder_cuda_graph_runner
        sequence_lengths = inputs["seq_lens"]
        if inputs.get("multi_item_part_lens") is not None:
            return None
        with runner.pad_batch(inputs, len(sequence_lengths)) as padded_inputs:
            graph_metadata, key = runner.maybe_get_cuda_graph(padded_inputs, metadata)
            if key is None:
                return None
            if runner.is_encoder_decoder:
                runner.retire_staging()
                model_inputs = runner.prepare_encoder_decoder_inputs(
                    padded_inputs, key, sequence_lengths
                )
            else:
                model_inputs = {**padded_inputs, "position_ids": padded_inputs.get("position_ids")}
            graph_metadata.prepare_encoder_cuda_graph_replay(model_inputs["seq_lens"], key[1])
            model_inputs["attn_metadata"] = graph_metadata
            return EncoderPreparedInputs(
                model_inputs,
                sequence_lengths=sequence_lengths,
                graph_key=key,
            )

    def _prepare_encoder_feature_graph_inputs(
        self,
        features: list[torch.Tensor],
        sequence_lengths: list[int],
        request_ids: list[int],
        build_metadata: Callable[[list[int], list[int]], AttentionMetadata],
    ) -> EncoderPreparedInputs | None:
        """Reuse feature graph metadata before allocating any eager metadata."""
        runner = self._encoder_cuda_graph_runner
        if not runner.enabled or not runner.feature_mode:
            return None

        fixed_seq_len = runner.config.fixed_seq_len
        for feature, sequence_length in zip(features, sequence_lengths):
            if (
                sequence_length != fixed_seq_len
                or tuple(feature.shape) != (1, *runner.config.feature_shape)
                or feature.dtype != runner.config.feature_dtype
            ):
                logger.warning_once(
                    "Encoder CUDA Graph request features do not match the "
                    "captured contract; the encoder phase stays eager.",
                    key="encoder_cuda_graph_feature_contract_warning",
                )
                return None
        graph_inputs = {"seq_lens": sequence_lengths, "input_features": features}
        with runner.pad_batch(graph_inputs, len(sequence_lengths)) as padded_inputs:
            padded_sequence_lengths = padded_inputs["seq_lens"]
            graph_metadata, key = runner.captured_graph_metadata(padded_inputs)
            if key is None:
                padded_request_ids = list(request_ids) + [
                    -(index + 1) for index in range(len(padded_sequence_lengths) - len(request_ids))
                ]
                eager_metadata = build_metadata(padded_sequence_lengths, padded_request_ids)
                graph_metadata, key = runner.maybe_get_cuda_graph(padded_inputs, eager_metadata)
            if key is None:
                return None
            padded_inputs["attn_metadata"] = graph_metadata
            if runner.needs_capture(key):
                padded_batch_size, padded_num_tokens, _ = key
                graph_metadata.prepare_encoder_cuda_graph_replay(
                    [fixed_seq_len] * padded_batch_size, padded_num_tokens
                )
            return EncoderPreparedInputs(
                padded_inputs,
                sequence_lengths=sequence_lengths,
                graph_key=key,
            )

    def _encoder_capture_shapes(self) -> Iterator[EncoderKeyType]:
        runner = self._encoder_cuda_graph_runner
        if runner.is_encoder_decoder:
            yield from sorted(runner.capture_keys, reverse=True)
            return

        batch_sizes = sorted(runner.config.cuda_graph_batch_sizes, reverse=True)
        num_tokens_list = sorted(runner.config.cuda_graph_num_tokens)
        seq_lens_list = sorted(runner.config.cuda_graph_seq_lens)
        for batch_size in batch_sizes:
            if batch_size > self._config.max_batch_size:
                continue
            for seq_len_index, max_seq_len in reversed(list(enumerate(seq_lens_list))):
                previous_seq_len = seq_lens_list[seq_len_index - 1] if seq_len_index > 0 else 0
                for token_index, num_tokens in reversed(list(enumerate(num_tokens_list))):
                    previous_num_tokens = num_tokens_list[token_index - 1] if token_index > 0 else 0
                    if (
                        num_tokens < previous_seq_len + batch_size
                        or previous_num_tokens >= batch_size * max_seq_len
                        or num_tokens > batch_size * max_seq_len
                        or max_seq_len > num_tokens
                    ):
                        continue
                    yield batch_size, num_tokens, max_seq_len

    @torch.inference_mode()
    def _capture_encoder_cuda_graphs(
        self,
        prepare_tokens: Callable[[list[int]], EncoderPreparedInputs],
        execute: Callable[[EncoderPreparedInputs], object],
        *,
        build_metadata: Callable[[list[int], list[int]], AttentionMetadata] | None = None,
    ) -> None:
        """Warm all encoder shapes before capturing any, using the same ordered pass."""
        runner = self._encoder_cuda_graph_runner
        if not runner.enabled:
            return
        with cuda_graph_capture(runner):
            runner.is_warmup_only = True
            self._run_encoder_capture_pass(prepare_tokens, execute, build_metadata)
            runner.is_warmup_only = False
            self._run_encoder_capture_pass(prepare_tokens, execute, build_metadata)

    def _run_encoder_capture_pass(
        self,
        prepare_tokens: Callable[[list[int]], EncoderPreparedInputs],
        execute: Callable[[EncoderPreparedInputs], object],
        build_metadata: Callable[[list[int], list[int]], AttentionMetadata] | None,
    ) -> None:
        runner = self._encoder_cuda_graph_runner
        operation = "warmup" if runner.is_warmup_only else "capture"
        num_processed = 0
        logger.info(f"Running encoder CUDA graph {operation} ...")
        for key in self._encoder_capture_shapes():
            if runner.is_encoder_decoder:
                sequence_lengths = runner.get_capture_warmup_sequence_lengths(key)
            else:
                sequence_lengths = runner.build_capture_sequence_lengths(*key)
            if sequence_lengths is None:
                continue
            logger.info(f"Encoder CUDA graph {operation}: key={key}")
            if runner.feature_mode:
                assert build_metadata is not None
                features = [
                    torch.zeros(
                        (1, *runner.config.feature_shape), dtype=runner.config.feature_dtype
                    )
                    for _ in sequence_lengths
                ]
                prepared = self._prepare_encoder_feature_graph_inputs(
                    features,
                    sequence_lengths,
                    list(range(len(sequence_lengths))),
                    build_metadata,
                )
            else:
                prepared = prepare_tokens(sequence_lengths)
            if prepared is None:
                continue
            execute(prepared)
            torch.cuda.synchronize()
            num_processed += 1
        logger.info(f"Completed encoder CUDA graph {operation} for {num_processed} graph shape(s).")

    def _execute_encoder_cuda_graph(
        self,
        prepared: EncoderPreparedInputs,
        forward: Callable[[dict[str, Any]], object],
    ) -> object:
        """Capture or replay a prepared encoder graph; the caller owns output semantics."""
        runner = self._encoder_cuda_graph_runner
        key = prepared.graph_key
        assert key is not None
        # Only token encoder graphs participate in MoE load-balancer iterations.
        moe_load_balancer = None if runner.feature_mode else self._deps.moe_load_balancer
        capture_outputs = None
        if runner.needs_capture(key):

            def capture_forward(inputs: dict[str, Any]) -> object:
                with MoeLoadBalancerIterContext(moe_load_balancer):
                    return forward(inputs)

            capture_outputs = runner.capture(key, capture_forward, prepared.kwargs)
        if runner.is_warmup_only:
            return capture_outputs
        with MoeLoadBalancerIterContext(moe_load_balancer):
            return runner.replay(key, prepared.kwargs)

    def release_graph(self) -> None:
        self._encoder_cuda_graph_runner.clear()


class EncoderRunner(EncoderMixin):
    """Run the direct EncodeOnly model contract.

    This runner deliberately knows nothing about ``LlmRequest``, decoder
    state, cross attention, or cross-KV resources.
    """

    def __init__(
        self,
        model: nn.Module,
        deps: RunnerDeps,
        config: EncoderRunnerConfig,
    ) -> None:
        if config.is_encoder_decoder:
            raise ValueError("EncoderRunner cannot run an encoder-decoder model.")
        self._initialize_encoder(model, deps, config)
        self._attn_metadata: AttentionMetadata | None = None

    def _setup_attention_metadata(self) -> AttentionMetadata:
        if self._attn_metadata is None:
            self._attn_metadata = self._create_attention_metadata()
        return self._attn_metadata

    def _prepare_encoder_inputs(
        self,
        input_ids: list[int] | torch.Tensor,
        sequence_lengths: list[int],
        *,
        position_ids: list[int] | torch.Tensor | None = None,
        multi_item_part_lens: list[list[int]] | None = None,
        model_inputs: dict[str, Any] | None = None,
    ) -> PreparedInputs:
        """Prepare one collected EncoderOnly batch for model execution."""
        model_inputs = model_inputs or {}
        actual_num_tokens = len(input_ids)
        batch_size = len(sequence_lengths)

        input_ids_cpu = torch.tensor(
            input_ids,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )
        if position_ids is None:
            position_ids_cpu = self._build_position_ids(
                sequence_lengths,
                actual_num_tokens,
                multi_item_part_lens,
            )
        elif isinstance(position_ids, torch.Tensor):
            position_ids_cpu = position_ids
        else:
            position_ids_cpu = torch.tensor(
                position_ids,
                dtype=torch.int,
                pin_memory=prefer_pinned(),
            )

        metadata = self._setup_attention_metadata()
        metadata.seq_lens = torch.tensor(sequence_lengths, dtype=torch.int)
        metadata.num_contexts = batch_size
        metadata.max_seq_len = self._config.max_seq_len
        metadata.request_ids = list(range(batch_size))
        if (
            multi_item_part_lens is not None
            and not self._config.attention_backend.support_multi_item_scoring()
        ):
            raise ValueError("The selected attention backend does not support multi-item scoring.")
        metadata.multi_item_part_lens = multi_item_part_lens
        metadata.prepare_encoder_only()

        self._deps.input_ids_cuda[:actual_num_tokens].copy_(input_ids_cpu, non_blocking=True)
        self._deps.position_ids_cuda[:actual_num_tokens].copy_(position_ids_cpu, non_blocking=True)
        return PreparedInputs(
            {
                **model_inputs,
                "attn_metadata": metadata,
                "input_ids": self._deps.input_ids_cuda[:actual_num_tokens],
                "position_ids": self._deps.position_ids_cuda[:actual_num_tokens].unsqueeze(0),
            }
        )

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager | None,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        **model_inputs: Any,
    ) -> EncoderPreparedInputs:
        del resource_manager, cuda_graph_lora_manager, runtime_draft_len
        reserved_inputs = model_inputs.keys() & {
            "input_ids",
            "seq_lens",
            "multi_item_part_lens",
            "attn_metadata",
            "return_context_logits",
        }
        if reserved_inputs:
            raise ValueError(
                f"Model inputs cannot override runner-managed fields: {sorted(reserved_inputs)}"
            )
        if model_inputs and self._encoder_cuda_graph_runner.enabled:
            raise NotImplementedError(
                "Model-specific encoder inputs are not supported when encoder "
                "CUDA graphs are enabled. Disable encoder CUDA graphs or omit "
                f"the inputs. Unsupported keys: {sorted(model_inputs)}"
            )
        input_ids, sequence_lengths, multi_item_part_lens = self._collect_scheduled_inputs(
            scheduled_requests
        )
        return self._prepare_encoder_batch(
            input_ids,
            sequence_lengths,
            multi_item_part_lens=multi_item_part_lens,
            model_inputs=model_inputs,
        )

    def _prepare_encoder_batch(
        self,
        input_ids: list[int],
        sequence_lengths: list[int],
        *,
        multi_item_part_lens: list[list[int]] | None = None,
        model_inputs: dict[str, Any] | None = None,
    ) -> EncoderPreparedInputs:
        inputs = {
            **(model_inputs or {}),
            "input_ids": input_ids,
            "seq_lens": sequence_lengths,
        }
        if multi_item_part_lens is not None:
            inputs["multi_item_part_lens"] = multi_item_part_lens
        graph_inputs = self._prepare_encoder_graph_inputs(inputs, self._setup_attention_metadata())
        if graph_inputs is not None:
            return graph_inputs
        prepared = self._prepare_encoder_inputs(
            input_ids,
            sequence_lengths,
            position_ids=inputs.get("position_ids"),
            multi_item_part_lens=multi_item_part_lens,
            model_inputs=inputs,
        )
        return EncoderPreparedInputs(
            prepared.kwargs,
            sequence_lengths=sequence_lengths,
        )

    @staticmethod
    def _collect_scheduled_inputs(
        scheduled_requests: ScheduledRequests,
    ) -> tuple[list[int], list[int], list[list[int]] | None]:
        requests = scheduled_requests.context_requests
        if not requests:
            raise ValueError("Encoder execution requires at least one request.")

        input_ids: list[int] = []
        sequence_lengths: list[int] = []
        multi_item_part_lens: list[list[int]] = []
        for request in requests:
            tokens = request.get_tokens(0)
            input_ids.extend(tokens)
            sequence_lengths.append(len(tokens))
            request_part_lens = getattr(request, "py_multi_item_part_lens", None)
            if request_part_lens is not None:
                multi_item_part_lens.append(request_part_lens)
        if multi_item_part_lens and len(multi_item_part_lens) != len(requests):
            raise ValueError(
                '"multi_item_part_lens" must either be provided for all requests or for none.'
            )
        return (
            input_ids,
            sequence_lengths,
            multi_item_part_lens or None,
        )

    def _build_position_ids(
        self,
        seq_lens: list[int],
        actual_num_tokens: int,
        multi_item_part_lens: list[list[int]] | None,
    ) -> torch.Tensor:
        if multi_item_part_lens is None:
            position_ids = torch.cat(
                [torch.arange(seq_len, dtype=torch.int) for seq_len in seq_lens]
            )[:actual_num_tokens]
            return maybe_pin_memory(position_ids)

        if len(multi_item_part_lens) != len(seq_lens):
            raise ValueError(
                '"multi_item_part_lens" must either be provided for all prompts or for none.'
            )
        starts = torch.tensor(
            [
                start
                for request_part_lens in multi_item_part_lens
                for start in [0] + [request_part_lens[0]] * (len(request_part_lens) - 1)
            ],
            pin_memory=prefer_pinned(),
            dtype=torch.int32,
        ).to(device=self._deps.position_ids_cuda.device, non_blocking=True)
        ends = torch.tensor(
            [
                end + 1
                for request_part_lens in multi_item_part_lens
                for end in [request_part_lens[0]]
                + [request_part_lens[0] + item_len for item_len in request_part_lens[1:]]
            ],
            pin_memory=prefer_pinned(),
            dtype=torch.int32,
        ).to(device=self._deps.position_ids_cuda.device, non_blocking=True)
        return torch_multi_arange(
            starts=starts,
            ends=ends,
            output_length=actual_num_tokens,
        )

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self._model.extra_attrs)
    def warmup(self, resource_manager: ResourceManager | None = None) -> None:
        AutoTuner.get()
        max_shape = (
            self._config.max_batch_size,
            self._config.max_num_tokens,
            self._config.max_seq_len,
        )
        warmup_shapes = list(
            dict.fromkeys(
                [
                    (1, 1, 1),
                    max_shape,
                    (1, 2, 2),
                ]
            )
        )
        self._run_warmup_shapes(warmup_shapes)
        gc.collect()
        torch.cuda.empty_cache()
        self._run_autotuner_warmup()

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self._model.extra_attrs)
    def capture_graphs(self, resource_manager: ResourceManager | None = None) -> None:
        self._capture_encoder_cuda_graphs(
            self._prepare_capture_inputs,
            self._execute_prepared,
        )
        self._run_warmup_shapes(
            [
                (
                    self._config.max_batch_size,
                    self._config.max_num_tokens,
                    self._config.max_seq_len,
                )
            ]
        )

    def _prepare_capture_inputs(self, sequence_lengths: list[int]) -> EncoderPreparedInputs:
        return self._prepare_encoder_batch([0] * sum(sequence_lengths), sequence_lengths)

    def _run_warmup_shapes(self, shapes: list[tuple[int, int, int]]) -> None:
        with cuda_graph_disabled(self._encoder_cuda_graph_runner):
            for batch_size, num_tokens, max_seq_len in shapes:
                sequence_lengths = self._encoder_cuda_graph_runner.build_capture_sequence_lengths(
                    batch_size, num_tokens, max_seq_len
                )
                if sequence_lengths is None:
                    continue
                try:
                    logger.info(
                        "Encoder general warmup: "
                        f"bs={batch_size}, nt={num_tokens}, sl={max_seq_len}"
                    )
                    prepared = self._prepare_capture_inputs(sequence_lengths)
                    self._execute_prepared(prepared)
                    torch.cuda.synchronize()
                except torch.OutOfMemoryError:
                    if self._is_distributed_forward():
                        raise
                    logger.warning(
                        "OOM during encoder general warmup with "
                        f"bs={batch_size}, nt={num_tokens}, sl={max_seq_len}. "
                        "Skipping."
                    )
                    torch.cuda.empty_cache()

    def _run_autotuner_warmup(self) -> None:
        if not self._encoder_config.enable_autotuner:
            return
        AutoTuner.get().setup_distributed_state(self._deps.mapping, self._deps.dist)
        logger.info("Running encoder autotuner warmup...")

        cache_path = os.environ.get("TLLM_AUTOTUNER_CACHE_PATH")
        with cuda_graph_disabled(self._encoder_cuda_graph_runner), autotune(cache_path=cache_path):
            sequence_lengths = self._encoder_cuda_graph_runner.build_capture_sequence_lengths(
                self._config.max_batch_size,
                self._config.max_num_tokens,
                self._config.max_seq_len,
            )
            if sequence_lengths is not None:
                prepared = self._prepare_capture_inputs(sequence_lengths)
                self._execute_prepared(prepared)
                torch.cuda.synchronize()

        logger.info(
            f"[Encoder Autotuner] Cache size after warmup is {len(AutoTuner.get().profiling_cache)}"
        )
        AutoTuner.get().print_profiling_cache()

    @nvtx_range("encoder_forward")
    def _execute_prepared(
        self,
        prepared: EncoderPreparedInputs,
        *,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        forward = partial(
            self._forward_step,
            gather_ids=prepared.gather_ids,
            gather_context_logits=gather_context_logits,
        )
        graph_runner = self._encoder_cuda_graph_runner
        with with_shared_pool(graph_runner.get_graph_pool()):
            if prepared.graph_key is None:
                with MoeLoadBalancerIterContext(self._deps.moe_load_balancer):
                    return forward(prepared.kwargs)
            graph_outputs = self._execute_encoder_cuda_graph(prepared, forward)
        if not isinstance(graph_outputs, dict):
            raise TypeError("Encoder CUDA Graph replay must return a dictionary.")
        outputs: dict[str, Any] = {}
        for name, value in graph_outputs.items():
            if isinstance(value, torch.Tensor):
                if name == "logits":
                    value = value[: len(prepared.sequence_lengths)]
                outputs[name] = value.clone()
            else:
                outputs[name] = value
        return outputs

    @torch.inference_mode()
    @with_model_extra_attrs(lambda self: self._model.extra_attrs)
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
        prepared = self.prepare_inputs(
            scheduled_requests,
            resource_manager=resource_manager,
            cuda_graph_lora_manager=cuda_graph_lora_manager,
            runtime_draft_len=runtime_draft_len,
            **model_inputs,
        )
        return self._execute_prepared(
            prepared,
            gather_context_logits=gather_context_logits,
        )

    def _forward_step(
        self,
        inputs: dict[str, Any],
        *,
        gather_ids: torch.Tensor | None,
        gather_context_logits: bool,
    ) -> dict[str, Any]:
        metadata = inputs.get("attn_metadata")
        if metadata is not None:
            metadata.on_update_kv_lens()

        outputs = self._deps.model_forward(
            **inputs,
            return_context_logits=(gather_ids is not None or gather_context_logits),
        )
        if self._config.without_logits:
            return outputs
        if isinstance(outputs, dict):
            logits = outputs.get("logits")
            if logits is None:
                return outputs
        else:
            logits = outputs
            outputs = {"logits": logits}
        if gather_ids is not None:
            outputs["logits"] = logits[gather_ids]
        return outputs
