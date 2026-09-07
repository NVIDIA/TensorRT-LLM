# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared execution template for model families that do not use a KV cache."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.vanilla import VanillaAttentionMetadata
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.models.modeling_multimodal_mixin import _build_request_multimodal_input
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.speculative import SpecMetadata, get_spec_metadata
from tensorrt_llm._torch.utils import set_per_request_prefill_cuda_graph_flag
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig, PrefillCudaGraphBackend
from tensorrt_llm.mapping import Mapping

from ..lora import LoraParamBuilder
from ..metadata import build_attention_metadata, update_spec_metadata
from .common import (
    apply_position_id_offset,
    get_all_rank_num_tokens,
    get_padding_params,
    prepare_multimodal_indices,
    set_spec_metadata_all_rank_num_tokens,
    ship_multimodal_indices,
)
from .interface import PreparedInputs, RunnerConfig, RunnerDeps


@dataclass(frozen=True)
class NoCacheRunnerConfig(RunnerConfig):
    """Settings used to prepare scheduled requests without a KV cache."""

    enable_attention_dp: bool
    prefill_cuda_graph_backend: PrefillCudaGraphBackend
    prefill_cuda_graph_num_tokens: list[int]
    mm_encoder_cache_enabled: bool
    spec_config: DecodingBaseConfig | None
    is_draft_model: bool
    num_seq_slots: int | None
    original_max_draft_len: int
    original_max_total_draft_tokens: int
    spec_dec_max_total_draft_tokens: int


def prepare_no_cache_inputs(
    scheduled_requests: ScheduledRequests,
    attn_metadata: AttentionMetadata,
    spec_metadata: SpecMetadata | None = None,
    resource_manager: ResourceManager | None = None,
    *,
    model: nn.Module,
    dist: Distributed | None,
    mapping: Mapping,
    enable_attention_dp: bool,
    enable_spec_decode: bool,
    cuda_graph_lora_manager: CudaGraphLoraManager | None,
    runtime_draft_len: int,
    max_num_tokens: int,
    max_seq_len: int,
    prefill_cuda_graph_backend: PrefillCudaGraphBackend,
    prefill_cuda_graph_num_tokens: list[int],
    mm_encoder_cache_enabled: bool,
    input_ids_cuda: torch.Tensor,
    position_ids_cuda: torch.Tensor,
    gather_ids_cuda: torch.Tensor | None,
    draft_tokens_cuda: torch.Tensor | None,
    lora: LoraParamBuilder,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """
    Prepare inputs for Pytorch Model.
    """
    sequence_lengths = []
    input_ids = []
    gather_ids = []
    position_ids = []
    multi_modal_data = []
    draft_lens = []
    request_ids = []
    multimodal_params_list = []

    for request in scheduled_requests.context_requests:
        prompt_tokens = request.get_tokens(0)
        # Start offset of this request's tokens within the flattened
        # input_ids (see _prepare_tp_inputs for rationale).
        context_start_idx = len(input_ids)
        input_ids.extend(prompt_tokens)
        request_ids.append(request.py_request_id)
        if request.position_ids is None:
            position_ids.extend(range(len(prompt_tokens)))
        else:
            position_ids.extend(request.position_ids)
        gather_ids.append(len(input_ids) - 1)
        sequence_lengths.append(len(prompt_tokens))
        draft_lens.append(0)
        multimodal_embedding = request.multimodal_embedding
        if multimodal_embedding is not None:
            multi_modal_data.append(multimodal_embedding)

        # Multimodal
        if request.py_multimodal_data is not None:
            multimodal_params = MultimodalParams(
                multimodal_input=_build_request_multimodal_input(
                    request,
                    mm_encoder_cache_enabled,
                ),
                multimodal_data=request.py_multimodal_data,
                mm_item_order=getattr(request, "py_mm_item_order", None),
                input_ids_start_offset=context_start_idx,
            )
            multimodal_params.to_device(
                "multimodal_data",
                "cuda",
                pin_memory=prefer_pinned(),
            )
            multimodal_params_list.append(multimodal_params)

        request.py_batch_idx = request.py_seq_slot

    num_tokens = len(input_ids)
    assert num_tokens <= max_num_tokens, "num_tokens should be less than or equal to max_num_tokens"
    # Compute MM/text token indices on CPU input_ids so that
    # fuse_input_embeds can skip its torch.where host sync. Must run before
    # the input_ids list is rebound to a tensor below. Skipped when
    # ``model`` is a vision encoder (no ``config.vocab_size`` to filter
    # against, and its forward doesn't consume the indices anyway); this
    # is a structural check on the model rather than a flag lookup, so it
    # naturally extends to any future "LLM-less" engine setup.
    _model_config = getattr(model, "config", None)
    if len(multimodal_params_list) > 0 and getattr(_model_config, "vocab_size", None) is not None:
        text_token_indices_cpu, mm_token_indices_cpu = prepare_multimodal_indices(
            input_ids,
            model=model,
        )
    else:
        text_token_indices_cpu = None
        mm_token_indices_cpu = None
    input_ids = torch.tensor(
        input_ids,
        dtype=torch.int,
        pin_memory=prefer_pinned(),
    )
    input_ids_cuda[:num_tokens].copy_(input_ids, non_blocking=True)

    position_ids = apply_position_id_offset(position_ids, model=model)
    position_ids = torch.tensor(
        position_ids,
        dtype=torch.int,
        pin_memory=prefer_pinned(),
    )
    position_ids_cuda[:num_tokens].copy_(position_ids, non_blocking=True)
    if enable_spec_decode:
        gather_ids_cuda[: len(gather_ids)].copy_(
            torch.tensor(
                gather_ids,
                dtype=torch.int,
                pin_memory=prefer_pinned(),
            ),
            non_blocking=True,
        )

    if not attn_metadata.is_cuda_graph:
        # No need to overwrite seq lens when using CUDA graphs -
        # CUDA graphs are only used for pure decoding batches
        # and have static batch size, so the seqlens never change.
        # Note that it's important to not free the seq_lens_cuda
        # buffer once the graph has been captured also - this will invalidate
        # the graph and force an expensive recapture.
        attn_metadata.seq_lens = torch.tensor(
            sequence_lengths,
            dtype=torch.int,
            pin_memory=prefer_pinned(),
        )

    attn_metadata.num_contexts = scheduled_requests.num_context_requests

    attn_all_rank_num_tokens = get_all_rank_num_tokens(
        attn_metadata,
        enable_attention_dp=enable_attention_dp,
        mapping=mapping,
        dist=dist,
    )
    padded_num_tokens, can_run_prefill_cuda_graph, attn_all_rank_num_tokens = get_padding_params(
        num_tokens,
        attn_metadata.num_contexts,
        attn_all_rank_num_tokens,
        dist=dist,
        enable_attention_dp=enable_attention_dp,
        prefill_cuda_graph_backend=prefill_cuda_graph_backend,
        prefill_cuda_graph_num_tokens=prefill_cuda_graph_num_tokens,
    )
    set_per_request_prefill_cuda_graph_flag(can_run_prefill_cuda_graph)
    attn_metadata.padded_num_tokens = padded_num_tokens if padded_num_tokens != num_tokens else None

    if enable_attention_dp:
        attn_metadata.all_rank_num_tokens = attn_all_rank_num_tokens

    virtual_num_tokens = num_tokens
    if attn_metadata.padded_num_tokens is not None:
        input_ids_cuda[num_tokens:padded_num_tokens].fill_(0)
        position_ids_cuda[num_tokens:padded_num_tokens].fill_(0)
        virtual_num_tokens = padded_num_tokens

    # this is for no cache attention, not for dummy attention
    if attn_metadata.kv_cache_manager is None:
        assert isinstance(
            attn_metadata,
            (VanillaAttentionMetadata, TrtllmAttentionMetadata),
        ), "Only vanilla and trtllm attention metadata are supported for no cache attention for now"
        attn_metadata.max_seq_len = max_seq_len
        attn_metadata.request_ids = request_ids
        attn_metadata.prepare()

    lora_params = lora.build(
        scheduled_requests,
        attn_metadata,
        cuda_graph_lora_manager=cuda_graph_lora_manager,
        enable_spec_decode=enable_spec_decode,
        runtime_draft_len=runtime_draft_len,
    )

    inputs = {
        "attn_metadata": attn_metadata,
        "input_ids": input_ids_cuda[:virtual_num_tokens],
        "position_ids": position_ids_cuda[:virtual_num_tokens].unsqueeze(0),
        "inputs_embeds": None,
        "multimodal_params": multimodal_params_list,
        "resource_manager": resource_manager,
    }

    if mm_token_indices_cpu is not None:
        # No extend/draft tokens in the no-cache path, so num_tokens covers
        # the full range and the helper's arange/cat branch is skipped.
        ship_multimodal_indices(
            inputs,
            mm_token_indices_cpu=mm_token_indices_cpu,
            text_token_indices_cpu=text_token_indices_cpu,
            num_ctx_tokens=num_tokens,
            total_num_tokens=num_tokens,
        )

    if bool(lora_params):
        inputs["lora_params"] = lora_params

    if spec_metadata is not None:
        total_draft_lens = sum(draft_lens)
        spec_metadata.draft_tokens = draft_tokens_cuda[:total_draft_lens]
        spec_metadata.request_ids = request_ids
        spec_metadata.gather_ids = gather_ids_cuda[: len(gather_ids)]
        spec_metadata.num_generations = len(scheduled_requests.generation_requests)
        spec_metadata.num_tokens = num_tokens
        spec_metadata.seq_lens = sequence_lengths
        spec_metadata.prepare()
        inputs["spec_metadata"] = spec_metadata

    # support attention dp
    if enable_attention_dp:
        assert dist is not None, "attention DP requires a distributed communicator"
        if spec_metadata is not None:
            all_rank_num_tokens = dist.tp_cp_allgather_int64(
                [
                    attn_metadata.num_tokens,
                    spec_metadata.num_tokens,
                    len(sequence_lengths),
                    spec_metadata.num_generations,
                ]
            ).tolist()
            attn_metadata.all_rank_num_tokens = [item[0] for item in all_rank_num_tokens]
            set_spec_metadata_all_rank_num_tokens(
                spec_metadata,
                [item[1] for item in all_rank_num_tokens],
                [item[2] for item in all_rank_num_tokens],
                [item[3] for item in all_rank_num_tokens],
            )
        else:
            all_rank_num_tokens = dist.tp_cp_allgather_int64([attn_metadata.num_tokens])[
                :, 0
            ].tolist()
            attn_metadata.all_rank_num_tokens = all_rank_num_tokens

    return inputs, None


class NoCacheRunner(ABC):
    """Common execution template for model families that do not use a KV cache."""

    def __init__(
        self,
        model: nn.Module,
        deps: RunnerDeps,
        config: NoCacheRunnerConfig,
    ) -> None:
        self._model = model
        self._deps = deps
        self._config = config
        self._attn_metadata: AttentionMetadata | None = None

    def setup_attn_metadata(self) -> AttentionMetadata:
        if self._attn_metadata is not None:
            return self._attn_metadata

        self._attn_metadata = build_attention_metadata(
            self._model.model_config,
            max_batch_size=self._config.max_batch_size,
            max_num_tokens=self._config.max_num_tokens,
            max_beam_width=self._config.max_beam_width,
            attention_backend=self._config.attention_backend,
            attention_runtime_features=self._config.attention_runtime_features,
            mapping=self._deps.mapping,
            cache_indirection=self._deps.cache_indirection,
            kv_cache_manager=None,
        )
        self._attn_metadata.block_ids_per_seq = None
        self._attn_metadata.kv_block_ids_per_seq = None
        return self._attn_metadata

    def setup_spec_metadata(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        attn_metadata: AttentionMetadata,
        runtime_draft_len: int,
    ) -> SpecMetadata | None:
        runner_config = self._config
        spec_config = runner_config.spec_config
        if spec_config is None:
            return None

        spec_resource_manager = resource_manager.get_resource_manager(
            ResourceManagerType.SPEC_RESOURCE_MANAGER
        )
        spec_tree_manager = getattr(spec_resource_manager, "spec_tree_manager", None)
        spec_metadata = get_spec_metadata(
            spec_config,
            self._model.config,
            runner_config.max_batch_size,
            max_num_tokens=runner_config.max_num_tokens,
            spec_resource_manager=spec_resource_manager,
            is_draft_model=runner_config.is_draft_model,
            max_seq_len=runner_config.max_seq_len,
            num_seq_slots=runner_config.num_seq_slots,
        )
        assert spec_metadata is not None
        update_spec_metadata(
            spec_metadata,
            scheduled_requests,
            attn_metadata,
            spec_tree_manager=spec_tree_manager,
            runtime_draft_len=runtime_draft_len,
            runtime_tokens_per_gen_step=spec_config.get_runtime_tokens_per_gen_step(
                runtime_draft_len
            ),
            is_draft_model=runner_config.is_draft_model,
            attention_backend=runner_config.attention_backend,
            original_max_draft_len=runner_config.original_max_draft_len,
            original_max_total_draft_tokens=(runner_config.original_max_total_draft_tokens),
            spec_dec_max_total_draft_tokens=(runner_config.spec_dec_max_total_draft_tokens),
        )
        return spec_metadata

    def prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
    ) -> PreparedInputs:
        runner_config = self._config
        attn_metadata = self.setup_attn_metadata()
        spec_metadata = self.setup_spec_metadata(
            scheduled_requests,
            resource_manager,
            attn_metadata,
            runtime_draft_len,
        )
        inputs, gather_ids = prepare_no_cache_inputs(
            scheduled_requests,
            attn_metadata,
            spec_metadata,
            resource_manager,
            model=self._model,
            dist=self._deps.dist,
            mapping=self._deps.mapping,
            enable_attention_dp=runner_config.enable_attention_dp,
            enable_spec_decode=runner_config.spec_config is not None,
            cuda_graph_lora_manager=cuda_graph_lora_manager,
            runtime_draft_len=runtime_draft_len,
            max_num_tokens=runner_config.max_num_tokens,
            max_seq_len=runner_config.max_seq_len,
            prefill_cuda_graph_backend=runner_config.prefill_cuda_graph_backend,
            prefill_cuda_graph_num_tokens=runner_config.prefill_cuda_graph_num_tokens,
            mm_encoder_cache_enabled=runner_config.mm_encoder_cache_enabled,
            input_ids_cuda=self._deps.input_ids_cuda,
            position_ids_cuda=self._deps.position_ids_cuda,
            gather_ids_cuda=self._deps.gather_ids_cuda,
            draft_tokens_cuda=self._deps.draft_tokens_cuda,
            lora=self._deps.lora,
        )
        return PreparedInputs(inputs, gather_ids)

    def warmup(self, resource_manager: ResourceManager) -> None:
        return

    def capture_graphs(self, resource_manager: ResourceManager) -> None:
        return

    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        *,
        resource_manager: ResourceManager,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        runtime_draft_len: int,
        moe_load_balancer: MoeLoadBalancer | None,
        gather_context_logits: bool,
    ) -> dict[str, Any]:
        prepared = self.prepare_inputs(
            scheduled_requests,
            resource_manager=resource_manager,
            cuda_graph_lora_manager=cuda_graph_lora_manager,
            runtime_draft_len=runtime_draft_len,
        )
        with MoeLoadBalancerIterContext(moe_load_balancer):
            return self._forward_step(
                prepared.kwargs,
                scheduled_requests,
                gather_ids=prepared.gather_ids,
                gather_context_logits=gather_context_logits,
            )

    @abstractmethod
    def _forward_step(
        self,
        inputs: dict[str, Any],
        scheduled_requests: ScheduledRequests,
        *,
        gather_ids: torch.Tensor | None = None,
        gather_context_logits: bool = False,
    ) -> dict[str, Any]:
        """Run the model-family-specific forward step."""
