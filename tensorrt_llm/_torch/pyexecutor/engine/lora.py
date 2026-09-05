# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LoRA parameter construction for the PyTorch model engine."""

import torch
from torch import nn

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._utils import torch_dtype_to_str
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
from tensorrt_llm.logger import logger

from ..resource_manager import PeftCacheManager
from ..scheduler import ScheduledRequests


def make_lora_model_config(
    model: nn.Module,
    lora_target_modules: list[str],
    trtllm_modules_to_hf_modules: dict[str, str],
    swap_gate_up_proj_lora_b_weight: bool = True,
) -> LoraModelConfig:
    """Describe the model to the LoRA stack, for the engine to record."""
    return LoraModelConfig(
        lora_target_modules=lora_target_modules,
        trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
        hidden_size=model.config.hidden_size,
        dtype=torch_dtype_to_str(model.config.torch_dtype),
        swap_gate_up_proj_lora_b_weight=swap_gate_up_proj_lora_b_weight,
    )


def make_cuda_graph_lora_manager(
    model: nn.Module,
    lora_config: LoraConfig,
    lora_model_config: LoraModelConfig,
    max_batch_size: int,
    max_tokens_per_seq: int,
) -> CudaGraphLoraManager:
    """Build the CUDA-graph LoRA manager. Only call this when graphs are enabled."""
    max_lora_size = lora_config.max_loras or 8  # Default fallback
    manager = CudaGraphLoraManager(
        max_lora_size=max_lora_size,
        max_batch_size=max_batch_size,
        max_lora_rank=lora_config.max_lora_rank,
        model=model,
        lora_model_config=lora_model_config,
        overlap_lora_and_base=lora_config.overlap_lora_and_base,
        device="cuda",
        max_tokens_per_seq=max_tokens_per_seq,
    )

    logger.info(
        f"Initialized CUDA Graph LoRA manager, "
        f"max {max_lora_size} adapters, max rank {lora_config.max_lora_rank}"
    )
    return manager


class LoraParamBuilder:
    """Builds the per-iteration ``lora_params`` dict for the model forward.

    Holds no model, no CUDA-graph manager and no device buffers -- only the two
    construction-time values the token-count logic needs. The engine passes
    everything else in on each :meth:`build` call.
    """

    def __init__(
        self, *, spec_config: DecodingBaseConfig | None, attn_backend: type[AttentionMetadata]
    ) -> None:
        self._spec_config = spec_config
        self._attn_backend = attn_backend

    def build(
        self,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        *,
        cuda_graph_lora_manager: CudaGraphLoraManager | None,
        enable_spec_decode: bool,
        runtime_draft_len: int,
        peft_cache_manager: PeftCacheManager | None = None,
        maybe_graph: bool = False,
        use_lora_graph: bool = False,
    ) -> dict | None:
        """Get LoRA parameters from scheduled requests.

        Uses CUDA Graph compatible mode in decode only batch, otherwise falls back to eager mode.

        Returns:
            Dictionary containing LoRA parameters, or None if no LoRA requests
        """
        use_cuda_graph_mode = cuda_graph_lora_manager is not None and maybe_graph

        if use_cuda_graph_mode:
            if not use_lora_graph:
                cuda_graph_lora_manager.prepare_base_only_batch(peft_cache_manager)
                return None
            # For spec decode verification (non-extend_ctx), each sequence has
            # runtime_draft_len + 1 tokens in the forward pass.
            tokens_per_seq = 1
            if (
                enable_spec_decode
                and runtime_draft_len > 0
                and self._spec_config.is_linear_tree
                and not self._spec_config.spec_dec_mode.extend_ctx(self._attn_backend)
            ):
                tokens_per_seq = runtime_draft_len + 1
            return cuda_graph_lora_manager.prepare_cuda_graph_lora_params(
                scheduled_requests, attn_metadata, peft_cache_manager, tokens_per_seq
            )
        else:
            if cuda_graph_lora_manager is not None:
                cuda_graph_lora_manager.adapter_slot_manager.remove_evicted_slots_in_cpp(
                    peft_cache_manager
                )
            peft_table = (
                peft_cache_manager.get_and_reset_batch_peft_table()
                if peft_cache_manager is not None
                else None
            )
            lora_params = peft_table and self._build_eager(
                scheduled_requests,
                attn_metadata,
                peft_table,
                enable_spec_decode=enable_spec_decode,
                runtime_draft_len=runtime_draft_len,
            )
            if lora_params:
                lora_params["data_type"] = peft_cache_manager.data_type
            return lora_params

    def _build_eager(
        self,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        peft_table: dict[int, list[TaskLayerModuleConfig]],
        *,
        enable_spec_decode: bool,
        runtime_draft_len: int,
    ) -> dict:
        """Eager mode LoRA parameter preparation logic.

        lora_params: dict
        {
            layer_id: dict
            {
                module_id: dict
                {
                    adapter_size: torch tensor: int
                    weight_pointers: torch tensor: int64
                }
            }
        }
        """
        lora_params = {}
        tmp_lora_params = {}

        request_list = scheduled_requests.all_requests()

        # trace all requests to get the union set of the lora params
        for request in request_list:
            if request.lora_task_id is None:
                continue

            layer_module_configs = peft_table[request.lora_task_id]

            for module in layer_module_configs:
                module_id = module.module_id
                layer_id = module.layer_id

                if layer_id not in lora_params:
                    lora_params[layer_id] = {}
                if module_id not in lora_params[layer_id]:
                    lora_params[layer_id][module_id] = {
                        "adapter_size": [],
                        "weight_pointers": [],
                    }

                scaling_vec_pointer = module.scaling_vec_pointer
                if scaling_vec_pointer is None:
                    scaling_vec_pointer = 0
                tmp_lora_params[(request.py_request_id, layer_id, module_id)] = {
                    "adapter_size": [module.adapter_size],
                    "weight_pointers": [
                        module.weights_in_pointer,
                        module.weights_out_pointer,
                        scaling_vec_pointer,
                    ],
                }

        for request in request_list:
            # Need to set default values for this case
            if request.lora_task_id is None:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        current_lora_params = lora_params[layer_id][module_id]
                        current_lora_params["adapter_size"].append(0)
                        current_lora_params["weight_pointers"] += [0, 0, 0]

            else:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        current_tmp_lora_params = tmp_lora_params.get(
                            (request.py_request_id, layer_id, module_id), None
                        )
                        current_lora_params = lora_params[layer_id][module_id]
                        if current_tmp_lora_params is None:
                            current_lora_params["adapter_size"].append(0)
                            current_lora_params["weight_pointers"] += [0, 0, 0]
                        else:
                            current_lora_params["adapter_size"] += current_tmp_lora_params[
                                "adapter_size"
                            ]
                            current_lora_params["weight_pointers"] += current_tmp_lora_params[
                                "weight_pointers"
                            ]

        for layer_id in lora_params:
            for module_id in lora_params[layer_id]:
                current_lora_params = lora_params[layer_id][module_id]
                current_lora_params["adapter_size"] = torch.IntTensor(
                    current_lora_params["adapter_size"]
                )
                current_lora_params["weight_pointers"] = torch.LongTensor(
                    current_lora_params["weight_pointers"]
                )

        if lora_params:
            host_request_types = attn_metadata.host_request_types
            prompt_lens_cpu = attn_metadata.prompt_lens_cpu
            num_seqs = attn_metadata.num_seqs
            num_contexts = attn_metadata.num_contexts
            num_generations = attn_metadata.num_generations

            # During spec decode verification (non-extend_ctx mode), each
            # generation request processes (runtime_draft_len + 1) tokens at
            # once. The LoRA op's C++ kernel only advances 1 token per
            # kGENERATION request, so we re-label generation requests as
            # kCONTEXT and set prompt_lens_cpu to the actual per-request token
            # count so the kernel correctly expands LoRA weights for all tokens.
            if (
                enable_spec_decode
                and runtime_draft_len > 0
                and self._spec_config.is_linear_tree
                and not self._spec_config.spec_dec_mode.extend_ctx(self._attn_backend)
                and num_generations > 0
            ):
                tokens_per_req = runtime_draft_len + 1
                host_request_types = host_request_types.clone()
                host_request_types[num_contexts:num_seqs].fill_(0)  # kCONTEXT
                prompt_lens_cpu = prompt_lens_cpu.clone()
                prompt_lens_cpu[num_contexts:num_seqs].fill_(tokens_per_req)

            lora_params["host_request_types"] = host_request_types
            lora_params["prompt_lens_cpu"] = prompt_lens_cpu
            lora_params["num_seqs"] = num_seqs

        return lora_params
