# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LoRA parameter construction for the PyTorch model engine."""

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm._torch.peft.lora.cuda_graph_lora_manager import CudaGraphLoraManager
from tensorrt_llm._torch.peft.lora.manager import LoraModelConfig
from tensorrt_llm._torch.peft.lora.ownership import DecoderLoraOwner
from tensorrt_llm._torch.peft.lora.ownership import resolve as resolve_owner
from tensorrt_llm._utils import torch_dtype_to_str
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.inputs.registry import InputProcessor
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
from tensorrt_llm.logger import logger

from ..resource_manager import PeftCacheManager
from ..scheduler import ScheduledRequests
from .multimodal import is_multimodal


def resolve_lora_owner(
    model: nn.Module, input_processor: InputProcessor
) -> tuple[DecoderLoraOwner, nn.Module]:
    """Bind the declared decoder owner before loading any adapters.

    Multimodal models must select a decoder submodule. Undeclared text-only
    models own adapters at the model root.
    """
    return resolve_owner(model, require_submodule=is_multimodal(model, input_processor))


def make_lora_model_config(
    model: nn.Module,
    lora_target_modules: list[str],
    trtllm_modules_to_hf_modules: dict[str, str],
    swap_gate_up_proj_lora_b_weight: bool = True,
    *,
    owner: DecoderLoraOwner,
) -> LoraModelConfig:
    """Describe the model to the LoRA stack, for the engine to record."""
    return LoraModelConfig(
        lora_target_modules=lora_target_modules,
        trtllm_modules_to_hf_modules=trtllm_modules_to_hf_modules,
        hidden_size=model.config.hidden_size,
        dtype=torch_dtype_to_str(model.config.torch_dtype),
        swap_gate_up_proj_lora_b_weight=swap_gate_up_proj_lora_b_weight,
        owner=owner,
    )


def make_cuda_graph_lora_manager(
    model: nn.Module,
    lora_config: LoraConfig,
    lora_model_config: LoraModelConfig,
    max_batch_size: int,
    max_tokens_per_seq: int,
    max_num_tokens: int,
) -> CudaGraphLoraManager:
    """Build a graph manager from the decoder module selected during initialization."""
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
        max_num_tokens=max_num_tokens,
    )

    logger.info(
        f"Initialized CUDA Graph LoRA manager, "
        f"max {max_lora_size} adapters, max rank {lora_config.max_lora_rank}"
    )
    return manager


class LoraParamBuilder:
    """Own LoRA preparation state and build per-iteration model parameters.

    ``model_config`` is set once LoRA adapters are configured. The optional
    graph manager is replaced whenever executor resources are rebuilt; None
    selects eager preparation only. Graph selection and parameter preparation
    use the same manager. Speculative execution decisions remain per-call inputs.
    """

    def __init__(
        self,
        *,
        attn_backend: type[AttentionMetadata],
        cuda_graph_manager: CudaGraphLoraManager | None,
        spec_config: DecodingBaseConfig | None = None,
    ) -> None:
        self._spec_config = spec_config
        self._attn_backend = attn_backend
        self.cuda_graph_manager = cuda_graph_manager
        self.model_config: LoraModelConfig | None = None

    def build(
        self,
        scheduled_requests: ScheduledRequests,
        attn_metadata: AttentionMetadata,
        *,
        enable_spec_decode: bool = False,
        runtime_draft_len: int = 0,
        peft_cache_manager: PeftCacheManager | None = None,
        maybe_graph: bool = False,
        use_lora_graph: bool = False,
    ) -> dict | None:
        """Get LoRA parameters from scheduled requests.

        Uses CUDA Graph compatible mode in decode only batch, otherwise falls back to eager mode.

        Returns:
            Dictionary containing LoRA parameters, or None if no LoRA requests
        """
        cuda_graph_lora_manager = self.cuda_graph_manager
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
        request_list = scheduled_requests.all_requests()
        if not request_list:
            return {}

        # Map each distinct lora_task_id (in first-seen order) to a column 1..num_adapters;
        # column 0 stays zero for base requests and adapters with empty config lists.
        adapter_to_col: dict[int, int] = {}
        adapter_configs: list[list[TaskLayerModuleConfig]] = []
        req_adapter_cols: list[int] = []

        for request in request_list:
            task_id = request.lora_task_id
            if task_id is None:
                req_adapter_cols.append(0)
                continue

            col = adapter_to_col.get(task_id)
            if col is None:
                configs = peft_table[task_id]
                if configs:
                    col = len(adapter_configs) + 1
                    adapter_configs.append(configs)
                else:
                    col = 0
                adapter_to_col[task_id] = col
            req_adapter_cols.append(col)

        if not adapter_configs:
            return {}

        num_cols = len(adapter_configs) + 1
        ptr_row_len = 3 * num_cols
        lora_params: dict = {}
        entry_rows: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
        entry_targets: list[tuple[dict, int]] = []
        sizes_by_entry: list[list[int]] = []
        ptrs_by_entry: list[list[int]] = []

        # Populate an entry-major [num_entries, num_cols] table once per unique adapter so
        # index_select along dim=1 yields contiguous per-entry [num_requests] rows directly.
        for adapter_col, configs in enumerate(adapter_configs, start=1):
            ptr_base = adapter_col * 3
            for module in configs:
                layer_id = module.layer_id
                module_id = module.module_id
                key = (layer_id, module_id)
                pair = entry_rows.get(key)
                if pair is None:
                    sizes_row = [0] * num_cols
                    ptrs_row = [0] * ptr_row_len
                    pair = (sizes_row, ptrs_row)
                    entry_rows[key] = pair
                    sizes_by_entry.append(sizes_row)
                    ptrs_by_entry.append(ptrs_row)
                    layer_dict = lora_params.get(layer_id)
                    if layer_dict is None:
                        layer_dict = {}
                        lora_params[layer_id] = layer_dict
                    layer_dict[module_id] = None
                    entry_targets.append((layer_dict, module_id))
                else:
                    sizes_row, ptrs_row = pair

                sizes_row[adapter_col] = module.adapter_size
                ptrs_row[ptr_base] = module.weights_in_pointer
                ptrs_row[ptr_base + 1] = module.weights_out_pointer
                scaling_vec_pointer = module.scaling_vec_pointer
                ptrs_row[ptr_base + 2] = 0 if scaling_vec_pointer is None else scaling_vec_pointer

        num_entries = len(entry_targets)
        num_requests = len(req_adapter_cols)
        req_idx = torch.tensor(req_adapter_cols, dtype=torch.int64, device="cpu")
        sizes_table = torch.tensor(sizes_by_entry, dtype=torch.int32, device="cpu")
        ptrs_table = torch.tensor(ptrs_by_entry, dtype=torch.int64, device="cpu").view(
            num_entries, num_cols, 3
        )

        sizes_out = sizes_table.index_select(1, req_idx).contiguous().unbind(0)
        ptrs_out = (
            ptrs_table.index_select(1, req_idx)
            .contiguous()
            .view(num_entries, 3 * num_requests)
            .unbind(0)
        )

        for (layer_dict, module_id), size_t, ptr_t in zip(entry_targets, sizes_out, ptrs_out):
            layer_dict[module_id] = {
                "adapter_size": size_t,
                "weight_pointers": ptr_t,
            }

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
