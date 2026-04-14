# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm.bindings.internal.runtime import TaskLayerModuleConfig
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig
from tensorrt_llm.lora_helper import LoraConfig
from tensorrt_llm.lora_manager import LoraManager, LoraModelConfig
from tensorrt_llm.runtime import ModelConfig as ModelConfigPython

from ...._utils import binding_to_str_dtype
from ....logger import logger
from ....mapping import Mapping
from ..llm_request import LlmRequest
from ..scheduler import ScheduledRequests
from .base import BaseResourceManager

ModelConfigCpp = tensorrt_llm.bindings.ModelConfig
PeftCacheManagerCpp = tensorrt_llm.bindings.internal.batch_manager.PeftCacheManager
WorldConfig = tensorrt_llm.bindings.WorldConfig


class PeftCacheManager(BaseResourceManager):
    def __init__(
        self,
        peft_cache_config: PeftCacheConfig,
        lora_config: LoraConfig,
        model_config: ModelConfigCpp,
        world_config: WorldConfig | None = None,
        execution_stream: Optional[torch.cuda.Stream] = None,
    ):
        import tensorrt_llm.bindings as _tb

        peft_cache_config = peft_cache_config._to_pybind()

        peft_cache_manager_config = _tb.PeftCacheManagerConfig(
            num_host_module_layer=peft_cache_config.num_host_module_layer,
            num_device_module_layer=peft_cache_config.num_device_module_layer,
            optimal_adapter_size=peft_cache_config.optimal_adapter_size,
            max_adapter_size=peft_cache_config.max_adapter_size,
            num_put_workers=peft_cache_config.num_put_workers,
            num_ensure_workers=peft_cache_config.num_ensure_workers,
            num_copy_streams=peft_cache_config.num_copy_streams,
            max_pages_per_block_host=peft_cache_config.max_pages_per_block_host,
            max_pages_per_block_device=peft_cache_config.max_pages_per_block_device,
            device_cache_percent=peft_cache_config.device_cache_percent,
            host_cache_size=peft_cache_config.host_cache_size,
            lora_prefetch_dir=peft_cache_config.lora_prefetch_dir,
        )

        if world_config is None:
            world_config = _tb.WorldConfig()

        BufferManager = tensorrt_llm.bindings.internal.runtime.BufferManager
        buffer_manager_stream = (
            execution_stream.cuda_stream
            if execution_stream is not None
            else torch.cuda.current_stream().cuda_stream
        )
        buffer_manager = BufferManager(buffer_manager_stream, True)
        logger.info(f"[PeftCacheManager] buffer_manager_stream: {buffer_manager_stream}")
        self.impl = PeftCacheManagerCpp(
            config=peft_cache_manager_config,
            model_config=model_config,
            world_config=world_config,
            buffer_manager=buffer_manager,
        )
        self._lora_config = lora_config
        self._lora_model_config = LoraModelConfig(
            lora_config.lora_target_modules,
            lora_config.trtllm_modules_to_hf_modules,
            model_config.hidden_size,
            binding_to_str_dtype(model_config.data_type),
            lora_config.swap_gate_up_proj_lora_b_weight,
        )
        mapping = Mapping(
            world_size=world_config.size,
            rank=world_config.rank,
            tp_size=world_config.tensor_parallelism,
            pp_size=world_config.pipeline_parallelism,
            gpus_per_node=world_config.gpus_per_node,
        )
        self._lora_manager = LoraManager(
            mapping=mapping,
            model_config=ModelConfigPython.from_model_config_cpp(model_config),
            cpp_peft_cache_manager=self.impl,
        )

        self._batch_peft_table: Optional[Dict[int, list[TaskLayerModuleConfig]]] = (
            None  # task_id -> layer-module-configs mapping for the current batch
        )

    def get_lora_manager(self) -> LoraManager:
        return self._lora_manager

    def add_request_peft(self, request: LlmRequest):
        if request.lora_task_id is not None:
            is_task_cached = self.impl.is_task_cached(request.lora_task_id)
            if is_task_cached:
                # PeftCacheManager::addRequestPeft in CPP doesn't allow having only one of [config tensor, weights
                # tensor] without the other. Since there's no need for any of them when the LoRA adapter is already
                # cached, we can safely remove both from the request.
                request.remove_lora_tensors()
            elif request.lora_weights is None and request.py_lora_path:
                self._lora_manager.load_from_ckpt(
                    [request.py_lora_path],
                    model_config=self._lora_model_config,
                    uids=[request.lora_task_id],
                    ckpt_source=self._lora_config.lora_ckpt_source,
                )
                request.lora_weights = self._lora_manager.cpp_lora_weights[request.lora_task_id]

            # PeftCacheManager CPP implementation expects an extra dim at index 0
            if request.lora_weights is not None:
                request.lora_weights = request.lora_weights.unsqueeze(0)
            if request.lora_config is not None:
                request.lora_config = request.lora_config.unsqueeze(0)
        self.impl.add_request_peft(request, True)

    def ensure_batch(
        self,
        context_batch: List[LlmRequest],
        generation_batch: List[LlmRequest],
        reset_gpu_cache: bool = False,
    ) -> List[LlmRequest]:
        return self.impl.ensure_batch(context_batch, generation_batch, reset_gpu_cache)

    def get_max_resource_count(self) -> int:
        return 0

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests
        for req in context_batch:
            self.add_request_peft(req)

        self._batch_peft_table, _ = self.impl.ensure_batch_map_task_id(
            context_batch, generation_batch, False
        )

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        self.impl.mark_request_done(request)

    def shutdown(self):
        pass

    def get_and_reset_batch_peft_table(self) -> Dict[int, list[TaskLayerModuleConfig]]:
        batch_peft_table = self._batch_peft_table
        self._batch_peft_table = None
        return batch_peft_table

    def is_task_cached_device(self, task_id: int) -> bool:
        return self.impl.is_task_cached_device(task_id)
