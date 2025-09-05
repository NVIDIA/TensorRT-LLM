"""
CUDA Graph compatible LoRA manager for PyTorch backend.

This module provides a manager that coordinates AdapterSlotManager, CudaGraphLoraParams,
and PeftCacheManager to enable CUDA Graph capture with multi-LoRA support.
"""

from typing import Dict, Optional

import torch

from ...._utils import nvtx_range
from ....logger import logger
from ....lora_manager import LoraManager, LoraModelConfig
from ...attention_backend.interface import AttentionMetadata
from ...pyexecutor.resource_manager import PeftCacheManager
from ...pyexecutor.scheduler import ScheduledRequests
from .adapter_slot_manager import AdapterSlotManager
from .cuda_graph_lora_params import CudaGraphLoraParams
from .layer import LoraLayer


class CudaGraphLoraManager:
    """
    Manager that coordinates adapter slots and CUDA Graph compatible LoRA parameters.

    This class bridges the gap between the current LoRA implementation and the new
    CUDA Graph compatible design by managing adapter slots and preparing persistent
    device tensors for group GEMM operations.
    """

    def __init__(self,
                 max_lora_size: int,
                 max_batch_size: int,
                 max_lora_rank: int,
                 model: torch.nn.Module,
                 lora_model_config: Optional[LoraModelConfig],
                 device: str = "cuda"):
        """
        Initialize the CUDA Graph LoRA manager.

        Args:
            max_lora_size: Maximum number of LoRA adapters that can be active
            max_batch_size: Maximum batch size for CUDA graphs
            max_lora_rank: Maximum LoRA rank across all layers
            model: Model to get layerwise LoRA info
            lora_model_config: LoRA model configuration
            device: Device to allocate tensors on
        """
        self.max_lora_size = max_lora_size
        self.max_batch_size = max_batch_size
        self.max_lora_rank = max_lora_rank
        self.device = device

        self.adapter_slot_manager = AdapterSlotManager(max_lora_size, device)
        self.lora_model_config = lora_model_config
        lora_target_modules = lora_model_config.lora_target_modules
        self.target_modules_ids: Optional[tuple[int, ...]] = tuple(
            map(LoraManager.LORA_MODULE_IDS.__getitem__,
                lora_target_modules)) if bool(lora_target_modules) else None
        # print(f"config target modules ids: {lora_model_config.lora_target_modules}")
        # print(f"converted target modules ids: {self.target_modules_ids}")
        if not self.target_modules_ids:
            logger.debug(
                "No LoRA target modules provided in LoRA config, using all modules in PyTorch Module!"
            )

        # Single CudaGraphLoraParams instance for all batch sizes
        # Will be initialized lazily when we have access to peft_table
        self.cuda_graph_lora_params: Optional[CudaGraphLoraParams] = None
        self.layer_info: Dict[
            CudaGraphLoraParams.LoraLayerKey, CudaGraphLoraParams.
            LoraLayerInfo] | None = None  # TODO: key value struct init from pytorch model's structure, value's content init from sample peft table; use target LoRA modules to only init valid modules
        self._initialize_from_model(model)
        self.cuda_graph_lora_params = CudaGraphLoraParams(
            max_batch_size=self.max_batch_size,
            max_lora_size=self.max_lora_size,
            max_rank=self.max_lora_rank,
            layer_info=self.layer_info,
            device=self.device)

    def _initialize_from_model(self, model: torch.nn.Module):
        """
        Initialize LoRALayerInfo from model.
        """
        self.layer_info = dict()

        def get_layer_idx(model: torch.nn.Module, lora_module: LoraLayer,
                          lora_module_name: str):
            module = lora_module
            name = lora_module_name
            while module is not None and ((not hasattr(module, 'layer_idx'))
                                          or module.layer_idx is None):
                name = name.rsplit(".", 1)
                name = name[0] if len(name) > 1 else None
                if name is not None:
                    module = model.get_submodule(name)
                else:
                    module = None
            if hasattr(module, 'layer_idx') and module.layer_idx is not None:
                return module.layer_idx
            return None

        # Ignore LoRA layers without at least one of the target modules. TODO: support disable some of the modules in a LoRA layer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                layer_idx = get_layer_idx(model, module, name)
                # if target_modules_ids is None, by default enable all modules
                if self.target_modules_ids and not any(
                        module_id in self.target_modules_ids
                        for module_id in module.lora_module_types):
                    logger.debug(
                        f"Layer {name} does not have any of the target modules, skipping"
                    )
                    continue
                layer_key = CudaGraphLoraParams.LoraLayerKey(
                    layer_idx=layer_idx,
                    module_ids=tuple(module.lora_module_types))
                assert layer_key not in self.layer_info, f"Layer {layer_key} already exists"

                self.layer_info[layer_key] = CudaGraphLoraParams.LoraLayerInfo()
                self.layer_info[
                    layer_key].output_sizes = module.output_hidden_sizes
                self.layer_info[layer_key].module_num = len(
                    module.lora_module_types)

    @nvtx_range("prepare_cuda_graph_lora_params")
    def prepare_cuda_graph_lora_params(
            self, scheduled_requests: "ScheduledRequests",
            attn_metadata: "AttentionMetadata",
            peft_cache_manager: PeftCacheManager) -> Optional[Dict]:
        """
        Prepare CUDA Graph compatible LoRA parameters from scheduled requests.

        This method replaces the original _get_lora_params_from_requests for CUDA Graph mode.

        Args:
            scheduled_requests: The scheduled requests for the current batch
            attn_metadata: Attention metadata containing batch information
            peft_table: PEFT table from cache manager

        Returns:
            Dictionary containing CUDA Graph compatible LoRA parameters, or None if no LoRA
        """
        # torch.cuda.current_stream().synchronize()   # failed!
        assert len(
            scheduled_requests.context_requests
        ) == 0, f"Context requests are not supported with LoRA CUDA Graph path. Have {len(scheduled_requests.context_requests)} context requests"
        request_list = scheduled_requests.generation_requests

        len(request_list)
        peft_table = peft_cache_manager.get_and_reset_batch_peft_table()

        # logger.info(f"num gen requests: {len(scheduled_requests.generation_requests)}, request_id: {[req.py_request_id for req in scheduled_requests.generation_requests]}")
        # Get slot assignments for this batch
        request_slot_ids = self.adapter_slot_manager.update_slots(
            request_list, peft_cache_manager)

        # print(f"request_id_to_slot_id: {request_to_slot_id}")

        # logger.info(f"request_slot_ids: {request_slot_ids}")
        cuda_graph_lora_params = self.cuda_graph_lora_params
        cuda_graph_lora_params.update_sorted_indices(request_slot_ids)

        # Get current slot to task mapping
        slot2task = self.adapter_slot_manager.get_slot_to_task_mapping()

        # logger.info(f"slot_to_task_mapping: {slot_to_task_mapping}")
        # torch.cuda.current_stream().synchronize()   # failed!
        # Update weight pointers if slot assignments changed
        # pass if always udapte weight pointers
        if self.adapter_slot_manager.has_slots_changed():
            cuda_graph_lora_params.update_weight_pointers(peft_table, slot2task)
            self.adapter_slot_manager.reset_changed_flag()

        # torch.cuda.current_stream().synchronize()   # failed!
        # Update GEMM sizes and offsets based on current batch
        cuda_graph_lora_params.update_slots_params(
            batch_slot_ids=request_slot_ids)

        # Create return dictionary compatible with current LoRA layer interface
        # This bridges the old and new interfaces
        lora_params = {
            'cuda_graph_params': cuda_graph_lora_params,
            'host_request_types': attn_metadata.host_request_types,
            'prompt_lens_cpu': attn_metadata.prompt_lens_cpu,
            'num_seqs': attn_metadata.num_seqs,
            'use_cuda_graph_mode': True,  # Flag to indicate new mode
        }

        return lora_params
