"""
CUDA Graph compatible LoRA parameters structure.

This module defines the new LoRA parameters structure that enables CUDA Graph capture
with multi-LoRA by using persistent device tensors for group GEMM operations.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class LoraLayerParams:
    """
    Parameters for a single LoRA layer that are compatible with CUDA Graph capture.

    All tensors are persistent device tensors that can be updated outside of graph replay.
    """
    # Weight pointers managed by PEFT cache manager
    # Shape: [layer_module_num, max_lora_size]
    d_b_ptrs: torch.Tensor  # Lora_in weight pointers
    d_b_prime_ptrs: torch.Tensor  # Lora_out weight pointers
    h_b_ptrs: torch.Tensor  # Lora_in weight pointers in host
    h_b_prime_ptrs: torch.Tensor  # Lora_out weight pointers in host


class CudaGraphLoraParams:
    """
    CUDA Graph compatible LoRA parameters for all layers and batch management.

    This structure maintains persistent device tensors that can be updated outside
    of CUDA Graph replay to support different LoRA combinations per batch.
    """

    LoraLayerKey = namedtuple('LoraLayerKey', ['layer_idx', 'module_ids'])

    @dataclass
    class LoraLayerInfo:
        module_num: int = 0
        output_sizes: List[int] | torch.Tensor | None = None
        input_hidden_size: int = 0

        def is_enabled(self) -> bool:
            """
            Check if the layer is enabled in the sample LoRA task. We assume that all LoRA tasks occupy the same layers. TODO: remove this restriction?
            Only input_hidden_size is initialized by the sample LoRA task, so use that to check if the layer is enabled.
            """
            return self.input_hidden_size > 0

    def __init__(self,
                 max_batch_size: int,
                 max_lora_size: int,
                 max_rank: int,
                 layer_info: Dict[LoraLayerKey, LoraLayerInfo],
                 device: str = "cuda"):
        """
        Initialize CUDA Graph compatible LoRA parameters.

        Args:
            max_batch_size: Maximum batch size for this graph
            max_lora_size: Maximum number of LoRA adapters
            max_rank: Maximum rank for all layers
            layers_info: Layer information for each layer
            device: Device to allocate tensors on
            dtype: Data type for size and offset tensors
        """
        self.max_batch_size = max_batch_size
        self.max_lora_size = max_lora_size
        self.max_rank = max_rank
        self.layer_info = layer_info
        self.layer_module2key = self._calculate_layer_module2key()
        self.device = device
        print(
            f'cuda graph lora params init max batch size: {max_batch_size}, max lora size: {max_lora_size}, max rank: {max_rank}'
        )

        # Sparse per-layer parameters (only for layers with LoRA modules)
        # Map: layer_idx -> LoraLayerParams (only contains layers with layer_module_nums[layer_idx] > 0)
        self.layer_params: Dict[self.LoraLayerKey, LoraLayerParams] = dict()

        # TODO: sort in graph instead
        # Sorted indices for gather/scatter operations (persistent device tensor)
        # Will contain indices that sort tokens by their slot_ids for efficient gather/scatter
        # Shape: [max_batch_size], values are indices in [0, max_batch_size-1]
        self.sorted_ids = torch.zeros(max_batch_size,
                                      dtype=torch.int64,
                                      device=device)
        self.sorted_ids_host = torch.zeros_like(self.sorted_ids,
                                                device='cpu',
                                                pin_memory=True)

        # persistent values for gen-only batch with cuda graph
        self.persistent_sorted_ids = self.sorted_ids

        # TODO: remove or move sorting to graph
        self.slot_ids = torch.zeros(max_batch_size,
                                    dtype=torch.int64,
                                    device=device)

        self.slot_counts = torch.zeros(max_lora_size,
                                       dtype=torch.int32,
                                       device=device)
        self.slot_counts_host = torch.zeros_like(self.slot_counts,
                                                 device='cpu',
                                                 pin_memory=True)
        self.slot_offsets_full = torch.zeros(max_lora_size + 1,
                                             dtype=torch.int64,
                                             device=device)
        self.slot_offsets = self.slot_offsets_full[:-1]
        self.slot_offsets_full_host = torch.zeros_like(self.slot_offsets_full,
                                                       device='cpu',
                                                       pin_memory=True)

        self.slot_ranks = torch.zeros(max_lora_size,
                                      dtype=torch.int32,
                                      device=device)
        self.slot_ranks_host = torch.zeros_like(self.slot_ranks,
                                                device='cpu',
                                                pin_memory=True)

        # Initialize per-layer parameters ONLY for layers with LoRA modules
        for key, info in self.layer_info.items():
            assert info.module_num > 0 and info.output_sizes is not None and len(
                info.output_sizes) == info.module_num
            # Create layer parameters
            self.layer_params[key] = self._create_layer_params(
                key, info.module_num, info.output_sizes)
        # print(f"layer_info: {self.layer_info}")
        # print(f"layer_params: {self.layer_params}")

    def _calculate_layer_module2key(self):
        layer_module2key = dict()
        for key in self.layer_info.keys():
            layer_id = key.layer_idx
            module_ids = key.module_ids
            for module_id in module_ids:
                layer_module2key[(layer_id, module_id)] = key
        return layer_module2key

    def _create_layer_params(
            self, key: LoraLayerKey, layer_module_num: int,
            module_output_sizes: torch.Tensor) -> LoraLayerParams:
        """
        Create LoraLayerParams for a specific layer.

        Args:
            key: Key of the layer
            layer_module_num: Number of modules in this layer
            module_output_sizes: Output sizes for each module in this layer

        Returns:
            LoraLayerParams for the specified layer
        """
        # GEMM parameter tensors only need max_lora_size (no dummy slot for base model)
        # Base model requests are handled separately and don't participate in GEMM operations
        shape_2d = (layer_module_num, self.max_lora_size)

        return LoraLayerParams(
            # Weight pointers - managed by PEFT cache manager
            d_b_ptrs=torch.zeros(shape_2d,
                                 dtype=torch.int64,
                                 device=self.device),
            d_b_prime_ptrs=torch.zeros(shape_2d,
                                       dtype=torch.int64,
                                       device=self.device),
            h_b_ptrs=torch.zeros(shape_2d, dtype=torch.int64, pin_memory=True),
            h_b_prime_ptrs=torch.zeros(shape_2d,
                                       dtype=torch.int64,
                                       pin_memory=True))

    @staticmethod
    def get_sorted_indices(slot_ids: List[int]) -> torch.Tensor:
        """
        Get sorted indices for the given slot IDs.
        """
        slot_ids = torch.tensor(slot_ids, dtype=torch.int64)

        # Compute sorted indices for gather/scatter operations
        # Use stable sort to maintain deterministic ordering for tokens with same slot_id
        sorted_slot_ids, sorted_indices = torch.sort(slot_ids, stable=True)

        # logger.info(f"sorted_slot_ids: {sorted_slot_ids}")
        # logger.info(f"sorted_indices: {sorted_indices}")
        return sorted_indices

    def update_sorted_indices(self, slot_ids: List[int]):
        """
        Update slot IDs for the current batch and compute sorted indices.

        Args:
            slot_ids: List of slot IDs for each token in the batch
            actual_batch_size: Actual batch size (may be less than max_batch_size)
        """
        actual_batch_size = len(slot_ids)
        assert actual_batch_size <= self.max_batch_size, \
            f"Actual batch size {actual_batch_size} exceeds max {self.max_batch_size}"
        sorted_indices = self.get_sorted_indices(slot_ids)

        # Update sorted_ids tensor with the computed indices
        assert actual_batch_size <= self.max_batch_size, f"CudaGraphLoraParams: Actual batch size {actual_batch_size} exceeds max {self.max_batch_size}!"
        # print(f"cuda graph lora params actual batch size: {actual_batch_size}")
        if actual_batch_size <= self.max_batch_size:
            # if can fit in persistent, use it
            self.sorted_ids = self.persistent_sorted_ids
            # self.slot_ids[:actual_batch_size] = slot_ids
            sorted_ids_host = self.sorted_ids_host[:actual_batch_size]
            sorted_ids_host.copy_(sorted_indices)
            self.sorted_ids[:actual_batch_size].copy_(sorted_ids_host,
                                                      non_blocking=True)
        else:
            # otherwise not an gen-only batch, use new allocated sorted_ids
            self.sorted_ids = sorted_indices.to(device=self.device)
        # logger.info(f"sorted_ids to be used: {self.sorted_ids}; actual_batch_size: {actual_batch_size}")

    def update_weight_pointers(self, peft_table: Dict[int, List],
                               slot_to_task_mapping: tuple[Optional[int], ...]):
        """
        Update weight pointers from PEFT cache manager.

        Args:
            peft_table: PEFT table from cache manager containing weight pointers, map task id to list of layer module configs
            slot_to_task_mapping: Mapping from slot_id to task_id, tuple of None for empty slots
        """

        # get slot ranks
        # assume ranks are the same within a slot,
        # input_hidden_size are the same within a layer
        # output sizes are the same within a layer
        def zero_out_weight_pointers(slot_id: int):
            """
            Zero out all weight pointers for a given slot_id for all layers
            """
            for layer_param in self.layer_params.values():
                layer_param.h_b_ptrs[:, slot_id] = 0
                layer_param.h_b_prime_ptrs[:, slot_id] = 0

        # TODO: sync last H2D memcpy before write to pinned host tensors?
        for slot_id in range(self.max_lora_size):
            task_id = slot_to_task_mapping[slot_id]
            if task_id is None:  # empty slot
                self.slot_ranks_host[slot_id] = 0
                zero_out_weight_pointers(slot_id)
            elif task_id not in peft_table:  # task has not changed in the slot, retain old rank / weight pointers
                continue
            else:  # task might have changed in the slot, update its rank
                task_configs = peft_table[task_id]
                config = task_configs[
                    0]  # assume all layerModuleConfigs have the same rank
                self.slot_ranks_host[slot_id] = config.adapter_size

                zero_out_weight_pointers(
                    slot_id
                )  # in case new task in slot do not have LoRA adapter for some module in some layer
                for config in task_configs:
                    layer_id = config.layer_id
                    module_id = config.module_id
                    key = self.layer_module2key[(layer_id, module_id)]
                    layer_param = self.layer_params[key]
                    local_module_id = key.module_ids.index(module_id)

                    # Only process layers that have LoRA modules
                    assert key in self.layer_params, f"Layer {layer_id} not found in layer_params, assumption that all LoRA has their adapters on the same layers is broken"

                    # Validate LoRA rank - TaskLayerModuleConfig uses 'rank' attribute
                    rank = config.adapter_size
                    assert rank <= self.max_rank, f"LoRA rank {rank} in layer {layer_id} exceeds configured max_rank {self.max_rank}. "

                    # Set weight pointers for this slot and module
                    layer_param.h_b_ptrs[local_module_id,
                                         slot_id] = config.weights_in_pointer
                    layer_param.h_b_prime_ptrs[
                        local_module_id, slot_id] = config.weights_out_pointer

        self.slot_ranks.copy_(self.slot_ranks_host, non_blocking=True)
        # print(f"slot_ranks: {self.slot_ranks}")

        for layer_param in self.layer_params.values():
            layer_param.d_b_ptrs.copy_(layer_param.h_b_ptrs, non_blocking=True)
            layer_param.d_b_prime_ptrs.copy_(layer_param.h_b_prime_ptrs,
                                             non_blocking=True)

    @staticmethod
    def get_offset_from_counts(counts: torch.Tensor,
                               full: bool = False,
                               out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            if full:
                offset = torch.empty(counts.shape[0] + 1,
                                     dtype=torch.int64,
                                     device=counts.device)
            else:
                offset = torch.empty(counts.shape[0],
                                     dtype=torch.int64,
                                     device=counts.device)
        else:
            assert (full and out.shape[0] == counts.shape[0] + 1) or (
                (not full) and out.shape[0] == counts.shape[0])
            offset = out

        offset[0] = 0

        if full:
            offset[1:] = counts
        else:
            offset[1:] = counts[:-1]
        offset[1:].cumsum_(dim=0)
        return offset

    @staticmethod
    def get_slot_counts(batch_slot_ids: List[int],
                        max_lora_size: int) -> torch.Tensor:
        """
        Get the number of tokens for each slot_id in the batch.
        """
        slot_counts = torch.bincount(torch.tensor(batch_slot_ids,
                                                  dtype=torch.int32),
                                     minlength=max_lora_size)
        assert slot_counts.size(0) <= max_lora_size + 1
        slot_counts = slot_counts[:max_lora_size]
        # logger.info(f"slot_counts: {slot_counts}")
        return slot_counts

    def update_slots_params(self, batch_slot_ids: List[int]):
        """
        Update GEMM sizes and buffer offsets based on current batch composition.

        Args:
            batch_slot_ids: Slot IDs for each token in the batch
        """
        slot_counts = self.get_slot_counts(batch_slot_ids, self.max_lora_size)
        self.slot_counts_host.copy_(slot_counts)
        self.get_offset_from_counts(slot_counts,
                                    full=True,
                                    out=self.slot_offsets_full_host)
        # logger.info(f"slot_token_offset: {slot_token_offset}")
        self.slot_counts.copy_(self.slot_counts_host, non_blocking=True)
        self.slot_offsets_full.copy_(self.slot_offsets_full_host,
                                     non_blocking=True)

    def get_problem_count(self, layer_key: LoraLayerKey) -> int:
        """
        Get the number of GEMM problems for a layer.

        Args:
            layer_key: Key of the layer

        Returns:
            Number of GEMM problems (layer_module_num * max_lora_size)
            Returns 0 if layer has no LoRA modules
            Note: Only actual LoRA slots are counted, not the dummy base model slot
        """
        if layer_key not in self.layer_params:
            return 0  # Layer has no LoRA modules
        return self.layer_info[layer_key].module_num * self.max_lora_size

    def get_layer_params(self,
                         layer_key: LoraLayerKey) -> Optional[LoraLayerParams]:
        """
        Get LoRA parameters for a specific layer.

        Args:
            layer_key: Key of the layer

        Returns:
            LoraLayerParams for the specified layer, or None if layer has no LoRA modules
        """
        return self.layer_params.get(layer_key)
