import os
import threading
from contextlib import nullcontext
from multiprocessing import resource_tracker, shared_memory
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...distributed import AllReduce
from ...utils import EventType
from ..multi_stream_utils import do_multi_stream


def _tensor_to_weight(t: torch.Tensor) -> _tbr.MoeWeight:
    """
    Convert a tensor to a MoeWeight object.

    Args:
        t: The tensor to convert
    """
    assert t.dim() <= 2, "t.dim() should be less than or equal to 2"
    shape = [1, 1]
    pitch = 1
    elt_size = torch.tensor([], dtype=t.dtype).element_size()
    if t.dim() == 2:
        shape[0] = t.size(0)
        shape[1] = t.size(1)
        pitch = t.stride(0)
    elif t.dim() == 1:
        shape[1] = t.size(0)
        pitch = t.size(0)
    else:
        pass
    mw = _tbr.MoeWeight()
    mw.height = shape[0]
    mw.width = shape[1] * elt_size
    mw.pitch = pitch * elt_size
    mw.weight_ptr = t.data_ptr()
    return mw


class HostMoeTensorSharer:
    """
    A class representing a host tensor sharer.
    """

    def __init__(self, layer_id: int, expert_count: int,
                 shared_mpi_comm: MPI.Comm):
        """
        Initialize a HostMoeTensorSharer instance.

        Args:
            shared_mpi_comm: The MPI communicator for shared memory
        """
        self.shared_mpi_comm = shared_mpi_comm
        self.layer_id = layer_id
        self.expert_count = expert_count
        self.shared_memory_base_name = None

        self.local_rank = self.shared_mpi_comm.Get_rank()
        self.local_size = self.shared_mpi_comm.Get_size()

        self.expert_start = self.local_rank * self.expert_count // self.local_size
        self.expert_end = (self.local_rank +
                           1) * self.expert_count // self.local_size

        self.name_info = {}  # key is weight name, value is (dtype, shape)
        self.host_weights = {}

        self.own_shm = None
        self.imported_shms = []

        self.shared_tensors = {}
        self.names = []

    def set_shared_memory_base_name(self, shared_memory_base_name):
        """
        Set the shared memory base name for the layer.

        Args:
            shared_memory_base_name: The base name for the shared memory
        """
        self.shared_memory_base_name = shared_memory_base_name

    def get_shared_memory_name(self, rank: Optional[int] = None):
        """
        Get the shared memory name for the layer.

        Args:
            rank: The rank who created the shared memory. Current rank if None
        """
        if rank is None:
            rank = self.local_rank
        assert 0 <= rank < self.local_size
        assert isinstance(self.shared_memory_base_name,
                          str), "self.shared_memory_base_name must be a string"
        shared_memory_name = f"{self.shared_memory_base_name}_l{self.layer_id}_lr{rank}_all"
        return shared_memory_name

    def pre_register_host_tensor_with_shape(self, expert_id: int, name: str,
                                            dtype, tensor_shape):
        """
        Pre-register a host tensor with shape.
        This function is invoked by the processes that don't load the weights.
        It just records the shape of the weight tensor, and the process that
        loads the weights will call share_host_tensor_with_shape to share the
        actual weight tensor. After the weight tensor is shared, e.g. after the
        barrier in finalize_model, the processes that pre-registers the weight
        tensor will add the weight tensor to the layer.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
        """
        assert len(tensor_shape
                   ) <= 2, "tensor_shape dim must be less than or equal to 2"
        assert 0 <= expert_id < self.expert_count
        assert expert_id < self.expert_start or expert_id >= self.expert_end
        if name not in self.name_info:
            self.name_info[name] = (dtype, tensor_shape)
        else:
            assert dtype == self.name_info[name][0] and tensor_shape == self.name_info[name][1], \
                f'weights name={name}, dtype={dtype}, shape={tensor_shape}, but already registered with dtype={self.name_info[name][0]}, shape={self.name_info[name][1]}'

    def share_host_tensor_with_shape(self, expert_id: int, name: str,
                                     t: torch.Tensor):
        """
        Share a host tensor with shape.
        This function is invoked by the processes that load the weights.
        It creates a shared memory for the weight tensor, and then shares the
        weight tensor with the processes that pre-register the weight tensor.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
            t: The weight tensor
        """
        assert len(
            t.shape) <= 2, "tensor_shape dim must be less than or equal to 2"
        assert t.is_contiguous() == True, "t.is_contiguous() must be True"
        assert (expert_id, name) not in self.shared_tensors.keys()
        assert self.expert_start <= expert_id < self.expert_end
        self.shared_tensors[(expert_id, name)] = t
        dtype = t.dtype
        tensor_shape = t.shape
        if name not in self.name_info:
            self.name_info[name] = (dtype, tensor_shape)
        else:
            assert dtype == self.name_info[name][0] and tensor_shape == self.name_info[name][1], \
                f'weights name={name}, dtype={dtype}, shape={tensor_shape}, but already registered with dtype={self.name_info[name][0]}, shape={self.name_info[name][1]}'

    @staticmethod
    def align_size(size: int):
        return (size + 256 - 1) // 256 * 256

    def finalize_layer_weights(self):
        self.names = list(sorted(self.name_info.keys()))
        assert len(
            self.shared_tensors.keys()) == (self.expert_end -
                                            self.expert_start) * len(self.names)

        total_size = 0
        for name in self.names:
            dtype, shape = self.name_info[name]
            for expert_id in range(self.expert_start, self.expert_end):
                t = self.shared_tensors[(expert_id, name)]
                assert dtype == t.dtype and shape == t.shape
                data_size = t.numel() * t.element_size()
                aligned_size = self.align_size(data_size)
                total_size += aligned_size

        shm_name = self.get_shared_memory_name()
        try:
            shm = shared_memory.SharedMemory(name=shm_name,
                                             create=True,
                                             size=total_size)
        except FileExistsError:
            tensorrt_llm.logger.warning(
                f'Found exist EPLB shared memory name: {shm_name}, unlinking...'
            )
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            existing_shm.close()
            existing_shm.unlink()
            shm = shared_memory.SharedMemory(name=shm_name,
                                             create=True,
                                             size=total_size)
        self.own_shm = shm

        offset = 0
        for name in self.names:
            for expert_id in range(self.expert_start, self.expert_end):
                t = self.shared_tensors[(expert_id, name)].contiguous().cpu()
                data_size = t.numel() * t.element_size()
                aligned_size = self.align_size(data_size)
                shm.buf[offset:offset + data_size] = t.flatten().view(
                    torch.int8).numpy().tobytes()
                dtype = t.dtype
                tensor_shape = t.shape
                elt_count = t.numel()
                st = torch.frombuffer(shm.buf,
                                      dtype=dtype,
                                      offset=offset,
                                      count=elt_count).view(tensor_shape)
                key = (expert_id, name)
                assert key not in self.host_weights.keys(
                ), f"key={key} already exists"
                self.host_weights[key] = st
                offset += aligned_size
        self.shared_tensors = {}

    def finalize_host_tensor_sharing(self, add_host_weight_fn: Callable = None):
        """
        Finalize the host tensor sharing.
        """
        for rank in range(self.local_size):
            if rank == self.local_rank:
                continue

            shm_name = self.get_shared_memory_name(rank)
            shm = shared_memory.SharedMemory(name=shm_name)
            self.imported_shms.append(shm)

            rank_expert_start = rank * self.expert_count // self.local_size
            rank_expert_end = (rank + 1) * self.expert_count // self.local_size

            offset = 0
            for name in self.names:
                dtype, shape = self.name_info[name]
                elt_count = int(np.prod(shape))
                data_size = torch.tensor([],
                                         dtype=dtype).element_size() * elt_count
                aligned_size = self.align_size(data_size)
                for expert_id in range(rank_expert_start, rank_expert_end):
                    t = torch.frombuffer(shm.buf,
                                         dtype=dtype,
                                         offset=offset,
                                         count=elt_count).view(shape)
                    key = (expert_id, name)
                    assert key not in self.host_weights.keys(
                    ), f"key={key} already exists"
                    self.host_weights[key] = t
                    offset += aligned_size

        if add_host_weight_fn is not None:
            for key, t in self.host_weights.items():
                add_host_weight_fn(key[0], key[1], t)

        self.host_weights.clear()

    def pre_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        for shm in self.imported_shms:
            shm.close()
            resource_tracker.unregister(shm._name, "shared_memory")
        self.imported_shms = None
        if self.own_shm:
            self.own_shm.close()

    def post_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        if self.own_shm:
            self.own_shm.unlink()
            self.own_shm = None


class SingleLayerMoeLoadBalancer:
    """
    A class representing a single layer of the Mixture of Experts (MoE) load balancer.
    This class wraps the C++ implementation of SingleLayerMoeLoadBalancer.
    """

    def __init__(
            self,
            single_layer_load_balancer_impl: _tbr.SingleLayerMoeLoadBalancer,
            shared_mpi_comm: MPI.Comm,
            expert_count: int,
            updates_enabled: bool = True,
            repeated_count=1,
            aux_stream: Optional[torch.cuda.Stream] = None):
        """
        Initialize a SingleLayerMoeLoadBalancer instance.

        Args:
            single_layer_load_balancer_impl: The C++ implementation of SingleLayerMoeLoadBalancer
            shared_mpi_comm: The MPI communicator for shared memory
            expert_count: total number of experts
            updates_enabled: whether to enable weight updates
            repeated_count: the repeated count of current layer, used when forward is repeated more than once like MTP.
        """
        self.single_layer_load_balancer_impl = single_layer_load_balancer_impl
        self.single_layer_load_balancer_ptr = single_layer_load_balancer_impl.get_pointer(
        )
        self.expert_count = expert_count
        self.updates_enabled = updates_enabled
        self.repeated_count = repeated_count
        layer_id = self.single_layer_load_balancer_impl.get_layer_id()
        self.host_tensor_sharer = HostMoeTensorSharer(
            layer_id, expert_count,
            shared_mpi_comm) if self.updates_enabled else None
        self.register_weight_fns = []
        self.to_migrate_weight_fns = []

        shared_rank = shared_mpi_comm.Get_rank()
        shared_size = shared_mpi_comm.Get_size()

        load_expert_start = shared_rank * self.expert_count // shared_size
        load_expert_end = min(
            (shared_rank + 1) * self.expert_count // shared_size,
            self.expert_count)
        self.load_expert_ids = list(range(load_expert_start, load_expert_end))

        if self.updates_enabled:
            self.aux_stream = aux_stream if aux_stream is not None else torch.cuda.Stream(
            )
            self.event_dict = {
                key: torch.cuda.Event()
                for key in [EventType.Main, EventType.MoeBalancer]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

        self.statistic_flag_tensor = None
        self.local_statistic_tensor = None
        self.func_called_count = {
            name: 0
            for name in [
                "start_wait_gpu_stage",
                "done_wait_gpu_stage",
                "start_set_cpu_stage",
                "done_set_cpu_stage",
                "update_local_statistic",
                "get_local_statistic_tensor",
                "update_statistic_with_gathered_statistic",
                "update_statistic_with_local_ids",
                "update_statistic_with_global_ids",
                "route",
            ]
        }

    def get_layer_idx(self):
        return self.single_layer_load_balancer_impl.get_layer_id()

    def get_load_expert_ids(self):
        assert self.updates_enabled, "should not call get_load_expert_ids when using statistic routing"
        return self.load_expert_ids

    def get_repeat_count(self):
        return self.repeated_count

    def is_static_routing(self):
        return not self.updates_enabled

    def need_load_shared_weights(self):
        return self.updates_enabled

    def set_shared_memory_base_name(self, shared_memory_base_name):
        """
        Set the shared memory base name for the layer.

        Args:
            shared_memory_base_name: The base name for the shared memory
        """
        if self.updates_enabled:
            self.host_tensor_sharer.set_shared_memory_base_name(
                shared_memory_base_name)

    def _add_weight_slot(self, slot_id: int, name: str,
                         weight_slot: _tbr.MoeWeight):
        """
        Add a weight slot to the layer.

        Args:
            slot_id: The ID of the slot
            name: The name of the weight
            weight_slot: The weight object
        """
        self.single_layer_load_balancer_impl.add_single_weight_slot(
            slot_id, name, weight_slot)

    def register_weight_slot(self, local_slot_id: int, name: str,
                             t: torch.Tensor):
        """
        Register a weight slot to the layer.

        Args:
            local_slot_id: The ID of the slot at local rank
            name: The name of the weight
            t: The weight tensor
        """
        moe_weight = _tensor_to_weight(t)
        self._add_weight_slot(local_slot_id, name, moe_weight)

    def _add_host_weight(self, expert_id: int, name: str,
                         host_weight: _tbr.MoeWeight):
        """
        Add a host weight to the layer.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
            host_weight: The host weight object
        """
        self.single_layer_load_balancer_impl.add_single_host_weight(
            expert_id, name, host_weight)

    def _add_host_weight_from_tensor(self, expert_id: int, name: str,
                                     t: torch.Tensor):
        """
        Add a host weight to the layer from a tensor.
        """
        host_weight = _tensor_to_weight(t)
        self._add_host_weight(expert_id, name, host_weight)

    def set_initial_weight_assignments(self,
                                       initial_weight_assignments: List[int]):
        """
        Set the initial weight assignments for the layer.

        Args:
            initial_weight_assignments: A list of initial weight assignments
        """
        self.single_layer_load_balancer_impl.set_initial_weight_assignments(
            initial_weight_assignments)

    def add_to_migrate_weight_fn(self,
                                 fn: Callable,
                                 args: Tuple,
                                 kwargs: Dict = {}):
        self.to_migrate_weight_fns.append((fn, args, kwargs))

    def add_register_weight_fn(self,
                               fn: Callable,
                               args: Tuple,
                               kwargs: Dict = {}):
        """
        Add weight register function, this function doesn't run fn directly but run all functions after model.to("cuda")
        so this function can be called when model is not on GPU yet.
        """
        self.register_weight_fns.append((fn, args, kwargs))

    def make_tensor_host_accessible(self, wt: torch.Tensor):
        torch.ops.trtllm.migrate_to_host_accessible(wt)
        torch.cuda.empty_cache()

    def register_weight_slots_after_to_cuda(self):
        """
        Register weights after model has been moved to cuda, should be invoked after model.to("cuda") and before finalize_model.
        """
        for fn, args, kwargs in self.to_migrate_weight_fns:
            fn(*args, **kwargs)

        self.to_migrate_weight_fns = []

        for fn, args, kwargs in self.register_weight_fns:
            fn(*args, **kwargs)

        self.register_weight_fns = []

    def py_finalize_model(self):
        """
        Finalize the model after all layers have been added.
        This must be called before starting any iterations.
        """
        if self.updates_enabled:
            self.host_tensor_sharer.finalize_host_tensor_sharing(
                self._add_host_weight_from_tensor)

    def start_wait_gpu_stage(self):
        """
        Start to wait for the GPU stage to complete.
        """
        assert self.func_called_count["start_wait_gpu_stage"] == 0
        self.func_called_count["start_wait_gpu_stage"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    self.statistic_flag_tensor = torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
                        self.single_layer_load_balancer_ptr)
                    self.event_dict[EventType.MoeBalancer].record()
            else:
                self.statistic_flag_tensor = torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
                    self.single_layer_load_balancer_ptr)

    def done_wait_gpu_stage(self):
        """
        Done waiting for the GPU stage to complete.
        """
        assert self.func_called_count["start_wait_gpu_stage"] == 1
        assert self.func_called_count["done_wait_gpu_stage"] == 0
        self.func_called_count["done_wait_gpu_stage"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.MoeBalancer].wait()

    def start_set_cpu_stage(self):
        """
        Start to set the CPU stage.
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count["start_set_cpu_stage"] == 0
        self.func_called_count["start_set_cpu_stage"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    torch.ops.trtllm.moe_load_balance_set_cpu_stage(
                        self.single_layer_load_balancer_ptr)
                    self.event_dict[EventType.MoeBalancer].record()
            else:
                torch.ops.trtllm.moe_load_balance_set_cpu_stage(
                    self.single_layer_load_balancer_ptr)

    def done_set_cpu_stage(self):
        """
        Done setting the CPU stage.
        """
        assert self.func_called_count["start_set_cpu_stage"] == 1
        for name in self.func_called_count:
            self.func_called_count[name] = 0
        self.statistic_flag_tensor = None
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.MoeBalancer].wait()

    def update_local_statistic(self, local_raw_expert_ids: torch.Tensor,
                               is_first_stage: bool, is_last_stage: bool):
        """
        Update local statistics of the expert IDs.

        Args:
            local_raw_expert_ids: The local raw expert IDs
            is_first_stage: Whether this is the first stage
            is_last_stage: Whether this is the last stage
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count["update_statistic_with_global_ids"] == 0
        self.func_called_count["update_local_statistic"] += 1
        if self.updates_enabled:
            if self.local_statistic_tensor is None:
                self.local_statistic_tensor = torch.empty(
                    (self.expert_count, ),
                    dtype=torch.int32,
                    device=torch.device('cuda'))
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    torch.ops.trtllm.moe_hierarchical_statistic_local_device(
                        local_raw_expert_ids, self.local_statistic_tensor,
                        self.statistic_flag_tensor,
                        self.single_layer_load_balancer_ptr, is_first_stage,
                        is_last_stage)
            else:
                torch.ops.trtllm.moe_hierarchical_statistic_local_device(
                    local_raw_expert_ids, self.local_statistic_tensor,
                    self.statistic_flag_tensor,
                    self.single_layer_load_balancer_ptr, is_first_stage,
                    is_last_stage)

    def get_local_statistic_tensor(self) -> Optional[torch.Tensor]:
        """
        Get the local statistic tensor.
        Returns:
            The local statistic tensor if using statistic else None
        """
        assert self.func_called_count["update_local_statistic"] > 0
        self.func_called_count["get_local_statistic_tensor"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.MoeBalancer].record()
                self.event_dict[EventType.MoeBalancer].wait()
            return self.local_statistic_tensor
        return None

    def update_statistic_with_gathered_statistic(
            self, gathered_local_statistic_tensor: torch.Tensor):
        """
        Update statistics of the expert IDs, using gathered local statistic tensors.

        Args:
            gathered_local_statistic_tensor: gathered local statistics info, should have shape (world_size, self.expert_count)
        """
        assert self.func_called_count["get_local_statistic_tensor"] > 0
        assert self.func_called_count["update_statistic_with_local_ids"] == 0
        assert self.func_called_count["update_statistic_with_global_ids"] == 0
        self.func_called_count["update_statistic_with_gathered_statistic"] += 1

        def _update_statistic():
            global_statistic_info = torch.sum(gathered_local_statistic_tensor,
                                              dim=0,
                                              dtype=torch.int32)
            torch.ops.trtllm.moe_hierarchical_statistic_update(
                global_statistic_info, self.statistic_flag_tensor,
                self.single_layer_load_balancer_ptr)

        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    _update_statistic()
            else:
                _update_statistic()

    def update_statistic_with_local_ids(self,
                                        local_raw_expert_ids: torch.Tensor,
                                        is_first_stage: bool,
                                        is_last_stage: bool,
                                        allreduce: Optional[AllReduce] = None):
        """
        Update statistics of the expert IDs, using local raw expert IDs.

        Args:
            local_raw_expert_ids: The local raw expert IDs
            is_first_stage: Whether this is the first stage
            is_last_stage: Whether this is the last stage
            allreduce: The allreduce object
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count[
            "update_statistic_with_gathered_statistic"] == 0
        assert self.func_called_count["update_statistic_with_global_ids"] == 0
        self.func_called_count["update_statistic_with_local_ids"] += 1

        def _update_statistic():
            if is_last_stage:
                global_statistic_info = allreduce(self.local_statistic_tensor)
                torch.ops.trtllm.moe_hierarchical_statistic_update(
                    global_statistic_info, self.statistic_flag_tensor,
                    self.single_layer_load_balancer_ptr)

        if self.updates_enabled:
            self.update_local_statistic(local_raw_expert_ids, is_first_stage,
                                        is_last_stage)
            if do_multi_stream():
                with torch.cuda.stream(self.aux_stream):
                    _update_statistic()
            else:
                _update_statistic()

    def update_statistic_with_global_ids(self,
                                         gathered_raw_expert_ids: torch.Tensor,
                                         is_first_stage: bool,
                                         is_last_stage: bool):
        """
        Update statistics of the expert IDs, using gathered raw expert IDs from all ranks.

        Args:
            gathered_raw_expert_ids: The gathered raw expert IDs from all ranks
            is_first_stage: Whether this is the first stage
            is_last_stage: Whether this is the last stage
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        assert self.func_called_count[
            "update_statistic_with_gathered_statistic"] == 0
        assert self.func_called_count["update_statistic_with_local_ids"] == 0
        self.func_called_count["update_statistic_with_global_ids"] += 1
        if self.updates_enabled:
            if do_multi_stream():
                self.event_dict[EventType.Main].record()
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.Main].wait()
                    torch.ops.trtllm.moe_load_balance_statistic(
                        gathered_raw_expert_ids, self.statistic_flag_tensor,
                        self.single_layer_load_balancer_ptr, is_first_stage,
                        is_last_stage)
            else:
                torch.ops.trtllm.moe_load_balance_statistic(
                    gathered_raw_expert_ids, self.statistic_flag_tensor,
                    self.single_layer_load_balancer_ptr, is_first_stage,
                    is_last_stage)

    def route(self,
              token_selected_experts: torch.Tensor,
              offset_by_ep_rank: bool = False) -> torch.Tensor:
        """
        Route the tokens to experts.

        Args:
            token_selected_experts: The experts selected by each token
            offset_by_ep_rank: Whether to offset the round robin position by ep_rank

        Returns:
            A tensor of routed slot IDs
        """
        assert self.func_called_count["done_wait_gpu_stage"] == 1
        self.func_called_count["route"] += 1
        return torch.ops.trtllm.moe_load_balance_routing(
            token_selected_experts, offset_by_ep_rank,
            self.single_layer_load_balancer_ptr)

    def py_pre_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        if self.updates_enabled:
            self.host_tensor_sharer.pre_shutdown_cleanup()

    def py_post_shutdown_cleanup(self):
        """
         Clean up the resources after C++ shutdown and barrier
         """
        if self.updates_enabled:
            self.host_tensor_sharer.post_shutdown_cleanup()


# Global variable to store the current active MoeLoadBalancer instance
_current_moe_load_balancer = threading.local()


class MoeLoadBalancer:
    """
    A class representing a Mixture of Experts (MoE) load balancer.
    This class can be used as a context manager to manage the lifecycle of a MoeLoadBalancer.
    """

    def __init__(self,
                 ep_rank: int,
                 ep_size: int,
                 layer_updates_per_iter: int,
                 shared_memory_base_name: Optional[str] = None):
        """
        Initialize a MoeLoadBalancer instance.

        Args:
            ep_rank: The rank of the current process in expert parallelism
            ep_size: The total number of processes in expert parallelism
            layer_updates_per_iter: The number of layers to update per iteration
            shared_memory_base_name: Shared memory base name, will use 'moe_shared' if None
        """
        self.is_shutdown = True
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.layer_updates_per_iter = layer_updates_per_iter
        self.load_balancer_impl = _tbr.MoeLoadBalancer(ep_rank, ep_size,
                                                       layer_updates_per_iter)
        self._previous_balancer = None
        self.single_layer_load_balancers = []
        self.shared_memory_base_name = shared_memory_base_name or os.getenv(
            'TRTLLM_EPLB_SHM_NAME', 'moe_shared')
        self._setup_mpi_comm()
        self.is_shutdown = False

        self.iter_id = 0
        self.in_iter = False

        self.enable_statistic = False
        self.enable_update_weights = False

        self.next_layer_repeated_count = None

    def __del__(self):
        if not self.is_shutdown:
            self.shutdown()

    def is_static_routing(self):
        # if we don't update, then it is statistic routing.
        return self.layer_updates_per_iter == 0

    def _setup_mpi_comm(self):
        global_mpi_comm = tensorrt_llm.mpi_comm()
        shared_mpi_comm = global_mpi_comm.Split_type(
            split_type=MPI.COMM_TYPE_SHARED)
        shared_size = shared_mpi_comm.Get_size()
        local_size = tensorrt_llm.local_mpi_size()
        assert shared_size == local_size, \
            f"Interesting, shared size {shared_size} is not same as local size {local_size}"
        self.shared_mpi_comm = shared_mpi_comm

    def set_use_gpu_memcpy(self, use_gpu_memcpy: bool):
        self.load_balancer_impl.set_use_gpu_memcpy(use_gpu_memcpy)

    def set_repeated_for_next_layer(self, repeated_count: int):
        """
        Set repeat count for next layer.

        Args:
            repeated_count: The repeat count for next layer
        """
        assert repeated_count > 0, "repeat count must be greater than 0"
        self.next_layer_repeated_count = repeated_count

    def add_layer(
        self,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        aux_stream: Optional[torch.cuda.Stream] = None
    ) -> SingleLayerMoeLoadBalancer:
        """
        Add a new layer to the load balancer.

        Args:
            expert_count: The number of experts in the layer
            top_k: The number of experts each token selects
            slot_count_per_rank: The number of slots per rank
            aux_stream: The auxiliary stream for overlapping

        Returns:
            A SingleLayerMoeLoadBalancer instance for the new layer
        """
        single_layer_load_balancer_impl = self.load_balancer_impl.add_layer(
            expert_count, top_k, slot_count_per_rank)
        updates_enabled = not self.is_static_routing()
        repeat_count = 1
        if self.next_layer_repeated_count is not None:
            repeat_count = self.next_layer_repeated_count
            self.next_layer_repeated_count = None
        single_layer_load_balancer = SingleLayerMoeLoadBalancer(
            single_layer_load_balancer_impl,
            self.shared_mpi_comm,
            expert_count,
            updates_enabled=updates_enabled,
            repeated_count=repeat_count,
            aux_stream=aux_stream)
        single_layer_load_balancer.set_shared_memory_base_name(
            self.shared_memory_base_name)
        self.single_layer_load_balancers.append(single_layer_load_balancer)
        return single_layer_load_balancer

    def register_weight_slots_after_to_cuda(self):
        """
        Register weights after model has been moved to cuda, should be invoked after model.to("cuda") and before finalize_model.
        """
        for layer in self.single_layer_load_balancers:
            layer.register_weight_slots_after_to_cuda()

    def finalize_model(self):
        """
        Finalize the model after all layers have been added.
        This must be called before starting any iterations.
        """

        # All shared memory create before this barrier.
        self.shared_mpi_comm.barrier()
        # All shared memory mapping after this barrier.
        for single_layer_load_balancer in self.single_layer_load_balancers:
            single_layer_load_balancer.py_finalize_model()
        self.load_balancer_impl.finalize_model()
        torch.cuda.empty_cache()

    def set_warm_up_iter_count(self, iter_count: int):
        """
        Set the number of warm-up iterations.

        Args:
            iter_count: The number of warm-up iterations
        """
        self.load_balancer_impl.set_warm_up_iter_count(iter_count)

    def set_iter_info(self, enable_statistic: Optional[bool],
                      enable_update_weights: Optional[bool]):
        if enable_statistic is not None:
            self.enable_statistic = enable_statistic
        if enable_update_weights is not None:
            self.enable_update_weights = enable_update_weights

    def start_iter(self):
        """
        Start a new iteration.
        """
        assert self.in_iter == False, "already in forward"
        self.in_iter = True
        self.load_balancer_impl.start_iter(self.iter_id, self.enable_statistic,
                                           self.enable_update_weights)

    def end_iter(self):
        """
        End the current iteration.

        Args:
            iter_id: The ID of the iteration to end
        """
        assert self.in_iter, "not in forward, cannot end_iter"
        self.load_balancer_impl.end_iter(self.iter_id)
        self.in_iter = False
        self.iter_id += 1

    def shutdown(self):
        """
        Shutdown the load balancer and release resources.
        """
        for single_layer_load_balancer in self.single_layer_load_balancers:
            single_layer_load_balancer.py_pre_shutdown_cleanup()
        self.load_balancer_impl.shutdown()
        # use this sync to make sure all the shm resources can be cleaned up
        self.shared_mpi_comm.barrier()
        for single_layer_load_balancer in self.single_layer_load_balancers:
            single_layer_load_balancer.py_post_shutdown_cleanup()
        self.shared_mpi_comm.barrier()
        self.is_shutdown = True

    def __repr__(self):
        """
        Return a string representation of the load balancer.

        Returns:
            A string representation of the load balancer
        """
        return f"MoeLoadBalancer(ep_rank={self.ep_rank}, ep_size={self.ep_size}, layer_updates_per_iter={self.layer_updates_per_iter})"

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            The MoeLoadBalancer instance
        """
        # Save the previous load balancer (if any)
        try:
            self._previous_balancer = getattr(_current_moe_load_balancer,
                                              'instance', None)
        except AttributeError:
            self._previous_balancer = None

        # Set the current instance as the active load balancer
        _current_moe_load_balancer.instance = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        Args:
            exc_type: The exception type
            exc_val: The exception value
            exc_tb: The exception traceback

        Returns:
            False to not suppress exceptions
        """
        # Restore the previous load balancer
        if self._previous_balancer is not None:
            _current_moe_load_balancer.instance = self._previous_balancer
        else:
            delattr(_current_moe_load_balancer, 'instance')

        # Do not suppress exceptions
        return False


moe_model_arch_list = [
    'DeepseekV3ForCausalLM',
    'MixtralForCausalLM',
    'Llama4ForConditionalGeneration',
    'Qwen2MoeForCausalLM',
    'Qwen3MoeForCausalLM',
]


def maybe_create_moe_load_balancer(
        model_config, mapping: Optional[Mapping]) -> Optional[MoeLoadBalancer]:
    ep_rank = model_config.mapping.moe_ep_rank
    ep_size = model_config.mapping.moe_ep_size
    model_arch = model_config.pretrained_config.architectures[0]
    using_ep = mapping and mapping.moe_ep_size > 1
    in_supported_model_arch = model_arch in moe_model_arch_list
    using_smart_router = mapping and mapping.moe_cluster_size > 1
    moe_load_balancer = nullcontext()
    if in_supported_model_arch and using_ep and not using_smart_router and model_config.moe_load_balancer is not None:
        model_config.moe_load_balancer.setup(ep_rank=ep_rank, ep_size=ep_size)
        if model_config.moe_load_balancer.layer_updates_per_iter > 0:
            # TODO: remove this when supported.
            # cpu_arch = platform.machine().lower()
            # assert cpu_arch == 'aarch64', "online load balancer only support aarch64, e.g. GB200 now, x86 coming soon."
            pass

        moe_load_balancer = MoeLoadBalancer(
            ep_rank=ep_rank,
            ep_size=ep_size,
            layer_updates_per_iter=model_config.moe_load_balancer.
            layer_updates_per_iter)
        logger.info(
            f"Created MoE LoadBalancer, layer_updates_per_iter={model_config.moe_load_balancer.layer_updates_per_iter}..."
        )
    return moe_load_balancer


class MoeLoadBalancerIterContext:

    def __init__(self,
                 moe_load_balancer: Optional[MoeLoadBalancer],
                 enable_statistic: Optional[bool] = None,
                 enable_updates: Optional[bool] = None):
        self.moe_load_balancer = moe_load_balancer
        self.enable_statistic = enable_statistic
        self.enable_updates = enable_updates

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            The MoeLoadBalancerIterContext instance
        """
        if self.moe_load_balancer is not None and not self.moe_load_balancer.is_static_routing(
        ):
            self.moe_load_balancer.set_iter_info(self.enable_statistic,
                                                 self.enable_updates)
            self.moe_load_balancer.start_iter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        Args:
            exc_type: The exception type
            exc_val: The exception value
            exc_tb: The exception traceback

        Returns:
            False to not suppress exceptions
        """
        if self.moe_load_balancer is not None and not self.moe_load_balancer.is_static_routing(
        ):
            self.moe_load_balancer.end_iter()
        return False


def get_moe_load_balancer() -> Optional[MoeLoadBalancer]:
    """
    Get the current active MoeLoadBalancer instance.

    Returns:
        The current active MoeLoadBalancer instance, or None if not in a MoeLoadBalancer context
    """
    try:
        return getattr(_current_moe_load_balancer, 'instance', None)
    except AttributeError:
        return None


def moe_load_balancer_set_repeated_for_next_layer(repeat_count: int):
    """
    Set repeated count for next Single Layer created.

    Args:
        repeat_count: repeated count
    """
    load_balancer = get_moe_load_balancer()
    if load_balancer is not None:
        load_balancer.set_repeated_for_next_layer(repeat_count)


def moe_load_balancer_add_single_layer(
    expert_count: int,
    top_k: int,
    slot_count_per_rank: int,
    aux_stream: Optional[torch.cuda.Stream] = None
) -> Optional[SingleLayerMoeLoadBalancer]:
    """
    Add a new layer to the current active MoeLoadBalancer.

    Args:
        expert_count: The number of experts in the layer
        top_k: The number of experts each token selects
        slot_count_per_rank: The number of slots per rank
        aux_stream: The auxiliary stream for overlapping

    Returns:
        A SingleLayerMoeLoadBalancer instance for the new layer, or None if not in a MoeLoadBalancer context
    """
    load_balancer = get_moe_load_balancer()
    if load_balancer is not None:
        return load_balancer.add_layer(expert_count, top_k, slot_count_per_rank,
                                       aux_stream)
    return None
