import atexit
import threading
from multiprocessing import shared_memory
from typing import Callable, List, Optional

import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings.internal.runtime as _tbr


def _tensor_to_weight(t: torch.Tensor) -> _tbr.MoeWeight:
    """
    Convert a tensor to a MoeWeight object.

    Args:
        t: The tensor to convert
    """
    assert t.dim() <= 2, "t.dim() should be less than or equal to 2"
    shape = [1, 1]
    pitch = 1
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
    mw.width = shape[1]
    mw.pitch = pitch
    mw.weight_ptr = t.data_ptr()
    return mw


class HostMoeTensorSharer:
    """
    A class representing a host tensor sharer.
    """

    def __init__(self, layer_id: int, shared_mpi_comm: MPI.Comm):
        """
        Initialize a HostMoeTensorSharer instance.

        Args:
            shared_mpi_comm: The MPI communicator for shared memory
        """
        self.shared_mpi_comm = shared_mpi_comm
        self.layer_id = layer_id
        self.shared_memory_base_name = None
        self.host_tensor_shapes = []
        self.host_weights = {}
        self.own_shms = {}
        self.all_shms = []

    def set_shared_memory_base_name(self, shared_memory_base_name):
        """
        Set the shared memory base name for the layer.

        Args:
            shared_memory_base_name: The base name for the shared memory
        """
        self.shared_memory_base_name = shared_memory_base_name

    def get_shared_memory_name(self, expert_id: int, name: str):
        """
        Get the shared memory name for the layer.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
        """
        assert isinstance(self.shared_memory_base_name,
                          str), "self.shared_memory_base_name must be a string"
        shared_memory_name = f"{self.shared_memory_base_name}_l{self.layer_id}_e{expert_id}_{name}"
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
        self.host_tensor_shapes.append((expert_id, name, dtype, tensor_shape))

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
        assert t.is_contiguous() == True, "t.is_contiguous() must be True"
        shm_name = self.get_shared_memory_name(expert_id, name)
        shm = shared_memory.SharedMemory(name=shm_name,
                                         create=True,
                                         size=t.numel() * t.element_size())
        shm.buf[:t.numel() * t.element_size()] = t.numpy().tobytes()
        dtype = t.dtype
        tensor_shape = t.shape
        t = torch.frombuffer(shm.buf,
                             dtype=dtype).view(tensor_shape).pin_memory()
        key = (expert_id, name)
        assert key not in self.host_weights.keys(), f"key={key} already exists"
        self.host_weights[key] = t
        self.own_shms[(expert_id, name)] = shm
        self.all_shms.append(shm)
        atexit.register(shm.unlink)

    def finalize_host_tensor_sharing(self, add_host_weight_fn: Callable = None):
        """
        Finalize the host tensor sharing.
        """
        for expert_weight_info in self.host_tensor_shapes:
            expert_id, name, dtype, tensor_shape = expert_weight_info
            shm_name = self.get_shared_memory_name(expert_id, name)
            shm = shared_memory.SharedMemory(name=shm_name)
            self.all_shms.append(shm)
            t = torch.frombuffer(shm.buf,
                                 dtype=dtype).view(tensor_shape).pin_memory()
            key = (expert_id, name)
            assert key not in self.host_weights.keys(
            ), f"key={key} already exists"
            self.host_weights[key] = t

        if add_host_weight_fn is not None:
            for key, t in self.host_weights.items():
                add_host_weight_fn(key[0], key[1], t)

        self.host_weights.clear()

    def pre_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        for shm in self.all_shms:
            shm.close()


class SingleLayerMoeLoadBalancer:
    """
    A class representing a single layer of the Mixture of Experts (MoE) load balancer.
    This class wraps the C++ implementation of SingleLayerMoeLoadBalancer.
    """

    def __init__(
            self,
            single_layer_load_balancer_impl: _tbr.SingleLayerMoeLoadBalancer,
            shared_mpi_comm: MPI.Comm):
        """
        Initialize a SingleLayerMoeLoadBalancer instance.

        Args:
            single_layer_load_balancer_impl: The C++ implementation of SingleLayerMoeLoadBalancer
            shared_mpi_comm: The MPI communicator for shared memory
        """
        self.single_layer_load_balancer_impl = single_layer_load_balancer_impl
        self.single_layer_load_balancer_ptr = single_layer_load_balancer_impl.get_pointer(
        )
        layer_id = self.single_layer_load_balancer_impl.get_layer_id()
        self.host_tensor_sharer = HostMoeTensorSharer(shared_mpi_comm, layer_id)

    def set_shared_memory_base_name(self, shared_memory_base_name):
        """
        Set the shared memory base name for the layer.

        Args:
            shared_memory_base_name: The base name for the shared memory
        """
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
        self.single_layer_load_balancer_impl.add_weight_slot(
            slot_id, name, weight_slot)

    def register_weight_slot(self, slot_id: int, name: str, t: torch.Tensor):
        """
        Register a weight slot to the layer.

        Args:
            slot_id: The ID of the slot
            name: The name of the weight
            t: The weight tensor
        """
        moe_weight = _tensor_to_weight(t)
        self._add_weight_slot(slot_id, name, moe_weight)

    def _add_host_weight(self, expert_id: int, name: str,
                         host_weight: _tbr.MoeWeight):
        """
        Add a host weight to the layer.

        Args:
            expert_id: The ID of the expert
            name: The name of the weight
            host_weight: The host weight object
        """
        self.single_layer_load_balancer_impl.add_host_weight(
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

    def py_finalize_model(self):
        """
        Finalize the model after all layers have been added.
        This must be called before starting any iterations.
        """
        self.host_tensor_sharer.finalize_host_tensor_sharing(
            self._add_host_weight_from_tensor)

    def wait_for_gpu_stage(self) -> torch.Tensor:
        """
        Wait for the GPU stage to complete.

        Returns:
            A tensor indicating whether the stage is enabled
        """
        return torch.ops.trtllm.moe_load_balance_wait_gpu_stage(
            self.single_layer_load_balancer_ptr)

    def set_cpu_stage(self):
        """
        Set the CPU stage.
        """
        torch.ops.trtllm.moe_load_balance_set_cpu_stage(
            self.single_layer_load_balancer_ptr)

    def statistic(self, gathered_raw_expert_ids: torch.Tensor,
                  enabled: torch.Tensor, is_first_stage: bool,
                  is_last_stage: bool):
        """
        Perform statistics on the expert IDs.

        Args:
            gathered_raw_expert_ids: The gathered raw expert IDs from all ranks
            enabled: A tensor indicating whether the operation is enabled
            is_first_stage: Whether this is the first stage
            is_last_stage: Whether this is the last stage
        """
        torch.ops.trtllm.moe_load_balance_statistic(
            gathered_raw_expert_ids, enabled,
            self.single_layer_load_balancer_ptr, is_first_stage, is_last_stage)

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
        return torch.ops.trtllm.moe_load_balance_routing(
            token_selected_experts, offset_by_ep_rank,
            self.single_layer_load_balancer_ptr)

    def py_pre_shutdown_cleanup(self):
        """
        Clean up the resources before C++ shutdown and barrier
        """
        self.host_tensor_sharer.pre_shutdown_cleanup()


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
                 shared_memory_base_name: str = 'moe_shared'):
        """
        Initialize a MoeLoadBalancer instance.

        Args:
            ep_rank: The rank of the current process in expert parallelism
            ep_size: The total number of processes in expert parallelism
            layer_updates_per_iter: The number of layers to update per iteration
            shared_memory_base_name: Shared memory base name
        """
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.layer_updates_per_iter = layer_updates_per_iter
        self.load_balancer_impl = _tbr.MoeLoadBalancer(ep_rank, ep_size,
                                                       layer_updates_per_iter)
        self._previous_balancer = None
        self.single_layer_load_balancers = []
        self.shared_memory_base_name = shared_memory_base_name
        self._setup_mpi_comm()

    def _setup_mpi_comm(self):
        global_mpi_comm = tensorrt_llm.mpi_comm()
        shared_mpi_comm = global_mpi_comm.Split_type(
            split_type=MPI.COMM_TYPE_SHARED)
        shared_size = shared_mpi_comm.Get_size()
        local_size = tensorrt_llm.local_mpi_size()
        assert shared_size == local_size, \
            f"Interesting, shared size {shared_size} is not same as local size {local_size}"
        self.shared_mpi_comm = shared_mpi_comm

    def add_layer(self, expert_count: int, top_k: int,
                  slot_count_per_rank: int) -> SingleLayerMoeLoadBalancer:
        """
        Add a new layer to the load balancer.

        Args:
            expert_count: The number of experts in the layer
            top_k: The number of experts each token selects
            slot_count_per_rank: The number of slots per rank

        Returns:
            A SingleLayerMoeLoadBalancer instance for the new layer
        """
        single_layer_load_balancer_impl = self.load_balancer_impl.add_layer(
            expert_count, top_k, slot_count_per_rank)
        single_layer_load_balancer = SingleLayerMoeLoadBalancer(
            single_layer_load_balancer_impl, self.shared_mpi_comm)
        single_layer_load_balancer.set_shared_memory_base_name(
            self.shared_memory_base_name)
        self.single_layer_load_balancers.append(single_layer_load_balancer)
        return single_layer_load_balancer

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

    def set_warm_up_iter_count(self, iter_count: int):
        """
        Set the number of warm-up iterations.

        Args:
            iter_count: The number of warm-up iterations
        """
        self.load_balancer_impl.set_warm_up_iter_count(iter_count)

    def start_iter(self, iter_id: int, enable_statistic: bool,
                   enable_update_weights: bool):
        """
        Start a new iteration.

        Args:
            iter_id: The ID of the iteration
            enable_statistic: Whether to enable statistics collection
            enable_update_weights: Whether to enable weight updates
        """
        self.load_balancer_impl.start_iter(iter_id, enable_statistic,
                                           enable_update_weights)

    def end_iter(self, iter_id: int):
        """
        End the current iteration.

        Args:
            iter_id: The ID of the iteration to end
        """
        self.load_balancer_impl.end_iter(iter_id)

    def shutdown(self):
        """
        Shutdown the load balancer and release resources.
        """
        for single_layer_load_balancer in self.single_layer_load_balancers:
            single_layer_load_balancer.py_pre_shutdown_cleanup()
        self.load_balancer_impl.shutdown()
        # use this sync to make sure all the shm resources can be cleaned up
        self.shared_mpi_comm.barrier()

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


def moe_load_balancer_add_single_layer(
        expert_count: int, top_k: int,
        slot_count_per_rank: int) -> Optional[SingleLayerMoeLoadBalancer]:
    """
    Add a new layer to the current active MoeLoadBalancer.

    Args:
        expert_count: The number of experts in the layer
        top_k: The number of experts each token selects
        slot_count_per_rank: The number of slots per rank

    Returns:
        A SingleLayerMoeLoadBalancer instance for the new layer, or None if not in a MoeLoadBalancer context
    """
    load_balancer = get_moe_load_balancer()
    if load_balancer is not None:
        return load_balancer.add_layer(expert_count, top_k, slot_count_per_rank)
    return None
