import threading
from typing import List, Optional

import torch

import tensorrt_llm.bindings.internal.runtime as _tbr


class SingleLayerMoeLoadBalancer:
    """
    A class representing a single layer of the Mixture of Experts (MoE) load balancer.
    This class wraps the C++ implementation of SingleLayerMoeLoadBalancer.
    """

    def __init__(
            self,
            single_layer_load_balancer_impl: _tbr.SingleLayerMoeLoadBalancer):
        """
        Initialize a SingleLayerMoeLoadBalancer instance.

        Args:
            single_layer_load_balancer_impl: The C++ implementation of SingleLayerMoeLoadBalancer
        """
        self.single_layer_load_balancer_impl = single_layer_load_balancer_impl
        self.single_layer_load_balancer_ptr = single_layer_load_balancer_impl.get_pointer(
        )

    def add_weight_slot(self, slot_id: int, name: str,
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

    def add_host_weight(self, expert_id: int, name: str,
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

    def set_initial_weight_assignments(self,
                                       initial_weight_assignments: List[int]):
        """
        Set the initial weight assignments for the layer.

        Args:
            initial_weight_assignments: A list of initial weight assignments
        """
        self.single_layer_load_balancer_impl.set_initial_weight_assignments(
            initial_weight_assignments)

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

    def route(self, token_selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Route the tokens to experts.

        Args:
            token_selected_experts: The experts selected by each token

        Returns:
            A tensor of routed slot IDs
        """
        return torch.ops.trtllm.moe_load_balance_routing(
            token_selected_experts, self.single_layer_load_balancer_ptr)


# Global variable to store the current active MoeLoadBalancer instance
_current_moe_load_balancer = threading.local()


class MoeLoadBalancer:
    """
    A class representing a Mixture of Experts (MoE) load balancer.
    This class can be used as a context manager to manage the lifecycle of a MoeLoadBalancer.
    """

    def __init__(self, ep_rank: int, ep_size: int, layer_updates_per_iter: int):
        """
        Initialize a MoeLoadBalancer instance.

        Args:
            ep_rank: The rank of the current process in expert parallelism
            ep_size: The total number of processes in expert parallelism
            layer_updates_per_iter: The number of layers to update per iteration
        """
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.layer_updates_per_iter = layer_updates_per_iter
        self.load_balancer_impl = _tbr.MoeLoadBalancer(ep_rank, ep_size,
                                                       layer_updates_per_iter)
        self._previous_balancer = None

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
        return SingleLayerMoeLoadBalancer(single_layer_load_balancer_impl)

    def finalize_model(self):
        """
        Finalize the model after all layers have been added.
        This must be called before starting any iterations.
        """
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
        self.load_balancer_impl.shutdown()

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
