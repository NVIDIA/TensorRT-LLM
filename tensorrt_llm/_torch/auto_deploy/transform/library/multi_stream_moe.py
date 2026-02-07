"""Transform for multi-stream execution of MoE layers that have shared experts and routed experts."""

from threading import RLock
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.fx import GraphModule

from tensorrt_llm._torch.utils import ActivationType

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


# Previously, CudaStreamManager and the custom ops that use the cuda streams and events were
# placed in custom_ops folder. However doing so resulted in CudaStreamManager
# being created only in the parent process, but we need each rank to have its own CudaStreamManager that
# manages the cuda streams and events for that rank. Placing the logic to instantiate
# CudaStreamManager and the custom ops that use the cuda streams and events at the transform level ensures that
# each rank has its own CudaStreamManager since each rank applies the transform independently.
class _Singleton(type):
    _instances: Dict[type, Any] = {}
    _lock = RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:  # double-checked locking
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# A singleton that holds the pointers to the cuda streams and events.
# Each device has its own cuda streams and events.
class CudaStreamManager(metaclass=_Singleton):
    AUX_STREAM_NAME = "aux"
    MAIN_STREAM_NAME = "main"
    devices: List[torch.device] = []
    events: Dict[torch.device, Dict[str, Any]] = {}
    streams: Dict[torch.device, Dict[str, Any]] = {}

    def __init__(self) -> None:
        # In case __init__ ever gets called twice, guard against re-init
        if hasattr(self, "streams"):
            return

        self._lock = RLock()
        self.add_device(torch.cuda.current_device())

    def add_device(self, device: int) -> None:
        if device not in self.devices:
            self.devices.append(device)
            with torch.cuda.device(device):
                self.events[device] = {
                    self.AUX_STREAM_NAME: torch.cuda.Event(),
                    self.MAIN_STREAM_NAME: torch.cuda.Event(),
                }
                self.streams[device] = {
                    self.AUX_STREAM_NAME: torch.cuda.Stream(),
                    self.MAIN_STREAM_NAME: torch.cuda.default_stream(),
                }
        else:
            ad_logger.warning(f"CudaStreamManager: Device {device} already added")

    def get_stream(self, device: int, stream_name: str) -> torch.cuda.Stream:
        return self.streams[device][stream_name]

    def get_event(self, device: int, event_name: str) -> torch.cuda.Event:
        return self.events[device][event_name]


# Every device will have a singleton instance of CudaStreamManager.
cuda_stream_manager = CudaStreamManager()


@torch.library.custom_op("auto_deploy::record_event", mutates_args=())
def record_event(device: int, stream_name: str) -> None:
    event = cuda_stream_manager.get_event(device, stream_name)
    event.record()


@torch.library.custom_op("auto_deploy::wait_event", mutates_args=())
def wait_event(device: int, stream_name: str) -> None:
    event = cuda_stream_manager.get_event(device, stream_name)
    event.wait()


@torch._dynamo.disable
def record_event_passthrough(
    x: torch.Tensor,
    *,
    device: int = -1,
) -> torch.Tensor:
    """Record a CUDA event on the main stream and return the input unchanged.

    Inserted after the gating/routing computation to mark a synchronization
    point.  The aux stream waits for this event before starting the MoE
    computation, enabling overlap between the shared expert (main stream)
    and routed experts (aux stream).
    """
    if device < 0:
        device = torch.cuda.current_device()
    torch.ops.auto_deploy.record_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    return x


@torch._dynamo.disable
def aux_stream_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    stream_name = cuda_stream_manager.AUX_STREAM_NAME
    device = kwargs.pop("device", torch.cuda.current_device())
    with torch.cuda.stream(cuda_stream_manager.get_stream(device, stream_name)):
        torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
        output = fn(*args, **kwargs)
        torch.ops.auto_deploy.record_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    return output


# trtllm bf16
@torch.library.custom_op("auto_deploy::trtllm_moe_fused_aux", mutates_args=())
def trtllm_moe_fused_aux(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    device = torch.cuda.current_device()
    with torch.cuda.stream(
        cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    ):
        torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
        output = torch.ops.auto_deploy.trtllm_moe_fused(
            x,
            selected_experts,
            routing_weights,
            w3_w1_stacked_weight,
            w2_stacked_weight,
            is_gated_mlp,
            act_fn,
        )
        torch.ops.auto_deploy.record_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    return output


@trtllm_moe_fused_aux.register_fake
def trtllm_moe_fused_aux_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    return torch.empty_like(x)


# triton bf16
@torch.library.custom_op("auto_deploy::triton_moe_fused_aux", mutates_args=())
def triton_moe_fused_aux(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    device = torch.cuda.current_device()
    with torch.cuda.stream(
        cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    ):
        torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
        output = torch.ops.auto_deploy.triton_moe_fused(
            x,
            selected_experts,
            routing_weights,
            w1_stacked_weight,
            w2_stacked_weight,
        )
        torch.ops.auto_deploy.record_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    return output


@triton_moe_fused_aux.register_fake
def triton_moe_fused_aux_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::trtllm_quant_fp8_moe_fused_aux", mutates_args=())
def trtllm_quant_fp8_moe_fused_aux(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_act_scale: torch.Tensor,
    fc1_dequant_scale: torch.Tensor,
    fc2_act_scale_reciprocal: torch.Tensor,
    fc2_dequant_scale: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    device = torch.cuda.current_device()
    with torch.cuda.stream(
        cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    ):
        torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
        output = torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused(
            x,
            selected_experts,
            routing_weights,
            fc1_expert_weights,
            fc2_expert_weights,
            fc1_act_scale,
            fc1_dequant_scale,
            fc2_act_scale_reciprocal,
            fc2_dequant_scale,
            is_gated_mlp,
            act_fn,
        )
        torch.ops.auto_deploy.record_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(device, cuda_stream_manager.AUX_STREAM_NAME)
    return output


@trtllm_quant_fp8_moe_fused_aux.register_fake
def trtllm_quant_fp8_moe_fused_aux_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_act_scale: torch.Tensor,
    fc1_dequant_scale: torch.Tensor,
    fc2_act_scale_reciprocal: torch.Tensor,
    fc2_dequant_scale: torch.Tensor,
    is_gated_mlp: bool = True,
    act_fn: int = int(ActivationType.Silu),
) -> torch.Tensor:
    return torch.empty_like(x)


def _execute_op_in_aux_stream(
    gm: GraphModule, op_dict: Dict[Callable, Callable]
) -> Tuple[GraphModule, int]:
    """Replace MoE ops with aux-stream variants and insert CUDA event synchronisation.

    For each MoE op we:
      1. Identify its *routing* inputs (selected_experts, routing_weights) —
         everything except the hidden-state tensor (``args[0]``) and static
         weight parameters (``get_attr`` nodes).
      2. Insert ``record_event_passthrough`` right after the latest routing
         input in graph order.  This records a CUDA event on the main stream
         *after* gating is done but *before* the shared expert starts,
         enabling the shared expert (main stream) and routed experts (aux
         stream) to execute concurrently.
      3. Replace the original MoE op with its ``_aux`` variant that runs on
         the auxiliary CUDA stream.
    """
    graph = gm.graph
    num_replaced = 0

    # Collect targets first to avoid mutating while iterating
    target_nodes = [n for n in graph.nodes if is_op(n, op_dict.keys())]

    # Topological position map — used to find the latest routing input.
    node_order = {node: i for i, node in enumerate(graph.nodes)}

    for n in target_nodes:
        # MoE ops follow the convention (x, selected_experts, routing_weights, ...weights...).
        # args[0] is the hidden-state tensor; the remaining *compute* (non-get_attr)
        # inputs are routing results produced by the gating computation.
        hidden_state_node = n.args[0]
        routing_inputs = [
            inp
            for inp in n.all_input_nodes
            if inp is not hidden_state_node and inp.op != "get_attr"
        ]
        assert routing_inputs, f"No routing inputs found for MoE node {n}"

        # Place the event record right after the last routing dependency so
        # that the aux stream's wait_event sees all routing results as ready
        # while the shared expert (later in graph order) can still overlap.
        latest_routing = max(routing_inputs, key=lambda inp: node_order.get(inp, 0))

        with graph.inserting_after(latest_routing):
            rec_node = graph.call_function(
                record_event_passthrough,
                args=(latest_routing,),
                kwargs={"device": torch.cuda.current_device()},
            )

        # Wire the event node into the MoE op so there is a true data
        # dependency that prevents reordering.
        n.args = tuple(rec_node if arg is latest_routing else arg for arg in n.args)

        # Replace MoE op with its aux-stream variant
        with graph.inserting_after(n):
            new_node = graph.call_function(op_dict[n.target], args=n.args, kwargs=n.kwargs)
        n.replace_all_uses_with(new_node)
        graph.erase_node(n)
        num_replaced += 1

    return gm, num_replaced


@TransformRegistry.register("multi_stream_moe")
class MultiStreamMOE(BaseTransform):
    """Multi-stream execution of MoE layers that have shared experts and routed experts."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        op_dict = {
            torch.ops.auto_deploy.trtllm_moe_fused: torch.ops.auto_deploy.trtllm_moe_fused_aux,
            torch.ops.auto_deploy.triton_moe_fused: torch.ops.auto_deploy.triton_moe_fused_aux,
            torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused: torch.ops.auto_deploy.trtllm_quant_fp8_moe_fused_aux,
        }
        with open("before_multi_stream.txt", "w") as f:
            f.write(str(gm.graph))
        # Ensure that aux stream and events for the current device are added to the CudaStreamManager.
        cuda_stream_manager.add_device(torch.cuda.current_device())
        gm, num_matches = _execute_op_in_aux_stream(gm, op_dict)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        with open("after_multi_stream.txt", "w") as f:
            f.write(str(gm.graph))
        return gm, info
