"""Transform for multi-stream execution of MoE layers that have shared experts and routed experts."""

from threading import RLock
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.fx import GraphModule

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


# skip during compilation
@torch._dynamo.disable
def record_event_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    device = kwargs.pop("device", torch.cuda.current_device())
    output = fn(*args, **kwargs)
    torch.ops.auto_deploy.record_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    return output


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
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
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
            mlp_style,
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
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
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


# trtllm fp8
@torch.library.custom_op("auto_deploy::trtllm_quant_fp8_moe_fused_aux", mutates_args=())
def trtllm_quant_fp8_moe_fused_aux(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,  # [E, I, H] stacked FP8 weights
    w2_weight: torch.Tensor,  # [E, H, I] stacked FP8 weights
    w3_weight: torch.Tensor,  # [E, I, H] for gated_mlp, unused for mlp
    w1_input_scale: torch.Tensor,  # [E] stacked input scales
    w2_input_scale: torch.Tensor,  # [E] stacked input scales
    w3_input_scale: torch.Tensor,  # [E] or unused
    w1_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w2_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w3_weight_scale: torch.Tensor,  # [E] or unused
    gemm1_dequant: torch.Tensor,  # [E]
    gemm2_act_quant: torch.Tensor,  # [E]
    gemm2_dequant: torch.Tensor,  # [E]
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
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
            w1_weight,
            w2_weight,
            w3_weight,
            w1_input_scale,
            w2_input_scale,
            w3_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
            gemm1_dequant,
            gemm2_act_quant,
            gemm2_dequant,
            mlp_style,
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
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    w1_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    w3_input_scale: torch.Tensor,
    w1_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w3_weight_scale: torch.Tensor,
    gemm1_dequant: torch.Tensor,
    gemm2_act_quant: torch.Tensor,
    gemm2_dequant: torch.Tensor,
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)


def _execute_op_in_aux_stream(
    gm: GraphModule, op_dict: Dict[Callable, Callable]
) -> Tuple[GraphModule, int]:
    graph = gm.graph
    num_replaced = 0

    # Collect targets first to avoid mutating while iterating
    target_nodes = [n for n in graph.nodes if is_op(n, op_dict.keys())]

    for n in target_nodes:
        target_input_node = None
        for input_node in n.all_input_nodes:
            if input_node.target == torch.ops.aten.view.default:
                target_input_node = input_node
                break

        assert target_input_node is not None, f"Target input node not found for node {n}"
        with graph.inserting_before(target_input_node):
            kwargs = target_input_node.kwargs.copy()
            kwargs["device"] = torch.cuda.current_device()
            new_node = graph.call_function(
                record_event_wrapper,
                args=(target_input_node.target, *target_input_node.args),
                kwargs=kwargs,
            )
        target_input_node.replace_all_uses_with(new_node)
        graph.erase_node(target_input_node)
        with graph.inserting_after(n):
            new_node = graph.call_function(op_dict[n.target], args=n.args, kwargs=n.kwargs)
        n.replace_all_uses_with(new_node)
        graph.erase_node(n)
        num_replaced += 1
    if num_replaced:
        graph.eliminate_dead_code()
        graph.lint()
        gm.recompile()

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
        # Ensure that aux stream and events for the current device are added to the CudaStreamManager.
        cuda_stream_manager.add_device(torch.cuda.current_device())
        gm, num_matches = _execute_op_in_aux_stream(gm, op_dict)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=False,
            has_valid_shapes=False,
        )

        return gm, info
