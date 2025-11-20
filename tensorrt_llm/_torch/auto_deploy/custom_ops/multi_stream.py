"""
Custom ops to enable multi-stream execution.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Dict, Tuple

import torch


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
# In multi-gpu scenario, each GPU/rank has its own CudaStreamManager.
class CudaStreamManager(metaclass=_Singleton):
    AUX_STREAM_NAME = "aux"
    MAIN_STREAM_NAME = "main"

    def __init__(self) -> None:
        # In case __init__ ever gets called twice, guard against re-init
        if hasattr(self, "streams"):
            return

        self._lock = RLock()

        # Events needed for stream synchronization
        self.events: Dict[str, Any] = {
            self.AUX_STREAM_NAME: torch.cuda.Event(),
            self.MAIN_STREAM_NAME: torch.cuda.Event(),
        }

        # Streams for multi-stream execution
        self.aux_stream = torch.cuda.Stream()
        self.streams: Dict[str, Any] = {
            self.AUX_STREAM_NAME: self.aux_stream,
            self.MAIN_STREAM_NAME: torch.cuda.default_stream(),
        }


cuda_stream_manager = CudaStreamManager()


@torch.library.custom_op("auto_deploy::record_event", mutates_args=())
def record_event(stream_name: str) -> None:
    event = cuda_stream_manager.events[stream_name]
    event.record()


@torch.library.custom_op("auto_deploy::wait_event", mutates_args=())
def wait_event(event_name: str) -> None:
    event = cuda_stream_manager.events[event_name]
    event.wait()


# skip during compilation
@torch._dynamo.disable
def record_event_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    output = fn(*args, **kwargs)
    torch.ops.auto_deploy.record_event(cuda_stream_manager.MAIN_STREAM_NAME)
    return output


@torch._dynamo.disable
def aux_stream_wrapper(
    fn: Callable,
    *args: Tuple[Any, ...],
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    stream_name = cuda_stream_manager.AUX_STREAM_NAME
    with torch.cuda.stream(cuda_stream_manager.streams[stream_name]):
        torch.ops.auto_deploy.wait_event(cuda_stream_manager.MAIN_STREAM_NAME)
        output = fn(*args, **kwargs)
        torch.ops.auto_deploy.record_event(cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(cuda_stream_manager.AUX_STREAM_NAME)
    return output


# bf16
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
    with torch.cuda.stream(cuda_stream_manager.streams[cuda_stream_manager.AUX_STREAM_NAME]):
        torch.ops.auto_deploy.wait_event(cuda_stream_manager.MAIN_STREAM_NAME)
        output = torch.ops.auto_deploy.trtllm_moe_fused(
            x,
            selected_experts,
            routing_weights,
            w3_w1_stacked_weight,
            w2_stacked_weight,
            mlp_style,
            act_fn,
        )
        torch.ops.auto_deploy.record_event(cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(cuda_stream_manager.AUX_STREAM_NAME)
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
    with torch.cuda.stream(cuda_stream_manager.streams[cuda_stream_manager.AUX_STREAM_NAME]):
        torch.ops.auto_deploy.wait_event(cuda_stream_manager.MAIN_STREAM_NAME)
        output = torch.ops.auto_deploy.triton_moe_fused(
            x,
            selected_experts,
            routing_weights,
            w1_stacked_weight,
            w2_stacked_weight,
        )
        torch.ops.auto_deploy.record_event(cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(cuda_stream_manager.AUX_STREAM_NAME)
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
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    with torch.cuda.stream(cuda_stream_manager.streams[cuda_stream_manager.AUX_STREAM_NAME]):
        torch.ops.auto_deploy.wait_event(cuda_stream_manager.MAIN_STREAM_NAME)
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
            mlp_style,
            act_fn,
        )
        torch.ops.auto_deploy.record_event(cuda_stream_manager.AUX_STREAM_NAME)
    torch.ops.auto_deploy.wait_event(cuda_stream_manager.AUX_STREAM_NAME)
    return output


@trtllm_quant_fp8_moe_fused_aux.register_fake
def trtllm_quant_fp8_moe_fused_fake(
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
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)
