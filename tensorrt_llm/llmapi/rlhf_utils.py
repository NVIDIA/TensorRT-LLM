# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import base64
import gc
from contextlib import contextmanager
from typing import Iterator, Optional

import torch

from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.executor.ray.utils import control_action_decorator
from tensorrt_llm.llmapi import serialization
from tensorrt_llm.logger import logger


@contextmanager
def _preserve_cuda_graph_refit_caches(model: torch.nn.Module) -> Iterator[None]:
    """Keep refit-derived tensor addresses stable across post-load hooks.

    Qwen3.5 eager fusion caches ``weight + 1`` for Gemma RMSNorm in the plain
    tensor attribute ``_fused_norm_weight``. Its post-load hook recomputes that
    cache by assigning a new tensor, but a CUDA Graph captured before refit
    continues to reference the original allocation. Preserve that allocation
    during refit and copy the refreshed value into it after post-load hooks run.
    """
    cached_tensors = []
    for module in model.modules():
        cached = getattr(module, "_fused_norm_weight", None)
        if isinstance(cached, torch.Tensor):
            cached_tensors.append((module, cached))

    yield

    for module, original in cached_tensors:
        refreshed = getattr(module, "_fused_norm_weight", None)
        if refreshed is original:
            continue
        if not isinstance(refreshed, torch.Tensor):
            raise RuntimeError("Refit removed the CUDA-Graph-visible _fused_norm_weight cache")
        if (
            refreshed.shape != original.shape
            or refreshed.dtype != original.dtype
            or refreshed.device != original.device
        ):
            raise RuntimeError(
                "Refit changed the shape, dtype, or device of the CUDA-Graph-visible "
                "_fused_norm_weight cache"
            )
        original.copy_(refreshed)
        module._fused_norm_weight = original


class WorkerExtension:
    """Worker extension class for extending TensorRT-LLM Ray workers with custom functionality.

    This class can be injected into tensorrt_llm.LLM() by specifying it via the
    ray_worker_extension_cls parameter in LLMArgs when using orchestrator_type='ray'.
    The extension methods will be available on each Ray worker and can be called via
    the LLM's collective RPC mechanism.

    Examples:
        Creating an LLM with worker extension:

        >>> llm = LLM(
        ...     model=model_dir,
        ...     orchestrator_type="ray",
        ...     ray_worker_extension_cls="rlhf_utils.WorkerExtension",
        ... )

        Calling extension methods via collective RPC:

        >>> llm._collective_rpc("update_weights", args=(ipc_handles,))
    """

    def finalize_weight_update(self) -> None:
        """Finalize a refit and refresh post-load state safely for CUDA Graph replay."""
        model_engine = self.engine.model_engine
        model = model_engine.model
        with _preserve_cuda_graph_refit_caches(model):
            model_engine.model_loader.finalize_update_weights()
            for module in model.modules():
                if hasattr(module, "process_weights_after_loading") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.process_weights_after_loading()
                if hasattr(module, "post_load_weights") and not getattr(
                    module, "_weights_removed", False
                ):
                    module.post_load_weights()

    @control_action_decorator
    def update_weights(self, ipc_handles: Optional[dict] = None):
        """Update model weights from IPC (Inter-Process Communication) handles.

        This method receives shared memory handles from another process (typically FSDP training),
        reconstructs tensors from these handles, and loads them into the TensorRT-LLM model.
        Uses the control_action_decorator to ensure all active requests are finished before
        updating weights.

        Args:
            ipc_handles: Dictionary mapping device UUIDs to lists of (param_name, tensor_handle) tuples.
                        Each tensor_handle is a tuple of (func, args) for reconstructing the tensor.

        Raises:
            ValueError: If the current device's UUID is not found in ipc_handles.
            Exception: Re-raises any exception encountered during weight update.
        """
        try:
            if not hasattr(self.engine.model_engine.model, "first_pre_reload_weights"):
                self.engine.model_engine.model_loader.begin_update_weights()
                for module in self.engine.model_engine.model.modules():
                    if hasattr(module, "pre_reload_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.pre_reload_weights()
                setattr(self.engine.model_engine.model, "first_pre_reload_weights", True)
            if ipc_handles is not None:
                logger.info("Update weights from IPC handles")
                device_uuid = get_device_uuid(self.device_id)

                if device_uuid not in ipc_handles:
                    raise ValueError(f"Device UUID {device_uuid} not found in ipc_handles")

                weights = {}

                serialized_handles = ipc_handles[device_uuid]
                if isinstance(serialized_handles, str):
                    # Data is base64-encoded pickled bytes - deserialize it
                    # using restricted unpickler from tensorrt_llm.llmapi.serialization
                    logger.info("Deserializing base64-encoded weight handles")
                    decoded_data = base64.b64decode(serialized_handles)
                    disallowed_imports = {
                        "torch.storage": ["_load_from_bytes"],
                        "torch.hub": ["_load_local"],
                        "torch": ["save"],
                    }
                    # CUDA IPC tensor handles serialize torch rebuild helpers.
                    # Keep deserialization default-deny by allowing only this
                    # call site to import torch symbols, with disallowed imports
                    # still taking precedence in serialization.Unpickler.
                    approved_imports = {
                        "builtins": [
                            "list",
                            "tuple",
                            "str",
                            "int",
                            "float",
                            "bool",
                            "bytes",
                            "dict",
                            "NoneType",
                            "type",
                        ],
                    }
                    all_handles = serialization.loads(
                        decoded_data,
                        approved_imports=approved_imports,
                        approved_module_patterns=[r"^torch.*"],
                        disallowed_imports=disallowed_imports,
                    )

                    # Verify the result is a list as expected
                    if not isinstance(all_handles, list):
                        raise ValueError(
                            f"Deserialized data must be a list, got {type(all_handles).__name__} instead"
                        )
                else:
                    # Data is already in the correct format (backward compatibility)
                    all_handles = serialized_handles

                for param_name, tensor_handle in all_handles:
                    func, args = tensor_handle
                    list_args = list(args)
                    list_args[6] = self.device_id
                    tensor = func(*list_args)
                    weights[param_name] = tensor

                logger.info(f"weights key size: {len(weights.keys())}")
                self.engine.model_engine.model_loader.reload(
                    self.engine.model_engine.model, weights, allow_partial_loading=True
                )
                del weights
                torch.cuda.ipc_collect()
            else:
                logger.info("Finalize update weights")
                self.finalize_weight_update()
                moe_load_balancer = getattr(self.engine.model_engine, "moe_load_balancer", None)
                if isinstance(moe_load_balancer, MoeLoadBalancer):
                    moe_load_balancer.register_weight_slots_after_to_cuda()
                    logger.info("moe_load_balancer finalizing model...")
                    moe_load_balancer.finalize_model()
                    logger.info("moe_load_balancer finalize model done")
                self.engine.reset_prefix_cache()
                delattr(self.engine.model_engine.model, "first_pre_reload_weights")

                torch.cuda.synchronize()
                # Done once after all buckets to avoid per-bucket cleanup overhead.
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

        except Exception as e:
            self.engine.model_engine.model_loader.abort_update_weights()
            if hasattr(self.engine.model_engine.model, "first_pre_reload_weights"):
                delattr(self.engine.model_engine.model, "first_pre_reload_weights")
            logger.error("Encountered an error in update_weights")
            raise e

    @control_action_decorator
    def reset_prefix_cache(self) -> None:
        """Invalidate the KV cache prefix reuse state after weight updates.

        Drains in-flight requests first, like update_weights(): clearing the reuse state
        detaches the whole radix tree, and a request that is still holding blocks from it
        would go on committing into the detached subtree.
        """
        self.engine.reset_prefix_cache()

    @control_action_decorator
    def wait_for_engine_idle(self) -> None:
        """Block until the engine has no active or queued requests."""
        pass

    def check_weights_updated(self) -> bool:
        """Check if the weights are updated to 0."""
        weights_updated = True
        for name, p in self.engine.model_engine.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated

    def start_profile(self):
        torch.cuda.profiler.start()

    def stop_profile(self):
        torch.cuda.profiler.stop()
