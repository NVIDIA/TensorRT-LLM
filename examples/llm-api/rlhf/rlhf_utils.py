import torch

from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.logger import logger


class WorkerExtension:
    """
    Worker extension class for extending TensorRT-LLM Ray workers with custom functionality.

    This class can be injected into tensorrt_llm.LLM() by specifying it via the
    ray_worker_extension_cls parameter in LLMArgs when using orchestrator_type='ray'.
    The extension methods will be available on each Ray worker and can be called via
    the LLM's collective RPC mechanism.

    Examples:
        Creating an LLM with worker extension:

        >>> llm = LLM(
        ...     model=model_dir,
        ...     orchestrator_type='ray',
        ...     ray_worker_extension_cls='rlhf_utils.WorkerExtension'
        ... )

        Calling extension methods via collective RPC:

        >>> llm._collective_rpc('update_weights', args=(ipc_handles,))
    """

    @control_action_decorator
    def update_weights(self, ipc_handles: dict):
        """
        Update model weights from IPC (Inter-Process Communication) handles.

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
            logger.info(f"Update weights from IPC handles")
            device_uuid = get_device_uuid(self.device_id)

            if device_uuid not in ipc_handles:
                raise ValueError(
                    f"Device UUID {device_uuid} not found in ipc_handles")

            weights = {}
            all_handles = ipc_handles[device_uuid]

            for param_name, tensor_handle in all_handles:
                func, args = tensor_handle
                list_args = list(args)
                list_args[6] = self.device_id  # Set target device
                tensor = func(*list_args)
                weights[param_name] = tensor

            self.engine.model_engine.model.load_weights(weights)
            torch.cuda.synchronize()
            self.engine.reset_prefix_cache()

        except Exception as e:
            logger.error(f"Encountered an error in update_weights")
            raise e

    def check_weights_updated(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.engine.model_engine.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated
