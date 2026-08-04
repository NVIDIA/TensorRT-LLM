import base64
import logging
from typing import Any, Callable, Dict, Tuple

import torch
from torch.multiprocessing import get_sharing_strategy, set_sharing_strategy
from torch.multiprocessing.reductions import (rebuild_cuda_tensor,
                                              rebuild_storage_filename,
                                              rebuild_tensor,
                                              rebuild_typed_storage,
                                              reduce_storage, reduce_tensor)

logger = logging.getLogger(__name__)

DTYPE_MAPPING = {
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.bool': torch.bool,
    'torch.uint8': torch.uint8,
    'torch.int8': torch.int8,
    'torch.int16': torch.int16,
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16,
}


def str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype
    """
    if dtype_str not in DTYPE_MAPPING:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported dtypes are: {list(DTYPE_MAPPING.keys())}"
        )
    return DTYPE_MAPPING[dtype_str]


class _SharedTensorRebuildMethodRegistry:
    """Registry for tensor rebuild methods with fixed keys for common methods.

    This registry is used by the SharedTensorContainer to manage PyTorch tensor
    rebuild methods on a consumer process.

    This class maintains a mapping of numeric keys to rebuild methods.
    Common methods are pre-registered with fixed keys for consistency.
    """
    # Fixed keys for common rebuild methods
    REBUILD_CUDA = 1
    REBUILD_CPU = 2

    _registry: Dict[int, Callable] = {}

    @classmethod
    def initialize(cls):
        """Initialize the registry with common rebuild methods."""
        # Register common methods with fixed keys
        cls._registry[cls.REBUILD_CUDA] = rebuild_cuda_tensor
        cls._registry[cls.REBUILD_CPU] = rebuild_tensor

    @classmethod
    def register(cls, method: Callable) -> int:
        """Register a rebuild method and return its key.

        Args:
            method: The rebuild method to register

        Returns:
            The numeric key assigned to the method
        """
        if method == rebuild_cuda_tensor:
            return cls.REBUILD_CUDA
        if method == rebuild_tensor:
            return cls.REBUILD_CPU
        raise NotImplementedError("Other rebuild methods are not supported yet")

    @classmethod
    def get_method(cls, key: int) -> Callable:
        """Get a rebuild method by its key.

        Args:
            key: The numeric key of the method

        Returns:
            The registered rebuild method

        Raises:
            KeyError: If the key is not found in the registry
        """
        if key not in cls._registry:
            raise KeyError(f"No rebuild method registered with key {key}")
        return cls._registry[key]


class SharedTensorContainer:
    """A class for sharing tensors between processes (non-Python multiprocessing processes).

    This is an intermediate solution to accommodate communication between two independent processes
    that are not spawned by Python multiprocessing. It uses shared memory or CUDA IPC to avoid
    serialization/IPC costs when communicating tensors between processes.

    Key Features:
    - Uses PyTorch's reduction methods (reduce_tensor, rebuild_tensor, etc.) to handle tensor sharing
    - Supports both CPU tensors (via shared memory) and CUDA tensors (via CUDA IPC)
    - Provides serialization capabilities for cross-process communication
    - Avoids expensive tensor serialization/deserialization/IPC overhead

    Architecture:
    - Producer process: Creates tensor -> reduces -> convert handle to dict -> sends via IPC
    - Consumer process: Receives dict -> convert dict to handle -> rebuilds tensor -> gets local view

    Note: This is a temporary solution. In the future, if we embrace more Python environment
    or PyTorch orchestration methods (like torch.multiprocessing, Ray, etc.), this intermediate
    layer would not be needed as those frameworks provide native tensor sharing capabilities.

    Note: Whenever you call reduce_tensor, you must call the corresponding rebuild method at
    consumer process(es), otherwise, the producer process cannot release the memory to caching
    allocator as the inner refcount never reaches zero.

    Note: This module can also be extended to transfer CUDA tensors between different GPUs managed by different processes
    using CE (Copy Engine) or torch.to() to initiate direct P2P transfers. This requires CUDA P2P support, i.e.,
    torch.cuda.can_device_access_peer(src_device, dst_device) must return True for the source and destination devices.

    """

    def __init__(self, method_key: int, tensor_handle: Dict[str, Any]):
        """Initialize the SharedTensorContainer.

        Args:
            method_key: Registry key for the rebuild method (CUDA, CPU, etc.)
            tensor_handle: Tensor handle that can be used to rebuild the tensor on a consumer process
        """
        self.method_key = method_key
        self.tensor_handle = tensor_handle

    @staticmethod
    def cuda_handle_to_dict(tensor_handle) -> Dict[str, Any]:
        """Convert CUDA tensor handle to serializable dictionary for IPC

        This method converts PyTorch's CUDA tensor reduction (cudaIPC) handle into a format that can be
        safely serialized and transmitted between any two processes. It handles binary data by
        encoding it in base64 to ensure JSON compatibility.

        The CUDA handle contains references to GPU memory, CUDA events, and reference counters
        that need to be properly serialized for cross-process sharing via CUDA IPC.

        Returns:
            Dictionary containing the serialized CUDA tensor information

        Raises:
            KeyError: If required tensor information is missing
            ValueError: If tensor information cannot be serialized
        """
        try:
            # tensor_handle is a tuple returned by reduce_tensor
            tensor_info = tensor_handle
            # Convert tensor info to a basic dict with only serializable values
            serializable_info = {
                # tensor_info[0] is the type of the tensor, which is "torch.Tensor"
                "tensor_size":
                list(tensor_info[1]),
                "tensor_stride":
                list(tensor_info[2]),
                "tensor_offset":
                tensor_info[3],
                "dtype":
                str(tensor_info[5]),
                "storage_device":
                tensor_info[6],
                "storage_handle":
                base64.b64encode(tensor_info[7]).decode('utf-8'),
                "storage_size_bytes":
                tensor_info[8],
                "storage_offset_bytes":
                tensor_info[9],
                "requires_grad":
                tensor_info[10],
                "ref_counter_handle":
                base64.b64encode(tensor_info[11]).decode('utf-8'),
                "ref_counter_offset":
                tensor_info[12],
                "event_handle":
                base64.b64encode(tensor_info[13]).decode('utf-8'),
                "event_sync_required":
                tensor_info[14]
            }
            return serializable_info
        except IndexError as e:
            raise KeyError(f"Missing required tensor information: {e}")
        except Exception as e:
            raise ValueError(f"Failed to serialize tensor information: {e}")

    @staticmethod
    def cpu_handle_to_dict(meta_data: Dict[str, Any],
                           storage_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert CPU tensor handle to serializable dictionary for IPC

        This method converts PyTorch's CPU tensor reduction handle into a format that can be
        safely serialized and transmitted between any two processes. For CPU tensors, we use
        shared memory via file_system sharing strategy as fd strategy is not serializable.

        Args:
            meta_data: Tensor metadata (size, stride, offset, etc.)
            storage_metadata: Storage metadata (handle, size, dtype, etc.)

        Returns:
            Dictionary containing the serialized CPU tensor information
        """
        try:
            serializable_info = {
                "tensor_storage_offset":
                meta_data[0],
                "tensor_size":
                list(meta_data[1]),
                "tensor_stride":
                list(meta_data[2]),
                "manager_handle":
                base64.b64encode(storage_metadata[0]).decode('utf-8'),
                "storage_handle":
                base64.b64encode(storage_metadata[1]).decode('utf-8'),
                "storage_size":
                storage_metadata[2],
                "storage_dtype":
                str(storage_metadata[3])
            }
            return serializable_info
        except IndexError as e:
            raise KeyError(f"Missing required tensor information: {e}")
        except Exception as e:
            raise ValueError(f"Failed to serialize tensor information: {e}")

    @staticmethod
    def dict_to_cuda_handle(tensor_info: Dict[str, Any]) -> Tuple:
        """Reconstruct CUDA tensor handle from serialized dictionary

        This method reconstructs a CUDA tensor handle from a previously serialized dictionary
        that was received from another process.

        Args:
            tensor_info: Dictionary containing the serialized CUDA tensor information
                with the same keys as returned by cuda_handle_to_dict()

        Returns:
            A tuple representing the CUDA tensor handle for PyTorch's rebuild_cuda_tensor
        """
        try:
            # Decode base64 encoded binary data
            storage_handle = base64.b64decode(tensor_info['storage_handle'])
            ref_counter_handle = base64.b64decode(
                tensor_info['ref_counter_handle'])
            event_handle = base64.b64decode(tensor_info['event_handle'])

            # Reconstruct the tensor handle
            tensor_handle = (torch.Tensor, tuple(tensor_info['tensor_size']),
                             tuple(tensor_info['tensor_stride']),
                             tensor_info['tensor_offset'],
                             torch.storage.TypedStorage,
                             str_to_torch_dtype(tensor_info['dtype']),
                             tensor_info['storage_device'], storage_handle,
                             tensor_info['storage_size_bytes'],
                             tensor_info['storage_offset_bytes'],
                             tensor_info['requires_grad'], ref_counter_handle,
                             tensor_info['ref_counter_offset'], event_handle,
                             tensor_info['event_sync_required'])

            return tensor_handle
        except KeyError as e:
            raise KeyError(f"Missing required tensor information: {e}")
        except Exception as e:
            raise ValueError(f"Failed to deserialize tensor information: {e}")

    @staticmethod
    def dict_to_cpu_handle(tensor_info: Dict[str, Any]) -> Tuple:
        """Reconstruct CPU tensor handle from serialized dictionary

        This method reconstructs a CPU tensor handle from a previously serialized dictionary
        that was received from another process.

        The reconstructed handle allows the consumer process to access the same memory
        region that was shared by the producer process, avoiding data copying.

        Args:
            tensor_info: Dictionary containing the serialized CPU tensor information
                with the same keys as returned by cpu_handle_to_dict()

        Returns:
            A tuple representing the CPU tensor handle for PyTorch's rebuild_tensor
        """
        try:
            manager_handle = base64.b64decode(tensor_info['manager_handle'])
            storage_handle = base64.b64decode(tensor_info['storage_handle'])
            storage_metadata = (torch.storage.TypedStorage, manager_handle,
                                storage_handle, tensor_info['storage_size'],
                                str_to_torch_dtype(
                                    tensor_info['storage_dtype']))
            storage = rebuild_storage_filename(*storage_metadata)
            if not isinstance(storage, torch.storage.TypedStorage):
                storage = rebuild_typed_storage(
                    storage, str_to_torch_dtype(tensor_info['storage_dtype']))

            meta_data = (tensor_info['tensor_storage_offset'],
                         tuple(tensor_info['tensor_size']),
                         tuple(tensor_info['tensor_stride']), False
                         )  # requires_grad is always False for cpu tensor
            tensor_handle = (torch.Tensor, storage, meta_data)
            return tensor_handle
        except KeyError as e:
            raise KeyError(f"Missing required tensor information: {e}")
        except Exception as e:
            raise ValueError(f"Failed to deserialize tensor information: {e}")

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'SharedTensorContainer':
        """Create a SharedTensorContainer from a local tensor (Producer side).

        This method is called by the producer process to prepare a tensor for sharing
        with other processes. It uses PyTorch's reduction methods to create a handle
        that can be efficiently transmitted and reconstructed by consumer processes.

        - CUDA tensors: Uses CUDA IPC for GPU memory sharing
        - CPU tensors: Uses file_system sharing strategy for shared memory

        Args:
            tensor: The tensor to share

        Returns:
            SharedTensorContainer instance that can be serialized later for IPC
        """
        rebuild_method, tensor_handle = reduce_tensor(tensor)
        method_key = _SharedTensorRebuildMethodRegistry.register(rebuild_method)
        return cls(method_key, tensor_handle)

    @classmethod
    def from_dict(cls, tensor_info: Dict[str, Any]) -> 'SharedTensorContainer':
        """Create a SharedTensorContainer from a serialized dictionary (Consumer side).

        This method is called by the consumer process to reconstruct a SharedTensorContainer
        from serialized data received via IPC.

        Args:
            tensor_info: Dictionary containing the serialized tensor information
                received from the producer process via IPC

        Returns:
            SharedTensorContainer instance ready for tensor reconstruction

        Raises:
            ValueError: If the method_key is not supported
        """
        method_key = tensor_info['method_key']
        if method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CUDA:
            tensor_handle = SharedTensorContainer.dict_to_cuda_handle(
                tensor_info)
        elif method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CPU:
            tensor_handle = SharedTensorContainer.dict_to_cpu_handle(
                tensor_info)
        else:
            raise ValueError(
                f"Unsupported shared tensor method key: {method_key}")
        return cls(method_key, tensor_handle)

    def get_local_view(self) -> torch.Tensor:
        """Convert the shared tensor back to a local tensor (Consumer side).

        This method is called by the consumer process to obtain the actual tensor
        from the SharedTensorContainer.

        Returns:
            The reconstructed tensor
        """
        rebuild_method = _SharedTensorRebuildMethodRegistry.get_method(
            self.method_key)
        return rebuild_method(*self.tensor_handle)

    def dump_to_dict(self) -> Dict[str, Any]:
        """Convert this container to a dictionary for direct IPC (Producer side).

        This method is called by the producer process to serialize the SharedTensorContainer instance
        into a format that is JSON compatible and can be transmitted via IPC.

        Returns:
            Dictionary containing the serialized tensor information
        """
        if self.method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CUDA:
            tensor_dict = SharedTensorContainer.cuda_handle_to_dict(
                self.tensor_handle)
        elif self.method_key == _SharedTensorRebuildMethodRegistry.REBUILD_CPU:
            sharing_strategy = get_sharing_strategy()
            # Here we use file_system sharing strategy to make it serializable between two non-python independent processes
            set_sharing_strategy("file_system")
            storage = self.tensor_handle[1]
            meta_data = self.tensor_handle[2]
            storage_handle = reduce_storage(storage)
            # restore the original sharing strategy
            set_sharing_strategy(sharing_strategy)
            # exclude the first element which is the type of the storage
            storage_metadata = storage_handle[-1][1:]
            tensor_dict = SharedTensorContainer.cpu_handle_to_dict(
                meta_data, storage_metadata)
        else:
            raise ValueError(f"Unsupported tensor device: {self.method_key}")

        tensor_dict["method_key"] = self.method_key
        return tensor_dict
