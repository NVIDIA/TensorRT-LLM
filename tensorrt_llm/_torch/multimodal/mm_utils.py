import logging
import base64
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor, rebuild_tensor, rebuild_meta_tensor

logger = logging.getLogger(__name__)


class _SharedTensorRebuildMethodRegistry:
    """Registry for tensor rebuild methods with fixed keys for common methods.

    This class maintains a mapping of numeric keys to rebuild methods.
    Common methods are pre-registered with fixed keys for consistency.
    """
    # Fixed keys for common rebuild methods
    REBUILD_CUDA = 1
    REBUILD_CPU = 2
    REBUILD_META = 3

    _registry: Dict[int, Callable] = {}

    @classmethod
    def initialize(cls):
        """Initialize the registry with common rebuild methods."""
        # Register common methods with fixed keys
        cls._registry[cls.REBUILD_CUDA] = rebuild_cuda_tensor
        cls._registry[cls.REBUILD_CPU] = rebuild_tensor
        cls._registry[cls.REBUILD_META] = rebuild_meta_tensor

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
        if method == rebuild_meta_tensor:
            return cls.REBUILD_META
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
    """A class for sharing tensors between processes.

    This class provides a simple way to share tensors between processes
    using Python's multiprocessing mechanisms.
    """
    def __init__(self, method_key: int, tensor_handle: Dict[str, Any]):
        self.method_key = method_key
        self.tensor_handle = tensor_handle

    @staticmethod
    def handle_to_dict(tensor_handle) -> Dict[str, Any]:
        """Convert the shared tensor handle to a dictionary that can be serialized.

        This method converts the tensor handle information into a format that can be
        safely serialized (e.g., to JSON). It handles binary data by encoding it in base64.

        Returns:
            Dictionary containing the serialized tensor information with the following keys:
            - method_key: The registry key for the rebuild method
            - tensor_size: List of tensor dimensions
            - tensor_stride: List of tensor strides
            - tensor_offset: Offset in the storage
            - dtype: String representation of the tensor's data type
            - storage_device: Device where the tensor is stored
            - storage_handle: Base64 encoded storage handle
            - storage_size_bytes: Size of the storage in bytes
            - storage_offset_bytes: Offset in the storage in bytes
            - requires_grad: Whether the tensor requires gradients
            - ref_counter_handle: Base64 encoded reference counter handle
            - ref_counter_offset: Offset in the reference counter
            - event_handle: Base64 encoded CUDA event handle
            - event_sync_required: Whether CUDA event synchronization is required

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
                "tensor_size": list(tensor_info[1]),
                "tensor_stride": list(tensor_info[2]),
                "tensor_offset": tensor_info[3],
                "dtype": str(tensor_info[5]),
                "storage_device": tensor_info[6],
                "storage_handle": base64.b64encode(tensor_info[7]).decode('utf-8'),
                "storage_size_bytes": tensor_info[8],
                "storage_offset_bytes": tensor_info[9],
                "requires_grad": tensor_info[10],
                "ref_counter_handle": base64.b64encode(tensor_info[11]).decode('utf-8'),
                "ref_counter_offset": tensor_info[12],
                "event_handle": base64.b64encode(tensor_info[13]).decode('utf-8'),
                "event_sync_required": tensor_info[14]
            }
            return serializable_info
        except IndexError as e:
            raise KeyError(f"Missing required tensor information: {e}")
        except Exception as e:
            raise ValueError(f"Failed to serialize tensor information: {e}")

    @staticmethod
    def dict_to_handle(tensor_info: Dict[str, Any]) -> Tuple:
        """Create a tensor handle from a serialized dictionary.

        This method reconstructs a tensor handle from a previously serialized
        dictionary. It handles base64 encoded binary data by decoding it back to bytes.

        Args:
            tensor_info: Dictionary containing the serialized tensor information
                with the same keys as returned by to_dict()

        Returns:
            A new SharedTensorContainer instance

        Raises:
            KeyError: If required tensor information is missing
            ValueError: If tensor information cannot be deserialized
        """
        try:
            # Decode base64 encoded binary data
            storage_handle = base64.b64decode(tensor_info['storage_handle'])
            ref_counter_handle = base64.b64decode(tensor_info['ref_counter_handle'])
            event_handle = base64.b64decode(tensor_info['event_handle'])

            # Reconstruct the tensor handle
            tensor_handle = (torch.Tensor,
                             tuple(tensor_info['tensor_size']),
                             tuple(tensor_info['tensor_stride']),
                             tensor_info['tensor_offset'],
                             torch.storage.TypedStorage,
                             eval(tensor_info['dtype']),
                             tensor_info['storage_device'],
                             storage_handle,
                             tensor_info['storage_size_bytes'],
                             tensor_info['storage_offset_bytes'],
                             tensor_info['requires_grad'],
                             ref_counter_handle,
                             tensor_info['ref_counter_offset'],
                             event_handle,
                             tensor_info['event_sync_required'])

            return tensor_handle
        except KeyError as e:
            raise KeyError(f"Missing required tensor information: {e}")
        except Exception as e:
            raise ValueError(f"Failed to deserialize tensor information: {e}")


    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'SharedTensorContainer':
        """Create a SharedTensorContainer from a local tensor.

        Args:
            tensor: The tensor to share

        Returns:
            SharedTensorContainer instance that can be shared between processes
        """
        rebuild_method, tensor_handle = reduce_tensor(tensor)
        method_key = _SharedTensorRebuildMethodRegistry.register(rebuild_method)
        # hack to make it serializable
        tensor_handle = SharedTensorContainer.handle_to_dict(tensor_handle)
        return cls(method_key, tensor_handle)

    @classmethod
    def from_dict(cls, tensor_info: Dict[str, Any]) -> 'SharedTensorContainer':
        """Create a SharedTensorContainer from a serialized dictionary.
        """
        method_key = tensor_info['method_key']
        tensor_handle = SharedTensorContainer.dict_to_handle(tensor_info)
        return cls(method_key, tensor_handle)

    def get_local_view(self) -> torch.Tensor:
        """Convert the shared tensor back to a local tensor.

        Returns:
            The reconstructed tensor
        """
        rebuild_method = _SharedTensorRebuildMethodRegistry.get_method(self.method_key)
        return rebuild_method(*self.tensor_handle)

    def dump_to_dict(self) -> Dict[str, Any]:
        """Convert this class instance to a dictionary that can be JSON serialized.

        Returns:
            Dictionary containing the serialized tensor information
        """
        result = self.tensor_handle.copy()
        result["method_key"] = self.method_key
        return result
