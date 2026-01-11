"""
Workaround for PyTorch tensor lifetime issue during CUDA graph capture.

This module provides a mechanism to keep tensors alive during CUDA graph capture
by storing Python references to them. This prevents premature destruction of tensors
returned from custom C++ operators that have custom deleters.

Issue: During CUDA graph capture, tensors returned from custom C++ operators
may have use_count=1, causing them to be destroyed immediately before PyTorch
binding can increment the reference count. This causes custom deleters to be
called prematurely, releasing buffers while NCCL operations are still using them.

Workaround: Store references to output tensors during graph capture, ensuring
they stay alive until graph execution completes.
"""

import threading
from typing import List, Union

import torch


class TensorLifetimeRegistry:
    """
    Thread-safe registry to store tensor references during CUDA graph capture.

    This ensures tensors returned from custom operators stay alive during graph
    capture and execution, preventing premature deleter calls.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # Store references per thread to handle multi-threaded scenarios
        self._thread_local = threading.local()

    def _get_storage(self) -> List[List[torch.Tensor]]:
        """Get thread-local storage for tensor references."""
        if not hasattr(self._thread_local, "tensor_refs"):
            self._thread_local.tensor_refs = []
        return self._thread_local.tensor_refs

    def register_tensors(self, tensors: Union[torch.Tensor, List[torch.Tensor], tuple]):
        """
        Register tensor(s) to keep them alive during graph capture.

        Args:
            tensors: Single tensor, list of tensors, or tuple of tensors to register
        """
        with self._lock:
            storage = self._get_storage()

            # Convert to list of tensors
            if isinstance(tensors, torch.Tensor):
                tensor_list = [tensors]
            elif isinstance(tensors, (list, tuple)):
                tensor_list = [t for t in tensors if isinstance(t, torch.Tensor)]
            else:
                return  # Not a tensor, ignore

            # Only register if we're in graph capture
            if self.is_capturing():
                storage.append(tensor_list)
                print(
                    f"[TensorLifetimeRegistry] Registered {len(tensor_list)} tensor(s) "
                    f"during graph capture (total batches: {len(storage)})"
                )

    def is_capturing(self) -> bool:
        """
        Check if we're currently in CUDA graph capture.

        Returns:
            True if currently capturing a CUDA graph, False otherwise
        """
        try:
            # Check if any stream is currently capturing
            # torch.cuda.is_current_stream_capturing() checks the current stream
            return torch.cuda.is_current_stream_capturing()
        except (AttributeError, RuntimeError):
            # Fallback: if the function doesn't exist or there's an error, assume not capturing
            return False

    def clear(self):
        """Clear all registered tensor references (call after graph execution completes)."""
        with self._lock:
            if hasattr(self._thread_local, "tensor_refs"):
                count = sum(len(batch) for batch in self._thread_local.tensor_refs)
                self._thread_local.tensor_refs.clear()
                print(f"[TensorLifetimeRegistry] Cleared {count} tensor reference(s)")

    def get_registered_count(self) -> int:
        """Get the number of registered tensor batches."""
        with self._lock:
            if hasattr(self._thread_local, "tensor_refs"):
                return len(self._thread_local.tensor_refs)
            return 0


# Global singleton instance
_tensor_registry = TensorLifetimeRegistry()


def register_tensor_references(tensors: Union[torch.Tensor, List[torch.Tensor], tuple]):
    """
    Register tensor(s) to keep them alive during CUDA graph capture.

    This is a convenience function that uses the global registry.

    Args:
        tensors: Single tensor, list of tensors, or tuple of tensors to register
    """
    _tensor_registry.register_tensors(tensors)


def clear_tensor_references():
    """Clear all registered tensor references."""
    _tensor_registry.clear()


def is_graph_capturing() -> bool:
    """Check if we're currently in CUDA graph capture."""
    return _tensor_registry.is_capturing()


def get_registered_count() -> int:
    """Get the number of registered tensor batches."""
    return _tensor_registry.get_registered_count()
