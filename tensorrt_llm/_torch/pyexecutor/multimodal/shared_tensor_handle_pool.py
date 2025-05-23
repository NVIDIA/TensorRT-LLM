import torch
import multiprocessing as mp
from collections import OrderedDict
import time
import signal
import logging
from queue import Empty
from typing import Any

logger = logging.getLogger(__name__)

class SharedTensorPool:
    """A pool for managing shared CUDA tensor handles across processes.

    This class manages a pool of shared CUDA tensor handles, ensuring proper cleanup
    of IPC resources when handles are no longer needed.

    Args:
        max_handles (int): Maximum number of active handles allowed in the pool.
        cleanup_timeout (float): Timeout in seconds for cleanup process operations.
    """

    def __init__(self, max_handles: int = 10, cleanup_timeout: float = 10.0):
        self.active_handles: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self.max_handles = max_handles
        self.cleanup_timeout = cleanup_timeout
        # TODO: This is problematic, as it will cause the new spawn process to inherit mpi context
        ctx = mp.get_context('spawn')
        self._lock = ctx.Lock()
        self.cleanup_queue = ctx.Queue()
        self.cleanup_process = ctx.Process(target=self._cleanup_worker, daemon=True)
        self.cleanup_process.start()

        # Verify cleanup process is running
        if not self.cleanup_process.is_alive():
            raise RuntimeError("Failed to start cleanup process")

    def _cleanup_worker(self) -> None:
        """Worker process that handles CUDA IPC cleanup.

        This process runs in the background and handles cleanup of CUDA IPC resources
        when requested through the cleanup queue.
        """
        def signal_handler(signum: int, frame: Any) -> None:
            logger.info("Cleanup worker received signal to terminate")
            return

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        while True:
            try:
                try:
                    task = self.cleanup_queue.get_nowait()
                    if task == "STOP":
                        torch.cuda.ipc_collect()
                        return  # Exit immediately on STOP signal
                    elif task == "CLEANUP":
                        torch.cuda.ipc_collect()
                except Empty:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}", exc_info=True)
                continue

    def _remove_handle(self, key: str) -> None:
        """Internal method to remove a handle without acquiring lock.

        Args:
            key: Identifier of the handle to remove
        """
        if key in self.active_handles:
            del self.active_handles[key]
            self.cleanup_queue.put("CLEANUP")

    def close_handle(self, key: str) -> None:
        """Close and remove a handle from the pool.

        Args:
            key: Identifier of the handle to close
        """
        with self._lock:
            self._remove_handle(key)

    def add_handle(self, key: str, tensor_info: Any) -> None:
        """Add a new tensor handle to the pool.

        Args:
            key: Unique identifier for the handle
            tensor_info: Information about the tensor to be stored
        """
        with self._lock:
            if len(self.active_handles) >= self.max_handles:
                oldest_key = next(iter(self.active_handles))
                self._remove_handle(oldest_key)  # Use internal method instead
            self.active_handles[key] = tensor_info

    def get_handle_count(self) -> int:
        """Get the current number of active handles.

        Returns:
            int: Number of active handles in the pool
        """
        with self._lock:
            return len(self.active_handles)

    def set_max_handles(self, max_handles: int) -> None:
        """Update the maximum number of allowed handles.

        Args:
            max_handles: New maximum number of handles
        """
        if max_handles < 1:
            raise ValueError("max_handles must be at least 1")

        # First update the max_handles value
        with self._lock:
            self.max_handles = max_handles
            if len(self.active_handles) <= max_handles:
                return
            # Get list of handles to remove while holding the lock
            handles_to_remove = list(self.active_handles.keys())[:-max_handles]

        # Remove handles outside the lock to reduce contention
        for key in handles_to_remove:
            try:
                self._remove_handle(key)
            except Exception as e:
                logger.error(f"Failed to remove handle {key}: {e}", exc_info=True)
                # Continue with other removals even if one fails

    def cleanup(self) -> None:
        """Clean up all handles and stop the cleanup worker.

        This method should be called when the pool is no longer needed.
        """
        with self._lock:
            self.active_handles.clear()
        self.cleanup_queue.put("STOP")
        self.cleanup_process.join(timeout=self.cleanup_timeout)
        if self.cleanup_process.is_alive():
            logger.warning("Cleanup process did not terminate within timeout")

class SharedTensorBuffer_NoCleanup:
    """A buffer for managing shared CUDA tensor reference.
    """

    def __init__(self, max_handles: int = 10):
        self.active_handles: OrderedDict[Any, torch.Tensor] = OrderedDict()
        self.max_handles = max_handles

    def _remove_handle(self, key: str) -> None:
        """Internal method to remove a handle without acquiring lock.

        Args:
            key: Identifier of the handle to remove
        """
        if key in self.active_handles:
            del self.active_handles[key]


    def add_handle(self, key: str, tensor_info: Any) -> None:
        """Add a new tensor handle to the pool.

        Args:
            key: Unique identifier for the handle
            tensor_info: Information about the tensor to be stored
        """
        if len(self.active_handles) >= self.max_handles:
            oldest_key = next(iter(self.active_handles))
            self._remove_handle(oldest_key)  # Use internal method instead
        self.active_handles[key] = tensor_info

# Global pool instance
_tensor_pool = None

def get_tensor_pool(async_ipc_release: bool = False):
    """Get or create the global tensor pool instance.

    This function ensures the tensor pool is created only when needed and after
    multiprocessing is properly set up.
    """
    global _tensor_pool
    if _tensor_pool is None:
        if async_ipc_release:
            _tensor_pool = SharedTensorPool()
        else:
            _tensor_pool = SharedTensorBuffer_NoCleanup()
    return _tensor_pool