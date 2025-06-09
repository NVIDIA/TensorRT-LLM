import torch
from collections import OrderedDict
from typing import Any

class SharedTensorHandleBuffer:
    """ This is a container to temporary hold the shared tensor handles recv by consumer process.
    Everytime when the consumer is done with accessing the shared tensor from producer, to avoid immediate 
    calling of release/close the cudaIPC handle (it could introduce severe overheads), we buffered it. For many 
    scenarios, it can be helpful.

    TODO: In fact, how is the impact of cudaIPC overhead needs to be studied. Ideally, we should manage a shared tensor pool
    in the producer; therefore we can avoid such overhead of open/close cudaIPC handles in consumer processes.
    
    Hopefully, NIXL integration can help address this issue.
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
            self._remove_handle(oldest_key)
        self.active_handles[key] = tensor_info

_tensor_pool = None

def get_handle_buffer():
    """Get or create the global tensor pool instance.

    This function ensures the tensor pool is created only when needed and after
    multiprocessing is properly set up.
    """
    global _tensor_pool
    if _tensor_pool is None:
        _tensor_pool = SharedTensorHandleBuffer()
        
    return _tensor_pool