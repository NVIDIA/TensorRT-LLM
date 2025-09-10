import math
from typing import Optional

import torch

# Intention to have this buffer is to reuse buffer tensors across graph and non-graph
# situation (across layer/round).
# When forward is under graph capturing, one stream is created and all tensors' memory
# is associated with this stream and be kept in a graph pool. Then, all buffer memory
# allocated during graph capture won't be released back to allocator/system.
# Then, in non-graph mode, additional buffers are allocated which give bigger pressure
# on memory consumption at runtime.
# Timeline example:
#   [t0] start cudagraph capture
#   [t1] A = torch.zeros(....) -> allocate buffer A and put into graph pool
#   [t2] end cudagraph capture


#   [t3] in non-graph forward
#   [t4] A = torch.zeros(....) -> allocate buffer A in allocator but not use memory in cudagraph pool
#        OOM may happen
# NOTE: it requires all tensors with the same identifier (buffer_name here) have the same dtype. Will
#       upgrade this.
class Buffers:
    """
+    Manages CUDA memory buffers for reuse between graph and non-graph contexts.
+
+    This buffer manager maintains two separate caches:
+    - allocated_buffer_in_graph_pool: Buffers allocated during CUDA graph capture
+    - allocated_buffer_in_runtime: Buffers allocated during regular runtime
+
+    Note: This implementation is not thread-safe. If used in multi-threaded contexts,
+    external synchronization is required.
+
+    Attributes:
+        allocated_buffer_in_graph_pool: Dict mapping buffer names to lists of graph-captured tensors
+        allocated_buffer_in_runtime: Dict mapping buffer names to single runtime tensors
+    """

    def __init__(self):
        self.allocated_buffer_in_graph_pool: dict[str, list[torch.Tensor]] = {}
        self.allocated_buffer_in_runtime: dict[str, torch.Tensor] = {}

    def get_buffer(self, tensor_shape: list[int], dtype: torch.dtype,
                   buffer_name: str, pin_memory: bool):

        def select_buffer_with_more_elements(
            graph_buffer: Optional[torch.Tensor],
            runtime_buffer: Optional[torch.Tensor]
        ) -> tuple[Optional[torch.Tensor], bool]:
            if graph_buffer is None:
                return runtime_buffer, False
            if runtime_buffer is None:
                return graph_buffer, True
            use_runtime = runtime_buffer.numel() > graph_buffer.numel()
            return (runtime_buffer if use_runtime else graph_buffer,
                    not use_runtime)

        capture_graph = pin_memory
        numel_like = math.prod(tensor_shape)
        runtime_buffer = None
        if buffer_name in self.allocated_buffer_in_runtime:
            buffer = self.allocated_buffer_in_runtime[buffer_name]
            numel_buffer = buffer.numel()
            runtime_buffer = buffer if numel_buffer >= numel_like else runtime_buffer

        graph_buffer = None
        # Safely get the list of candidates. Defaults to an empty list if key is missing.
        candidate_buffers = self.allocated_buffer_in_graph_pool.get(
            buffer_name, [])
        for buffer in candidate_buffers:
            numel_buffer = buffer.numel()
            # buffer just needs to be large enough.
            if numel_buffer >= numel_like:
                graph_buffer = buffer
                break

        if capture_graph and graph_buffer is not None:
            return graph_buffer[0:numel_like].view(tensor_shape)
        else:
            buffer, use_graph = select_buffer_with_more_elements(
                graph_buffer, runtime_buffer)
            if buffer is not None:
                if not use_graph and capture_graph:
                    self.allocated_buffer_in_graph_pool.setdefault(
                        buffer_name, []).append(buffer)
                    del self.allocated_buffer_in_runtime[buffer_name]
                return buffer[0:numel_like].view(tensor_shape)

        # Reach here, no buffer is found. Then, we will use a new buffer to replace the small one. Release the memory first.
        if buffer_name in self.allocated_buffer_in_runtime:
            del self.allocated_buffer_in_runtime[buffer_name]

        # If we get here, no suitable buffer was found in the cache. Create a new one.
        new_buffer = torch.zeros(tensor_shape, device='cuda', dtype=dtype)
        if capture_graph:
            self.allocated_buffer_in_graph_pool.setdefault(
                buffer_name, []).append(new_buffer)
        else:
            self.allocated_buffer_in_runtime[buffer_name] = new_buffer
        return new_buffer


_buffer = Buffers()


def get_memory_buffer():
    global _buffer
    return _buffer
