from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from cuda.bindings import driver as cuda_driver
from cuda.bindings import runtime as cudart
from mpi4py.MPI import COMM_WORLD

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._utils import global_mpi_rank, nvtx_range
from tensorrt_llm.llmapi.llm_args import DwdpConfig

# Parameter names to collect handles for
WEIGHT_PARAMS = ["w3_w1_weight", "w2_weight"]
BIAS_PARAMS = ["w3_w1_bias", "w2_bias"]
# Quant scale params vary by quantization method
QUANT_SCALE_PARAMS = [
    "w3_w1_weight_scale",
    "w2_weight_scale",  # NVFP4/MXFP4
    "fc31_alpha",
    "fc2_alpha",  # NVFP4 alpha
]


_global_dwdp_manager: Optional["DwdpManager"] = None


def set_global_dwdp_manager(manager: "DwdpManager"):
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


def check_cuda_error(err, context: str = ""):
    """Check CUDA error."""
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error in {context}: {err}")


class DwdpLayerHandleCollector:
    """
    Dwdp Layer Handle Collector for IPC handle coordination and prefetch buffer management.
    """

    def __init__(
        self,
        layer_idx: int,
    ):
        self.layer_idx = layer_idx

        # Local IPC handles: param_name -> handle_bytes
        self.local_ipc_handles: Dict[str, bytes] = {}
        # Local pointers: param_name -> data_ptr (for verification)
        self.local_ptrs: Dict[str, int] = {}
        # Local offsets: param_name -> offset from allocation base
        # IPC handle points to allocation base, we need offset to get actual tensor data
        self.local_offsets: Dict[str, int] = {}
        # Parameter shapes: param_name -> shape (without expert dim)
        self.param_shapes: Dict[str, torch.Size] = {}
        # Parameter dtypes: param_name -> dtype
        self.param_dtypes: Dict[str, torch.dtype] = {}
        # Peer pointers: (peer_rank, param_name) -> ptr (already adjusted with offset)
        self.peer_ptrs: Dict[Tuple[int, str], int] = {}

    def register_weights(self, module: nn.Module):
        """
        Register weights from a MoE module and create IPC handles.

        Called after module.load_weights() completes.

        Args:
            module: The MoE module with loaded weights
        """
        params_to_register = []
        # Weights (check if present and not None)
        for param_name in WEIGHT_PARAMS:
            if hasattr(module, param_name) and getattr(module, param_name, None) is not None:
                params_to_register.append(param_name)
        # Bias (optional)
        if hasattr(module, "bias"):
            params_to_register.extend(BIAS_PARAMS)
        # Quant scales (optional, depends on quant method)
        for param_name in QUANT_SCALE_PARAMS:
            if hasattr(module, param_name) and getattr(module, param_name, None) is not None:
                params_to_register.append(param_name)

        # Register each parameter
        for param_name in params_to_register:
            param = getattr(module, param_name)
            if isinstance(param, nn.Parameter):
                param = param.data
            if param is None:
                continue
            if not param.is_cuda or not param.is_contiguous():
                raise ValueError(f"Parameter {param_name} is not on GPU or is not contiguous")
            self._register_param(param_name, param)

    def _register_param(self, param_name: str, param: torch.Tensor):
        # Get IPC handle - note: handle points to the CUDA allocation base, not tensor's data_ptr
        tensor_ptr = param.data_ptr()
        err, handle = cudart.cudaIpcGetMemHandle(tensor_ptr)
        check_cuda_error(err, f"get handle for {param_name}")

        # Get allocation base address using Driver API cuMemGetAddressRange
        # This returns the actual base address and size of the CUDA allocation
        # cudaPointerGetAttributes.devicePointer returns the input pointer, not base!
        err, alloc_base, alloc_size = cuda_driver.cuMemGetAddressRange(tensor_ptr)
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemGetAddressRange failed for {param_name}: {err}")

        # Calculate offset from allocation base
        # Convert CUdeviceptr to int for arithmetic
        offset = tensor_ptr - int(alloc_base)

        self.local_ipc_handles[param_name] = bytes(handle.reserved)
        self.local_ptrs[param_name] = tensor_ptr
        self.local_offsets[param_name] = offset
        self.param_shapes[param_name] = param.shape[1:]
        self.param_dtypes[param_name] = param.dtype

    def get_peer_ptr(self, peer_rank: int, param_name: str) -> int:
        """Get pointer to parameter on peer rank."""
        return self.peer_ptrs[(peer_rank, param_name)]

    def cleanup(self):
        """Clean up peer handles."""
        for _, ptr in self.peer_ptrs.items():
            cudart.cudaIpcCloseMemHandle(ptr)
        self.peer_ptrs.clear()


class DwdpPrefetchBuffer:
    """
    Ping-pong buffer for expert weight prefetching.

    Buffer Selection Strategy:
    - Even layers (0, 2, 4, ...) use buffer[0]
    - Odd layers (1, 3, 5, ...) use buffer[1]
    - This ensures layer N-1's prefetch doesn't overwrite layer N's data

    Synchronization Strategy:
    - prefetch_events[buffer_idx][layer_idx]: Recorded when prefetch completes
      Waited by forward() before using prefetched data
    - compute_events[buffer_idx][layer_idx]: Recorded when forward() completes
      Waited by next prefetch before overwriting buffer

    Buffer Layout (organized by rank):
    - buffers[buffer_idx][param_name] = List[Optional[Tensor]]
    - len(list) == dwdp_size
    - list[peer_rank] = Tensor[num_prefetch_experts, ...] for peer_rank != dwdp_rank
    - list[dwdp_rank] = None (local weight used directly, not prefetched)
    """

    def __init__(
        self,
        dwdp_size: int,
        dwdp_rank: int,
        num_experts_per_worker: int,
        num_prefetch_experts: int,
        num_layers: int,
        first_moe_layer_idx: int,
        param_shapes: Dict[str, torch.Size],
        param_dtypes: Dict[str, torch.dtype],
    ):
        self.dwdp_size = dwdp_size
        self.num_prefetch_experts = num_prefetch_experts
        self.num_experts_per_worker = num_experts_per_worker
        self.num_layers = num_layers
        self.first_moe_layer_idx = first_moe_layer_idx
        self.num_buffers = 2  # Ping-pong
        self.dwdp_rank = dwdp_rank

        self.param_shapes = param_shapes
        self.param_dtypes = param_dtypes

        self.device = torch.cuda.current_device()

        # buffers[buffer_idx][param_name] = List[Optional[Tensor]]
        # list[peer_rank] contains prefetched weights from that rank
        # list[dwdp_rank] = None (local weights used directly)
        self.buffers: List[Dict[str, List[Optional[torch.Tensor]]]] = []

        for _ in range(self.num_buffers):
            buffer = {}
            for param_name, shape in param_shapes.items():
                dtype = param_dtypes[param_name]
                # Pre-allocate list of length dwdp_size, one slot per rank
                # tensor_list[dwdp_rank] = None (local weights used directly)
                # tensor_list[peer_rank] = Tensor for prefetched weights from peer
                tensor_list: List[Optional[torch.Tensor]] = [None] * dwdp_size
                for peer_rank in range(dwdp_size):
                    if peer_rank != dwdp_rank:
                        buffer_shape = (self.num_prefetch_experts,) + tuple(shape)
                        tensor_list[peer_rank] = torch.empty(
                            buffer_shape,
                            dtype=dtype,
                            device=self.device,
                        )
                buffer[param_name] = tensor_list
            self.buffers.append(buffer)

        self.max_layer_idx = num_layers + first_moe_layer_idx
        self.prefetch_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(self.max_layer_idx // self.num_buffers + 1)]
            for _ in range(self.num_buffers)
        ]
        self.compute_events: List[List[torch.cuda.Event]] = [
            [torch.cuda.Event() for _ in range(self.max_layer_idx // self.num_buffers + 1)]
            for _ in range(self.num_buffers)
        ]
        self.prefetch_stream = torch.cuda.Stream(device=self.device)

    def initialize_compute_events(self):
        for buffer_idx in range(self.num_buffers):
            self.compute_events[buffer_idx][0].record(torch.cuda.current_stream())

    def record_prefetch_event(self, layer_idx: int):
        self.prefetch_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers].record(
            self.prefetch_stream
        )

    def record_compute_event(self, layer_idx: int):
        self.compute_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers].record(
            torch.cuda.current_stream()
        )

    def wait_prefetch_event(self, layer_idx: int):
        torch.cuda.current_stream().wait_event(
            self.prefetch_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers]
        )

    def wait_compute_event(self, layer_idx: int):
        self.prefetch_stream.wait_event(
            self.compute_events[layer_idx % self.num_buffers][layer_idx // self.num_buffers]
        )


class DwdpManager:
    """
    Dwdp Manager for IPC handle coordination and prefetch buffer management.

    This manager:
    - Tracks IPC handles for all MoE layers across Context workers
    - Manages double-buffered prefetch buffers for remote expert weights
    - Provides expert tensor routing (local vs. prefetched)

    """

    def __init__(
        self,
        config: DwdpConfig,
        dist: Optional[object] = None,
    ):
        self.config = config
        self.dist = dist
        self.dwdp_size = config.dwdp_size
        self.num_experts_per_worker = config.num_experts_per_worker
        self.num_groups = config.num_groups

        self._init_dwdp_group()

        # Per-layer IPC handle collectors (indexed by layer_idx)
        self.ipc_collectors: List[DwdpLayerHandleCollector] = []

        # Prefetch buffer (initialized later in create_py_executor)
        self.prefetch_buffer: Optional[DwdpPrefetchBuffer] = None
        # Auto-detected from first add_layer() call
        self.first_moe_layer_idx: Optional[int] = None

        # Peer expert ranges: (peer_rank, (start_expert_id, end_expert_id))
        self.peer_expert_ranges: Dict[int, Tuple[int, int]] = {}

        self.dwdp_rank = self.rank % self.dwdp_size
        self.num_prefetch_experts = config.num_prefetch_experts
        self.start_expert_id = self.num_prefetch_experts * self.dwdp_rank
        self.end_expert_id = self.start_expert_id + self.num_experts_per_worker

    def __enter__(self):
        set_global_dwdp_manager(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        set_global_dwdp_manager(None)
        return False

    def _init_dwdp_group(self):
        if not isinstance(self.dist, MPIDist):
            raise RuntimeError("DWDP requires MPI backend (MPIDist)")

        self.rank = global_mpi_rank()

        # Calculate which group this rank belongs to
        # With num_groups=2, dwdp_size=4:
        #   Group 0: ranks [0, 1, 2, 3]
        #   Group 1: ranks [4, 5, 6, 7]
        self.group_id = self.rank // self.dwdp_size
        group_start_rank = self.group_id * self.dwdp_size
        ranks = list(range(group_start_rank, group_start_rank + self.dwdp_size))

        new_group = COMM_WORLD.group.Incl(ranks)
        self.dwdp_group = COMM_WORLD.Create_group(new_group)

    def is_enabled(self) -> bool:
        return self.dwdp_size > 1

    def cleanup(self):
        """Release all IPC handles and clean up resources."""
        for collector in self.ipc_collectors:
            collector.cleanup()
        self.ipc_collectors.clear()
        if self.dwdp_group is not None:
            self.dwdp_group.Free()
            self.dwdp_group = None

    def add_layer(
        self,
        layer_idx: int,
    ) -> "DwdpLayerHandleCollector":
        """
        Add a new layer IPC handle collector.

        Called from CuteDslFusedMoE.__init__() during model construction.
        """
        if self.first_moe_layer_idx is None:
            self.first_moe_layer_idx = layer_idx
        collector = DwdpLayerHandleCollector(layer_idx=layer_idx)
        self.ipc_collectors.append(collector)
        return collector

    def exchange_all_handles(self):
        """
        Exchange IPC handles with peer Context workers via Dwdp Group AllGather.

        Called after all weights are loaded, before creating prefetch buffer.
        """

        # Collect all local handles with explicit worker info
        local_data = {
            "dwdp_rank": self.dwdp_rank,
            "expert_start_id": self.start_expert_id,
            "expert_end_id": self.end_expert_id,
            "ipc_collectors": [],
        }
        for collector in self.ipc_collectors:
            local_data["ipc_collectors"].append(
                {
                    "layer_idx": collector.layer_idx,
                    "handles": collector.local_ipc_handles,
                    "offsets": collector.local_offsets,
                }
            )

        # AllGather from all Context workers in DWDP group
        all_data = self.dwdp_group.allgather(local_data)

        # Open handles from peer workers
        for peer_data in all_data:
            peer_rank = peer_data["dwdp_rank"]
            self.peer_expert_ranges[peer_rank] = (
                peer_data["expert_start_id"],
                peer_data["expert_end_id"],
            )

            if peer_rank == self.dwdp_rank:
                continue
            for layer_idx, ipc_collector in enumerate(peer_data["ipc_collectors"]):
                collector = self.ipc_collectors[layer_idx]
                peer_offsets = ipc_collector["offsets"]
                for param_name, handle_bytes in ipc_collector["handles"].items():
                    # Reconstruct and open handle
                    handle = cudart.cudaIpcMemHandle_t()
                    handle.reserved = list(handle_bytes)

                    err, base_ptr = cudart.cudaIpcOpenMemHandle(
                        handle, cudart.cudaIpcMemLazyEnablePeerAccess
                    )
                    check_cuda_error(err, f"open handle rank={peer_rank}")

                    # Apply offset to get actual tensor pointer
                    # IPC handle points to allocation base, offset gives us the tensor location
                    offset = peer_offsets[param_name]
                    actual_ptr = base_ptr + offset
                    collector.peer_ptrs[(peer_rank, param_name)] = actual_ptr

    def initialize_prefetch_buffer(self):
        """
        Initialize the prefetch buffer.

        Called in create_py_executor() after model loading.
        """
        self.prefetch_buffer = DwdpPrefetchBuffer(
            dwdp_size=self.dwdp_size,
            dwdp_rank=self.dwdp_rank,
            num_experts_per_worker=self.num_experts_per_worker,
            num_prefetch_experts=self.num_prefetch_experts,
            num_layers=len(self.ipc_collectors),
            first_moe_layer_idx=self.first_moe_layer_idx,
            param_shapes=self.ipc_collectors[0].param_shapes,
            param_dtypes=self.ipc_collectors[0].param_dtypes,
        )
        self.prefetch_buffer.initialize_compute_events()

    def prefetch_first_layers(self):
        """Prefetch the first num_buffers layers as warmup."""
        if self.prefetch_buffer is None:
            raise RuntimeError("Prefetch buffer is not initialized")
        start = self.first_moe_layer_idx
        for layer_idx in range(start, start + self.prefetch_buffer.num_buffers):
            self.prefetch_layer(layer_idx)
            self.prefetch_buffer.record_prefetch_event(layer_idx)

    def build_weight_view(self, layer_idx: int, backend):
        """Build NvFp4WeightView from prefetch buffer and local weights.

        Assembles weight tensors from all DWDP ranks:
        - Peer ranks: uses prefetched buffer tensors
        - Local rank: uses backend's actual model weights

        Args:
            layer_idx: The MoE layer index.
            backend: The CuteDslFusedMoE backend holding local model weights.

        Returns:
            NvFp4WeightView with all weights assembled.
        """
        from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import NvFp4WeightView

        buffer_data = self.wait_prefetch_and_get_buffer(layer_idx)
        required_keys = (
            "w3_w1_weight",
            "w3_w1_weight_scale",
            "fc31_alpha",
            "w2_weight",
            "w2_weight_scale",
            "fc2_alpha",
        )
        missing_keys = [key for key in required_keys if key not in buffer_data]
        if missing_keys:
            raise ValueError(
                f"DWDP buffer missing required keys {missing_keys} for layer {layer_idx}."
            )

        w3_w1_weight_list = buffer_data["w3_w1_weight"]
        fc1_weight_scale_list = buffer_data["w3_w1_weight_scale"]
        fc1_global_scale_list = buffer_data["fc31_alpha"]
        w2_weight_list = buffer_data["w2_weight"]
        fc2_weight_scale_list = buffer_data["w2_weight_scale"]
        fc2_global_scale_list = buffer_data["fc2_alpha"]

        w3_w1_weight_list[self.dwdp_rank] = backend.w3_w1_weight
        fc1_weight_scale_list[self.dwdp_rank] = backend.quant_scales.fc1_weight_block
        fc1_global_scale_list[self.dwdp_rank] = backend.quant_scales.fc1_global
        w2_weight_list[self.dwdp_rank] = backend.w2_weight
        fc2_weight_scale_list[self.dwdp_rank] = backend.quant_scales.fc2_weight_block
        fc2_global_scale_list[self.dwdp_rank] = backend.quant_scales.fc2_global

        return NvFp4WeightView(
            w3_w1_weight=w3_w1_weight_list,
            fc1_weight_scale=fc1_weight_scale_list,
            fc1_global_scale=fc1_global_scale_list,
            w2_weight=w2_weight_list,
            fc2_weight_scale=fc2_weight_scale_list,
            fc2_global_scale=fc2_global_scale_list,
            expert_size_per_partition=backend.num_slots,
            slot_start=0,
        )

    def wait_prefetch_and_get_buffer(
        self, layer_idx: int
    ) -> Optional[Dict[str, List[Optional[torch.Tensor]]]]:
        """Wait for prefetch to complete and return the buffer for this layer.

        Returns:
            Dict mapping param_name to List[Optional[Tensor]] where:
            - list[peer_rank] = Tensor for prefetched weights from that peer
            - list[dwdp_rank] = None (local weights used directly)
        """
        if self.prefetch_buffer is None:
            raise RuntimeError("Prefetch buffer is not initialized")
        self.prefetch_buffer.wait_prefetch_event(layer_idx)
        buffer_idx = layer_idx % self.prefetch_buffer.num_buffers
        return self.prefetch_buffer.buffers[buffer_idx]

    def record_compute_and_prefetch_next(self, layer_idx: int):
        """Record compute completion and trigger prefetch for layer_idx + num_buffers."""
        if self.prefetch_buffer is None:
            raise RuntimeError("Prefetch buffer is not initialized")
        # Record compute event for current layer
        self.prefetch_buffer.record_compute_event(layer_idx)

        next_layer_idx = layer_idx + self.prefetch_buffer.num_buffers
        if next_layer_idx >= self.prefetch_buffer.max_layer_idx:
            return
        # prefetch_layer handles stream internally: local copy on default stream, peer copy on prefetch stream
        self.prefetch_layer(next_layer_idx, wait_compute_layer_idx=layer_idx)
        self.prefetch_buffer.record_prefetch_event(next_layer_idx)

    def _get_prefetch_src_offset_from_peer(self, peer_rank: int) -> int:
        """
        Calculate the source offset (in number of experts) to fetch from a peer.

        Returns:
            src_offset: Offset into peer's local expert tensor to start copying from

        Example: 256 experts, rank0: [0, 200), rank1: [56, 256)
        - rank0 needs [200, 256) from rank1:
          src_offset = 200 - 56 = 144 (fetch last 56 experts from rank1)
        - rank1 needs [0, 56) from rank0:
          src_offset = 0 - 0 = 0 (fetch first 56 experts from rank0)
        """
        peer_start, peer_end = self.peer_expert_ranges[peer_rank]

        # What I need = global - what I have
        # From peer = what I need ∩ what peer has
        if self.dwdp_rank < peer_rank:
            # I'm earlier rank, need experts after my end (tail of peer's experts)
            prefetch_end = peer_end
            prefetch_start = prefetch_end - self.num_prefetch_experts
        else:
            # I'm later rank, need experts before my start (head of peer's experts)
            prefetch_start = peer_start

        src_offset = prefetch_start - peer_start
        return src_offset

    @nvtx_range("dwdp_prefetch_layer")
    def prefetch_layer(self, layer_idx: int, wait_compute_layer_idx: Optional[int] = None):
        """
        Prefetch layer data from peer ranks.

        Args:
            layer_idx: The layer to prefetch
            wait_compute_layer_idx: If provided, wait for this layer's compute to complete
                                    before overwriting buffer (used when prefetching next layer)

        Note: Local weights are used directly by the kernel, no copy needed.
        Peer copy runs on prefetch stream.
        """
        moe_idx = layer_idx - self.first_moe_layer_idx
        param_names = self.ipc_collectors[moe_idx].param_shapes.keys()
        collector = self.ipc_collectors[moe_idx]
        buffer_idx = layer_idx % self.prefetch_buffer.num_buffers

        # Peer copy on prefetch stream
        # Local weights are used directly - no local copy needed
        with torch.cuda.stream(self.prefetch_buffer.prefetch_stream):
            # Wait for compute to complete before overwriting buffer
            if wait_compute_layer_idx is not None:
                self.prefetch_buffer.wait_compute_event(wait_compute_layer_idx)

            for peer_rank in range(self.dwdp_size):
                if peer_rank == self.dwdp_rank:
                    continue  # Skip local rank - local weights used directly

                src_expert_offset = self._get_prefetch_src_offset_from_peer(peer_rank)

                for param_name in param_names:
                    param_shape = collector.param_shapes[param_name]
                    param_dtype = collector.param_dtypes[param_name]
                    expert_size = param_shape.numel() * param_dtype.itemsize

                    # src_ptr points to peer's tensor start, add offset for specific experts
                    base_ptr = collector.get_peer_ptr(peer_rank, param_name)
                    src_ptr = base_ptr + src_expert_offset * expert_size

                    # dst_tensor is directly indexed by peer_rank in the list
                    dst_tensor = self.prefetch_buffer.buffers[buffer_idx][param_name][peer_rank]
                    dst_ptr = dst_tensor.data_ptr()

                    data_size = self.num_prefetch_experts * expert_size

                    (err,) = cudart.cudaMemcpyAsync(
                        dst_ptr,
                        src_ptr,
                        data_size,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                        self.prefetch_buffer.prefetch_stream.cuda_stream,
                    )
                    check_cuda_error(
                        err, f"prefetch layer {layer_idx} peer_rank {peer_rank} {param_name}"
                    )
