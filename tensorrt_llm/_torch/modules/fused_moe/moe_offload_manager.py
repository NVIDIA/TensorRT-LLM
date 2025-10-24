from typing import List, Tuple

import torch


class MoeOffloadManager:
    """
    MoE offload manager for MoE weight offloading.
    It manages the offloaded layers and buffers.
    """

    def __init__(
        self, moe_layer_indices: List[int], moe_offload_capacity: int, moe_offload_stride: int
    ):
        # tp and ep support only
        self.moe_layer_indices = moe_layer_indices
        self.offload_capacity = moe_offload_capacity  # of buffers
        self.offload_stride = moe_offload_stride  # offload every [stride] moe layers
        self.weights = {}
        # set on first register_weight call
        self.weight_dtype = None
        self.weight_shapes = None
        self.device_weight_buffers = []

        # offloaded layer indices (stride-spaced)
        self.offload_layer_indices = [
            self.moe_layer_indices[i]
            for i in range(self.offload_stride - 1, len(self.moe_layer_indices), moe_offload_stride)
        ]

        # prefill needed layers outside decoder layers
        self.initial_offload_layers = self.offload_layer_indices[:moe_offload_capacity]

        # cuda streams and events for offloading (per buffer)
        self.offload_streams = [torch.cuda.Stream() for _ in range(self.offload_capacity)]

        # map of layer index to buffer index
        self.layer_to_buffer = {
            layer_idx: i % self.offload_capacity
            for i, layer_idx in enumerate(self.offload_layer_indices)
        }

        # map of current offloaded layer to next-offloaded layer sharing same buffer
        self.next_offload_layer = {
            layer_idx: (
                self.offload_layer_indices[i + self.offload_capacity]
                if i + self.offload_capacity < len(self.offload_layer_indices)
                else None
            )
            for i, layer_idx in enumerate(self.offload_layer_indices)
        }

    def register_weight(self, layer_id: int, weights: List[torch.Tensor]):
        weight_names = [
            "w3_w1_weight",
            "w2_weight",
        ]  # should match with the order of weights tensor list
        assert len(weights) == len(weight_names), (
            "Expected {len(weight_names)} weight tensors per moe layer"
        )

        if self.weight_dtype is None or self.weight_shapes is None:
            self.weight_dtype = weights[0].dtype
            self.weight_shapes = {name: weights[i].shape for i, name in enumerate(weight_names)}

            # allocate device (gpu) memory for buffers storing offloaded weights
            self._allocate_device_weight_buffers(*self.weight_shapes.values())
        else:
            assert self.weight_dtype == weights[0].dtype, (
                f"MoE Dtype mismatch on layer {layer_id}: {self.weight_dtype} != {weights[0].dtype}"
            )
            for i, name in enumerate(weight_names):
                assert weights[i].shape == self.weight_shapes[name], (
                    f"MoE {name} Weight shapes mismatch on layer {layer_id}: "
                    f"{weights[i].shape} != {self.weight_shapes[name]}"
                )

        self.weights[layer_id] = weights

    def _allocate_device_weight_buffers(
        self, w3_w1_shape: Tuple[int, int, int], w2_shape: Tuple[int, int, int]
    ):
        assert self.weight_dtype is not None, (
            "MoE Offloaded weights dtype must be set before allocating device weight buffers"
        )
        for _ in range(self.offload_capacity):
            self.device_weight_buffers.append(
                {
                    "w3_w1_weight": torch.empty(
                        w3_w1_shape, dtype=self.weight_dtype, device=torch.device("cuda")
                    ),
                    "w2_weight": torch.empty(
                        w2_shape, dtype=self.weight_dtype, device=torch.device("cuda")
                    ),
                }
            )

    def offload_initial_weights(self, cur_stream: torch.cuda.Stream):
        prev_stream = cur_stream
        for i, layer_idx in enumerate(self.initial_offload_layers):
            assert layer_idx in self.weights, (
                f"Layer {layer_idx} not found in registered offloaded weights"
            )

            cur_stream = self.offload_streams[i]
            cur_stream.wait_stream(prev_stream)

            with torch.cuda.stream(cur_stream):
                self.device_weight_buffers[i]["w3_w1_weight"].copy_(
                    self.weights[layer_idx][0].view(self.weight_dtype), non_blocking=True
                )
                self.device_weight_buffers[i]["w2_weight"].copy_(
                    self.weights[layer_idx][1].view(self.weight_dtype), non_blocking=True
                )

            prev_stream = cur_stream

    def get_offload_stream(self, layer_idx: int):
        assert layer_idx in self.layer_to_buffer, (
            f"Layer {layer_idx} not found in registered offloaded layers"
        )
        buffer_idx = self.layer_to_buffer[layer_idx]
        return self.offload_streams[buffer_idx]

    def get_dst_device_buffers(self, layer_idx: int):
        assert layer_idx in self.layer_to_buffer, (
            f"Layer {layer_idx} not found in registered offloaded layers"
        )
        buffer_idx = self.layer_to_buffer[layer_idx]
        return self.device_weight_buffers[buffer_idx]["w3_w1_weight"], self.device_weight_buffers[
            buffer_idx
        ]["w2_weight"]

    def get_next_offload_buffers(self, layer_idx: int):
        assert layer_idx in self.offload_layer_indices, (
            f"Layer {layer_idx} not found in registered offloaded layers"
        )
        offload_layer_idx = self.next_offload_layer[layer_idx]
        if offload_layer_idx is not None:
            assert offload_layer_idx in self.layer_to_buffer, (
                f"Layer {offload_layer_idx} to be offloaded not found in registered offloaded layers"
            )
            buffer_idx = self.layer_to_buffer[offload_layer_idx]
            return self.device_weight_buffers[buffer_idx], self.weights[offload_layer_idx]
        else:
            # last [offload_capacity] layers
            return None, None


class MoeOffloadProxy:
    """
    MoE offload proxy for MoE weight offloading.
    It manages the offload logic of a single layer.
    """

    def __init__(self, layer_id: int, offload_manager: MoeOffloadManager):
        self.layer_id = layer_id
        self.offload_manager = offload_manager

        # will set when creating weights for the moe layer
        self.w3_w1_weight_shape = None
        self.w2_weight_shape = None
        self.weight_dtype = None
        self.w3_w1_dst_buffer = None
        self.w2_dst_buffer = None

        # get offload stream for this layer from offload manager
        self.offload_stream = self.offload_manager.get_offload_stream(self.layer_id)

    def register_weight(self, w3_w1_weight: torch.Tensor, w2_weight: torch.Tensor):
        self.w3_w1_weight_shape = w3_w1_weight.shape
        self.w2_weight_shape = w2_weight.shape
        self.weight_dtype = w3_w1_weight.dtype
        self.offload_manager.register_weight(self.layer_id, [w3_w1_weight, w2_weight])
        self.w3_w1_dst_buffer, self.w2_dst_buffer = self.offload_manager.get_dst_device_buffers(
            self.layer_id
        )

    def start_next_layer_offloading(self, cur_stream: torch.cuda.Stream):
        offload_dst, offload_src = self.offload_manager.get_next_offload_buffers(self.layer_id)
        if offload_dst is not None and offload_src is not None:
            self.offload_stream.wait_stream(cur_stream)
            with torch.cuda.stream(self.offload_stream):
                offload_dst["w3_w1_weight"].copy_(
                    offload_src[0].view(self.weight_dtype), non_blocking=True
                )
                offload_dst["w2_weight"].copy_(
                    offload_src[1].view(self.weight_dtype), non_blocking=True
                )
