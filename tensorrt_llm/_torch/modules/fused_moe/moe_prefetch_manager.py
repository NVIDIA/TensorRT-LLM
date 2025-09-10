from typing import List, Tuple

import torch


class MoEPrefetchManager:
    """
    MoE prefetch manager for MoE weight prefetching.
    It manages the prefetched layers and buffers.
    """

    def __init__(self, moe_layer_indices: List[int], moe_prefetch_capacity: int,
                 moe_prefetch_stride: int):
        # tp and ep support only
        self.moe_layer_indices = moe_layer_indices
        self.prefetch_capacity = moe_prefetch_capacity  # of buffers
        self.prefetch_stride = moe_prefetch_stride  # prefetch every [stride] moe layers
        self.weights = {}
        # set on first register_weight call
        self.weight_dtype = None
        self.weight_shapes = None
        self.device_weight_buffers = []

        # prefetch layer indices (stride-spaced)
        self.prefetch_layer_indices = [
            self.moe_layer_indices[i]
            for i in range(self.prefetch_stride -
                           1, len(self.moe_layer_indices), moe_prefetch_stride)
        ]

        # prefill needed layers outside decoder layers
        self.initial_prefetch_layers = self.prefetch_layer_indices[:
                                                                   moe_prefetch_capacity]

        # cuda streams and events for prefetching (per buffer)
        self.prefetch_streams = [
            torch.cuda.Stream() for _ in range(self.prefetch_capacity)
        ]

        # map of layer index to buffer index
        self.layer_to_buffer = {
            layer_idx: i % self.prefetch_capacity
            for i, layer_idx in enumerate(self.prefetch_layer_indices)
        }

        # map of current prefetched layer to next-prefetched layer sharing same buffer
        self.next_prefetch_layer = {
            layer_idx:
            (self.prefetch_layer_indices[i + self.prefetch_capacity] if i +
             self.prefetch_capacity < len(
                 self.prefetch_layer_indices) else None)
            for i, layer_idx in enumerate(self.prefetch_layer_indices)
        }

    def register_weight(self, layer_id: int, weights: List[torch.Tensor]):
        weight_names = ["w3_w1_weight", "w2_weight"
                        ]  # should match with the order of weights tensor list
        assert len(weights) == len(
            weight_names
        ), "Expected {len(weight_names)} weight tensors per moe layer"

        if self.weight_dtype is None or self.weight_shapes is None:
            self.weight_dtype = weights[0].dtype
            self.weight_shapes = {
                name: weights[i].shape
                for i, name in enumerate(weight_names)
            }

            # allocate device (gpu) memory for buffers storing prefetched weights
            self._allocate_device_weight_buffers(*self.weight_shapes.values())
        else:
            assert self.weight_dtype == weights[
                0].dtype, f"MoE Dtype mismatch on layer {layer_id}: {self.weight_dtype} != {weights[0].dtype}"
            for i, name in enumerate(weight_names):
                assert weights[i].shape == self.weight_shapes[
                    name], f"MoE {name} Weight shapes mismatch on layer {layer_id}: {self.weight_shapes[name]} != {weights[i].shape}"

        self.weights[layer_id] = weights

    def _allocate_device_weight_buffers(self, w3_w1_shape: Tuple[int, int, int],
                                        w2_shape: Tuple[int, int, int]):
        assert self.weight_dtype is not None, "MoE Prefetched Weight dtype must be set before allocating device weight buffers"
        for _ in range(self.prefetch_capacity):
            self.device_weight_buffers.append({
                "w3_w1_weight":
                torch.empty(w3_w1_shape,
                            dtype=self.weight_dtype,
                            device=torch.device("cuda")),
                "w2_weight":
                torch.empty(w2_shape,
                            dtype=self.weight_dtype,
                            device=torch.device("cuda"))
            })

    def prefetch_initial_weights(self, cur_stream: torch.cuda.Stream):
        prev_stream = cur_stream
        for i, layer_idx in enumerate(self.initial_prefetch_layers):
            assert layer_idx in self.weights, f"Layer {layer_idx} not found in registered prefteched weights"

            cur_stream = self.prefetch_streams[i]
            cur_stream.wait_stream(prev_stream)

            with torch.cuda.stream(cur_stream):
                self.device_weight_buffers[i]["w3_w1_weight"].copy_(
                    self.weights[layer_idx][0].view(self.weight_dtype),
                    non_blocking=True)
                self.device_weight_buffers[i]["w2_weight"].copy_(
                    self.weights[layer_idx][1].view(self.weight_dtype),
                    non_blocking=True)

            prev_stream = cur_stream

    def get_prefetched_stream(self, layer_idx: int):
        assert layer_idx in self.layer_to_buffer, f"Layer {layer_idx} not found in registered prefetched layers"
        buffer_idx = self.layer_to_buffer[layer_idx]
        return self.prefetch_streams[buffer_idx]

    def get_dst_device_buffers(self, layer_idx: int):
        assert layer_idx in self.layer_to_buffer, f"Layer {layer_idx} not found in registered prefetched layers"
        buffer_idx = self.layer_to_buffer[layer_idx]
        return self.device_weight_buffers[buffer_idx][
            'w3_w1_weight'], self.device_weight_buffers[buffer_idx]['w2_weight']

    def get_next_prefetch_buffers(self, layer_idx: int):
        assert layer_idx in self.prefetch_layer_indices, f"Layer {layer_idx} not found in registered prefetched layers"
        prefetch_layer_idx = self.next_prefetch_layer[layer_idx]
        if prefetch_layer_idx is not None:
            assert prefetch_layer_idx in self.layer_to_buffer, f"Layer {prefetch_layer_idx} to be prefetched not found in registered prefetched layers"
            buffer_idx = self.layer_to_buffer[prefetch_layer_idx]
            return self.device_weight_buffers[buffer_idx], self.weights[
                prefetch_layer_idx]
        else:
            # last [prefetch_capacity] layers
            return None, None


class MoEPrefetchProxy:
    """
    MoE prefetch proxy for MoE weight prefetching.
    It manages the prefetch logic of a single layer.
    """

    def __init__(self, layer_id: int, prefetch_manager: MoEPrefetchManager):
        self.layer_id = layer_id
        self.prefetch_manager = prefetch_manager

        # will set when creating weights for the moe layer
        self.w3_w1_weight_shape = None
        self.w2_weight_shape = None
        self.weight_dtype = None
        self.w3_w1_dst_buffer = None
        self.w2_dst_buffer = None

        # get prefetch stream for this layer from prefetch manager
        self.prefetch_stream = self.prefetch_manager.get_prefetched_stream(
            self.layer_id)

    def register_weight(self, w3_w1_weight: torch.Tensor,
                        w2_weight: torch.Tensor):
        self.w3_w1_weight_shape = w3_w1_weight.shape
        self.w2_weight_shape = w2_weight.shape
        self.weight_dtype = w3_w1_weight.dtype
        self.prefetch_manager.register_weight(self.layer_id,
                                              [w3_w1_weight, w2_weight])
        self.w3_w1_dst_buffer, self.w2_dst_buffer = self.prefetch_manager.get_dst_device_buffers(
            self.layer_id)

    def start_next_layer_prefetching(self, cur_stream: torch.cuda.Stream):
        prefetch_dst, prefetch_src = self.prefetch_manager.get_next_prefetch_buffers(
            self.layer_id)
        if prefetch_dst is not None and prefetch_src is not None:
            self.prefetch_stream.wait_stream(cur_stream)
            with torch.cuda.stream(self.prefetch_stream):
                prefetch_dst["w3_w1_weight"].copy_(prefetch_src[0].view(
                    self.weight_dtype),
                                                   non_blocking=True)
                prefetch_dst["w2_weight"].copy_(prefetch_src[1].view(
                    self.weight_dtype),
                                                non_blocking=True)
