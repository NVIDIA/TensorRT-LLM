# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-load U1 static-FP8 gate-up/SG fusion with explicit BF16 intermediate rounding."""

import time
import types

import breuse_gateup_kernel as kernel
import cuda.bindings.driver as cuda
import torch

ACTIVE = None


@torch.library.custom_op("h3_fp8_long3::gateup_swiglu", mutates_args=())
def gateup_swiglu(
    activation: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    sg_scale: torch.Tensor,
) -> torch.Tensor:
    runtime = ACTIVE
    if runtime is None:
        raise RuntimeError("Install the post-load gate-up runtime first")
    runtime.calls += 1
    output = torch.empty(
        (activation.shape[0], 14336), device=activation.device, dtype=torch.float8_e4m3fn
    )
    kernel.run(
        activation,
        weight,
        input_scale.reshape(1),
        weight_scale.reshape(1),
        sg_scale.reshape(1),
        runtime.packed_weights[weight.data_ptr()],
        output,
    )
    return output


@gateup_swiglu.register_fake
def _(activation, weight, input_scale, weight_scale, sg_scale):
    return torch.empty(
        (activation.shape[0], 14336), device=activation.device, dtype=torch.float8_e4m3fn
    )


def forward(self, x, all_rank_num_tokens=None, final_all_reduce_params=None, lora_params=None):
    if lora_params is not None:
        raise ValueError("LoRA is outside this qualified experiment")
    if x.ndim != 2 or x.shape[1] != 5376 or x.dtype != torch.float8_e4m3fn:
        raise ValueError("Expected static-FP8 H3 activation [M,5376]")
    up, down = self.gate_up_proj, self.down_proj
    codes = gateup_swiglu(x, up.weight, up.input_scale, up.weight_scale, down.input_scale)
    return down(codes, all_reduce_params=final_all_reduce_params, layer_idx=self.layer_idx)


class GateupRuntime:
    """One active research pipeline per process; retain original weights for control runs."""

    def __init__(self, blocks):
        global ACTIVE
        device = blocks[0].ff.gate_up_proj.weight.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 7):
            raise ValueError("The B-reuse kernel is qualified on actual SM107 only")
        status, capacity = cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute(150), cuda.CUdevice(device.index)
        )
        # Public CUDA13.4 oversized capacity, separately verified with real launches.
        # Ordinary shared_memory_per_block_optin does not describe this mode.
        config = kernel._CFG
        cluster_n = config["cluster_shape_mn"][1]
        n_tiles = 28672 // 256
        if n_tiles % cluster_n or (n_tiles // cluster_n) % config["swizzle_size"]:
            raise ValueError("The fixed swizzle requires a complete band of N clusters")
        required = 3 * 128 * config["mma_tile_k"] * config["num_ab_stage"] + 4096
        if int(status) != 0 or int(capacity) < required:
            raise RuntimeError("The verified oversized shared-memory configuration is unavailable")
        self.blocks = blocks
        self.original = [block.ff.forward for block in blocks]
        self.calls = 0
        self.packed_weights = {}
        for block in blocks:
            if block.ff.gate_up_proj.bias is not None or block.ff.down_proj.bias is not None:
                raise ValueError("Only the original bias-free H3 MLP is qualified")
            if block.ff.mapping.tp_size != 1:
                raise ValueError("Only U1/TP1 is qualified")
        start = time.perf_counter()
        for block in blocks:
            weight = block.ff.gate_up_proj.weight
            if weight.shape != (28672, 5376) or not weight.is_contiguous():
                raise ValueError("Expected contiguous original H3 gate/up weights")
            self.packed_weights[weight.data_ptr()] = kernel.pack_weight(weight)
        torch.cuda.synchronize(device)
        self.prepack_time_s = time.perf_counter() - start
        self.extra_weight_bytes = sum(
            value.numel() * value.element_size() for value in self.packed_weights.values()
        )
        kernel._get_compiled(device.index)
        ACTIVE = self

    def configure(self, enabled):
        for block, original in zip(self.blocks, self.original):
            block.ff.forward = types.MethodType(forward, block.ff) if enabled else original
