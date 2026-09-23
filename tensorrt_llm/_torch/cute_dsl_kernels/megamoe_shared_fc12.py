# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-expert use of MegaMoE's complete fused FC1/activation/FC2 kernel."""

from functools import lru_cache

import torch

__all__ = ["run_shared_fc12", "shared_fc12_cache_info"]


class _SharedFc12Runner:
    def __init__(
        self,
        device_index: int,
        hidden_size: int,
        intermediate_size: int,
        max_tokens: int,
        sm_count: int,
        swiglu_limit: float | None,
        stream_handle: int,
    ) -> None:
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils
        from cutlass.cute.nvgpu import OperandMajorMode
        from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_ptr
        from cutlass.cute.typing import AddressSpace

        from .cutedsl_megamoe.api import ImplDesc, ProblemDesc
        from .cutedsl_megamoe.kernel_src.blackwell.inference.mega.block_scaled_swap_ab_fc12_kernel import (
            BlockScaledSwapAbFc12Kernel,
        )
        from .cutedsl_megamoe.quant_def import CombineFormat

        device = torch.device("cuda", device_index)
        properties = torch.cuda.get_device_properties(device)
        if not 0 < sm_count <= properties.multi_processor_count:
            raise ValueError("shared FC12 sm_count must be within the device SM count")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Create the shared FC12 runner before CUDA graph capture")
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.reserved_sms = properties.multi_processor_count - sm_count
        self.stream_handle = stream_handle
        cluster_size = 2
        hardware_clusters = utils.HardwareInfo(device_index).get_max_active_clusters(cluster_size)
        launch_clusters = min(sm_count // cluster_size, hardware_clusters)
        if launch_clusters <= 0:
            raise ValueError("shared FC12 requires at least two available SMs")

        problem = ProblemDesc(
            {
                "expert_count": 1,
                "intermediate_gateup_size": 2 * intermediate_size,
                "hidden_size": hidden_size,
                "quant_kind": "mxfp8_e4m3",
                "a_major_mode": OperandMajorMode.K,
                "b_major_mode": OperandMajorMode.K,
                "combine_format": CombineFormat.parse("bf16"),
                "gate_up_clamp": swiglu_limit,
            }
        )
        implementation = ImplDesc(
            {
                "mma_tiler_mnk": (256, 128, 128),
                "cluster_shape_mn": (2, 1),
                "use_2cta_instrs": True,
                "group_hint": launch_clusters,
                "token_padding_block": 64,
                "sf_padding_block": 128,
                "work_id_mode": "grid_stride",
                "max_tokens": max_tokens,
                "launch_cluster_count": launch_clusters,
                "fc2_use_bulk": True,
                "fc1_epi_flag_batch": 1,
                "fc2_epi_flag_batch": 1,
            }
        )
        kernel = BlockScaledSwapAbFc12Kernel(problem, implementation)
        local_bytes, shared_bytes = kernel.get_workspace_sizes()
        if shared_bytes:
            raise RuntimeError("shared FC12 must not allocate a communication workspace")
        self.workspace = torch.empty(local_bytes, dtype=torch.uint8, device=device)
        local_zero_bytes, shared_zero_bytes = kernel.require_zero_workspace_leading_bytes
        if shared_zero_bytes:
            raise RuntimeError("shared FC12 must not initialize communication state")
        self.workspace_pointer = self.workspace.data_ptr()
        self.workspace_reset_bytes = local_zero_bytes
        self._memset_async = cuda.cuMemsetD8Async
        self._cuda_success = cuda.CUresult.CUDA_SUCCESS
        self._cuda_stream = cuda.CUstream(stream_handle)
        # A slice supplies the current E=1 token count without a per-call GPU write.
        self.token_counts = torch.arange(max_tokens + 1, dtype=torch.int32, device=device)

        token_rows = cute.sym_int64(divisibility=1)
        scale_rows = cute.sym_int64(divisibility=128)
        padded_gateup = (2 * intermediate_size + 127) // 128 * 128
        fc1_sf_columns = padded_gateup * (hidden_size // 32)
        fc2_sf_columns = hidden_size * (intermediate_size // 32)
        fake_arguments = {
            "activation": make_fake_compact_tensor(
                cutlass.Float8E4M3FN,
                (token_rows, hidden_size),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            "fc1_weight": make_fake_compact_tensor(
                cutlass.Float8E4M3FN,
                (1, hidden_size, 2 * intermediate_size),
                stride_order=(2, 0, 1),
                assumed_align=16,
            ),
            "activation_sf": make_fake_compact_tensor(
                cutlass.Float8E8M0FNU,
                (scale_rows, hidden_size // 32),
                stride_order=(1, 0),
                assumed_align=16,
            ),
            "fc1_weight_sf": make_fake_compact_tensor(
                cutlass.Float8E8M0FNU, (1, fc1_sf_columns), stride_order=(1, 0), assumed_align=16
            ),
            "fc2_weight": make_fake_compact_tensor(
                cutlass.Float8E4M3FN,
                (1, intermediate_size, hidden_size),
                stride_order=(2, 0, 1),
                assumed_align=16,
            ),
            "fc2_weight_sf": make_fake_compact_tensor(
                cutlass.Float8E8M0FNU, (1, fc2_sf_columns), stride_order=(1, 0), assumed_align=16
            ),
            "fc2_output": make_fake_compact_tensor(
                cutlass.BFloat16,
                (token_rows, 1, hidden_size),
                stride_order=(2, 1, 0),
                assumed_align=16,
            ),
            "local_workspace": make_ptr(cutlass.Uint8, 0, AddressSpace.gmem, assumed_align=128),
            "max_active_clusters": hardware_clusters,
            "stream": make_fake_stream(),
            "expert_token_sizes": make_fake_compact_tensor(
                cutlass.Int32, (1,), stride_order=(0,), assumed_align=4
            ),
            "expert_token_prefix_sum": None,
            "topk_scores": None,
            "fc1_alpha": None,
            "fc2_alpha": None,
            "fc1_norm_const": None,
        }
        self.compiled = cute.compile[cute.EnableTVMFFI(True)](kernel, **fake_arguments)
        self.audit = {
            "kernel_class": type(kernel).__name__,
            "upstream_commit": "522fbb019942ad6bbe0f418dba02541f837ea46c",
            "kernel_descriptor_architecture": kernel.architecture,
            "device_compute_capability": [properties.major, properties.minor],
            "input_quantizer": "fp8_quantize_1x128_packed_ue8m0.sm_budget",
            "input_quantization_block": 128,
            "fc1_output_quantization_block": 32,
            "requested_sm_count": sm_count,
            "input_quantizer_reserved_sms": self.reserved_sms,
            "launch_cluster_count": launch_clusters,
            "launch_grid": [2, 1, launch_clusters],
            "max_tokens": max_tokens,
            "dynamic_token_dimension": True,
            "routing_or_communication": False,
            "reset_strategy": "cuMemsetD8Async_before_fc12_same_stream",
            "workspace_reset_bytes": self.workspace_reset_bytes,
        }

    def __call__(
        self,
        activation: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_sf: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_sf: torch.Tensor,
    ) -> torch.Tensor:
        token_rows = activation.shape[0]
        with torch.cuda.nvtx.range("MEGAMOE_SHARED_INPUT_QUANT"):
            quantized, scales = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0.sm_budget(
                activation, reserved_sms=self.reserved_sms
            )
        output = torch.empty(
            (token_rows, 1, self.hidden_size), dtype=torch.bfloat16, device=activation.device
        )
        with torch.cuda.nvtx.range("MEGAMOE_SHARED_FC12"):
            # The standalone FC12 readiness counters require reset before every invocation.
            (status,) = self._memset_async(
                self.workspace_pointer, 0, self.workspace_reset_bytes, self._cuda_stream
            )
            if status != self._cuda_success:
                raise RuntimeError(f"shared FC12 asynchronous counter reset failed: {status}")
            self.compiled(
                activation=quantized,
                fc1_weight=fc1_weight.transpose(1, 2),
                activation_sf=scales.view(torch.float8_e8m0fnu).view(-1, self.hidden_size // 32),
                fc1_weight_sf=fc1_weight_sf.view(torch.float8_e8m0fnu),
                fc2_weight=fc2_weight.transpose(1, 2),
                fc2_weight_sf=fc2_weight_sf.view(torch.float8_e8m0fnu),
                fc2_output=output,
                local_workspace=self.workspace_pointer,
                stream=self.stream_handle,
                expert_token_sizes=self.token_counts[token_rows : token_rows + 1],
                expert_token_prefix_sum=None,
                topk_scores=None,
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
            )
        return output[:, 0, :]


@lru_cache(maxsize=None)
def _get_runner(
    device_index: int,
    hidden_size: int,
    intermediate_size: int,
    max_tokens: int,
    sm_count: int,
    swiglu_limit: float | None,
    stream_handle: int,
) -> _SharedFc12Runner:
    with torch.cuda.device(device_index):
        return _SharedFc12Runner(
            device_index,
            hidden_size,
            intermediate_size,
            max_tokens,
            sm_count,
            swiglu_limit,
            stream_handle,
        )


def shared_fc12_cache_info() -> dict[str, int | None]:
    """Return runner-cache counters; changing token counts must not create a new entry."""
    return _get_runner.cache_info()._asdict()


def run_shared_fc12(
    activation: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    *,
    swiglu_limit: float | None,
    sm_count: int,
    max_tokens: int = 8192,
) -> torch.Tensor:
    """Run one complete FC12 kernel after the existing shared-expert input quantizer.

    ``activation`` is contiguous BF16 ``[T, H]``. FP8 weights have shapes
    ``[1, 2*I, H]`` (gate/up interleaved in groups of 16) and ``[1, H, I]``.
    Their uint8 scales are flattened R128c4 UE8M0 ``[1, padded_rows * K/32]``.
    ``H`` and ``I`` must be divisible by 128; output is BF16 ``[T, H]``.
    ``max_tokens`` is a fixed model capacity, not the current batch size.
    """
    if activation.dtype != torch.bfloat16 or not activation.is_cuda:
        raise ValueError("shared FC12 requires CUDA BF16 activations")
    if activation.ndim != 2 or not activation.is_contiguous():
        raise ValueError("shared FC12 requires contiguous [T, H] activations")
    if max_tokens <= 0 or activation.shape[0] > max_tokens:
        raise ValueError("shared FC12 token count exceeds its configured capacity")
    hidden_size = activation.shape[1]
    if fc2_weight.ndim != 3 or fc2_weight.shape[:2] != (1, hidden_size):
        raise ValueError("shared FC12 weight must have shape [1, H, I]")
    intermediate_size = fc2_weight.shape[2]
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("shared FC12 hidden and intermediate dimensions must be multiples of 128")
    if fc1_weight.shape != (1, 2 * intermediate_size, hidden_size):
        raise ValueError("shared FC1 weight must have shape [1, 2*I, H]")
    for weight in (fc1_weight, fc2_weight):
        if weight.dtype != torch.float8_e4m3fn or not weight.is_contiguous():
            raise ValueError("shared FC12 weights must be contiguous E4M3 tensors")
    for scale in (fc1_weight_sf, fc2_weight_sf):
        if scale.dtype != torch.uint8 or not scale.is_contiguous():
            raise ValueError("shared FC12 scales must be contiguous uint8 tensors")
    if any(
        t.device != activation.device
        for t in (fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf)
    ):
        raise ValueError("shared FC12 inputs and weights must be on the same CUDA device")
    if activation.shape[0] == 0:
        return torch.empty_like(activation)
    stream = torch.cuda.current_stream(activation.device)
    runner = _get_runner(
        activation.device.index,
        hidden_size,
        intermediate_size,
        max_tokens,
        sm_count,
        swiglu_limit,
        stream.cuda_stream,
    )
    with torch.cuda.device(activation.device):
        return runner(activation, fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf)
