# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared FP8 MLP using a single MegaMoE FC1/FC2 compute kernel."""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from .gated_mlp import GatedMLP


def _unpack_r128c4(scales: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    if scales.dtype != torch.uint8 or scales.ndim != 1:
        raise ValueError("Shared FC12 requires loaded uint8 R128c4 weight scales")
    if not scales.is_contiguous() or scales.numel() != padded_rows * padded_columns:
        raise ValueError("Shared FC12 weight scale storage has an incompatible shape")
    return (
        scales.reshape(padded_rows // 128, padded_columns // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(padded_rows, padded_columns)[:rows, :columns]
        .contiguous()
    )


def _pack_r128c4(scales: torch.Tensor) -> torch.Tensor:
    rows, columns = scales.shape
    padded = F.pad(scales, (0, -columns % 4, 0, -rows % 128))
    return (
        padded.reshape((rows + 127) // 128, 4, 32, (columns + 3) // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(1, -1)
    )


def _interleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    intermediate = tensor.shape[0] // 2
    # Shared Linear stores gate || up; the FC12 epilogue consumes alternating 16-row blocks.
    return (
        tensor.reshape(2, intermediate // 16, 16, tensor.shape[1])
        .permute(1, 0, 2, 3)
        .contiguous()
        .reshape(tensor.shape)
    )


class MegaMoESharedMLP(GatedMLP):
    """Preserve GatedMLP loading and TP shards while fusing both projections.

    Linear owns checkpoint loading and its existing FP8 resmoothing. This adapter
    only rearranges the loaded FP8/UE8M0 bytes. The caller owns shared-output
    scaling and reduction; this module always returns the local TP contribution.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        bias: bool,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        dtype: torch.dtype | None = None,
        config: ModelConfig | None = None,
        overridden_tp_size: int | None = None,
        reduce_output: bool = False,
        layer_idx: int | None = None,
        use_cute_dsl_blockscaling_mm: bool = False,
        disable_deep_gemm: bool = False,
        use_custom_cublas_mm: bool = False,
        is_shared_expert: bool = False,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        fc12_reserved_sms: int = 0,
    ):
        if bias or activation is not F.silu:
            raise ValueError("Shared FC12 requires bias=False and SwiGLU activation")
        if swiglu_alpha not in (None, 1.0) or swiglu_beta not in (None, 0.0):
            raise ValueError("Shared FC12 requires SwiGLU alpha=1 and beta=0")
        if swiglu_limit is not None and not float(swiglu_limit) > 0:
            raise ValueError("Shared FC12 clamp must be positive or None")
        if dtype not in (None, torch.bfloat16):
            raise ValueError("Shared FC12 requires BF16 activations")
        if reduce_output or use_custom_cublas_mm:
            raise ValueError("Shared FC12 requires local output and the CuTe DSL backend")
        if type(fc12_reserved_sms) is not int or fc12_reserved_sms < 0:
            raise ValueError("Shared FC12 reserved-SM budget must be a non-negative integer")
        config = config or ModelConfig()
        if config.use_cuda_graph:
            raise ValueError("Shared FC12 currently requires eager execution")
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
            activation=activation,
            dtype=dtype,
            config=config,
            overridden_tp_size=overridden_tp_size,
            reduce_output=False,
            layer_idx=layer_idx,
            # This selects the existing Linear post-load UE8M0 conversion, including when
            # the caller's dense GEMM preference is unset.
            use_cute_dsl_blockscaling_mm=True,
            disable_deep_gemm=disable_deep_gemm,
            is_shared_expert=is_shared_expert,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )
        self._validate_fp8_quantization()
        self._fc12_reserved_sms = fc12_reserved_sms
        self._fc12_max_tokens = int(config.max_num_tokens)
        if self._fc12_max_tokens <= 0:
            raise ValueError("Shared FC12 max_num_tokens must be positive")
        self._fc12_source_state: tuple[tuple[int, int, int | None], ...] | None = None
        for name in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
            self.register_buffer(f"_fc12_{name}", None, persistent=False)

    backend_name = "cutedsl_megamoe_shared_fc12"
    kernel_name = "BlockScaledSwapAbFc12Kernel"
    quantization_kind = "mxfp8_e4m3"

    def _validate_fp8_quantization(self) -> None:
        for projection in (self.gate_up_proj, self.down_proj):
            quant_config = projection.quant_config
            if quant_config is None or not quant_config.layer_quant_mode.has_fp8_block_scales():
                raise ValueError("Shared FC12 requires FP8_BLOCK_SCALES for both projections")

    @property
    def sm_count(self) -> int:
        if not self.gate_up_proj._weights_created:
            raise RuntimeError("Shared FC12 SM count requires materialized CUDA weights")
        device = self.gate_up_proj.weight.device
        if device.type != "cuda":
            raise RuntimeError("Shared FC12 SM count requires CUDA weights")
        total_sms = torch.cuda.get_device_properties(device).multi_processor_count
        reserved = self._fc12_reserved_sms
        if not 0 <= reserved < total_sms:
            raise ValueError("Shared FC12 reserved-SM budget must leave at least one SM")
        return total_sms - reserved

    def cache_derived_state(self) -> None:
        """Drop derived views after loading, including staged in-place restoration."""
        self._fc12_source_state = None
        for name in ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf"):
            setattr(self, f"_fc12_{name}", None)

    def post_load_weights(self) -> None:
        self.cache_derived_state()

    @torch.no_grad()
    def prepare_fc12_weights(self) -> None:
        """Cache FC12 layouts after both Linear post-load transformations finish."""
        gate_up, down = self.gate_up_proj, self.down_proj
        self._validate_fp8_quantization()
        if not gate_up._weights_transformed or not down._weights_transformed:
            raise RuntimeError("Shared FC12 requires both Linear post_load_weights calls first")
        sources = (gate_up.weight, gate_up.weight_scale, down.weight, down.weight_scale)
        source_state = tuple(
            (id(t), t.data_ptr(), None if t.is_inference() else t._version) for t in sources
        )
        if source_state == self._fc12_source_state:
            return
        fc1, fc1_sf, fc2, fc2_sf = sources
        if fc1.dtype != torch.float8_e4m3fn or fc2.dtype != torch.float8_e4m3fn:
            raise ValueError("Shared FC12 requires loaded E4M3 weights")
        if fc1.ndim != 2 or fc2.ndim != 2 or not fc1.is_contiguous() or not fc2.is_contiguous():
            raise ValueError("Shared FC12 weights must be contiguous matrices")
        intermediate = fc2.shape[1]
        if fc1.shape != (2 * intermediate, self.hidden_size) or fc2.shape[0] != self.hidden_size:
            raise ValueError("Shared FC12 projection shapes do not describe the same TP shard")
        if (
            intermediate <= 0
            or self.hidden_size <= 0
            or intermediate % 128
            or self.hidden_size % 128
        ):
            raise ValueError(
                "Shared FC12 hidden and local intermediate sizes must be divisible by 128"
            )
        if any(t.device != fc1.device for t in sources):
            raise ValueError("Shared FC12 weights and scales must be on the same device")
        logical_fc1_sf = _unpack_r128c4(fc1_sf, 2 * intermediate, self.hidden_size // 32)
        _unpack_r128c4(fc2_sf, self.hidden_size, intermediate // 32)
        # Byte views keep packing independent of PyTorch FP8 indexing support.
        packed_fc1 = _interleave_gate_up(fc1.view(torch.uint8)).view(fc1.dtype).unsqueeze(0)
        packed_fc1_sf = _pack_r128c4(_interleave_gate_up(logical_fc1_sf))
        self._fc12_fc1_weight = packed_fc1
        self._fc12_fc1_weight_sf = packed_fc1_sf
        self._fc12_fc2_weight = fc2.unsqueeze(0)
        self._fc12_fc2_weight_sf = fc2_sf.unsqueeze(0)
        self._fc12_source_state = source_state

    def forward(
        self,
        x: torch.Tensor,
        all_rank_num_tokens: list[int] | None = None,
        final_all_reduce_params: AllReduceParams | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if lora_params or kwargs:
            raise ValueError("Shared FC12 does not support LoRA or extra forward options")
        if final_all_reduce_params is not None:
            raise ValueError("Shared FC12 reduction belongs to the enclosing MoE module")
        if not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError("Shared FC12 input must be a CUDA BF16 tensor")
        if x.ndim != 2 or x.shape[1] != self.hidden_size or not x.is_contiguous():
            raise ValueError("Shared FC12 input must be contiguous [tokens, hidden_size]")
        if x.shape[0] > self._fc12_max_tokens:
            raise ValueError("Shared FC12 input exceeds configured max_num_tokens")
        self.prepare_fc12_weights()
        if x.device != self._fc12_fc1_weight.device:
            raise ValueError("Shared FC12 input and weights must be on the same device")
        from ..cute_dsl_kernels.megamoe_shared_fc12 import run_shared_fc12

        return run_shared_fc12(
            x,
            self._fc12_fc1_weight,
            self._fc12_fc1_weight_sf,
            self._fc12_fc2_weight,
            self._fc12_fc2_weight_sf,
            swiglu_limit=self.swiglu_limit,
            sm_count=self.sm_count,
            max_tokens=self._fc12_max_tokens,
        )
