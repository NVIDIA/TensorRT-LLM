"""``NVFP4Linear``: NVFP4 GEMM via ``ltx_kernels.nvfp4``.
Weights are expected already quantized (``weight`` / ``weight_scale`` /
``weight_scale_2``) — conversion happens in policy ``SDOps`` at load time, not here.
Parameter layout:
* ``weight`` — packed NVFP4 ``uint8`` ``[out, in // 2]`` (two E2M1 values per byte;
  element ``2j`` in the high nibble)
* ``weight_scale`` — E4M3 block scales as ``uint8``, cuBLAS 128x4 tiled layout
* ``weight_scale_2`` — scalar FP32 weight-only per-tensor decode scale
* ``input_scale`` — scalar FP32 activation per-tensor decode scale (``ActScale.STATIC``)
"""

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import assert_never

from tensorrt_llm._torch.visual_gen.models.ltx_core.quantization.nvfp4.kernels import require_nvfp4
from tensorrt_llm._torch.visual_gen.models.ltx_core.quantization.nvfp4.types import ActScale

E4M3_MAX = 448.0
E2M1_MAX = 6.0
GLOBAL_RANGE = E4M3_MAX * E2M1_MAX


class NVFP4Linear(nn.Module):
    """Drop-in ``nn.Linear`` replacement using NVFP4 quantize + cuBLASLt GEMM.
    ``act_scale`` controls the activation *per-tensor* decode scale:
    * ``ActScale.FIXED_1`` — constant 1.0 (no activation amax; ``alpha = weight_scale_2``)
    * ``ActScale.DYNAMIC`` — per-forward ``amax / GLOBAL_RANGE``
    * ``ActScale.STATIC`` — calibrated ``input_scale`` from the checkpoint (``alpha =
      input_scale * weight_scale_2``). Per-16 E4M3 block scales are still computed each
      forward.
    """

    in_features: int
    out_features: int
    act_scale: ActScale

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        act_scale: ActScale = ActScale.FIXED_1,
        weight_scale_rows: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if in_features % 16 != 0:
            raise ValueError(f"in_features must be divisible by 16, got {in_features}")

        self.in_features = in_features
        self.out_features = out_features
        self.act_scale = act_scale

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        # Default: cuBLAS-tiled row pad (matches ``ltx_kernels.nvfp4.quantize_nvfp4``).
        rows = weight_scale_rows if weight_scale_rows is not None else (out_features + 127) // 128 * 128
        scale_cols = ((in_features // 16) + 3) // 4 * 4
        self.weight_scale = nn.Parameter(
            torch.empty(rows, scale_cols, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.weight_scale_2 = nn.Parameter(torch.empty((), dtype=torch.float32, device=device), requires_grad=False)

        match act_scale:
            case ActScale.STATIC:
                self.input_scale = nn.Parameter(
                    torch.empty((), dtype=torch.float32, device=device), requires_grad=False
                )
            case ActScale.FIXED_1 | ActScale.DYNAMIC:
                self.register_parameter("input_scale", None)
            case _:
                assert_never(act_scale)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        act_scale: ActScale = ActScale.FIXED_1,
    ) -> "NVFP4Linear":
        match act_scale:
            case ActScale.STATIC:
                raise ValueError(
                    "act_scale=ActScale.STATIC requires a calibrated checkpoint input_scale; "
                    "use build_nvfp4_prequant_policy, not online convert / from_linear."
                )
            case ActScale.FIXED_1 | ActScale.DYNAMIC:
                pass
            case _:
                assert_never(act_scale)
        # Online convert writes cuBLAS-tiled block scales (rows padded to 128).
        scale_rows = (linear.out_features + 127) // 128 * 128
        return cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            act_scale=act_scale,
            weight_scale_rows=scale_rows,
            device=linear.weight.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nvfp4 = require_nvfp4()
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])
        x_bf16 = x_2d if x_2d.dtype in (torch.bfloat16, torch.float16) else x_2d.to(torch.bfloat16)

        match self.act_scale:
            case ActScale.FIXED_1:
                tensor_scale_a: torch.Tensor | None = None
            case ActScale.DYNAMIC:
                tensor_scale_a = nvfp4.per_tensor_decode_scale(x_bf16)
            case ActScale.STATIC:
                if self.input_scale is None:
                    raise RuntimeError("act_scale=ActScale.STATIC requires a loaded input_scale parameter")
                tensor_scale_a = self.input_scale
            case _:
                assert_never(self.act_scale)

        out = nvfp4.linear_nvfp4(
            x_bf16,
            self.weight,
            self.weight_scale,
            self.weight_scale_2,
            bias=self.bias,
            act_scale=tensor_scale_a,
        )
        return out.view(*original_shape[:-1], self.out_features)
