# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeCo-Attention CuTe DSL Backend for Visual Generation Models

Runs self-attention on the DeCo-Attention kernels (``cute_dsl_kernels.blackwell.deco_attention``):
each call runs the DeCo preparation and then the attention kernel of the recipe, two stateless
functions that work in a workspace buffer this backend hands them.

* ``qk_dtype="nvfp4", v_dtype="fp8"``: NVFP4 Q/K with compensation channels and FP8 PV, on SM100,
  SM103, SM107 (tcgen05 MMA) and SM120 (warp-level MMA).
* ``qk_dtype="nvfp4", v_dtype="nvfp4"``: the same Q/K with NVFP4 PV, on SM120.

The preparation rotates Q and K, quantizes them to NVFP4 and appends compensation channels for
the first-order quantization error of the scores; it smooths V over blocks of keys before
quantizing it. The kernels take NHD layout ([B, S, H, D]) with D = 128, as many K/V heads as Q
heads, queries and keys of one sequence, bfloat16 and no mask; other calls (cross-attention, a
causal or key padding mask, another dtype) run on the VANILLA backend.
"""

from typing import Any

import torch

from tensorrt_llm.visual_gen.args import QuantAttentionConfig

from .....attention.backends.interface import PredefinedAttentionMask
from ..interface import AttentionBackend, AttentionTensorLayout
from ..vanilla import VanillaAttention

_deco_import_error: BaseException | None = None
try:
    from tensorrt_llm._torch.visual_gen.cute_dsl_kernels.blackwell import deco_attention
except (ImportError, OSError) as e:
    deco_attention = None
    _deco_import_error = e


class DeCoAttention(AttentionBackend):
    """DeCo-Attention backend for diffusion self-attention (see the module docstring).

    The DeCo layers of one model component share one workspace through
    ``attention_metadata_state``: the layers run one after another on one stream, the kernels keep
    nothing in the workspace between calls, and each call writes its output into its own tensor.

    The kernels compile, and the inputs of the preparation that depend only on the sequence length
    or the device are built, on first use. A CUDA graph capture therefore follows an eager forward
    of the same shape, which the CUDA graph runner's warm-up provides; a first use under capture
    raises a ``RuntimeError``.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
        attention_metadata_state: dict[str, Any] | None = None,
        **kwargs,
    ):
        if _deco_import_error is not None:
            raise ImportError(
                f"DeCo-Attention kernels are not available. Import error: {_deco_import_error}"
            ) from _deco_import_error
        if quant_attention_config is None or not quant_attention_config.deco_attention:
            raise ValueError(
                "DeCoAttention needs a quant_attention_config with deco_attention=True."
            )
        num_kv_heads = num_kv_heads or num_heads
        kernel_head_dim = deco_attention.constants.HEAD_DIM
        if head_dim != kernel_head_dim or num_kv_heads != num_heads:
            raise ValueError(
                f"DeCo-Attention needs head_dim {kernel_head_dim} and as many K/V heads as Q "
                f"heads; got head_dim {head_dim}, {num_heads} Q heads and {num_kv_heads} K/V heads."
            )
        # The V format of the recipe names the PV format of the kernel; Q/K are always NVFP4.
        try:
            pv_formats = deco_attention.pv_formats()
        except deco_attention.target.NoKernelError as e:
            raise ValueError(
                "DeCo-Attention has kernels for SM100, SM103, SM107 and SM120 GPUs only."
            ) from e
        qk_dtype, v_dtype = quant_attention_config.qk_dtype, quant_attention_config.v_dtype
        if qk_dtype != "nvfp4" or v_dtype not in pv_formats:
            raise ValueError(
                f"DeCo-Attention has no kernel for qk_dtype={qk_dtype!r}, v_dtype={v_dtype!r} on "
                f"this GPU, whose kernels take qk_dtype='nvfp4' with v_dtype in {pv_formats}."
            )

        if attention_metadata_state is None:
            raise ValueError(
                "DeCoAttention requires `attention_metadata_state` from DiffusionModelConfig: "
                "the layers of a model component share one workspace buffer."
            )

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        self._state = attention_metadata_state
        self._preferred_layout = AttentionTensorLayout.NHD
        self._vanilla = VanillaAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
        )

    def _can_serve(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any) -> bool:
        """Whether the DeCo kernels apply to this call; every other call is delegated."""
        if not q.is_cuda or q.numel() == 0:
            return False
        # Cross-attention: the kernels attend one sequence to itself.
        if not q.shape == k.shape == v.shape:
            return False
        if not q.dtype == k.dtype == v.dtype == torch.bfloat16:
            return False
        if kwargs.get("key_padding_mask") is not None:
            return False
        mask = kwargs.get("attention_mask", PredefinedAttentionMask.FULL)
        return mask == PredefinedAttentionMask.FULL

    def _delegate(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Run a call the kernels do not take on VANILLA, which works in HND ([B, H, S, D]).

        A model's cross-attention layers take this path on every call, inside the compiled
        model, so it logs nothing.
        """
        out = self._vanilla.forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
        )
        return out.transpose(1, 2)

    @torch.compile
    def _compact_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Slices of a fused QKV projection are strided and the kernels read gap-free
        # [B, S, H, 128] tensors; the compiled copy keeps them on a vectorized kernel, while
        # already contiguous inputs pass through without a copy.
        return q.contiguous(), k.contiguous(), v.contiguous()

    def _workspace(self, num_bytes: int, device: torch.device) -> torch.Tensor:
        """The workspace of a call: the component's buffer, grown to its largest call.

        A CUDA graph keeps the addresses it was captured with, so a call under capture takes
        its workspace from the graph's memory pool instead, like its other temporaries.
        """
        if torch.cuda.is_current_stream_capturing():
            return torch.empty(num_bytes, dtype=torch.uint8, device=device)
        workspace = self._state.get("deco_workspace")
        if workspace is None or workspace.numel() < num_bytes or workspace.device != device:
            # Release the smaller buffer before allocating the larger one.
            workspace = self._state["deco_workspace"] = None
            workspace = torch.empty(num_bytes, dtype=torch.uint8, device=device)
            self._state["deco_workspace"] = workspace
        return workspace

    @torch.compiler.disable
    def _launch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, _ = q.shape
        pv = self.quant_attention_config.v_dtype
        num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, q.device, pv)
        operands = deco_attention.prepare(q, k, v, self._workspace(num_bytes, q.device), pv)
        return deco_attention.attention(operands, torch.empty_like(q))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Attention over q, k, v ``[B, S, H, D]``; returns ``[B, S, H, D]`` in q's dtype."""
        if not self._can_serve(q, k, v, **kwargs):
            return self._delegate(q, k, v, **kwargs)
        return self._launch(*self._compact_qkv(q, k, v))

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False
