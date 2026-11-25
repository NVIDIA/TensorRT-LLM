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
"""DeCo-Attention CuTe DSL backend for visual generation models.

Self-attention on the DeCo-Attention kernels (``cute_dsl_kernels.blackwell.deco_attention``):
NVFP4 Q/K with compensation channels for the quantization error of the scores, and V smoothed
over blocks of keys and quantized to FP8 or NVFP4 (``v_dtype``; ``supported_v_dtypes()`` lists
the formats of the current GPU). The backend is a complete attention layer: a model builds it
like any other backend.

* Kernel calls: one sequence attending itself, bfloat16 q, k and v of one shape
  ``[B, S, H, 128]`` on a CUDA device, at least ``MIN_SEQ_LEN`` rows, no mask other than padding.
* VANILLA calls (unquantized): every other call, that is cross-attention, a causal
  ``attention_mask``, other dtypes or devices, shorter sequences, and every call of a layer whose
  head dimension or K/V head count the kernels do not take.
* Padding: ``key_padding_mask`` (``[B, S]``, True for a token) is read as one valid prefix per
  sample, its count of tokens, on the kernels and on VANILLA alike
  (``requires_trailing_padding``): the model puts the tokens of every sample first. Padded rows
  get no attention weight and come back zero.
* One sequence: a call whose q, k and v have one shape, on the kernels and on VANILLA alike. Its
  padded rows are excluded as keys and come back zero as queries. The Attention module builds
  its cross-attention layers on VANILLA, so such a call is self-attention; a direct caller
  whose cross-attention has as many queries as keys must not use this backend.
* Token tags: with ``token_tags``, half of the tokens that the kernels sample for the
  compensation channels are text and audio tokens of a packed sequence, for every sample from
  its own tokens.
* Device only: the token counts and the sampled tokens of a call are computed on the device and
  nothing is read back to the host. The layers of a forward share one draw: the kernel package
  keeps the last one while the model passes the same mask and tags tensors.
* CUDA graphs: first use (a kernel variant, the inputs of a sequence length) runs eagerly, as
  in the warm-up of a graph runner, and raises ``RuntimeError`` under capture. A captured call
  computes its token counts and sampled tokens inside the graph: a replay follows the content
  of the captured mask and tags tensors.
"""

from typing import Any

import torch

from tensorrt_llm.logger import logger
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

# Self-attention over fewer rows runs on VANILLA, such as the text of a token refiner.
MIN_SEQ_LEN = 8192


def supported_v_dtypes() -> tuple[str, ...]:
    """Return the ``v_dtype`` values with a DeCo-Attention kernel on the current GPU; empty when
    the kernel package does not import or the GPU has no kernels."""
    if deco_attention is None:
        return ()
    try:
        return tuple(deco_attention.pv_formats())
    except NotImplementedError:
        return ()


class DeCoAttention(AttentionBackend):
    """DeCo-Attention backend for diffusion self-attention (see the module docstring).

    A layer keeps nothing between calls. The workspace of a call comes from the caching
    allocator and lives for the call, as the other temporaries of a forward do.
    """

    def __init__(
        self,
        layer_idx: int = 0,
        num_heads: int = 8,
        head_dim: int = 128,
        num_kv_heads: int | None = None,
        dtype: torch.dtype | None = None,
        quant_attention_config: QuantAttentionConfig | None = None,
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
        # The V format of the recipe names the PV format of the kernel; Q/K are always NVFP4.
        v_dtypes = supported_v_dtypes()
        if not v_dtypes:
            raise ValueError(
                "DeCo-Attention has kernels for SM100, SM103, SM107 and SM120 GPUs only."
            )
        qk_dtype, v_dtype = quant_attention_config.qk_dtype, quant_attention_config.v_dtype
        if qk_dtype != "nvfp4" or v_dtype not in v_dtypes:
            raise ValueError(
                f"DeCo-Attention has no kernel for qk_dtype={qk_dtype!r}, v_dtype={v_dtype!r} on "
                f"this GPU, whose kernels take qk_dtype='nvfp4' with v_dtype in {v_dtypes}."
            )

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        self.quant_attention_config = quant_attention_config
        # A layer of another head dimension or K/V head count runs all its calls on VANILLA.
        kernel_head_dim = deco_attention.constants.HEAD_DIM
        self._has_kernel_geometry = head_dim == kernel_head_dim and num_kv_heads == num_heads
        if not self._has_kernel_geometry:
            logger.warning_once(
                f"DeCo-Attention runs the attention layers with head_dim {head_dim}, {num_heads} "
                f"Q heads and {num_kv_heads} K/V heads unquantized on VANILLA (first seen in "
                f"layer {layer_idx}); its kernels take head_dim {kernel_head_dim} and as many "
                "K/V heads as Q heads.",
                key=f"deco_attention_geometry_{head_dim}_{num_heads}_{num_kv_heads}",
            )
        self._noted_unquantized = False
        self._preferred_layout = AttentionTensorLayout.NHD
        self._vanilla = VanillaAttention(
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            dtype=dtype,
        )

    def _delegation_reason(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: Any
    ) -> str | None:
        """Return why a self-attention call of the kernels' geometry runs on VANILLA, None when
        the kernels take it."""
        if not q.is_cuda:
            return "inputs that are not on a CUDA device"
        if q.numel() == 0:
            return "empty inputs"
        if q.shape[1] < MIN_SEQ_LEN:
            return f"a sequence of {q.shape[1]} rows, under the {MIN_SEQ_LEN} the kernels take"
        if not q.dtype == k.dtype == v.dtype == torch.bfloat16:
            dtypes = sorted({str(x.dtype) for x in (q, k, v)} - {"torch.bfloat16"})
            return f"{' and '.join(dtypes)} inputs"
        if kwargs.get("attention_mask", PredefinedAttentionMask.FULL) != (
            PredefinedAttentionMask.FULL
        ):
            return "an attention mask"
        return None

    def _delegate(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, one_sequence: bool, **kwargs: Any
    ) -> torch.Tensor:
        """Run a call on VANILLA. For one sequence, ``key_padding_mask`` is read as a prefix, as
        the kernels read it, and the rows it excludes come back zero, as from the kernels; what
        they hold has no influence: a padded V row enters SDPA with weight zero, which spreads a
        NaN in it over the valid rows, so the padded K and V rows are zeroed first."""
        key_padding_mask = kwargs.get("key_padding_mask")
        padded = None
        if key_padding_mask is not None and one_sequence:
            # the mask as a prefix of its count of tokens, as _launch reads it
            token_counts = key_padding_mask.sum(dim=1, dtype=torch.int32)
            positions = torch.arange(q.shape[1], dtype=torch.int32, device=q.device)
            key_padding_mask = kwargs["key_padding_mask"] = positions[None] < token_counts[:, None]
            padded = ~key_padding_mask[:, :, None, None]
            k = k.masked_fill(padded, 0)
            v = v.masked_fill(padded, 0)
        # VANILLA works in HND ([B, H, S, D]).
        out = self._vanilla.forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), **kwargs
        ).transpose(1, 2)
        if padded is not None:
            out = out.masked_fill(padded, 0)
        return out

    @staticmethod
    def _kernel_input(x: torch.Tensor) -> bool:
        """Whether the kernels read x in place: contiguous or a view of a fused projection, at an
        aligned address."""
        return (
            x.is_contiguous() or deco_attention.token_rows(x)
        ) and x.data_ptr() % deco_attention.TENSOR_ALIGNMENT == 0

    @torch.compiler.disable
    def _launch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: torch.Tensor | None,
        token_tags: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the preparation and the attention kernel. Everything that reads a tensor address
        or launches a kernel is here, outside compiled code.

        What the padding and the tags give the kernels is computed on the device:

        * token_counts: Tokens per sample, int32 ``[B]`` (the kernels' ``valid_seq_lens``); None
          without a mask.
        * rows: Tokens that estimate the compensation basis, drawn by modality from the tokens
          of every sample (the kernels' ``sample_rows``, which keeps the last draw); None, the
          kernels' own draw, without tags.
        """
        # The kernels read the slices of a fused projection in place; a tensor of another
        # layout, or at an address that is not aligned, is copied.
        q, k, v = (
            x if self._kernel_input(x) else x.clone(memory_format=torch.contiguous_format)
            for x in (q, k, v)
        )
        batch_size, seq_len, num_heads, _ = q.shape
        if key_padding_mask is not None and key_padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"key_padding_mask must have shape {(batch_size, seq_len)}, got "
                f"{tuple(key_padding_mask.shape)}."
            )
        if token_tags is not None and token_tags.numel() != seq_len:
            raise ValueError(
                f"DeCo-Attention got {token_tags.numel()} token tags for a sequence of "
                f"{seq_len} tokens."
            )
        token_counts, rows = None, None
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(q.device)
            token_counts = key_padding_mask.sum(dim=1, dtype=torch.int32)
        if token_tags is not None:
            rows = deco_attention.sample_rows(
                seq_len, q.device, token_tags, tokens=key_padding_mask
            )
        pv = self.quant_attention_config.v_dtype
        num_bytes = deco_attention.workspace_size(batch_size, seq_len, num_heads, q.device, pv)
        # The kernels compile and launch on the current device, which is the device of q here.
        with torch.cuda.device(q.device):
            operands = deco_attention.prepare(
                q,
                k,
                v,
                torch.empty(num_bytes, dtype=torch.uint8, device=q.device),
                pv,
                rows=rows,
                valid_seq_lens=token_counts,
            )
            return deco_attention.attention(operands, torch.empty_like(q))

    @torch.compiler.disable
    def _note_unquantized(self, reason: str) -> None:
        """Note that the layer delegates a call: a layer notes it once, and the process logs each
        reason once."""
        self._noted_unquantized = True
        logger.warning_once(
            f"DeCo-Attention runs attention unquantized on VANILLA because of {reason} (first "
            f"seen in layer {self.layer_idx}); its kernels take bfloat16 CUDA inputs of at least "
            f"{MIN_SEQ_LEN} rows of self-attention and padding, no other mask.",
            key=f"deco_attention_unquantized_{reason}",
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Attention over q, k, v ``[B, S, H, D]``; returns ``[B, S, H, D]`` in q's dtype.

        The module docstring says how the backend reads the keywords ``key_padding_mask`` and
        ``token_tags``.
        """
        key_padding_mask = kwargs.get("key_padding_mask")
        if key_padding_mask is not None and key_padding_mask.dtype != torch.bool:
            key_padding_mask = kwargs["key_padding_mask"] = key_padding_mask.bool()
        # The kernels attend one sequence to itself at their head geometry; another geometry
        # (noted when the layer is built) and cross-attention are delegated on every call.
        one_sequence = q.shape == k.shape == v.shape
        if not (self._has_kernel_geometry and one_sequence):
            if self._has_kernel_geometry and not self._noted_unquantized:
                self._note_unquantized("cross-attention (q and k of different shapes)")
            return self._delegate(q, k, v, one_sequence, **kwargs)
        reason = self._delegation_reason(q, k, v, **kwargs)
        if reason is not None:
            if not self._noted_unquantized:
                self._note_unquantized(reason)
            return self._delegate(q, k, v, one_sequence, **kwargs)
        return self._launch(q, k, v, key_padding_mask, kwargs.get("token_tags"))

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_key_padding_mask(cls) -> bool:
        return True

    @classmethod
    def requires_trailing_padding(cls) -> bool:
        return True
