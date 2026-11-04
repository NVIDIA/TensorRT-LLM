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
"""Single-launch draft-token embedding for the MTP draft loop.

The native ``Embedding`` of a model built without tensor parallelism runs
``pre_comm_embedding_ops``: a seven-kernel range mask, ``F.embedding`` and a
``masked_fill_``. This module produces the identical tensor in one launch:
row ``r`` is ``weight[ids[r]]`` when ``0 <= ids[r] < weight.shape[0]`` and
``+0.0`` otherwise. Values are copied bit for bit, so the result equals the
native output exactly.
"""

import weakref
from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_H = 2048
_NUM_WARPS = 4
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SUPPORTED_ID_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _draft_embedding_kernel(
    ids_ptr,  # int32/int64 [num_rows]
    weight_ptr,  # [vocab_size, hidden_size]
    out_ptr,  # [num_rows, hidden_size]
    vocab_size,
    hidden_size,
    weight_row_stride,
    out_row_stride,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= 0) & (token < vocab_size)
    safe_token = tl.where(valid, token, 0)
    cols = col_block * BLOCK_H + tl.arange(0, BLOCK_H)
    col_mask = cols < hidden_size
    values = tl.load(
        weight_ptr + safe_token * weight_row_stride + cols, mask=col_mask & valid, other=0.0
    )
    tl.store(out_ptr + row.to(tl.int64) * out_row_stride + cols, values, mask=col_mask)


def draft_embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Embed ``input_ids`` with the native out-of-range-to-zero semantics.

    Args:
        input_ids: ``[num_rows]`` int32 or int64 CUDA tensor, contiguous.
        weight: ``[vocab_size, hidden_size]`` contiguous CUDA tensor
            (bfloat16, float16 or float32).

    Returns:
        ``[num_rows, hidden_size]`` tensor with ``weight``'s dtype.
    """
    num_rows = input_ids.shape[0]
    vocab_size, hidden_size = weight.shape
    out = torch.empty((num_rows, hidden_size), dtype=weight.dtype, device=weight.device)
    if num_rows == 0:
        return out
    grid = (num_rows, triton.cdiv(hidden_size, _BLOCK_H))
    _draft_embedding_kernel[grid](
        input_ids,
        weight,
        out,
        vocab_size,
        hidden_size,
        weight.stride(0),
        out.stride(0),
        BLOCK_H=_BLOCK_H,
        num_warps=_NUM_WARPS,
    )
    return out


def module_ineligibility(embedding: torch.nn.Module) -> Optional[str]:
    """Return why ``embedding`` cannot use the fused kernel, or None.

    The fused kernel reproduces only the communication-free native path:
    ``tp_size == 1`` (the replicated full-vocabulary table the DeepSeek-V3 /
    GLM models build), an unquantized 2-D contiguous CUDA weight.
    """
    if getattr(embedding, "tp_size", None) != 1:
        return "embedding is tensor-parallel"
    if getattr(embedding, "has_any_quant", False):
        return "embedding weight is quantized"
    weight = getattr(embedding, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return "embedding has no weight tensor"
    if weight.dim() != 2 or not weight.is_cuda or not weight.is_contiguous():
        return "embedding weight is not a contiguous 2-D CUDA tensor"
    if weight.dtype not in _SUPPORTED_DTYPES:
        return f"unsupported embedding dtype {weight.dtype}"
    return None


def ids_ineligibility(input_ids: torch.Tensor, weight: torch.Tensor) -> Optional[str]:
    """Return why ``input_ids`` cannot use the fused kernel, or None."""
    if input_ids.dim() != 1:
        return "draft ids are not 1-D"
    if input_ids.dtype not in _SUPPORTED_ID_DTYPES:
        return f"unsupported id dtype {input_ids.dtype}"
    if input_ids.device != weight.device or not input_ids.is_contiguous():
        return "draft ids are not contiguous on the weight's device"
    return None


class DraftEmbedding:
    """Callable stand-in for a native ``Embedding`` inside the draft loop.

    ``__call__`` runs the fused kernel when the module and the ids are in the
    supported domain and the native module otherwise; the decision is made on
    the host before any device work. Attribute reads fall through to the native
    module, so consumers that inspect ``weight`` or other fields see the native
    object.
    """

    def __init__(self, native: torch.nn.Module):
        self._native = native
        self.ineligible_reason = module_ineligibility(native)

    @property
    def native(self) -> torch.nn.Module:
        return self._native

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.ineligible_reason is None:
            weight = self._native.weight
            if ids_ineligibility(input_ids, weight) is None:
                return draft_embedding(input_ids, weight)
        return self._native(input_ids)

    def __getattr__(self, name):
        if name == "_native":
            raise AttributeError(name)
        return getattr(self._native, name)


_WRAPPERS: "weakref.WeakKeyDictionary[torch.nn.Module, DraftEmbedding]" = (
    weakref.WeakKeyDictionary()
)


def wrap_draft_embedding(native: torch.nn.Module) -> DraftEmbedding:
    """Return the cached ``DraftEmbedding`` for ``native``."""
    wrapper = _WRAPPERS.get(native)
    if wrapper is None:
        wrapper = DraftEmbedding(native)
        _WRAPPERS[native] = wrapper
        from ... import dispatch

        dispatch.log_capability(
            "draft_loop",
            "draft_embedding",
            wrapper.ineligible_reason is None,
            wrapper.ineligible_reason or "",
        )
    return wrapper
