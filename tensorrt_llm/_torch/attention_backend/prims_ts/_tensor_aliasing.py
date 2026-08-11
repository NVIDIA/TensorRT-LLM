# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Storage-alias checks shared by task-scheduled attention launchers."""

from typing import Optional

import torch


def _tensor_byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Return a conservative half-open byte span covering a strided tensor.

    The interval includes stride holes. That deliberate over-approximation
    keeps alias rejection safe for paged-cache views whose outer page stride is
    larger than one compact page. ``data_ptr`` already includes the tensor's
    storage offset, so no storage implementation API is needed.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.layout != torch.strided:
        raise TypeError("tensor must have strided layout")

    byte_start = tensor.data_ptr()
    numel = tensor.numel()
    if numel == 0:
        return byte_start, byte_start

    element_size = tensor.element_size()
    if tensor.is_contiguous():
        return byte_start, byte_start + numel * element_size

    min_element_offset = 0
    max_element_offset = 0
    for extent, stride in zip(tensor.shape, tensor.stride(), strict=True):
        last_offset = (int(extent) - 1) * int(stride)
        min_element_offset += min(last_offset, 0)
        max_element_offset += max(last_offset, 0)

    return (
        byte_start + min_element_offset * element_size,
        byte_start + (max_element_offset + 1) * element_size,
    )


def _byte_spans_overlap(lhs_span: tuple[int, int], rhs_span: tuple[int, int]) -> bool:
    """Return whether two half-open byte spans overlap."""

    lhs_start, lhs_end = lhs_span
    rhs_start, rhs_end = rhs_span
    if lhs_start == lhs_end or rhs_start == rhs_end:
        return False
    return lhs_start < rhs_end and rhs_start < lhs_end


def _tensors_overlap(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether conservative storage spans for two tensors overlap."""

    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        raise TypeError("overlap operands must be torch.Tensor")
    if lhs.device != rhs.device:
        return False
    return _byte_spans_overlap(_tensor_byte_span(lhs), _tensor_byte_span(rhs))


def _validate_out_does_not_overlap_inputs(
    out: torch.Tensor,
    *named_inputs: tuple[str, Optional[torch.Tensor]],
) -> None:
    """Reject output aliasing with any named live launch input."""

    _validate_tensor_does_not_overlap_inputs(out, "out", *named_inputs)


def _validate_tensor_does_not_overlap_inputs(
    checked_tensor: torch.Tensor,
    tensor_name: str,
    *named_inputs: tuple[str, Optional[torch.Tensor]],
) -> None:
    """Reject one named tensor aliasing with any named live launch input."""

    tensor_span = _tensor_byte_span(checked_tensor)
    for name, input_tensor in named_inputs:
        if (
            input_tensor is not None
            and input_tensor.device == checked_tensor.device
            and _byte_spans_overlap(tensor_span, _tensor_byte_span(input_tensor))
        ):
            raise ValueError(f"{tensor_name} must not overlap {name} storage")


__all__ = [
    "_tensor_byte_span",
    "_tensors_overlap",
    "_validate_out_does_not_overlap_inputs",
    "_validate_tensor_does_not_overlap_inputs",
]
