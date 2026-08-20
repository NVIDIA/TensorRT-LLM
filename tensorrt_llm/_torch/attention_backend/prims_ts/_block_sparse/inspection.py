# Copyright (c) 2026 by FlashInfer team.
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

"""Host side of one-shot raw-BSR validation and capacity reduction.

The one-shot public entry point validates the tensor ABI before this module
checks Int32 addressing limits, launches the GPU inspector, and decodes its
fixed Int64 summary into Python planning facts. Reusable plans instead receive
an explicit row-capacity bound. Inspection deliberately performs one packed
device-to-host copy and does not inspect token-mask contents or build run-time
route metadata.
"""

from dataclasses import dataclass

import torch

from .common import _SIGNED_INT32_MAX


# Device summary ABI; keep this order synchronized with block_sparse_inspect.py:
# validation error, maximum semantic BSR blocks in one row.
_SUMMARY_FIELDS = 2


@dataclass(frozen=True)
class _BlockSparseInspection:
    """Planning facts derived from caller-owned canonical BSR metadata.

    ``max_row_block_count`` is the semantic BSR-block bound used by the
    one-shot entry point to create a temporary capacity-only plan.
    """

    max_row_block_count: int


def _validate_int32_extent(value: int, name: str) -> None:
    """Reject a validated positive extent that device Int32 cannot represent."""

    if value > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name} must fit in signed int32")


def _raise_for_noncanonical_bsr(summary_values: tuple[int, ...]) -> None:
    """Translate the device validation code after the one packed copy."""

    error_code = summary_values[0]
    if error_code == 0:
        return
    reason = {
        1: "indices in each row must be strictly increasing and unique",
        2: "indices must select an in-range KV block",
        3: "each indptr row range must be bounded and monotone",
    }.get(error_code, f"unknown validation error {error_code}")
    raise ValueError(f"block_indptr/block_indices must form canonical BSR: {reason}")


def _inspect_block_sparse_bsr(
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    q_block_size: int,
    kv_block_size: int,
    stream: torch.cuda.Stream,
) -> _BlockSparseInspection:
    """Inspect raw BSR on its CUDA device and return host planning facts.

    ``block_indptr`` is contiguous Int32
    ``[B, Hkv, ceil(Sq / q_block_size) + 1]`` and stores absolute ranges into
    contiguous Int32 ``block_indices[nnz]``.

    The GPU validates each referenced range before reading it, reduces all rows
    into one zero-initialized Int64[2], and this function performs the sole D2H
    synchronization before returning an immutable ``_BlockSparseInspection``.
    The run-time prepare kernel resolves current BSR indices and token bits into
    fixed-stride route metadata.
    """

    for value, name in (
        (batch_size, "batch_size"),
        (num_kv_heads, "num_kv_heads"),
        (seq_len_q, "seq_len_q"),
        (seq_len_kv, "seq_len_kv"),
    ):
        _validate_int32_extent(value, name)

    device = block_indptr.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    # Checking the caller's current stream first avoids entering another
    # device/stream context while an enclosing graph capture is active.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "block-sparse inspection is unsupported during CUDA Graph capture"
        )

    from ..kernels.fmha_decode.block_sparse_inspect import (
        compile_block_sparse_inspection,
    )

    with torch.cuda.device(device_index), torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "block-sparse inspection is unsupported during CUDA Graph capture"
            )
        summary_gpu = torch.zeros(
            _SUMMARY_FIELDS,
            dtype=torch.int64,
            device=device,
        )
        inspect_bsr = compile_block_sparse_inspection(
            device_index=device_index,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
        )
        inspect_bsr(
            block_indptr,
            block_indices,
            summary_gpu,
        )
        summary_values = tuple(int(value) for value in summary_gpu.tolist())

    _raise_for_noncanonical_bsr(summary_values)
    # This positional decode is the host mirror of the device summary ABI.
    return _BlockSparseInspection(
        max_row_block_count=summary_values[1],
    )


__all__ = ["_BlockSparseInspection", "_inspect_block_sparse_bsr"]
