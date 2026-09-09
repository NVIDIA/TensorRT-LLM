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

"""Host side of one-shot live-metadata validation and capacity reduction.

The one-shot public entry points validate tensor structure before this module
launches GPU inspectors and decodes their fixed Int64 summary into Python
planning facts. Reusable plans instead receive an explicit row-capacity bound.
Inspection deliberately performs one packed device-to-host copy and does not
inspect token-mask contents or build run-time route metadata.
"""

from collections.abc import Callable

import torch

from .config import _BlockSparseStaticProfile


# Device summary ABI; keep this order synchronized with block_sparse_inspect.py:
# validation error, maximum semantic BSR blocks in one row.
_SUMMARY_FIELDS = 2


def _bsr_error_reason(error_code: int) -> str:
    return {
        1: "indices in each row must be strictly increasing and unique",
        2: "indices must select an in-range KV block",
        3: "each indptr row range must be bounded and monotone",
    }.get(error_code, f"unknown validation error {error_code}")


def _raise_for_noncanonical_bsr(summary_values: tuple[int, ...]) -> None:
    """Translate the device validation code after the one packed copy."""

    error_code = summary_values[0]
    if error_code == 0:
        return
    reason = _bsr_error_reason(error_code)
    raise ValueError(f"block_indptr/block_indices must form canonical BSR: {reason}")


def _raise_for_invalid_paged_metadata(
    summary_values: tuple[int, ...],
    *,
    minimum_seq_len_kv: int,
    max_seq_len_kv: int,
) -> None:
    """Decode the strongest paged request or live-BSR validation failure."""

    error_code = summary_values[0]
    if error_code == 0:
        return
    reason = {
        4: (f"seq_lens_kv values must lie in [{minimum_seq_len_kv}, {max_seq_len_kv}]"),
        5: (
            "paged_kv_indptr must start at zero and each row must be "
            "bounded and monotone"
        ),
        6: "paged_kv_indptr rows must contain enough pages for seq_lens_kv",
        7: "paged_kv_indices must contain an in-range physical page ID",
    }.get(error_code)
    if reason is None:
        reason = (
            "block_indptr/block_indices must form canonical BSR for live "
            f"seq_lens_kv: {_bsr_error_reason(error_code)}"
        )
    raise ValueError(reason)


def _collect_inspection_summary(
    device: torch.device,
    stream: torch.cuda.Stream,
    launch: Callable[[torch.Tensor, int], None],
) -> tuple[int, ...]:
    """Run inspectors under one capture-safe stream context and copy once."""

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "block-sparse inspection is unsupported during CUDA Graph capture"
        )
    with torch.cuda.device(device_index), torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "block-sparse inspection is unsupported during CUDA Graph capture"
            )
        summary_gpu = torch.zeros(_SUMMARY_FIELDS, dtype=torch.int64, device=device)
        launch(summary_gpu, device_index)
        return tuple(int(value) for value in summary_gpu.tolist())


def _inspect_block_sparse_bsr(
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    static: _BlockSparseStaticProfile,
    stream: torch.cuda.Stream,
) -> int:
    """Inspect raw BSR and return its maximum semantic row width.

    ``block_indptr`` is contiguous Int32
    ``[B, Hkv, ceil(Sq / q_block_size) + 1]`` and stores absolute ranges into
    contiguous Int32 ``block_indices[nnz]``.

    The GPU validates each referenced range before reading it, reduces all rows
    into one zero-initialized Int64[2], and this function performs the sole D2H
    synchronization before returning the semantic BSR-block bound. The run-time
    prepare kernel resolves current BSR indices and token bits into fixed-stride
    route metadata.
    """

    from ..kernels.fmha_decode.block_sparse_inspect import (
        compile_block_sparse_inspection,
    )

    def launch(summary: torch.Tensor, device_index: int) -> None:
        inspect_bsr = compile_block_sparse_inspection(
            device_index=device_index,
            batch_size=static.batch_size,
            num_kv_heads=static.num_kv_heads,
            seq_len_q=static.seq_len_q,
            seq_len_kv=static.seq_len_kv,
            q_block_size=static.q_block_size,
            kv_block_size=static.kv_block_size,
        )
        inspect_bsr(block_indptr, block_indices, None, summary)

    summary_values = _collect_inspection_summary(
        block_indptr.device,
        stream,
        launch,
    )

    _raise_for_noncanonical_bsr(summary_values)
    return summary_values[1]


def _inspect_paged_block_sparse_metadata(
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    static: _BlockSparseStaticProfile,
    num_physical_kv_pages: int,
    stream: torch.cuda.Stream,
) -> int:
    """Inspect paged requests and return the maximum live BSR row width."""

    page_size = static.page_size
    if page_size is None:
        raise TypeError("paged inspection requires a paged static profile")
    minimum_seq_len_kv = static.seq_len_q if static.mask_type == "causal" else 1

    from ..kernels.fmha_decode.block_sparse_inspect import (
        compile_paged_block_sparse_metadata_inspection,
    )

    def launch(summary: torch.Tensor, device_index: int) -> None:
        inspect_metadata = compile_paged_block_sparse_metadata_inspection(
            device_index=device_index,
            batch_size=static.batch_size,
            num_kv_heads=static.num_kv_heads,
            seq_len_q=static.seq_len_q,
            minimum_seq_len_kv=minimum_seq_len_kv,
            max_seq_len_kv=static.seq_len_kv,
            q_block_size=static.q_block_size,
            kv_block_size=static.kv_block_size,
            page_size=page_size,
        )
        inspect_metadata(
            block_indptr,
            block_indices,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens_kv,
            num_physical_kv_pages,
            summary,
        )

    summary_values = _collect_inspection_summary(
        block_indptr.device,
        stream,
        launch,
    )

    _raise_for_invalid_paged_metadata(
        summary_values,
        minimum_seq_len_kv=minimum_seq_len_kv,
        max_seq_len_kv=static.seq_len_kv,
    )
    return summary_values[1]


__all__ = [
    "_inspect_block_sparse_bsr",
    "_inspect_paged_block_sparse_metadata",
]
