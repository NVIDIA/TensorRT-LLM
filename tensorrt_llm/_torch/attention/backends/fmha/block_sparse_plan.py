# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal

import torch

from tensorrt_llm._torch.attention.backends.prims_ts import (
    BlockSparsePagedTSWrapper,
    BlockSparseTSWrapper,
    SageAttentionConfig,
)
from tensorrt_llm._torch.attention.backends.prims_ts._block_sparse.config import (
    _validate_block_sparse_static_profile,
)


@dataclass(frozen=True, slots=True)
class BlockSparsePlan:
    """Cache key and plan parameters for one static PrimTS block-sparse kernel profile."""

    device: torch.device
    batch_size: int
    seq_len_q: int
    kv_capacity: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int | None
    q_block_size: int
    kv_block_size: int
    max_blocks_per_row: int | None
    mask_type: Literal["dense", "causal"]
    dtype: torch.dtype
    qk_dtype: torch.dtype
    v_dtype: torch.dtype
    use_kv_valid_bits: bool
    sparse_format: Literal["bsr", "bitmask"]
    use_proxy_routes: bool
    sage: SageAttentionConfig | None
    use_block_sparse: bool = True

    def unsupported_reason(self) -> str | None:
        # A dense plan has no route capacity.
        capacity = {"max_blocks_per_row": self.max_blocks_per_row} if self.use_block_sparse else {}
        try:
            _validate_block_sparse_static_profile(
                batch_size=self.batch_size,
                seq_len_q=self.seq_len_q,
                seq_len_kv=self.kv_capacity,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                q_block_size=self.q_block_size,
                kv_block_size=self.kv_block_size,
                use_kv_valid_bits=self.use_kv_valid_bits,
                mask_type=self.mask_type,
                q_dtype=self.qk_dtype,
                kv_dtype=self.qk_dtype,
                v_dtype=self.v_dtype,
                output_dtype=self.dtype,
                page_size=self.page_size,
                sage=self.sage,
                use_block_sparse=self.use_block_sparse,
                **capacity,
            )
        except (ValueError, NotImplementedError, OverflowError) as error:
            return str(error)
        return None

    def plan(self) -> "BlockSparseTSWrapper | BlockSparsePagedTSWrapper":
        paged = self.page_size is not None
        wrapper_type = BlockSparsePagedTSWrapper if paged else BlockSparseTSWrapper
        wrapper = wrapper_type()
        plan_args = (
            self.batch_size,
            self.seq_len_q,
            self.kv_capacity,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.q_block_size,
            self.kv_block_size,
        )
        plan_kwargs = {
            "device": self.device,
            "max_blocks_per_row": self.max_blocks_per_row,
            "use_kv_valid_bits": self.use_kv_valid_bits,
            "mask_type": self.mask_type,
            "q_data_type": self.qk_dtype,
            "o_data_type": self.dtype,
        }
        if paged:
            plan_args += (self.page_size,)
            plan_kwargs.update(kv_data_type=self.qk_dtype)
        else:
            plan_kwargs.update(
                use_block_sparse=self.use_block_sparse,
                sparse_format=self.sparse_format,
                use_proxy_routes=self.use_proxy_routes,
                k_data_type=self.qk_dtype,
                v_data_type=self.v_dtype,
                sage_config=self.sage,
            )
        wrapper.plan(*plan_args, **plan_kwargs)
        return wrapper
