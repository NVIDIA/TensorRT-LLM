# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for FlashInfer sparse MLA."""


def get_sparse_mla_op():
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    return _sparse_mla_sm120_paged_attention
