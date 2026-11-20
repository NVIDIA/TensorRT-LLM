# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility import for Blackwell FP8 MLA decode."""

from tensorrt_llm._torch.kernels.blackwell.mla_decode_fp8 import (
    MAX_SPLITS,
    BlackwellMultiHeadLatentAttentionForwardFP8,
    run,
)

__all__ = [
    "BlackwellMultiHeadLatentAttentionForwardFP8",
    "MAX_SPLITS",
    "run",
]


if __name__ == "__main__":
    from tensorrt_llm._torch.kernels.blackwell.mla_decode_fp8 import _main

    _main()
