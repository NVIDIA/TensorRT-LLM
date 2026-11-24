# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp8 target's index scorer runner: the family ``kernels/m3_index_score`` with this target's two schedule
options on (``prestage_old`` in the early-issue kernels, ``early_trigger`` in every kernel) and its own compiled
kernels."""

from tensorrt_llm.logger import logger

from ...kernels import m3_index_score as _family
from ...kernels.m3_index_score import IndexDecodeScoreKernel

__all__ = [
    "INDEX_SCORE_PRESTAGE_OLD",
    "INDEX_SCORE_SET",
    "IndexDecodeScoreKernel",
    "IndexScoreRunner",
]

# With the early issue, a CTA whose first block holds only tokens older than this step also TMA-loads that block's
# index-K tile before the grid wait (bit-identical scores).
INDEX_SCORE_PRESTAGE_OLD = True
# Every CTA releases the dependent block selector right after its setup instead of at its exit (the selector's own
# grid wait still returns only once the whole scoring grid is complete, so the picks are unchanged), at every key.
INDEX_SCORE_SET = True


class IndexScoreRunner(_family.IndexScoreRunner):
    """The family runner with ``INDEX_SCORE_PRESTAGE_OLD`` / ``INDEX_SCORE_SET`` and its own compiled kernels."""

    kernel_cache: dict = {}
    PRESTAGE_OLD = INDEX_SCORE_PRESTAGE_OLD
    EARLY_TRIGGER = INDEX_SCORE_SET

    @classmethod
    def _compile(cls, *key) -> None:
        if key in cls.kernel_cache:
            return
        if cls.EARLY_TRIGGER and not cls.kernel_cache:
            logger.info(
                "tp8 index scorer: the block selector is released after the scorer's setup (INDEX_SCORE_SET)"
            )
        super()._compile(*key)
