# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft KV dtype follows the POOL, not the target's quant config.

``SpecDecOneEngineForCausalLM.__init__`` used to stamp the target's
``kv_cache_quant_algo`` onto every draft config it built. For a draft that
reads the target's pool that is required; for one with its own pool it is
wrong, and two downstream sites in ``_util.py`` existed only to undo it
(``_get_draft_kv_model_config`` for the budget, the ``draft_kv_config.dtype``
reset for the allocation). The stamp is now guarded by
``should_use_separate_draft_kv_cache``.

These tests pin the guard's premises rather than the stamp itself: building a
``SpecDecOneEngineForCausalLM`` needs a real checkpoint, while the premises are
what would silently stop holding.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.speculative.interface import (
    SpeculativeDecodingMode,
    should_use_separate_draft_kv_cache,
)


def _spec_config(mode, *, shared_kv=False, embedded=False, allow=True):
    # _allow_separate_draft_kv_cache defaults True (llm_args.py:1890) and is
    # cleared at runtime by py_executor_creator (:286, :631) when a separate
    # pool cannot be used -- the draft then reads the target's pool, so taking
    # the target's dtype is right there.
    return SimpleNamespace(
        spec_dec_mode=mode,
        _use_shared_kv_cache=shared_kv,
        draft_is_embedded_in_target=embedded,
        _allow_separate_draft_kv_cache=allow,
    )


@pytest.mark.parametrize(
    "mode",
    [
        SpeculativeDecodingMode.DFLASH,
        SpeculativeDecodingMode.DSPARK,
        SpeculativeDecodingMode.PARD,
    ],
)
def test_external_drafter_is_one_engine(mode):
    """The guard reads a one-engine predicate, so this has to keep holding.

    If an external drafter ever stopped being one-engine,
    should_use_separate_draft_kv_cache would return False for it and the target's
    KV algo would silently start being stamped onto a separate draft pool again.
    """
    assert mode.is_external_drafter()
    assert mode.use_one_engine()


@pytest.mark.parametrize(
    "mode",
    [
        SpeculativeDecodingMode.DFLASH,
        SpeculativeDecodingMode.DSPARK,
        SpeculativeDecodingMode.PARD,
    ],
)
def test_external_drafter_owns_its_pool_so_it_keeps_its_dtype(mode):
    assert should_use_separate_draft_kv_cache(_spec_config(mode)) is True


def test_embedded_dspark_draft_shares_the_target_pool():
    """The embedded DSv4 draft reads the target's pool, so it must inherit."""
    assert (
        should_use_separate_draft_kv_cache(
            _spec_config(SpeculativeDecodingMode.DSPARK, embedded=True)
        )
        is False
    )


def test_an_explicitly_shared_pool_inherits():
    assert (
        should_use_separate_draft_kv_cache(
            _spec_config(SpeculativeDecodingMode.DSPARK, shared_kv=True)
        )
        is False
    )


@pytest.mark.parametrize(
    "mode",
    [SpeculativeDecodingMode.DFLASH, SpeculativeDecodingMode.DSPARK],
)
def test_runtime_fallback_to_a_shared_pool_inherits(mode):
    """py_executor_creator clears the allow flag when a separate pool is out.

    The draft then reads the target's pool, so it has to take the target's KV
    dtype -- the guard has to follow that, not just the structural mode.
    """
    assert should_use_separate_draft_kv_cache(_spec_config(mode, allow=False)) is False


def test_two_model_modes_are_untouched():
    """Not one-engine -> the guard leaves the previous behaviour in place."""
    for mode in (SpeculativeDecodingMode.DRAFT_TARGET, SpeculativeDecodingMode.NGRAM):
        assert mode.use_one_engine() is False
        assert should_use_separate_draft_kv_cache(_spec_config(mode)) is False
