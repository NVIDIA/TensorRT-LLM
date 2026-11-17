# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CoReba configuration checks.

dtype='fp8_ds_mla' allows pool rebalancing only with copy-based rebalancing, and
kv_pool_rebalance_config applies only with enable_kv_pool_rebalance.
"""

import pytest

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _copy_based_rebalancing_flags,
)
from tensorrt_llm.llmapi.llm_args import (
    CacheTransceiverConfig,
    KvCacheConfig,
    KvPoolRebalanceConfig,
    TorchLlmArgs,
)


def _config(rebalance: bool, copy_based: bool, in_place: bool = False) -> KvCacheConfig:
    return KvCacheConfig(
        dtype="fp8_ds_mla",
        enable_kv_pool_rebalance=rebalance,
        kv_pool_rebalance_config=KvPoolRebalanceConfig(
            copy_based_rebalancing=copy_based, in_place_relocation=in_place
        ),
    )


def test_fp8_ds_mla_vmm_rebalance_rejected():
    with pytest.raises(ValueError, match="copy_based_rebalancing"):
        _config(rebalance=True, copy_based=False)


@pytest.mark.parametrize("in_place", [False, True])
def test_fp8_ds_mla_copy_based_rebalance_allowed(in_place):
    cfg = _config(rebalance=True, copy_based=True, in_place=in_place)
    assert cfg.enable_kv_pool_rebalance and cfg.kv_pool_rebalance_config.copy_based_rebalancing


def test_fp8_ds_mla_without_rebalance_allowed():
    assert not _config(rebalance=False, copy_based=False).enable_kv_pool_rebalance


@pytest.mark.parametrize("disagg", [False, True])
def test_copy_based_rebalancing_rejected_with_disaggregated_serving(tmp_path, disagg):
    """The arena's pools all start at its base, so per-pool NIXL registrations would overlap."""
    kwargs = dict(model=str(tmp_path), kv_cache_config=_config(rebalance=True, copy_based=True))
    if disagg:
        kwargs["cache_transceiver_config"] = CacheTransceiverConfig(backend="NIXL")
        with pytest.raises(ValueError, match="not supported with disaggregated serving"):
            TorchLlmArgs(**kwargs)
    else:
        TorchLlmArgs(**kwargs)


def test_copy_based_rebalancing_without_rebalance_allowed_with_disaggregated_serving(tmp_path):
    """Without enable_kv_pool_rebalance no arena is built, so there is nothing to reject."""
    TorchLlmArgs(
        model=str(tmp_path),
        kv_cache_config=_config(rebalance=False, copy_based=True),
        cache_transceiver_config=CacheTransceiverConfig(backend="NIXL"),
    )


@pytest.mark.parametrize(
    "rebalance, expected",
    [(True, (True, True, True)), (False, (False, False, False))],
    ids=["rebalance", "no_rebalance"],
)
def test_copy_based_flags_follow_enable_kv_pool_rebalance(rebalance, expected):
    """The manager builds the arena only with rebalancing on.

    Built without it, the arena would bypass the checks keyed on enable_kv_pool_rebalance (sparse
    KV cache offload rejects rebalancing there) and hit the arena's own sparse-pool check instead.
    """
    cfg = KvCacheConfig(
        enable_kv_pool_rebalance=rebalance,
        kv_pool_rebalance_config=KvPoolRebalanceConfig(
            copy_based_rebalancing=True,
            drop_unreferenced_on_rebalance=True,
            in_place_relocation=True,
        ),
    )
    assert _copy_based_rebalancing_flags(cfg) == expected
