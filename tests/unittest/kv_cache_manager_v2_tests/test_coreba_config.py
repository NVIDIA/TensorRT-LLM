# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KvCacheConfig: dtype='fp8_ds_mla' allows pool rebalancing only with copy-based rebalancing."""

import pytest

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
def test_copy_based_rebalancing_allowed_with_disaggregated_serving(tmp_path, disagg):
    """The KV cache transceiver registers the arena as one NIXL region, so the combination is valid."""
    kwargs = dict(model=str(tmp_path), kv_cache_config=_config(rebalance=True, copy_based=True))
    if disagg:
        kwargs["cache_transceiver_config"] = CacheTransceiverConfig(backend="NIXL")
    TorchLlmArgs(**kwargs)
