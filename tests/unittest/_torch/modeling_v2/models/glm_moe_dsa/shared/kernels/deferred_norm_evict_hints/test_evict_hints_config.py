# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Host-only tests of the deferred_norm_evict_hints configuration parsing (no GPU work)."""

import dataclasses


def test_pre_wait_prefetch_is_a_test_arm():
    """The served hop configuration prefetches nothing before its dependency wait; tests build
    the pre-wait arms explicitly."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.hop import (  # noqa: E501
        PRE_WAIT_PREFETCH,
        EvictHintsHopConfig,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.projection import (  # noqa: E501
        parse_cfg,
    )

    assert EvictHintsHopConfig().poll_prefetch == "none"
    assert EvictHintsHopConfig().poll_prefetch not in PRE_WAIT_PREFETCH
    for mode in PRE_WAIT_PREFETCH + ("none", "wait3", "t3"):
        assert parse_cfg(f"poll_prefetch={mode}", EvictHintsHopConfig).poll_prefetch == mode


def test_defaults_are_deferred_norm_launch():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm import (  # noqa: E501
        DeferredNormProjConfig,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints import (  # noqa: E501
        EvictHintsHopConfig,
        EvictHintsProjConfig,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.projection import (  # noqa: E501
        EVICT_HINTS_CONFIG_KEYS,
    )

    z = EvictHintsProjConfig()
    w = DeferredNormProjConfig()
    for f in dataclasses.fields(DeferredNormProjConfig):
        assert getattr(z, f.name) == getattr(w, f.name), f.name
    extra = [
        f.name
        for f in dataclasses.fields(EvictHintsProjConfig)
        if f.name not in {g.name for g in dataclasses.fields(DeferredNormProjConfig)}
    ]
    assert tuple(extra) == EVICT_HINTS_CONFIG_KEYS
    # Served default: deferred_norm's launch plus the evict_first policy on the weight copies.
    assert not any(getattr(z, k) for k in EVICT_HINTS_CONFIG_KEYS if k != "copy_hint")
    assert z.copy_hint == "evict_first"
    h = EvictHintsHopConfig()
    assert (h.carveout, h.poll_prefetch, h.clear_hint, h.ctas, h.ffi, h.dyn_smem) == (
        -1,
        "none",
        "none",
        8,
        True,
        0,
    )


def test_full_indexer_layers_keep_deferred_norm_launch():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints import (  # noqa: E501
        EvictHintsProjConfig,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.deferred_norm_evict_hints.projection import (  # noqa: E501
        config_for,
        parse_cfg,
    )

    cfg = parse_cfg("prod_warp=1,split_chunk=1,split_delay_ns=3000", EvictHintsProjConfig)
    assert config_for(cfg, 0) is cfg
    assert config_for(cfg, 160) == EvictHintsProjConfig(copy_hint="none")
    assert config_for(EvictHintsProjConfig(), 160).copy_hint == "none"
