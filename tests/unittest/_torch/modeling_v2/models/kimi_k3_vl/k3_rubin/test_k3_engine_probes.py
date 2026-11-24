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
"""Engine names the Kimi-K3 Rubin package probes behind the switches its Rubin deployment runs.

The package reaches main's engine through duck-typed probes (``hasattr``, ``getattr(..., None)``,
``inspect.signature``). A probe for a name the runtime object does not have returns its fallback,
so the switch behind it stops working without an error: ``TRTLLM_K3_LD_FUSED`` on main's FC12 leaf
was one such case. Each test pins the probes of one switch against the class the package gets at
runtime (the leaf ``create_moe`` builds, the KV cache manager the executor creates, ...), so a
rename on either side fails here rather than as a slower step in a perf run.

CPU only: classes and signatures, no kernels.
"""

import importlib
import inspect
from dataclasses import fields

from torch import nn

K3_RUBIN = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin"


def _k3_module(name):
    """A module of the Kimi-K3 Rubin package (``text_model``, ``spec.modeling_dspark_mla``)."""
    return importlib.import_module(f"{K3_RUBIN}.{name}")


def _bare(cls):
    """An instance of the ``nn.Module`` ``cls`` without running its constructor."""
    instance = cls.__new__(cls)
    nn.Module.__init__(instance)
    return instance


def _is_property(cls, name):
    return isinstance(inspect.getattr_static(cls, name, None), property)


def test_fc12_prezero_hooks_on_the_fc12_leaf():
    """``TRTLLM_K3_FC12_PREZERO`` (generation): the MoE runtime zeroes FC12's output and counters
    ahead of the call when ``isinstance(backend, CuteDslFc12FusedMoE)``, ``backend.has_nvfp4``
    and ``_k3_fc12_hooks(backend)`` hold for the ``CUTEDSL_FC12`` leaf. The context server's
    grouped-GEMM leaf zeroes its own output, so the switch stays generation-only."""
    from tensorrt_llm._torch.moe.fused_moe.cutedsl import (
        TrtllmCutedslFusedFc12Nvfp4Impl,
        TrtllmCutedslGroupedGemmNvfp4Impl,
    )
    from tensorrt_llm._torch.moe.fused_moe.fused_moe_cute_dsl_fc12 import CuteDslFc12FusedMoE

    hooks = _k3_module("text_model")._k3_fc12_hooks
    assert CuteDslFc12FusedMoE is TrtllmCutedslFusedFc12Nvfp4Impl
    assert _is_property(TrtllmCutedslFusedFc12Nvfp4Impl, "has_nvfp4")
    assert hooks(_bare(TrtllmCutedslFusedFc12Nvfp4Impl))
    assert not hooks(_bare(TrtllmCutedslGroupedGemmNvfp4Impl))


def test_fp8_kv_layer_selector_on_the_cache_policy():
    """``TLLM_KIMI_K3_FP8_KV_LAYERS`` (both roles): the MLA layers take their FP8 KV-cache
    fallback only where ``kimi_k3_cache_policy`` provides the selector the cache manager reads;
    otherwise every layer keeps the model-wide NVFP4 dtype and a warning names the gap."""
    from tensorrt_llm._torch import kimi_k3_cache_policy

    selector = getattr(kimi_k3_cache_policy, "get_kimi_k3_fp8_kv_layer_ids", None)
    assert callable(selector)
    assert "nvfp4_enabled" in inspect.signature(selector).parameters


def test_fp4_mla_probe_on_trtllm_attention():
    """The FP4 MLA paths (packed decode, ``TRTLLM_K3_ULL4_MLA_PLAN_EARLY``) key off
    ``mqa.uses_fp4_mla_attention`` on main's ``TrtllmAttention``."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    assert _is_property(TrtllmAttention, "uses_fp4_mla_attention")


def test_dspark_unified_draft_cache_probes_on_the_k3_hybrid_manager():
    """DSpark's drafter context KV in the target manager (``context_kv_only`` /
    ``context_kv_dtype``): the package's worker treats a manager without ``draft_layer_ids`` as
    having no unified draft cache, and runs the KDA bookkeeping after accept only when the
    manager is a ``MambaHybridCacheManager``. K3's manager is ``KimiK3HybridCacheManager``."""
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
    from tensorrt_llm._torch.speculative.dflash import DFlashSpecMetadata
    from tensorrt_llm.llmapi import DSparkDecodingConfig

    assert issubclass(KimiK3HybridCacheManager, MambaHybridCacheManager)
    assert _is_property(KimiK3HybridCacheManager, "draft_layer_ids")
    assert {"context_kv_only", "context_kv_dtype"} <= set(DSparkDecodingConfig.model_fields)
    assert "project_captures" in {f.name for f in fields(DFlashSpecMetadata)}
    assert callable(getattr(DFlashSpecMetadata, "bind_capture_projection", None))


def test_kda_midchunk_snapshot_probe_on_the_k3_hybrid_manager():
    """``TRTLLM_KDA_MIDCHUNK_SNAPSHOT`` (context): ``KimiK3MambaMetadata.prepare`` builds the
    snapshot plan only from the cache manager's ``get_kda_midchunk_snapshots(request_ids)``,
    ``(context index, pool slot, offset)`` per snapshot; without that method the KDA prefill
    never stores a mid-chunk state."""
    from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager

    fetch = getattr(KimiK3HybridCacheManager, "get_kda_midchunk_snapshots", None)
    assert callable(fetch)
    assert list(inspect.signature(fetch).parameters) == ["self", "request_ids"]


def test_dspark_helix_drafter_probes():
    """The DSpark drafter under Helix (generation): its latent cache is page-sharded through
    ``Mapping.helix_mapping_for_kv_heads``; its split-KV plan passes the K3 piece cost and its
    Helix exchange keeps the NCCL reformats eager only where main's functions take them."""
    from tensorrt_llm.mapping import Mapping

    drafter = _k3_module("spec.modeling_dspark_mla")
    assert callable(getattr(Mapping, "helix_mapping_for_kv_heads", None))
    assert drafter._mla_split_plan_kwargs(3) == (("piece_cost", 3),)
    assert drafter._helix_post_process_fn()[1]
