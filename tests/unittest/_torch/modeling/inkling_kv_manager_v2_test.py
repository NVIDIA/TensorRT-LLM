#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic, CPU-only proof that the Inkling runtime dispatches KVCacheManagerV2.

The resolved ``kv_cache_config.use_kv_cache_manager_v2`` FLAG and the concrete KV
cache manager CLASS are two different things. The flag defaults to ``"auto"`` and,
absent a model default, resolves to ``False`` (that is the ``Resolved
use_kv_cache_manager_v2='auto' -> False`` line seen in the serve log). The CLASS,
however, is chosen by ``get_kv_cache_manager_cls`` -> ``_non_hybrid_kv_cache_manager_cls``,
which forces ``KVCacheManagerV2`` for Inkling via the ``is_inkling`` branch
(Inkling's local-16 / global-8 per-layer KV-head split needs V2's per-layer
``num_kv_heads`` geometry). ``_fallback_if_unsupported_kv_cache_manager_v2`` RAISES
for Inkling rather than silently downgrading, so the runtime cannot end up on V1.

This test proves both facts without a GPU, a running engine, or any log-capture /
``set -x`` grep (which can false-match its own trace):

  1. ``InklingForConditionalGeneration.get_model_defaults()`` declares
     ``kv_cache_config.use_kv_cache_manager_v2=True`` so the resolved flag agrees
     with the manager class on every launch path (LLM API, trtllm-serve,
     trtllm-eval).
  2. For the REAL Inkling checkpoint config, ``is_inkling`` is True,
     ``is_hybrid_linear`` is False, and the runtime's manager-class selector
     returns a ``KVCacheManagerV2`` subclass EVEN with the flag left at ``"auto"``.

Run: python -m pytest -q -s inkling_kv_manager_v2_test.py
Override the checkpoint with INKLING_CHECKPOINT=/path/to/Inkling-NVFP4-full.
"""
import os

import pytest

CKPT = os.environ.get(
    "INKLING_CHECKPOINT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/"
    "users/kleinc/hf_data/Inkling-NVFP4-full")


def test_get_model_defaults_declares_v2():
    """The model default must declare V2 so the resolved flag matches reality."""
    from tensorrt_llm._torch.models.modeling_inkling import \
        InklingForConditionalGeneration
    defaults = InklingForConditionalGeneration.get_model_defaults(None)
    assert isinstance(defaults, dict), defaults
    kv = defaults.get("kv_cache_config", {})
    assert kv.get("use_kv_cache_manager_v2") is True, defaults
    print(f"GET_MODEL_DEFAULTS_V2 kv_cache_config={kv}", flush=True)


def _load_inkling_pretrained_config():
    from transformers import AutoConfig

    # Import registers the Inkling auto-model / auto-config for the
    # trust_remote_code checkpoint.
    import tensorrt_llm._torch.models.modeling_inkling  # noqa: F401
    return AutoConfig.from_pretrained(CKPT, trust_remote_code=True)


@pytest.mark.skipif(not os.path.isdir(CKPT),
                    reason=f"Inkling checkpoint not found at {CKPT}")
def test_selector_returns_v2_for_real_inkling_config():
    """The runtime's manager-class selector returns V2 for the real Inkling
    config even with the flag at 'auto' (structural is_inkling override)."""
    from tensorrt_llm._torch.pyexecutor._util import \
        _non_hybrid_kv_cache_manager_cls
    from tensorrt_llm._torch.pyexecutor.config_utils import (is_hybrid_linear,
                                                             is_inkling)
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import \
        KVCacheManagerV2
    from tensorrt_llm.llmapi import KvCacheConfig

    config = _load_inkling_pretrained_config()
    assert is_inkling(config) is True, getattr(config, "model_type", None)
    # Inkling is not a nemotron/qwen3 hybrid-linear model, so it takes the
    # non-hybrid route where is_inkling forces V2 (not the mamba-hybrid route).
    assert is_hybrid_linear(config) is False

    # Flag deliberately left at the "auto" default to prove the CLASS choice is
    # independent of the flag.
    kv_cfg = KvCacheConfig()
    assert kv_cfg.use_kv_cache_manager_v2 == "auto", kv_cfg.use_kv_cache_manager_v2
    cls = _non_hybrid_kv_cache_manager_cls(config, kv_cfg)
    assert issubclass(cls, KVCacheManagerV2), cls.__name__
    print(
        f"KV_MANAGER_SELECTOR cls={cls.__name__} is_v2=True "
        f"flag={kv_cfg.use_kv_cache_manager_v2} model_type="
        f"{getattr(config, 'model_type', None)}",
        flush=True)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s", "-p", "no:cacheprovider"]))
