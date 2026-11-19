# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The KV cache creator keeps the Kimi K3.1 DFlash2 drafter out of the engine's shared target + draft layout.

The engine can place a DFlash drafter's KV layers in the target's KV cache manager, after the target's layers. The
K3.1 DFlash2 drafter keeps its context latents in its own unified draft cache instead
(ops/spec/dflash2_draft_pages), and the K3.1 cache manager accepts no appended draft layers
(``accepts_appended_draft_layers``), so the creator refuses the shared layout, aggregated or disaggregated. The
refusal is decided before the DFlash attention backend is read: the drafter runs the TRTLLM backend here."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.kda_dsa_cache_manager import (
    KimiK3DSAHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator
from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.quantization.mode import QuantMode

pytestmark = pytest.mark.cpu_only

PACKAGE = "tensorrt_llm._torch.modeling_v2.models.kimi_k31"
TARGETS = {
    "kimi_k31_pro_mxfp4__sm_10x__default": "ModelingV2KimiK31ProMxfp4Sm10xDefault",
}


def _creator(target: str, is_disagg: bool) -> KvCacheCreator:
    """A KV cache creator for a K3.1 target serving DFlash2 as the K3.1 configs do, with the manager class the target
    declares (``runtime_hooks``); every other input allows the shared layout."""
    module = importlib.import_module(f"{PACKAGE}.{target}.modeling")
    target_cls = getattr(module, TARGETS[target])
    spec_config = DFlashDecodingConfig(
        max_draft_len=4,
        speculative_model="Kimi-K3.1-Dflash2-Preview",
        attention_backend="TRTLLM",
        use_rejection_sampling=True,
        skip_ctx_buffer_budget_check=True,
    )
    kv_cache_config = KvCacheConfig(
        use_kv_cache_manager_v2=True, tokens_per_block=64, enable_block_reuse=False
    )
    model_config = SimpleNamespace(
        sparse_attention_config=SimpleNamespace(algorithm="dsa"),
        quant_config=SimpleNamespace(quant_mode=QuantMode.from_quant_algo()),
        pretrained_config=SimpleNamespace(torch_dtype=torch.bfloat16),
    )
    creator = object.__new__(KvCacheCreator)
    creator._speculative_config = spec_config
    # The engine builds the subclass the target declares in place of the class it resolves
    # (test_modeling_v2_runtime.py covers the declared class).
    creator._kv_cache_manager_cls = target_cls.runtime_hooks.kv_cache_manager_cls
    # The creator reads the drafter off the loaded target, whose wrapper forwards ``draft_model``; an instance built
    # without __init__ stands for it.
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(
            draft_model=object.__new__(module.DraftModel),
            model_config=model_config,
            config=model_config.pretrained_config,
        )
    )
    creator._draft_config = SimpleNamespace(
        pretrained_config=SimpleNamespace(num_hidden_layers=5, torch_dtype=torch.bfloat16)
    )
    creator._mapping = SimpleNamespace(
        pp_size=1,
        cp_size=1,
        tp_size=8,
        enable_attention_dp=False,
        helix_repurposed_cp_size=1,
        has_cp_helix=lambda: False,
        pp_layers=lambda num_layers: list(range(num_layers)),
    )
    creator._llm_args = SimpleNamespace(attn_backend="TRTLLM", kv_cache_compression_config=None)
    creator._kv_cache_config = kv_cache_config
    creator._tokens_per_block = kv_cache_config.tokens_per_block
    creator._is_disagg = is_disagg
    creator._is_kv_cache_manager_v2 = True
    return creator


@pytest.mark.parametrize("is_disagg", [False, True], ids=["agg", "disagg"])
@pytest.mark.parametrize("target", list(TARGETS))
def test_the_dflash2_drafter_never_shares_the_target_manager(monkeypatch, target, is_disagg):
    # A launcher that asks for the shared layout under disaggregated serving.
    monkeypatch.setenv("TRTLLM_DISAGG_SHARED_DRAFT_KV", "1")
    creator = _creator(target, is_disagg)

    assert issubclass(creator._kv_cache_manager_cls, KimiK3DSAHybridCacheManager)
    assert not creator._kv_cache_manager_cls.accepts_appended_draft_layers()
    assert creator._dflash_unify_blocker() is not None
    assert not creator._should_unify_dflash_kv_cache()
    # A disaggregated server refuses the request for the shared layout at startup.
    if is_disagg:
        with pytest.raises(ValueError, match="TRTLLM_DISAGG_SHARED_DRAFT_KV=1 requires"):
            creator._check_shared_draft_kv_request()
    else:
        creator._check_shared_draft_kv_request()
