# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The config-time DFlash buffer check sizes the Kimi K3.1 drafter context as the target's worker
allocates it (a window-bounded ring of 576-wide latents), through the family's
``@dflash_context_for`` declaration; a model without one keeps the engine's dense estimate.

The size checks run on the CPU stage (cpu_only); the comparison with the worker's real allocation needs a GPU and runs
on the GB300 stage."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2 import modeling_v2_dflash_ctx_bytes_per_slot
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_ring import (
    dflash2_ctx_bytes_per_slot,
)
from tensorrt_llm._torch.speculative import dflash

GIB = 1024**3
# The DFlash2 MLA drafter's geometry: five sliding-window MLA layers, one 512 + 64 latent each.
DRAFT_CONFIG = {
    "architectures": ["DFlash2DraftModel"],
    "block_size": 5,
    "dflash_config": {"attention_mode": "mla", "block_size": 5},
    "dtype": "bfloat16",
    "head_dim": 64,
    "hidden_size": 7168,
    "kv_lora_rank": 512,
    "layer_types": ["sliding_attention"] * 5,
    "max_position_embeddings": 1048576,
    "num_attention_heads": 96,
    "num_hidden_layers": 5,
    "num_key_value_heads": 96,
    "qk_rope_head_dim": 64,
    "sliding_window": 1024,
}
# 5 layers x 1056 ring rows (1024 + 5 + 5, whole 32-row pages) x 576 x 2 bytes.
RING_PER_SLOT = 5 * 1056 * 576 * 2
# The per-GPU memory headroom the budget check is given in these tests.
BUDGET = int(84.86 * GIB)


@pytest.mark.cpu_only
def test_k31_drafter_slot_is_its_window_ring():
    assert dflash2_ctx_bytes_per_slot(DRAFT_CONFIG, 1048576, 4) == RING_PER_SLOT
    # Width 8 (max_draft_len 7) still fits the same pages: 1024 + 8 + 8 -> 1056.
    assert dflash2_ctx_bytes_per_slot(DRAFT_CONFIG, 1048576, 7) == RING_PER_SLOT
    # A context shorter than the ring keeps the dense capacity: 512 + 5 -> 544 rows.
    assert dflash2_ctx_bytes_per_slot(DRAFT_CONFIG, 512, 4) == 5 * 544 * 576 * 2
    # Without a window on every layer there is no ring.
    dense = dict(DRAFT_CONFIG, layer_types=["sliding_attention"] * 4 + ["full_attention"])
    assert dflash2_ctx_bytes_per_slot(dense, 32768, 4) == 5 * 32800 * 576 * 2
    # No MLA geometry: the family cannot say.
    assert dflash2_ctx_bytes_per_slot({"num_hidden_layers": 5}, 32768, 4) is None


def _args(tmp_path, modeling_v2="require", arch="KimiK31ForConditionalGeneration", **extra):
    from tensorrt_llm.llmapi.llm_args import DFlashDecodingConfig, TorchLlmArgs

    target = tmp_path / "target"
    target.mkdir(exist_ok=True)
    (target / "config.json").write_text(json.dumps({"architectures": [arch]}))
    drafter = tmp_path / "drafter"
    drafter.mkdir(exist_ok=True)
    (drafter / "config.json").write_text(json.dumps(DRAFT_CONFIG))
    kwargs = dict(
        model=str(target),
        modeling_v2=modeling_v2,
        speculative_config=DFlashDecodingConfig(max_draft_len=4, speculative_model=str(drafter)),
        max_batch_size=8,
        max_num_tokens=16384,
        max_seq_len=1048576,
        tensor_parallel_size=8,
    )
    kwargs.update(extra)
    # model_construct skips model-path validation; only the DFlash budget check is under test.
    return TorchLlmArgs.model_construct(**kwargs)


@pytest.mark.cpu_only
def test_family_declares_the_ring_bytes(tmp_path):
    args = _args(tmp_path)
    assert modeling_v2_dflash_ctx_bytes_per_slot(args, DRAFT_CONFIG) == RING_PER_SLOT
    # No drafter config yet (a hub id before download): the engine estimate applies.
    assert modeling_v2_dflash_ctx_bytes_per_slot(args, None) is None
    # modeling_v2 off, or a model whose family declares nothing: the engine estimate applies.
    assert modeling_v2_dflash_ctx_bytes_per_slot(_args(tmp_path, "off"), DRAFT_CONFIG) is None
    (tmp_path / "other").mkdir()
    other = _args(tmp_path / "other", arch="OtherForCausalLM")
    assert modeling_v2_dflash_ctx_bytes_per_slot(other, DRAFT_CONFIG) is None


@pytest.mark.cpu_only
def test_k31_at_1m_mbs8_is_approved_with_the_ring_bytes(tmp_path):
    args = _args(tmp_path)
    with patch.object(
        dflash,
        "validate_dflash_ctx_buffer_budget",
        wraps=dflash.validate_dflash_ctx_buffer_budget,
    ) as validate:
        args._validate_dflash_ctx_budget(memory_budget_bytes=BUDGET)
    assert validate.call_args.kwargs["ctx_bytes_per_slot"] == RING_PER_SLOT
    # Nine slots (eight requests and the scratch slot) of the ring: about 52 MiB.
    assert 9 * RING_PER_SLOT < 64 * 1024**2


@pytest.mark.cpu_only
def test_dense_drafter_estimate_is_unchanged(tmp_path):
    # The same serve without modeling_v2 is sized as the engine's dense arena and refused:
    # 2 x 9 slots x 5 layers x (1048576 + 5) x 12 kv heads per rank x 64 x 2 bytes = 135.00 GiB.
    with pytest.raises(ValueError, match=r"need 135\.00 GiB per GPU"):
        _args(tmp_path, "off")._validate_dflash_ctx_budget(memory_budget_bytes=BUDGET)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match=r"need 135\.00 GiB per GPU"):
        _args(other, arch="OtherForCausalLM")._validate_dflash_ctx_budget(
            memory_budget_bytes=BUDGET
        )


@pytest.mark.cpu_only
def test_ring_bytes_still_refuse_a_budget_they_overrun(tmp_path):
    with pytest.raises(ValueError, match="largest max_batch_size estimated to fit is 6") as excinfo:
        _args(tmp_path)._validate_dflash_ctx_budget(memory_budget_bytes=7 * RING_PER_SLOT + 1)
    assert "drafter context per slot" in str(excinfo.value)


class _FakeMLADrafter:
    """What the target's DFlash worker reads off the DFlash2 MLA drafter to size its arena."""

    config = SimpleNamespace(max_position_embeddings=1 << 20)
    dflash_attention_backend = "VANILLA"
    _paged_ctx_cache = True
    _ctx_ring_rows = None

    def __init__(self, window, block_size):
        self.block_size = block_size
        self.fc = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16, device="cuda"))
        self._ctx_ring_window = window
        self._num_attn_layers = 2
        self._num_heads = 4
        self._num_kv_heads = 1
        self._head_dim = 16
        self._kv_factor = 1

    def _build_fused_kv_buffers(self):
        pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the worker allocates CUDA buffers")
# A ring at 1M and 32k; at 40 the dense 45-row capacity (64 in pages) is below the 96-row ring.
@pytest.mark.parametrize("max_seq_len", [1 << 20, 32768, 40])
@pytest.mark.parametrize("max_draft_len", [4, 7])
def test_estimate_matches_the_worker_allocation(max_seq_len, max_draft_len):
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        K31DFlashWorker,
    )
    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    window = 64
    worker = K31DFlashWorker(
        DFlashDecodingConfig(max_draft_len=max_draft_len, attention_backend="VANILLA"), Mapping()
    )
    draft = _FakeMLADrafter(window, max_draft_len + 1)
    worker.set_draft_model(draft)
    worker._lazy_init_ctx_buffers(
        draft, SimpleNamespace(max_num_requests=2), SimpleNamespace(max_seq_len=max_seq_len)
    )
    buf = worker._ctx_kv_buf
    allocated_per_slot = buf.numel() * buf.element_size() // 3  # two requests and the scratch slot
    config = dict(
        DRAFT_CONFIG,
        num_hidden_layers=2,
        kv_lora_rank=12,
        qk_rope_head_dim=4,
        sliding_window=window,
        layer_types=["sliding_attention"] * 2,
    )
    assert dflash2_ctx_bytes_per_slot(config, max_seq_len, max_draft_len) == allocated_per_slot


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the worker allocates CUDA buffers")
@pytest.mark.parametrize(
    "target",
    [
        "kimi_k31_pro_mxfp4__sm_10x__default",
        "kimi_k31_pro_mxfp4__sm_103__tep",
        "kimi_k31_pro_mxfp4__sm_103__dep",
    ],
)
def test_arena_follows_the_served_length_after_kv_cache_estimation(target):
    """KV cache estimation runs a probe manager whose small quota clamps the served length, then
    the real manager: the drafter arena (and the drafter's position ceiling) is rebuilt for the
    real one, and kept while the manager stays."""
    import importlib

    from tensorrt_llm.llmapi import DFlashDecodingConfig
    from tensorrt_llm.mapping import Mapping

    module = importlib.import_module(
        f"tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.{target}.modeling"
    )
    worker = module.K31DFlashWorker(
        DFlashDecodingConfig(max_draft_len=4, attention_backend="VANILLA"), Mapping()
    )
    draft = _FakeMLADrafter(64, 5)
    worker.set_draft_model(draft)
    spec_metadata = SimpleNamespace(max_num_requests=2)
    probe = SimpleNamespace(kv_cache_manager=SimpleNamespace(max_seq_len=15869))
    served = SimpleNamespace(kv_cache_manager=SimpleNamespace(max_seq_len=32779))
    worker._lazy_init_ctx_buffers(draft, spec_metadata, probe)
    assert worker._max_ctx == 15869
    worker._lazy_init_ctx_buffers(draft, spec_metadata, served)
    assert worker._max_ctx == 32779
    assert draft._runtime_position_ceiling == dflash.dflash_position_ceiling(
        32779, worker._compute_block_size, worker.max_draft_len
    )
    arena = worker._ctx_kv_buf
    worker._lazy_init_ctx_buffers(draft, spec_metadata, served)
    assert worker._ctx_kv_buf is arena
