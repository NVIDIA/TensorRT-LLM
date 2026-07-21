#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit guard for the Inkling ``INKLING_MOE_BACKEND=TRTLLM`` config override.

Root cause of the iter64 runtime failure: ``InklingMoE`` selected the trtllm-gen
routed-expert backend by shallow-copying ``model_config`` and assigning
``moe_backend = "TRTLLM"``. But ``ModelConfig`` freezes itself after construction
(``_frozen=True``) and its ``__setattr__`` rejects every field except a small
allowlist, so the assignment raised ``AttributeError: Cannot modify
ModelConfig.'moe_backend' - instance is frozen`` during model construction --
before any trtllm-gen dispatch could run, on BOTH the baseline and enabled rows
of the source_logit_replay job (Slurm 5493576).

``_moe_config_with_trtllm_backend`` is the frozen-safe replacement: it uses the
escape hatch documented in ``ModelConfig.__setattr__`` (``_frozen`` is itself
writable) to unfreeze the copy, retarget only ``moe_backend``, and re-freeze,
leaving the original config untouched. This CPU test pins that contract so the
regression is caught in seconds instead of after a multi-GPU allocation.

No GPU or checkpoint needed: ``ModelConfig(pretrained_config=None)`` constructs on
CPU and we freeze it exactly as ``ModelConfig.from_pretrained`` does
(``model_config._frozen = True``).
"""

import copy
import os
import unittest.mock as mock

import pytest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_inkling import (
    _inkling_trtllm_moe_backend, _moe_config_with_trtllm_backend)


def _frozen_model_config(moe_backend: str = "CUTLASS") -> ModelConfig:
    """A frozen ModelConfig, mirroring the post-``from_pretrained`` state that is
    actually handed to ``InklingMoE.__init__`` (model_config.py sets
    ``model_config._frozen = True`` at the end of ``from_pretrained``)."""
    mc = ModelConfig(pretrained_config=None)
    mc.moe_backend = moe_backend
    mc._frozen = True  # '_frozen' is writable even when frozen (the escape hatch)
    return mc


def test_naive_assignment_on_frozen_config_raises():
    """Pin the iter64 failure mode: the naive shallow-copy + direct assignment
    raises the exact frozen-instance AttributeError. This is what the helper must
    avoid; if a future edit reverts to it, this test documents why it breaks."""
    mc = _frozen_model_config()
    bad = copy.copy(mc)
    with pytest.raises(AttributeError, match="instance is frozen"):
        bad.moe_backend = "TRTLLM"


def test_helper_flips_backend_on_copy_only():
    """``_moe_config_with_trtllm_backend`` returns a copy whose ``moe_backend`` is
    ``TRTLLM`` while leaving the original config frozen and unchanged."""
    mc = _frozen_model_config("CUTLASS")

    moe_cfg = _moe_config_with_trtllm_backend(mc)

    # The returned config selects trtllm-gen ...
    assert moe_cfg.moe_backend == "TRTLLM"
    # ... and is a distinct object from the shared/global config ...
    assert moe_cfg is not mc
    # ... which is left byte-unchanged on the default backend and still frozen.
    assert mc.moe_backend == "CUTLASS"
    assert mc._frozen is True


def test_returned_config_is_refrozen():
    """The copy must be re-frozen so later stray writes are still rejected (the
    override is a one-shot backend selection, not a general unfreeze)."""
    mc = _frozen_model_config("CUTLASS")
    moe_cfg = _moe_config_with_trtllm_backend(mc)
    assert moe_cfg._frozen is True
    with pytest.raises(AttributeError, match="instance is frozen"):
        moe_cfg.attn_backend = "FLASHINFER"


def test_helper_shares_pretrained_and_quant_config():
    """Shallow copy: the routed-expert MoE config must still point at the same
    ``pretrained_config``/``quant_config`` objects the rest of the model uses, so
    ``create_moe`` sees the real Inkling config and NVFP4 quant, not a stub."""
    mc = _frozen_model_config("CUTLASS")
    moe_cfg = _moe_config_with_trtllm_backend(mc)
    assert moe_cfg.pretrained_config is mc.pretrained_config
    assert moe_cfg.quant_config is mc.quant_config


def test_env_gate_reads_inkling_moe_backend():
    """``_inkling_trtllm_moe_backend`` is the single env gate (case-insensitive)
    that turns the whole trtllm-gen path on; the default is off (CUTLASS)."""
    with mock.patch.dict(os.environ, {}, clear=True):
        assert _inkling_trtllm_moe_backend() is False
    with mock.patch.dict(os.environ, {"INKLING_MOE_BACKEND": "TRTLLM"}):
        assert _inkling_trtllm_moe_backend() is True
    with mock.patch.dict(os.environ, {"INKLING_MOE_BACKEND": "trtllm"}):
        assert _inkling_trtllm_moe_backend() is True
    with mock.patch.dict(os.environ, {"INKLING_MOE_BACKEND": "CUTLASS"}):
        assert _inkling_trtllm_moe_backend() is False


if __name__ == "__main__":
    test_naive_assignment_on_frozen_config_raises()
    test_helper_flips_backend_on_copy_only()
    test_returned_config_is_refrozen()
    test_helper_shares_pretrained_and_quant_config()
    test_env_gate_reads_inkling_moe_backend()
    print("INKLING_MOE_BACKEND_SELECT_UNIT_OK")
