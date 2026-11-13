# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The features a model opts into through RuntimeHooks. The executor arms the process-wide ones
(``_arm_runtime_opt_ins``): the SM107 MXFP8 GEMM tactic table and the late-compile detector. The
speculative context one-hot index_fill and the DFlash host-side context allocation stay on the
model's own spec metadata and worker (``DecoderMetadata.setup_speculative``). A model that declares
none leaves them off, even at a table shape on SM107; the MiniMax-M3 sm_107 targets declare the first
two."""

import importlib

import pytest
import torch

from tensorrt_llm._torch import late_compile
from tensorrt_llm._torch.custom_ops import mxfp8_cute_pinned_tactics as pin
from tensorrt_llm._torch.pyexecutor.py_executor import _arm_runtime_opt_ins
from tensorrt_llm._torch.pyexecutor.runtime_hooks import runtime_hooks_of

pytestmark = pytest.mark.cpu_only

M3_SHAPE = (64, 6144, 6144)  # (m, n, k): a table entry
M3_TARGETS = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__{}.modeling"


class _PlainModel(torch.nn.Module):
    """A model that declares no hooks."""


@pytest.fixture(autouse=True)
def _sm107_unarmed(monkeypatch):
    monkeypatch.setattr(pin, "_ARMED", [False])
    monkeypatch.setattr(pin, "ENABLED", True)
    monkeypatch.setattr(pin, "get_sm_version", lambda: 107)
    monkeypatch.setattr(late_compile, "_ready", [False])


def test_a_model_without_the_opt_in_arms_nothing_at_an_m3_shape():
    hooks = runtime_hooks_of(_PlainModel())

    assert _arm_runtime_opt_ins(hooks) is False
    assert pin.tactic_for(*M3_SHAPE) is None
    assert not late_compile._ready[0]


@pytest.mark.parametrize("tp", ["tp2", "tp4", "tp8"])
def test_the_m3_sm107_targets_arm_both(tp):
    hooks = importlib.import_module(M3_TARGETS.format(tp))._runtime_hooks(None)

    assert hooks.mxfp8_pinned_tactics and hooks.late_compile_detect
    assert _arm_runtime_opt_ins(hooks) is True
    m, n, k = M3_SHAPE
    assert pin.tactic_for(m, n, k) == pin.TACTICS[(n, k, pin._bucket(m))]


def test_the_kill_switch_still_wins_once_armed(monkeypatch):
    pin.arm()
    monkeypatch.setattr(pin, "ENABLED", False)

    assert pin.tactic_for(*M3_SHAPE) is None


class _SpecWorker:
    def __init__(self):
        self.host_ctx_alloc_calls = 0

    def enable_host_ctx_alloc(self):
        self.host_ctx_alloc_calls += 1
        return True


def _setup_speculative(monkeypatch, hooks):
    """DecoderMetadata.setup_speculative of one model with ``hooks``: (spec metadata, spec worker)."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder import metadata as decoder_metadata

    monkeypatch.setattr(decoder_metadata, "create_spec_metadata", lambda *args: SimpleNamespace())
    runner_metadata = object.__new__(decoder_metadata.DecoderMetadata)
    runner_metadata._config = None
    runner_metadata.mapping = None
    runner_metadata.model = SimpleNamespace(config=None, spec_worker=_SpecWorker())
    runner_metadata.runtime_hooks = hooks
    runner_metadata.spec_metadata = None
    spec_metadata = runner_metadata.setup_speculative(None)
    return spec_metadata, runner_metadata.model.spec_worker


@pytest.mark.parametrize("opt_in", [False, True])
def test_the_speculative_opt_ins_stay_on_their_model(monkeypatch, opt_in):
    from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

    monkeypatch.delenv("TRTLLM_M31_DISABLE_ONEHOT_INDEX_FILL", raising=False)
    hooks = RuntimeHooks(context_onehot_index_fill=opt_in, dflash_host_ctx_alloc=opt_in)
    spec_metadata, worker = _setup_speculative(monkeypatch, hooks)
    assert spec_metadata.context_onehot_index_fill is opt_in
    assert worker.host_ctx_alloc_calls == int(opt_in)
    # A second model in the same process that opts into nothing keeps main's behavior.
    plain_metadata, plain_worker = _setup_speculative(monkeypatch, RuntimeHooks())
    assert plain_metadata.context_onehot_index_fill is False
    assert plain_worker.host_ctx_alloc_calls == 0


def test_spec_metadata_and_dflash_worker_default_to_main():
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker
    from tensorrt_llm._torch.speculative.interface import SpecMetadata

    assert SpecMetadata.__dataclass_fields__["context_onehot_index_fill"].default is False
    assert DFlashWorker._host_ctx_alloc_enabled is False


@pytest.mark.parametrize("killed", [False, True])
def test_the_dflash_host_ctx_alloc_kill_switch(monkeypatch, killed):
    from tensorrt_llm._torch.speculative.dflash import DFlashWorker

    if killed:
        monkeypatch.setenv("TRTLLM_DFLASH_DISABLE_HOST_CTX_ALLOC", "1")
    else:
        monkeypatch.delenv("TRTLLM_DFLASH_DISABLE_HOST_CTX_ALLOC", raising=False)
    worker = object.__new__(DFlashWorker)
    assert worker.enable_host_ctx_alloc() is not killed
    assert worker._host_ctx_alloc_enabled is not killed
    assert DFlashWorker._host_ctx_alloc_enabled is False
