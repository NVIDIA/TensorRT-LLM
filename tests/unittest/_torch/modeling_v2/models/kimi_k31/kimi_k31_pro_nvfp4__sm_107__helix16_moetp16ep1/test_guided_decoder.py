# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The helix16 shell attaches a guided decoder to its DFlash worker, and the worker verifies against
the gathered logits while one is attached: the engine DFlash worker's ``set_guided_decoder`` and
execute / rollback hooks, no vocab-shard selection."""

import importlib

import pytest

from tensorrt_llm._torch.speculative.dflash import DFlashWorker as EngineDFlashWorker

_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(_PKG + ".modeling")


def _shell(mod, worker):
    shell = object.__new__(mod.ModelingV2KimiK31ProNvfp4Sm107Helix16Moetp16ep1)
    object.__setattr__(shell, "spec_worker", worker)
    return shell


def test_the_shell_forwards_to_its_spec_worker(mod):
    seen = []

    class Worker:
        def set_guided_decoder(self, guided_decoder):
            seen.append(guided_decoder)
            return False

    guided = object()
    assert _shell(mod, Worker()).set_guided_decoder(guided) is False
    assert seen == [guided]


def test_the_dflash_worker_takes_the_engine_hooks(mod):
    for name in (
        "set_guided_decoder",
        "_execute_guided_decoder_if_present",
        "_rollback_guided_decoder_after_verify",
    ):
        assert getattr(mod.DFlashWorker, name) is getattr(EngineDFlashWorker, name), name
    worker = object.__new__(mod.DFlashWorker)
    worker._vocab_parallel = "bound before"
    guided = object()
    assert _shell(mod, worker).set_guided_decoder(guided) is True
    assert worker.guided_decoder is guided
    assert worker._vocab_parallel is None


def test_a_guided_decoder_keeps_the_gathered_logits(mod):
    """With a guided decoder (it masks the gathered logits) the worker binds no vocab-shard
    selection, so target tokens never come from one rank's shard."""
    worker = object.__new__(mod.DFlashWorker)
    worker.guided_decoder = object()
    worker._vocab_parallel = "stale"
    worker.bind_target_lm_head(None, None)
    assert worker._vocab_parallel is None
    assert not worker._target_tokens_from_shard(None)
