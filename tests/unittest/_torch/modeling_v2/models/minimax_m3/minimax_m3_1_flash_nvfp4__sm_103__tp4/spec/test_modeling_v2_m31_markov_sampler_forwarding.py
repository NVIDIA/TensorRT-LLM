# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every MiniMax-M3.1-Flash target's Markov draft sampler (``dspark_markov_worker.S4WorkerMixin``) hands a step the
chain did not answer to the base sampler with the caller's arguments unchanged."""

import importlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

pytestmark = pytest.mark.cpu_only

_TARGETS = ("sm_103__tp4", "sm_103__tp8", "sm_107__tp2", "sm_107__tp4", "sm_107__tp8")


class _RecordingBase:
    def sample_draft_tokens(self, logits, spec_metadata, batch_size, **kwargs):
        self.forwarded = (logits, spec_metadata, batch_size, kwargs)
        return "in-tree"


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("with_out", [False, True])
def test_the_fallback_forwards_every_base_parameter(target, with_out):
    """Without chain tokens the sampler is the base's, called with the caller's arguments unchanged."""
    try:
        mod = importlib.import_module(
            "tensorrt_llm._torch.modeling_v2.models.minimax_m3."
            f"minimax_m3_1_flash_nvfp4__{target}.dspark_markov_worker"
        )
    except NoTarget as no_target:
        pytest.skip(str(no_target))

    class _Recorder(mod.S4WorkerMixin, _RecordingBase):
        pass

    wk = _Recorder()
    wk._s4_tokens = None
    logits, meta = torch.zeros(4, 8), object()
    out = torch.empty(4, dtype=torch.int32) if with_out else None
    kwargs = dict(num_contexts=1, draft_step=2, mapping_lm_head_tp="tp")
    if with_out:
        kwargs["out"] = out
    assert wk.sample_draft_tokens(logits, meta, 4, **kwargs) == "in-tree"
    got_logits, got_meta, got_batch, got_kwargs = wk.forwarded
    assert got_logits is logits and got_meta is meta and got_batch == 4
    assert got_kwargs == dict(num_contexts=1, draft_step=2, mapping_lm_head_tp="tp", out=out)
    assert wk.sample_draft_tokens(logits, meta, 4) == "in-tree"
    assert wk.forwarded[3] == dict(
        num_contexts=0, draft_step=None, mapping_lm_head_tp=None, out=None
    )
