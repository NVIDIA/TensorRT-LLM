# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""reset_process_overrides returns every model override registry to its default, so a model built
later in the same process does not inherit an earlier model's overrides. CPU only."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch import flashinfer_autotune_cache
from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.models import modeling_dflash
from tensorrt_llm._torch.modules import logits_processor
from tensorrt_llm._torch.pyexecutor import perf_metrics_manager
from tensorrt_llm._torch.pyexecutor.model_overrides import reset_process_overrides
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_ops
from tensorrt_llm._torch.speculative import dspark
from tensorrt_llm._torch.speculative import interface as spec_interface

pytestmark = pytest.mark.cpu_only

# (module, registry, setter): installing through the setter fills the registry.
_REGISTRIES = [
    (
        spec_interface,
        "_CHAIN_SPEC_OVERRIDE",
        spec_interface.set_chain_speculative_sampling_override,
    ),
    (
        spec_interface,
        "_CHAIN_SPEC_SLOT_OVERRIDE",
        spec_interface.set_chain_speculative_sampling_slot_override,
    ),
    (spec_interface, "_PADDING_FILL_OVERRIDE", spec_interface.set_padding_rows_fill_override),
    (flashinfer_ops, "_SOFTMAX_OVERRIDE", flashinfer_ops.set_softmax_override),
    (flashinfer_ops, "_TOP_P_RENORM_OVERRIDE", flashinfer_ops.set_top_p_renorm_override),
    (
        flashinfer_ops,
        "_SAMPLING_FROM_PROBS_OVERRIDE",
        flashinfer_ops.set_sampling_from_probs_override,
    ),
    (dspark, "_ANCESTRAL_GLUE_OVERRIDE", dspark.set_ancestral_glue_override),
    (dspark, "_ANCESTRAL_BLOCK_OVERRIDE", dspark.set_ancestral_block_override),
    (logits_processor, "_LOGITS_WIDEN_OVERRIDE", logits_processor.set_logits_widen_override),
    (modeling_dflash, "_MERGE_FUTURE_OVERRIDE", modeling_dflash.set_dflash_merge_future_override),
]


def _sentinel(*args, **kwargs):
    return None


def test_reset_clears_every_registry(monkeypatch):
    for module, name, _ in _REGISTRIES:
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(spec_interface, "_CHEAP_PADDING_ROWS", spec_interface._CHEAP_PADDING_ROWS)
    monkeypatch.setitem(flashinfer_autotune_cache._STATE, "bundled", None)
    # A stand-in tuner: constructing the real one queries a CUDA device.
    monkeypatch.setattr(AutoTuner, "_instance", SimpleNamespace(bundle_reporter=None))
    monkeypatch.setattr(
        perf_metrics_manager, "_GPU_TIMESTAMPS", perf_metrics_manager._GPU_TIMESTAMPS
    )

    for module, name, setter in _REGISTRIES:
        setter(_sentinel)
        assert getattr(module, name) is _sentinel, name
    spec_interface.set_cheap_padding_rows(True)
    flashinfer_autotune_cache.use_bundled_file("/nonexistent/bundle.json")
    AutoTuner.get().bundle_reporter = object()
    perf_metrics_manager._GPU_TIMESTAMPS = True

    reset_process_overrides()

    for module, name, _ in _REGISTRIES:
        assert getattr(module, name) is None, name
    assert spec_interface._CHEAP_PADDING_ROWS is False
    assert flashinfer_autotune_cache._STATE["bundled"] is None
    assert AutoTuner.get().bundle_reporter is None
    assert perf_metrics_manager._GPU_TIMESTAMPS is False
