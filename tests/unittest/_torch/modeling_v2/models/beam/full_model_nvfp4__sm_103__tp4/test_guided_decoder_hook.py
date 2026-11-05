# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the Beam NVFP4 target's guided-decoder hand-off to its DFlash worker (stubs)."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.beam.full_model_nvfp4__sm_103__tp4.modeling import (  # noqa: E501
    ModelingV2BeamFullModelNvfp4Sm103Tp4,
)
from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.runner import DecoderRunner
from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine

pytestmark = pytest.mark.cpu_only


class _Worker:
    """Stands in for the DFlash worker: records the decoder it is given."""

    def __init__(self, accepts: bool):
        self.accepts = accepts
        self.received = []

    def set_guided_decoder(self, guided_decoder) -> bool:
        self.received.append(guided_decoder)
        return self.accepts


def _target(spec_worker):
    model = ModelingV2BeamFullModelNvfp4Sm103Tp4.__new__(ModelingV2BeamFullModelNvfp4Sm103Tp4)
    model.__dict__.update(spec_worker=spec_worker)
    return model


@pytest.mark.parametrize("accepts", [True, False])
def test_forwards_the_decoder_to_the_spec_worker_and_returns_its_result(accepts):
    worker = _Worker(accepts)
    decoder = object()
    assert _target(worker).set_guided_decoder(decoder) is accepts
    assert worker.received == [decoder]


@pytest.mark.parametrize(
    "spec_worker",
    [None, SimpleNamespace(set_draft_model=lambda draft_model: None)],
    ids=["no_spec_worker", "worker_without_hook"],
)
def test_declines_without_a_worker_that_takes_the_decoder(spec_worker):
    assert _target(spec_worker).set_guided_decoder(object()) is False


def test_model_engine_hands_the_capturable_decoder_to_the_target():
    worker = _Worker(accepts=True)
    decoder = object()
    runner = DecoderRunner.__new__(DecoderRunner)
    runner.__dict__.update(_inputs=SimpleNamespace(guided_decoder=None))
    engine = SimpleNamespace(model=_target(worker), _runner=runner)
    assert PyTorchModelEngine.set_guided_decoder(engine, decoder) is True
    assert worker.received == [decoder]
    assert runner.guided_decoder is decoder


def test_model_engine_reports_a_target_that_declines():
    runner = DecoderRunner.__new__(DecoderRunner)
    runner.__dict__.update(_inputs=SimpleNamespace(guided_decoder=None))
    engine = SimpleNamespace(model=_target(None), _runner=runner)
    assert PyTorchModelEngine.set_guided_decoder(engine, object()) is False
    assert runner.guided_decoder is None
