# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MMMU references count questions in total; lm-eval limits each of its 30 subjects."""

from types import SimpleNamespace

import pytest

from tensorrt_llm.llmapi import DSparkDecodingConfig

__extra_import_path__ = ["~/tests/integration"]
from defs.accuracy import accuracy_core  # noqa: E402

pytestmark = pytest.mark.cpu_only

MODEL = "mistralai/Mistral-Large-4-1T-A52B"


@pytest.mark.parametrize("profile,total,per_subject", [("quick", 90, 3), ("full", 900, 30)])
@pytest.mark.parametrize("spec_dec_algo", [None, "DSpark"])
def test_reference_total_reaches_the_evaluator_per_subject(
    monkeypatch, profile, total, per_subject, spec_dec_algo
):
    monkeypatch.delenv("INTEGRATION_TEST", raising=False)
    task = accuracy_core.MMMU(MODEL)
    specs = dict(
        dtype="auto",
        quant_algo="MIXED_PRECISION",
        kv_cache_quant_algo="FP8",
        spec_dec_algo=spec_dec_algo,
        extra_acc_spec=f"native-mm-{profile}",
    )
    assert task.get_hypothesis_testing_params(**specs).num_samples == total
    built = {}

    class Evaluator:
        def __init__(self, **kwargs):
            built.update(kwargs)

        def evaluate(self, llm, sampling_params, streaming, **kwargs):
            return 100.0

    monkeypatch.setattr(task, "EVALUATOR_CLS", Evaluator)
    quant = SimpleNamespace(quant_algo="MIXED_PRECISION", kv_cache_quant_algo="FP8")
    spec_config = DSparkDecodingConfig(max_draft_len=3) if spec_dec_algo else None
    args = SimpleNamespace(speculative_config=spec_config, dtype="auto", quant_config=quant)
    task.evaluate(SimpleNamespace(args=args), extra_acc_spec=specs["extra_acc_spec"])
    assert built["num_samples"] == per_subject


def test_integration_smoke_keeps_one_question_per_subject():
    assert accuracy_core.MMMU(MODEL).evaluator_num_samples(1) == 1


def test_total_must_split_evenly_across_subjects():
    with pytest.raises(AssertionError, match="30 subjects"):
        accuracy_core.MMMU(MODEL).evaluator_num_samples(100)
