# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import pytest
import scipy
import torch
import yaml

import tensorrt_llm.evaluate
from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.evaluate.audio_asr import AudioASREvaluator
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
from tensorrt_llm.logger import logger

from ..conftest import llm_models_root
from .video_mme import VideoMME as VideoMMEEvaluator


def compute_theta(num_samples: int,
                  sigma: float,
                  alpha: float = 0.05,
                  beta: float = 0.2):
    scale = (2 * sigma**2 / num_samples)**0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    z_beta = scipy.stats.norm.ppf(beta)
    theta = -(z_alpha + z_beta) * scale
    return theta


def compute_threshold(num_samples: int,
                      ref_accuracy: float,
                      sigma: float,
                      alpha: float = 0.05,
                      higher_is_better: bool = True):
    scale = (2 * sigma**2 / num_samples)**0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    if higher_is_better:
        return ref_accuracy + z_alpha * scale
    else:
        return ref_accuracy - z_alpha * scale


@dataclass(slots=True)
class HypothesisTestingParams:
    ref_accuracy: float
    num_samples: int
    metric_name: str = "accuracy"
    alpha: float = 0.05
    beta: float = 0.2
    sigma: float = 50.0
    higher_is_better: bool = True
    theta: float = field(init=False)
    threshold: float = field(init=False)

    def __post_init__(self) -> None:
        self.theta = compute_theta(self.num_samples,
                                   sigma=self.sigma,
                                   alpha=self.alpha,
                                   beta=self.beta)
        self.threshold = compute_threshold(
            self.num_samples,
            self.ref_accuracy,
            sigma=self.sigma,
            alpha=self.alpha,
            higher_is_better=self.higher_is_better)

    def report(self, accuracy: Optional[float] = None) -> str:
        metric_name = self.metric_name.upper()
        report = f"""===========================================================
= {metric_name} HYPOTHESIS TESTING
===========================================================
Alpha (Type I:  False Positive): {self.alpha:.3f}
Beta  (Type II: False Negative): {self.beta:.3f}
Sigma (Standard deviation): {self.sigma:.3f}
#Samples: {self.num_samples}
Higher is better: {self.higher_is_better}
Theta (Minimum detectable effect): {self.theta:.3f}
Reference {self.metric_name}: {self.ref_accuracy:.3f}
Threshold: {self.threshold:.3f}
==========================================================="""
        if accuracy is not None:
            report = f"""{report}
Evaluated {self.metric_name}: {accuracy:.3f}
==========================================================="""
        return report

    def assert_passing(self, accuracy: float) -> None:
        compare_op = ">=" if self.higher_is_better else "<="
        err_msg = (
            f"Reference {self.metric_name} is {self.ref_accuracy:.3f}, threshold is {self.threshold:.3f}. "
            f"Expected {self.metric_name} {compare_op} threshold, but got {accuracy:.3f}. "
            f"Please see hypothesis testing report:\n{self.report(accuracy)}")
        if self.higher_is_better:
            assert accuracy >= self.threshold, err_msg
        else:
            assert accuracy <= self.threshold, err_msg


def assert_acceptance_length(test_key: str, al_value: float) -> None:
    """Assert acceptance length meets the registered minimum.

    Reads ``references/acceptance_length.yaml`` and checks
    ``al_value >= entry["min_al"]``.

    Args:
        test_key: Key in acceptance_length.yaml identifying the test variant,
            e.g. ``"TestGPTOSS::test_dflash"``.
        al_value: Observed mean acceptance length to check.

    Population:
        Set ``TRTLLM_POPULATE_ACCEPTANCE_LENGTH=1`` to write the observed
        value as ``ref_al`` and set ``min_al`` to 95% of it. The YAML key
        must already exist; add a ``ref_al: null`` / ``min_al: null`` stub
        when introducing a new test baseline.

    Raises:
        KeyError: If test_key is absent from the YAML.
        ValueError: If the YAML entry has ``min_al: null`` (not yet populated).
        AssertionError: If al_value < min_al.
    """
    populate = os.getenv("TRTLLM_POPULATE_ACCEPTANCE_LENGTH") == "1"
    if os.getenv("TRTLLM_ACCURACY_NO_REFERENCE") == "1" and not populate:
        return

    _ref_dir = f"{os.path.dirname(__file__)}/references"
    yaml_path = f"{_ref_dir}/acceptance_length.yaml"
    with open(yaml_path) as _f:
        baselines: dict = yaml.safe_load(_f)

    entry = baselines.get(test_key)
    if entry is None:
        raise KeyError(f"No acceptance-length baseline for '{test_key}'. "
                       f"Add an entry to {yaml_path} after a GPU run.")

    if populate:
        entry["ref_al"] = al_value
        entry["min_al"] = al_value * 0.95

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(baselines, f, sort_keys=False)

        print(f"[AL] populated {test_key}: "
              f"ref_al={entry['ref_al']:.6f}, min_al={entry['min_al']:.6f}")
        return
    min_al = entry.get("min_al")
    if min_al is None:
        raise ValueError(
            f"Acceptance-length baseline for '{test_key}' has min_al=null. "
            "Populate min_al after a GPU run, or set "
            "TRTLLM_ACCURACY_NO_REFERENCE=1 to skip the check.")

    ref_al = entry.get("ref_al")
    ref_str = f"{ref_al:.3f}" if isinstance(ref_al, (int, float)) else "null"
    assert al_value >= min_al, (
        f"[AL] Regression: {test_key}: "
        f"acceptance_length={al_value:.3f} < min_al={min_al:.3f} "
        f"(ref_al={ref_str})")


class AccuracyTask:
    REFERENCE_DIR = f"{os.path.dirname(__file__)}/references"

    # Dataset
    DATASET = None
    DATASET_DIR = None
    HIGHER_IS_BETTER = True
    METRIC_NAME = "accuracy"

    # Hypothesis testing parameters
    ALPHA = None
    BETA = None
    SIGMA = None
    NUM_SAMPLES = None

    # Input and output sizes
    MAX_INPUT_LEN = None
    MAX_OUTPUT_LEN = None
    MAX_BATCH_SIZE = None

    # Evaluator
    EVALUATOR_CLS = None
    EVALUATOR_KWARGS = None

    def __init__(self, model_name: str):
        with open(f"{self.REFERENCE_DIR}/{self.DATASET}.yaml") as f:
            self.reference: List[dict] = yaml.safe_load(f).get(model_name, [])

    def get_hypothesis_testing_params(self,
                                      **acc_specs) -> HypothesisTestingParams:
        """Get hypothesis testing parameters via accuracy specifications.

        Args:
            acc_specs: Accuracy specifications, currently including:
                dtype (str): Model data type. Defaults to 'auto'.
                quant_algo (str): Quantizaion algorithm. Defaults to None.
                kv_cache_quant_algo (str): KV cache quantizaion algorithm. Defaults to None.
                spec_dec_algo (str): Speculative decoding algorithm. Defaults to None.
                extra_acc_spec (str): Extra accuracy specifications. Defaults to None.
        """
        for entry in self.reference:
            matched = True
            for key, value in acc_specs.items():
                default = 'auto' if key == 'dtype' else None
                if entry.get(key, default) != value:
                    matched = False
                    break
            if matched:
                break
        else:
            if os.getenv("TRTLLM_ACCURACY_NO_REFERENCE") == "1":
                metric_key = self.METRIC_NAME.lower()
                entry = {metric_key: 0 if self.HIGHER_IS_BETTER else math.inf}
            else:
                raise ValueError(f"Not registered specs: {acc_specs}.")

        metric_key = self.METRIC_NAME.lower()
        return HypothesisTestingParams(
            ref_accuracy=entry.get(metric_key, entry.get("accuracy")),
            metric_name=self.METRIC_NAME,
            alpha=entry.get("alpha", self.ALPHA),
            beta=entry.get("beta", self.BETA),
            sigma=entry.get("sigma", self.SIGMA),
            num_samples=entry.get("num_samples", self.NUM_SAMPLES),
            higher_is_better=entry.get("higher_is_better",
                                       self.HIGHER_IS_BETTER))

    def evaluate(self,
                 llm: Union[PyTorchLLM, AutoDeployLLM],
                 extra_acc_spec: Optional[str] = None,
                 extra_evaluator_kwargs: Optional[dict] = None,
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False,
                 is_integration_test: bool = False):
        assert self.EVALUATOR_CLS is not None

        if llm.args.speculative_config is None:
            spec_dec_algo = None
        elif isinstance(llm.args.speculative_config, DecodingBaseConfig):
            spec_dec_algo = llm.args.speculative_config.decoding_type
            if spec_dec_algo == 'AUTO':
                spec_dec_algo = 'NGram'
        else:
            raise ValueError(
                f"Not recognized speculative_config: {llm.args.speculative_config}."
            )
        is_integration_test = is_integration_test or os.getenv(
            'INTEGRATION_TEST', '0') == '1'

        if is_integration_test:
            logger.info(
                "Running in INTEGRATION_TEST mode: using only 1 sample and skipping accuracy verification"
            )
            hypothesis_testing_params = HypothesisTestingParams(
                ref_accuracy=0 if self.HIGHER_IS_BETTER else math.inf,
                num_samples=1,
                metric_name=self.METRIC_NAME,
                higher_is_better=self.HIGHER_IS_BETTER)
        else:
            hypothesis_testing_params = self.get_hypothesis_testing_params(
                dtype=llm.args.dtype,
                quant_algo=llm.args.quant_config.quant_algo,
                kv_cache_quant_algo=llm.args.quant_config.kv_cache_quant_algo,
                spec_dec_algo=spec_dec_algo,
                extra_acc_spec=extra_acc_spec)

        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=self.MAX_OUTPUT_LEN,
                truncate_prompt_tokens=self.MAX_INPUT_LEN)
        else:
            if sampling_params.max_tokens is None:
                sampling_params.max_tokens = self.MAX_OUTPUT_LEN
            if sampling_params.truncate_prompt_tokens is None:
                sampling_params.truncate_prompt_tokens = self.MAX_INPUT_LEN

        evaluator_kwargs = {}
        if self.EVALUATOR_KWARGS is not None:
            evaluator_kwargs.update(self.EVALUATOR_KWARGS)
        if extra_evaluator_kwargs is not None:
            evaluator_kwargs.update(extra_evaluator_kwargs)
        evaluator = self.EVALUATOR_CLS(
            num_samples=hypothesis_testing_params.num_samples,
            **evaluator_kwargs)
        evaluate_kwargs = {}
        if hasattr(self, 'EVALUATE_KWARGS'):
            evaluate_kwargs.update(self.EVALUATE_KWARGS)
        score = evaluator.evaluate(llm, sampling_params, streaming,
                                   **evaluate_kwargs)

        logger.info(
            f"Hypothesis testing report:\n{hypothesis_testing_params.report(score)}"
        )
        hypothesis_testing_params.assert_passing(score)
        return score


class VoxPopuli(AccuracyTask):
    """ASR accuracy task on the facebook/voxpopuli dataset, scored by WER (lower is better)."""

    DATASET = "voxpopuli"
    DATASET_DIR = f"{llm_models_root()}/datasets/facebook/voxpopuli"
    METRIC_NAME = "WER"
    HIGHER_IS_BETTER = False

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50.0
    NUM_SAMPLES = 32

    MAX_INPUT_LEN = 8192
    MAX_OUTPUT_LEN = 128
    MAX_BATCH_SIZE = 64

    EVALUATOR_CLS = AudioASREvaluator
    EVALUATOR_KWARGS = {
        "dataset_path": DATASET_DIR,
        "split": "test",
        "text_column": "normalized_text",
    }


class VideoMME(AccuracyTask):
    """Multiple-choice video QA accuracy task on the local Video-MME short shard."""

    DATASET = "videomme"
    DATASET_DIR = f"{llm_models_root()}/datasets/lmms-lab__Video-MME-short-v1"

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50.0
    NUM_SAMPLES = 300

    MAX_BATCH_SIZE = 128
    MAX_INPUT_LEN = 32768
    # The prompt asks for one option letter; keep a compact cap with room for
    # short extra text.
    MAX_OUTPUT_LEN = 32

    EVALUATOR_CLS = VideoMMEEvaluator
    EVALUATOR_KWARGS = {
        "dataset_path": DATASET_DIR,
        "random_seed": 0,
        "num_frames": 8,
    }


class CnnDailymail(AccuracyTask):
    DATASET = "cnn_dailymail"
    DATASET_DIR = f"{llm_models_root()}/datasets/ccdv/cnn_dailymail"
    ROUGE_DIR = f"{llm_models_root()}/rouge"

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 11.06
    NUM_SAMPLES = 512

    MAX_BATCH_SIZE = 128
    MAX_INPUT_LEN = 924
    MAX_OUTPUT_LEN = 100

    EVALUATOR_CLS = tensorrt_llm.evaluate.CnnDailymail
    EVALUATOR_KWARGS = dict(dataset_path=DATASET_DIR,
                            random_seed=0,
                            rouge_path=ROUGE_DIR)


class Humaneval(AccuracyTask):
    DATASET = "humaneval"
    DATASET_DIR = f"{llm_models_root()}/datasets/openai_humaneval"
    ROUGE_DIR = f"{llm_models_root()}/rouge"

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 15.08
    NUM_SAMPLES = 164  # Full sample

    MAX_BATCH_SIZE = 16
    MAX_INPUT_LEN = 924
    MAX_OUTPUT_LEN = 100


class ZeroScrolls(AccuracyTask):
    DATASET = "zero_scrolls"
    DATASET_DIR = f"{llm_models_root()}/datasets/tau/zero_scrolls"
    ROUGE_DIR = f"{llm_models_root()}/rouge"

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 6.97
    NUM_SAMPLES = 80  # Full sample

    MAX_BATCH_SIZE = 16
    MAX_INPUT_LEN = 24576
    MAX_OUTPUT_LEN = 8192


class SlimPajama6B(AccuracyTask):
    DATASET = "SlimPajama-6B"
    DATASET_DIR = f"{llm_models_root()}/datasets/SlimPajama-6B"
    HIGHER_IS_BETTER = False
    ROUGE_DIR = f"{llm_models_root()}/rouge"

    ALPHA = 0.01
    BETA = 0.2
    SIGMA = 4.48
    NUM_SAMPLES = 86  # Full sample with length >= 10000

    MAX_BATCH_SIZE = 1
    MAX_INPUT_LEN = 16 * 1024
    MIN_INPUT_LEN = 10000
    MAX_OUTPUT_LEN = 1


class MMLU(AccuracyTask):
    DATASET = "mmlu"
    DATASET_DIR = f"{llm_models_root()}/datasets/mmlu"

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50
    NUM_SAMPLES = 4096

    MAX_BATCH_SIZE = 128
    MAX_INPUT_LEN = 4094
    MAX_OUTPUT_LEN = 2

    EVALUATOR_CLS = tensorrt_llm.evaluate.MMLU
    EVALUATOR_KWARGS = dict(dataset_path=DATASET_DIR, random_seed=0)


class GSM8K(AccuracyTask):
    DATASET = "gsm8k"
    DATASET_DIR = f"{llm_models_root()}/datasets/openai/gsm8k"

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50
    NUM_SAMPLES = 1319  # Full sample

    MAX_INPUT_LEN = 4096
    MAX_OUTPUT_LEN = 256

    EVALUATOR_CLS = tensorrt_llm.evaluate.GSM8K
    EVALUATOR_KWARGS = dict(dataset_path=DATASET_DIR, random_seed=0)

    EVALUATE_KWARGS = dict(scores_filter=None)


class GPQADiamond(AccuracyTask):
    DATASET = "gpqa_diamond"
    DATASET_DIR = f"{llm_models_root()}/datasets/gpqa"

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50
    NUM_SAMPLES = 198  # Full sample

    MAX_INPUT_LEN = 4096
    MAX_OUTPUT_LEN = 32768

    EVALUATOR_CLS = tensorrt_llm.evaluate.GPQADiamond
    EVALUATOR_KWARGS = dict(dataset_path=DATASET_DIR, random_seed=0)


class JsonModeEval(AccuracyTask):
    DATASET = "json_mode_eval"
    DATASET_DIR = f"{llm_models_root()}/datasets/NousResearch/json-mode-eval"

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50
    NUM_SAMPLES = 100  # Full sample

    MAX_INPUT_LEN = 1024
    MAX_OUTPUT_LEN = 512

    EVALUATOR_CLS = tensorrt_llm.evaluate.JsonModeEval
    EVALUATOR_KWARGS = dict(dataset_path=DATASET_DIR,
                            random_seed=0,
                            apply_chat_template=True)


class MMMU(AccuracyTask):
    DATASET = "mmmu"
    DATASET_DIR = f"{llm_models_root()}/datasets/MMMU"

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50
    NUM_SAMPLES = 900

    MAX_BATCH_SIZE = 128
    MAX_INPUT_LEN = 8192
    MAX_OUTPUT_LEN = 512

    EVALUATOR_CLS = tensorrt_llm.evaluate.MMMU
    EVALUATOR_KWARGS = dict(dataset_path=DATASET_DIR,
                            random_seed=0,
                            is_multimodal=True,
                            apply_chat_template=True)

    EVALUATE_KWARGS = dict(model_type=None, is_force_single_image=False)


class PassKeyRetrieval64k(AccuracyTask):
    DATASET = "passkey_retrieval_64k"
    LEVEL = 3

    # Threshold is set equal to reference accuracy
    ALPHA = 0.5
    BETA = 0.2
    SIGMA = 0
    NUM_SAMPLES = 20

    MAX_BATCH_SIZE = 1
    MAX_INPUT_LEN = 64 * 1024
    MAX_OUTPUT_LEN = 50


class PassKeyRetrieval128k(AccuracyTask):
    DATASET = "passkey_retrieval_128k"
    LEVEL = 4

    # Threshold is set equal to reference accuracy
    ALPHA = 0.5
    BETA = 0.2
    SIGMA = 0
    NUM_SAMPLES = 20

    MAX_BATCH_SIZE = 1
    MAX_INPUT_LEN = 128 * 1024
    MAX_OUTPUT_LEN = 50


class LongBenchV2(AccuracyTask):
    DATASET = "longbench_v2"
    DATASET_DIR = f"{llm_models_root()}/zai-org/LongBench-v2"

    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50.0
    NUM_SAMPLES = 215

    MAX_BATCH_SIZE = 32
    MAX_INPUT_LEN = 1280000
    MAX_OUTPUT_LEN = 32000

    EVALUATOR_CLS = tensorrt_llm.evaluate.LongBenchV2
    EVALUATOR_KWARGS = dict(
        dataset_path=DATASET_DIR,
        length="medium",
        max_input_length=120000,
        apply_chat_template=True,
        random_seed=0,
    )


class LongBenchV1(AccuracyTask):
    DATASET = "longbench_v1"
    # Keep the dataset local like other accuracy tasks (avoid HF hub traffic).
    # Expected to be populated in CI image / test environment.
    DATASET_DIR = f"{llm_models_root()}/datasets/Xnhyacinth/LongBench"

    # NOTE: LongBench v1 is driven by lm-evaluation-harness task configs.
    # We intentionally do not pin dataset_path here (it can be resolved by lm-eval
    # via HF Hub or local cache).
    ALPHA = 0.05
    BETA = 0.2
    SIGMA = 50.0

    # Full sample
    NUM_SAMPLES = 4750

    # These are used by AccuracyTask to construct SamplingParams defaults.
    # LongBench v1 tasks provide per-task gen_kwargs, so these are mainly a safe fallback.
    MAX_BATCH_SIZE = 256
    MAX_INPUT_LEN = 128000
    MAX_OUTPUT_LEN = 1024

    EVALUATOR_CLS = tensorrt_llm.evaluate.LongBenchV1
    EVALUATOR_KWARGS = dict(dataset_path=DATASET_DIR,
                            random_seed=0,
                            apply_chat_template=True)


class LlmapiAccuracyTestHarness:
    # Model
    MODEL_NAME = None
    MODEL_PATH = None

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup_class(cls):
        original_level = logger.level
        logger.set_level("info")
        yield
        logger.set_level(original_level)

    @pytest.fixture(autouse=True)
    def _cleanup_cuda_between_tests(self):
        # Force Python GC + CUDA cache release after each test method.
        # The LLM context manager's __exit__ schedules destruction of CUDA
        # resources (streams, graph captures, KV cache pools), but objects in
        # reference cycles aren't reclaimed until the next GC cycle. Without
        # this teardown, a leftover CUDA stream/graph from a previous test can
        # land in the next test's allocations and corrupt them, producing
        # cross-test IMA reports that look like the current test crashed.
        yield
        gc.collect()
        torch.cuda.empty_cache()


def get_accuracy_task(dataset_name: str):
    try:
        task_class = globals()[dataset_name]
        if issubclass(task_class, AccuracyTask):
            return task_class
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}.")
    except KeyError:
        raise ValueError(f"Not registered dataset: {dataset_name}.")
