# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pytest
import scipy
import yaml

import tensorrt_llm.evaluate
from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..common import venv_check_call, venv_mpi_check_call
from ..conftest import llm_models_root
from ..trt_test_alternative import check_call, exists


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
        report = f"""===========================================================
= ACCURACY HYPOTHESIS TESTING
===========================================================
Alpha (Type I:  False Positive): {self.alpha:.3f}
Beta  (Type II: False Negative): {self.beta:.3f}
Sigma (Standard deviation): {self.sigma:.3f}
#Samples: {self.num_samples}
Higher is better: {self.higher_is_better}
Theta (Minimum detectable effect): {self.theta:.3f}
Reference accuracy: {self.ref_accuracy:.3f}
Threshold: {self.threshold:.3f}
==========================================================="""
        if accuracy is not None:
            report = f"""{report}
Evaluated accuracy: {accuracy:.3f}
==========================================================="""
        return report

    def assert_passing(self, accuracy: float) -> None:
        compare_op = ">=" if self.higher_is_better else "<="
        err_msg = f"Reference accuracy is {self.ref_accuracy:.3f}, threshold is {self.threshold:.3f}. Expected accuracy {compare_op} threshold, but got {accuracy:.3f}. Please see hypothesis testing report:\n{self.report(accuracy)}"
        if self.higher_is_better:
            assert accuracy >= self.threshold, err_msg
        else:
            assert accuracy <= self.threshold, err_msg


class AccuracyTask:
    REFERENCE_DIR = f"{os.path.dirname(__file__)}/references"

    # Dataset
    DATASET = None
    DATASET_DIR = None
    HIGHER_IS_BETTER = True

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
                entry = {"accuracy": 0}
            else:
                raise ValueError(f"Not registered specs: {acc_specs}.")

        return HypothesisTestingParams(
            ref_accuracy=entry.get("accuracy"),
            alpha=entry.get("alpha", self.ALPHA),
            beta=entry.get("beta", self.BETA),
            sigma=entry.get("sigma", self.SIGMA),
            num_samples=entry.get("num_samples", self.NUM_SAMPLES),
            higher_is_better=entry.get("higher_is_better",
                                       self.HIGHER_IS_BETTER))

    def evaluate(self,
                 llm: Union[LLM, PyTorchLLM, AutoDeployLLM],
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
            hypothesis_testing_params = HypothesisTestingParams(ref_accuracy=0,
                                                                num_samples=1)
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
        accuracy = evaluator.evaluate(llm, sampling_params, streaming,
                                      **evaluate_kwargs)

        logger.info(
            f"Hypothesis testing report:\n{hypothesis_testing_params.report(accuracy)}"
        )
        hypothesis_testing_params.assert_passing(accuracy)


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


class CliFlowAccuracyTestHarness:
    # Model
    MODEL_NAME = None
    MODEL_PATH = None
    MODEL_FORMAT = "HF"
    EXAMPLE_FOLDER = None

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def setup_class(cls, request):
        cls.llm_venv = request.getfixturevalue("llm_venv")
        cls.llm_root = request.getfixturevalue("llm_root")

    @pytest.fixture(autouse=True, scope="function")
    def setup_method(self):
        with tempfile.TemporaryDirectory(
                prefix=self.MODEL_NAME.replace("/", "-"),
                dir=self.llm_venv.get_working_directory()) as workspace:
            self.ckpt_dir = f"{workspace}/cmodels"
            self.engine_dir = f"{workspace}/engines"
            yield

    def install_requirements(self):
        requirements = f"{self.llm_root}/examples/{self.EXAMPLE_FOLDER}/requirements.txt"
        if exists(requirements):
            self.llm_venv.run_cmd(
                ["-m", "pip", "install", "-r", requirements],
                env={
                    "CMAKE_POLICY_VERSION_MINIMUM":
                    "3.5"  # https://github.com/google/sentencepiece/issues/1111
                })

    def initialize_case(self,
                        tasks: Optional[List[AccuracyTask]] = None,
                        dtype: str = 'auto',
                        quant_algo: Optional[str] = None,
                        kv_cache_quant_algo: Optional[str] = None,
                        spec_dec_algo: Optional[str] = None,
                        extra_acc_spec: Optional[str] = None,
                        tp_size: int = 1,
                        pp_size: int = 1,
                        cp_size: int = 1,
                        extra_convert_args: Optional[list] = None,
                        extra_build_args: Optional[list] = None,
                        extra_summarize_args: Optional[list] = None,
                        extra_mmlu_args: Optional[list] = None,
                        extra_eval_long_context_args: Optional[list] = None,
                        env: Optional[Dict[str, str]] = None):
        self.tasks = [CnnDailymail(self.MODEL_NAME)] if tasks is None else tasks
        self.dtype = dtype
        self.quant_algo = quant_algo
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.spec_dec_algo = spec_dec_algo
        self.extra_acc_spec = extra_acc_spec
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.cp_size = cp_size
        self.extra_convert_args = extra_convert_args
        self.extra_build_args = extra_build_args
        self.extra_summarize_args = extra_summarize_args
        self.extra_mmlu_args = extra_mmlu_args
        self.extra_eval_long_context_args = extra_eval_long_context_args
        self.env = env

    def convert(self):
        logger.info("Converting model to TensorRT LLM checkpoint...")

        is_prequantized = False
        for quant_config_file in [
                "hf_quant_config.json", "quant_config.json",
                "quantize_config.json"
        ]:
            if exists(f"{self.MODEL_PATH}/{quant_config_file}"):
                is_prequantized = True
                break
        if not is_prequantized and exists(f"{self.MODEL_PATH}/config.json"):
            with open(f"{self.MODEL_PATH}/config.json") as f:
                hf_config = json.load(f)
            if "quantization_config" in hf_config:
                is_prequantized = True

        quant_config = QuantConfig(self.quant_algo, self.kv_cache_quant_algo)
        if not is_prequantized and quant_config._requires_modelopt_quantization:
            script = f"{self.llm_root}/examples/quantization/quantize.py"
        else:
            script = f"{self.llm_root}/examples/{self.EXAMPLE_FOLDER}/convert_checkpoint.py"

        convert_cmd = [
            script,
            f"--output_dir={self.ckpt_dir}",
            f"--dtype={self.dtype}",
        ]

        if "nemotron_nas" in self.EXAMPLE_FOLDER:
            convert_cmd.append("--trust_remote_code")

        if self.MODEL_FORMAT == "NEMO":
            convert_cmd.append(f"--nemo_ckpt_path={self.MODEL_PATH}")
        else:
            convert_cmd.append(f"--model_dir={self.MODEL_PATH}")

        if self.tp_size > 1:
            convert_cmd.append(f"--tp_size={self.tp_size}")
        if self.pp_size > 1:
            convert_cmd.append(f"--pp_size={self.pp_size}")
        if self.cp_size > 1:
            convert_cmd.append(f"--cp_size={self.cp_size}")

        if not is_prequantized and quant_config._requires_modelopt_quantization:
            if self.quant_algo == QuantAlgo.MIXED_PRECISION:
                assert self.extra_convert_args is not None
                assert any(
                    x.startswith("--autoq_format")
                    for x in self.extra_convert_args)
            else:
                convert_cmd.append(
                    f"--qformat={quant_config._get_modelopt_qformat()}")
            if (kv_cache_dtype :=
                    quant_config._get_modelopt_kv_cache_dtype()) is not None:
                convert_cmd.append(f"--kv_cache_dtype={kv_cache_dtype}")
        else:
            if self.quant_algo == QuantAlgo.NVFP4:
                convert_cmd.append("--use_nvfp4")
            elif self.quant_algo == QuantAlgo.FP8:
                if self.EXAMPLE_FOLDER != "models/core/gpt":  # --use_fp8 flag is not needed for gpt.
                    convert_cmd.append("--use_fp8")
            elif self.quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN:
                convert_cmd.append("--use_fp8_rowwise")
            elif quant_config._use_plugin_sq:
                convert_cmd.append("--smoothquant=0.5")
                if "PER_TOKEN" in self.quant_algo:
                    convert_cmd.append("--per_token")
                if "PER_CHANNEL" in self.quant_algo:
                    convert_cmd.append("--per_channel")
            elif self.quant_algo == QuantAlgo.W8A16:
                convert_cmd.extend(
                    ["--use_weight_only", "--weight_only_precision=int8"])
            elif self.quant_algo == QuantAlgo.W4A16:
                convert_cmd.extend(
                    ["--use_weight_only", "--weight_only_precision=int4"])
            elif self.quant_algo == QuantAlgo.W8A16_GPTQ:
                convert_cmd.extend([
                    "--use_weight_only", "--weight_only_precision=int8_gptq",
                    "--per_group", "--group_size=64"
                ])
            elif self.quant_algo == QuantAlgo.W4A16_GPTQ:
                convert_cmd.extend([
                    "--use_weight_only", "--weight_only_precision=int4_gptq",
                    "--per_group"
                ])

            if self.kv_cache_quant_algo == QuantAlgo.INT8:
                convert_cmd.append("--int8_kv_cache")
            elif self.kv_cache_quant_algo == QuantAlgo.FP8:
                if self.EXAMPLE_FOLDER != "models/core/gpt":  # --fp8_kv_cache flag is not needed for gpt.
                    convert_cmd.append("--fp8_kv_cache")

        if quant_config._requires_calibration:
            convert_cmd.append(
                f"--calib_dataset={llm_models_root()}/datasets/cnn_dailymail")

        if self.extra_convert_args:
            convert_cmd.extend(self.extra_convert_args)

        venv_check_call(self.llm_venv, convert_cmd)

    def build(self):
        logger.info("Building engines...")
        max_batch_size = max(task.MAX_BATCH_SIZE for task in self.tasks)
        max_input_len = max(task.MAX_INPUT_LEN for task in self.tasks)
        max_seq_len = max(task.MAX_INPUT_LEN + task.MAX_OUTPUT_LEN
                          for task in self.tasks)
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={self.ckpt_dir}",
            f"--output_dir={self.engine_dir}",
            f"--max_batch_size={max_batch_size}",
            f"--max_input_len={max_input_len}",
            f"--max_seq_len={max_seq_len}",
            f"--workers={self.tp_size * self.pp_size * self.cp_size}",
        ]
        if self.extra_build_args:
            build_cmd.extend(self.extra_build_args)
        check_call(" ".join(build_cmd), shell=True, env=self.llm_venv._new_env)

    def summarize(self, task: AccuracyTask):
        logger.info("Running summarize...")
        summarize_cmd = [
            f"{self.llm_root}/examples/summarize.py",
            f"--engine_dir={self.engine_dir}",
            f"--hf_model_dir={self.MODEL_PATH}",
            f"--max_input_length={task.MAX_INPUT_LEN}",
            f"--output_len={task.MAX_OUTPUT_LEN}",
            f"--dataset_dir={task.DATASET_DIR}",
            f"--rouge_dir={task.ROUGE_DIR}", "--test_trt_llm",
            "--random_seed=0", "--check_accuracy"
        ]
        if self.MODEL_FORMAT == "NEMO":
            summarize_cmd.extend([
                f"--vocab_file={self.ckpt_dir}/tokenizer.model",
                "--no_add_special_tokens"
            ])

        hypothesis_testing_params = task.get_hypothesis_testing_params(
            dtype=self.dtype,
            quant_algo=self.quant_algo,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            spec_dec_algo=self.spec_dec_algo,
            extra_acc_spec=self.extra_acc_spec)
        logger.info(
            f"Hypothesis testing report:\n{hypothesis_testing_params.report()}")
        num_samples = hypothesis_testing_params.num_samples
        threshold = hypothesis_testing_params.threshold

        if num_samples < task.MAX_BATCH_SIZE:
            max_ite = 1
            batch_size = num_samples
        else:
            max_ite = math.ceil(num_samples / task.MAX_BATCH_SIZE)
            batch_size = task.MAX_BATCH_SIZE
        summarize_cmd.extend([
            f"--batch_size={batch_size}", f"--max_ite={max_ite}",
            f"--tensorrt_llm_rouge1_threshold={threshold}"
        ])

        if isinstance(task, Humaneval):
            summarize_cmd.append("--eval_task=code_completion")
        elif isinstance(task, ZeroScrolls):
            summarize_cmd.append("--eval_task=summarize_long")
        elif isinstance(task, SlimPajama6B):
            max_tokens_in_paged_kv_cache = int(
                batch_size * (task.MAX_INPUT_LEN + task.MAX_OUTPUT_LEN) * 1.1)
            summarize_cmd.extend([
                "--eval_task=eval_context_ppl",
                f"--min_input_length={task.MIN_INPUT_LEN}",
                f"--max_tokens_in_paged_kv_cache={max_tokens_in_paged_kv_cache}"
            ])

        if task.MAX_INPUT_LEN + task.MAX_OUTPUT_LEN > BuildConfig.max_num_tokens:
            summarize_cmd.append("--enable_chunked_context")

        if self.extra_summarize_args:
            summarize_cmd.extend(self.extra_summarize_args)

        world_size = self.tp_size * self.pp_size * self.cp_size
        if world_size == 1:
            venv_check_call(self.llm_venv, summarize_cmd, env=self.env)
        else:
            venv_mpi_check_call(
                self.llm_venv,
                ["mpirun", "-n",
                 str(world_size), "--allow-run-as-root"], summarize_cmd)

    def mmlu(self, task: AccuracyTask):
        logger.info("Running mmlu...")
        hypothesis_testing_params = task.get_hypothesis_testing_params(
            dtype=self.dtype,
            quant_algo=self.quant_algo,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            spec_dec_algo=self.spec_dec_algo,
            extra_acc_spec=self.extra_acc_spec)
        logger.info(
            f"Hypothesis testing report:\n{hypothesis_testing_params.report()}")
        num_samples = hypothesis_testing_params.num_samples
        threshold = hypothesis_testing_params.threshold

        mmlu_cmd = [
            "trtllm-eval",
            f"--model={self.engine_dir}",
            f"--tokenizer={self.MODEL_PATH}",
            "--backend=tensorrt",
        ]

        if self.extra_mmlu_args:
            mmlu_cmd.extend(self.extra_mmlu_args)

        mmlu_cmd.extend([
            "mmlu", f"--dataset_path={task.DATASET_DIR}",
            f"--num_samples={num_samples}", "--random_seed=0",
            "--check_accuracy", f"--accuracy_threshold={threshold}"
        ])

        check_call(" ".join(mmlu_cmd), shell=True, env=self.llm_venv._new_env)

    def eval_long_context(self, task: AccuracyTask):
        logger.info("Running construct_synthetic_dataset...")
        data_gen_cmd = [
            f"{self.llm_root}/examples/infinitebench/construct_synthetic_dataset.py",
            "--test_case=build_passkey", f"--test_level={task.LEVEL}"
        ]
        venv_check_call(self.llm_venv, data_gen_cmd)

        logger.info("Running eval_long_context...")
        eval_cmd = [
            f"{self.llm_root}/examples/eval_long_context.py", "--task=passkey",
            f"--engine_dir={self.engine_dir}",
            f"--tokenizer_dir={self.MODEL_PATH}",
            f"--max_input_length={task.MAX_INPUT_LEN}",
            "--enable_chunked_context"
        ]
        hypothesis_testing_params = task.get_hypothesis_testing_params(
            dtype=self.dtype,
            quant_algo=self.quant_algo,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            spec_dec_algo=self.spec_dec_algo,
            extra_acc_spec=self.extra_acc_spec)
        logger.info(
            f"Hypothesis testing report:\n{hypothesis_testing_params.report()}")
        num_samples = hypothesis_testing_params.num_samples
        threshold = hypothesis_testing_params.threshold

        batch_size = min(task.MAX_BATCH_SIZE, num_samples)
        eval_cmd.extend([
            f"--batch_size={batch_size}", f"--stop_idx={num_samples}",
            f"--tensorrt_llm_accuracy_threshold={threshold}"
        ])

        if self.extra_eval_long_context_args:
            eval_cmd.extend(self.extra_eval_long_context_args)

        world_size = self.tp_size * self.pp_size * self.cp_size
        if world_size == 1:
            venv_check_call(self.llm_venv, eval_cmd, env=self.env)
        else:
            venv_mpi_check_call(
                self.llm_venv,
                ["mpirun", "-n",
                 str(world_size), "--allow-run-as-root"], eval_cmd)

    def evaluate(self):
        for task in self.tasks:
            if isinstance(task,
                          (CnnDailymail, Humaneval, ZeroScrolls, SlimPajama6B)):
                self.summarize(task)
            elif isinstance(task, MMLU):
                self.mmlu(task)
            elif isinstance(task, (PassKeyRetrieval64k, PassKeyRetrieval128k)):
                self.eval_long_context(task)
            else:
                raise ValueError(f"Not registered dataset: {task.DATASET}.")

    def run(self,
            tasks: Optional[List[AccuracyTask]] = None,
            dtype: str = 'auto',
            quant_algo: Optional[str] = None,
            kv_cache_quant_algo: Optional[str] = None,
            spec_dec_algo: Optional[str] = None,
            extra_acc_spec: Optional[str] = None,
            tp_size: int = 1,
            pp_size: int = 1,
            cp_size: int = 1,
            extra_convert_args: Optional[list] = None,
            extra_build_args: Optional[list] = None,
            extra_summarize_args: Optional[list] = None,
            extra_eval_long_context_args: Optional[list] = None,
            env: Optional[Dict[str, str]] = None,
            timeout_manager=None):
        """
        Run all accuracy test phases with timeout management.
        If timeout_manager is provided, each phase will be wrapped to track and deduct remaining timeout.
        """
        # Use timeout_manager to manage timeout for each phase
        if timeout_manager is not None:
            with timeout_manager.timed_operation("install_requirements"):
                self.install_requirements()
            with timeout_manager.timed_operation("initialize_case"):
                self.initialize_case(
                    tasks=tasks,
                    dtype=dtype,
                    quant_algo=quant_algo,
                    kv_cache_quant_algo=kv_cache_quant_algo,
                    spec_dec_algo=spec_dec_algo,
                    extra_acc_spec=extra_acc_spec,
                    tp_size=tp_size,
                    pp_size=pp_size,
                    cp_size=cp_size,
                    extra_convert_args=extra_convert_args,
                    extra_build_args=extra_build_args,
                    extra_summarize_args=extra_summarize_args,
                    extra_eval_long_context_args=extra_eval_long_context_args,
                    env=env)
            with timeout_manager.timed_operation("convert"):
                self.convert()
            with timeout_manager.timed_operation("build"):
                self.build()
            with timeout_manager.timed_operation("evaluate"):
                self.evaluate()
        else:
            # fallback: no timeout management
            self.install_requirements()
            self.initialize_case(
                tasks=tasks,
                dtype=dtype,
                quant_algo=quant_algo,
                kv_cache_quant_algo=kv_cache_quant_algo,
                spec_dec_algo=spec_dec_algo,
                extra_acc_spec=extra_acc_spec,
                tp_size=tp_size,
                pp_size=pp_size,
                cp_size=cp_size,
                extra_convert_args=extra_convert_args,
                extra_build_args=extra_build_args,
                extra_summarize_args=extra_summarize_args,
                extra_eval_long_context_args=extra_eval_long_context_args,
                env=env)
            self.convert()
            self.build()
            self.evaluate()


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


def get_accuracy_task(dataset_name: str):
    try:
        task_class = globals()[dataset_name]
        if issubclass(task_class, AccuracyTask):
            return task_class
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}.")
    except KeyError:
        raise ValueError(f"Not registered dataset: {dataset_name}.")
