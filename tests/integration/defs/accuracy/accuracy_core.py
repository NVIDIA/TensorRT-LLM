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
from typing import Dict, List, Optional

import pytest
import scipy
import yaml

from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ..common import venv_check_call, venv_mpi_check_call
from ..conftest import llm_models_root
from ..trt_test_alternative import check_call, exists, makedirs


def compute_threshold(num_samples: int,
                      ref_accuracy: float,
                      sigma: float,
                      alpha: float = 0.05,
                      beta: float = 0.2,
                      higher_is_better: bool = True):
    scale = (2 * sigma**2 / num_samples)**0.5

    # Single-tail testing
    z_alpha = scipy.stats.norm.ppf(alpha)
    if higher_is_better:
        threshold = ref_accuracy + z_alpha * scale
    else:
        threshold = ref_accuracy - z_alpha * scale

    z_beta = scipy.stats.norm.ppf(beta)
    theta = -(z_alpha + z_beta) * scale
    return threshold, theta


class AccuracyTask:
    REFERENCE_DIR = f"{os.path.dirname(__file__)}/references"

    # Dataset
    DATASET = None
    DATASET_DIR = f"{llm_models_root()}/datasets"
    ROUGE_DIR = f"{llm_models_root()}/rouge"
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

    def __init__(self, model_name: str):
        with open(f"{self.REFERENCE_DIR}/{self.DATASET}.yaml") as f:
            self.reference = yaml.safe_load(f)[model_name]

    def get_num_samples_and_threshold(self, **acc_specs):
        """Get num_samples and threshold via accuracy specifications.

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

        accuracy = entry.get("accuracy")
        alpha = entry.get("alpha", self.ALPHA)
        beta = entry.get("beta", self.BETA)
        sigma = entry.get("sigma", self.SIGMA)
        num_samples = entry.get("num_samples", self.NUM_SAMPLES)
        higher_is_better = entry.get("higher_is_better", self.HIGHER_IS_BETTER)
        threshold, theta = compute_threshold(num_samples,
                                             accuracy,
                                             sigma=sigma,
                                             alpha=alpha,
                                             beta=beta,
                                             higher_is_better=higher_is_better)
        print("===========================================================\n"
              "= ACCURACY HYPOTHESIS TESTING\n"
              "===========================================================\n"
              f"Alpha (Type I:  False Positive): {alpha:.3f}\n"
              f"Beta  (Type II: False Negative): {beta:.3f}\n"
              f"Sigma (Standard deviation): {sigma:.3f}\n"
              f"#Samples: {num_samples}\n"
              f"Theta (Minimum detectable effect): {theta:.3f}\n"
              f"Reference accuracy: {accuracy:.3f}\n"
              f"Threshold: {threshold:.3f}\n"
              "===========================================================\n")
        return num_samples, threshold


class CnnDailymail(AccuracyTask):
    DATASET = "cnn_dailymail"

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 11.06
    NUM_SAMPLES = 512

    MAX_BATCH_SIZE = 128
    MAX_INPUT_LEN = 924
    MAX_OUTPUT_LEN = 100


class Humaneval(AccuracyTask):
    DATASET = "humaneval"

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 15.08
    NUM_SAMPLES = 164  # Full sample

    MAX_BATCH_SIZE = 16
    MAX_INPUT_LEN = 924
    MAX_OUTPUT_LEN = 100


class ZeroScrolls(AccuracyTask):
    DATASET = "zero_scrolls"

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 6.97
    NUM_SAMPLES = 80  # Full sample

    MAX_BATCH_SIZE = 16
    MAX_INPUT_LEN = 24576
    MAX_OUTPUT_LEN = 8192


class SlimPajama6B(AccuracyTask):
    DATASET = "SlimPajama-6B"
    HIGHER_IS_BETTER = False

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 4.48
    NUM_SAMPLES = 86  # Full sample with length >= 10000

    MAX_BATCH_SIZE = 1
    MAX_INPUT_LEN = 16 * 1024
    MIN_INPUT_LEN = 10000
    MAX_OUTPUT_LEN = 1


class Mmlu(AccuracyTask):
    DATASET = "mmlu"
    DATASET_DIR = f"{llm_models_root()}/datasets/mmlu"

    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 50
    NUM_SAMPLES = 4096

    MAX_BATCH_SIZE = 128
    MAX_INPUT_LEN = 4094
    MAX_OUTPUT_LEN = 2


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


class AccuracyTestHarness:
    # Model
    MODEL_NAME = None
    MODEL_PATH = None
    MODEL_FORMAT = "HF"
    EXAMPLE_FOLDER = None

    @pytest.fixture(autouse=True)
    @classmethod
    def setup_class(cls, request):
        cls.llm_venv = request.getfixturevalue("llm_venv")
        cls.llm_root = request.getfixturevalue("llm_root")

    @property
    def example_dir(self):
        return f"{self.llm_root}/examples/{self.EXAMPLE_FOLDER}"

    @property
    def ckpt_dir(self):
        ckpt_dir = f"{self.llm_venv.get_working_directory()}/cmodels/{self.MODEL_NAME}"
        if not exists(ckpt_dir):
            makedirs(ckpt_dir)
        return ckpt_dir

    @property
    def engine_dir(self):
        engine_dir = f"{self.llm_venv.get_working_directory()}/engines/{self.MODEL_NAME}"
        if not exists(engine_dir):
            makedirs(engine_dir)
        return engine_dir

    def install_requirements(self):
        requirements = f"{self.example_dir}/requirements.txt"
        if exists(requirements):
            self.llm_venv.run_cmd(["-m", "pip", "install", "-r", requirements])

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
        print("Converting model to TensorRT-LLM checkpoint...")

        is_pre_quantized = False
        for quant_config_file in [
                "hf_quant_config.json", "quant_config.json",
                "quantize_config.json"
        ]:
            if exists(f"{self.MODEL_PATH}/{quant_config_file}"):
                is_pre_quantized = True
                break
        if not is_pre_quantized and exists(f"{self.MODEL_PATH}/config.json"):
            with open(f"{self.MODEL_PATH}/config.json") as f:
                hf_config = json.load(f)
            if "quantization_config" in hf_config:
                is_pre_quantized = True

        quant_config = QuantConfig(self.quant_algo, self.kv_cache_quant_algo)
        if not is_pre_quantized and quant_config._requires_modelopt_quantization:
            script = "../quantization/quantize.py"
        else:
            script = "convert_checkpoint.py"

        convert_cmd = [
            f"{self.example_dir}/{script}",
            f"--output_dir={self.ckpt_dir}",
            f"--dtype={self.dtype}",
        ]

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

        if not is_pre_quantized and quant_config._requires_modelopt_quantization:
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
                if self.EXAMPLE_FOLDER != "gpt":  # --use_fp8 flag is not needed for gpt.
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
                if self.EXAMPLE_FOLDER != "gpt":  # --fp8_kv_cache flag is not needed for gpt.
                    convert_cmd.append("--fp8_kv_cache")

        if quant_config._requires_calibration:
            convert_cmd.append(
                f"--calib_dataset={llm_models_root()}/datasets/cnn_dailymail")

        if self.extra_convert_args:
            convert_cmd.extend(self.extra_convert_args)

        venv_check_call(self.llm_venv, convert_cmd)

    def build(self):
        print("Building engines...")
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
        print("Running summarize...")
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

        num_samples, threshold = task.get_num_samples_and_threshold(
            dtype=self.dtype,
            quant_algo=self.quant_algo,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            spec_dec_algo=self.spec_dec_algo,
            extra_acc_spec=self.extra_acc_spec)

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
        print("Running mmlu...")
        mmlu_cmd = [
            f"{self.llm_root}/examples/mmlu_llmapi.py",
            f"--engine_dir={self.engine_dir}",
            f"--hf_model_dir={self.MODEL_PATH}",
            f"--data_dir={task.DATASET_DIR}", "--backend=tensorrt",
            "--random_seed=0", "--check_accuracy"
        ]

        num_samples, threshold = task.get_num_samples_and_threshold(
            dtype=self.dtype,
            quant_algo=self.quant_algo,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            spec_dec_algo=self.spec_dec_algo,
            extra_acc_spec=self.extra_acc_spec)
        mmlu_cmd.extend([
            f"--num_samples={num_samples}", f"--accuracy_threshold={threshold}"
        ])

        if task.MAX_INPUT_LEN + task.MAX_OUTPUT_LEN > BuildConfig.max_num_tokens:
            mmlu_cmd.append("--enable_chunked_prefill")

        if self.extra_mmlu_args:
            mmlu_cmd.extend(self.extra_mmlu_args)

        venv_check_call(self.llm_venv, mmlu_cmd, env=self.env)

    def eval_long_context(self, task: AccuracyTask):
        print("Running construct_synthetic_dataset...")
        data_gen_cmd = [
            f"{self.llm_root}/examples/infinitebench/construct_synthetic_dataset.py",
            "--test_case=build_passkey", f"--test_level={task.LEVEL}"
        ]
        venv_check_call(self.llm_venv, data_gen_cmd)

        print("Running eval_long_context...")
        eval_cmd = [
            f"{self.llm_root}/examples/eval_long_context.py", "--task=passkey",
            f"--engine_dir={self.engine_dir}",
            f"--tokenizer_dir={self.MODEL_PATH}",
            f"--max_input_length={task.MAX_INPUT_LEN}",
            "--enable_chunked_context"
        ]
        num_samples, threshold = task.get_num_samples_and_threshold(
            dtype=self.dtype,
            quant_algo=self.quant_algo,
            kv_cache_quant_algo=self.kv_cache_quant_algo,
            spec_dec_algo=self.spec_dec_algo,
            extra_acc_spec=self.extra_acc_spec)

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
            elif isinstance(task, Mmlu):
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
            env: Optional[Dict[str, str]] = None):
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
