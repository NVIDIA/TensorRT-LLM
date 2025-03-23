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
import math
import os
from typing import Dict, Optional

import pytest
import yaml

try:
    from tensorrt_llm.builder import BuildConfig
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo
except ImportError:
    BuildConfig = None
    QuantConfig = None
    QuantAlgo = None

from ..common import venv_check_call, venv_mpi_check_call
from ..conftest import llm_models_root, skip_pre_ada, skip_pre_hopper
from ..trt_test_alternative import check_call, exists, makedirs
from .accuracy_core import compute_threshold


class AccuracyTestHarness:
    REFERENCE_DIR = f"{os.path.dirname(__file__)}/references"

    # Dataset
    DATASET = None
    DATASET_DIR = f"{llm_models_root()}/datasets"
    ROUGE_DIR = f"{llm_models_root()}/rouge"

    # Model
    MODEL_NAME = None
    MODEL_PATH = None
    MODEL_FORMAT = "HF"
    EXAMPLE_FOLDER = None

    # Hypothesis testing parameters
    ALPHA = None
    BETA = None
    SIGMA = None
    NUM_SAMPLES = None

    # Input and output sizes
    MAX_INPUT_LEN = None
    MAX_OUTPUT_LEN = None
    MAX_BATCH_SIZE = None

    @pytest.fixture(autouse=True)
    @classmethod
    def setup_class(cls, request):
        with open(f"{cls.REFERENCE_DIR}/{cls.DATASET}.yaml") as f:
            cls.reference = yaml.safe_load(f)[cls.MODEL_NAME]

        cls.llm_venv = request.getfixturevalue("llm_venv")
        cls.llm_root = request.getfixturevalue("llm_root")

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
        threshold, theta = compute_threshold(num_samples,
                                             accuracy,
                                             sigma=sigma,
                                             alpha=alpha,
                                             beta=beta)
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

    def is_pre_quantized(self):
        for quant_config_file in [
                "hf_quant_config.json", "quant_config.json",
                "quantize_config.json"
        ]:
            if exists(f"{self.MODEL_PATH}/{quant_config_file}"):
                return True
        return False

    def convert(self,
                dtype: str = 'auto',
                quant_algo: Optional[str] = None,
                kv_cache_quant_algo: Optional[str] = None,
                tp_size: int = 1,
                pp_size: int = 1,
                cp_size: int = 1,
                extra_convert_args: Optional[list] = None):
        print("Converting model to TensorRT-LLM checkpoint...")

        quant_config = QuantConfig(quant_algo, kv_cache_quant_algo)
        if not self.is_pre_quantized(
        ) and quant_config._requires_modelopt_quantization:
            script = "../quantization/quantize.py"
        else:
            script = "convert_checkpoint.py"

        convert_cmd = [
            f"{self.example_dir}/{script}",
            f"--output_dir={self.ckpt_dir}",
            f"--dtype={dtype}",
        ]

        if self.MODEL_FORMAT == "NEMO":
            convert_cmd.append(f"--nemo_ckpt_path={self.MODEL_PATH}")
        else:
            convert_cmd.append(f"--model_dir={self.MODEL_PATH}")

        if tp_size > 1:
            convert_cmd.append(f"--tp_size={tp_size}")
        if pp_size > 1:
            convert_cmd.append(f"--pp_size={pp_size}")
        if cp_size > 1:
            convert_cmd.append(f"--cp_size={cp_size}")

        if not self.is_pre_quantized(
        ) and quant_config._requires_modelopt_quantization:
            convert_cmd.append(
                f"--qformat={quant_config._get_modelopt_qformat()}")
            if (kv_cache_dtype :=
                    quant_config._get_modelopt_kv_cache_dtype()) is not None:
                convert_cmd.append(f"--kv_cache_dtype={kv_cache_dtype}")
        else:
            if quant_algo == QuantAlgo.NVFP4:
                convert_cmd.append("--use_nvfp4")
            elif quant_algo == QuantAlgo.FP8:
                convert_cmd.append("--use_fp8")
            elif quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN:
                convert_cmd.append("--use_fp8_rowwise")
            elif quant_config._use_plugin_sq:
                convert_cmd.append("--smoothquant=0.5")
                if "PER_TOKEN" in quant_algo:
                    convert_cmd.append("--per_token")
                if "PER_CHANNEL" in quant_algo:
                    convert_cmd.append("--per_channel")
            elif quant_algo == QuantAlgo.W8A16:
                convert_cmd.extend(
                    ["--use_weight_only", "--weight_only_precision=int8"])
            elif quant_algo == QuantAlgo.W4A16:
                convert_cmd.extend(
                    ["--use_weight_only", "--weight_only_precision=int4"])
            elif quant_algo == QuantAlgo.W8A16_GPTQ:
                convert_cmd.extend([
                    "--use_weight_only", "--weight_only_precision=int8_gptq",
                    "--per_group", "--group_size=64"
                ])
            elif quant_algo == QuantAlgo.W4A16_GPTQ:
                convert_cmd.extend([
                    "--use_weight_only", "--weight_only_precision=int4_gptq",
                    "--per_group"
                ])

            if kv_cache_quant_algo == QuantAlgo.INT8:
                convert_cmd.append("--int8_kv_cache")
            elif kv_cache_quant_algo == QuantAlgo.FP8:
                convert_cmd.append("--fp8_kv_cache")

        if quant_config._requires_calibration:
            convert_cmd.append(
                f"--calib_dataset={self.DATASET_DIR}/cnn_dailymail")

        if extra_convert_args:
            convert_cmd.extend(extra_convert_args)

        venv_check_call(self.llm_venv, convert_cmd)

    def build(self,
              tp_size: int = 1,
              pp_size: int = 1,
              cp_size: int = 1,
              extra_build_args: Optional[list] = None):
        print("Building engines...")
        max_seq_len = self.MAX_INPUT_LEN + self.MAX_OUTPUT_LEN
        max_num_tokens = max(BuildConfig.max_num_tokens, max_seq_len)
        build_cmd = [
            "trtllm-build",
            f"--checkpoint_dir={self.ckpt_dir}",
            f"--output_dir={self.engine_dir}",
            f"--max_batch_size={self.MAX_BATCH_SIZE}",
            f"--max_input_len={self.MAX_INPUT_LEN}",
            f"--max_seq_len={max_seq_len}",
            f"--max_num_tokens={max_num_tokens}",
            f"--workers={tp_size * pp_size * cp_size}",
        ]
        if extra_build_args:
            build_cmd.extend(extra_build_args)
        check_call(" ".join(build_cmd), shell=True, env=self.llm_venv._new_env)

    def evaluate(self,
                 dtype: str = 'auto',
                 quant_algo: Optional[str] = None,
                 kv_cache_quant_algo: Optional[str] = None,
                 spec_dec_algo: Optional[str] = None,
                 extra_acc_spec: Optional[str] = None,
                 tp_size: int = 1,
                 pp_size: int = 1,
                 cp_size: int = 1,
                 extra_eval_args: Optional[list] = None,
                 env: Optional[Dict[str, str]] = None):
        print("Running evaluation...")
        eval_cmd = [
            f"{self.example_dir}/../summarize.py",
            f"--engine_dir={self.engine_dir}",
            f"--dataset_dir={self.DATASET_DIR}",
            f"--rouge_dir={self.ROUGE_DIR}", "--test_trt_llm",
            "--random_seed=0", "--check_accuracy"
        ]
        if self.MODEL_FORMAT == "NEMO":
            eval_cmd.extend([
                f"--vocab_file={self.ckpt_dir}/tokenizer.model",
                "--no_add_special_tokens"
            ])
        else:
            eval_cmd.append(f"--tokenizer_dir={self.MODEL_PATH}")

        num_samples, threshold = self.get_num_samples_and_threshold(
            dtype=dtype,
            quant_algo=quant_algo,
            kv_cache_quant_algo=kv_cache_quant_algo,
            spec_dec_algo=spec_dec_algo,
            extra_acc_spec=extra_acc_spec)

        if num_samples < self.MAX_BATCH_SIZE:
            max_ite = 1
            batch_size = num_samples
        else:
            max_ite = math.ceil(num_samples / self.MAX_BATCH_SIZE)
            batch_size = self.MAX_BATCH_SIZE
        eval_cmd.extend([
            f"--batch_size={batch_size}", f"--max_ite={max_ite}",
            f"--tensorrt_llm_rouge1_threshold={threshold}"
        ])

        if extra_eval_args:
            eval_cmd.extend(extra_eval_args)

        world_size = tp_size * pp_size * cp_size
        if world_size == 1:
            venv_check_call(self.llm_venv, eval_cmd, env=env)
        else:
            venv_mpi_check_call(
                self.llm_venv,
                ["mpirun", "-n",
                 str(world_size), "--allow-run-as-root"], eval_cmd)

    def run(self,
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
            extra_eval_args: Optional[list] = None,
            env: Optional[Dict[str, str]] = None):
        self.install_requirements()
        self.convert(dtype=dtype,
                     quant_algo=quant_algo,
                     kv_cache_quant_algo=kv_cache_quant_algo,
                     tp_size=tp_size,
                     pp_size=pp_size,
                     cp_size=cp_size,
                     extra_convert_args=extra_convert_args)
        self.build(tp_size=tp_size,
                   pp_size=pp_size,
                   cp_size=cp_size,
                   extra_build_args=extra_build_args)
        self.evaluate(dtype=dtype,
                      quant_algo=quant_algo,
                      kv_cache_quant_algo=kv_cache_quant_algo,
                      spec_dec_algo=spec_dec_algo,
                      extra_acc_spec=extra_acc_spec,
                      tp_size=tp_size,
                      pp_size=pp_size,
                      cp_size=cp_size,
                      extra_eval_args=extra_eval_args,
                      env=env)


class CnnDailymailTestHarness(AccuracyTestHarness):
    DATASET = "cnn_dailymail"
    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 11.06
    NUM_SAMPLES = 512

    MAX_BATCH_SIZE = 128
    MAX_INPUT_LEN = 924
    MAX_OUTPUT_LEN = 100


class HumanevalTestHarness(AccuracyTestHarness):
    DATASET = "humaneval"
    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 15.08
    NUM_SAMPLES = 164  # Full sample

    MAX_BATCH_SIZE = 16
    MAX_INPUT_LEN = 924
    MAX_OUTPUT_LEN = 100


class ZeroScrollsTestHarness(AccuracyTestHarness):
    DATASET = "zero_scrolls"
    ALPHA = 0.002
    BETA = 0.2
    SIGMA = 6.97
    NUM_SAMPLES = 80  # Full sample

    MAX_BATCH_SIZE = 16
    MAX_INPUT_LEN = 24576
    MAX_OUTPUT_LEN = 8192


class TestGpt2(CnnDailymailTestHarness):
    MODEL_NAME = "gpt2"
    MODEL_PATH = f"{llm_models_root()}/gpt2"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        # float16
        self.run(dtype='auto')

    def test_gemm_plugin(self):
        self.run(extra_build_args=["--gemm_plugin=auto"])

    def test_attention_ootb(self):
        self.run(extra_build_args=[
            "--gpt_attention_plugin=disable", "--context_fmha=disable",
            "--paged_kv_cache=disable", "--remove_input_padding=disable"
        ])

    def test_context_fmha_disabled(self):
        self.run(extra_build_args=["--context_fmha=disable"])

    def test_context_fmha_fp32_acc(self):
        self.run(extra_eval_args=["--enable_context_fmha_fp32_acc"])

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo)

    def test_int8_kv_cache(self):
        self.run(kv_cache_quant_algo=QuantAlgo.INT8)

    @pytest.mark.parametrize("per_token,per_channel", [(False, False),
                                                       (True, True)],
                             ids=["", "per_token-per_channel"])
    def test_smooth_quant(self, per_token: bool, per_channel: bool):
        if per_token:
            if per_channel:
                quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN
            else:
                quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN
        else:
            if per_channel:
                quant_algo = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN
            else:
                quant_algo = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN
        self.run(quant_algo=quant_algo)

    def test_beam_search(self):
        self.run(extra_acc_spec="beam_width=4",
                 extra_build_args=["--max_beam_width=4"],
                 extra_eval_args=["--num_beams=4", "--length_penalty=2.0"])

    def test_beam_search_large(self, mocker):
        mocker.patch.object(self.__class__, "MAX_BATCH_SIZE", 8)
        self.run(extra_acc_spec="beam_width=256",
                 extra_build_args=["--max_beam_width=256"],
                 extra_eval_args=["--num_beams=256"])

    def test_weight_streaming_ootb(self):
        self.run(
            extra_build_args=[
                "--gpt_attention_plugin=disable", "--weight_streaming",
                "--remove_input_padding=disable", "--paged_kv_cache=disable"
            ],
            extra_eval_args=["--gpu_weights_percent=0.5", "--use_py_session"])

    def test_weight_streaming_plugin(self):
        self.run(extra_build_args=["--weight_streaming"],
                 extra_eval_args=["--gpu_weights_percent=0"])

    def test_cuda_graph(self):
        self.run(extra_eval_args=["--cuda_graph_mode"])


class TestGpt2Medium(CnnDailymailTestHarness):
    MODEL_NAME = "gpt2-medium"
    MODEL_PATH = f"{llm_models_root()}/gpt2-medium"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8)

    @skip_pre_ada
    def test_fp8_lm_head(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 extra_convert_args=["--quantize_lm_head"])


class TestSantacoder(HumanevalTestHarness):
    MODEL_NAME = "bigcode/santacoder"
    MODEL_PATH = f"{llm_models_root()}/santacoder"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        # float16
        self.run(dtype='auto', extra_eval_args=["--eval_task=code_completion"])


class TestStarcoder2_3B(HumanevalTestHarness):
    MODEL_NAME = "bigcode/starcoder2-3b"
    MODEL_PATH = f"{llm_models_root()}/starcoder2-3b"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        self.run(dtype='auto', extra_eval_args=["--eval_task=code_completion"])


class TestStarcoder2_15B(HumanevalTestHarness):
    MODEL_NAME = "bigcode/starcoder2-15b"
    MODEL_PATH = f"{llm_models_root()}/starcoder2-model"
    EXAMPLE_FOLDER = "gpt"

    def test_smooth_quant_ootb(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL,
                 extra_eval_args=["--eval_task=code_completion"])


class TestGptNext(CnnDailymailTestHarness):
    MODEL_NAME = "gpt-next"
    MODEL_PATH = f"{llm_models_root()}/gpt-next/megatron_converted_843m_tp1_pp1.nemo"
    MODEL_FORMAT = "NEMO"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        # bfloat16
        self.run(dtype='auto')


class TestMinitron4BBase(HumanevalTestHarness):
    MODEL_NAME = "nvidia/Minitron-4B-Base"
    MODEL_PATH = f"{llm_models_root()}/nemotron/Minitron-4B-Base"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        self.run(dtype='auto', extra_eval_args=["--eval_task=code_completion"])

    @skip_pre_ada
    def test_fp8(self, mocker):
        # Accuracy regression when using large batch size
        mocker.patch.object(self.__class__, "MAX_BATCH_SIZE", 1)
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 extra_eval_args=["--eval_task=code_completion"])


class TestGptJ6B(CnnDailymailTestHarness):
    MODEL_NAME = "EleutherAI/gpt-j-6b"
    MODEL_PATH = f"{llm_models_root()}/gpt-j-6b"
    EXAMPLE_FOLDER = "gptj"

    def test_auto_dtype(self):
        # float16
        self.run(dtype='auto')

    def test_float32(self):
        self.run(dtype='float32')

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5166352")
    def test_cyclic_kv_cache(self):
        self.run(extra_acc_spec="max_attention_window_size=900",
                 extra_eval_args=["--max_attention_window_size=900"])

    @pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5166352")
    def test_cyclic_kv_cache_beam_search(self):
        self.run(extra_acc_spec="max_attention_window_size=900;beam_width=4",
                 extra_build_args=["--max_beam_width=4"],
                 extra_eval_args=[
                     "--max_attention_window_size=900", "--num_beams=4"
                 ])


@pytest.mark.skip_less_device_memory(50000)
class TestPhi2(CnnDailymailTestHarness):
    MODEL_NAME = "microsoft/phi-2"
    MODEL_PATH = f"{llm_models_root()}/phi-2"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


@pytest.mark.skip_less_device_memory(50000)
class TestPhi3Mini4kInstruct(CnnDailymailTestHarness):
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-3/Phi-3-mini-4k-instruct"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


@pytest.mark.skip_less_device_memory(50000)
class TestPhi3Mini128kInstruct(CnnDailymailTestHarness):
    MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-3/Phi-3-mini-128k-instruct"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


# Long sequence length test:
# Model FP16 7B + 32K tokens in KV cache = 14 * 1024 MB + 32K * 0.5 MB = 30720 MB + scratch memory
@pytest.mark.skip_less_device_memory(40000)
class TestLongAlpaca7B(ZeroScrollsTestHarness):
    MODEL_NAME = "Yukang/LongAlpaca-7B"
    MODEL_PATH = f"{llm_models_root()}/LongAlpaca-7B"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(extra_eval_args=[
            "--eval_task=summarize_long", "--max_input_length=24576",
            "--output_len=8192"
        ])

    def test_multiblock_aggressive(self):
        # MMHA + aggressive Multi_block_mode (export TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG=1)
        self.run(extra_build_args=["--gemm_plugin=auto"],
                 extra_eval_args=[
                     "--eval_task=summarize_long", "--max_input_length=24576",
                     "--output_len=8192"
                 ],
                 env={
                     "TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG": "1",
                     "TRTLLM_MMHA_BLOCKS_PER_SEQUENCE": "32"
                 })


class TestMamba130M(CnnDailymailTestHarness):
    MODEL_NAME = "state-spaces/mamba-130m-hf"
    MODEL_PATH = f"{llm_models_root()}/mamba/mamba-130m-hf"
    EXAMPLE_FOLDER = "mamba"

    def test_auto_dtype(self):
        self.run(dtype='auto')


class TestVicuna7B(CnnDailymailTestHarness):
    MODEL_NAME = "lmsys/vicuna-7b-v1.3"
    MODEL_PATH = f"{llm_models_root()}/vicuna-7b-v1.3"
    EXAMPLE_FOLDER = "llama"
    MEDUSA_MODEL_NAME = "FasterDecoding/medusa-vicuna-7b-v1.3"
    MEDUSA_MODEL_PATH = f"{llm_models_root()}/medusa-vicuna-7b-v1.3"
    EAGLE_MODEL_NAME = "yuhuili/EAGLE-Vicuna-7B-v1.3"
    EAGLE_MODEL_PATH = f"{llm_models_root()}/EAGLE-Vicuna-7B-v1.3"

    def test_lookahead(self, mocker):
        mocker.patch.object(self.__class__, "MAX_BATCH_SIZE", 8)

        self.run(spec_dec_algo="lookahead",
                 extra_build_args=[
                     "--max_draft_len=83",
                     "--speculative_decoding_mode=lookahead_decoding"
                 ],
                 extra_eval_args=["--lookahead_config=[7,7,7]"])

    @pytest.mark.parametrize("cuda_graph", [False, True],
                             ids=["", "cuda_graph"])
    def test_medusa(self, cuda_graph, mocker):
        mocker.patch.object(self.__class__, "EXAMPLE_FOLDER", "medusa")
        mocker.patch.object(self.__class__, "MAX_BATCH_SIZE", 8)

        extra_eval_args = [
            "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]"
        ]
        if cuda_graph:
            extra_eval_args.append("--cuda_graph_mode")

        self.run(dtype="float16",
                 spec_dec_algo="medusa",
                 extra_convert_args=[
                     f"--medusa_model_dir={self.MEDUSA_MODEL_PATH}",
                     "--num_medusa_heads=4"
                 ],
                 extra_build_args=["--speculative_decoding_mode=medusa"],
                 extra_eval_args=extra_eval_args)

    @pytest.mark.parametrize("cuda_graph,chunked_context,typical_acceptance",
                             [(False, False, False), (True, False, False),
                              (True, True, False), (True, False, True)],
                             ids=[
                                 "", "cuda_graph", "cuda_graph-chunked_context",
                                 "cuda_graph-typical_acceptance"
                             ])
    def test_eagle(self, cuda_graph, chunked_context, typical_acceptance,
                   mocker):
        mocker.patch.object(self.__class__, "EXAMPLE_FOLDER", "eagle")
        mocker.patch.object(self.__class__, "MAX_BATCH_SIZE", 8)

        extra_eval_args = [
            "--eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]"
        ]
        if cuda_graph:
            extra_eval_args.append("--cuda_graph_mode")
        if chunked_context:
            extra_eval_args.append("--enable_chunked_context")
        if typical_acceptance:
            extra_eval_args.extend(
                ["--eagle_posterior_threshold=0.09", "--temperature=0.7"])

        self.run(spec_dec_algo="eagle",
                 extra_convert_args=[
                     f"--eagle_model_dir={self.EAGLE_MODEL_PATH}",
                     "--max_draft_len=63", "--num_eagle_layers=4",
                     "--max_non_leaves_per_layer=10"
                 ],
                 extra_build_args=[
                     "--speculative_decoding_mode=eagle", "--max_draft_len=63"
                 ],
                 extra_eval_args=extra_eval_args)


class TestLlama7B(CnnDailymailTestHarness):
    MODEL_NAME = "llama-7b-hf"
    MODEL_PATH = f"{llm_models_root()}/llama-models/llama-7b-hf"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_beam_search(self):
        self.run(extra_acc_spec="beam_width=5",
                 extra_build_args=["--max_beam_width=5"],
                 extra_eval_args=["--num_beams=5"])

    def test_int4_gptq(self):
        self.run(
            quant_algo=QuantAlgo.W4A16_GPTQ,
            extra_convert_args=[
                f"--quant_ckpt_path={llm_models_root()}/int4-quantized-gptq-awq/llama-7b-4bit-gs128.safetensors"
            ])

    def test_streamingllm(self):
        self.run(extra_acc_spec="streamingllm",
                 extra_build_args=["--streamingllm=enable"],
                 extra_eval_args=[
                     "--max_attention_window_size=2048", "--sink_token_length=4"
                 ])

    def test_manage_weights(self):
        self.run(extra_build_args=["--fast_build"])


class TestLlama2_7B(CnnDailymailTestHarness):
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    MODEL_PATH = f"{llm_models_root()}/llama-models-v2/llama-v2-7b-hf"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_smooth_quant(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_ada
    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize("tp_size,pp_size,cp_size", [(2, 1, 1), (1, 2, 1),
                                                         (1, 1, 2)],
                             ids=["tp2", "pp2", "cp2"])
    def test_fp8_2gpus(self, tp_size, pp_size, cp_size):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 tp_size=tp_size,
                 pp_size=pp_size,
                 cp_size=cp_size)

    @skip_pre_ada
    @pytest.mark.skip_less_device(4)
    def test_tp2cp2(self):
        self.run(tp_size=2, cp_size=2)

    @skip_pre_ada
    def test_fp8_gemm_plugin(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 extra_build_args=["--gemm_plugin=fp8"])

    @skip_pre_ada
    def test_fp8_gemm_swiglu_plugin(self):
        self.run(
            quant_algo=QuantAlgo.FP8,
            kv_cache_quant_algo=QuantAlgo.FP8,
            extra_build_args=["--gemm_plugin=fp8", "--gemm_swiglu_plugin=fp8"])

    @skip_pre_ada
    def test_fp8_low_latency_gemm_plugin(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 extra_build_args=["--low_latency_gemm_plugin=fp8"])

    @pytest.mark.skip_less_device(2)
    def test_smooth_quant_ootb_tp2(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL, tp_size=2)

    @pytest.mark.skip_less_device(2)
    def test_int4_awq_tp2(self):
        self.run(quant_algo=QuantAlgo.W4A16_AWQ, tp_size=2)

    @pytest.mark.skip_less_device(2)
    def test_int4_awq_pre_quantized_tp2(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/llama-models-v2/Llama-2-7B-AWQ")
        self.run(quant_algo=QuantAlgo.W4A16_AWQ, tp_size=2)

    @pytest.mark.skip_less_device(2)
    def test_int4_gptq_pre_quantized_tp2(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/llama-models-v2/Llama-2-7B-GPTQ")
        self.run(quant_algo=QuantAlgo.W4A16_GPTQ, tp_size=2)

    def test_weight_sparsity(self):
        self.run(extra_build_args=["--weight_sparsity"])


class TestTinyLlama1_1BChat(CnnDailymailTestHarness):
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MODEL_PATH = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo)

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only_int8_kv_cache(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, kv_cache_quant_algo=QuantAlgo.INT8)

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only_manage_weights(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, extra_build_args=["--fast_build"])

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @pytest.mark.skip_less_device(4)
    def test_pp4(self):
        # Test num_hidden_layers (22) undivisible by pp_size (4)
        self.run(extra_acc_spec="pp_size=4", pp_size=4)


class TestLlama3_8BInstruct(CnnDailymailTestHarness):
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-models-v3/llama-v3-8b-instruct-hf"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    def test_int8_gptq(self):
        self.run(
            quant_algo=QuantAlgo.W8A16_GPTQ,
            extra_convert_args=[
                f"--quant_ckpt_path={llm_models_root()}/int8-quantized-gptq/llama-3-8b-8bit-gs64-gptq.safetensors"
            ])


class TestLlama3_1_8B(CnnDailymailTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_smooth_quant(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_ada
    def test_fp8_rowwise(self):
        self.run(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

    @skip_pre_ada
    def test_fp8_rowwise_meta_recipe(self):
        self.run(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
                 extra_acc_spec="meta_recipe",
                 extra_convert_args=["--use_meta_fp8_rowwise_recipe"])


class TestLlama3_1_8BInstruct(CnnDailymailTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @skip_pre_ada
    def test_fp8_pre_quantized(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8")
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)


class TestLlama3_2_1B(CnnDailymailTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_smooth_quant(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)

    def test_smooth_quant_ootb(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL)

    def test_smooth_quant_ootb_manage_weights(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL,
                 extra_build_args=["--fast_build"])

    def test_int4_awq(self):
        self.run(quant_algo=QuantAlgo.W4A16_AWQ)

    def test_int4_awq_int8_kv_cache(self):
        self.run(quant_algo=QuantAlgo.W4A16_AWQ,
                 kv_cache_quant_algo=QuantAlgo.INT8)

    def test_int4_awq_manage_weights(self):
        self.run(quant_algo=QuantAlgo.W4A16_AWQ,
                 extra_build_args=["--fast_build"])

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_ada
    @pytest.mark.skip_less_device(2)
    @pytest.mark.parametrize(
        "fp8_context_fmha", [False, True],
        ids=["disable_fp8_context_fmha", "enable_fp8_context_fmha"])
    @pytest.mark.parametrize(
        "reduce_fusion", [False, True],
        ids=["disable_reduce_fusion", "enable_reduce_fusion"])
    def test_fp8_tp2(self, fp8_context_fmha: bool, reduce_fusion: bool):
        if fp8_context_fmha:
            extra_build_args = [
                "--use_fp8_context_fmha=enable",
                "--use_paged_context_fmha=enable"
            ]
        else:
            extra_build_args = [
                "--use_fp8_context_fmha=disable",
                "--use_paged_context_fmha=disable"
            ]

        if reduce_fusion:
            extra_build_args.append("--reduce_fusion=enable")
        else:
            extra_build_args.append("--reduce_fusion=disable")

        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 tp_size=2,
                 extra_build_args=extra_build_args)

    @skip_pre_ada
    @pytest.mark.skip_less_device(2)
    def test_fp8_pp2(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 pp_size=2)

    @skip_pre_ada
    def test_fp8_rowwise(self):
        self.run(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

    @skip_pre_ada
    def test_fp8_rowwise_meta_recipe(self):
        self.run(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
                 extra_acc_spec="meta_recipe",
                 extra_convert_args=["--use_meta_fp8_rowwise_recipe"])

    @pytest.mark.parametrize("max_gpu_percent", [0.1, 1.0])
    def test_weight_streaming(self, max_gpu_percent: float):
        self.run(extra_build_args=["--weight_streaming"],
                 extra_eval_args=["--gpu_weights_percent=0"])

        for gpu_percent in [0.1, 0.5, 0.9, 1]:
            if gpu_percent > max_gpu_percent:
                break
            self.evaluate(
                extra_eval_args=[f"--gpu_weights_percent={gpu_percent}"])


class TestMixtral8x7B(CnnDailymailTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-v0.1"
    EXAMPLE_FOLDER = "llama"

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(80000)
    def test_auto_dtype(self):
        self.run(dtype='auto', tp_size=2)

    @skip_pre_ada
    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(80000)
    def test_fp8_tp2(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 tp_size=2)

    @skip_pre_ada
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(40000)
    def test_fp8_tp2pp2(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 tp_size=2,
                 pp_size=2)


class TestGemma2B(CnnDailymailTestHarness):
    MODEL_NAME = "google/gemma-2b"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-2b"
    EXAMPLE_FOLDER = "gemma"

    def test_auto_dtype(self):
        self.run(dtype='auto', extra_convert_args=["--ckpt-type=hf"])

    @pytest.mark.parametrize("precision", ["int8"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, extra_convert_args=["--ckpt-type=hf"])

    def test_smooth_quant(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
                 extra_convert_args=[
                     "--ckpt-type=hf",
                     f"--tokenizer_dir={self.MODEL_PATH}/tokenizer.model"
                 ])

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    def test_int4_awq(self):
        self.run(quant_algo=QuantAlgo.W4A16_AWQ)


@pytest.mark.skip_less_device_memory(40000)
class TestGemma7B(CnnDailymailTestHarness):
    MODEL_NAME = "google/gemma-7b"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-7b"
    EXAMPLE_FOLDER = "gemma"

    def test_auto_dtype(self):
        self.run(dtype='auto', extra_convert_args=["--ckpt-type=hf"])

    @pytest.mark.parametrize("precision", ["int8"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, extra_convert_args=["--ckpt-type=hf"])

    @pytest.mark.skip_less_device_memory(50000)
    def test_smooth_quant(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN,
                 extra_convert_args=[
                     "--ckpt-type=hf",
                     f"--tokenizer_dir={self.MODEL_PATH}/tokenizer.model"
                 ])

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    def test_int4_awq(self):
        self.run(quant_algo=QuantAlgo.W4A16_AWQ)


@pytest.mark.skip_less_device_memory(40000)
class TestGemma2_9BIt(CnnDailymailTestHarness):
    MODEL_NAME = "google/gemma-2-9b-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-2-9b-it"
    EXAMPLE_FOLDER = "gemma"

    def test_auto_dtype(self):
        self.run(dtype='auto', extra_convert_args=["--ckpt-type=hf"])

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, extra_convert_args=["--ckpt-type=hf"])

    @skip_pre_hopper
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 extra_convert_args=["--device_map=sequential"])
