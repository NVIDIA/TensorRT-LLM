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
import pytest

from tensorrt_llm.llmapi import EagleDecodingConfig, MedusaDecodingConfig
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import (get_sm_version, llm_models_root, parametrize_with_ids,
                        skip_no_nvls, skip_post_blackwell, skip_pre_ada,
                        skip_pre_hopper)
from .accuracy_core import (MMLU, CliFlowAccuracyTestHarness, CnnDailymail,
                            Humaneval, PassKeyRetrieval64k, ZeroScrolls)

# skip trt flow cases on post-Blackwell-Ultra
if get_sm_version() >= 103:
    pytest.skip(
        "TRT workflow tests are not supported on post Blackwell-Ultra architecture",
        allow_module_level=True)


class TestStarcoder2_15B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "bigcode/starcoder2-15b"
    MODEL_PATH = f"{llm_models_root()}/starcoder2-model"
    EXAMPLE_FOLDER = "models/core/gpt"


class TestGptNext(CliFlowAccuracyTestHarness):
    MODEL_NAME = "gpt-next"
    MODEL_PATH = f"{llm_models_root()}/gpt-next/megatron_converted_843m_tp1_pp1.nemo"
    MODEL_FORMAT = "NEMO"
    EXAMPLE_FOLDER = "models/core/gpt"

    def test_auto_dtype(self):
        # bfloat16
        self.run(dtype='auto')


class TestMinitron4BBase(CliFlowAccuracyTestHarness):
    MODEL_NAME = "nvidia/Minitron-4B-Base"
    MODEL_PATH = f"{llm_models_root()}/nemotron/Minitron-4B-Base"
    EXAMPLE_FOLDER = "models/core/gpt"

    def test_auto_dtype(self):
        self.run(tasks=[Humaneval(self.MODEL_NAME)], dtype='auto')

    @skip_pre_ada
    def test_fp8(self, mocker):
        # Accuracy regression when using large batch size
        mocker.patch.object(Humaneval, "MAX_BATCH_SIZE", 1)
        self.run(tasks=[Humaneval(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8)


class TestNemotronMini4BInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "nvidia/Nemotron-Mini-4B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/nemotron/Nemotron-Mini-4B-Instruct"
    EXAMPLE_FOLDER = "models/core/gpt"

    @skip_pre_ada
    def test_fp8_prequantized(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/nemotron/nemotron-mini-4b-instruct_vfp8-fp8-bf16-export"
        )
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)


# Long sequence length test:
# Model FP16 7B + 32K tokens in KV cache = 14 * 1024 MB + 32K * 0.5 MB = 30720 MB + scratch memory
@pytest.mark.skip_less_device_memory(40000)
class TestLongAlpaca7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Yukang/LongAlpaca-7B"
    MODEL_PATH = f"{llm_models_root()}/LongAlpaca-7B"
    EXAMPLE_FOLDER = "models/core/llama"

    def test_auto_dtype(self):
        self.run(tasks=[ZeroScrolls(self.MODEL_NAME)])

    def test_multiblock_aggressive(self):
        # MMHA + aggressive Multi_block_mode (export TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG=1)
        self.run(tasks=[ZeroScrolls(self.MODEL_NAME)],
                 extra_build_args=["--gemm_plugin=auto"],
                 env={
                     "TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG": "1",
                     "TRTLLM_MMHA_BLOCKS_PER_SEQUENCE": "32"
                 })


class TestMamba130M(CliFlowAccuracyTestHarness):
    MODEL_NAME = "state-spaces/mamba-130m-hf"
    MODEL_PATH = f"{llm_models_root()}/mamba/mamba-130m-hf"
    EXAMPLE_FOLDER = "models/core/mamba"

    def test_auto_dtype(self):
        self.run(dtype='auto')


class TestVicuna7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "lmsys/vicuna-7b-v1.3"
    MODEL_PATH = f"{llm_models_root()}/vicuna-7b-v1.3"
    EXAMPLE_FOLDER = "models/core/llama"
    MEDUSA_MODEL_NAME = "FasterDecoding/medusa-vicuna-7b-v1.3"
    MEDUSA_MODEL_PATH = f"{llm_models_root()}/medusa-vicuna-7b-v1.3"
    EAGLE_MODEL_NAME = "yuhuili/EAGLE-Vicuna-7B-v1.3"
    EAGLE_MODEL_PATH = f"{llm_models_root()}/EAGLE-Vicuna-7B-v1.3"

    @skip_post_blackwell
    @parametrize_with_ids("cuda_graph", [False, True])
    def test_medusa(self, cuda_graph, mocker):
        mocker.patch.object(self.__class__, "EXAMPLE_FOLDER", "medusa")
        mocker.patch.object(CnnDailymail, "MAX_BATCH_SIZE", 8)

        extra_summarize_args = [
            "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]"
        ]
        if cuda_graph:
            extra_summarize_args.append("--cuda_graph_mode")

        self.run(dtype="float16",
                 spec_dec_algo=MedusaDecodingConfig.
                 model_fields["decoding_type"].default,
                 extra_convert_args=[
                     f"--medusa_model_dir={self.MEDUSA_MODEL_PATH}",
                     "--num_medusa_heads=4"
                 ],
                 extra_build_args=["--speculative_decoding_mode=medusa"],
                 extra_summarize_args=extra_summarize_args)

    @skip_post_blackwell
    @parametrize_with_ids("cuda_graph,chunked_context,typical_acceptance",
                          [(False, False, False), (True, False, False),
                           (True, True, False), (True, False, True)])
    def test_eagle(self, cuda_graph, chunked_context, typical_acceptance,
                   mocker):
        mocker.patch.object(self.__class__, "EXAMPLE_FOLDER", "eagle")
        mocker.patch.object(CnnDailymail, "MAX_BATCH_SIZE", 8)

        extra_summarize_args = [
            "--eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]"
        ]
        if cuda_graph:
            extra_summarize_args.append("--cuda_graph_mode")
        if chunked_context:
            extra_summarize_args.append("--enable_chunked_context")
        if typical_acceptance:
            extra_summarize_args.extend(
                ["--eagle_posterior_threshold=0.09", "--temperature=0.7"])

        self.run(spec_dec_algo=EagleDecodingConfig.
                 model_fields["decoding_type"].default,
                 extra_convert_args=[
                     f"--eagle_model_dir={self.EAGLE_MODEL_PATH}",
                     "--max_draft_len=63", "--num_eagle_layers=4",
                     "--max_non_leaves_per_layer=10"
                 ],
                 extra_build_args=[
                     "--speculative_decoding_mode=eagle", "--max_draft_len=63"
                 ],
                 extra_summarize_args=extra_summarize_args)


class TestTinyLlama1_1BChat(CliFlowAccuracyTestHarness):
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MODEL_PATH = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
    EXAMPLE_FOLDER = "models/core/llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @skip_post_blackwell
    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo)

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @pytest.mark.skip_less_device(4)
    def test_pp4(self):
        # Test num_hidden_layers (22) undivisible by pp_size (4)
        self.run(extra_acc_spec="pp_size=4", pp_size=4)


class TestLlama3_1_8B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Meta-Llama-3.1-8B"
    EXAMPLE_FOLDER = "models/core/llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_ada
    @skip_post_blackwell
    def test_fp8_rowwise(self):
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

    @skip_pre_ada
    @skip_post_blackwell
    def test_fp8_rowwise_meta_recipe(self):
        self.run(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
                 extra_acc_spec="meta_recipe",
                 extra_convert_args=["--use_meta_fp8_rowwise_recipe"])

    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize(
        "gemm_allreduce", [False, pytest.param(True, marks=skip_no_nvls)],
        ids=["disable_gemm_allreduce_plugin", "enable_gemm_allreduce_plugin"])
    def test_tp4(self, gemm_allreduce: bool):
        extra_build_args = None
        if gemm_allreduce:
            extra_build_args = ["--gemm_allreduce_plugin=bfloat16"]
        self.run(
            tasks=[PassKeyRetrieval64k(self.MODEL_NAME),
                   MMLU(self.MODEL_NAME)],
            tp_size=4,
            extra_build_args=extra_build_args)

    @skip_pre_hopper
    @skip_post_blackwell
    @pytest.mark.skip_less_device(4)
    @pytest.mark.parametrize(
        "gemm_allreduce", [False, pytest.param(True, marks=skip_no_nvls)],
        ids=["disable_gemm_allreduce_plugin", "enable_gemm_allreduce_plugin"])
    def test_fp8_rowwise_tp4(self, gemm_allreduce: bool):
        extra_build_args = None
        if gemm_allreduce:
            extra_build_args = ["--gemm_allreduce_plugin=bfloat16"]
        self.run(
            tasks=[PassKeyRetrieval64k(self.MODEL_NAME),
                   MMLU(self.MODEL_NAME)],
            quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
            tp_size=4,
            extra_build_args=extra_build_args)


class TestLlama3_1_8BInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    EXAMPLE_FOLDER = "models/core/llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @skip_pre_hopper
    def test_fp8_prequantized(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8")
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_hopper
    @skip_post_blackwell
    def test_medusa_fp8_prequantized(self, mocker):
        # nvidia/Llama-3.1-8B-Medusa-FP8
        mocker.patch.object(self.__class__, "MODEL_PATH",
                            f"{llm_models_root()}/llama3.1-medusa-8b-hf_v0.1")
        mocker.patch.object(self.__class__, "EXAMPLE_FOLDER", "medusa")
        mocker.patch.object(CnnDailymail, "MAX_BATCH_SIZE", 8)

        extra_summarize_args = [
            "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [1, 6], [0, 7, 0]]"
        ]
        self.run(dtype="float16",
                 spec_dec_algo=MedusaDecodingConfig.
                 model_fields["decoding_type"].default,
                 extra_build_args=["--speculative_decoding_mode=medusa"],
                 extra_summarize_args=extra_summarize_args)


class TestGemma2B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "google/gemma-2b"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-2b"
    EXAMPLE_FOLDER = "models/core/gemma"

    def test_auto_dtype(self):
        self.run(dtype='auto', extra_convert_args=["--ckpt-type=hf"])

    @pytest.mark.parametrize("precision", ["int8"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, extra_convert_args=["--ckpt-type=hf"])

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)


@pytest.mark.skip_less_device_memory(40000)
class TestGemma7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "google/gemma-7b"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-7b"
    EXAMPLE_FOLDER = "models/core/gemma"

    def test_auto_dtype(self):
        self.run(dtype='auto', extra_convert_args=["--ckpt-type=hf"])

    @pytest.mark.parametrize("precision", ["int8"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, extra_convert_args=["--ckpt-type=hf"])

    @skip_pre_ada
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)
