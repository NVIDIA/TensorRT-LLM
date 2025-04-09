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
import pytest

from tensorrt_llm.llmapi import (EagleDecodingConfig, LookaheadDecodingConfig,
                                 MedusaDecodingConfig)
from tensorrt_llm.quantization import QuantAlgo

from ..conftest import (llm_models_root, parametrize_with_ids, skip_no_nvls,
                        skip_pre_ada, skip_pre_blackwell, skip_pre_hopper)
from .accuracy_core import (MMLU, CliFlowAccuracyTestHarness, CnnDailymail,
                            Humaneval, PassKeyRetrieval64k,
                            PassKeyRetrieval128k, SlimPajama6B, ZeroScrolls)


class TestGpt2(CliFlowAccuracyTestHarness):
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
        self.run(extra_summarize_args=["--enable_context_fmha_fp32_acc"])

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo)

    def test_int8_kv_cache(self):
        self.run(kv_cache_quant_algo=QuantAlgo.INT8)

    @parametrize_with_ids("per_token,per_channel", [(False, False),
                                                    (True, True)])
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
                 extra_summarize_args=["--num_beams=4", "--length_penalty=2.0"])

    def test_beam_search_large(self, mocker):
        mocker.patch.object(CnnDailymail, "MAX_BATCH_SIZE", 8)
        self.run(extra_acc_spec="beam_width=256",
                 extra_build_args=["--max_beam_width=256"],
                 extra_summarize_args=["--num_beams=256"])

    def test_weight_streaming_ootb(self):
        self.run(extra_build_args=[
            "--gpt_attention_plugin=disable", "--weight_streaming",
            "--remove_input_padding=disable", "--paged_kv_cache=disable"
        ],
                 extra_summarize_args=[
                     "--gpu_weights_percent=0.5", "--use_py_session"
                 ])

    def test_weight_streaming_plugin(self):
        self.run(extra_build_args=["--weight_streaming"],
                 extra_summarize_args=["--gpu_weights_percent=0"])

    def test_cuda_graph(self):
        self.run(extra_summarize_args=["--cuda_graph_mode"])


class TestGpt2Medium(CliFlowAccuracyTestHarness):
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


class TestSantacoder(CliFlowAccuracyTestHarness):
    MODEL_NAME = "bigcode/santacoder"
    MODEL_PATH = f"{llm_models_root()}/santacoder"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        # float16
        self.run(tasks=[Humaneval(self.MODEL_NAME)], dtype='auto')


class TestStarcoder2_3B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "bigcode/starcoder2-3b"
    MODEL_PATH = f"{llm_models_root()}/starcoder2-3b"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        self.run(tasks=[Humaneval(self.MODEL_NAME)], dtype='auto')


class TestStarcoder2_15B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "bigcode/starcoder2-15b"
    MODEL_PATH = f"{llm_models_root()}/starcoder2-model"
    EXAMPLE_FOLDER = "gpt"

    def test_smooth_quant_ootb(self):
        self.run(tasks=[Humaneval(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL)


class TestGptNext(CliFlowAccuracyTestHarness):
    MODEL_NAME = "gpt-next"
    MODEL_PATH = f"{llm_models_root()}/gpt-next/megatron_converted_843m_tp1_pp1.nemo"
    MODEL_FORMAT = "NEMO"
    EXAMPLE_FOLDER = "gpt"

    def test_auto_dtype(self):
        # bfloat16
        self.run(dtype='auto')


class TestMinitron4BBase(CliFlowAccuracyTestHarness):
    MODEL_NAME = "nvidia/Minitron-4B-Base"
    MODEL_PATH = f"{llm_models_root()}/nemotron/Minitron-4B-Base"
    EXAMPLE_FOLDER = "gpt"

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
    EXAMPLE_FOLDER = "gpt"

    @skip_pre_ada
    def test_fp8_prequantized(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/nemotron/nemotron-mini-4b-instruct_vfp8-fp8-bf16-export"
        )
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)


class TestPhi2(CliFlowAccuracyTestHarness):
    MODEL_NAME = "microsoft/phi-2"
    MODEL_PATH = f"{llm_models_root()}/phi-2"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @pytest.mark.skip_less_device(2)
    def test_tp2(self):
        self.run(tp_size=2)


class TestPhi3Mini4kInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-3/Phi-3-mini-4k-instruct"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


class TestPhi3Mini128kInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-3/Phi-3-mini-128k-instruct"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


class TestPhi3Small8kInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-3-small-8k-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-3/Phi-3-small-8k-instruct"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


class TestPhi3Small128kInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-3-small-128k-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-3/Phi-3-small-128k-instruct"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


class TestPhi3_5MiniInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
    MODEL_PATH = f"{llm_models_root()}/Phi-3.5/Phi-3.5-mini-instruct"
    EXAMPLE_FOLDER = "phi"

    def test_auto_dtype(self):
        self.run(dtype='auto')


# Long sequence length test:
# Model FP16 7B + 32K tokens in KV cache = 14 * 1024 MB + 32K * 0.5 MB = 30720 MB + scratch memory
@pytest.mark.skip_less_device_memory(40000)
class TestLongAlpaca7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Yukang/LongAlpaca-7B"
    MODEL_PATH = f"{llm_models_root()}/LongAlpaca-7B"
    EXAMPLE_FOLDER = "llama"

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
    EXAMPLE_FOLDER = "mamba"

    def test_auto_dtype(self):
        self.run(dtype='auto')


class TestVicuna7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "lmsys/vicuna-7b-v1.3"
    MODEL_PATH = f"{llm_models_root()}/vicuna-7b-v1.3"
    EXAMPLE_FOLDER = "llama"
    MEDUSA_MODEL_NAME = "FasterDecoding/medusa-vicuna-7b-v1.3"
    MEDUSA_MODEL_PATH = f"{llm_models_root()}/medusa-vicuna-7b-v1.3"
    EAGLE_MODEL_NAME = "yuhuili/EAGLE-Vicuna-7B-v1.3"
    EAGLE_MODEL_PATH = f"{llm_models_root()}/EAGLE-Vicuna-7B-v1.3"

    def test_lookahead(self, mocker):
        mocker.patch.object(CnnDailymail, "MAX_BATCH_SIZE", 8)

        self.run(spec_dec_algo=LookaheadDecodingConfig.decoding_type,
                 extra_build_args=[
                     "--max_draft_len=83",
                     "--speculative_decoding_mode=lookahead_decoding"
                 ],
                 extra_summarize_args=["--lookahead_config=[7,7,7]"])

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
                 spec_dec_algo=MedusaDecodingConfig.decoding_type,
                 extra_convert_args=[
                     f"--medusa_model_dir={self.MEDUSA_MODEL_PATH}",
                     "--num_medusa_heads=4"
                 ],
                 extra_build_args=["--speculative_decoding_mode=medusa"],
                 extra_summarize_args=extra_summarize_args)

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

        self.run(spec_dec_algo=EagleDecodingConfig.decoding_type,
                 extra_convert_args=[
                     f"--eagle_model_dir={self.EAGLE_MODEL_PATH}",
                     "--max_draft_len=63", "--num_eagle_layers=4",
                     "--max_non_leaves_per_layer=10"
                 ],
                 extra_build_args=[
                     "--speculative_decoding_mode=eagle", "--max_draft_len=63"
                 ],
                 extra_summarize_args=extra_summarize_args)


class TestLlama7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "llama-7b-hf"
    MODEL_PATH = f"{llm_models_root()}/llama-models/llama-7b-hf"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_beam_search(self):
        self.run(extra_acc_spec="beam_width=5",
                 extra_build_args=["--max_beam_width=5"],
                 extra_summarize_args=["--num_beams=5"])

    def test_int4_gptq(self):
        self.run(
            quant_algo=QuantAlgo.W4A16_GPTQ,
            extra_convert_args=[
                f"--quant_ckpt_path={llm_models_root()}/int4-quantized-gptq-awq/llama-7b-4bit-gs128.safetensors"
            ])

    def test_streamingllm(self):
        self.run(extra_acc_spec="streamingllm",
                 extra_build_args=["--streamingllm=enable"],
                 extra_summarize_args=[
                     "--max_attention_window_size=2048", "--sink_token_length=4"
                 ])

    def test_manage_weights(self):
        self.run(extra_build_args=["--fast_build"])


class TestLlama2_7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    MODEL_PATH = f"{llm_models_root()}/llama-models-v2/llama-v2-7b-hf"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_smooth_quant(self):
        self.run(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)

    @skip_pre_ada
    def test_fp8(self):
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8)

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
    def test_int4_awq_prequantized_tp2(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/llama-models-v2/Llama-2-7B-AWQ")
        self.run(quant_algo=QuantAlgo.W4A16_AWQ, tp_size=2)

    @pytest.mark.skip_less_device(2)
    def test_int4_gptq_prequantized_tp2(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/llama-models-v2/Llama-2-7B-GPTQ")
        self.run(quant_algo=QuantAlgo.W4A16_GPTQ, tp_size=2)

    def test_weight_sparsity(self):
        self.run(extra_build_args=["--weight_sparsity"])


class TestTinyLlama1_1BChat(CliFlowAccuracyTestHarness):
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    MODEL_PATH = f"{llm_models_root()}/llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_float32(self):
        self.run(dtype='float32')

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


class TestLlama3_8BInstruct(CliFlowAccuracyTestHarness):
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

    @skip_pre_blackwell
    def test_nvfp4(self):
        self.run(tasks=[MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.NVFP4,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 extra_build_args=["--gemm_plugin=disable"])

    @pytest.mark.skip(
        reason="Broken by modelopt. Will be fixed in next release")
    @skip_pre_blackwell
    @pytest.mark.parametrize("fuse_fp4_quant", [False, True],
                             ids=["disable_fused_quant", "enable_fused_quant"])
    @pytest.mark.parametrize(
        "norm_quant_fusion", [False, True],
        ids=["disable_norm_quant_fusion", "enable_norm_quant_fusion"])
    def test_nvfp4_gemm_plugin(self, fuse_fp4_quant: bool,
                               norm_quant_fusion: bool):
        extra_build_args = ["--gemm_plugin=nvfp4"]
        if fuse_fp4_quant:
            extra_build_args.extend([
                "--use_paged_context_fmha=enable",
                "--use_fp8_context_fmha=enable", "--fuse_fp4_quant=enable"
            ])
        if norm_quant_fusion:
            extra_build_args.append("--norm_quant_fusion=enable")
        self.run(tasks=[MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.NVFP4,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 extra_build_args=extra_build_args)


class TestLlama3_8BInstructGradient1048k(CliFlowAccuracyTestHarness):
    MODEL_NAME = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
    MODEL_PATH = f"{llm_models_root()}/llama-models-v3/Llama-3-8B-Instruct-Gradient-1048k"
    EXAMPLE_FOLDER = "llama"

    @pytest.mark.skip_less_device_memory(60000)
    def test_long_context(self):
        self.run(tasks=[PassKeyRetrieval128k(self.MODEL_NAME)])

    @pytest.mark.skip_less_device_memory(60000)
    def test_long_context_ppl(self):
        self.run(tasks=[SlimPajama6B(self.MODEL_NAME)],
                 extra_build_args=["--gather_context_logits"])


class TestLlama3_1_8B(CliFlowAccuracyTestHarness):
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
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)

    @skip_pre_ada
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

    @skip_pre_ada
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

    @skip_pre_ada
    def test_autoq(self):
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.MIXED_PRECISION,
                 extra_acc_spec=
                 "autoq_format=int4_awq,fp8,w4a8_awq;auto_quantize_bits=5.8",
                 extra_convert_args=[
                     "--autoq_format=int4_awq,fp8,w4a8_awq",
                     "--auto_quantize_bits=5.8", "--calib_size=4",
                     "--batch_size=4"
                 ])


class TestLlama3_1_8BInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    EXAMPLE_FOLDER = "llama"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @skip_pre_ada
    def test_fp8_prequantized(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8")
        self.run(quant_algo=QuantAlgo.FP8, kv_cache_quant_algo=QuantAlgo.FP8)

    @skip_pre_ada
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
                 spec_dec_algo=MedusaDecodingConfig.decoding_type,
                 extra_build_args=["--speculative_decoding_mode=medusa"],
                 extra_summarize_args=extra_summarize_args)


class TestLlama3_2_1B(CliFlowAccuracyTestHarness):
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
                 extra_summarize_args=["--gpu_weights_percent=0"])

        for gpu_percent in [0.1, 0.5, 0.9, 1]:
            if gpu_percent > max_gpu_percent:
                break
            self.extra_summarize_args = [f"--gpu_weights_percent={gpu_percent}"]
            self.evaluate()

    def test_cyclic_kv_cache(self):
        self.run(extra_acc_spec="max_attention_window_size=960",
                 extra_summarize_args=["--max_attention_window_size=960"])

    @pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5166352")
    def test_cyclic_kv_cache_beam_search(self):
        self.run(extra_acc_spec="max_attention_window_size=960;beam_width=4",
                 extra_build_args=["--max_beam_width=4"],
                 extra_summarize_args=[
                     "--max_attention_window_size=960", "--num_beams=4"
                 ])


class TestMixtral8x7B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"
    MODEL_PATH = f"{llm_models_root()}/Mixtral-8x7B-v0.1"
    EXAMPLE_FOLDER = "llama"

    @pytest.mark.skip_less_device(2)
    @pytest.mark.skip_less_device_memory(80000)
    def test_tp2(self):
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
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 tp_size=2,
                 pp_size=2)

    @skip_pre_ada
    @pytest.mark.skip_less_device(4)
    @pytest.mark.skip_less_device_memory(40000)
    def test_fp8_tp2pp2_manage_weights(self):
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 tp_size=2,
                 pp_size=2,
                 extra_build_args=["--fast_build"])

    @skip_pre_blackwell
    def test_nvfp4_prequantized(self, mocker):
        mocker.patch.object(
            self.__class__, "MODEL_PATH",
            f"{llm_models_root()}/nvfp4-quantized/Mixtral-8x7B-Instruct-v0.1")
        self.run(tasks=[MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.NVFP4,
                 kv_cache_quant_algo=QuantAlgo.FP8)


class TestGemma2B(CliFlowAccuracyTestHarness):
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
class TestGemma7B(CliFlowAccuracyTestHarness):
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
class TestGemma2_9BIt(CliFlowAccuracyTestHarness):
    MODEL_NAME = "google/gemma-2-9b-it"
    MODEL_PATH = f"{llm_models_root()}/gemma/gemma-2-9b-it"
    EXAMPLE_FOLDER = "gemma"

    def test_auto_dtype(self):
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 dtype='auto',
                 extra_convert_args=["--ckpt-type=hf"])

    @pytest.mark.parametrize("precision", ["int8", "int4"])
    def test_weight_only(self, precision: str):
        quant_algo = QuantAlgo.W8A16 if precision == "int8" else QuantAlgo.W4A16
        self.run(quant_algo=quant_algo, extra_convert_args=["--ckpt-type=hf"])

    @skip_pre_hopper
    def test_fp8(self):
        self.run(quant_algo=QuantAlgo.FP8,
                 kv_cache_quant_algo=QuantAlgo.FP8,
                 extra_convert_args=["--device_map=sequential"])


class TestQwen7BChat(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen-7B-Chat"
    MODEL_PATH = f"{llm_models_root()}/Qwen-7B-Chat"
    EXAMPLE_FOLDER = "qwen"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_weight_only(self):
        self.run(quant_algo=QuantAlgo.W8A16)

    def test_int4_gptq_prequantized(self, mocker):
        mocker.patch.object(self.__class__, "MODEL_PATH",
                            f"{llm_models_root()}/Qwen-7B-Chat-Int4")
        self.run(quant_algo=QuantAlgo.W4A16_GPTQ)


@pytest.mark.skip_less_device_memory(40000)
class TestQwen1_5MoeA2_7BChat(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    MODEL_PATH = f"{llm_models_root()}/Qwen1.5-MoE-A2.7B-Chat"
    EXAMPLE_FOLDER = "qwen"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    @pytest.mark.skip(reason="https://nvbugs/5100102")
    def test_weight_only(self):
        self.run(quant_algo=QuantAlgo.W8A16)


class TestQwen2_0_5BInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-0.5B-Instruct"
    EXAMPLE_FOLDER = "qwen"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_weight_only(self):
        self.run(quant_algo=QuantAlgo.W8A16)

    @skip_pre_ada
    def test_fp8(self):
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8)


class TestQwen2_7BInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-7B-Instruct"
    EXAMPLE_FOLDER = "qwen"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_weight_only(self):
        self.run(quant_algo=QuantAlgo.W8A16)

    def test_int4_awq_prequantized(self, mocker):
        mocker.patch.object(self.__class__, "MODEL_PATH",
                            f"{llm_models_root()}/Qwen2-7B-Instruct-AWQ")
        self.run(quant_algo=QuantAlgo.W4A16_AWQ)


@pytest.mark.skip_less_device_memory(40000)
class TestQwen2_57B_A14B(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2-57B-A14B"
    MODEL_PATH = f"{llm_models_root()}/Qwen2-57B-A14B"
    EXAMPLE_FOLDER = "qwen"

    @pytest.mark.skip(reason="https://nvbugs/5063469")
    @pytest.mark.skip_less_device(4)
    def test_tp4(self):
        self.run(tp_size=4)

    @pytest.mark.skip(reason="https://nvbugs/5063469")
    @pytest.mark.skip_less_device(4)
    def test_tp2pp2(self):
        self.run(tp_size=2, pp_size=2)


class TestQwen2_5_1_5BInstruct(CliFlowAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/Qwen2.5-1.5B-Instruct"
    EXAMPLE_FOLDER = "qwen"

    def test_auto_dtype(self):
        self.run(dtype='auto')

    def test_weight_only(self):
        self.run(quant_algo=QuantAlgo.W8A16)

    @skip_pre_ada
    def test_fp8(self):
        self.run(tasks=[CnnDailymail(self.MODEL_NAME),
                        MMLU(self.MODEL_NAME)],
                 quant_algo=QuantAlgo.FP8)
