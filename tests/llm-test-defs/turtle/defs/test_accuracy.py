# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
from copy import copy

import pytest

from .common import PluginOptions, venv_check_call
from .conftest import skip_fp8_pre_ada
from .trt_test_alternative import check_call, exists, makedirs


class AccuracyTestConfig:

    def __init__(self,
                 name,
                 model="gpt2",
                 dtype="float16",
                 max_attention_window=None,
                 smooth_quant=False,
                 int8_kv_cache=False,
                 modelopt_quant=False,
                 convert_args=[],
                 build_args=[],
                 summarize_args=[],
                 rouge1_threshold=14,
                 model_subfolder="1-gpu",
                 env=None):
        self.name = name
        self.model = model
        self.dtype = dtype
        self.max_attention_window = max_attention_window
        self.smooth_quant = smooth_quant
        self.int8_kv_cache = int8_kv_cache
        self.modelopt_quant = modelopt_quant
        self.extra_convert_args = convert_args
        self.extra_build_args = build_args
        self.extra_summarize_args = summarize_args
        self.rouge1_threshold = rouge1_threshold
        self.model_subfolder = model_subfolder
        self.env = env

    # Creates a new instance of AccuracyTestConfig by replacing some of the original instance's
    # data members. e.g., new_config = config.replace(name="new_name")
    def replace(self, **kwargs):
        instance = copy(self)
        for k, v in kwargs.items():
            setattr(instance, k, v)
        return instance

    def example(self):
        if "gptj" in self.model:
            return "gptj"
        elif "gpt" in self.model or self.model in [
                "santacoder", "starcoder", "starcoder2"
        ]:
            # GPT, SantaCoder, StarCoder (v1 and v2), and GPT-Next share the same example code
            return "gpt"
        elif "opt" in self.model:
            return "opt"
        elif "bloom" in self.model:
            return "bloom"
        elif "llama" in self.model:
            return "llama"
        elif "phi" in self.model:
            return "phi"
        elif "medusa" in self.model:
            return "medusa"
        elif "mamba" in self.model:
            return "mamba"
        elif "lookahead" in self.model:
            return "llama"
        elif "eagle" in self.model:
            return "eagle"
        else:
            assert False

    def convert_cmd(self, example_root, llm_model_root, llm_datasets_root,
                    output):
        if self.modelopt_quant:
            script = "../quantization/quantize.py"
        else:
            script = "convert_checkpoint.py"

        if self.model in [
                "gpt2", "santacoder", "starcoder", "starcoder2", "gptj",
                "bloom", "opt", "llama", "phi", "mamba", "lookahead"
        ]:
            convert_cmd = [
                f"{example_root}/{script}",
                f"--model_dir={llm_model_root}",
                f"--output_dir={output}/{self.model_subfolder}",
                f"--dtype={self.dtype}",
            ]
            if self.smooth_quant:
                convert_cmd.append("--smoothquant=0.5")
            if self.int8_kv_cache:
                convert_cmd.append("--int8_kv_cache")
            if self.smooth_quant or self.int8_kv_cache:
                convert_cmd.append(
                    f"--calib_dataset={llm_datasets_root}/cimec/lambada")
        elif self.model == "gpt-next":
            convert_cmd = [
                f"{example_root}/{script}",
                f"--nemo_ckpt_path={llm_model_root}",
                f"--output_dir={output}/{self.model_subfolder}",
                f"--dtype={self.dtype}",
            ]
        elif self.model == 'medusa':
            convert_cmd = [
                f"{example_root}/{script}", f"--model_dir={llm_model_root[0]}",
                f"--medusa_model_dir={llm_model_root[1]}",
                f"--output_dir={output}/{self.model_subfolder}",
                "--dtype=float16", "--num_medusa_heads=4"
            ]
        elif self.model == 'eagle':
            convert_cmd = [
                f"{example_root}/{script}", f"--model_dir={llm_model_root[0]}",
                f"--eagle_model_dir={llm_model_root[1]}",
                f"--output_dir={output}/{self.model_subfolder}",
                "--dtype=float16", "--max_draft_len=63", "--num_eagle_layers=4",
                "--max_non_leaves_per_layer=10"
            ]
        else:
            assert False, "Unsupported model"

        if self.modelopt_quant:
            convert_cmd.append(
                f"--calib_dataset={llm_datasets_root}/cnn_dailymail")

        convert_cmd.extend(self.extra_convert_args)
        return convert_cmd

    def _convert_extra_build_args(self, extra_build_args):
        ''' Ugly convert, we should remove this when all of the build
        scripts are migrated to `tensorrt_llm/commands/build.py`
        '''
        res_extra_build_args = []
        for i in range(len(extra_build_args)):
            legacy_arg = extra_build_args[i]
            if legacy_arg == "--use_gpt_attention_plugin=float16":
                arg = ["--gpt_attention_plugin", "float16"]
            elif legacy_arg == "--use_gemm_plugin=float16":
                arg = ["--gemm_plugin", "float16"]
            elif legacy_arg == "--enable_context_fmha":
                arg = ["--context_fmha", "enable"]
            elif legacy_arg == "--paged_kv_cache":
                arg = ["--paged_kv_cache", "enable"]
            elif legacy_arg == "--remove_input_padding":
                arg = ["--remove_input_padding", "enable"]
            else:
                arg = [legacy_arg]
            res_extra_build_args += arg
        return res_extra_build_args

    def build_cmd(self, example_root, converted_model_path, engine_dir):

        model_dir = os.path.join(converted_model_path, self.model_subfolder)

        if self.model in [
                "gpt2", "santacoder", "starcoder", "starcoder2", "gpt-next",
                "gptj", "bloom", "opt", "llama", "medusa", "phi", "mamba",
                "lookahead", "eagle"
        ]:
            build_cmd = [
                "trtllm-build",
                f"--checkpoint_dir={model_dir}",
                f"--output_dir={engine_dir}",
            ]
            # Use smaller max batch size (default is 2048) to avoid OOM
            if self.model in ["phi", "mamba", "opt"]:
                build_cmd.extend([
                    "--max_batch_size=256",
                ])

            if "gpt" in self.model or self.model in [
                    "santacoder", "starcoder", "starcoder2"
            ]:
                build_cmd.extend([
                    "--max_batch_size=8",
                    "--max_input_len=924",
                    "--max_seq_len=1024",
                ])
            elif self.model == 'mamba':
                build_cmd.extend(
                    PluginOptions(None, None, self.dtype).to_args())
                build_cmd.extend([
                    "--remove_input_padding=disable",
                    "--paged_kv_cache=disable",
                ])
            else:
                build_cmd.extend(
                    PluginOptions(self.dtype, None, self.dtype).to_args())

            # Use always IFB models
            if self.model != 'mamba':
                if not any("gpt_attention_plugin" in a for a in build_cmd):
                    build_cmd += ["--gpt_attention_plugin", "float16"]

            if "context-fmha-disabled" in self.name:
                build_cmd.extend(["--context_fmha", "disable"])
            build_cmd.extend(
                self._convert_extra_build_args(self.extra_build_args))
        else:
            build_cmd = [
                f"{example_root}/build.py",
                "--log_level=verbose",
                f"--model_dir={model_dir}",
                f"--output_dir={engine_dir}",
            ]
            build_cmd.extend(
                PluginOptions(self.dtype, None, self.dtype).to_legacy_args())
            build_cmd.extend(self.extra_build_args)
        return build_cmd

    def summarize_cmd(self,
                      example_root,
                      llm_model_root,
                      llm_datasets_root,
                      converted_model_path,
                      engine_dir,
                      llm_rouge_root,
                      llm_tokenizer_root=None):
        summarize_cmd = [
            f"{example_root}/../summarize.py", f"--engine_dir={engine_dir}",
            f"--dataset_dir={llm_datasets_root}",
            f"--rouge_dir={llm_rouge_root}"
        ]
        summarize_cmd.extend(self.extra_summarize_args)

        if self.model == "gptj":
            summarize_cmd.extend([
                "--test_trt_llm", "--check_accuracy", "--hf_model_dir",
                llm_model_root
            ])
        elif self.model == "gpt-next":
            vocab_path = os.path.join(converted_model_path, "1-gpu",
                                      "tokenizer.model")
            summarize_cmd.extend([
                "--test_trt_llm", "--check_accuracy", "--vocab_file",
                vocab_path, "--no_add_special_tokens"
            ])
        elif "gpt" in self.model:
            summarize_cmd.extend([
                "--test_trt_llm", "--check_accuracy", "--hf_model_dir",
                llm_model_root, "--no_add_special_tokens"
            ])
        elif self.model in ["santacoder", "starcoder", "starcoder2"]:
            summarize_cmd.extend([
                "--test_trt_llm", "--check_accuracy", "--eval_task",
                "code_completion", "--hf_model_dir", llm_model_root,
                "--no_add_special_tokens"
            ])
        elif self.model == "bloom":
            summarize_cmd.extend([
                "--test_trt_llm", "--hf_model_dir", llm_model_root,
                "--data_type", "fp16"
            ])
        elif "opt" in self.model:
            summarize_cmd.extend([
                "--test_trt_llm", "--hf_model_dir", llm_model_root,
                "--data_type", "fp16", "--check_accuracy",
                "--no_add_special_tokens"
            ])
        elif "llama" in self.model or "phi" in self.model:
            summarize_cmd.extend([
                "--test_trt_llm", "--hf_model_dir", llm_model_root,
                "--data_type", "fp16", "--check_accuracy",
                "--no_add_special_tokens"
            ])
        elif self.model == "medusa":
            summarize_cmd.extend([
                "--data_type",
                "fp16",
                "--test_trt_llm",
                "--hf_model_dir",
                llm_model_root[0],
                "--tokenizer_dir",
                llm_model_root[0],
                "--check_accuracy",
                "--medusa_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]",
            ])
        elif self.model == "mamba":
            summarize_cmd.extend([
                "--test_trt_llm", "--use_py_session", "--hf_model_dir",
                llm_model_root, "--tokenizer_dir", llm_tokenizer_root,
                "--data_type", "fp16"
            ])
        elif self.model == "lookahead":
            summarize_cmd.extend([
                "--data_type", "fp16", "--test_trt_llm", "--hf_model_dir",
                llm_model_root, "--tokenizer_dir", llm_model_root,
                "--check_accuracy"
            ])
        elif self.model == "eagle":
            summarize_cmd.extend([
                "--data_type",
                "fp16",
                "--test_trt_llm",
                "--hf_model_dir",
                llm_model_root[0],
                "--tokenizer_dir",
                llm_model_root[0],
                "--check_accuracy",
                "--eagle_choices=[[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]",
            ])
        elif self.max_attention_window:
            summarize_cmd.extend(
                [f"--max_attention_window_size {self.max_attention_window}"])

        summarize_cmd.append(
            f"--tensorrt_llm_rouge1_threshold={self.rouge1_threshold}")

        return summarize_cmd


GPT_ACCURACY_TESTS = [
    AccuracyTestConfig("gpt-context-fmha-disabled",
                       build_args=["--context_fmha=disable"],
                       summarize_args=["--batch_size=8", "--max_ite=40"]),
    AccuracyTestConfig("gpt-context-fmha-enabled",
                       build_args=["--context_fmha=enable"],
                       summarize_args=["--batch_size=8", "--max_ite=40"]),
    AccuracyTestConfig("gpt-mmha-multi-block-mode",
                       build_args=["--context_fmha=enable"],
                       summarize_args=["--batch_size=8", "--max_ite=40"]),
    AccuracyTestConfig("gpt-context-fmha-fp32-acc-enabled",
                       build_args=["--context_fmha=enable"],
                       summarize_args=[
                           "--batch_size=8", "--max_ite=40",
                           "--enable_context_fmha_fp32_acc"
                       ]),
    AccuracyTestConfig(
        "gpt-use-int8-weight-only-quant",
        convert_args=["--use_weight_only", "--weight_only_precision=int8"],
        summarize_args=["--batch_size=8", "--max_ite=40"]),

    # https://nvbugs/4181031, change to 12.8 to WAR the test stability issue.
    AccuracyTestConfig(
        "gpt-use-int4-weight-only-quant",
        convert_args=["--use_weight_only", "--weight_only_precision=int4"],
        build_args=["--context_fmha=disable"],
        summarize_args=["--batch_size=8", "--max_ite=40"],
        rouge1_threshold=11),
    AccuracyTestConfig("gpt-use-int8-kv-cache",
                       int8_kv_cache=True,
                       summarize_args=["--batch_size=8", "--max_ite=40"],
                       rouge1_threshold=13),
    AccuracyTestConfig(
        "gpt-smooth-quant",
        model="gpt2",
        smooth_quant=True,
        summarize_args=["--batch_size=8", "--max_ite=40"],
        # TODO: https://nvbugs/4240447, check threshold
        rouge1_threshold=10.5),
    AccuracyTestConfig("gpt-smooth-quant-per-token-per-channel",
                       model="gpt2",
                       convert_args=["--per_token", "--per_channel"],
                       smooth_quant=True,
                       summarize_args=["--batch_size=8", "--max_ite=40"],
                       rouge1_threshold=13),
    AccuracyTestConfig("gpt-beam-search",
                       model="gpt2",
                       build_args=["--max_beam_width=4"],
                       summarize_args=[
                           "--num_beams=4", "--length_penalty=2.0",
                           "--batch_size=8", "--max_ite=40"
                       ],
                       rouge1_threshold=18),
    AccuracyTestConfig("gpt-remove-padding",
                       model="gpt2",
                       build_args=["--remove_input_padding"],
                       summarize_args=["--batch_size=8", "--max_ite=40"]),
    AccuracyTestConfig(
        "gpt-remove-padding-beam-search",
        model="gpt2",
        build_args=["--remove_input_padding", "--max_beam_width=4"],
        summarize_args=["--num_beams=4", "--batch_size=8", "--max_ite=40"],
        rouge1_threshold=17),
    AccuracyTestConfig("gpt-paged-kv-cache",
                       model="gpt2",
                       build_args=["--paged_kv_cache"],
                       summarize_args=["--batch_size=8", "--max_ite=40"]),
    AccuracyTestConfig("gpt-weight-streaming-ootb",
                       model="gpt2",
                       build_args=[
                           "--gpt_attention_plugin=disable",
                           "--weight_streaming",
                           "--remove_input_padding=disable",
                           "--paged_kv_cache=disable"
                       ],
                       summarize_args=[
                           "--gpu_weights_percent=0.5", "--batch_size=8",
                           "--max_ite=40", "--use_py_session"
                       ]),
    AccuracyTestConfig("gpt-weight-streaming-mha-plugin",
                       model="gpt2",
                       build_args=["--weight_streaming"],
                       summarize_args=[
                           "--gpu_weights_percent=0", "--batch_size=8",
                           "--max_ite=40"
                       ]),
    AccuracyTestConfig(
        "gpt-cuda-graph",
        model="gpt2",
        summarize_args=["--cuda_graph_mode", "--batch_size=8", "--max_ite=40"]),
]

SANTACODER_ACCURACY_TESTS = [
    c.replace(model="santacoder",
              name=c.name.replace("gpt", "santacoder"),
              rouge1_threshold=21,
              summarize_args=["--eval_task", "code_completion"])
    for c in GPT_ACCURACY_TESTS
    if "quant" not in c.name and "int8" not in c.name
]

GPT_NEXT_ACCURACY_TESTS = [
    AccuracyTestConfig("gpt-next",
                       model="gpt-next",
                       build_args=['--context_fmha=disable'],
                       rouge1_threshold=16)
]

GPTJ_ACCURACY_TESTS = [
    AccuracyTestConfig("gptj-context-fmha-disabled", model="gptj"),
    AccuracyTestConfig("gptj-context-fmha-enabled",
                       model="gptj",
                       build_args=['--enable_context_fmha']),
    AccuracyTestConfig("gptj-mmha-multi-block-mode",
                       model="gptj",
                       build_args=["--enable_context_fmha"]),
    # Add fp8 kv cache tests
    AccuracyTestConfig("gptj-fp8-kv-cache",
                       model="gptj",
                       modelopt_quant=True,
                       convert_args=['--qformat=fp8', '--kv_cache_dtype=fp8'],
                       rouge1_threshold=19,
                       summarize_args=[
                           "--max_ite=40",
                       ]),
    # Add gptj cyclic kv cache tests.
    AccuracyTestConfig("gptj-cyclic-kv-cache",
                       model="gptj",
                       dtype="float16",
                       max_attention_window=900),
    AccuracyTestConfig("gptj-cyclic-and-paged-kv-cache",
                       model="gptj",
                       dtype="float16",
                       max_attention_window=900,
                       build_args=['--paged_kv_cache']),
    AccuracyTestConfig("gptj-cyclic-kv-cache-beam-search",
                       model="gptj",
                       dtype="float16",
                       max_attention_window=900,
                       build_args="--max_beam_width 4".split(" "),
                       summarize_args="--num_beams 4".split(" "),
                       rouge1_threshold=17),
    # Add float32 head_size = 256 case for MMHA tests.
    AccuracyTestConfig("gptj-float32",
                       model="gptj",
                       dtype="float32",
                       build_args=['--gpt_attention_plugin=float32']),
]

PHI_ACCURACY_TESTS = [
    AccuracyTestConfig("phi-context-fmha-disabled",
                       model="phi",
                       rouge1_threshold=22),
    AccuracyTestConfig("phi-context-fmha-enabled",
                       model="phi",
                       build_args=['--enable_context_fmha'],
                       rouge1_threshold=21.5),
    AccuracyTestConfig("phi-mmha-multi-block-mode",
                       model="phi",
                       build_args=["--enable_context_fmha"],
                       rouge1_threshold=21.5),
]

OPT_ACCURACY_TESTS = [
    AccuracyTestConfig("opt-context-fmha-disabled",
                       model="opt",
                       rouge1_threshold=14),
    AccuracyTestConfig("opt-context-fmha-enabled",
                       model="opt",
                       build_args=[
                           "--enable_context_fmha",
                       ],
                       rouge1_threshold=14),
    AccuracyTestConfig(
        "opt-weight-streaming",
        model="opt",
        build_args=["--enable_context_fmha", "--weight_streaming"],
        summarize_args=["--gpu_weights_percent=0.01"],
        rouge1_threshold=14),
    AccuracyTestConfig("opt-mmha-multi-block-mode",
                       model="opt",
                       build_args=["--enable_context_fmha"],
                       rouge1_threshold=14),
    AccuracyTestConfig("opt-context-fmha-fp32-acc-enabled",
                       model="opt",
                       summarize_args=["--enable_context_fmha_fp32_acc"],
                       rouge1_threshold=14),
    AccuracyTestConfig("opt-context-fmha-fp32-acc-enabled",
                       model="opt",
                       summarize_args=["--enable_context_fmha_fp32_acc"],
                       rouge1_threshold=14),
]

BLOOM_ACCURACY_TESTS = [
    AccuracyTestConfig(
        "bloom-context-fmha-enabled",
        model="bloom",
        build_args=["--enable_context_fmha", "--max_batch_size=8"],
        rouge1_threshold=16),
    AccuracyTestConfig(
        "bloom-mmha-multi-block-mode",
        model="bloom",
        build_args=["--enable_context_fmha", "--max_batch_size=8"],
        rouge1_threshold=16),
    AccuracyTestConfig(
        "bloom-context-fmha-disabled",
        model="bloom",
        build_args=[f"--max_batch_size={1}", f"--max_input_len={1024}"],
        rouge1_threshold=16)
]

LONG_ALPACA_ACCURACY_TEST = [
    # Baseline
    AccuracyTestConfig(
        "long-alpaca-7b",
        model="llama",  # belongs to LLaMA family of models
        build_args=[
            "--remove_input_padding", "--enable_context_fmha",
            "--max_input_len", "32768", "--max_seq_len", "32768",
            "--max_batch_size", "1", "--max_num_tokens", "32768"
        ],
        summarize_args=[
            "--eval_task", "summarize_long", "--max_input_length", "24576",
            "--output_len", "8192"
        ],
        rouge1_threshold=26.8,  # FIXME: was 27.x before Dec8 rebase
        model_subfolder=''),
    # MMHA Multi_block_mode
    AccuracyTestConfig(
        "long-alpaca-7b-multiblock",
        model="llama",  # belongs to LLaMA family of models
        build_args=[
            "--remove_input_padding", "--enable_context_fmha",
            "--max_input_len", "32768", "--max_seq_len", "32768",
            "--max_batch_size", "1", "--max_num_tokens", "32768"
        ],
        summarize_args=[
            "--eval_task", "summarize_long", "--max_input_length", "24576",
            "--output_len", "8192"
        ],
        rouge1_threshold=26.8,  # FIXME: was 27.x before Dec8 rebase
        model_subfolder=''),
    # MMHA + aggressive Multi_block_mode (export TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG=1)
    AccuracyTestConfig(
        "long-alpaca-7b-multiblock-aggressive",
        model="llama",  # belongs to LLaMA family of models
        build_args=[
            "--remove_input_padding", "--enable_context_fmha",
            "--max_input_len", "32768", "--max_seq_len", "32768",
            "--max_batch_size", "1", "--max_num_tokens", "32768",
            "--use_gpt_attention_plugin=float16", "--use_gemm_plugin=float16"
        ],
        summarize_args=[
            "--eval_task", "summarize_long", "--max_input_length", "24576",
            "--output_len", "8192"
        ],
        rouge1_threshold=26.89,  # FIXME: was 27.x before Dec8 rebase
        model_subfolder='',
        env={
            "TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG": "1",
            "TRTLLM_MMHA_BLOCKS_PER_SEQUENCE": "32"
        })
]

MEDUSA_VICUNA_ACCURACY_TEST = [
    AccuracyTestConfig(
        "medusa-vicuna-7b",
        model="medusa",  # belongs to Medusa models
        build_args=[
            "--use_gpt_attention_plugin=float16", "--use_gemm_plugin=float16",
            "--max_beam_width", "1", "--max_batch_size", "1", "--max_input_len",
            "1024", "--max_seq_len", "1280",
            "--speculative_decoding_mode=medusa"
        ],
        summarize_args=[
            "--eval_task",
            "summarize",
            "--use_py_session",
            "--temperature",
            "1.0",
            "--batch_size",
            "1",
            "--max_ite",
            "100",
        ],
        rouge1_threshold=25,
        model_subfolder=''),
    AccuracyTestConfig(
        "medusa-vicuna-7b-cuda-graph",
        model="medusa",  # belongs to Medusa models
        build_args=[
            "--use_gpt_attention_plugin=float16", "--use_gemm_plugin=float16",
            "--max_beam_width", "1", "--max_batch_size", "8", "--max_input_len",
            "1024", "--max_seq_len", "1280",
            "--speculative_decoding_mode=medusa"
        ],
        summarize_args=[
            "--eval_task", "summarize", "--temperature", "1.0", "--batch_size",
            "8", "--max_ite", "40", "--cuda_graph_mode"
        ],
        rouge1_threshold=25,
        model_subfolder=''),
]

MAMBA_ACCURACY_TESTS = [
    AccuracyTestConfig("mamba-130m", model="mamba", rouge1_threshold=15.4),
]

LOOKAHEAD_VICUNA_ACCURACY_TEST = [
    AccuracyTestConfig("lookahead-vicuna-7b",
                       model="lookahead",
                       build_args=[
                           "--gpt_attention_plugin=float16",
                           "--gemm_plugin=float16", "--max_batch_size=8",
                           "--max_beam_width=1", "--paged_kv_cache=enable",
                           "--remove_input_padding=enable",
                           "--max_draft_len=83",
                           "--speculative_decoding_mode=lookahead_decoding"
                       ],
                       summarize_args=[
                           "--max_ite=100", "--lookahead_config=[7,7,7]",
                           "--batch_size=2"
                       ],
                       rouge1_threshold=25,
                       model_subfolder='')
]

EAGLE_VICUNA_ACCURACY_TEST = [
    AccuracyTestConfig(
        "eagle-vicuna-7b",
        model="eagle",
        build_args=[
            "--gpt_attention_plugin=float16", "--gemm_plugin=float16",
            "--max_batch_size=8", "--max_beam_width=1",
            "--paged_kv_cache=enable", "--remove_input_padding=enable",
            "--use_paged_context_fmha=enable", "--max_draft_len=63",
            "--speculative_decoding_mode=eagle"
        ],
        summarize_args=["--max_ite=100", "--batch_size=2"],
        rouge1_threshold=25,
        model_subfolder=''),
    AccuracyTestConfig(
        "eagle-vicuna-7b-cuda-graph",
        model="eagle",
        build_args=[
            "--gpt_attention_plugin=float16", "--gemm_plugin=float16",
            "--max_batch_size=8", "--max_beam_width=1",
            "--paged_kv_cache=enable", "--remove_input_padding=enable",
            "--use_paged_context_fmha=enable", "--max_draft_len=63",
            "--speculative_decoding_mode=eagle"
        ],
        summarize_args=["--max_ite=100", "--batch_size=2", "--cuda_graph_mode"],
        rouge1_threshold=25,
        model_subfolder=''),
    AccuracyTestConfig(
        "eagle-vicuna-7b-cuda-graph-chunked-context",
        model="eagle",
        build_args=[
            "--gpt_attention_plugin=float16", "--gemm_plugin=float16",
            "--max_batch_size=2", "--max_beam_width=1", "--max_num_tokens=128",
            "--paged_kv_cache=enable", "--remove_input_padding=enable",
            "--use_paged_context_fmha=enable", "--max_draft_len=63",
            "--speculative_decoding_mode=eagle"
        ],
        summarize_args=[
            "--max_ite=100", "--batch_size=2", "--cuda_graph_mode",
            "--enable_chunked_context"
        ],
        # Discussed occasional failure of this test with Nikita, who confirmed
        # that 24.6 instead of 25 is fine and that test can be flaky depending
        # on the trt version and HW you run on.
        # Lowered rouge threshold from 25 to 24.5 to reduce flakiness.
        rouge1_threshold=24.5,
        model_subfolder=''),
    AccuracyTestConfig(
        "eagle-vicuna-7b-cuda-graph-typical-acceptance",
        model="eagle",
        build_args=[
            "--gpt_attention_plugin=float16", "--gemm_plugin=float16",
            "--max_batch_size=8", "--max_beam_width=1",
            "--paged_kv_cache=enable", "--remove_input_padding=enable",
            "--use_paged_context_fmha=enable", "--max_draft_len=63",
            "--speculative_decoding_mode=eagle"
        ],
        summarize_args=[
            "--max_ite=100", "--batch_size=8", "--cuda_graph_mode",
            "--eagle_posterior_threshold=0.09", "--temperature=0.7"
        ],
        rouge1_threshold=23,
        model_subfolder='')
]

LARGE_BEAM_WIDTH_SEARCH_ACCURACY_TEST = [
    AccuracyTestConfig("large-beam-width-search",
                       model="gpt2",
                       build_args=[
                           "--gpt_attention_plugin=float16",
                           "--gemm_plugin=float16",
                           "--max_batch_size=1",
                           "--max_beam_width=256",
                           "--use_paged_context_fmha=enable",
                       ],
                       summarize_args=[
                           "--max_ite=10",
                           "--batch_size=1",
                           "--num_beams=256",
                           "--max_input_length=572",
                       ],
                       rouge1_threshold=9.5,
                       model_subfolder=''),
]


def accuracy_test_harness(llm_root,
                          llm_venv,
                          case,
                          llm_model_root,
                          llm_datasets_root,
                          llm_rouge_root,
                          llm_tokenizer_root=None):
    example_root = os.path.join(llm_root, "examples", case.example())
    requirements = os.path.join(example_root, "requirements.txt")
    converted_model_path = os.path.join(llm_venv.get_working_directory(),
                                        "cmodels", case.name)
    engine_dir = os.path.join(llm_venv.get_working_directory(), "engines",
                              case.name)

    if exists(requirements):
        llm_venv.run_cmd(["-m", "pip", "install", "-r", requirements])

    if not exists(converted_model_path):
        makedirs(converted_model_path)

    if not exists(engine_dir):
        makedirs(engine_dir)

    print("Converting model to TensorRT-LLM checkpoint...")
    convert_cmd = case.convert_cmd(example_root, llm_model_root,
                                   llm_datasets_root, converted_model_path)
    venv_check_call(llm_venv, convert_cmd)

    print("Building engines...")
    build_cmd = case.build_cmd(example_root, converted_model_path, engine_dir)

    if case.model in [
            "gpt2", "santacoder", "starcoder", "starcoder2", "gpt-next", "gptj",
            "opt", "bloom", "llama", "medusa", "phi", "mamba", "lookahead",
            "eagle"
    ]:
        check_call(" ".join(build_cmd), shell=True, env=llm_venv._new_env)
    else:
        venv_check_call(llm_venv, build_cmd)

    print("Running summarize...")
    summarize_cmd = case.summarize_cmd(example_root, llm_model_root,
                                       llm_datasets_root, converted_model_path,
                                       engine_dir, llm_rouge_root,
                                       llm_tokenizer_root)
    venv_check_call(llm_venv, summarize_cmd, env=case.env)


@pytest.mark.parametrize("case", GPT_ACCURACY_TESTS, ids=lambda c: c.name)
def test_accuracy_gpt(llm_root, llm_venv, case, llm_gpt2_model_root,
                      llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, llm_gpt2_model_root,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("case",
                         SANTACODER_ACCURACY_TESTS,
                         ids=lambda c: c.name)
def test_accuracy_santacoder(llm_root, llm_venv, case,
                             llm_gpt2_santacoder_model_root, llm_datasets_root,
                             llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case,
                          llm_gpt2_santacoder_model_root, llm_datasets_root,
                          llm_rouge_root)


@pytest.mark.parametrize("case", GPT_NEXT_ACCURACY_TESTS, ids=lambda c: c.name)
def test_accuracy_gpt_next(llm_root, llm_venv, case, gpt_next_root,
                           llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, gpt_next_root,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("case", GPTJ_ACCURACY_TESTS, ids=lambda c: c.name)
def test_accuracy_gptj(llm_root, llm_venv, case, llm_gptj_model_root,
                       llm_datasets_root, llm_rouge_root):
    skip_fp8_pre_ada('fp8' in case.name)
    accuracy_test_harness(llm_root, llm_venv, case, llm_gptj_model_root,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.skip_less_device_memory(50000)
@pytest.mark.parametrize("case", PHI_ACCURACY_TESTS, ids=lambda c: c.name)
@pytest.mark.parametrize(
    "llm_phi_model_root",
    ["phi-2", "Phi-3-mini-4k-instruct", "Phi-3-mini-128k-instruct"],
    indirect=True)
def test_accuracy_phi(llm_root, llm_venv, case, llm_phi_model_root,
                      gpt_next_root, llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, llm_phi_model_root,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("case", BLOOM_ACCURACY_TESTS, ids=lambda c: c.name)
def test_accuracy_bloom(llm_root, llm_venv, case, llm_bloom_3b_model_root,
                        gpt_next_root, llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, llm_bloom_3b_model_root,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("case", OPT_ACCURACY_TESTS, ids=lambda c: c.name)
def test_accuracy_opt(llm_root, llm_venv, case, llm_opt_model_root,
                      llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, llm_opt_model_root,
                          llm_datasets_root, llm_rouge_root)


# Long sequence length test:
# Model FP16 7B + 32K tokens in KV cache = 14 * 1024 MB + 32K * 0.5 MB = 30720 MB + scratch memory
@pytest.mark.skip_less_device_memory(40000)
@pytest.mark.parametrize("case",
                         LONG_ALPACA_ACCURACY_TEST,
                         ids=lambda c: c.name)
def test_accuracy_long_alpaca(llm_root, llm_venv, case,
                              llm_long_alpaca_model_root, llm_datasets_root,
                              llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, llm_long_alpaca_model_root,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("medusa_model_roots", ['medusa-vicuna-7b-v1.3'],
                         indirect=True)
@pytest.mark.parametrize("case",
                         MEDUSA_VICUNA_ACCURACY_TEST,
                         ids=lambda c: c.name)
def test_accuracy_medusa(llm_root, llm_venv, case, medusa_model_roots,
                         llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, medusa_model_roots,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("lookahead_model_roots", ['vicuna-7b-v1.3'],
                         indirect=True)
@pytest.mark.parametrize("case",
                         LOOKAHEAD_VICUNA_ACCURACY_TEST,
                         ids=lambda c: c.name)
def test_accuracy_lookahead(llm_root, llm_venv, case, lookahead_model_roots,
                            llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, lookahead_model_roots,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("eagle_model_roots", ["EAGLE-Vicuna-7B-v1.3"],
                         indirect=True)
@pytest.mark.parametrize("case",
                         EAGLE_VICUNA_ACCURACY_TEST,
                         ids=lambda c: c.name)
def test_accuracy_eagle(llm_root, llm_venv, case, eagle_model_roots,
                        llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, eagle_model_roots,
                          llm_datasets_root, llm_rouge_root)


@pytest.mark.parametrize("case", MAMBA_ACCURACY_TESTS, ids=lambda c: c.name)
def test_accuracy_mamba(llm_root, llm_venv, case, mamba_model_root,
                        llm_datasets_root, llm_gptneox_model_root,
                        mamba_example_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, mamba_model_root,
                          llm_datasets_root, llm_rouge_root,
                          llm_gptneox_model_root)


@pytest.mark.parametrize("case",
                         LARGE_BEAM_WIDTH_SEARCH_ACCURACY_TEST,
                         ids=lambda c: c.name)
def test_accuracy_large_beam_width_search(llm_root, llm_venv, case,
                                          llm_gpt2_model_root,
                                          llm_datasets_root, llm_rouge_root):
    accuracy_test_harness(llm_root, llm_venv, case, llm_gpt2_model_root,
                          llm_datasets_root, llm_rouge_root)
