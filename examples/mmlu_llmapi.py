# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
# Copyright (c) 2023 Deep Cognition and Language Research (DeCLaRe) Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""A duplication of examples/mmlu.py, but exclusively targeting LLM API.
The duplication is to prevent from breaking CI test that relies on examples/mmlu.py.
TODO: Should be merged with examples/mmlu.py
Example usage:
    mkdir data; wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
    tar -xf data/mmlu.tar -C data && mv data/data data/mmlu

    To eval LLM API with pytorch backend (default):
    python mmlu_llmapi.py --hf_model_dir <HF model path> --backend pytorch
    To eval LLM API with tensorrt backend:
    python mmlu_llmapi.py --hf_model_dir <HF model path> --engine_dir <(Optional) TRTLLM engine path> --backend tensorrt
"""

import argparse
import math
import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi import LLM, KvCacheConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(get_choices()[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def generate_samples(data_dir: str,
                     subjects: List[str],
                     k: int = 5,
                     num_samples_per_subject: Optional[int] = None):
    for subject in subjects:
        dev_df = pd.read_csv(f"{data_dir}/dev/{subject}_dev.csv", header=None)
        train_prompt = gen_prompt(dev_df, subject, k)

        test_df = pd.read_csv(f"{data_dir}/test/{subject}_test.csv",
                              header=None)
        if num_samples_per_subject is not None and num_samples_per_subject < test_df.shape[
                0]:
            test_df = test_df.sample(num_samples_per_subject)

        for i in range(test_df.shape[0]):
            prompt_end = format_example(test_df, i, include_answer=False)
            prompt = train_prompt + prompt_end
            label = test_df.iloc[i, test_df.shape[1] - 1]
            yield subject, prompt, label


def parse_args():
    # Model args
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir",
                        type=str,
                        required=True,
                        help="HF model dir")
    parser.add_argument("--engine_dir",
                        type=str,
                        default=None,
                        help="TensorRT Engine dir (only for tensorrt backend)")
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--backend",
                        type=str,
                        choices=["pytorch", "tensorrt"],
                        default="pytorch",
                        help="Choose the backend to run the model")
    parser.add_argument('--torch_compile',
                        action="store_true",
                        help="Enable torch compile for pytorch backend")
    parser.add_argument("--tp_size",
                        type=int,
                        default=1,
                        help="Tensor Parallel size (only for pytorch backend)")
    parser.add_argument("--ep_size",
                        type=int,
                        default=1,
                        help="Expert Parallel size (only for pytorch backend)")
    parser.add_argument("--enable_attention_dp",
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--attn_backend',
        type=str,
        default='TRTLLM',
        choices=['TRTLLM', 'FLASHINFER'],
        help='Attention kernel for PyTorch. Ignored for TRT backend.')
    parser.add_argument("--enable_chunked_prefill",
                        action="store_true",
                        help="Exercises the chunked prefill inference feature.")
    parser.add_argument('--enable_overlap_scheduler',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--kv_cache_free_gpu_memory_fraction',
        default=0.9,
        type=float,
        help='Specify the free gpu memory fraction.',
    )

    # MMLU args
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu",
        help=("Path to the data directory. If not available, "
              "download https://people.eecs.berkeley.edu/~hendrycks/data.tar"),
    )
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument('--max_input_length', type=int, default=4094)
    parser.add_argument('--output_len', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--num_samples_per_subject', type=int, default=None)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--accuracy_threshold', type=float, default=30)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir
    if args.engine_dir is not None:
        args.backend = "tensorrt"
    if args.num_samples is not None:
        assert args.num_samples_per_subject is None
        args.num_samples_per_subject = math.ceil(args.num_samples /
                                                 len(get_subcategories()))

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    profiler.start("trtllm init")
    if args.enable_chunked_prefill:
        # Use a small max_num_tokens/tokens_per_block to guarantee
        # that chunked context features get exercised.
        build_config = BuildConfig(max_num_tokens=256)
        # Chunk size.
        build_config.plugin_config.tokens_per_block = 64
        # Required to use chunked prefill in the TRT backend.
        build_config.plugin_config.use_paged_context_fmha = True
    else:
        build_config = None

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction)

    if args.backend == "pytorch":
        assert args.engine_dir is None, "pytorch backend does not need TRT Engine"
        config = PyTorchConfig(
            attn_backend=args.attn_backend,
            enable_overlap_scheduler=args.enable_overlap_scheduler,
            torch_compile_enabled=args.torch_compile)
        llm = tensorrt_llm._torch.LLM(
            model=args.hf_model_dir,
            tokenizer=args.tokenizer_dir,
            tensor_parallel_size=args.tp_size,
            moe_expert_parallel_size=args.ep_size,
            pytorch_backend_config=config,
            enable_chunked_prefill=args.enable_chunked_prefill,
            build_config=build_config,
            kv_cache_config=kv_cache_config,
            enable_attention_dp=args.enable_attention_dp)
    else:
        llm = LLM(model=args.engine_dir or args.hf_model_dir,
                  tokenizer=args.tokenizer_dir,
                  tensor_parallel_size=args.tp_size,
                  enable_chunked_prefill=args.enable_chunked_prefill,
                  build_config=build_config,
                  kv_cache_config=kv_cache_config)
    profiler.stop("trtllm init")
    elapsed_time = profiler.elapsed_time_in_sec("trtllm init")
    print(f"TRTLLM initialization time: {elapsed_time:.3f} seconds.")

    subjects = list(get_subcategories().keys())
    subcategories = list(
        set(subcat for subcats in get_subcategories().values()
            for subcat in subcats))
    categories = list(get_categories().keys())

    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        top_k=1,
        temperature=0.0,
        truncate_prompt_tokens=args.max_input_length)

    profiler.start("trtllm exec")
    data = []
    for subject, prompt, label in tqdm(
            generate_samples(args.data_dir, subjects, args.ntrain,
                             args.num_samples_per_subject),
            "Submitting requests"):
        output = llm.generate_async(prompt, sampling_params)
        data.append([subject, prompt, label, output])

    subject_corrections = {key: [] for key in subjects}
    for subject, prompt, label, output in tqdm(data, "Fetching responses"):
        output = output.result()
        correction = output.outputs[0].text.strip().startswith(label)
        subject_corrections[subject].append(correction)
    profiler.stop("trtllm exec")
    elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
    print(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")

    subcategory_corrections = {key: [] for key in subcategories}
    category_corrections = {key: [] for key in categories}
    all_corrections = []
    for subject, corrections in subject_corrections.items():
        for subcat in get_subcategories()[subject]:
            subcategory_corrections[subcat].extend(corrections)
            for cat, subcats in get_categories().items():
                if subcat in subcats:
                    category_corrections[cat].extend(corrections)
        all_corrections.extend(corrections)

    for subject, corrections in subject_corrections.items():
        acc = np.mean(corrections) * 100
        print(f"Average accuracy {acc:.2f} ({len(corrections)}) - {subject}")

    for subcat, corrections in subcategory_corrections.items():
        acc = np.mean(corrections) * 100
        print(f"Average accuracy {acc:.2f} ({len(corrections)}) - {subcat}")

    for cat, corrections in category_corrections.items():
        acc = np.mean(corrections) * 100
        print(f"Average accuracy {acc:.2f} ({len(corrections)}) - {cat}")

    weighted_acc = np.mean(all_corrections) * 100
    print(
        f"MMLU weighted average accuracy: {weighted_acc:.2f} ({len(all_corrections)})"
    )

    if args.check_accuracy:
        assert weighted_acc >= args.accuracy_threshold, f"Expected accuracy >= {args.accuracy_threshold} while got {weighted_acc}"
    return weighted_acc


if __name__ == "__main__":
    main()
