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
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.llmapi.llm import LLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DTYPE_STR_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
RAND_SEED = 1234


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


def evaluate(args, subject, pipeline, dev_df, test_df):
    cors = []
    all_probs = []
    for i in range(test_df.shape[0]):
        if i >= args.max_ite:
            break
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while not pipeline.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1] - 1]
        pred = pipeline(prompt)

        probs = [0 for _ in get_choices()]
        cor = pred.strip().startswith(label)
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def get_tokenizer(ckpt_path, max_seq_len):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class Pipeline:

    def __init__(self, model: LLM):
        """Initialize Pipeline with basic components.

        Args:
            tokenizer: Tokenizer instance
            model: The language model instance
            model_name: Name of the model being used
        """

        self.model = model
        # Fixed output length for MMLU-style tasks, use greedy decoding
        self.sampling_params = SamplingParams(max_tokens=2,
                                              top_k=1,
                                              temperature=0.0)

    def __call__(self, prompt):
        """Process the input prompt and generate a response.

        Args:
            prompt: Input text prompt

        Returns:
            str: Generated response with special tokens removed
        """
        # Encode and prepare input
        # TODO: batched inference
        batch = [prompt]

        # Generate response
        batch_outputs = self.model.generate(batch,
                                            self.sampling_params,
                                            use_tqdm=False)

        # Extract generated tokens
        text = batch_outputs[0].outputs[0].text

        # Decode and return response
        return text

    def check_valid_length(self, prompt):
        """Check if prompt length is valid for model.

        Args:
            prompt: Input text prompt

        Returns:
            bool: True if the prompt is within valid length
        """
        # TODO: Check whether this is needed for LLM API.
        # input_len = len(self.tokenizer.encode(prompt))
        # build_config = self.model.args.build_config
        # return (input_len <= build_config.max_input_len and
        #             input_len + build_config.max_output_len <= build_config.max_seq_len)
        return True


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
    parser.add_argument("--tp_size",
                        type=int,
                        default=1,
                        help="Tensor Parallel size (only for pytorch backend)")
    parser.add_argument(
        '--attn_backend',
        type=str,
        default='TRTLLM',
        choices=['TRTLLM', 'FLASHINFER'],
        help='Attention kernel for PyTorch. Ignored for TRT backend.')
    parser.add_argument("--enable_chunked_prefill",
                        action="store_true",
                        help="Exercises the chunked prefill inference feature.")

    # MMLU args
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mmlu",
        help=("Path to the data directory. If not available, "
              "download https://people.eecs.berkeley.edu/~hendrycks/data.tar"),
    )
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument('--check_accuracy', action='store_true')
    parser.add_argument('--accuracy_threshold', type=float, default=0.3)
    parser.add_argument('--max_ite', type=int, default=10000000)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.tokenizer_dir is None:
        args.tokenizer_dir = args.hf_model_dir
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    os.path.dirname(os.path.abspath(__file__))
    data_fullpath = os.path.join(args.data_dir, "test")

    subjects = sorted([
        f.split("_test.csv")[0] for f in os.listdir(data_fullpath)
        if "_test.csv" in f
    ])

    all_cors = []
    subcat_cors = {
        subcat: []
        for subcat_lists in get_subcategories().values()
        for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in get_categories()}

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir)

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

    if args.backend == "pytorch":
        assert args.engine_dir is None, "pytorch backend does not need TRT Engine"
        config = PyTorchConfig(attn_backend=args.attn_backend)
        model = tensorrt_llm._torch.LLM(
            model=args.hf_model_dir,
            tokenizer=tokenizer,
            tensor_parallel_size=args.tp_size,
            pytorch_backend_config=config,
            enable_chunked_prefill=args.enable_chunked_prefill,
            build_config=build_config)
    else:
        model = LLM(model=args.engine_dir or args.hf_model_dir,
                    tokenizer=tokenizer,
                    tensor_parallel_size=args.tp_size,
                    enable_chunked_prefill=args.enable_chunked_prefill,
                    build_config=build_config)

    pipeline = Pipeline(model)

    t = time.time()
    for subject in tqdm(subjects):
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev",
                                          subject + "_dev.csv"),
                             header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test",
                                           subject + "_test.csv"),
                              header=None)

        cors, acc, probs = evaluate(args, subject, pipeline, dev_df, test_df)
        subcats = get_subcategories()[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in get_categories().keys():
                if subcat in get_categories()[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

    t = time.time() - t
    print(f"Finished in {t:.3f} seconds")

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    if args.check_accuracy:
        assert weighted_acc >= args.accuracy_threshold, f"Expected accuracy >= {args.accuracy_threshold} while got {weighted_acc}"
    return weighted_acc


if __name__ == "__main__":
    main()
