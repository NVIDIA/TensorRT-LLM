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
import random
from typing import Iterable, List, Optional, Union

import click
import numpy as np
import pandas as pd

from .._torch import LLM as PyTorchLLM
from ..llmapi import LLM, RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator


class MMLU(Evaluator):
    CHOICES = ["A", "B", "C", "D"]
    SUBJECT_TO_SUBCATEGORIES = {
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
    CATEGORY_TO_SUBCATEGORIES = {
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

    def __init__(self,
                 dataset_path: str,
                 num_samples: int = None,
                 num_train: int = 5,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 system_prompt: Optional[str] = None):
        super().__init__(apply_chat_template=apply_chat_template,
                         system_prompt=system_prompt)
        self.dataset_path = dataset_path
        if num_samples is None:
            self.num_samples_per_subject = None
        else:
            self.num_samples_per_subject = math.ceil(
                num_samples / len(self.SUBJECT_TO_SUBCATEGORIES))
        self.num_train = num_train
        random.seed(random_seed)
        np.random.seed(random_seed)

    def format_subject(self, subject):
        line = subject.split("_")
        s = ""
        for entry in line:
            s += " " + entry
        return s

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(self.CHOICES[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            self.format_subject(subject))
        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(train_df, i)
        return prompt

    def generate_samples(self) -> Iterable[tuple]:
        for subject in self.SUBJECT_TO_SUBCATEGORIES.keys():
            dev_df = pd.read_csv(f"{self.dataset_path}/dev/{subject}_dev.csv",
                                 header=None)
            train_prompt = self.gen_prompt(dev_df, subject, self.num_train)

            test_df = pd.read_csv(
                f"{self.dataset_path}/test/{subject}_test.csv", header=None)
            if self.num_samples_per_subject is not None and self.num_samples_per_subject < test_df.shape[
                    0]:
                test_df = test_df.sample(self.num_samples_per_subject)

            for i in range(test_df.shape[0]):
                prompt_end = self.format_example(test_df,
                                                 i,
                                                 include_answer=False)
                prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1] - 1]
                yield prompt, label, subject

    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      subjects: List[str]) -> float:
        subject_corrections = {
            key: []
            for key in self.SUBJECT_TO_SUBCATEGORIES.keys()
        }
        for output, ref, sub in zip(outputs, references, subjects):
            correction = output.outputs[0].text.strip().startswith(ref)
            subject_corrections[sub].append(correction)

        subcategory_corrections = {
            key: []
            for subcats in self.SUBJECT_TO_SUBCATEGORIES.values()
            for key in subcats
        }
        category_corrections = {
            key: []
            for key in self.CATEGORY_TO_SUBCATEGORIES.keys()
        }
        all_corrections = []
        for sub, corrections in subject_corrections.items():
            for subcat in self.SUBJECT_TO_SUBCATEGORIES[sub]:
                subcategory_corrections[subcat].extend(corrections)
                for cat, subcats in self.CATEGORY_TO_SUBCATEGORIES.items():
                    if subcat in subcats:
                        category_corrections[cat].extend(corrections)
            all_corrections.extend(corrections)

        for subject, corrections in subject_corrections.items():
            acc = np.mean(corrections) * 100
            logger.info(
                f"Average accuracy {acc:.2f} ({len(corrections)}) - {subject}")

        for subcat, corrections in subcategory_corrections.items():
            acc = np.mean(corrections) * 100
            logger.info(
                f"Average accuracy {acc:.2f} ({len(corrections)}) - {subcat}")

        for cat, corrections in category_corrections.items():
            acc = np.mean(corrections) * 100
            logger.info(
                f"Average accuracy {acc:.2f} ({len(corrections)}) - {cat}")

        weighted_acc = np.mean(all_corrections) * 100
        logger.info(
            f"MMLU weighted average accuracy: {weighted_acc:.2f} ({len(all_corrections)})"
        )
        return weighted_acc

    @click.command("mmlu")
    @click.option("--dataset_path", type=str, required=True)
    @click.option("--num_samples", type=int, default=None)
    @click.option("--num_train", type=int, default=5)
    @click.option("--random_seed", type=int, default=0)
    @click.option("--max_input_length", type=int, default=4094)
    @click.option("--max_output_length", type=int, default=2)
    @click.option("--check_accuracy", is_flag=True, default=False)
    @click.option("--accuracy_threshold", type=float, default=30)
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: str, num_samples: int, num_train: int,
                random_seed: int, max_input_length: int, max_output_length: int,
                check_accuracy: bool, accuracy_threshold: float) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        sampling_params = SamplingParams(
            max_tokens=max_output_length,
            truncate_prompt_tokens=max_input_length)
        evaluator = MMLU(dataset_path,
                         num_samples=num_samples,
                         num_train=num_train,
                         random_seed=random_seed)
        accuracy = evaluator.evaluate(llm, sampling_params)
        llm.shutdown()

        if check_accuracy:
            assert accuracy >= accuracy_threshold, f"Expected accuracy >= {accuracy_threshold}, but got {accuracy}"
