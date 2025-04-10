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
from abc import abstractmethod
from contextlib import contextmanager
from typing import Iterable, List, Optional, Tuple, Union

import click
import lm_eval
from tqdm import tqdm

from .._torch import LLM as PyTorchLLM
from ..llmapi import LLM, RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator


class LmEvalWrapper(lm_eval.api.model.TemplateLM):

    def __init__(self, llm: Union[LLM, PyTorchLLM]):
        super().__init__()
        self.llm = llm

    @property
    def eot_token_id(self) -> int:
        return self.llm.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.llm.tokenizer.encode(string, **kwargs)

    def _loglikelihood_tokens(self, requests,
                              **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError()

    def loglikelihood_rolling(self,
                              requests,
                              disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError()

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        outputs = []
        for request in tqdm(requests,
                            desc="Submitting requests",
                            disable=disable_tqdm):
            prompt, gen_kwargs = request.args
            sampling_params = SamplingParams(
                max_tokens=gen_kwargs.get("max_gen_toks", 256),
                top_p=gen_kwargs.get("top_p", None),
                temperature=gen_kwargs.get("temperature", None),
                stop=gen_kwargs.get("until", None),
                include_stop_str_in_output=False)
            output = self.llm.generate_async(prompt,
                                             sampling_params=sampling_params)
            outputs.append(output)

        for output in tqdm(outputs,
                           desc="Fetching responses",
                           disable=disable_tqdm):
            output.result()

        return [output.outputs[0].text for output in outputs]


class LmEvalEvaluator(Evaluator):

    def __init__(self,
                 task_name: str,
                 dataset_path: str = None,
                 num_samples: Optional[int] = None,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 system_prompt: Optional[str] = None):
        if system_prompt is not None:
            raise NotImplementedError("lm-eval does not support system_prompt.")
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template)
        self.dataset_path = dataset_path
        self.num_samples = num_samples
        with self._patch_lm_eval():
            self.task_dict = lm_eval.tasks.get_task_dict(task_name)

    @contextmanager
    def _patch_lm_eval(self):
        if self.dataset_path is None:
            yield
            return

        self._task_config_post_init = lm_eval.api.task.TaskConfig.__post_init__

        def _patched(task_config, *args, **kwargs):
            task_config.dataset_path = self.dataset_path
            self._task_config_post_init(task_config, *args, **kwargs)

        lm_eval.api.task.TaskConfig.__post_init__ = _patched

        try:
            yield
        finally:
            lm_eval.api.task.TaskConfig.__post_init__ = self._task_config_post_init

    def generate_samples(self) -> Iterable[tuple]:
        raise NotImplementedError()

    def compute_score(self, outputs: List[RequestOutput], references: List[str],
                      *auxiliaries) -> float:
        raise NotImplementedError()

    @abstractmethod
    def get_score(self, results: dict):
        raise NotImplementedError()

    def evaluate(self,
                 llm: Union[LLM, PyTorchLLM],
                 sampling_params: Optional[SamplingParams] = None) -> float:
        if sampling_params is not None:
            raise NotImplementedError("lm-eval handles sampling internally.")
        results = lm_eval.evaluate(lm=LmEvalWrapper(llm),
                                   task_dict=self.task_dict,
                                   limit=self.num_samples,
                                   apply_chat_template=self.apply_chat_template)
        logger.info(f"Lm eval results:\n{lm_eval.utils.make_table(results)}")
        return self.get_score(results)


class GSM8K(LmEvalEvaluator):

    def __init__(self,
                 dataset_path: str = None,
                 num_samples: int = None,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 system_prompt: Optional[str] = None):
        super().__init__(task_name="gsm8k",
                         dataset_path=dataset_path,
                         num_samples=num_samples,
                         random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         system_prompt=system_prompt)

    def get_score(self, results: dict):
        return results["results"]["gsm8k"]["exact_match,strict-match"] * 100

    @click.command("gsm8k")
    @click.option("--dataset_path", type=str, default=None)
    @click.option("--num_samples", type=int, default=None)
    @click.option("--random_seed", type=int, default=0)
    @click.option("--check_accuracy", is_flag=True, default=False)
    @click.option("--accuracy_threshold", type=float, default=50)
    @click.pass_context
    @staticmethod
    def command(ctx, dataset_path: str, num_samples: int, random_seed: int,
                check_accuracy: bool, accuracy_threshold: float) -> None:
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        evaluator = GSM8K(dataset_path,
                          num_samples=num_samples,
                          random_seed=random_seed)
        accuracy = evaluator.evaluate(llm)
        llm.shutdown()

        if check_accuracy:
            assert accuracy >= accuracy_threshold, f"Expected accuracy >= {accuracy_threshold}, but got {accuracy}"
