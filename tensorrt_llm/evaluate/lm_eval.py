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
import copy
import os
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple, Union

import click
import numpy as np
from tqdm import tqdm

import tensorrt_llm.profiler as profiler

try:
    from lm_eval.api.model import TemplateLM
except ImportError:
    TemplateLM = object

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator


class LmEvalWrapper(TemplateLM):

    def __init__(self,
                 llm: Union[LLM, PyTorchLLM],
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False):
        super().__init__()
        self.llm = llm
        self.sampling_params = sampling_params
        self.streaming = streaming

    @property
    def eot_token_id(self) -> int:
        return self.llm.tokenizer.eos_token_id

    def apply_chat_template(self,
                            chat_history: List[Dict[str, str]],
                            add_generation_prompt: bool = True) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.llm.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.llm.tokenizer.name_or_path.replace("/", "__")

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.llm.tokenizer.encode(string, **kwargs)

    def _loglikelihood_tokens(self, requests,
                              **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError()

    def loglikelihood_rolling(self,
                              requests,
                              disable_tqdm: bool = False) -> List[float]:
        raise NotImplementedError()

    def _get_sampling_params(self, gen_kwargs: dict) -> SamplingParams:
        params_mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "max_gen_toks": "max_tokens",
            "until": "stop",
        }
        if self.sampling_params is None:
            sampling_params = SamplingParams()
        else:
            sampling_params = copy.deepcopy(self.sampling_params)
        for lm_eval_key, trtllm_key in params_mapping.items():
            value = gen_kwargs.pop(lm_eval_key, None)
            if value is not None:
                setattr(sampling_params, trtllm_key, value)
        return sampling_params

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        profiler.start("trtllm exec")
        results = []
        for request in tqdm(requests,
                            desc="Submitting requests",
                            disable=disable_tqdm):
            prompt, gen_kwargs = request.args
            sampling_params = self._get_sampling_params(gen_kwargs)
            output = self.llm.generate_async(prompt,
                                             sampling_params=sampling_params,
                                             streaming=self.streaming)
            results.append(output)

        outputs = []
        for output in tqdm(results,
                           desc="Fetching responses",
                           disable=disable_tqdm):
            outputs.append(output.result())

        profiler.stop("trtllm exec")
        elapsed_time = profiler.elapsed_time_in_sec("trtllm exec")
        logger.info(f"TRTLLM execution time: {elapsed_time:.3f} seconds.")
        profiler.reset("trtllm exec")

        return [output.outputs[0].text for output in outputs]


class LmEvalEvaluator(Evaluator):

    def __init__(self,
                 task_name: str,
                 dataset_path: str = None,
                 num_samples: Optional[int] = None,
                 random_seed: int = 0,
                 apply_chat_template: bool = False,
                 fewshot_as_multiturn: bool = False,
                 system_prompt: Optional[str] = None):
        try:
            import lm_eval
        except ImportError as e:
            raise ImportError(
                f"Evaluation task {self.__class__.__name__} requires `lm_eval`. "
                "Please install the package first, e.g., `pip install lm_eval`."
            ) from e
        import lm_eval.tasks
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         fewshot_as_multiturn=fewshot_as_multiturn,
                         system_prompt=system_prompt)
        self.task_name = task_name
        self.dataset_path = dataset_path
        self.num_samples = num_samples

        task_manager = lm_eval.tasks.TaskManager(
            include_path=f"{os.path.dirname(__file__)}/lm_eval_tasks")
        with self._patch_lm_eval():
            self.task_dict = lm_eval.tasks.get_task_dict(
                task_name, task_manager=task_manager)
        # Few-shot random seed
        self.task_dict[self.task_name].set_fewshot_seed(random_seed)
        # Shuffle dataset
        data = self.task_dict[self.task_name].dataset
        for split in data.keys():
            data[split] = data[split].shuffle(random_seed)

    @contextmanager
    def _patch_lm_eval(self):
        if self.dataset_path is None:
            yield
            return

        import lm_eval
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

    def evaluate(self,
                 llm: Union[LLM, PyTorchLLM],
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False,
                 scores_filter: str = None) -> float:
        import lm_eval
        results = lm_eval.evaluate(
            lm=LmEvalWrapper(llm, sampling_params, streaming),
            task_dict=self.task_dict,
            limit=self.num_samples,
            apply_chat_template=self.apply_chat_template,
            fewshot_as_multiturn=self.fewshot_as_multiturn,
            system_instruction=self.system_prompt)
        # Normalize scores to range 0~100
        scores = results["results"][self.task_name]
        for metric in scores.keys():
            if isinstance(scores[metric], (float, int)):
                scores[metric] *= 100
        logger.info(
            f"lm-eval {self.task_name} results (scores normalized to range 0~100):\n{lm_eval.utils.make_table(results)}"
        )
        if scores_filter is not None:
            result_acc = results["results"][self.task_name][scores_filter]
            logger.info(
                f"lm-eval {self.task_name} {scores_filter} accuracy: {result_acc:.2f}"
            )
        else:
            result_acc = np.mean(
                [acc for m, acc in scores.items() if "_stderr" not in m])
            logger.info(
                f"lm-eval {self.task_name} average accuracy: {result_acc:.2f}")
        return result_acc

    @classmethod
    def command_harness(cls, ctx, **kwargs):
        llm: Union[LLM, PyTorchLLM] = ctx.obj
        evaluator = cls(dataset_path=kwargs.pop("dataset_path", None),
                        num_samples=kwargs.pop("num_samples", None),
                        random_seed=kwargs.pop("random_seed", 0),
                        apply_chat_template=kwargs.pop("apply_chat_template",
                                                       False),
                        fewshot_as_multiturn=kwargs.pop("fewshot_as_multiturn",
                                                        False),
                        system_prompt=kwargs.pop("system_prompt", None))
        sampling_params = SamplingParams(
            max_tokens=kwargs.pop("max_output_length"),
            truncate_prompt_tokens=kwargs.pop("max_input_length"))
        evaluator.evaluate(llm, sampling_params)
        llm.shutdown()


class GSM8K(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gsm8k", **kwargs)

    @click.command("gsm8k")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GSM8K dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option("--fewshot_as_multiturn",
                  is_flag=True,
                  default=False,
                  help="Apply fewshot as multiturn.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=256,
                  help="Maximum generation length.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        if kwargs.get("fewshot_as_multiturn", False):
            assert kwargs.get(
                "apply_chat_template", False
            ), "apply_chat_template must be True when fewshot_as_multiturn is True"
        GSM8K.command_harness(ctx, **kwargs)


class GPQADiamond(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gpqa_diamond_cot_zeroshot_aa", **kwargs)

    @click.command("gpqa_diamond")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GPQA dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=32768,
                  help="Maximum generation length.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQADiamond.command_harness(ctx, **kwargs)


class GPQAMain(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gpqa_main_cot_zeroshot_aa", **kwargs)

    @click.command("gpqa_main")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GPQA dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=32768,
                  help="Maximum generation length.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQAMain.command_harness(ctx, **kwargs)


class GPQAExtended(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("gpqa_extended_cot_zeroshot_aa", **kwargs)

    @click.command("gpqa_extended")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to GPQA dataset. "
                  "If unspecified, the dataset is downloaded from HF hub.")
    @click.option(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to run the evaluation; None means full dataset."
    )
    @click.option("--random_seed",
                  type=int,
                  default=0,
                  help="Random seed for dataset processing.")
    @click.option("--apply_chat_template",
                  is_flag=True,
                  default=False,
                  help="Whether to apply chat template.")
    @click.option("--system_prompt",
                  type=str,
                  default=None,
                  help="System prompt.")
    @click.option("--max_input_length",
                  type=int,
                  default=4096,
                  help="Maximum prompt length.")
    @click.option("--max_output_length",
                  type=int,
                  default=32768,
                  help="Maximum generation length.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQAExtended.command_harness(ctx, **kwargs)
