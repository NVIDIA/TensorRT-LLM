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
import json
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import click
import numpy as np
from tqdm import tqdm

import tensorrt_llm.profiler as profiler
from tensorrt_llm.inputs import prompt_inputs

try:
    from lm_eval.api.model import TemplateLM
    from lm_eval.tasks import TaskManager
except ImportError:
    TemplateLM = object

from .. import LLM as PyTorchLLM
from .._tensorrt_engine import LLM
from ..inputs import (ConversationMessage, MultimodalDataTracker,
                      add_multimodal_placeholders, convert_image_mode)
from ..inputs.utils import apply_chat_template as trtllm_apply_chat_template
from ..llmapi import RequestOutput
from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator

# NOTE: lm_eval uses "<image>" as the default image placeholder
# https://github.com/EleutherAI/lm-evaluation-harness/blob/7f04db12d2f8e7a99a0830d99eb78130e1ba2122/lm_eval/models/hf_vlms.py#L25
LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER = "<image>"


class LmEvalWrapper(TemplateLM):

    def __init__(self,
                 llm: Union[LLM, PyTorchLLM],
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None):
        super().__init__()
        self.llm = llm
        self.sampling_params = sampling_params
        self.streaming = streaming
        self.chat_template_kwargs = chat_template_kwargs

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
            **(self.chat_template_kwargs or {}),
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


class MultimodalLmEvalWrapper(LmEvalWrapper):
    """
    Multimodal wrapper for lm-evaluation-harness that handles vision-language models.

    This wrapper extends the base LmEvalWrapper to support multimodal inputs,
    particularly for tasks that require both text and image processing.
    """

    def __init__(self,
                 llm: Union[LLM, PyTorchLLM],
                 sampling_params: Optional[SamplingParams] = None,
                 streaming: bool = False,
                 max_images: int = 999,
                 chat_template_kwargs: Optional[dict[str, Any]] = None):
        """
        Initialize the multimodal wrapper.

        Args:
            llm: The language model instance (either TensorRT or PyTorch)
            sampling_params: Parameters for text generation
            streaming: Whether to use streaming generation
            max_images: Maximum number of images per prompt (currently unlimited in TRT-LLM), set to 999 from lm_eval's default value.
        """
        super().__init__(llm, sampling_params, streaming)

        # NOTE: Required by lm_eval to identify this as a multimodal model
        self.MULTIMODAL = True
        self.max_images = max_images
        self.chat_template_kwargs = chat_template_kwargs
        self.model_type = self._get_model_type(llm)

        # NOTE: In TRT-LLM, currently we do not support interleaved text and image. Instead, we are adding image placeholders at the end of the text or at the beginning of the text.
        # So, until we support interleaved text and image, we set this to False.
        self.interleave = False

    def _get_model_type(self, llm: Union[LLM, PyTorchLLM]) -> str:
        """Extract model type from the model configuration."""
        config_path = os.path.join(llm._hf_model_dir, 'config.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model configuration file not found: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in model configuration file {config_path}: {e}")

        if 'model_type' not in config:
            raise KeyError(
                f"'model_type' key not found in model configuration: {config_path}"
            )

        return config['model_type']

    def apply_chat_template(self,
                            chat_history: List[Dict[str, str]],
                            add_generation_prompt: bool = True) -> str:
        """
        Apply chat template to multimodal conversation history.

        Converts text with image placeholders into structured format expected by
        the multimodal processor.

        Adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/7f04db12d2f8e7a99a0830d99eb78130e1ba2122/lm_eval/models/hf_vlms.py#L225
        """
        mm_placeholder_counts = []
        for i in range(len(chat_history)):
            content = chat_history[i]
            text = content["content"]
            image_count = min(self.max_images,
                              text.count(LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER))

            if self.interleave:
                # TODO: Implement interleaved text and image.
                text.split(LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER)
                ...
            else:
                text = text.replace(LM_EVAL_DEFAULT_IMAGE_PLACEHOLDER, "")

            conv = ConversationMessage(role="user", content=text)
            mm_data_tracker = MultimodalDataTracker(self.model_type)

            # NOTE: Since we already have loaded images, for the placeholder purpose, we add data here.
            for _ in range(image_count):
                mm_data_tracker.add_data("image", None)
            mm_placeholder_count = mm_data_tracker.placeholder_counts()
            if mm_placeholder_count:
                # TODO: This is an assumption of not interleaving text and image. Need to extend to interleaved texts.
                conv["content"] = add_multimodal_placeholders(
                    self.model_type, conv["content"], mm_placeholder_count)
            mm_placeholder_counts.append(mm_placeholder_count)
            chat_history[i] = conv

        output = trtllm_apply_chat_template(
            model_type=self.model_type,
            tokenizer=self.llm.tokenizer,
            processor=self.llm.input_processor.processor,
            conversation=chat_history,
            add_generation_prompt=add_generation_prompt,
            mm_placeholder_counts=mm_placeholder_counts,
            tools=None,
            chat_template_kwargs={
                **(self.chat_template_kwargs or {}),
                "continue_final_message":
                not add_generation_prompt,
            })
        return output

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """
        Generate text responses for multimodal requests.

        This method processes multimodal requests that include both text prompts
        and visual data (images).

        Args:
            requests: List of multimodal generation requests
            disable_tqdm: Whether to disable progress bars

        Returns:
            List of generated text responses
        """
        profiler.start("trtllm exec")
        results = []
        for request in tqdm(requests,
                            desc="Submitting requests",
                            disable=disable_tqdm):

            # NOTE: For now, only this part is different from the original generate_until
            prompt, gen_kwargs, media_data = request.args
            prompt = prompt_inputs(prompt)

            # NOTE: Convert RGBA format to RGB format
            images = [
                convert_image_mode(img, "RGB") for img in media_data["visual"]
            ]
            prompt["multi_modal_data"] = {"image": images}

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
                 system_prompt: Optional[str] = None,
                 is_multimodal: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None):
        try:
            import lm_eval
        except ImportError as e:
            raise ImportError(
                f"Evaluation task {self.__class__.__name__} requires `lm_eval`. "
                "Please install the package first, e.g., `pip install lm_eval`."
            ) from e
        import lm_eval.tasks
        self.MULTIMODAL = is_multimodal
        if self.MULTIMODAL:
            apply_chat_template = True
            logger.info(
                "Chat template automatically enabled for multimodal evaluation."
            )
        super().__init__(random_seed=random_seed,
                         apply_chat_template=apply_chat_template,
                         fewshot_as_multiturn=fewshot_as_multiturn,
                         system_prompt=system_prompt,
                         chat_template_kwargs=chat_template_kwargs)
        self.task_name = task_name
        self.dataset_path = dataset_path
        self.num_samples = num_samples

        task_manager = TaskManager(
            include_path=f"{os.path.dirname(__file__)}/lm_eval_tasks")
        with self._patch_lm_eval():
            self.task_dict = lm_eval.tasks.get_task_dict(
                task_name, task_manager=task_manager)

        # Adopted from https://github.com/EleutherAI/lm-evaluation-harness/blob/7f04db12d2f8e7a99a0830d99eb78130e1ba2122/lm_eval/evaluator.py#L290
        def _adjust_config(task_dict, random_seed):
            adjusted_task_dict = {}
            for task_name, task_obj in task_dict.items():
                if isinstance(task_obj, dict):
                    adjusted_task_dict = {
                        **adjusted_task_dict,
                        **{
                            task_name: _adjust_config(task_obj, random_seed)
                        },
                    }
                else:
                    # NOTE: Few-shot random seed
                    task_obj.set_fewshot_seed(seed=random_seed)
                    adjusted_task_dict[task_name] = task_obj

                    # NOTE: Shuffle dataset
                    data = adjusted_task_dict[task_name].dataset
                    for split in data.keys():
                        data[split] = data[split].shuffle(random_seed)

            return adjusted_task_dict

        self.task_dict = _adjust_config(self.task_dict, random_seed)

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
        lm_cls = MultimodalLmEvalWrapper if self.MULTIMODAL else LmEvalWrapper
        results = lm_eval.evaluate(
            lm=lm_cls(llm,
                      sampling_params=sampling_params,
                      streaming=streaming,
                      chat_template_kwargs=self.chat_template_kwargs),
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
                        system_prompt=kwargs.pop("system_prompt", None),
                        is_multimodal=kwargs.pop("is_multimodal", False),
                        chat_template_kwargs=kwargs.pop("chat_template_kwargs",
                                                        None))
        sampling_params = SamplingParams(
            max_tokens=kwargs.pop("max_output_length"),
            truncate_prompt_tokens=kwargs.pop("max_input_length"),
            stop=kwargs.pop("stop", None))
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
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
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
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
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
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
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
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
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


class MMMU(LmEvalEvaluator):

    def __init__(self, **kwargs):
        super().__init__("mmmu_val", **kwargs)

    @click.command("mmmu")
    @click.option("--dataset_path",
                  type=str,
                  default=None,
                  help="The path to MMMU dataset. "
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
    @click.option(
        "--chat_template_kwargs",
        type=str,
        default=None,
        callback=lambda ctx, param, value: json.loads(value) if value else None,
        help=
        'Chat template kwargs as JSON string, e.g., \'{"thinking_budget": 0}\'')
    @click.option(
        "--system_prompt",
        type=str,
        default=None,
        help=
        "The system prompt to be added on the prompt. If specified, it will add {'role': 'system', 'content': system_prompt} to the prompt."
    )
    @click.option("--max_input_length",
                  type=int,
                  default=8192,
                  help="Maximum prompt length.")
    @click.option(
        "--max_output_length",
        type=int,
        default=
        512,  # NOTE: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmmu/_template_yaml#L13
        help="Maximum generation length.")
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        # NOTE: MMMU is a multimodal task, so we need to set the is_multimodal and apply_chat_template flags to True
        kwargs["is_multimodal"] = True
        kwargs["apply_chat_template"] = True
        kwargs[
            "stop"] = "<|endoftext|>"  # NOTE: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmmu/_template_yaml#L10
        MMMU.command_harness(ctx, **kwargs)
