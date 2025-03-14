from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

from transformers import AutoTokenizer

from tensorrt_llm import bindings as tllm
from tensorrt_llm._torch.pyexecutor.config import (PyTorchConfig,
                                                   update_executor_config)
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.executor import GenerationExecutor, GenerationRequest
from tensorrt_llm.sampling_params import SamplingParams

from .task import GenerationTask, Task, TaskStatus

ExecutorCls = GenerationExecutor


class Worker(ABC):

    @abstractmethod
    async def run_task(self, task: Task) -> TaskStatus:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self):
        self.shutdown()


class ProposerWorker(Worker):

    def __init__(self,
                 model_dir: str,
                 *,
                 pytorch_backend_config: PyTorchConfig = None,
                 sampling_params: SamplingParams = None,
                 max_batch_size: int = 64,
                 max_num_tokens: int = 8192,
                 max_seq_len: int = 16384):
        self.pytorch_backend_config = pytorch_backend_config
        self.executor, self.tokenizer = self._create_executor(
            model_dir, max_batch_size, max_num_tokens, max_seq_len)
        # TODO: enable Top-P or Top-K Sampling for Best-Of-N
        self.sampling_params = self._prepare_sampling_params(sampling_params)

    def _create_executor(self,
                         model_dir: str,
                         max_batch_size: int = 64,
                         max_num_tokens: int = 8192,
                         max_seq_len: int = 16384):
        # TODO: maybe need common interface for create pyExecutor with LLM.
        scheduler_config = tllm.executor.SchedulerConfig()
        kv_cache_config = tllm.executor.KvCacheConfig(enable_block_reuse=True)
        executor_config = tllm.executor.ExecutorConfig(
            max_beam_width=1,
            scheduler_config=scheduler_config,
            batching_type=tllm.executor.BatchingType.INFLIGHT,
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens)
        executor_config.kv_cache_config = kv_cache_config
        build_config = BuildConfig()
        build_config.max_seq_len = max_seq_len

        update_executor_config(
            executor_config,
            backend='pytorch',
            pytorch_backend_config=self.pytorch_backend_config,
            build_config=build_config,
            hf_model_dir=model_dir,
            trt_engine_dir=None,
        )
        executor = ExecutorCls.create(None, executor_config)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=False,
            use_fast=True,
        )

        return executor, tokenizer

    def _prepare_sampling_params(
        self,
        sampling_params: Optional[SamplingParams] = None,
    ) -> SamplingParams:
        """From LLM._prepare_sampling_params"""
        if sampling_params is None:
            if self.tokenizer is None:
                raise ValueError(
                    "tokenizer is required to initialize a default sampling_params, or you can explicitly specify a sampling_params"
                )
            sampling_param = SamplingParams(end_id=self.tokenizer.eos_token_id,
                                            pad_id=self.tokenizer.pad_token_id,
                                            max_tokens=4096)

        if not isinstance(sampling_params, SamplingParams):
            raise TypeError(
                f"The sampling_params must be type SamplingParams or None, but got {type(sampling_params)}"
            )

        if sampling_params.end_id is None and self.tokenizer is None:
            raise ValueError(
                "tokenizer is required to reset end_id if it is None, or you can explicitly specify the end_id for sampling_params"
            )

        # TODO(zhenhuanc): sync this with LLM._prepare_sampling_params
        if sampling_params.stop is None and self.tokenizer is not None:
            sampling_params.stop = []
            vocab = self.tokenizer.get_vocab()
            # Many models' eos_token_id is not aligned with tokenizer.eos_token, so we add stop words here.
            # Sometimes llama3.1-intruct will use "\r\n\r\n\n" as end token, not sure should we add it here
            for token in [
                    "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>",
                    "<｜end▁of▁sentence｜>"
            ]:
                if token in vocab and token != self.tokenizer.eos_token:
                    sampling_params.stop.append(token)
        sampling_params._setup(self.tokenizer)
        return sampling_params

    def _combine_sampling_params(self, base: SamplingParams, custom: dict):
        base = deepcopy(base)
        for key, value in custom.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base

    def _create_generation_request_from_task(
            self, task: GenerationTask) -> GenerationRequest:
        if not task.skip_tokenizer:
            task.input_tokens = self.tokenizer.encode(task.input_str)
        else:
            assert task.input_tokens
        if hasattr(task, "custom_sampling_params"
                   ) and task.custom_sampling_params is not None:
            sampling_params = self._combine_sampling_params(
                self.sampling_params, task.custom_sampling_params)
        else:
            sampling_params = self.sampling_params

        generation_request = GenerationRequest(
            prompt_token_ids=task.input_tokens, sampling_params=sampling_params)
        return generation_request

    async def run_task(self, task: Task) -> TaskStatus:
        assert isinstance(
            task, GenerationTask
        ), 'Only GenerationTask(role="generation") should be dispatched to ProposerWorker'
        assert task.type == 'generate', 'ProposerWorker only supports generation tasks with type "generate"'
        generation_request = self._create_generation_request_from_task(task)
        result = self.executor.submit(generation_request)
        await result.aresult()
        task.output_tokens = result.outputs[0].token_ids
        task.cumulative_logprob = result.outputs[0].cumulative_logprob
        task.logprobs = result.outputs[0].logprobs
        task.output_str = None
        if not task.skip_detokenizer:
            task.output_str = self.tokenizer.decode(task.output_tokens)
        # TODO: handle status
        status = TaskStatus()
        return status

    def shutdown(self):
        if self.executor:
            self.executor.shutdown()
            self.executor = None
