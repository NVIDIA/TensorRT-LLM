from transformers import AutoTokenizer

from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.scaffolding import GenerationTask, TaskStatus, Worker


class PytorchWorker(Worker):

    def __init__(
        self,
        model_path: str,
        max_batch_size: int = 32,
        max_num_tokens: int = 4096,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False,
            trust_remote_code=False,
            use_fast=True,
        )
        self.llm = LLM(
            model_dir=model_path,  # Use model_dir for consistency
            tokenizer=self.tokenizer,  # Pass the tokenizer to LLM
            backend='pytorch',
            pytorch_backend_config={'device': 'cpu'},
            max_batch_size=max_batch_size,
            max_num_tokens=max_num_tokens,
        )

    @classmethod
    def convert_task_params(self, task: GenerationTask) -> SamplingParams:
        sampling_params = SamplingParams(
            max_tokens=task.max_tokens,
            temperature=task.temperature,
            top_p=task.top_p,
            top_k=task.top_k,
            return_context_logits=task.return_context_logits)
        return sampling_params

    async def generation_handler(self, task: GenerationTask) -> TaskStatus:
        sampling_params = self.convert_task_params(task)
        result = await self.llm.generate_async(task.input_str,
                                               sampling_params=sampling_params)

        task.output_tokens = result.outputs[0].token_ids
        task.cumulative_logprob = result.outputs[0].cumulative_logprob
        task.logprobs = result.outputs[0].logprobs
        task.output_str = result.outputs[0].text
        task.context_logits = result.context_logits

        # TODO: error handle
        return TaskStatus.SUCCESS

    def shutdown(self):
        # There is no clean-up needed
        pass

    task_handlers = {GenerationTask: generation_handler}
