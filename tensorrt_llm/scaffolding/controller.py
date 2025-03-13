from abc import ABC

from tensorrt_llm.scaffolding.math_utils import (extract_answer,
                                                 get_majority_result)
from tensorrt_llm.scaffolding.task import GenerationTask


class ScaffoldingOutput:

    def __init__(self):
        self.output_str = None


class Controller(ABC):

    def generate(self, prompt: str, **kwargs) -> ScaffoldingOutput:
        raise NotImplementedError

    @staticmethod
    def create_task_from_prompt(prompt):
        generation_task = GenerationTask()
        generation_task.input_str = prompt
        generation_task.skip_tokenizer = False
        generation_task.skip_detokenizer = False
        return generation_task

    @staticmethod
    def create_output_from_generation_task(task: GenerationTask):
        scaffolding_output = ScaffoldingOutput()
        scaffolding_output.output_str = task.output_str
        return scaffolding_output


class SimpleController(Controller):
    # Simple Controller which is just a single autoregressive LLM call.

    def __init__(self):
        super().__init__()

    def generate(self, prompt: str, **kwargs) -> ScaffoldingOutput:
        task = self.create_task_from_prompt(prompt)
        # We only need a single generation task to handle simple case
        yield [task]
        # Should come here with task done and return field in task filled
        return self.create_output_from_generation_task(task)


class BestOfNController(Controller):

    def __init__(self, n: int = 5, custom_sampling_params: dict = None):
        super().__init__()
        self.n = n
        self.custom_sampling_params = custom_sampling_params

    def create_task_from_prompt(self, prompt):
        task = super().create_task_from_prompt(prompt)
        task.custom_sampling_params = self.custom_sampling_params
        return task

    def generate(self, prompt: str, **kwargs) -> ScaffoldingOutput:
        tasks = [self.create_task_from_prompt(prompt) for _ in range(self.n)]
        yield tasks
        return self.select_best_of_n(tasks)

    def select_best_of_n(self, tasks):

        def is_digit(result: str):
            extracted_answer = extract_answer(result)
            if extracted_answer is None:
                return False
            return extracted_answer.isdigit()

        results = [task.output_str for task in tasks]
        final_result, extracted_result = get_majority_result(
            results, result_extractor=extract_answer, result_validator=is_digit)
        if final_result is None:
            final_result = results[0]
        scaffolding_output = ScaffoldingOutput()
        scaffolding_output.output_str = final_result
        return scaffolding_output
