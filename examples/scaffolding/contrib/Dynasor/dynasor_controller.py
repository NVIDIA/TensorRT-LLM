from enum import Enum
from typing import List

from evaluator import equal_group

from tensorrt_llm.scaffolding.controller import Controller, ScaffoldingOutput
from tensorrt_llm.scaffolding.task import GenerationTask


class DynasorController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"

    # Certainty_threshold and chunk_size controls the compute saving level
    # Decreasing the certainty_threshold and chunk_size will save tokens but may risk at compromising accuracy.
    def __init__(self, max_tokens=8192, certainty_threshold=3, chunk_size=64):
        super().__init__()
        self.max_tokens = max_tokens
        self.certainty_threshold = certainty_threshold
        self.chunk_size = chunk_size

    def generate(self, prompt: str, **kwargs) -> ScaffoldingOutput:
        proposer_task = GenerationTask.create_from_prompt(prompt)
        proposer_task.max_tokens = self.chunk_size
        proposer_task.temperature = 0.6
        proposer_task.top_p = 0.95
        proposer_task.worker_tag = self.WorkerTag.GENERATION

        probe_task = GenerationTask.create_from_prompt(prompt)
        probe_task.max_tokens = 20
        probe_task.temperature = 0.6
        probe_task.top_p = 0.95
        probe_task.worker_tag = self.WorkerTag.GENERATION

        result = yield from self.process([proposer_task, probe_task], **kwargs)
        scaffolding_output = ScaffoldingOutput()
        scaffolding_output.output_str = result
        return scaffolding_output

    def process(self, tasks: List[GenerationTask], **kwargs):
        proposer_task, probe_task = tasks
        current_prompt = proposer_task.input_str

        probe_answers = []
        probe_responses = []
        probeing_suffix: str = (
            "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
        )
        uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]
        for _ in range(0, self.max_tokens, self.chunk_size):
            proposer_task.input_str = current_prompt
            probe_task.input_str = current_prompt + probeing_suffix

            yield [probe_task]

            probe_text = probe_task.output_str
            answer = self.obtain_answer(probe_text)
            probe_answers.append(answer)
            probe_responses.append(probe_text)
            probe_certain_count = [
                not any(word in res.lower() for word in uncertain_words)
                for res in probe_responses[-self.certainty_threshold:]
            ]

            # if generation in current round is considered to be confident enough, return it
            if (equal_group(probe_answers[-self.certainty_threshold:])
                    and self.count_not_empty(
                        probe_answers[-self.certainty_threshold:])
                    == self.certainty_threshold
                    and sum(probe_certain_count) == self.certainty_threshold):
                if "</think>" in current_prompt:
                    return (
                        current_prompt +
                        "\n\n... Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
                        + probe_answers[-1] + "}\n\\]")

                else:
                    return (
                        current_prompt +
                        "\n\n...</think>\n Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
                        + probe_answers[-1] + "}\n\\]")

            # if not confident, do another round of generation
            yield [proposer_task]

            current_prompt += proposer_task.output_str

        # if exceed max_tokens
        return current_prompt

    @staticmethod
    def obtain_answer(s):
        # Find first unpaired } by counting { and }
        stack = []
        for i, c in enumerate(s):
            if c == "{":
                stack.append(c)
            elif c == "}":
                if not stack:  # No matching { found
                    return s[:i]
                stack.pop()
        return ""

    @staticmethod
    def count_not_empty(answers):
        return sum(1 for answer in answers if answer != "")
