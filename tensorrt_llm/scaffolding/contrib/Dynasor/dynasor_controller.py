from enum import Enum
from typing import List

from transformers import AutoTokenizer

from tensorrt_llm.scaffolding import Controller, GenerationTask

from .evaluator import equal_group


class DynasorGenerationController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation_with_dynasor_cot"

    # Certainty_threshold and chunk_size controls the compute saving level
    # Decreasing the certainty_threshold and chunk_size will save tokens but may risk at compromising accuracy.
    def __init__(
        self,
        generation_dir,
        max_tokens=8192,
        certainty_threshold=3,
        chunk_size=64,
        streaming=False,
    ):
        """
        Initializes the controller with parameters controlling token limits and certainty thresholds.

        Args:
            max_tokens (int): Maximum number of tokens to generate in total.
            certainty_threshold (int): Number of consecutive identical and confident probe answers
                                       required to consider the generation as certain.
            chunk_size (int): Number of tokens to generate per proposal round.
        """
        super().__init__()
        self.generation_dir = generation_dir
        self.max_tokens = max_tokens
        self.certainty_threshold = certainty_threshold
        self.chunk_size = chunk_size
        self.uncertain_words = ["wait", "hold", "but", "okay", "no", "hmm"]
        self.probe_suffix = "... Oh, I suddenly got the answer to the whole problem, **Final Answer**\n\n\\[ \\boxed{"
        self.answer_suffix = "\n\n... Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
        self.answer_suffix_with_marker = "\n\n...</think>\n Oh, I have got the answer to the whole problem\n**Final Answer:**\n\\[\n \\boxed{"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.generation_dir,
            legacy=False,
            padding_side='left',
            truncation_side='left',
            trust_remote_code=False,
            use_fast=True,
        )
        self.streaming = streaming

    def process(self, tasks: List[GenerationTask], **kwargs):
        """
        Process the generation task using an iterative approach:
        1. Generate a probe response with an extra suffix to simulate chain-of-thought.
        2. Evaluate the probe response to extract a potential answer.
        3. Check for consistency over several rounds (using certainty_threshold).
        4. If consistent, finalize the answer and return. Otherwise, continue appending new proposals.

        Args:
            tasks (List[GenerationTask]): A list of generation tasks to process.
                                          The first task is assumed to hold the initial prompt.

        Yields:
            A list of GenerationTask objects to be executed in further processing steps.
        """
        # Start with the initial prompt provided by the first task.
        initial_prompt = tasks[0].input_str

        proposer_task = GenerationTask()
        proposer_task.max_tokens = self.chunk_size
        proposer_task.temperature = 0.6
        proposer_task.top_p = 0.95
        proposer_task.worker_tag = self.WorkerTag.GENERATION
        proposer_task.streaming = self.streaming

        probe_task = GenerationTask()
        probe_task.max_tokens = 20
        probe_task.temperature = 0.6
        probe_task.top_p = 0.95
        probe_task.worker_tag = self.WorkerTag.GENERATION
        probe_task.streaming = False

        probe_answers = []
        probe_responses = []

        initial_prompt_token_num = len(
            self.tokenizer.encode(initial_prompt, add_special_tokens=False))
        probe_suffix_token_num = len(
            self.tokenizer.encode(self.probe_suffix, add_special_tokens=False))

        current_prompt = initial_prompt

        # Iterate over generation rounds until the maximum tokens limit is reached.
        # Make sure length of prefilling is always smaller than the max_tokens in TRTLLMWorker.init_with_new_llm
        # Otherwise it will through an assertion fail, stated in issue #3576
        for _ in range(initial_prompt_token_num + probe_suffix_token_num,
                       self.max_tokens, self.chunk_size):
            proposer_task.input_str = current_prompt
            probe_task.input_str = current_prompt + self.probe_suffix

            # For the probe task, append the suffix to force a chain-of-thought leading to an answer.
            yield [proposer_task, probe_task]

            # Retrieve the output from the probe task.
            probe_text = probe_task.output_str

            # Extract the potential answer from the probe response.
            answer = self.obtain_answer(probe_text)
            probe_answers.append(answer)
            probe_responses.append(probe_text)

            # Determine if the last few probe responses are considered confident enough.
            # A response is flagged as confident if it does not contain any of the uncertain words.
            probe_certain_count = [
                not any(word in res.lower() for word in self.uncertain_words)
                for res in probe_responses[-self.certainty_threshold:]
            ]

            # Check if the last 'certainty_threshold' probe answers are identical (by equal_group)
            # and they are not empty, and all responses are confident.
            if (equal_group(probe_answers[-self.certainty_threshold:])
                    and self.count_not_empty(
                        probe_answers[-self.certainty_threshold:])
                    == self.certainty_threshold
                    and sum(probe_certain_count) == self.certainty_threshold):
                tasks[0].result = probe_task.result
                # If the current prompt indicates the chain-of-thought phase has ended, use one type of suffix.
                if "</think>" in current_prompt:
                    tasks[0].output_str = (current_prompt + self.answer_suffix +
                                           probe_answers[-1] + "}\n\\]")
                    return
                else:
                    # Otherwise, use the suffix with marker to transition clearly.
                    tasks[0].output_str = (current_prompt +
                                           self.answer_suffix_with_marker +
                                           probe_answers[-1] + "}\n\\]")
                    return

            # If not confident, do another round of generation
            # Append the newly generated text from the proposer to the current prompt for the next iteration.
            current_prompt += proposer_task.output_str

        # If the maximum token limit is reached without satisfying the certainty condition,
        # output the accumulated prompt as the final output.
        tasks[0].result = proposer_task.result
        tasks[0].output_str = current_prompt
        return

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
