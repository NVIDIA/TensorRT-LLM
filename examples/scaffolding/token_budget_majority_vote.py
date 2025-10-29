import argparse
import copy
from typing import List

from tensorrt_llm.scaffolding import (Controller, GenerationTokenCounter,
                                      NativeGenerationController,
                                      ParallelProcess, ScaffoldingLlm, Task,
                                      TRTLLMWorker, extract_answer_from_boxed,
                                      get_digit_majority_vote_result,
                                      with_task_collection)


@with_task_collection("token_counter", GenerationTokenCounter)
class TokenBudgetMajorityVoteController(Controller):

    def __init__(self, generation_controller: Controller, token_budget: int,
                 sumple_num_per_turn: int):
        super().__init__()
        self.generation_controller = generation_controller
        self.token_budget = token_budget
        self.sumple_num_per_turn = sumple_num_per_turn

    def clone(self):
        generation_controller = self.generation_controller.clone()
        return TokenBudgetMajorityVoteController(generation_controller,
                                                 self.token_budget,
                                                 self.sumple_num_per_turn)

    def process(self, tasks: List[Task], **kwargs):
        candidates = []
        # use GenerationTokenCounter to get the total token count from this controller
        while self.task_collections[
                "token_counter"].generation_token_count < self.token_budget:
            sample_num = self.sumple_num_per_turn
            generation_controllers = [
                self.generation_controller.clone() for _ in range(sample_num)
            ]
            tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
            generation_kwargs_list = [
                copy.deepcopy(kwargs) for _ in range(sample_num)
            ]

            yield ParallelProcess(generation_controllers, tasks_list,
                                  generation_kwargs_list)

            for task_list in tasks_list:
                candidates.extend([task.output_str for task in task_list])

        result = self.majority_vote(candidates, **kwargs)
        print(
            'final token count: ',
            str(self.task_collections["token_counter"].generation_token_count))

        assert isinstance(result, str), "majority_vote failed"
        # The task returned by majority vote does not have output_tokens and logits.
        tasks[0].output_str = result

    def majority_vote(self, candidates: List[str], **kwargs) -> str:
        return get_digit_majority_vote_result(candidates)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # .e.g. DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    parser.add_argument('--token_budget', type=int, default=30384)
    parser.add_argument('--sumple_num_per_turn', type=int, default=3)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    workers = {}

    llm_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        max_batch_size=32,
        max_num_tokens=4096,
    )

    prototype_generation_controller = NativeGenerationController(
        sampling_params={
            "max_tokens": 4096,
            "top_p": 0.9,
            "temperature": 0.9,
        })
    workers[NativeGenerationController.WorkerTag.GENERATION] = llm_worker

    prototype_majority_vote_controller = TokenBudgetMajorityVoteController(
        generation_controller=prototype_generation_controller,
        token_budget=args.token_budget,
        sumple_num_per_turn=args.sumple_num_per_turn,
    )

    llm = ScaffoldingLlm(
        prototype_majority_vote_controller,
        workers=workers,
    )
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n"

    result = llm.generate(prompt)
    extracted_answer = extract_answer_from_boxed(result.output.output_str)
    print(f'extracted_answer={extracted_answer}')

    llm.shutdown(shutdown_workers=True)
    print(f'main shut down done')


if __name__ == '__main__':
    main()
