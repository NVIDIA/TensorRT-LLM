import argparse
import json

from tensorrt_llm.scaffolding.controller import (MajorityVoteController,
                                                 NativeGenerationController)
from tensorrt_llm.scaffolding.math_utils import *
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import TRTLLMWorker


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generation_dir',
        type=str,
        default=
        "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"
    )
    # https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/aime24/test.jsonl
    parser.add_argument('--jsonl_file', type=str, default='./test.jsonl')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--sample_num', type=int, default=10)
    args = parser.parse_args()
    return args


def load_test_file(jsonl_file: str):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    args = parse_arguments()
    workers = {}

    llm_worker = TRTLLMWorker.init_with_new_llm(args.generation_dir,
                                                backend="pytorch",
                                                max_batch_size=32,
                                                max_num_tokens=4096,
                                                temperature=0.9)

    prototype_generation_controller = NativeGenerationController(
        custom_sampling_params={
            "max_tokens": 4096,
            "top_p": 0.9,
        })
    workers[NativeGenerationController.WorkerTag.GENERATION] = llm_worker

    prototype_majority_vote_controller = MajorityVoteController(
        generation_controller=prototype_generation_controller,
        default_sample_num=args.sample_num,
    )

    llm = ScaffoldingLlm(
        prototype_majority_vote_controller,
        workers=workers,
    )
    test_dataset = load_test_file(args.jsonl_file)
    total_count = 0
    correct_count = 0
    controller_name = "MajorityVoteController"

    prompts = []
    for i in range(len(test_dataset)):
        test_case = test_dataset[i]
        prompts.append(test_case["problem"])

    results = llm.generate(prompts)

    for i in range(len(results)):
        result = results[i]
        test_case = test_dataset[i]
        ref_answer = int(test_case["answer"])
        result.result()
        output = result.output
        extracted_answer = extract_answer(output.output_str)
        try:
            # print(f"[QUESTION]:\n{prompt}\n\n[OUTPUT]\n\n{output.output_str}\n\n")
            answer = int(extracted_answer)
            print(f'Answer={answer}, reference={ref_answer}')
            if answer == ref_answer:
                correct_count += 1
        except:
            print(f'extracted_answer={extracted_answer}, not integer.')
        total_count += 1
    print(
        f'Controller {controller_name} Accuracy: {correct_count} out of {total_count}'
    )
    if args.threshold is not None:
        assert correct_count >= args.threshold * total_count, \
                f'Accuracy check failed with {correct_count}/{total_count} < {args.threshold}'
        print(f'Accuracy check passed with threshold={args.threshold}')
    print(f'main shutting down...')
    llm.shutdown()
    llm_worker.shutdown()
    print(f'main shut down done')


if __name__ == '__main__':
    main()
