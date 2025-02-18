import argparse
import json

from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.scaffolding.controller import BestOfNController
from tensorrt_llm.scaffolding.math_utils import *
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.worker import ProposerWorker, SamplingParams


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

    proposer_worker = ProposerWorker(
        args.generation_dir,
        pytorch_backend_config=PyTorchConfig(
            mixed_decoder=True,
            enable_overlap_scheduler=True,
        ),
        sampling_params=SamplingParams(max_tokens=2048),
    )

    llm = ScaffoldingLlm(
        BestOfNController,
        {"custom_sampling_params": {
            "max_tokens": 4096,
            "top_p": 0.9,
        }},
        {'generation': proposer_worker},
    )
    test_dataset = load_test_file(args.jsonl_file)
    total_count = 0
    correct_count = 0
    controller_name = "BestOfNController"

    results = []
    for i in range(len(test_dataset)):
        test_case = test_dataset[i]
        prompt = test_case["problem"]
        result = llm.generate_async(prompt)
        results.append(result)
    for i in range(len(results)):
        result = results[i]
        test_case = test_dataset[i]
        prompt = test_case["problem"]
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
        assert correct_count >= args.threshold * total_count
        print(f'Accuracy check passed with threshold={args.threshold}')
    print(f'main shutting down...')
    llm.shutdown()
    proposer_worker.shutdown()
    print(f'main shut down done')


if __name__ == '__main__':
    main()
