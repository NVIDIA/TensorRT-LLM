import argparse
import asyncio
import json

from tensorrt_llm.scaffolding import (GenerationTokenCounter,
                                      MajorityVoteController,
                                      NativeGenerationController,
                                      ScaffoldingBenchRequest, ScaffoldingLlm,
                                      TRTLLMWorker, async_scaffolding_benchmark,
                                      extract_answer_from_boxed)


def parse_arguments():
    parser = argparse.ArgumentParser()
    # .e.g. DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    # https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation/data/aime24/test.jsonl
    parser.add_argument('--jsonl_file', type=str, default='./test.jsonl')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--sample_num', type=int, default=10)
    parser.add_argument('--concurrency', type=int, default=None)
    parser.add_argument('--static_with_benchmark', action='store_true')
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

    llm_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
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

    if args.static_with_benchmark or args.concurrency:
        if args.concurrency == None:
            args.concurrency = 1

        if args.static_with_benchmark:
            task_collection_types = {"token_counter": GenerationTokenCounter}

        requests = [
            ScaffoldingBenchRequest(prompt=prompt) for prompt in prompts
        ]

        results, requests_execution_time, total_time = asyncio.run(
            async_scaffolding_benchmark(llm, task_collection_types, requests,
                                        args.concurrency))
    else:
        results = llm.generate(prompts)

    print(f'main shutting down...')
    llm.shutdown()
    llm_worker.shutdown()
    print(f'main shut down done')

    for i in range(len(results)):
        result = results[i]
        test_case = test_dataset[i]
        ref_answer = int(test_case["answer"])
        output = result.outputs[0]
        extracted_answer = extract_answer_from_boxed(output.text)
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
        accuracy = correct_count / total_count
        if accuracy < args.threshold:
            print(
                f'Accuracy check failed with {correct_count}/{total_count} < {args.threshold}'
            )
        else:
            print(f'Accuracy check passed with threshold={args.threshold}')

    if args.static_with_benchmark:
        print(f'Total time: {total_time}')
        print(
            f'Average requests execution time: {sum(requests_execution_time) / len(requests_execution_time)}'
        )
        total_token_count = 0
        for result in results:
            total_token_count += result.task_collections[
                "token_counter"].generation_token_count
        print(f'Average output token count: {total_token_count / len(results)}')


if __name__ == '__main__':
    main()
