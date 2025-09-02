import argparse
import json

from tensorrt_llm.scaffolding import (NativeGenerationController,
                                      NativeRewardController, ScaffoldingLlm,
                                      TOTController, TRTLLMWorker)


def load_test_file(jsonl_file: str):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--reward_model_dir', type=str)
    parser.add_argument('--jsonl_file', type=str, default='./test.jsonl')
    parser.add_argument('--max_depth', type=int, default=3)
    parser.add_argument('--max_iterations', type=int, default=15)
    parser.add_argument('--num_thoughts_per_step', type=int, default=3)
    parser.add_argument('--selection_strategy',
                        type=str,
                        default="best",
                        choices=["best", "vote", "random"])
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    workers = {}

    # Initialize generation worker
    gen_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=16,
        max_num_tokens=4096,
        kv_cache_free_gpu_memory_fraction=0.4)
    workers[NativeGenerationController.WorkerTag.GENERATION] = gen_worker

    # Initialize reward worker if provided
    reward_controller = None
    if args.reward_model_dir:
        reward_worker = TRTLLMWorker.init_with_new_llm(
            args.reward_model_dir,
            backend="pytorch",
            max_batch_size=8,
            max_num_tokens=2048,
            kv_cache_free_gpu_memory_fraction=0.2)
        workers[NativeRewardController.WorkerTag.REWARD] = reward_worker
        reward_controller = NativeRewardController()

    # Create generation controller
    generation_controller = NativeGenerationController(sampling_params={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.9,
    })

    # Create TOT controller
    controller = TOTController(generation_controller=generation_controller,
                               reward_controller=reward_controller,
                               max_depth=args.max_depth,
                               max_iterations=args.max_iterations,
                               num_thoughts_per_step=args.num_thoughts_per_step,
                               selection_strategy=args.selection_strategy)

    llm = ScaffoldingLlm(controller, workers=workers)

    # Load problems from JSONL
    test_dataset = load_test_file(args.jsonl_file)
    prompts = []
    ref_answers = []
    for case in test_dataset:
        prompt = case.get("problem") or case.get("question")
        if prompt is None:
            continue
        prompts.append(prompt)
        ref_answers.append(case.get("answer"))

    for i, problem in enumerate(prompts):
        print(f"\nProblem {i+1}: {problem}")
        if i < len(ref_answers) and ref_answers[i] is not None:
            print(f"Reference Answer: {ref_answers[i]}")

        try:
            results = llm.generate([problem])

            for result in results:
                print(f"TOT Solution:")
                print(result.outputs[0].text)
                print(f"Tokens generated: {len(result.outputs[0].token_ids)}")

        except Exception as e:
            print(e)

    llm.shutdown(shutdown_workers=True)
    print(f'main shut down done')


if __name__ == "__main__":
    main()
