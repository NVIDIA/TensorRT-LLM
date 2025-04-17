import argparse

from stream_generation_controller import NativeStreamGenerationController
from stream_generation_task import (StreamGenerationTask,
                                    stream_generation_handler)

from tensorrt_llm.scaffolding import ScaffoldingLlm, TRTLLMWorker


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generation_dir',
        type=str,
        default=
        "/home/scratch.trt_llm_data/llm-models/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument('--run_type', type=str, default='original')
    args = parser.parse_args()
    return args


def test(prompts, proposer_worker):
    prototype_controller = NativeStreamGenerationController()

    llm = ScaffoldingLlm(
        prototype_controller,
        {NativeStreamGenerationController.WorkerTag.STREAM: proposer_worker},
    )
    results = llm.generate(prompts)
    for result in results:
        print(result.output.output_str)
    print(f'test main shutting down...')
    llm.shutdown()
    print(f'test worker shutting down...')
    proposer_worker.shutdown()
    print(f'test main shut down done')


def test_step(prompts, proposer_worker):
    prototype_controller = NativeStreamGenerationController()
    prototype_controller.set_stream_step(20)

    llm = ScaffoldingLlm(
        prototype_controller,
        {NativeStreamGenerationController.WorkerTag.STREAM: proposer_worker},
    )
    results = llm.generate(prompts)
    for result in results:
        print(result.output.output_str)
    print(f'test step main shutting down...')
    llm.shutdown()
    print(f'test step worker shutting down...')
    proposer_worker.shutdown()
    print(f'test step main shut down done')


def test_cancel(prompts, proposer_worker):
    prototype_controller = NativeStreamGenerationController()
    prototype_controller.set_output_threshold(200)

    llm = ScaffoldingLlm(
        prototype_controller,
        {NativeStreamGenerationController.WorkerTag.STREAM: proposer_worker},
    )
    results = llm.generate(prompts)
    for result in results:
        print(result.output.output_str)
    print(f'test cancel main shutting down...')
    llm.shutdown()
    print(f'test cancel worker shutting down...')
    proposer_worker.shutdown()
    print(f'test cancel main shut down done')


def main():
    args = parse_arguments()

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]
    llm_worker = TRTLLMWorker.init_with_new_llm(args.generation_dir,
                                                backend="pytorch",
                                                max_batch_size=32,
                                                max_num_tokens=4096,
                                                temperature=0.9)

    print(f'main llm worker init done')
    llm_worker.register_task_handler(StreamGenerationTask,
                                     stream_generation_handler)
    if args.run_type == 'original':
        test(prompts, llm_worker)
    elif args.run_type == 'step':
        test_step(prompts, llm_worker)
    elif args.run_type == 'cancel':
        test_cancel(prompts, llm_worker)


if __name__ == "__main__":
    main()
