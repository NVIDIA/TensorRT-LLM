import argparse
import asyncio

from tensorrt_llm.scaffolding.contrib.PytorchCPU import PytorchWorker
from tensorrt_llm.scaffolding.native_controller import \
    NativeGenerationController
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm


def parse_arguments():
    parser = argparse.ArgumentParser()
    # .e.g. DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    parser.add_argument('--run_async', action='store_true')
    args = parser.parse_args()
    return args


def test_sync(prompts, proposer_worker):
    prototype_controller = NativeGenerationController(
        sampling_params={"temperature": 0.9})

    llm = ScaffoldingLlm(
        prototype_controller,
        {NativeGenerationController.WorkerTag.GENERATION: proposer_worker},
    )
    results = llm.generate(prompts)
    for result in results:
        print(result.output.output_str)
    print(f'main shutting down...')
    llm.shutdown()
    print(f'worker shutting down...')
    proposer_worker.shutdown()
    print(f'main shut down done')


def test_async(prompt, proposer_worker):

    async def test_async_func(prompt, proposer_worker):
        prototype_controller = NativeGenerationController(
            sampling_params={"temperature": 0.9})
        llm = ScaffoldingLlm(
            prototype_controller,
            {NativeGenerationController.WorkerTag.GENERATION: proposer_worker},
        )

        future = llm.generate_async(prompt)

        result = await future.aresult()
        print(result.output.output_str)

        print(f'main shutting down...')
        llm.shutdown()
        print(f'worker shutting down...')
        proposer_worker.shutdown()
        print(f'main shut down done')

    asyncio.run(test_async_func(prompt, proposer_worker))


def main():
    args = parse_arguments()

    prompts = [
        "Anton sold GPUs to 48 of his friends in April, and then he sold half as many GPUs in May. How many GPUs did Anton sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]

    llm_worker = PytorchWorker(
        args.model_dir,
        max_batch_size=32,
        max_num_tokens=4096,
    )

    if args.run_async:
        test_async(prompts[0], llm_worker)
    else:
        test_sync(prompts, llm_worker)


if __name__ == "__main__":
    main()
