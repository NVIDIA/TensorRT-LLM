import argparse
import asyncio

from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.scaffolding.controller import SimpleController
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
    parser.add_argument('--verifier_dir', type=str, default=None)
    parser.add_argument('--run_async', action='store_true')
    args = parser.parse_args()
    return args


def test_sync(prompts, proposer_worker):
    llm = ScaffoldingLlm(
        SimpleController,
        {},
        {'generation': proposer_worker},
    )
    results = llm.generate(prompts)
    for result in results:
        print(result.output.output_str)
    print(f'main shutting down...')
    llm.shutdown()
    print(f'worker shutting down...')
    proposer_worker.shutdown()
    print(f'main shut down done')


def test_async(prompts, proposer_worker):

    async def test_async_func(prompts, proposer_worker):
        llm = ScaffoldingLlm(
            SimpleController,
            {},
            {'generation': proposer_worker},
        )

        futures = []
        for prompt in prompts:
            future = llm.generate_async(prompt)
            futures.append(future)

        for future in futures:
            result = await future.aresult()
            print(result.output.output_str)

        print(f'main shutting down...')
        llm.shutdown()
        print(f'worker shutting down...')
        proposer_worker.shutdown()
        print(f'main shut down done')

    asyncio.run(test_async_func(prompts, proposer_worker))


def main():
    args = parse_arguments()

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]
    proposer_worker = ProposerWorker(
        args.generation_dir,
        pytorch_backend_config=PyTorchConfig(
            mixed_decoder=True,
            enable_overlap_scheduler=True,
        ),
        sampling_params=SamplingParams(max_tokens=2048),
    )

    if args.run_async:
        test_async(prompts, proposer_worker)
    else:
        test_sync(prompts, proposer_worker)


if __name__ == "__main__":
    main()
