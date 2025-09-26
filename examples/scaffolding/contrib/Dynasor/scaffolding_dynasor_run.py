import argparse
import asyncio

from tensorrt_llm.scaffolding import (MajorityVoteController, ScaffoldingLlm,
                                      TRTLLMWorker)
from tensorrt_llm.scaffolding.contrib.Dynasor import DynasorGenerationController


def parse_arguments():
    parser = argparse.ArgumentParser()
    # .e.g. DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    parser.add_argument("--max_num_tokens", type=int, default=7000)
    parser.add_argument("--majority_vote", action='store_true')
    parser.add_argument('--sample_num', type=int, default=3)
    parser.add_argument('--streaming', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]

    generation_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir, backend="pytorch", max_num_tokens=args.max_num_tokens)

    dynasor_generation_controller = DynasorGenerationController(
        generation_dir=args.model_dir,
        max_tokens=args.max_num_tokens,
        streaming=args.streaming)

    # If majority voting is requested, wrap the controller in MajorityVoteController
    if args.majority_vote:
        majority_vote_controller = MajorityVoteController(
            generation_controller=dynasor_generation_controller,
            default_sample_num=args.sample_num,
        )
        llm = ScaffoldingLlm(
            prototype_controller=majority_vote_controller,
            workers={
                DynasorGenerationController.WorkerTag.GENERATION:
                generation_worker
            },
        )
    else:
        # Otherwise Use Dynasor controller directly
        llm = ScaffoldingLlm(
            prototype_controller=dynasor_generation_controller,
            workers={
                DynasorGenerationController.WorkerTag.GENERATION:
                generation_worker
            },
        )

    if args.streaming:

        async def task(prompt: str):
            i = 0
            async for result in llm.generate_async(prompt):
                i += 1
                print(">>>", i, result)
                async for output in result.cur_output:
                    print(">>>", i, len(output.outputs[0].token_ids), "\n",
                          output.outputs[0].text)
            print(f">>> final output {len(result.outputs[0].token_ids)}\n",
                  result.outputs[0].text)

        # Need to provide LLM's event loop to get results in the middle of the whole process.
        asyncio.run_coroutine_threadsafe(task(prompts[0]), llm.loop).result()
    else:
        results = llm.generate(prompts)
        for result in results:
            print(result.outputs[0].text)

    print(f"main shutting down...")
    llm.shutdown()
    print(f"worker shutting down...")
    generation_worker.shutdown()
    print(f"main shut down done")


if __name__ == "__main__":
    main()
