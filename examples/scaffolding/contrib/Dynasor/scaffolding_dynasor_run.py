import argparse

from dynasor_controller import DynasorGenerationController

from tensorrt_llm.scaffolding import (MajorityVoteController, ScaffoldingLlm,
                                      TRTLLMWorker)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_dir",
                        type=str,
                        default="./models/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--max_num_tokens", type=int, default=7000)
    parser.add_argument("--majority_vote", action='store_true')
    parser.add_argument('--sample_num', type=int, default=3)
    args = parser.parse_args()
    return args


def test_sync(prompts, proposer_worker, args):
    dynasor_generation_controller = DynasorGenerationController(
        generation_dir=args.generation_dir, max_tokens=args.max_num_tokens)

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
                proposer_worker
            },
        )
    else:
        # Otherwise Use Dynasor controller directly
        llm = ScaffoldingLlm(
            prototype_controller=dynasor_generation_controller,
            workers={
                DynasorGenerationController.WorkerTag.GENERATION:
                proposer_worker
            },
        )

    results = llm.generate(prompts)
    for result in results:
        print(result.output.output_str)
    print(f"main shutting down...")
    llm.shutdown()
    print(f"worker shutting down...")
    proposer_worker.shutdown()
    print(f"main shut down done")


def main():
    args = parse_arguments()

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]

    llm_worker = TRTLLMWorker.init_with_new_llm(
        args.generation_dir,
        backend="pytorch",
        max_num_tokens=args.max_num_tokens)

    test_sync(prompts, llm_worker, args)


if __name__ == "__main__":
    main()
