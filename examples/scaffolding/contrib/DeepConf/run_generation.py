import argparse
import asyncio

from tensorrt_llm.scaffolding import (NativeGenerationController,
                                      ScaffoldingLlm, TRTLLMWorker)
from tensorrt_llm.scaffolding.contrib.DeepConf import DeepConfOfflineController, DeepConfOnlineController


def parse_arguments():
    parser = argparse.ArgumentParser()
    # .e.g. DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
    parser.add_argument(
        '--model_dir',
        type=str,
        required=True,
        help="Path to the directory containing the generation model")
    args = parser.parse_args()
    return args

def run_scaffolding_llm(prompts, proposer_worker, controller):
    print('run_scaffolding_llm')
    llm = ScaffoldingLlm(
        controller,
        {controller.WorkerTag.GENERATION: proposer_worker},
    )
    print('run_scaffolding_llm 22222')
    results = llm.generate(prompts)
    for result in results:
        print(result.outputs[0].text)
    llm.shutdown(shutdown_workers=False)

def test_offline_controller(prompts, proposer_worker):
    prototype_controller = DeepConfOfflineController(
        conf_group_size=10, 
        conf_threshold=0.5, 
        logprobs_topk=20,
        sampling_params={
        "temperature": 0.9,
        "max_tokens": 1024,
    })

    run_scaffolding_llm(prompts, proposer_worker, prototype_controller)


def test_online_controller(prompts, proposer_worker):
    prototype_controller = DeepConfOnlineController(
        conf_group_size=10, 
        conf_threshold=0.5, 
        logprobs_topk=20,
        sampling_params={
        "temperature": 0.9,
        "max_tokens": 1024,
    })

    run_scaffolding_llm(prompts, proposer_worker, prototype_controller)

def main():
    args = parse_arguments()

    prompts = [
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n",
        "There exist real numbers $x$ and $y$, both greater than 1, such that $\\log_x\\left(y^x\\right)=\\log_y\\left(x^{4y}\\right)=10$. Find $xy$.",
        "Find the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.",
    ]

    llm_worker = TRTLLMWorker.init_with_new_llm(
        args.model_dir,
        backend="pytorch",
        max_batch_size=32,
        max_num_tokens=4096,
    )

    test_offline_controller(prompts, llm_worker)
    test_online_controller(prompts, llm_worker)

    llm_worker.shutdown()
    print('llm worker shutdown done')


if __name__ == "__main__":
    main()
