# Generate text with Ray orchestrator.
import argparse

from tensorrt_llm import LLM, SamplingParams


def main():
    # model could accept HF model name or a path to local HF model.
    llm = LLM(
        model=args.model_dir,
        orchestrator_type="ray",  # Enable Ray orchestrator
        # Enable 2-way tensor parallelism
        tensor_parallel_size=args.tp_size,
        # Enable 2-way pipeline parallelism if needed
        pipeline_parallel_size=args.pp_size,
        # Enable 2-way expert parallelism for MoE model's expert weights
        moe_expert_parallel_size=args.moe_ep_size)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    # Got output like
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='LLM Inference with Ray orchestrator')
    parser.add_argument('--model_dir',
                        type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model checkpoint directory")
    parser.add_argument('--tp_size', type=int, default=2)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--moe_ep_size', type=int, default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    main()
