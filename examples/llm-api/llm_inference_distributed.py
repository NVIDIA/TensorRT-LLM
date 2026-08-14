### :section Basics
### :title Distributed LLM Generation
### :order 3
from tensorrt_llm import LLM, SamplingParams


def main():
    # model could accept HF model name or a path to local HF model.
    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # Enable 2-way tensor parallelism
        tensor_parallel_size=2
        # Enable 2-way pipeline parallelism if needed
        # pipeline_parallel_size=2
        # Enable 2-way expert parallelism for MoE model's expert weights
        # moe_expert_parallel_size=2
        # Enable 2-way tensor parallelism for MoE model's expert weights
        # moe_tensor_parallel_size=2
    )

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


# The entry point of the program need to be protected for spawning processes.
if __name__ == '__main__':
    main()
