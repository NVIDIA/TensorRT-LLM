### :section Basics
### :title Generate text asynchronously
### :order 1
import asyncio

from tensorrt_llm import LLM, SamplingParams


def main():
    # model could accept HF model name or a path to local HF model.
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Async based on Python coroutines
    async def task(prompt: str):
        output = await llm.generate_async(prompt, sampling_params)
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    async def main():
        tasks = [task(prompt) for prompt in prompts]
        await asyncio.gather(*tasks)

    asyncio.run(main())

    # Got output like follows:
    # Prompt: 'Hello, my name is', Generated text: '\n\nJane Smith. I am a student pursuing my degree in Computer Science at [university]. I enjoy learning new things, especially technology and programming'
    # Prompt: 'The capital of France is', Generated text: 'Paris.'
    # Prompt: 'The future of AI is', Generated text: 'an exciting time for us. We are constantly researching, developing, and improving our platform to create the most advanced and efficient model available. We are'


if __name__ == '__main__':
    main()
