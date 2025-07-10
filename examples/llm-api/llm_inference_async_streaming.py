### :section Basics
### :title Generate text in streaming
### :order 2
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
    async def task(id: int, prompt: str):

        # streaming=True is used to enable streaming generation.
        async for output in llm.generate_async(prompt,
                                               sampling_params,
                                               streaming=True):
            print(f"Generation for prompt-{id}: {output.outputs[0].text!r}")

    async def main():
        tasks = [task(id, prompt) for id, prompt in enumerate(prompts)]
        await asyncio.gather(*tasks)

    asyncio.run(main())

    # Got output like follows:
    # Generation for prompt-0: '\n'
    # Generation for prompt-3: 'an'
    # Generation for prompt-2: 'Paris'
    # Generation for prompt-1: 'likely'
    # Generation for prompt-0: '\n\n'
    # Generation for prompt-3: 'an exc'
    # Generation for prompt-2: 'Paris.'
    # Generation for prompt-1: 'likely to'
    # Generation for prompt-0: '\n\nJ'
    # Generation for prompt-3: 'an exciting'
    # Generation for prompt-2: 'Paris.'
    # Generation for prompt-1: 'likely to nomin'
    # Generation for prompt-0: '\n\nJane'
    # Generation for prompt-3: 'an exciting time'
    # Generation for prompt-1: 'likely to nominate'
    # Generation for prompt-0: '\n\nJane Smith'
    # Generation for prompt-3: 'an exciting time for'
    # Generation for prompt-1: 'likely to nominate a'
    # Generation for prompt-0: '\n\nJane Smith.'
    # Generation for prompt-3: 'an exciting time for us'
    # Generation for prompt-1: 'likely to nominate a new'


if __name__ == '__main__':
    main()
