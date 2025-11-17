import argparse
import asyncio

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import MCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.DeepResearch import create_open_deep_research_scaffolding_llm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_research_iter", type=int, default=10)
    parser.add_argument("--max_concurrent_research_units", type=int, default=10)
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="gpt-oss-20b")
    return parser.parse_args()


async def main():
    args = parse_arguments()
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)

    generation_worker = TRTOpenaiWorker(client, args.model)

    mcp_worker = MCPWorker.init_with_urls(["http://0.0.0.0:8082/sse"])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_open_deep_research_scaffolding_llm(
        generation_worker, mcp_worker, args.max_research_iter, args.max_concurrent_research_units
    )

    prompt = """
        From 2020 to 2050, how many elderly people will there be in Japan? What is their consumption \
        potential across various aspects such as clothing, food, housing, and transportation? \
        Based on population projections, elderly consumer willingness, and potential changes in their \
        consumption habits, please produce a market size analysis report for the elderly demographic.
    """

    future = llm.generate_async(prompt)
    result = await future.aresult()

    assert result.outputs[0].text is not None
    print("final output:" + result.outputs[0].text)

    llm.shutdown()
    generation_worker.shutdown()
    await mcp_worker.async_shutdown()
    return


if __name__ == "__main__":
    asyncio.run(main())
