import argparse
import asyncio

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import OpenaiWorker, ScaffoldingLlm
from tensorrt_llm.scaffolding.contrib.DeepResearch import Researcher, Supervisor
from tensorrt_llm.scaffolding.contrib.mcp import ChatTask, MCPController, MCPWorker, chat_handler


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_research_iter", type=int, default=10)
    parser.add_argument("--max_concurrent_research_units", type=int, default=10)
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="gpt-oss-20b")
    return parser.parse_args()


async def main():
    args = parse_arguments()
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)

    generation_worker = OpenaiWorker(client, args.model)
    generation_worker.register_task_handler(ChatTask, chat_handler)

    mcp_worker = await MCPWorker.init_with_urls(["http://0.0.0.0:8082/sse"])

    supervisor = Supervisor(
        max_research_iter=args.max_research_iter,
        max_concurrent_research_units=args.max_concurrent_research_units,
    )

    llm = ScaffoldingLlm(
        prototype_controller=supervisor,
        workers={
            Supervisor.WorkerTag.GENERATION: generation_worker,
            Researcher.WorkerTag.GENERATION: generation_worker,
            MCPController.WorkerTag.MCP: mcp_worker,
        },
    )

    prompt = """
        From 2020 to 2050, how many elderly people will there be in Japan? What is their consumption \
        potential across various aspects such as clothing, food, housing, and transportation? \
        Based on population projections, elderly consumer willingness, and potential changes in their \
        consumption habits, please produce a market size analysis report for the elderly demographic.
    """

    future = llm.generate_async(prompt)
    result = await future.aresult()

    print(result.outputs[0].text)

    llm.shutdown()
    generation_worker.shutdown()
    mcp_worker.shutdown()

    return


if __name__ == "__main__":
    asyncio.run(main())
