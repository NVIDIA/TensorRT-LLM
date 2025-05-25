import argparse
import asyncio

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import OpenaiWorker, ScaffoldingLlm
from tensorrt_llm.scaffolding.contrib import (ChatTask, MCPController,
                                              MCPWorker, chat_handler)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_url',
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        '--model',
        type=str,
        default="qwen-plus-latest",
    )
    parser.add_argument('--API_KEY', type=str)
    args = parser.parse_args()
    return args


from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import OpenaiWorker, ScaffoldingLlm
from tensorrt_llm.scaffolding.contrib import MCPController, MCPWorker


async def main():
    args = parse_arguments()
    prompts = [
        # "What's the weather like today in LA?"
        # 'Solve the problem with running python code: What is the number of Fibonacci array 20th element? The array goes like 0,1,1,2,3...'
        # 'Which game won TGA Best Action Game and Players Voice awards in 2024?'
        'What was the score of the NBA playoffs game 7 between the Thunder and the Nuggets in 2025?'
    ]
    API_KEY = args.API_KEY
    urls = [
        "http://0.0.0.0:8080/sse", "http://0.0.0.0:8081/sse",
        "http://0.0.0.0:8082/sse"
    ]
    print(f"API_KEY {API_KEY}")
    client = AsyncOpenAI(api_key=API_KEY, base_url=args.base_url)
    qwen_worker = OpenaiWorker(client, args.model)
    qwen_worker.register_task_handler(ChatTask, chat_handler)
    mcp_worker = await MCPWorker.init_with_urls(urls)

    prototype_controller = MCPController()
    llm = ScaffoldingLlm(
        prototype_controller,
        {
            MCPController.WorkerTag.GENERATION: qwen_worker,
            MCPController.WorkerTag.MCP: mcp_worker
        },
    )

    future = llm.generate_async(prompts[0])
    result = await future.aresult()
    print(f"\nresult is {result.output.output_str}\n")

    print(f'main shutting down...')
    llm.shutdown()
    print(f'worker shutting down...')
    qwen_worker.shutdown()
    mcp_worker.shutdown()

    print(f'main shut down done')
    return


if __name__ == '__main__':
    asyncio.run(main())
