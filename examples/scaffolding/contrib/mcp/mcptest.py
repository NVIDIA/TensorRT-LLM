import asyncio
import os

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import OpenaiWorker, ScaffoldingLlm
from tensorrt_llm.scaffolding.contrib import MCPController, MCPWorker


async def main():
    prompts = ["What's the weather like today in LA?"]
    API_KEY = os.environ.get("API_KEY")
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    qwen_worker = OpenaiWorker(client, "qwen-plus")

    mcp_worker = await MCPWorker.init_with_url("http://0.0.0.0:8080/sse")

    prototype_controller = MCPController()
    llm = ScaffoldingLlm(
        prototype_controller,
        {
            MCPController.WorkerTag.GENERATION: qwen_worker,
            MCPController.WorkerTag.MCPLIST: mcp_worker,
            MCPController.WorkerTag.MCPCALL: mcp_worker
        },
    )

    future = llm.generate_async(prompts[0])
    result = await future.aresult()
    print(f"result is {result.output.output_str}")
    mcp_worker.shutdown()
    print(f'main shutting down...')
    llm.shutdown()
    print(f'worker shutting down...')
    qwen_worker.shutdown()

    print(f'main shut down done')
    return


if __name__ == '__main__':
    asyncio.run(main())
