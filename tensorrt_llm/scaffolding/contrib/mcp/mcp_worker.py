import asyncio
import json
from typing import List

from tensorrt_llm.scaffolding import TaskStatus, Worker

from .mcp_task import MCPCallTask, MCPListTask
from .mcp_utils import MCPClient


class MCPWorker(Worker):

    def __init__(
        self,
        mcp_clients: List,
    ):
        self.mcp_clients = mcp_clients

    @classmethod
    async def init_with_urls(cls, urls):
        clients = []
        for url in urls:
            client = MCPClient()
            await client.connect_to_sse_server(server_url=url)
            clients.append(client)
        return cls(clients)

    async def call_handler(self, task: MCPCallTask) -> TaskStatus:
        for mcp_client in self.mcp_clients:
            response = await mcp_client.list_tools()
            for tool in response.tools:
                if task.tool_name not in tool.name:
                    continue
                print(f"\ncall handler {tool.name} and {task.tool_name}\n")
                tool_name = task.tool_name
                args = json.loads(task.args)
                response = await mcp_client.call_tool(tool_name, args)
                task.output_str = response.content[0].text
                return TaskStatus.SUCCESS

    async def list_handler(self, task: MCPListTask) -> TaskStatus:
        result_tools = []
        for mcp_client in self.mcp_clients:
            response = await mcp_client.list_tools()
            result_tools.extend(response.tools)
        task.result_tools = result_tools
        return TaskStatus.SUCCESS

    def shutdown(self):
        loop = asyncio.get_event_loop()
        for mcp_client in self.mcp_clients:
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(mcp_client.cleanup(), loop)
            else:
                loop.run_until_complete(self.mcp_client.cleanup())

    task_handlers = {MCPListTask: list_handler, MCPCallTask: call_handler}
