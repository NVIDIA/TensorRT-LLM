import json

from tensorrt_llm.scaffolding import TaskStatus, Worker

from .mcp_task import MCPCallTask, MCPListTask
from .mcp_utils import MCPClient


class MCPWorker(Worker):

    def __init__(
        self,
        mcp_client: MCPClient,
    ):
        self.mcp_client = mcp_client

    @classmethod
    async def init_with_url(cls, url):
        client = MCPClient()
        try:
            await client.connect_to_sse_server(server_url=url)
        finally:
            return cls(client)

    async def call_handler(self, task: MCPCallTask) -> TaskStatus:
        tool_name = task.tool_name
        args = json.loads(task.args)
        response = await self.mcp_client.call_tool(tool_name, args)
        print(f"mcp call tool response {response}")
        task.output_str = response.content[0].text
        return TaskStatus.SUCCESS

    async def list_handler(self, task: MCPListTask) -> TaskStatus:
        response = await self.mcp_client.list_tools()
        task.output_str = response
        task.result_tools = response.tools
        return TaskStatus.SUCCESS

    async def shutdown(self):
        await self.mcp_client.cleanup()

    task_handlers = {MCPListTask: list_handler, MCPCallTask: call_handler}
