import copy
from enum import Enum
from typing import List

from tensorrt_llm.scaffolding import Controller, Task

from .chat_task import ChatTask
from .mcp_task import MCPCallTask, MCPListTask


class MCPController(Controller):

    class WorkerTag(Enum):
        GENERATION = "generation"
        MCP = "mcp"

    def __init__(self, custom_sampling_params: dict = None):
        super().__init__()
        self.custom_sampling_params = copy.deepcopy(
            custom_sampling_params) if custom_sampling_params else None

    def process(self, tasks: List[Task], **kwargs):
        list_task = MCPListTask.create_mcptask()
        list_task.worker_tag = MCPController.WorkerTag.MCP
        yield [list_task]
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in list_task.result_tools]

        print(f"\navailable_tools {available_tools}\n")
        # return
        assert (len(tasks) == 1)
        system_message = (
            "You are a helpful assistant with access tools:\n\n"
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above.")
        messages = [{"role": "system", "content": system_message}]
        chattask = ChatTask.create_from_prompt(messages, tasks[0].input_str,
                                               available_tools)
        result_task = tasks[0]
        chattask.worker_tag = self.WorkerTag.GENERATION
        if self.custom_sampling_params:
            for key, value in self.custom_sampling_params.items():
                if hasattr(tasks[0], key) and getattr(tasks[0], key) is None:
                    setattr(tasks[0], key, value)
        yield [chattask]
        if chattask.finish_reason != 'tool_calls':
            result_task.output_str = chattask.output_str
            return
        tool_calls = chattask.tool_calls
        mcp_call_tasks = [
            MCPCallTask.create_mcptask(tool_call.function.name,
                                       tool_call.function.arguments)
            for tool_call in tool_calls
        ]
        for task in mcp_call_tasks:
            task.worker_tag = MCPController.WorkerTag.MCP
        print(f"\nmcp_call_tasks is {mcp_call_tasks}\n")
        yield mcp_call_tasks
        mcp_result = mcp_call_tasks[0].output_str
        print(f"\nmcp_result is {mcp_result}\n")
        messages.append({"role": "assistant", "content": chattask.output_str})
        finalchattask = ChatTask.create_from_prompt(messages, mcp_result,
                                                    available_tools)
        finalchattask.worker_tag = self.WorkerTag.GENERATION
        yield [finalchattask]
        result_task.output_str = finalchattask.output_str
        return
