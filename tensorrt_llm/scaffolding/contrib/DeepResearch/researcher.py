from dataclasses import dataclass
from typing import List, Dict, Any

from tensorrt_llm.scaffolding import Controller, Task
from tensorrt_llm.scaffolding.contrib.mcp import ChatTask, MCPCallTask, MCPController
from .prompts import research_system_prompt, compress_system_prompt, compress_research_simple_human_message
from .utils import RoleMessage, get_today_str


@dataclass
class ResearchTask(Task):
    research_topic: str
    research_result: str

    @staticmethod
    def from_topic(topic: str) -> "ResearchTask":
        task = ResearchTask()
        task.research_topic = topic
        task.research_result = ""
        return task


class Researcher(Controller):
    def __init__(self, tools: List[Dict[str, Any]], max_tools_iter: int = 3, max_compress_iter: int = 3):
        self.tools = tools
        self.max_tools_iter = max_tools_iter
        self.max_compress_iter = max_compress_iter

    def process(self, tasks: List[Task], **kwargs):
        research_task = tasks[0]
        assert isinstance(research_task, ResearchTask)

        research_prompt_messages = [RoleMessage(role="system", content=research_system_prompt.format(
            date=get_today_str(),
            mcp_prompt='\n'.join(
                [f"**{tool['name']}**: {tool['description']}" for tool in self.tools]),
        ))]

        research_tools_messages = []
        chat_with_tools_task = ChatTask.from_messages(
            research_prompt_messages + research_tools_messages, self.tools)

        for _ in range(self.max_tools_iter):
            yield [chat_with_tools_task]

            if chat_with_tools_task.finish_reason != 'tool_calls':
                break

            research_tools_messages.append(RoleMessage(
                role="assistant", content=chat_with_tools_task.output_str))

            mcp_call_tasks = [
                MCPCallTask.create_mcptask(tool_call.function.name,
                                           tool_call.function.arguments)
                for tool_call in chat_with_tools_task.tool_calls
            ]
            for task in mcp_call_tasks:
                task.worker_tag = MCPController.WorkerTag.MCP

            yield mcp_call_tasks

            for task in mcp_call_tasks:
                research_tools_messages.append(RoleMessage(
                    role="user", content=task.output_str))

            chat_with_tools_task = ChatTask.from_messages(
                research_prompt_messages + research_tools_messages, self.tools)

        compress_prompt_messages = [RoleMessage(
            role="system", content=compress_system_prompt.format(date=get_today_str()))]

        compress_messages = research_tools_messages + \
            [RoleMessage(
                role="user", content=compress_research_simple_human_message)]
        compress_task = ChatTask.from_messages(
            compress_prompt_messages + compress_messages)

        for _ in range(self.max_compress_iter):
            yield [compress_task]
            if compress_task.finish_reason == 'finish':
                break

        research_task.research_result = compress_task.output_str
        return
