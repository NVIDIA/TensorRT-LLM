from tensorrt_llm.scaffolding.controller import Controller
from tensorrt_llm.scaffolding.task import Task
from tensorrt_llm.scaffolding.contrib.mcp import ChatTask
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from .researcher import Researcher,  ResearchTask
from .prompts import supervisor_system_prompt, final_report_generation_prompt, generate_research_brief_prompt
from .utils import get_today_str, RoleMessage


@dataclass
class SupervisorTask(Task):
    user_prompt: str
    research_brief: Optional[str] = None
    final_report: Optional[str] = None

    @staticmethod
    def create_from_prompt(prompt: str) -> "SupervisorTask":
        task = SupervisorTask()
        task.user_prompt = prompt
        task.research_brief = ""
        task.final_report = ""
        return task


class Supervisor(Controller):
    def __init__(self, tools: List[Dict[str, Any]], max_research_iter: int = 3, max_concurrent_research_units: int = 3):
        super().__init__()
        self.tools = tools
        self.max_research_iter = max_research_iter
        self.max_concurrent_research_units = max_concurrent_research_units

        # TODO: Definition of researcher tools subject to certain specifications.
        # TODO: Add more tools (e.g., MCP tools) beyond search.
        self.researcher_tools = [
            {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        ]
        self.researcher_controller = Researcher(self.researcher_tools)

    def process(self, tasks: List[Task], **kwargs):
        supervisor_task = tasks[0]
        assert isinstance(supervisor_task, SupervisorTask)

        # Generate research brief by wrapping the user original prompt with the
        # system prompt for generating research brief.
        research_brief_messages = [
            RoleMessage(role="user", content=generate_research_brief_prompt.format(
                date=get_today_str(),
                messages=[RoleMessage(
                    role="user", content=supervisor_task.user_prompt)],
            ))
        ]

        research_brief_task = ChatTask.from_messages(
            messages=research_brief_messages)
        yield [research_brief_task]
        supervisor_task.research_brief = research_brief_task.output_str

        supervisor_prompt_messages = [
            RoleMessage(role="system", content=supervisor_system_prompt.format(
                date=get_today_str(),
                max_researcher_iterations=self.max_research_iter,
                max_concurrent_research_units=self.max_concurrent_research_units
            )),
            RoleMessage(role="user", content=supervisor_task.research_brief)
        ]

        # The messages that the supervisor clarify the research brief with the user.
        supervisor_chat_messages = []

        # TODO: Clarify the research brief with the user.

        # The messages that the supervisor use tools to conduct research.
        supervisor_tools_messages = []

        chat_with_tools_task = ChatTask.from_messages(
            messages=supervisor_prompt_messages + supervisor_tools_messages,
            tools=self.tools
        )

        for _ in range(self.max_research_iter):
            yield [chat_with_tools_task]
            if chat_with_tools_task.finish_reason != 'tool_calls':
                break
            supervisor_tools_messages.append(
                RoleMessage(role="assistant", content=chat_with_tools_task.output_str))

            if len(chat_with_tools_task.tool_calls) > 1:
                print(
                    f"Warning: Multiple tool calls detected. Only the first tool call is processed.")

            tool_call = chat_with_tools_task.tool_calls[0]
            if tool_call.function.name == 'ThinkTool':
                supervisor_tools_messages.append(
                    RoleMessage(role="user", content=f"Reflection recorded: {tool_call.function.arguments['think']}"))
            elif tool_call.function.name == 'ConductResearch':
                researcher_task = ResearchTask.from_topic(
                    tool_call.function.arguments['research_topic'])
                yield from self.researcher_controller.process([researcher_task])
                supervisor_tools_messages.append(
                    RoleMessage(role="user", content=researcher_task.research_result))
            elif tool_call.function.name == 'CompleteResearch':
                break
            chat_with_tools_task = ChatTask.from_messages(
                messages=supervisor_prompt_messages + supervisor_tools_messages,
                tools=self.tools
            )

        final_report_generation_task = ChatTask.from_messages(
            messages=[
                RoleMessage(role="system", content=final_report_generation_prompt.format(
                    research_brief=supervisor_task.research_brief,
                    messages=supervisor_system_prompt + supervisor_chat_messages,
                    findings=supervisor_tools_messages,
                    date=get_today_str()
                ))
            ]
        )
        yield [final_report_generation_task]
        supervisor_task.final_report = final_report_generation_task.output_str
