import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from tensorrt_llm.scaffolding.contrib.mcp import ChatTask
from tensorrt_llm.scaffolding.controller import Controller
from tensorrt_llm.scaffolding.task import Task

from .prompts import (
    final_report_generation_prompt,
    generate_research_brief_prompt,
    supervisor_system_prompt,
)
from .researcher import Researcher, ResearchTask
from .utils import AssistantMessage, SystemMessage, UserMessage, get_today_str


@dataclass
class SupervisorTask(Task):
    user_prompt: str = field(default=None)
    research_brief: str = field(default=None)
    final_report: str = field(default=None)

    @staticmethod
    def create_from_prompt(prompt: str) -> "SupervisorTask":
        task = SupervisorTask()
        task.user_prompt = prompt
        task.research_brief = None
        task.final_report = None
        return task


class Supervisor(Controller):
    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self, max_research_iter: int = 3, max_concurrent_research_units: int = 3):
        super().__init__()
        self.max_research_iter = max_research_iter
        self.max_concurrent_research_units = max_concurrent_research_units

        # TODO: Definition of researcher tools subject to certain specifications.
        # TODO: Add more tools (e.g., MCP tools) beyond search.
        self.researcher_tools = [
            {
                "type": "function",
                "function": {
                    "name": "conduct_research",
                    "description": "Conduct research on a given topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "research_topic": {
                                "type": "string",
                                "description": "The topic of the research",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_research",
                    "description": "Complete the research",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "think_tool",
                    "description": "Think about the research",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "think": {
                                "type": "string",
                                "description": "The reflection of the research",
                            }
                        },
                    },
                },
            },
        ]

        self.researcher_controller = Researcher()

    def process(self, tasks: List[Task], **kwargs):
        supervisor_task = tasks[0]

        # For now, user messages only contain the user's prompt. Later, the user's
        # interactions with the supervisor (e.g., clarifying the research question)
        # can be added.
        user_messages = [UserMessage(content=supervisor_task.input_str).to_dict()]

        # Generate research brief by wrapping the user original prompt with the
        # system prompt for generating research brief.
        research_brief_messages = [
            UserMessage(
                generate_research_brief_prompt.format(date=get_today_str(), messages=user_messages)
            ).to_dict()
        ]

        research_brief_task = ChatTask.from_messages(messages=research_brief_messages)
        research_brief_task.worker_tag = Supervisor.WorkerTag.GENERATION

        yield [research_brief_task]

        research_brief = research_brief_task.output_str

        supervisor_prompt_messages = [
            SystemMessage(
                supervisor_system_prompt.format(
                    date=get_today_str(),
                    max_researcher_iterations=self.max_research_iter,
                    max_concurrent_research_units=self.max_concurrent_research_units,
                )
            ).to_dict(),
            UserMessage(research_brief).to_dict(),
        ]

        # TODO: Clarify the research brief with the user.
        # The messages that the supervisor clarify the research brief with the user.

        # The messages that the supervisor use tools to conduct research.
        supervisor_tools_messages = []

        chat_with_tools_task = ChatTask.from_messages(
            messages=supervisor_prompt_messages + supervisor_tools_messages, tools=self.tools
        )
        chat_with_tools_task.worker_tag = Supervisor.WorkerTag.GENERATION

        for _ in range(self.max_research_iter):
            yield [chat_with_tools_task]
            if chat_with_tools_task.finish_reason != "tool_calls":
                break

            if chat_with_tools_task.output_str is not None:
                supervisor_tools_messages.append(
                    AssistantMessage(chat_with_tools_task.output_str).to_dict()
                )

            research_tasks = []

            for tool_call in chat_with_tools_task.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                supervisor_tools_messages.append(
                    AssistantMessage(
                        f"I have called the tool {tool_name} with arguments: {tool_call.function.arguments}"
                    ).to_dict()
                )

                if tool_name == "think_tool":
                    supervisor_tools_messages.append(
                        UserMessage(f"Reflection recorded: {arguments['think']}").to_dict()
                    )
                elif tool_name == "conduct_research":
                    research_tasks.append(ResearchTask.from_topic(arguments["research_topic"]))

                elif tool_name == "complete_research":
                    break

            # In a single research iteration, the supervisor may invoke multiple tools.
            # For example, it might generate several research topics and assign them
            # concurrently to multiple researchers. We gather these in research_tasks
            # to take advantage of the researcher_controller's capability to process
            # them concurrently.
            if len(research_tasks) != 0:
                yield from self.researcher_controller.process(research_tasks)
                for task in research_tasks:
                    supervisor_tools_messages.append(UserMessage(task.research_result).to_dict())

            chat_with_tools_task = ChatTask.from_messages(
                messages=supervisor_prompt_messages + supervisor_tools_messages, tools=self.tools
            )
            chat_with_tools_task.worker_tag = Supervisor.WorkerTag.GENERATION

        final_report_generation_task = ChatTask.from_messages(
            messages=[
                SystemMessage(
                    final_report_generation_prompt.format(
                        research_brief=research_brief,
                        messages=supervisor_prompt_messages + supervisor_tools_messages,
                        findings=supervisor_tools_messages,
                        date=get_today_str(),
                    )
                ).to_dict()
            ]
        )
        final_report_generation_task.worker_tag = Supervisor.WorkerTag.GENERATION

        yield [final_report_generation_task]

        final_report = final_report_generation_task.output_str

        tasks[0].output_str = final_report
        return
