import copy
import json
from dataclasses import dataclass, field
from typing import List

from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    Controller,
    NativeGenerationController,
    ParallelProcess,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    AssistantMessage,
    ChatTask,
    SystemMessage,
    Task,
    UserMessage,
)
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    drop_kv_cache_scope,
    sub_request_node,
)
from tensorrt_llm.scaffolding.worker import Worker

from .prompts import (
    compress_system_prompt,
    compress_system_prompt_prefix,
    final_report_generation_prompt,
    final_report_generation_prompt_prefix,
    generate_research_brief_prompt,
    generate_research_brief_prompt_prefix,
    supervisor_system_prompt,
    supervisor_system_prompt_prefix,
)
from .researcher import Compressor, Researcher, ResearchTask
from .tools import complete_research_tool, conduct_research_tool, think_tool
from .utils import get_today_str


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


@sub_request_node("OpenDeepResearch", is_top_level=True)
@drop_kv_cache_scope()
class Supervisor(Controller):
    tools = [conduct_research_tool, complete_research_tool, think_tool]
    max_research_iter = 3
    max_concurrent_research_units = 5

    def __init__(
        self,
        brief_controller: Controller,
        research_planning_controller: Controller,
        research_with_tools_controller: Controller,
        final_report_controller: Controller,
    ):
        super().__init__()
        self.brief_controller = brief_controller
        self.research_planning_controller = research_planning_controller
        self.research_with_tools_controller = research_with_tools_controller
        self.final_report_controller = final_report_controller

    def clone(self):
        return Supervisor(
            brief_controller=self.brief_controller.clone(),
            research_planning_controller=self.research_planning_controller.clone(),
            research_with_tools_controller=self.research_with_tools_controller.clone(),
            final_report_controller=self.final_report_controller.clone(),
        )

    def process(self, tasks: List[Task], **kwargs):
        supervisor_task = tasks[0]

        user_topic = [UserMessage(content=supervisor_task.input_str)]

        research_brief_messages = [
            UserMessage(
                content=generate_research_brief_prompt.format(
                    date=get_today_str(), messages=str(user_topic)
                ),
                prefix=generate_research_brief_prompt_prefix,
            )
        ]

        research_brief_task = ChatTask.create_from_messages(messages=research_brief_messages)

        yield from self.brief_controller.process([research_brief_task])

        research_brief = research_brief_task.messages[-1].content

        research_planning_task = ChatTask.create_from_prompt(
            research_brief,
            [
                SystemMessage(
                    content=supervisor_system_prompt.format(
                        date=get_today_str(),
                        max_researcher_iterations=self.max_research_iter,
                        max_concurrent_research_units=self.max_concurrent_research_units,
                    ),
                    prefix=supervisor_system_prompt_prefix,
                )
            ],
            tools=self.tools,
        )
        prompt_index = len(research_planning_task.messages)

        for _ in range(self.max_research_iter):
            yield from self.research_planning_controller.process([research_planning_task])

            if research_planning_task.finish_reason != "tool_calls":
                break

            research_tasks_list = []

            for tool_call in research_planning_task.messages[-1].tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                research_planning_task.add_message(
                    AssistantMessage(
                        f"I have called the tool {tool_name} with arguments: {tool_call.function.arguments}"
                    )
                )

                if tool_name == "think_tool":
                    research_planning_task.add_message(
                        UserMessage(f"Reflection recorded: {arguments['think']}")
                    )
                elif tool_name == "conduct_research":
                    research_tasks_list.append(
                        [ResearchTask.from_topic(arguments["research_topic"])]
                    )

                elif tool_name == "complete_research":
                    break

            if len(research_tasks_list) > 0:
                researcher_controllers = [
                    self.research_with_tools_controller.clone()
                    for _ in range(len(research_tasks_list))
                ]
                kwargs_list = [copy.deepcopy(kwargs) for _ in range(len(research_tasks_list))]

                yield ParallelProcess(researcher_controllers, research_tasks_list, kwargs_list)

                for research_tasks in research_tasks_list:
                    research_planning_task.add_message(
                        UserMessage(research_tasks[0].research_result)
                    )

        final_report_generation_task = ChatTask.create_from_prompt(
            user_prompt=research_brief,
            system_prompts=[
                SystemMessage(
                    final_report_generation_prompt.format(
                        research_brief=research_brief,
                        messages=research_planning_task.messages_to_dict_content(),
                        findings=research_planning_task.messages_to_dict_content(prompt_index + 1),
                        date=get_today_str(),
                    ),
                    prefix=final_report_generation_prompt_prefix,
                ),
            ],
        )

        yield from self.final_report_controller.process([final_report_generation_task])

        final_report = final_report_generation_task.messages[-1].content

        tasks[0].output_str = final_report
        return


def create_open_deep_research_scaffolding_llm(
    generation_worker: Worker, mcp_worker: Worker
) -> ScaffoldingLlm:
    gerneration_controller = NativeGenerationController(
        sampling_params={
            "temperature": 0.9,
            "max_tokens": 16386,
        }
    )

    research_chat_with_tools_controller = ChatWithMCPController(gerneration_controller)
    research_compress_controller = Compressor(
        gerneration_controller,
        system_prompts=[
            SystemMessage(
                compress_system_prompt.format(date=get_today_str()),
                prefix=compress_system_prompt_prefix,
            )
        ],
    )

    research_controller = Researcher(
        research_chat_with_tools_controller, research_compress_controller
    )

    brief_controller = gerneration_controller
    research_planning_controller = gerneration_controller
    final_report_controller = gerneration_controller

    supervisor_controller = Supervisor(
        brief_controller, research_planning_controller, research_controller, final_report_controller
    )
    scaffolding_llm = ScaffoldingLlm(
        supervisor_controller,
        {
            NativeGenerationController.WorkerTag.GENERATION: generation_worker,
            ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
            DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
        },
    )

    return scaffolding_llm
