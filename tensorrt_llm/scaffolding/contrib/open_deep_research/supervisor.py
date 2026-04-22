# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
from typing import List

from tensorrt_llm.scaffolding.contrib.iter_research.agent import VisitController
from tensorrt_llm.scaffolding.controller import (
    ChatWithMCPController,
    Controller,
    NativeGenerationController,
    ParallelProcess,
)
from tensorrt_llm.scaffolding.scaffolding_llm import ScaffoldingLlm
from tensorrt_llm.scaffolding.task import (
    ChatTask,
    MCPCallTask,
    SystemMessage,
    Task,
    ToolMessage,
    UserMessage,
)
from tensorrt_llm.scaffolding.task_collection import (
    DropKVCacheWorkerTag,
    QueryCollector,
    TaskMetricsCollector,
    TokenizeWorkerTag,
    sub_request_node,
    tokenize_trace_scope,
    with_execution_tracing,
    with_task_collection,
)
from tensorrt_llm.scaffolding.worker import Worker

from .prompts import (
    COMPRESSOR_SYSTEM_PROMPT,
    FINAL_REPORT_GENERATION_PROMPT,
    GENERATE_RESEARCH_BRIEF_SYSTEM_PROMPT,
    GENERATE_RESEARCH_BRIEF_USER_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from .researcher import Compressor, ResearchChatWithMCPController, Researcher, ResearchTask
from .tools import complete_research_tool, conduct_research_tool, think_tool
from .utils import get_today_str


@sub_request_node("agent_deep_research", is_top_level=True)
# @drop_kv_cache_scope()
class Supervisor(Controller):
    tools = [conduct_research_tool, complete_research_tool, think_tool]
    max_research_iter = 12
    max_concurrent_research_units = 8

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
            SystemMessage(
                content=GENERATE_RESEARCH_BRIEF_SYSTEM_PROMPT.format(
                    date=get_today_str(),
                ),
            ),
            UserMessage(
                content=GENERATE_RESEARCH_BRIEF_USER_PROMPT.format(
                    messages=str(user_topic),
                ),
            ),
        ]

        research_brief_task = ChatTask.create_from_messages(messages=research_brief_messages)

        yield from self.brief_controller.process([research_brief_task])

        research_brief = research_brief_task.messages[-1].content

        research_planning_task = ChatTask.create_from_prompt(
            research_brief,
            [
                SystemMessage(
                    content=SUPERVISOR_SYSTEM_PROMPT.format(
                        date=get_today_str(),
                        max_researcher_iterations=self.max_research_iter,
                        max_concurrent_research_units=self.max_concurrent_research_units,
                    ),
                )
            ],
            tools=self.tools,
        )

        research_findings = {}
        for _ in range(self.max_research_iter):
            yield from self.research_planning_controller.process([research_planning_task])

            if research_planning_task.finish_reason != "tool_calls":
                break

            research_tasks_list = []

            for tool_call in research_planning_task.messages[-1].tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                if tool_name == "think_tool":
                    research_planning_task.add_message(
                        ToolMessage(
                            f"Reflection recorded: {arguments['think']}", tool_call_id=tool_call.id
                        )
                    )
                elif tool_name == "conduct_research":
                    research_task = ResearchTask.from_topic(
                        arguments["research_topic"], tool_call.id
                    )
                    research_tasks_list.append([research_task])

                elif tool_name == "complete_research":
                    research_planning_task.add_message(
                        ToolMessage("Research completed.", tool_call_id=tool_call.id)
                    )
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
                        ToolMessage(
                            research_tasks[0].research_findings, research_tasks[0].tool_call_id
                        )
                    )
                    topic = research_tasks[0].research_topic
                    findings = research_tasks[0].research_findings
                    research_findings[topic] = findings

        # Generate final report based on interactions with the user and the research findings
        # gathered by the researchers.
        research_planning_task.add_message(
            UserMessage(FINAL_REPORT_GENERATION_PROMPT.format(date=get_today_str()))
        )
        yield from self.final_report_controller.process([research_planning_task])
        final_report = research_planning_task.messages[-1].content

        tasks[0].output_str = final_report
        tasks[0].output_tokens = tasks[0].output_tokens or []
        return


class BriefController(NativeGenerationController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, tasks: List[Task], **kwargs):
        yield from super().process(tasks, **kwargs)
        return


class ResearchPlanningController(NativeGenerationController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, tasks: List[Task], **kwargs):
        yield from super().process(tasks, **kwargs)
        return


class FinalReportController(NativeGenerationController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, tasks: List[Task], **kwargs):
        yield from super().process(tasks, **kwargs)
        return


def create_open_deep_research_controller(
    max_tokens: int = 16384,
    max_webpage_tokens: int = 48000,
    enable_statistics: bool = False,
    enable_query_collector: bool = False,
    enable_tracing: bool = False,
) -> Controller:
    """Create the prototype controller for open deep research scaffolding."""
    sampling_params = {
        "temperature": 0.9,
        "max_tokens": max_tokens,
    }
    gerneration_controller = NativeGenerationController(sampling_params=sampling_params)
    visit_controller = VisitController(
        generation_controller=gerneration_controller,
        max_webpage_tokens=max_webpage_tokens,
    )

    supervisor_type = Supervisor
    compressor_type = Compressor
    chat_with_mcp_type = ResearchChatWithMCPController
    researcher_type = Researcher
    brief_type = BriefController
    research_planning_type = ResearchPlanningController
    final_report_type = FinalReportController

    if enable_statistics:

        def wrap_with_detailed_profiler(controller_type, controller_name):
            return with_task_collection(
                f"{controller_name}TaskCollection",
                TaskMetricsCollector,
                controller_name=controller_name,
                task_types=[ChatTask, MCPCallTask],
                enable_print=True,
                capture_messages=True,
            )(controller_type)

        supervisor_type = wrap_with_detailed_profiler(supervisor_type, "Supervisor")
        compressor_type = wrap_with_detailed_profiler(compressor_type, "Compressor")
        chat_with_mcp_type = wrap_with_detailed_profiler(chat_with_mcp_type, "ChatWithMCP")
        researcher_type = wrap_with_detailed_profiler(researcher_type, "Researcher")
        brief_type = wrap_with_detailed_profiler(brief_type, "Brief")
        research_planning_type = wrap_with_detailed_profiler(
            research_planning_type, "ResearchPlanning"
        )
        final_report_type = wrap_with_detailed_profiler(final_report_type, "FinalReport")

    if enable_query_collector:
        chat_with_mcp_type = with_task_collection(
            "query_collect",
            QueryCollector,
        )(chat_with_mcp_type)

    if enable_tracing:
        supervisor_type = with_execution_tracing("Supervisor")(supervisor_type)
        supervisor_type = tokenize_trace_scope()(supervisor_type)

    research_chat_with_tools_controller = chat_with_mcp_type(
        gerneration_controller,
        visit_controller,
        max_iterations=12,
    )
    research_compress_controller = compressor_type(
        gerneration_controller,
        system_prompts=[
            SystemMessage(
                COMPRESSOR_SYSTEM_PROMPT.format(date=get_today_str()),
            )
        ],
    )

    research_controller = researcher_type(
        research_chat_with_tools_controller, research_compress_controller
    )

    brief_controller = brief_type(sampling_params=sampling_params)
    research_planning_controller = research_planning_type(sampling_params=sampling_params)
    final_report_controller = final_report_type(sampling_params=sampling_params)

    return supervisor_type(
        brief_controller, research_planning_controller, research_controller, final_report_controller
    )


def create_open_deep_research_scaffolding_llm(
    generation_worker: Worker,
    mcp_worker: Worker,
    max_tokens: int = 16384,
    max_webpage_tokens: int = 48000,
    max_parallel_requests: int = 1024,
    enable_statistics: bool = False,
    enable_query_collector: bool = False,
    enable_tracing: bool = False,
) -> ScaffoldingLlm:
    supervisor_controller = create_open_deep_research_controller(
        max_tokens=max_tokens,
        max_webpage_tokens=max_webpage_tokens,
        enable_statistics=enable_statistics,
        enable_query_collector=enable_query_collector,
        enable_tracing=enable_tracing,
    )

    workers = {
        NativeGenerationController.WorkerTag.GENERATION: generation_worker,
        ChatWithMCPController.WorkerTag.TOOLCALL: mcp_worker,
        VisitController.WorkerTag.TOOL_CALL: mcp_worker,
        DropKVCacheWorkerTag.DROP_KV_CACHE: generation_worker,
    }
    if enable_tracing:
        workers[TokenizeWorkerTag.TOKENIZE] = generation_worker

    scaffolding_llm = ScaffoldingLlm(
        supervisor_controller,
        workers,
        max_parallel_requests=max_parallel_requests,
    )

    if enable_tracing:
        scaffolding_llm.enable_output_task_collection()

    return scaffolding_llm
