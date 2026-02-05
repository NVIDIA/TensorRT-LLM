import logging
from dataclasses import dataclass, field
from typing import List

from tensorrt_llm.scaffolding import (
    AssistantMessage,
    ChatTask,
    Controller,
    SystemMessage,
    Task,
    UserMessage,
)
from tensorrt_llm.scaffolding.task_collection import drop_kv_cache_scope, sub_request_node

from .prompts import research_system_prompt, research_system_prompt_prefix
from .tools import reflection_tool, web_search_tool
from .utils import get_today_str

LOGGER = logging.getLogger()


@dataclass
class ResearchTask(Task):
    research_topic: str = field(default=None)
    research_result: str = field(default=None)

    @staticmethod
    def from_topic(topic: str) -> "ResearchTask":
        task = ResearchTask()
        task.research_topic = topic
        task.research_result = ""
        return task


class Compressor(Controller):
    def __init__(
        self,
        generation_controller: Controller,
        system_prompts: list[SystemMessage],
        max_iterations: int = 3,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.system_prompts = system_prompts
        self.max_iterations = max_iterations

    def clone(self):
        return Compressor(
            self.generation_controller.clone(), self.system_prompts, self.max_iterations
        )

    def process(self, tasks: List[Task], **kwargs):
        assert len(tasks) == 1 and isinstance(tasks[0], ChatTask), (
            "Compressor only supports one ChatTask"
        )
        compress_task = ChatTask.create_from_prompt(None, self.system_prompts)
        compress_task.add_message(UserMessage(str([str(message) for message in tasks[0].messages])))

        for i in range(self.max_iterations):
            yield from self.generation_controller.process([compress_task])
            if compress_task.finish_reason == "stop":
                break
            if i < self.max_iterations - 1:
                compress_task.messages.pop()

        last_message = compress_task.messages[-1]
        assert isinstance(last_message, AssistantMessage), (
            f"last_message is not AssistantMessage, {type(last_message)=}"
        )
        tasks[0].output_str = last_message.content
        return


@sub_request_node("Researcher")
@drop_kv_cache_scope()
class Researcher(Controller):
    tools = [web_search_tool, reflection_tool]

    def __init__(self, chat_with_tools_controller: Controller, compress_controller: Controller):
        super().__init__()
        self.chat_with_tools_controller = chat_with_tools_controller
        self.compress_controller = compress_controller

    def clone(self):
        return Researcher(
            chat_with_tools_controller=self.chat_with_tools_controller.clone(),
            compress_controller=self.compress_controller.clone(),
        )

    def process(self, research_tasks: List[ResearchTask], **kwargs):
        chat_task = ChatTask.create_from_prompt(
            research_tasks[0].research_topic,
            [
                SystemMessage(
                    research_system_prompt.format(date=get_today_str()),
                    prefix=research_system_prompt_prefix,
                )
            ],
            tools=self.tools,
        )

        yield from self.chat_with_tools_controller.process([chat_task])

        yield from self.compress_controller.process([chat_task])

        research_tasks[0].research_result = chat_task.output_str
        return
