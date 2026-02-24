from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tensorrt_llm.scaffolding import GenerationTask


@dataclass
class ChatTask(GenerationTask):
    messages: list = None
    tools = None
    finish_reason = None
    tool_calls = None

    @staticmethod
    def create_from_prompt(messages: list, prompt: str, tools) -> "ChatTask":
        task = ChatTask()
        messages.append({"role": "user", "content": prompt})
        task.messages = messages
        task.tools = tools
        return task

    @staticmethod
    def from_messages(
            messages: List[str],
            tools: Optional[List[Dict[str, Any]]] = None) -> "ChatTask":
        task = ChatTask()
        task.messages = messages
        task.tools = tools
        return task
