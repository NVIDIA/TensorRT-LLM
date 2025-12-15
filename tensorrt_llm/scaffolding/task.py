import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

from tensorrt_llm.executor.result import TokenLogprobs

from .result import ScaffoldingOutput


@dataclass
class Task:
    # Scaffolding delivers the task to the Worker by worker_tag.
    worker_tag: str = field(default=None)

    # For streaming output.
    streaming_output_flag: bool = field(default=False)
    streaming_output_list: list[Any] = field(default_factory=list)

    # Reserve for custom input params.
    custom_input_params: Optional[dict] = None

    # Reserve for custom output params.
    custom_output_params: Optional[dict] = None

    @staticmethod
    def create_from_prompt(prompt: str) -> "Task":
        pass

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        pass

    def create_scaffolding_output_stream(self) -> List[ScaffoldingOutput]:
        pass


class TaskStatus(Enum):
    SUCCESS = "success"
    WORKER_NOT_SUPPORTED = "worker_not_supported"
    WORKER_EXECEPTION = "worker_exception"


@dataclass
class GenerationTask(Task):
    # input field
    input_tokens: Optional[List[int]] = None
    input_str: Optional[str] = None
    skip_tokenizer: bool = False
    skip_detokenizer: bool = False
    #streaming: bool = False

    # sampling params for openai
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    # The special case is `num_logprobs`, its original name si `logprobs` but conflicted by the result field
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    num_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = field(default_factory=list)
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    ignore_eos: bool = False

    # sampling params
    top_k: Optional[int] = None
    return_context_logits: Optional[bool] = False

    # suggest to use Controller.WorkerTag
    # anyway, users need to ensure that the value of the worker_tag can be found in the scaffoldingLlm's workers map
    worker_tag: Union[str, "Controller.WorkerTag"] = None

    # result field
    output_str: Optional[str] = None
    output_tokens: Optional[List[int]] = None
    finish_reason: Optional[str] = None
    # TODO: support openai API format context logits
    context_logits: Optional[torch.Tensor] = None
    # TODO: don't not use TokenLogprobs for general support
    logprobs: Optional[TokenLogprobs] = None
    customized_result_fields: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def create_from_prompt(prompt: str) -> "GenerationTask":
        task = GenerationTask()
        task.input_str = prompt
        task.skip_tokenizer = False
        task.skip_detokenizer = False
        return task

    def create_scaffolding_output(self) -> ScaffoldingOutput:
        return ScaffoldingOutput(self.output_str, self.output_tokens)


@dataclass
class StreamGenerationTask(GenerationTask):
    # input field
    # if the flag is set to True, the worker will cancel the generation work
    cancel_flag: Optional[bool] = field(default=False)
    # the task will be returned to the controller with at least new streaming_step tokens
    # if the streaming_step is set to 0,
    # the task will be returned to the controller immediately with
    # new tokens that have already been generated.
    streaming_step: Optional[int] = field(default=1)

    #result field
    # worker set this field and identify the same task by this field
    request_handle: Any = field(default=None)
    # worker set this field to True when the generation is finished
    end_flag: bool = field(default=False)

    @staticmethod
    def create_from_generation_task(task: GenerationTask,
                                    streaming_step) -> "StreamGenerationTask":
        stream_task = StreamGenerationTask()
        for k, v in task.__dict__.items():
            stream_task.__dict__[k] = v
        stream_task.streaming_step = streaming_step
        return stream_task


@dataclass
class RewardTask(Task):
    # input field
    input_tokens: Optional[List[int]] = field(default=None)
    input_str: Optional[str] = field(default=None)


@dataclass
class RoleMessage:
    role: Optional[str] = field(default=None)
    content: Optional[str] = field(default=None)
    prefix: Optional[str] = field(default=None)

    def __str__(self) -> str:
        return json.dumps({
            "role": self.role,
            "content": self.content,
        })

    def __repr__(self) -> str:
        return f"{self.role}: {self.content}\n"

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(role=data["role"], content=data["content"])


@dataclass
class UserMessage(RoleMessage):

    def __init__(self, content: str, prefix: Optional[str] = None):
        super().__init__(role="user", content=content, prefix=prefix)


@dataclass
class AssistantMessage(RoleMessage):
    reasoning: Optional[str] = field(default=None)
    reasoning_content: Optional[str] = field(default=None)
    tool_calls: Optional[List[Any]] = field(default=None)

    def __init__(self,
                 content: str,
                 reasoning: Optional[str] = None,
                 reasoning_content: Optional[str] = None,
                 tool_calls: Optional[List[Any]] = None):
        super().__init__(role="assistant", content=content)
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls

    def __str__(self) -> str:
        # return f"role: assistant, content: {self.content}, reasoning: {self.reasoning}, reasoning_content: {self.reasoning_content}, tool_calls: {self.tool_calls}"
        return json.dumps({
            "role":
            "assistant",
            "content":
            self.content,
            "reasoning":
            self.reasoning,
            "reasoning_content":
            self.reasoning_content,
            "tool_calls": [str(tool) for tool in self.tool_calls]
            if self.tool_calls is not None else None,
        })


@dataclass
class SystemMessage(RoleMessage):

    def __init__(self, content: str, prefix: Optional[str] = None):
        super().__init__(role="system", content=content, prefix=prefix)


class ToolDescription:

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> Dict[str, Any]:
        pass


class OpenAIToolDescription(ToolDescription):

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                },
            },
        }


@dataclass
class ChatTask(StreamGenerationTask):
    messages: list[RoleMessage] = field(default_factory=list)
    tools: Any = field(default=None)

    # for token counting
    enable_token_counting: bool = field(default=False)
    prompt_tokens_num: int = field(default=0)
    completion_tokens_num: int = field(default=0)
    reasoning_tokens_num: int = field(default=0)

    # for sub request marker
    sub_request_markers: list[tuple[str, int]] = field(default_factory=list)
    unique_id: Optional[int] = field(default=None)

    def messages_to_dict_content(self,
                                 start_index: int = 0
                                 ) -> list[Mapping[str, str]]:
        ret = []
        for message in self.messages[start_index:]:
            if message.content is not None:
                ret.append(message.to_dict())
        return ret

    def add_message(self, message: RoleMessage):
        self.messages.append(message)

    def add_messages(self, messages: list[RoleMessage]):
        self.messages.extend(messages)

    @staticmethod
    def create_from_prompt(user_prompt: Optional[str],
                           system_prompts: Optional[list[SystemMessage]] = None,
                           tools: Optional[Any] = None) -> "ChatTask":
        task = ChatTask()
        if system_prompts is not None:
            task.messages.extend(system_prompts)
        if user_prompt is not None:
            task.add_message(UserMessage(user_prompt))
        task.tools = tools
        return task

    @staticmethod
    def create_from_messages(messages: list[RoleMessage],
                             tools: Optional[Any] = None) -> "ChatTask":
        task = ChatTask()
        task.messages = messages
        task.tools = tools
        return task


@dataclass
class MCPCallTask(Task):
    # mcp inputs
    tool_name: Optional[str] = field(default=None)
    args: Optional[dict] = field(default=None)

    #result field
    result_str: Optional[str] = None

    @staticmethod
    def create_mcptask(tool_name: str, args: dict,
                       worker_tag: str) -> "MCPCallTask":
        task = MCPCallTask()
        task.tool_name = tool_name
        task.args = args
        task.worker_tag = worker_tag
        return task


@dataclass
class DropKVCacheTask(Task):
    messages_to_retain: list[RoleMessage] = field(default_factory=list)
    partial_prefix: Optional[str] = field(
        default=None)  # Currently unused since it's hard to tackle
    chat_task: ChatTask = field(default=None)

    def __init__(self, chat_task: ChatTask, worker_tag: str):
        self.worker_tag = worker_tag

        self.messages_to_retain = []
        self.partial_prefix = None
        self.chat_task = chat_task

        retain = True
        for message in chat_task.messages:
            if (message.role == "system" or message.role == "user") and retain:
                if message.prefix is not None:
                    self.messages_to_retain.append(message)
                if message.prefix is None or len(message.prefix) < len(
                        message.content):
                    self.partial_prefix = message.prefix
                    retain = False
