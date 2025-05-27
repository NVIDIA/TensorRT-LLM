from dataclasses import dataclass, field
from typing import Optional, Union

from tensorrt_llm.scaffolding.task import Task


@dataclass
class MCPCallTask(Task):
    # mcp inputs
    tool_name: Optional[str] = field(default=None)
    args: Optional[dict] = field(default=None)
    # retrying control
    retry: Optional[int] = field(default=1)
    delay: Optional[float] = field(default=10)

    worker_tag: Union[str, "Controller.WorkerTag"] = None

    #result field
    result_str: Optional[str] = None

    @staticmethod
    def create_mcptask(tool_name: str,
                       args: dict,
                       retry: int = 1,
                       delay: float = 1) -> "MCPCallTask":
        task = MCPCallTask()
        task.tool_name = tool_name
        task.args = args
        task.retry = retry
        task.delay = delay
        return task


@dataclass
class MCPListTask(Task):
    worker_tag: Union[str, "Controller.WorkerTag"] = None

    #result field
    result_str: Optional[str] = None
    result_tools = None

    @staticmethod
    def create_mcptask() -> "MCPListTask":
        task = MCPListTask()
        return task
