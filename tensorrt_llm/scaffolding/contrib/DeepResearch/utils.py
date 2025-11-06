from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


@dataclass
class RoleMessage:
    role: str
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def __repr__(self) -> str:
        return f"{self.role}: {self.content}\n"

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(role=data["role"], content=data["content"])


@dataclass
class UserMessage(RoleMessage):
    def __init__(self, content: str):
        super().__init__(role="user", content=content)


@dataclass
class AssistantMessage(RoleMessage):
    def __init__(self, content: str):
        super().__init__(role="assistant", content=content)


@dataclass
class SystemMessage(RoleMessage):
    def __init__(self, content: str):
        super().__init__(role="system", content=content)
