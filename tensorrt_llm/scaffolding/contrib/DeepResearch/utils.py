import datetime
from typing import TypedDict, Literal

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

class RoleMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def __repr__(self) -> str:
        return f"{self.role}: {self.content}\n"
