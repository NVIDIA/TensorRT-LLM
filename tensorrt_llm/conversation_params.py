from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class ConversationParams:
    conversation_id: str
