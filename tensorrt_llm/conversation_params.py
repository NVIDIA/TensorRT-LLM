from dataclasses import dataclass


@dataclass(slots=True, kw_only=True)
class ConversationParams:
    """Conversation parameters.

    Args:
        conversation_id (str): Stable multi-turn conversation id used for routing.
    """

    conversation_id: str

    def __post_init__(self) -> None:
        if self.conversation_id is None:
            raise ValueError("conversation_id must be non-empty")
        conversation_id = str(self.conversation_id).strip()
        if not conversation_id:
            raise ValueError("conversation_id must be non-empty")
        self.conversation_id = conversation_id
