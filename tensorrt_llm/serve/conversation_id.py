from typing import Any, Mapping, Optional, Protocol

CONVERSATION_ID_HEADERS = (
    "x-session-id",
    "x-correlation-id",
    "x-session-affinity",
    "x-multi-turn-session-id",
)


class RequestWithConversationParams(Protocol):
    conversation_params: Any


def get_request_conversation_id(request: RequestWithConversationParams) -> Optional[str]:
    conversation_params = request.conversation_params
    if conversation_params is None:
        return None
    return conversation_params.conversation_id


def extract_conversation_id_from_headers(headers: Optional[Mapping[str, str]]) -> Optional[str]:
    if headers is None:
        return None
    lower_headers = {str(key).lower(): value for key, value in headers.items()}
    for header_name in CONVERSATION_ID_HEADERS:
        conversation_id = lower_headers.get(header_name)
        if conversation_id is None:
            continue
        conversation_id = str(conversation_id).strip()
        if conversation_id:
            return conversation_id
    return None


def resolve_request_conversation_id(
    request: RequestWithConversationParams,
    headers: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return conversation_params.conversation_id populated at the serve edge.

    Body ``conversation_params.conversation_id`` is canonical. Headers are used
    only when the body does not provide an id.
    """
    conversation_params = request.conversation_params
    if conversation_params is not None:
        return conversation_params.conversation_id

    conversation_id = extract_conversation_id_from_headers(headers)
    if conversation_id is not None:
        from tensorrt_llm.serve.openai_protocol import ConversationParams

        request.conversation_params = ConversationParams(conversation_id=conversation_id)
    return conversation_id
