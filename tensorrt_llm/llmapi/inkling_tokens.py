# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/parser/inkling_tokenizer.py
"""Inkling's control-token alphabet, in one place.

Two independent consumers need the same list -- the reasoning parser, which
routes ``<|content_thinking|>`` and ``<|content_text|>`` runs into separate
channels, and the tool parser, which has to strip every control token out of
the text it reports as visible. Defining it twice would let the two drift, and
a token missing from one list is not a crash: it silently leaks framing into
user-visible output.

Copied verbatim from SGLang's ``INKLING_CONTROL_TOKENS``, which is the
reference implementation for this model.
"""

INKLING_ENDOFTEXT = "<|endoftext|>"
INKLING_MESSAGE_USER = "<|message_user|>"
INKLING_MESSAGE_MODEL = "<|message_model|>"
INKLING_MESSAGE_SYSTEM = "<|message_system|>"
INKLING_MESSAGE_TOOL = "<|message_tool|>"
INKLING_CONTENT_TEXT = "<|content_text|>"
INKLING_CONTENT_IMAGE = "<|content_image|>"
INKLING_CONTENT_MODEL_END_SAMPLING = "<|content_model_end_sampling|>"
INKLING_CONTENT_THINKING = "<|content_thinking|>"
INKLING_CONTENT_AUDIO_INPUT = "<|content_audio_input|>"
INKLING_CONTENT_TOOL_ERROR = "<|content_tool_error|>"
INKLING_CONTENT_XML = "<|content_xml|>"
INKLING_END_MESSAGE = "<|end_message|>"
INKLING_AUDIO_END = "<|audio_end|>"
INKLING_INVOKE_TOOL_JSON = "<|content_invoke_tool_json|>"
INKLING_INVOKE_TOOL_TEXT = "<|content_invoke_tool_text|>"
INKLING_INVOKE_TOOL = "<|content_invoke_tool|>"
INKLING_MODEL_TRIGGER_GENERATION = "<|model_trigger_generation|>"

# Blocks whose payload is a tool invocation. They route to visible content
# rather than to reasoning, matching SGLang, so that a tool parser downstream
# still sees the invocation.
INKLING_TOOL_CONTENT_TOKENS = frozenset(
    {
        INKLING_INVOKE_TOOL_JSON,
        INKLING_INVOKE_TOOL_TEXT,
        INKLING_INVOKE_TOOL,
    }
)

INKLING_CONTROL_TOKENS = frozenset(
    {
        INKLING_ENDOFTEXT,
        INKLING_MESSAGE_USER,
        INKLING_MESSAGE_MODEL,
        INKLING_MESSAGE_SYSTEM,
        INKLING_MESSAGE_TOOL,
        INKLING_CONTENT_TEXT,
        INKLING_CONTENT_IMAGE,
        INKLING_CONTENT_MODEL_END_SAMPLING,
        INKLING_CONTENT_THINKING,
        INKLING_CONTENT_AUDIO_INPUT,
        INKLING_CONTENT_TOOL_ERROR,
        INKLING_CONTENT_XML,
        INKLING_END_MESSAGE,
        INKLING_AUDIO_END,
        INKLING_INVOKE_TOOL_JSON,
        INKLING_INVOKE_TOOL_TEXT,
        INKLING_INVOKE_TOOL,
        INKLING_MODEL_TRIGGER_GENERATION,
    }
)

INKLING_MAX_CONTROL_LEN = max(len(t) for t in INKLING_CONTROL_TOKENS)
