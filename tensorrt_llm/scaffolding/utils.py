# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Scaffolding utilities.

:func:`system_prompt` wraps a system-prompt template with a stable
``system_prompt_id`` (UUIDv5 derived from a registered name).  The tagged
value is a ``str`` subclass, so it passes through ``SystemMessage(content=...)``
unchanged and is invisible to call sites that just consume the text.

The id survives ``.format()`` / ``.format_map()`` so templates with
runtime-context placeholders (e.g. ``{date}``, configuration values, or even
user-provided fields) keep the same identity after substitution — the id
identifies the *template*, not the rendered string.

The :class:`ExecutionTracer` reads ``system_prompt_id`` off the message
content when building a ``role="system"`` :class:`TraceEvent` and emits it
into both compact and full traces.
"""

import uuid

# Fixed namespace so UUIDv5(name) is reproducible across runs and processes.
_SYSTEM_PROMPT_NAMESPACE = uuid.UUID("4f9c4d4e-1c7e-4d8a-9f8a-3c1c0e1c1f3d")


class _TaggedSystemPrompt(str):
    """A :class:`str` carrying a stable ``system_prompt_id``.

    Behaves like the underlying string for every operation; ``.format()`` and
    ``.format_map()`` re-wrap their result so substitution preserves identity.
    Other string ops (slicing, ``+``, ``replace``) return a plain ``str`` and
    drop the tag — that is intentional: those represent meaningfully different
    content.
    """

    __slots__ = ("system_prompt_id",)

    def __new__(cls, content: str, system_prompt_id: str) -> "_TaggedSystemPrompt":
        instance = super().__new__(cls, content)
        instance.system_prompt_id = system_prompt_id
        return instance

    def format(self, *args, **kwargs) -> "_TaggedSystemPrompt":
        return _TaggedSystemPrompt(str.format(self, *args, **kwargs), self.system_prompt_id)

    def format_map(self, mapping) -> "_TaggedSystemPrompt":
        return _TaggedSystemPrompt(str.format_map(self, mapping), self.system_prompt_id)

    def __reduce__(self):
        # Make deepcopy / pickle round-trip the tag (dataclasses.asdict deepcopies
        # field values, and the default str.__reduce__ would call __new__ with one
        # positional arg, dropping the id).
        return (_TaggedSystemPrompt, (str(self), self.system_prompt_id))


def system_prompt(content: str, *, name: str) -> _TaggedSystemPrompt:
    """Tag a system-prompt template with a stable UUIDv5 derived from *name*.

    *name* must be unique across the codebase (recommended convention:
    ``"<project>.<prompt_const_name>"``, e.g. ``"coder.coder_system_prompt"``).
    The returned value is a ``str`` subclass usable wherever the original
    template was used; downstream code only sees the text.
    """
    system_prompt_id = str(uuid.uuid5(_SYSTEM_PROMPT_NAMESPACE, name))
    return _TaggedSystemPrompt(content, system_prompt_id)
