# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate caller controls against the active Jinja chat template."""

import hashlib
import os
from collections.abc import Collection
from functools import lru_cache

import jinja2
import jinja2.ext
import jinja2.nodes

from ..logger import logger

ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS_ENV_VAR = "TRTLLM_ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS"


class UnusedChatTemplateKwargsError(ValueError):
    """A caller-supplied chat_template_kwargs key is not read by the template.

    A distinct type so serving layers can map this client mistake to HTTP 400
    without treating every ValueError raised during rendering the same way.
    """


# Renderer parameters and standard context overrides retain their existing
# behavior. Only additional template controls are checked.
#
# `enable_thinking`, `thinking` and `force_nonempty_content` are not
# template-only controls: the reasoning parsers read them from
# chat_template_kwargs after generation (llmapi/reasoning_parser.py,
# serve/postprocess_handlers.py) to decide how to split reasoning from the
# visible answer, and the servers set `enable_thinking`/`thinking` from
# API-level fields (Responses `reasoning.effort`, Anthropic `thinking`).
# `force_nonempty_content` is consumed by NemotronV3ReasoningParser
# independently of the template. A template that never reads them still
# leaves them with a consumer, so they are accepted everywhere.
ALWAYS_ALLOWED_CHAT_TEMPLATE_KWARGS = frozenset(
    {
        "enable_thinking",
        "thinking",
        "force_nonempty_content",
        "messages",
        "tools",
        "documents",
        "add_generation_prompt",
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
        "chat_template",
        "continue_final_message",
        "tokenize",
        "padding",
        "truncation",
        "max_length",
        "return_tensors",
        "return_dict",
        "return_assistant_tokens_mask",
        "tokenizer_kwargs",
    }
)


class _GenerationTagExtension(jinja2.ext.Extension):
    """Preserve the body of Transformers' generation tag for AST analysis."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Scope:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body).set_lineno(lineno)


@lru_cache(maxsize=128)
def _referenced_template_variables(template_source: str) -> frozenset[str] | None:
    """Find possible reads; return None when static analysis is incomplete."""
    # This environment only parses templates into an AST (env.parse below); it
    # never renders, so autoescape has no effect here. Enable it anyway for the
    # secure default (bandit B701).
    env = jinja2.Environment(
        autoescape=True,
        extensions=[jinja2.ext.loopcontrols, _GenerationTagExtension],
    )
    try:
        ast = env.parse(template_source)
    except jinja2.TemplateSyntaxError:
        # Leave syntax errors and custom extensions to the actual renderer.
        # A parse failure is not evidence that a caller's control is unused.
        return None
    if any(
        ast.find_all(
            (
                jinja2.nodes.Include,
                jinja2.nodes.Import,
                jinja2.nodes.FromImport,
                jinja2.nodes.Extends,
            )
        )
    ):
        return None
    # Conservatively include all reads, including globals, self-defaulted
    # variables and macro arguments. Meta analysis excludes Jinja globals
    # even though caller kwargs can override them.
    return frozenset(node.name for node in ast.find_all(jinja2.nodes.Name) if node.ctx == "load")


@lru_cache(maxsize=1024)
def _warn_unused_key(template_hash: str, key: str) -> None:
    logger.warning(
        f"chat_template_kwargs key '{key}' is not referenced by the active chat template."
    )


def validate_chat_template_kwargs(
    template_source: str | None,
    chat_template_kwargs: dict[str, object] | None,
    strict: bool | None = None,
    injected_keys: Collection[str] | None = None,
) -> None:
    """Reject unused controls, or warn when the compatibility escape hatch is set.

    Non-Jinja renderers and templates that cannot be fully analyzed retain
    their existing behavior. Standard renderer options are always accepted.

    Args:
        template_source: Resolved Jinja source, or None for a native renderer.
        chat_template_kwargs: The controls handed to the renderer.
        strict: Override the default rejection policy. When omitted,
            TRTLLM_ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS=1 selects warnings.
        injected_keys: Keys the server derived from API-level fields rather
            than the caller. The Anthropic adapter, for instance, sends both
            `clear_thinking` (GLM templates) and `drop_thinking` (DeepSeek-V4)
            because it cannot know which one the template reads. The caller
            cannot remove these, so they are exempt; a template that does not
            read one simply ignores it, as before.

    Raises:
        UnusedChatTemplateKwargsError: A caller-supplied control is not read
            by an analyzable template.
    """
    if not chat_template_kwargs or not isinstance(template_source, str):
        return
    referenced = _referenced_template_variables(template_source)
    if referenced is None:
        return
    unknown = set(chat_template_kwargs) - referenced - ALWAYS_ALLOWED_CHAT_TEMPLATE_KWARGS
    if injected_keys:
        unknown -= set(injected_keys)
    unknown = sorted(unknown)
    if not unknown:
        return
    if strict is None:
        strict = os.getenv(ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS_ENV_VAR, "0") != "1"
    if strict:
        raise UnusedChatTemplateKwargsError(
            f"chat_template_kwargs {unknown} are not referenced by the active chat template. "
            "Remove them, or set "
            f"{ALLOW_UNUSED_CHAT_TEMPLATE_KWARGS_ENV_VAR}=1 to allow them with a warning."
        )
    template_hash = hashlib.sha256(template_source.encode("utf-8")).hexdigest()
    for key in unknown:
        _warn_unused_key(template_hash, key)
