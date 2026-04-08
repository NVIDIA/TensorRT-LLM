# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Content format detection for multimodal chat templates.

Detects whether a Jinja chat template handles multimodal content natively
(OpenAI-style dicts with type/text/image fields) or expects plain strings.
"""

import enum
from functools import lru_cache

from jinja2 import Environment
from jinja2 import nodes as jinja_nodes
from jinja2.exceptions import TemplateSyntaxError


class ContentFormat(enum.Enum):
    """Content format expected by a chat template.

    OPENAI: Template iterates over `message['content']` expecting dicts with 'type' keys
        (e.g., {"type": "text", ...}, {"type": "image"}).
    STRING: Template expects `message['content']` to be a plain string.
    PASSTHROUGH: Skip chat template rendering entirely; just concatenate message content strings.
        Used by models whose input processor already produces the final prompt (e.g. VILA /
        llava_llama, mistral_large_3 native tokenizer path).
    """

    OPENAI = "openai"
    STRING = "string"
    PASSTHROUGH = "passthrough"


def _ast_has_content_iteration(ast: jinja_nodes.Template) -> bool:
    """Helper for determining whether the jinja template has a content iteration loop.

    Walk the Jinja AST to find a `For` node iterating over message content
    with inner type-checking (content['type'] or content.type).

    This detects templates that handle multimodal content natively, e.g.:

    ```
        {% for content in message['content'] %}
            {% if content['type'] == 'image' %}
    ```
    """
    for node in ast.find_all(jinja_nodes.For):
        # Check if iterating over message['content'] or message.content
        iter_node = node.iter
        is_content_iter = False

        if isinstance(iter_node, jinja_nodes.Getitem):
            # message['content']
            if isinstance(iter_node.arg, jinja_nodes.Const) and iter_node.arg.value == "content":
                is_content_iter = True
        elif isinstance(iter_node, jinja_nodes.Getattr):
            # message.content
            if iter_node.attr == "content":
                is_content_iter = True

        if not is_content_iter:
            continue

        # Now check if the loop body accesses `['type']` or `.type` on the loop variable
        loop_var = node.target
        if not isinstance(loop_var, jinja_nodes.Name):
            continue
        loop_var_name = loop_var.name

        for inner_node in node.find_all((jinja_nodes.Getitem, jinja_nodes.Getattr)):
            if isinstance(inner_node, jinja_nodes.Getitem):
                if (
                    isinstance(inner_node.node, jinja_nodes.Name)
                    and inner_node.node.name == loop_var_name
                    and isinstance(inner_node.arg, jinja_nodes.Const)
                    and inner_node.arg.value == "type"
                ):
                    return True
            elif isinstance(inner_node, jinja_nodes.Getattr):
                if (
                    isinstance(inner_node.node, jinja_nodes.Name)
                    and inner_node.node.name == loop_var_name
                    and inner_node.attr == "type"
                ):
                    return True

    return False


@lru_cache(maxsize=128)
def detect_content_format(chat_template: str) -> ContentFormat:
    """Detect whether a chat template handles multimodal content natively.

    Parses the Jinja template AST and looks for patterns like:

    ```
        {% for content in message['content'] %}
            {% if content['type'] == 'image' %} ...
    ```

    If found, the template handles OpenAI-style content dicts.
    Otherwise, it expects plain string content.

    Args:
        chat_template: The Jinja2 chat template string.

    Returns:
        `ContentFormat.OPENAI` if template handles multimodal content dicts,
            `ContentFormat.STRING` otherwise.
    """
    try:
        env = Environment(autoescape=True)
        ast = env.parse(chat_template)
        if _ast_has_content_iteration(ast):
            return ContentFormat.OPENAI
    except TemplateSyntaxError:
        # If parsing fails, fall back to STRING format
        pass
    return ContentFormat.STRING
