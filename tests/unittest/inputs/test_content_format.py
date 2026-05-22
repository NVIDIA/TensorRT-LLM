# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for content format detection via Jinja AST analysis."""

from tensorrt_llm.inputs.content_format import ContentFormat, detect_content_format

# A template that iterates over message['content'] and checks content['type']
# (OpenAI-style multimodal template)
OPENAI_TEMPLATE = """\
{% for message in messages %}
{% if message['role'] == 'user' %}
{% for content in message['content'] %}
{% if content['type'] == 'text' %}
{{ content['text'] }}
{% elif content['type'] == 'image' %}
<image>
{% endif %}
{% endfor %}
{% endif %}
{% endfor %}
"""

# Same pattern but using dot-access: message.content and content.type
OPENAI_TEMPLATE_DOT_ACCESS = """\
{% for message in messages %}
{% for content in message.content %}
{% if content.type == 'text' %}
{{ content.text }}
{% elif content.type == 'image' %}
<image>
{% endif %}
{% endfor %}
{% endfor %}
"""

# A plain string template that just uses message['content'] as a string
STRING_TEMPLATE = """\
{% for message in messages %}
<|{{ message['role'] }}|>
{{ message['content'] }}
{% endfor %}
"""

# A template that iterates over messages but not over content items
STRING_TEMPLATE_WITH_LOOP = """\
{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>{{ message['content'] }}<|end|>
{% elif message['role'] == 'user' %}
<|user|>{{ message['content'] }}<|end|>
{% endif %}
{% endfor %}
"""

# An empty template
EMPTY_TEMPLATE = ""

# A real-world-like llava template that handles multimodal content
LLAVA_LIKE_TEMPLATE = """\
{% for message in messages %}
{% if message['role'] == 'user' %}
USER: {% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] }}{% elif content['type'] == 'image' %}<image>{% endif %}{% endfor %}
{% elif message['role'] == 'assistant' %}
ASSISTANT: {{ message['content'] }}
{% endif %}
{% endfor %}
"""  # noqa: E501


class TestDetectContentFormat:
    def test_openai_template_bracket_access(self):
        assert detect_content_format(OPENAI_TEMPLATE) == ContentFormat.OPENAI

    def test_openai_template_dot_access(self):
        assert detect_content_format(OPENAI_TEMPLATE_DOT_ACCESS) == ContentFormat.OPENAI

    def test_string_template(self):
        assert detect_content_format(STRING_TEMPLATE) == ContentFormat.STRING

    def test_string_template_with_loop(self):
        assert detect_content_format(STRING_TEMPLATE_WITH_LOOP) == ContentFormat.STRING

    def test_empty_template(self):
        assert detect_content_format(EMPTY_TEMPLATE) == ContentFormat.STRING

    def test_llava_like_template(self):
        assert detect_content_format(LLAVA_LIKE_TEMPLATE) == ContentFormat.OPENAI

    def test_invalid_jinja_falls_back_to_string(self):
        assert detect_content_format("{% invalid jinja %}") == ContentFormat.STRING

    def test_caching(self):
        """Verify that repeated calls return cached results via lru_cache."""
        detect_content_format.cache_clear()
        template = STRING_TEMPLATE

        detect_content_format(template)
        info_after_first = detect_content_format.cache_info()
        assert info_after_first.misses >= 1

        detect_content_format(template)
        info_after_second = detect_content_format.cache_info()
        assert info_after_second.hits == info_after_first.hits + 1
        assert info_after_second.misses == info_after_first.misses


class TestContentFormatEnum:
    def test_values(self):
        assert ContentFormat.OPENAI.value == "openai"
        assert ContentFormat.STRING.value == "string"
        assert ContentFormat.PASSTHROUGH.value == "passthrough"

    def test_from_string(self):
        assert ContentFormat("openai") == ContentFormat.OPENAI
        assert ContentFormat("string") == ContentFormat.STRING
        assert ContentFormat("passthrough") == ContentFormat.PASSTHROUGH
