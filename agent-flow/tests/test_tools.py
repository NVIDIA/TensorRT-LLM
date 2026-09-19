from __future__ import annotations

import copy
from enum import Enum
from types import SimpleNamespace
from typing import Annotated, Any, Literal, NotRequired, Required, TypedDict
from unittest.mock import AsyncMock

import pytest

from agent_flow.tools import (
    ToolDefinition,
    normalize_input_schema,
    normalize_tool,
    normalize_tool_result,
    tool,
    tool_result_to_claude,
    tool_result_to_codex,
)


class Location(TypedDict):
    path: str
    line: NotRequired[int]


class SearchArgs(TypedDict, total=False):
    query: Required[str]
    locations: list[Location]
    limit: int | None
    modes: list[Literal["text", "image"]]


class ExtendedSearchArgs(SearchArgs):
    exact: bool


class RecursiveArgs(TypedDict):
    child: NotRequired[RecursiveArgs]


def test_shorthand_schema_supports_primitives_and_a_field_named_type():
    schema = normalize_input_schema({"type": str, "count": int, "ratio": float, "ok": bool})
    assert schema == {
        "type": "object",
        "properties": {
            "type": {"type": "string"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "ok": {"type": "boolean"},
        },
        "required": ["type", "count", "ratio", "ok"],
    }


def test_shorthand_fields_can_use_json_schema_keyword_names():
    schema = normalize_input_schema({"properties": list[str], "required": NotRequired[bool]})
    assert schema["properties"] == {
        "properties": {"type": "array", "items": {"type": "string"}},
        "required": {"type": "boolean"},
    }
    assert schema["required"] == ["properties"]


def test_typeddict_required_and_optional_fields_survive_future_annotations():
    schema = normalize_input_schema(ExtendedSearchArgs)
    assert schema["required"] == ["query", "exact"]
    assert schema["properties"]["locations"] == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "line": {"type": "integer"}},
            "required": ["path"],
        },
    }
    assert schema["properties"]["limit"] == {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    assert schema["properties"]["modes"] == {"type": "array", "items": {"enum": ["text", "image"]}}


def test_schema_supports_annotations_mapping_and_tuples():
    schema = normalize_input_schema(
        {
            "metadata": dict[str, Any],
            "numbers": tuple[int, ...],
            "pair": tuple[str, int],
            "label": Annotated[str, "Label"],
            "optional": Annotated[NotRequired[bool], "Optional flag"],
        }
    )
    assert schema["properties"]["metadata"] == {"type": "object", "additionalProperties": {}}
    assert schema["properties"]["numbers"] == {"type": "array", "items": {"type": "integer"}}
    assert schema["properties"]["pair"] == {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "minItems": 2,
        "maxItems": 2,
    }
    assert schema["properties"]["label"] == {"type": "string"}
    assert "optional" not in schema["required"]


def test_enum_schema():
    class Mode(Enum):
        READ = "read"
        WRITE = "write"

    assert normalize_input_schema({"mode": Mode})["properties"]["mode"] == {
        "enum": ["read", "write"]
    }


def test_explicit_json_schema_is_copied_without_reinterpretation():
    original = {
        "type": "object",
        "properties": {"names": {"type": "array", "items": {"type": "string"}}},
    }
    pristine = copy.deepcopy(original)
    normalized = normalize_input_schema(original)
    assert normalized == pristine
    assert normalized is not original
    normalized["properties"]["names"]["items"]["type"] = "integer"
    assert original == pristine


@pytest.mark.parametrize(
    "original, expected",
    [
        (
            {"type": "object", "additionalProperties": {"type": "string"}},
            {"type": "object", "properties": {}, "additionalProperties": {"type": "string"}},
        ),
        (
            {"$ref": "#/$defs/arguments", "$defs": {"arguments": {"type": "object"}}},
            {
                "type": "object",
                "properties": {},
                "allOf": [{"$ref": "#/$defs/arguments"}],
                "$defs": {"arguments": {"type": "object"}},
            },
        ),
        (
            {
                "type": ["object", "null"],
                "properties": {"name": {"type": "string"}},
                "allOf": [{"required": ["name"]}],
            },
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "allOf": [{"type": ["object", "null"]}, {"required": ["name"]}],
            },
        ),
        (
            {"anyOf": [{"required": ["a"]}, {"required": ["b"]}]},
            {
                "type": "object",
                "properties": {},
                "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
            },
        ),
    ],
)
def test_explicit_json_schema_is_normalized_to_mcp_object_shape(original, expected):
    # The Claude SDK only recognises a JSON Schema when it has a string ``type`` and a
    # ``properties`` key; anything else is re-read as Python shorthand. Root keywords that
    # do not fit that shape move into ``allOf`` with their meaning intact.
    pristine = copy.deepcopy(original)
    assert normalize_input_schema(original) == expected
    assert original == pristine


@pytest.mark.parametrize(
    "schema, message",
    [
        ({"bad": object}, "Unsupported Python type"),
        ({"bad": dict[int, str]}, "keys must be strings"),
        (RecursiveArgs, "Recursive TypedDict"),
        ({"type": "object", "default": object()}, "JSON-serializable"),
        (str, "must be a JSON Schema"),
    ],
)
def test_invalid_schema_fails_during_registration(schema, message):
    with pytest.raises(TypeError, match=message):
        normalize_input_schema(schema)


async def test_framework_decorator_produces_callable_handler_without_sdk_dependency():
    @tool("echo", "Return the input", {"text": str}, annotations={"readOnlyHint": True})
    async def echo(args):
        return {"content": [{"type": "text", "text": args["text"]}]}

    assert isinstance(echo, ToolDefinition)
    assert echo.annotations == {"readOnlyHint": True}
    assert echo.input_schema["properties"] == {"text": {"type": "string"}}
    assert await echo.handler({"text": "hello"}) == {"content": [{"type": "text", "text": "hello"}]}


async def test_legacy_sdk_tool_conversion_preserves_annotations_and_handler():
    from claude_agent_sdk import tool as claude_tool
    from mcp.types import ToolAnnotations

    @claude_tool(
        "read",
        "Read data",
        SearchArgs,
        annotations=ToolAnnotations(readOnlyHint=True, maxResultSizeChars=500),
    )
    async def read(args):
        return {"content": [{"type": "text", "text": args["query"]}]}

    normalized = normalize_tool(read)
    assert normalized.handler is read.handler
    assert normalized.input_schema["required"] == ["query"]
    assert normalized.annotations["readOnlyHint"] is True
    assert normalized.meta == {"anthropic/maxResultSizeChars": 500}
    assert await normalized.handler({"query": "value"}) == {
        "content": [{"type": "text", "text": "value"}]
    }


def test_normalizing_tool_does_not_mutate_annotations():
    original = SimpleNamespace(
        name="read",
        description="Read",
        input_schema={},
        handler=AsyncMock(),
        annotations={"readOnlyHint": True},
    )
    normalized = normalize_tool(original)
    normalized.annotations["readOnlyHint"] = False
    assert original.annotations == {"readOnlyHint": True}


@pytest.mark.parametrize("flag", ["is_error", "isError"])
def test_error_results_preserve_both_mcp_spellings(flag):
    result = {"content": [{"type": "text", "text": "failed"}], flag: True}
    assert tool_result_to_codex(result) == {
        "contentItems": [{"type": "inputText", "text": "failed"}],
        "success": False,
    }
    assert tool_result_to_claude(result)["is_error"] is True


def test_resources_are_readable_instead_of_empty_success():
    result = {
        "content": [
            {
                "type": "resource_link",
                "name": "Report",
                "uri": "file:///report.txt",
                "description": "Findings",
            },
            {"type": "resource", "resource": {"uri": "memo://one", "text": "The result is 42."}},
        ]
    }
    assert tool_result_to_codex(result) == {
        "contentItems": [
            {"type": "inputText", "text": "Report\nfile:///report.txt\nFindings"},
            {"type": "inputText", "text": "The result is 42."},
        ],
        "success": True,
    }
    assert tool_result_to_claude(result)["content"][1] == {
        "type": "text",
        "text": "The result is 42.",
    }


def test_images_use_data_urls_in_codex_and_mcp_bytes_in_claude():
    result = {"content": [{"type": "image", "data": "YWJj", "mimeType": "image/png"}]}
    assert tool_result_to_codex(result) == {
        "contentItems": [{"type": "inputImage", "imageUrl": "data:image/png;base64,YWJj"}],
        "success": True,
    }
    assert tool_result_to_claude(result)["content"] == result["content"]


def test_image_data_url_round_trip():
    result = {"content": [{"type": "image", "image_url": "data:image/webp;base64,YWJj"}]}
    converted = tool_result_to_claude(result)["content"][0]
    assert converted["data"] == "YWJj"
    assert converted["mimeType"] == "image/webp"
    assert (
        tool_result_to_codex(result)["contentItems"][0]["imageUrl"] == "data:image/webp;base64,YWJj"
    )


def test_remote_image_url_is_supported_by_codex_and_explicitly_rejected_by_claude():
    result = {"content": [{"type": "image", "image_url": "https://example.com/image.png"}]}
    assert (
        tool_result_to_codex(result)["contentItems"][0]["imageUrl"]
        == "https://example.com/image.png"
    )
    with pytest.raises(ValueError, match="remote image URLs"):
        tool_result_to_claude(result)


@pytest.mark.parametrize(
    "content, message",
    [
        (
            [{"type": "audio", "data": "YWJj", "mimeType": "audio/wav"}],
            "Unsupported tool result content",
        ),
        (
            [{"type": "resource", "resource": {"uri": "binary://one", "blob": "YWJj"}}],
            "Binary embedded",
        ),
        ([{"type": "image", "data": "YWJj"}], "mimeType"),
        ([{"type": "image"}], "requires base64 data"),
        ([{"type": "text", "text": None}], "requires a string"),
        ([{"type": "resource_link"}], "resource link needs"),
    ],
)
def test_unrepresentable_content_raises_instead_of_disappearing(content, message):
    with pytest.raises(ValueError, match=message):
        tool_result_to_codex({"content": content})


def test_pydantic_mcp_result_conversion():
    from mcp.types import CallToolResult, TextContent

    result = CallToolResult(content=[TextContent(type="text", text="failure")], isError=True)
    assert tool_result_to_codex(result) == {
        "contentItems": [{"type": "inputText", "text": "failure"}],
        "success": False,
    }


def test_structured_content_remains_visible():
    result = {"content": [], "structuredContent": {"answer": 42}}
    assert tool_result_to_codex(result)["contentItems"] == [
        {"type": "inputText", "text": '{"answer": 42}'}
    ]


def test_result_normalization_is_idempotent_and_does_not_mutate_input():
    original = {
        "content": [],
        "structuredContent": {"answer": 42},
        "isError": True,
        "is_error": True,
    }
    pristine = copy.deepcopy(original)
    normalized = normalize_tool_result(original)
    assert normalize_tool_result(normalized) == normalized
    assert "isError" not in normalized
    assert original == pristine


def test_empty_result_retains_its_error_status():
    assert tool_result_to_codex({"content": [], "isError": True}) == {
        "contentItems": [{"type": "inputText", "text": ""}],
        "success": False,
    }


@pytest.mark.parametrize(
    "result", [None, "text", {"content": "text"}, {"content": ""}, {"content": ["text"]}]
)
def test_invalid_result_shape_is_rejected(result):
    with pytest.raises(TypeError):
        normalize_tool_result(result)
