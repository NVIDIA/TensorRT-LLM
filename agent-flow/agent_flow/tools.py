"""Backend-independent tool definitions and MCP result conversion.

Tool annotations describe behavior; they never grant permission to execute a tool.
Resources become readable text on both backends. Content that cannot be represented
is rejected explicitly so that a tool cannot appear to succeed with missing output.
"""

from __future__ import annotations

import copy
import json
import types
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class ToolDefinition:
    """A tool that can be translated into either backend's native API."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    annotations: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump(mode="json", by_alias=True, exclude_none=True)
    raise TypeError(f"{label} must be a mapping, got {type(value).__name__}")


def _json_schema(value: Any, active_types: set[type]) -> dict[str, Any]:
    if value is Any:
        return {}
    if value is None or value is type(None):
        return {"type": "null"}
    primitive_types = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if isinstance(value, type) and value in primitive_types:
        return {"type": primitive_types[value]}

    origin = get_origin(value)
    args = get_args(value)
    if origin in (Annotated, Required, NotRequired):
        return _json_schema(args[0], active_types)
    if origin in (Union, types.UnionType):
        return {"anyOf": [_json_schema(arg, active_types) for arg in args]}
    if origin is Literal:
        return {"enum": list(args)}
    if isinstance(value, type) and issubclass(value, Enum):
        return {"enum": [member.value for member in value]}
    if origin is list or value is list:
        return {"type": "array", "items": _json_schema(args[0], active_types) if args else {}}
    if origin is tuple or value is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 2 and args[1] is Ellipsis:
            return {"type": "array", "items": _json_schema(args[0], active_types)}
        return {
            "type": "array",
            "prefixItems": [_json_schema(arg, active_types) for arg in args],
            "minItems": len(args),
            "maxItems": len(args),
        }
    if origin in (dict, Mapping) or value is dict:
        if args and args[0] not in (str, Any):
            raise TypeError("Tool schema dictionary keys must be strings")
        return {
            "type": "object",
            "additionalProperties": _json_schema(args[1], active_types) if args else {},
        }
    if is_typeddict(value):
        if value in active_types:
            raise TypeError("Recursive TypedDict tool schemas require an explicit JSON Schema")
        active_types.add(value)
        try:
            hints = get_type_hints(value, include_extras=True)
            properties = {key: _json_schema(hint, active_types) for key, hint in hints.items()}
            required = []
            for key, hint in hints.items():
                qualifier = _field_qualifier(hint)
                if qualifier is Required or (
                    qualifier is not NotRequired and key in value.__required_keys__
                ):
                    required.append(key)
            return {"type": "object", "properties": properties, "required": required}
        finally:
            active_types.remove(value)
    raise TypeError(f"Unsupported Python type in tool schema: {value!r}")


def _field_qualifier(hint: Any) -> Any:
    while get_origin(hint) is Annotated:
        hint = get_args(hint)[0]
    return get_origin(hint)


def _mcp_object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Give an explicit JSON Schema the root shape MCP tool schemas require.

    Both backends advertise an object schema with a ``properties`` key. The Claude SDK
    also relies on that shape to recognise a JSON Schema: a dict without a string ``type``
    and a ``properties`` key is re-read as Python shorthand, so keywords such as ``$ref``
    become required parameters. Root constraints that do not fit the shape move into
    ``allOf``, which keeps their meaning across JSON Schema drafts.
    """
    constraints = []
    if "$ref" in schema:
        constraints.append({"$ref": schema.pop("$ref")})
    if "type" in schema and not isinstance(schema["type"], str):
        constraints.append({"type": schema.pop("type")})
    if constraints:
        schema["allOf"] = constraints + list(schema.get("allOf", []))
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return schema


def normalize_input_schema(schema: Any) -> dict[str, Any]:
    """Copy JSON Schema into the MCP object shape, or convert shorthand/TypedDict input."""
    if isinstance(schema, Mapping):
        # A field named "type" can also occur in shorthand, e.g. {"type": str}.
        schema_type = schema.get("type")
        is_json_schema = isinstance(schema_type, (str, list)) or any(
            isinstance(schema.get(key), expected)
            for key, expected in (
                ("$ref", str),
                ("$defs", Mapping),
                ("$schema", str),
                ("properties", Mapping),
                ("additionalProperties", (bool, Mapping)),
                ("required", list),
                ("allOf", list),
                ("anyOf", list),
                ("oneOf", list),
            )
        )
        if is_json_schema:
            normalized = _mcp_object_schema(copy.deepcopy(dict(schema)))
        else:
            properties = {key: _json_schema(hint, set()) for key, hint in schema.items()}
            normalized = {
                "type": "object",
                "properties": properties,
                "required": [
                    key for key, hint in schema.items() if _field_qualifier(hint) is not NotRequired
                ],
            }
    elif is_typeddict(schema):
        normalized = _json_schema(schema, set())
    else:
        raise TypeError("Tool input_schema must be a JSON Schema, shorthand mapping, or TypedDict")
    # Fail at registration instead of in a backend's transport encoder.
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("Tool input_schema must contain JSON-serializable values") from exc
    return normalized


def normalize_tool(value: Any) -> ToolDefinition:
    """Convert a framework or legacy SDK tool without importing either backend SDK."""
    name = value.name
    description = value.description
    if not isinstance(name, str) or not name:
        raise ValueError("A tool must have a nonempty string name")
    if not isinstance(description, str):
        raise TypeError("A tool description must be a string")
    if not callable(value.handler):
        raise TypeError(f"Tool {name!r} must have a callable handler")
    annotations = getattr(value, "annotations", None)
    if annotations is not None:
        annotations = _mapping(annotations, label="Tool annotations")
    meta = getattr(value, "meta", None)
    if meta is not None:
        meta = _mapping(meta, label="Tool metadata")
    if annotations and annotations.get("maxResultSizeChars") is not None:
        meta = dict(meta or {})
        meta.setdefault("anthropic/maxResultSizeChars", annotations["maxResultSizeChars"])
    return ToolDefinition(
        name=name,
        description=description,
        input_schema=normalize_input_schema(value.input_schema),
        handler=value.handler,
        annotations=copy.deepcopy(annotations),
        meta=copy.deepcopy(meta),
    )


def tool(
    name: str,
    description: str,
    input_schema: Any,
    annotations: Any = None,
    *,
    meta: Mapping[str, Any] | None = None,
) -> Callable[[ToolHandler], ToolDefinition]:
    """Define an async tool using the same call shape as the Claude SDK decorator."""

    def decorate(handler: ToolHandler) -> ToolDefinition:
        return normalize_tool(
            ToolDefinition(name, description, input_schema, handler, annotations, meta)
        )

    return decorate


def normalize_tool_result(result: Any) -> dict[str, Any]:
    """Normalize MCP blocks and error flags, preserving images and top-level metadata."""
    result = _mapping(result, label="Tool result")
    raw_content = result.get("content", [])
    if raw_content is None:
        raw_content = []
    if not isinstance(raw_content, (list, tuple)):
        raise TypeError("Tool result content must be a list of content blocks")
    content = []
    for raw_item in raw_content:
        item = _mapping(raw_item, label="Tool result content block")
        kind = item.get("type")
        if kind == "text":
            if not isinstance(item.get("text"), str):
                raise ValueError("Tool text content requires a string text field")
            content.append(item)
        elif kind == "image":
            content.append(item)
        elif kind == "resource_link":
            parts = [str(item[key]) for key in ("name", "uri", "description") if item.get(key)]
            if not parts:
                raise ValueError("A tool resource link needs a name, URI, or description")
            content.append({"type": "text", "text": "\n".join(parts)})
        elif kind == "resource":
            resource = _mapping(item.get("resource"), label="Embedded tool resource")
            if not isinstance(resource.get("text"), str):
                raise ValueError(
                    "Binary embedded tool resources are not supported; return a resource link"
                )
            content.append({"type": "text", "text": resource["text"]})
        else:
            raise ValueError(f"Unsupported tool result content type: {kind!r}")
    if "structuredContent" in result:
        # Dynamic tools and the Claude SDK's in-process adapter do not forward
        # structured content; include it as JSON text so it remains visible.
        content.append(
            {
                "type": "text",
                "text": json.dumps(result.pop("structuredContent"), ensure_ascii=False),
            }
        )
    result["content"] = content
    camel_case_error = result.pop("isError", False)
    result["is_error"] = bool(result.get("is_error") or camel_case_error)
    return result


def _image_url(item: dict[str, Any]) -> str:
    data = item.get("data")
    if isinstance(data, str) and data:
        if data.startswith("data:"):
            return data
        mime_type = item.get("mimeType")
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise ValueError("A base64 tool image requires an image/* mimeType")
        return f"data:{mime_type};base64,{data}"
    image_url = item.get("image_url")
    if isinstance(image_url, Mapping):
        image_url = image_url.get("url")
    if isinstance(image_url, str) and image_url:
        return image_url
    raise ValueError("A tool image requires base64 data and mimeType, or image_url")


def tool_result_to_codex(result: Any) -> dict[str, Any]:
    """Translate MCP results into the Codex dynamic-tool response wire format."""
    normalized = normalize_tool_result(result)
    items = []
    for item in normalized["content"]:
        if item["type"] == "image":
            items.append({"type": "inputImage", "imageUrl": _image_url(item)})
        else:
            items.append({"type": "inputText", "text": item["text"]})
    return {
        "contentItems": items or [{"type": "inputText", "text": ""}],
        "success": not normalized["is_error"],
    }


def tool_result_to_claude(result: Any) -> dict[str, Any]:
    """Translate results into Claude SDK MCP format, retaining image bytes and MIME."""
    normalized = normalize_tool_result(result)
    for index, item in enumerate(normalized["content"]):
        if item["type"] != "image":
            continue
        url = _image_url(item)
        if not url.startswith("data:") or ";base64," not in url:
            raise ValueError(
                "Claude SDK tool images require base64 data; remote image URLs are unsupported"
            )
        header, data = url.split(";base64,", 1)
        normalized["content"][index] = {
            **item,
            "data": data,
            "mimeType": header.removeprefix("data:"),
        }
    return normalized
