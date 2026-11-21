# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Control-token spellings in chat inputs, encoded as text.

A chat template renders to one string, and encoding that string turns every control-token
spelling in it into the control token, also one inside a message, a tool or a document: a
user message can open a turn of its own. Here such spellings are wrapped in private-use
markers before rendering; the marked spans are then encoded as plain text and the rest of
the prompt, with the template's own control tokens, as usual.

The marking and the encoding match ``apply_chat_template(..., sanitize_control_tokens=True)``
of transformers 5.20 (huggingface/transformers#47386), which this module can delegate to once
the transformers pin reaches 5.20. It needs only the standard library, ``jinja2`` and a
Hugging Face fast tokenizer.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import jinja2
import jinja2.ext
import jinja2.nodes

_START = "\U000f0000"
_END = "\U000f0001"
_WRAPPED = _START + r"\g<0>" + _END


class _GenerationTag(jinja2.ext.Extension):
    """Transformers' ``{% generation %}`` tag, parsed so that the text inside it is seen."""

    tags = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body).set_lineno(lineno)


@functools.lru_cache(maxsize=64)
def _template_literals(chat_template: str) -> str:
    """The template's raw text and string constants, joined so that no search spans two."""
    # Parsed with the options Transformers renders with, and never rendered here.
    env = jinja2.Environment(
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols, _GenerationTag],
    )
    try:
        ast = env.parse(chat_template)
    except jinja2.TemplateSyntaxError:
        # Rendering reports it.
        return ""
    literals = {node.data for node in ast.find_all(jinja2.nodes.TemplateData)}
    literals.update(
        node.value for node in ast.find_all(jinja2.nodes.Const) if isinstance(node.value, str)
    )
    return "\0".join(literals)


def _trie_pattern(node: Dict[str, Any]) -> str:
    branches = [re.escape(char) + _trie_pattern(child) for char, child in node.items() if char]
    if not branches:
        return ""
    pattern = branches[0] if len(branches) == 1 else f"(?:{'|'.join(branches)})"
    # A token that ends here loses to a longer one that goes on.
    return f"(?:{pattern})?" if "" in node else pattern


def _alternation(tokens: Sequence[str]) -> re.Pattern:
    """A pattern for any of ``tokens``, the longest when several match.

    Shaped as a trie, so that a match attempt costs the length of a token, not the number of
    tokens (Mistral has 771 control tokens, most of them starting with ``[``).
    """
    trie: Dict[str, Any] = {}
    for token in tokens:
        node = trie
        for char in token:
            node = node.setdefault(char, {})
        node[""] = {}
    return re.compile(_trie_pattern(trie) or "(?!)")


def _strings(value: Any) -> Iterator[str]:
    """Every string in ``value`` and in its nested lists, tuples and dict values."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            yield item
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)


def _spells(pattern: re.Pattern, values: Any) -> bool:
    """Whether a string in ``values`` (dict values, not keys) matches ``pattern``."""
    search = pattern.search
    return any(search(text) for text in _strings(values))


def _without_markers(text: str) -> str:
    return text.replace(_START, "").replace(_END, "")


def _marked(value: Any, pattern: re.Pattern) -> Any:
    """A copy of ``value`` whose strings have every match of ``pattern`` wrapped in markers.

    Dict keys stay as they are. Markers already in a string are dropped first, so they cannot
    fake a marked token.
    """
    if isinstance(value, str):
        return pattern.sub(_WRAPPED, _without_markers(value))
    if isinstance(value, dict):
        return {key: _marked(item, pattern) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_marked(item, pattern) for item in value]
    return value


@dataclass(frozen=True)
class ControlTokens:
    """The control tokens of a tokenizer under one chat template.

    They are the tokenizer's special tokens, plus its other added tokens that the template's
    text or string constants contain. Tokens that are only whitespace are left out.
    """

    # Any control token.
    pattern: re.Pattern
    # The special tokens alone; None when every control token is special.
    special_pattern: Optional[re.Pattern]
    # A marked control token (captured), or a marker on its own.
    marked: re.Pattern
    # The id of each control token.
    ids: Dict[str, int]
    # A newline, encoded as usual and as plain text.
    newline_ids: List[int]
    newline_text_ids: List[int]

    def found_in(self, *values: Any) -> bool:
        """Whether a string in ``values`` (dict values, not keys) spells a control token."""
        return _spells(self.pattern, values)


@functools.lru_cache(maxsize=32)
def control_tokens(tokenizer: Any, chat_template: str) -> Optional[ControlTokens]:
    """The control tokens of a Hugging Face fast tokenizer under ``chat_template``.

    None when the tokenizer is not a fast one (only its Rust backend can encode text without
    matching added tokens) or has no control tokens.
    """
    if getattr(tokenizer, "backend_tokenizer", None) is None:
        return None
    literals = _template_literals(chat_template)
    ids: Dict[str, int] = {}
    special = []
    for token_id, token in tokenizer.added_tokens_decoder.items():
        text = token.content
        if not text or text.isspace() or not (token.special or text in literals):
            continue
        ids[text] = token_id
        if token.special:
            special.append(text)
    if not ids:
        return None
    pattern = _alternation(list(ids))
    return ControlTokens(
        pattern=pattern,
        special_pattern=None if len(special) == len(ids) else _alternation(special),
        marked=re.compile(f"{_START}({pattern.pattern}){_END}|[{_START}{_END}]"),
        ids=ids,
        newline_ids=_encode(tokenizer, "\n"),
        newline_text_ids=_encode_as_text(tokenizer, "\n"),
    )


def _encode(tokenizer: Any, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _encode_as_text(tokenizer: Any, text: str) -> List[int]:
    """Encode ``text`` without matching any added token, special or not."""
    backend = tokenizer.backend_tokenizer
    if backend.normalizer is not None:
        text = backend.normalizer.normalize_str(text)
    words = [text]
    if backend.pre_tokenizer is not None:
        words = [word for word, _ in backend.pre_tokenizer.pre_tokenize_str(text)]
    return [token.id for word in words for token in backend.model.tokenize(word)]


def _encode_piece(
    encode: Callable[[Any, str], List[int]], tokenizer: Any, piece: str, newline_ids: List[int]
) -> List[int]:
    """Encode a piece from inside a prompt.

    It is encoded after a newline, which is then removed, so that a SentencePiece tokenizer
    does not add the space it adds at the start of a text.
    """
    ids = encode(tokenizer, "\n" + piece)
    if ids[: len(newline_ids)] == newline_ids:
        return ids[len(newline_ids) :]
    return encode(tokenizer, piece)


def _encode_marked(tokens: ControlTokens, tokenizer: Any, text: str) -> List[int]:
    """Encode a prompt rendered from marked inputs.

    A marked control token is encoded as plain text, without its markers; the rest is encoded
    as usual. A marker on its own (a template string method such as ``split`` can separate a
    token from its markers) is dropped, but the text is still split there, so that the text
    on either side cannot join into a control token.

    Raises:
        ValueError: The tokenizer makes a marked control token from its plain text.
    """
    # [text, token, text, token, ..., text], with None as the token of a marker on its own.
    pieces = tokens.marked.split(text)
    ids = _encode(tokenizer, pieces[0])
    for index, piece in enumerate(pieces[1:], start=1):
        if not piece:
            continue
        if index % 2:
            piece_ids = _encode_piece(_encode_as_text, tokenizer, piece, tokens.newline_text_ids)
            if tokens.ids[piece] in piece_ids:
                raise ValueError(
                    f"Cannot encode {piece!r} as text: the tokenizer makes it from plain text."
                )
        else:
            piece_ids = _encode_piece(_encode, tokenizer, piece, tokens.newline_ids)
        ids.extend(piece_ids)
    return ids


def _special_token_affixes(tokenizer: Any) -> Tuple[List[int], List[int]]:
    """The ids that ``encode(text, add_special_tokens=True)`` puts before and after a text."""
    plain = _encode(tokenizer, "x")
    full = tokenizer.encode("x", add_special_tokens=True)
    for start in range(len(full) - len(plain) + 1):
        if full[start : start + len(plain)] == plain:
            return full[:start], full[start + len(plain) :]
    raise ValueError(f"Cannot tell which special tokens {type(tokenizer).__name__} adds.")


def _add_special_tokens_and_truncate(
    tokenizer: Any,
    ids: List[int],
    *,
    add_special_tokens: bool,
    truncate_prompt_tokens: Optional[int],
) -> List[int]:
    """``ids`` as ``tokenizer.encode(..., truncation=True, max_length=...)`` would return them.

    The special tokens count towards ``truncate_prompt_tokens``, and the tokenizer's
    ``truncation_side`` decides which end is kept. The tokenizer itself is not changed.
    """
    head, tail = _special_token_affixes(tokenizer) if add_special_tokens else ([], [])
    if truncate_prompt_tokens is not None:
        keep = max(truncate_prompt_tokens - len(head) - len(tail), 0)
        if len(ids) > keep:
            ids = ids[len(ids) - keep :] if tokenizer.truncation_side == "left" else ids[:keep]
    return head + ids + tail


def encode_as_text(
    tokens: ControlTokens,
    tokenizer: Any,
    render: Callable[..., str],
    rendered: str,
    inputs: Sequence[Any],
    *,
    suffix: str = "",
    add_special_tokens: bool = False,
    truncate_prompt_tokens: Optional[int] = None,
) -> Optional[Tuple[str, List[int]]]:
    """Render ``inputs`` with their control-token spellings marked, and encode the prompt.

    ``render(*inputs)`` renders the chat template, and ``rendered`` is its output for the
    inputs as they are; the inputs themselves are never changed. A template that reads the
    inputs with string methods (``content.split('</think>')`` and then a ``strip``) can render
    marked copies differently. Then only the special tokens are marked: the other control
    tokens in the inputs stay tokens, as they are without marking. If even that renders
    differently, the marked rendering is used, as Transformers does. ``suffix`` is appended
    to the prompt and encoded as usual.

    Returns:
        The prompt and its ids, or None when the inputs spell no control token (or, on the
        second rendering, no special token), so that ``rendered`` stands as it is.
    """
    patterns = [tokens.pattern]
    if tokens.special_pattern is not None:
        patterns.append(tokens.special_pattern)
    expected = _without_markers(rendered)
    for pattern in patterns:
        if not _spells(pattern, inputs):
            return None
        text = render(*[_marked(value, pattern) for value in inputs])
        if _without_markers(text) == expected:
            break
    ids = _encode_marked(tokens, tokenizer, text + suffix)
    ids = _add_special_tokens_and_truncate(
        tokenizer,
        ids,
        add_special_tokens=add_special_tokens,
        truncate_prompt_tokens=truncate_prompt_tokens,
    )
    return _without_markers(text), ids
