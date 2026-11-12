# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source parity of a family copy and the Modeling V1 code it replaces, read with ``ast``.

:func:`function_digest` hashes a function's signature and body (docstring, comments, formatting
and decorators left out), so a copy whose code is the same token for token has the same digest.
Standard library only: ``python3 _k3_source_parity.py <file> <function>...`` prints digests.
"""

import ast
import hashlib
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPO = Path(__file__).resolve().parents[6]


def _strip_docstring(body: list) -> list:
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            return body[1:]
    return body


def function_dump(path: Path, name: str, edits: Iterable[Tuple[str, str]] = ()) -> str:
    """``ast.dump`` of the arguments and body of the top-level function ``name`` in ``path``.

    ``edits``: ``(old, new)`` source substitutions applied first, each of which must occur once:
    the documented local edits of a copy, named where it is compared.
    """
    text = Path(path).read_text(encoding="utf-8")
    for old, new in edits:
        assert text.count(old) == 1, f"{path}: edit {old!r} occurs {text.count(old)} times"
        text = text.replace(old, new)
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            body = ast.Module(body=_strip_docstring(node.body), type_ignores=[])
            return ast.dump(node.args) + ast.dump(body)
    raise KeyError(f"{name} is not a top-level function of {path}")


def function_digest(path: Path, name: str, edits: Iterable[Tuple[str, str]] = ()) -> str:
    return hashlib.sha256(function_dump(path, name, edits).encode()).hexdigest()


def digests(path: Path, names: Iterable[str]) -> Dict[str, str]:
    return {name: function_digest(path, name) for name in names}


if __name__ == "__main__":
    for fn, digest in digests(Path(sys.argv[1]), sys.argv[2:]).items():
        print(f"{fn} {digest}")
